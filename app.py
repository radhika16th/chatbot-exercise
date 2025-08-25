import json, os, threading
from flask import Flask, request, render_template, render_template_string, Response, stream_with_context
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import time

# --------------------------
# System prompt & grounding
# --------------------------
SYSTEM_PROMPT = (
    "You are a cautious exercise assistant. You do NOT diagnose or replace medical care. "
    "Rules: First, check for red flags. If present, advise urgent or prompt in-person care and do not give exercises. "
    "If no red flags, offer 3–6 gentle, low-risk movement suggestions tailored to the described symptom. "
    "Use simple, actionable steps with reps/sets/time, plus what to avoid. Include a brief safety disclaimer at the end. "
    "If unsure, ask 1–2 clarifying questions (max) before suggesting movement. Keep responses under 180 words."
)

NON_SYMPTOM_MSG = (
    "I’m an exercise-suggestion bot for minor aches and pains. "
    "Tell me what hurts, how it started, and any red flags (fever, numbness, recent trauma). "
    "I don’t answer general questions like math or trivia."
)

SYMPTOM_WORDS = {
    "pain","ache","sore","stiff","tight","strain","sprain","spasm",
    "tingle","numb","weak","swollen","swelling","bruise","hurt", "discomfort"
}
BODY_PARTS = {
    "back","neck","shoulder","knee","hip","ankle","elbow","wrist",
    "hand","foot","leg","arm","hamstring","quad","glute","calf",
    "chest","rib","head","headache"
}

RED_MSG = (
    "I’m noticing possible red flags based on what you wrote. "
    "Please seek prompt in-person care before starting exercise. "
    "General advice: rest, gentle breathing, comfortable positions. "
    "This is not medical advice."
)


def looks_like_symptom(text: str) -> bool:
    s = (text or "").lower()
    return (any(w in s for w in SYMPTOM_WORDS | BODY_PARTS) and len(s.split()) >= 3)

EXDB_PATH = os.path.join("data", "exercises.json")
with open(EXDB_PATH, "r") as f:
    EXDB = json.load(f)

# --------------------------
# Device & model setup (MPS‑safe)
# --------------------------
MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
ATTN_IMPL = os.getenv("ATTENTION_IMPL", "eager")  # "eager" is safest on MPS

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "mps" and torch.backends.mps.is_built()) else torch.float32
# ----- generation knobs -----
DO_SAMPLE = os.getenv("SAMPLE", "1") != "0"   # set SAMPLE=0 for deterministic
TEMPERATURE = float(os.getenv("TEMP", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REP_PEN", "1.05"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "500"))  # was 120; increases completion chance

NON_SYMPTOM_CHUNK = int(os.getenv("NS_CHUNK", "1"))   # 1 = letter-by-letter
NON_SYMPTOM_DELAY = float(os.getenv("NS_DELAY", "0")) # seconds between chunks


# Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or "<|endoftext|>"
pad_id = tok.pad_token_id
eos_id = tok.eos_token_id or pad_id

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=None,            # force full load; avoids accelerate/meta sharding surprises
    low_cpu_mem_usage=False,
    attn_implementation=ATTN_IMPL,
)
model.to(DEVICE)  # type: ignore
model.eval()

# --------------------------
# Simple keyword screener
# --------------------------
RED_FLAGS = [
    "chest pain", "shortness of breath", "fainting", "bladder", "bowel",
    "progressive weakness", "fever", "recent trauma"
]

def classify(sym: str):
    s = (sym or "").lower()
    if any(r in s for r in RED_FLAGS):
        return "red"
    if "back" in s:
        return "lower_back_pain_mild"
    if "neck" in s:
        return "neck_stiffness"
    if "knee" in s:
        return "knee_pain_mild"
    return None

# Cache the static system part once
SYS_MSG = [{"role": "system", "content": SYSTEM_PROMPT}]

def build_prompt(user_text: str, grounding_json: str) -> str:
    msgs = SYS_MSG + [{
        "role": "user",
        "content": (
            f"User symptom description: {user_text}\n"
            f"Grounded exercise library: {grounding_json}\n"
            "Respond with: screening confirmation, 3–6 specific movements (sets/reps/time), "
            "things to avoid, brief disclaimer."
        )
    }]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def stream_model_prompt(prompt_text: str):
    inputs = tok(prompt_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    def event_stream():
        try:
            for chunk in stream_generate(inputs):  # <-- same TextIteratorStreamer path
                yield f"data: {chunk}\n\n"
        except GeneratorExit:
            return
        except Exception as e:
            yield f"event: error\ndata: {type(e).__name__}: {e}\n\n"
        finally:
            yield "event: done\ndata: end\n\n"

    return Response(stream_with_context(event_stream()),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    })

def build_repeat_prompt(text: str) -> str:
    """
    Build a chat prompt that makes the model output *exactly* `text`.
    We use a narrow 'repeat server' system instruction to avoid extra words.
    """
    messages = [
        {"role": "system",
         "content": "You are a repeat server. Output exactly the user-provided text, "
                    "with no extra words, no quotes, no formatting."},
        {"role": "user", "content": text}
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --------------------------
# Streaming helper (SSE)
# --------------------------
def stream_generate(inputs: dict):
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE if DO_SAMPLE else None,
        top_p=TOP_P if DO_SAMPLE else None,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=True,
        streamer=streamer,
    )

    def _worker():
        with torch.inference_mode():
            model.generate(**{k:v for k,v in gen_kwargs.items() if v is not None}) # type: ignore
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    for new_text in streamer:
        yield new_text

def sse_response(q: str):
    # Non-symptom → stream fixed message via model streamer
    if not looks_like_symptom(q):
        prompt = build_repeat_prompt(NON_SYMPTOM_MSG)
        return stream_model_prompt(prompt)

    # Red flags → stream fixed message via model streamer
    if classify(q) == "red":
        prompt = build_repeat_prompt(RED_MSG)
        return stream_model_prompt(prompt)

    # Normal symptom → stream generated answer via the same streamer
    key = classify(q)
    g = EXDB.get(key, {}) if key else {}
    g = {"moves": (g.get("moves") or [])[:6]} if g else {}
    grounding = json.dumps(g, ensure_ascii=False)

    prompt = build_prompt(q, grounding)
    return stream_model_prompt(prompt)

app = Flask(__name__, template_folder='.')

@app.get("/events")
def events():
    q = request.args.get("q", "")
    return sse_response(q)

@app.route("/", methods=["GET", "POST"])
def root():
    if request.method == "POST":
        q = request.form.get("q", "")
        return render_template("index.html", q=q)
    return render_template("index.html", q=None)

if __name__ == "__main__":
    # Bound to loopback only (localhost) for safety; change host to "0.0.0.0" if you need LAN access
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)