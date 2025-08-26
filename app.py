import json, os, threading
from flask import Flask, request, render_template, render_template_string, Response, stream_with_context
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

# --------------------------
# System prompt & grounding
# --------------------------

# Safety-forward instruction: detect red flags first, otherwise offer suggestions and keep answers concise.
SYSTEM_PROMPT = (
    "You are a cautious exercise assistant. You do NOT diagnose or replace medical care. "
    "Rules: First, check for red flags. If present, advise urgent or prompt in-person care and do not give exercises. "
    "If no red flags, offer 3–6 gentle, low-risk movement suggestions tailored to the described symptom. "
    "Use simple, actionable steps with reps/sets/time, plus what to avoid. Include a brief safety disclaimer at the end. "
    "If unsure, ask 1–2 clarifying questions (max) before suggesting movement. Keep responses under 180 words."
)

# Message used when the input doesn’t look like a symptom description.
NON_SYMPTOM_MSG = (
    "I’m an exercise-suggestion bot for minor aches and pains. "
    "Tell me what hurts, how it started, and any red flags (fever, numbness, recent trauma). "
    "I don’t answer general questions like math or trivia."
)

# If text contains any of these, we treat it as a symptom description.
SYMPTOM_WORDS = {
    "pain","ache","sore","stiff","tight","strain","sprain","spasm",
    "tingle","numb","weak","swollen","swelling","bruise","hurt", "discomfort"
}

BODY_PARTS = {
    "back","neck","shoulder","knee","hip","ankle","elbow","wrist",
    "hand","foot","leg","arm","hamstring","quad","glute","calf",
    "chest","rib","head","headache"
}

# Red-flag caution text
RED_MSG = (
    "I’m noticing possible red flags based on what you wrote. "
    "Please seek prompt in-person care before starting exercise. "
    "General advice: rest, gentle breathing, comfortable positions. "
    "This is not medical advice."
)

# Red flags
RED_FLAGS = [
    "chest pain", "shortness of breath", "fainting", "bladder", "bowel",
    "progressive weakness", "fever", "recent trauma"
]

def looks_like_symptom(text: str) -> bool:
    """
    looks like a symptom if it mentions a symptom word/body part
    AND has at least 3 tokens (helps filter out single words like 'fever').
    """
    s = (text or "").lower()
    return (any(w in s for w in SYMPTOM_WORDS | BODY_PARTS) and len(s.split()) >= 3)

# Load the local JSON exercise library
EXDB_PATH = os.path.join("data", "exercises.json")
with open(EXDB_PATH, "r") as f:
    EXDB = json.load(f)

# --------------------------
# Device & model setup
# --------------------------
MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
ATTN_IMPL = os.getenv("ATTENTION_IMPL", "eager")  # "eager" is safest on MPS

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "mps" and torch.backends.mps.is_built()) else torch.float32

# Generation knobs (used in stream_generate)
DO_SAMPLE = os.getenv("SAMPLE", "1") != "0"   # set SAMPLE=0 for deterministic
TEMPERATURE = float(os.getenv("TEMP", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REP_PEN", "1.05"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "500")) # upper bound to avoid early cutoffs

# Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or "<|endoftext|>"
pad_id = tok.pad_token_id
eos_id = tok.eos_token_id or pad_id

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, # model ID
    torch_dtype=DTYPE, # torch dtype
    device_map=None, # device map
    low_cpu_mem_usage=False, # low CPU memory usage
    attn_implementation=ATTN_IMPL, # attention implementation
)
model.to(DEVICE)  # type: ignore
model.eval()

# --------------------------
# Simple keyword screener
# --------------------------
def classify(sym: str):
    """
    Map a free-text symptom description to a key in EXDB.
    Return "red" if red flags present; else return a condition key or None.
    """
    s = (sym or "").lower()
    if any(r in s for r in RED_FLAGS):
        return "red"
    if "back" in s:
        return "lower_back_pain_mild"
    if "neck" in s:
        return "neck_stiffness"
    if "knee" in s:
        return "knee_pain_mild"
    if "impingement" in s or ("shoulder" in s and "pinch" in s): 
        return "shoulder_impingement_mild"
    if "frozen shoulder" in s or ("shoulder" in s and "stiff" in s): 
        return "frozen_shoulder_early"
    if "de quervain" in s or ("thumb" in s and "wrist" in s): 
        return "de_quervain_tenosynovitis_mild"
    if "carpal" in s or ("wrist" in s and "tingl" in s): 
        return "carpal_tunnel_early"
    if "headache" in s and ("neck" in s or "shoulder" in s): 
        return "tension_headache_muscular"
    if "tmj" in s or "jaw" in s: 
        return "tmj_jaw_discomfort_mild"
    if "rib" in s or "intercostal" in s: 
        return "rib_strain_mild"
    if "outer hip" in s or "lateral hip" in s or "trochanter" in s: 
        return "lateral_hip_pain_gtps"
    if "hip flexor" in s or ("front hip" in s and "tight" in s): 
        return "hip_flexor_tightness_sitting"
    if "adductor" in s or "groin" in s: 
        return "adductor_strain_grade1"
    if "quad" in s or "quadriceps" in s: 
        return "quad_strain_grade1"
    if "calf" in s: 
        return "calf_strain_grade1"
    if "runner" in s or "patellofemoral" in s: 
        return "patellofemoral_pain"
    if "patellar" in s or "jumper" in s: 
        return "patellar_tendinopathy_mild"
    if "immobilization" in s or ("ankle" in s and "stiff" in s): 
        return "ankle_dorsiflexion_stiffness_post_immobilization"
    if "sciatica" in s or ("leg" in s and "nerve" in s): 
        return "sciatic_irritation_mild_no_red_flags"
    if "sore" in s and "workout" in s: 
        return "doms_general_soreness"
    return None

# System message
SYS_MSG = [{"role": "system", "content": SYSTEM_PROMPT}]

def build_prompt(user_text: str, grounding_json: str) -> str:
    """
    Build the chat template prompt containing:
      - system message (safety/spec)
      - user message with the user's symptom + grounded exercise list
    """
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
    """
    Tokenize a ready-to-go prompt and stream the model's output as SSE.
    Used by ALL paths (normal, non-symptom, red flags) for consistent streaming.
    """
    inputs = tok(prompt_text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    def event_stream():
        try:
            for chunk in stream_generate(inputs):
                # Each chunk is plain text; front-end converts to formatted HTML.
                yield f"data: {chunk}\n\n"
        except GeneratorExit:
            # Client closed the connection; just stop.
            return
        except Exception as e:
            # Report errors as an SSE event so the client can surface it.
            yield f"event: error\ndata: {type(e).__name__}: {e}\n\n"
        finally:
            # Always close the stream.
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
    Build a tiny 'repeat server' chat where the model is asked to output
    exactly the provided text (no extra words). This lets us stream fixed
    messages (non-symptom / red flags) via the SAME streamer.
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
    """
    Shared TextIteratorStreamer pipeline:
      - Starts a background thread calling model.generate(...)
      - Yields decoded text chunks as they arrive
    """
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
    gen_kwargs = dict(
        **inputs,  # model inputs
        max_new_tokens=MAX_NEW_TOKENS,  # max tokens to generate
        do_sample=DO_SAMPLE,  # whether to sample
        temperature=TEMPERATURE if DO_SAMPLE else None,  # sampling temperature
        top_p=TOP_P if DO_SAMPLE else None,  # nucleus sampling
        repetition_penalty=REPETITION_PENALTY,  # repetition penalty
        eos_token_id=eos_id,  # end-of-sequence token
        pad_token_id=pad_id,  # padding token
        use_cache=True,  # whether to use cache
        streamer=streamer,  # streamer for text generation
    )

    def _worker():
        with torch.inference_mode():
            # Filter out None values (temperature/top_p when greedy)
            model.generate(**{k:v for k,v in gen_kwargs.items() if v is not None}) # type: ignore

    # Run generation in a daemon thread so Flask can stream the results
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # The streamer yields incremental strings as tokens decode.
    for new_text in streamer:
        yield new_text

def sse_response(q: str):
    """
    Central routing for SSE:
      1) (Recommended) Red flags first → always caution.
      2) Non-symptom → about-the-bot blurb.
      3) Otherwise → build grounded prompt and stream a full answer.
    NOTE: If you want one-word inputs like "fever" to trigger red flags,
          keep the 'red' check BEFORE looks_like_symptom().
    """

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

    # Keep grounding compact: only pass up to 6 moves to the model
    g = {"moves": (g.get("moves") or [])[:6]} if g else {}
    grounding = json.dumps(g, ensure_ascii=False)

    prompt = build_prompt(q, grounding)
    return stream_model_prompt(prompt)

# --------------------------
# Flask app + routes
# --------------------------
app = Flask(__name__, template_folder='.')

@app.get("/events")
def events():
    """
    SSE endpoint. Front-end calls: new EventSource('/events?q=...').
    """
    q = request.args.get("q", "")
    return sse_response(q)

@app.route("/", methods=["GET", "POST"])
def root():
    """
    Main page:
      - GET  -> render the form
      - POST -> render the page with the user's last query (non-streaming fallback not needed here)
    """
    if request.method == "POST":
        q = request.form.get("q", "")
        return render_template("index.html", q=q)
    return render_template("index.html", q=None)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)