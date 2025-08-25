# chat.py
import json, os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SYSTEM_PROMPT = "You are a cautious exercise assistant. You do NOT diagnose or replace medical care. Rules: First, " \
"check for red flags. If present, advise urgent or prompt in-person care and do not give exercises. If no red flags, " \
"offer 3–6 gentle, low-risk movement suggestions tailored to the described symptom. Use simple, actionable steps with " \
"reps/sets/time, plus what to avoid. Include a brief safety disclaimer at the end. If unsure, ask 1–2 clarifying " \
"questions (max) before suggesting movement. Keep responses under 180 words."

with open("data/exercises.json", "r") as f:
    EXDB = json.load(f)

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE=="mps" else torch.float32,
        device_map="auto"
    )
    return tok, model

tokenizer, model = load_model()

def build_user_context(user_text: str):
    # Simple heuristic mapping to our library (extend as needed)
    key = None
    lt = user_text.lower()
    if any(k in lt for k in ["lower back", "low back", "lumbar"]):
        key = "lower_back_pain_mild"
    elif any(k in lt for k in ["neck", "cervical", "stiff neck"]):
        key = "neck_stiffness"
    elif any(k in lt for k in ["knee", "patella"]):
        key = "knee_pain_mild"
    # Embed a small JSON context the model can reference
    context = EXDB.get(key, {})
    return key, context

def make_prompt(user_text: str):
    key, ctx = build_user_context(user_text)
    grounding = json.dumps(ctx) if ctx else "{}"
    return (
f"<|system|>\n{SYSTEM_PROMPT}\n</|system|>\n"
f"<|user|>\nUser symptom description: {user_text}\n"
f"Grounded exercise library: {grounding}\n"
f"Respond with: screening confirmation, 3–6 specific movements (sets/reps/time), things to avoid, brief disclaimer.\n</|user|>\n<|assistant|>\n"
    )

def stream_chat(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=300, temperature=0.2, top_p=0.9, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for text in streamer:
        print(text, end="", flush=True)
    print()

if __name__ == "__main__":
    print("Exercise Bot (type 'exit' to quit)")
    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in {"exit", "quit"}: sys.exit(0)
        # quick red-flag triage
        hits = [h for h in ["chest pain","shortness of breath","bladder","bowel","numbness in groin","fever","recent trauma"] if h in msg.lower()]
        if hits:
            print("Assistant: I’m noticing possible red flags (" + ", ".join(hits) + "). Please seek prompt in-person care. I can provide general posture and relaxation tips, but won’t suggest exercises until you’re cleared.")
            continue
        prompt = make_prompt(msg)
        stream_chat(prompt)
