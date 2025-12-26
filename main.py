from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIG =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"  # ✅ MODELLO CORRETTO

SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente AI professionale.

REGOLE ASSOLUTE:
- Rispondi SEMPRE in italiano
- NON fare domande
- NON chiedere chiarimenti
- Se l'utente chiede di scrivere un libro, INIZIA SUBITO
- Usa titoli, capitoli e struttura professionale
- Risposte lunghe, complete e ben scritte
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return {"text": "❌ GROQ_API_KEY non configurata sul server."}

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ],
        "temperature": 0.9,
        "max_tokens": 1500
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        return {"text": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"text": f"⚠️ Errore Groq: {e}"}
