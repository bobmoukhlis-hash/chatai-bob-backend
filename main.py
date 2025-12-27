import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("MODEL")

SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente AI avanzato e professionale.
Rispondi SEMPRE in italiano.
Non fare domande.
Inizia subito a creare ciò che l'utente chiede.
"""

# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "env_model": os.getenv("MODEL")
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return {"text": "❌ GROQ_API_KEY non configurata."}

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ],
        "temperature": 0.8,
        "max_tokens": 2000
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
