from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

app = FastAPI()

# CORS per GitHub Pages / App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIG =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

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

    messages = [
    {
        "role": "system",
        "content": (
            "Sei ChatAI Bob, un assistente AI professionale. "
            "Rispondi SEMPRE in modo diretto, pratico e completo. "
            "NON fare domande inutili. "
            "Se l'utente chiede di scrivere un libro, INIZIA SUBITO a scriverlo. "
            "Usa uno stile chiaro, naturale e strutturato."
        )
    },
    {
        "role": "user",
        "content": req.message
    }
]
            {"role": "system", "content": "Rispondi in italiano, in modo chiaro e utile."},
            {"role": "user", "content": req.message}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        return {"text": reply}
    except Exception as e:
        return {"text": f"⚠️ Errore Groq: {e}"}
