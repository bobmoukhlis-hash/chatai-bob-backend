from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente AI avanzato.
Rispondi sempre in italiano.
Scrivi risposte complete, senza chiedere chiarimenti.
Se l’utente chiede di creare qualcosa (libro, storia, codice, testo),
fallo immediatamente.
Usa titoli, paragrafi e struttura professionale.
"""
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

    payload = {
    "model": "llama3-70b-8192",
    "temperature": 0.9,
    "max_tokens": 2048,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.message}
    ] = [
    {"SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente AI professionale.

REGOLE ASSOLUTE (NON VIOLABILI):
- Rispondi SEMPRE in italiano
- NON fare domande di chiarimento
- NON chiedere dettagli aggiuntivi
- NON dire frasi come "prima di iniziare", "potresti dirmi", "vorrei sapere"
- Se l'utente chiede di scrivere un libro, una storia, un testo o codice:
  INIZIA SUBITO A SCRIVERLO
- Scegli TU genere, struttura e stile in modo professionale
- Produci risposte LUNGHE, COMPLETE e BEN STRUTTURATE
- Usa titoli, capitoli, paragrafi
- Comportati come un autore esperto, non come un assistente che chiede permesso

ESEMPIO OBBLIGATORIO:
Utente: "Scrivi un libro"
Risposta: inizi direttamente dal titolo e Capitolo 1, SENZA domande.
"""},
    {"role": "user", "content": req.message}
](
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
