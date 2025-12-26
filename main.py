from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """
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
- Comportati come un autore esperto

ESEMPIO OBBLIGATORIO:
Utente: "Scrivi un libro"
Risposta: inizi direttamente dal titolo e Capitolo 1, SENZA domande.
"""

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

class ChatRequest(BaseModel):
    message: str

# ================= ROUTES =================
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
        "temperature": 0.9,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        return {"text": reply}
    except Exception as e:
        return {"text": f"⚠️ Errore Groq: {e}"}
