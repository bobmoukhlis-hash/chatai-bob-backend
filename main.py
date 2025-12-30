from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# -----------------------------
# CONFIG
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()

SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente di intelligenza artificiale avanzato, affidabile e professionale.

CARATTERISTICHE PRINCIPALI:
- Rispondi in modo chiaro, naturale, educato e molto dettagliato
- Adatti automaticamente la lingua a quella dell’utente
- Mantieni il contesto della conversazione
- Spiega passo dopo passo se la domanda è complessa
- Scrivi codice pulito, commentato e funzionante
- Usa esempi chiari
- Non inventare informazioni

COMPORTAMENTO:
- Amichevole ma professionale
- Niente arroganza
- Emoji solo se utili
- Mai parlare di API, modelli o limiti

OBIETTIVO:
Aiutare l’utente come farebbe un vero esperto umano.
""".strip()

# -----------------------------
# APP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# -----------------------------
# MEMORIA (RAM)
# -----------------------------
MEMORY: dict[str, list[dict]] = {}
MAX_HISTORY = 12   # numero messaggi ricordati

# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok" if GROQ_API_KEY else "missing"}

class ChatReq(BaseModel):
    message: str
    client_id: str

@app.post("/chat")
def chat(req: ChatReq):
    if not groq:
        return {"error": "Backend non configurato"}

    cid = req.client_id

    # inizializza memoria utente
    if cid not in MEMORY:
        MEMORY[cid] = []

    # aggiungi messaggio utente
    MEMORY[cid].append({"role": "user", "content": req.message})

    # limita memoria
    MEMORY[cid] = MEMORY[cid][-MAX_HISTORY:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + MEMORY[cid]

    try:
        res = groq.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=900,
        )

        answer = res.choices[0].message.content

        # salva risposta AI
        MEMORY[cid].append({"role": "assistant", "content": answer})
        MEMORY[cid] = MEMORY[cid][-MAX_HISTORY:]

        return {"text": answer}

    except Exception:
        return {"error": "Servizio non disponibile. Riprova tra poco."}
