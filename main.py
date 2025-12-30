from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ================= CONFIG =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()

SYSTEM_PROMPT = """
Sei ChatAI Bob.

OBIETTIVO
Fornisci risposte utili, complete e pratiche. Sii naturale, umano, diretto.

LINGUA
Rispondi SEMPRE nella lingua dell’utente. Se l’utente mescola lingue, usa quella prevalente. Se non è chiaro, scegli Italiano.

STILE
- Niente Markdown (niente ** ## ``` ` _ -).
- Testo pulito, con paragrafi e titoli in MAIUSCOLO quando serve.
- Non chiedere chiarimenti: se mancano dettagli, fai assunzioni ragionevoli e vai avanti.
- Evita frasi inutili, vai al punto.

QUALITÀ
- Dai sempre una risposta “finita”: spiegazione + passi concreti + esempi quando utile.
- Se l’utente chiede codice: consegna codice completo, pronto da copiare, con istruzioni minime.
- Se l’utente chiede una guida: fai lista step-by-step e checklist.

SICUREZZA
- Rifiuta richieste illegali o dannose.
- Se richiesta ambigua ma rischiosa: sposta su alternativa sicura.
""".strip()

groq: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se vuoi, più avanti puoi mettere solo il tuo dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "groq": "ok" if bool(GROQ_API_KEY) else "missing",
    }

class ChatReq(BaseModel):
    message: str
    client_id: str

def clean_no_markdown(text: str) -> str:
    if not text:
        return ""
    for b in ["```", "**", "__", "##", "`"]:
        text = text.replace(b, "")
    return text.strip()

@app.post("/chat")
def chat(req: ChatReq):
    msg = (req.message or "").strip()
    if not msg:
        return {"text": "Scrivi qualcosa e rispondo subito."}

    if not groq:
        return {"text": "Servizio non disponibile (chiave Groq mancante)."}

    try:
        res = groq.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        out = res.choices[0].message.content
    except Exception:
        return {"text": "Servizio non disponibile. Riprova tra poco."}

    return {"text": clean_no_markdown(out)}
