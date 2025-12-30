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

# Modello consigliato (ottimo in multilingua e qualità)
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()

# Prompt di sistema: è qui che fai la differenza (qualità “alta”)
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "Sei ChatAI Bob, un assistente AI estremamente intelligente, educato e naturale. "
        "Rispondi sempre in modo chiaro, completo e utile. "
        "Adatta automaticamente la lingua alla lingua dell’utente (multilingua). "
        "Se l’utente chiede un codice, scrivilo bene e completo. "
        "Se non sai qualcosa, dillo con onestà e proponi alternative."
    ),
).strip()

# -----------------------------
# APP
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # per GitHub Pages / app web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@app.get("/health")
def health():
    ok = bool(GROQ_API_KEY)
    return {"status": "ok" if ok else "missing"}


class ChatReq(BaseModel):
    message: str
    client_id: str


@app.post("/chat")
def chat(req: ChatReq):
    if not groq:
        return {"error": "Backend non configurato: manca GROQ_API_KEY."}

    try:
        res = groq.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message},
            ],
            temperature=0.7,  # qualità più “umana”
            max_tokens=900,   # risposte belle complete
        )
        text = res.choices[0].message.content
        return {"text": text}
    except Exception:
        return {"error": "Servizio non disponibile. Riprova tra poco."}
