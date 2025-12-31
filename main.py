# main.py
from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

# =========================
# CONFIG (ENV)
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()

MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()
SQLITE_PATH = os.getenv("SQLITE_PATH", "data.sqlite3").strip()

HF_VISION_MODEL = os.getenv(
    "HF_VISION_MODEL",
    "Salesforce/blip-image-captioning-large",
).strip()

HF_OCR_MODEL = os.getenv(
    "HF_OCR_MODEL",
    "microsoft/trocr-base-printed",
).strip()

HF_TIMEOUT = int((os.getenv("HF_TIMEOUT", "60") or "60").strip())

# =========================
# SYSTEM PROMPTS
# =========================
SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente professionale.
Rispondi in modo chiaro, umano e diretto.
""".strip()

VISION_PROMPT = """
Devi aiutare l’utente a capire una FOTO partendo dalla descrizione.
Non inventare dettagli.
""".strip()

OCR_PROMPT = """
Devi aiutare l’utente a capire il TESTO letto da una foto.
Se il testo è confuso, dillo chiaramente.
""".strip()

# =========================
# CLIENTS
# =========================
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# =========================
# DB (SQLite)
# =========================
def now_ts() -> int:
    return int(time.time())


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS convo_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


DB = db_connect()


def save_msg(client_id: str, role: str, content: str) -> None:
    DB.execute(
        "INSERT INTO convo_messages (client_id, role, content, created_at) VALUES (?,?,?,?)",
        (client_id, role, content, now_ts()),
    )
    DB.commit()


def load_history(client_id: str, limit: int = 12) -> List[Dict[str, str]]:
    rows = DB.execute(
        """
        SELECT role, content
        FROM convo_messages
        WHERE client_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (client_id, limit),
    ).fetchall()

    rows = list(rows)[::-1]
    out: List[Dict[str, str]] = []
    for role, content in rows:
        out.append(
            {
                "role": "user" if role == "user" else "assistant",
                "content": str(content),
            }
        )
    return out


def clear_history(client_id: str) -> None:
    DB.execute("DELETE FROM convo_messages WHERE client_id=?", (client_id,))
    DB.commit()


# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "groq": "ok" if bool(GROQ_API_KEY) else "missing",
        "hf": "ok" if bool(HF_API_KEY) else "missing",
        "model": MODEL,
        "vision_model": HF_VISION_MODEL,
        "ocr_model": HF_OCR_MODEL,
    }


# =========================
# CHAT
# =========================
class ChatReq(BaseModel):
    message: str
    client_id: str


@app.post("/chat")
def chat(req: ChatReq) -> Dict[str, str]:
    if not groq_client:
        return {"text": "Servizio non disponibile al momento."}

    client_id = (req.client_id or "").strip() or "client_anon"
    user_text = (req.message or "").strip()
    if not user_text:
        return {"text": "Scrivi un messaggio e rispondo subito."}

    history = load_history(client_id)

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )
        reply = (res.choices[0].message.content or "").strip() or "Non riesco a rispondere ora."

        save_msg(client_id, "user", user_text)
        save_msg(client_id, "assistant", reply)

        return {"text": reply}
    except Exception:
        return {"text": "Errore temporaneo. Riprova."}


# =========================
# CLEAR
# =========================
class ClearReq(BaseModel):
    client_id: str


@app.post("/clear")
def clear(req: ClearReq) -> Dict[str, bool]:
    client_id = (req.client_id or "").strip() or "client_anon"
    clear_history(client_id)
    return {"ok": True}


# =========================
# OCR (funzione)
# =========================
def hf_ocr_image(image_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    if not HF_API_KEY:
        return None, "Servizio OCR non disponibile."

    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_OCR_MODEL}",
            headers=HF_HEADERS,
            files={"file": ("image.png", image_bytes, "application/octet-stream")},
            timeout=HF_TIMEOUT,
        )
    except Exception:
        return None, "Errore di rete durante OCR."

    if r.status_code != 200:
        return None, "OCR non disponibile al momento."

    try:
        data = r.json()
    except Exception:
        return None, "Risposta OCR non valida."

    text = ""
    if isinstance(data, dict):
        text = str(data.get("text", "")).strip()

    if not text:
        return None, "Non riesco a leggere il testo nella foto."

    return text, None
# =========================
# OCR ENDPOINT (CORRETTO)
# =========================
@app.post("/ocr_photo")
async def ocr_photo(
    file: UploadFile = File(...),
    question: str = Form(""),
    client_id: str = Form("client_anon"),
):
    # 1️⃣ controllo servizio AI
    if not groq_client:
        return {"text": "Servizio non disponibile al momento."}

    # 2️⃣ leggo il file UNA SOLA VOLTA
    img_bytes = await file.read()
    if not img_bytes:
        return {"text": "File vuoto."}

    # 3️⃣ OCR HuggingFace
    ocr_text, err = hf_ocr_image(img_bytes)
    if err:
        return {"text": err}

    # 4️⃣ preparo dati utente
    client_id = (client_id or "").strip() or "client_anon"
    user_question = (question or "").strip() or "Cosa c’è scritto?"

    # 5️⃣ prompt per Groq
    messages = [
        {"role": "system", "content": OCR_PROMPT},
        {
            "role": "user",
            "content": f"TESTO OCR:\n{ocr_text}\n\nDOMANDA UTENTE:\n{user_question}",
        },
    ]

    # 6️⃣ chiamata Groq
    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )

        reply = (res.choices[0].message.content or "").strip()
        if not reply:
            reply = "Non riesco a spiegare il testo al momento."

        # 7️⃣ salvo storico
        save_msg(client_id, "user", f"[OCR] {user_question}")
        save_msg(client_id, "assistant", reply)

        # 8️⃣ risposta finale
        return {"text": reply}

    except Exception:
        return {"text": "Errore durante l’analisi OCR. Riprova tra poco."}
