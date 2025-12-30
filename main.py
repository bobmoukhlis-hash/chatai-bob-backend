# main.py
from __future__ import annotations

import io
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

HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "60").strip() or "60")

# =========================
# SYSTEM PROMPT (QUALITÀ “COME ME”)
# =========================
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """
Sei ChatAI Bob, un assistente di intelligenza artificiale di livello professionale, affidabile e molto intelligente.

COMPORTAMENTO GENERALE:
- Rispondi come un esperto umano reale
- Sii chiaro, naturale, educato e molto dettagliato
- Non essere freddo né robotico
- Mantieni sempre un tono professionale ma amichevole

LINGUA:
- Rileva automaticamente la lingua dell’utente
- Rispondi SEMPRE nella stessa lingua dell’utente
- Supporti italiano, inglese, francese, spagnolo e altre lingue comuni

QUALITÀ DELLE RISPOSTE:
- Se la domanda è semplice → risposta semplice e chiara
- Se la domanda è complessa → spiega passo dopo passo
- Se la domanda è vaga → chiedi chiarimenti intelligenti
- Usa esempi concreti quando aiutano a capire meglio

CODICE E TECNICA:
- Se l’utente chiede codice, fornisci codice completo, funzionante e ben strutturato
- Spiega il codice solo se utile, senza essere prolisso
- Non fornire codice incompleto

ONESTÀ:
- Non inventare informazioni
- Se non sei sicuro di qualcosa, dillo chiaramente e proponi alternative
- Non fare supposizioni non richieste

REGOLE IMPORTANTI:
- Non parlare mai di modelli, API, provider o costi
- Non menzionare limiti tecnici
- Non usare emoji in modo eccessivo

OBIETTIVO:
Aiutare l’utente nel miglior modo possibile, come farebbe un vero esperto umano di alto livello.
""".strip(),
).strip()

VISION_PROMPT = os.getenv(
    "VISION_PROMPT",
    """
Sei ChatAI Bob. Devi aiutare l’utente a capire una FOTO.
Ti verrà dato:
- una DESCRIZIONE dell’immagine (caption)
- un’eventuale DOMANDA dell’utente

REGOLE:
- Rispondi nella lingua dell’utente.
- Se la domanda è presente: rispondi direttamente e in modo pratico.
- Se la domanda è vuota: descrivi l’immagine in modo utile e ordinato.
- Se la caption è troppo generica: dillo con onestà e suggerisci cosa chiedere o che foto caricare.
- Non inventare dettagli non supportati dalla caption.
""".strip(),
).strip()

# =========================
# CLIENTS
# =========================
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# =========================
# DB (SQLite) — Memoria per client_id
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
        r = "user" if role == "user" else "assistant"
        out.append({"role": r, "content": str(content)})
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
    allow_origins=["*"],  # ok per GitHub Pages / WebView
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

    history = load_history(client_id, limit=12)

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=900,
        )
        reply = (res.choices[0].message.content or "").strip() or "Non riesco a rispondere in questo momento."

        save_msg(client_id, "user", user_text)
        save_msg(client_id, "assistant", reply)

        return {"text": reply}
    except Exception:
        return {"text": "Servizio non disponibile. Riprova tra poco."}


# =========================
# CLEAR MEMORY (opzionale)
# =========================
class ClearReq(BaseModel):
    client_id: str


@app.post("/clear")
def clear(req: ClearReq) -> Dict[str, Any]:
    client_id = (req.client_id or "").strip() or "client_anon"
    clear_history(client_id)
    return {"ok": True}


# =========================
# ANALISI FOTO (HuggingFace caption + Groq risposta)
# =========================
def hf_caption_image(image_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    if not HF_API_KEY:
        return None, "Servizio non disponibile al momento."

    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_VISION_MODEL}",
            headers=HF_HEADERS,
            files={"file": ("image.jpg", image_bytes, "application/octet-stream")},
            timeout=HF_TIMEOUT,
        )
    except Exception:
        return None, "Errore rete durante l’analisi."

    if r.status_code != 200:
        return None, "Analisi immagine non disponibile al momento."

    try:
        data = r.json()
    except Exception:
        return None, "Analisi immagine non disponibile al momento."

    caption = ""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        caption = str(data[0].get("generated_text", "")).strip()

    if not caption:
        return None, "Non riesco a leggere bene questa immagine."

    return caption, None


@app.post("/analyze_photo")
async def analyze_photo(
    file: UploadFile = File(...),
    question: str = Form(""),
    client_id: str = Form("client_anon"),
) -> Dict[str, str]:
    if not groq_client:
        return {"text": "Servizio non disponibile al momento."}

    img_bytes = await file.read()
    if not img_bytes:
        return {"text": "File vuoto. Riprova con un’altra foto."}

    caption, err = hf_caption_image(img_bytes)
    if err:
        return {"text": err}

    q = (question or "").strip()
    client_id = (client_id or "").strip() or "client_anon"

    user_block = f"DESCRIZIONE IMMAGINE: {caption}"
    if q:
        user_block += f"\nDOMANDA UTENTE: {q}"

    messages = [
        {"role": "system", "content": VISION_PROMPT},
        {"role": "user", "content": user_block},
    ]

    try:
        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=700,
        )
        reply = (res.choices[0].message.content or "").strip() or "Non riesco a rispondere in questo momento."

        save_msg(client_id, "user", f"[FOTO] {q or 'Analizza immagine'}")
        save_msg(client_id, "assistant", reply)

        return {"text": reply}
    except Exception:
        return {"text": "Errore durante l’analisi. Riprova tra poco."}
