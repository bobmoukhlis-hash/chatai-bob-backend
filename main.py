# ============================================================
# ChatAI Bob Backend V2
# FastAPI + Groq + Hugging Face OCR
#
# Compatibile con:
#   POST /chat
#   POST /clear
#   POST /ocr_photo
#   GET  /health
#
# MODEL consigliato:
#   openai/gpt-oss-20b
# ============================================================

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


# ============================================================
# CONFIG
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()

# Manteniamo il modello gratuito che stai usando
MODEL = os.getenv(
    "MODEL",
    "openai/gpt-oss-20b"
).strip()

SQLITE_PATH = os.getenv(
    "SQLITE_PATH",
    "data.sqlite3"
).strip()

HF_VISION_MODEL = os.getenv(
    "HF_VISION_MODEL",
    "Salesforce/blip-image-captioning-large"
).strip()

HF_OCR_MODEL = os.getenv(
    "HF_OCR_MODEL",
    "microsoft/trocr-base-printed"
).strip()

HF_TIMEOUT = int(
    (os.getenv("HF_TIMEOUT", "60") or "60").strip()
)


# ============================================================
# SYSTEM PROMPTS
# ============================================================

SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente AI professionale, amichevole e utile.

Rispondi in italiano quando l'utente scrive in italiano.
Puoi aiutare normalmente con:

- programmazione
- sviluppo di siti web
- sviluppo di app
- Android
- Python
- JavaScript
- HTML e CSS
- API e backend
- database
- blockchain
- criptovalute
- token
- smart contract
- wallet
- Web3
- app di mining
- simulatori di mining
- sistemi di ricompense
- dashboard crypto
- progettazione di applicazioni blockchain

Quando l'utente chiede di creare un'app di mining crypto,
puoi aiutare con architettura, codice, database, API, interfaccia,
calcolo delle ricompense, sicurezza, anti-cheat e simulazione.

Non inventare informazioni tecniche.
Se una cosa non è certa, dichiaralo chiaramente.

Dai risposte pratiche e comprensibili.
Quando serve codice, fornisci codice completo e utilizzabile.

Non rifiutare una normale domanda tecnica solo perché contiene
le parole crypto, blockchain, token o mining.
Valuta il contenuto reale della richiesta.
""".strip()


VISION_PROMPT = """
Sei ChatAI Bob e devi aiutare l'utente a capire una FOTO
partendo dalla descrizione disponibile.

Non inventare dettagli che non sono visibili o non sono presenti
nella descrizione.
Se qualcosa non è leggibile o non è sicuro, dichiaralo chiaramente.
""".strip()


OCR_PROMPT = """
Sei ChatAI Bob e devi aiutare l'utente a capire il TESTO
letto da una fotografia tramite OCR.

Spiega il testo in modo chiaro e semplice.
Puoi tradurlo, riassumerlo o spiegare cosa significa.

Se il testo OCR è incompleto, confuso o contiene errori,
dillo chiaramente e non inventare le parti mancanti.
""".strip()


# ============================================================
# CLIENTS
# ============================================================

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)

HF_HEADERS = (
    {"Authorization": f"Bearer {HF_API_KEY}"}
    if HF_API_KEY
    else {}
)


# ============================================================
# DATABASE SQLITE
# ============================================================

def now_ts() -> int:
    return int(time.time())


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        SQLITE_PATH,
        check_same_thread=False
    )

    conn.execute(
        "PRAGMA journal_mode=WAL;"
    )

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


def save_msg(
    client_id: str,
    role: str,
    content: str
) -> None:

    DB.execute(
        """
        INSERT INTO convo_messages
        (client_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            client_id,
            role,
            content,
            now_ts()
        )
    )

    DB.commit()


def load_history(
    client_id: str,
    limit: int = 12
) -> List[Dict[str, str]]:

    rows = DB.execute(
        """
        SELECT role, content
        FROM convo_messages
        WHERE client_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            client_id,
            limit
        )
    ).fetchall()

    rows = list(rows)[::-1]

    out: List[Dict[str, str]] = []

    for role, content in rows:

        out.append(
            {
                "role": (
                    "user"
                    if role == "user"
                    else "assistant"
                ),
                "content": str(content)
            }
        )

    return out


def clear_history(
    client_id: str
) -> None:

    DB.execute(
        """
        DELETE FROM convo_messages
        WHERE client_id=?
        """,
        (client_id,)
    )

    DB.commit()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ChatAI Bob Backend",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:

    return {
        "status": "ok",
        "groq": (
            "ok"
            if bool(GROQ_API_KEY)
            else "missing"
        ),
        "hf": (
            "ok"
            if bool(HF_API_KEY)
            else "missing"
        ),
        "model": MODEL,
        "vision_model": HF_VISION_MODEL,
        "ocr_model": HF_OCR_MODEL,
        "version": "2.0.0"
    }


# ============================================================
# CHAT REQUEST
# ============================================================

class ChatReq(BaseModel):
    message: str
    client_id: str


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    req: ChatReq
) -> Dict[str, str]:

    # --------------------------------------------------------
    # Controllo Groq
    # --------------------------------------------------------

    if not groq_client:

        return {
            "text":
            "Servizio AI non disponibile al momento."
        }


    # --------------------------------------------------------
    # Dati utente
    # --------------------------------------------------------

    client_id = (
        (req.client_id or "").strip()
        or "client_anon"
    )

    user_text = (
        (req.message or "").strip()
    )


    if not user_text:

        return {
            "text":
            "Scrivi un messaggio e rispondo subito."
        }


    # --------------------------------------------------------
    # Cronologia
    # --------------------------------------------------------

    history = load_history(
        client_id,
        limit=12
    )


    # --------------------------------------------------------
    # Messaggi
    # --------------------------------------------------------

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_text
        }
    )


    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print(
        f"CHATAI BOB V2 | MODEL={MODEL}"
    )

    print(
        f"CHATAI BOB V2 | CLIENT={client_id}"
    )

    print(
        f"CHATAI BOB V2 | MESSAGE={user_text[:300]}"
    )


    # --------------------------------------------------------
    # GROQ
    #
    # IMPORTANTE:
    # Non definiamo tool/function.
    # Non definiamo tool_choice.
    # --------------------------------------------------------

    try:

        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )


        # ----------------------------------------------------
        # Risposta
        # ----------------------------------------------------

        reply = (
            res.choices[0].message.content
            or ""
        ).strip()


        if not reply:

            reply = (
                "Non riesco a rispondere "
                "in questo momento."
            )


        # ----------------------------------------------------
        # LOG RISPOSTA
        # ----------------------------------------------------

        print(
            f"CHATAI BOB V2 | REPLY="
            f"{reply[:500]}"
        )


        # ----------------------------------------------------
        # SALVATAGGIO
        # ----------------------------------------------------

        save_msg(
            client_id,
            "user",
            user_text
        )

        save_msg(
            client_id,
            "assistant",
            reply
        )


        return {
            "text": reply
        }


    except Exception as e:

        # ----------------------------------------------------
        # LOG COMPLETO SOLO SERVER
        # ----------------------------------------------------

        print(
            "ERRORE GROQ V2: "
            f"{type(e).__name__}: {e}"
        )


        # ----------------------------------------------------
        # RISPOSTA PUBBLICA
        # ----------------------------------------------------

        return {
            "text":
            "Errore temporaneo del servizio AI. "
            "Riprova tra poco."
        }


# ============================================================
# CLEAR HISTORY
# ============================================================

class ClearReq(BaseModel):
    client_id: str


@app.post("/clear")
def clear(
    req: ClearReq
) -> Dict[str, bool]:

    client_id = (
        (req.client_id or "").strip()
        or "client_anon"
    )

    clear_history(
        client_id
    )

    return {
        "ok": True
    }


# ============================================================
# HUGGING FACE OCR
# ============================================================

def hf_ocr_image(
    image_bytes: bytes
) -> Tuple[Optional[str], Optional[str]]:

    if not HF_API_KEY:

        return (
            None,
            "Servizio OCR non disponibile."
        )


    try:

        response = requests.post(
            (
                "https://api-inference.huggingface.co/"
                f"models/{HF_OCR_MODEL}"
            ),
            headers=HF_HEADERS,
            files={
                "file": (
                    "image.png",
                    image_bytes,
                    "application/octet-stream"
                )
            },
            timeout=HF_TIMEOUT
        )


    except Exception as e:

        print(
            "ERRORE HF OCR RETE: "
            f"{type(e).__name__}: {e}"
        )

        return (
            None,
            "Errore di rete durante OCR."
        )


    # --------------------------------------------------------
    # Status HTTP
    # --------------------------------------------------------

    if response.status_code != 200:

        print(
            "ERRORE HF OCR HTTP: "
            f"{response.status_code}"
        )

        print(
            response.text[:500]
        )

        return (
            None,
            "OCR non disponibile al momento."
        )


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except Exception:

        return (
            None,
            "Risposta OCR non valida."
        )


    # --------------------------------------------------------
    # Estrazione testo
    # --------------------------------------------------------

    text = ""


    if isinstance(data, dict):

        text = str(
            data.get(
                "text",
                ""
            )
        ).strip()


    elif isinstance(data, list):

        parts = []

        for item in data:

            if isinstance(item, dict):

                value = item.get(
                    "text",
                    ""
                )

                if value:

                    parts.append(
                        str(value)
                    )

        text = " ".join(
            parts
        ).strip()


    if not text:

        return (
            None,
            "Non riesco a leggere il testo nella foto."
        )


    return (
        text,
        None
    )


# ============================================================
# OCR PHOTO
# ============================================================

@app.post("/ocr_photo")
async def ocr_photo(
    file: UploadFile = File(...),
    question: str = Form(""),
    client_id: str = Form("client_anon")
):

    # --------------------------------------------------------
    # Controllo AI
    # --------------------------------------------------------

    if not groq_client:

        return {
            "text":
            "Servizio AI non disponibile al momento."
        }


    # --------------------------------------------------------
    # Leggo file una sola volta
    # --------------------------------------------------------

    try:

        img_bytes = await file.read()

    except Exception as e:

        print(
            "ERRORE LETTURA FOTO: "
            f"{type(e).__name__}: {e}"
        )

        return {
            "text":
            "Non riesco a leggere la foto."
        }


    if not img_bytes:

        return {
            "text":
            "File vuoto."
        }


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    ocr_text, err = hf_ocr_image(
        img_bytes
    )


    if err:

        return {
            "text": err
        }


    # --------------------------------------------------------
    # Dati
    # --------------------------------------------------------

    client_id = (
        (client_id or "").strip()
        or "client_anon"
    )

    user_question = (
        (question or "").strip()
        or "Cosa c'è scritto?"
    )


    # --------------------------------------------------------
    # Prompt OCR
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": OCR_PROMPT
        },

        {
            "role": "user",
            "content":
            (
                "TESTO OCR:\n"
                f"{ocr_text}\n\n"
                "DOMANDA UTENTE:\n"
                f"{user_question}"
            )
        }

    ]


    # --------------------------------------------------------
    # Groq OCR
    # --------------------------------------------------------

    try:

        res = groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=600
        )


        reply = (
            res.choices[0].message.content
            or ""
        ).strip()


        if not reply:

            reply = (
                "Non riesco a spiegare "
                "il testo al momento."
            )


        # ----------------------------------------------------
        # Salvataggio storico
        # ----------------------------------------------------

        save_msg(
            client_id,
            "user",
            f"[OCR] {user_question}"
        )

        save_msg(
            client_id,
            "assistant",
            reply
        )


        print(
            "CHATAI BOB V2 | OCR OK"
        )


        return {
            "text": reply
        }


    except Exception as e:

        print(
            "ERRORE OCR GROQ V2: "
            f"{type(e).__name__}: {e}"
        )


        return {
            "text":
            "Errore durante l'analisi OCR. "
            "Riprova tra poco."
        }


# ============================================================
# STARTUP INFO
# ============================================================

@app.on_event("startup")
def startup_event():

    print(
        "=========================================="
    )

    print(
        "      ChatAI Bob Backend V2 AVVIATO"
    )

    print(
        f"      MODEL: {MODEL}"
    )

    print(
        "      Crypto/Blockchain support: ON"
    )

    print(
        "      Mining app support: ON"
    )

    print(
        f"      Groq API: "
        f"{'OK' if GROQ_API_KEY else 'MISSING'}"
    )

    print(
        f"      HuggingFace API: "
        f"{'OK' if HF_API_KEY else 'MISSING'}"
    )

    print(
        "=========================================="
    )
