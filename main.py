# ============================================================
# ChatAI Bob Backend V2.1
# FastAPI + Groq + Hugging Face OCR
#
# Compatibile con:
#   POST /chat
#   POST /clear
#   POST /ocr_photo
#   GET  /health
#
# MODEL:
#   openai/gpt-oss-20b
#
# FIX:
#   Riduzione cronologia per evitare errore Groq 413 / TPM
# ============================================================

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
).strip()

HF_API_KEY = os.getenv(
    "HF_API_KEY",
    ""
).strip()
# ============================================================
# LUMA API
# ============================================================

LUMA_API_KEY = os.getenv(
    "LUMA_API_KEY",
    ""
).strip()

LUMA_API_URL = (
    "https://api.lumalabs.ai/"
    "dream-machine/v1/generations"
)

# ============================================================
# SUPABASE CREDITS
# ============================================================

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
).strip()

SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    ""
).strip()

SUPABASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json"
}


# ============================================================
# MODEL
# ============================================================

MODEL = os.getenv(
    "MODEL",
    "openai/gpt-oss-20b"
).strip()


# ============================================================
# SQLITE
# ============================================================

SQLITE_PATH = os.getenv(
    "SQLITE_PATH",
    "data.sqlite3"
).strip()


# ============================================================
# HUGGING FACE
# ============================================================

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
# LIMITI CHAT
# ============================================================

# Manteniamo pochissima cronologia per restare
# molto sotto il limite TPM di Groq.

MAX_HISTORY_MESSAGES = 4
MAX_HISTORY_CHARS = 1200
MAX_USER_CHARS = 4000
MAX_SYSTEM_CHARS = 1800
MAX_REPLY_TOKENS = 500


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente AI professionale, amichevole e utile.

Rispondi nella lingua usata dall'utente.

Aiuta con programmazione, siti web, app, Android, Python,
JavaScript, HTML, CSS, API, backend, database, blockchain,
crypto, token, smart contract, wallet, Web3 e simulatori di mining.

Dai risposte pratiche e comprensibili.
Quando serve codice, fornisci codice completo e utilizzabile.

Non inventare informazioni tecniche.
Se non sei sicuro, dichiaralo chiaramente.

Non rifiutare una normale domanda tecnica solo perché contiene
parole come crypto, blockchain, token o mining.
Valuta il contenuto reale della richiesta.
""".strip()


VISION_PROMPT = """
Sei ChatAI Bob.
Aiuta l'utente a capire una foto partendo dalla descrizione disponibile.
Non inventare dettagli non visibili.
Se qualcosa non è sicuro o leggibile, dichiaralo.
""".strip()


OCR_PROMPT = """
Sei ChatAI Bob.
Aiuta l'utente a capire il testo letto da una fotografia tramite OCR.

Puoi tradurlo, riassumerlo o spiegarlo.

Se il testo OCR è incompleto o contiene errori,
dillo chiaramente e non inventare le parti mancanti.
""".strip()


# ============================================================
# CLIENT GROQ
# ============================================================

groq_client = (
    Groq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)


# ============================================================
# HUGGING FACE
# ============================================================

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


# ============================================================
# SALVA MESSAGGIO
# ============================================================

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


# ============================================================
# CARICA CRONOLOGIA RIDOTTA
# ============================================================

def load_history(
    client_id: str,
    limit: int = MAX_HISTORY_MESSAGES
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

        clean_content = str(
            content or ""
        ).strip()

        # Limita la dimensione di ogni messaggio storico.
        if len(clean_content) > MAX_HISTORY_CHARS:

            clean_content = (
                clean_content[:MAX_HISTORY_CHARS]
                + "\n[contenuto precedente abbreviato]"
            )

        if not clean_content:
            continue

        out.append(
            {
                "role": (
                    "user"
                    if role == "user"
                    else "assistant"
                ),
                "content": clean_content
            }
        )

    return out


# ============================================================
# CANCELLA CRONOLOGIA
# ============================================================

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
# SUPABASE CREDITS FUNCTIONS
# ============================================================

def supabase_get_credits(
    client_id: str
) -> int:

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return 0

    try:

        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/bob_credits",
            headers=SUPABASE_HEADERS,
            params={
                "client_id": f"eq.{client_id}",
                "select": "credits"
            },
            timeout=10
        )

        if response.status_code != 200:
            print(
                "Supabase get credits error:",
                response.status_code,
                response.text
            )
            return 0

        data = response.json()

        if not data:
            return 0

        return int(
            data[0].get("credits", 0)
        )

    except Exception as e:

        print(
            "Supabase get credits error:",
            e
        )

        return 0


def supabase_create_client(
    client_id: str
) -> int:

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return 0

    try:

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/bob_credits",
            headers={
                **SUPABASE_HEADERS,
                "Prefer": "return=representation"
            },
            json={
                "client_id": client_id,
                "credits": 100
            },
            timeout=10
        )

        print(
            "SUPABASE CREATE:",
            response.status_code,
            response.text
        )

        if response.status_code not in (200, 201):
            return 0

        data = response.json()

        if not data:
            return 0

        return int(
            data[0].get("credits", 100)
        )

    except Exception as e:

        print(
            "Supabase create client error:",
            e
        )

        return 0


def supabase_use_credits(
    client_id: str,
    amount: int
):

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False, 0

    try:

        current = supabase_get_credits(
            client_id
        )

        if current < amount:
            return False, current

        new_balance = current - amount

        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/bob_credits",
            headers={
                **SUPABASE_HEADERS,
                "Prefer": "return=representation"
            },
            params={
                "client_id": f"eq.{client_id}"
            },
            json={
                "credits": new_balance
            },
            timeout=10
        )

        if response.status_code not in (200, 204):

            print(
                "Supabase use credits error:",
                response.status_code,
                response.text
            )

            return False, current

        return True, new_balance

    except Exception as e:

        print(
            "Supabase use credits error:",
            e
        )

        return False, 0


def supabase_add_credits(
    client_id: str,
    amount: int
):

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False, 0

    try:

        current = supabase_get_credits(
            client_id
        )

        new_balance = current + amount

        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/bob_credits",
            headers={
                **SUPABASE_HEADERS,
                "Prefer": "return=representation"
            },
            params={
                "client_id": f"eq.{client_id}"
            },
            json={
                "credits": new_balance
            },
            timeout=10
        )

        if response.status_code not in (200, 204):

            print(
                "Supabase add credits error:",
                response.status_code,
                response.text
            )

            return False, current

        return True, new_balance

    except Exception as e:

        print(
            "Supabase add credits error:",
            e
        )

        return False, 0


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ChatAI Bob Backend",
    version="2.1.0"
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
        "version": "2.1.0",
        "history_limit": MAX_HISTORY_MESSAGES,
        "max_reply_tokens": MAX_REPLY_TOKENS
    }
   # ============================================================
# LUMA TEST - AGENTS API
# ============================================================

@app.get("/luma-test")
def luma_test() -> Dict[str, Any]:

    luma_key = os.getenv(
        "LUMA_AGENTS_API_KEY",
        ""
    ).strip()

    if not luma_key:
        return {
            "ok": False,
            "luma": "missing",
            "detail": "LUMA_AGENTS_API_KEY non configurata"
        }

    try:

        response = requests.post(
            "https://agents.lumalabs.ai/v1/generations",
            headers={
                "Authorization": f"Bearer {luma_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json={},
            timeout=15
        )

        print(
            "LUMA AGENTS TEST:",
            response.status_code,
            response.text[:500]
        )

        return {
            "ok": response.status_code != 401
            and response.status_code != 403,

            "status": response.status_code,

            "luma": (
                "authenticated"
                if response.status_code != 401
                and response.status_code != 403
                else "not_authenticated"
            ),

            "detail": response.text[:500]
        }

    except Exception as e:

        print(
            "LUMA AGENTS TEST ERROR:",
            type(e).__name__,
            e
        )

        return {
            "ok": False,
            "luma": "connection_error",
            "detail": str(e)
        }
# ============================================================
# CREDITS API
# ============================================================

class CreditsReq(BaseModel):
    client_id: str


class CreditsChangeReq(BaseModel):
    client_id: str
    amount: int


@app.post("/credits")
def credits(
    req: CreditsReq
) -> Dict[str, Any]:

    client_id = (
        (req.client_id or "").strip()
        or "client_anon"
    )

    current = supabase_get_credits(
        client_id
    )

    if current == 0:

        current = supabase_create_client(
            client_id
        )

    return {
        "ok": True,
        "client_id": client_id,
        "credits": current
    }


@app.post("/credits/use")
def credits_use(
    req: CreditsChangeReq
) -> Dict[str, Any]:

    client_id = (
        (req.client_id or "").strip()
        or "client_anon"
    )

    amount = int(req.amount)

    if amount <= 0:

        raise HTTPException(
            status_code=400,
            detail="Importo crediti non valido"
        )

    success, balance = supabase_use_credits(
        client_id,
        amount
    )

    if not success:

        raise HTTPException(
            status_code=402,
            detail="Crediti insufficienti"
        )

    return {
        "ok": True,
        "credits": balance
    }


@app.post("/credits/add")
def credits_add(
    req: CreditsChangeReq
) -> Dict[str, Any]:

    client_id = (
        (req.client_id or "").strip()
        or "client_anon"
    )

    amount = int(req.amount)

    if amount <= 0:

        raise HTTPException(
            status_code=400,
            detail="Importo crediti non valido"
        )

    success, balance = supabase_add_credits(
        client_id,
        amount
    )

    if not success:

        raise HTTPException(
            status_code=500,
            detail="Impossibile aggiungere crediti"
        )

    return {
        "ok": True,
        "credits": balance
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
    # CONTROLLO GROQ
    # --------------------------------------------------------

    if not groq_client:

        return {
            "text":
            "Servizio AI non disponibile al momento."
        }


    # --------------------------------------------------------
    # CLIENT
    # --------------------------------------------------------

    client_id = (
        (req.client_id or "").strip()
        or "client_anon"
    )


    # --------------------------------------------------------
    # MESSAGGIO UTENTE
    # --------------------------------------------------------

    user_text = (
        (req.message or "").strip()
    )


    if not user_text:

        return {
            "text":
            "Scrivi un messaggio e rispondo subito."
        }


    # --------------------------------------------------------
    # LIMITE MESSAGGIO
    # --------------------------------------------------------

    if len(user_text) > MAX_USER_CHARS:

        user_text = (
            user_text[:MAX_USER_CHARS]
            + "\n[Messaggio abbreviato]"
        )


    # --------------------------------------------------------
    # CRONOLOGIA RIDOTTA
    # --------------------------------------------------------

    history = load_history(
        client_id,
        limit=MAX_HISTORY_MESSAGES
    )


    # --------------------------------------------------------
    # SYSTEM PROMPT LIMITATO
    # --------------------------------------------------------

    system_content = SYSTEM_PROMPT[
        :MAX_SYSTEM_CHARS
    ]


    # --------------------------------------------------------
    # MESSAGGI GROQ
    # --------------------------------------------------------

    messages: List[Dict[str, str]] = [

        {
            "role": "system",
            "content": system_content
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

    total_chars = sum(
        len(str(item.get("content", "")))
        for item in messages
    )


    print(
        f"CHATAI BOB V2.1 | MODEL={MODEL}"
    )

    print(
        f"CHATAI BOB V2.1 | CLIENT={client_id}"
    )

    print(
        f"CHATAI BOB V2.1 | HISTORY={len(history)}"
    )

    print(
        f"CHATAI BOB V2.1 | TOTAL_CHARS={total_chars}"
    )

    print(
        f"CHATAI BOB V2.1 | MESSAGE={user_text[:300]}"
    )


    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    try:

        res = groq_client.chat.completions.create(

            model=MODEL,

            messages=messages,

            temperature=0.7,

            max_tokens=MAX_REPLY_TOKENS

        )


        # ----------------------------------------------------
        # RISPOSTA
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
        # LOG
        # ----------------------------------------------------

        print(
            f"CHATAI BOB V2.1 | REPLY="
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


        # ----------------------------------------------------
        # RISPOSTA
        # ----------------------------------------------------

        return {
            "text": reply
        }


    except Exception as e:

        print(
            "ERRORE GROQ V2.1: "
            f"{type(e).__name__}: {e}"
        )


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
    # STATUS HTTP
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
    # TESTO
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
# =====================================================
# FEEDBACK
# =====================================================

class FeedbackRequest(BaseModel):

    client_id: str
    message: str
    feedback: str


@app.post("/feedback")
async def feedback(
    data: FeedbackRequest
):

    feedback = (
        data.feedback
        .lower()
        .strip()
    )

    if feedback not in [
        "like",
        "dislike"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Feedback non valido"
        )

    print(
        f"[FEEDBACK] "
        f"CLIENT_ID={data.client_id} "
        f"TYPE={feedback} "
        f"MESSAGE={data.message[:200]}"
    )

    return {
        "ok": True,
        "feedback": feedback
    }

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
    # CONTROLLO AI
    # --------------------------------------------------------

    if not groq_client:

        return {
            "text":
            "Servizio AI non disponibile al momento."
        }


    # --------------------------------------------------------
    # FILE
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
    # CLIENT
    # --------------------------------------------------------

    client_id = (
        (client_id or "").strip()
        or "client_anon"
    )


    user_question = (
        (question or "").strip()
        or "Cosa c'è scritto?"
    )


    # Limitiamo anche OCR per sicurezza.
    ocr_text = str(
        ocr_text or ""
    )[:4000]


    user_question = user_question[:1000]


    # --------------------------------------------------------
    # PROMPT OCR
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
    # GROQ OCR
    # --------------------------------------------------------

    try:

        res = groq_client.chat.completions.create(

            model=MODEL,

            messages=messages,

            temperature=0.3,

            max_tokens=400

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
        # SALVA STORICO
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
            "CHATAI BOB V2.1 | OCR OK"
        )


        return {
            "text": reply
        }


    except Exception as e:

        print(
            "ERRORE OCR GROQ V2.1: "
            f"{type(e).__name__}: {e}"
        )


        return {
            "text":
            "Errore durante l'analisi OCR. "
            "Riprova tra poco."
        }
# ============================================================
# GENERAZIONE + MODIFICA IMMAGINI
# ============================================================

from huggingface_hub import InferenceClient
from fastapi.responses import Response


# ============================================================
# COSTI
# ============================================================

IMAGE_COST = 15
EDIT_IMAGE_COST = 20


# ============================================================
# MODELLI
# ============================================================

HF_IMAGE_MODEL = os.getenv(
    "HF_IMAGE_MODEL",
    "black-forest-labs/FLUX.1-dev"
).strip()

HF_EDIT_IMAGE_MODEL = os.getenv(
    "HF_EDIT_IMAGE_MODEL",
    "Qwen/Qwen-Image-Edit"
).strip()


# ============================================================
# CREA IMMAGINE DA TESTO
# ============================================================

@app.post("/generate_image")
async def generate_image(
    prompt: str = Form(...),
    client_id: str = Form("client_anon")
):

    client_id = (
        (client_id or "").strip()
        or "client_anon"
    )

    prompt = (
        (prompt or "").strip()
    )

    if not prompt:

        raise HTTPException(
            status_code=400,
            detail="Inserisci una descrizione dell'immagine."
        )

    if len(prompt) > 1000:

        prompt = prompt[:1000]

    # --------------------------------------------------------
    # CREDITI
    # --------------------------------------------------------

    success, balance = supabase_use_credits(
        client_id,
        IMAGE_COST
    )

    if not success:

        raise HTTPException(
            status_code=402,
            detail="Crediti insufficienti"
        )

    try:

        print(
            f"IMAGE | CLIENT={client_id}"
        )

        print(
            f"IMAGE | MODEL={HF_IMAGE_MODEL}"
        )

        print(
            f"IMAGE | PROMPT={prompt[:300]}"
        )

        client = InferenceClient(
            api_key=HF_API_KEY
        )

        image = client.text_to_image(
            prompt=prompt,
            model=HF_IMAGE_MODEL
        )

        # ----------------------------------------------------
        # PNG
        # ----------------------------------------------------

        import io

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        image_bytes = buffer.getvalue()

        print(
            f"IMAGE | OK | BYTES={len(image_bytes)}"
        )

        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "X-Bob-Credits": str(balance)
            }
        )

    except Exception as e:

        print(
            "ERRORE GENERAZIONE IMMAGINE: "
            f"{type(e).__name__}: {e}"
        )

        # ----------------------------------------------------
        # RIMBORSO
        # ----------------------------------------------------

        refund_ok, refund_balance = (
            supabase_add_credits(
                client_id,
                IMAGE_COST
            )
        )

        print(
            f"IMAGE | REFUND="
            f"{'OK' if refund_ok else 'FAILED'}"
        )

        raise HTTPException(
            status_code=500,
            detail="Generazione immagine non riuscita."
        )


# ============================================================
# MODIFICA IMMAGINI
# ============================================================

@app.post("/edit_image")
async def edit_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    client_id: str = Form("client_anon")
):

    client_id = (
        (client_id or "").strip()
        or "client_anon"
    )

    prompt = (
        (prompt or "").strip()
    )

    if not prompt:

        raise HTTPException(
            status_code=400,
            detail="Scrivi cosa vuoi modificare nella foto."
        )

    if len(prompt) > 1000:

        prompt = prompt[:1000]

    try:

        image_bytes = await file.read()

    except Exception as e:

        print(
            "EDIT IMAGE | ERRORE LETTURA:",
            type(e).__name__,
            e
        )

        raise HTTPException(
            status_code=400,
            detail="Impossibile leggere la foto."
        )

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Il file della foto è vuoto."
        )

    if len(image_bytes) > 12 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="La foto è troppo grande. Massimo 12 MB."
        )

    # --------------------------------------------------------
    # COSTO MODIFICA
    # --------------------------------------------------------

    success, balance = supabase_use_credits(
        client_id,
        EDIT_IMAGE_COST
    )

    if not success:

        raise HTTPException(
            status_code=402,
            detail="Crediti insufficienti"
        )

    try:

        print(
            f"EDIT IMAGE | CLIENT={client_id}"
        )

        print(
            f"EDIT IMAGE | MODEL={HF_EDIT_IMAGE_MODEL}"
        )

        print(
            f"EDIT IMAGE | PROMPT={prompt[:300]}"
        )

        print(
            f"EDIT IMAGE | INPUT_BYTES={len(image_bytes)}"
        )

        client = InferenceClient(
            api_key=HF_API_KEY
        )

        image = client.image_to_image(
            image=image_bytes,
            prompt=prompt,
            model=HF_EDIT_IMAGE_MODEL
        )

        import io

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        output_bytes = (
            buffer.getvalue()
        )

        print(
            f"EDIT IMAGE | OK | BYTES={len(output_bytes)}"
        )

        return Response(
            content=output_bytes,
            media_type="image/png",
            headers={
                "X-Bob-Credits":
                    str(balance)
            }
        )

    except Exception as e:

        print(
            "ERRORE MODIFICA IMMAGINE: "
            f"{type(e).__name__}: {e}"
        )

        # ----------------------------------------------------
        # RIMBORSO AUTOMATICO
        # ----------------------------------------------------

        refund_ok, refund_balance = (
            supabase_add_credits(
                client_id,
                EDIT_IMAGE_COST
            )
        )

        print(
            f"EDIT IMAGE | REFUND="
            f"{'OK' if refund_ok else 'FAILED'}"
        )

        raise HTTPException(
            status_code=500,
            detail="Modifica immagine non riuscita."
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    print(
        "=========================================="
    )

    print(
        "      ChatAI Bob Backend V2.1 AVVIATO"
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
        f"      History: "
        f"{MAX_HISTORY_MESSAGES} messaggi"
    )

    print(
        f"      Max reply: "
        f"{MAX_REPLY_TOKENS} token"
    )

    print(
        "=========================================="
    )
