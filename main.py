# main.py
from __future__ import annotations

import base64
import io
import os
import re
import secrets
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from passlib.hash import bcrypt
from PIL import Image
from pydantic import BaseModel


# =============================
# CONFIG (ENV on Render)
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()

# HF models (change in Render if you want)
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/sdxl-turbo").strip()
HF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "Salesforce/blip-image-captioning-large").strip()

DB_PATH = os.getenv("SQLITE_PATH", "data.sqlite3").strip()

# FREE limits per client_id/day
LIMITS_FREE = {
    "chat": 15,
    "image": 3,
    "video": 3,
    "photo": 3,  # analyze_photo
}

# Premium: unlimited by default (you can still keep caps if you want)
PREMIUM_UNLIMITED = True


# =============================
# HELPERS
# =============================
def now_ts() -> int:
    return int(time.time())


def today_day() -> int:
    return int(time.time() // 86400)


def clean_text(text: str) -> str:
    if not text:
        return ""
    # remove common markdown artifacts that broke your UI before
    for s in ["```", "###", "##", "**", "__", "`", "---"]:
        text = text.replace(s, "")
    return text.strip()


def style_prompt(style: str) -> str:
    s = (style or "").lower().strip()
    if s == "anime":
        return "anime style, sharp lines, vibrant colors"
    if s == "realistic":
        return "photorealistic, ultra detailed, cinematic lighting"
    return "3D cartoon, pixar style, soft lighting"


def hf_headers() -> Dict[str, str]:
    if not HF_API_KEY:
        return {}
    return {"Authorization": f"Bearer {HF_API_KEY}"}


# =============================
# DB
# =============================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute(
        """CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            premium INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL
        )"""
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS tokens(
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            last_seen INTEGER NOT NULL
        )"""
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS usage_free(
            client_id TEXT NOT NULL,
            day INTEGER NOT NULL,
            action TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (client_id, day, action)
        )"""
    )

    conn.commit()
    return conn


DB = db()


# =============================
# AUTH
# =============================
def new_token() -> str:
    return secrets.token_urlsafe(32)


def auth_user_optional(authorization: Optional[str]) -> Tuple[Optional[int], bool]:
    """
    Returns (user_id, premium). If missing/invalid token -> (None, False).
    """
    if not authorization:
        return None, False

    m = re.match(r"Bearer\s+(.+)", authorization.strip(), re.IGNORECASE)
    if not m:
        return None, False

    token = m.group(1).strip()
    row = DB.execute(
        """
        SELECT u.id, u.premium
        FROM tokens t
        JOIN users u ON u.id=t.user_id
        WHERE t.token=?
        """,
        (token,),
    ).fetchone()

    if not row:
        return None, False

    DB.execute("UPDATE tokens SET last_seen=? WHERE token=?", (now_ts(), token))
    DB.commit()
    return int(row[0]), bool(int(row[1]) == 1)


def auth_user_required(authorization: Optional[str]) -> Tuple[int, bool]:
    uid, prem = auth_user_optional(authorization)
    if uid is None:
        raise HTTPException(status_code=401, detail="Missing/invalid token")
    return uid, prem


# =============================
# FREE LIMITS
# =============================
def free_get(client_id: str, action: str) -> int:
    row = DB.execute(
        "SELECT count FROM usage_free WHERE client_id=? AND day=? AND action=?",
        (client_id, today_day(), action),
    ).fetchone()
    return int(row[0]) if row else 0


def free_can_use(client_id: str, action: str) -> bool:
    if action not in LIMITS_FREE:
        return True
    used = free_get(client_id, action)
    return used < int(LIMITS_FREE[action])


def free_inc(client_id: str, action: str) -> None:
    if action not in LIMITS_FREE:
        return
    DB.execute(
        """
        INSERT INTO usage_free (client_id, day, action, count)
        VALUES (?,?,?,1)
        ON CONFLICT(client_id, day, action) DO UPDATE SET count=count+1
        """,
        (client_id, today_day(), action),
    )
    DB.commit()


# =============================
# GROQ (CHAT)
# =============================
BASE_RULES = """
Sei ChatAI Bob.

REGOLE:
- Rispondi in modo naturale, chiaro e diretto.
- Se l’utente chiede codice: scrivi codice completo pronto da copiare.
- Se chiede HTML/CSS/JS: fornisci file completi.
- Se chiede libri: scrivi capitoli interi.
- Se chiede trend/info: rispondi completo con esempi pratici.
- Evita markdown pesante; testo pulito.
"""

def groq_chat(user_text: str) -> str:
    if not GROQ_API_KEY:
        return "Servizio chat non disponibile (GROQ_API_KEY mancante)."

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": BASE_RULES.strip()},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.8,
        "max_tokens": 1400,
    }
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    if not r.ok:
        try:
            return f"Errore chat: {r.status_code} {r.json()}"
        except Exception:
            return f"Errore chat: {r.status_code} {r.text}"

    data = r.json()
    out = data["choices"][0]["message"]["content"]
    return clean_text(out)


# =============================
# HUGGINGFACE (IMAGE + VISION)
# =============================
def hf_image_generate(prompt: str) -> bytes:
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY mancante")

    url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
    r = requests.post(
        url,
        headers={**hf_headers(), "Accept": "image/png"},
        json={"inputs": prompt},
        timeout=120,
    )
    # Some HF models return 503 while loading
    if r.status_code == 503:
        raise RuntimeError("Modello HF in caricamento, riprova tra 10-20 secondi.")
    r.raise_for_status()
    return r.content


def hf_vision_caption(image_bytes: bytes) -> str:
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY mancante")

    url = f"https://api-inference.huggingface.co/models/{HF_VISION_MODEL}"
    r = requests.post(
        url,
        headers={**hf_headers(), "Accept": "application/json"},
        data=image_bytes,
        timeout=120,
    )
    if r.status_code == 503:
        raise RuntimeError("Modello Vision in caricamento, riprova tra 10-20 secondi.")
    r.raise_for_status()
    data = r.json()

    # BLIP often returns: [{"generated_text": "..."}]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"]).strip()

    # fallback
    if isinstance(data, dict):
        return str(data)

    return str(data)


def make_gif(frames_bytes: List[bytes], duration_ms: int = 750) -> bytes:
    frames: List[Image.Image] = []
    for b in frames_bytes:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        frames.append(img)

    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return out.getvalue()


# =============================
# API MODELS
# =============================
class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str
    client_id: str


class ImageRequest(BaseModel):
    prompt: str
    client_id: str
    style: str = "cartoon"


class VideoRequest(BaseModel):
    prompt: str
    client_id: str
    style: str = "cartoon"


# =============================
# APP
# =============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chat": bool(GROQ_API_KEY),
        "hf": bool(HF_API_KEY),
        "image_model": HF_IMAGE_MODEL,
        "vision_model": HF_VISION_MODEL,
    }


# =============================
# AUTH ROUTES (Premium)
# =============================
@app.post("/auth/signup")
def signup(req: SignupRequest):
    email = (req.email or "").strip().lower()
    pw = req.password or ""
    if not email or "@" not in email or len(pw) < 6:
        return {"ok": False, "error": "Email o password non valida (min 6 caratteri)."}

    try:
        DB.execute(
            "INSERT INTO users(email,password_hash,premium,created_at) VALUES(?,?,0,?)",
            (email, bcrypt.hash(pw), now_ts()),
        )
        DB.commit()
        return {"ok": True}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Email già registrata."}


@app.post("/auth/login")
def login(req: LoginRequest):
    email = (req.email or "").strip().lower()
    row = DB.execute("SELECT id,password_hash,premium FROM users WHERE email=?", (email,)).fetchone()
    if not row or not bcrypt.verify(req.password, row[1]):
        return {"ok": False, "error": "Credenziali errate."}

    token = new_token()
    DB.execute(
        "INSERT INTO tokens(token,user_id,created_at,last_seen) VALUES(?,?,?,?)",
        (token, int(row[0]), now_ts(), now_ts()),
    )
    DB.commit()

    return {"ok": True, "token": token, "premium": bool(int(row[2]) == 1)}


@app.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user_required(authorization)
    email = DB.execute("SELECT email FROM users WHERE id=?", (user_id,)).fetchone()[0]
    return {"ok": True, "email": email, "premium": premium}


@app.post("/admin/set_premium")
def set_premium(email: str = Form(...), key: str = Form(...)):
    if not ADMIN_KEY or key != ADMIN_KEY:
        return {"ok": False}
    DB.execute("UPDATE users SET premium=1 WHERE email=?", ((email or "").strip().lower(),))
    DB.commit()
    return {"ok": True}


# =============================
# CHAT (FREE + Premium)
# =============================
@app.post("/chat")
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    client_id = (req.client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    if user_id is None:
        if not free_can_use(client_id, "chat"):
            return {"text": "Limite FREE chat raggiunto oggi. Passa a Premium per continuare."}
        free_inc(client_id, "chat")

    reply = groq_chat(req.message or "")
    return {"text": reply}


# =============================
# IMAGE (HF)
# =============================
@app.post("/image")
def image(req: ImageRequest, authorization: Optional[str] = Header(default=None)):
    client_id = (req.client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    if user_id is None:
        if not free_can_use(client_id, "image"):
            return {"error": "Limite FREE immagini raggiunto oggi. Passa a Premium."}
        free_inc(client_id, "image")

    style = (req.style or "cartoon").strip().lower()
    final_prompt = f"{style_prompt(style)}, no text, no watermark, {req.prompt}"

    try:
        img_bytes = hf_image_generate(final_prompt)
        return {"url": "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}


# =============================
# VIDEO (GIF via HF images)
# =============================
@app.post("/video")
def video(req: VideoRequest, authorization: Optional[str] = Header(default=None)):
    client_id = (req.client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    if user_id is None:
        if not free_can_use(client_id, "video"):
            return {"error": "Limite FREE video/GIF raggiunto oggi. Passa a Premium."}
        free_inc(client_id, "video")

    style = (req.style or "cartoon").strip().lower()
    base = f"{style_prompt(style)}, same character, smooth motion, cinematic"
    prompts = [
        f"{base}, {req.prompt}, scene 1",
        f"{base}, {req.prompt}, scene 2",
        f"{base}, {req.prompt}, scene 3",
        f"{base}, {req.prompt}, scene 4",
    ]

    try:
        frames = [hf_image_generate(p) for p in prompts]
        gif_bytes = make_gif(frames, duration_ms=750)
        return {"url": "data:image/gif;base64," + base64.b64encode(gif_bytes).decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}


# =============================
# ANALYZE PHOTO (HF vision + Groq answer)
# =============================
@app.post("/analyze_photo")
async def analyze_photo(
    file: UploadFile = File(...),
    question: str = Form(""),
    client_id: str = Form("client_anon"),
    authorization: Optional[str] = Header(default=None),
):
    client_id = (client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    if user_id is None:
        if not free_can_use(client_id, "photo"):
            return {"text": "Limite FREE analisi foto raggiunto oggi. Passa a Premium."}
        free_inc(client_id, "photo")

    img_bytes = await file.read()
    if not img_bytes:
        return {"text": "File immagine vuoto."}

    try:
        caption = hf_vision_caption(img_bytes)
    except Exception as e:
        return {"text": f"Errore analisi foto: {e}"}

    q = (question or "").strip()
    if not q:
        # solo descrizione
        return {"text": clean_text(f"DESCRIZIONE FOTO:\n{caption}")}

    # risposta completa con Groq usando caption come contesto
    prompt = (
        "Rispondi alla domanda usando SOLO le info coerenti con la descrizione.\n\n"
        f"DESCRIZIONE: {caption}\n"
        f"DOMANDA: {q}\n"
        "RISPOSTA:"
    )
    answer = groq_chat(prompt)
    return {"text": clean_text(f"DESCRIZIONE FOTO:\n{caption}\n\nRISPOSTA:\n{answer}")}
