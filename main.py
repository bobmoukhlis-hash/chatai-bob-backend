# main.py
from __future__ import annotations

import base64
import io
import os
import re
import secrets
import sqlite3
import time
from typing import Optional, Tuple, List

import requests
from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from passlib.hash import bcrypt
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader

# ================= CONFIG =================
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")
DB_PATH = os.getenv("SQLITE_PATH", "data.sqlite3")
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

LIMITS_FREE = {
    "chat": 15,
    "image": 3,
    "video": 3,
    "pdf": 3,
}

def today() -> int:
    return int(time.time() // 86400)

# ================= DB =================
def db() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA foreign_keys=ON;")

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE,
        password_hash TEXT,
        premium INTEGER DEFAULT 0
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS tokens(
        token TEXT PRIMARY KEY,
        user_id INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS usage_free(
        client_id TEXT,
        day INTEGER,
        action TEXT,
        count INTEGER,
        PRIMARY KEY(client_id, day, action)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        role TEXT,
        content TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)

    c.commit()
    return c

DB = db()

# ================= AUTH =================
def new_token() -> str:
    return secrets.token_urlsafe(32)

def auth_user(authorization: Optional[str]) -> Tuple[Optional[int], bool]:
    if not authorization:
        return None, False

    m = re.match(r"Bearer\s+(.+)", authorization.strip())
    if not m:
        return None, False

    token = m.group(1)
    row = DB.execute(
        "SELECT u.id, u.premium FROM tokens t JOIN users u ON u.id=t.user_id WHERE t.token=?",
        (token,)
    ).fetchone()

    if not row:
        return None, False

    return int(row[0]), bool(row[1])

# ================= LIMITS =================
def free_can(client_id: str, action: str) -> bool:
    row = DB.execute(
        "SELECT count FROM usage_free WHERE client_id=? AND day=? AND action=?",
        (client_id, today(), action)
    ).fetchone()
    used = int(row[0]) if row else 0
    return used < int(LIMITS_FREE[action])

def free_inc(client_id: str, action: str) -> None:
    DB.execute("""
        INSERT INTO usage_free(client_id, day, action, count) VALUES(?,?,?,1)
        ON CONFLICT(client_id,day,action) DO UPDATE SET count=count+1
    """, (client_id, today(), action))
    DB.commit()

# ================= AI RULES =================
BASE_RULES = """
Sei ChatAI Bob.

REGOLE ASSOLUTE:
- Rispondi come un umano esperto, chiaro e diretto.
- NON usare markdown, simboli strani o codice incompleto.
- Se l’utente chiede codice: scrivi SEMPRE codice completo.
- Se chiede HTML/CSS/JS: fornisci file completi pronti all’uso.
- Se chiede libri o testi lunghi: scrivi capitoli interi.
- Non fare domande di chiarimento: decidi tu e procedi.
- Se l’utente chiede informazioni, includi contesto, esempi e trend attuali.
- Rispondi nella lingua dell’utente.
""".strip()

# ================= GROQ =================
def groq_chat(messages) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "Servizio non disponibile: manca GROQ_API_KEY."

    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL, "messages": messages, "temperature": 0.8},
            timeout=40
        )
    except Exception:
        return "Errore di rete verso Groq."

    if r.status_code != 200:
        return f"Errore Groq (HTTP {r.status_code})."

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return "Errore Groq: risposta non valida."

# ================= MEDIA =================
def pollinations_image(prompt: str) -> bytes:
    url = POLLINATIONS_BASE + requests.utils.quote(prompt)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def make_gif(prompts: List[str]) -> bytes:
    frames: List[Image.Image] = []
    for p in prompts:
        img_bytes = pollinations_image(p)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frames.append(img)

    out = io.BytesIO()
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=800,
        loop=0
    )
    return out.getvalue()

# ================= API MODELS =================
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

class VideoRequest(BaseModel):
    prompt: str
    client_id: str

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= AUTH ROUTES =================
@app.post("/auth/signup")
def signup(req: SignupRequest):
    email = req.email.strip().lower()
    if not email or not req.password:
        return {"ok": False, "error": "Email o password mancanti."}

    try:
        DB.execute(
            "INSERT INTO users(email,password_hash) VALUES(?,?)",
            (email, bcrypt.hash(req.password))
        )
        DB.commit()
        return {"ok": True}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Email già registrata."}
    except Exception:
        return {"ok": False, "error": "Errore interno signup."}

@app.post("/auth/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    row = DB.execute(
        "SELECT id,password_hash,premium FROM users WHERE email=?",
        (email,)
    ).fetchone()

    if not row:
        return {"ok": False, "error": "Credenziali non valide."}

    if not bcrypt.verify(req.password, row[1]):
        return {"ok": False, "error": "Credenziali non valide."}

    token = new_token()
    DB.execute("INSERT INTO tokens(token,user_id) VALUES(?,?)", (token, int(row[0])))
    DB.commit()
    return {"ok": True, "token": token, "premium": bool(row[2])}

# ================= CHAT =================
@app.post("/chat")
def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    user_id, premium = auth_user(authorization)

    if user_id is None:
        if not free_can(req.client_id, "chat"):
            return {"text": "Limite FREE raggiunto."}
        free_inc(req.client_id, "chat")

    reply = groq_chat([
        {"role": "system", "content": BASE_RULES},
        {"role": "user", "content": req.message},
    ])
    return {"text": reply}

# ================= IMAGE =================
@app.post("/image")
def image(req: ImageRequest, authorization: Optional[str] = Header(None)):
    user_id, _premium = auth_user(authorization)

    if user_id is None:
        if not free_can(req.client_id, "image"):
            return {"error": "Limite immagini FREE"}
        free_inc(req.client_id, "image")

    try:
        img = pollinations_image(req.prompt)
        return {"url": "data:image/png;base64," + base64.b64encode(img).decode()}
    except Exception:
        return {"error": "Errore generazione immagine (Pollinations offline o prompt non valido)."}

# ================= VIDEO =================
@app.post("/video")
def video(req: VideoRequest, authorization: Optional[str] = Header(None)):
    user_id, _premium = auth_user(authorization)

    if user_id is None:
        if not free_can(req.client_id, "video"):
            return {"error": "Limite video FREE"}
        free_inc(req.client_id, "video")

    try:
        prompts = [f"{req.prompt}, scene {i}" for i in range(1, 5)]
        gif = make_gif(prompts)
        return {"url": "data:image/gif;base64," + base64.b64encode(gif).decode()}
    except Exception:
        return {"error": "Errore generazione GIF (Pollinations offline o errore immagini)."}

# ================= PDF =================
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    client_id: str = Form("client"),
    authorization: Optional[str] = Header(None),
):
    user_id, _premium = auth_user(authorization)

    if user_id is None:
        if not free_can(client_id, "pdf"):
            return {"text": "Limite PDF FREE"}
        free_inc(client_id, "pdf")

    try:
        reader = PdfReader(io.BytesIO(await file.read()))
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:20])
    except Exception:
        return {"text": "Errore: PDF non leggibile."}

    reply = groq_chat([
        {"role": "system", "content": "Analizza il documento."},
        {"role": "user", "content": prompt + "\n\n" + text},
    ])
    return {"text": reply}
