# main.py
from __future__ import annotations

import base64, io, os, re, secrets, sqlite3, time
from typing import Optional, Tuple, Dict, List, Any

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

def now_ts(): return int(time.time())
def today(): return int(time.time() // 86400)

# ================= DB =================
def db():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;")

    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE,
        password_hash TEXT,
        premium INTEGER DEFAULT 0
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS tokens(
        token TEXT PRIMARY KEY,
        user_id INTEGER
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS usage_free(
        client_id TEXT,
        day INTEGER,
        action TEXT,
        count INTEGER,
        PRIMARY KEY(client_id, day, action)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        role TEXT,
        content TEXT
    )""")

    c.commit()
    return c

DB = db()

# ================= AUTH =================
def new_token():
    return secrets.token_urlsafe(32)

def auth_user(authorization: Optional[str]) -> Tuple[Optional[int], bool]:
    if not authorization:
        return None, False

    m = re.match(r"Bearer\s+(.+)", authorization)
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
def free_can(client_id: str, action: str):
    row = DB.execute(
        "SELECT count FROM usage_free WHERE client_id=? AND day=? AND action=?",
        (client_id, today(), action)
    ).fetchone()
    used = row[0] if row else 0
    return used < LIMITS_FREE[action]

def free_inc(client_id: str, action: str):
    DB.execute("""
        INSERT INTO usage_free VALUES(?,?,?,1)
        ON CONFLICT(client_id,day,action) DO UPDATE SET count=count+1
    """, (client_id, today(), action))
    DB.commit()

# ================= GROQ =================
def groq_chat(messages):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "Servizio non disponibile."

    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "messages": messages, "temperature": 0.8},
        timeout=40
    )
    return r.json()["choices"][0]["message"]["content"]

# ================= MEDIA =================
def pollinations_image(prompt: str) -> bytes:
    r = requests.get(POLLINATIONS_BASE + requests.utils.quote(prompt), timeout=60)
    r.raise_for_status()
    return r.content

def make_gif(prompts: List[str]) -> bytes:
    frames = []
    for p in prompts:
        img = Image.open(io.BytesIO(pollinations_image(p))).convert("RGB")
        frames.append(img)
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=800, loop=0)
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
    allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ================= AUTH ROUTES =================
@app.post("/auth/signup")
def signup(req: SignupRequest):
    DB.execute(
        "INSERT INTO users(email,password_hash) VALUES(?,?)",
        (req.email.lower(), bcrypt.hash(req.password))
    )
    DB.commit()
    return {"ok": True}

@app.post("/auth/login")
def login(req: LoginRequest):
    row = DB.execute(
        "SELECT id,password_hash,premium FROM users WHERE email=?",
        (req.email.lower(),)
    ).fetchone()
    if not row or not bcrypt.verify(req.password, row[1]):
        return {"ok": False}

    token = new_token()
    DB.execute("INSERT INTO tokens VALUES(?,?)", (token, row[0]))
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
        {"role": "system", "content": "Sei ChatAI Bob, rispondi in modo naturale."},
        {"role": "user", "content": req.message}
    ])
    return {"text": reply}

# ================= IMAGE =================
@app.post("/image")
def image(req: ImageRequest, authorization: Optional[str] = Header(None)):
    user_id, _ = auth_user(authorization)
    if user_id is None:
        if not free_can(req.client_id, "image"):
            return {"error": "Limite immagini FREE"}
        free_inc(req.client_id, "image")

    img = pollinations_image(req.prompt)
    return {"url": "data:image/png;base64," + base64.b64encode(img).decode()}

# ================= VIDEO =================
@app.post("/video")
def video(req: VideoRequest, authorization: Optional[str] = Header(None)):
    user_id, _ = auth_user(authorization)
    if user_id is None:
        if not free_can(req.client_id, "video"):
            return {"error": "Limite video FREE"}
        free_inc(req.client_id, "video")

    gif = make_gif([req.prompt + f", scene {i}" for i in range(1,4)])
    return {"url": "data:image/gif;base64," + base64.b64encode(gif).decode()}

# ================= PDF =================
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    client_id: str = Form("client"),
    authorization: Optional[str] = Header(None)
):
    user_id, _ = auth_user(authorization)
    if user_id is None:
        if not free_can(client_id, "pdf"):
            return {"text": "Limite PDF FREE"}
        free_inc(client_id, "pdf")

    reader = PdfReader(io.BytesIO(await file.read()))
    text = "\n".join(p.extract_text() or "" for p in reader.pages[:20])

    reply = groq_chat([
        {"role": "system", "content": "Analizza il documento."},
        {"role": "user", "content": prompt + "\n\n" + text}
    ])
    return {"text": reply}
