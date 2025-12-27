# =========================
# REPO STRUCTURE
# =========================
# backend/
#   app.py
#   requirements.txt
#   start.sh
# frontend/
#   index.html
# android/
#   app/src/main/AndroidManifest.xml
#   app/src/main/java/com/chatai/bob/MainActivity.java
#   app/build.gradle
#   settings.gradle


# =========================
# backend/requirements.txt
# =========================
fastapi
uvicorn
requests
python-multipart
passlib[bcrypt]
pillow
PyPDF2


# =========================
# backend/start.sh
# =========================
#!/usr/bin/env bash
set -e
uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}


# =========================
# backend/app.py
# =========================
from __future__ import annotations

import base64
import io
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from passlib.hash import bcrypt
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader

# -------------------------
# CONFIG
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()
DB_PATH = os.getenv("SQLITE_PATH", "data.sqlite3").strip()

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

# FREE LIMITS
LIMITS_FREE = {
    "chat": 15,
    "voice": 15,   # per ora contatore, la voce lato browser non consuma backend
    "mic": 15,     # idem, ma lo contiamo quando il client manda "mic_used": true
    "image": 3,
    "video": 3,
    "pdf": 3,
    "other": 3,
}

# -------------------------
# DB
# -------------------------
def now_ts() -> int:
    return int(time.time())

def today_day() -> int:
    return int(time.time() // 86400)

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          email TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          premium INTEGER DEFAULT 0,
          created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tokens (
          token TEXT PRIMARY KEY,
          user_id INTEGER NOT NULL,
          created_at INTEGER NOT NULL,
          last_seen INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage (
          user_id INTEGER NOT NULL,
          day INTEGER NOT NULL,
          action TEXT NOT NULL,
          count INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (user_id, day, action)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          type TEXT NOT NULL,           -- BOOK / COACH / CODER / GENERAL
          state TEXT NOT NULL,          -- key=value;key=value
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          project_id INTEGER,
          role TEXT NOT NULL,           -- user / assistant
          content TEXT NOT NULL,
          created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn

DB = db()

def kv_dump(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in d.items():
        parts.append(f"{k}={str(v).replace(';',' ').replace('\\n',' ')}")
    return ";".join(parts)

def kv_load(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not s:
        return out
    for part in s.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out

# -------------------------
# AUTH
# -------------------------
def new_token() -> str:
    return secrets.token_urlsafe(32)

def auth_user(authorization: Optional[str]) -> Tuple[int, bool]:
    """
    Returns (user_id, premium).
    Authorization: Bearer <token>
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    m = re.match(r"Bearer\s+(.+)", authorization.strip(), re.IGNORECASE)
    if not m:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = m.group(1).strip()
    row = DB.execute(
        """
        SELECT t.user_id, u.premium
        FROM tokens t
        JOIN users u ON u.id = t.user_id
        WHERE t.token=?
        """,
        (token,),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid token")

    DB.execute("UPDATE tokens SET last_seen=? WHERE token=?", (now_ts(), token))
    DB.commit()
    return int(row[0]), bool(row[1] == 1)

def set_premium_by_email(email: str, premium: int) -> None:
    DB.execute("UPDATE users SET premium=? WHERE email=?", (premium, email.strip().lower()))
    DB.commit()

# -------------------------
# LIMITS
# -------------------------
def can_use(user_id: int, premium: bool, action: str) -> bool:
    if premium:
        return True
    if action not in LIMITS_FREE:
        action = "other"
    day = today_day()
    row = DB.execute(
        "SELECT count FROM usage WHERE user_id=? AND day=? AND action=?",
        (user_id, day, action),
    ).fetchone()
    used = int(row[0]) if row else 0
    return used < int(LIMITS_FREE[action])

def inc_use(user_id: int, action: str) -> None:
    if action not in LIMITS_FREE:
        action = "other"
    day = today_day()
    DB.execute(
        """
        INSERT INTO usage (user_id, day, action, count)
        VALUES (?,?,?,1)
        ON CONFLICT(user_id, day, action) DO UPDATE SET count=count+1
        """,
        (user_id, day, action),
    )
    DB.commit()

def usage_snapshot(user_id: int) -> Dict[str, int]:
    day = today_day()
    rows = DB.execute("SELECT action, count FROM usage WHERE user_id=? AND day=?", (user_id, day)).fetchall()
    out = {k: 0 for k in LIMITS_FREE.keys()}
    for a, c in rows:
        out[str(a)] = int(c)
    return out

# -------------------------
# AI PROMPTS (NO MARKDOWN)
# -------------------------
BASE_RULES = """
Sei ChatAI Bob.

REGOLE ASSOLUTE:
- Rispondi come un umano esperto: naturale, fluido, diretto.
- NON usare Markdown (vietato **, ##, ``` , `, _, -).
- NON fare domande di chiarimento.
- Se mancano dettagli: DECIDI TU e vai avanti.
- Output pronto da usare con TITOLI IN MAIUSCOLO e paragrafi.
- Se l’utente chiede di creare (libro, storia, codice, testo): INIZIA SUBITO.
"""

AUTHOR_PROMPT = BASE_RULES + """
MODALITÀ AUTORE:
- Se l’utente chiede un libro/romanzo: produci TITOLO + MINI-INDICE + CAPITOLO 1 completo.
- Se dice "continua": scrivi il capitolo successivo coerente.
"""

COACH_PROMPT = BASE_RULES + """
MODALITÀ COACH:
- Piano pratico con passi concreti e routine.
"""

CODER_PROMPT = BASE_RULES + """
MODALITÀ CODER:
- Se chiede codice: consegna codice completo pronto da copiare.
- Spiegazione breve (max 6 righe) e poi codice.
"""

AUTOCORE_PROMPT = BASE_RULES + """
AUTOCORE:
- Se l’utente è vago o scrive pochissimo: scegli tu l’output più utile e crealo subito.
"""

PDF_ANALYST_PROMPT = BASE_RULES + """
MODALITÀ ANALISI DOCUMENTI:
- Produci: RIASSUNTO, PUNTI CHIAVE, AZIONI, DATI IMPORTANTI.
"""

def detect_intent(user_text: str) -> str:
    t = (user_text or "").lower().strip()
    if any(x in t for x in ["libro", "romanzo", "storia", "racconto", "capitolo", "scrivi un libro"]):
        return "AUTHOR"
    if any(x in t for x in ["motivazione", "ansia", "disciplina", "abitudini", "crescita personale", "coach"]):
        return "COACH"
    if any(x in t for x in ["html", "css", "javascript", "python", "codice", "api", "app", "bug", "errore", "programma"]):
        return "CODER"
    if len(t.split()) <= 3:
        return "AUTO"
    return "GENERAL"

def pick_prompt(mode: str) -> str:
    if mode == "AUTHOR":
        return AUTHOR_PROMPT
    if mode == "COACH":
        return COACH_PROMPT
    if mode == "CODER":
        return CODER_PROMPT
    if mode == "AUTO":
        return AUTOCORE_PROMPT
    return BASE_RULES

def clean_text(text: str) -> str:
    if not text:
        return ""
    for b in ["**", "__", "##", "```", "`", "---"]:
        text = text.replace(b, "")
    text = text.replace("* ", "• ")
    return text.strip()

def language_rule(user_text: str) -> str:
    t = user_text or ""
    if re.search(r"[\u0600-\u06FF]", t):
        return "أجب باللغة العربية المغربية إن أمكن، وبأسلوب طبيعي وإنساني."
    # english quick signals
    low = t.lower()
    if any(x in low for x in ["hello", "write", "book", "plan", "business", "how to", "why", "what is"]):
        return "Reply in English, natural, human."
    # otherwise: mirror user language
    return "Rispondi nella stessa lingua usata dall’utente. Se non è chiara, scegli la lingua più probabile."

# -------------------------
# GROQ
# -------------------------
def groq_chat(messages: List[Dict[str, str]], temperature: float = 0.75, max_tokens: int = 1400) -> str:
    if not GROQ_API_KEY:
        return "ERRORE: GROQ_API_KEY non configurata sul server."

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=45)

    if not r.ok:
        try:
            return f"ERRORE GROQ HTTP {r.status_code}: {r.json()}"
        except Exception:
            return f"ERRORE GROQ HTTP {r.status_code}: {r.text}"

    data = r.json()
    return data["choices"][0]["message"]["content"]

# -------------------------
# MEDIA (IMAGE/VIDEO)
# -------------------------
def pollinations_image(prompt: str) -> bytes:
    url = POLLINATIONS_BASE + requests.utils.quote(prompt)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def make_gif(prompts: List[str], duration_ms: int = 850) -> bytes:
    frames: List[Image.Image] = []
    for p in prompts:
        img_bytes = pollinations_image(p)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frames.append(img)
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return out.getvalue()

# -------------------------
# PDF TEXT
# -------------------------
def extract_pdf_text(file_bytes: bytes, max_pages: int = 40) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts: List[str] = []
    for page in reader.pages[:max_pages]:
        txt = (page.extract_text() or "").strip()
        if txt:
            texts.append(txt)
    return "\n\n".join(texts).strip()

# -------------------------
# API MODELS
# -------------------------
class SignupRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    project_id: Optional[int] = None
    mic_used: Optional[bool] = False

class ProjectCreateRequest(BaseModel):
    name: str
    type: str  # BOOK/COACH/CODER/GENERAL

class ProjectSelectRequest(BaseModel):
    project_id: int

class ImageRequest(BaseModel):
    prompt: str

class VideoRequest(BaseModel):
    prompt: str

# -------------------------
# APP
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

# -------------------------
# AUTH ROUTES
# -------------------------
@app.post("/auth/signup")
def signup(req: SignupRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email or len(req.password) < 6:
        return {"ok": False, "error": "Email o password non valida (min 6 caratteri)."}

    try:
        DB.execute(
            "INSERT INTO users (email, password_hash, premium, created_at) VALUES (?,?,0,?)",
            (email, bcrypt.hash(req.password), now_ts()),
        )
        DB.commit()
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Email già registrata."}

    return {"ok": True}

@app.post("/auth/login")
def login(req: LoginRequest):
    email = req.email.strip().lower()
    row = DB.execute("SELECT id, password_hash, premium FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        return {"ok": False, "error": "Credenziali errate."}
    if not bcrypt.verify(req.password, row[1]):
        return {"ok": False, "error": "Credenziali errate."}

    token = new_token()
    DB.execute("INSERT INTO tokens (token, user_id, created_at, last_seen) VALUES (?,?,?,?)",
               (token, int(row[0]), now_ts(), now_ts()))
    DB.commit()

    return {"ok": True, "token": token, "premium": bool(int(row[2]) == 1), "email": email}

@app.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user(authorization)
    email = DB.execute("SELECT email FROM users WHERE id=?", (user_id,)).fetchone()[0]
    return {"ok": True, "email": email, "premium": premium, "usage": usage_snapshot(user_id), "limits": LIMITS_FREE}

# -------------------------
# PREMIUM (NO WEBHOOK) - ADMIN MANUAL
# -------------------------
@app.post("/premium/admin_set")
def admin_set_premium(email: str = Form(...), premium: int = Form(...), admin_key: str = Form(...)):
    """
    Imposta PREMIUM manualmente.
    Su Render imposta env ADMIN_KEY e usalo qui per sicurezza.
    """
    ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    set_premium_by_email(email, 1 if premium else 0)
    return {"ok": True}

# -------------------------
# PROJECTS
# -------------------------
@app.post("/projects/create")
def projects_create(req: ProjectCreateRequest, authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user(authorization)
    name = (req.name or "").strip() or "Progetto"
    ptype = (req.type or "GENERAL").strip().upper()
    if ptype not in {"BOOK", "COACH", "CODER", "GENERAL"}:
        ptype = "GENERAL"

    state: Dict[str, Any] = {}
    if ptype == "BOOK":
        state = {"chapter": 1, "style": "narrativo"}
    elif ptype == "COACH":
        state = {"plan": "base"}
    elif ptype == "CODER":
        state = {"stack": "web"}

    ts = now_ts()
    cur = DB.execute(
        "INSERT INTO projects (user_id, name, type, state, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (user_id, name, ptype, kv_dump(state), ts, ts),
    )
    DB.commit()
    return {"ok": True, "project_id": int(cur.lastrowid)}

@app.get("/projects/list")
def projects_list(authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user(authorization)
    rows = DB.execute(
        "SELECT id, name, type, state, updated_at FROM projects WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    projects = []
    for r in rows:
        projects.append({"id": int(r[0]), "name": r[1], "type": r[2], "state": kv_load(r[3]), "updated_at": int(r[4])})
    return {"ok": True, "projects": projects}

# -------------------------
# CHAT
# -------------------------
def last_messages(user_id: int, project_id: Optional[int], limit: int = 8) -> List[Dict[str, str]]:
    if project_id:
        rows = DB.execute(
            "SELECT role, content FROM messages WHERE user_id=? AND project_id=? ORDER BY id DESC LIMIT ?",
            (user_id, project_id, limit),
        ).fetchall()
    else:
        rows = DB.execute(
            "SELECT role, content FROM messages WHERE user_id=? AND project_id IS NULL ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    rows = list(rows)[::-1]
    return [{"role": "user" if r[0] == "user" else "assistant", "content": r[1]} for r in rows]

@app.post("/chat")
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user(authorization)

    # limit
    if not can_use(user_id, premium, "chat"):
        return {"text": "LIMITE GIORNALIERO CHAT RAGGIUNTO. PASSA A PREMIUM PER ILLIMITATO."}

    user_text = (req.message or "").strip()
    if not user_text:
        return {"text": "SCRIVI QUALCOSA E PARTO SUBITO."}

    if req.mic_used:
        if can_use(user_id, premium, "mic"):
            inc_use(user_id, "mic")

    project_id = req.project_id

    # project bias
    if project_id:
        pr = DB.execute("SELECT type, state FROM projects WHERE id=? AND user_id=?", (project_id, user_id)).fetchone()
        if pr:
            ptype = pr[0]
            state = kv_load(pr[1])
            # book continue logic
            if ptype == "BOOK":
                if user_text.lower() in {"continua", "vai avanti", "prosegui", "avanti", "ok"}:
                    ch = int(state.get("chapter", "1"))
                    ch += 1
                    state["chapter"] = str(ch)
                    DB.execute("UPDATE projects SET state=?, updated_at=? WHERE id=? AND user_id=?",
                               (kv_dump(state), now_ts(), project_id, user_id))
                    DB.commit()
                    user_text = f"CONTINUA IL LIBRO DAL CAPITOLO {ch} MANTENENDO TRAMA E STILE COERENTI."
                elif "inizia" in user_text.lower() and "libro" in user_text.lower():
                    state["chapter"] = "1"
                    DB.execute("UPDATE projects SET state=?, updated_at=? WHERE id=? AND user_id=?",
                               (kv_dump(state), now_ts(), project_id, user_id))
                    DB.commit()

    intent = detect_intent(user_text)
    if project_id:
        pr = DB.execute("SELECT type FROM projects WHERE id=? AND user_id=?", (project_id, user_id)).fetchone()
        if pr:
            ptype = pr[0]
            if ptype == "BOOK":
                intent = "AUTHOR"
            elif ptype == "COACH":
                intent = "COACH"
            elif ptype == "CODER":
                intent = "CODER"

    sys = pick_prompt(intent) + "\n" + language_rule(user_text)

    history = last_messages(user_id, project_id, limit=8)
    messages = [{"role": "system", "content": sys}]
    for m in history[-4:]:
        messages.append(m)
    messages.append({"role": "user", "content": user_text})

    DB.execute(
        "INSERT INTO messages (user_id, project_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (user_id, project_id, "user", user_text, now_ts()),
    )
    DB.commit()

    reply = groq_chat(messages, temperature=0.8, max_tokens=1600)
    reply = clean_text(reply)

    DB.execute(
        "INSERT INTO messages (user_id, project_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (user_id, project_id, "assistant", reply, now_ts()),
    )
    DB.commit()

    inc_use(user_id, "chat")
    return {"text": reply}

# -------------------------
# IMAGE
# -------------------------
@app.post("/image")
def image(req: ImageRequest, authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user(authorization)
    if not can_use(user_id, premium, "image"):
        return {"url": "", "error": "LIMITE IMMAGINI GIORNALIERO RAGGIUNTO. PASSA A PREMIUM."}

    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"url": ""}

    safe_prompt = f"high quality, detailed, realistic, {prompt}"
    img_bytes = pollinations_image(safe_prompt)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    inc_use(user_id, "image")
    return {"url": f"data:image/png;base64,{b64}"}

# -------------------------
# VIDEO (GIF 3 scene)
# -------------------------
@app.post("/video")
def video(req: VideoRequest, authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user(authorization)
    if not can_use(user_id, premium, "video"):
        return {"url": "", "error": "LIMITE VIDEO GIORNALIERO RAGGIUNTO. PASSA A PREMIUM."}

    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"url": ""}

    prompts = [
        f"{prompt}, scene 1, cinematic, high quality",
        f"{prompt}, scene 2, cinematic, high quality",
        f"{prompt}, scene 3, cinematic, high quality",
    ]
    gif_bytes = make_gif(prompts, duration_ms=850)
    b64 = base64.b64encode(gif_bytes).decode("utf-8")

    inc_use(user_id, "video")
    return {"url": f"data:image/gif;base64,{b64}"}

# -------------------------
# ANALYZE (PDF)
# -------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    project_id: str = Form(""),
    authorization: Optional[str] = Header(default=None),
):
    user_id, premium = auth_user(authorization)
    if not can_use(user_id, premium, "pdf"):
        return {"text": "LIMITE PDF GIORNALIERO RAGGIUNTO. PASSA A PREMIUM."}

    prompt = (prompt or "").strip()
    if not prompt:
        return {"text": "SCRIVI COSA VUOI FARE SUL FILE (RIASSUMI, ESTRAI DATI, TROVA DATE, ECC.)."}

    content = await file.read()
    filename = (file.filename or "").lower()

    if not filename.endswith(".pdf"):
        return {"text": "PER ORA SUPPORTO COMPLETO SOLO PDF. PER FOTO OCR LO AGGIUNGIAMO DOPO."}

    extracted = extract_pdf_text(content)
    if not extracted:
        return {"text": "NON RIESCO A LEGGERE TESTO DAL PDF (FORSE È SCANSIONE). OCR LO AGGIUNGIAMO DOPO."}

    sys = PDF_ANALYST_PROMPT + "\n" + language_rule(extracted[:4000])
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"RICHIESTA: {prompt}\n\nCONTENUTO PDF:\n{extracted[:14000]}"},
    ]
    reply = groq_chat(msgs, temperature=0.4, max_tokens=1400)
    reply = clean_text(reply)

    inc_use(user_id, "pdf")
    return {"text": reply}


# =========================
# frontend/index.html
# (GitHub Pages + Login + Premium UI + Tabs + Limits)
# =========================
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ChatAI Bob FREE</title>
  <style>
    :root{
      --bg:#0b0f1a; --panel:#020617; --panel2:#111827;
      --text:#e5e7eb; --muted:#9ca3af; --accent:#38bdf8;
      --ok:#34d399; --bad:#fb7185; --border:rgba(255,255,255,.10);
      --r:16px;
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:Arial;background:radial-gradient(1200px 800px at 10% 0%, #0f172a, var(--bg));color:var(--text)}
    .top{padding:12px 14px;background:linear-gradient(180deg, rgba(2,6,23,.95), rgba(2,6,23,.75));display:flex;justify-content:space-between;gap:10px;align-items:center;border-bottom:1px solid var(--border)}
    .brand{font-weight:900}
    .pill{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid var(--border);font-size:12px;color:var(--muted)}
    .pill.premium{color:var(--ok);border-color:rgba(52,211,153,.35);background:rgba(52,211,153,.08)}
    .status{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:8px}
    .dot{width:9px;height:9px;border-radius:50%;background:#999;display:inline-block}
    .dot.ok{background:var(--ok)}
    .dot.bad{background:var(--bad)}
    .wrap{max-width:1100px;margin:0 auto;padding:12px;display:grid;gap:10px;min-height:calc(100vh - 60px)}
    .grid{display:grid;gap:10px}
    @media(min-width:900px){ .grid{grid-template-columns:1.25fr .75fr} }
    .card{border:1px solid var(--border);background:rgba(15,23,42,.65);border-radius:18px;overflow:hidden}
    .card h3{margin:0;font-size:14px;padding:12px;border-bottom:1px solid var(--border);background:rgba(17,24,39,.5)}
    .body{padding:12px}
    input, textarea, select, button{
      width:100%; padding:12px; border-radius:var(--r); border:1px solid var(--border);
      background:rgba(2,6,23,.45); color:var(--text); font-size:15px; outline:none;
    }
    textarea{min-height:110px; resize:vertical}
    button{cursor:pointer; font-weight:900; background:rgba(56,189,248,.16); border-color:rgba(56,189,248,.35)}
    button.secondary{background:rgba(255,255,255,.06); border-color:var(--border); font-weight:800}
    .row{display:grid;gap:10px}
    .tabs{display:flex;gap:8px;flex-wrap:wrap}
    .tab{padding:10px 12px;border-radius:999px;border:1px solid var(--border);background:rgba(255,255,255,.03);cursor:pointer;font-size:14px}
    .tab.active{border-color:rgba(56,189,248,.4);background:rgba(56,189,248,.12);color:var(--accent);font-weight:900}
    .messages{max-height:56vh;overflow:auto;display:flex;flex-direction:column;gap:10px;padding-right:4px}
    .msg{border:1px solid var(--border);border-radius:16px;padding:12px;white-space:pre-wrap;word-break:break-word}
    .msg.user{border-color:rgba(56,189,248,.25);background:rgba(56,189,248,.10)}
    .msg.ai{background:rgba(17,24,39,.55)}
    .meta{font-size:12px;color:var(--muted);display:flex;justify-content:space-between;margin-bottom:6px}
    .hint{margin-top:6px;font-size:12px;color:var(--muted);line-height:1.35}
    .two{display:grid;gap:10px}
    @media(min-width:900px){ .two{grid-template-columns:1fr 1fr} }
    .preview{border:1px solid var(--border);border-radius:16px;background:rgba(2,6,23,.35);padding:10px;overflow:hidden}
    img, video{width:100%;border-radius:12px;display:block}
    .mini{font-size:12px;color:var(--muted)}
  </style>
</head>
<body>
  <div class="top">
    <div class="brand">🤖 ChatAI Bob <span id="tier" class="pill">FREE</span></div>
    <div class="status">
      <span id="statusDot" class="dot"></span>
      <span id="statusText">Backend: non verificato</span>
    </div>
  </div>

  <div class="wrap grid">
    <!-- MAIN -->
    <div class="card">
      <h3 id="mainTitle">💬 Chat</h3>
      <div class="body">
        <div class="tabs" id="tabs">
          <div class="tab active" data-tab="chat">💬 Chat</div>
          <div class="tab" data-tab="image">🖼️ Immagini</div>
          <div class="tab" data-tab="video">🎥 Video breve</div>
          <div class="tab" data-tab="pdf">📄 PDF</div>
          <div class="tab" data-tab="projects">📁 Progetti</div>
        </div>

        <div style="height:12px"></div>

        <!-- CHAT -->
        <div id="view-chat">
          <div class="messages" id="messages"></div>

          <div class="row" style="margin-top:10px">
            <textarea id="chatInput" placeholder="Scrivi... (es: Scrivi un libro fantasy)"></textarea>

            <div class="two">
              <button id="sendBtn">Invia</button>
              <button id="micBtn" class="secondary">🎙️ Parla (STT)</button>
            </div>

            <div class="two">
              <button id="voiceBtn" class="secondary">🔊 Voce: ON (TTS)</button>
              <button id="clearBtn" class="secondary">🧹 Pulisci chat</button>
            </div>

            <div class="hint">
              Voce (TTS) è lato browser. Microfono (STT) funziona meglio su Chrome/Android.
            </div>
          </div>
        </div>

        <!-- IMAGE -->
        <div id="view-image" style="display:none">
          <div class="row">
            <input id="imgPrompt" placeholder="Prompt immagine (es: cane realistico in spiaggia)"/>
            <button id="imgBtn">Genera immagine</button>
            <div class="preview" id="imgPrev" style="display:none">
              <div class="mini" id="imgUrl"></div>
              <img id="imgTag" alt="immagine"/>
            </div>
          </div>
        </div>

        <!-- VIDEO -->
        <div id="view-video" style="display:none">
          <div class="row">
            <input id="vidPrompt" placeholder="Prompt video breve (GIF 3 scene)"/>
            <button id="vidBtn">Genera video</button>
            <div class="preview" id="vidPrev" style="display:none">
              <div class="mini" id="vidUrl"></div>
              <video id="vidTag" controls playsinline></video>
            </div>
          </div>
        </div>

        <!-- PDF -->
        <div id="view-pdf" style="display:none">
          <div class="row">
            <input type="file" id="pdfFile" accept=".pdf"/>
            <textarea id="pdfPrompt" placeholder="Cosa devo fare sul PDF? (es: Riassumi e estrai punti chiave)"></textarea>
            <button id="pdfBtn">Analizza PDF</button>
            <div class="preview" id="pdfPrev" style="display:none">
              <div class="msg ai" id="pdfOut"></div>
            </div>
          </div>
        </div>

        <!-- PROJECTS -->
        <div id="view-projects" style="display:none">
          <div class="row">
            <div class="preview">
              <div class="mini">Crea un progetto (memoria separata)</div>
              <input id="projName" placeholder="Nome progetto (es: Libro Fantasy)"/>
              <select id="projType">
                <option value="BOOK">📘 Libro</option>
                <option value="COACH">🧠 Coaching</option>
                <option value="CODER">💻 Codice</option>
                <option value="GENERAL">✨ Generale</option>
              </select>
              <button id="projCreate">Crea progetto</button>
              <div class="hint">Dopo aver selezionato un progetto, la chat lo usa automaticamente.</div>
            </div>

            <div class="preview">
              <div class="mini">I tuoi progetti</div>
              <button id="projRefresh" class="secondary">Aggiorna lista</button>
              <div id="projList" style="margin-top:10px"></div>
              <div class="hint">Progetto attivo: <b id="projActive">Nessuno</b></div>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- SIDE -->
    <div class="card">
      <h3>⚙️ Account + Backend + Premium</h3>
      <div class="body">
        <div class="row">
          <div class="mini">Backend URL</div>
          <input id="apiBase" placeholder="https://tuo-backend.onrender.com"/>
          <button id="saveApi">Salva backend URL</button>
          <button id="checkHealth" class="secondary">Verifica backend</button>
          <div class="hint">Inserisci SOLO dominio base. Esempio: https://chatai-bob-backend.onrender.com</div>

          <hr style="border:0;border-top:1px solid rgba(255,255,255,.08)">

          <div class="mini">Login</div>
          <input id="email" placeholder="Email"/>
          <input id="password" type="password" placeholder="Password (min 6)"/>
          <div class="two">
            <button id="signupBtn" class="secondary">Crea account</button>
            <button id="loginBtn">Login</button>
          </div>
          <button id="logoutBtn" class="secondary" style="display:none">Logout</button>
          <div class="hint" id="loginHint">Devi fare login per usare la chat.</div>

          <hr style="border:0;border-top:1px solid rgba(255,255,255,.08)">

          <div class="mini">Premium</div>
          <div class="preview">
            <div><b>Premium – €4,99 / mese</b></div>
            <div class="mini" style="margin-top:6px">Sblocca tutto: chat, immagini, video, PDF, limiti illimitati.</div>
            <div style="height:8px"></div>
            <a id="paypalLink" href="https://www.paypal.me/bobbob1979/4.99" target="_blank" style="color:#38bdf8;font-weight:900;text-decoration:none">
              💳 Paga con PayPal
            </a>
            <div class="hint" style="margin-top:8px">
              Dopo il pagamento ti attivo Premium manualmente (per ora, senza webhook).
            </div>
          </div>

          <div class="preview">
            <div class="mini">Uso giornaliero (FREE)</div>
            <div id="usageBox" class="mini" style="margin-top:8px;line-height:1.6">—</div>
          </div>

          <div class="hint">
            Se vedi limiti: Premium = illimitato.
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
/* ===== STORAGE ===== */
function getApiBase(){ return (localStorage.getItem("API_BASE")||"").trim().replace(/\/+$/,""); }
function setApiBase(v){ const x=(v||"").trim().replace(/\/+$/,""); localStorage.setItem("API_BASE", x); return x; }
function getToken(){ return localStorage.getItem("TOKEN")||""; }
function setToken(t){ localStorage.setItem("TOKEN", t||""); }
function getProjectId(){ const v=localStorage.getItem("PROJECT_ID"); return v ? parseInt(v,10) : null; }
function setProjectId(id){ if(id==null){ localStorage.removeItem("PROJECT_ID"); } else { localStorage.setItem("PROJECT_ID", String(id)); } }
let VOICE_ON = (localStorage.getItem("VOICE_ON") || "1") === "1";
function setVoiceOn(on){ VOICE_ON = !!on; localStorage.setItem("VOICE_ON", VOICE_ON ? "1":"0"); }

/* ===== UI ===== */
const els = {
  tier: document.getElementById("tier"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  apiBase: document.getElementById("apiBase"),
  saveApi: document.getElementById("saveApi"),
  checkHealth: document.getElementById("checkHealth"),

  email: document.getElementById("email"),
  password: document.getElementById("password"),
  signupBtn: document.getElementById("signupBtn"),
  loginBtn: document.getElementById("loginBtn"),
  logoutBtn: document.getElementById("logoutBtn"),
  loginHint: document.getElementById("loginHint"),

  tabs: document.getElementById("tabs"),
  mainTitle: document.getElementById("mainTitle"),

  messages: document.getElementById("messages"),
  chatInput: document.getElementById("chatInput"),
  sendBtn: document.getElementById("sendBtn"),
  micBtn: document.getElementById("micBtn"),
  voiceBtn: document.getElementById("voiceBtn"),
  clearBtn: document.getElementById("clearBtn"),

  imgPrompt: document.getElementById("imgPrompt"),
  imgBtn: document.getElementById("imgBtn"),
  imgPrev: document.getElementById("imgPrev"),
  imgTag: document.getElementById("imgTag"),
  imgUrl: document.getElementById("imgUrl"),

  vidPrompt: document.getElementById("vidPrompt"),
  vidBtn: document.getElementById("vidBtn"),
  vidPrev: document.getElementById("vidPrev"),
  vidTag: document.getElementById("vidTag"),
  vidUrl: document.getElementById("vidUrl"),

  pdfFile: document.getElementById("pdfFile"),
  pdfPrompt: document.getElementById("pdfPrompt"),
  pdfBtn: document.getElementById("pdfBtn"),
  pdfPrev: document.getElementById("pdfPrev"),
  pdfOut: document.getElementById("pdfOut"),

  projName: document.getElementById("projName"),
  projType: document.getElementById("projType"),
  projCreate: document.getElementById("projCreate"),
  projRefresh: document.getElementById("projRefresh"),
  projList: document.getElementById("projList"),
  projActive: document.getElementById("projActive"),

  usageBox: document.getElementById("usageBox"),
};

function nowTime(){
  const d = new Date();
  return d.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
}

function addMessage(role, text){
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "ai");

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `<span>${role === "user" ? "👤 Tu" : "🤖 AI"}</span><span>${nowTime()}</span>`;

  const body = document.createElement("div");
  body.textContent = text;

  div.appendChild(meta);
  div.appendChild(body);
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setBackendStatus(ok, extra){
  if(ok){
    els.statusDot.className = "dot ok";
    els.statusText.textContent = "Backend: online" + (extra ? (" • " + extra) : "");
  } else {
    els.statusDot.className = "dot bad";
    els.statusText.textContent = "Backend: offline / URL errato";
  }
}

/* ===== API ===== */
async function apiFetch(path, opts={}){
  const base = getApiBase();
  if(!base) throw new Error("API_BASE non impostato");
  const headers = opts.headers || {};
  const token = getToken();
  if(token) headers["Authorization"] = "Bearer " + token;
  opts.headers = headers;

  const res = await fetch(base + path, opts);
  const ct = res.headers.get("content-type") || "";
  const data = ct.includes("application/json") ? await res.json() : await res.text();
  if(!res.ok){
    const msg = (data && data.detail) ? data.detail : (data.error || JSON.stringify(data));
    throw new Error(msg);
  }
  return data;
}

async function checkHealth(){
  const base = getApiBase();
  if(!base){
    els.statusDot.className = "dot";
    els.statusText.textContent = "Backend: non verificato";
    return;
  }
  try{
    const r = await fetch(base + "/health");
    const d = await r.json();
    setBackendStatus(d.status === "ok", d.model || "");
  }catch{
    setBackendStatus(false, "");
  }
}

/* ===== AUTH ===== */
function setLoggedUI(isLogged){
  els.logoutBtn.style.display = isLogged ? "block" : "none";
  els.loginBtn.style.display = isLogged ? "none" : "block";
  els.signupBtn.style.display = isLogged ? "none" : "block";
  els.loginHint.textContent = isLogged ? "Login OK ✅" : "Devi fare login per usare la chat.";
}

async function refreshMe(){
  try{
    const me = await apiFetch("/me");
    els.tier.textContent = me.premium ? "PREMIUM" : "FREE";
    els.tier.className = me.premium ? "pill premium" : "pill";
    setLoggedUI(true);

    const u = me.usage || {};
    const limits = me.limits || {};
    let lines = [];
    for(const k of Object.keys(limits)){
      lines.push(`${k}: ${u[k]||0} / ${limits[k]}`);
    }
    els.usageBox.textContent = lines.join("   |   ");
  }catch{
    els.tier.textContent = "FREE";
    els.tier.className = "pill";
    els.usageBox.textContent = "—";
    setLoggedUI(false);
  }
}

els.signupBtn.onclick = async () => {
  try{
    await apiFetch("/auth/signup", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ email: els.email.value.trim(), password: els.password.value })
    });
    alert("Account creato ✅ Ora fai Login.");
  }catch(e){
    alert("Errore signup: " + e.message);
  }
};

els.loginBtn.onclick = async () => {
  try{
    const d = await apiFetch("/auth/login", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ email: els.email.value.trim(), password: els.password.value })
    });
    if(!d.ok) throw new Error(d.error || "Login fallito");
    setToken(d.token);
    await refreshMe();
    addMessage("ai", "LOGIN OK. ORA POSSO LAVORARE AL MASSIMO.");
  }catch(e){
    alert("Errore login: " + e.message);
  }
};

els.logoutBtn.onclick = async () => {
  setToken("");
  setProjectId(null);
  els.projActive.textContent = "Nessuno";
  await refreshMe();
  alert("Logout OK");
};

/* ===== TABS ===== */
const views = {
  chat: document.getElementById("view-chat"),
  image: document.getElementById("view-image"),
  video: document.getElementById("view-video"),
  pdf: document.getElementById("view-pdf"),
  projects: document.getElementById("view-projects"),
};
const titles = {
  chat: "💬 Chat",
  image: "🖼️ Immagini",
  video: "🎥 Video breve",
  pdf: "📄 PDF",
  projects: "📁 Progetti"
};

function switchTab(name){
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  Object.keys(views).forEach(k => views[k].style.display = (k===name ? "block":"none"));
  els.mainTitle.textContent = titles[name] || "ChatAI";
}

els.tabs.addEventListener("click", (e)=>{
  const t = e.target.closest(".tab");
  if(!t) return;
  switchTab(t.dataset.tab);
});

/* ===== VOICE (TTS in browser) ===== */
function speak(text){
  if(!VOICE_ON) return;
  if(!("speechSynthesis" in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  // lingua auto: se arabo usa ar, altrimenti it/eng base
  utter.lang = /[\u0600-\u06FF]/.test(text) ? "ar-MA" : "it-IT";
  speechSynthesis.cancel();
  speechSynthesis.speak(utter);
}

els.voiceBtn.onclick = ()=>{
  setVoiceOn(!VOICE_ON);
  els.voiceBtn.textContent = VOICE_ON ? "🔊 Voce: ON (TTS)" : "🔇 Voce: OFF (TTS)";
};

/* ===== MIC (STT) ===== */
function startMic(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){ alert("SpeechRecognition non supportato su questo browser."); return; }
  const rec = new SR();
  rec.lang = "it-IT";
  rec.interimResults = false;
  rec.maxAlternatives = 1;
  rec.onresult = (e)=>{
    const t = e.results[0][0].transcript;
    els.chatInput.value = (els.chatInput.value ? (els.chatInput.value + "\n") : "") + t;
  };
  rec.start();
}
els.micBtn.onclick = startMic;

/* ===== CHAT ===== */
els.sendBtn.onclick = async ()=>{
  try{
    const text = (els.chatInput.value || "").trim();
    if(!text) return;
    addMessage("user", text);
    els.chatInput.value = "";

    const pid = getProjectId();
    const data = await apiFetch("/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ message: text, project_id: pid, mic_used: false })
    });

    addMessage("ai", data.text || "…");
    speak(data.text || "");
    await refreshMe();
  }catch(e){
    addMessage("ai", "ERRORE: " + e.message);
  }
};

els.chatInput.addEventListener("keydown",(ev)=>{
  if(ev.key==="Enter" && (ev.ctrlKey||ev.metaKey)) els.sendBtn.click();
});

els.clearBtn.onclick = ()=>{
  els.messages.innerHTML = "";
  addMessage("ai", "CHAT PULITA. DIMMI COSA CREIAMO.");
};

/* ===== IMAGE ===== */
els.imgBtn.onclick = async ()=>{
  try{
    const prompt = (els.imgPrompt.value||"").trim();
    if(!prompt) return;
    const data = await apiFetch("/image", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ prompt })
    });
    if(data.error){ alert(data.error); return; }
    els.imgTag.src = data.url;
    els.imgUrl.textContent = "OK";
    els.imgPrev.style.display = data.url ? "block":"none";
    await refreshMe();
  }catch(e){
    alert("Errore immagine: " + e.message);
  }
};

/* ===== VIDEO ===== */
els.vidBtn.onclick = async ()=>{
  try{
    const prompt = (els.vidPrompt.value||"").trim();
    if(!prompt) return;
    const data = await apiFetch("/video", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ prompt })
    });
    if(data.error){ alert(data.error); return; }
    els.vidTag.src = data.url || "";
    els.vidUrl.textContent = "OK";
    els.vidPrev.style.display = data.url ? "block":"none";
    await refreshMe();
  }catch(e){
    alert("Errore video: " + e.message);
  }
};

/* ===== PDF ===== */
els.pdfBtn.onclick = async ()=>{
  try{
    const f = els.pdfFile.files && els.pdfFile.files[0];
    const p = (els.pdfPrompt.value||"").trim();
    if(!f) return alert("Seleziona un PDF");
    if(!p) return alert("Scrivi cosa vuoi fare sul PDF");

    const form = new FormData();
    form.append("file", f);
    form.append("prompt", p);
    form.append("project_id", String(getProjectId() || ""));

    const data = await apiFetch("/analyze", { method:"POST", body: form });
    els.pdfOut.textContent = data.text || "—";
    els.pdfPrev.style.display = "block";
    speak(data.text || "");
    await refreshMe();
  }catch(e){
    alert("Errore PDF: " + e.message);
  }
};

/* ===== PROJECTS ===== */
async function loadProjects(){
  const d = await apiFetch("/projects/list");
  els.projList.innerHTML = "";
  (d.projects || []).forEach(p=>{
    const b = document.createElement("button");
    b.className = "secondary";
    b.textContent = `📂 ${p.name} (${p.type})`;
    b.onclick = ()=>{
      setProjectId(p.id);
      els.projActive.textContent = p.name;
      alert("Progetto attivo: " + p.name);
    };
    els.projList.appendChild(b);
    els.projList.appendChild(document.createElement("div")).style.height="8px";
  });
}

els.projCreate.onclick = async ()=>{
  try{
    const name = (els.projName.value||"").trim();
    const type = els.projType.value;
    if(!name) return alert("Inserisci nome progetto");
    const d = await apiFetch("/projects/create", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ name, type })
    });
    els.projName.value = "";
    await loadProjects();
    if(d.project_id){
      setProjectId(d.project_id);
      els.projActive.textContent = name;
    }
  }catch(e){
    alert("Errore progetto: " + e.message);
  }
};

els.projRefresh.onclick = ()=> loadProjects();

/* ===== BACKEND SETTINGS ===== */
els.saveApi.onclick = ()=>{
  const v = setApiBase(els.apiBase.value);
  els.apiBase.value = v;
  checkHealth();
};
els.checkHealth.onclick = checkHealth;

/* ===== INIT ===== */
(function init(){
  const base = getApiBase();
  if(base) els.apiBase.value = base;
  els.voiceBtn.textContent = VOICE_ON ? "🔊 Voce: ON (TTS)" : "🔇 Voce: OFF (TTS)";

  checkHealth();
  refreshMe();

  addMessage("ai", "CIAO. FAI LOGIN, POI DIMMI COSA VUOI CREARE: LIBRO, CODICE, IDEE, VIDEO, IMMAGINI, PDF.");
})();
</script>
</body>
</html>


# =========================
# android/app/src/main/AndroidManifest.xml
# =========================
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
  package="com.chatai.bob">

  <uses-permission android:name="android.permission.INTERNET" />

  <application
    android:usesCleartextTraffic="true"
    android:label="ChatAI Bob"
    android:theme="@style/Theme.AppCompat.Light.NoActionBar">

    <activity
      android:name=".MainActivity"
      android:exported="true">
      <intent-filter>
        <action android:name="android.intent.action.MAIN"/>
        <category android:name="android.intent.category.LAUNCHER"/>
      </intent-filter>
    </activity>

  </application>
</manifest>


# =========================
# android/app/src/main/java/com/chatai/bob/MainActivity.java
# =========================
package com.chatai.bob;

import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    WebView webView = new WebView(this);
    setContentView(webView);

    WebSettings ws = webView.getSettings();
    ws.setJavaScriptEnabled(true);
    ws.setDomStorageEnabled(true);
    ws.setMediaPlaybackRequiresUserGesture(false);

    webView.setWebViewClient(new WebViewClient());

    // METTI QUI IL TUO GITHUB PAGES (frontend)
    webView.loadUrl("https://bobmoukhlis-hash.github.io/");
  }
}


# =========================
# android/app/build.gradle  (minimo)
# =========================
plugins {
  id 'com.android.application'
}

android {
  namespace 'com.chatai.bob'
  compileSdk 34

  defaultConfig {
    applicationId "com.chatai.bob"
    minSdk 24
    targetSdk 34
    versionCode 1
    versionName "1.0"
  }

  buildTypes {
    release {
      minifyEnabled false
    }
  }
}

dependencies {
  implementation 'androidx.appcompat:appcompat:1.7.0'
}


# =========================
# android/settings.gradle
# =========================
pluginManagement {
  repositories {
    google()
    mavenCentral()
    gradlePluginPortal()
  }
}
dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
  }
}
rootProject.name = "ChatAIBob"
include ':app'
