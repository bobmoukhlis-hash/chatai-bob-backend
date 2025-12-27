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
def get_groq_key():
    return os.environ.get("GROQ_API_KEY")
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
    )conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_limits (
            client_id TEXT,
            day INTEGER,
            chat_count INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            video_count INTEGER DEFAULT 0,
            pdf_count INTEGER DEFAULT 0,
            voice_count INTEGER DEFAULT 0,
            PRIMARY KEY (client_id, day)
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
def groq_chat(messages: List[Dict[str, str]], temperature: float = 0.75, max_tokens: int = 1400):
    api_key = get_groq_key()
    if not api_key:
        return "⚠️ Servizio in avvio, riprova tra qualche secondo."
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
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
