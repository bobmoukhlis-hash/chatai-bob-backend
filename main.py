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
from PyPDF2 import PdfReader

# -------------------------
# CONFIG
# -------------------------
def get_groq_key() -> Optional[str]:
    return os.environ.get("GROQ_API_KEY")


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()
DB_PATH = os.getenv("SQLITE_PATH", "data.sqlite3").strip()
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

# FREE LIMITS (per client_id / giorno)
LIMITS_FREE = {
    "chat": 15,
    "voice": 15,
    "mic": 15,
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
          type TEXT NOT NULL,
          state TEXT NOT NULL,
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
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_limits (
            client_id TEXT,
            day INTEGER,
            chat_count INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            video_count INTEGER DEFAULT 0,
            pdf_count INTEGER DEFAULT 0,
            voice_count INTEGER DEFAULT 0,
            mic_count INTEGER DEFAULT 0,
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
# AUTH (Premium)
# -------------------------
def new_token() -> str:
    return secrets.token_urlsafe(32)


def auth_user_required(authorization: Optional[str]) -> Tuple[int, bool]:
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
    return int(row[0]), bool(int(row[1]) == 1)


def auth_user_optional(authorization: Optional[str]) -> Tuple[Optional[int], bool]:
    if not authorization:
        return None, False
    try:
        return auth_user_required(authorization)
    except HTTPException:
        return None, False


def set_premium_by_email(email: str, premium: int) -> None:
    DB.execute("UPDATE users SET premium=? WHERE email=?", (premium, email.strip().lower()))
    DB.commit()


# -------------------------
# LIMITS (Premium per user_id)
# -------------------------
def can_use_user(user_id: int, premium: bool, action: str) -> bool:
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


def inc_use_user(user_id: int, action: str) -> None:
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
# LIMITS (FREE per client_id)
# -------------------------
def _free_row(client_id: str) -> Tuple[int, int, int, int, int, int]:
    day = today_day()
    row = DB.execute(
        """
        SELECT chat_count, image_count, video_count, pdf_count, voice_count, mic_count
        FROM usage_limits
        WHERE client_id=? AND day=?
        """,
        (client_id, day),
    ).fetchone()
    if row:
        return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))

    DB.execute(
        "INSERT INTO usage_limits (client_id, day) VALUES (?, ?)",
        (client_id, day),
    )
    DB.commit()
    return (0, 0, 0, 0, 0, 0)


def free_can_use(client_id: str, action: str) -> bool:
    action = action if action in LIMITS_FREE else "other"
    chat_c, img_c, vid_c, pdf_c, voice_c, mic_c = _free_row(client_id)
    used_map = {
        "chat": chat_c,
        "image": img_c,
        "video": vid_c,
        "pdf": pdf_c,
        "voice": voice_c,
        "mic": mic_c,
        "other": 0,
    }
    return used_map.get(action, 0) < int(LIMITS_FREE.get(action, 0))


def free_inc_use(client_id: str, action: str) -> None:
    action = action if action in LIMITS_FREE else "other"
    day = today_day()
    col = f"{action}_count" if action in {"chat", "image", "video", "pdf", "voice", "mic"} else None
    if not col:
        return
    DB.execute(
        f"UPDATE usage_limits SET {col}={col}+1 WHERE client_id=? AND day=?",
        (client_id, day),
    )
    DB.commit()


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
    low = t.lower()
    if any(x in low for x in ["hello", "write", "book", "plan", "business", "how to", "why", "what is"]):
        return "Reply in English, natural, human."
    return "Rispondi nella stessa lingua usata dall’utente. Se non è chiara, scegli la lingua più probabile."


# -------------------------
# GROQ
# -------------------------
def groq_chat(messages: List[Dict[str, str]], temperature: float = 0.75, max_tokens: int = 1400) -> str:
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
        "Content-Type": "application/json",
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
    client_id: Optional[str] = None  # <-- AGGIUNTO (FREE)
    project_id: Optional[int] = None
    mic_used: Optional[bool] = False


class ProjectCreateRequest(BaseModel):
    name: str
    type: str  # BOOK/COACH/CODER/GENERAL


class ImageRequest(BaseModel):
    prompt: str
    client_id: Optional[str] = None  # <-- AGGIUNTO (FREE)


class VideoRequest(BaseModel):
    prompt: str
    client_id: Optional[str] = None  # <-- AGGIUNTO (FREE)


# -------------------------
# APP
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://obmoukhlish-hash.github.io",
        "https://bobmoukhlis-hash.github.io",
        "http://localhost:3000",
        "http://localhost:5500",
    ],
    allow_credentials=True,
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
# AUTH ROUTES (Premium)
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
    DB.execute(
        "INSERT INTO tokens (token, user_id, created_at, last_seen) VALUES (?,?,?,?)",
        (token, int(row[0]), now_ts(), now_ts()),
    )
    DB.commit()

    return {"ok": True, "token": token, "premium": bool(int(row[2]) == 1), "email": email}


@app.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user_required(authorization)
    email = DB.execute("SELECT email FROM users WHERE id=?", (user_id,)).fetchone()[0]
    return {"ok": True, "email": email, "premium": premium, "usage": usage_snapshot(user_id), "limits": LIMITS_FREE}


# -------------------------
# PREMIUM (NO WEBHOOK) - ADMIN MANUAL
# -------------------------
@app.post("/premium/admin_set")
def admin_set_premium(email: str = Form(...), premium: int = Form(...), admin_key: str = Form(...)):
    admin_env = os.getenv("ADMIN_KEY", "").strip()
    if not admin_env or admin_key != admin_env:
        raise HTTPException(status_code=403, detail="Forbidden")

    set_premium_by_email(email, 1 if premium else 0)
    return {"ok": True}


# -------------------------
# PROJECTS (Premium only)
# -------------------------
@app.post("/projects/create")
def projects_create(req: ProjectCreateRequest, authorization: Optional[str] = Header(default=None)):
    user_id, premium = auth_user_required(authorization)
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
    user_id, premium = auth_user_required(authorization)
    rows = DB.execute(
        "SELECT id, name, type, state, updated_at FROM projects WHERE user_id=? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    projects = []
    for r in rows:
        projects.append({"id": int(r[0]), "name": r[1], "type": r[2], "state": kv_load(r[3]), "updated_at": int(r[4])})
    return {"ok": True, "projects": projects}


# -------------------------
# CHAT (FREE without token, Premium with token)
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
    client_id = (req.client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    # LIMITS
    if user_id is not None:
        if not can_use_user(user_id, premium, "chat"):
            return {"text": "LIMITE GIORNALIERO CHAT RAGGIUNTO. PASSA A PREMIUM PER ILLIMITATO."}
    else:
        if not free_can_use(client_id, "chat"):
            return {"text": "LIMITE CHAT FREE RAGGIUNTO. PASSA A PREMIUM PER CONTINUARE."}

    user_text = (req.message or "").strip()
    if not user_text:
        return {"text": "SCRIVI QUALCOSA E PARTO SUBITO."}

    if req.mic_used:
        if user_id is not None:
            if can_use_user(user_id, premium, "mic"):
                inc_use_user(user_id, "mic")
        else:
            if free_can_use(client_id, "mic"):
                free_inc_use(client_id, "mic")

    project_id = req.project_id if user_id is not None else None  # FREE non salva su progetti server

    # project bias (Premium)
    if user_id is not None and project_id:
        pr = DB.execute("SELECT type, state FROM projects WHERE id=? AND user_id=?", (project_id, user_id)).fetchone()
        if pr:
            ptype = pr[0]
            state = kv_load(pr[1])
            if ptype == "BOOK":
                if user_text.lower() in {"continua", "vai avanti", "prosegui", "avanti", "ok"}:
                    ch = int(state.get("chapter", "1"))
                    ch += 1
                    state["chapter"] = str(ch)
                    DB.execute(
                        "UPDATE projects SET state=?, updated_at=? WHERE id=? AND user_id=?",
                        (kv_dump(state), now_ts(), project_id, user_id),
                    )
                    DB.commit()
                    user_text = f"CONTINUA IL LIBRO DAL CAPITOLO {ch} MANTENENDO TRAMA E STILE COERENTI."
                elif "inizia" in user_text.lower() and "libro" in user_text.lower():
                    state["chapter"] = "1"
                    DB.execute(
                        "UPDATE projects SET state=?, updated_at=? WHERE id=? AND user_id=?",
                        (kv_dump(state), now_ts(), project_id, user_id),
                    )
                    DB.commit()

    intent = detect_intent(user_text)

    if user_id is not None and project_id:
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

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]

    # Premium history only
    if user_id is not None:
        history = last_messages(user_id, project_id, limit=8)
        for m in history[-4:]:
            messages.append(m)

    messages.append({"role": "user", "content": user_text})

    # save user message (Premium only)
    if user_id is not None:
        DB.execute(
            "INSERT INTO messages (user_id, project_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (user_id, project_id, "user", user_text, now_ts()),
        )
        DB.commit()

    reply = groq_chat(messages, temperature=0.8, max_tokens=1600)
    reply = clean_text(reply)

    # save assistant message (Premium only)
    if user_id is not None:
        DB.execute(
            "INSERT INTO messages (user_id, project_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (user_id, project_id, "assistant", reply, now_ts()),
        )
        DB.commit()
        inc_use_user(user_id, "chat")
    else:
        free_inc_use(client_id, "chat")

    return {"text": reply}


# -------------------------
# IMAGE (FREE without token, Premium with token)
# -------------------------
@app.post("/image")
def image(req: ImageRequest, authorization: Optional[str] = Header(default=None)):
    client_id = (req.client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    if user_id is not None:
        if not can_use_user(user_id, premium, "image"):
            return {"url": "", "error": "LIMITE IMMAGINI GIORNALIERO RAGGIUNTO. PASSA A PREMIUM."}
    else:
        if not free_can_use(client_id, "image"):
            return {"url": "", "error": "LIMITE IMMAGINI FREE RAGGIUNTO. PASSA A PREMIUM."}

    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"url": ""}

    safe_prompt = f"high quality, detailed, realistic, {prompt}"
    img_bytes = pollinations_image(safe_prompt)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    if user_id is not None:
        inc_use_user(user_id, "image")
    else:
        free_inc_use(client_id, "image")

    return {"url": f"data:image/png;base64,{b64}"}


# -------------------------
# VIDEO (GIF 3 scene) (FREE without token, Premium with token)
# -------------------------
@app.post("/video")
def video(req: VideoRequest, authorization: Optional[str] = Header(default=None)):
    client_id = (req.client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    if user_id is not None:
        if not can_use_user(user_id, premium, "video"):
            return {"url": "", "error": "LIMITE VIDEO GIORNALIERO RAGGIUNTO. PASSA A PREMIUM."}
    else:
        if not free_can_use(client_id, "video"):
            return {"url": "", "error": "LIMITE VIDEO FREE RAGGIUNTO. PASSA A PREMIUM."}

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

    if user_id is not None:
        inc_use_user(user_id, "video")
    else:
        free_inc_use(client_id, "video")

    return {"url": f"data:image/gif;base64,{b64}"}


# -------------------------
# ANALYZE (PDF) (FREE without token, Premium with token)
# -------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    project_id: str = Form(""),
    client_id: str = Form("client_anon"),  # <-- AGGIUNTO (FREE)
    authorization: Optional[str] = Header(default=None),
):
    client_id = (client_id or "").strip() or "client_anon"
    user_id, premium = auth_user_optional(authorization)

    if user_id is not None:
        if not can_use_user(user_id, premium, "pdf"):
            return {"text": "LIMITE PDF GIORNALIERO RAGGIUNTO. PASSA A PREMIUM."}
    else:
        if not free_can_use(client_id, "pdf"):
            return {"text": "LIMITE PDF FREE RAGGIUNTO. PASSA A PREMIUM."}

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

    if user_id is not None:
        inc_use_user(user_id, "pdf")
    else:
        free_inc_use(client_id, "pdf")

    return {"text": reply}
