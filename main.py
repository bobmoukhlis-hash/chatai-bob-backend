from __future__ import annotations

import base64
import io
import os
import re
import sqlite3
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIG =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
Sei ChatAI Bob, un assistente AI avanzato e professionale.

REGOLE ASSOLUTE:
- Rispondi SEMPRE in italiano
- NON fare domande
- NON chiedere chiarimenti
- Se l'utente chiede di creare qualcosa (libro, codice, storia, idea):
  INIZIA SUBITO a crearlo
- Usa titoli, sezioni, struttura chiara
- Produci risposte COMPLETE e di ALTA QUALITÀ
"""

class ChatRequest(BaseModel):
    message: str
    client_id: str | None = None

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return {"text": "❌ GROQ_API_KEY non configurata sul server."}

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ],
        "temperature": 0.85,
        "max_tokens": 2000
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        return {"text": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"text": f"⚠️ Errore Groq: {e}"}

# Metti MODEL in Render ENV per cambiare senza toccare codice.
# Se vedi 400, cambia MODEL.
MODEL = os.getenv("MODEL", "llama3-8b-8192").strip()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.75"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1700"))

DB_PATH = os.getenv("SQLITE_PATH", "data.sqlite3").strip()

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

# OCR SETTINGS
OCR_LANGS = os.getenv("OCR_LANGS", "it,en").split(",")  # usato da EasyOCR se presente
OCR_MAX_IMAGE_SIDE = int(os.getenv("OCR_MAX_IMAGE_SIDE", "1600"))  # riduce peso OCR
OCR_PDF_MAX_PAGES = int(os.getenv("OCR_PDF_MAX_PAGES", "3"))  # OCR su poche pagine per performance

# =========================
# PROMPTS (NO MARKDOWN)
# =========================
BASE_RULES = """
Sei ChatAI Bob.

REGOLE ASSOLUTE:
- Rispondi SEMPRE in italiano.
- NON usare MAI Markdown: vietato **, ##, ``` , `, _, -.
- NON fare domande di chiarimento.
- Se mancano dettagli: DECIDI TU in modo professionale e vai avanti.
- Output pronto da usare: testo reale, titoli in MAIUSCOLO, paragrafi con a capo.
- Non dire frasi tipo "Sono un'AI", "Come posso aiutarti?".
"""

AUTHOR_PROMPT = (
    BASE_RULES
    + """
MODALITÀ AUTORE:
- Se l'utente chiede un libro/romanzo/storia: produci TITOLO + MINI-INDICE + CAPITOLO 1 completo subito.
- Se l'utente dice "continua": scrivi il capitolo successivo coerente con trama e stile già creati.
- Stile: narrativo, coinvolgente, ritmo alto, immagini forti.
"""
)

COACH_PROMPT = (
    BASE_RULES
    + """
MODALITÀ COACH:
- Dai un piano pratico con passi concreti, esercizi e routine.
- Niente domande: proponi TU obiettivi realistici e misurabili.
- Tono: motivazionale, diretto, umano.
"""
)

CODER_PROMPT = (
    BASE_RULES
    + """
MODALITÀ CODER:
- Se l'utente chiede codice: consegna codice completo pronto da copiare.
- Spiegazione breve (massimo 6 righe) e poi codice.
- Se chiede un progetto: struttura + file + contenuto.
"""
)

AUTOCORE_PROMPT = (
    BASE_RULES
    + """
AUTOCORE™ (AUTONOMO):
- Se l'utente è vago o scrive pochissimo, scegli tu la cosa più utile e potente e creala.
- Esempio: un testo motivazionale, un piano, una storia breve, un'idea business, un progetto codice.
- Niente domande.
"""
)

PDF_ANALYST_PROMPT = (
    BASE_RULES
    + """
MODALITÀ ANALISI DOCUMENTI:
- Analizza il contenuto fornito (testo estratto da PDF o OCR).
- Produci: RIASSUNTO, PUNTI CHIAVE, AZIONI/INSIGHT.
- Se richiesto: estrai date, numeri, entità, checklist.
- Niente domande.
"""
)

# =========================
# APP
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DB (PROGETTI + MEMORIA PERSISTENTE)
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            client_id TEXT PRIMARY KEY,
            active_project_id INTEGER,
            updated_at INTEGER NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,            -- BOOK / COACH / CODER / GENERAL
            state_json TEXT NOT NULL,      -- json-like string semplice
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            project_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
        """
    )

    conn.commit()
    return conn


DB = db()


def now_ts() -> int:
    return int(time.time())


def _json_dump(d: Dict[str, Any]) -> str:
    # JSON minimale senza import json (per semplicità e robustezza)
    # Nota: usiamo un formato key=value;.. per evitare problemi
    # (sufficiente per state interno)
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={str(v).replace(';',' ').replace('\\n',' ')}")
    return ";".join(parts)


def _json_load(s: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not s:
        return out
    for part in s.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def get_or_create_session(client_id: str) -> Dict[str, Any]:
    cur = DB.execute("SELECT active_project_id, updated_at FROM sessions WHERE client_id=?", (client_id,))
    row = cur.fetchone()
    if not row:
        DB.execute(
            "INSERT INTO sessions (client_id, active_project_id, updated_at) VALUES (?,?,?)",
            (client_id, None, now_ts()),
        )
        DB.commit()
        return {"active_project_id": None, "updated_at": now_ts()}
    return {"active_project_id": row[0], "updated_at": row[1]}


def set_active_project(client_id: str, project_id: Optional[int]) -> None:
    DB.execute(
        """
        INSERT INTO sessions (client_id, active_project_id, updated_at)
        VALUES (?,?,?)
        ON CONFLICT(client_id) DO UPDATE SET
            active_project_id=excluded.active_project_id,
            updated_at=excluded.updated_at
        """,
        (client_id, project_id, now_ts()),
    )
    DB.commit()


def create_project(client_id: str, name: str, ptype: str, state: Dict[str, Any]) -> int:
    ts = now_ts()
    cur = DB.execute(
        """
        INSERT INTO projects (client_id, name, type, state_json, created_at, updated_at)
        VALUES (?,?,?,?,?,?)
        """,
        (client_id, name, ptype, _json_dump(state), ts, ts),
    )
    DB.commit()
    return int(cur.lastrowid)


def list_projects(client_id: str) -> List[Dict[str, Any]]:
    cur = DB.execute(
        "SELECT id, name, type, state_json, created_at, updated_at FROM projects WHERE client_id=? ORDER BY updated_at DESC",
        (client_id,),
    )
    out = []
    for r in cur.fetchall():
        out.append(
            {
                "id": int(r[0]),
                "name": r[1],
                "type": r[2],
                "state": _json_load(r[3]),
                "created_at": int(r[4]),
                "updated_at": int(r[5]),
            }
        )
    return out


def get_project(client_id: str, project_id: int) -> Optional[Dict[str, Any]]:
    cur = DB.execute(
        "SELECT id, name, type, state_json FROM projects WHERE client_id=? AND id=?",
        (client_id, project_id),
    )
    r = cur.fetchone()
    if not r:
        return None
    return {"id": int(r[0]), "name": r[1], "type": r[2], "state": _json_load(r[3])}


def update_project_state(client_id: str, project_id: int, state: Dict[str, Any]) -> None:
    DB.execute(
        "UPDATE projects SET state_json=?, updated_at=? WHERE client_id=? AND id=?",
        (_json_dump(state), now_ts(), client_id, project_id),
    )
    DB.commit()


def add_msg(client_id: str, project_id: Optional[int], role: str, content: str) -> None:
    DB.execute(
        "INSERT INTO messages (client_id, project_id, role, content, created_at) VALUES (?,?,?,?,?)",
        (client_id, project_id, role, content, now_ts()),
    )
    DB.commit()


def get_last_messages(client_id: str, project_id: Optional[int], limit: int = 10) -> List[Dict[str, str]]:
    if project_id is None:
        cur = DB.execute(
            "SELECT role, content FROM messages WHERE client_id=? ORDER BY id DESC LIMIT ?",
            (client_id, limit),
        )
    else:
        cur = DB.execute(
            "SELECT role, content FROM messages WHERE client_id=? AND project_id=? ORDER BY id DESC LIMIT ?",
            (client_id, project_id, limit),
        )
    rows = list(cur.fetchall())[::-1]
    return [{"role": r[0], "content": r[1]} for r in rows]


# =========================
# UTILS
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    bad = ["**", "__", "##", "```", "`", "---"]
    for b in bad:
        text = text.replace(b, "")
    text = text.replace("* ", "• ")
    return text.strip()


def detect_intent(user_text: str) -> str:
    t = (user_text or "").lower().strip()
    if any(x in t for x in ["libro", "romanzo", "storia", "racconto", "capitolo", "scrivi un libro"]):
        return "AUTHOR"
    if any(x in t for x in ["motivazione", "stanco", "ansia", "vita", "disciplina", "abitudini", "crescita personale"]):
        return "COACH"
    if any(x in t for x in ["html", "css", "javascript", "python", "codice", "api", "app", "bug", "errore", "programma", "gioco"]):
        return "CODER"
    if len(t.split()) <= 3:
        return "AUTO"
    return "GENERAL"


def pick_system_prompt(mode: str) -> str:
    if mode == "AUTHOR":
        return AUTHOR_PROMPT
    if mode == "COACH":
        return COACH_PROMPT
    if mode == "CODER":
        return CODER_PROMPT
    if mode == "AUTO":
        return AUTOCORE_PROMPT
    return BASE_RULES


def groq_chat(messages: List[Dict[str, str]], temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
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


# =========================
# IMAGE + VIDEO (FREE)
# =========================
def pollinations_image(prompt: str) -> bytes:
    url = POLLINATIONS_BASE + requests.utils.quote(prompt)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def make_gif_from_prompts(prompts: List[str], duration_ms: int = 850) -> bytes:
    frames: List[Image.Image] = []
    for p in prompts:
        img_bytes = pollinations_image(p)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frames.append(img)
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return out.getvalue()


# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_pdf_text(file_bytes: bytes, max_pages: int = 40) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts: List[str] = []
    for page in reader.pages[:max_pages]:
        txt = (page.extract_text() or "").strip()
        if txt:
            texts.append(txt)
    return "\n\n".join(texts).strip()


# =========================
# OCR (A)
# =========================
def _resize_for_ocr(img: Image.Image) -> Image.Image:
    w, h = img.size
    mx = max(w, h)
    if mx <= OCR_MAX_IMAGE_SIDE:
        return img
    scale = OCR_MAX_IMAGE_SIDE / float(mx)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh))


def ocr_image_bytes(image_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (text, engine_name).
    Engine priority: EasyOCR -> pytesseract -> none.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _resize_for_ocr(img)

    # 1) EasyOCR (best, but heavy)
    try:
        import easyocr  # type: ignore

        reader = easyocr.Reader([l.strip() for l in OCR_LANGS if l.strip()], gpu=False)
        result = reader.readtext(
            image=np.array(img),  # type: ignore[name-defined]
            detail=0,
            paragraph=True,
        )
        text = "\n".join([r for r in result if r and isinstance(r, str)]).strip()
        if text:
            return text, "easyocr"
    except Exception:
        pass

    # 2) pytesseract (needs system tesseract installed)
    try:
        import pytesseract  # type: ignore

        text = (pytesseract.image_to_string(img) or "").strip()
        if text:
            return text, "pytesseract"
    except Exception:
        pass

    # 3) none
    return "", "none"


def ocr_pdf_bytes(file_bytes: bytes) -> Tuple[str, str]:
    """
    OCR PDF scans if possible.
    This requires pdf2image + poppler (system dep).
    If not available, returns ("", "none").
    """
    # Try text first
    txt = extract_pdf_text(file_bytes)
    if txt:
        return txt, "pdf_text"

    # Try pdf2image OCR
    try:
        from pdf2image import convert_from_bytes  # type: ignore

        images = convert_from_bytes(file_bytes, first_page=1, last_page=min(OCR_PDF_MAX_PAGES, 10))
        chunks = []
        engine_used = "none"
        for img in images:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            t, eng = ocr_image_bytes(buf.getvalue())
            if t:
                chunks.append(t)
            if eng != "none":
                engine_used = eng
        return "\n\n".join(chunks).strip(), f"pdf_ocr_{engine_used}"
    except Exception:
        return "", "none"


# numpy optional for easyocr path
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


# =========================
# API MODELS
# =========================
class ChatRequest(BaseModel):
    message: str
    client_id: Optional[str] = None


class ProjectCreateRequest(BaseModel):
    client_id: Optional[str] = None
    name: str
    type: str  # BOOK / COACH / CODER / GENERAL


class ProjectSelectRequest(BaseModel):
    client_id: Optional[str] = None
    project_id: int


class ImageRequest(BaseModel):
    prompt: str


class VideoRequest(BaseModel):
    prompt: str


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


# ===== PROJECTS (B) =====
@app.post("/projects/create")
def projects_create(req: ProjectCreateRequest):
    client_id = (req.client_id or "default").strip()
    name = (req.name or "").strip() or "Progetto"
    ptype = (req.type or "GENERAL").strip().upper()

    if ptype not in {"BOOK", "COACH", "CODER", "GENERAL"}:
        ptype = "GENERAL"

    # state iniziale
    state = {}
    if ptype == "BOOK":
        state = {"chapter": 1, "title": "", "style": "narrativo"}
    elif ptype == "COACH":
        state = {"goal": "", "plan": "base"}
    elif ptype == "CODER":
        state = {"stack": "web", "notes": ""}

    pid = create_project(client_id, name, ptype, state)
    set_active_project(client_id, pid)
    return {"ok": True, "project_id": pid}


@app.get("/projects/list")
def projects_list(client_id: str = "default"):
    client_id = (client_id or "default").strip()
    get_or_create_session(client_id)
    return {"projects": list_projects(client_id), "active_project_id": get_or_create_session(client_id)["active_project_id"]}


@app.post("/projects/select")
def projects_select(req: ProjectSelectRequest):
    client_id = (req.client_id or "default").strip()
    pr = get_project(client_id, req.project_id)
    if not pr:
        return {"ok": False, "error": "Progetto non trovato"}
    set_active_project(client_id, req.project_id)
    return {"ok": True, "active_project_id": req.project_id}


@app.get("/projects/current")
def projects_current(client_id: str = "default"):
    client_id = (client_id or "default").strip()
    sess = get_or_create_session(client_id)
    pid = sess["active_project_id"]
    if not pid:
        return {"active_project_id": None, "project": None}
    return {"active_project_id": pid, "project": get_project(client_id, int(pid))}


# ===== CHAT (AUTOCORE + PROGETTI) =====
@app.post("/chat")
def chat(req: ChatRequest):
    client_id = (req.client_id or "default").strip()
    user_text = (req.message or "").strip()
    if not user_text:
        return {"text": "Scrivi qualcosa e parto subito."}

    sess = get_or_create_session(client_id)
    active_pid = sess["active_project_id"]
    active_project = get_project(client_id, int(active_pid)) if active_pid else None

    # Intent automatico
    intent = detect_intent(user_text)

    # Se c'è progetto attivo, usa il suo tipo come “bias”
    if active_project:
        if active_project["type"] == "BOOK":
            intent = "AUTHOR"
        elif active_project["type"] == "COACH":
            intent = "COACH"
        elif active_project["type"] == "CODER":
            intent = "CODER"

    sys_prompt = pick_system_prompt(intent)

    # gestione libro: "continua" = capitolo successivo nel progetto
    if active_project and active_project["type"] == "BOOK":
        state = active_project["state"]
        if user_text.lower() in {"continua", "vai avanti", "prosegui", "avanti", "ok"}:
            ch = int(state.get("chapter", 1))
            ch += 1
            state["chapter"] = ch
            update_project_state(client_id, active_project["id"], state)
            user_text = f"Continua il libro dal CAPITOLO {ch} mantenendo trama e stile coerenti."
        elif any(x in user_text.lower() for x in ["nuovo libro", "inizia un libro", "scrivi un libro"]):
            state["chapter"] = 1
            update_project_state(client_id, active_project["id"], state)

    # memoria conversazione per progetto attivo
    history = get_last_messages(client_id, active_project["id"] if active_project else None, limit=10)

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]
    # metti 4 messaggi recenti per continuità
    for m in history[-4:]:
        if m["role"] in {"user", "assistant"}:
            messages.append(m)
    messages.append({"role": "user", "content": user_text})

    add_msg(client_id, active_project["id"] if active_project else None, "user", user_text)
    reply = groq_chat(messages)
    reply = clean_text(reply)
    add_msg(client_id, active_project["id"] if active_project else None, "assistant", reply)

    return {"text": reply}


# ===== IMAGE (FREE) =====
@app.post("/image")
def image(req: ImageRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"url": ""}

    safe_prompt = f"high quality, detailed, realistic, {prompt}"
    img_bytes = pollinations_image(safe_prompt)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return {"url": f"data:image/png;base64,{b64}"}


# ===== VIDEO (FREE GIF 3 SCENE) =====
@app.post("/video")
def video(req: VideoRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"url": ""}

    prompts = [
        f"{prompt}, scene 1, cinematic, high quality",
        f"{prompt}, scene 2, cinematic, high quality",
        f"{prompt}, scene 3, cinematic, high quality",
    ]
    gif_bytes = make_gif_from_prompts(prompts, duration_ms=850)
    b64 = base64.b64encode(gif_bytes).decode("utf-8")
    return {"url": f"data:image/gif;base64,{b64}"}


# ===== ANALYZE (PDF + FOTO OCR) =====
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    client_id: str = Form("default"),
):
    client_id = (client_id or "default").strip()
    prompt = (prompt or "").strip()
    if not prompt:
        return {"text": "Scrivi cosa devo fare sul file (es. 'riassumi', 'estrai punti chiave', 'trova date', ecc.)."}

    content = await file.read()
    filename = (file.filename or "").lower()

    extracted = ""
    engine = "none"

    if filename.endswith(".pdf"):
        extracted, engine = ocr_pdf_bytes(content)
        if not extracted:
            return {
                "text": (
                    "Non riesco a leggere testo dal PDF. "
                    "Se è una scansione immagine, serve OCR completo. "
                    "Posso attivarlo meglio installando pdf2image + poppler su server."
                )
            }

    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        extracted, engine = ocr_image_bytes(content)
        if not extracted:
            return {
                "text": (
                    "OCR non disponibile sul server. "
                    "Per attivarlo: installa easyocr (pesante) oppure pytesseract + tesseract."
                )
            }
    else:
        return {"text": "Formato non supportato. Carica PDF o immagine."}

    messages = [
        {"role": "system", "content": PDF_ANALYST_PROMPT},
        {"role": "user", "content": f"RICHIESTA UTENTE: {prompt}\n\nTESTO ESTRATTO ({engine}):\n{extracted[:14000]}"},
    ]
    reply = groq_chat(messages, temperature=0.4, max_tokens=1400)
    return {"text": clean_text(reply)}
