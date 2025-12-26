from __future__ import annotations

import base64
import io
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions").strip()

# Modello: usa quello che funziona nel tuo account.
# Se vedi 400, cambia MODEL da Render (ENV) senza toccare codice.
MODEL = os.getenv("MODEL", "llama3-8b-8192").strip()

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.75"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1600"))

DB_PATH = os.getenv("SQLITE_PATH", "data.sqlite3").strip()

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"

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
- Se l'utente chiede codice: consegna codice completo pronto da copiare (HTML/JS/Python ecc).
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
- Analizza il contenuto fornito (testo estratto da PDF).
- Produci: RIASSUNTO, PUNTI CHIAVE, AZIONI/INSIGHT, e se richiesto estrai dati/tabelle in modo testuale.
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
# DB (MEMORIA PERSISTENTE)
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            client_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            book_title TEXT,
            book_style TEXT,
            book_chapter INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
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

DB = db()

def now_ts() -> int:
    return int(time.time())

def get_session(client_id: str) -> Dict[str, Any]:
    cur = DB.execute(
        "SELECT mode, book_title, book_style, book_chapter, updated_at FROM sessions WHERE client_id=?",
        (client_id,),
    )
    row = cur.fetchone()
    if not row:
        st = {
            "mode": "GENERAL",
            "book_title": None,
            "book_style": None,
            "book_chapter": 1,
            "updated_at": now_ts(),
        }
        DB.execute(
            "INSERT INTO sessions (client_id, mode, book_title, book_style, book_chapter, updated_at) VALUES (?,?,?,?,?,?)",
            (client_id, st["mode"], st["book_title"], st["book_style"], st["book_chapter"], st["updated_at"]),
        )
        DB.commit()
        return st
    return {
        "mode": row[0],
        "book_title": row[1],
        "book_style": row[2],
        "book_chapter": int(row[3]),
        "updated_at": int(row[4]),
    }

def set_session(client_id: str, st: Dict[str, Any]) -> None:
    DB.execute(
        """
        INSERT INTO sessions (client_id, mode, book_title, book_style, book_chapter, updated_at)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(client_id) DO UPDATE SET
            mode=excluded.mode,
            book_title=excluded.book_title,
            book_style=excluded.book_style,
            book_chapter=excluded.book_chapter,
            updated_at=excluded.updated_at
        """,
        (
            client_id,
            st.get("mode", "GENERAL"),
            st.get("book_title"),
            st.get("book_style"),
            int(st.get("book_chapter", 1)),
            now_ts(),
        ),
    )
    DB.commit()

def add_msg(client_id: str, role: str, content: str) -> None:
    DB.execute(
        "INSERT INTO messages (client_id, role, content, created_at) VALUES (?,?,?,?)",
        (client_id, role, content, now_ts()),
    )
    DB.commit()

def get_last_messages(client_id: str, limit: int = 8) -> List[Dict[str, str]]:
    cur = DB.execute(
        "SELECT role, content FROM messages WHERE client_id=? ORDER BY id DESC LIMIT ?",
        (client_id, limit),
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

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
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

def pollinations_image(prompt: str) -> bytes:
    url = POLLINATIONS_BASE + requests.utils.quote(prompt)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def make_gif_from_prompts(prompts: List[str], duration_ms: int = 800) -> bytes:
    frames: List[Image.Image] = []
    for p in prompts:
        img_bytes = pollinations_image(p)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frames.append(img)

    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return out.getvalue()

def extract_pdf_text(file_bytes: bytes, max_pages: int = 30) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts: List[str] = []
    for i, page in enumerate(reader.pages[:max_pages]):
        txt = page.extract_text() or ""
        txt = txt.strip()
        if txt:
            texts.append(txt)
    return "\n\n".join(texts).strip()

# =========================
# API MODELS
# =========================
class ChatRequest(BaseModel):
    message: str
    client_id: Optional[str] = None

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

@app.post("/chat")
def chat(req: ChatRequest):
    client_id = (req.client_id or "default").strip()
    user_text = (req.message or "").strip()
    if not user_text:
        return {"text": "Scrivi qualcosa e parto subito."}

    st = get_session(client_id)
    intent = detect_intent(user_text)

    # Se user dice "continua" e ultima modalità era AUTHOR, continua il libro
    if user_text.lower() in {"continua", "vai avanti", "prosegui", "avanti", "ok"} and st.get("mode") == "AUTHOR":
        st["book_chapter"] = int(st.get("book_chapter", 1)) + 1
        intent = "AUTHOR"
        user_text = f"Continua il libro dal CAPITOLO {st['book_chapter']} mantenendo trama e stile coerenti."

    # Se user chiede libro inizia capitolo 1 (reset)
    if intent == "AUTHOR" and any(x in user_text.lower() for x in ["scrivi un libro", "nuovo libro", "inizia un libro", "romanzo"]):
        st["book_chapter"] = 1
        st["book_title"] = None
        st["book_style"] = None

    st["mode"] = intent
    set_session(client_id, st)

    sys_prompt = pick_system_prompt(intent)

    # Memoria conversazione: ultimi messaggi (pochi) per continuità
    history = get_last_messages(client_id, limit=6)

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]

    # metti un po' di history, ma senza “sporcare” troppo
    for m in history[-4:]:
        if m["role"] in {"user", "assistant"}:
            messages.append(m)

    messages.append({"role": "user", "content": user_text})

    add_msg(client_id, "user", user_text)
    reply = groq_chat(messages)
    reply = clean_text(reply)
    add_msg(client_id, "assistant", reply)

    return {"text": reply}

@app.post("/image")
def image(req: ImageRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"url": ""}

    # miglioramento automatico prompt
    safe_prompt = f"high quality, detailed, realistic, {prompt}"
    img_bytes = pollinations_image(safe_prompt)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return {"url": f"data:image/png;base64,{b64}"}

@app.post("/video")
def video(req: VideoRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"url": ""}

    # video breve = GIF con 3 scene
    prompts = [
        f"{prompt}, scene 1, cinematic",
        f"{prompt}, scene 2, cinematic",
        f"{prompt}, scene 3, cinematic",
    ]
    gif_bytes = make_gif_from_prompts(prompts, duration_ms=850)
    b64 = base64.b64encode(gif_bytes).decode("utf-8")
    return {"url": f"data:image/gif;base64,{b64}"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), prompt: str = Form(...), client_id: str = Form("default")):
    prompt = (prompt or "").strip()
    if not prompt:
        return {"text": "Scrivi cosa devo fare sul file (es. 'riassumi', 'estrai punti chiave', 'trova date', ecc.)."}

    content = await file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        extracted = extract_pdf_text(content)
        if not extracted:
            return {"text": "Non riesco a leggere testo dal PDF (potrebbe essere scansione immagine). Se vuoi, posso aggiungere OCR."}

        messages = [
            {"role": "system", "content": PDF_ANALYST_PROMPT},
            {"role": "user", "content": f"RICHIESTA UTENTE: {prompt}\n\nTESTO PDF ESTRATTO:\n{extracted[:12000]}"},
        ]
        reply = groq_chat(messages, temperature=0.4, max_tokens=1400)
        return {"text": clean_text(reply)}

    # Immagini: senza modello vision, non posso descrivere contenuto visivo.
    # Posso però implementare OCR se vuoi (step successivo).
    if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return {"text": "Analisi immagini: al momento Groq chat non vede foto. Se vuoi, aggiungo OCR (legge testo dentro foto) oppure integriamo un modello vision."}

    return {"text": "Formato non supportato. Carica PDF o immagine."}
