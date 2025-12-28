# main.py
from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ================= CONFIG =================
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0").strip()
HF_CAPTION_MODEL = os.getenv("HF_CAPTION_MODEL", "Salesforce/blip-image-captioning-large").strip()

HF_INFER_BASE = "https://api-inference.huggingface.co/models/"

LIMITS_FREE = {"chat": 15, "image": 3, "video": 3, "photo": 5}
# Per produzione: collega limiti a DB (qui minimal: in-memory per semplicità)
FREE_USAGE: Dict[str, Dict[str, int]] = {}  # {client_id: {day:count...}} semplificato

# ================= UTILS =================
def _day_key() -> str:
    return str(int(time.time() // 86400))

def _free_get(client_id: str, action: str) -> int:
    d = _day_key()
    FREE_USAGE.setdefault(client_id, {})
    FREE_USAGE[client_id].setdefault(d, 0)
    # separazione azioni semplice: client_id|action
    key = f"{d}:{action}"
    return FREE_USAGE[client_id].get(key, 0)

def _free_inc(client_id: str, action: str) -> None:
    d = _day_key()
    key = f"{d}:{action}"
    FREE_USAGE.setdefault(client_id, {})
    FREE_USAGE[client_id][key] = FREE_USAGE[client_id].get(key, 0) + 1

def _free_can(client_id: str, action: str) -> bool:
    limit = int(LIMITS_FREE.get(action, 0))
    return _free_get(client_id, action) < limit

def _groq_key() -> str:
    return (os.getenv("GROQ_API_KEY") or "").strip()

def _hf_headers() -> Dict[str, str]:
    if not HF_TOKEN:
        return {}
    return {"Authorization": f"Bearer {HF_TOKEN}"}

def _data_url(mime: str, b: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

def _clean_markdown_like(text: str) -> str:
    if not text:
        return ""
    for bad in ["```", "##", "**", "__"]:
        text = text.replace(bad, "")
    return text.strip()

# ================= GROQ CHAT =================
BASE_RULES = (
    "Sei ChatAI Bob.\n"
    "Regole:\n"
    "- Rispondi naturale e chiaro.\n"
    "- Se l’utente chiede codice: fornisci codice completo.\n"
    "- Se chiede HTML/CSS/JS: fornisci file completi.\n"
    "- Rispondi nella lingua dell’utente.\n"
)

def groq_chat(user_text: str, extra_context: str = "") -> str:
    key = _groq_key()
    if not key:
        return "Servizio chat non disponibile (manca GROQ_API_KEY)."

    sys = BASE_RULES
    if extra_context:
        sys += "\nCONTESTO EXTRA:\n" + extra_context

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.8,
        "max_tokens": 1400,
    }
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    if not r.ok:
        return f"Errore chat ({r.status_code})."
    data = r.json()
    out = data["choices"][0]["message"]["content"]
    return _clean_markdown_like(out)

# ================= HF: TEXT->IMAGE =================
def hf_text_to_image(prompt: str, timeout_s: int = 120) -> bytes:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN mancante")

    url = HF_INFER_BASE + HF_IMAGE_MODEL
    # Inference API: per molti modelli text-to-image accetta JSON {"inputs": "..."}
    r = requests.post(
        url,
        headers={**_hf_headers(), "Accept": "image/png", "Content-Type": "application/json"},
        json={"inputs": prompt},
        timeout=timeout_s,
    )
    if r.status_code == 503:
        # model loading
        raise RuntimeError("Modello immagini in avvio (riprovare).")
    if not r.ok:
        raise RuntimeError(f"HF image error {r.status_code}: {r.text[:200]}")
    return r.content

# ================= HF: IMAGE CAPTION =================
def hf_caption_image(image_bytes: bytes, timeout_s: int = 60) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN mancante")

    url = HF_INFER_BASE + HF_CAPTION_MODEL
    # Image-to-text: spesso accetta raw bytes come body
    r = requests.post(
        url,
        headers={**_hf_headers(), "Accept": "application/json"},
        data=image_bytes,
        timeout=timeout_s,
    )
    if r.status_code == 503:
        raise RuntimeError("Modello analisi foto in avvio (riprovare).")
    if not r.ok:
        raise RuntimeError(f"HF caption error {r.status_code}: {r.text[:200]}")

    data = r.json()
    # BLIP ritorna tipicamente [{"generated_text":"..."}]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("generated_text", "")).strip()
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]).strip()
    return str(data).strip()

# ================= GIF (VIDEO) =================
def make_gif_from_prompts(prompts: List[str], duration_ms: int = 650) -> bytes:
    frames: List[Image.Image] = []
    for p in prompts:
        img_b = hf_text_to_image(p, timeout_s=140)
        img = Image.open(io.BytesIO(img_b)).convert("RGB")
        frames.append(img)

    out = io.BytesIO()
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out.getvalue()

# ================= API MODELS =================
class ChatRequest(BaseModel):
    message: str
    client_id: Optional[str] = "client_anon"

class ImageRequest(BaseModel):
    prompt: str
    client_id: Optional[str] = "client_anon"

class VideoRequest(BaseModel):
    prompt: str
    client_id: Optional[str] = "client_anon"

# ================= APP =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # per Play Store: restringi al tuo dominio/app
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, str]:
    client_id = (req.client_id or "client_anon").strip()

    if not _free_can(client_id, "chat"):
        return {"text": "Limite chat FREE raggiunto. Passa a Premium."}

    _free_inc(client_id, "chat")
    reply = groq_chat(req.message)
    return {"text": reply}

@app.post("/image")
def image(req: ImageRequest) -> Dict[str, str]:
    client_id = (req.client_id or "client_anon").strip()

    if not _free_can(client_id, "image"):
        return {"error": "Limite immagini FREE raggiunto. Passa a Premium."}

    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"error": "Scrivi un prompt per l’immagine."}

    # prompt pulito, senza watermark/loghi
    enhanced = (
        "high quality, ultra detailed, clean composition, "
        "no text, no watermark, no logo, no brand, "
        + prompt
    )

    try:
        _free_inc(client_id, "image")
        img = hf_text_to_image(enhanced)
        return {"url": _data_url("image/png", img)}
    except Exception:
        return {"error": "Errore generazione immagine (servizio occupato o offline). Riprova."}

@app.post("/video")
def video(req: VideoRequest) -> Dict[str, str]:
    client_id = (req.client_id or "client_anon").strip()

    if not _free_can(client_id, "video"):
        return {"error": "Limite video FREE raggiunto. Passa a Premium."}

    prompt = (req.prompt or "").strip()
    if not prompt:
        return {"error": "Scrivi un prompt per il video/GIF."}

    # GIF = 4 frame con “movimento”
    # (stabile in produzione: niente “vero video model”)
    base = (
        "high quality, ultra detailed, clean composition, "
        "no text, no watermark, no logo, no brand, "
    )
    motion = "same character, consistent face, consistent outfit, smooth motion"
    frames = [
        f"{base}{motion}, frame 1, start of action, {prompt}",
        f"{base}{motion}, frame 2, mid action, {prompt}",
        f"{base}{motion}, frame 3, continue action, {prompt}",
        f"{base}{motion}, frame 4, end action, {prompt}",
    ]

    try:
        _free_inc(client_id, "video")
        gif = make_gif_from_prompts(frames, duration_ms=650)
        return {"url": _data_url("image/gif", gif)}
    except Exception:
        return {"error": "Errore generazione GIF (servizio occupato o offline). Riprova."}

@app.post("/analyze_photo")
async def analyze_photo(
    file: UploadFile = File(...),
    question: str = Form(""),
    client_id: str = Form("client_anon"),
) -> Dict[str, str]:
    client_id = (client_id or "client_anon").strip()

    if not _free_can(client_id, "photo"):
        return {"text": "Limite analisi foto FREE raggiunto. Passa a Premium."}

    content = await file.read()
    if not content:
        return {"text": "File vuoto."}

    q = (question or "").strip() or "Descrivi la foto in modo dettagliato e dimmi cosa è importante."
    try:
        _free_inc(client_id, "photo")
        caption = hf_caption_image(content)
        # Risposta “intelligente”: caption + domanda -> Groq
        ctx = f"DESCRIZIONE FOTO (da modello visivo): {caption}"
        reply = groq_chat(q, extra_context=ctx)
        return {"text": reply, "caption": caption}
    except Exception:
        return {"text": "Errore analisi foto (servizio occupato o offline). Riprova."}
