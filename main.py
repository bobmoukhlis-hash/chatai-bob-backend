from __future__ import annotations

import os
import io
import base64
import requests
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()

MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile").strip()
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "stabilityai/sdxl-turbo").strip()
HF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "Salesforce/blip-image-captioning-large").strip()

groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

HF_TIMEOUT = 60


@app.get("/health")
def health():
    ok = bool(GROQ_API_KEY) and bool(HF_API_KEY)
    return {"status": "ok", "keys": "ok" if ok else "missing"}


class ChatReq(BaseModel):
    message: str
    client_id: str


@app.post("/chat")
def chat(req: ChatReq):
    if not groq:
        return {"text": "Servizio non disponibile."}

    try:
        res = groq.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Sei ChatAI Bob, rispondi in modo naturale, chiaro e utile."},
                {"role": "user", "content": req.message},
            ],
        )
        return {"text": res.choices[0].message.content}
    except Exception:
        return {"text": "Servizio non disponibile."}


def style_prompt(style: str) -> str:
    s = (style or "").strip().lower()
    if s == "anime":
        return "anime style, sharp lines, vibrant colors"
    if s == "realistic":
        return "photorealistic, ultra detailed, cinematic lighting"
    return "3D cartoon, pixar style, soft lighting"


def hf_image_bytes(prompt: str) -> tuple[Optional[bytes], Optional[str]]:
    if not HF_API_KEY:
        return None, "HF_API_KEY mancante"

    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
            headers=HF_HEADERS,
            json={"inputs": prompt},
            timeout=HF_TIMEOUT,
        )
    except Exception:
        return None, "Errore rete HuggingFace"

    if r.status_code != 200:
        detail = (r.text or "").strip()
        if len(detail) > 300:
            detail = detail[:300] + "..."
        return None, f"Errore HuggingFace ({r.status_code})"

    ctype = (r.headers.get("content-type") or "").lower()
    if "image" not in ctype:
        detail = (r.text or "").strip()
        if len(detail) > 300:
            detail = detail[:300] + "..."
        return None, "HuggingFace non ha restituito un'immagine"

    return r.content, None


def bytes_to_data_png(img_bytes: bytes) -> tuple[Optional[str], Optional[str]]:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None, "Immagine non valida"

    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
    except Exception:
        return None, "Errore salvataggio immagine"

    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), None


class ImageReq(BaseModel):
    prompt: str
    client_id: str
    style: Optional[str] = "cartoon"


@app.post("/image")
def image(req: ImageReq):
    style = req.style or "cartoon"
    final_prompt = f"{style_prompt(style)}, no text, no watermark, {req.prompt}".strip()

    img_bytes, err = hf_image_bytes(final_prompt)
    if err or not img_bytes:
        return {"error": "Errore generazione immagine"}

    url, err2 = bytes_to_data_png(img_bytes)
    if err2 or not url:
        return {"error": "Errore generazione immagine"}

    return {"url": url}


class VideoReq(BaseModel):
    prompt: str
    client_id: str
    style: Optional[str] = "cartoon"


@app.post("/video")
def video(req: VideoReq):
    style = req.style or "cartoon"
    base_prompt = f"{style_prompt(style)}, same character, smooth motion, no text, no watermark, {req.prompt}".strip()

    frames = []
    for _ in range(4):
        img_bytes, err = hf_image_bytes(base_prompt)
        if err or not img_bytes:
            return {"error": "Errore generazione GIF"}

        try:
            frame = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return {"error": "Errore generazione GIF"}

        frames.append(frame)

    if not frames:
        return {"error": "Errore generazione GIF"}

    out = io.BytesIO()
    try:
        frames[0].save(
            out,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=600,
            loop=0,
        )
    except Exception:
        return {"error": "Errore generazione GIF"}

    return {"url": "data:image/gif;base64," + base64.b64encode(out.getvalue()).decode()}


@app.post("/analyze_photo")
async def analyze_photo(
    file: UploadFile = File(...),
    question: str = Form(""),
    client_id: str = Form(...),
):
    if not HF_API_KEY:
        return {"text": "Servizio non disponibile."}

    img_bytes = await file.read()

    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_VISION_MODEL}",
            headers=HF_HEADERS,
            files={"file": ("image.jpg", img_bytes, file.content_type or "application/octet-stream")},
            timeout=HF_TIMEOUT,
        )
    except Exception:
        return {"text": "Errore analisi immagine"}

    if r.status_code != 200:
        return {"text": "Errore analisi immagine"}

    try:
        data = r.json()
    except Exception:
        return {"text": "Errore analisi immagine"}

    caption = ""
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        caption = str(data[0].get("generated_text", "")).strip()

    if not caption:
        return {"text": "Errore analisi immagine"}

    q = (question or "").strip()
    if q:
        return {"text": f"Descrizione: {caption}\n\nDomanda: {q}"}
    return {"text": caption}
