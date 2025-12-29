from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from PIL import Image
import os, io, base64, requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY   = os.getenv("HF_API_KEY")
ADMIN_KEY    = os.getenv("ADMIN_KEY")

MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")
HF_IMAGE_MODEL  = os.getenv("HF_IMAGE_MODEL", "stabilityai/sdxl-turbo")
HF_VISION_MODEL = os.getenv("HF_VISION_MODEL", "Salesforce/blip-image-captioning-large")

groq = Groq(api_key=GROQ_API_KEY)

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

@app.get("/health")
def health():
    return {"status": "ok"}

class ChatReq(BaseModel):
    message: str
    client_id: str

@app.post("/chat")
def chat(req: ChatReq):
    res = groq.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Sei ChatAI Bob, rispondi in modo naturale."},
            {"role": "user", "content": req.message}
        ]
    )
    return {"text": res.choices[0].message.content}

def style_prompt(style: str) -> str:
    if style == "anime":
        return "anime style, sharp lines, vibrant colors"
    if style == "realistic":
        return "photorealistic, ultra detailed, cinematic lighting"
    return "3D cartoon, pixar style, soft lighting"

class ImageReq(BaseModel):
    prompt: str
    client_id: str
    style: str | None = "cartoon"

@app.post("/image")
def image(req: ImageReq):
    style = req.style or "cartoon"
    final_prompt = f"{style_prompt(style)}, no text, no watermark, {req.prompt}"

    r = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
        headers=HF_HEADERS,
        json={"inputs": final_prompt},
        timeout=60
    )

    if r.status_code != 200:
        return {"error": "Errore generazione immagine"}

    img = Image.open(io.BytesIO(r.content))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return {
        "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    }
class VideoReq(BaseModel):
    prompt: str
    client_id: str
    style: str | None = "cartoon"

@app.post("/video")
def video(req: VideoReq):
    style = req.style or "cartoon"
    base = f"{style_prompt(style)}, same character, smooth motion, {req.prompt}"

    frames = []
    for _ in range(4):
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
            headers=HF_HEADERS,
            json={"inputs": base},
            timeout=60
        )
        frames.append(Image.open(io.BytesIO(r.content)))

    out = io.BytesIO()
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=600,
        loop=0
    )

    return {
        "url": "data:image/gif;base64," + base64.b64encode(out.getvalue()).decode()
    }

@app.post("/analyze_photo")
async def analyze_photo(
    file: UploadFile = File(...),
    question: str = Form(""),
    client_id: str = Form(...)
):
    img_bytes = await file.read()

    r = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_VISION_MODEL}",
        headers=HF_HEADERS,
        files={"file": img_bytes},
        timeout=60
    )

    if r.status_code != 200:
        return {"text": "Errore analisi immagine"}

    caption = r.json()[0].get("generated_text", "")
    if question:
        return {"text": f"Descrizione: {caption}\n\nDomanda: {question}"}
    return {"text": caption}


