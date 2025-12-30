from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIG =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL = "llama-3.3-70b-versatile"

groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ===== HEALTH =====
@app.get("/health")
def health():
    return {"status": "ok" if groq else "missing"}

# ===== REQUEST =====
class ChatReq(BaseModel):
    message: str
    client_id: str

# ===== CHAT MULTILINGUA =====
@app.post("/chat")
def chat(req: ChatReq):
    if not groq:
        return {"text": "Service not available."}

    try:
        res = groq.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ChatAI Bob, a multilingual assistant. "
                        "Always reply in the SAME language used by the user. "
                        "Write clearly, naturally, and helpfully. "
                        "Answer any question on any topic."
                    )
                },
                {"role": "user", "content": req.message}
            ]
        )
        return {"text": res.choices[0].message.content}
    except Exception:
        return {"text": "Temporary service error."}
