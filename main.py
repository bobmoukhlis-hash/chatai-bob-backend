from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(data: dict):
    r = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": data["message"]}],
            "max_tokens": 400,
        },
        timeout=30,
    )
    return {"text": r.json()["choices"][0]["message"]["content"]}
