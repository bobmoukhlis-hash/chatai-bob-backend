from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# -------- MODELLI --------
class ChatRequest(BaseModel):
    message: str

# -------- ROOT --------
@app.get("/")
def root():
    return {"status": "ok"}

# ✅ QUESTO È IL BLOCCO CHE CHIEDEVI
@app.get("/chat")
def chat_get():
    return {"status": "chat endpoint OK (use POST)"}

# -------- CHAT (POST) --------
@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "text": f"Hai scritto: {req.message}"
    }
