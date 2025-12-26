from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# -------- MODELLI --------
class ChatRequest(BaseModel):
    message: str

# -------- ROUTE BASE --------
@app.get("/")
def root():
    return {"status": "ok"}

# -------- CHAT --------
@app.post("/chat")
def chat(req: ChatRequest):
    # risposta di test (poi collegherai Groq)
   return {
    "reply": f"Hai scritto: {req.message}"
}
