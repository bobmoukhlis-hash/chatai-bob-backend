from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ok"}

# 🔑 QUESTA ROUTE SERVE AL FRONTEND
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "reply": f"Hai scritto: {req.message}"
    }
