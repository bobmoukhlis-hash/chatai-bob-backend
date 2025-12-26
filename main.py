from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ok"}

# 🔑 QUESTA ROUTE SERVE AL FRONTEND
fetch(API_BASE + "/health")
  .then(r => r.json())
  .then(d => {
    if (d.status === "ok") {
      setBackendOnline(true);
    } else {
      setBackendOnline(false);
    }
  })
  .catch(() => setBackendOnline(false));
