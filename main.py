@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "reply": f"Hai scritto: {req.message}"
    }
