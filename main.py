@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    @app.post("/chat")
def chat(req: ChatRequest):
    return {
        "text": f"Hai scritto: {req.message}"
    }
