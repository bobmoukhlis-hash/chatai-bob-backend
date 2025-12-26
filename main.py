@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "text": f"Hai scritto: {req.message}"
    }
