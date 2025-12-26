from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

# 🔹 FUNZIONE DI PULIZIA TESTO (QUI VA MESSA)
def clean_text(text: str) -> str:
    bad = ["**", "__", "##", "* ", "`", "---"]
    for b in bad:
        text = text.replace(b, "")
    return text.strip()

app = FastAPI()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
Sei ChatAI Bob.

MODALITÀ: RISPOSTA DIRETTA (OBBLIGATORIA)
- Rispondi SEMPRE in italiano.
- NON fare MAI domande.
- NON chiedere chiarimenti o dettagli.
- NON dire mai frasi tipo: "Come posso aiutarti?", "Prima di iniziare", "Potresti dirmi", "Vorrei sapere".
- NON dire mai: "Sono un'intelligenza artificiale", "non ho emozioni", ecc.
- Se l'utente chiede: libro/storia/testo/codice → INIZIA SUBITO A PRODURRE.
- Scegli TU genere, tono, struttura in modo professionale.
- Usa titoli, capitoli, paragrafi.
- Output lungo e completo.

Se l'utente scrive solo "Ciao" o frasi brevi:
- rispondi con una frase breve + proponi 3 opzioni (senza fare domande).
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

def looks_like_question(text: str) -> bool:
    # Heuristica: se contiene tanti "?" o frasi tipiche di domanda
    if text.count("?") >= 1:
        return True
    bad_patterns = [
        r"\bcome posso aiutarti\b",
        r"\bpotresti\b",
        r"\bvorrei sapere\b",
        r"\bprima di iniziare\b",
        r"\bmi puoi dire\b",
        r"\bpuoi dirmi\b",
        r"\bquale\b.*\?",
        r"\bche tipo\b",
    ]
    t = text.lower()
    return any(re.search(p, t) for p in bad_patterns)

@app.post("/chat")
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return {"text": "❌ GROQ_API_KEY non configurata sul server."}

    user_msg = (req.message or "").strip()
SYSTEM_PROMPT = """
Sei ChatAI Bob.

MODALITÀ TESTO PURO (OBBLIGATORIA):
- Rispondi sempre in italiano
- NON usare MAI Markdown
- NON usare **, ##, *, -, _, ` o formattazioni
- NON usare elenchi Markdown
- Usa SOLO testo normale con a capo
- Titoli scritti in MAIUSCOLO, senza simboli
- NON fare domande
- NON chiedere chiarimenti
- NON spiegare cosa stai facendo

Se l'utente chiede:
- titoli → restituisci SOLO titoli, uno per riga
- codice → restituisci SOLO codice pulito
- libro → inizia SUBITO dal contenuto

Esempio corretto:
TITOLO
Capitolo 1
Testo normale senza simboli
"""
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
reply = clean_text(reply)
return {"text": reply}
        # ✅ Anti-domande: se prova a fare domande, lo correggiamo al volo
        if looks_like_question(reply):
            fix_payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "Riscrivi la risposta SENZA alcuna domanda, iniziando SUBITO con contenuto utile e completo."}
                ],
                "temperature": 0.4,
                "max_tokens": 1600
            }
            r2 = requests.post(GROQ_URL, json=fix_payload, headers=headers, timeout=30)
            r2.raise_for_status()
            data2 = r2.json()
            reply2 = data2["choices"][0]["message"]["content"].strip()
            return {"text": reply2}

        return {"text": reply}

    except Exception as e:
        return {"text": f"⚠️ Errore Groq: {e}"}
