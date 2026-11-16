"""Minimal document RAG API with local TF-IDF retrieval."""
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Enterprise RAG Knowledge Assistant", version="1.0.0")
chunks: List[dict] = []
vectorizer = TfidfVectorizer(stop_words="english")
matrix = None

class TextInput(BaseModel):
    name: str = "manual.txt"
    text: str = Field(min_length=20, max_length=200000)

class AskInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

def add_document(name: str, text: str):
    global matrix
    words = text.split()
    parts = [" ".join(words[i:i+180]) for i in range(0, len(words), 150)]
    for part in parts:
        if part.strip():
            chunks.append({"source": name, "text": part})
    if chunks:
        matrix = vectorizer.fit_transform([item["text"] for item in chunks])
    return len(parts)

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(chunks)}

@app.post("/ingest/text")
def ingest_text(body: TextInput):
    count = add_document(body.name, body.text)
    return {"message": "Document indexed", "chunks_added": count}

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    data = await file.read()
    import io
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise HTTPException(400, "No extractable text found in PDF.")
    count = add_document(file.filename, text)
    return {"message": "PDF indexed", "chunks_added": count}

@app.post("/ask")
def ask(body: AskInput):
    if not chunks or matrix is None:
        raise HTTPException(400, "Ingest a document before asking questions.")
    query_vec = vectorizer.transform([body.question])
    scores = cosine_similarity(query_vec, matrix).ravel()
    indexes = scores.argsort()[::-1][:body.top_k]
    sources = [{"source": chunks[i]["source"], "score": round(float(scores[i]), 4), "excerpt": chunks[i]["text"]} for i in indexes if scores[i] > 0]
    if not sources:
        return {"answer": "I could not find relevant information in the indexed documents.", "sources": []}
    context = "\n\n".join(f"Source: {s['source']}\n{s['excerpt']}" for s in sources)
    answer = "\n\n".join(f"From {s['source']}: {s['excerpt']}" for s in sources)
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        response = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer only from the supplied context. If the answer is absent, say so. Cite source filenames."},
                {"role": "user", "content": f"Question: {body.question}\nContext:\n{context}"},
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content or answer
    return {"answer": answer, "sources": sources}
