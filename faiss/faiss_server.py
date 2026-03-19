import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
from pydantic import BaseModel
from pathlib import Path
import shutil
import requests
import fitz

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# ================== CONFIG ==================
INDEX_PATH = Path("faiss_index")
UPLOAD_DIR = Path("uploaded_books")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()

embedding_model = HuggingFaceEmbeddings(
    model_name="multi-qa-mpnet-base-dot-v1"
)

# ================== LOAD / INIT FAISS ==================
db: FAISS | None = None

if INDEX_PATH.exists():
    db = FAISS.load_local(
        INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# ================== UTILS ==================
def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)

    raise ValueError("Unsupported file format")


def add_text_to_faiss(text: str, source_name: str) -> int:
    global db

    splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=0.7
    )

    documents = splitter.create_documents([text])

    texts = []
    metas = []

    for i, doc in enumerate(documents):
        texts.append(doc.page_content)
        metas.append({
            "chunk_id": i,
            "source": source_name
        })

    if db is None:
        db = FAISS.from_texts(texts, embedding_model, metadatas=metas)
    else:
        db.add_texts(texts, metadatas=metas)

    db.save_local(INDEX_PATH)
    return len(texts)

# ================== API MODELS ==================
class SearchQuery(BaseModel):
    text: str
    k: int = 5
    source: str | None = None


class IngestUrlRequest(BaseModel):
    url: str

# ================== SEARCH ==================
@app.post("/search")
def search(q: SearchQuery):
    if db is None:
        return []

    if q.source:
        results = db.similarity_search_with_score(
            q.text,
            k=q.k,
            filter={"source": q.source}
        )
    else:
        results = db.similarity_search_with_score(q.text, k=q.k)

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        }
        for doc, score in results
    ]

# ================== INGEST FILE ==================
@app.post("/ingest")
def ingest_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".txt"]:
        raise HTTPException(400, "Only PDF and TXT supported")

    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    text = load_text_from_file(save_path)
    chunks = add_text_to_faiss(text, file.filename)

    return {
        "status": "ok",
        "file": file.filename,
        "chunks_added": chunks
    }

# ================== LIST BOOKS ==================
@app.get("/books")
def list_books():
    if db is None:
        return []

    sources = set()
    for doc in db.docstore._dict.values():
        src = doc.metadata.get("source")
        if src:
            sources.add(src)

    return sorted(sources)

# ================== RESET ==================
@app.post("/reset")
def reset_index():
    global db
    db = None

    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH)

    return {
        "status": "ok",
        "message": "FAISS index cleared"
    }

