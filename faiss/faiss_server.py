from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

INDEX_PATH = "faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="multi-qa-mpnet-base-dot-v1"
)

db = FAISS.load_local(
    INDEX_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


class Query(BaseModel):
    text: str
    k: int = 5


@app.post("/search")
def search(q: Query):
    results = db.similarity_search_with_score(q.text, k=q.k)

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        }
        for doc, score in results
    ]


