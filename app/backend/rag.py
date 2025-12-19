from backend.faiss_client import faiss_search
from backend.llm import get_llm
from backend.prompts import rag_prompt

llm = get_llm()


def retrieve_context(query: str, n: int = 5):
    data = faiss_search(query, k=n)
    if not data:
        return "", [], [], []

    docs = [item["text"] for item in data]
    metas = [item["metadata"] for item in data]
    scores = [item.get("score", 0.0) for item in data] 

    return "\n---\n".join(docs), docs, metas, scores


def get_rag_answer(question: str):
    context, docs, metas, scores = retrieve_context(question)
    response = llm.invoke(
        rag_prompt.format(context=context, question=question)
    ).content
    return response, docs, metas, scores
