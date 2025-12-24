from backend.faiss_client import faiss_search
from backend.llm import get_llm
from backend.prompts import rag_prompt

llm = get_llm()


def retrieve_context(query: str, n: int = 5, book: str | None = None):
    data = faiss_search(
        query=query,
        k=n,
        source=book
    )

    if not data:
        return "", [], [], []

    texts = [item["text"] for item in data]
    metas = [item.get("metadata") for item in data]
    scores = [
    round(item.get("score", 0.0), 2)
    if item.get("score") is not None else None
    for item in data
    ]

    return "\n".join(texts), texts, metas, scores


def get_rag_answer(question: str, book: str | None = None):
    """
    Возвращает ответ RAG, а также контекстные фрагменты.
    Можно указать конкретную книгу через book.
    """
    context, docs, metas, scores = retrieve_context(question, book=book)
    response = llm.invoke(
        rag_prompt.format(context=context, question=question)
    ).content
    return response, docs, metas, scores

