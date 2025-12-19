import json
import random

from backend.faiss_client import faiss_search
from backend.llm import get_llm
from backend.prompts import quiz_generate_prompt
from backend.rag import retrieve_context

llm = get_llm()


def extract_first_json_array(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def auto_generate_quiz_topics(n_chunks: int = 10) -> list[str]:
    data = faiss_search("главы содержание структура", k=50)
    if not data:
        return []

    random.shuffle(data)
    selected = data[:n_chunks]

    text = "\n".join(
        item.get("text", "") for item in selected if item.get("text")
    )

    prompt = f"""
Проанализируй текст из нескольких фрагментов книги.
Сформируй ровно 5 ключевых тем / глав книги в формате JSON массива:
["...", "...", "...", "...", "..."]

Контекст:
{text}
    """

    res = llm.invoke(prompt).content
    topics = extract_first_json_array(res)
    if not topics:
        raise ValueError("Не удалось извлечь темы квиза")

    return list({t.strip() for t in topics if t.strip()})


def generate_quiz(topic: str):
    data = faiss_search(topic, k=6)
    if not data:
        return None

    random.shuffle(data)
    selected = data[:4]

    context = "\n---\n".join(
        item.get("text", "")[:3500] for item in selected
    )

    raw = llm.invoke(
        quiz_generate_prompt.format(context=context, topic=topic)
    ).content

    quiz = extract_first_json_array(raw)
    if not quiz:
        raise ValueError("Ошибка парсинга квиза")

    for q in quiz:
        random.shuffle(q["options"])

    return quiz[:5]


def explain_correct_answer(answer: str) -> str:
    context, _, _, _ = retrieve_context(answer, n=3)

    prompt = f"""
Ты преподаватель.
Объясни правильный ответ понятным языком на основе контекста.

Контекст:
{context}

Объяснение:
    """
    return llm.invoke(prompt).content

