import json
import random
from typing import List, Optional

from backend.faiss_client import faiss_search
from backend.llm import get_llm
from backend.prompts import quiz_generate_prompt
from backend.rag import retrieve_context

llm = get_llm()


# ---------- utils ----------

def extract_first_json_array(text: str) -> Optional[list]:
    """
    Извлекает первый JSON-массив из текста LLM.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# ---------- topics ----------

def auto_generate_quiz_topics(
    n_chunks: int = 10,
    book: str | None = None
) -> List[str]:
    """
    Автоматически генерирует темы квиза по книге.
    """
    data = faiss_search(
        query="главы содержание структура",
        k=50,
        source=book
    )

    if not data:
        return []

    random.shuffle(data)
    selected = data[:n_chunks]

    context = "\n".join(
        item.get("text", "") for item in selected if item.get("text")
    )

    prompt = f"""
Проанализируй текст из фрагментов книги.
Сформируй ровно 5 ключевых тем / глав
в формате JSON массива:

["...", "...", "...", "...", "..."]

Контекст:
{context}
    """

    res = llm.invoke(prompt).content
    topics = extract_first_json_array(res)

    if not topics:
        raise ValueError("Не удалось извлечь темы квиза")

    # убираем дубли и мусор
    return list({t.strip() for t in topics if isinstance(t, str) and t.strip()})


# ---------- quiz ----------

def generate_quiz(
    topic: str,
    book: str | None = None
) -> list:
    """
    Генерирует квиз по теме и книге.
    """
    data = faiss_search(
        query=topic,
        k=6,
        source=book
    )

    if not data:
        raise ValueError("Не найден контекст для квиза")

    random.shuffle(data)
    selected = data[:4]

    context = "\n---\n".join(
        item.get("text", "")[:3500] for item in selected
    )

    raw = llm.invoke(
        quiz_generate_prompt.format(
            context=context,
            topic=topic
        )
    ).content

    quiz = extract_first_json_array(raw)
    if not quiz:
        raise ValueError("Ошибка парсинга квиза")

    # перемешиваем варианты ответов
    for q in quiz:
        if "options" in q:
            random.shuffle(q["options"])

    return quiz[:5]


# ---------- explanation ----------

def explain_correct_answer(
    answer: str,
    book: str | None = None
) -> str:
    """
    Объясняет правильный ответ на основе контекста из книги.
    """
    context, _, _, _ = retrieve_context(
        query=answer,
        n=3,
        book=book
    )

    prompt = f"""
Ты преподаватель.
Объясни правильный ответ понятным языком,
опираясь на контекст из книги.

Контекст:
{context}

Объяснение:
    """

    return llm.invoke(prompt).content