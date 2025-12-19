from langchain_core.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Ты — профессор, эксперт по книге. Отвечай строго по контексту.\n"
        "Если информации недостаточно — скажи, что точный ответ невозможен.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )
)

quiz_generate_prompt = PromptTemplate(
    input_variables=["context", "topic"],
    template="""
Ты — преподаватель, создающий учебный квиз по теме: {topic}.
Используй только контекст.

Сгенерируй 5 вопросов с 4 вариантами ответов.
Формат строго JSON:

[
  {{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "..."
  }}
]

Не копируй текст из контекста дословно.
Перемешай варианты.
Контекст:
{context}
"""
)
