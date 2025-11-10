import os
from pathlib import Path
import toml  

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# ШАГ 1: НАСТРОЙКА И ЗАГРУЗКА RAG-СИСТЕМЫ

PROJECT_ROOT = Path.cwd()
FAISS_INDEX_PATH = str(PROJECT_ROOT / "faiss_index")
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"

# Функция для загрузки ключа из secrets.toml 
def load_api_key(secrets_path):
    """Загружает GROQ_API_KEY из файла secrets.toml."""
    try:
        secrets = toml.load(secrets_path)
        return secrets.get("GROQ_API_KEY")
    except FileNotFoundError:
        print(f"Ошибка: Файл секретов не найден по пути {secrets_path}")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла секретов: {e}")
        return None

print("Загрузка API ключа")
api_key = load_api_key(SECRETS_PATH)
if not api_key:
    print("Не удалось загрузить API ключ.")
    exit() 


# Загрузка компонентов RAG 
print("Загрузка компонентов RAG-системы")
embedding_model = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Ты — интеллектуальный помощник, эксперт по содержимому книги.\n"
        "Используй приведённый контекст, чтобы ответить на вопрос пользователя.\n"
        "Если в контексте нет нужной информации — скажи, что ты не уверен.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )
)

# СОЗДАНИЕ ТЕСТОВОГО НАБОРА ДАННЫХ
print("Создание тестового набора данных")
test_questions = [
    "Что такое регуляризация и зачем она нужна?",
    "Объясни разницу между L1 и L2 регуляризацией.",
    "Какова роль функции активации в нейронной сети?",
    "Что такое переобучение?",
]
ground_truths = [
    "Регуляризация - это любая модификация алгоритма обучения, направленная на уменьшение его ошибки обобщения, но не ошибки обучения. Она нужна для борьбы с переобучением.",
    "L1 регуляризация добавляет к функции потерь сумму модулей весов и приводит к разреженным весам (обнулению некоторых из них), в то время как L2 регуляризация добавляет сумму квадратов весов и приводит к малым, но ненулевым весам.",
    "Функция активации вводит нелинейность в модель, позволяя нейронной сети изучать сложные зависимости в данных, которые не могут быть смоделированы линейными функциями.",
    "Переобучение — это явление, при котором модель машинного обучения слишком хорошо 'запоминает' обучающие данные, включая случайный шум и выбросы, вместо того чтобы изучать общие закономерности. В результате такая модель показывает отличную производительность на обучающей выборке, но очень плохо обобщается и делает много ошибок на новых, невиданных ранее данных.",
]


# ЗАПУСК RAG КОНВЕЙЕРА ДЛЯ ПОЛУЧЕНИЯ ОТВЕТОВ
print("Генерация ответов для тестовых вопросов")
answers = []
contexts = []

for question in test_questions:
    results = db.similarity_search(question, k=3)
    retrieved_contexts = [doc.page_content for doc in results]
    contexts.append(retrieved_contexts)

    context_str = "\n---\n".join(retrieved_contexts)
    prompt_text = prompt_template.format(context=context_str, question=question)
    generated_answer = llm.invoke(prompt_text)
    answers.append(generated_answer.content)

# ОЦЕНКА С ПОМОЩЬЮ RAGAS
print("Подготовка данных для оценки")
ragas_data = {
    "question": test_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}
dataset = Dataset.from_dict(ragas_data)

print("Запуск оценки RAGAs")
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

result = evaluate(
    dataset=dataset, 
    metrics=metrics,
    llm=llm,
    embeddings=embedding_model
)

print("="*50)
print("РЕЗУЛЬТАТЫ ОЦЕНКИ КАЧЕСТВА RAG:")
print("="*50)
print(result)
print("="*50)