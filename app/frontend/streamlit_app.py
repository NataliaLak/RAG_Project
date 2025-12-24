import streamlit as st
import requests

from backend.rag import get_rag_answer
from backend.quiz import (
    auto_generate_quiz_topics,
    generate_quiz,
    explain_correct_answer
)

# ================== CONFIG ==================
st.set_page_config(page_title="📘 RAG & Quiz Ассистент", layout="wide")
st.title("📘 RAG & Quiz Ассистент")

FAISS_API_URL = "http://faiss:8001"  # ВАЖНО: имя сервиса из docker-compose

# ================== SIDEBAR: ЗАГРУЗКА КНИГ ==================
st.sidebar.header("📚 Управление книгами")

# --- Upload file ---
uploaded_file = st.sidebar.file_uploader(
    "Загрузить книгу (PDF или TXT)",
    type=["pdf", "txt"]
)

if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    try:
        res = requests.post(f"{FAISS_API_URL}/ingest", files=files)
        res.raise_for_status()
        st.sidebar.success(f"Книга '{uploaded_file.name}' загружена")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки: {e}")

# ================== LOAD BOOKS ==================
try:
    books_res = requests.get(f"{FAISS_API_URL}/books")
    books_res.raise_for_status()
    books = books_res.json()
except Exception as e:
    st.error(f"Ошибка получения списка книг: {e}")
    books = []

if not books:
    st.warning("Загрузите хотя бы одну книгу, чтобы начать работу")
    st.stop()

# ================== GLOBAL BOOK SELECTION ==================
if "selected_book" not in st.session_state:
    st.session_state.selected_book = "Все"

st.subheader("📚 Выбор книги")

st.session_state.selected_book = st.selectbox(
    "По какой книге работать:",
    ["Все"] + books,
    index=(["Все"] + books).index(st.session_state.selected_book)
)

selected_book = st.session_state.selected_book
book_param = None if selected_book == "Все" else selected_book

st.markdown(f"**Текущая книга:** `{selected_book}`")
st.divider()

# ================== TABS ==================
tab_rag, tab_quiz = st.tabs(["🔍 RAG", "🧠 Квизы"])

# ================== RAG ==================
with tab_rag:
    st.subheader("🔍 Вопрос по книге")

    question = st.text_input("Введите вопрос")

    if question:
        try:
            answer, docs, metas, scores = get_rag_answer(
                question,
                book=book_param
            )
        except Exception as e:
            st.error(f"Ошибка RAG: {e}")
            st.stop()

        st.markdown("### 💬 Ответ")
        st.write(answer)

        if docs:
            st.markdown("### 📚 Фрагменты из книги")
            for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
                st.markdown(
                    f"**Фрагмент {i+1}**  "
                    f"(source={meta.get('source')}, "
                    f"chunk={meta.get('chunk_id')}, "
                    f"score={score:.2f})"
                )
                st.write(doc[:800] + "…")
                st.divider()
        else:
            st.info("Контекст не найден")

# ================== QUIZ ==================
with tab_quiz:
    st.subheader("🧠 Квизы по книге")

    # Генерация тем (ОДИН РАЗ НА КНИГУ)
    if (
        "quiz_topics" not in st.session_state
        or st.session_state.get("quiz_book") != selected_book
    ):
        try:
            st.session_state.quiz_topics = auto_generate_quiz_topics(
                book=book_param
            )
            st.session_state.quiz_book = selected_book
        except Exception as e:
            st.error(f"Ошибка генерации тем: {e}")
            st.stop()

    topics = st.session_state.quiz_topics

    if not topics:
        st.warning("Темы не найдены для выбранной книги")
        st.stop()

    topic = st.selectbox("Выберите тему:", topics)

    if st.button("🎲 Сгенерировать квиз"):
        try:
            st.session_state.quiz = generate_quiz(
                topic,
                book=book_param
            )
            st.session_state.answers = {}
            st.session_state.checked = False
        except Exception as e:
            st.error(f"Ошибка генерации квиза: {e}")
            st.stop()

    quiz = st.session_state.get("quiz")

    if quiz:
        for i, q in enumerate(quiz):
            st.markdown(f"### ❓ {q['question']}")
            st.session_state.answers[i] = st.radio(
                "Ответ:",
                q["options"],
                key=f"q_{i}"
            )
            st.divider()

    if st.button("📊 Проверить"):
        st.session_state.checked = True

    if st.session_state.get("checked") and quiz:
        correct = 0
        for i, q in enumerate(quiz):
            user_answer = st.session_state.answers.get(i)
            correct_answer = q["answer"]

            if user_answer == correct_answer:
                st.success("Верно ✅")
                st.markdown(f"**Правильный ответ:**\n\n{correct_answer}")
                correct += 1
            else:
                st.error("Неверно ❌")
                explanation = explain_correct_answer(correct_answer)
                st.markdown(
                    f"**Вы ответили:** {user_answer}\n\n"
                    f"**Правильный ответ:** {correct_answer}\n\n"
                    f"{explanation}"
                )

        st.info(f"Результат: {correct} / {len(quiz)}")
