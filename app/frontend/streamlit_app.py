import streamlit as st
from backend.rag import get_rag_answer
from backend.quiz import (
    auto_generate_quiz_topics,
    generate_quiz,
    explain_correct_answer
)

st.set_page_config(page_title="📘 RAG Ассистент", layout="wide")
st.title("📘 RAG & Quiz Ассистент")

tab_rag, tab_quiz = st.tabs(["🔍 RAG", "🧠 Квизы"])

# ---------------- RAG ----------------
with tab_rag:
    question = st.text_input("Введите вопрос:")

    if question:
        try:
            answer, docs, metas, scores = get_rag_answer(question)
        except Exception as e:
            st.error(f"Ошибка при получении ответа: {e}")
            st.stop()

        st.markdown("### 💬 Ответ")
        st.write(answer)

        st.markdown("### 📚 Фрагменты")
        for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
            score_str = f"{score:.4f}" if score is not None else "-"
            chunk_id = meta.get("chunk_id") if meta else "-"
            st.markdown(
            f"**Фрагмент {i+1}** — score={score_str}, chunk_id={chunk_id}"
            )
            st.write(doc[:700] + "…")
            st.divider()

# ---------------- QUIZ ----------------
with tab_quiz:
    if "quiz_topics" not in st.session_state:
        try:
            st.session_state.quiz_topics = auto_generate_quiz_topics()
        except Exception as e:
            st.error(f"Ошибка генерации тем: {e}")
            st.stop()

    topics = st.session_state.quiz_topics
    if not topics:
        st.warning("Темы не найдены")
        st.stop()

    topic = st.selectbox("Выберите тему:", topics)

    if st.button("🎲 Сгенерировать квиз"):
        try:
            st.session_state.quiz = generate_quiz(topic)
        except Exception as e:
            st.error(f"Ошибка генерации квиза: {e}")
            st.stop()

        st.session_state.answers = {}
        st.session_state.checked = False

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
                    f"**Вы ответили:**  \n{user_answer}\n\n"
                    f"**Правильный ответ:**  \n{correct_answer}\n\n"
                    f"{explanation}"
                )
        st.info(f"Результат: {correct} / {len(quiz)}")
