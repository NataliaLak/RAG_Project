import streamlit as st
# Импорты должны идти первыми
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # <-- ИЗМЕНЕН ИМПОРТ
from langchain_core.prompts import PromptTemplate
from pathlib import Path

# ====================================================================
st.set_page_config(page_title="RAG Ассистент", page_icon="📘")
# ====================================================================

# -------------------------
# Пути проекта
# -------------------------
PROJECT_ROOT = Path.cwd()
faiss_index_path = str(PROJECT_ROOT / "faiss_index") 

# -------------------------
# Кэширование ресурсов 
# -------------------------
@st.cache_resource
def load_embedding_model():
    """Загружает модель эмбеддингов."""
    return HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

@st.cache_resource
def load_faiss_db(_embedding_model):
    """Загружает существующую базу данных FAISS."""
    return FAISS.load_local(
        faiss_index_path, 
        _embedding_model, 
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm():
    """Загружает языковую модель Llama 3 через Groq."""
    try:
        return ChatGroq(
            # Используем отличную и очень быструю модель Llama 3
            model="llama-3.3-70b-versatile", 
            groq_api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.1
        )
    except Exception as e:
        st.error(f"Ошибка при загрузке модели Groq: {e}")
        st.error("Убедитесь, что ваш API ключ GROQ_API_KEY корректен")
        st.stop()

# -------------------------
# Инициализация приложения
# -------------------------
embedding_model = load_embedding_model()
db = load_faiss_db(embedding_model)
llm = load_llm()

# -------------------------
st.title("📘 RAG Ассистент")
# -------------------------

# -------------------------
# Prompt шаблон 
# -------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Ты — профессор, эксперт по содержимому научной книги. Твоя задача — дать ясный и точный ответ на вопрос студента.\n"
        "Действуй по шагам:\n"
        "1. Внимательно изучи предоставленный контекст.\n"
        "2. Найди в контексте все факты, относящиеся к вопросу.\n"
        "3. Сформулируй исчерпывающий ответ на русском языке, основываясь ИСКЛЮЧИТЕЛЬНО на найденных фактах.\n"
        "4. Если контекст не содержит достаточной информации для ответа, прямо скажи: 'На основе предоставленных фрагментов я не могу дать точный ответ'. Не придумывай ничего.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )
)

# -------------------------
# Функции поиска и ответа
# -------------------------
def retrieve_context(question: str, n_results: int = 5):
    results = db.similarity_search_with_score(question, k=n_results)
    docs = [doc.page_content for doc, score in results]
    metas = [doc.metadata for doc, score in results]
    scores = [score for doc, score in results]
    context = "\n---\n".join(docs)
    return context, docs, metas, scores

def get_answer(question: str):
    context, docs, metas, scores = retrieve_context(question)
    prompt_text = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt_text)
    return answer.content, docs, metas, scores

# -------------------------
# Streamlit интерфейс (основная часть)
# -------------------------
question = st.text_input("Введите ваш вопрос о книге:")

if question:
    with st.spinner("Генерация ответа..."):
        answer, docs, metas, scores = get_answer(question)
    st.markdown(f"### 💬 Ответ:\n{answer}")

    st.divider()
    st.markdown("**🔍 Найденные фрагменты контекста:**")
    for i, (doc, meta, score) in enumerate(zip(docs, metas, scores)):
        st.markdown(f"**Фрагмент {i+1}** (score={score:.4f})")
        st.write(doc[:500] + "...")
        st.divider()