# RAG & Quiz Assistant

Проект реализует **RAG-ассистент (Retrieval-Augmented Generation)** с возможностью:
- загружать книги в формате **PDF / TXT**
- задавать вопросы по содержимому книги
- автоматически генерировать **квизы по книге**
- получать объяснения правильных ответов

Проект состоит из двух сервисов:
- **FAISS service** — хранение и поиск семантических чанков
- **Streamlit app** — интерфейс RAG и квизов

Репозиторий:  
https://github.com/NataliaLak/RAG_Project

---

## Архитектура

Streamlit UI → RAG / Quiz logic → FAISS API → Vector Store

- FAISS используется как векторное хранилище
- Текст разбивается с помощью `SemanticChunker`
- LLM используется для:
  - генерации ответов (RAG)
  - генерации тем квизов
  - генерации вопросов и объяснений

---

## Технологии

- Python 
- FastAPI
- Streamlit
- FAISS
- LangChain
- HuggingFace Embeddings (`multi-qa-mpnet-base-dot-v1`)
- Docker & Docker Compose
- GitHub Actions (CI)

---

## Структура проекта

RAG_Project/
├── app/ # Streamlit приложение
├── faiss/ # FAISS сервис
│ ├── faiss_server.py
│ └── Dockerfile.faiss
├── docker-compose.yml
├── .env # секреты (в .gitignore)
├── .gitignore
├── notebooks/
└── README.md


---

## Запуск проекта локально

### Клонировать репозиторий

```bash
git clone https://github.com/NataliaLak/RAG_Project.git
cd RAG_Project

### Создать .env файл

В корне проекта создать файл .env:

GROQ_API_KEY=your_api_key_here

### Запуск через Docker Compose

```bash
docker compose up --build


При первом запуске:

модель эмбеддингов загружается автоматически

индексация больших PDF может занимать продолжительное время

### Доступ к сервисам

После запуска:

Streamlit UI:
http://localhost:8501

FAISS API:
http://localhost:8001

