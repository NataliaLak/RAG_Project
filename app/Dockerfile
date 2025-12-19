FROM python:3.11-slim

WORKDIR /app

# Копируем общий requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем frontend и backend
COPY frontend/ ./frontend
COPY backend/ ./backend  

# Открываем порт Streamlit
EXPOSE 8501

# Запуск Streamlit
CMD ["bash", "-c", "PYTHONPATH=/app streamlit run frontend/streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]

