FROM python:3.12-slim 
WORKDIR /app

# Копируем зависимости
COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код (все содержимое папки app теперь в /app)
COPY app/ .

EXPOSE 8501

# Устанавливаем PYTHONPATH, чтобы Python видел корень /app
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
