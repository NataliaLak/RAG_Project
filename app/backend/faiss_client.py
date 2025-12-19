import requests

FAISS_API_URL = "http://faiss-index:8001/search"


def faiss_search(query: str, k: int = 5):
    """
    Отправляем ТЕКСТ запроса в FAISS.
    FAISS:
    - считает embedding
    - ищет
    - возвращает score
    """
    try:
        response = requests.post(
            FAISS_API_URL,
            json={
                "text": query,
                "k": k
            },
            timeout=15
        )
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"Неверный формат ответа FAISS API: {data}")

        return data

    except Exception as e:
        print(f"[FAISS ERROR] {e}")
        return []

