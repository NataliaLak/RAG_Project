import requests
from typing import List

FAISS_API_URL = "http://faiss:8001"


def faiss_search(query: str, k: int = 5, source: str | None = None) -> List[dict]:
    try:
        payload = {
            "text": query,
            "k": k,
        }
        if source:
            payload["source"] = source

        resp = requests.post(
            f"{FAISS_API_URL}/search",
            json=payload,
            timeout=15
        )
        resp.raise_for_status()

        return resp.json()

    except Exception as e:
        print(f"[FAISS ERROR] {e}")
        return []
