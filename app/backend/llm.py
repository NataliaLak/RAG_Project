import os
from functools import lru_cache

from langchain_groq import ChatGroq


# ---------- LLM ----------

@lru_cache
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.1
    )


def get_llm():
    return load_llm()
