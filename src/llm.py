# =========================================
# DOCUMIND AI - OpenRouter LLM
# =========================================

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm():

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    model = os.getenv(
        "OPENROUTER_CHAT_MODEL"
    )

    if not api_key:

        raise ValueError(
            "OPENROUTER_API_KEY is missing "
            "from .env"
        )

    if not model:

        raise ValueError(
            "OPENROUTER_CHAT_MODEL is missing "
            "from .env"
        )

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_tokens=1000
    )

    return llm