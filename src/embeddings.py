import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings

load_dotenv()


class OpenRouterEmbeddings(Embeddings):
    """
    OpenRouter embedding implementation.

    Uses OpenRouter's OpenAI-compatible embeddings API
    and explicitly requests float vectors.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1"
        )
        self.model = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            "nvidia/nemotron-3-embed-1b:free"
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing from .env"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple document chunks into embedding vectors.
        """

        if not texts:
            return []

        all_embeddings = []

        # Process in small batches for reliability
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float"
            )

            # OpenAI-compatible APIs return data with an index.
            # Sort to preserve the original text order.
            sorted_data = sorted(
                response.data,
                key=lambda item: item.index
            )

            batch_embeddings = [
                item.embedding for item in sorted_data
            ]

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Convert a user question into one embedding vector.
        """

        if not text or not text.strip():
            raise ValueError(
                "Cannot create an embedding for empty text."
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )

        return response.data[0].embedding


def get_embeddings():
    """
    Return the OpenRouter embedding object used by the RAG pipeline.
    """

    return OpenRouterEmbeddings()