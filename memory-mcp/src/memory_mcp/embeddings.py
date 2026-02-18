"""Embedding provider abstraction layer."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Convert texts to embedding vectors."""
        ...

    @abstractmethod
    def embed_query(self, texts: list[str]) -> list[np.ndarray]:
        """Convert query texts to embedding vectors."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension."""
        ...


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers (in-process)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Convert texts to embedding vectors (passage prefix for e5)."""
        prefixed = [f"passage: {t}" for t in texts]
        result = self._model.encode(prefixed)
        return [np.array(v, dtype=np.float32) for v in result]

    def embed_query(self, texts: list[str]) -> list[np.ndarray]:
        """Convert query texts to embedding vectors (query prefix for e5)."""
        prefixed = [f"query: {t}" for t in texts]
        result = self._model.encode(prefixed)
        return [np.array(v, dtype=np.float32) for v in result]

    def dimension(self) -> int:
        return self._dim


class EmbeddingAPIProvider(EmbeddingProvider):
    """Embedding provider using HTTP API server (e.g. embedding-api container)."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        import httpx

        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._dim: int | None = None

    def _post(self, texts: list[str], prefix: str) -> list[np.ndarray]:
        resp = self._client.post(
            f"{self._base_url}/embed",
            json={"texts": texts, "prefix": prefix},
        )
        resp.raise_for_status()
        data = resp.json()
        if self._dim is None:
            self._dim = data["dimension"]
        return [np.array(v, dtype=np.float32) for v in data["embeddings"]]

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Convert texts to embedding vectors via HTTP API."""
        return self._post(texts, prefix="passage: ")

    def embed_query(self, texts: list[str]) -> list[np.ndarray]:
        """Convert query texts to embedding vectors via HTTP API."""
        return self._post(texts, prefix="query: ")

    def dimension(self) -> int:
        if self._dim is None:
            # Probe the API to get dimension
            result = self._post(["probe"], prefix="query: ")
            if self._dim is None:
                self._dim = len(result[0])
        return self._dim
