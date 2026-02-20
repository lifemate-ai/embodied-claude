"""カスタム埋め込み関数モジュール。

intfloat/multilingual-e5-base を ChromaDB で使うためのラッパー。
e5 モデルはクエリと文書で異なるプレフィックスが必要なため、
標準の SentenceTransformerEmbeddingFunction は使えない。
"""

from __future__ import annotations

import logging
from typing import Any

from chromadb import EmbeddingFunction

logger = logging.getLogger(__name__)


class E5EmbeddingFunction(EmbeddingFunction):
    """intfloat/multilingual-e5-base 用カスタム埋め込み関数。

    e5 モデルの仕様:
    - 文書（passage）保存時: "passage: {text}" としてエンコード
    - クエリ検索時: "query: {text}" としてエンコード

    ChromaDB の EmbeddingFunction プロトコルに準拠:
    - __call__(input: Documents) -> Embeddings  # 文書保存用（passage プレフィックス）

    クエリ用は encode_query() メソッドで別途提供。

    Args:
        model_name: SentenceTransformer モデル名
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy load; actual type is SentenceTransformer

    @staticmethod
    def name() -> str:
        return "E5EmbeddingFunction"

    def get_config(self) -> dict[str, str]:
        return {"model_name": self._model_name}

    def _load_model(self) -> None:
        """モデルを遅延ロード。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                logger.info("E5EmbeddingFunction: loaded model %s", self._model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers が必要です。"
                    "`uv add sentence-transformers` を実行してください。"
                ) from e

    def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]
        """文書保存用埋め込み（passage: プレフィックス）。

        ChromaDB の EmbeddingFunction プロトコルに準拠。

        Args:
            input: エンコードするテキストのリスト

        Returns:
            埋め込みベクトルのリスト（各要素は float のリスト）
        """
        self._load_model()
        prefixed = [f"passage: {doc}" for doc in input]
        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, texts: list[str]) -> list[list[float]]:
        """クエリ検索用埋め込み（query: プレフィックス）。

        collection.query() の query_embeddings に渡す際に使用。

        Args:
            texts: クエリテキストのリスト

        Returns:
            埋め込みベクトルのリスト
        """
        self._load_model()
        prefixed = [f"query: {t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
