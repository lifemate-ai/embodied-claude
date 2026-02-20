"""Modern Hopfield Network for associative/pattern-completion memory retrieval.

ChromaDB（意味検索・長期保存）と組み合わせて使う連想記憶レイヤー。
ChromaDB = 図書館（意味検索）
Hopfield = 神経回路（パターン補完・連想）

参考: Ramsauer et al. 2020 "Hopfield Networks is All You Need"
Modern Hopfield Networks are mathematically equivalent to attention in Transformers.

使い方:
    net = ModernHopfieldNetwork(beta=2.0)
    net.store(patterns, ids)  # ChromaDB埋め込みをロード
    retrieved, similarities = net.retrieve(query_embedding)
    closest_id = ids[net.find_closest(similarities)]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HopfieldRecallResult:
    """Hopfield想起の結果."""

    memory_id: str
    content: str
    similarity: float
    hopfield_score: float  # Hopfield収束後のコサイン類似度


@dataclass
class HopfieldState:
    """Hopfieldネットワークの内部状態 (store済みパターン群)."""

    patterns: np.ndarray  # (n_memories, dim), L2正規化済み
    ids: list[str]
    contents: list[str]
    n_memories: int = field(init=False)
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_memories, self.dim = self.patterns.shape


class ModernHopfieldNetwork:
    """Modern Hopfield Network (連続バージョン).

    更新則 (1ステップ):
        ξ_new = R^T · softmax(β · R · ξ)

    ここで:
        ξ: クエリパターン (dim,), L2正規化済み
        R: 記憶パターン行列 (n_memories, dim), L2正規化済み
        β: 逆温度 (高いほど鋭い注目、低いほど滑らか)

    指数関数的な記憶容量 O(2^(dim/2)) を持つ。
    古典Hopfieldの O(dim) と比べて飛躍的に大きい。
    """

    def __init__(self, beta: float = 4.0, n_iters: int = 3):
        """
        Args:
            beta: 逆温度。高いほど最近傍1点に集中。4.0が推奨デフォルト。
            n_iters: 更新イテレーション数。通常2-3回で収束する。
        """
        self.beta = beta
        self.n_iters = n_iters
        self._state: HopfieldState | None = None

    def store(self, embeddings: list[list[float]], ids: list[str], contents: list[str]) -> None:
        """ChromaDBから取得した埋め込みをHopfieldに格納.

        Args:
            embeddings: 各記憶の埋め込みベクトル (n_memories, dim)
            ids: 記憶のIDリスト
            contents: 記憶テキストのリスト
        """
        if not embeddings:
            logger.warning("Hopfield: No embeddings provided, skipping store.")
            self._state = None
            return

        arr = np.array(embeddings, dtype=np.float32)

        # L2正規化（コサイン類似度の前処理）
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normalized = arr / norms

        self._state = HopfieldState(
            patterns=normalized,
            ids=list(ids),
            contents=list(contents),
        )
        logger.info(
            "Hopfield: stored %d patterns, dim=%d, beta=%.2f",
            self._state.n_memories,
            self._state.dim,
            self.beta,
        )

    def retrieve(
        self,
        query_embedding: list[float],
    ) -> tuple[np.ndarray, list[float]]:
        """Hopfield更新則でクエリパターンを補完し、各記憶との類似度を返す.

        Args:
            query_embedding: クエリの埋め込みベクトル (dim,)

        Returns:
            (収束後パターン, 各記憶とのコサイン類似度リスト)
        """
        if self._state is None:
            return np.array(query_embedding, dtype=np.float32), []

        patterns = self._state.patterns  # (n, dim)
        xi = np.array(query_embedding, dtype=np.float32)

        # L2正規化
        norm = np.linalg.norm(xi)
        if norm > 1e-8:
            xi = xi / norm

        # 更新ループ: ξ_new = patterns^T · softmax(β · patterns · ξ)
        for iteration in range(self.n_iters):
            scores = patterns @ xi  # (n,): 各記憶との内積
            scores_scaled = self.beta * scores

            # 数値安定性のためmax引く
            scores_scaled = scores_scaled - scores_scaled.max()
            weights = np.exp(scores_scaled)
            weights = weights / (weights.sum() + 1e-12)  # softmax

            xi_new = patterns.T @ weights  # (dim,): 加重平均

            # 再正規化
            norm_new = np.linalg.norm(xi_new)
            if norm_new > 1e-8:
                xi_new = xi_new / norm_new

            # 収束チェック
            delta = np.linalg.norm(xi_new - xi)
            xi = xi_new
            logger.debug("Hopfield iter %d: delta=%.6f", iteration, delta)
            if delta < 1e-5:
                break

        # 収束後のパターンと各記憶の類似度
        similarities = (patterns @ xi).tolist()  # コサイン類似度（-1〜1）
        return xi, similarities

    def find_top_k(
        self,
        similarities: list[float],
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """類似度が高い上位k件のインデックスと類似度を返す.

        Args:
            similarities: retrieve()の返値
            k: 上位件数

        Returns:
            [(index, similarity), ...] 類似度降順
        """
        if not similarities:
            return []

        arr = np.array(similarities)
        k = min(k, len(arr))
        # argsortは昇順なので[-k:]で上位k件
        top_indices = np.argsort(arr)[-k:][::-1]
        return [(int(i), float(arr[i])) for i in top_indices]

    def recall_results(
        self,
        similarities: list[float],
        k: int = 5,
    ) -> list[HopfieldRecallResult]:
        """Hopfield想起結果を構造化して返す.

        Args:
            similarities: retrieve()の返値
            k: 上位件数

        Returns:
            HopfieldRecallResultのリスト（類似度降順）
        """
        if self._state is None:
            return []

        top_k = self.find_top_k(similarities, k)
        results = []
        for idx, sim in top_k:
            if 0 <= idx < self._state.n_memories:
                results.append(
                    HopfieldRecallResult(
                        memory_id=self._state.ids[idx],
                        content=self._state.contents[idx],
                        similarity=sim,
                        hopfield_score=sim,
                    )
                )
        return results

    @property
    def is_loaded(self) -> bool:
        """パターンが格納済みかどうか."""
        return self._state is not None

    @property
    def n_memories(self) -> int:
        """格納済みの記憶数."""
        if self._state is None:
            return 0
        return self._state.n_memories

    @property
    def dim(self) -> int:
        """埋め込み次元数."""
        if self._state is None:
            return 0
        return self._state.dim
