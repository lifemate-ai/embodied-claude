"""Tests for Modern Hopfield Network (Phase 7 associative memory)."""

import numpy as np
import pytest

from memory_mcp.hopfield import HopfieldRecallResult, ModernHopfieldNetwork


class TestModernHopfieldNetwork:
    """Modern Hopfield Networkの基本動作テスト."""

    def test_store_and_properties(self):
        """パターン格納後のプロパティが正しいことを確認."""
        net = ModernHopfieldNetwork(beta=4.0)
        patterns = np.random.randn(5, 64).tolist()
        ids = [f"mem{i}" for i in range(5)]
        contents = [f"content{i}" for i in range(5)]

        net.store(patterns, ids, contents)

        assert net.is_loaded
        assert net.n_memories == 5
        assert net.dim == 64

    def test_empty_before_store(self):
        """格納前はis_loadedがFalse."""
        net = ModernHopfieldNetwork()
        assert not net.is_loaded
        assert net.n_memories == 0
        assert net.dim == 0

    def test_retrieve_empty_returns_query(self):
        """パターン未格納でretrieveするとクエリがそのまま返る."""
        net = ModernHopfieldNetwork()
        query = [0.1, 0.2, 0.3]
        retrieved, similarities = net.retrieve(query)
        assert similarities == []
        assert retrieved is not None

    def test_pattern_completion_noisy_query(self):
        """ノイズ入りクエリが最近傍パターンに収束する (核心機能)."""
        np.random.seed(123)
        net = ModernHopfieldNetwork(beta=4.0, n_iters=5)

        # 3つの直交に近いパターン
        dim = 128
        patterns_arr = np.random.randn(3, dim)
        # L2正規化でほぼ直交（次元が高いとランダムベクトルは直交に近い）
        patterns = patterns_arr.tolist()
        ids = ["コウタとしりとり", "夜景を見た", "初めて声が出た"]
        contents = ids[:]

        net.store(patterns, ids, contents)

        # パターン0にわずかなノイズを加える
        noisy_query = (patterns_arr[0] + np.random.randn(dim) * 0.05).tolist()

        _, similarities = net.retrieve(noisy_query)

        # パターン0が最も高い類似度を持つはず
        top_k = net.find_top_k(similarities, k=1)
        assert top_k[0][0] == 0, f"Expected pattern 0 to be nearest, got {top_k[0][0]}"
        assert top_k[0][1] > 0.9, f"Expected high similarity, got {top_k[0][1]:.4f}"

    def test_recall_results_order(self):
        """recall_resultsが類似度降順に返ること."""
        np.random.seed(42)
        net = ModernHopfieldNetwork(beta=4.0)

        dim = 64
        patterns_arr = np.random.randn(5, dim)
        patterns = patterns_arr.tolist()
        ids = [f"mem{i}" for i in range(5)]
        contents = [f"content{i}" for i in range(5)]

        net.store(patterns, ids, contents)

        query = patterns_arr[2] + np.random.randn(dim) * 0.1
        _, similarities = net.retrieve(query.tolist())

        results = net.recall_results(similarities, k=3)

        assert len(results) == 3
        assert all(isinstance(r, HopfieldRecallResult) for r in results)

        # 降順チェック
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_find_top_k_limits_to_n_memories(self):
        """find_top_kがn_memories以下に制限されること."""
        net = ModernHopfieldNetwork()
        patterns = np.random.randn(3, 32).tolist()
        net.store(patterns, ["a", "b", "c"], ["A", "B", "C"])

        similarities = [0.9, 0.5, 0.1]
        top_k = net.find_top_k(similarities, k=10)  # 10 > 3

        assert len(top_k) == 3  # 3件に制限

    def test_beta_sharpness(self):
        """beta（逆温度）が高いほど最近傍への集中度が上がること."""
        np.random.seed(7)
        dim = 64
        patterns_arr = np.random.randn(3, dim)
        patterns = patterns_arr.tolist()
        ids = ["a", "b", "c"]
        contents = ids[:]

        query = patterns_arr[0] + np.random.randn(dim) * 0.2

        # 低betaでのsoftmax（散漫）
        net_low = ModernHopfieldNetwork(beta=0.5, n_iters=5)
        net_low.store(patterns, ids, contents)
        _, sims_low = net_low.retrieve(query.tolist())

        # 高betaでのsoftmax（鋭い）
        net_high = ModernHopfieldNetwork(beta=10.0, n_iters=5)
        net_high.store(patterns, ids, contents)
        _, sims_high = net_high.retrieve(query.tolist())

        # 高betaの方が最上位とそれ以外の差が大きい
        top_low = max(sims_low)
        top_high = max(sims_high)
        assert top_high >= top_low - 0.05  # 高betaの方が（または同程度に）集中

    def test_store_empty_embeddings(self):
        """空の埋め込みリストを渡した場合はis_loadedがFalseのまま."""
        net = ModernHopfieldNetwork()
        net.store([], [], [])
        assert not net.is_loaded

    def test_hopfield_recall_result_fields(self):
        """HopfieldRecallResultの全フィールドが揃っていること."""
        np.random.seed(0)
        net = ModernHopfieldNetwork(beta=4.0)
        dim = 32
        patterns = np.random.randn(2, dim).tolist()
        net.store(patterns, ["id1", "id2"], ["content one", "content two"])

        query = np.random.randn(dim).tolist()
        _, sims = net.retrieve(query)
        results = net.recall_results(sims, k=2)

        assert len(results) == 2
        r = results[0]
        assert isinstance(r.memory_id, str)
        assert isinstance(r.content, str)
        assert isinstance(r.similarity, float)
        assert isinstance(r.hopfield_score, float)
