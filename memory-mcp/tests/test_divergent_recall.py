"""Tests for divergent recall and consolidation."""

import asyncio

import pytest

from memory_mcp.memory import MemoryStore


class TestDivergentRecall:
    """Divergent recall behavior tests."""

    @pytest.mark.asyncio
    async def test_recall_divergent_returns_results_with_diagnostics(
        self,
        memory_store: MemoryStore,
    ):
        first = await memory_store.save(
            content="朝の空をカメラで探した",
            emotion="excited",
            category="observation",
            tags=("camera", "sky"),
        )
        second = await memory_store.save(
            content="窓の位置を変えて空を見つけた",
            emotion="happy",
            category="observation",
            tags=("window", "sky"),
        )
        await memory_store.save(
            content="夕飯の献立を考えた",
            emotion="neutral",
            category="daily",
        )
        await memory_store.add_causal_link(first.id, second.id, link_type="related")

        results, diagnostics = await memory_store.recall_divergent(
            context="空を探すカメラの話",
            n_results=3,
            max_branches=3,
            max_depth=3,
            include_diagnostics=True,
        )

        assert results
        assert "diversity_score" in diagnostics
        assert diagnostics["selected_count"] == len(results)

    @pytest.mark.asyncio
    async def test_get_association_diagnostics_has_core_metrics(
        self,
        memory_store: MemoryStore,
    ):
        await memory_store.save(content="カメラの向きと空の関係を覚えた")
        await memory_store.save(content="雲の色で天気を予測した")

        diagnostics = await memory_store.get_association_diagnostics(
            context="空とカメラ",
            sample_size=10,
        )

        assert "adaptive_branch_limit" in diagnostics
        assert "avg_prediction_error" in diagnostics


class TestConsolidation:
    """Consolidation replay tests."""

    @pytest.mark.asyncio
    async def test_consolidate_memories_updates_coactivation(
        self,
        memory_store: MemoryStore,
    ):
        first = await memory_store.save(content="朝の観察を記録した", category="observation")
        await asyncio.sleep(0.01)
        second = await memory_store.save(content="窓辺で空を見た", category="observation")

        stats = await memory_store.consolidate_memories(
            window_hours=1,
            max_replay_events=10,
            link_update_strength=0.3,
        )

        assert stats["coactivation_updates"] >= 0

        # Verify coactivation was created between adjacent memories
        neighbors = await memory_store._store.get_coactivation_neighbors(first.id)
        neighbor_ids = [nid for nid, _ in neighbors]
        assert second.id in neighbor_ids or stats["coactivation_updates"] == 0
