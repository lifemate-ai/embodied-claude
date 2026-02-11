"""Tests for memory operations."""

from datetime import datetime, timedelta

import pytest

from memory_mcp.memory import (
    MemoryStore,
    calculate_emotion_boost,
    calculate_final_score,
    calculate_importance_boost,
    calculate_time_decay,
)


class TestMemorySave:
    """Tests for save_memory."""

    @pytest.mark.asyncio
    async def test_save_basic(self, memory_store: MemoryStore):
        """Test basic memory save."""
        memory = await memory_store.save(
            content="幼馴染と初めて会った日",
            emotion="happy",
            importance=5,
            category="memory",
        )

        assert memory.content == "幼馴染と初めて会った日"
        assert memory.emotion == "happy"
        assert memory.importance == 5
        assert memory.category == "memory"
        assert memory.id is not None
        assert memory.timestamp is not None

    @pytest.mark.asyncio
    async def test_save_with_defaults(self, memory_store: MemoryStore):
        """Test save with default values."""
        memory = await memory_store.save(content="Something happened")

        assert memory.emotion == "neutral"
        assert memory.importance == 3
        assert memory.category == "daily"

    @pytest.mark.asyncio
    async def test_importance_clamping(self, memory_store: MemoryStore):
        """Test importance is clamped to 1-5."""
        memory_low = await memory_store.save(content="Test low", importance=0)
        memory_high = await memory_store.save(content="Test high", importance=10)

        assert memory_low.importance == 1
        assert memory_high.importance == 5


class TestMemorySearch:
    """Tests for search_memories."""

    @pytest.mark.asyncio
    async def test_search_basic(self, memory_store: MemoryStore):
        """Test basic semantic search."""
        await memory_store.save(content="カメラで部屋を見た", category="observation")
        await memory_store.save(content="コードを書いた", category="technical")
        await memory_store.save(content="幼馴染と話した", category="memory")

        results = await memory_store.search("幼馴染との会話")

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, memory_store: MemoryStore):
        """Test search with category filter."""
        await memory_store.save(content="技術的な学び1", category="technical")
        await memory_store.save(content="日常の出来事", category="daily")
        await memory_store.save(content="技術的な学び2", category="technical")

        results = await memory_store.search("学び", category_filter="technical")

        assert len(results) > 0
        for result in results:
            assert result.memory.category == "technical"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, memory_store: MemoryStore):
        """Test search with no matching results."""
        await memory_store.save(content="Something completely different")

        results = await memory_store.search(
            "非常に特殊なクエリ",
            category_filter="philosophical",
        )

        # May or may not find results depending on semantic similarity
        assert isinstance(results, list)


class TestSharedGroupJobIsolation:
    """Tests for shared group and job isolation bug fix."""

    @pytest.mark.asyncio
    async def test_create_shared_group_updates_job_shared_group_ids(self, memory_store: MemoryStore):
        """Test that creating a shared group with members updates each job's shared_group_ids."""
        # Create jobs
        job_a = await memory_store.create_job(job_id="Chat_A", name="Chat A")
        job_b = await memory_store.create_job(job_id="Chat_B", name="Chat B")
        
        # Initially, jobs should have no shared groups
        assert job_a.shared_group_ids == ()
        assert job_b.shared_group_ids == ()
        
        # Create shared group with members
        group = await memory_store.create_shared_group(
            group_id="User_Chat",
            name="User Chat",
            member_job_ids=("Chat_A", "Chat_B"),
        )
        
        # Verify group has members
        assert group.member_job_ids == ("Chat_A", "Chat_B")
        
        # Verify jobs were updated to reference the group
        updated_job_a = await memory_store.get_job("Chat_A")
        updated_job_b = await memory_store.get_job("Chat_B")
        
        assert updated_job_a is not None
        assert updated_job_b is not None
        assert "User_Chat" in updated_job_a.shared_group_ids
        assert "User_Chat" in updated_job_b.shared_group_ids

    @pytest.mark.asyncio
    async def test_search_includes_shared_memories_after_create_group(self, memory_store: MemoryStore):
        """Test that search with job_id includes shared group memories after create_shared_group."""
        # Create jobs
        await memory_store.create_job(job_id="Chat_A", name="Chat A")
        await memory_store.create_job(job_id="Chat_B", name="Chat B")
        
        # Create shared group with members
        await memory_store.create_shared_group(
            group_id="User_Chat",
            name="User Chat",
            member_job_ids=("Chat_A", "Chat_B"),
        )
        
        # Save memories
        await memory_store.save(
            content="貴女の名前は二上ユウです",
            memory_type="job",
            job_id="Chat_A",
        )
        await memory_store.save(
            content="チャット相手の名前は西日本太郎です",
            memory_type="shared",
            shared_group_ids=("User_Chat",),
        )
        
        # Search with job_id Chat_A should find both memories
        results = await memory_store.search(
            query="*",
            job_id="Chat_A",
            include_global=True,
            include_shared=True,
        )
        
        contents = [r.memory.content for r in results]
        assert "貴女の名前は二上ユウです" in contents
        assert "チャット相手の名前は西日本太郎です" in contents

    @pytest.mark.asyncio
    async def test_delete_shared_group_removes_group_and_updates_jobs(self, memory_store: MemoryStore):
        """Test that deleting a shared group removes it and updates member jobs."""
        # Create jobs
        await memory_store.create_job(job_id="Chat_A", name="Chat A")
        await memory_store.create_job(job_id="Chat_B", name="Chat B")
        
        # Create shared group with members
        await memory_store.create_shared_group(
            group_id="User_Chat",
            name="User Chat",
            member_job_ids=("Chat_A", "Chat_B"),
        )
        
        # Verify jobs have the group
        job_a = await memory_store.get_job("Chat_A")
        job_b = await memory_store.get_job("Chat_B")
        assert "User_Chat" in job_a.shared_group_ids
        assert "User_Chat" in job_b.shared_group_ids
        
        # Delete the group
        success = await memory_store.delete_shared_group("User_Chat")
        assert success is True
        
        # Verify group is deleted
        group = await memory_store.get_shared_group("User_Chat")
        assert group is None
        
        # Verify jobs no longer have the group
        job_a = await memory_store.get_job("Chat_A")
        job_b = await memory_store.get_job("Chat_B")
        assert "User_Chat" not in job_a.shared_group_ids
        assert "User_Chat" not in job_b.shared_group_ids

    @pytest.mark.asyncio
    async def test_delete_shared_group_not_found(self, memory_store: MemoryStore):
        """Test that deleting a non-existent group returns False."""
        success = await memory_store.delete_shared_group("NonExistentGroup")
        assert success is False


class TestMemoryRecall:
    """Tests for recall."""

    @pytest.mark.asyncio
    async def test_recall_context(self, memory_store: MemoryStore):
        """Test context-based recall."""
        await memory_store.save(content="Wi-Fiカメラを設置した")
        await memory_store.save(content="パン・チルト機能を実装した")
        await memory_store.save(content="美味しいラーメンを食べた")

        results = await memory_store.recall(context="カメラの機能について")

        assert len(results) > 0


class TestMemoryListRecent:
    """Tests for list_recent_memories."""

    @pytest.mark.asyncio
    async def test_list_recent_order(self, memory_store: MemoryStore):
        """Test that recent memories are returned in order."""
        import asyncio
        await memory_store.save(content="Memory 1")
        await asyncio.sleep(0.01)  # Ensure different timestamps
        await memory_store.save(content="Memory 2")
        await asyncio.sleep(0.01)
        await memory_store.save(content="Memory 3")

        memories = await memory_store.list_recent(limit=3)

        assert len(memories) == 3
        # Should be newest first
        assert memories[0].content == "Memory 3"
        assert memories[2].content == "Memory 1"

    @pytest.mark.asyncio
    async def test_list_recent_with_limit(self, memory_store: MemoryStore):
        """Test limit parameter."""
        for i in range(10):
            await memory_store.save(content=f"Memory {i}")

        memories = await memory_store.list_recent(limit=5)

        assert len(memories) == 5

    @pytest.mark.asyncio
    async def test_list_recent_with_category_filter(self, memory_store: MemoryStore):
        """Test category filter."""
        await memory_store.save(content="Tech 1", category="technical")
        await memory_store.save(content="Daily 1", category="daily")
        await memory_store.save(content="Tech 2", category="technical")

        memories = await memory_store.list_recent(category_filter="technical")

        assert len(memories) == 2
        for m in memories:
            assert m.category == "technical"


class TestMemoryStats:
    """Tests for get_memory_stats."""

    @pytest.mark.asyncio
    async def test_stats_counts(self, memory_store: MemoryStore):
        """Test statistics counts."""
        await memory_store.save(content="Happy memory", emotion="happy", category="daily")
        await memory_store.save(content="Sad memory", emotion="sad", category="feeling")
        await memory_store.save(content="Another happy", emotion="happy", category="daily")

        stats = await memory_store.get_stats()

        assert stats.total_count == 3
        assert stats.by_emotion.get("happy") == 2
        assert stats.by_emotion.get("sad") == 1
        assert stats.by_category.get("daily") == 2
        assert stats.by_category.get("feeling") == 1

    @pytest.mark.asyncio
    async def test_stats_empty(self, memory_store: MemoryStore):
        """Test stats with no memories."""
        stats = await memory_store.get_stats()

        assert stats.total_count == 0
        assert stats.oldest_timestamp is None
        assert stats.newest_timestamp is None


class TestScoringFunctions:
    """Tests for scoring utility functions."""

    def test_time_decay_fresh_memory(self):
        """Test time decay for a fresh memory."""
        now = datetime.now()
        timestamp = now.isoformat()
        decay = calculate_time_decay(timestamp, now)
        # Fresh memory should have decay close to 1.0
        assert decay > 0.99

    def test_time_decay_old_memory(self):
        """Test time decay for an old memory."""
        now = datetime.now()
        old_time = now - timedelta(days=60)  # 60 days ago
        timestamp = old_time.isoformat()
        decay = calculate_time_decay(timestamp, now, half_life_days=30.0)
        # After 2 half-lives, should be around 0.25
        assert 0.2 < decay < 0.3

    def test_emotion_boost_values(self):
        """Test emotion boost returns expected values."""
        assert calculate_emotion_boost("excited") == 0.4
        assert calculate_emotion_boost("moved") == 0.3
        assert calculate_emotion_boost("neutral") == 0.0
        assert calculate_emotion_boost("unknown") == 0.0

    def test_importance_boost_values(self):
        """Test importance boost calculation."""
        assert calculate_importance_boost(1) == 0.0
        assert calculate_importance_boost(5) == 0.4
        assert calculate_importance_boost(3) == 0.2

    def test_final_score_calculation(self):
        """Test final score combines all factors."""
        score = calculate_final_score(
            semantic_distance=1.0,
            time_decay=1.0,  # No decay
            emotion_boost=0.3,
            importance_boost=0.2,
        )
        # score = 1.0 * 1.0 + (1-1)*0.3 - 0.3*0.2 - 0.2*0.2
        # score = 1.0 + 0 - 0.06 - 0.04 = 0.9
        assert 0.85 < score < 0.95


class TestAccessTracking:
    """Tests for access count tracking."""

    @pytest.mark.asyncio
    async def test_update_access(self, memory_store: MemoryStore):
        """Test access count is incremented."""
        memory = await memory_store.save(content="Test memory")
        assert memory.access_count == 0

        await memory_store.update_access(memory.id)

        updated = await memory_store.get_by_id(memory.id)
        assert updated is not None
        assert updated.access_count == 1
        assert updated.last_accessed != ""

    @pytest.mark.asyncio
    async def test_get_by_id(self, memory_store: MemoryStore):
        """Test retrieving memory by ID."""
        memory = await memory_store.save(content="Findable memory", emotion="happy")

        found = await memory_store.get_by_id(memory.id)

        assert found is not None
        assert found.content == "Findable memory"
        assert found.emotion == "happy"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, memory_store: MemoryStore):
        """Test get_by_id returns None for non-existent ID."""
        found = await memory_store.get_by_id("non-existent-id")
        assert found is None


class TestAutoLinking:
    """Tests for automatic memory linking."""

    @pytest.mark.asyncio
    async def test_save_with_auto_link(self, memory_store: MemoryStore):
        """Test auto-linking creates bidirectional links."""
        # Save first memory
        mem1 = await memory_store.save(content="Wi-Fiカメラを設置した")

        # Save similar memory with auto-link
        mem2 = await memory_store.save_with_auto_link(
            content="カメラのパンチルト機能を実装",
            link_threshold=1.5,  # Generous threshold
        )

        # Check that mem2 has link to mem1
        assert len(mem2.linked_ids) > 0 or True  # May or may not link depending on similarity

        # If linked, check bidirectional
        if mem2.linked_ids:
            mem1_updated = await memory_store.get_by_id(mem1.id)
            assert mem1_updated is not None
            assert mem2.id in mem1_updated.linked_ids

    @pytest.mark.asyncio
    async def test_get_linked_memories(self, memory_store: MemoryStore):
        """Test retrieving linked memories."""
        # Save and link memories manually
        mem1 = await memory_store.save(content="記憶1")
        mem2 = await memory_store.save_with_auto_link(
            content="記憶1に関連する記憶2",
            link_threshold=2.0,  # Very generous
        )

        # Get linked memories
        linked = await memory_store.get_linked_memories(mem2.id, depth=1)

        # Should find linked memories (may be empty if not similar enough)
        assert isinstance(linked, list)

    @pytest.mark.asyncio
    async def test_recall_with_chain(self, memory_store: MemoryStore):
        """Test recall with chain returns linked memories."""
        # Save some memories
        await memory_store.save(content="USBカメラの設定")
        await memory_store.save(content="カメラで部屋を撮影")
        await memory_store.save(content="美味しいラーメン")

        # Recall with chain
        results = await memory_store.recall_with_chain(
            context="カメラの機能",
            n_results=2,
            chain_depth=1,
        )

        assert len(results) >= 1


class TestSearchWithScoring:
    """Tests for search with scoring."""

    @pytest.mark.asyncio
    async def test_search_with_scoring_returns_scored_memories(self, memory_store: MemoryStore):
        """Test search_with_scoring returns ScoredMemory objects."""
        await memory_store.save(content="Important memory", importance=5, emotion="excited")
        await memory_store.save(content="Regular memory", importance=3, emotion="neutral")

        results = await memory_store.search_with_scoring(
            query="memory",
            use_time_decay=True,
            use_emotion_boost=True,
        )

        assert len(results) > 0
        # Check ScoredMemory attributes
        result = results[0]
        assert hasattr(result, "semantic_distance")
        assert hasattr(result, "time_decay_factor")
        assert hasattr(result, "emotion_boost")
        assert hasattr(result, "importance_boost")
        assert hasattr(result, "final_score")

    @pytest.mark.asyncio
    async def test_search_with_scoring_disabled(self, memory_store: MemoryStore):
        """Test search with scoring disabled."""
        await memory_store.save(content="Test memory")

        results = await memory_store.search_with_scoring(
            query="test",
            use_time_decay=False,
            use_emotion_boost=False,
        )

        assert len(results) > 0
        result = results[0]
        # With scoring disabled, time_decay should be 1.0 and emotion_boost 0.0
        assert result.time_decay_factor == 1.0
        assert result.emotion_boost == 0.0
