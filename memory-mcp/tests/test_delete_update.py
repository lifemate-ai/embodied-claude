"""Tests for delete_memory and update_memory operations."""

import pytest

from memory_mcp.memory import MemoryStore


class TestDeleteMemory:
    """Tests for MemoryStore.delete()."""

    @pytest.mark.asyncio
    async def test_delete_basic(self, memory_store: MemoryStore):
        """Deleted memory should no longer be retrievable."""
        memory = await memory_store.save(content="削除テスト記憶")
        assert await memory_store.get_by_id(memory.id) is not None

        deleted = await memory_store.delete(memory.id)
        assert deleted is True
        assert await memory_store.get_by_id(memory.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_store: MemoryStore):
        """Deleting a nonexistent memory returns False."""
        result = await memory_store.delete("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_removes_from_linked_ids(self, memory_store: MemoryStore):
        """Deleting a memory removes its ID from linked_ids of related memories."""
        m1 = await memory_store.save(content="記憶A")
        m2 = await memory_store.save(content="記憶B")
        await memory_store._add_bidirectional_link(m1.id, m2.id)

        m2_before = await memory_store.get_by_id(m2.id)
        assert m2_before is not None
        assert m1.id in m2_before.linked_ids

        await memory_store.delete(m1.id)

        m2_after = await memory_store.get_by_id(m2.id)
        assert m2_after is not None
        assert m1.id not in m2_after.linked_ids

    @pytest.mark.asyncio
    async def test_delete_removes_from_search(self, memory_store: MemoryStore):
        """Deleted memory should not appear in search results."""
        memory = await memory_store.save(content="ユニークな削除テスト文字列xyzzy")
        results_before = await memory_store.search(query="ユニークな削除テスト文字列xyzzy", n_results=5)
        assert any(r.memory.id == memory.id for r in results_before)

        await memory_store.delete(memory.id)

        results_after = await memory_store.search(query="ユニークな削除テスト文字列xyzzy", n_results=5)
        assert not any(r.memory.id == memory.id for r in results_after)


class TestUpdateMemory:
    """Tests for MemoryStore.update()."""

    @pytest.mark.asyncio
    async def test_update_emotion(self, memory_store: MemoryStore):
        """Updating emotion only changes emotion."""
        memory = await memory_store.save(content="更新テスト", emotion="neutral", importance=3)
        result = await memory_store.update(memory.id, emotion="happy")
        assert result is True

        updated = await memory_store.get_by_id(memory.id)
        assert updated is not None
        assert updated.emotion == "happy"
        assert updated.content == "更新テスト"
        assert updated.importance == 3

    @pytest.mark.asyncio
    async def test_update_importance(self, memory_store: MemoryStore):
        """Updating importance clamps to 1-5."""
        memory = await memory_store.save(content="重要度テスト", importance=2)
        await memory_store.update(memory.id, importance=5)
        updated = await memory_store.get_by_id(memory.id)
        assert updated is not None
        assert updated.importance == 5

    @pytest.mark.asyncio
    async def test_update_content_re_embeds(self, memory_store: MemoryStore):
        """Updating content changes content and allows search by new text."""
        memory = await memory_store.save(content="古いコンテンツ abcdef")
        await memory_store.update(memory.id, content="新しいコンテンツ uvwxyz")

        updated = await memory_store.get_by_id(memory.id)
        assert updated is not None
        assert updated.content == "新しいコンテンツ uvwxyz"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, memory_store: MemoryStore):
        """Updating a nonexistent memory returns False."""
        result = await memory_store.update("nonexistent-id", emotion="happy")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_category(self, memory_store: MemoryStore):
        """Updating category works independently."""
        memory = await memory_store.save(content="カテゴリ変更テスト", category="daily")
        await memory_store.update(memory.id, category="technical")
        updated = await memory_store.get_by_id(memory.id)
        assert updated is not None
        assert updated.category == "technical"
