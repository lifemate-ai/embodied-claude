"""Tests for the unified memory_links table."""

import pytest

from memory_mcp.memory import MemoryStore


class TestMigration:
    """Tests for migration from legacy linked_ids / links columns."""

    @pytest.mark.asyncio
    async def test_migration_idempotent(self, memory_store: MemoryStore):
        """Running migration twice does not duplicate rows."""
        m1 = await memory_store.save(content="記憶A", auto_link=False)
        m2 = await memory_store.save(content="記憶B", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "similar")

        links_before = await memory_store.get_links_from(m1.id)
        assert len(links_before) == 1

        # Simulate re-connect (migration runs again)
        await memory_store.disconnect()
        await memory_store.connect()

        links_after = await memory_store.get_links_from(m1.id)
        assert len(links_after) == 1  # not duplicated


class TestAddLink:
    """Tests for add_link."""

    @pytest.mark.asyncio
    async def test_add_link_basic(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="ソース", auto_link=False)
        m2 = await memory_store.save(content="ターゲット", auto_link=False)

        await memory_store.add_link(m1.id, m2.id, "similar")

        links = await memory_store.get_links_from(m1.id)
        assert len(links) == 1
        assert links[0]["target_id"] == m2.id
        assert links[0]["link_type"] == "similar"

    @pytest.mark.asyncio
    async def test_add_link_bidirectional(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)

        await memory_store.add_link(m1.id, m2.id, "similar", bidirectional=True)

        links_from_m1 = await memory_store.get_links_from(m1.id)
        links_from_m2 = await memory_store.get_links_from(m2.id)
        assert any(lk["target_id"] == m2.id for lk in links_from_m1)
        assert any(lk["target_id"] == m1.id for lk in links_from_m2)

    @pytest.mark.asyncio
    async def test_add_link_duplicate_prevention(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)

        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m1.id, m2.id, "similar")  # duplicate

        links = await memory_store.get_links_from(m1.id)
        assert len(links) == 1

    @pytest.mark.asyncio
    async def test_add_link_different_types_same_pair(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)

        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m1.id, m2.id, "caused_by")

        links = await memory_store.get_links_from(m1.id)
        assert len(links) == 2
        types = {lk["link_type"] for lk in links}
        assert types == {"similar", "caused_by"}


class TestRemoveLink:
    """Tests for remove_link."""

    @pytest.mark.asyncio
    async def test_remove_link_specific_type(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m1.id, m2.id, "caused_by")

        removed = await memory_store.remove_link(m1.id, m2.id, "similar")
        assert removed is True

        links = await memory_store.get_links_from(m1.id)
        assert len(links) == 1
        assert links[0]["link_type"] == "caused_by"

    @pytest.mark.asyncio
    async def test_remove_link_all_types(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m1.id, m2.id, "related")

        removed = await memory_store.remove_link(m1.id, m2.id)  # no link_type = all
        assert removed is True

        links = await memory_store.get_links_from(m1.id)
        assert len(links) == 0

    @pytest.mark.asyncio
    async def test_remove_link_nonexistent(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)

        removed = await memory_store.remove_link(m1.id, m2.id)
        assert removed is False


class TestGetLinks:
    """Tests for get_links_from / get_links_to."""

    @pytest.mark.asyncio
    async def test_get_links_from(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)
        m3 = await memory_store.save(content="C", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m1.id, m3.id, "related")

        links = await memory_store.get_links_from(m1.id)
        assert len(links) == 2
        targets = {lk["target_id"] for lk in links}
        assert targets == {m2.id, m3.id}

    @pytest.mark.asyncio
    async def test_get_links_from_with_type_filter(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)
        m3 = await memory_store.save(content="C", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m1.id, m3.id, "caused_by")

        links = await memory_store.get_links_from(m1.id, link_type="similar")
        assert len(links) == 1
        assert links[0]["target_id"] == m2.id

    @pytest.mark.asyncio
    async def test_get_links_to_reverse_query(self, memory_store: MemoryStore):
        """get_links_to can find incoming links — impossible with CSV linked_ids."""
        m1 = await memory_store.save(content="原因", auto_link=False)
        m2 = await memory_store.save(content="結果", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "leads_to")

        # m2 has no outgoing links, but has incoming link from m1
        outgoing = await memory_store.get_links_from(m2.id)
        assert len(outgoing) == 0

        incoming = await memory_store.get_links_to(m2.id)
        assert len(incoming) == 1
        assert incoming[0]["source_id"] == m1.id


class TestAutoLink:
    """Tests for save() auto_link behavior."""

    @pytest.mark.asyncio
    async def test_save_auto_links_by_default(self, memory_store: MemoryStore):
        """save() should auto-link to similar existing memories."""
        m1 = await memory_store.save(content="東京の桜が美しい春の日", auto_link=False)

        # Save a similar memory — should auto-link
        m2 = await memory_store.save(content="東京の桜が美しい春の朝")

        links = await memory_store.get_links_from(m2.id)
        # Should have linked to m1 (similar content)
        assert any(lk["target_id"] == m1.id for lk in links)

    @pytest.mark.asyncio
    async def test_save_auto_link_disabled(self, memory_store: MemoryStore):
        """save(auto_link=False) should not create any links."""
        await memory_store.save(content="東京の桜が美しい春の日", auto_link=False)
        m2 = await memory_store.save(content="東京の桜が美しい春の朝", auto_link=False)

        links = await memory_store.get_links_from(m2.id)
        assert len(links) == 0

    @pytest.mark.asyncio
    async def test_save_auto_link_first_memory(self, memory_store: MemoryStore):
        """First memory in empty store should save without error even with auto_link=True."""
        m = await memory_store.save(content="最初の記憶")
        assert m.id is not None
        # No links since there are no other memories to link to
        links = await memory_store.get_links_from(m.id)
        assert len(links) == 0


class TestGetLinkedMemories:
    """Tests for get_linked_memories using memory_links table."""

    @pytest.mark.asyncio
    async def test_get_linked_memories_uses_all_link_types(self, memory_store: MemoryStore):
        """get_linked_memories should traverse all link types, not just linked_ids."""
        m1 = await memory_store.save(content="記憶1", auto_link=False)
        m2 = await memory_store.save(content="記憶2", auto_link=False)
        m3 = await memory_store.save(content="記憶3", auto_link=False)

        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m1.id, m3.id, "caused_by")

        linked = await memory_store.get_linked_memories(m1.id, depth=1)
        linked_ids = {m.id for m in linked}
        assert m2.id in linked_ids
        assert m3.id in linked_ids

    @pytest.mark.asyncio
    async def test_get_linked_memories_depth2(self, memory_store: MemoryStore):
        """get_linked_memories should follow chains at depth 2."""
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)
        m3 = await memory_store.save(content="C", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "similar")
        await memory_store.add_link(m2.id, m3.id, "similar")

        linked = await memory_store.get_linked_memories(m1.id, depth=2)
        linked_ids = {m.id for m in linked}
        assert m2.id in linked_ids
        assert m3.id in linked_ids

    @pytest.mark.asyncio
    async def test_delete_cascades_memory_links(self, memory_store: MemoryStore):
        """Deleting a memory removes its memory_links rows."""
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)
        await memory_store.add_link(m1.id, m2.id, "similar", bidirectional=True)

        await memory_store.delete(m1.id)

        # m2 should have no links after m1 is deleted (CASCADE)
        links = await memory_store.get_links_from(m2.id)
        links_to = await memory_store.get_links_to(m2.id)
        assert len(links) == 0
        assert len(links_to) == 0


class TestCoactivationDecay:
    """Tests for coactivation decay."""

    @pytest.mark.asyncio
    async def test_decay_coactivation(self, memory_store: MemoryStore):
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)
        await memory_store.bump_coactivation(m1.id, m2.id, delta=0.5)

        pruned = await memory_store.decay_coactivation(factor=0.5)
        # weight 0.5 * 0.5 = 0.25, still > 0.01, so not pruned
        assert pruned == 0

        # Decay to near-zero — weight 0.25 * 0.01 = 0.0025 < 0.01, pruned immediately
        pruned2 = await memory_store.decay_coactivation(factor=0.01)
        assert pruned2 > 0

    @pytest.mark.asyncio
    async def test_bump_coactivation_no_full_load(self, memory_store: MemoryStore):
        """bump_coactivation should work without loading full Memory objects."""
        m1 = await memory_store.save(content="A", auto_link=False)
        m2 = await memory_store.save(content="B", auto_link=False)

        result = await memory_store.bump_coactivation(m1.id, m2.id, delta=0.3)
        assert result is True

    @pytest.mark.asyncio
    async def test_bump_coactivation_nonexistent(self, memory_store: MemoryStore):
        """bump_coactivation returns False for nonexistent IDs."""
        m1 = await memory_store.save(content="A", auto_link=False)
        result = await memory_store.bump_coactivation(m1.id, "nonexistent", delta=0.1)
        assert result is False
