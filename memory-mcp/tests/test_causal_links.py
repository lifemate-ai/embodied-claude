"""Tests for Phase 5: Causal Links."""

import pytest

from memory_mcp.types import LinkType, MemoryLink


class TestLinkType:
    """Tests for LinkType enum."""

    def test_link_type_values(self) -> None:
        """Test all link type values."""
        assert LinkType.SIMILAR.value == "similar"
        assert LinkType.CAUSED_BY.value == "caused_by"
        assert LinkType.LEADS_TO.value == "leads_to"
        assert LinkType.RELATED.value == "related"

    def test_link_type_is_string_enum(self) -> None:
        """Test that LinkType is a string enum."""
        assert isinstance(LinkType.SIMILAR, str)
        assert LinkType.CAUSED_BY == "caused_by"


class TestMemoryLink:
    """Tests for MemoryLink dataclass."""

    def test_memory_link_creation(self) -> None:
        """Test creating a MemoryLink."""
        link = MemoryLink(
            target_id="test-target-id",
            link_type="caused_by",
            created_at="2026-02-01T12:00:00",
            note="Test note",
        )
        assert link.target_id == "test-target-id"
        assert link.link_type == "caused_by"
        assert link.created_at == "2026-02-01T12:00:00"
        assert link.note == "Test note"

    def test_memory_link_without_note(self) -> None:
        """Test creating a MemoryLink without note."""
        link = MemoryLink(
            target_id="test-target-id",
            link_type="leads_to",
            created_at="2026-02-01T12:00:00",
        )
        assert link.note is None

    def test_memory_link_to_dict(self) -> None:
        """Test converting MemoryLink to dict."""
        link = MemoryLink(
            target_id="test-target-id",
            link_type="caused_by",
            created_at="2026-02-01T12:00:00",
            note="Test note",
        )
        data = link.to_dict()
        assert data == {
            "target_id": "test-target-id",
            "link_type": "caused_by",
            "created_at": "2026-02-01T12:00:00",
            "note": "Test note",
        }

    def test_memory_link_from_dict(self) -> None:
        """Test creating MemoryLink from dict."""
        data = {
            "target_id": "test-target-id",
            "link_type": "related",
            "created_at": "2026-02-01T12:00:00",
            "note": "Related memory",
        }
        link = MemoryLink.from_dict(data)
        assert link.target_id == "test-target-id"
        assert link.link_type == "related"
        assert link.created_at == "2026-02-01T12:00:00"
        assert link.note == "Related memory"

    def test_memory_link_from_dict_without_note(self) -> None:
        """Test creating MemoryLink from dict without note."""
        data = {
            "target_id": "test-target-id",
            "link_type": "similar",
            "created_at": "2026-02-01T12:00:00",
        }
        link = MemoryLink.from_dict(data)
        assert link.note is None

    def test_memory_link_immutable(self) -> None:
        """Test that MemoryLink is immutable (frozen dataclass)."""
        link = MemoryLink(
            target_id="test-target-id",
            link_type="caused_by",
            created_at="2026-02-01T12:00:00",
        )
        with pytest.raises(AttributeError):
            link.target_id = "new-id"  # type: ignore


# Integration tests use memory_store fixture from conftest.py


class TestCausalLinksIntegration:
    """Integration tests for causal links in MemoryStore."""

    @pytest.mark.asyncio
    async def test_add_causal_link(self, memory_store) -> None:
        """Test adding a causal link between memories."""
        mem1 = await memory_store.save(content="Memory 1: The cause")
        mem2 = await memory_store.save(content="Memory 2: The effect")

        await memory_store.add_causal_link(
            source_id=mem2.id,
            target_id=mem1.id,
            link_type="caused_by",
            note="mem1 caused mem2",
        )

        # Verify link via get_links on the store
        links = await memory_store._store.get_links(mem2.id)
        assert len(links) == 1
        assert links[0].target_id == mem1.id
        assert links[0].link_type == "caused_by"
        assert links[0].note == "mem1 caused mem2"

    @pytest.mark.asyncio
    async def test_add_causal_link_duplicate_prevention(self, memory_store) -> None:
        """Test that duplicate links are not added."""
        mem1 = await memory_store.save(content="Memory 1")
        mem2 = await memory_store.save(content="Memory 2")

        await memory_store.add_causal_link(mem2.id, mem1.id, "caused_by")
        await memory_store.add_causal_link(mem2.id, mem1.id, "caused_by")

        links = await memory_store._store.get_links(mem2.id)
        assert len(links) == 1

    @pytest.mark.asyncio
    async def test_add_causal_link_different_types(self, memory_store) -> None:
        """Test adding different link types between same memories."""
        mem1 = await memory_store.save(content="Memory 1")
        mem2 = await memory_store.save(content="Memory 2")

        await memory_store.add_causal_link(mem2.id, mem1.id, "caused_by")
        await memory_store.add_causal_link(mem2.id, mem1.id, "related")

        links = await memory_store._store.get_links(mem2.id)
        assert len(links) == 2

    @pytest.mark.asyncio
    async def test_add_causal_link_invalid_source(self, memory_store) -> None:
        """Test adding link with invalid source ID."""
        import uuid
        mem1 = await memory_store.save(content="Memory 1")
        fake_id = str(uuid.uuid4())

        with pytest.raises(ValueError, match="Source memory not found"):
            await memory_store.add_causal_link(
                source_id=fake_id,
                target_id=mem1.id,
                link_type="caused_by",
            )

    @pytest.mark.asyncio
    async def test_add_causal_link_invalid_target(self, memory_store) -> None:
        """Test adding link with invalid target ID."""
        import uuid
        mem1 = await memory_store.save(content="Memory 1")
        fake_id = str(uuid.uuid4())

        with pytest.raises(ValueError, match="Target memory not found"):
            await memory_store.add_causal_link(
                source_id=mem1.id,
                target_id=fake_id,
                link_type="caused_by",
            )

    @pytest.mark.asyncio
    async def test_get_causal_chain_backward(self, memory_store) -> None:
        """Test tracing causal chain backward (finding causes)."""
        # Create chain: mem1 -> mem2 -> mem3
        mem1 = await memory_store.save(content="The root cause")
        mem2 = await memory_store.save(content="The intermediate effect")
        mem3 = await memory_store.save(content="The final effect")

        # Link: mem2 caused_by mem1, mem3 caused_by mem2
        await memory_store.add_causal_link(mem2.id, mem1.id, "caused_by")
        await memory_store.add_causal_link(mem3.id, mem2.id, "caused_by")

        # Trace backward from mem3
        chain = await memory_store.get_causal_chain(mem3.id, "backward", max_depth=5)

        assert len(chain) == 2
        assert chain[0][0].id == mem2.id
        assert chain[0][1] == "caused_by"
        assert chain[1][0].id == mem1.id
        assert chain[1][1] == "caused_by"

    @pytest.mark.asyncio
    async def test_get_causal_chain_forward(self, memory_store) -> None:
        """Test tracing causal chain forward (finding effects)."""
        # Create chain: mem1 -> mem2 -> mem3
        mem1 = await memory_store.save(content="The root cause")
        mem2 = await memory_store.save(content="The intermediate effect")
        mem3 = await memory_store.save(content="The final effect")

        # Link: mem1 leads_to mem2, mem2 leads_to mem3
        await memory_store.add_causal_link(mem1.id, mem2.id, "leads_to")
        await memory_store.add_causal_link(mem2.id, mem3.id, "leads_to")

        # Trace forward from mem1
        chain = await memory_store.get_causal_chain(mem1.id, "forward", max_depth=5)

        assert len(chain) == 2
        assert chain[0][0].id == mem2.id
        assert chain[0][1] == "leads_to"
        assert chain[1][0].id == mem3.id
        assert chain[1][1] == "leads_to"

    @pytest.mark.asyncio
    async def test_get_causal_chain_max_depth(self, memory_store) -> None:
        """Test that max_depth limits the chain length."""
        # Create long chain
        mems = []
        for i in range(5):
            mem = await memory_store.save(content=f"Memory {i}")
            mems.append(mem)

        # Link all in sequence
        for i in range(len(mems) - 1):
            await memory_store.add_causal_link(mems[i + 1].id, mems[i].id, "caused_by")

        # Trace with max_depth=2
        chain = await memory_store.get_causal_chain(mems[4].id, "backward", max_depth=2)

        assert len(chain) == 2  # Should stop at depth 2

    @pytest.mark.asyncio
    async def test_get_causal_chain_empty(self, memory_store) -> None:
        """Test getting causal chain for memory with no links."""
        mem = await memory_store.save(content="Isolated memory")

        chain = await memory_store.get_causal_chain(mem.id, "backward")

        assert len(chain) == 0

    @pytest.mark.asyncio
    async def test_get_causal_chain_invalid_direction(self, memory_store) -> None:
        """Test invalid direction parameter."""
        mem = await memory_store.save(content="Memory")

        with pytest.raises(ValueError, match="Invalid direction"):
            await memory_store.get_causal_chain(mem.id, "sideways")
