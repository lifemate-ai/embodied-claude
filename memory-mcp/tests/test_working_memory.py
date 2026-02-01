"""Tests for WorkingMemoryBuffer."""

import pytest
from datetime import datetime, timezone

from src.memory_mcp.types import Memory
from src.memory_mcp.working_memory import WorkingMemoryBuffer


class TestWorkingMemoryBasic:
    """Basic working memory operations."""

    @pytest.mark.asyncio
    async def test_add_and_get_recent(self):
        """Test adding memories and retrieving recent ones."""
        buffer = WorkingMemoryBuffer(capacity=5)

        # Add 3 memories
        mem1 = Memory(
            id="1",
            content="First memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="neutral",
            importance=3,
            category="daily",
        )
        mem2 = Memory(
            id="2",
            content="Second memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="happy",
            importance=4,
            category="daily",
        )
        mem3 = Memory(
            id="3",
            content="Third memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="excited",
            importance=5,
            category="observation",
        )

        await buffer.add(mem1)
        await buffer.add(mem2)
        await buffer.add(mem3)

        # Get recent 2
        recent = await buffer.get_recent(n=2)

        assert len(recent) == 2
        assert recent[0].id == "3"  # Most recent first
        assert recent[1].id == "2"

    @pytest.mark.asyncio
    async def test_get_all(self):
        """Test getting all memories in buffer."""
        buffer = WorkingMemoryBuffer(capacity=3)

        mem1 = Memory(
            id="1",
            content="Memory 1",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="neutral",
            importance=3,
            category="daily",
        )
        mem2 = Memory(
            id="2",
            content="Memory 2",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="neutral",
            importance=3,
            category="daily",
        )

        await buffer.add(mem1)
        await buffer.add(mem2)

        all_memories = await buffer.get_all()

        assert len(all_memories) == 2
        assert all_memories[0].id == "2"  # Newest first
        assert all_memories[1].id == "1"

    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = WorkingMemoryBuffer(capacity=3)

        # Add 5 memories (exceeds capacity)
        for i in range(5):
            mem = Memory(
                id=str(i),
                content=f"Memory {i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                emotion="neutral",
                importance=3,
                category="daily",
            )
            await buffer.add(mem)

        all_memories = await buffer.get_all()

        # Should only have last 3
        assert len(all_memories) == 3
        assert all_memories[0].id == "4"  # Most recent
        assert all_memories[1].id == "3"
        assert all_memories[2].id == "2"

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the buffer."""
        buffer = WorkingMemoryBuffer(capacity=5)

        mem = Memory(
            id="1",
            content="Memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="neutral",
            importance=3,
            category="daily",
        )
        await buffer.add(mem)

        assert buffer.size() == 1

        await buffer.clear()

        assert buffer.size() == 0
        all_memories = await buffer.get_all()
        assert len(all_memories) == 0

    @pytest.mark.asyncio
    async def test_size(self):
        """Test size tracking."""
        buffer = WorkingMemoryBuffer(capacity=10)

        assert buffer.size() == 0

        mem = Memory(
            id="1",
            content="Memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="neutral",
            importance=3,
            category="daily",
        )
        await buffer.add(mem)

        assert buffer.size() == 1


class TestWorkingMemoryEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_recent_empty_buffer(self):
        """Test get_recent on empty buffer."""
        buffer = WorkingMemoryBuffer(capacity=5)

        recent = await buffer.get_recent(n=10)

        assert len(recent) == 0

    @pytest.mark.asyncio
    async def test_get_recent_more_than_available(self):
        """Test requesting more memories than available."""
        buffer = WorkingMemoryBuffer(capacity=5)

        mem = Memory(
            id="1",
            content="Memory",
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="neutral",
            importance=3,
            category="daily",
        )
        await buffer.add(mem)

        recent = await buffer.get_recent(n=10)

        # Should return all available (1)
        assert len(recent) == 1
