"""Tests for EpisodeManager."""

import pytest
from datetime import datetime, timezone

from src.memory_mcp.config import MemoryConfig
from src.memory_mcp.episode import EpisodeManager
from src.memory_mcp.memory import MemoryStore
from src.memory_mcp.types import Episode


@pytest.fixture
async def memory_store():
    """Create a MemoryStore instance for testing."""
    config = MemoryConfig(
        db_path=":memory:",  # In-memory for testing
        collection_name="test_memories",
    )
    store = MemoryStore(config)
    await store.connect()
    yield store
    await store.disconnect()


@pytest.fixture
async def episode_manager(memory_store):
    """Create an EpisodeManager instance."""
    collection = memory_store.get_episodes_collection()
    return EpisodeManager(memory_store, collection)


class TestEpisodeCreation:
    """Test episode creation."""

    @pytest.mark.asyncio
    async def test_create_episode_basic(self, memory_store, episode_manager):
        """Test creating a basic episode."""
        # Create 3 memories
        mem1 = await memory_store.save(
            content="Look around to find the sky",
            emotion="curious",
            importance=3,
        )
        mem2 = await memory_store.save(
            content="Found a window on the left",
            emotion="curious",
            importance=3,
        )
        mem3 = await memory_store.save(
            content="Found the morning sky! Beautiful blue and golden light",
            emotion="excited",
            importance=5,
        )

        # Create episode
        episode = await episode_manager.create_episode(
            title="Morning sky search",
            memory_ids=[mem1.id, mem2.id, mem3.id],
            participants=["幼馴染"],
        )

        assert episode.title == "Morning sky search"
        assert len(episode.memory_ids) == 3
        assert episode.participants == ("幼馴染",)
        assert episode.emotion == "excited"  # From highest importance memory
        assert episode.importance == 5
        assert episode.start_time == mem1.timestamp
        assert episode.end_time == mem3.timestamp

    @pytest.mark.asyncio
    async def test_create_episode_auto_summarize(self, memory_store, episode_manager):
        """Test auto-summarize feature."""
        mem1 = await memory_store.save(content="First memory", importance=3)
        mem2 = await memory_store.save(content="Second memory", importance=3)

        episode = await episode_manager.create_episode(
            title="Test Episode",
            memory_ids=[mem1.id, mem2.id],
            auto_summarize=True,
        )

        # Summary should contain both memory contents
        assert "First memory" in episode.summary
        assert "Second memory" in episode.summary
        assert " → " in episode.summary

    @pytest.mark.asyncio
    async def test_create_episode_no_auto_summarize(
        self, memory_store, episode_manager
    ):
        """Test creating episode without auto-summarize."""
        mem1 = await memory_store.save(content="Memory 1", importance=3)

        episode = await episode_manager.create_episode(
            title="Test Episode",
            memory_ids=[mem1.id],
            auto_summarize=False,
        )

        assert episode.summary == ""

    @pytest.mark.asyncio
    async def test_create_episode_updates_memory_episode_id(
        self, memory_store, episode_manager
    ):
        """Test that creating an episode updates memory episode_ids."""
        mem1 = await memory_store.save(content="Memory 1", importance=3)
        mem2 = await memory_store.save(content="Memory 2", importance=3)

        episode = await episode_manager.create_episode(
            title="Test Episode",
            memory_ids=[mem1.id, mem2.id],
        )

        # Retrieve memories and check episode_id
        memories = await memory_store.get_by_ids([mem1.id, mem2.id])

        for memory in memories:
            assert memory.episode_id == episode.id

    @pytest.mark.asyncio
    async def test_create_episode_empty_memory_ids(self, episode_manager):
        """Test creating episode with empty memory_ids raises error."""
        with pytest.raises(ValueError, match="memory_ids cannot be empty"):
            await episode_manager.create_episode(
                title="Empty Episode",
                memory_ids=[],
            )

    @pytest.mark.asyncio
    async def test_create_episode_single_memory(self, memory_store, episode_manager):
        """Test creating episode with single memory."""
        mem = await memory_store.save(content="Single memory", importance=4)

        episode = await episode_manager.create_episode(
            title="Single Memory Episode",
            memory_ids=[mem.id],
        )

        assert len(episode.memory_ids) == 1
        assert episode.start_time == mem.timestamp
        assert episode.end_time is None  # Only one memory


class TestEpisodeSearch:
    """Test episode search."""

    @pytest.mark.asyncio
    async def test_search_episodes(self, memory_store, episode_manager):
        """Test searching episodes by query."""
        # Create memories and episode
        mem1 = await memory_store.save(content="Morning sky search", importance=5)
        mem2 = await memory_store.save(content="Found beautiful sky", importance=4)

        await episode_manager.create_episode(
            title="Sky Search Adventure",
            memory_ids=[mem1.id, mem2.id],
        )

        # Search
        results = await episode_manager.search_episodes(query="sky", n_results=5)

        assert len(results) > 0
        assert any("sky" in ep.summary.lower() for ep in results)

    @pytest.mark.asyncio
    async def test_search_episodes_no_results(self, episode_manager):
        """Test searching with no matching episodes."""
        results = await episode_manager.search_episodes(
            query="nonexistent query xyz",
            n_results=5,
        )

        # May return empty or unrelated results
        assert isinstance(results, list)


class TestEpisodeRetrieval:
    """Test episode retrieval."""

    @pytest.mark.asyncio
    async def test_get_episode_by_id(self, memory_store, episode_manager):
        """Test getting episode by ID."""
        mem = await memory_store.save(content="Test memory", importance=3)

        created = await episode_manager.create_episode(
            title="Test Episode",
            memory_ids=[mem.id],
        )

        # Retrieve by ID
        retrieved = await episode_manager.get_episode_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == "Test Episode"

    @pytest.mark.asyncio
    async def test_get_episode_by_id_not_found(self, episode_manager):
        """Test getting non-existent episode returns None."""
        result = await episode_manager.get_episode_by_id("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_episode_memories(self, memory_store, episode_manager):
        """Test getting memories in an episode."""
        mem1 = await memory_store.save(content="First", importance=3)
        mem2 = await memory_store.save(content="Second", importance=3)
        mem3 = await memory_store.save(content="Third", importance=3)

        episode = await episode_manager.create_episode(
            title="Three Memories",
            memory_ids=[mem1.id, mem2.id, mem3.id],
        )

        # Get episode memories
        memories = await episode_manager.get_episode_memories(episode.id)

        assert len(memories) == 3
        # Should be in chronological order
        assert memories[0].content == "First"
        assert memories[1].content == "Second"
        assert memories[2].content == "Third"

    @pytest.mark.asyncio
    async def test_get_episode_memories_not_found(self, episode_manager):
        """Test getting memories for non-existent episode."""
        with pytest.raises(ValueError, match="Episode not found"):
            await episode_manager.get_episode_memories("nonexistent-id")

    @pytest.mark.asyncio
    async def test_list_all_episodes(self, memory_store, episode_manager):
        """Test listing all episodes."""
        # Create 2 episodes
        mem1 = await memory_store.save(content="Memory 1", importance=3)
        mem2 = await memory_store.save(content="Memory 2", importance=3)

        ep1 = await episode_manager.create_episode(
            title="Episode 1",
            memory_ids=[mem1.id],
        )
        ep2 = await episode_manager.create_episode(
            title="Episode 2",
            memory_ids=[mem2.id],
        )

        # List all
        all_episodes = await episode_manager.list_all_episodes()

        assert len(all_episodes) >= 2
        episode_ids = [ep.id for ep in all_episodes]
        assert ep1.id in episode_ids
        assert ep2.id in episode_ids


class TestEpisodeDeletion:
    """Test episode deletion."""

    @pytest.mark.asyncio
    async def test_delete_episode(self, memory_store, episode_manager):
        """Test deleting an episode."""
        mem = await memory_store.save(content="Test memory", importance=3)

        episode = await episode_manager.create_episode(
            title="To Be Deleted",
            memory_ids=[mem.id],
        )

        # Delete
        await episode_manager.delete_episode(episode.id)

        # Should not be found
        retrieved = await episode_manager.get_episode_by_id(episode.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_episode_clears_memory_episode_id(
        self, memory_store, episode_manager
    ):
        """Test that deleting episode clears memory episode_ids."""
        mem = await memory_store.save(content="Test memory", importance=3)

        episode = await episode_manager.create_episode(
            title="To Be Deleted",
            memory_ids=[mem.id],
        )

        # Verify episode_id is set
        memories_before = await memory_store.get_by_ids([mem.id])
        assert memories_before[0].episode_id == episode.id

        # Delete episode
        await episode_manager.delete_episode(episode.id)

        # Verify episode_id is cleared (None represents no episode)
        memories_after = await memory_store.get_by_ids([mem.id])
        assert memories_after[0].episode_id is None
