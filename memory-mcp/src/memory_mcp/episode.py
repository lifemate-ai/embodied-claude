"""Episode memory management with PostgreSQL backend."""

import uuid
from typing import TYPE_CHECKING

from .types import Episode, Memory

if TYPE_CHECKING:
    from .memory import MemoryStore


class EpisodeManager:
    """Manage episodic memories using PostgreSQL backend."""

    def __init__(self, memory_store: "MemoryStore"):
        self._memory_store = memory_store

    @property
    def _store(self):
        return self._memory_store._store

    async def create_episode(
        self,
        title: str,
        memory_ids: list[str],
        participants: list[str] | None = None,
        auto_summarize: bool = True,
    ) -> Episode:
        """Create an episode from a set of memories."""
        if not memory_ids:
            raise ValueError("memory_ids cannot be empty")

        memories = await self._memory_store.get_by_ids(memory_ids)
        if not memories:
            raise ValueError("No memories found for the given IDs")

        memories.sort(key=lambda m: m.timestamp)

        if auto_summarize:
            summary = " → ".join(m.content[:50] for m in memories)
        else:
            summary = ""

        most_important = max(memories, key=lambda m: m.importance)
        emotion = most_important.emotion

        episode = Episode(
            id=str(uuid.uuid4()),
            title=title,
            start_time=memories[0].timestamp,
            end_time=memories[-1].timestamp if len(memories) > 1 else None,
            memory_ids=tuple(m.id for m in memories),
            participants=tuple(participants or []),
            location_context=None,
            summary=summary,
            emotion=emotion,
            importance=max(m.importance for m in memories),
        )

        await self._store.save_episode(episode)

        for memory in memories:
            await self._memory_store.update_episode_id(memory.id, episode.id)

        return episode

    async def search_episodes(self, query: str, n_results: int = 5) -> list[Episode]:
        """Search episodes by summary."""
        return await self._store.search_episodes(query, n_results)

    async def get_episode_by_id(self, episode_id: str) -> Episode | None:
        """Get episode by ID."""
        return await self._store.get_episode_by_id(episode_id)

    async def get_episode_memories(self, episode_id: str) -> list[Memory]:
        """Get memories in an episode, in chronological order."""
        episode = await self._store.get_episode_by_id(episode_id)
        if episode is None:
            raise ValueError(f"Episode not found: {episode_id}")

        memories = await self._memory_store.get_by_ids(list(episode.memory_ids))
        memories.sort(key=lambda m: m.timestamp)
        return memories

    async def list_all_episodes(self) -> list[Episode]:
        """List all episodes."""
        return await self._store.list_all_episodes()

    async def delete_episode(self, episode_id: str) -> None:
        """Delete an episode (memories are kept)."""
        await self._store.delete_episode(episode_id)
