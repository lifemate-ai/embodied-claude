"""Memory operations with PostgreSQL backend."""

import math
from datetime import datetime
from typing import Any

from .association import (
    AssociationDiagnostics,
    adaptive_search_params,
)
from .config import MemoryConfig
from .postgres_store import PostgresStore
from .predictive import (
    PredictiveDiagnostics,
    calculate_context_relevance,
    calculate_novelty_score,
    calculate_prediction_error,
)
from .types import (
    CameraPosition,
    Memory,
    MemorySearchResult,
    MemoryStats,
    ScoredMemory,
    SensoryData,
)
from .working_memory import WorkingMemoryBuffer
from .workspace import (
    WorkspaceCandidate,
    diversity_score,
    select_workspace_candidates,
)

# Emotion boost map: strong emotions make memories stickier
EMOTION_BOOST_MAP: dict[str, float] = {
    "excited": 0.4,
    "surprised": 0.35,
    "moved": 0.3,
    "sad": 0.25,
    "happy": 0.2,
    "nostalgic": 0.15,
    "curious": 0.1,
    "neutral": 0.0,
}


def calculate_time_decay(
    timestamp: str,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> float:
    """Calculate time decay factor.

    Returns:
        0.0 (forgotten) to 1.0 (fresh)
    """
    if now is None:
        now = datetime.now()

    try:
        memory_time = datetime.fromisoformat(timestamp)
    except ValueError:
        return 1.0

    age_seconds = (now - memory_time).total_seconds()
    if age_seconds < 0:
        return 1.0

    age_days = age_seconds / 86400
    decay = math.pow(2, -age_days / half_life_days)
    return max(0.0, min(1.0, decay))


def calculate_emotion_boost(emotion: str) -> float:
    """Return emotion-based boost value."""
    return EMOTION_BOOST_MAP.get(emotion, 0.0)


def calculate_importance_boost(importance: int) -> float:
    """Return importance-based boost (0.0 to 0.4)."""
    clamped = max(1, min(5, importance))
    return (clamped - 1) / 10


def calculate_final_score(
    semantic_distance: float,
    time_decay: float,
    emotion_boost: float,
    importance_boost: float,
    semantic_weight: float = 1.0,
    decay_weight: float = 0.3,
    emotion_weight: float = 0.2,
    importance_weight: float = 0.2,
) -> float:
    """Calculate final score (lower is better)."""
    decay_penalty = (1.0 - time_decay) * decay_weight
    total_boost = emotion_boost * emotion_weight + importance_boost * importance_weight
    final = semantic_distance * semantic_weight + decay_penalty - total_boost
    return max(0.0, final)


class MemoryStore:
    """PostgreSQL-backed memory storage.

    Wraps PostgresStore while maintaining the same external interface.
    """

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._store = PostgresStore(config)
        self._working_memory = WorkingMemoryBuffer(capacity=20)

    async def connect(self) -> None:
        """Initialize PostgreSQL connection."""
        await self._store.connect()

    async def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        await self._store.disconnect()

    async def save(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "daily",
        episode_id: str | None = None,
        sensory_data: tuple[SensoryData, ...] = (),
        camera_position: CameraPosition | None = None,
        tags: tuple[str, ...] = (),
    ) -> Memory:
        """Save a new memory."""
        memory = await self._store.save(
            content=content,
            emotion=emotion,
            importance=importance,
            category=category,
            episode_id=episode_id,
            sensory_data=sensory_data,
            camera_position=camera_position,
            tags=tags,
        )
        await self._working_memory.add(memory)
        return memory

    async def search(
        self,
        query: str,
        n_results: int = 5,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity."""
        return await self._store.search(
            query=query,
            n_results=n_results,
            emotion_filter=emotion_filter,
            category_filter=category_filter,
            date_from=date_from,
            date_to=date_to,
        )

    async def recall(
        self,
        context: str,
        n_results: int = 3,
    ) -> list[MemorySearchResult]:
        """Recall relevant memories with time decay and emotion boost."""
        scored_results = await self.search_with_scoring(
            query=context,
            n_results=n_results,
            use_time_decay=True,
            use_emotion_boost=True,
        )
        return [
            MemorySearchResult(memory=sr.memory, distance=sr.final_score)
            for sr in scored_results
        ]

    async def list_recent(
        self,
        limit: int = 10,
        category_filter: str | None = None,
    ) -> list[Memory]:
        """List recent memories sorted by timestamp."""
        return await self._store.list_recent(limit=limit, category_filter=category_filter)

    async def get_stats(self) -> MemoryStats:
        """Get statistics about stored memories."""
        return await self._store.get_stats()

    async def search_with_scoring(
        self,
        query: str,
        n_results: int = 5,
        use_time_decay: bool = True,
        use_emotion_boost: bool = True,
        decay_half_life_days: float | None = None,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ScoredMemory]:
        """Search with time decay + emotion boost scoring."""
        return await self._store.search_with_scoring(
            query=query,
            n_results=n_results,
            use_time_decay=use_time_decay,
            use_emotion_boost=use_emotion_boost,
            decay_half_life_days=decay_half_life_days,
            emotion_filter=emotion_filter,
            category_filter=category_filter,
            date_from=date_from,
            date_to=date_to,
        )

    async def update_access(self, memory_id: str) -> None:
        """Update access info."""
        await self._store.update_access(memory_id)

    async def get_by_id(self, memory_id: str) -> Memory | None:
        """Get memory by ID."""
        return await self._store.get_by_id(memory_id)

    async def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        """Get multiple memories by IDs."""
        return await self._store.get_by_ids(memory_ids)

    async def save_with_auto_link(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 3,
        category: str = "daily",
        link_threshold: float = 0.8,
        max_links: int = 5,
    ) -> Memory:
        """Save with auto-linking to similar memories."""
        memory = await self._store.save_with_auto_link(
            content=content,
            emotion=emotion,
            importance=importance,
            category=category,
            link_threshold=link_threshold,
            max_links=max_links,
        )
        await self._working_memory.add(memory)
        return memory

    async def get_linked_memories(
        self,
        memory_id: str,
        depth: int = 1,
    ) -> list[Memory]:
        """Get linked memories up to given depth."""
        return await self._store.get_linked_memories(memory_id, depth)

    async def recall_with_chain(
        self,
        context: str,
        n_results: int = 3,
        chain_depth: int = 1,
    ) -> list[MemorySearchResult]:
        """Recall + linked memories."""
        main_results = await self.recall(context=context, n_results=n_results)

        seen_ids: set[str] = {r.memory.id for r in main_results}
        linked_memories: list[Memory] = []

        for result in main_results:
            linked = await self._store.get_linked_memories(
                memory_id=result.memory.id,
                depth=chain_depth,
            )
            for mem in linked:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    linked_memories.append(mem)

        linked_results = [
            MemorySearchResult(memory=mem, distance=999.0)
            for mem in linked_memories
        ]

        return main_results + linked_results

    def get_working_memory(self) -> WorkingMemoryBuffer:
        """Access working memory buffer."""
        return self._working_memory

    async def update_episode_id(self, memory_id: str, episode_id: str) -> None:
        """Update episode_id on a memory."""
        await self._store.update_episode_id(memory_id, episode_id)

    async def search_important_memories(
        self,
        min_importance: int = 4,
        min_access_count: int = 5,
        since: str | None = None,
        n_results: int = 10,
    ) -> list[Memory]:
        """Get important, frequently accessed memories."""
        return await self._store.search_important_memories(
            min_importance=min_importance,
            min_access_count=min_access_count,
            since=since,
            n_results=n_results,
        )

    async def get_all(self) -> list[Memory]:
        """Get all memories."""
        return await self._store.get_all()

    # Phase 5: Causal links

    async def add_causal_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str = "caused_by",
        note: str | None = None,
    ) -> None:
        """Add a causal link."""
        # Verify both memories exist
        source = await self._store.get_by_id(source_id)
        if source is None:
            raise ValueError(f"Source memory not found: {source_id}")
        target = await self._store.get_by_id(target_id)
        if target is None:
            raise ValueError(f"Target memory not found: {target_id}")

        await self._store.add_link(source_id, target_id, link_type, note)

    async def get_causal_chain(
        self,
        memory_id: str,
        direction: str = "backward",
        max_depth: int = 5,
    ) -> list[tuple[Memory, str]]:
        """Trace causal chain."""
        return await self._store.get_causal_chain(memory_id, direction, max_depth)

    # Phase 6: Divergent recall + consolidation

    async def update_memory_fields(self, memory_id: str, **fields: Any) -> bool:
        """Partial update of memory fields."""
        return await self._store.update_memory_fields(memory_id, **fields)

    async def record_activation(
        self,
        memory_id: str,
        prediction_error: float | None = None,
    ) -> bool:
        """Record activation on recall."""
        return await self._store.record_activation(memory_id, prediction_error)

    async def bump_coactivation(
        self,
        source_id: str,
        target_id: str,
        delta: float = 0.1,
    ) -> bool:
        """Increase coactivation weight."""
        return await self._store.bump_coactivation(source_id, target_id, delta)

    async def maybe_add_related_link(
        self,
        source_id: str,
        target_id: str,
        threshold: float = 0.6,
    ) -> bool:
        """Add related link if coactivation exceeds threshold."""
        return await self._store.maybe_add_related_link(source_id, target_id, threshold)

    async def recall_divergent(
        self,
        context: str,
        n_results: int = 5,
        max_branches: int = 3,
        max_depth: int = 3,
        temperature: float = 0.7,
        include_diagnostics: bool = False,
        record_activation: bool = True,
    ) -> tuple[list[MemorySearchResult], dict[str, Any]]:
        """Divergent recall with associative expansion + workspace competition."""
        n_results = max(1, min(20, n_results))
        seed_size = max(3, min(25, n_results * 3))
        seeds = await self.search_with_scoring(query=context, n_results=seed_size)
        if not seeds:
            return [], {}

        branch_limit, depth_limit = adaptive_search_params(
            context=context,
            requested_branches=max_branches,
            requested_depth=max_depth,
            seed_count=len(seeds),
        )

        # Use SQL-based association spreading
        seed_ids = [item.memory.id for item in seeds]
        expanded_tuples = await self._store.spread_associations(
            seed_ids=seed_ids,
            max_depth=depth_limit,
            max_branches=branch_limit,
        )

        # Build diagnostics from the SQL results
        expanded_memories = [m for m, _, _ in expanded_tuples if m.id not in set(seed_ids)]
        assoc_diag = AssociationDiagnostics(
            branches_used=branch_limit,
            depth_used=depth_limit,
            traversed_edges=len(expanded_tuples),
            expanded_nodes=len(expanded_memories),
            avg_branching_factor=len(expanded_tuples) / max(1, len(seed_ids)),
        )

        distance_map = {item.memory.id: item.semantic_distance for item in seeds}
        all_candidates: dict[str, Memory] = {}
        for item in seeds:
            all_candidates[item.memory.id] = item.memory
        for memory in expanded_memories:
            all_candidates[memory.id] = memory

        workspace_candidates: list[WorkspaceCandidate] = []
        prediction_errors: list[float] = []
        novelty_scores: list[float] = []

        for memory in all_candidates.values():
            semantic_distance = distance_map.get(memory.id)
            if semantic_distance is None:
                relevance = calculate_context_relevance(context, memory)
            else:
                relevance = 1.0 / (1.0 + max(0.0, semantic_distance))

            pe = calculate_prediction_error(context, memory)
            novelty = calculate_novelty_score(memory, pe)
            eb = calculate_emotion_boost(memory.emotion)
            normalized_emotion = max(0.0, min(1.0, eb / 0.4))

            prediction_errors.append(pe)
            novelty_scores.append(novelty)

            workspace_candidates.append(
                WorkspaceCandidate(
                    memory=memory,
                    relevance=relevance,
                    novelty=novelty,
                    prediction_error=pe,
                    emotion_boost=normalized_emotion,
                )
            )

        selected = select_workspace_candidates(
            candidates=workspace_candidates,
            max_results=n_results,
            temperature=temperature,
        )

        results: list[MemorySearchResult] = []
        selected_memories: list[Memory] = []
        for candidate, utility in selected:
            selected_memories.append(candidate.memory)
            if record_activation:
                await self.record_activation(
                    candidate.memory.id,
                    prediction_error=candidate.prediction_error,
                )
                await self.update_memory_fields(
                    candidate.memory.id,
                    novelty_score=candidate.novelty,
                    prediction_error=candidate.prediction_error,
                )
            score_distance = max(0.0, 1.0 - utility)
            results.append(MemorySearchResult(memory=candidate.memory, distance=score_distance))

        if not include_diagnostics:
            return results, {}

        diagnostics = self._build_divergent_diagnostics(
            context=context,
            association=assoc_diag,
            selected=selected_memories,
            prediction_errors=prediction_errors,
            novelty_scores=novelty_scores,
            branch_limit=branch_limit,
            depth_limit=depth_limit,
        )
        return results, diagnostics

    async def get_association_diagnostics(
        self,
        context: str,
        sample_size: int = 20,
    ) -> dict[str, Any]:
        """Return association diagnostics without committing activation updates."""
        n_results = max(3, min(20, sample_size))
        _, diagnostics = await self.recall_divergent(
            context=context,
            n_results=n_results,
            max_branches=4,
            max_depth=3,
            include_diagnostics=True,
            record_activation=False,
        )
        return diagnostics

    async def consolidate_memories(
        self,
        window_hours: int = 24,
        max_replay_events: int = 200,
        link_update_strength: float = 0.2,
    ) -> dict[str, int]:
        """Run consolidation using batch SQL."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max(1, window_hours))
        delta = max(0.05, min(1.0, link_update_strength))

        stats = await self._store.consolidate_batch(
            cutoff_timestamp=cutoff,
            delta=delta,
        )

        # Count recent memories for refreshed count
        recent = await self._store.list_recent(limit=max_replay_events)
        refreshed = len([m for m in recent if m.timestamp >= cutoff.isoformat()])

        return {
            "replay_events": stats.get("coactivation_updates", 0),
            "coactivation_updates": stats.get("coactivation_updates", 0),
            "link_updates": stats.get("link_updates", 0),
            "refreshed_memories": refreshed,
        }

    def _build_divergent_diagnostics(
        self,
        context: str,
        association: AssociationDiagnostics,
        selected: list[Memory],
        prediction_errors: list[float],
        novelty_scores: list[float],
        branch_limit: int,
        depth_limit: int,
    ) -> dict[str, Any]:
        """Build divergent recall diagnostics."""
        avg_pe = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
        avg_nov = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        predictive = PredictiveDiagnostics(
            avg_prediction_error=avg_pe,
            avg_novelty=avg_nov,
        )

        return {
            "context": context,
            "branches_used": association.branches_used,
            "depth_used": association.depth_used,
            "adaptive_branch_limit": branch_limit,
            "adaptive_depth_limit": depth_limit,
            "traversed_edges": association.traversed_edges,
            "expanded_nodes": association.expanded_nodes,
            "avg_branching_factor": association.avg_branching_factor,
            "selected_count": len(selected),
            "diversity_score": diversity_score(selected),
            "avg_prediction_error": predictive.avg_prediction_error,
            "avg_novelty": predictive.avg_novelty,
        }
