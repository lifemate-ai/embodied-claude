"""Sleep-like replay and consolidation routines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import MemoryStore
    from .types import Memory


@dataclass(frozen=True)
class ConsolidationStats:
    """Summary statistics for replay execution."""

    replay_events: int
    coactivation_updates: int
    link_updates: int
    refreshed_memories: int

    def to_dict(self) -> dict[str, int]:
        return {
            "replay_events": self.replay_events,
            "coactivation_updates": self.coactivation_updates,
            "link_updates": self.link_updates,
            "refreshed_memories": self.refreshed_memories,
        }


class ConsolidationEngine:
    """Replay memories and update association strengths."""

    async def run(
        self,
        store: "MemoryStore",
        window_hours: int = 24,
        max_replay_events: int = 200,
        link_update_strength: float = 0.2,
        job_id: str | None = None,
        include_global: bool = True,
        include_shared: bool = True,
    ) -> ConsolidationStats:
        """Run consolidation replay.

        Args:
            store: MemoryStore instance
            window_hours: Lookback window in hours
            max_replay_events: Maximum replay events
            link_update_strength: Strength for coactivation updates
            job_id: Job ID for job isolation
            include_global: Include global memories
            include_shared: Include shared memories
        """
        cutoff = datetime.now() - timedelta(hours=max(1, window_hours))
        recent = await store.list_recent(
            limit=max(max_replay_events * 2, 50),
            job_id=job_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        recent = [m for m in recent if self._is_after(m, cutoff)]

        if len(recent) < 2:
            return ConsolidationStats(0, 0, 0, len(recent))

        replay_events = 0
        coactivation_updates = 0
        link_updates = 0
        refreshed_ids: set[str] = set()

        for idx in range(len(recent) - 1):
            if replay_events >= max_replay_events:
                break

            left = recent[idx]
            right = recent[idx + 1]

            delta = max(0.05, min(1.0, link_update_strength))
            await store.bump_coactivation(
                left.id,
                right.id,
                delta=delta,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            coactivation_updates += 2

            left_error = max(0.0, left.prediction_error * 0.9)
            right_error = max(0.0, right.prediction_error * 0.9)
            await store.record_activation(
                left.id,
                prediction_error=left_error,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            await store.record_activation(
                right.id,
                prediction_error=right_error,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            refreshed_ids.add(left.id)
            refreshed_ids.add(right.id)

            if await store.maybe_add_related_link(
                left.id,
                right.id,
                threshold=0.6,
                job_id=job_id,
                include_global=include_global,
                include_shared=include_shared,
            ):
                link_updates += 1

            replay_events += 1

        return ConsolidationStats(
            replay_events=replay_events,
            coactivation_updates=coactivation_updates,
            link_updates=link_updates,
            refreshed_memories=len(refreshed_ids),
        )

    def _is_after(self, memory: "Memory", cutoff: datetime) -> bool:
        try:
            timestamp = datetime.fromisoformat(memory.timestamp)
        except ValueError:
            return False
        return timestamp >= cutoff
