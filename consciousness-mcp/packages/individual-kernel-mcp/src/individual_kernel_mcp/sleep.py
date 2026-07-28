"""SleepConsolidator — quiet-hours-gated scheduler glue.

Phase 2.1 ships only the SCHEDULING SHELL: quiet-hours gate, briefing
assembly from existing typed records (counterfactuals + interpretation
shifts), and morning_briefing.json emission for autonomous-action.sh
to surface on first session of day.

Memory-mcp consolidate / recall_divergent / append_daybook calls are
out of scope for Phase 2.1 — they are wired in a subsequent PR once the
tick-cadence question is settled. The current implementation queries
social-core counters and counterfactuals directly so the shell is
testable and useful immediately.

See consciousness-mcp/AGENTS.md for vocabulary discipline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from boundary_mcp.store import BoundaryStore
from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

from individual_kernel_mcp.counterfactual import (
    CounterfactualRecord,
    CounterfactualStore,
)

DEFAULT_BRIEFING_PATH = Path.home() / ".claude" / "morning_briefing.json"

DEFAULT_WINDOW_HOURS = 24

QuietPredicate = Callable[[str], bool]
Clock = Callable[[], str]


class MorningBriefing(BaseModel):
    """One-page summary emitted by SleepConsolidator for first-session-of-day surfacing.

    `summary_line` is the load-bearing field — autonomous-action.sh injects
    it verbatim into MORNING_PROMPT, so it must stay short (<=120 chars).
    Everything else is structured detail for the kernel or for a /consolidate
    skill invocation to read.
    """

    model_config = ConfigDict(extra="ignore")

    generated_at: str
    window_start: str
    window_end: str
    summary_line: str = Field(max_length=120)
    counterfactual_count: int = 0
    interpretation_shift_count: int = 0
    top_counterfactuals: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SleepConsolidationResult(BaseModel):
    """Outcome of SleepConsolidator.run()."""

    model_config = ConfigDict(extra="ignore")

    ran: bool
    skipped_reason: str | None = None
    briefing: MorningBriefing | None = None
    briefing_path: str | None = None


def quiet_hours_from_policy(store: BoundaryStore) -> QuietPredicate:
    """Ask the configured policy whether an instant is inside quiet hours.

    The default used to be a predicate that returned True unconditionally, so
    the gate was open at every hour and the consolidator ran whenever anything
    called it. Reusing `BoundaryStore` keeps one definition of quiet hours: the
    `[global] quiet_hours` and `timezone` in socialPolicy.toml that already
    govern speaking and posting.
    """

    def predicate(ts: str) -> bool:
        return store.get_quiet_mode_state(ts=ts).active

    return predicate


def _always_quiet(_ts: str) -> bool:
    """Fallback for a caller that supplies no policy.

    Kept permissive on purpose: a consolidator constructed without a boundary
    store has no way to know the hour, and refusing every run would be a silent
    failure rather than a safe one. Callers that can supply a store should.
    """
    return True


class SleepConsolidator:
    """Compose existing engines into a quiet-hours-gated scheduled job.

    Phase 2.1 scope: gate by quiet-hours predicate, assemble a
    MorningBriefing from CounterfactualStore + social-core tables,
    write to disk, return result. Memory-mcp engine integration is
    deferred to Phase 2.2+.
    """

    def __init__(
        self,
        *,
        db: SocialDB,
        counterfactual_store: CounterfactualStore,
        briefing_path: Path = DEFAULT_BRIEFING_PATH,
        is_quiet_hours: QuietPredicate = _always_quiet,
        clock: Clock = utc_now,
        window_hours: int = DEFAULT_WINDOW_HOURS,
    ) -> None:
        self._db = db
        self._cf = counterfactual_store
        self._briefing_path = Path(briefing_path)
        self._is_quiet_hours = is_quiet_hours
        self._clock = clock
        self._window_hours = window_hours

    def run(
        self,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> SleepConsolidationResult:
        now = self._clock()
        if not force and not self._is_quiet_hours(now):
            return SleepConsolidationResult(
                ran=False,
                skipped_reason="not in quiet hours (use force=True to override)",
            )

        window_start = _shift_iso(now, -self._window_hours)
        cfs = self._cf.query(since=window_start, limit=500)
        shift_count = self._count_interpretation_shifts(window_start)
        top_cf_payload = _top_counterfactuals(cfs, k=3)

        summary_line = _build_summary_line(
            cf_count=len(cfs),
            shift_count=shift_count,
        )

        briefing = MorningBriefing(
            generated_at=now,
            window_start=window_start,
            window_end=now,
            summary_line=summary_line,
            counterfactual_count=len(cfs),
            interpretation_shift_count=shift_count,
            top_counterfactuals=top_cf_payload,
            notes=[],
        )

        briefing_path: str | None = None
        if not dry_run:
            self._briefing_path.parent.mkdir(parents=True, exist_ok=True)
            self._briefing_path.write_text(
                json.dumps(briefing.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            briefing_path = str(self._briefing_path)

        return SleepConsolidationResult(
            ran=True,
            briefing=briefing,
            briefing_path=briefing_path,
        )

    def _count_interpretation_shifts(self, since: str) -> int:
        row = self._db.fetchone(
            "SELECT COUNT(*) AS n FROM interpretation_shifts WHERE ts >= ?",
            (since,),
        )
        return int(row["n"]) if row is not None else 0


def _build_summary_line(*, cf_count: int, shift_count: int) -> str:
    return (
        f"前夜の振り返り：拒否した選択肢 {cf_count} 件、解釈の更新 {shift_count} 件"
    )


def _top_counterfactuals(
    cfs: list[CounterfactualRecord], *, k: int = 3
) -> list[dict]:
    ranked = sorted(
        cfs,
        key=lambda c: (c.importance, c.ts),
        reverse=True,
    )[:k]
    return [
        {
            "counterfactual_id": c.counterfactual_id,
            "rejected_action": c.rejected_action,
            "source": c.source,
            "reason": c.reason,
            "importance": c.importance,
            "ts": c.ts,
        }
        for c in ranked
    ]


def _shift_iso(ts: str, delta_hours: int) -> str:
    dt = _parse_iso(ts)
    shifted = dt + timedelta(hours=delta_hours)
    return shifted.isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> datetime:
    normalized = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
