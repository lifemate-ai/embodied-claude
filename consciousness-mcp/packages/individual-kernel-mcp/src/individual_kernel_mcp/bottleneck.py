"""ActionBottleneck — one external action per tick (Phase 2.3).

Critique B finding 1 of the plan-of-record is decisive: LLM verbal output
must become a first-class gated action, otherwise the unity invariant of
"one externally visible effect per tick" leaks. The bottleneck enforces
that on the data layer.

Behavior:
  - First commit_action() for a tick_id succeeds, installs
    tick_frames.chosen_action_ref, and returns CommitActionResult(ok=True,
    deferred=False).
  - Second (and later) commit_action() calls for the same tick_id are
    deferred: they return CommitActionResult(ok=False, deferred=True) and
    write a counterfactuals row with source='attention_lost_bid', keyed by
    tick_id. The first action's chosen_action_ref is never overwritten.
  - Internal cognition (recall, hypothesize, reflect_attention_schema,
    introspect) is intentionally NOT routed through here — the bottleneck
    only governs externally-visible action. Free internal access stays free.

See consciousness-mcp/AGENTS.md for vocabulary discipline.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from social_core.db import SocialDB

from individual_kernel_mcp.counterfactual import (
    CounterfactualInput,
    CounterfactualStore,
)
from individual_kernel_mcp.frame import TickFrameStore


class CommitActionResult(BaseModel):
    """Result of ActionBottleneck.commit_action().

    ok=True, deferred=False  -> action installed as chosen_action_ref
    ok=False, deferred=True  -> deferred to next tick + counterfactual recorded
    """

    model_config = ConfigDict(extra="ignore")

    ok: bool
    deferred: bool
    tick_id: str
    action_ref: str
    counterfactual_id: str | None = None


class ActionBottleneck:
    """Single-action-per-tick gate over tick_frames.chosen_action_ref."""

    def __init__(
        self,
        *,
        db: SocialDB,
        frame_store: TickFrameStore,
        counterfactual_store: CounterfactualStore,
    ) -> None:
        self._db = db
        self._frames = frame_store
        self._cfs = counterfactual_store

    def commit_action(
        self,
        *,
        tick_id: str,
        action_ref: str,
        action_payload: dict | None = None,
        person_id: str | None = None,
    ) -> CommitActionResult:
        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT chosen_action_ref FROM tick_frames WHERE tick_id = ?",
                (tick_id,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"tick_frame {tick_id!r} does not exist; "
                    f"record the frame via TickFrameStore before committing an action."
                )
            claimed = connection.execute(
                """
                UPDATE tick_frames SET chosen_action_ref = ?
                WHERE tick_id = ?
                  AND (chosen_action_ref IS NULL OR chosen_action_ref = '')
                """,
                (action_ref, tick_id),
            )
            if claimed.rowcount == 1:
                return CommitActionResult(
                    ok=True,
                    deferred=False,
                    tick_id=tick_id,
                    action_ref=action_ref,
                )
            existing_row = connection.execute(
                "SELECT chosen_action_ref FROM tick_frames WHERE tick_id = ?",
                (tick_id,),
            ).fetchone()
            existing = existing_row["chosen_action_ref"]
            cf = self._cfs.record(
                CounterfactualInput(
                    tick_id=tick_id,
                    person_id=person_id,
                    chosen_action_ref=existing,
                    rejected_action=action_ref,
                    rejected_action_payload=action_payload or {},
                    reason=(
                        "action bottleneck: another action already committed this tick"
                    ),
                    source="attention_lost_bid",
                    evidence_type="remembered",
                )
            )
            return CommitActionResult(
                ok=False,
                deferred=True,
                tick_id=tick_id,
                action_ref=action_ref,
                counterfactual_id=cf.counterfactual_id,
            )
