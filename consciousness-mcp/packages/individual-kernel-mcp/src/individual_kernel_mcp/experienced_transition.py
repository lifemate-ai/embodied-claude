"""Unified experienced-transition records.

One row per arrived-at field: previous state, attributed action, observed
result, component prediction errors, and affect deltas. `next_field_id` is
UNIQUE, so recording is idempotent; learning consumption is guarded by
`applied_at` (see `CountBasedGenerativeFieldModel.update`).

Source discipline: a transition row never carries imagined or remembered
source modes, and `knowledge_source` is fixed to 'experienced' in this
version (told/imagined/replayed ingestion arrives with the fork experiments;
the schema already reserves the values).
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from social_core.db import SocialDB
from social_core.time import utc_now

from .workspace import SourceMode

_ALLOWED_SOURCE_MODES = {SourceMode.LIVE, SourceMode.INFERRED, SourceMode.MIXED}
_KNOWLEDGE_SOURCES_V1 = {"experienced"}


class ExperiencedTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transition_id: str = Field(default_factory=lambda: f"etx_{secrets.token_urlsafe(12)}")
    owner_id: str = "self"
    previous_field_id: str | None = None
    next_field_id: str
    previous_tick_id: str | None = None
    next_tick_id: str
    action_ref: str | None = None
    outcome_ref: str | None = None
    intended_trajectory_id: str | None = None
    distribution_id: str | None = None
    context_signature: str
    action_kind: str
    outcome_bucket: str
    observed_external: dict[str, Any] = Field(default_factory=dict)
    observed_internal: dict[str, float] = Field(default_factory=dict)
    observed_social: dict[str, Any] = Field(default_factory=dict)
    prediction_errors: dict[str, float] = Field(default_factory=dict)
    mean_prediction_error: float = Field(ge=0.0, le=1.0)
    valence_before: float = Field(ge=-1.0, le=1.0)
    valence_after: float = Field(ge=-1.0, le=1.0)
    valence_change: float = Field(ge=-2.0, le=2.0)
    arousal_before: float = Field(ge=0.0, le=1.0)
    arousal_after: float = Field(ge=0.0, le=1.0)
    controllability_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    agency_confidence: float = Field(ge=0.0, le=1.0)
    ownership_confidence: float = Field(ge=0.0, le=1.0)
    success: bool | None = None
    latency_ms: int | None = None
    expected_latency_ms: int | None = None
    source_mode: SourceMode
    knowledge_source: str = "experienced"
    info_content_hash: str | None = None
    pose_before_ref: str | None = None
    pose_after_ref: str | None = None
    body_delta: dict[str, Any] = Field(default_factory=dict)
    process_meta_ref: str | None = None
    allostatic_snapshot: dict[str, Any] = Field(default_factory=dict)
    hor_ref: str | None = None
    fork_id: str | None = None
    applied_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)

    @field_validator("source_mode")
    @classmethod
    def _reject_non_enacted_sources(cls, value: SourceMode) -> SourceMode:
        if value not in _ALLOWED_SOURCE_MODES:
            raise ValueError(
                "experienced transitions require live/inferred/mixed source modes; "
                "imagined or remembered content must not be recorded as experienced"
            )
        return value

    @field_validator("knowledge_source")
    @classmethod
    def _knowledge_source_v1(cls, value: str) -> str:
        if value not in _KNOWLEDGE_SOURCES_V1:
            raise ValueError(
                "knowledge_source is fixed to 'experienced' in this version"
            )
        return value


class ExperiencedTransitionStore:
    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def record(
        self, transition: ExperiencedTransition
    ) -> tuple[ExperiencedTransition, bool]:
        """Insert idempotently; each field is arrived-at exactly once."""
        existing = self.get_by_next_field(transition.next_field_id)
        if existing is not None:
            return existing, False
        try:
            with self._db.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO experienced_transitions(
                        transition_id, owner_id, previous_field_id, next_field_id,
                        previous_tick_id, next_tick_id, action_ref, outcome_ref,
                        intended_trajectory_id, distribution_id, context_signature,
                        action_kind, outcome_bucket, observed_external_json,
                        observed_internal_json, observed_social_json,
                        prediction_errors_json, mean_prediction_error,
                        valence_before, valence_after, valence_change,
                        arousal_before, arousal_after, controllability_delta,
                        agency_confidence, ownership_confidence, success,
                        latency_ms, expected_latency_ms, source_mode,
                        knowledge_source, info_content_hash, pose_before_ref,
                        pose_after_ref, body_delta_json, process_meta_ref,
                        allostatic_snapshot_json, hor_ref, fork_id, applied_at,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transition.transition_id,
                        transition.owner_id,
                        transition.previous_field_id,
                        transition.next_field_id,
                        transition.previous_tick_id,
                        transition.next_tick_id,
                        transition.action_ref,
                        transition.outcome_ref,
                        transition.intended_trajectory_id,
                        transition.distribution_id,
                        transition.context_signature,
                        transition.action_kind,
                        transition.outcome_bucket,
                        json.dumps(transition.observed_external, sort_keys=True),
                        json.dumps(transition.observed_internal, sort_keys=True),
                        json.dumps(transition.observed_social, sort_keys=True),
                        json.dumps(transition.prediction_errors, sort_keys=True),
                        transition.mean_prediction_error,
                        transition.valence_before,
                        transition.valence_after,
                        transition.valence_change,
                        transition.arousal_before,
                        transition.arousal_after,
                        transition.controllability_delta,
                        transition.agency_confidence,
                        transition.ownership_confidence,
                        None if transition.success is None else int(transition.success),
                        transition.latency_ms,
                        transition.expected_latency_ms,
                        transition.source_mode.value,
                        transition.knowledge_source,
                        transition.info_content_hash,
                        transition.pose_before_ref,
                        transition.pose_after_ref,
                        json.dumps(transition.body_delta, sort_keys=True),
                        transition.process_meta_ref,
                        json.dumps(transition.allostatic_snapshot, sort_keys=True),
                        transition.hor_ref,
                        transition.fork_id,
                        transition.applied_at,
                        json.dumps(transition.metadata, sort_keys=True),
                        transition.created_at,
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self.get_by_next_field(transition.next_field_id)
            if existing is not None:
                return existing, False
            raise
        return transition, True

    def _row_to_transition(self, row: Any) -> ExperiencedTransition:
        return ExperiencedTransition(
            transition_id=row["transition_id"],
            owner_id=row["owner_id"],
            previous_field_id=row["previous_field_id"],
            next_field_id=row["next_field_id"],
            previous_tick_id=row["previous_tick_id"],
            next_tick_id=row["next_tick_id"],
            action_ref=row["action_ref"],
            outcome_ref=row["outcome_ref"],
            intended_trajectory_id=row["intended_trajectory_id"],
            distribution_id=row["distribution_id"],
            context_signature=row["context_signature"],
            action_kind=row["action_kind"],
            outcome_bucket=row["outcome_bucket"],
            observed_external=json.loads(row["observed_external_json"]),
            observed_internal=json.loads(row["observed_internal_json"]),
            observed_social=json.loads(row["observed_social_json"]),
            prediction_errors=json.loads(row["prediction_errors_json"]),
            mean_prediction_error=row["mean_prediction_error"],
            valence_before=row["valence_before"],
            valence_after=row["valence_after"],
            valence_change=row["valence_change"],
            arousal_before=row["arousal_before"],
            arousal_after=row["arousal_after"],
            controllability_delta=row["controllability_delta"],
            agency_confidence=row["agency_confidence"],
            ownership_confidence=row["ownership_confidence"],
            success=None if row["success"] is None else bool(row["success"]),
            latency_ms=row["latency_ms"],
            expected_latency_ms=row["expected_latency_ms"],
            source_mode=SourceMode(row["source_mode"]),
            knowledge_source=row["knowledge_source"],
            info_content_hash=row["info_content_hash"],
            pose_before_ref=row["pose_before_ref"],
            pose_after_ref=row["pose_after_ref"],
            body_delta=json.loads(row["body_delta_json"]),
            process_meta_ref=row["process_meta_ref"],
            allostatic_snapshot=json.loads(row["allostatic_snapshot_json"]),
            hor_ref=row["hor_ref"],
            fork_id=row["fork_id"],
            applied_at=row["applied_at"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def get(self, transition_id: str) -> ExperiencedTransition | None:
        row = self._db.fetchone(
            "SELECT * FROM experienced_transitions WHERE transition_id = ?",
            (transition_id,),
        )
        return None if row is None else self._row_to_transition(row)

    def get_by_next_field(self, next_field_id: str) -> ExperiencedTransition | None:
        row = self._db.fetchone(
            "SELECT * FROM experienced_transitions WHERE next_field_id = ?",
            (next_field_id,),
        )
        return None if row is None else self._row_to_transition(row)

    def query(
        self,
        *,
        since: str | None = None,
        action_kind: str | None = None,
        context_signature: str | None = None,
        owner_id: str = "self",
        limit: int = 50,
    ) -> list[ExperiencedTransition]:
        clauses = ["owner_id = ?"]
        params: list[Any] = [owner_id]
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        if action_kind is not None:
            clauses.append("action_kind = ?")
            params.append(action_kind)
        if context_signature is not None:
            clauses.append("context_signature = ?")
            params.append(context_signature)
        params.append(max(1, min(int(limit), 200)))
        rows = self._db.fetchall(
            "SELECT * FROM experienced_transitions "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at DESC, transition_id LIMIT ?",
            tuple(params),
        )
        return [self._row_to_transition(row) for row in rows]

    def link_trajectory(
        self,
        transition_id: str,
        *,
        intended_trajectory_id: str | None,
        distribution_id: str | None,
    ) -> bool:
        """Complete the prediction linkage exactly once (never rewrites)."""
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE experienced_transitions "
                "SET intended_trajectory_id = ?, distribution_id = ? "
                "WHERE transition_id = ? AND intended_trajectory_id IS NULL "
                "AND distribution_id IS NULL",
                (intended_trajectory_id, distribution_id, transition_id),
            )
            return cursor.rowcount > 0
