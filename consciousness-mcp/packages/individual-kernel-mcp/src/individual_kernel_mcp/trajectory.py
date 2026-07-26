"""Imagined trajectories and protention distributions.

A trajectory is a persisted rollout of the generative model. Its status
lifecycle is the only mutable part; probabilities and steps are immutable
after insert. Imagined records never auto-promote to enacted: every legal
transition is explicit and audited in `status_history`.
"""

from __future__ import annotations

import json
import math
import secrets
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from social_core.db import SocialDB
from social_core.time import utc_now

from .generative_model import MAX_HORIZON, MODEL_VERSION, TrajectoryStep
from .workspace import SourceMode

PROBABILITY_TOLERANCE = 1e-6


class TrajectoryStatus(StrEnum):
    IMAGINED = "imagined"
    INTENDED = "intended"
    ENACTED = "enacted"
    OBSERVED = "observed"
    CONTRADICTED = "contradicted"
    PARTIALLY_OBSERVED = "partially_observed"
    EXPIRED = "expired"


_LEGAL_TRANSITIONS: dict[TrajectoryStatus, set[TrajectoryStatus]] = {
    TrajectoryStatus.IMAGINED: {TrajectoryStatus.INTENDED, TrajectoryStatus.EXPIRED},
    TrajectoryStatus.INTENDED: {TrajectoryStatus.ENACTED, TrajectoryStatus.IMAGINED},
    TrajectoryStatus.ENACTED: {
        TrajectoryStatus.OBSERVED,
        TrajectoryStatus.CONTRADICTED,
        TrajectoryStatus.PARTIALLY_OBSERVED,
    },
    TrajectoryStatus.OBSERVED: set(),
    TrajectoryStatus.CONTRADICTED: set(),
    TrajectoryStatus.PARTIALLY_OBSERVED: set(),
    TrajectoryStatus.EXPIRED: set(),
}


def normalized_entropy(probabilities: list[float]) -> float:
    """Shannon entropy normalized to [0, 1] over the positive entries."""
    values = [p for p in probabilities if p > 0.0]
    if len(values) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in values)
    return min(1.0, entropy / math.log(len(values)))


class ImaginedTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str = Field(default_factory=lambda: f"traj_{secrets.token_urlsafe(12)}")
    distribution_id: str
    owner_id: str = "self"
    field_id: str
    tick_id: str
    action_ref: str | None = None
    action_kind: str
    context_signature: str
    horizon: int = Field(ge=1, le=MAX_HORIZON)
    steps: list[TrajectoryStep] = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    source_mode: SourceMode = SourceMode.IMAGINED
    status: TrajectoryStatus = TrajectoryStatus.IMAGINED
    status_history: list[dict[str, str]] = Field(default_factory=list)
    expires_at: str | None = None
    fork_id: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @field_validator("source_mode")
    @classmethod
    def _imagined_only(cls, value: SourceMode) -> SourceMode:
        if value is not SourceMode.IMAGINED:
            raise ValueError("trajectories are imagined records; source_mode must be imagined")
        return value

    @model_validator(mode="after")
    def _steps_match_horizon(self) -> "ImaginedTrajectory":
        if len(self.steps) != self.horizon:
            raise ValueError(
                f"trajectory has {len(self.steps)} steps but horizon {self.horizon}"
            )
        return self


class ProtentionDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution_id: str = Field(default_factory=lambda: f"prot_{secrets.token_urlsafe(12)}")
    owner_id: str = "self"
    field_id: str
    tick_id: str
    action_ref: str | None = None
    trajectories: list[ImaginedTrajectory] = Field(min_length=1)
    entropy: float = Field(ge=0.0, le=1.0)
    model_version: str = MODEL_VERSION
    fork_id: str | None = None
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _probabilities_sum_to_one(self) -> "ProtentionDistribution":
        total = sum(t.probability for t in self.trajectories)
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError(f"trajectory probabilities sum to {total}, expected 1.0")
        return self


class TrajectoryStore:
    """Persistence for distributions and the only legal status mutator."""

    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def create_distribution(
        self, distribution: ProtentionDistribution
    ) -> ProtentionDistribution:
        with self._db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO protention_distributions(
                    distribution_id, owner_id, field_id, tick_id, action_ref,
                    entropy, model_version, trajectory_count, fork_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    distribution.distribution_id,
                    distribution.owner_id,
                    distribution.field_id,
                    distribution.tick_id,
                    distribution.action_ref,
                    distribution.entropy,
                    distribution.model_version,
                    len(distribution.trajectories),
                    distribution.fork_id,
                    distribution.created_at,
                ),
            )
            for trajectory in distribution.trajectories:
                connection.execute(
                    """
                    INSERT INTO imagined_trajectories(
                        trajectory_id, distribution_id, owner_id, field_id, tick_id,
                        action_ref, action_kind, context_signature, horizon,
                        steps_json, probability, uncertainty, source_mode, status,
                        status_history_json, expires_at, fork_id, created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trajectory.trajectory_id,
                        trajectory.distribution_id,
                        trajectory.owner_id,
                        trajectory.field_id,
                        trajectory.tick_id,
                        trajectory.action_ref,
                        trajectory.action_kind,
                        trajectory.context_signature,
                        trajectory.horizon,
                        json.dumps(
                            [step.model_dump(mode="json") for step in trajectory.steps],
                            sort_keys=True,
                        ),
                        trajectory.probability,
                        trajectory.uncertainty,
                        trajectory.source_mode.value,
                        trajectory.status.value,
                        json.dumps(trajectory.status_history, sort_keys=True),
                        trajectory.expires_at,
                        trajectory.fork_id,
                        trajectory.created_at,
                        trajectory.updated_at,
                    ),
                )
        return distribution

    def _row_to_trajectory(self, row: Any) -> ImaginedTrajectory:
        return ImaginedTrajectory(
            trajectory_id=row["trajectory_id"],
            distribution_id=row["distribution_id"],
            owner_id=row["owner_id"],
            field_id=row["field_id"],
            tick_id=row["tick_id"],
            action_ref=row["action_ref"],
            action_kind=row["action_kind"],
            context_signature=row["context_signature"],
            horizon=row["horizon"],
            steps=[
                TrajectoryStep.model_validate(step)
                for step in json.loads(row["steps_json"])
            ],
            probability=row["probability"],
            uncertainty=row["uncertainty"],
            source_mode=SourceMode(row["source_mode"]),
            status=TrajectoryStatus(row["status"]),
            status_history=json.loads(row["status_history_json"]),
            expires_at=row["expires_at"],
            fork_id=row["fork_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_trajectory(self, trajectory_id: str) -> ImaginedTrajectory | None:
        row = self._db.fetchone(
            "SELECT * FROM imagined_trajectories WHERE trajectory_id = ?",
            (trajectory_id,),
        )
        return None if row is None else self._row_to_trajectory(row)

    def get_distribution(self, distribution_id: str) -> ProtentionDistribution | None:
        row = self._db.fetchone(
            "SELECT * FROM protention_distributions WHERE distribution_id = ?",
            (distribution_id,),
        )
        if row is None:
            return None
        trajectory_rows = self._db.fetchall(
            "SELECT * FROM imagined_trajectories WHERE distribution_id = ? "
            "ORDER BY probability DESC, trajectory_id",
            (distribution_id,),
        )
        return ProtentionDistribution(
            distribution_id=row["distribution_id"],
            owner_id=row["owner_id"],
            field_id=row["field_id"],
            tick_id=row["tick_id"],
            action_ref=row["action_ref"],
            trajectories=[self._row_to_trajectory(item) for item in trajectory_rows],
            entropy=row["entropy"],
            model_version=row["model_version"],
            fork_id=row["fork_id"],
            created_at=row["created_at"],
        )

    def query(
        self,
        *,
        field_id: str | None = None,
        action_ref: str | None = None,
        status: TrajectoryStatus | str | None = None,
        owner_id: str = "self",
        limit: int = 50,
    ) -> list[ImaginedTrajectory]:
        clauses = ["owner_id = ?"]
        params: list[Any] = [owner_id]
        if field_id is not None:
            clauses.append("field_id = ?")
            params.append(field_id)
        if action_ref is not None:
            clauses.append("action_ref = ?")
            params.append(action_ref)
        if status is not None:
            clauses.append("status = ?")
            params.append(TrajectoryStatus(status).value)
        params.append(max(1, min(int(limit), 200)))
        rows = self._db.fetchall(
            "SELECT * FROM imagined_trajectories "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at DESC, trajectory_id LIMIT ?",
            tuple(params),
        )
        return [self._row_to_trajectory(row) for row in rows]

    def find_active_for_action(self, action_ref: str) -> ImaginedTrajectory | None:
        row = self._db.fetchone(
            "SELECT * FROM imagined_trajectories "
            "WHERE action_ref = ? AND status IN ('intended', 'enacted') "
            "ORDER BY created_at DESC LIMIT 1",
            (action_ref,),
        )
        return None if row is None else self._row_to_trajectory(row)

    def transition(
        self,
        trajectory_id: str,
        new_status: TrajectoryStatus,
        *,
        reason: str,
    ) -> ImaginedTrajectory:
        current = self.get_trajectory(trajectory_id)
        if current is None:
            raise ValueError(f"unknown trajectory {trajectory_id!r}")
        target = TrajectoryStatus(new_status)
        if target not in _LEGAL_TRANSITIONS[current.status]:
            raise ValueError(
                f"illegal trajectory transition {current.status.value!r} -> "
                f"{target.value!r}"
            )
        now = utc_now()
        history = [*current.status_history, {"status": target.value, "ts": now, "reason": reason}]
        self._db.execute(
            "UPDATE imagined_trajectories "
            "SET status = ?, status_history_json = ?, updated_at = ? "
            "WHERE trajectory_id = ?",
            (target.value, json.dumps(history, sort_keys=True), now, trajectory_id),
        )
        refreshed = self.get_trajectory(trajectory_id)
        assert refreshed is not None
        return refreshed
