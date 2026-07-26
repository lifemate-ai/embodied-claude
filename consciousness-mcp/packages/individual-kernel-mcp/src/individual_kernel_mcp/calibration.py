"""Prediction-calibration metrics over stored trajectory predictions.

Pure scoring functions (no database) plus a `CalibrationScorer` that resolves
step-1 predictions against experienced transitions and reports Brier score,
log loss, expected calibration error, and per-component mean absolute errors,
always alongside persistence and uniform baselines. Numbers below the
reliability floor are reported but flagged provisional. Deeper-step (horizon
>= 2) resolution reuses the `prediction_resolutions` table and lands with the
calibration batch work in a later change.
"""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

from .generative_model import parse_outcome_bucket

RELIABLE_FLOOR = 20
LOG_LOSS_EPSILON = 1e-3
ECE_BINS = 10
UNIFORM_BASELINE_PROBABILITY = 1.0 / 3.0

_COMPONENT_KEYS = (
    "exteroceptive",
    "interoceptive",
    "social",
    "mnemonic",
    "latency",
    "focus_kind_error",
    "valence_delta_error",
)


def brier_score(pairs: list[tuple[float, bool]]) -> float:
    if not pairs:
        return 0.0
    return sum((p - (1.0 if hit else 0.0)) ** 2 for p, hit in pairs) / len(pairs)


def log_loss(pairs: list[tuple[float, bool]], *, epsilon: float = LOG_LOSS_EPSILON) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for p, hit in pairs:
        clipped = min(1.0 - epsilon, max(epsilon, p))
        total += -math.log(clipped if hit else 1.0 - clipped)
    return total / len(pairs)


def expected_calibration_error(
    pairs: list[tuple[float, bool]], *, bins: int = ECE_BINS
) -> tuple[float, list[dict[str, float]]]:
    if not pairs:
        return 0.0, []
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, hit in pairs:
        index = min(bins - 1, int(p * bins))
        buckets[index].append((p, hit))
    total = len(pairs)
    ece = 0.0
    summaries: list[dict[str, float]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_confidence = sum(p for p, _ in bucket) / len(bucket)
        hit_rate = sum(1.0 for _, hit in bucket if hit) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_confidence - hit_rate)
        summaries.append(
            {
                "bin": float(index),
                "mean_confidence": mean_confidence,
                "hit_rate": hit_rate,
                "count": float(len(bucket)),
            }
        )
    return ece, summaries


class CalibrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_resolved: int = Field(ge=0)
    reliable: bool
    brier: float = Field(ge=0.0)
    log_loss: float = Field(ge=0.0)
    ece: float = Field(ge=0.0)
    bins: list[dict[str, float]] = Field(default_factory=list)
    top_1: float = Field(ge=0.0, le=1.0)
    per_component_mae: dict[str, float] = Field(default_factory=dict)
    baseline_persistence_brier: float = Field(ge=0.0)
    baseline_uniform_brier: float = Field(ge=0.0)
    uncertainty_notes: list[str] = Field(default_factory=list)


class CalibrationScorer:
    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def resolve_step_one(self, owner_id: str = "self") -> int:
        """Match linked step-1 predictions to their observed transitions.

        Idempotent: `prediction_resolutions` is unique per (trajectory, step).
        """
        rows = self._db.fetchall(
            """
            SELECT t.transition_id, t.outcome_bucket, t.intended_trajectory_id,
                   t.prediction_errors_json, tr.steps_json
            FROM experienced_transitions t
            JOIN imagined_trajectories tr
              ON tr.trajectory_id = t.intended_trajectory_id
            WHERE t.owner_id = ? AND t.intended_trajectory_id IS NOT NULL
            """,
            (owner_id,),
        )
        resolved = 0
        for row in rows:
            steps = json.loads(row["steps_json"])
            if not steps:
                continue
            predicted_bucket = str(steps[0]["outcome_bucket"])
            hit = predicted_bucket == row["outcome_bucket"]
            errors = json.loads(row["prediction_errors_json"])
            component_errors = {
                key: float(errors[key]) for key in _COMPONENT_KEYS if key in errors
            }
            try:
                with self._db.transaction() as connection:
                    connection.execute(
                        """
                        INSERT INTO prediction_resolutions(
                            resolution_id, trajectory_id, step_index,
                            transition_id, hit, component_errors_json, resolved_at
                        ) VALUES (?, ?, 1, ?, ?, ?, ?)
                        """,
                        (
                            f"res_{secrets.token_urlsafe(12)}",
                            row["intended_trajectory_id"],
                            row["transition_id"],
                            1 if hit else 0,
                            json.dumps(component_errors, sort_keys=True),
                            utc_now(),
                        ),
                    )
                resolved += 1
            except sqlite3.IntegrityError:
                continue
        return resolved

    def report(self, *, owner_id: str = "self", window: int = 200) -> CalibrationReport:
        self.resolve_step_one(owner_id)
        limit = max(1, min(int(window), 1000))
        rows = self._db.fetchall(
            """
            SELECT r.hit, r.component_errors_json, tr.steps_json,
                   t.context_signature, t.outcome_bucket
            FROM prediction_resolutions r
            JOIN imagined_trajectories tr ON tr.trajectory_id = r.trajectory_id
            JOIN experienced_transitions t
              ON t.intended_trajectory_id = r.trajectory_id
            WHERE tr.owner_id = ? AND r.step_index = 1
            ORDER BY r.resolved_at DESC, r.resolution_id LIMIT ?
            """,
            (owner_id, limit),
        )
        pairs: list[tuple[float, bool]] = []
        persistence_pairs: list[tuple[float, bool]] = []
        uniform_pairs: list[tuple[float, bool]] = []
        component_sums: dict[str, list[float]] = {}
        for row in rows:
            steps = json.loads(row["steps_json"])
            probability = float(steps[0]["conditional_probability"]) if steps else 0.0
            hit = bool(row["hit"])
            pairs.append((probability, hit))
            uniform_pairs.append((UNIFORM_BASELINE_PROBABILITY, hit))
            previous_focus = row["context_signature"].split("|", 1)[0]
            observed_focus = parse_outcome_bucket(row["outcome_bucket"])[
                "next_focus_kind"
            ]
            persistence_pairs.append((1.0, previous_focus == observed_focus))
            for key, value in json.loads(row["component_errors_json"]).items():
                component_sums.setdefault(key, []).append(float(value))

        n_resolved = len(pairs)
        ece, bins = expected_calibration_error(pairs)
        reliable = n_resolved >= RELIABLE_FLOOR
        notes = [
            "Calibration measures the count model's stated probabilities against "
            "observed outcome buckets; it is not a phenomenal claim.",
        ]
        if not reliable:
            notes.append(
                f"Provisional: only {n_resolved} resolved predictions "
                f"(reliability floor is {RELIABLE_FLOOR})."
            )
        return CalibrationReport(
            n_resolved=n_resolved,
            reliable=reliable,
            brier=brier_score(pairs),
            log_loss=log_loss(pairs),
            ece=ece,
            bins=bins,
            top_1=(
                sum(1.0 for _, hit in pairs if hit) / n_resolved if n_resolved else 0.0
            ),
            per_component_mae={
                key: sum(values) / len(values)
                for key, values in sorted(component_sums.items())
            },
            baseline_persistence_brier=brier_score(persistence_pairs),
            baseline_uniform_brier=brier_score(uniform_pairs),
            uncertainty_notes=notes,
        )

    def detail(self, *, owner_id: str = "self", window: int = 200) -> dict[str, Any]:
        report = self.report(owner_id=owner_id, window=window)
        return {
            "brier": report.brier,
            "log_loss": report.log_loss,
            "ece": report.ece,
            "top_1": report.top_1,
            "n_resolved": report.n_resolved,
            "reliable": report.reliable,
        }
