"""Deterministic workspace competition for enacted-field ticks.

Candidates point at source records and carry only a compact summary. Competition
is performed outside the language model so that winner selection, ignition, and
attention intensity are inspectable and reproducible.
"""

from __future__ import annotations

import json
import math
import secrets
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now


class CandidateKind(StrEnum):
    PERCEPT = "percept"
    MEMORY = "memory"
    INTEROCEPTIVE = "interoceptive"
    SOCIAL = "social"
    GOAL = "goal"
    ACTION = "action"
    SELF_MODEL = "self_model"


class CandidateSource(StrEnum):
    USER_EVENT = "user_event"
    SENSOR = "sensor"
    MEMORY = "memory"
    INTEROCEPTION = "interoception"
    DESIRE = "desire"
    SOCIAL = "social"
    PREVIOUS_FIELD = "previous_field"
    ATTENTION_SCHEMA = "attention_schema"
    HOR = "hor"
    TOOL_RESULT = "tool_result"
    EXPLICIT = "explicit"


class SourceMode(StrEnum):
    LIVE = "live"
    INFERRED = "inferred"
    REMEMBERED = "remembered"
    IMAGINED = "imagined"
    MIXED = "mixed"


class CompetitionWeights(BaseModel):
    """Configurable coefficients for workspace scoring."""

    model_config = ConfigDict(extra="forbid")

    surprise: float = 1.0
    need: float = 0.9
    goal: float = 0.9
    information: float = 0.7
    continuity: float = 0.65
    control: float = 0.7
    social: float = 0.65
    conflict: float = 0.8
    switching: float = 0.55
    ignition_threshold: float = 0.9
    conflict_margin: float = 0.12
    entropy_conflict_threshold: float = 0.78


class WorkspaceCandidate(BaseModel):
    """One compact bid for access to the committed field."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(default_factory=lambda: f"cand_{secrets.token_urlsafe(12)}")
    tick_id: str
    kind: CandidateKind
    content_ref: str
    content_summary: str = Field(default="", max_length=320)
    source: CandidateSource = CandidateSource.EXPLICIT
    source_mode: SourceMode
    modality: str = "internal"
    precision: float = Field(default=0.5, ge=0.0, le=1.0)
    prediction_error: float = Field(default=0.0, ge=-1.0, le=1.0)
    need_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_information_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    continuity_with_previous: float = Field(default=0.0, ge=0.0, le=1.0)
    controllability: float = Field(default=0.0, ge=0.0, le=1.0)
    social_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    switching_cost: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float = 0.0
    reportability: str = "mentionable"


class CompetitionResult(BaseModel):
    """Auditable result of one workspace competition."""

    model_config = ConfigDict(extra="forbid")

    tick_id: str
    winner: WorkspaceCandidate
    top_rejected: list[WorkspaceCandidate] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    margin: float
    entropy: float = Field(ge=0.0, le=1.0)
    ignited: bool
    conflicted: bool
    attention_intensity: float = Field(ge=0.0, le=1.0)
    deterministic_order: list[str] = Field(default_factory=list)


class WorkspaceEngine:
    """SQLite-backed candidate market with deterministic tie breaking."""

    def __init__(
        self,
        db: SocialDB,
        *,
        weights: CompetitionWeights | None = None,
    ) -> None:
        self._db = db
        self.weights = weights or CompetitionWeights()

    def score_candidate(self, candidate: WorkspaceCandidate) -> float:
        w = self.weights
        score = (
            w.surprise * candidate.precision * abs(candidate.prediction_error)
            + w.need * candidate.need_relevance
            + w.goal * candidate.goal_relevance
            + w.information * candidate.expected_information_gain
            + w.continuity * candidate.continuity_with_previous
            + w.control * candidate.controllability
            + w.social * candidate.social_relevance
            - w.conflict * candidate.conflict_penalty
            - w.switching * candidate.switching_cost
        )
        return round(score, 8)

    def add_candidate(self, candidate: WorkspaceCandidate) -> WorkspaceCandidate:
        scored = candidate.model_copy(update={"score": self.score_candidate(candidate)})
        self._db.execute(
            """
            INSERT INTO workspace_candidates(
                candidate_id, tick_id, kind, content_ref, content_summary,
                source, source_mode, modality, precision, prediction_error,
                need_relevance, goal_relevance, expected_information_gain,
                continuity_with_previous, controllability, social_relevance,
                conflict_penalty, switching_cost, score, reportability, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scored.candidate_id,
                scored.tick_id,
                scored.kind.value,
                scored.content_ref,
                scored.content_summary,
                scored.source.value,
                scored.source_mode.value,
                scored.modality,
                scored.precision,
                scored.prediction_error,
                scored.need_relevance,
                scored.goal_relevance,
                scored.expected_information_gain,
                scored.continuity_with_previous,
                scored.controllability,
                scored.social_relevance,
                scored.conflict_penalty,
                scored.switching_cost,
                scored.score,
                scored.reportability,
                utc_now(),
            ),
        )
        return scored

    def candidates_for_tick(self, tick_id: str) -> list[WorkspaceCandidate]:
        rows = self._db.fetchall(
            """
            SELECT * FROM workspace_candidates
            WHERE tick_id = ?
            ORDER BY score DESC, candidate_id ASC
            """,
            (tick_id,),
        )
        return [self._row_to_candidate(row) for row in rows]

    def compete(self, tick_id: str, *, top_rejected: int = 5) -> CompetitionResult:
        candidates = self.candidates_for_tick(tick_id)
        if not candidates:
            raise ValueError(f"tick {tick_id!r} has no workspace candidates")

        ranked = sorted(
            candidates,
            key=lambda item: (-item.score, item.content_ref, item.candidate_id),
        )
        winner = ranked[0]
        runner_score = ranked[1].score if len(ranked) > 1 else 0.0
        margin = max(0.0, winner.score - runner_score)
        entropy = self._normalized_entropy([item.score for item in ranked])
        ignited = winner.score >= self.weights.ignition_threshold
        conflicted = len(ranked) > 1 and (
            margin < self.weights.conflict_margin
            or entropy >= self.weights.entropy_conflict_threshold
        )
        intensity = self._attention_intensity(
            winner_score=winner.score,
            margin=margin,
            entropy=entropy,
            ignited=ignited,
        )

        with self._db.transaction() as connection:
            for rank, item in enumerate(ranked, start=1):
                connection.execute(
                    """
                    UPDATE workspace_candidates
                    SET rank = ?, selected = ?
                    WHERE candidate_id = ?
                    """,
                    (rank, 1 if rank == 1 else 0, item.candidate_id),
                )

        return CompetitionResult(
            tick_id=tick_id,
            winner=winner,
            top_rejected=ranked[1 : 1 + max(0, top_rejected)],
            scores={item.candidate_id: item.score for item in ranked},
            margin=round(margin, 8),
            entropy=round(entropy, 8),
            ignited=ignited,
            conflicted=conflicted,
            attention_intensity=round(intensity, 8),
            deterministic_order=[item.candidate_id for item in ranked],
        )

    def _attention_intensity(
        self,
        *,
        winner_score: float,
        margin: float,
        entropy: float,
        ignited: bool,
    ) -> float:
        threshold = max(0.01, self.weights.ignition_threshold)
        score_term = _clamp01(winner_score / (2.0 * threshold))
        margin_term = _clamp01(margin / threshold)
        certainty_term = 1.0 - entropy
        value = 0.5 * score_term + 0.3 * margin_term + 0.2 * certainty_term
        if not ignited:
            value *= 0.6
        return _clamp01(value)

    @staticmethod
    def _normalized_entropy(scores: list[float]) -> float:
        if len(scores) <= 1:
            return 0.0
        peak = max(scores)
        exps = [math.exp(min(40.0, score - peak)) for score in scores]
        total = sum(exps)
        probabilities = [value / total for value in exps]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
        return _clamp01(entropy / math.log(len(probabilities)))

    @staticmethod
    def _row_to_candidate(row) -> WorkspaceCandidate:
        return WorkspaceCandidate(
            candidate_id=row["candidate_id"],
            tick_id=row["tick_id"],
            kind=row["kind"],
            content_ref=row["content_ref"],
            content_summary=row["content_summary"],
            source=row["source"],
            source_mode=row["source_mode"],
            modality=row["modality"],
            precision=float(row["precision"]),
            prediction_error=float(row["prediction_error"]),
            need_relevance=float(row["need_relevance"]),
            goal_relevance=float(row["goal_relevance"]),
            expected_information_gain=float(row["expected_information_gain"]),
            continuity_with_previous=float(row["continuity_with_previous"]),
            controllability=float(row["controllability"]),
            social_relevance=float(row["social_relevance"]),
            conflict_penalty=float(row["conflict_penalty"]),
            switching_cost=float(row["switching_cost"]),
            score=float(row["score"]),
            reportability=row["reportability"],
        )


def competition_trace(result: CompetitionResult) -> dict[str, object]:
    """Return the stable subset persisted in a field's epistemic trace."""

    return {
        "winner_id": result.winner.candidate_id,
        "winner_ref": result.winner.content_ref,
        "scores": json.loads(json.dumps(result.scores, sort_keys=True)),
        "margin": result.margin,
        "entropy": result.entropy,
        "ignited": result.ignited,
        "conflicted": result.conflicted,
        "attention_intensity": result.attention_intensity,
        "top_rejected": [item.candidate_id for item in result.top_rejected],
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
