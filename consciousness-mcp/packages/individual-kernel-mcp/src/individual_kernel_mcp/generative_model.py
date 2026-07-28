"""Action-conditioned generative model over discretized field context.

`count_v1`: Dirichlet-smoothed categorical outcome statistics per
(context signature, action kind), with a deterministic hierarchical backoff
instead of nearest-neighbor retrieval. No RNG anywhere; all learned state
lives in the shared SQLite database so hook subprocesses see one model.

This records and predicts functional state transitions. It makes no claim
about phenomenal properties of the modeled process.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator
from social_core.db import SocialDB
from social_core.time import utc_now

from .enacted_field import EnactedField
from .fork import LEARNABLE_KNOWLEDGE_SOURCE
from .workspace import SourceMode

if TYPE_CHECKING:
    from .experienced_transition import ExperiencedTransition

MODEL_VERSION = "count_v1"
DIRICHLET_ALPHA = 0.5
MAX_HORIZON = 5
NO_ACTION = "no_action"
NOVEL_BUCKET = "novel"

VALENCE_DEADBAND = 0.15
VALENCE_DELTA_DEADBAND = 0.05
AROUSAL_LOW = 0.33
AROUSAL_HIGH = 0.66
LATENCY_SHORT_MS = 1000
LATENCY_MID_MS = 10000

DEFAULT_ACTION_PRIOR_BUCKETS = (
    "ok/short/tool_result/=",
    "fail/short/tool_result/=",
)
DEFAULT_NO_ACTION_PRIOR_BUCKETS = (
    "na/na/desire/=",
    "na/na/event/=",
)

_STATS_FEATURE_COLUMNS = (
    "focus_kind",
    "trigger_kind",
    "dominant_desire",
    "valence_bucket",
    "arousal_bucket",
    "action_kind",
)


def focus_kind_of(content_ref: str | None) -> str:
    if not content_ref:
        return "none"
    if ":" in content_ref:
        return content_ref.split(":", 1)[0]
    return content_ref


def dominant_desire_of(need_vector: dict[str, float]) -> str:
    if not need_vector:
        return ""
    ranked = sorted(need_vector.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def valence_bucket_of(valence: float) -> str:
    if valence < -VALENCE_DEADBAND:
        return "neg"
    if valence > VALENCE_DEADBAND:
        return "pos"
    return "neu"


def arousal_bucket_of(arousal: float) -> str:
    if arousal < AROUSAL_LOW:
        return "low"
    if arousal > AROUSAL_HIGH:
        return "high"
    return "mid"


def latency_bucket_of(latency_ms: int | None) -> str:
    if latency_ms is None:
        return "na"
    if latency_ms < LATENCY_SHORT_MS:
        return "short"
    if latency_ms < LATENCY_MID_MS:
        return "mid"
    return "long"


def success_facet_of(success: bool | None) -> str:
    if success is None:
        return "na"
    return "ok" if success else "fail"


def valence_delta_facet_of(delta: float) -> str:
    if delta < -VALENCE_DELTA_DEADBAND:
        return "-"
    if delta > VALENCE_DELTA_DEADBAND:
        return "+"
    return "="


def action_kind_of(tool_name: str | None) -> str:
    return f"tool:{tool_name}" if tool_name else NO_ACTION


def outcome_bucket_of(
    *,
    success: bool | None,
    latency_ms: int | None,
    next_focus_kind: str,
    valence_delta: float,
) -> str:
    return "/".join(
        [
            success_facet_of(success),
            latency_bucket_of(latency_ms),
            next_focus_kind,
            valence_delta_facet_of(valence_delta),
        ]
    )


def parse_outcome_bucket(bucket: str) -> dict[str, str]:
    parts = bucket.split("/")
    if len(parts) != 4:
        raise ValueError(f"malformed outcome bucket: {bucket!r}")
    return {
        "success": parts[0],
        "latency": parts[1],
        "next_focus_kind": parts[2],
        "valence_delta": parts[3],
    }


def _shift_valence_bucket(bucket: str, delta_facet: str) -> str:
    order = ["neg", "neu", "pos"]
    index = order.index(bucket) if bucket in order else 1
    if delta_facet == "+":
        index = min(len(order) - 1, index + 1)
    elif delta_facet == "-":
        index = max(0, index - 1)
    return order[index]


class FieldBelief(BaseModel):
    """Compact belief-state projection of one enacted field or an imagined one."""

    model_config = ConfigDict(extra="forbid")

    belief_id: str = Field(default_factory=lambda: f"blf_{secrets.token_urlsafe(12)}")
    field_ref: str | None = None
    source_mode: SourceMode = SourceMode.INFERRED
    external: dict[str, Any] = Field(default_factory=dict)
    internal: dict[str, float] = Field(default_factory=dict)
    social: dict[str, Any] = Field(default_factory=dict)
    self_state: dict[str, Any] = Field(default_factory=dict)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    precision: dict[str, float] = Field(default_factory=dict)
    ts: str = Field(default_factory=utc_now)

    @classmethod
    def from_enacted_field(cls, field: EnactedField) -> "FieldBelief":
        intero = field.interoception
        need_vector = dict(intero.need_vector or {})
        return cls(
            field_ref=field.field_id,
            source_mode=field.reality_mode,
            external={
                "focal_content_ref": field.focal_content_ref,
                "focus_kind": focus_kind_of(field.focal_content_ref),
                "reality_score": field.reality_score,
                "trigger_kind": field.trigger_kind.value,
            },
            internal={
                "valence": intero.valence,
                "arousal": intero.arousal,
                "uncertainty": intero.uncertainty,
                "controllability": intero.controllability,
                "dominant_need_level": max(need_vector.values(), default=0.0),
            },
            social={"person_id": field.person_id},
            self_state={
                "ownership_score": field.agency_state.ownership_score,
                "pending_intention": field.agency_state.pending_intention_ref is not None,
                "attention_intensity": field.attention_intensity,
                "self_model_precision": field.precision.self_model,
            },
            uncertainty=intero.uncertainty,
            precision=field.precision.model_dump(),
        )


class ContextSignature(BaseModel):
    """Discretized (field, action) context; column-aligned with the stats table."""

    model_config = ConfigDict(extra="forbid")

    focus_kind: str
    trigger_kind: str
    dominant_desire: str = ""
    valence_bucket: str
    arousal_bucket: str
    action_kind: str

    @classmethod
    def from_field(cls, field: EnactedField, *, action_kind: str) -> "ContextSignature":
        intero = field.interoception
        return cls(
            focus_kind=focus_kind_of(field.focal_content_ref),
            trigger_kind=field.trigger_kind.value,
            dominant_desire=dominant_desire_of(dict(intero.need_vector or {})),
            valence_bucket=valence_bucket_of(intero.valence),
            arousal_bucket=arousal_bucket_of(intero.arousal),
            action_kind=action_kind,
        )

    @classmethod
    def from_key(cls, key: str) -> "ContextSignature":
        parts = key.split("|")
        if len(parts) != 6:
            raise ValueError(f"malformed context signature key: {key!r}")
        return cls(
            focus_kind=parts[0],
            trigger_kind=parts[1],
            dominant_desire=parts[2],
            valence_bucket=parts[3],
            arousal_bucket=parts[4],
            action_kind=parts[5],
        )

    def key(self) -> str:
        return "|".join(
            [
                self.focus_kind,
                self.trigger_kind,
                self.dominant_desire,
                self.valence_bucket,
                self.arousal_bucket,
                self.action_kind,
            ]
        )

    def backoff_chain(self) -> list[dict[str, str]]:
        """Progressively generalized column filters, most specific first."""
        full = {
            "focus_kind": self.focus_kind,
            "trigger_kind": self.trigger_kind,
            "dominant_desire": self.dominant_desire,
            "valence_bucket": self.valence_bucket,
            "arousal_bucket": self.arousal_bucket,
            "action_kind": self.action_kind,
        }
        level_1 = {k: v for k, v in full.items() if k != "dominant_desire"}
        level_2 = {k: v for k, v in level_1.items() if k != "arousal_bucket"}
        level_3 = {k: v for k, v in level_2.items() if k != "valence_bucket"}
        level_4 = {"action_kind": self.action_kind}
        return [full, level_1, level_2, level_3, level_4]


class OutcomePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bucket: str
    probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    support: float = Field(ge=0.0)
    basis: str


class OutcomeForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[OutcomePrediction]
    novel_probability: float = Field(ge=0.0, le=1.0)
    basis: str
    uncertainty: float = Field(ge=0.0, le=1.0)
    total_observations: float = Field(ge=0.0)


class TrajectoryStep(BaseModel):
    """One imagined step of a rollout; beliefs are always imagined-mode."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=1, le=MAX_HORIZON)
    predicted_belief: FieldBelief
    outcome_bucket: str
    conditional_probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    support_observations: float = Field(ge=0.0)
    basis: str

    @field_validator("predicted_belief")
    @classmethod
    def _belief_must_be_imagined(cls, value: FieldBelief) -> FieldBelief:
        if value.source_mode is not SourceMode.IMAGINED:
            raise ValueError("predicted beliefs must carry source_mode=imagined")
        return value


class GenerativeFieldModel(Protocol):
    def infer(self, field: EnactedField, *, action_kind: str = NO_ACTION) -> ContextSignature: ...

    def rollout(
        self,
        field: EnactedField,
        *,
        action_kind: str,
        horizon: int,
        first_bucket: str | None = None,
    ) -> list[TrajectoryStep]: ...

    def update(self, transition: "ExperiencedTransition") -> bool: ...


class CountBasedGenerativeFieldModel:
    """Dirichlet-smoothed count model over `generative_transition_stats`."""

    def __init__(
        self,
        db: SocialDB,
        *,
        owner_id: str = "self",
        alpha: float = DIRICHLET_ALPHA,
    ) -> None:
        self._db = db
        self.owner_id = owner_id
        self.alpha = alpha

    def infer(self, field: EnactedField, *, action_kind: str = NO_ACTION) -> ContextSignature:
        return ContextSignature.from_field(field, action_kind=action_kind)

    def _bucket_counts(self, filters: dict[str, str]) -> dict[str, float]:
        clauses = ["owner_id = ?"]
        params: list[Any] = [self.owner_id]
        for column in _STATS_FEATURE_COLUMNS:
            if column in filters:
                clauses.append(f"{column} = ?")
                params.append(filters[column])
        rows = self._db.fetchall(
            "SELECT outcome_bucket, SUM(observation_count) AS n "
            "FROM generative_transition_stats "
            f"WHERE {' AND '.join(clauses)} "
            "GROUP BY outcome_bucket",
            tuple(params),
        )
        return {
            row["outcome_bucket"]: float(row["n"])
            for row in rows
            if row["n"] is not None and float(row["n"]) > 0.0
        }

    def _prior_buckets(self, action_kind: str) -> tuple[str, ...]:
        if action_kind == NO_ACTION:
            return DEFAULT_NO_ACTION_PRIOR_BUCKETS
        return DEFAULT_ACTION_PRIOR_BUCKETS

    def forecast(self, signature: ContextSignature) -> OutcomeForecast:
        for level_index, filters in enumerate(signature.backoff_chain()):
            counts = self._bucket_counts(filters)
            total = sum(counts.values())
            if total > 0.0:
                basis = "count" if level_index == 0 else f"backoff:{level_index}"
                return self._smoothed_forecast(counts, total, basis)
        return self._prior_forecast(signature)

    def _smoothed_forecast(
        self, counts: dict[str, float], total: float, basis: str
    ) -> OutcomeForecast:
        k = len(counts) + 1  # +1 novel bucket
        denominator = total + self.alpha * k
        uncertainty = min(1.0, (self.alpha * k) / denominator)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        predictions = [
            OutcomePrediction(
                bucket=bucket,
                probability=(count + self.alpha) / denominator,
                uncertainty=uncertainty,
                support=count,
                basis=basis,
            )
            for bucket, count in ranked
        ]
        return OutcomeForecast(
            predictions=predictions,
            novel_probability=self.alpha / denominator,
            basis=basis,
            uncertainty=uncertainty,
            total_observations=total,
        )

    def _prior_forecast(self, signature: ContextSignature) -> OutcomeForecast:
        buckets = self._prior_buckets(signature.action_kind)
        k = len(buckets) + 1
        probability = 1.0 / k
        predictions = [
            OutcomePrediction(
                bucket=bucket,
                probability=probability,
                uncertainty=1.0,
                support=0.0,
                basis="prior",
            )
            for bucket in buckets
        ]
        return OutcomeForecast(
            predictions=predictions,
            novel_probability=probability,
            basis="prior",
            uncertainty=1.0,
            total_observations=0.0,
        )

    def probability_of(
        self, signature: ContextSignature, outcome_bucket: str
    ) -> tuple[float, str, float]:
        forecast = self.forecast(signature)
        for prediction in forecast.predictions:
            if prediction.bucket == outcome_bucket:
                return prediction.probability, forecast.basis, forecast.uncertainty
        return forecast.novel_probability, forecast.basis, forecast.uncertainty

    def expected_valence_delta(self, signature: ContextSignature) -> float:
        """How much affect this context is expected to move, in [-1, 1].

        Each reachable outcome bucket contributes its learned mean valence
        change, weighted by how likely that bucket is. With no history the
        forecast falls back to the uniform prior, whose buckets carry small
        symmetric deltas, so the estimate sits near zero rather than inventing
        an expectation.
        """

        forecast = self.forecast(signature)
        expected = sum(
            prediction.probability
            * self._mean_valence_delta(signature, prediction.bucket)
            for prediction in forecast.predictions
        )
        return max(-1.0, min(1.0, expected))

    def _mean_valence_delta(self, signature: ContextSignature, bucket: str) -> float:
        for filters in signature.backoff_chain():
            clauses = ["owner_id = ?", "outcome_bucket = ?"]
            params: list[Any] = [self.owner_id, bucket]
            for column in _STATS_FEATURE_COLUMNS:
                if column in filters:
                    clauses.append(f"{column} = ?")
                    params.append(filters[column])
            row = self._db.fetchone(
                "SELECT SUM(observation_count) AS n, SUM(sum_valence_delta) AS delta "
                "FROM generative_transition_stats "
                f"WHERE {' AND '.join(clauses)}",
                tuple(params),
            )
            if row is not None and row["n"] and float(row["n"]) > 0.0:
                return float(row["delta"] or 0.0) / float(row["n"])
        facet = parse_outcome_bucket(bucket)["valence_delta"]
        if facet == "+":
            return 0.1
        if facet == "-":
            return -0.1
        return 0.0

    def _imagined_next_belief(
        self,
        previous: FieldBelief,
        bucket: str,
        mean_valence_delta: float,
        uncertainty: float,
    ) -> FieldBelief:
        facets = parse_outcome_bucket(bucket)
        internal = dict(previous.internal)
        valence = max(-1.0, min(1.0, internal.get("valence", 0.0) + mean_valence_delta))
        internal["valence"] = valence
        external = dict(previous.external)
        external["focus_kind"] = facets["next_focus_kind"]
        external["focal_content_ref"] = None
        return FieldBelief(
            field_ref=None,
            source_mode=SourceMode.IMAGINED,
            external=external,
            internal=internal,
            social=dict(previous.social),
            self_state=dict(previous.self_state),
            uncertainty=uncertainty,
            precision=dict(previous.precision),
        )

    def _chained_signature(
        self, signature: ContextSignature, bucket: str
    ) -> ContextSignature:
        facets = parse_outcome_bucket(bucket)
        acted = signature.action_kind != NO_ACTION
        return ContextSignature(
            focus_kind=facets["next_focus_kind"],
            trigger_kind="tool_result" if acted else signature.trigger_kind,
            dominant_desire=signature.dominant_desire,
            valence_bucket=_shift_valence_bucket(
                signature.valence_bucket, facets["valence_delta"]
            ),
            arousal_bucket=signature.arousal_bucket,
            action_kind=NO_ACTION,
        )

    def rollout(
        self,
        field: EnactedField,
        *,
        action_kind: str,
        horizon: int,
        first_bucket: str | None = None,
    ) -> list[TrajectoryStep]:
        if not 1 <= horizon <= MAX_HORIZON:
            raise ValueError(f"horizon must be in 1..{MAX_HORIZON}, got {horizon}")
        signature = self.infer(field, action_kind=action_kind)
        belief = FieldBelief.from_enacted_field(field)
        steps: list[TrajectoryStep] = []
        for index in range(1, horizon + 1):
            forecast = self.forecast(signature)
            chosen: OutcomePrediction | None = None
            if index == 1 and first_bucket is not None:
                chosen = next(
                    (p for p in forecast.predictions if p.bucket == first_bucket),
                    None,
                )
                if chosen is None:
                    probability, basis, uncertainty = self.probability_of(
                        signature, first_bucket
                    )
                    chosen = OutcomePrediction(
                        bucket=first_bucket,
                        probability=probability,
                        uncertainty=uncertainty,
                        support=0.0,
                        basis=basis,
                    )
            if chosen is None:
                chosen = forecast.predictions[0]
            mean_delta = self._mean_valence_delta(signature, chosen.bucket)
            belief = self._imagined_next_belief(
                belief, chosen.bucket, mean_delta, chosen.uncertainty
            )
            steps.append(
                TrajectoryStep(
                    step_index=index,
                    predicted_belief=belief,
                    outcome_bucket=chosen.bucket,
                    conditional_probability=chosen.probability,
                    uncertainty=chosen.uncertainty,
                    support_observations=chosen.support,
                    basis=chosen.basis,
                )
            )
            signature = self._chained_signature(signature, chosen.bucket)
        return steps

    def update(self, transition: "ExperiencedTransition") -> bool:
        """Count one experienced transition into the stats, exactly once.

        Returns False when the transition was already applied (idempotent
        across processes via the `applied_at` row guard), and False for any
        route other than `experienced`: being told a contingency, or imagining
        one, must not train it. This is the gate that makes the route recorded
        on a transition consequential rather than cosmetic.
        """
        if transition.knowledge_source != LEARNABLE_KNOWLEDGE_SOURCE:
            return False
        signature = ContextSignature.from_key(transition.context_signature)
        now = utc_now()
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE experienced_transitions SET applied_at = ? "
                "WHERE transition_id = ? AND applied_at IS NULL",
                (now, transition.transition_id),
            )
            if cursor.rowcount == 0:
                return False
            connection.execute(
                """
                INSERT INTO generative_transition_stats(
                    owner_id, focus_kind, trigger_kind, dominant_desire,
                    valence_bucket, arousal_bucket, action_kind, outcome_bucket,
                    observation_count, sum_valence_delta, sum_latency_ms,
                    sum_prediction_error, last_transition_id, model_version,
                    first_observed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, focus_kind, trigger_kind, dominant_desire,
                            valence_bucket, arousal_bucket, action_kind, outcome_bucket)
                DO UPDATE SET
                    observation_count = observation_count + 1,
                    sum_valence_delta = sum_valence_delta + excluded.sum_valence_delta,
                    sum_latency_ms = sum_latency_ms + excluded.sum_latency_ms,
                    sum_prediction_error =
                        sum_prediction_error + excluded.sum_prediction_error,
                    last_transition_id = excluded.last_transition_id,
                    updated_at = excluded.updated_at
                """,
                (
                    transition.owner_id,
                    signature.focus_kind,
                    signature.trigger_kind,
                    signature.dominant_desire,
                    signature.valence_bucket,
                    signature.arousal_bucket,
                    signature.action_kind,
                    transition.outcome_bucket,
                    transition.valence_change,
                    float(transition.latency_ms or 0),
                    transition.mean_prediction_error,
                    transition.transition_id,
                    MODEL_VERSION,
                    now,
                    now,
                ),
            )
        return True
