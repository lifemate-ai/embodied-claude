"""Sensorimotor contingency: did the body move the way it was told to?

`exclusive_causal_fit` previously restated whether the tool call returned
successfully. That cannot tell a movement the runtime commanded apart from one
somebody else caused, which is the only thing the term was meant to measure.

This module scores the reafference instead: the observed body change against
the commanded one, in direction, magnitude, and timing. Everything is a pure
function over explicit arguments, so a verdict is reproducible from the stored
record alone.

Claim policy: this relates one functional variable to another. Nothing in it is
a phenomenal claim.
"""

from __future__ import annotations

import json
import math
import secrets
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

# Direction dominates. Moving the wrong way is stronger evidence that something
# else caused the change than moving the right way by the wrong amount.
DIRECTION_WEIGHT = 0.5
MAGNITUDE_WEIGHT = 0.3
TIMING_WEIGHT = 0.2

# Ratio error tolerated in full before the magnitude term starts to fall.
MAGNITUDE_TOLERANCE = 0.35

# Movement below this is indistinguishable from encoder noise on the PTZ
# hardware, so it counts as no movement.
NOISE_FLOOR_DEGREES = 1.0

# Ceilings applied after scoring. "Something else moved me" must not keep a high
# score just because the timing happened to look right.
EXTERNALLY_CAUSED_CEILING = 0.10
INVERTED_CEILING = 0.10
UNRESPONSIVE_CEILING = 0.15

# Neither commanded nor observed: consistent, but no evidence either way.
NO_CHANGE_SCORE = 0.5


class ContingencyVerdict(str, Enum):
    """How an observed body change relates to the commanded one."""

    SELF_CAUSED = "self_caused"
    INVERTED = "inverted"
    UNRESPONSIVE = "unresponsive"
    EXTERNALLY_CAUSED = "externally_caused"
    NO_CHANGE = "no_change"
    UNVERIFIED = "unverified"


class BodyPose(BaseModel):
    """Absolute pose of a controllable body channel, in degrees."""

    model_config = ConfigDict(extra="forbid")

    pan: float = 0.0
    tilt: float = 0.0

    def delta_to(self, other: "BodyPose") -> dict[str, float]:
        return {"pan": other.pan - self.pan, "tilt": other.tilt - self.tilt}


class BodyObservation(BaseModel):
    """A measured or declared body state before and after an action."""

    model_config = ConfigDict(extra="forbid")

    channel: str = "camera_pose"
    before: BodyPose
    after: BodyPose
    observed_latency_ms: int | None = None
    # `measured` came from hardware; `declared` is the caller's assertion and is
    # kept separable so a later audit can discount it.
    source: Literal["measured", "declared"] = "measured"


class ReafferenceVerdict(BaseModel):
    """The scored comparison, carrying every term that produced the score."""

    model_config = ConfigDict(extra="forbid")

    verdict: ContingencyVerdict
    score: float = Field(ge=0.0, le=1.0)
    direction_score: float = Field(ge=0.0, le=1.0)
    magnitude_score: float = Field(ge=0.0, le=1.0)
    timing_score: float = Field(ge=0.0, le=1.0)
    magnitude_ratio: float | None = None
    commanded_delta: dict[str, float] = Field(default_factory=dict)
    observed_delta: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)


def evaluate_reafference(
    *,
    commanded_delta: dict[str, float] | None,
    observation: BodyObservation | None,
    expected_latency_ms: int | None = None,
) -> ReafferenceVerdict:
    """Score an observed body change against the one that was commanded.

    Returns an UNVERIFIED verdict when there is nothing to compare, so callers
    can fall back to whatever they did before rather than inventing evidence.
    """
    if observation is None or commanded_delta is None:
        return ReafferenceVerdict(
            verdict=ContingencyVerdict.UNVERIFIED,
            score=NO_CHANGE_SCORE,
            direction_score=0.0,
            magnitude_score=0.0,
            timing_score=0.0,
            commanded_delta=dict(commanded_delta or {}),
            observed_delta={},
            rationale=["no body observation for this action"],
        )

    observed_delta = observation.before.delta_to(observation.after)
    commanded = _as_vector(commanded_delta)
    observed = _as_vector(observed_delta)
    commanded_magnitude = _magnitude(commanded)
    observed_magnitude = _magnitude(observed)

    commanded_moved = commanded_magnitude >= NOISE_FLOOR_DEGREES
    observed_moved = observed_magnitude >= NOISE_FLOOR_DEGREES

    timing = _timing_score(observation.observed_latency_ms, expected_latency_ms)

    if not commanded_moved and not observed_moved:
        return ReafferenceVerdict(
            verdict=ContingencyVerdict.NO_CHANGE,
            score=NO_CHANGE_SCORE,
            direction_score=0.0,
            magnitude_score=0.0,
            timing_score=timing,
            commanded_delta=dict(commanded_delta),
            observed_delta=observed_delta,
            rationale=["nothing commanded and nothing moved"],
        )

    if not commanded_moved and observed_moved:
        return ReafferenceVerdict(
            verdict=ContingencyVerdict.EXTERNALLY_CAUSED,
            score=EXTERNALLY_CAUSED_CEILING,
            direction_score=0.0,
            magnitude_score=0.0,
            timing_score=timing,
            commanded_delta=dict(commanded_delta),
            observed_delta=observed_delta,
            magnitude_ratio=None,
            rationale=[
                f"body moved {observed_magnitude:.1f} deg with nothing commanded"
            ],
        )

    if commanded_moved and not observed_moved:
        return ReafferenceVerdict(
            verdict=ContingencyVerdict.UNRESPONSIVE,
            score=UNRESPONSIVE_CEILING,
            direction_score=0.0,
            magnitude_score=0.0,
            timing_score=timing,
            commanded_delta=dict(commanded_delta),
            observed_delta=observed_delta,
            magnitude_ratio=0.0,
            rationale=[
                f"commanded {commanded_magnitude:.1f} deg but the body did not move"
            ],
        )

    cosine = _cosine(commanded, observed)
    direction = (cosine + 1.0) / 2.0
    ratio = observed_magnitude / commanded_magnitude
    magnitude = _magnitude_score(ratio)
    score = (
        DIRECTION_WEIGHT * direction
        + MAGNITUDE_WEIGHT * magnitude
        + TIMING_WEIGHT * timing
    )

    if cosine < 0.0:
        return ReafferenceVerdict(
            verdict=ContingencyVerdict.INVERTED,
            score=min(score, INVERTED_CEILING),
            direction_score=direction,
            magnitude_score=magnitude,
            timing_score=timing,
            magnitude_ratio=ratio,
            commanded_delta=dict(commanded_delta),
            observed_delta=observed_delta,
            rationale=[f"observed movement opposes the command (cos={cosine:.2f})"],
        )

    return ReafferenceVerdict(
        verdict=ContingencyVerdict.SELF_CAUSED,
        score=_clamp01(score),
        direction_score=direction,
        magnitude_score=magnitude,
        timing_score=timing,
        magnitude_ratio=ratio,
        commanded_delta=dict(commanded_delta),
        observed_delta=observed_delta,
        rationale=[
            f"direction cos={cosine:.2f}",
            f"magnitude ratio={ratio:.2f}",
        ],
    )


def commanded_delta_from_tool(
    tool_name: str, tool_input: dict[str, object]
) -> dict[str, float] | None:
    """Read the intended pose change out of a camera tool call.

    Returns None for tools that do not move the body, which is what makes the
    verdict UNVERIFIED rather than a fabricated zero.
    """
    axis_sign = _CAMERA_TOOLS.get(tool_name)
    if axis_sign is None:
        return None
    axis, sign = axis_sign
    degrees = tool_input.get("degrees")
    if degrees is None:
        degrees = _DEFAULT_DEGREES[axis]
    try:
        amount = float(degrees)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    delta = {"pan": 0.0, "tilt": 0.0}
    delta[axis] = sign * amount
    return delta


# Tool name -> (axis, sign). Left and down are negative by the same convention
# the camera package uses for its own position tracking.
_CAMERA_TOOLS: dict[str, tuple[str, float]] = {
    "look_left": ("pan", -1.0),
    "look_right": ("pan", 1.0),
    "look_up": ("tilt", 1.0),
    "look_down": ("tilt", -1.0),
    "right_eye_look_left": ("pan", -1.0),
    "right_eye_look_right": ("pan", 1.0),
    "right_eye_look_up": ("tilt", 1.0),
    "right_eye_look_down": ("tilt", -1.0),
    "both_eyes_look_left": ("pan", -1.0),
    "both_eyes_look_right": ("pan", 1.0),
    "both_eyes_look_up": ("tilt", 1.0),
    "both_eyes_look_down": ("tilt", -1.0),
}

# Matches the camera tool defaults, so an omitted `degrees` is still checkable.
_DEFAULT_DEGREES = {"pan": 30.0, "tilt": 20.0}


def _as_vector(delta: dict[str, float]) -> tuple[float, float]:
    return (float(delta.get("pan", 0.0)), float(delta.get("tilt", 0.0)))


def _magnitude(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    denominator = _magnitude(a) * _magnitude(b)
    if denominator == 0.0:
        return 0.0
    return _clamp((a[0] * b[0] + a[1] * b[1]) / denominator, -1.0, 1.0)


def _magnitude_score(ratio: float) -> float:
    """1.0 while the ratio error stays inside tolerance, falling to 0 at 2x."""
    error = abs(ratio - 1.0)
    if error <= MAGNITUDE_TOLERANCE:
        return 1.0
    return _clamp01(1.0 - (error - MAGNITUDE_TOLERANCE) / (1.0 - MAGNITUDE_TOLERANCE))


def _timing_score(
    observed_latency_ms: int | None, expected_latency_ms: int | None
) -> float:
    if observed_latency_ms is None or expected_latency_ms is None:
        return 0.5
    scale = max(500.0, float(expected_latency_ms))
    return _clamp01(
        math.exp(-abs(observed_latency_ms - expected_latency_ms) / (2.0 * scale))
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


class BodyContingencyStore:
    """Append-only ledger of reafference verdicts, one per action at most."""

    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def record(
        self,
        *,
        verdict: ReafferenceVerdict,
        observation: BodyObservation | None,
        owner_id: str = "self",
        action_id: str | None = None,
        outcome_ref: str | None = None,
        field_id: str | None = None,
        tick_id: str | None = None,
        expected_latency_ms: int | None = None,
    ) -> str:
        """Store one verdict, returning the row id.

        An action already carrying a verdict keeps the one it has: outcomes are
        closed once, and a second write would mean the ledger disagreed with
        the ownership score that was actually used.
        """
        if action_id is not None:
            existing = self._db.fetchone(
                "SELECT contingency_id FROM body_contingencies WHERE action_id = ?",
                (action_id,),
            )
            if existing is not None:
                return str(existing["contingency_id"])

        contingency_id = f"bct_{secrets.token_urlsafe(12)}"
        before = observation.before.model_dump() if observation else {}
        after = observation.after.model_dump() if observation else {}
        with self._db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO body_contingencies(
                    contingency_id, owner_id, action_id, outcome_ref, field_id,
                    tick_id, channel, commanded_delta_json, observed_before_json,
                    observed_after_json, observed_delta_json, verdict,
                    reafference_score, direction_score, magnitude_score,
                    timing_score, magnitude_ratio, observed_latency_ms,
                    expected_latency_ms, observation_source, rationale_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contingency_id,
                    owner_id,
                    action_id,
                    outcome_ref,
                    field_id,
                    tick_id,
                    observation.channel if observation else "camera_pose",
                    _dumps(verdict.commanded_delta),
                    _dumps(before),
                    _dumps(after),
                    _dumps(verdict.observed_delta),
                    verdict.verdict.value,
                    verdict.score,
                    verdict.direction_score,
                    verdict.magnitude_score,
                    verdict.timing_score,
                    verdict.magnitude_ratio,
                    observation.observed_latency_ms if observation else None,
                    expected_latency_ms,
                    observation.source if observation else "measured",
                    _dumps(verdict.rationale),
                    utc_now(),
                ),
            )
        return contingency_id

    def for_action(self, action_id: str) -> dict[str, Any] | None:
        row = self._db.fetchone(
            "SELECT * FROM body_contingencies WHERE action_id = ?", (action_id,)
        )
        return None if row is None else dict(row)

    def recent(
        self,
        *,
        owner_id: str = "self",
        verdict: ContingencyVerdict | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if verdict is None:
            rows = self._db.fetchall(
                """
                SELECT * FROM body_contingencies
                WHERE owner_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (owner_id, limit),
            )
        else:
            rows = self._db.fetchall(
                """
                SELECT * FROM body_contingencies
                WHERE owner_id = ? AND verdict = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (owner_id, verdict.value, limit),
            )
        return [dict(row) for row in rows]


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
