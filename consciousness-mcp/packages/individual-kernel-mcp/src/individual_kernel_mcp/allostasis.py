"""Allostatic body state: desire projection and valence composition.

Two problems in the previous body state are addressed here.

The desire file is written by an external updater on a timer. When that timer
stops, every level freezes at whatever was last written. `project_desires`
re-derives current levels from the recorded snapshot plus elapsed time, so the
writer only supplies corrections and the kernel owns the clock.

Valence was `-0.6 * max_discomfort`, whose upper bound is zero: no outcome could
produce a positive state. `compose_valence` builds it from an appetitive term
(expected improvement, weighted by how much control we actually measured) minus
an aversive term (unmet needs, uncertainty, unresolved prediction error).

Everything here is a pure function of its arguments, including `now`, so the
caller supplies the clock and tests stay deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))

# Weights for the two valence terms. Chosen so that a fully good context reaches
# +1 and a fully bad one reaches -1; see tests/test_valence_composition.py.
APPETITIVE_GAIN = 1.0
DISCOMFORT_WEIGHT = 0.55
UNCERTAINTY_WEIGHT = 0.25
UNRESOLVED_ERROR_WEIGHT = 0.20


@dataclass(frozen=True)
class DesireSpec:
    """How one need grows over time and where it is comfortable."""

    satisfaction_hours: float
    set_point: float


# Mirrors desire-system/desire_updater.py. That package has its own virtualenv
# and is not part of this workspace, so the constants are duplicated rather than
# imported; mcpBehavior.toml can override them without touching code.
DEFAULT_DESIRE_SPECS: dict[str, DesireSpec] = {
    "look_outside": DesireSpec(satisfaction_hours=1.0, set_point=0.3),
    "browse_curiosity": DesireSpec(satisfaction_hours=2.0, set_point=0.3),
    "miss_companion": DesireSpec(satisfaction_hours=3.0, set_point=0.3),
    "miss_kouta": DesireSpec(satisfaction_hours=3.0, set_point=0.3),
    "observe_room": DesireSpec(satisfaction_hours=0.167, set_point=0.2),
    "identity_coherence": DesireSpec(satisfaction_hours=1.0, set_point=0.9),
    "cognitive_load": DesireSpec(satisfaction_hours=1.5, set_point=0.3),
}

FALLBACK_SPEC = DesireSpec(satisfaction_hours=2.0, set_point=0.3)


@dataclass(frozen=True)
class DesireProjection:
    """Desire levels re-derived for a specific instant."""

    desires: dict[str, float]
    discomforts: dict[str, float]
    dominant: str | None
    stale_seconds: float

    @property
    def mean_discomfort(self) -> float:
        if not self.discomforts:
            return 0.0
        return sum(self.discomforts.values()) / len(self.discomforts)


@dataclass(frozen=True)
class AllostaticVariable:
    """The composed body state, with its terms kept separable for inspection."""

    appetitive: float
    aversive: float
    controllability: float
    uncertainty: float
    unresolved_error: float
    expected_valence_delta: float

    @property
    def valence(self) -> float:
        return _clamp(self.appetitive - self.aversive, -1.0, 1.0)

    def as_snapshot(self) -> dict[str, float]:
        return {
            "valence": self.valence,
            "appetitive": self.appetitive,
            "aversive": self.aversive,
            "controllability": self.controllability,
            "uncertainty": self.uncertainty,
            "unresolved_error": self.unresolved_error,
            "expected_valence_delta": self.expected_valence_delta,
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def extrapolate_desire_level(
    recorded_level: float,
    recorded_at: datetime,
    satisfaction_hours: float,
    now: datetime,
) -> float:
    """Advance a recorded level to ``now``.

    Levels grow linearly as ``elapsed / satisfaction_hours``, so the recorded
    level implies when the need was last met and the same ramp can be evaluated
    at any later instant. A level already at 1.0 carries no such implication and
    stays saturated. Time running backwards (clock skew, a snapshot written by a
    machine slightly ahead) never lowers the level below what was recorded.
    """

    level = _clamp01(recorded_level)
    if satisfaction_hours <= 0.0 or level >= 1.0:
        return 1.0
    elapsed_hours = (now - recorded_at).total_seconds() / 3600.0
    if elapsed_hours <= 0.0:
        return level
    return _clamp01(level + elapsed_hours / satisfaction_hours)


def allostatic_set_point(name: str, spec: DesireSpec, now: datetime) -> float:
    """Adjust a set point for the time of day.

    Being alone at 3am is not the same as being alone at noon, so the social and
    outward-looking needs settle overnight. Identity coherence does not: it is
    wanted equally at every hour.
    """

    if name == "identity_coherence":
        return spec.set_point
    hour = now.astimezone(JST).hour
    if 0 <= hour < 5:
        if name in ("miss_companion", "miss_kouta"):
            return max(0.0, spec.set_point - 0.15)
        if name in ("look_outside", "observe_room"):
            return max(0.0, spec.set_point - 0.1)
    if 5 <= hour < 7 and name in ("miss_companion", "miss_kouta"):
        return max(0.0, spec.set_point - 0.05)
    return spec.set_point


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def project_desires(
    snapshot: dict[str, Any],
    now: datetime,
    specs: dict[str, DesireSpec] | None = None,
) -> DesireProjection:
    """Re-derive desire levels and discomforts for ``now``.

    An unreadable or absent timestamp is treated as "just written": the recorded
    levels are used as-is rather than extrapolated from a guessed instant.
    """

    specs = specs or DEFAULT_DESIRE_SPECS
    recorded = snapshot.get("desires") or {}
    if not isinstance(recorded, dict) or not recorded:
        return DesireProjection({}, {}, None, 0.0)

    recorded_at = _parse_timestamp(snapshot.get("updated_at"))
    stale_seconds = (
        max(0.0, (now - recorded_at).total_seconds()) if recorded_at else 0.0
    )

    desires: dict[str, float] = {}
    discomforts: dict[str, float] = {}
    for name, raw_level in recorded.items():
        try:
            level = float(raw_level)
        except (TypeError, ValueError):
            continue
        spec = specs.get(str(name), FALLBACK_SPEC)
        projected = (
            extrapolate_desire_level(level, recorded_at, spec.satisfaction_hours, now)
            if recorded_at
            else _clamp01(level)
        )
        desires[str(name)] = projected
        discomforts[str(name)] = abs(
            projected - allostatic_set_point(str(name), spec, now)
        )

    dominant = (
        max(sorted(discomforts), key=lambda name: discomforts[name])
        if discomforts
        else None
    )
    return DesireProjection(desires, discomforts, dominant, stale_seconds)


def compose_valence(
    expected_valence_delta: float,
    mean_discomfort: float,
    controllability: float,
    uncertainty: float,
    unresolved_error: float,
) -> AllostaticVariable:
    """Combine an appetitive and an aversive term into a body state.

    Expected improvement only counts to the extent it is actually reachable, so
    it is scaled by measured controllability; an expected gain nobody can bring
    about is not good news. Expected worsening is aversive on its own terms and
    is not discounted that way.
    """

    delta = _clamp(expected_valence_delta, -1.0, 1.0)
    control = _clamp01(controllability)
    appetitive = APPETITIVE_GAIN * max(0.0, delta) * control
    aversive = (
        max(0.0, -delta)
        + DISCOMFORT_WEIGHT * _clamp01(mean_discomfort)
        + UNCERTAINTY_WEIGHT * _clamp01(uncertainty)
        + UNRESOLVED_ERROR_WEIGHT * _clamp01(unresolved_error)
    )
    return AllostaticVariable(
        appetitive=appetitive,
        aversive=aversive,
        controllability=control,
        uncertainty=_clamp01(uncertainty),
        unresolved_error=_clamp01(unresolved_error),
        expected_valence_delta=delta,
    )
