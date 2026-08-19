"""Allostatic body state: desire projection and valence composition.

Two problems in the previous body state are addressed here.

The desire file is written by an external updater on a timer. When that timer
stops, every level freezes at whatever was last written. `project_desires`
re-derives current levels from the recorded snapshot plus elapsed time, so the
writer only supplies corrections and the kernel owns the clock. Owning the clock
is not the same as owning the input: levels ramp to 1.0 within hours, so a
snapshot the writer abandoned reports every need maxed out forever. Past
`MAX_SNAPSHOT_AGE_HOURS` the projection says so instead of extrapolating.

Valence was `-0.6 * max_discomfort`, whose upper bound is zero: no outcome could
produce a positive state. `compose_valence` builds it from an appetitive term
(expected improvement, weighted by how much control we actually measured) minus
an aversive term (unmet needs, uncertainty, unresolved prediction error).

Everything here is a pure function of its arguments, including `now`, so the
caller supplies the clock and tests stay deterministic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Quiet-hours clock. Mirrors desire-system/desire_updater.py: the same four
# environment variables, the same defaults, so both projections of the same
# need agree on when night is. Hardcoding JST and 0-5 meant a non-JST
# deployment ran its "night" in the local afternoon, silently.
DEFAULT_TIMEZONE = "Asia/Tokyo"
DEFAULT_NIGHT_START = 0
DEFAULT_NIGHT_END = 5
DEFAULT_DAWN_END = 7

_OFFSET_RE = re.compile(r"^([+-])(\d{1,2})(?::?(\d{2}))?$")


def _fixed_offset(spec: str) -> tzinfo | None:
    match = _OFFSET_RE.match(spec.strip())
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3) or 0)
    if hours > 23 or minutes > 59:
        return None
    return timezone(sign * timedelta(hours=hours, minutes=minutes), spec.strip())


def resolve_timezone(spec: str | None = None) -> tzinfo:
    """Resolve ``DESIRE_TIMEZONE`` (IANA name or ``+09:00``), defaulting to Asia/Tokyo.

    A name zoneinfo cannot resolve falls back to the fixed +09:00 this module
    always used, rather than failing the body state over a configuration string.
    """
    raw = (spec if spec is not None else os.getenv("DESIRE_TIMEZONE", "")).strip()
    if not raw:
        raw = DEFAULT_TIMEZONE
    fixed = _fixed_offset(raw)
    if fixed is not None:
        return fixed
    try:
        return ZoneInfo(raw)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone(timedelta(hours=9), "JST")


def _env_hour(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw) % 24
    except ValueError:
        return default


def quiet_bands() -> tuple[int, int, int]:
    """``(night_start, night_end, dawn_end)`` hours from the environment."""
    return (
        _env_hour("DESIRE_NIGHT_START", DEFAULT_NIGHT_START),
        _env_hour("DESIRE_NIGHT_END", DEFAULT_NIGHT_END),
        _env_hour("DESIRE_DAWN_END", DEFAULT_DAWN_END),
    )


def hour_in_band(hour: int, start: int, end: int) -> bool:
    """``start <= hour < end`` on a 24-hour clock, wrapping past midnight if ``end < start``."""
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end

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


# Mirrors desire-system/desire_updater.py. That package is a separate
# distribution, so the constants are duplicated rather than imported;
# mcpBehavior.toml can override them without touching code.
DEFAULT_DESIRE_SPECS: dict[str, DesireSpec] = {
    "look_outside": DesireSpec(satisfaction_hours=1.0, set_point=0.3),
    "browse_curiosity": DesireSpec(satisfaction_hours=2.0, set_point=0.3),
    "miss_companion": DesireSpec(satisfaction_hours=3.0, set_point=0.3),
    "observe_room": DesireSpec(satisfaction_hours=0.167, set_point=0.2),
    "identity_coherence": DesireSpec(satisfaction_hours=1.0, set_point=0.9),
    "cognitive_load": DesireSpec(satisfaction_hours=1.5, set_point=0.3),
}

FALLBACK_SPEC = DesireSpec(satisfaction_hours=2.0, set_point=0.3)

# Beyond this, a snapshot is an artifact rather than a record. The longest
# satisfaction window is three hours, so everything saturates well inside a day
# and a further extrapolation adds no information -- it only makes an absent
# writer look identical to a genuinely neglected agent. A day is far longer than
# any gap a running writer produces and far shorter than the 71 days the live
# snapshot actually sat unwritten.
MAX_SNAPSHOT_AGE_HOURS = 24.0


@dataclass(frozen=True)
class DesireProjection:
    """Desire levels re-derived for a specific instant."""

    desires: dict[str, float]
    discomforts: dict[str, float]
    dominant: str | None
    stale_seconds: float
    usable: bool = True

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
    wanted equally at every hour. "Night" is read from DESIRE_TIMEZONE /
    DESIRE_NIGHT_START / DESIRE_NIGHT_END / DESIRE_DAWN_END (default JST, 0-5,
    5-7), the same variables desire-system uses.
    """

    if name == "identity_coherence":
        return spec.set_point
    hour = now.astimezone(resolve_timezone()).hour
    night_start, night_end, dawn_end = quiet_bands()
    if hour_in_band(hour, night_start, night_end):
        if name == "miss_companion":
            return max(0.0, spec.set_point - 0.15)
        if name in ("look_outside", "observe_room"):
            return max(0.0, spec.set_point - 0.1)
    if hour_in_band(hour, night_end, dawn_end) and name == "miss_companion":
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

    Past ``MAX_SNAPSHOT_AGE_HOURS`` the snapshot no longer describes anyone, and
    extrapolating it would report every need maxed out however long the writer
    has been gone. Such a projection is marked unusable and reports the resting
    state: levels at their set points, no discomfort, no dominant need. The age
    is still returned, because refusing to extrapolate is not the same as
    claiming the snapshot is current.
    """

    specs = specs or DEFAULT_DESIRE_SPECS
    recorded = snapshot.get("desires") or {}
    if not isinstance(recorded, dict) or not recorded:
        return DesireProjection({}, {}, None, 0.0)

    recorded_at = _parse_timestamp(snapshot.get("updated_at"))
    stale_seconds = (
        max(0.0, (now - recorded_at).total_seconds()) if recorded_at else 0.0
    )
    usable = stale_seconds <= MAX_SNAPSHOT_AGE_HOURS * 3600.0

    desires: dict[str, float] = {}
    discomforts: dict[str, float] = {}
    for name, raw_level in recorded.items():
        try:
            level = float(raw_level)
        except (TypeError, ValueError):
            continue
        spec = specs.get(str(name), FALLBACK_SPEC)
        set_point = allostatic_set_point(str(name), spec, now)
        if not usable:
            desires[str(name)] = set_point
            discomforts[str(name)] = 0.0
            continue
        projected = (
            extrapolate_desire_level(level, recorded_at, spec.satisfaction_hours, now)
            if recorded_at
            else _clamp01(level)
        )
        desires[str(name)] = projected
        discomforts[str(name)] = abs(projected - set_point)

    dominant = (
        max(sorted(discomforts), key=lambda name: discomforts[name])
        if discomforts and usable
        else None
    )
    return DesireProjection(desires, discomforts, dominant, stale_seconds, usable)


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
