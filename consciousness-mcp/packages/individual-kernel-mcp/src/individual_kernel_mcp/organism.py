"""Internal time: what changes when nothing is said to it.

Every tick so far was caused from outside -- a prompt, a tool result, a hook
firing on someone else's action. `TriggerKind.AUTONOMOUS` existed in the enum
and nothing produced it, which is to say the runtime had no clock of its own:
with no input, nothing in it moved.

This module is that clock, and it is deliberately not a model. Every quantity
is read from recorded state and combined arithmetically, so a step is
reproducible from the database and the instant it ran at. Three things drift
and one fires:

    decay        transition counts lose weight with elapsed time, so a
                 contingency that stopped holding stops being believed
    expiry       an imagined trajectory nobody took up stops being pending
    projection   needs grow along their own ramps (see `allostasis`)
    ignition     when unmet need and elapsed silence are both high enough,
                 a tick opens with TriggerKind.AUTONOMOUS

Claim policy: this is a scheduler over recorded state. Nothing in it is a
phenomenal claim.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

from individual_kernel_mcp._behavior import get_behavior
from individual_kernel_mcp.allostasis import project_desires
from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.trajectory import TrajectoryStatus, TrajectoryStore

# A week: a contingency observed once and never again is worth about half as
# much a week later. Slow enough that ordinary gaps between sessions do not
# erase what was learned.
STATS_HALF_LIFE_HOURS = 168.0

# Counts stop decaying here rather than being deleted. A row at the floor still
# records that the transition was seen; it just stops dominating the estimate.
MIN_OBSERVATION_COUNT = 0.05

# How long an imagined trajectory stays pending before it is no longer a
# prediction about anything.
PROTENTION_TTL_SECONDS = 1800.0

IGNITION_NEED_WEIGHT = 0.6
IGNITION_SILENCE_WEIGHT = 0.4
IGNITION_SILENCE_SATURATION_SECONDS = 7200.0
IGNITION_THRESHOLD = 0.55
MIN_IGNITION_INTERVAL_SECONDS = 900.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_iso(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def decay_factor(elapsed_seconds: float, half_life_hours: float) -> float:
    """Weight retained after `elapsed_seconds` at the given half-life.

    Compounding this over successive steps gives the same result as applying it
    once over the total elapsed time, so the daemon may run at any cadence --
    or twice in a row -- without changing what a count ends up at.
    """

    if elapsed_seconds <= 0.0 or half_life_hours <= 0.0:
        return 1.0
    return 0.5 ** ((elapsed_seconds / 3600.0) / half_life_hours)


def ignition_score(
    *, max_discomfort: float, seconds_since_last_field: float
) -> float:
    """Combine unmet need with elapsed silence.

    Both terms are load-bearing. Need alone would fire continuously during a
    session, when the runtime is already awake and the need is being addressed.
    Silence alone would fire on a schedule, which is a cron job in a costume.
    """

    need = _clamp01(max_discomfort)
    silence = _clamp01(seconds_since_last_field / IGNITION_SILENCE_SATURATION_SECONDS)
    return _clamp01(IGNITION_NEED_WEIGHT * need + IGNITION_SILENCE_WEIGHT * silence)


def should_ignite(
    score: float,
    *,
    seconds_since_last_field: float,
    threshold: float = IGNITION_THRESHOLD,
    min_interval: float = MIN_IGNITION_INTERVAL_SECONDS,
) -> tuple[bool, str]:
    """Decide, and say why not when the answer is no.

    The interval is a hard floor rather than another weighted term: a tick that
    just happened means the runtime is already awake, and no amount of need
    justifies opening a second one on top of it.
    """

    if seconds_since_last_field < min_interval:
        return False, (
            f"a field was committed {seconds_since_last_field:.0f}s ago; "
            f"the floor is {min_interval:.0f}s"
        )
    if score < threshold:
        return False, f"score {score:.3f} is below the threshold {threshold:.3f}"
    return True, f"score {score:.3f} reached the threshold {threshold:.3f}"


def expiry_deadline(
    *, created_at: Any, expires_at: Any, ttl_seconds: float
) -> datetime | None:
    """When a pending trajectory stops being a prediction.

    `expires_at` is honoured when a producer set one; otherwise the deadline is
    derived from creation, so trajectories written before anything populated
    that column still expire.
    """

    explicit = parse_iso(expires_at)
    if explicit is not None:
        return explicit
    created = parse_iso(created_at)
    if created is None:
        return None
    return created + timedelta(seconds=ttl_seconds)


class OrganismStep(BaseModel):
    """What one turn of the clock did."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: f"org_{secrets.token_urlsafe(10)}")
    owner_id: str = "self"
    ran_at: str = Field(default_factory=utc_now)
    elapsed_seconds: float = 0.0
    stats_decayed: int = 0
    decay_factor: float = 1.0
    protentions_expired: int = 0
    max_discomfort: float = 0.0
    dominant_desire: str | None = None
    seconds_since_last_field: float = 0.0
    ignition_score: float = 0.0
    ignited: bool = False
    tick_id: str | None = None
    reason: str = ""


class OrganismRunStore:
    """Append-only ledger of daemon steps; also the daemon's own clock."""

    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def record(self, step: OrganismStep) -> OrganismStep:
        self._db.execute(
            """
            INSERT INTO organism_runs(
                run_id, owner_id, ran_at, elapsed_seconds, stats_decayed,
                decay_factor, protentions_expired, max_discomfort,
                dominant_desire, seconds_since_last_field, ignition_score,
                ignited, tick_id, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.run_id,
                step.owner_id,
                step.ran_at,
                step.elapsed_seconds,
                step.stats_decayed,
                step.decay_factor,
                step.protentions_expired,
                step.max_discomfort,
                step.dominant_desire,
                step.seconds_since_last_field,
                step.ignition_score,
                int(step.ignited),
                step.tick_id,
                step.reason,
                step.ran_at,
            ),
        )
        return step

    def last_run_at(self, owner_id: str = "self") -> datetime | None:
        row = self._db.fetchone(
            "SELECT ran_at FROM organism_runs WHERE owner_id = ? "
            "ORDER BY ran_at DESC LIMIT 1",
            (owner_id,),
        )
        return None if row is None else parse_iso(row["ran_at"])

    def recent(self, owner_id: str = "self", limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.fetchall(
            "SELECT * FROM organism_runs WHERE owner_id = ? "
            "ORDER BY ran_at DESC LIMIT ?",
            (owner_id, max(1, min(int(limit), 200))),
        )
        return [dict(row) for row in rows]


def organism_enabled() -> bool:
    """Whether a scheduler may run the clock. Re-read on every call."""
    return bool(get_behavior("individual-kernel", "organism_daemon", False))


class OrganismDaemon:
    """One turn of the clock: drift what drifts, then decide whether to wake.

    There is no long-lived process here, so a step derives its elapsed time
    from the previous recorded run rather than from how long it slept. A
    scheduler may call it at any cadence, or twice in a row, and the state it
    arrives at is the same.

    The behaviour flag is checked by callers, not here: this is the mechanism,
    and a test that wants to exercise it should not have to edit configuration.
    """

    def __init__(
        self,
        db: SocialDB,
        producer: TickProducer,
        *,
        owner_id: str = "self",
        half_life_hours: float = STATS_HALF_LIFE_HOURS,
        protention_ttl_seconds: float = PROTENTION_TTL_SECONDS,
        ignition_threshold: float = IGNITION_THRESHOLD,
        min_ignition_interval: float = MIN_IGNITION_INTERVAL_SECONDS,
    ) -> None:
        self._db = db
        self._producer = producer
        self._owner_id = owner_id
        self._half_life_hours = half_life_hours
        self._ttl_seconds = protention_ttl_seconds
        self._threshold = ignition_threshold
        self._min_interval = min_ignition_interval
        self.runs = OrganismRunStore(db)
        self.trajectories = TrajectoryStore(db)

    def step(
        self,
        *,
        now: datetime | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> OrganismStep:
        """Advance internal time once and record what it did.

        `force` overrides the ignition decision only; the drift still follows
        elapsed time. `dry_run` computes everything and writes nothing, which
        is how a scheduler can ask what would happen.
        """

        moment = now or datetime.now(timezone.utc)
        ran_at = moment.isoformat().replace("+00:00", "Z")
        previous = self.runs.last_run_at(self._owner_id)
        elapsed = max(0.0, (moment - previous).total_seconds()) if previous else 0.0

        factor = decay_factor(elapsed, self._half_life_hours)
        decayed = 0 if dry_run else self._decay_stats(factor, ran_at)
        expired = 0 if dry_run else self._expire_protentions(moment)

        projection = project_desires(self._read_desires(), now=moment)
        max_discomfort = max(projection.discomforts.values(), default=0.0)
        silence = self._seconds_since_last_field(moment)
        score = ignition_score(
            max_discomfort=max_discomfort, seconds_since_last_field=silence
        )
        ignite, reason = should_ignite(
            score,
            seconds_since_last_field=silence,
            threshold=self._threshold,
            min_interval=self._min_interval,
        )
        if force:
            ignite, reason = True, "ignition was forced"

        tick_id: str | None = None
        if ignite and not dry_run:
            try:
                tick_id = self._open_autonomous_tick()
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                ignite = False
                reason = f"ignition failed: {error}"

        step = OrganismStep(
            owner_id=self._owner_id,
            ran_at=ran_at,
            elapsed_seconds=elapsed,
            stats_decayed=decayed,
            decay_factor=factor,
            protentions_expired=expired,
            max_discomfort=_clamp01(max_discomfort),
            dominant_desire=projection.dominant,
            seconds_since_last_field=silence,
            ignition_score=score,
            ignited=ignite,
            tick_id=tick_id,
            reason=reason,
        )
        if not dry_run:
            self.runs.record(step)
        return step

    def _decay_stats(self, factor: float, ran_at: str) -> int:
        """Shrink observation counts, stopping at the floor rather than deleting."""
        if factor >= 1.0:
            return 0
        with self._db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE generative_transition_stats "
                "SET observation_count = observation_count * ?, updated_at = ? "
                "WHERE owner_id = ? AND observation_count > ?",
                (factor, ran_at, self._owner_id, MIN_OBSERVATION_COUNT),
            )
            return int(cursor.rowcount)

    def _expire_protentions(self, moment: datetime) -> int:
        """Retire imaginings nobody took up.

        Only `imagined` trajectories expire. One that reached `intended` was
        claimed by an intention and has to be resolved by its outcome, not by
        the clock.
        """

        rows = self._db.fetchall(
            "SELECT trajectory_id, created_at, expires_at FROM imagined_trajectories "
            "WHERE owner_id = ? AND status = 'imagined' "
            "ORDER BY created_at LIMIT 500",
            (self._owner_id,),
        )
        expired = 0
        for row in rows:
            deadline = expiry_deadline(
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                ttl_seconds=self._ttl_seconds,
            )
            if deadline is None or deadline > moment:
                continue
            self.trajectories.transition(
                row["trajectory_id"],
                TrajectoryStatus.EXPIRED,
                reason="the moment this predicted has passed",
            )
            expired += 1
        return expired

    def _seconds_since_last_field(self, moment: datetime) -> float:
        row = self._db.fetchone(
            "SELECT created_at FROM enacted_fields WHERE owner_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (self._owner_id,),
        )
        last = parse_iso(row["created_at"]) if row is not None else None
        if last is None:
            # Nothing has ever happened. That is the most silent a runtime gets,
            # so a first step is allowed to be the one that starts it.
            return IGNITION_SILENCE_SATURATION_SECONDS
        return max(0.0, (moment - last).total_seconds())

    def _open_autonomous_tick(self) -> str:
        opened = self._producer.begin_tick(TriggerKind.AUTONOMOUS)
        self._producer.compete_and_commit(opened.tick_id)
        return opened.tick_id

    def _read_desires(self) -> dict[str, Any]:
        path = Path(self._producer.desires_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}
