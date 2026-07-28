"""Representing the processing, not only what was processed.

The runtime already records higher-order representations: at the end of a tick
it asserts things like "I am attending X". Until now those records went into a
table and stopped there. Nothing downstream read them, so a HOR could be absent,
present, or wrong and every other quantity in the system stayed identical.

This module closes that loop in one narrow place: a HOR about a channel raises
the precision assigned to that channel on the following tick. It also records a
ProcessMetaRepresentation -- how the field was produced (what competed, by how
much, whether it ignited) as distinct from what the field is about.

Claim policy: this couples one recorded functional variable to another. Nothing
in it is a phenomenal claim.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

# How much a maximally confident HOR can raise its channel's precision.
HOR_PRECISION_GAIN = 0.15

# Total ceiling across however many HORs point at the same channel. Without
# this, a burst of self-directed records could drive precision to 1.0 on
# nothing but its own say-so.
HOR_PRECISION_CAP = 0.20

# Which precision channel each asserted mode speaks to. `attending`,
# `intending` and `wanting` are all about the runtime's own processing rather
# than about a sense, so they land on self_model.
MODE_TO_CHANNEL: dict[str, str] = {
    "seeing": "extero",
    "feeling": "intero",
    "remembering": "mnemonic",
    "attending": "self_model",
    "intending": "self_model",
    "wanting": "self_model",
}

CHANNELS = ("extero", "intero", "mnemonic", "social", "self_model")


class ProcessMetaRepresentation(BaseModel):
    """What the production of one field looked like from the inside.

    Every field already says what it is about. This says how it came to be:
    how close the competition was, whether it ignited, how many candidates
    were in play. It is the object a report about one's own processing would
    have to be a report *of*.
    """

    model_config = ConfigDict(extra="forbid")

    process_meta_id: str
    owner_id: str = "self"
    tick_id: str
    field_id: str
    producer: str = "deterministic"
    trigger_kind: str
    candidate_count: int = Field(ge=0)
    winner_kind: str = ""
    competition_margin: float = Field(ge=0.0)
    competition_entropy: float = Field(ge=0.0, le=1.0)
    ignited: bool
    conflicted: bool
    attention_intensity: float = Field(ge=0.0, le=1.0)
    registered_intention: bool = False
    hor_channel_bias: dict[str, float] = Field(default_factory=dict)
    created_at: str = ""

    def canonical_statement(self) -> str:
        """A first-person sentence that reports the processing, not the content.

        Kept here rather than in the surface layer so the assertion and the
        numbers it summarises cannot drift apart.
        """
        settled = "settled" if self.ignited else "did not settle"
        contested = "contested" if self.conflicted else "uncontested"
        return (
            f"{self.candidate_count} candidates competed and it {settled}; "
            f"the choice was {contested} by a margin of "
            f"{self.competition_margin:.3f}"
        )


class PrecisionBias(BaseModel):
    """Per-channel precision adjustment derived from recent HORs."""

    model_config = ConfigDict(extra="forbid")

    extero: float = 0.0
    intero: float = 0.0
    mnemonic: float = 0.0
    social: float = 0.0
    self_model: float = 0.0

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, channel) == 0.0 for channel in CHANNELS)

    def as_dict(self) -> dict[str, float]:
        return {channel: getattr(self, channel) for channel in CHANNELS}


def precision_bias_from_hors(records: list[Any]) -> PrecisionBias:
    """Turn recent higher-order records into a per-channel precision bump.

    Each record contributes `gain * confidence` to the channel its asserted
    mode speaks to, and every channel is capped, so repeating an assertion
    cannot substitute for evidence.
    """
    totals = dict.fromkeys(CHANNELS, 0.0)
    for record in records:
        mode = getattr(record, "asserted_mode", None)
        mode_value = getattr(mode, "value", mode)
        channel = MODE_TO_CHANNEL.get(str(mode_value))
        if channel is None:
            continue
        confidence = float(getattr(record, "confidence", 0.0) or 0.0)
        totals[channel] += HOR_PRECISION_GAIN * max(0.0, min(1.0, confidence))
    return PrecisionBias(
        **{channel: min(HOR_PRECISION_CAP, value) for channel, value in totals.items()}
    )


def apply_precision_bias(precision: Any, bias: PrecisionBias) -> Any:
    """Add the bias to a PrecisionState, clamped to the model's own bounds.

    Takes and returns the caller's type so this module does not have to import
    the field schema.
    """
    if bias.is_empty:
        return precision
    updated = {
        channel: max(0.0, min(1.0, getattr(precision, channel) + getattr(bias, channel)))
        for channel in CHANNELS
    }
    return precision.model_copy(update=updated)


class ProcessMetaStore:
    """Append-only ledger of process-level self representations."""

    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def record(
        self,
        *,
        tick_id: str,
        field_id: str,
        trigger_kind: str,
        competition: dict[str, Any],
        winner_kind: str = "",
        registered_intention: bool = False,
        hor_channel_bias: dict[str, float] | None = None,
        owner_id: str = "self",
    ) -> ProcessMetaRepresentation:
        scores = competition.get("scores") or {}
        record = ProcessMetaRepresentation(
            process_meta_id=f"pmr_{secrets.token_urlsafe(12)}",
            owner_id=owner_id,
            tick_id=tick_id,
            field_id=field_id,
            trigger_kind=trigger_kind,
            candidate_count=len(scores),
            winner_kind=winner_kind,
            competition_margin=max(0.0, float(competition.get("margin", 0.0) or 0.0)),
            competition_entropy=_clamp01(competition.get("entropy", 0.0)),
            ignited=bool(competition.get("ignited", False)),
            conflicted=bool(competition.get("conflicted", False)),
            attention_intensity=_clamp01(competition.get("attention_intensity", 0.0)),
            registered_intention=registered_intention,
            hor_channel_bias=hor_channel_bias or {},
            created_at=utc_now(),
        )
        with self._db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO process_meta_representations(
                    process_meta_id, owner_id, tick_id, field_id, producer,
                    trigger_kind, candidate_count, winner_kind,
                    competition_margin, competition_entropy, ignited, conflicted,
                    attention_intensity, registered_intention,
                    hor_channel_bias_json, canonical_statement, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.process_meta_id,
                    record.owner_id,
                    record.tick_id,
                    record.field_id,
                    record.producer,
                    record.trigger_kind,
                    record.candidate_count,
                    record.winner_kind,
                    record.competition_margin,
                    record.competition_entropy,
                    1 if record.ignited else 0,
                    1 if record.conflicted else 0,
                    record.attention_intensity,
                    1 if record.registered_intention else 0,
                    json.dumps(record.hor_channel_bias, sort_keys=True),
                    record.canonical_statement(),
                    record.created_at,
                ),
            )
        return record

    def for_tick(self, tick_id: str) -> dict[str, Any] | None:
        row = self._db.fetchone(
            "SELECT * FROM process_meta_representations WHERE tick_id = ?",
            (tick_id,),
        )
        return None if row is None else dict(row)

    def recent(
        self, *, owner_id: str = "self", limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = self._db.fetchall(
            """
            SELECT * FROM process_meta_representations
            WHERE owner_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (owner_id, limit),
        )
        return [dict(row) for row in rows]


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))
