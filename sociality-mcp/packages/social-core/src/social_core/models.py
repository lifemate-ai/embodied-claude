"""Shared Pydantic models for the sociality stack."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .time import ensure_iso8601

EVENT_KINDS = (
    "scene_parse",
    "audio_transcript",
    "human_utterance",
    "agent_utterance",
    "touchpoint",
    "health_summary",
    "tweet_posted",
    "mention_received",
    "commitment_created",
    "commitment_completed",
    "boundary_updated",
    "ritual_done",
)

EventKind = Literal[
    "scene_parse",
    "audio_transcript",
    "human_utterance",
    "agent_utterance",
    "touchpoint",
    "health_summary",
    "tweet_posted",
    "mention_received",
    "commitment_created",
    "commitment_completed",
    "boundary_updated",
    "ritual_done",
]


class SocialEventCreate(BaseModel):
    """Shared append-only event payload."""

    model_config = ConfigDict(extra="forbid")

    ts: str
    source: str = Field(min_length=1)
    kind: EventKind
    person_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    payload_json: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("payload_json", "payload"),
    )

    @field_validator("ts")
    @classmethod
    def _normalize_ts(cls, value: str) -> str:
        return ensure_iso8601(value)


class SocialEvent(SocialEventCreate):
    """Stored event with deterministic identifier and sequence."""

    event_id: str
    event_seq: int | None = None


class RankedDecision(BaseModel):
    """Small helper model for ranked, confidence-bearing outputs."""

    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
