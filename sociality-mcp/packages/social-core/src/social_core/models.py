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


EVIDENCE_TYPES = (
    "observed",
    "inferred",
    "remembered",
    "heard",
    "assumed",
)

EvidenceType = Literal[
    "observed",
    "inferred",
    "remembered",
    "heard",
    "assumed",
]


class EpistemicClaim(BaseModel):
    """A claim tagged with how the agent came to know it.

    Primitive type for the Agent Grammar / Epistemic Effect System.

    - observed:   directly perceived via a tool or sensor (camera, mic, etc.)
    - inferred:   derived from one or more upstream observations
    - remembered: recalled from the long-term memory store
    - heard:      reported by another agent or person
    - assumed:    prior / default, treated as low-confidence
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    evidence_type: EvidenceType
    source: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    derived_from: list[str] = Field(default_factory=list)
