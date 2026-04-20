"""Schemas for relationship-mcp."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CommitmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    due_at: str | None = None
    source: str


class OpenLoopRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    topic: str
    status: str


class RitualRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    cadence: str


class SuggestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    reason: str


class PreferenceRecord(BaseModel):
    """A preference with evidence, confidence, and provenance."""

    model_config = ConfigDict(extra="forbid")

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    source: Literal["explicit", "inferred", "seeded"] = "inferred"


class PersonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str
    canonical_name: str
    aliases: list[str]
    role: str | None = None
    salient_preferences: list[PreferenceRecord]
    open_loops: list[OpenLoopRecord]
    active_commitments: list[CommitmentRecord]
    rituals: list[RitualRecord]
    boundaries: list[str]
    relationship_summary: str
    last_updated: str
