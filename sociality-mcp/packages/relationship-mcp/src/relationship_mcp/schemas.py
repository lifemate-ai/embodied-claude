"""Schemas for relationship-mcp."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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


class PersonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str
    canonical_name: str
    aliases: list[str]
    role: str | None = None
    salient_preferences: list[str]
    open_loops: list[OpenLoopRecord]
    active_commitments: list[CommitmentRecord]
    rituals: list[RitualRecord]
    boundaries: list[str]
    relationship_summary: str
    last_updated: str
