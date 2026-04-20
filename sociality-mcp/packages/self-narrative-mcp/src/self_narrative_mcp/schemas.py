"""Schemas for self-narrative-mcp."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ArcRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    status: str
    importance: float = Field(ge=0.0, le=1.0)
    summary: str


class DaybookRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: str
    summary: str
    concrete_events: list[str] = Field(default_factory=list)
    noticed_changes: list[str] = Field(default_factory=list)
    relationship_moments: list[str] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    boundaries_respected: list[str] = Field(default_factory=list)
    private_reflections: list[str] = Field(default_factory=list)
    next_gentle_actions: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)


class SelfSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    recent_concrete_events: list[str] = Field(default_factory=list)
    recent_interpretation_shifts: list[str] = Field(default_factory=list)
    latest_daybook: str | None = None
