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


class SelfSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
