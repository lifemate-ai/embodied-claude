"""Schemas for boundary-mcp."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluateActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["allow", "deny", "allow_with_override"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    safer_alternatives: list[str] = Field(default_factory=list)
    # The person's ``preferred_nudge_style`` from socialPolicy.toml, when a
    # person rule exists. Callers shaping a nudge or a reply read it here.
    nudge_style: str | None = None


class ReviewSocialPostResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: Literal["low", "medium", "high"]
    issues: list[str]
    recommendation: Literal["post_ok", "rewrite", "deny"]


class QuietModeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    until: str | None = None
