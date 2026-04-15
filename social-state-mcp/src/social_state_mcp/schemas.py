"""Schemas for social-state-mcp."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendedMove(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    style: str | None = None


class AffectGuess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class SocialStateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    person_id: str | None = None
    presence: Literal["absent", "possible", "present", "speaking"]
    activity: Literal[
        "working", "commuting", "eating", "resting", "sleeping", "chatting", "unknown"
    ]
    availability: Literal["interruptible", "maybe_interruptible", "do_not_interrupt"]
    interaction_phase: Literal[
        "opening", "ongoing", "awaiting_reply", "quiet_focus", "cooling_down", "closing", "idle"
    ]
    energy: Literal["high", "medium", "low", "unknown"]
    affect_guess: AffectGuess
    interrupt_cost: float = Field(ge=0.0, le=1.0)
    recommended_moves: list[RecommendedMove]
    summary_for_prompt: str
    evidence: list[str] = Field(default_factory=list)


class ShouldInterruptResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["yes", "no"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    cooldown_seconds: int = Field(ge=0)


class TurnTakingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["respond", "hold", "listen"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class SocialContextSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
