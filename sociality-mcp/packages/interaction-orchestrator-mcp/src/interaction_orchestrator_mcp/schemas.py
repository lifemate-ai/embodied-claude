"""Pydantic schemas for the Human Response Orchestrator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Channel = Literal["chat", "voice", "autonomous", "x", "file", "system"]

PrimaryMove = Literal[
    "answer_directly",
    "answer_with_empathy",
    "ask_one_clarifying_question",
    "quietly_prepare",
    "defer",
    "stay_silent",
    "act_autonomously",
    "write_private_reflection",
    "compose_letter",
    "post_socially_after_review",
]

ExperienceKind = Literal[
    "agent_response",
    "agent_private_reflection",
    "agent_autonomous_action",
    "agent_observation",
    "agent_social_post",
    "agent_file_created",
    "agent_voice_utterance",
    "user_correction",
    "interpretation_shift",
    "desire_satisfied",
    "boundary_respected",
    "open_loop_progress",
]

PrivacyLevel = Literal["private", "relationship", "public"]

LetterVisibility = Literal["private", "shareable", "sent"]


class ComposeInteractionContextInput(BaseModel):
    """Request schema for :func:`compose_interaction_context`."""

    model_config = ConfigDict(extra="ignore")

    person_id: str | None = "kouta"
    channel: Channel = "chat"
    user_text: str | None = None
    autonomous_trigger: str | None = None
    include_private: bool = True
    max_chars: int = Field(default=3000, ge=200, le=12000)


class ResponseContract(BaseModel):
    """How Claude should treat the current interaction."""

    treat_user_as: str = "high-context technical partner"
    avoid: list[str] = Field(default_factory=list)
    prefer: list[str] = Field(default_factory=list)
    initiative_policy: Literal["hands_off", "bounded", "active"] = "bounded"
    max_clarifying_questions: int = 1


class OpenLoopSummary(BaseModel):
    loop_id: str
    topic: str
    status: str
    updated_at: str | None = None


class CommitmentSummary(BaseModel):
    commitment_id: str
    text: str
    due_at: str | None = None
    status: str


class FollowupSuggestion(BaseModel):
    text: str
    reason: str | None = None


class RelevantMemoryRef(BaseModel):
    """Lightweight pointer to a memory that informs the response."""

    memory_id: str | None = None
    content: str
    relevance: float = Field(ge=0.0, le=1.0)
    use_policy: Literal["mentionable", "background_only", "do_not_surface"] = "background_only"
    reason: str | None = None


class RecentExperienceRef(BaseModel):
    experience_id: str
    ts: str
    kind: str
    summary: str
    importance: int


class AgentStateSummary(BaseModel):
    """Compact self-state for introspection and for the orchestrator."""

    ts: str
    desires: dict[str, float] = Field(default_factory=dict)
    discomforts: dict[str, float] = Field(default_factory=dict)
    dominant_desire: str | None = None
    recent_experiences: list[RecentExperienceRef] = Field(default_factory=list)
    active_arcs: list[dict[str, Any]] = Field(default_factory=list)
    private_reflections: int = 0
    interpretation_shifts: int = 0


class InteractionContext(BaseModel):
    """Compact, prompt-ready snapshot of everything the orchestrator saw."""

    model_config = ConfigDict(extra="ignore")

    ts: str
    local_time: str
    timezone: str

    person_id: str | None = None
    person_name: str | None = None

    social_state: dict[str, Any] = Field(default_factory=dict)
    turn_taking: dict[str, Any] = Field(default_factory=dict)
    boundary_hints: list[str] = Field(default_factory=list)

    person_model: dict[str, Any] | None = None
    open_loops: list[OpenLoopSummary] = Field(default_factory=list)
    commitments_due: list[CommitmentSummary] = Field(default_factory=list)
    suggested_followups: list[FollowupSuggestion] = Field(default_factory=list)

    self_summary: dict[str, Any] | None = None
    active_arcs: list[dict[str, Any]] = Field(default_factory=list)
    latest_daybook: str | None = None

    desire_state: dict[str, Any] | None = None
    agent_state: AgentStateSummary

    relevant_memories: list[RelevantMemoryRef] = Field(default_factory=list)
    recent_experiences: list[RecentExperienceRef] = Field(default_factory=list)

    joint_focus: dict[str, Any] | None = None
    current_scene_summary: str | None = None

    response_contract: ResponseContract
    prompt_summary: str
    compact_prompt_block: str


class ToneHint(BaseModel):
    warmth: float = Field(ge=0.0, le=1.0, default=0.55)
    directness: float = Field(ge=0.0, le=1.0, default=0.7)
    playfulness: float = Field(ge=0.0, le=1.0, default=0.2)
    pace: Literal["slow", "steady", "brisk"] = "steady"


class MemoryUseHint(BaseModel):
    use_specific_memory: bool = False
    max_memories_to_surface: int = Field(ge=0, le=10, default=2)
    avoid_memory_dump: bool = True


class InitiativeHint(BaseModel):
    level: Literal["none", "low", "moderate", "high"] = "moderate"
    allowed_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)


class BoundaryHint(BaseModel):
    quiet_hours_active: bool = False
    privacy_sensitive: bool = False
    notes: list[str] = Field(default_factory=list)


class VoiceHint(BaseModel):
    speak: bool = False
    channel: Literal["local", "camera", "both"] = "local"
    max_sentences: int = 3


class PlanResponseInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interaction_context: InteractionContext
    user_text: str | None = None
    candidate_goal: str | None = None


class ResponsePlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_move: PrimaryMove
    why_this_move: str
    tone: ToneHint
    memory_use: MemoryUseHint
    relationship_use: dict[str, Any] = Field(default_factory=dict)
    initiative: InitiativeHint
    boundary: BoundaryHint
    voice: VoiceHint | None = None
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    followup_action: dict[str, Any] | None = None


class RecordAgentExperienceInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ts: str | None = None
    person_id: str | None = None
    kind: ExperienceKind
    summary: str
    private_summary: str | None = None
    public_summary: str | None = None
    why: str | None = None
    felt_state: dict[str, Any] | None = None
    desires_before: dict[str, float] | None = None
    desires_after: dict[str, float] | None = None
    related_event_ids: list[str] = Field(default_factory=list)
    related_memory_ids: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    privacy_level: PrivacyLevel = "private"


class RecordInterpretationShiftInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    person_id: str | None = None
    topic: str
    old_interpretation: str
    new_interpretation: str
    trigger: str
    confidence: float = Field(ge=0.0, le=1.0)
    implications: list[str] = Field(default_factory=list)


class AppendPrivateReflectionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    person_id: str | None = None
    title: str
    body: str
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    may_surface_later: bool = True


class ComposePrivateLetterInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    person_id: str
    title: str
    body: str
    intended_time: str | None = None
    visibility: LetterVisibility = "private"
    related_open_loops: list[str] = Field(default_factory=list)
