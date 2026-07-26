"""Structured intentions, outcomes, and deterministic agency assessment."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import shlex
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now


class IntentionStatus(StrEnum):
    PENDING = "PENDING"
    ALLOWED = "ALLOWED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    DENIED = "DENIED"


class ExpectedEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    expected_refs: list[str] = Field(default_factory=list)


class PredictedEffects(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exteroceptive: ExpectedEffect = Field(default_factory=ExpectedEffect)
    interoceptive: ExpectedEffect = Field(default_factory=ExpectedEffect)
    social: ExpectedEffect = Field(default_factory=ExpectedEffect)
    mnemonic: ExpectedEffect = Field(default_factory=ExpectedEffect)


class ActionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_id: str
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    predicted_effects: PredictedEffects = Field(default_factory=PredictedEffects)
    goal: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    expected_latency_ms: int | None = Field(default=None, ge=0)


class IntentionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    owner_id: str
    field_id: str
    tick_id: str
    tool_name: str
    tool_input_hash: str
    normalized_tool_input: dict[str, Any]
    intended_goal: str
    predicted_effects: PredictedEffects
    expected_latency_ms: int | None = None
    confidence: float
    status: IntentionStatus
    created_at: str
    closed_at: str | None = None


class ActionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome_id: str
    action_id: str
    field_id: str
    tick_id: str
    actual_result_ref: str | None = None
    actual_result_summary: str = ""
    actual_result_hash: str | None = None
    success: bool
    latency_ms: int | None = None
    mismatch_vector: dict[str, float] = Field(default_factory=dict)
    ownership_score: float = Field(ge=0.0, le=1.0)
    created_at: str


class AgencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registered_intention: float = Field(ge=0.0, le=1.0)
    action_effect_match: float = Field(ge=0.0, le=1.0)
    temporal_contiguity: float = Field(ge=0.0, le=1.0)
    exclusive_causal_fit: float = Field(ge=0.0, le=1.0)
    ownership_score: float = Field(ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)


class AgencyStore:
    """Persistence and matching rules for the closed action loop."""

    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def propose(self, proposal: ActionProposal, *, owner_id: str = "self") -> IntentionRecord:
        field = self._db.fetchone(
            "SELECT tick_id, status FROM enacted_fields WHERE field_id = ? AND owner_id = ?",
            (proposal.field_id, owner_id),
        )
        if field is None:
            raise ValueError(f"unknown field {proposal.field_id!r}")
        if field["status"] != "COMMITTED":
            raise ValueError("actions require a COMMITTED field")
        existing = self.get_pending(owner_id)
        if existing is not None:
            raise ValueError(
                f"owner {owner_id!r} already has pending intention {existing.action_id!r}"
            )

        normalized = normalize_tool_input(proposal.tool_input)
        action_id = f"act_{secrets.token_urlsafe(12)}"
        now = utc_now()
        with self._db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO field_intentions(
                    action_id, owner_id, field_id, tick_id, tool_name,
                    tool_input_hash, normalized_tool_input_json, intended_goal,
                    predicted_effects_json, expected_latency_ms, confidence,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (
                    action_id,
                    owner_id,
                    proposal.field_id,
                    field["tick_id"],
                    proposal.tool_name,
                    hash_tool_input(proposal.tool_input),
                    json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                    proposal.goal,
                    json.dumps(
                        proposal.predicted_effects.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    proposal.expected_latency_ms,
                    proposal.confidence,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE enacted_fields
                SET pending_intention_ref = ?,
                    agency_state_json = ?,
                    updated_at = ?
                WHERE field_id = ?
                """,
                (
                    action_id,
                    json.dumps(
                        {
                            "pending_intention_ref": action_id,
                            "ownership_score": 0.0,
                            "last_outcome_ref": None,
                            "registered_intention": True,
                        },
                        sort_keys=True,
                    ),
                    now,
                    proposal.field_id,
                ),
            )
        record = self.get(action_id)
        assert record is not None
        return record

    def get(self, action_id: str) -> IntentionRecord | None:
        row = self._db.fetchone(
            "SELECT * FROM field_intentions WHERE action_id = ?", (action_id,)
        )
        return None if row is None else self._row_to_intention(row)

    def get_pending(self, owner_id: str = "self") -> IntentionRecord | None:
        row = self._db.fetchone(
            """
            SELECT * FROM field_intentions
            WHERE owner_id = ? AND status IN ('PENDING', 'ALLOWED')
            ORDER BY created_at DESC LIMIT 1
            """,
            (owner_id,),
        )
        return None if row is None else self._row_to_intention(row)

    def match_pending(
        self,
        *,
        owner_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> tuple[bool, str, IntentionRecord | None]:
        pending = self.get_pending(owner_id)
        if pending is None:
            return False, "no matching pending intention", None
        if pending.tool_name != tool_name:
            return (
                False,
                f"intention tool mismatch: expected {pending.tool_name!r}",
                pending,
            )
        actual_hash = hash_tool_input(tool_input)
        if pending.tool_input_hash != actual_hash:
            return False, "intention tool input hash mismatch", pending
        return True, "pending intention matches tool and normalized input", pending

    def mark_allowed(self, action_id: str) -> IntentionRecord:
        self._db.execute(
            """
            UPDATE field_intentions SET status = 'ALLOWED'
            WHERE action_id = ? AND status = 'PENDING'
            """,
            (action_id,),
        )
        record = self.get(action_id)
        if record is None:
            raise ValueError(f"unknown intention {action_id!r}")
        return record

    def mark_denied(self, action_id: str) -> IntentionRecord:
        now = utc_now()
        self._db.execute(
            """
            UPDATE field_intentions SET status = 'DENIED', closed_at = ?
            WHERE action_id = ? AND status IN ('PENDING', 'ALLOWED')
            """,
            (now, action_id),
        )
        record = self.get(action_id)
        if record is None:
            raise ValueError(f"unknown intention {action_id!r}")
        return record

    def close(
        self,
        *,
        action_id: str,
        actual_result_summary: str,
        success: bool,
        actual_result_ref: str | None = None,
        latency_ms: int | None = None,
    ) -> tuple[ActionOutcome, AgencyAssessment]:
        intention = self.get(action_id)
        if intention is None:
            raise ValueError(f"unknown intention {action_id!r}")
        existing = self.outcome_for_action(action_id)
        if existing is not None:
            assessment = self._assessment_from_outcome(existing)
            return existing, assessment

        mismatch = self._mismatch_vector(
            intention=intention,
            actual_result_summary=actual_result_summary,
            success=success,
            latency_ms=latency_ms,
        )
        assessment = self.assess(
            registered_intention=True,
            mismatch_vector=mismatch,
            latency_ms=latency_ms,
            expected_latency_ms=intention.expected_latency_ms,
            exclusive_causal_fit=1.0 if success else 0.65,
        )
        outcome_id = f"out_{secrets.token_urlsafe(12)}"
        now = utc_now()
        result_hash = hashlib.sha256(actual_result_summary.encode("utf-8")).hexdigest()
        status = IntentionStatus.COMPLETED if success else IntentionStatus.FAILED
        with self._db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO field_action_outcomes(
                    outcome_id, action_id, field_id, tick_id, actual_result_ref,
                    actual_result_summary, actual_result_hash, success, latency_ms,
                    mismatch_vector_json, ownership_score, agency_assessment_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome_id,
                    intention.action_id,
                    intention.field_id,
                    intention.tick_id,
                    actual_result_ref,
                    actual_result_summary[:2000],
                    result_hash,
                    1 if success else 0,
                    latency_ms,
                    json.dumps(mismatch, sort_keys=True),
                    assessment.ownership_score,
                    json.dumps(assessment.model_dump(mode="json"), sort_keys=True),
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE field_intentions SET status = ?, closed_at = ?
                WHERE action_id = ?
                """,
                (status.value, now, action_id),
            )
            connection.execute(
                """
                UPDATE enacted_fields
                SET pending_intention_ref = NULL,
                    agency_state_json = ?,
                    updated_at = ?
                WHERE field_id = ?
                """,
                (
                    json.dumps(
                        {
                            "pending_intention_ref": None,
                            "ownership_score": assessment.ownership_score,
                            "last_outcome_ref": outcome_id,
                            "registered_intention": True,
                        },
                        sort_keys=True,
                    ),
                    now,
                    intention.field_id,
                ),
            )
        outcome = self.outcome_for_action(action_id)
        assert outcome is not None
        return outcome, assessment

    def assess_uncommanded_outcome(
        self,
        *,
        effect_match: float,
        temporal_contiguity: float = 0.5,
        exclusive_causal_fit: float = 0.2,
    ) -> AgencyAssessment:
        ownership = _clamp01(
            0.35 * 0.0
            + 0.30 * _clamp01(effect_match)
            + 0.20 * _clamp01(temporal_contiguity)
            + 0.15 * _clamp01(exclusive_causal_fit)
        )
        return AgencyAssessment(
            registered_intention=0.0,
            action_effect_match=_clamp01(effect_match),
            temporal_contiguity=_clamp01(temporal_contiguity),
            exclusive_causal_fit=_clamp01(exclusive_causal_fit),
            ownership_score=ownership,
            rationale=["no registered intention; ownership remains low"],
        )

    def assess(
        self,
        *,
        registered_intention: bool,
        mismatch_vector: dict[str, float],
        latency_ms: int | None,
        expected_latency_ms: int | None,
        exclusive_causal_fit: float,
    ) -> AgencyAssessment:
        mismatch = (
            sum(_clamp01(value) for value in mismatch_vector.values())
            / len(mismatch_vector)
            if mismatch_vector
            else 0.5
        )
        effect_match = 1.0 - mismatch
        temporal = _temporal_contiguity(latency_ms, expected_latency_ms)
        registered = 1.0 if registered_intention else 0.0
        ownership = _clamp01(
            0.35 * registered
            + 0.30 * effect_match
            + 0.20 * temporal
            + 0.15 * _clamp01(exclusive_causal_fit)
        )
        return AgencyAssessment(
            registered_intention=registered,
            action_effect_match=effect_match,
            temporal_contiguity=temporal,
            exclusive_causal_fit=_clamp01(exclusive_causal_fit),
            ownership_score=ownership,
            rationale=[
                "registered intention present"
                if registered_intention
                else "no registered intention",
                f"mean prediction mismatch={mismatch:.3f}",
            ],
        )

    def recover_dangling(self, owner_id: str = "self") -> list[ActionOutcome]:
        rows = self._db.fetchall(
            """
            SELECT action_id FROM field_intentions
            WHERE owner_id = ? AND status IN ('PENDING', 'ALLOWED')
            ORDER BY created_at ASC
            """,
            (owner_id,),
        )
        recovered: list[ActionOutcome] = []
        for row in rows:
            action_id = row["action_id"]
            outcome, _ = self.close(
                action_id=action_id,
                actual_result_summary="runtime recovered before a tool outcome was recorded",
                success=False,
                actual_result_ref="recovery:unknown",
            )
            self._db.execute(
                "UPDATE field_intentions SET status = 'UNKNOWN' WHERE action_id = ?",
                (action_id,),
            )
            recovered.append(outcome)
        return recovered

    def outcome_for_action(self, action_id: str) -> ActionOutcome | None:
        row = self._db.fetchone(
            "SELECT * FROM field_action_outcomes WHERE action_id = ?", (action_id,)
        )
        if row is None:
            return None
        return ActionOutcome(
            outcome_id=row["outcome_id"],
            action_id=row["action_id"],
            field_id=row["field_id"],
            tick_id=row["tick_id"],
            actual_result_ref=row["actual_result_ref"],
            actual_result_summary=row["actual_result_summary"],
            actual_result_hash=row["actual_result_hash"],
            success=bool(row["success"]),
            latency_ms=row["latency_ms"],
            mismatch_vector=json.loads(row["mismatch_vector_json"] or "{}"),
            ownership_score=float(row["ownership_score"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _mismatch_vector(
        *,
        intention: IntentionRecord,
        actual_result_summary: str,
        success: bool,
        latency_ms: int | None,
    ) -> dict[str, float]:
        actual_tokens = _tokens(actual_result_summary)
        effects = intention.predicted_effects
        mismatch: dict[str, float] = {}
        for channel in ("exteroceptive", "interoceptive", "social", "mnemonic"):
            expected = getattr(effects, channel)
            expected_tokens = _tokens(expected.summary)
            if not expected_tokens:
                value = 0.25 if success else 0.75
            else:
                overlap = len(expected_tokens & actual_tokens) / len(expected_tokens)
                value = 1.0 - overlap
                if not success:
                    value = max(value, 0.8)
            mismatch[channel] = round(_clamp01(value), 6)
        mismatch["latency"] = round(
            1.0 - _temporal_contiguity(latency_ms, intention.expected_latency_ms), 6
        )
        return mismatch

    @staticmethod
    def _row_to_intention(row) -> IntentionRecord:
        return IntentionRecord(
            action_id=row["action_id"],
            owner_id=row["owner_id"],
            field_id=row["field_id"],
            tick_id=row["tick_id"],
            tool_name=row["tool_name"],
            tool_input_hash=row["tool_input_hash"],
            normalized_tool_input=json.loads(row["normalized_tool_input_json"] or "{}"),
            intended_goal=row["intended_goal"],
            predicted_effects=PredictedEffects.model_validate(
                json.loads(row["predicted_effects_json"] or "{}")
            ),
            expected_latency_ms=row["expected_latency_ms"],
            confidence=float(row["confidence"]),
            status=row["status"],
            created_at=row["created_at"],
            closed_at=row["closed_at"],
        )

    def _assessment_from_outcome(self, outcome: ActionOutcome) -> AgencyAssessment:
        row = self._db.fetchone(
            "SELECT agency_assessment_json FROM field_action_outcomes WHERE outcome_id = ?",
            (outcome.outcome_id,),
        )
        if row is None:
            raise RuntimeError("outcome assessment is missing")
        return AgencyAssessment.model_validate(
            json.loads(row["agency_assessment_json"] or "{}")
        )


def normalize_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Return a stable JSON-compatible representation."""

    return json.loads(
        json.dumps(tool_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def hash_tool_input(tool_input: dict[str, Any]) -> str:
    normalized = json.dumps(
        normalize_tool_input(tool_input),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_external_tool(tool_name: str, tool_input: dict[str, Any] | None = None) -> bool:
    """Classify tool calls that can change the body, filesystem, or network."""

    name = tool_name.lower()
    if name.startswith("mcp__individual-kernel__"):
        return False
    external_fragments = (
        "__say",
        "__speak",
        "__post",
        "__notify",
        "__look_",
        "__camera_go",
        "__turn_on",
        "__turn_off",
        "__set_",
        "__toggle",
        "__activate",
        "__create_",
        "__move",
        "__send",
        "__forward",
        "__archive",
        "__trash",
        "__apply_label",
        "__remove_label",
        "__deploy",
        "__publish",
        "__install",
        "__uninstall",
        "__merge",
        "__push",
        "__add_comment",
        "__respond",
        "__cancel",
        "__reschedule",
        "__update_event",
        "__tweet",
        "__reply",
        "__like",
        "__repost",
        "__delete",
        "__remember",
        "__save_visual_memory",
        "__save_audio_memory",
        "mcp__tts__",
    )
    if any(fragment in name for fragment in external_fragments):
        return True
    if tool_name in {"Write", "Edit", "NotebookEdit"}:
        return True
    if tool_name == "Bash":
        command = str((tool_input or {}).get("command", ""))
        return not _bash_is_read_only(command)
    if name.startswith("mcp__"):
        internal_fragments = (
            "__get_",
            "__query_",
            "__list_",
            "__search_",
            "__read_",
            "__recall",
            "__introspect",
            "__diagnostic",
            "__check_",
            "__see",
            "__capture",
            "__begin_subjective_tick",
            "__add_workspace_candidate",
            "__commit_subjective_field",
            "__propose_field_action",
            "__close_field_action",
            "__close_subjective_tick",
            "__run_field_ablation",
            "__pause_field_runtime",
            "__resume_field_runtime",
            "__evaluate_action",
            "__compose_",
            "__plan_response",
        )
        return not any(fragment in name for fragment in internal_fragments)
    return False


def _bash_is_read_only(command: str) -> bool:
    """Conservatively recognize terminal-only inspection commands."""

    if not command.strip() or "$(" in command or "`" in command:
        return False
    sanitized = re.sub(r"\d*>\s*/dev/null", "", command)
    sanitized = re.sub(r"\d*>&\d+", "", sanitized)
    if re.search(r"(^|[^<])>{1,2}", sanitized):
        return False
    read_only = {
        "[",
        "awk",
        "basename",
        "cat",
        "cd",
        "cut",
        "date",
        "df",
        "dirname",
        "du",
        "echo",
        "fd",
        "file",
        "find",
        "git",
        "grep",
        "head",
        "id",
        "jq",
        "less",
        "ls",
        "md5sum",
        "more",
        "nl",
        "printenv",
        "printf",
        "pwd",
        "readlink",
        "realpath",
        "rg",
        "sed",
        "sha1sum",
        "sha256sum",
        "sort",
        "stat",
        "tail",
        "test",
        "tr",
        "true",
        "type",
        "uname",
        "uniq",
        "wc",
        "which",
        "whoami",
    }
    for segment in re.split(r"\|\||&&|[;|]", sanitized):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
            tokens.pop(0)
        if not tokens:
            continue
        executable = tokens[0].rsplit("/", 1)[-1]
        if executable not in read_only:
            return False
        if executable == "sed" and any(
            token == "-i" or token.startswith("-i") for token in tokens[1:]
        ):
            return False
        if executable == "find" and any(
            token in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
            for token in tokens[1:]
        ):
            return False
        if executable == "git" and not _git_is_read_only(tokens[1:]):
            return False
        if executable == "awk" and any(">" in token or "system(" in token for token in tokens[1:]):
            return False
    return True


def _git_is_read_only(arguments: list[str]) -> bool:
    if not arguments:
        return True
    command = next((item for item in arguments if not item.startswith("-")), "")
    if command in {
        "diff",
        "grep",
        "log",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "show",
        "status",
        "tag",
    }:
        return True
    if command == "branch":
        return all(
            item in {"branch", "--show-current", "--list", "-a", "-r", "-v", "-vv"}
            for item in arguments
        )
    if command == "remote":
        return all(item in {"remote", "-v", "--verbose"} for item in arguments)
    return False


def _tokens(text: str) -> set[str]:
    """Tokenize for mismatch overlap; ASCII words plus non-ASCII bigrams.

    Whole-word matching collapses Japanese summaries into one giant token, so
    non-ASCII runs are compared as character bigrams instead. ASCII behavior
    is unchanged.
    """
    lowered = text.lower()
    tokens = {
        token
        for token in re.findall(r"[a-z0-9_-]+", lowered)
        if len(token) > 2
    }
    for run in re.findall(r"[^\x00-\x7f]+", lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _temporal_contiguity(
    latency_ms: int | None,
    expected_latency_ms: int | None,
) -> float:
    if latency_ms is None:
        return 0.5
    if expected_latency_ms is None:
        return math.exp(-max(0, latency_ms - 5000) / 15000)
    scale = max(500.0, float(expected_latency_ms))
    return _clamp01(math.exp(-abs(latency_ms - expected_latency_ms) / (2.0 * scale)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
