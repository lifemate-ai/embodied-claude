"""Enacted-field tick producer and closed-loop runtime."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.events import EventStore
from social_core.models import SocialEventCreate
from social_core.time import utc_now

from individual_kernel_mcp._behavior import get_behavior
from individual_kernel_mcp.agency import (
    ActionOutcome,
    ActionProposal,
    AgencyAssessment,
    AgencyStore,
    IntentionRecord,
    is_external_tool,
)
from individual_kernel_mcp.allostasis import compose_valence, project_desires
from individual_kernel_mcp.attention_schema import AttentionSchema, AttentionSchemaTracker
from individual_kernel_mcp.bottleneck import ActionBottleneck
from individual_kernel_mcp.counterfactual import CounterfactualStore
from individual_kernel_mcp.enacted_field import (
    AgencyState,
    EnactedField,
    EnactedFieldStore,
    FieldStatus,
    InteroceptionState,
    PrecisionState,
    Protention,
    SelfOrigin,
    TriggerKind,
)
from individual_kernel_mcp.frame import ConsciousFrameInput, TickFrameStore
from individual_kernel_mcp.generative_model import NO_ACTION, ContextSignature
from individual_kernel_mcp.hor import HORInput, HORRecord, HORStore
from individual_kernel_mcp.prediction_loop import FieldPredictionLoop, generative_enabled
from individual_kernel_mcp.quality_geometry import QualityGeometry, QualitySignature
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    CompetitionResult,
    SourceMode,
    WorkspaceCandidate,
    WorkspaceEngine,
    competition_trace,
)


class TickCommit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: EnactedField
    competition: CompetitionResult
    attention_schema: AttentionSchema
    hor: HORRecord
    quality_signature: QualitySignature


class RecoveryReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_id: str
    continuity_token: str
    recovered_action_ids: list[str] = Field(default_factory=list)
    aborted_tick_ids: list[str] = Field(default_factory=list)
    resumed_field_id: str | None = None


class ToolGateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow: bool
    reason: str
    external: bool
    field_id: str | None = None
    action_id: str | None = None
    deferred: bool = False


class TemporalSequenceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chronological_fraction: float = Field(ge=0.0, le=1.0)
    continuity: float = Field(ge=0.0, le=1.0)
    protention_accuracy: float = Field(ge=0.0, le=1.0)


BoundaryEvaluator = Callable[[str, dict[str, Any], EnactedField], tuple[bool, str]]

# Control and surprise both default to the midpoint when nothing has been
# measured yet, so an unmeasured body state is neither confident nor alarmed.
UNMEASURED_CONTROLLABILITY = 0.5
UNRESOLVED_ERROR_WHEN_UNKNOWN = 0.5


def allostatic_enabled() -> bool:
    return bool(get_behavior("individual-kernel", "allostatic_valence", False))


def default_interoception_path() -> Path:
    """Return the platform-native interoception state path."""
    override = os.getenv("INTEROCEPTION_STATE_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "interoception_state.json"


class TickProducer:
    """Deterministically produces and commits one field per tick."""

    def __init__(
        self,
        db: SocialDB,
        *,
        workspace: WorkspaceEngine | None = None,
        fields: EnactedFieldStore | None = None,
        frames: TickFrameStore | None = None,
        attention: AttentionSchemaTracker | None = None,
        hors: HORStore | None = None,
        agency: AgencyStore | None = None,
        quality: QualityGeometry | None = None,
        prediction: FieldPredictionLoop | None = None,
        interoception_path: str | Path | None = None,
        desires_path: str | Path | None = None,
    ) -> None:
        self.db = db
        self.workspace = workspace or WorkspaceEngine(db)
        self.fields = fields or EnactedFieldStore(db)
        self.frames = frames or TickFrameStore(db)
        self.attention = attention or AttentionSchemaTracker(db=db, owner_id="self")
        self.hors = hors or HORStore(db)
        self.agency = agency or AgencyStore(db)
        self.quality = quality or QualityGeometry(db)
        self.prediction = prediction or FieldPredictionLoop(db, agency=self.agency)
        self.events = EventStore(db=db)
        self.interoception_path = (
            Path(interoception_path)
            if interoception_path is not None
            else default_interoception_path()
        )
        self.desires_path = (
            Path(desires_path).expanduser()
            if desires_path is not None
            else Path.home() / ".claude" / "desires.json"
        )

    def begin_tick(
        self,
        trigger: TriggerKind | str,
        owner_id: str = "self",
        *,
        person_id: str | None = None,
        user_text: str | None = None,
        session_id: str | None = None,
    ) -> EnactedField:
        trigger_kind = TriggerKind(trigger)
        runtime = self.fields.runtime_state(owner_id)
        if runtime is not None and runtime.state == "PAUSED":
            raise RuntimeError(f"field runtime for {owner_id!r} is paused")

        dangling = self.agency.get_pending(owner_id)
        if dangling is not None:
            self.agency.recover_dangling(owner_id)
            dangling_field = self.fields.get(dangling.field_id)
            if dangling_field and dangling_field.status == FieldStatus.COMMITTED:
                self.fields.invalidate(
                    dangling.field_id,
                    reason="new tick recovered a dangling intention before execution",
                )

        previous = self.fields.get_current(owner_id)
        if previous is None:
            recent = self.fields.query(owner_id=owner_id, limit=1)
            previous = recent[0] if recent else None
        continuity_token = (
            runtime.continuity_token
            if runtime is not None
            else f"cont_{secrets.token_urlsafe(16)}"
        )
        interoception = self._read_interoception(previous)
        desire_snapshot = self._read_json(self.desires_path)
        dominant_desire = str(desire_snapshot.get("dominant") or "") or None

        with self.db.transaction():
            frame = self.frames.record(
                ConsciousFrameInput(
                    person_id=person_id,
                    ignited=False,
                    conflicted=False,
                    dominant_desire=dominant_desire,
                    prediction_error={
                        "extero": 0.0,
                        "intero": interoception.uncertainty,
                        "mnemonic": 0.0,
                    },
                    affect_summary=(
                        f"valence={interoception.valence:.3f};"
                        f"arousal={interoception.arousal:.3f}"
                    ),
                )
            )
            field = self.fields.create_open(
                EnactedField(
                    owner_id=owner_id,
                    tick_id=frame.tick_id,
                    continuity_token=continuity_token,
                    previous_field_id=previous.field_id if previous else None,
                    trigger_kind=trigger_kind,
                    person_id=person_id,
                    session_id=session_id,
                    self_origin=SelfOrigin(
                        controllable_channels=[
                            "tool_action",
                            "camera_pose",
                            "voice",
                            "memory_write",
                        ],
                        body_resource_refs=[str(self.interoception_path)],
                        ownership_boundary_refs=["boundary:default"],
                    ),
                    retention_refs=self._retention_refs(previous),
                    interoception=interoception,
                    precision=self._initial_precision(previous),
                    agency_state=previous.agency_state if previous else AgencyState(),
                    epistemic_trace={
                        "producer": "deterministic",
                        "trigger": trigger_kind.value,
                        "raw_user_text_is_not_field_evidence": True,
                        "retention_decay": 0.72,
                    },
                )
            )
            self._add_trigger_candidate(field, trigger_kind)
            if user_text:
                self._ingest_user_event(
                    field=field,
                    user_text=user_text,
                    person_id=person_id,
                    session_id=session_id,
                )
            self._ingest_desires(field, desire_snapshot)
            self._ingest_interoceptive_candidate(field)
            self._ingest_previous_field(field, previous)
            self._ingest_recent_scene(field)
            self._ingest_recent_social(field, exclude_session_id=session_id)
            self.generate_self_model_candidate(field.tick_id, owner_id=owner_id)

        if user_text:
            self._ingest_memory_http(field.tick_id, user_text)
        return field

    def ingest_observation(
        self,
        tick_id: str,
        *,
        content_ref: str,
        summary: str,
        source_mode: SourceMode | str = SourceMode.LIVE,
        modality: str = "visual",
        precision: float = 0.8,
        prediction_error: float = 0.2,
        controllability: float = 0.5,
        expected_information_gain: float = 0.5,
        source: CandidateSource | str = CandidateSource.SENSOR,
    ) -> WorkspaceCandidate:
        return self.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=tick_id,
                kind=CandidateKind.PERCEPT,
                content_ref=content_ref,
                content_summary=summary[:320],
                source=CandidateSource(source),
                source_mode=SourceMode(source_mode),
                modality=modality,
                precision=precision,
                prediction_error=prediction_error,
                controllability=controllability,
                expected_information_gain=expected_information_gain,
            )
        )

    def ingest_memory_candidates(
        self,
        tick_id: str,
        candidates: Iterable[dict[str, Any]],
    ) -> list[WorkspaceCandidate]:
        stored: list[WorkspaceCandidate] = []
        for raw in candidates:
            memory_id = str(raw.get("memory_id") or raw.get("id") or "")
            if not memory_id:
                continue
            relevance = _clamp01(float(raw.get("relevance", raw.get("score", 0.5))))
            stored.append(
                self.workspace.add_candidate(
                    WorkspaceCandidate(
                        tick_id=tick_id,
                        kind=CandidateKind.MEMORY,
                        content_ref=f"memory:{memory_id}",
                        content_summary=str(
                            raw.get("content") or raw.get("summary") or memory_id
                        )[:320],
                        source=CandidateSource.MEMORY,
                        source_mode=SourceMode.REMEMBERED,
                        modality="internal",
                        precision=relevance,
                        goal_relevance=relevance * 0.6,
                        expected_information_gain=relevance * 0.5,
                        continuity_with_previous=relevance,
                    )
                )
            )
        return stored

    def ingest_interoception(
        self,
        tick_id: str,
        state: InteroceptionState | dict[str, Any],
    ) -> WorkspaceCandidate:
        field = self._require_open_tick(tick_id)
        intero = (
            state
            if isinstance(state, InteroceptionState)
            else InteroceptionState.model_validate(state)
        )
        updated = field.model_copy(update={"interoception": intero})
        self.fields.update(updated)
        return self._ingest_interoceptive_candidate(updated)

    def ingest_social_context(
        self,
        tick_id: str,
        *,
        content_ref: str,
        summary: str,
        social_relevance: float = 0.8,
        source_mode: SourceMode | str = SourceMode.INFERRED,
    ) -> WorkspaceCandidate:
        return self.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=tick_id,
                kind=CandidateKind.SOCIAL,
                content_ref=content_ref,
                content_summary=summary[:320],
                source=CandidateSource.SOCIAL,
                source_mode=SourceMode(source_mode),
                modality="social",
                precision=0.75,
                social_relevance=social_relevance,
                goal_relevance=social_relevance * 0.5,
            )
        )

    def generate_self_model_candidate(
        self,
        tick_id: str,
        *,
        owner_id: str = "self",
    ) -> WorkspaceCandidate | None:
        schema_row = self.db.fetchone(
            """
            SELECT * FROM attention_schemas
            WHERE owner_id = ?
            ORDER BY ts DESC, created_at DESC LIMIT 1
            """,
            (owner_id,),
        )
        hor_row = self.db.fetchone(
            """
            SELECT * FROM hor_records
            WHERE owner_id = ?
            ORDER BY ts DESC, created_at DESC LIMIT 1
            """,
            (owner_id,),
        )
        if schema_row is None and hor_row is None:
            return None
        schema_focus = schema_row["focal_target_ref"] if schema_row else None
        hor_target = hor_row["target_ref"] if hor_row else None
        grounded_same_commit = bool(
            schema_row
            and hor_row
            and hor_row["schema_snapshot_id"] == schema_row["schema_id"]
            and hor_row["source_tick_id"] == schema_row["source_tick_id"]
        )
        consistency = (
            1.0
            if grounded_same_commit
            or (schema_focus and hor_target and schema_focus == hor_target)
            else 0.65
            if schema_focus or hor_target
            else 0.35
        )
        predicted = schema_row["predicted_next_focus"] if schema_row else None
        return self.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=tick_id,
                kind=CandidateKind.SELF_MODEL,
                content_ref=(
                    f"attention_schema:{schema_row['schema_id']}"
                    if schema_row
                    else f"hor:{hor_row['hor_id']}"
                ),
                content_summary=(
                    f"attention={schema_focus or 'unknown'}; "
                    f"predicted_next={predicted or 'unknown'}"
                ),
                source=(
                    CandidateSource.ATTENTION_SCHEMA
                    if schema_row
                    else CandidateSource.HOR
                ),
                source_mode=SourceMode.INFERRED,
                modality="internal",
                precision=consistency,
                continuity_with_previous=consistency,
                expected_information_gain=0.2,
                conflict_penalty=1.0 - consistency,
            )
        )

    def compete_and_commit(self, tick_id: str) -> TickCommit:
        open_field = self._require_open_tick(tick_id)
        previous = (
            self.fields.get(open_field.previous_field_id)
            if open_field.previous_field_id
            else None
        )
        with self.db.transaction() as connection:
            result = self.workspace.compete(tick_id)
            winner = result.winner
            reality_score = calculate_reality_score(
                source_mode=winner.source_mode,
                sensor_coupling=_sensor_coupling(winner),
                action_controllability=winner.controllability,
                prediction_match=1.0 - abs(winner.prediction_error),
                recency=_source_recency(winner.source_mode),
            )
            protention = self._build_protention(result)
            precision = self._precision_from_competition(open_field, result)
            trace = dict(open_field.epistemic_trace)
            trace.update(
                {
                    "competition": competition_trace(result),
                    "focal_summary": winner.content_summary,
                    "provenance": {
                        "source": winner.source.value,
                        "source_mode": winner.source_mode.value,
                        "content_ref": winner.content_ref,
                    },
                    "evidence_refs": [
                        winner.content_ref,
                        *[item.content_ref for item in result.top_rejected],
                    ],
                    "ablation_flags": [],
                    "predicted_next_focus": protention.expected_content_ref,
                }
            )
            field = open_field.model_copy(
                update={
                    "reality_mode": winner.source_mode,
                    "reality_score": reality_score,
                    "focal_content_ref": winner.content_ref,
                    "focal_summary": winner.content_summary,
                    "peripheral_content_refs": [
                        item.content_ref for item in result.top_rejected
                    ],
                    "retention_refs": self._retention_refs(previous),
                    "protention": protention,
                    "precision": precision,
                    "affordance_refs": self._affordances_for(winner),
                    "attention_intensity": result.attention_intensity,
                    "predicted_next_focus": protention.expected_content_ref,
                    "epistemic_trace": trace,
                }
            )
            self.fields.update(field)
            connection.execute(
                """
                UPDATE tick_frames
                SET ignited = ?, conflicted = ?, attention_target_ref = ?,
                    winning_memory_ids_json = ?, prediction_error_json = ?,
                    reportability = ?
                WHERE tick_id = ?
                """,
                (
                    1 if result.ignited else 0,
                    1 if result.conflicted else 0,
                    winner.content_ref,
                    json.dumps(
                        [
                            item.content_ref.removeprefix("memory:")
                            for item in [winner, *result.top_rejected]
                            if item.kind == CandidateKind.MEMORY
                        ]
                    ),
                    json.dumps(
                        {
                            "extero": (
                                abs(winner.prediction_error)
                                if winner.modality in {"visual", "auditory"}
                                else 0.0
                            ),
                            "intero": (
                                abs(winner.prediction_error)
                                if winner.kind == CandidateKind.INTEROCEPTIVE
                                else open_field.interoception.uncertainty
                            ),
                            "mnemonic": (
                                abs(winner.prediction_error)
                                if winner.kind == CandidateKind.MEMORY
                                else 0.0
                            ),
                        },
                        sort_keys=True,
                    ),
                    winner.reportability,
                    tick_id,
                ),
            )
            committed = self.fields.commit(field.field_id)
            frame = self.frames.get_by_tick_id(tick_id)
            if frame is None:
                raise RuntimeError("tick frame disappeared during commit")
            schema = self.attention.update_from_competition(frame, result)
            self.attention.flush()
            hor = self._record_commit_hor(committed, winner, schema, result)
            quality = self.quality.local_signature(
                winner.content_ref,
                modality=winner.modality,
                source_mode=winner.source_mode,
                recent_valence=committed.interoception.valence,
            )
            final = committed.model_copy(
                update={
                    "attention_schema_ref": schema.schema_id,
                    "attention_intensity": result.attention_intensity,
                    "predicted_next_focus": schema.predicted_next_focus,
                    "hor_refs": [hor.hor_id],
                    "quality_signature_ref": quality.signature_id,
                }
            )
            final = self.fields.update(final)
            transition_action_id = self._transition_action_id(previous)
            self.quality.record_transition(
                owner_id=final.owner_id,
                from_field_id=previous.field_id if previous else None,
                to_field_id=final.field_id,
                from_content_ref=previous.focal_content_ref if previous else None,
                to_content_ref=final.focal_content_ref,
                action_id=transition_action_id,
                continuity_score=self._continuity_score(previous, final),
                prediction_match=self._prediction_match(previous, final),
                temporal_order=self._next_temporal_order(final.owner_id),
                metadata={"trigger": final.trigger_kind.value},
            )
            if generative_enabled():
                try:
                    self.prediction.record_transition_and_update(
                        previous=previous,
                        committed=final,
                        action_id=transition_action_id,
                    )
                except Exception:
                    # A prediction-layer fault must never abort a field commit
                    # (same tolerance as the memory HTTP ingest).
                    pass
        return TickCommit(
            field=final,
            competition=result,
            attention_schema=schema,
            hor=hor,
            quality_signature=quality,
        )

    def get_current_field(self, owner_id: str = "self") -> EnactedField | None:
        return self.fields.get_current(owner_id)

    def invalidate_field(
        self,
        reason: str,
        *,
        owner_id: str = "self",
        field_id: str | None = None,
    ) -> EnactedField:
        field = self.fields.get(field_id) if field_id else self.fields.get_current(owner_id)
        if field is None:
            raise ValueError("no current committed field to invalidate")
        return self.fields.invalidate(field.field_id, reason=reason)

    def close_tick(self, tick_id: str, output_summary: str | None = None) -> EnactedField:
        field = self.fields.get_by_tick(tick_id)
        if field is None:
            raise ValueError(f"unknown tick {tick_id!r}")
        return self.fields.close(field.field_id, output_summary=output_summary)

    def recover_stale_runtime(self, owner_id: str = "self") -> RecoveryReport:
        runtime = self.fields.runtime_state(owner_id)
        token = (
            runtime.continuity_token
            if runtime is not None
            else f"cont_{secrets.token_urlsafe(16)}"
        )
        outcomes = self.agency.recover_dangling(owner_id)
        recovered_ids = [item.action_id for item in outcomes]
        for outcome in outcomes:
            field = self.fields.get(outcome.field_id)
            if field and field.status in {FieldStatus.OPEN, FieldStatus.COMMITTED}:
                self.fields.invalidate(
                    field.field_id,
                    reason="session recovery closed dangling action as unknown",
                )
        open_rows = self.db.fetchall(
            "SELECT field_id, tick_id FROM enacted_fields WHERE owner_id = ? AND status = 'OPEN'",
            (owner_id,),
        )
        aborted: list[str] = []
        with self.db.transaction() as connection:
            for row in open_rows:
                connection.execute(
                    """
                    UPDATE enacted_fields
                    SET status = 'ABORTED', closed_at = ?, updated_at = ?
                    WHERE field_id = ?
                    """,
                    (utc_now(), utc_now(), row["field_id"]),
                )
                aborted.append(row["tick_id"])
            connection.execute(
                """
                INSERT INTO field_runtime_state(
                    owner_id, continuity_token, current_field_id, open_tick_id,
                    state, last_recovery_at, updated_at
                ) VALUES (?, ?, NULL, NULL, 'ACTIVE', ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    continuity_token = excluded.continuity_token,
                    open_tick_id = NULL,
                    last_recovery_at = excluded.last_recovery_at,
                    updated_at = excluded.updated_at
                """,
                (owner_id, token, utc_now(), utc_now()),
            )
        current = self.fields.get_current(owner_id)
        return RecoveryReport(
            owner_id=owner_id,
            continuity_token=token,
            recovered_action_ids=recovered_ids,
            aborted_tick_ids=aborted,
            resumed_field_id=current.field_id if current else None,
        )

    def pause_field_runtime(self, owner_id: str = "self"):
        return self.fields.pause(owner_id)

    def resume_field_runtime(self, owner_id: str = "self"):
        return self.fields.resume(owner_id)

    def _add_trigger_candidate(
        self,
        field: EnactedField,
        trigger: TriggerKind,
    ) -> WorkspaceCandidate:
        kind = (
            CandidateKind.GOAL
            if trigger in {TriggerKind.HEARTBEAT, TriggerKind.AUTONOMOUS}
            else CandidateKind.PERCEPT
        )
        return self.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=field.tick_id,
                kind=kind,
                content_ref=f"trigger:{trigger.value}",
                content_summary=f"runtime trigger: {trigger.value}",
                source=CandidateSource.EXPLICIT,
                source_mode=SourceMode.INFERRED,
                modality="internal",
                precision=0.6,
                goal_relevance=0.5,
                continuity_with_previous=0.2,
            )
        )

    def _ingest_user_event(
        self,
        *,
        field: EnactedField,
        user_text: str,
        person_id: str | None,
        session_id: str | None,
    ) -> WorkspaceCandidate:
        digest = hashlib.sha256(user_text.encode("utf-8")).hexdigest()[:16]
        event = self.events.ingest(
            SocialEventCreate(
                ts=utc_now(),
                source="efpf_hook",
                kind="human_utterance",
                person_id=person_id,
                session_id=session_id,
                correlation_id=f"efpf:{session_id or 'none'}:{digest}",
                confidence=1.0,
                payload_json={"text": user_text},
            )
        )
        return self.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=field.tick_id,
                kind=CandidateKind.SOCIAL,
                content_ref=f"event:{event.event_id}",
                content_summary=f"user utterance: {user_text}"[:320],
                source=CandidateSource.USER_EVENT,
                source_mode=SourceMode.LIVE,
                modality="social",
                precision=0.95,
                prediction_error=0.35,
                goal_relevance=0.65,
                expected_information_gain=0.45,
                social_relevance=1.0,
                controllability=0.35,
            )
        )

    def _ingest_desires(
        self,
        field: EnactedField,
        snapshot: dict[str, Any],
    ) -> list[WorkspaceCandidate]:
        desires = snapshot.get("desires") or {}
        discomforts = snapshot.get("discomforts") or {}
        dominant = snapshot.get("dominant")
        stored: list[WorkspaceCandidate] = []
        for name, level in desires.items():
            relevance = _clamp01(
                float(discomforts.get(name, level if name == dominant else 0.0))
            )
            stored.append(
                self.workspace.add_candidate(
                    WorkspaceCandidate(
                        tick_id=field.tick_id,
                        kind=CandidateKind.GOAL,
                        content_ref=f"desire:{name}",
                        content_summary=f"need {name}",
                        source=CandidateSource.DESIRE,
                        source_mode=SourceMode.INFERRED,
                        modality="internal",
                        precision=0.7,
                        need_relevance=relevance,
                        goal_relevance=1.0 if name == dominant else relevance * 0.5,
                        continuity_with_previous=0.25,
                        controllability=0.6,
                    )
                )
            )
        return stored

    def _ingest_interoceptive_candidate(
        self,
        field: EnactedField,
    ) -> WorkspaceCandidate:
        state = field.interoception
        salience = max(
            abs(state.valence),
            state.uncertainty,
            max(state.need_vector.values(), default=0.0),
        )
        return self.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=field.tick_id,
                kind=CandidateKind.INTEROCEPTIVE,
                content_ref=f"interoception:{field.started_at}",
                content_summary=(
                    f"bounded body state valence={state.valence:.2f}, "
                    f"arousal={state.arousal:.2f}"
                ),
                source=CandidateSource.INTEROCEPTION,
                source_mode=SourceMode.LIVE,
                modality="internal",
                precision=0.7,
                prediction_error=state.uncertainty,
                need_relevance=salience,
                controllability=state.controllability,
            )
        )

    def _ingest_previous_field(
        self,
        field: EnactedField,
        previous: EnactedField | None,
    ) -> WorkspaceCandidate | None:
        if previous is None or previous.focal_content_ref is None:
            return None
        return self.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=field.tick_id,
                kind=CandidateKind.SELF_MODEL,
                content_ref=f"field:{previous.field_id}",
                content_summary=previous.focal_summary or previous.focal_content_ref,
                source=CandidateSource.PREVIOUS_FIELD,
                source_mode=SourceMode.MIXED,
                modality="internal",
                precision=previous.precision.self_model,
                continuity_with_previous=0.9,
                switching_cost=0.2,
                controllability=previous.interoception.controllability,
            )
        )

    def _ingest_recent_scene(self, field: EnactedField) -> WorkspaceCandidate | None:
        row = self.db.fetchone(
            """
            SELECT frame_id, scene_summary FROM scene_frames
            ORDER BY ts DESC LIMIT 1
            """
        )
        if row is None:
            return None
        return self.ingest_observation(
            field.tick_id,
            content_ref=f"scene:{row['frame_id']}",
            summary=row["scene_summary"],
            source_mode=SourceMode.LIVE,
            modality="visual",
            precision=0.85,
            prediction_error=0.25,
            controllability=0.65,
            expected_information_gain=0.55,
        )

    def _ingest_recent_social(
        self,
        field: EnactedField,
        *,
        exclude_session_id: str | None,
    ) -> WorkspaceCandidate | None:
        row = self.db.fetchone(
            """
            SELECT event_id, payload_json, source, session_id FROM events
            WHERE kind IN ('human_utterance', 'mention_received', 'touchpoint')
            ORDER BY ts DESC, event_seq DESC LIMIT 1
            """
        )
        if row is None or (
            exclude_session_id and row["session_id"] == exclude_session_id
        ):
            return None
        payload = _json_loads(row["payload_json"], {})
        summary = str(payload.get("text") or payload.get("summary") or row["event_id"])
        return self.ingest_social_context(
            field.tick_id,
            content_ref=f"event:{row['event_id']}",
            summary=summary,
            social_relevance=0.7,
            source_mode=SourceMode.REMEMBERED,
        )

    def _ingest_memory_http(self, tick_id: str, user_text: str) -> None:
        port = int(os.getenv("MEMORY_HTTP_PORT", "18900"))
        url = (
            f"http://127.0.0.1:{port}/recall?"
            + urllib.parse.urlencode({"q": user_text, "n": 4})
        )
        try:
            with urllib.request.urlopen(url, timeout=1.2) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except Exception:
            return
        items = raw if isinstance(raw, list) else raw.get("memories", raw.get("results", []))
        if isinstance(items, list):
            self.ingest_memory_candidates(
                tick_id,
                [item for item in items if isinstance(item, dict)],
            )

    def _read_interoception(
        self,
        previous: EnactedField | None,
    ) -> InteroceptionState:
        if allostatic_enabled():
            try:
                return self._allostatic_interoception(previous)
            except Exception:
                # The body state must always be producible; fall back rather
                # than block a tick on a prediction or storage fault.
                pass
        return self._legacy_interoception(previous)

    def _measured_controllability(self, previous: EnactedField | None) -> float:
        """Control as actually measured, not as inferred from discomfort.

        Ownership is only meaningful once an intention has been registered and
        closed; before that there is nothing measured to report, so the neutral
        midpoint stands in.
        """

        if previous is None:
            return UNMEASURED_CONTROLLABILITY
        agency = previous.agency_state
        if agency is None or not agency.registered_intention:
            return UNMEASURED_CONTROLLABILITY
        return _clamp01(agency.ownership_score)

    def _expected_valence_delta(self, previous: EnactedField | None) -> float:
        if previous is None:
            return 0.0
        signature = ContextSignature.from_field(previous, action_kind=NO_ACTION)
        return self.prediction.model.expected_valence_delta(signature)

    def _unresolved_prediction_error(self) -> float:
        recent = self.prediction.transitions.query(limit=1)
        if not recent:
            return UNRESOLVED_ERROR_WHEN_UNKNOWN
        return _clamp01(recent[0].mean_prediction_error)

    def _allostatic_interoception(
        self,
        previous: EnactedField | None,
    ) -> InteroceptionState:
        """Body state composed from projected needs and the learned model."""

        raw = self._read_json(self.interoception_path)
        now_readings = raw.get("now") or {}
        # utc_now() yields an ISO string for storage; the projection needs a
        # real instant to subtract from.
        projection = project_desires(
            self._read_json(self.desires_path), now=datetime.now(timezone.utc)
        )
        control = self._measured_controllability(previous)
        uncertainty = 0.35 if raw else 0.6
        state = compose_valence(
            expected_valence_delta=self._expected_valence_delta(previous),
            mean_discomfort=projection.mean_discomfort,
            controllability=control,
            uncertainty=uncertainty,
            unresolved_error=self._unresolved_prediction_error(),
        )
        valence = state.valence
        arousal = _clamp01(float(now_readings.get("arousal", 0.0)) / 100.0)
        exposure = 0.0
        if previous and valence < -0.2:
            exposure = min(
                86400.0,
                previous.interoception.negative_exposure_seconds
                + max(0.0, _seconds_between(previous.updated_at, utc_now())),
            )
        return InteroceptionState(
            valence=valence,
            arousal=arousal,
            uncertainty=uncertainty,
            controllability=control,
            need_vector=projection.discomforts or projection.desires,
            resource_refs=([str(self.interoception_path)] if raw else []),
            negative_exposure_seconds=exposure,
        )

    def _legacy_interoception(
        self,
        previous: EnactedField | None,
    ) -> InteroceptionState:
        raw = self._read_json(self.interoception_path)
        now = raw.get("now") or {}
        desire = self._read_json(self.desires_path)
        desires = {
            str(key): _clamp01(float(value))
            for key, value in (desire.get("desires") or {}).items()
        }
        discomforts = {
            str(key): _clamp01(float(value))
            for key, value in (desire.get("discomforts") or {}).items()
        }
        max_discomfort = max(discomforts.values(), default=0.0)
        raw_valence = -0.6 * max_discomfort
        previous_valence = previous.interoception.valence if previous else 0.0
        valence = 0.65 * previous_valence + 0.35 * raw_valence
        if raw_valence >= previous_valence and previous_valence < 0.0:
            valence += 0.04
        valence = max(-0.8, min(0.8, valence))
        arousal = _clamp01(float(now.get("arousal", 0.0)) / 100.0)
        uncertainty = 0.35 if raw else 0.6
        control = _clamp01(0.45 + 0.25 * (1.0 - max_discomfort))
        exposure = 0.0
        if previous and valence < -0.2:
            exposure = min(
                86400.0,
                previous.interoception.negative_exposure_seconds
                + max(0.0, _seconds_between(previous.updated_at, utc_now())),
            )
        return InteroceptionState(
            valence=valence,
            arousal=arousal,
            uncertainty=uncertainty,
            controllability=control,
            need_vector=discomforts or desires,
            resource_refs=([str(self.interoception_path)] if raw else []),
            negative_exposure_seconds=exposure,
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _retention_refs(previous: EnactedField | None) -> list[str]:
        if previous is None:
            return []
        refs: list[str] = []
        if previous.focal_content_ref:
            refs.append(previous.focal_content_ref)
        refs.extend(previous.retention_refs[:3])
        return list(dict.fromkeys(refs))

    @staticmethod
    def _initial_precision(previous: EnactedField | None) -> PrecisionState:
        if previous is None:
            return PrecisionState()
        return PrecisionState(
            extero=0.6 * previous.precision.extero + 0.2,
            intero=0.6 * previous.precision.intero + 0.2,
            mnemonic=0.6 * previous.precision.mnemonic + 0.2,
            social=0.6 * previous.precision.social + 0.2,
            self_model=0.7 * previous.precision.self_model + 0.15,
        )

    @staticmethod
    def _precision_from_competition(
        field: EnactedField,
        result: CompetitionResult,
    ) -> PrecisionState:
        winner = result.winner
        self_feedback = _clamp01(
            field.precision.self_model * 0.65
            + result.attention_intensity * 0.25
            + (0.1 if not result.conflicted else 0.0)
        )
        return PrecisionState(
            extero=(
                winner.precision
                if winner.modality in {"visual", "auditory"}
                else field.precision.extero * 0.8
            ),
            intero=(
                winner.precision
                if winner.kind == CandidateKind.INTEROCEPTIVE
                else field.precision.intero * 0.8
            ),
            mnemonic=(
                winner.precision
                if winner.kind == CandidateKind.MEMORY
                else field.precision.mnemonic * 0.8
            ),
            social=(
                winner.precision
                if winner.kind == CandidateKind.SOCIAL
                else field.precision.social * 0.8
            ),
            self_model=self_feedback,
        )

    @staticmethod
    def _build_protention(result: CompetitionResult) -> Protention:
        predicted = result.top_rejected[0] if result.top_rejected else result.winner
        confidence = _clamp01(
            0.45 * result.attention_intensity
            + 0.35 * (1.0 - result.entropy)
            + 0.20 * (0.0 if result.conflicted else 1.0)
        )
        return Protention(
            expected_content_ref=predicted.content_ref,
            summary=f"next likely focus: {predicted.content_summary or predicted.content_ref}",
            confidence=confidence,
        )

    @staticmethod
    def _affordances_for(candidate: WorkspaceCandidate) -> list[str]:
        mapping = {
            CandidateKind.PERCEPT: ["inspect", "compare_prediction", "orient"],
            CandidateKind.MEMORY: ["use_memory", "verify_source", "recontextualize"],
            CandidateKind.INTEROCEPTIVE: ["regulate", "defer", "seek_information"],
            CandidateKind.SOCIAL: ["respond", "remember", "clarify"],
            CandidateKind.GOAL: ["propose_action", "defer", "gather_evidence"],
            CandidateKind.ACTION: ["intend", "predict_effect", "gate_action"],
            CandidateKind.SELF_MODEL: ["stabilize_focus", "recompete", "inspect_conflict"],
        }
        return mapping[candidate.kind]

    def _record_commit_hor(
        self,
        field: EnactedField,
        winner: WorkspaceCandidate,
        schema: AttentionSchema,
        result: CompetitionResult,
    ) -> HORRecord:
        if winner.kind == CandidateKind.MEMORY:
            mode, target_kind, target_ref = "remembering", "memory", winner.content_ref
        elif winner.kind == CandidateKind.GOAL:
            mode, target_kind, target_ref = "wanting", "desire", winner.content_ref
        elif winner.kind == CandidateKind.ACTION:
            mode, target_kind, target_ref = "intending", "action", winner.content_ref
        elif winner.kind == CandidateKind.INTEROCEPTIVE:
            mode, target_kind, target_ref = "feeling", "frame", field.tick_id
        elif winner.modality == "visual" and winner.source_mode == SourceMode.LIVE:
            mode, target_kind, target_ref = "seeing", "frame", field.tick_id
        else:
            mode, target_kind, target_ref = "attending", "frame", field.tick_id
        return self.hors.record(
            HORInput(
                owner_id=field.owner_id,
                target_kind=target_kind,  # type: ignore[arg-type]
                target_ref=target_ref,
                asserted_mode=mode,  # type: ignore[arg-type]
                asserted_content=f"I am {mode} {winner.content_ref}",
                schema_snapshot_id=schema.schema_id,
                source_tick_id=field.tick_id,
                confidence=_clamp01(
                    0.45 * result.attention_intensity
                    + 0.35 * winner.precision
                    + 0.20 * (1.0 - result.entropy)
                ),
                source="schema_readout",
            )
        )

    @staticmethod
    def _continuity_score(
        previous: EnactedField | None,
        current: EnactedField,
    ) -> float:
        if previous is None:
            return 0.0
        if previous.focal_content_ref in current.retention_refs:
            return 0.9
        if previous.focal_content_ref == current.focal_content_ref:
            return 1.0
        return 0.3

    @staticmethod
    def _prediction_match(
        previous: EnactedField | None,
        current: EnactedField,
    ) -> float:
        if previous is None or previous.protention.expected_content_ref is None:
            return 0.5
        return (
            1.0
            if previous.protention.expected_content_ref == current.focal_content_ref
            else 0.0
        )

    def _next_temporal_order(self, owner_id: str) -> int:
        row = self.db.fetchone(
            "SELECT MAX(temporal_order) AS n FROM field_transitions WHERE owner_id = ?",
            (owner_id,),
        )
        return int(row["n"] or 0) + 1 if row else 1

    def _transition_action_id(self, previous: EnactedField | None) -> str | None:
        if previous is None:
            return None
        if previous.agency_state.pending_intention_ref:
            return previous.agency_state.pending_intention_ref
        outcome_ref = previous.agency_state.last_outcome_ref
        if not outcome_ref:
            return None
        row = self.db.fetchone(
            "SELECT action_id FROM field_action_outcomes WHERE outcome_id = ?",
            (outcome_ref,),
        )
        return str(row["action_id"]) if row else None

    def _require_open_tick(self, tick_id: str) -> EnactedField:
        field = self.fields.get_by_tick(tick_id)
        if field is None:
            raise ValueError(f"unknown tick {tick_id!r}")
        if field.status != FieldStatus.OPEN:
            raise ValueError(f"tick {tick_id!r} is not OPEN")
        return field


class FieldRuntime:
    """High-level action gate that closes tool outcomes into a new field."""

    def __init__(
        self,
        db: SocialDB,
        *,
        producer: TickProducer | None = None,
        boundary_evaluator: BoundaryEvaluator | None = None,
        compatibility_mode: bool = False,
    ) -> None:
        self.db = db
        self.producer = producer or TickProducer(db)
        self.agency = self.producer.agency
        self.frames = self.producer.frames
        self.fields = self.producer.fields
        self.counterfactuals = CounterfactualStore(db)
        self.bottleneck = ActionBottleneck(
            db=db,
            frame_store=self.frames,
            counterfactual_store=self.counterfactuals,
        )
        self.boundary_evaluator = boundary_evaluator
        self.compatibility_mode = compatibility_mode

    def propose_action(
        self,
        proposal: ActionProposal,
        *,
        owner_id: str = "self",
    ) -> IntentionRecord:
        intention = self.agency.propose(proposal, owner_id=owner_id)
        field = self.fields.get(intention.field_id)
        if field is not None:
            if generative_enabled():
                try:
                    distribution = self.producer.prediction.imagine_for_intention(
                        field, intention
                    )
                except Exception:
                    distribution = None
                if distribution is not None:
                    trace = dict(field.epistemic_trace)
                    trace["protention_distribution_ref"] = distribution.distribution_id
                    field = field.model_copy(update={"epistemic_trace": trace})
            self.fields.update(field)
        return intention

    def gate_tool(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, Any],
        owner_id: str = "self",
    ) -> ToolGateDecision:
        external = is_external_tool(tool_name, tool_input)
        if not external:
            return ToolGateDecision(
                allow=True,
                reason="internal/read-only tool; outward action gate not required",
                external=False,
            )
        runtime_state = self.fields.runtime_state(owner_id)
        if runtime_state is not None and runtime_state.state == "PAUSED":
            return ToolGateDecision(
                allow=False,
                reason="field runtime is paused; resume before outward action",
                external=True,
                field_id=runtime_state.current_field_id,
            )
        field = self.fields.get_current(owner_id)
        if field is None:
            if self.compatibility_mode:
                return ToolGateDecision(
                    allow=True,
                    reason="compatibility mode: no committed field",
                    external=True,
                )
            return ToolGateDecision(
                allow=False,
                reason="no COMMITTED field; refresh the field before outward action",
                external=True,
            )
        if field.status != FieldStatus.COMMITTED:
            return ToolGateDecision(
                allow=False,
                reason="current field is stale or invalidated; refresh before outward action",
                external=True,
                field_id=field.field_id,
            )
        matches, reason, intention = self.agency.match_pending(
            owner_id=owner_id,
            tool_name=tool_name,
            tool_input=tool_input,
        )
        if not matches or intention is None:
            return ToolGateDecision(
                allow=False,
                reason=reason + "; call propose_field_action first",
                external=True,
                field_id=field.field_id,
                action_id=intention.action_id if intention else None,
            )
        if intention.field_id != field.field_id:
            return ToolGateDecision(
                allow=False,
                reason="pending intention belongs to a superseded field",
                external=True,
                field_id=field.field_id,
                action_id=intention.action_id,
            )
        if self.boundary_evaluator is not None:
            allowed, boundary_reason = self.boundary_evaluator(
                tool_name, tool_input, field
            )
            if not allowed:
                self.agency.mark_denied(intention.action_id)
                if generative_enabled():
                    try:
                        self.producer.prediction.mark_denied_trajectory(
                            intention.action_id
                        )
                    except Exception:
                        pass
                return ToolGateDecision(
                    allow=False,
                    reason=f"boundary policy denied action: {boundary_reason}",
                    external=True,
                    field_id=field.field_id,
                    action_id=intention.action_id,
                )
        bottleneck = self.bottleneck.commit_action(
            tick_id=field.tick_id,
            action_ref=intention.action_id,
            action_payload={"tool_name": tool_name, "tool_input_hash": intention.tool_input_hash},
            person_id=field.person_id,
        )
        if not bottleneck.ok:
            return ToolGateDecision(
                allow=False,
                reason="one outward action is already committed for this field",
                external=True,
                field_id=field.field_id,
                action_id=intention.action_id,
                deferred=True,
            )
        self.agency.mark_allowed(intention.action_id)
        if generative_enabled():
            try:
                self.producer.prediction.mark_enacted(intention.action_id)
            except Exception:
                pass
        return ToolGateDecision(
            allow=True,
            reason="field, intention, input hash, boundary, and bottleneck checks passed",
            external=True,
            field_id=field.field_id,
            action_id=intention.action_id,
        )

    def close_tool(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, Any],
        actual_result_summary: str,
        success: bool,
        owner_id: str = "self",
        actual_result_ref: str | None = None,
        latency_ms: int | None = None,
        session_id: str | None = None,
    ) -> tuple[ActionOutcome | None, AgencyAssessment | None, EnactedField | None]:
        external = is_external_tool(tool_name, tool_input)
        outcome: ActionOutcome | None = None
        assessment: AgencyAssessment | None = None
        previous = self.fields.get_current(owner_id)
        if external:
            matches, _, intention = self.agency.match_pending(
                owner_id=owner_id,
                tool_name=tool_name,
                tool_input=tool_input,
            )
            if not matches or intention is None:
                return None, self.agency.assess_uncommanded_outcome(effect_match=0.5), None
            outcome, assessment = self.agency.close(
                action_id=intention.action_id,
                actual_result_summary=actual_result_summary,
                success=success,
                actual_result_ref=actual_result_ref,
                latency_ms=latency_ms,
            )
        if previous is not None:
            self.fields.invalidate(
                previous.field_id,
                reason=(
                    "tool result changed the available evidence"
                    if success
                    else "tool failure changed prediction confidence"
                ),
            )
        if not external and not is_field_refreshing_tool(tool_name):
            return outcome, assessment, None

        micro = self.producer.begin_tick(
            TriggerKind.TOOL_RESULT,
            owner_id=owner_id,
            person_id=previous.person_id if previous else None,
            session_id=session_id,
        )
        result_ref = (
            f"outcome:{outcome.outcome_id}"
            if outcome
            else actual_result_ref
            or f"tool_result:{hashlib.sha256(actual_result_summary.encode()).hexdigest()[:16]}"
        )
        self.producer.ingest_observation(
            micro.tick_id,
            content_ref=result_ref,
            summary=actual_result_summary,
            source_mode=SourceMode.LIVE if success else SourceMode.INFERRED,
            modality=_tool_modality(tool_name),
            precision=0.9 if success else 0.65,
            prediction_error=(
                _mean_mismatch(outcome.mismatch_vector)
                if outcome
                else (0.2 if success else 0.9)
            ),
            controllability=assessment.ownership_score if assessment else 0.2,
            expected_information_gain=0.8,
            source=CandidateSource.TOOL_RESULT,
        )
        commit = self.producer.compete_and_commit(micro.tick_id)
        if outcome is not None and generative_enabled():
            try:
                self.producer.prediction.reconcile_trajectories(
                    action_id=outcome.action_id,
                    next_field_id=commit.field.field_id,
                )
            except Exception:
                pass
        return outcome, assessment, commit.field

    def close_action(
        self,
        *,
        action_id: str,
        actual_result_summary: str,
        success: bool,
        actual_result_ref: str | None = None,
        latency_ms: int | None = None,
        session_id: str | None = None,
    ) -> tuple[ActionOutcome, AgencyAssessment, EnactedField]:
        intention = self.agency.get(action_id)
        if intention is None:
            raise ValueError(f"unknown intention {action_id!r}")
        source_field = self.fields.get(intention.field_id)
        existing_outcome = self.agency.outcome_for_action(action_id)
        outcome, assessment = self.agency.close(
            action_id=action_id,
            actual_result_summary=actual_result_summary,
            success=success,
            actual_result_ref=actual_result_ref,
            latency_ms=latency_ms,
        )
        if existing_outcome is not None:
            current = self.fields.get_current(intention.owner_id)
            if current is None:
                raise RuntimeError("action was already closed but no current field is available")
            return outcome, assessment, current
        if source_field and source_field.status in {
            FieldStatus.OPEN,
            FieldStatus.COMMITTED,
        }:
            self.fields.invalidate(
                source_field.field_id,
                reason="structured action outcome requires field refresh",
            )
        micro = self.producer.begin_tick(
            TriggerKind.TOOL_RESULT,
            owner_id=intention.owner_id,
            person_id=source_field.person_id if source_field else None,
            session_id=session_id,
        )
        self.producer.ingest_observation(
            micro.tick_id,
            content_ref=f"outcome:{outcome.outcome_id}",
            summary=actual_result_summary,
            source_mode=SourceMode.LIVE if success else SourceMode.INFERRED,
            modality=_tool_modality(intention.tool_name),
            precision=0.9 if success else 0.65,
            prediction_error=_mean_mismatch(outcome.mismatch_vector),
            controllability=assessment.ownership_score,
            expected_information_gain=0.8,
            source=CandidateSource.TOOL_RESULT,
        )
        commit = self.producer.compete_and_commit(micro.tick_id)
        if generative_enabled():
            try:
                self.producer.prediction.reconcile_trajectories(
                    action_id=action_id,
                    next_field_id=commit.field.field_id,
                )
            except Exception:
                pass
        return outcome, assessment, commit.field


def calculate_reality_score(
    *,
    source_mode: SourceMode | str,
    sensor_coupling: float,
    action_controllability: float,
    prediction_match: float,
    recency: float,
) -> float:
    """Reality marking based on coupling, control, prediction, and recency."""

    SourceMode(source_mode)
    return _clamp01(
        0.35 * _clamp01(sensor_coupling)
        + 0.30 * _clamp01(action_controllability)
        + 0.20 * _clamp01(prediction_match)
        + 0.15 * _clamp01(recency)
    )


def temporal_sequence_metrics(fields: list[EnactedField]) -> TemporalSequenceMetrics:
    """Measure order-sensitive retention and next-focus calibration."""

    if len(fields) < 2:
        return TemporalSequenceMetrics(
            chronological_fraction=1.0,
            continuity=1.0,
            protention_accuracy=1.0,
        )
    chronological = 0
    retention_hits = 0
    protention_hits = 0
    for previous, current in zip(fields, fields[1:]):
        try:
            left = datetime.fromisoformat(previous.started_at.replace("Z", "+00:00"))
            right = datetime.fromisoformat(current.started_at.replace("Z", "+00:00"))
            chronological += int(right >= left)
        except ValueError:
            pass
        retention_hits += int(
            previous.focal_content_ref is not None
            and previous.focal_content_ref in current.retention_refs
        )
        protention_hits += int(
            previous.protention.expected_content_ref is not None
            and previous.protention.expected_content_ref == current.focal_content_ref
        )
    denominator = len(fields) - 1
    chronology = chronological / denominator
    return TemporalSequenceMetrics(
        chronological_fraction=chronology,
        continuity=(retention_hits / denominator) * chronology,
        protention_accuracy=(protention_hits / denominator) * chronology,
    )


def is_field_refreshing_tool(tool_name: str) -> bool:
    name = tool_name.lower()
    if name in {"read", "grep", "glob", "webfetch", "websearch", "bash"}:
        return True
    return any(
        fragment in name
        for fragment in (
            "__see",
            "__listen",
            "__recall",
            "__search_memories",
            "__get_social_state",
            "__get_current_joint_focus",
            "__look_around",
            "__camera_info",
        )
    )


def _sensor_coupling(candidate: WorkspaceCandidate) -> float:
    if candidate.source_mode == SourceMode.LIVE:
        return 1.0 if candidate.source in {
            CandidateSource.SENSOR,
            CandidateSource.TOOL_RESULT,
            CandidateSource.INTEROCEPTION,
        } else 0.75
    if candidate.source_mode == SourceMode.MIXED:
        return 0.5
    if candidate.source_mode == SourceMode.INFERRED:
        return 0.3
    if candidate.source_mode == SourceMode.REMEMBERED:
        return 0.1
    return 0.0


def _source_recency(source_mode: SourceMode) -> float:
    return {
        SourceMode.LIVE: 1.0,
        SourceMode.MIXED: 0.7,
        SourceMode.INFERRED: 0.6,
        SourceMode.REMEMBERED: 0.35,
        SourceMode.IMAGINED: 0.2,
    }[source_mode]


def _tool_modality(tool_name: str) -> str:
    name = tool_name.lower()
    if any(value in name for value in ("see", "camera", "look")):
        return "visual"
    if any(value in name for value in ("listen", "audio", "say", "tts")):
        return "auditory"
    if any(value in name for value in ("social", "post", "notify", "send")):
        return "social"
    return "internal"


def _mean_mismatch(values: dict[str, float]) -> float:
    return (
        sum(_clamp01(value) for value in values.values()) / len(values)
        if values
        else 0.5
    )


def _json_loads(raw: str | None, default):
    try:
        return json.loads(raw or "")
    except (TypeError, json.JSONDecodeError):
        return default


def _seconds_between(start: str, end: str) -> float:
    try:
        left = datetime.fromisoformat(start.replace("Z", "+00:00"))
        right = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return (right - left).total_seconds()
    except ValueError:
        return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
