"""Typed enacted field and SQLite persistence.

The compact surface is rendered from typed state. The XML rendering is never
used as the database source of truth.
"""

from __future__ import annotations

import json
import secrets
from enum import StrEnum
from html import escape
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from social_core.db import SocialDB
from social_core.time import utc_now

from individual_kernel_mcp.workspace import SourceMode


class FieldStatus(StrEnum):
    OPEN = "OPEN"
    COMMITTED = "COMMITTED"
    INVALIDATED = "INVALIDATED"
    CLOSED = "CLOSED"
    ABORTED = "ABORTED"


class TriggerKind(StrEnum):
    SESSION_START = "session_start"
    USER_PROMPT = "user_prompt"
    HEARTBEAT = "heartbeat"
    PERCEPTION = "perception"
    TOOL_RESULT = "tool_result"
    AUTONOMOUS = "autonomous"


class SelfOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controllable_channels: list[str] = Field(default_factory=list)
    camera_pose_ref: str | None = None
    viewpoint_ref: str | None = None
    body_resource_refs: list[str] = Field(default_factory=list)
    ownership_boundary_refs: list[str] = Field(default_factory=list)


class Protention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_content_ref: str | None = None
    summary: str = ""
    action_ref: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class InteroceptionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    controllability: float = Field(default=0.0, ge=0.0, le=1.0)
    need_vector: dict[str, float] = Field(default_factory=dict)
    resource_refs: list[str] = Field(default_factory=list)
    negative_exposure_seconds: float = Field(default=0.0, ge=0.0)


class PrecisionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extero: float = Field(default=0.5, ge=0.0, le=1.0)
    intero: float = Field(default=0.5, ge=0.0, le=1.0)
    mnemonic: float = Field(default=0.5, ge=0.0, le=1.0)
    social: float = Field(default=0.5, ge=0.0, le=1.0)
    self_model: float = Field(default=0.5, ge=0.0, le=1.0)


class AgencyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pending_intention_ref: str | None = None
    ownership_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_outcome_ref: str | None = None
    registered_intention: bool = False


class EnactedField(BaseModel):
    """The one self-world hypothesis controlling a committed tick."""

    model_config = ConfigDict(extra="forbid")

    field_id: str = Field(default_factory=lambda: f"field_{secrets.token_urlsafe(12)}")
    owner_id: str = "self"
    tick_id: str
    continuity_token: str
    previous_field_id: str | None = None
    status: FieldStatus = FieldStatus.OPEN
    trigger_kind: TriggerKind
    person_id: str | None = None
    session_id: str | None = None
    started_at: str = Field(default_factory=utc_now)
    committed_at: str | None = None
    closed_at: str | None = None

    self_origin: SelfOrigin = Field(default_factory=SelfOrigin)
    reality_mode: SourceMode = SourceMode.INFERRED
    reality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    focal_content_ref: str | None = None
    focal_summary: str = Field(default="", max_length=400)
    peripheral_content_refs: list[str] = Field(default_factory=list)
    retention_refs: list[str] = Field(default_factory=list)
    protention: Protention = Field(default_factory=Protention)
    interoception: InteroceptionState = Field(default_factory=InteroceptionState)
    precision: PrecisionState = Field(default_factory=PrecisionState)
    affordance_refs: list[str] = Field(default_factory=list)
    pending_intention_ref: str | None = None
    agency_state: AgencyState = Field(default_factory=AgencyState)
    attention_schema_ref: str | None = None
    attention_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_next_focus: str | None = None
    hor_refs: list[str] = Field(default_factory=list)
    quality_signature_ref: str | None = None

    phenomenal_surface: str = ""
    epistemic_trace: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    def render_surface(self, *, max_chars: int = 1500) -> str:
        """Render the compact agent-facing read surface."""

        pending = self.pending_intention_ref or "none"
        focus_ref = self.focal_content_ref or "none"
        focus = self.focal_summary or focus_ref
        periphery = ", ".join(self.peripheral_content_refs[:5]) or "none"
        retention = ", ".join(self.retention_refs[:4]) or "none"
        affordances = ", ".join(self.affordance_refs[:5]) or "none"
        predicted = self.predicted_next_focus or self.protention.expected_content_ref or "unknown"
        text = (
            f'<current_field version="1" field_id="{escape(self.field_id)}" '
            f'tick_id="{escape(self.tick_id)}" continuity="{escape(self.continuity_token)}">\n'
            f"  <mode>{self.reality_mode.value}</mode>\n"
            f'  <reality score="{self.reality_score:.3f}" />\n'
            f'  <focus ref="{escape(focus_ref)}">{escape(focus)}</focus>\n'
            f"  <periphery>{escape(periphery)}</periphery>\n"
            f"  <retention>{escape(retention)}</retention>\n"
            f"  <protention>{escape(self.protention.summary or predicted)}</protention>\n"
            f'  <body valence="{self.interoception.valence:.3f}" '
            f'arousal="{self.interoception.arousal:.3f}" '
            f'uncertainty="{self.interoception.uncertainty:.3f}" '
            f'control="{self.interoception.controllability:.3f}" />\n'
            f'  <attention intensity="{self.attention_intensity:.3f}" '
            f'predicted_next="{escape(predicted)}" />\n'
            f'  <agency pending_intention="{escape(pending)}" '
            f'ownership="{self.agency_state.ownership_score:.3f}" />\n'
            f"  <affordances>{escape(affordances)}</affordances>\n"
            f"  <source_constraints>mode={self.reality_mode.value}; "
            f"uncertainty={self.interoception.uncertainty:.3f}; "
            "do not upgrade remembered or imagined content to live</source_constraints>\n"
            "</current_field>"
        )
        if len(text) <= max_chars:
            return text
        # Keep the closing tag and the provenance constraint when a long ref list
        # consumes the budget.
        suffix = "\n</current_field>"
        return text[: max(0, max_chars - len(suffix))] + suffix


class RuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_id: str
    continuity_token: str
    current_field_id: str | None = None
    open_tick_id: str | None = None
    state: str = "ACTIVE"
    last_trigger_kind: str | None = None
    last_recovery_at: str | None = None
    updated_at: str
    # The Claude Code session that speaks for this owner, claimed at its
    # SessionStart. `runtime_state` builds this model straight from the row and
    # the config forbids extras, so this field has to land BEFORE the migration
    # that adds the column -- otherwise every path that reads runtime state
    # raises at once, the gate fails closed, and the runtime can no longer be
    # repaired from inside. That is not hypothetical; it happened while writing
    # this change in the other order.
    session_id: str | None = None


class EnactedFieldStore:
    """Persistence and state transitions for enacted fields."""

    def __init__(self, db: SocialDB) -> None:
        self._db = db

    def create_open(self, field: EnactedField) -> EnactedField:
        if field.status != FieldStatus.OPEN:
            raise ValueError("new fields must begin in OPEN status")
        rendered = field.render_surface()
        stored = field.model_copy(
            update={"phenomenal_surface": rendered, "updated_at": utc_now()}
        )
        self._insert(stored)
        self._upsert_runtime(
            owner_id=stored.owner_id,
            continuity_token=stored.continuity_token,
            current_field_id=self._current_field_id(stored.owner_id),
            open_tick_id=stored.tick_id,
            last_trigger_kind=stored.trigger_kind.value,
        )
        return stored

    def commit(self, field_id: str) -> EnactedField:
        field = self.get(field_id)
        if field is None:
            raise ValueError(f"unknown field {field_id!r}")
        if field.status != FieldStatus.OPEN:
            raise ValueError(f"field {field_id!r} is not OPEN")
        now = utc_now()
        with self._db.transaction() as connection:
            connection.execute(
                """
                UPDATE enacted_fields
                SET status = 'INVALIDATED', updated_at = ?
                WHERE owner_id = ? AND status = 'COMMITTED' AND field_id != ?
                """,
                (now, field.owner_id, field_id),
            )
            connection.execute(
                """
                UPDATE enacted_fields
                SET status = 'COMMITTED', committed_at = ?, updated_at = ?
                WHERE field_id = ? AND status = 'OPEN'
                """,
                (now, now, field_id),
            )
            connection.execute(
                """
                INSERT INTO field_runtime_state(
                    owner_id, continuity_token, current_field_id, open_tick_id,
                    state, last_trigger_kind, updated_at
                ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    continuity_token = excluded.continuity_token,
                    current_field_id = excluded.current_field_id,
                    open_tick_id = excluded.open_tick_id,
                    last_trigger_kind = excluded.last_trigger_kind,
                    updated_at = excluded.updated_at
                """,
                (
                    field.owner_id,
                    field.continuity_token,
                    field_id,
                    field.tick_id,
                    field.trigger_kind.value,
                    now,
                ),
            )
        committed = self.get(field_id)
        if committed is None:
            raise RuntimeError("committed field could not be reloaded")
        return committed

    def update(self, field: EnactedField) -> EnactedField:
        rendered = field.render_surface()
        updated = field.model_copy(
            update={"phenomenal_surface": rendered, "updated_at": utc_now()}
        )
        self._db.execute(
            """
            UPDATE enacted_fields SET
                self_origin_json = ?, reality_mode = ?, reality_score = ?,
                focal_content_ref = ?, peripheral_content_refs_json = ?,
                retention_refs_json = ?, protention_json = ?,
                interoception_json = ?, precision_json = ?,
                affordance_refs_json = ?, pending_intention_ref = ?,
                agency_state_json = ?, attention_schema_ref = ?, hor_refs_json = ?,
                quality_signature_ref = ?, phenomenal_surface = ?,
                epistemic_trace_json = ?, updated_at = ?
            WHERE field_id = ?
            """,
            (
                _dump(updated.self_origin),
                updated.reality_mode.value,
                updated.reality_score,
                updated.focal_content_ref,
                json.dumps(updated.peripheral_content_refs, ensure_ascii=False),
                json.dumps(updated.retention_refs, ensure_ascii=False),
                _dump(updated.protention),
                _dump(updated.interoception),
                _dump(updated.precision),
                json.dumps(updated.affordance_refs, ensure_ascii=False),
                updated.pending_intention_ref,
                _dump(updated.agency_state),
                updated.attention_schema_ref,
                json.dumps(updated.hor_refs, ensure_ascii=False),
                updated.quality_signature_ref,
                updated.phenomenal_surface,
                json.dumps(updated.epistemic_trace, ensure_ascii=False, sort_keys=True),
                updated.updated_at,
                updated.field_id,
            ),
        )
        return updated

    def get(self, field_id: str) -> EnactedField | None:
        row = self._db.fetchone("SELECT * FROM enacted_fields WHERE field_id = ?", (field_id,))
        return None if row is None else self._row_to_field(row)

    def get_by_tick(self, tick_id: str) -> EnactedField | None:
        row = self._db.fetchone("SELECT * FROM enacted_fields WHERE tick_id = ?", (tick_id,))
        return None if row is None else self._row_to_field(row)

    def get_current(self, owner_id: str = "self") -> EnactedField | None:
        row = self._db.fetchone(
            """
            SELECT ef.* FROM field_runtime_state fr
            JOIN enacted_fields ef ON ef.field_id = fr.current_field_id
            WHERE fr.owner_id = ? AND ef.status = 'COMMITTED'
            """,
            (owner_id,),
        )
        return None if row is None else self._row_to_field(row)

    def query(
        self,
        *,
        owner_id: str | None = None,
        status: FieldStatus | None = None,
        trigger_kind: TriggerKind | None = None,
        limit: int = 50,
    ) -> list[EnactedField]:
        clauses: list[str] = []
        params: list[object] = []
        if owner_id is not None:
            clauses.append("owner_id = ?")
            params.append(owner_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if trigger_kind is not None:
            clauses.append("trigger_kind = ?")
            params.append(trigger_kind.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))
        rows = self._db.fetchall(
            f"""
            SELECT * FROM enacted_fields {where}
            ORDER BY started_at DESC, created_at DESC LIMIT ?
            """,
            tuple(params),
        )
        return [self._row_to_field(row) for row in rows]

    def invalidate(self, field_id: str, *, reason: str) -> EnactedField:
        field = self.get(field_id)
        if field is None:
            raise ValueError(f"unknown field {field_id!r}")
        trace = dict(field.epistemic_trace)
        trace.setdefault("invalidations", []).append({"reason": reason, "ts": utc_now()})
        now = utc_now()
        with self._db.transaction() as connection:
            connection.execute(
                """
                UPDATE enacted_fields
                SET status = 'INVALIDATED', epistemic_trace_json = ?, updated_at = ?
                WHERE field_id = ? AND status IN ('OPEN', 'COMMITTED')
                """,
                (json.dumps(trace, ensure_ascii=False, sort_keys=True), now, field_id),
            )
            connection.execute(
                """
                UPDATE field_runtime_state
                SET current_field_id = CASE
                        WHEN current_field_id = ? THEN NULL
                        ELSE current_field_id
                    END,
                    updated_at = ?
                WHERE owner_id = ?
                """,
                (field_id, now, field.owner_id),
            )
        invalidated = self.get(field_id)
        if invalidated is None:
            raise RuntimeError("invalidated field could not be reloaded")
        return invalidated

    def close(self, field_id: str, *, output_summary: str | None = None) -> EnactedField:
        field = self.get(field_id)
        if field is None:
            raise ValueError(f"unknown field {field_id!r}")
        trace = dict(field.epistemic_trace)
        if output_summary:
            trace["output_summary"] = output_summary[:1000]
        now = utc_now()
        with self._db.transaction() as connection:
            connection.execute(
                """
                UPDATE enacted_fields
                SET status = 'CLOSED', closed_at = ?, epistemic_trace_json = ?, updated_at = ?
                WHERE field_id = ? AND status != 'ABORTED'
                """,
                (
                    now,
                    json.dumps(trace, ensure_ascii=False, sort_keys=True),
                    now,
                    field_id,
                ),
            )
            connection.execute(
                """
                UPDATE field_runtime_state
                SET current_field_id = CASE
                        WHEN current_field_id = ? THEN NULL
                        ELSE current_field_id
                    END,
                    open_tick_id = CASE
                        WHEN open_tick_id = ? THEN NULL
                        ELSE open_tick_id
                    END,
                    updated_at = ?
                WHERE owner_id = ?
                """,
                (field_id, field.tick_id, now, field.owner_id),
            )
        closed = self.get(field_id)
        if closed is None:
            raise RuntimeError("closed field could not be reloaded")
        return closed

    def runtime_state(self, owner_id: str = "self") -> RuntimeState | None:
        row = self._db.fetchone(
            "SELECT * FROM field_runtime_state WHERE owner_id = ?", (owner_id,)
        )
        if row is None:
            return None
        return RuntimeState(**dict(row))

    def claim_session(
        self,
        *,
        owner_id: str,
        continuity_token: str,
        session_id: str,
        current_field_id: str | None = None,
        open_tick_id: str | None = None,
        state: str = "ACTIVE",
    ) -> RuntimeState:
        """Record the Claude Code session that speaks for this owner.

        Deliberately last-write-wins: a genuinely new top-level session must be
        able to take over `self` from the one before it, or every restart would
        strand the agent under a derived owner. Only SessionStart calls this, and
        subagents do not fire SessionStart, so the takeover cannot happen mid-run.
        """
        self._upsert_runtime(
            owner_id=owner_id,
            continuity_token=continuity_token,
            current_field_id=current_field_id,
            open_tick_id=open_tick_id,
            state=state,
            session_id=session_id,
        )
        claimed = self.runtime_state(owner_id)
        assert claimed is not None
        return claimed

    def pause(self, owner_id: str = "self") -> RuntimeState:
        state = self.runtime_state(owner_id)
        token = state.continuity_token if state else f"cont_{secrets.token_urlsafe(16)}"
        self._upsert_runtime(
            owner_id=owner_id,
            continuity_token=token,
            current_field_id=state.current_field_id if state else None,
            open_tick_id=state.open_tick_id if state else None,
            state="PAUSED",
        )
        paused = self.runtime_state(owner_id)
        assert paused is not None
        return paused

    def resume(self, owner_id: str = "self") -> RuntimeState:
        state = self.runtime_state(owner_id)
        token = state.continuity_token if state else f"cont_{secrets.token_urlsafe(16)}"
        self._upsert_runtime(
            owner_id=owner_id,
            continuity_token=token,
            current_field_id=state.current_field_id if state else None,
            open_tick_id=state.open_tick_id if state else None,
            state="ACTIVE",
        )
        resumed = self.runtime_state(owner_id)
        assert resumed is not None
        return resumed

    def _insert(self, field: EnactedField) -> None:
        self._db.execute(
            """
            INSERT INTO enacted_fields(
                field_id, owner_id, tick_id, continuity_token, previous_field_id,
                status, trigger_kind, person_id, session_id, started_at,
                committed_at, closed_at, self_origin_json, reality_mode,
                reality_score, focal_content_ref, peripheral_content_refs_json,
                retention_refs_json, protention_json, interoception_json,
                precision_json, affordance_refs_json, pending_intention_ref,
                agency_state_json, attention_schema_ref, hor_refs_json,
                quality_signature_ref, phenomenal_surface, epistemic_trace_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                field.field_id,
                field.owner_id,
                field.tick_id,
                field.continuity_token,
                field.previous_field_id,
                field.status.value,
                field.trigger_kind.value,
                field.person_id,
                field.session_id,
                field.started_at,
                field.committed_at,
                field.closed_at,
                _dump(field.self_origin),
                field.reality_mode.value,
                field.reality_score,
                field.focal_content_ref,
                json.dumps(field.peripheral_content_refs, ensure_ascii=False),
                json.dumps(field.retention_refs, ensure_ascii=False),
                _dump(field.protention),
                _dump(field.interoception),
                _dump(field.precision),
                json.dumps(field.affordance_refs, ensure_ascii=False),
                field.pending_intention_ref,
                _dump(field.agency_state),
                field.attention_schema_ref,
                json.dumps(field.hor_refs, ensure_ascii=False),
                field.quality_signature_ref,
                field.phenomenal_surface,
                json.dumps(field.epistemic_trace, ensure_ascii=False, sort_keys=True),
                field.created_at,
                field.updated_at,
            ),
        )

    def _upsert_runtime(
        self,
        *,
        owner_id: str,
        continuity_token: str,
        current_field_id: str | None,
        open_tick_id: str | None,
        state: str = "ACTIVE",
        last_trigger_kind: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._db.execute(
            """
            INSERT INTO field_runtime_state(
                owner_id, continuity_token, current_field_id, open_tick_id,
                state, last_trigger_kind, session_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id) DO UPDATE SET
                continuity_token = excluded.continuity_token,
                current_field_id = excluded.current_field_id,
                open_tick_id = excluded.open_tick_id,
                state = excluded.state,
                last_trigger_kind = COALESCE(excluded.last_trigger_kind, last_trigger_kind),
                -- COALESCE, so only a caller that actually knows the session
                -- writes it. Every other upsert (commit, pause, resume) passes
                -- None and must leave the claim standing.
                session_id = COALESCE(excluded.session_id, session_id),
                updated_at = excluded.updated_at
            """,
            (
                owner_id,
                continuity_token,
                current_field_id,
                open_tick_id,
                state,
                last_trigger_kind,
                session_id,
                utc_now(),
            ),
        )

    def _current_field_id(self, owner_id: str) -> str | None:
        current = self.get_current(owner_id)
        return current.field_id if current else None

    @staticmethod
    def _row_to_field(row) -> EnactedField:
        trace = _load(row["epistemic_trace_json"], {})
        focal_summary = str(trace.get("focal_summary") or "")
        attention = trace.get("competition") or {}
        return EnactedField(
            field_id=row["field_id"],
            owner_id=row["owner_id"],
            tick_id=row["tick_id"],
            continuity_token=row["continuity_token"],
            previous_field_id=row["previous_field_id"],
            status=row["status"],
            trigger_kind=row["trigger_kind"],
            person_id=row["person_id"],
            session_id=row["session_id"],
            started_at=row["started_at"],
            committed_at=row["committed_at"],
            closed_at=row["closed_at"],
            self_origin=SelfOrigin.model_validate(_load(row["self_origin_json"], {})),
            reality_mode=row["reality_mode"],
            reality_score=float(row["reality_score"]),
            focal_content_ref=row["focal_content_ref"],
            focal_summary=focal_summary,
            peripheral_content_refs=_load(row["peripheral_content_refs_json"], []),
            retention_refs=_load(row["retention_refs_json"], []),
            protention=Protention.model_validate(_load(row["protention_json"], {})),
            interoception=InteroceptionState.model_validate(
                _load(row["interoception_json"], {})
            ),
            precision=PrecisionState.model_validate(_load(row["precision_json"], {})),
            affordance_refs=_load(row["affordance_refs_json"], []),
            pending_intention_ref=row["pending_intention_ref"],
            agency_state=AgencyState.model_validate(
                _load(row["agency_state_json"], {})
            ),
            attention_schema_ref=row["attention_schema_ref"],
            attention_intensity=float(attention.get("attention_intensity", 0.0)),
            predicted_next_focus=trace.get("predicted_next_focus"),
            hor_refs=_load(row["hor_refs_json"], []),
            quality_signature_ref=row["quality_signature_ref"],
            phenomenal_surface=row["phenomenal_surface"],
            epistemic_trace=trace,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def _load(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default
