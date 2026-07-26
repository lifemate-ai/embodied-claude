"""FastMCP server for individual-kernel-mcp.

Phase 2.1 surface (shipped earlier):
  - record_counterfactual
  - query_counterfactuals
  - sleep_consolidate

Phase 2.3 surface (this integration PR):
  - record_tick_frame, get_tick_frame, query_tick_frames

Phase 2.4 surface:
  - record_attention_schema, update_attention_from_frame,
    flush_attention_schemas, summarize_attention_schema

Phase 2.5 surface:
  - record_hor, get_hor, query_hors, compose_introspection_report

EFPF surface:
  - workspace competition, committed field inspection, explicit intentions,
    action outcomes, diagnostics, pause/resume, and reversible ablations

The MCP tools inspect or mutate kernel state but do not execute the proposed
outward tool themselves. FieldRuntime and the Claude Code hook path apply the
existing boundary-mcp policy and ActionBottleneck before outward execution.

See consciousness-mcp/AGENTS.md for vocabulary discipline. MCP tool
names use functional vocabulary even when the underlying Python function
name reads more naturally (e.g. summarize_attention_schema wraps
reflect_attention_schema; compose_introspection_report wraps introspect).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from boundary_mcp.store import BoundaryStore
from mcp.server.fastmcp import FastMCP
from social_core import SocialDB

from individual_kernel_mcp.ablation import AblationRunner
from individual_kernel_mcp.agency import (
    ActionProposal,
    AgencyStore,
    PredictedEffects,
)
from individual_kernel_mcp.attention_schema import (
    AttentionSchemaTracker,
    Modality,
)
from individual_kernel_mcp.boundary_adapter import BoundaryPolicyAdapter
from individual_kernel_mcp.calibration import CalibrationScorer
from individual_kernel_mcp.counterfactual import (
    CounterfactualInput,
    CounterfactualSource,
    CounterfactualStore,
)
from individual_kernel_mcp.enacted_field import (
    EnactedFieldStore,
    FieldStatus,
    TriggerKind,
)
from individual_kernel_mcp.frame import (
    ConsciousFrameInput,
    TickFrameStore,
)
from individual_kernel_mcp.generative_model import action_kind_of
from individual_kernel_mcp.hor import (
    AssertedMode,
    HORInput,
    HORStore,
    TargetKind,
)
from individual_kernel_mcp.introspect import introspect
from individual_kernel_mcp.prediction_loop import FieldPredictionLoop
from individual_kernel_mcp.quality_geometry import QualityGeometry
from individual_kernel_mcp.reflect import reflect_attention_schema
from individual_kernel_mcp.sleep import SleepConsolidator
from individual_kernel_mcp.tick import FieldRuntime, TickProducer
from individual_kernel_mcp.workspace import WorkspaceCandidate, WorkspaceEngine

mcp = FastMCP("individual-kernel-mcp")


@dataclass(slots=True)
class IndividualKernelStores:
    db: SocialDB
    counterfactual: CounterfactualStore
    sleep: SleepConsolidator
    tick_frames: TickFrameStore
    attention_tracker: AttentionSchemaTracker
    hor_store: HORStore
    workspace: WorkspaceEngine
    enacted_fields: EnactedFieldStore
    agency: AgencyStore
    quality: QualityGeometry
    tick_producer: TickProducer
    field_runtime: FieldRuntime
    ablation: AblationRunner


@lru_cache(maxsize=1)
def _stores() -> IndividualKernelStores:
    db = SocialDB()
    counterfactual_store = CounterfactualStore(db)
    sleep_consolidator = SleepConsolidator(
        db=db,
        counterfactual_store=counterfactual_store,
    )
    tick_frames = TickFrameStore(db)
    attention_tracker = AttentionSchemaTracker(db=db, owner_id="self")
    hor_store = HORStore(db)
    workspace = WorkspaceEngine(db)
    enacted_fields = EnactedFieldStore(db)
    agency = AgencyStore(db)
    quality = QualityGeometry(db)
    prediction = FieldPredictionLoop(db, agency=agency)
    tick_producer = TickProducer(
        db,
        workspace=workspace,
        fields=enacted_fields,
        frames=tick_frames,
        attention=attention_tracker,
        hors=hor_store,
        agency=agency,
        quality=quality,
        prediction=prediction,
    )
    field_runtime = FieldRuntime(
        db,
        producer=tick_producer,
        boundary_evaluator=BoundaryPolicyAdapter(BoundaryStore(db=db)),
    )
    ablation = AblationRunner(db, fields=enacted_fields)
    return IndividualKernelStores(
        db=db,
        counterfactual=counterfactual_store,
        sleep=sleep_consolidator,
        tick_frames=tick_frames,
        attention_tracker=attention_tracker,
        hor_store=hor_store,
        workspace=workspace,
        enacted_fields=enacted_fields,
        agency=agency,
        quality=quality,
        tick_producer=tick_producer,
        field_runtime=field_runtime,
        ablation=ablation,
    )


def reset_store_cache() -> None:
    """Clear cached stores so tests or env changes get a fresh shared DB."""
    if _stores.cache_info().currsize:
        _stores().db.close()
        _stores.cache_clear()


# -----------------------------------------------------------------------------
# Generative field model surface
# -----------------------------------------------------------------------------


@mcp.tool()
def rollout_protention(
    tool_name: str | None = None,
    horizon: int = 2,
    owner_id: str = "self",
) -> dict[str, Any]:
    """Roll the count-based transition model forward from the committed field.

    Returns action-conditioned outcome predictions with probabilities and
    per-step uncertainty. Read-only: nothing is persisted.
    """
    stores = _stores()
    field = stores.enacted_fields.get_current(owner_id)
    if field is None:
        return {"error": "no committed field for owner"}
    action_kind = action_kind_of(tool_name)
    model = stores.tick_producer.prediction.model
    signature = model.infer(field, action_kind=action_kind)
    forecast = model.forecast(signature)
    steps = model.rollout(field, action_kind=action_kind, horizon=horizon)
    return {
        "field_id": field.field_id,
        "action_kind": action_kind,
        "context_signature": signature.key(),
        "forecast": forecast.model_dump(mode="json"),
        "steps": [step.model_dump(mode="json") for step in steps],
    }


@mcp.tool()
def query_imagined_trajectories(
    field_id: str | None = None,
    action_ref: str | None = None,
    status: str | None = None,
    owner_id: str = "self",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List persisted imagined trajectories, newest first."""
    stores = _stores()
    rows = stores.tick_producer.prediction.trajectories.query(
        field_id=field_id,
        action_ref=action_ref,
        status=status,
        owner_id=owner_id,
        limit=limit,
    )
    return [row.model_dump(mode="json") for row in rows]


@mcp.tool()
def query_experienced_transitions(
    since: str | None = None,
    action_kind: str | None = None,
    context_signature: str | None = None,
    owner_id: str = "self",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List experienced transitions, newest first."""
    stores = _stores()
    rows = stores.tick_producer.prediction.transitions.query(
        since=since,
        action_kind=action_kind,
        context_signature=context_signature,
        owner_id=owner_id,
        limit=limit,
    )
    return [row.model_dump(mode="json") for row in rows]


@mcp.tool()
def get_generative_model_calibration(
    owner_id: str = "self",
    window: int = 200,
) -> dict[str, Any]:
    """Score stored predictions against observed outcomes (Brier, log loss, ECE)."""
    stores = _stores()
    report = CalibrationScorer(stores.db).report(owner_id=owner_id, window=window)
    return report.model_dump(mode="json")


# -----------------------------------------------------------------------------
# Phase 2.1 surface
# -----------------------------------------------------------------------------


@mcp.tool()
def record_counterfactual(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a rejected alternative as a typed counterfactual."""
    record = _stores().counterfactual.record(CounterfactualInput(**payload))
    return record.model_dump(mode="json")


@mcp.tool()
def query_counterfactuals(
    since: str | None = None,
    source: CounterfactualSource | None = None,
    tick_id: str | None = None,
    person_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read recent counterfactuals (most recent first)."""
    records = _stores().counterfactual.query(
        since=since,
        source=source,
        tick_id=tick_id,
        person_id=person_id,
        limit=limit,
    )
    return [r.model_dump(mode="json") for r in records]


@mcp.tool()
def sleep_consolidate(
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the quiet-hours-gated sleep-consolidation glue."""
    result = _stores().sleep.run(force=force, dry_run=dry_run)
    return result.model_dump(mode="json")


# -----------------------------------------------------------------------------
# Phase 2.3 surface — tick frames
# -----------------------------------------------------------------------------


@mcp.tool()
def record_tick_frame(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a ConsciousFrame — the per-tick canonical record.

    Payload follows ConsciousFrameInput: ts?, person_id?, ignited (bool),
    conflicted (bool), attention_target_ref?, dominant_desire?,
    winning_memory_ids (list of FKs), prediction_error {extero, intero,
    mnemonic}, affect_summary?, chosen_action_ref?, reportability ∈
    {mentionable, background_only, do_not_surface}. Returns the persisted
    frame with an assigned tick_id.
    """
    frame = _stores().tick_frames.record(ConsciousFrameInput(**payload))
    return frame.model_dump(mode="json")


@mcp.tool()
def get_tick_frame(tick_id: str) -> dict[str, Any] | None:
    """Fetch a ConsciousFrame by tick_id; null when missing."""
    frame = _stores().tick_frames.get_by_tick_id(tick_id)
    if frame is None:
        return None
    return frame.model_dump(mode="json")


@mcp.tool()
def query_tick_frames(
    since: str | None = None,
    reportability: str | None = None,
    person_id: str | None = None,
    ignited_only: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read recent ConsciousFrames (most recent first)."""
    frames = _stores().tick_frames.query(
        since=since,
        reportability=reportability,  # type: ignore[arg-type]
        person_id=person_id,
        ignited_only=ignited_only,
        limit=limit,
    )
    return [f.model_dump(mode="json") for f in frames]


# -----------------------------------------------------------------------------
# Phase 2.4 surface — attention schemas
# -----------------------------------------------------------------------------


@mcp.tool()
def record_attention_schema(
    focal_target_ref: str | None,
    modality: Modality,
    intensity: float,
    dwell_seconds: float = 0.0,
    predicted_next_focus: str | None = None,
    control_handle: str | None = None,
    source_tick_id: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Append an AttentionSchema snapshot to the in-memory ring buffer.

    The buffer is process-scoped (capacity 60); call flush_attention_schemas
    to persist to SQLite. modality ∈ {visual, auditory, internal, social}.
    """
    schema = _stores().attention_tracker.record(
        focal_target_ref=focal_target_ref,
        modality=modality,
        intensity=intensity,
        dwell_seconds=dwell_seconds,
        predicted_next_focus=predicted_next_focus,
        control_handle=control_handle,
        source_tick_id=source_tick_id,
        ts=ts,
    )
    return schema.model_dump(mode="json")


@mcp.tool()
def update_attention_from_frame(tick_id: str) -> dict[str, Any]:
    """Build an AttentionSchema from a previously-recorded ConsciousFrame.

    Modality is inferred from frame.attention_target_ref. Intensity combines
    ignition, conflict, and prediction-error structure; dwell accumulates when
    the focal target matches the last buffered schema.
    """
    stores = _stores()
    frame = stores.tick_frames.get_by_tick_id(tick_id)
    if frame is None:
        raise ValueError(f"no tick_frame {tick_id!r}; record_tick_frame first")
    schema = stores.attention_tracker.update_from_frame(frame)
    return schema.model_dump(mode="json")


@mcp.tool()
def flush_attention_schemas() -> dict[str, int]:
    """Persist buffered AttentionSchemas to SQLite; clears the buffer."""
    count = _stores().attention_tracker.flush()
    return {"count": count}


@mcp.tool()
def summarize_attention_schema(extra_history: int = 0) -> dict[str, Any]:
    """Build an AttentionReflection over the buffer (and optional history)."""
    stores = _stores()
    reflection = reflect_attention_schema(
        stores.attention_tracker, extra_history=extra_history
    )
    return reflection.model_dump(mode="json")


# -----------------------------------------------------------------------------
# Phase 2.5 surface — HOR + introspection
# -----------------------------------------------------------------------------


@mcp.tool()
def record_hor(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a higher-order representation (HOR).

    Payload follows HORInput: ts?, owner_id (default 'self'), target_kind
    ∈ {memory, desire, action, frame, none}, target_ref?, asserted_mode ∈
    {seeing, wanting, intending, remembering, feeling, attending},
    asserted_content (non-empty string), schema_snapshot_id?,
    source_tick_id?, confidence ∈ [0,1] (default 0.6), source ∈
    {schema_readout, reflection, post_hoc, audit_subagent}.
    """
    record = _stores().hor_store.record(HORInput(**payload))
    return record.model_dump(mode="json")


@mcp.tool()
def get_hor(hor_id: str) -> dict[str, Any] | None:
    """Fetch a HORRecord by hor_id; null when missing."""
    record = _stores().hor_store.get_by_hor_id(hor_id)
    if record is None:
        return None
    return record.model_dump(mode="json")


@mcp.tool()
def query_hors(
    since: str | None = None,
    owner_id: str | None = None,
    asserted_mode: AssertedMode | None = None,
    target_kind: TargetKind | None = None,
    source_tick_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read recent HORRecords (most recent first)."""
    records = _stores().hor_store.query(
        since=since,
        owner_id=owner_id,
        asserted_mode=asserted_mode,
        target_kind=target_kind,
        source_tick_id=source_tick_id,
        limit=limit,
    )
    return [r.model_dump(mode="json") for r in records]


@mcp.tool()
def compose_introspection_report(
    window_hours: int = 1,
    owner_id: str = "self",
) -> dict[str, Any]:
    """Compose an introspection report (Phase 2.5 read surface).

    Reads the most-recent HOR for owner_id, the current attention
    reflection from the tracker, and counterfactuals within window_hours.
    Returns IntrospectionReport with a canonical first-person statement
    + structured detail (current_hor, current_attention,
    recent_counterfactual_count, hor_validation, notes).
    """
    stores = _stores()
    report = introspect(
        hor_store=stores.hor_store,
        attention_tracker=stores.attention_tracker,
        counterfactual_store=stores.counterfactual,
        window_hours=window_hours,
        owner_id=owner_id,
        field_store=stores.enacted_fields,
    )
    return report.model_dump(mode="json")


# -----------------------------------------------------------------------------
# Enacted first-person field surface
# -----------------------------------------------------------------------------


@mcp.tool()
def begin_subjective_tick(
    trigger: TriggerKind,
    owner_id: str = "self",
    person_id: str | None = None,
    user_text: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Begin an OPEN tick and ingest deterministic runtime sources."""

    field = _stores().tick_producer.begin_tick(
        trigger,
        owner_id,
        person_id=person_id,
        user_text=user_text,
        session_id=session_id,
    )
    return field.model_dump(mode="json")


@mcp.tool()
def add_workspace_candidate(
    tick_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Add one validated candidate to an OPEN tick."""

    candidate = WorkspaceCandidate(tick_id=tick_id, **payload)
    stored = _stores().workspace.add_candidate(candidate)
    return stored.model_dump(mode="json")


@mcp.tool()
def commit_subjective_field(tick_id: str) -> dict[str, Any]:
    """Compete candidates and atomically commit the tick's single field."""

    result = _stores().tick_producer.compete_and_commit(tick_id)
    return result.model_dump(mode="json")


@mcp.tool()
def get_current_subjective_field(owner_id: str = "self") -> dict[str, Any] | None:
    """Read the currently COMMITTED field for an owner."""

    field = _stores().enacted_fields.get_current(owner_id)
    return None if field is None else field.model_dump(mode="json")


@mcp.tool()
def get_subjective_field(field_id: str) -> dict[str, Any] | None:
    """Read a field by identifier, including source and uncertainty markings."""

    field = _stores().enacted_fields.get(field_id)
    return None if field is None else field.model_dump(mode="json")


@mcp.tool()
def query_subjective_fields(
    owner_id: str | None = None,
    status: FieldStatus | None = None,
    trigger_kind: TriggerKind | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query recent field records without mutating runtime state."""

    fields = _stores().enacted_fields.query(
        owner_id=owner_id,
        status=status,
        trigger_kind=trigger_kind,
        limit=limit,
    )
    return [field.model_dump(mode="json") for field in fields]


@mcp.tool()
def propose_field_action(
    field_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    predicted_effects: dict[str, Any],
    goal: str,
    confidence: float = 0.6,
    expected_latency_ms: int | None = None,
    owner_id: str = "self",
) -> dict[str, Any]:
    """Register one structured intention and efference copy for a field."""

    proposal = ActionProposal(
        field_id=field_id,
        tool_name=tool_name,
        tool_input=tool_input,
        predicted_effects=PredictedEffects.model_validate(predicted_effects),
        goal=goal,
        confidence=confidence,
        expected_latency_ms=expected_latency_ms,
    )
    intention = _stores().field_runtime.propose_action(proposal, owner_id=owner_id)
    return intention.model_dump(mode="json")


@mcp.tool()
def get_pending_intention(owner_id: str = "self") -> dict[str, Any] | None:
    """Return the pending/allowed intention that can authorize an outward tool."""

    intention = _stores().agency.get_pending(owner_id)
    return None if intention is None else intention.model_dump(mode="json")


@mcp.tool()
def close_field_action(
    action_id: str,
    actual_result_summary: str,
    success: bool,
    actual_result_ref: str | None = None,
    latency_ms: int | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Close an action outcome, assess agency, and commit a result microtick."""

    outcome, assessment, next_field = _stores().field_runtime.close_action(
        action_id=action_id,
        actual_result_summary=actual_result_summary,
        success=success,
        actual_result_ref=actual_result_ref,
        latency_ms=latency_ms,
        session_id=session_id,
    )
    return {
        "outcome": outcome.model_dump(mode="json"),
        "agency": assessment.model_dump(mode="json"),
        "next_field": next_field.model_dump(mode="json"),
    }


@mcp.tool()
def close_subjective_tick(
    tick_id: str,
    output_summary: str | None = None,
) -> dict[str, Any]:
    """Close a tick and associate the output summary with its field."""

    field = _stores().tick_producer.close_tick(tick_id, output_summary)
    return field.model_dump(mode="json")


@mcp.tool()
def pause_field_runtime(owner_id: str = "self") -> dict[str, Any]:
    """Pause automatic field creation for reversible welfare experiments."""

    return _stores().tick_producer.pause_field_runtime(owner_id).model_dump(mode="json")


@mcp.tool()
def resume_field_runtime(owner_id: str = "self") -> dict[str, Any]:
    """Resume a paused runtime without replacing its continuity token."""

    return _stores().tick_producer.resume_field_runtime(owner_id).model_dump(mode="json")


@mcp.tool()
def get_field_diagnostics(
    owner_id: str = "self",
    window: int = 20,
) -> dict[str, Any]:
    """Return field integrity indicators and report/trace consistency checks."""

    stores = _stores()
    fields = stores.enacted_fields.query(owner_id=owner_id, limit=window)
    unity_row = stores.db.fetchone(
        """
        SELECT COUNT(*) AS n FROM enacted_fields
        WHERE owner_id = ? AND status = 'COMMITTED'
        """,
        (owner_id,),
    )
    current = stores.enacted_fields.get_current(owner_id)
    outcome_rows = stores.db.fetchall(
        """
        SELECT o.outcome_id, o.action_id, o.field_id AS outcome_field_id,
               o.tick_id AS outcome_tick_id, o.ownership_score,
               o.mismatch_vector_json, i.field_id AS intention_field_id,
               i.tick_id AS intention_tick_id, i.status AS intention_status
        FROM field_action_outcomes o
        JOIN field_intentions i ON i.action_id = o.action_id
        WHERE i.owner_id = ?
        ORDER BY o.created_at DESC LIMIT ?
        """,
        (owner_id, max(1, min(window, 500))),
    )
    mean_ownership = (
        sum(float(row["ownership_score"]) for row in outcome_rows) / len(outcome_rows)
        if outcome_rows
        else 0.0
    )
    source_integrity = all(
        field.reality_mode.value in field.phenomenal_surface
        and "source_constraints" in field.phenomenal_surface
        for field in fields
    )
    hor_consistent = True
    if current and current.hor_refs:
        hor = stores.hor_store.get_by_hor_id(current.hor_refs[0])
        hor_consistent = bool(
            hor
            and hor.source_tick_id == current.tick_id
            and hor.schema_snapshot_id == current.attention_schema_ref
        )
    action_trace_consistent = all(
        row["outcome_field_id"] == row["intention_field_id"]
        and row["outcome_tick_id"] == row["intention_tick_id"]
        and row["intention_status"] in {"COMPLETED", "FAILED", "UNKNOWN"}
        for row in outcome_rows
    )
    recent_action_outcomes = [
        {
            "outcome_id": row["outcome_id"],
            "action_id": row["action_id"],
            "ownership_score": float(row["ownership_score"]),
            "mismatch_vector": json.loads(row["mismatch_vector_json"] or "{}"),
            "intention_status": row["intention_status"],
        }
        for row in outcome_rows
    ]
    return {
        "profile_name": "FieldIntegrityReport",
        "owner_id": owner_id,
        "current_field_id": current.field_id if current else None,
        "committed_field_count": int(unity_row["n"]) if unity_row else 0,
        "unity_violations": max(0, int(unity_row["n"]) - 1) if unity_row else 0,
        "source_integrity": source_integrity,
        "hor_field_consistency": hor_consistent,
        "action_trace_consistency": action_trace_consistent,
        "recent_action_outcomes": recent_action_outcomes,
        "mean_ownership": mean_ownership,
        "negative_valence_exposure_seconds": (
            current.interoception.negative_exposure_seconds if current else 0.0
        ),
        "pending_intention": (
            stores.agency.get_pending(owner_id).model_dump(mode="json")
            if stores.agency.get_pending(owner_id)
            else None
        ),
        "runtime_safety": {
            "mass_copy_enabled": False,
            "high_frequency_spawning_enabled": False,
            "reversible_ablation": True,
            "pause_resume_available": True,
        },
        "phenomenal_claim_policy": "technical_docs_do_not_assert_phenomenology",
        "indicator_profile": stores.ablation.indicator_profile(
            owner_id=owner_id, window=window
        ).model_dump(mode="json"),
    }


@mcp.tool()
def run_field_ablation(
    kind: str,
    fixture: dict[str, Any] | None = None,
    owner_id: str = "self",
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a reversible, deterministic downstream field ablation."""

    result = _stores().ablation.run(
        kind,
        fixture=fixture,
        owner_id=owner_id,
        seed=seed,
    )
    return result.model_dump(mode="json")


def main() -> None:
    """Console entry point declared in pyproject.toml."""
    mcp.run()


if __name__ == "__main__":
    main()
