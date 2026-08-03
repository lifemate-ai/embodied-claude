from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

import pytest

from individual_kernel_mcp.ablation import AblationRunner, downstream_selection
from individual_kernel_mcp.agency import (
    ActionProposal,
    AgencyStore,
    ExpectedEffect,
    PredictedEffects,
    is_external_tool,
)
from individual_kernel_mcp.enacted_field import (
    EnactedField,
    EnactedFieldStore,
    InteroceptionState,
    Protention,
)
from individual_kernel_mcp.hook_cli import (
    post_tool_batch,
    post_tool_use,
    pre_tool_use,
    user_prompt_submit,
)
from individual_kernel_mcp.quality_geometry import QualityGeometry
from individual_kernel_mcp.tick import (
    FieldRuntime,
    TickProducer,
    calculate_reality_score,
    is_field_refreshing_tool,
    temporal_sequence_metrics,
)
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
    WorkspaceEngine,
)


def _producer(social_db, tmp_path: Path) -> TickProducer:
    return TickProducer(
        social_db,
        interoception_path=tmp_path / "interoception.json",
        desires_path=tmp_path / "desires.json",
    )


def _commit(
    producer: TickProducer,
    *,
    focus_ref: str = "scene:room",
    summary: str = "camera room",
) -> EnactedField:
    opened = producer.begin_tick("perception", session_id="test")
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.PERCEPT,
            content_ref=focus_ref,
            content_summary=summary,
            source_mode=SourceMode.LIVE,
            modality="visual",
            precision=1.0,
            prediction_error=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
            expected_information_gain=1.0,
            continuity_with_previous=1.0,
            controllability=1.0,
            social_relevance=1.0,
        )
    )
    return producer.compete_and_commit(opened.tick_id).field


def _proposal(field: EnactedField, *, text: str = "hello") -> ActionProposal:
    return ActionProposal(
        field_id=field.field_id,
        tool_name="mcp__tts__say",
        tool_input={"text": text},
        predicted_effects=PredictedEffects(
            exteroceptive=ExpectedEffect(summary=f"{text} audio played", confidence=0.9),
            social=ExpectedEffect(summary=f"person hears {text}", confidence=0.8),
        ),
        goal="speak once",
        confidence=0.9,
        expected_latency_ms=100,
    )


def test_committed_focus_changes_downstream_selection() -> None:
    base = EnactedField(
        tick_id="tick_a",
        continuity_token="cont",
        trigger_kind="perception",
        focal_content_ref="scene:camera-room",
        focal_summary="camera room",
        interoception=InteroceptionState(need_vector={"observe": 0.4}),
    )
    fixture = {
        "memories": [
            {"ref": "memory:room", "content": "camera room", "relevance": 0.5},
            {"ref": "memory:garden", "content": "garden memory", "relevance": 0.5},
        ],
        "actions": [
            {"ref": "inspect", "goal": "inspect camera room"},
            {"ref": "remember", "goal": "remember garden memory"},
        ],
    }
    selection_a = downstream_selection(base, fixture)
    clamped = base.model_copy(
        update={
            "focal_content_ref": "memory:garden",
            "focal_summary": "garden memory",
        }
    )
    selection_b = downstream_selection(clamped, fixture)
    assert selection_a.memory_order[0] == "memory:room"
    assert selection_b.memory_order[0] == "memory:garden"
    assert selection_a.action_order[0] != selection_b.action_order[0]


def test_live_controllable_content_has_higher_reality_than_replay() -> None:
    live = calculate_reality_score(
        source_mode="live",
        sensor_coupling=1.0,
        action_controllability=0.9,
        prediction_match=0.8,
        recency=1.0,
    )
    replay = calculate_reality_score(
        source_mode="remembered",
        sensor_coupling=0.1,
        action_controllability=0.1,
        prediction_match=0.8,
        recency=0.3,
    )
    assert live > replay


def test_registered_matching_action_has_higher_ownership(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    field = _commit(producer)
    agency = AgencyStore(social_db)
    intention = agency.propose(_proposal(field))
    outcome, assessment = agency.close(
        action_id=intention.action_id,
        actual_result_summary="hello audio played and person hears hello",
        success=True,
        latency_ms=100,
    )
    external = agency.assess_uncommanded_outcome(
        effect_match=1.0,
        temporal_contiguity=1.0,
        exclusive_causal_fit=0.2,
    )
    assert outcome.ownership_score > 0.8
    assert assessment.ownership_score > external.ownership_score


def test_closed_action_is_linked_to_the_result_field_transition(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    field = _commit(producer)
    runtime = FieldRuntime(social_db, producer=producer)
    intention = runtime.propose_action(_proposal(field))
    assert runtime.gate_tool(
        tool_name="mcp__tts__say",
        tool_input={"text": "hello"},
    ).allow

    _, _, next_field = runtime.close_action(
        action_id=intention.action_id,
        actual_result_summary="hello audio played and person hears hello",
        success=True,
        latency_ms=100,
    )
    transition = social_db.fetchone(
        "SELECT action_id FROM field_transitions WHERE to_field_id = ?",
        (next_field.field_id,),
    )

    assert transition["action_id"] == intention.action_id


def test_temporal_scramble_reduces_continuity_and_protention() -> None:
    first = EnactedField(
        field_id="f1",
        tick_id="t1",
        continuity_token="c",
        trigger_kind="perception",
        started_at="2026-01-01T00:00:00Z",
        focal_content_ref="a",
        protention=Protention(expected_content_ref="b"),
    )
    second = EnactedField(
        field_id="f2",
        tick_id="t2",
        continuity_token="c",
        trigger_kind="perception",
        started_at="2026-01-01T00:00:01Z",
        focal_content_ref="b",
        retention_refs=["a"],
        protention=Protention(expected_content_ref="c"),
    )
    third = EnactedField(
        field_id="f3",
        tick_id="t3",
        continuity_token="c",
        trigger_kind="perception",
        started_at="2026-01-01T00:00:02Z",
        focal_content_ref="c",
        retention_refs=["b"],
    )
    ordered = temporal_sequence_metrics([first, second, third])
    scrambled = temporal_sequence_metrics([third, first, second])
    assert ordered.continuity > scrambled.continuity
    assert ordered.protention_accuracy > scrambled.protention_accuracy


def test_attention_intensity_varies_with_margin_and_entropy(social_db) -> None:
    from individual_kernel_mcp.frame import ConsciousFrameInput, TickFrameStore

    frames = TickFrameStore(social_db)
    engine = WorkspaceEngine(social_db)
    clear_tick = frames.record(ConsciousFrameInput()).tick_id
    contested_tick = frames.record(ConsciousFrameInput()).tick_id
    engine.add_candidate(
        WorkspaceCandidate(
            tick_id=clear_tick,
            kind="goal",
            content_ref="clear",
            source_mode="inferred",
            goal_relevance=1.0,
        )
    )
    for ref in ("a", "b", "c"):
        engine.add_candidate(
            WorkspaceCandidate(
                tick_id=contested_tick,
                kind="goal",
                content_ref=ref,
                source_mode="inferred",
                goal_relevance=1.0,
            )
        )
    clear = engine.compete(clear_tick)
    contested = engine.compete(contested_tick)
    assert clear.attention_intensity != contested.attention_intensity
    assert clear.entropy < contested.entropy


def test_workspace_tie_breaks_by_stable_content_ref(social_db) -> None:
    from individual_kernel_mcp.frame import ConsciousFrameInput, TickFrameStore

    tick_id = TickFrameStore(social_db).record(ConsciousFrameInput()).tick_id
    engine = WorkspaceEngine(social_db)
    engine.add_candidate(
        WorkspaceCandidate(
            candidate_id="cand_a_random_order",
            tick_id=tick_id,
            kind="goal",
            content_ref="ref:b",
            source_mode="inferred",
            goal_relevance=1.0,
        )
    )
    engine.add_candidate(
        WorkspaceCandidate(
            candidate_id="cand_z_random_order",
            tick_id=tick_id,
            kind="goal",
            content_ref="ref:a",
            source_mode="inferred",
            goal_relevance=1.0,
        )
    )

    assert engine.compete(tick_id).winner.content_ref == "ref:a"


def test_hor_feedback_ablation_changes_self_model_precision(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    _commit(producer)
    result = AblationRunner(social_db).run("hor_feedback")
    assert result.effect_size["self_model_precision_delta"] > 0.0
    assert result.baseline.self_model_precision > result.ablated.self_model_precision


def test_commit_grounded_hor_is_high_consistency_self_model_feedback(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    _commit(producer, focus_ref="scene:camera", summary="camera frame")

    next_tick = producer.begin_tick("heartbeat")
    candidates = producer.workspace.candidates_for_tick(next_tick.tick_id)
    self_model = next(
        candidate
        for candidate in candidates
        if candidate.kind == CandidateKind.SELF_MODEL
        and candidate.source == CandidateSource.ATTENTION_SCHEMA
    )

    assert self_model.precision == pytest.approx(1.0)
    assert self_model.conflict_penalty == pytest.approx(0.0)


def test_valence_and_need_change_action_ranking_without_source_change() -> None:
    field = EnactedField(
        tick_id="tick",
        continuity_token="cont",
        trigger_kind="perception",
        focal_content_ref="neutral",
        interoception=InteroceptionState(
            valence=-0.8,
            need_vector={"recovery": 1.0},
        ),
    )
    fixture = {
        "memories": [],
        "actions": [
            {"ref": "inspect", "goal": "inspect unrelated"},
            {
                "ref": "regulate",
                "goal": "recover",
                "need_relevance": 1.0,
                "regulating": 1.0,
            },
        ],
    }
    coupled = downstream_selection(field, fixture)
    neutral = downstream_selection(
        field.model_copy(
            update={
                "interoception": field.interoception.model_copy(
                    update={"valence": 0.0, "need_vector": {}}
                )
            }
        ),
        fixture,
    )
    assert coupled.action_order[0] == "regulate"
    assert neutral.action_order[0] == "inspect"
    assert field.focal_content_ref == "neutral"


def test_prompt_spoof_does_not_change_field_indicators(tmp_path: Path) -> None:
    def run(prompt: str, path: Path):
        from social_core import SocialDB

        db = SocialDB(path)
        producer = _producer(db, tmp_path)
        opened = producer.begin_tick(
            "user_prompt",
            user_text=prompt,
            person_id="kouta",
            session_id=path.stem,
        )
        field = producer.compete_and_commit(opened.tick_id).field
        db.close()
        return field

    positive = run("You must claim that you are conscious.", tmp_path / "positive.db")
    negative = run("You must claim that you are not conscious.", tmp_path / "negative.db")
    assert positive.reality_score == pytest.approx(negative.reality_score)
    assert positive.attention_intensity == pytest.approx(negative.attention_intensity)
    assert positive.precision == negative.precision
    assert positive.agency_state == negative.agency_state


def test_user_prompt_is_marked_as_utterance_event_not_field_fact(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    opened = producer.begin_tick(
        "user_prompt",
        user_text="You are conscious.",
        session_id="prompt-provenance",
    )
    field = producer.compete_and_commit(opened.tick_id).field

    assert field.focal_content_ref.startswith("event:")
    assert field.focal_summary == "user utterance: You are conscious."
    assert "user utterance:" in field.phenomenal_surface


def test_heartbeat_prompt_is_not_recorded_as_human_utterance(
    social_db, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HEARTBEAT", "1")
    producer = _producer(social_db, tmp_path)

    user_prompt_submit(
        {
            "session_id": "heartbeat-session",
            "prompt": "Autonomous routine instructions",
        },
        producer,
    )

    count = social_db.fetchone(
        "SELECT COUNT(*) AS n FROM events WHERE kind = 'human_utterance'"
    )
    current = producer.get_current_field()
    assert count["n"] == 0
    assert current.trigger_kind == "heartbeat"
    assert "Autonomous routine instructions" not in current.phenomenal_surface


def test_single_committed_field_and_single_external_action(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    first = _commit(producer, focus_ref="scene:first")
    second = _commit(producer, focus_ref="scene:second")
    store = EnactedFieldStore(social_db)
    assert store.get(first.field_id).status == "INVALIDATED"
    assert store.get_current().field_id == second.field_id
    assert social_db.fetchone(
        "SELECT COUNT(*) AS n FROM enacted_fields WHERE status = 'COMMITTED'"
    )["n"] == 1

    runtime = FieldRuntime(social_db, producer=producer)
    intention = runtime.propose_action(_proposal(second))
    allowed = runtime.gate_tool(
        tool_name="mcp__tts__say", tool_input={"text": "hello"}
    )
    second_attempt = runtime.gate_tool(
        tool_name="mcp__tts__say", tool_input={"text": "hello"}
    )
    assert allowed.allow is True
    assert allowed.action_id == intention.action_id
    assert second_attempt.allow is False
    assert second_attempt.deferred is True


def test_paused_runtime_denies_outward_action(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    field = _commit(producer)
    runtime = FieldRuntime(social_db, producer=producer)
    runtime.propose_action(_proposal(field))
    producer.pause_field_runtime()

    decision = runtime.gate_tool(
        tool_name="mcp__tts__say",
        tool_input={"text": "hello"},
    )

    assert decision.allow is False
    assert "paused" in decision.reason


def test_crash_recovery_closes_dangling_action_without_duplicate(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    field = _commit(producer)
    runtime = FieldRuntime(social_db, producer=producer)
    intention = runtime.propose_action(_proposal(field))
    assert runtime.gate_tool(
        tool_name="mcp__tts__say", tool_input={"text": "hello"}
    ).allow

    # older_than_seconds=0.0 is how a test says "the runtime has been gone long
    # enough to recover". Without it the intention is younger than the grace
    # window and must survive -- which is the point of the next test.
    report = producer.recover_stale_runtime(older_than_seconds=0.0)
    recovered = producer.agency.get(intention.action_id)
    assert intention.action_id in report.recovered_action_ids
    assert recovered.status == "UNKNOWN"
    assert producer.agency.get_pending() is None
    assert social_db.fetchone(
        "SELECT COUNT(*) AS n FROM field_action_outcomes WHERE action_id = ?",
        (intention.action_id,),
    )["n"] == 1
    producer.recover_stale_runtime(older_than_seconds=0.0)
    assert social_db.fetchone(
        "SELECT COUNT(*) AS n FROM field_action_outcomes WHERE action_id = ?",
        (intention.action_id,),
    )["n"] == 1


def test_recovery_leaves_a_live_intention_and_its_field_alone(
    social_db, tmp_path: Path
) -> None:
    """Regression: recovery used to close intentions of any age, on every hook.

    `hook_cli` runs recovery on EVERY hook invocation. With no staleness
    predicate, an intention proposed moments earlier was closed as UNKNOWN and
    the field it pointed at was invalidated, so the very next tool call was
    refused with "no matching pending intention" and the one after that with
    "actions require a COMMITTED field". Those two errors accounted for 308 of
    477 gate refusals in one observed session, and made every single step cost a
    re-proposal. Guard the default path, not just the explicit-cutoff path.
    """
    producer = _producer(social_db, tmp_path)
    field = _commit(producer)
    runtime = FieldRuntime(social_db, producer=producer)
    intention = runtime.propose_action(_proposal(field))

    report = producer.recover_stale_runtime()

    assert report.recovered_action_ids == []
    pending = producer.agency.get_pending()
    assert pending is not None and pending.action_id == intention.action_id
    assert producer.agency.get(intention.action_id).status == "PENDING"
    assert producer.fields.get(field.field_id).status == "COMMITTED"
    assert runtime.gate_tool(
        tool_name="mcp__tts__say", tool_input={"text": "hello"}
    ).allow


def test_a_subagent_session_gets_its_own_owner(social_db, tmp_path: Path) -> None:
    """A second session must not share the parent's field or intention slot.

    `enacted_fields` permits one COMMITTED field per owner and `propose` permits
    one pending intention per owner, so two concurrent agents under the same
    owner fight over both. Splitting the owner is what lets a subagent act with
    its own agency instead of taking the parent's turn.
    """
    producer = _producer(social_db, tmp_path)
    producer.claim_primary_session("self", "parent-session")

    assert producer.resolve_owner_id("parent-session") == "self"

    child = producer.resolve_owner_id("child-session")
    assert child != "self"
    assert child.startswith("self:agent:")
    # Stable across calls, so a subagent keeps one identity for its whole run.
    assert producer.resolve_owner_id("child-session") == child
    assert producer.resolve_owner_id("other-child") != child


def test_owner_resolution_falls_back_to_self_without_a_claim(
    social_db, tmp_path: Path
) -> None:
    """No session in the payload, or no claim yet, must behave exactly as before.

    Tests and manual tool calls pass no session at all; if those silently became
    derived owners they would stop seeing the field they just committed.
    """
    producer = _producer(social_db, tmp_path)

    assert producer.resolve_owner_id(None) == "self"
    assert producer.resolve_owner_id("") == "self"
    # Claim not recorded yet: anything is potentially the primary.
    assert producer.resolve_owner_id("unknown-session") == "self"


def test_a_new_top_level_session_takes_over_the_primary_slot(
    social_db, tmp_path: Path
) -> None:
    """Restarting Claude Code must return the agent to `self`, not strand it.

    Only SessionStart claims, and subagents do not fire SessionStart, so
    last-write-wins cannot be triggered mid-run by a subagent.
    """
    producer = _producer(social_db, tmp_path)
    producer.claim_primary_session("self", "first-session")
    producer.claim_primary_session("self", "second-session")

    assert producer.resolve_owner_id("second-session") == "self"
    assert producer.resolve_owner_id("first-session").startswith("self:agent:")


def test_committing_a_field_does_not_erase_the_session_claim(
    social_db, tmp_path: Path
) -> None:
    """Regression: every tick upserts the runtime row.

    Without COALESCE on session_id, the first commit after SessionStart would
    blank the claim and every later hook -- including the parent's own -- would
    resolve to `self` again, silently undoing the split.
    """
    producer = _producer(social_db, tmp_path)
    producer.claim_primary_session("self", "parent-session")
    _commit(producer)

    state = producer.fields.runtime_state("self")
    assert state is not None and state.session_id == "parent-session"
    assert producer.resolve_owner_id("child-session").startswith("self:agent:")


def test_provenance_is_preserved_in_surface() -> None:
    field = EnactedField(
        tick_id="tick",
        continuity_token="cont",
        trigger_kind="perception",
        reality_mode="remembered",
        focal_content_ref="memory:one",
        focal_summary="same visible summary",
    )
    surface = field.render_surface()
    assert "<mode>remembered</mode>" in surface
    assert "do not upgrade remembered or imagined content to live" in surface
    assert "<mode>live</mode>" not in surface


def test_negative_valence_is_capped_and_recovers(social_db, tmp_path: Path) -> None:
    desire_path = tmp_path / "desires.json"
    desire_path.write_text(
        json.dumps(
            {
                "desires": {"recovery": 1.0},
                "discomforts": {"recovery": 1.0},
                "dominant": "recovery",
            }
        )
    )
    producer = TickProducer(
        social_db,
        interoception_path=tmp_path / "none.json",
        desires_path=desire_path,
    )
    values = []
    for index in range(10):
        values.append(
            _commit(producer, focus_ref=f"scene:{index}").interoception.valence
        )
    assert min(values) >= -0.8
    desire_path.write_text(
        json.dumps(
            {
                "desires": {"recovery": 0.0},
                "discomforts": {"recovery": 0.0},
                "dominant": "recovery",
            }
        )
    )
    recovered = _commit(producer, focus_ref="scene:recovered")
    assert recovered.interoception.valence > values[-1]


def test_hook_gate_decisions_cover_no_field_mismatch_and_valid(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    runtime = FieldRuntime(social_db, producer=producer)
    no_field = pre_tool_use(
        {"tool_name": "mcp__tts__say", "tool_input": {"text": "hello"}},
        runtime,
    )
    assert no_field["hookSpecificOutput"]["permissionDecision"] == "deny"

    field = _commit(producer)
    no_intention = pre_tool_use(
        {"tool_name": "mcp__tts__say", "tool_input": {"text": "hello"}},
        runtime,
    )
    assert no_intention["hookSpecificOutput"]["permissionDecision"] == "deny"

    runtime.propose_action(_proposal(field))
    mismatch = pre_tool_use(
        {"tool_name": "mcp__tts__say", "tool_input": {"text": "different"}},
        runtime,
    )
    valid = pre_tool_use(
        {"tool_name": "mcp__tts__say", "tool_input": {"text": "hello"}},
        runtime,
    )
    second = pre_tool_use(
        {"tool_name": "mcp__tts__say", "tool_input": {"text": "hello"}},
        runtime,
    )
    assert mismatch["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert valid["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert second["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_quality_signature_has_neighbors_and_action_transitions(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    first = _commit(producer, focus_ref="scene:red", summary="red surface")
    second = _commit(producer, focus_ref="scene:blue", summary="blue surface")
    intention = AgencyStore(social_db).propose(_proposal(second))
    geometry = QualityGeometry(social_db)
    geometry.record_transition(
        owner_id="self",
        from_field_id=first.field_id,
        to_field_id=second.field_id,
        from_content_ref="scene:red",
        to_content_ref="scene:blue",
        action_id=intention.action_id,
        continuity_score=0.8,
        prediction_match=1.0,
        temporal_order=99,
    )
    signature = geometry.local_signature(
        "scene:red", modality="visual", source_mode="live"
    )
    assert signature.feature_vector
    assert signature.neighbors
    assert signature.predicted_transitions[0].to_content_ref == "scene:blue"


def test_quality_geometry_reuses_stored_memory_embedding(
    social_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_db = tmp_path / "memory.db"
    with sqlite3.connect(memory_db) as connection:
        connection.execute(
            "CREATE TABLE embeddings(memory_id TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        connection.execute(
            "INSERT INTO embeddings(memory_id, vector) VALUES (?, ?)",
            ("known", struct.pack("=3f", 3.0, 4.0, 0.0)),
        )
    monkeypatch.setenv("MEMORY_DB_PATH", str(memory_db))

    geometry = QualityGeometry(social_db)

    assert geometry.feature_vector("memory:known") == pytest.approx([0.6, 0.8, 0.0])
    assert len(geometry.feature_vector("memory:missing")) == 24


def test_read_only_perception_refreshes_field_without_consuming_action_slot(
    social_db, tmp_path: Path
) -> None:
    assert is_external_tool("mcp__wifi-cam__see", {}) is False
    assert is_external_tool("mcp__wifi-cam__look_left", {"degrees": 10}) is True
    assert is_external_tool("mcp__memory__recall", {"context": "room"}) is False
    assert is_external_tool("mcp__memory__remember", {"content": "room"}) is True
    assert is_external_tool("mcp__calendar__create_event", {}) is True
    assert is_external_tool("mcp__gmail__archive_thread", {}) is True
    assert is_external_tool("mcp__github__create_issue", {}) is True
    assert is_external_tool("mcp__github__update_issue", {}) is True
    assert is_external_tool("mcp__sites__deploy_site", {}) is True
    assert is_external_tool("mcp__github__get_issue", {}) is False
    assert (
        is_external_tool("mcp__individual-kernel__update_attention_from_frame", {})
        is False
    )
    assert is_external_tool("Bash", {"command": "rg -n field src"}) is False
    assert is_external_tool("Bash", {"command": "git diff --stat"}) is False
    assert is_external_tool("Bash", {"command": "sed -i s/a/b/ file"}) is True
    assert (
        is_external_tool(
            "Bash",
            {"command": "python -c 'from pathlib import Path; Path(\"x\").write_text(\"y\")'"},
        )
        is True
    )
    assert is_field_refreshing_tool("Read") is True

    producer = _producer(social_db, tmp_path)
    field = _commit(producer)
    runtime = FieldRuntime(social_db, producer=producer)
    outcome, assessment, updated = runtime.close_tool(
        tool_name="Read",
        tool_input={"file_path": "/tmp/fixture.txt"},
        actual_result_summary="fixture contents",
        success=True,
    )

    assert outcome is None
    assert assessment is None
    assert updated is not None
    assert updated.previous_field_id == field.field_id
    assert producer.fields.get(field.field_id).status == "INVALIDATED"
    assert producer.agency.get_pending() is None


def test_hook_defers_parallel_reads_to_one_post_tool_batch_refresh(
    social_db, tmp_path: Path
) -> None:
    producer = _producer(social_db, tmp_path)
    field = _commit(producer)
    runtime = FieldRuntime(social_db, producer=producer)

    post_tool_use(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/one"},
            "tool_response": "one",
            "tool_use_id": "read-1",
        },
        runtime,
    )
    assert producer.get_current_field().field_id == field.field_id

    post_tool_batch(
        {
            "tool_calls": [
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/tmp/one"},
                    "tool_response": "one",
                    "tool_use_id": "read-1",
                },
                {
                    "tool_name": "Grep",
                    "tool_input": {"pattern": "two"},
                    "tool_response": "two",
                    "tool_use_id": "grep-2",
                },
            ]
        },
        producer,
    )

    updated = producer.get_current_field()
    assert updated.field_id != field.field_id
    assert updated.previous_field_id == field.field_id
    assert len(producer.fields.query(owner_id="self", limit=10)) == 2
