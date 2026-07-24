"""SQLite migrations for the shared sociality database."""

from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection

from .time import utc_now


@dataclass(frozen=True, slots=True)
class Migration:
    name: str
    sql: str


_MIGRATION_002_SQL = """
CREATE TABLE IF NOT EXISTS agent_experiences (
    experience_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    person_id TEXT,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    private_summary TEXT,
    public_summary TEXT,
    why TEXT,
    felt_state_json TEXT NOT NULL DEFAULT '{}',
    desires_before_json TEXT NOT NULL DEFAULT '{}',
    desires_after_json TEXT NOT NULL DEFAULT '{}',
    related_event_ids TEXT NOT NULL DEFAULT '',
    related_memory_ids TEXT NOT NULL DEFAULT '',
    artifacts_json TEXT NOT NULL DEFAULT '[]',
    importance INTEGER NOT NULL DEFAULT 3,
    privacy_level TEXT NOT NULL DEFAULT 'private',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_experiences_ts
    ON agent_experiences(ts DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_experiences_person
    ON agent_experiences(person_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_agent_experiences_kind
    ON agent_experiences(kind, ts DESC);

CREATE TABLE IF NOT EXISTS private_reflections (
    reflection_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    person_id TEXT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    importance INTEGER NOT NULL DEFAULT 3,
    may_surface_later INTEGER NOT NULL DEFAULT 1,
    surfaced_at TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_private_reflections_ts
    ON private_reflections(ts DESC);
CREATE INDEX IF NOT EXISTS idx_private_reflections_person
    ON private_reflections(person_id, ts DESC);

CREATE TABLE IF NOT EXISTS interpretation_shifts (
    shift_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    person_id TEXT,
    topic TEXT NOT NULL,
    old_interpretation TEXT NOT NULL,
    new_interpretation TEXT NOT NULL,
    trigger TEXT NOT NULL,
    confidence REAL NOT NULL,
    implications_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interpretation_shifts_ts
    ON interpretation_shifts(ts DESC);
CREATE INDEX IF NOT EXISTS idx_interpretation_shifts_topic
    ON interpretation_shifts(topic, ts DESC);

CREATE TABLE IF NOT EXISTS private_letters (
    letter_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    person_id TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    intended_time TEXT,
    visibility TEXT NOT NULL DEFAULT 'private',
    related_open_loops TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_private_letters_person_ts
    ON private_letters(person_id, ts DESC);
"""


_MIGRATION_006_SQL = """
CREATE TABLE IF NOT EXISTS hor_records (
    hor_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_ref TEXT,
    asserted_mode TEXT NOT NULL,
    asserted_content TEXT NOT NULL,
    schema_snapshot_id TEXT,
    source_tick_id TEXT,
    confidence REAL NOT NULL DEFAULT 0.6,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hor_records_ts
    ON hor_records(ts DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hor_records_owner
    ON hor_records(owner_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_hor_records_asserted_mode
    ON hor_records(asserted_mode, ts DESC);
CREATE INDEX IF NOT EXISTS idx_hor_records_target_kind
    ON hor_records(target_kind, ts DESC);
CREATE INDEX IF NOT EXISTS idx_hor_records_source_tick
    ON hor_records(source_tick_id);
"""


_MIGRATION_005_SQL = """
CREATE TABLE IF NOT EXISTS attention_schemas (
    schema_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    focal_target_ref TEXT,
    modality TEXT NOT NULL,
    intensity REAL NOT NULL DEFAULT 0.0,
    dwell_seconds REAL NOT NULL DEFAULT 0.0,
    predicted_next_focus TEXT,
    control_handle TEXT,
    source_tick_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attention_schemas_ts
    ON attention_schemas(ts DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_attention_schemas_owner
    ON attention_schemas(owner_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_attention_schemas_modality
    ON attention_schemas(modality, ts DESC);
"""


_MIGRATION_004_SQL = """
CREATE TABLE IF NOT EXISTS tick_frames (
    tick_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    person_id TEXT,
    ignited INTEGER NOT NULL DEFAULT 0,
    conflicted INTEGER NOT NULL DEFAULT 0,
    attention_target_ref TEXT,
    dominant_desire TEXT,
    winning_memory_ids_json TEXT NOT NULL DEFAULT '[]',
    prediction_error_json TEXT NOT NULL DEFAULT '{}',
    affect_summary TEXT,
    chosen_action_ref TEXT,
    reportability TEXT NOT NULL DEFAULT 'mentionable',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tick_frames_ts
    ON tick_frames(ts DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tick_frames_reportability
    ON tick_frames(reportability, ts DESC);
CREATE INDEX IF NOT EXISTS idx_tick_frames_person
    ON tick_frames(person_id, ts DESC);
"""


_MIGRATION_003_SQL = """
CREATE TABLE IF NOT EXISTS counterfactuals (
    counterfactual_id TEXT PRIMARY KEY,
    tick_id TEXT,
    ts TEXT NOT NULL,
    person_id TEXT,
    chosen_action_ref TEXT,
    rejected_action TEXT NOT NULL,
    rejected_action_payload_json TEXT NOT NULL DEFAULT '{}',
    reason TEXT,
    source TEXT NOT NULL,
    expected_outcome TEXT,
    evidence_type TEXT,
    importance INTEGER NOT NULL DEFAULT 3,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_counterfactuals_ts
    ON counterfactuals(ts DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_counterfactuals_source
    ON counterfactuals(source, ts DESC);
CREATE INDEX IF NOT EXISTS idx_counterfactuals_tick
    ON counterfactuals(tick_id);
CREATE INDEX IF NOT EXISTS idx_counterfactuals_person
    ON counterfactuals(person_id, ts DESC);
"""


_MIGRATION_007_SQL = """
CREATE TABLE IF NOT EXISTS workspace_candidates (
    candidate_id TEXT PRIMARY KEY,
    tick_id TEXT NOT NULL REFERENCES tick_frames(tick_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    content_ref TEXT NOT NULL,
    content_summary TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    modality TEXT NOT NULL,
    precision REAL NOT NULL,
    prediction_error REAL NOT NULL,
    need_relevance REAL NOT NULL,
    goal_relevance REAL NOT NULL,
    expected_information_gain REAL NOT NULL,
    continuity_with_previous REAL NOT NULL,
    controllability REAL NOT NULL,
    social_relevance REAL NOT NULL,
    conflict_penalty REAL NOT NULL,
    switching_cost REAL NOT NULL,
    score REAL NOT NULL,
    reportability TEXT NOT NULL,
    rank INTEGER,
    selected INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workspace_candidates_tick
    ON workspace_candidates(tick_id, rank, score DESC);
CREATE INDEX IF NOT EXISTS idx_workspace_candidates_content
    ON workspace_candidates(content_ref);

CREATE TABLE IF NOT EXISTS enacted_fields (
    field_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    tick_id TEXT NOT NULL UNIQUE REFERENCES tick_frames(tick_id) ON DELETE CASCADE,
    continuity_token TEXT NOT NULL,
    previous_field_id TEXT REFERENCES enacted_fields(field_id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    trigger_kind TEXT NOT NULL,
    person_id TEXT,
    session_id TEXT,
    started_at TEXT NOT NULL,
    committed_at TEXT,
    closed_at TEXT,
    self_origin_json TEXT NOT NULL DEFAULT '{}',
    reality_mode TEXT NOT NULL,
    reality_score REAL NOT NULL,
    focal_content_ref TEXT,
    peripheral_content_refs_json TEXT NOT NULL DEFAULT '[]',
    retention_refs_json TEXT NOT NULL DEFAULT '[]',
    protention_json TEXT NOT NULL DEFAULT '{}',
    interoception_json TEXT NOT NULL DEFAULT '{}',
    precision_json TEXT NOT NULL DEFAULT '{}',
    affordance_refs_json TEXT NOT NULL DEFAULT '[]',
    pending_intention_ref TEXT,
    agency_state_json TEXT NOT NULL DEFAULT '{}',
    attention_schema_ref TEXT,
    hor_refs_json TEXT NOT NULL DEFAULT '[]',
    quality_signature_ref TEXT,
    phenomenal_surface TEXT NOT NULL DEFAULT '',
    epistemic_trace_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(status IN ('OPEN', 'COMMITTED', 'INVALIDATED', 'CLOSED', 'ABORTED')),
    CHECK(reality_mode IN ('live', 'inferred', 'remembered', 'imagined', 'mixed')),
    CHECK(reality_score >= 0.0 AND reality_score <= 1.0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_enacted_fields_one_committed_owner
    ON enacted_fields(owner_id)
    WHERE status = 'COMMITTED';
CREATE INDEX IF NOT EXISTS idx_enacted_fields_owner_updated
    ON enacted_fields(owner_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_enacted_fields_previous
    ON enacted_fields(previous_field_id);
CREATE INDEX IF NOT EXISTS idx_enacted_fields_tick
    ON enacted_fields(tick_id);

CREATE TABLE IF NOT EXISTS field_runtime_state (
    owner_id TEXT PRIMARY KEY,
    continuity_token TEXT NOT NULL,
    current_field_id TEXT REFERENCES enacted_fields(field_id) ON DELETE SET NULL,
    open_tick_id TEXT REFERENCES tick_frames(tick_id) ON DELETE SET NULL,
    state TEXT NOT NULL DEFAULT 'ACTIVE',
    last_trigger_kind TEXT,
    last_recovery_at TEXT,
    updated_at TEXT NOT NULL,
    CHECK(state IN ('ACTIVE', 'PAUSED'))
);

CREATE TABLE IF NOT EXISTS field_intentions (
    action_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    field_id TEXT NOT NULL REFERENCES enacted_fields(field_id) ON DELETE CASCADE,
    tick_id TEXT NOT NULL REFERENCES tick_frames(tick_id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    tool_input_hash TEXT NOT NULL,
    normalized_tool_input_json TEXT NOT NULL,
    intended_goal TEXT NOT NULL,
    predicted_effects_json TEXT NOT NULL,
    expected_latency_ms INTEGER,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    closed_at TEXT,
    CHECK(status IN ('PENDING', 'ALLOWED', 'COMPLETED', 'FAILED', 'UNKNOWN', 'DENIED'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_field_intentions_one_pending_owner
    ON field_intentions(owner_id)
    WHERE status IN ('PENDING', 'ALLOWED');
CREATE INDEX IF NOT EXISTS idx_field_intentions_tick
    ON field_intentions(tick_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_intentions_field
    ON field_intentions(field_id, created_at DESC);

CREATE TABLE IF NOT EXISTS field_action_outcomes (
    outcome_id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL UNIQUE REFERENCES field_intentions(action_id) ON DELETE CASCADE,
    field_id TEXT NOT NULL REFERENCES enacted_fields(field_id) ON DELETE CASCADE,
    tick_id TEXT NOT NULL REFERENCES tick_frames(tick_id) ON DELETE CASCADE,
    actual_result_ref TEXT,
    actual_result_summary TEXT NOT NULL DEFAULT '',
    actual_result_hash TEXT,
    success INTEGER NOT NULL,
    latency_ms INTEGER,
    mismatch_vector_json TEXT NOT NULL DEFAULT '{}',
    ownership_score REAL NOT NULL,
    agency_assessment_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    CHECK(ownership_score >= 0.0 AND ownership_score <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_field_action_outcomes_tick
    ON field_action_outcomes(tick_id, created_at DESC);

CREATE TABLE IF NOT EXISTS field_transitions (
    transition_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    from_field_id TEXT REFERENCES enacted_fields(field_id) ON DELETE SET NULL,
    to_field_id TEXT REFERENCES enacted_fields(field_id) ON DELETE SET NULL,
    action_id TEXT REFERENCES field_intentions(action_id) ON DELETE SET NULL,
    from_content_ref TEXT,
    to_content_ref TEXT,
    continuity_score REAL NOT NULL,
    prediction_match REAL NOT NULL,
    temporal_order INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_field_transitions_owner
    ON field_transitions(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_transitions_content
    ON field_transitions(from_content_ref, to_content_ref);

CREATE TABLE IF NOT EXISTS quality_signatures (
    signature_id TEXT PRIMARY KEY,
    content_ref TEXT NOT NULL,
    modality TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    feature_vector_json TEXT NOT NULL,
    neighbors_json TEXT NOT NULL DEFAULT '[]',
    predicted_transitions_json TEXT NOT NULL DEFAULT '[]',
    valence_associations_json TEXT NOT NULL DEFAULT '{}',
    discriminability REAL NOT NULL,
    adaptation_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quality_signatures_content
    ON quality_signatures(content_ref, updated_at DESC);

CREATE TABLE IF NOT EXISTS field_ablation_runs (
    ablation_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    fixture_ref TEXT,
    seed INTEGER,
    source_field_id TEXT REFERENCES enacted_fields(field_id) ON DELETE SET NULL,
    baseline_json TEXT NOT NULL,
    ablated_json TEXT NOT NULL,
    effect_size_json TEXT NOT NULL,
    reversible_snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_field_ablation_runs_owner
    ON field_ablation_runs(owner_id, created_at DESC);
"""

_MIGRATION_008_SQL = """
CREATE INDEX IF NOT EXISTS idx_field_action_outcomes_field
    ON field_action_outcomes(field_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_transitions_from_field
    ON field_transitions(from_field_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_transitions_to_field
    ON field_transitions(to_field_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_transitions_action
    ON field_transitions(action_id, created_at DESC);
"""


MIGRATIONS = [
    Migration(
        name="001_initial_schema",
        sql="""
        CREATE TABLE IF NOT EXISTS events (
            event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            person_id TEXT,
            session_id TEXT,
            correlation_id TEXT,
            confidence REAL NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS social_state_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            person_id TEXT,
            state_json TEXT NOT NULL,
            summary_for_prompt TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS persons (
            person_id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            role TEXT,
            profile_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS person_aliases (
            alias_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            alias TEXT NOT NULL,
            UNIQUE(person_id, alias),
            UNIQUE(alias)
        );

        CREATE TABLE IF NOT EXISTS relationship_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            ts TEXT NOT NULL,
            warmth REAL NOT NULL,
            trust REAL NOT NULL,
            fragility REAL NOT NULL,
            expected_response_latency REAL NOT NULL,
            recent_stress REAL NOT NULL,
            reciprocity_balance REAL NOT NULL,
            relationship_summary TEXT NOT NULL,
            notes_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS commitments (
            commitment_id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            text TEXT NOT NULL,
            due_at TEXT,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS open_loops (
            loop_id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            topic TEXT NOT NULL,
            status TEXT NOT NULL,
            source_event_id TEXT,
            updated_at TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS rituals (
            ritual_id TEXT PRIMARY KEY,
            person_id TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
            kind TEXT NOT NULL,
            cadence TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS person_boundaries (
            boundary_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            rule TEXT NOT NULL,
            source_text TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scene_frames (
            frame_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            person_id TEXT,
            session_id TEXT,
            camera_pose_json TEXT NOT NULL,
            scene_summary TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scene_people (
            frame_person_id TEXT PRIMARY KEY,
            frame_id TEXT NOT NULL REFERENCES scene_frames(frame_id) ON DELETE CASCADE,
            person_id TEXT,
            display_name TEXT,
            relative_position TEXT,
            distance TEXT,
            gaze_target TEXT,
            confidence REAL NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scene_objects (
            frame_object_id TEXT PRIMARY KEY,
            frame_id TEXT NOT NULL REFERENCES scene_frames(frame_id) ON DELETE CASCADE,
            object_id TEXT NOT NULL,
            label TEXT NOT NULL,
            attributes_json TEXT NOT NULL,
            relations_json TEXT NOT NULL,
            salience REAL NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS joint_focus (
            focus_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            person_id TEXT,
            target_id TEXT NOT NULL,
            initiator TEXT NOT NULL,
            confidence REAL NOT NULL,
            based_on_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS consents (
            consent_id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            consent_type TEXT NOT NULL,
            value INTEGER NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            UNIQUE(person_id, consent_type)
        );

        CREATE TABLE IF NOT EXISTS narrative_daybooks (
            daybook_id TEXT PRIMARY KEY,
            day TEXT NOT NULL UNIQUE,
            ts TEXT NOT NULL,
            summary TEXT NOT NULL,
            evidence_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS identity_facets (
            facet_id TEXT PRIMARY KEY,
            facet_key TEXT NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            confidence REAL NOT NULL,
            updated_at TEXT NOT NULL,
            evidence_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS narrative_arcs (
            arc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            importance REAL NOT NULL,
            summary TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_events_source_correlation
            ON events(source, correlation_id)
            WHERE correlation_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC, event_seq DESC);
        CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_commitments_person_status
            ON commitments(person_id, status, due_at);
        CREATE INDEX IF NOT EXISTS idx_open_loops_person_status
            ON open_loops(person_id, status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_scene_frames_person_ts
            ON scene_frames(person_id, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_joint_focus_person_ts
            ON joint_focus(person_id, ts DESC);
        """,
    ),
    Migration(
        name="002_interaction_orchestrator",
        sql=_MIGRATION_002_SQL,
    ),
    Migration(
        name="003_counterfactuals",
        sql=_MIGRATION_003_SQL,
    ),
    Migration(
        name="004_tick_frames",
        sql=_MIGRATION_004_SQL,
    ),
    Migration(
        name="005_attention_schemas",
        sql=_MIGRATION_005_SQL,
    ),
    Migration(
        name="006_hor_records",
        sql=_MIGRATION_006_SQL,
    ),
    Migration(
        name="007_enacted_first_person_field",
        sql=_MIGRATION_007_SQL,
    ),
    Migration(
        name="008_enacted_field_indexes",
        sql=_MIGRATION_008_SQL,
    ),
]


def apply_migrations(connection: Connection) -> None:
    """Apply pending migrations exactly once."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied = {
        row[0] for row in connection.execute("SELECT name FROM schema_migrations").fetchall()
    }
    for migration in MIGRATIONS:
        if migration.name in applied:
            continue
        connection.executescript(migration.sql)
        connection.execute(
            "INSERT INTO schema_migrations(name, applied_at) VALUES(?, ?)",
            (migration.name, utc_now()),
        )
    connection.commit()
