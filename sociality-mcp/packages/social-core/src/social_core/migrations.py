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
