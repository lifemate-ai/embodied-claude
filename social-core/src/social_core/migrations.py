"""SQLite migrations for the shared sociality database."""

from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection

from .time import utc_now


@dataclass(frozen=True, slots=True)
class Migration:
    name: str
    sql: str


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
