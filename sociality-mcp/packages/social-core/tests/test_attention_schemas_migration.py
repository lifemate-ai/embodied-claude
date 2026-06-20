"""Tests for migration 005 — attention_schemas table.

attention_schemas stores low-dimensional snapshots of "what attention is
doing" — the Attention Schema Theory surface (Theory B R4). owner_id='self'
is Kokone modeling her own attention; non-'self' owner_id values are
ToM-like models of other persons' attention (Phase 2.4.1+).

Backs Phase 2.4's AttentionSchemaTracker in consciousness-mcp.
"""

from __future__ import annotations

from social_core.db import SocialDB
from social_core.migrations import MIGRATIONS


def test_attention_schemas_table_created(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    tables = {
        row["name"]
        for row in db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "attention_schemas" in tables


def test_attention_schemas_columns(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    columns = {
        row["name"] for row in db.fetchall("PRAGMA table_info(attention_schemas)")
    }
    expected = {
        "schema_id",
        "ts",
        "owner_id",
        "focal_target_ref",
        "modality",
        "intensity",
        "dwell_seconds",
        "predicted_next_focus",
        "control_handle",
        "source_tick_id",
        "created_at",
    }
    assert expected <= columns, f"missing: {expected - columns}"


def test_attention_schemas_indexes(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    indexes = {
        row["name"]
        for row in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='attention_schemas'"
        )
    }
    assert "idx_attention_schemas_ts" in indexes
    assert "idx_attention_schemas_owner" in indexes
    assert "idx_attention_schemas_modality" in indexes


def test_migration_005_registered(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    applied = {row["name"] for row in db.fetchall("SELECT name FROM schema_migrations")}
    assert "005_attention_schemas" in applied
    assert len(applied) == len(MIGRATIONS)


def test_attention_schemas_insert_round_trip(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    db.execute(
        """
        INSERT INTO attention_schemas(
            schema_id, ts, owner_id, focal_target_ref, modality,
            intensity, dwell_seconds, predicted_next_focus,
            control_handle, source_tick_id, created_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "schema_test_001",
            "2026-06-21T03:00:00Z",
            "self",
            "wifi_cam.see",
            "visual",
            0.8,
            12.5,
            "person:kouta",
            "wifi_cam.look_left",
            "tick_001",
            "2026-06-21T03:00:00Z",
        ),
    )
    row = db.fetchone(
        "SELECT * FROM attention_schemas WHERE schema_id = ?",
        ("schema_test_001",),
    )
    assert row is not None
    assert row["owner_id"] == "self"
    assert row["modality"] == "visual"
    assert row["intensity"] == 0.8
    assert row["dwell_seconds"] == 12.5
    assert row["source_tick_id"] == "tick_001"


def test_schema_id_unique(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    db.execute(
        """
        INSERT INTO attention_schemas(
            schema_id, ts, owner_id, modality, intensity, created_at
        ) VALUES(?, ?, ?, ?, ?, ?)
        """,
        ("sch_dupe", "2026-06-21T03:00:00Z", "self", "visual", 0.5, "2026-06-21T03:00:00Z"),
    )
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO attention_schemas(
                schema_id, ts, owner_id, modality, intensity, created_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            ("sch_dupe", "2026-06-21T04:00:00Z", "self", "visual", 0.5, "2026-06-21T04:00:00Z"),
        )


def test_owner_id_default_to_self_via_non_null(temp_db_path) -> None:
    """owner_id is NOT NULL so omitting it is an error — defaults are caller-side."""
    db = SocialDB(temp_db_path)
    db.connect()
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO attention_schemas(
                schema_id, ts, modality, intensity, created_at
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (
                "sch_no_owner",
                "2026-06-21T03:00:00Z",
                "visual",
                0.5,
                "2026-06-21T03:00:00Z",
            ),
        )
