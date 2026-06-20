"""Tests for migration 006 — hor_records table.

Higher-Order Representation records (HOT / Theory B R1-R3, R9-R10):
specializations of EpistemicClaim where evidence_type='inferred' and
content paraphrases "I am in state X targeting first-order ref Y".

Backs Phase 2.5's HORStore in consciousness-mcp.
"""

from __future__ import annotations

from social_core.db import SocialDB
from social_core.migrations import MIGRATIONS


def test_hor_records_table_created(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    tables = {
        row["name"]
        for row in db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "hor_records" in tables


def test_hor_records_columns(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    columns = {
        row["name"] for row in db.fetchall("PRAGMA table_info(hor_records)")
    }
    expected = {
        "hor_id",
        "ts",
        "owner_id",
        "target_kind",
        "target_ref",
        "asserted_mode",
        "asserted_content",
        "schema_snapshot_id",
        "source_tick_id",
        "confidence",
        "source",
        "created_at",
    }
    assert expected <= columns, f"missing: {expected - columns}"


def test_hor_records_indexes(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    indexes = {
        row["name"]
        for row in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='hor_records'"
        )
    }
    assert "idx_hor_records_ts" in indexes
    assert "idx_hor_records_owner" in indexes
    assert "idx_hor_records_asserted_mode" in indexes
    assert "idx_hor_records_target_kind" in indexes
    assert "idx_hor_records_source_tick" in indexes


def test_migration_006_registered(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    applied = {row["name"] for row in db.fetchall("SELECT name FROM schema_migrations")}
    assert "006_hor_records" in applied
    assert len(applied) == len(MIGRATIONS)


def test_hor_records_insert_round_trip(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    db.execute(
        """
        INSERT INTO hor_records(
            hor_id, ts, owner_id, target_kind, target_ref,
            asserted_mode, asserted_content, schema_snapshot_id,
            source_tick_id, confidence, source, created_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "hor_test_001",
            "2026-06-21T03:00:00Z",
            "self",
            "memory",
            "mem_001",
            "remembering",
            "I am remembering Kouta sat at the desk earlier",
            "sch_001",
            "tick_001",
            0.8,
            "schema_readout",
            "2026-06-21T03:00:00Z",
        ),
    )
    row = db.fetchone(
        "SELECT * FROM hor_records WHERE hor_id = ?",
        ("hor_test_001",),
    )
    assert row is not None
    assert row["owner_id"] == "self"
    assert row["target_kind"] == "memory"
    assert row["target_ref"] == "mem_001"
    assert row["asserted_mode"] == "remembering"
    assert row["confidence"] == 0.8
    assert row["source"] == "schema_readout"


def test_hor_id_unique(temp_db_path) -> None:
    db = SocialDB(temp_db_path)
    db.connect()
    db.execute(
        """
        INSERT INTO hor_records(
            hor_id, ts, owner_id, target_kind, asserted_mode,
            asserted_content, confidence, source, created_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "hor_dupe",
            "2026-06-21T03:00:00Z",
            "self",
            "none",
            "attending",
            "I am attending to nothing in particular",
            0.5,
            "reflection",
            "2026-06-21T03:00:00Z",
        ),
    )
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO hor_records(
                hor_id, ts, owner_id, target_kind, asserted_mode,
                asserted_content, confidence, source, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hor_dupe",
                "2026-06-21T04:00:00Z",
                "self",
                "none",
                "attending",
                "again",
                0.5,
                "reflection",
                "2026-06-21T04:00:00Z",
            ),
        )


def test_owner_and_modes_not_null(temp_db_path) -> None:
    """owner_id, target_kind, asserted_mode, asserted_content are NOT NULL."""
    db = SocialDB(temp_db_path)
    db.connect()
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO hor_records(hor_id, ts, created_at)
            VALUES(?, ?, ?)
            """,
            ("hor_partial", "2026-06-21T03:00:00Z", "2026-06-21T03:00:00Z"),
        )
