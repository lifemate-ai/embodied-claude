"""Tests for the desire-system memory backend adapters."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend import (
    ChromaMemoryAdapter,
    NullMemoryAdapter,
    SQLiteMemoryAdapter,
    make_default_adapter,
)


def _bootstrap_memory_db(path: Path) -> None:
    """Create the minimal memories table the adapter queries."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                normalized_content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL DEFAULT 'neutral',
                importance INTEGER NOT NULL DEFAULT 3,
                category TEXT NOT NULL DEFAULT 'daily',
                tags TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert(path: Path, memory_id: str, content: str, ts: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO memories(id, content, normalized_content, timestamp) VALUES (?, ?, ?, ?)",
            (memory_id, content, content.lower(), ts),
        )
        conn.commit()
    finally:
        conn.close()


def test_sqlite_adapter_returns_none_when_db_missing(tmp_path):
    adapter = SQLiteMemoryAdapter(tmp_path / "missing.db")
    assert adapter.latest_satisfaction_ts(["外を見た"]) is None


def test_sqlite_adapter_finds_latest_matching_keyword(tmp_path):
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    _insert(db, "a", "今日は晴れです", "2026-04-18T08:00:00+00:00")
    _insert(db, "b", "外を見た、空が青い", "2026-04-18T09:30:00+00:00")
    _insert(db, "c", "外を見た、雨が降ってる", "2026-04-18T11:00:00+00:00")

    adapter = SQLiteMemoryAdapter(db)
    latest = adapter.latest_satisfaction_ts(["外を見た", "空を見た"])
    assert latest is not None
    assert latest.hour == 11
    assert latest.tzinfo is not None


def test_sqlite_adapter_ignores_non_matching_rows(tmp_path):
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    _insert(db, "a", "コード書いた", "2026-04-18T11:00:00+00:00")

    adapter = SQLiteMemoryAdapter(db)
    assert adapter.latest_satisfaction_ts(["外を見た"]) is None


def test_sqlite_adapter_record_satisfaction_round_trip(tmp_path):
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    adapter = SQLiteMemoryAdapter(db)
    ts = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)

    memory_id = adapter.record_satisfaction(
        desire_name="look_outside",
        summary="ベランダから夕焼け見た",
        ts=ts,
        metadata={"outcome": "satisfied"},
    )
    assert memory_id.startswith("desire_look_outside_")

    found = adapter.latest_satisfaction_ts(["ベランダから夕焼け見た"])
    assert found is not None
    assert found == ts


def test_sqlite_adapter_empty_keywords_returns_none(tmp_path):
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    adapter = SQLiteMemoryAdapter(db)
    assert adapter.latest_satisfaction_ts([]) is None
    assert adapter.latest_satisfaction_ts([""]) is None


def test_null_adapter_is_inert():
    adapter = NullMemoryAdapter()
    assert adapter.latest_satisfaction_ts(["anything"]) is None
    assert adapter.record_satisfaction(
        desire_name="x", summary="y", ts=datetime.now(timezone.utc)
    ) == ""


def test_make_default_adapter_prefers_sqlite_when_db_exists(monkeypatch, tmp_path):
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    monkeypatch.setenv("MEMORY_DB_FILE", str(db))
    monkeypatch.delenv("DESIRE_MEMORY_BACKEND", raising=False)
    adapter = make_default_adapter()
    assert isinstance(adapter, SQLiteMemoryAdapter)


def test_make_default_adapter_respects_explicit_sqlite(monkeypatch, tmp_path):
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    monkeypatch.setenv("DESIRE_MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("MEMORY_DB_FILE", str(db))
    adapter = make_default_adapter()
    assert isinstance(adapter, SQLiteMemoryAdapter)


def test_make_default_adapter_respects_explicit_chroma(monkeypatch, tmp_path):
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    monkeypatch.setenv("DESIRE_MEMORY_BACKEND", "chroma")
    monkeypatch.setenv("MEMORY_DB_PATH", str(chroma_path))
    adapter = make_default_adapter()
    assert isinstance(adapter, ChromaMemoryAdapter)


def test_make_default_adapter_null_when_nothing_available(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DB_FILE", str(tmp_path / "missing.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "missing-chroma"))
    monkeypatch.delenv("DESIRE_MEMORY_BACKEND", raising=False)
    adapter = make_default_adapter()
    assert isinstance(adapter, NullMemoryAdapter)


def test_make_default_adapter_explicit_null(monkeypatch):
    monkeypatch.setenv("DESIRE_MEMORY_BACKEND", "none")
    adapter = make_default_adapter()
    assert isinstance(adapter, NullMemoryAdapter)


def test_satisfaction_write_makes_future_lookup_succeed(tmp_path):
    """Regression guard: record_satisfaction + latest_satisfaction_ts must agree."""
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    adapter = SQLiteMemoryAdapter(db)

    first = datetime(2026, 4, 19, 10, 0, 0, tzinfo=timezone.utc)
    adapter.record_satisfaction(desire_name="browse_curiosity", summary="WebSearch", ts=first)

    second = first + timedelta(hours=1)
    adapter.record_satisfaction(
        desire_name="browse_curiosity", summary="WebSearchで調べた", ts=second
    )

    latest = adapter.latest_satisfaction_ts(["WebSearchで調べた"])
    assert latest == second


@pytest.mark.parametrize("bad_ts", ["", "not-a-date", "2026/04/19"])
def test_sqlite_adapter_ignores_malformed_timestamps(tmp_path, bad_ts):
    db = tmp_path / "memory.db"
    _bootstrap_memory_db(db)
    _insert(db, "good", "WebSearch", "2026-04-18T11:00:00+00:00")
    _insert(db, "bad", "WebSearch", bad_ts)
    adapter = SQLiteMemoryAdapter(db)
    result = adapter.latest_satisfaction_ts(["WebSearch"])
    # Malformed row may be ignored; at minimum the good row's timestamp is returned.
    assert result is not None
    assert result.year == 2026
