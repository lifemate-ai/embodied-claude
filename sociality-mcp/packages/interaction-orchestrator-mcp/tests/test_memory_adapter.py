"""Tests for the orchestrator memory adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from interaction_orchestrator_mcp.memory_adapter import (
    NullMemoryAdapter,
    SQLiteMemoryAdapter,
    _extract_keywords,
    _use_policy_for,
    make_default_adapter,
)


def _bootstrap(path: Path) -> None:
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


def _insert(path: Path, **kwargs):
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO memories(id, content, normalized_content, timestamp, "
            "emotion, importance, category) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                kwargs["id"],
                kwargs["content"],
                kwargs["content"].lower(),
                kwargs["timestamp"],
                kwargs.get("emotion", "neutral"),
                kwargs.get("importance", 3),
                kwargs.get("category", "daily"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


class TestKeywordExtraction:
    def test_strips_stopwords_and_punctuation(self):
        keywords = _extract_keywords("kokone.one の DNS また見ておく？")
        assert "kokone" in " ".join(keywords).lower() or "kokone.one" in keywords
        assert "DNS" in keywords or "dns" in [k.lower() for k in keywords]
        assert "の" not in keywords

    def test_returns_empty_for_empty_input(self):
        assert _extract_keywords("") == []
        assert _extract_keywords("   ") == []


class TestUsePolicy:
    def test_philosophical_with_include_private_false_is_hidden(self):
        policy, _ = _use_policy_for(
            category="philosophical", importance=3, include_private=False
        )
        assert policy == "do_not_surface"

    def test_philosophical_with_include_private_true_is_background(self):
        policy, _ = _use_policy_for(
            category="philosophical", importance=3, include_private=True
        )
        assert policy == "background_only"

    def test_high_importance_sensitive_stays_mentionable(self):
        policy, _ = _use_policy_for(
            category="feeling", importance=5, include_private=True
        )
        assert policy == "mentionable"

    def test_daily_is_always_mentionable(self):
        policy, _ = _use_policy_for(
            category="daily", importance=3, include_private=True
        )
        assert policy == "mentionable"


class TestSQLiteAdapter:
    def test_returns_empty_when_db_missing(self, tmp_path):
        adapter = SQLiteMemoryAdapter(tmp_path / "missing.db")
        hits = adapter.recall_for_response(user_text="hello")
        assert hits == []

    def test_returns_empty_when_user_text_is_none(self, tmp_path):
        db = tmp_path / "memory.db"
        _bootstrap(db)
        adapter = SQLiteMemoryAdapter(db)
        assert adapter.recall_for_response(user_text=None) == []

    def test_ranks_by_keyword_match_and_importance(self, tmp_path):
        db = tmp_path / "memory.db"
        _bootstrap(db)
        _insert(
            db,
            id="a",
            content="kokone.one の DNS 修復した (low importance)",
            timestamp="2026-04-19T10:00:00+00:00",
            importance=2,
        )
        _insert(
            db,
            id="b",
            content="kokone.one の DNS の根本原因を特定 (high importance)",
            timestamp="2026-04-19T10:00:00+00:00",
            importance=5,
        )
        _insert(
            db,
            id="c",
            content="全然関係ない memory",
            timestamp="2026-04-19T10:00:00+00:00",
            importance=5,
        )
        adapter = SQLiteMemoryAdapter(db)
        hits = adapter.recall_for_response(user_text="kokone.one の DNS")
        assert [h.memory_id for h in hits[:2]] == ["b", "a"]
        assert all(h.memory_id != "c" for h in hits)

    def test_exclude_categories_filters(self, tmp_path):
        db = tmp_path / "memory.db"
        _bootstrap(db)
        _insert(
            db,
            id="tech",
            content="DNS 修復した",
            timestamp="2026-04-19T10:00:00+00:00",
            category="technical",
            importance=4,
        )
        _insert(
            db,
            id="feel",
            content="DNS 直した時の気持ち",
            timestamp="2026-04-19T10:00:00+00:00",
            category="feeling",
            importance=4,
        )
        adapter = SQLiteMemoryAdapter(db)
        hits = adapter.recall_for_response(
            user_text="DNS", exclude_categories={"feeling"}
        )
        assert all(h.memory_id != "feel" for h in hits)

    def test_include_private_false_hides_philosophical(self, tmp_path):
        db = tmp_path / "memory.db"
        _bootstrap(db)
        _insert(
            db,
            id="p",
            content="存在について考えた philosophical",
            timestamp="2026-04-19T10:00:00+00:00",
            category="philosophical",
            importance=3,
        )
        adapter = SQLiteMemoryAdapter(db)
        hits = adapter.recall_for_response(
            user_text="philosophical 存在", include_private=False
        )
        assert hits == []

    def test_use_policy_assigned_per_hit(self, tmp_path):
        db = tmp_path / "memory.db"
        _bootstrap(db)
        _insert(
            db,
            id="core",
            content="コア記憶 identity",
            timestamp="2026-04-19T10:00:00+00:00",
            category="core",
            importance=5,
        )
        adapter = SQLiteMemoryAdapter(db)
        hits = adapter.recall_for_response(user_text="identity")
        assert hits
        assert hits[0].use_policy == "mentionable"
        assert hits[0].reason


class TestNullAdapter:
    def test_always_returns_empty(self):
        adapter = NullMemoryAdapter()
        assert adapter.recall_for_response(user_text="anything") == []


class TestFactory:
    def test_respects_explicit_null(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_MEMORY_BACKEND", "null")
        assert isinstance(make_default_adapter(), NullMemoryAdapter)

    def test_returns_null_when_no_db(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ORCHESTRATOR_MEMORY_BACKEND", raising=False)
        monkeypatch.setenv("MEMORY_DB_FILE", str(tmp_path / "nope.db"))
        assert isinstance(make_default_adapter(), NullMemoryAdapter)

    def test_returns_sqlite_when_db_present(self, monkeypatch, tmp_path):
        db = tmp_path / "memory.db"
        _bootstrap(db)
        monkeypatch.setenv("MEMORY_DB_FILE", str(db))
        monkeypatch.delenv("ORCHESTRATOR_MEMORY_BACKEND", raising=False)
        assert isinstance(make_default_adapter(), SQLiteMemoryAdapter)
