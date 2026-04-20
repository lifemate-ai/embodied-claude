"""Tests for the shared SQLite store."""

from social_core.db import DEFAULT_SOCIAL_DB_PATH, SocialDB, get_social_db_path


def test_get_social_db_path_uses_env(monkeypatch, tmp_path):
    override = tmp_path / "override.db"
    monkeypatch.setenv("SOCIAL_DB_PATH", str(override))
    assert get_social_db_path() == override
    monkeypatch.delenv("SOCIAL_DB_PATH")
    assert get_social_db_path() == DEFAULT_SOCIAL_DB_PATH


def test_migrations_are_idempotent(temp_db_path):
    from social_core.migrations import MIGRATIONS

    db = SocialDB(temp_db_path)
    db.connect()
    db.connect()
    tables = {
        row["name"] for row in db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "events" in tables
    assert "schema_migrations" in tables
    assert "agent_experiences" in tables
    assert "private_reflections" in tables
    assert "interpretation_shifts" in tables
    applied = db.fetchall("SELECT name FROM schema_migrations")
    assert len(applied) == len(MIGRATIONS)
    db.close()
