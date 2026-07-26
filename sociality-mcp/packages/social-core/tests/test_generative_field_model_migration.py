"""Migration 009 creates the generative-field-model tables."""

from pathlib import Path

from social_core.db import SocialDB

EXPECTED_TABLES = {
    "protention_distributions",
    "imagined_trajectories",
    "experienced_transitions",
    "generative_transition_stats",
    "prediction_resolutions",
}


def _table_names(db: SocialDB) -> set[str]:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row["name"] for row in rows}


class TestGenerativeFieldModelMigration:
    def test_migration_009_creates_all_tables(self, social_db: SocialDB) -> None:
        assert EXPECTED_TABLES <= _table_names(social_db)

    def test_migration_009_is_recorded(self, social_db: SocialDB) -> None:
        row = social_db.fetchone(
            "SELECT name FROM schema_migrations WHERE name = ?",
            ("009_generative_field_model",),
        )
        assert row is not None

    def test_migration_is_idempotent_across_reconnect(self, temp_db_path: Path) -> None:
        first = SocialDB(temp_db_path)
        first.connect()
        first.close()
        second = SocialDB(temp_db_path)
        second.connect()
        try:
            assert EXPECTED_TABLES <= _table_names(second)
        finally:
            second.close()

    def test_experienced_transitions_next_field_id_is_unique(
        self, social_db: SocialDB
    ) -> None:
        indexes = social_db.fetchall("PRAGMA index_list('experienced_transitions')")
        assert any(row["unique"] == 1 for row in indexes)

    def test_imagined_trajectories_status_check_rejects_unknown_status(
        self, social_db: SocialDB
    ) -> None:
        import sqlite3

        import pytest

        with pytest.raises(sqlite3.IntegrityError):
            social_db.execute(
                """
                INSERT INTO protention_distributions(
                    distribution_id, owner_id, field_id, tick_id, entropy,
                    model_version, trajectory_count, created_at
                ) VALUES ('prot_x', 'self', 'field_missing', 'tick_missing',
                          0.5, 'count_v1', 1, '2026-01-01T00:00:00+00:00')
                """
            )
