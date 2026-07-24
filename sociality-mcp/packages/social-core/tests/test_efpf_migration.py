from __future__ import annotations

from social_core import SocialDB


def test_efpf_migration_is_idempotent_and_has_required_tables(tmp_path) -> None:
    db = SocialDB(tmp_path / "social.db")
    db.connect()
    db.close()

    reopened = SocialDB(tmp_path / "social.db")
    connection = reopened.connect()
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {
        "workspace_candidates",
        "enacted_fields",
        "field_runtime_state",
        "field_intentions",
        "field_action_outcomes",
        "field_transitions",
        "quality_signatures",
        "field_ablation_runs",
    } <= tables
    indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert {
        "idx_enacted_fields_tick",
        "idx_enacted_fields_previous",
        "idx_field_intentions_tick",
        "idx_field_intentions_field",
        "idx_field_action_outcomes_field",
        "idx_field_transitions_from_field",
        "idx_field_transitions_to_field",
        "idx_field_transitions_action",
    } <= indexes
    migrations = connection.execute(
        """
        SELECT COUNT(*) FROM schema_migrations
        WHERE name = '007_enacted_first_person_field'
        """
    ).fetchone()[0]
    assert migrations == 1
    index_migration = connection.execute(
        """
        SELECT COUNT(*) FROM schema_migrations
        WHERE name = '008_enacted_field_indexes'
        """
    ).fetchone()[0]
    assert index_migration == 1
    reopened.close()
