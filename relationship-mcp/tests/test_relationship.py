"""Tests for relationship abstractions."""

from relationship_mcp.store import RelationshipStore


def test_alias_matching_works(store):
    store.upsert_person(
        person_id="kouta",
        canonical_name="水島宏太",
        aliases=["こうちゃん", "宏太"],
        role="companion",
    )

    assert store.resolve_person_id("宏太") == "kouta"
    assert store.resolve_person_id("こうちゃん") == "kouta"


def test_commitments_survive_restart(tmp_path):
    db_path = tmp_path / "social.db"
    first = RelationshipStore(db_path)
    first.upsert_person(person_id="kouta", canonical_name="水島宏太", aliases=[], role="companion")
    created = first.create_commitment(
        person_id="kouta",
        text="remind about dentist tomorrow morning",
        due_at="2026-04-16T08:00:00+09:00",
        source="conversation",
    )
    first.close()

    second = RelationshipStore(db_path)
    model = second.get_person_model(person_id="kouta")
    second.close()

    assert any(commitment.id == created["commitment_id"] for commitment in model.active_commitments)


def test_repeated_mentions_of_future_task_create_open_loop(store):
    store.upsert_person(person_id="kouta", canonical_name="水島宏太", aliases=[], role="companion")
    store.ingest_interaction(
        person_id="kouta",
        channel="voice",
        direction="human_to_ai",
        text="明日の PR review 忘れんようにしたい",
        ts="2026-04-15T19:12:00+09:00",
    )
    store.ingest_interaction(
        person_id="kouta",
        channel="voice",
        direction="human_to_ai",
        text="PR review 明日やるの覚えといて",
        ts="2026-04-15T19:20:00+09:00",
    )
    loops = store.list_open_loops(person_id="kouta")

    assert len(loops) == 1
    assert loops[0].topic == "pr review"


def test_completed_commitment_disappears_from_active_list(store):
    store.upsert_person(person_id="kouta", canonical_name="水島宏太", aliases=[], role="companion")
    created = store.create_commitment(
        person_id="kouta",
        text="remind about dentist tomorrow morning",
        due_at="2026-04-16T08:00:00+09:00",
        source="conversation",
    )
    store.complete_commitment(created["commitment_id"])
    model = store.get_person_model(person_id="kouta")

    assert all(commitment.id != created["commitment_id"] for commitment in model.active_commitments)


def test_person_model_stays_compact_and_followup_uses_same_day_disclosure(store):
    store.upsert_person(
        person_id="kouta", canonical_name="水島宏太", aliases=["こうちゃん"], role="companion"
    )
    store.record_boundary(
        person_id="kouta",
        kind="communication",
        rule="quiet_after_midnight",
        source_text="夜中は静かめで頼む",
    )
    store.ingest_interaction(
        person_id="kouta",
        channel="voice",
        direction="human_to_ai",
        text="今日は会議多くて疲れた",
        ts="2026-04-15T19:12:00+09:00",
    )
    model = store.get_person_model(person_id="kouta")
    suggestions = store.suggest_followup(person_id="kouta", context="evening_checkin")

    assert "今日は会議多くて疲れた" not in model.relationship_summary
    assert len(model.relationship_summary) < 120
    assert "prefers gentle brief nudges while working" in model.salient_preferences
    assert "会議多くて疲れた" in suggestions[0].text
