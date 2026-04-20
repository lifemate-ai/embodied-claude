"""Tests for self narrative summaries."""

from social_core import SocialEventCreate


def test_append_daybook_and_active_arcs(store):
    store.events.ingest(
        SocialEventCreate(
            ts="2026-04-15T08:00:00+00:00",
            source="camera",
            kind="scene_parse",
            person_id="kouta",
            confidence=0.9,
            payload={"scene_summary": "Desk scene"},
        )
    )
    store.events.ingest(
        SocialEventCreate(
            ts="2026-04-15T09:00:00+00:00",
            source="voice",
            kind="human_utterance",
            person_id="kouta",
            confidence=0.95,
            payload={"text": "今日は会議多い"},
        )
    )

    daybook = store.append_daybook(day="2026-04-15")
    arcs = store.list_active_arcs()

    assert "2026-04-15" in daybook.summary
    assert arcs
    assert any("continuity" in arc.title or "daily life" in arc.title for arc in arcs)


def test_daybook_surfaces_agent_experiences_and_shifts(store):
    import json
    import uuid

    # Seed an agent experience and an interpretation shift on the same day.
    with store.db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO agent_experiences(
                experience_id, ts, person_id, kind, summary,
                felt_state_json, desires_before_json, desires_after_json,
                related_event_ids, related_memory_ids, artifacts_json,
                importance, privacy_level, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"exp_{uuid.uuid4().hex[:10]}",
                "2026-04-19T11:30:00+00:00",
                "kouta",
                "agent_response",
                "replied with v0.3 spec",
                "{}",
                "{}",
                "{}",
                "",
                "",
                "[]",
                4,
                "relationship",
                "2026-04-19T11:30:05+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO interpretation_shifts(
                shift_id, ts, person_id, topic,
                old_interpretation, new_interpretation, trigger,
                confidence, implications_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"shft_{uuid.uuid4().hex[:10]}",
                "2026-04-19T11:40:00+00:00",
                "kouta",
                "late-night behavior",
                "sample wording is a hard rule",
                "policy purpose is the rule",
                "Kouta clarified",
                0.92,
                json.dumps(["Honor the purpose, not the wording"]),
                "2026-04-19T11:40:05+00:00",
            ),
        )
        # Also ingest an event so append_daybook has a latest_ts.
    store.events.ingest(
        SocialEventCreate(
            ts="2026-04-19T11:00:00+00:00",
            source="voice",
            kind="human_utterance",
            person_id="kouta",
            confidence=0.99,
            payload={"text": "OK"},
        )
    )

    daybook = store.append_daybook(day="2026-04-19")
    assert any("replied with v0.3 spec" in item for item in daybook.concrete_events)
    assert any("late-night behavior" in item for item in daybook.noticed_changes)
    assert any("kouta" in moment for moment in daybook.relationship_moments)

    summary = store.get_self_summary()
    assert summary.recent_concrete_events
    assert summary.recent_interpretation_shifts
    assert summary.latest_daybook is not None


def test_self_summary_and_reflect_on_change(store):
    for day, text in [
        ("2026-04-14", "Desk scene"),
        ("2026-04-15", "Balcony scene"),
    ]:
        store.events.ingest(
            SocialEventCreate(
                ts=f"{day}T08:00:00+00:00",
                source="camera",
                kind="scene_parse",
                person_id="kouta",
                confidence=0.9,
                payload={"scene_summary": text},
            )
        )
        store.append_daybook(day=day)

    summary = store.get_self_summary()
    change = store.reflect_on_change(horizon_days=2)

    assert "Kokone" in summary.summary
    assert change.summary
