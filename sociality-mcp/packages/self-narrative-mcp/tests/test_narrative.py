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
