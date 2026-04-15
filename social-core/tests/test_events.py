"""Tests for the shared event store."""

from social_core.events import EventStore
from social_core.models import SocialEventCreate


def test_ingest_deduplicates_by_source_and_correlation_id(social_db):
    store = EventStore(social_db)
    event = SocialEventCreate(
        ts="2026-04-15T08:21:00+09:00",
        source="human_mcp",
        kind="human_utterance",
        person_id="kouta",
        correlation_id="abc123",
        confidence=0.98,
        payload={"text": "静かめで頼む"},
    )
    first = store.ingest(event)
    second = store.ingest(event)

    assert first.event_id == second.event_id
    rows = social_db.fetchall("SELECT COUNT(*) AS count FROM events")
    assert rows[0]["count"] == 1


def test_replay_orders_by_timestamp(social_db):
    store = EventStore(social_db)
    results = store.replay(
        [
            {
                "ts": "2026-04-15T08:22:00+09:00",
                "source": "manual",
                "kind": "touchpoint",
                "confidence": 0.7,
                "payload": {"text": "second"},
            },
            {
                "ts": "2026-04-15T08:21:00+09:00",
                "source": "manual",
                "kind": "touchpoint",
                "confidence": 0.7,
                "payload": {"text": "first"},
            },
        ]
    )

    assert [event.payload_json["text"] for event in results] == ["first", "second"]
