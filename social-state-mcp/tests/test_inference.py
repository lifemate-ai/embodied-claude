"""Tests for social state inference."""

from social_state_mcp.inference import should_interrupt_result


def _ingest(store, event):
    store.ingest_social_event(event)


def test_focused_work_low_body_battery_reduces_interruptibility(store):
    _ingest(
        store,
        {
            "ts": "2026-04-15T08:10:00+09:00",
            "source": "camera",
            "kind": "scene_parse",
            "person_id": "kouta",
            "confidence": 0.9,
            "payload": {
                "scene_summary": "Kouta is at his desk working on the laptop.",
                "activity": "working",
            },
        },
    )
    _ingest(
        store,
        {
            "ts": "2026-04-15T08:12:00+09:00",
            "source": "garmin",
            "kind": "health_summary",
            "person_id": "kouta",
            "confidence": 0.94,
            "payload": {"body_battery": 21},
        },
    )
    state = store.get_social_state(window_seconds=900, person_id="kouta")

    assert state.activity == "working"
    assert state.energy == "low"
    assert state.availability in {"maybe_interruptible", "do_not_interrupt"}
    assert state.interrupt_cost >= 0.5


def test_recent_direct_question_sets_awaiting_reply(store):
    _ingest(
        store,
        {
            "ts": "2026-04-15T19:12:00+09:00",
            "source": "human_mcp",
            "kind": "human_utterance",
            "person_id": "kouta",
            "confidence": 0.99,
            "payload": {"text": "その PR どう見る？"},
        },
    )
    state = store.get_social_state(window_seconds=900, person_id="kouta")

    assert state.interaction_phase == "awaiting_reply"
    assert state.availability == "interruptible"


def test_sleeping_hours_without_direct_address_do_not_interrupt(store):
    _ingest(
        store,
        {
            "ts": "2026-04-16T01:10:00+09:00",
            "source": "camera",
            "kind": "scene_parse",
            "person_id": "kouta",
            "confidence": 0.78,
            "payload": {"scene_summary": "The room is dark and quiet."},
        },
    )
    state = store.get_social_state(window_seconds=900, person_id="kouta")

    assert state.activity == "sleeping"
    assert state.availability == "do_not_interrupt"


def test_repeated_nudges_raise_interrupt_cost(store):
    for minute in (0, 10, 20):
        _ingest(
            store,
            {
                "ts": f"2026-04-15T18:{minute:02d}:00+09:00",
                "source": "agent",
                "kind": "touchpoint",
                "person_id": "kouta",
                "confidence": 0.7,
                "payload": {"action": "nudge_human"},
            },
        )
    state = store.get_social_state(window_seconds=3600, person_id="kouta")
    decision = should_interrupt_result(state, candidate_action="say", urgency="low")

    assert state.interrupt_cost >= 0.55
    assert decision.decision == "no"


def test_absence_of_evidence_stays_unknown(store):
    state = store.get_social_state(window_seconds=900, person_id="kouta")

    assert state.activity == "unknown"
    assert state.energy == "unknown"
    assert state.affect_guess.label == "uncertain"
