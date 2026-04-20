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


def test_policy_timezone_applies_to_utc_timestamp(tmp_path):
    """UTC events in JST's late-night window must resolve to do_not_interrupt."""
    from social_state_mcp.store import SocialStateStore

    tz_store = SocialStateStore(
        tmp_path / "tz.db",
        quiet_hours_windows=["00:00-07:00"],
        policy_timezone="Asia/Tokyo",
    )
    try:
        # 16:30 UTC = 01:30 JST — inside quiet window.
        tz_store.ingest_social_event(
            {
                "ts": "2026-04-18T16:30:00Z",
                "source": "camera",
                "kind": "scene_parse",
                "person_id": "kouta",
                "confidence": 0.8,
                "payload": {"scene_summary": "Room is dim but not dark."},
            }
        )
        state = tz_store.get_social_state(window_seconds=900, person_id="kouta")
        assert state.availability == "do_not_interrupt"
        assert any("quiet hours" in item for item in state.evidence)
    finally:
        tz_store.close()


def test_policy_timezone_awake_hours_stay_interactive(tmp_path):
    """23:30 UTC = 08:30 JST — not quiet; availability may be permissive."""
    from social_state_mcp.store import SocialStateStore

    tz_store = SocialStateStore(
        tmp_path / "tz2.db",
        quiet_hours_windows=["00:00-07:00"],
        policy_timezone="Asia/Tokyo",
    )
    try:
        tz_store.ingest_social_event(
            {
                "ts": "2026-04-18T23:30:00Z",
                "source": "human_mcp",
                "kind": "human_utterance",
                "person_id": "kouta",
                "confidence": 0.99,
                "payload": {"text": "おはよう"},
            }
        )
        state = tz_store.get_social_state(window_seconds=900, person_id="kouta")
        # Quiet-hour boost must NOT apply at 08:30 JST.
        assert "local time is within quiet hours" not in state.evidence
    finally:
        tz_store.close()
