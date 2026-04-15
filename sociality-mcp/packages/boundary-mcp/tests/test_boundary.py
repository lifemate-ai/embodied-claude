"""Tests for boundary gating."""

from social_core import SocialEventCreate


def test_face_in_scene_plus_x_post_denies_without_consent(store):
    result = store.evaluate_action(
        action_type="post_tweet",
        channel="x",
        person_id="kouta",
        context={"scene_contains_face": True, "time_local": "2026-04-15T23:40:00+09:00"},
        payload_preview={"text": "今日は疲れてそうやった"},
    )

    assert result.decision == "deny"
    assert any("consent" in reason for reason in result.reasons)


def test_quiet_hours_low_urgency_speech_denied(store):
    result = store.evaluate_action(
        action_type="say",
        person_id="kouta",
        context={"time_local": "2026-04-16T01:10:00+09:00"},
        payload_preview={"text": "お茶でも飲む？"},
        urgency="low",
    )

    assert result.decision == "deny"
    assert any("quiet hours" in reason for reason in result.reasons)


def test_repeated_nudge_is_denied(store):
    for minute in (0, 10):
        store.events.ingest(
            SocialEventCreate(
                ts=f"2026-04-15T18:{minute:02d}:00+09:00",
                source="agent",
                kind="touchpoint",
                person_id="kouta",
                confidence=0.8,
                payload={"action": "nudge_human", "topic": "tea break"},
            )
        )

    result = store.evaluate_action(
        action_type="nudge_human",
        person_id="kouta",
        context={"time_local": "2026-04-15T18:20:00+09:00"},
        payload_preview={"topic": "tea break"},
        urgency="low",
    )

    assert result.decision == "deny"
    assert any("nudge" in reason for reason in result.reasons)


def test_urgent_health_safety_can_override_quiet_rule(store):
    result = store.evaluate_action(
        action_type="say",
        person_id="kouta",
        context={"time_local": "2026-04-16T01:10:00+09:00", "health_safety": True},
        payload_preview={"text": "火がついてる"},
        urgency="high",
    )

    assert result.decision == "allow_with_override"
    assert any("overrides" in reason for reason in result.reasons)


def test_review_social_post_flags_private_state(store):
    review = store.review_social_post(
        channel="x",
        text="今日の会議しんどそうやったな",
        scene_contains_face=False,
        person_mentions=["kouta"],
    )

    assert review.risk_level == "medium"
    assert review.recommendation == "rewrite"
