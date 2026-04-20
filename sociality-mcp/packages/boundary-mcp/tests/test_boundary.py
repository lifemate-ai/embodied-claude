"""Tests for boundary gating."""

from pathlib import Path

from social_core import SocialEventCreate


def test_get_policy_path_walks_up_from_cwd(tmp_path: Path, monkeypatch):
    """If cwd has no policy, climb parents to find the repo-root socialPolicy.toml."""
    from boundary_mcp.policy import get_policy_path

    repo_root = tmp_path / "fake_repo"
    sub_dir = repo_root / "sociality-mcp"
    sub_dir.mkdir(parents=True)
    policy_file = repo_root / "socialPolicy.toml"
    policy_file.write_text('[global]\ntimezone = "Asia/Tokyo"\n', encoding="utf-8")

    monkeypatch.delenv("SOCIAL_POLICY_PATH", raising=False)
    monkeypatch.chdir(sub_dir)

    found = get_policy_path()
    assert found.resolve() == policy_file.resolve()


def test_get_policy_path_prefers_env_over_walk(tmp_path: Path, monkeypatch):
    from boundary_mcp.policy import get_policy_path

    override = tmp_path / "custom.toml"
    override.write_text("", encoding="utf-8")
    monkeypatch.setenv("SOCIAL_POLICY_PATH", str(override))
    assert get_policy_path().resolve() == override.resolve()


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


def test_quiet_hours_uses_policy_timezone_utc_input(store):
    # Spec §8.2 acceptance: 16:30Z is 01:30 JST and must be quiet.
    quiet_state = store.get_quiet_mode_state(ts="2026-04-18T16:30:00Z")
    assert quiet_state.active is True

    # 23:30Z is 08:30 JST and must NOT be quiet.
    awake_state = store.get_quiet_mode_state(ts="2026-04-18T23:30:00Z")
    assert awake_state.active is False


def test_quiet_hours_uses_policy_timezone_through_evaluate(store):
    # Low-urgency speech at 16:30Z (= 01:30 JST) must be denied because
    # boundary-mcp now resolves the policy timezone before comparing windows.
    result = store.evaluate_action(
        action_type="say",
        person_id="kouta",
        context={"time_local": "2026-04-18T16:30:00Z"},
        payload_preview={"text": "お茶でも飲む？"},
        urgency="low",
    )
    assert result.decision == "deny"
    assert any("quiet hours" in reason for reason in result.reasons)


def test_review_social_post_flags_private_state(store):
    review = store.review_social_post(
        channel="x",
        text="今日の会議しんどそうやったな",
        scene_contains_face=False,
        person_mentions=["kouta"],
    )

    assert review.risk_level == "medium"
    assert review.recommendation == "rewrite"
