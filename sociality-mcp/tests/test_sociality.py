"""Integration tests for the unified sociality MCP facade."""

from sociality_mcp import server


def test_sociality_facade_handles_social_state_and_relationship_flow():
    server.upsert_person(
        person_id="kouta",
        canonical_name="水島宏太",
        aliases=["こうちゃん"],
        role="companion",
    )
    server.ingest_social_event(
        {
            "ts": "2026-04-15T08:21:00+09:00",
            "source": "human_mcp",
            "kind": "human_utterance",
            "person_id": "kouta",
            "confidence": 0.98,
            "payload": {"text": "ちょっと集中したいから静かめで頼む"},
        }
    )
    server.ingest_social_event(
        {
            "ts": "2026-04-15T08:22:00+09:00",
            "source": "garmin",
            "kind": "health_summary",
            "person_id": "kouta",
            "confidence": 0.87,
            "payload": {"body_battery": 23, "energy": "low"},
        }
    )
    server.ingest_interaction(
        person_id="kouta",
        channel="voice",
        direction="human_to_ai",
        text="今日は会議多くて疲れた",
        ts="2026-04-15T19:12:00+09:00",
    )
    commitment = server.create_commitment(
        person_id="kouta",
        text="remind about dentist tomorrow morning",
        due_at="2026-04-16T08:00:00+09:00",
        source="conversation",
    )

    state = server.get_social_state(window_seconds=900, person_id="kouta", include_evidence=True)
    decision = server.should_interrupt(
        candidate_action="say",
        urgency="low",
        person_id="kouta",
        message_preview="お茶でも飲む？",
    )
    model = server.get_person_model(person_id="kouta")
    suggestions = server.suggest_followup(person_id="kouta", context="evening_checkin")

    assert state["availability"] in {"maybe_interruptible", "do_not_interrupt"}
    assert state["interrupt_cost"] >= 0.65
    assert decision["decision"] == "no"
    assert any(item["id"] == commitment["commitment_id"] for item in model["active_commitments"])
    assert suggestions["suggestions"]
    assert "会議多くて疲れた" in suggestions["suggestions"][0]["text"]


def test_sociality_facade_handles_joint_attention_and_boundary_gating():
    server.upsert_person(
        person_id="kouta",
        canonical_name="水島宏太",
        aliases=[],
        role="companion",
    )
    server.ingest_scene_parse(
        {
            "ts": "2026-04-15T20:00:00+09:00",
            "camera_pose": {"pan_deg": 12.0, "tilt_deg": -6.0, "zoom": 1.0},
            "scene_summary": (
                "Kouta is at his desk with a blue mug left of the laptop and a notebook on the"
                " right."
            ),
            "people": [
                {
                    "person_id": "kouta",
                    "display_name": "Kouta",
                    "relative_position": "center",
                    "distance": "near",
                    "gaze_target": "laptop",
                    "confidence": 0.93,
                }
            ],
            "objects": [
                {
                    "object_id": "obj_blue_mug_1",
                    "label": "mug",
                    "attributes": {"color": "blue"},
                    "relative_position": ["left_of:obj_laptop_1"],
                    "salience": 0.72,
                },
                {
                    "object_id": "obj_laptop_1",
                    "label": "laptop",
                    "attributes": {"open": True},
                    "relative_position": ["center"],
                    "salience": 0.91,
                },
            ],
        }
    )

    resolution = server.resolve_reference(
        expression="その青いマグ",
        person_id="kouta",
        lookback_frames=5,
    )
    focus = server.get_current_joint_focus(person_id="kouta")
    review = server.review_social_post(
        channel="x",
        text="今日の会議しんどそうやったな",
        scene_contains_face=False,
        person_mentions=["kouta"],
    )
    evaluation = server.evaluate_action(
        action_type="post_tweet",
        channel="x",
        person_id="kouta",
        context={
            "scene_contains_face": True,
            "time_local": "2026-04-15T23:40:00+09:00",
        },
        payload_preview={"text": "今日は疲れてそうやった"},
    )

    assert resolution["matches"][0]["object_id"] == "obj_blue_mug_1"
    assert focus["focus_target"] == "obj_laptop_1"
    assert review["recommendation"] == "rewrite"
    assert evaluation["decision"] == "deny"


def test_sociality_facade_handles_self_narrative_tools():
    server.ingest_social_event(
        {
            "ts": "2026-04-15T07:30:00+00:00",
            "source": "manual",
            "kind": "scene_parse",
            "person_id": "kouta",
            "confidence": 0.91,
            "payload": {"scene_summary": "morning desk check"},
        }
    )

    daybook = server.append_daybook(day="2026-04-15")
    summary = server.get_self_summary()
    arcs = server.list_active_arcs()
    reflection = server.reflect_on_change(horizon_days=7)

    assert daybook["day"] == "2026-04-15"
    assert summary["summary"]
    assert arcs
    assert reflection["summary"]
