"""Tests for reference resolution and scene diffing."""


def _scene(ts, mug_salience=0.72, mug_relation="left_of:obj_laptop_1", include_blue=True):
    objects = [
        {
            "object_id": "obj_laptop_1",
            "label": "laptop",
            "attributes": {"open": True},
            "relative_position": ["center"],
            "salience": 0.91,
        },
        {
            "object_id": "obj_blue_mug_1",
            "label": "mug",
            "attributes": {"color": "blue"} if include_blue else {},
            "relative_position": [mug_relation],
            "salience": mug_salience,
        },
        {
            "object_id": "obj_white_mug_2",
            "label": "mug",
            "attributes": {"color": "white"},
            "relative_position": ["right_of:obj_laptop_1"],
            "salience": 0.44,
        },
    ]
    return {
        "ts": ts,
        "camera_pose": {"pan_deg": 12.0, "tilt_deg": -6.0, "zoom": 1.0},
        "scene_summary": "Kouta is at his desk with two mugs around the laptop.",
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
        "objects": objects,
    }


def test_resolve_that_mug_uses_recency_and_salience(store):
    store.ingest_scene_parse(_scene("2026-04-15T20:00:00+09:00", mug_salience=0.40))
    store.ingest_scene_parse(_scene("2026-04-15T20:02:00+09:00", mug_salience=0.85))
    store.set_joint_focus(person_id="kouta", target_id="obj_blue_mug_1", initiator="human")

    result = store.resolve_reference(expression="that mug", person_id="kouta")

    assert result.matches[0].object_id == "obj_blue_mug_1"
    assert result.matches[0].confidence >= 0.6


def test_resolve_blue_mug_uses_attribute(store):
    store.ingest_scene_parse(_scene("2026-04-15T20:00:00+09:00"))

    result = store.resolve_reference(expression="その青いマグ", person_id="kouta")

    assert result.matches[0].object_id == "obj_blue_mug_1"
    assert any("color=blue" in why for why in result.matches[0].why)


def test_scene_diff_detects_disappearance_and_movement(store):
    store.ingest_scene_parse(
        _scene("2026-04-15T20:00:00+09:00", mug_relation="left_of:obj_laptop_1")
    )
    later = _scene("2026-04-15T20:10:00+09:00", mug_relation="right_of:obj_laptop_1")
    later["objects"] = [obj for obj in later["objects"] if obj["object_id"] != "obj_white_mug_2"]
    store.ingest_scene_parse(later)

    changes = store.compare_recent_scenes(person_id="kouta", window_minutes=30)["changes"]

    assert any(
        "moved from left_of:obj_laptop_1 to right_of:obj_laptop_1" in change for change in changes
    )
    assert any("mug disappeared from view" in change for change in changes)


def test_no_match_returns_low_confidence_ambiguity(store):
    store.ingest_scene_parse(_scene("2026-04-15T20:00:00+09:00"))

    result = store.resolve_reference(expression="the stapler", person_id="kouta")

    assert result.matches
    assert result.matches[0].confidence < 0.5
