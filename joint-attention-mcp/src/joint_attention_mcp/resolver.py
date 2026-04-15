"""Reference resolution over recent structured scene parses."""

from __future__ import annotations

from typing import Any

from social_core import clamp01, confidence_from_evidence

from .schemas import ReferenceMatch

COLOR_MAP = {
    "blue": {"blue", "青"},
    "red": {"red", "赤"},
    "green": {"green", "緑"},
    "white": {"white", "白"},
    "black": {"black", "黒"},
}

LABEL_ALIASES = {
    "mug": {"mug", "cup", "マグ", "コップ"},
    "laptop": {"laptop", "pc", "computer", "ノートpc", "ノートパソコン"},
    "notebook": {"notebook", "note", "ノート"},
}

RELATION_MARKERS = {
    "left": {"left", "左"},
    "right": {"right", "右"},
}


def resolve_reference(
    expression: str,
    objects_by_frame: list[list[dict[str, Any]]],
    prior_focus: str | None = None,
) -> list[ReferenceMatch]:
    """Rank scene objects against a referring expression."""

    expr = expression.lower().strip()
    ranked: dict[str, tuple[float, list[str]]] = {}
    for frame_index, objects in enumerate(objects_by_frame):
        recency_bonus = max(0.0, 0.25 - frame_index * 0.04)
        for obj in objects:
            score = 0.0
            reasons: list[str] = []
            label = str(obj["label"]).lower()
            label_family = _label_family(label)
            if label_family and any(alias in expr for alias in LABEL_ALIASES[label_family]):
                score += 0.45
                reasons.append(f"label={label}")
            elif _is_deictic(expr):
                score += 0.18
                reasons.append("deictic reference")

            color = str(obj["attributes"].get("color", "")).lower()
            if color and _matches_color(expr, color):
                score += 0.25
                reasons.append(f"attribute color={color}")

            relation_hit = _relation_match(expr, obj["relative_position"])
            if relation_hit:
                score += 0.12
                reasons.append(relation_hit)

            salience = float(obj.get("salience", 0.0))
            if salience > 0:
                score += salience * 0.2
                if salience >= 0.6:
                    reasons.append("recently salient")

            score += recency_bonus
            if recency_bonus:
                reasons.append("recent frame")

            if prior_focus and obj["object_id"] == prior_focus:
                score += 0.18
                reasons.append("matches prior joint focus")

            if score <= 0:
                continue
            confidence = clamp01(score)
            current = ranked.get(obj["object_id"])
            if current is None or confidence > current[0]:
                ranked[obj["object_id"]] = (confidence, reasons)

    ordered = sorted(
        (
            ReferenceMatch(object_id=object_id, confidence=score, why=reasons)
            for object_id, (score, reasons) in ranked.items()
        ),
        key=lambda item: item.confidence,
        reverse=True,
    )
    if not ordered:
        return []
    top = ordered[0].confidence
    if top < 0.35:
        return ordered[:2]
    return ordered[:3]


def infer_joint_focus(
    latest_frame_objects: list[dict[str, Any]],
    latest_people: list[dict[str, Any]],
    explicit_focus: dict[str, Any] | None,
) -> dict[str, Any]:
    """Infer current joint focus from explicit records or gaze targets."""

    if explicit_focus is not None:
        return {
            "focus_target": explicit_focus["target_id"],
            "confidence": float(explicit_focus["confidence"]),
            "based_on": list(explicit_focus["based_on"]),
        }
    for person in latest_people:
        gaze_target = str(person.get("gaze_target") or "").lower()
        if not gaze_target:
            continue
        for obj in latest_frame_objects:
            label = str(obj["label"]).lower()
            if gaze_target == label or gaze_target in label:
                return {
                    "focus_target": obj["object_id"],
                    "confidence": confidence_from_evidence(
                        [person.get("confidence", 0.5), obj.get("salience", 0.4)], 0.1
                    ),
                    "based_on": [f"person gaze_target={gaze_target}", "ongoing scene context"],
                }
    return {
        "focus_target": None,
        "confidence": 0.18,
        "based_on": ["no explicit joint focus evidence"],
    }


def compare_scenes(earlier: list[dict[str, Any]], later: list[dict[str, Any]]) -> list[str]:
    """Produce a compact scene diff."""

    earlier_map = {item["object_id"]: item for item in earlier}
    later_map = {item["object_id"]: item for item in later}
    changes: list[str] = []
    for object_id, obj in earlier_map.items():
        if object_id not in later_map:
            changes.append(f"{obj['label']} disappeared from view")
            continue
        later_obj = later_map[object_id]
        if obj["relative_position"] != later_obj["relative_position"]:
            before = obj["relative_position"][0] if obj["relative_position"] else "unknown"
            after = (
                later_obj["relative_position"][0] if later_obj["relative_position"] else "unknown"
            )
            changes.append(f"{obj['label']} moved from {before} to {after}")
        if obj["attributes"].get("open") is False and later_obj["attributes"].get("open") is True:
            changes.append(f"{obj['label']} opened")
    for object_id, obj in later_map.items():
        if object_id not in earlier_map:
            changes.append(f"{obj['label']} appeared in view")
    return changes


def _label_family(label: str) -> str | None:
    for family, aliases in LABEL_ALIASES.items():
        if label in aliases:
            return family
    return None


def _is_deictic(expression: str) -> bool:
    return any(token in expression for token in ("that", "this", "その", "あの", "これ"))


def _matches_color(expression: str, color: str) -> bool:
    return any(alias in expression for alias in COLOR_MAP.get(color, {color}))


def _relation_match(expression: str, relations: list[str]) -> str | None:
    lowered_relations = [relation.lower() for relation in relations]
    for key, markers in RELATION_MARKERS.items():
        if any(marker in expression for marker in markers) and any(
            key in relation for relation in lowered_relations
        ):
            return lowered_relations[0]
    return None
