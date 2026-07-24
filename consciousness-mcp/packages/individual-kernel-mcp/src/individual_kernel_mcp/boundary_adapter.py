"""Adapter from exact tool calls to the existing social boundary policy."""

from __future__ import annotations

from typing import Any

from boundary_mcp.store import BoundaryStore
from social_core.time import utc_now

from individual_kernel_mcp.enacted_field import EnactedField


class BoundaryPolicyAdapter:
    """Callable boundary evaluator used by `FieldRuntime`."""

    def __init__(self, store: BoundaryStore) -> None:
        self._store = store

    def __call__(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        field: EnactedField,
    ) -> tuple[bool, str]:
        action_type, channel = _classify_action(tool_name)
        result = self._store.evaluate_action(
            action_type=action_type,
            channel=channel,
            person_id=field.person_id,
            context={
                "time_local": utc_now(),
                "scene_contains_face": bool(
                    field.epistemic_trace.get("scene_contains_face")
                ),
                "health_safety": bool(tool_input.get("health_safety")),
                "field_id": field.field_id,
            },
            payload_preview=tool_input,
            urgency=str(tool_input.get("urgency") or "low"),
        )
        allowed = result.decision in {"allow", "allow_with_override"}
        reason = "; ".join(result.reasons) or result.decision
        return allowed, reason


def _classify_action(tool_name: str) -> tuple[str, str | None]:
    name = tool_name.lower()
    leaf = name.rsplit("__", 1)[-1]
    if any(token in leaf for token in ("say", "speak")):
        return "say", "voice"
    if any(token in leaf for token in ("post", "tweet", "repost")):
        return "post_tweet", "x"
    if "notify" in leaf or "nudge" in leaf:
        return "nudge_human", "notification"
    if any(token in leaf for token in ("look_", "move", "camera_go")):
        return "camera_motion", "camera"
    if tool_name in {"Write", "Edit", "NotebookEdit"}:
        return "file_write", "filesystem"
    if tool_name == "Bash":
        return "shell_side_effect", "terminal"
    return leaf or "external_tool", None
