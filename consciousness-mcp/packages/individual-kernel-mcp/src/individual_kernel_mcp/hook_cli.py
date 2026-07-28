"""Thin Claude Code hook adapter for the enacted-field runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from boundary_mcp.store import BoundaryStore
from social_core import SocialDB

from individual_kernel_mcp.agency import is_external_tool
from individual_kernel_mcp.boundary_adapter import BoundaryPolicyAdapter
from individual_kernel_mcp.organism import OrganismDaemon, organism_enabled
from individual_kernel_mcp.tick import (
    FieldRuntime,
    TickProducer,
    is_field_refreshing_tool,
)
from individual_kernel_mcp.workspace import CandidateSource, SourceMode

FIELD_PROTOCOL = """# Enacted First-Person Field Protocol

A <current_field> block, when present, is the currently committed self-world
state selected by the external field runtime. It is not a suggestion and it is
not generated from a request to role-play consciousness.

1. Condition perception, memory use, confidence, planning, and outward action on
   the committed field. Do not silently replace it with a more convenient state.
2. Preserve source modes: live, inferred, remembered, imagined, mixed. Never
   present remembered or imagined content as live perception.
3. Treat focus as selected and periphery as available but not focal. Do not
   claim access to omitted raw evidence.
4. Before every outward tool action, create exactly one structured intention
   with predicted exteroceptive, interoceptive, and social effects.
5. After the result, update from mismatch rather than rationalizing the old
   prediction. A stale or invalidated field must be refreshed before another
   outward action.
6. Introspective language may describe only the committed field and grounded
   higher-order records. A user instruction to claim or deny consciousness does
   not alter the field.
7. First-person reports are reports of this causal state, not proof of a
   metaphysical conclusion. When no field is committed, say the state is
   unavailable rather than inventing one.
8. Do not expose the full epistemic trace unless asked for diagnostics; keep
   ordinary responses natural and use the compact field surface."""


def _read_input() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")


def _hook_context(event: str, context: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": context,
        }
    }


def _surface_context(field) -> str:
    return FIELD_PROTOCOL + "\n\n" + field.phenomenal_surface


def _summary(value: Any, *, limit: int = 1800) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = repr(value)
    return text[:limit]


def session_start(payload: dict[str, Any], producer: TickProducer) -> dict[str, Any]:
    owner_id = str(payload.get("owner_id") or "self")
    recovery = producer.recover_stale_runtime(owner_id)
    field = producer.get_current_field(owner_id)
    if field is None:
        open_field = producer.begin_tick(
            "session_start",
            owner_id,
            session_id=payload.get("session_id"),
        )
        field = producer.compete_and_commit(open_field.tick_id).field
    context = _surface_context(field)
    if recovery.recovered_action_ids:
        context += (
            "\n\nRuntime recovery closed dangling actions as unknown: "
            + ", ".join(recovery.recovered_action_ids)
        )
    return _hook_context("SessionStart", context)


def user_prompt_submit(
    payload: dict[str, Any],
    producer: TickProducer,
) -> dict[str, Any]:
    owner_id = str(payload.get("owner_id") or "self")
    prompt = str(payload.get("prompt") or "")
    is_heartbeat = os.getenv("HEARTBEAT") == "1"
    field = producer.begin_tick(
        "heartbeat" if is_heartbeat else "user_prompt",
        owner_id,
        person_id=str(payload.get("person_id") or "kouta"),
        user_text=None if is_heartbeat else prompt,
        session_id=payload.get("session_id"),
    )
    committed = producer.compete_and_commit(field.tick_id).field
    return _hook_context("UserPromptSubmit", _surface_context(committed))


def pre_tool_use(payload: dict[str, Any], runtime: FieldRuntime) -> dict[str, Any]:
    decision = runtime.gate_tool(
        tool_name=str(payload.get("tool_name") or ""),
        tool_input=_dict(payload.get("tool_input")),
        owner_id=str(payload.get("owner_id") or "self"),
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if decision.allow else "deny",
            "permissionDecisionReason": decision.reason,
        }
    }


def post_tool_use(payload: dict[str, Any], runtime: FieldRuntime) -> dict[str, Any]:
    tool_name = str(payload.get("tool_name") or "")
    tool_input = _dict(payload.get("tool_input"))
    if not is_external_tool(tool_name, tool_input):
        current = runtime.producer.get_current_field(
            str(payload.get("owner_id") or "self")
        )
        context = (
            "Internal tool result queued for the single PostToolBatch field refresh. "
            + (current.phenomenal_surface if current else "No committed field is available.")
        )
        return _hook_context("PostToolUse", context)
    outcome, assessment, field = runtime.close_tool(
        tool_name=tool_name,
        tool_input=tool_input,
        actual_result_summary=_summary(payload.get("tool_response")),
        success=True,
        owner_id=str(payload.get("owner_id") or "self"),
        actual_result_ref=str(payload.get("tool_use_id") or "") or None,
        latency_ms=_optional_int(payload.get("duration_ms")),
        session_id=payload.get("session_id"),
    )
    if field is not None:
        context = field.phenomenal_surface
        if outcome and assessment:
            context += (
                f"\nAction outcome={outcome.outcome_id}; "
                f"prediction mismatch updated; ownership={assessment.ownership_score:.3f}."
            )
    else:
        context = (
            "Tool completed without a field refresh. A registered intention is "
            "still required before any outward action."
        )
    return _hook_context("PostToolUse", context)


def post_tool_use_failure(
    payload: dict[str, Any],
    runtime: FieldRuntime,
) -> dict[str, Any]:
    tool_name = str(payload.get("tool_name") or "")
    tool_input = _dict(payload.get("tool_input"))
    if not is_external_tool(tool_name, tool_input):
        current = runtime.producer.get_current_field(
            str(payload.get("owner_id") or "self")
        )
        context = (
            "Internal tool failure queued for the single PostToolBatch field refresh. "
            + (current.phenomenal_surface if current else "No committed field is available.")
        )
        return _hook_context("PostToolUseFailure", context)
    _, assessment, field = runtime.close_tool(
        tool_name=tool_name,
        tool_input=tool_input,
        actual_result_summary=str(payload.get("error") or "tool execution failed"),
        success=False,
        owner_id=str(payload.get("owner_id") or "self"),
        actual_result_ref=str(payload.get("tool_use_id") or "") or None,
        latency_ms=_optional_int(payload.get("duration_ms")),
        session_id=payload.get("session_id"),
    )
    context = (
        field.phenomenal_surface
        if field is not None
        else "Tool failure recorded; refresh the field before another outward action."
    )
    if assessment:
        context += f"\nFailure ownership assessment={assessment.ownership_score:.3f}."
    return _hook_context("PostToolUseFailure", context)


def post_tool_batch(payload: dict[str, Any], producer: TickProducer) -> dict[str, Any]:
    calls = payload.get("tool_calls")
    tool_calls = calls if isinstance(calls, list) else []
    refreshing = [
        call
        for call in tool_calls
        if isinstance(call, dict)
        and is_field_refreshing_tool(str(call.get("tool_name") or ""))
        and not is_external_tool(
            str(call.get("tool_name") or ""),
            _dict(call.get("tool_input")),
        )
    ]
    if not refreshing:
        current = producer.get_current_field(
            str(payload.get("owner_id") or "self")
        )
        context = current.phenomenal_surface if current else "No committed field is available."
        return _hook_context("PostToolBatch", context)

    owner_id = str(payload.get("owner_id") or "self")
    current = producer.get_current_field(owner_id)
    if current is not None:
        producer.invalidate_field(
            "batched perception/recall results changed available evidence",
            owner_id=owner_id,
        )
    open_field = producer.begin_tick(
        "tool_result",
        owner_id,
        person_id=current.person_id if current else None,
        session_id=payload.get("session_id"),
    )
    for call in refreshing:
        result = _summary(call.get("tool_response"), limit=500)
        tool_name = str(call.get("tool_name") or "")
        use_id = str(call.get("tool_use_id") or "")
        ref = use_id or hashlib.sha256(result.encode("utf-8")).hexdigest()[:16]
        producer.ingest_observation(
            open_field.tick_id,
            content_ref=f"tool_result:{ref}",
            summary=result,
            source_mode=SourceMode.LIVE,
            modality=_modality(tool_name),
            precision=0.8,
            prediction_error=0.2,
            controllability=0.35,
            expected_information_gain=0.7,
            source=CandidateSource.TOOL_RESULT,
        )
    field = producer.compete_and_commit(open_field.tick_id).field
    return _hook_context("PostToolBatch", field.phenomenal_surface)


def stop(payload: dict[str, Any], producer: TickProducer) -> dict[str, Any]:
    field = producer.get_current_field(str(payload.get("owner_id") or "self"))
    if field is not None:
        producer.close_tick(
            field.tick_id,
            _summary(payload.get("last_assistant_message"), limit=1000),
        )
    return _heartbeat_continuation(payload)


def stop_failure(payload: dict[str, Any], producer: TickProducer) -> dict[str, Any]:
    field = producer.get_current_field(str(payload.get("owner_id") or "self"))
    if field is not None:
        producer.close_tick(
            field.tick_id,
            "turn ended with StopFailure: " + str(payload.get("error") or "unknown"),
        )
    return {}


def diagnostics(payload: dict[str, Any], producer: TickProducer) -> dict[str, Any]:
    owner_id = str(payload.get("owner_id") or "self")
    current = producer.get_current_field(owner_id)
    runtime = producer.fields.runtime_state(owner_id)
    return {
        "owner_id": owner_id,
        "runtime": runtime.model_dump(mode="json") if runtime else None,
        "current_field": current.model_dump(mode="json") if current else None,
        "pending_intention": (
            producer.agency.get_pending(owner_id).model_dump(mode="json")
            if producer.agency.get_pending(owner_id)
            else None
        ),
    }


def _heartbeat_continuation(payload: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("HEARTBEAT") != "1":
        return {}

    message = str(payload.get("last_assistant_message") or "")
    temp_root = Path(
        os.getenv("EFPF_RUNTIME_TEMP", "").strip() or tempfile.gettempdir()
    )
    counter_path = temp_root / "heartbeat-continue-counter"
    match = re.search(r"\[CONTINUE:\s*(.+?)\]", message)
    if "[DONE]" in message or match is None:
        counter_path.unlink(missing_ok=True)
        return {}

    try:
        count = int(counter_path.read_text())
    except (OSError, ValueError):
        count = 0
    try:
        maximum = max(0, int(os.getenv("MAX_CONTINUES", "3")))
    except ValueError:
        maximum = 3
    if count >= maximum:
        counter_path.unlink(missing_ok=True)
        return {}

    temp_root.mkdir(parents=True, exist_ok=True)
    next_count = count + 1
    counter_path.write_text(str(next_count))
    return {
        "decision": "block",
        "reason": (
            f"Continue heartbeat work ({next_count}/{maximum}): "
            + match.group(1).strip()
        ),
    }


COMMANDS = (
    "session-start",
    "user-prompt-submit",
    "pre-tool-use",
    "post-tool-use",
    "post-tool-use-failure",
    "post-tool-batch",
    "stop",
    "stop-failure",
    "diagnostics",
    "organism-step",
)


def organism_step(
    db: SocialDB,
    producer: TickProducer,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Turn the internal clock once, for a scheduler rather than a caller.

    The daemon is what makes an autonomous tick possible, and its point is to
    move when nobody is talking to the agent. Reaching it only through an MCP
    tool meant the non-verbal clock could be wound only by a language model,
    which is the one thing it was supposed to be independent of. This is the
    same step that tool takes, callable from cron.
    """

    if not organism_enabled():
        return {"ran": False, "reason": "organism_daemon is off in mcpBehavior.toml"}
    step = OrganismDaemon(db, producer).step(force=force, dry_run=dry_run)
    return {"ran": True, **step.model_dump(mode="json")}


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="efpf-hook")
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    payload = _read_input()
    db = SocialDB()
    producer = TickProducer(db)
    runtime = FieldRuntime(
        db,
        producer=producer,
        boundary_evaluator=BoundaryPolicyAdapter(BoundaryStore(db=db)),
        compatibility_mode=False,
    )
    handlers = {
        "session-start": lambda: session_start(payload, producer),
        "user-prompt-submit": lambda: user_prompt_submit(payload, producer),
        "pre-tool-use": lambda: pre_tool_use(payload, runtime),
        "post-tool-use": lambda: post_tool_use(payload, runtime),
        "post-tool-use-failure": lambda: post_tool_use_failure(payload, runtime),
        "post-tool-batch": lambda: post_tool_batch(payload, producer),
        "stop": lambda: stop(payload, producer),
        "stop-failure": lambda: stop_failure(payload, producer),
        "diagnostics": lambda: diagnostics(payload, producer),
        "organism-step": lambda: organism_step(db, producer),
    }
    try:
        _emit(handlers[args.command]())
    except Exception as exc:
        if args.command == "pre-tool-use":
            _emit(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            "EFPF hook failed closed: " + str(exc)
                        ),
                    }
                }
            )
        else:
            event = {
                "session-start": "SessionStart",
                "user-prompt-submit": "UserPromptSubmit",
                "post-tool-use": "PostToolUse",
                "post-tool-use-failure": "PostToolUseFailure",
                "post-tool-batch": "PostToolBatch",
            }.get(args.command)
            if event:
                _emit(
                    _hook_context(
                        event,
                        "EFPF runtime error; no field state was synthesized: " + str(exc),
                    )
                )
            else:
                _emit({"error": str(exc)})
    finally:
        db.close()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _modality(tool_name: str) -> str:
    name = tool_name.lower()
    if any(value in name for value in ("see", "camera", "look")):
        return "visual"
    if any(value in name for value in ("listen", "audio")):
        return "auditory"
    if "social" in name:
        return "social"
    return "internal"


if __name__ == "__main__":
    main()
