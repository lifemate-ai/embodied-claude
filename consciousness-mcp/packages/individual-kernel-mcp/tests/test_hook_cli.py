from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from social_core import SocialDB

from individual_kernel_mcp import tick
from individual_kernel_mcp.agency import ActionProposal
from individual_kernel_mcp.hook_cli import stop
from individual_kernel_mcp.tick import FieldRuntime, TickProducer


def _run_hook(tmp_path, command: str, payload: dict) -> dict:
    env = dict(os.environ)
    env["SOCIAL_DB_PATH"] = str(tmp_path / "hook-social.db")
    result = subprocess.run(
        [sys.executable, "-m", "individual_kernel_mcp.hook_cli", command],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    assert result.stderr == ""
    return json.loads(result.stdout)


def test_user_prompt_hook_emits_official_additional_context_shape(tmp_path) -> None:
    result = _run_hook(
        tmp_path,
        "user-prompt-submit",
        {
            "session_id": "session-1",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Please inspect the current work.",
        },
    )
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "UserPromptSubmit"
    assert "<current_field" in output["additionalContext"]
    assert "Enacted First-Person Field Protocol" in output["additionalContext"]


def test_pre_tool_hook_fails_closed_without_intention(tmp_path) -> None:
    _run_hook(
        tmp_path,
        "session-start",
        {
            "session_id": "session-1",
            "hook_event_name": "SessionStart",
            "source": "startup",
        },
    )
    result = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "session_id": "session-1",
            "hook_event_name": "PreToolUse",
            "tool_name": "mcp__tts__say",
            "tool_input": {"text": "hello"},
            "tool_use_id": "toolu_1",
        },
    )
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert "propose_field_action" in output["permissionDecisionReason"]


def test_pre_tool_stdin_json_covers_no_field_mismatch_valid_and_second(
    tmp_path,
) -> None:
    tool_name = "Write"
    tool_input = {"file_path": "/tmp/efpf-fixture", "content": "one"}
    no_field = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": "write-0",
        },
    )
    assert no_field["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "no COMMITTED field" in no_field["hookSpecificOutput"][
        "permissionDecisionReason"
    ]

    _run_hook(
        tmp_path,
        "user-prompt-submit",
        {
            "session_id": "session-gate",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "write fixture",
        },
    )
    no_intention = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": "write-1",
        },
    )
    assert no_intention["hookSpecificOutput"]["permissionDecision"] == "deny"

    db = SocialDB(tmp_path / "hook-social.db")
    producer = TickProducer(
        db,
        interoception_path=tmp_path / "interoception.json",
        desires_path=tmp_path / "desires.json",
    )
    field = producer.get_current_field()
    assert field is not None
    FieldRuntime(db, producer=producer).propose_action(
        ActionProposal(
            field_id=field.field_id,
            tool_name=tool_name,
            tool_input=tool_input,
            goal="write fixture once",
        )
    )
    db.close()

    mismatch = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": {**tool_input, "content": "different"},
            "tool_use_id": "write-2",
        },
    )
    valid = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": "write-3",
        },
    )
    second = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": "write-4",
        },
    )

    assert mismatch["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "hash mismatch" in mismatch["hookSpecificOutput"][
        "permissionDecisionReason"
    ]
    assert valid["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert second["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "one outward action" in second["hookSpecificOutput"][
        "permissionDecisionReason"
    ]


def test_default_interoception_path_uses_platform_temp_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("INTEROCEPTION_STATE_PATH", raising=False)
    monkeypatch.setattr(tick.tempfile, "gettempdir", lambda: str(tmp_path))

    assert tick.default_interoception_path() == (
        tmp_path / "interoception_state.json"
    )


def test_heartbeat_stop_continuation_is_cross_platform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HEARTBEAT", "1")
    monkeypatch.setenv("EFPF_RUNTIME_TEMP", str(tmp_path))
    db = SocialDB(tmp_path / "hook-social.db")
    producer = TickProducer(
        db,
        interoception_path=tmp_path / "interoception.json",
        desires_path=tmp_path / "desires.json",
    )

    result = stop(
        {
            "stop_hook_active": False,
            "last_assistant_message": "[CONTINUE: finish the release check]",
        },
        producer,
    )

    assert result["decision"] == "block"
    assert "finish the release check" in result["reason"]
    db.close()
