from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from social_core import SocialDB

from individual_kernel_mcp import tick
from individual_kernel_mcp.agency import ActionProposal
from individual_kernel_mcp.hook_cli import (
    UNREADABLE_INPUT_REASON,
    _emit,
    _read_input,
    kernel_server_configured,
    stop,
)
from individual_kernel_mcp.tick import FieldRuntime, TickProducer


def _run_hook(
    tmp_path,
    command: str,
    payload: dict | None,
    *,
    raw_input: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict:
    env = dict(os.environ)
    env["SOCIAL_DB_PATH"] = str(tmp_path / "hook-social.db")
    # Keep the hook off the machine's real recall endpoint; 0 means "none".
    env["MEMORY_HTTP_PORT"] = "0"
    env.update(env_overrides or {})
    result = subprocess.run(
        [sys.executable, "-m", "individual_kernel_mcp.hook_cli", command],
        input=json.dumps(payload) if raw_input is None else raw_input,
        text=True,
        # Claude Code speaks UTF-8 on the hook pipes; without this the parent
        # side of the test would encode with the locale (cp932 on Japanese
        # Windows) and never exercise the #152 path.
        encoding="utf-8",
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
    assert valid["hookSpecificOutput"]["permissionDecision"] == "allow", valid
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


_OWNER_TOOL = "Write"
_OWNER_INPUT = {"file_path": "/tmp/efpf-owner-fixture", "content": "parent"}


def _field_id(result: dict) -> str:
    context = result["hookSpecificOutput"]["additionalContext"]
    match = re.search(r'field_id="([^"]+)"', context)
    assert match is not None, context
    return match.group(1)


def _producer(tmp_path) -> tuple[SocialDB, TickProducer]:
    db = SocialDB(tmp_path / "hook-social.db")
    return db, TickProducer(
        db,
        interoception_path=tmp_path / "interoception.json",
        desires_path=tmp_path / "desires.json",
    )


def _register_intention(tmp_path, field_id: str) -> None:
    db, producer = _producer(tmp_path)
    FieldRuntime(db, producer=producer).propose_action(
        ActionProposal(
            field_id=field_id,
            tool_name=_OWNER_TOOL,
            tool_input=_OWNER_INPUT,
            goal="write the fixture once",
        )
    )
    db.close()


def _committed_field_id(tmp_path, owner_id: str) -> str | None:
    db, producer = _producer(tmp_path)
    field = producer.get_current_field(owner_id)
    db.close()
    return None if field is None else field.field_id


def _start_parent_and_register(tmp_path) -> str:
    _run_hook(
        tmp_path,
        "session-start",
        {
            "session_id": "parent-session",
            "hook_event_name": "SessionStart",
            "source": "startup",
        },
    )
    parent = _run_hook(
        tmp_path,
        "user-prompt-submit",
        {
            "session_id": "parent-session",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "write the fixture",
        },
    )
    parent_field_id = _field_id(parent)
    _register_intention(tmp_path, parent_field_id)
    return parent_field_id


def test_a_subagent_turn_leaves_the_parents_field_and_intention_alone(
    tmp_path,
) -> None:
    """Drive the hooks end to end the way a subagent actually arrives.

    Every hook used to act as owner "self", and only one COMMITTED field is
    allowed per owner, so a subagent's UserPromptSubmit took the parent's
    single slot. Resolving the owner from the session id gives the child its
    own field and its own intention slot; the parent keeps both.
    """
    parent_field_id = _start_parent_and_register(tmp_path)

    child = _run_hook(
        tmp_path,
        "user-prompt-submit",
        {
            "session_id": "child-session",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "look something up for the parent",
        },
    )
    assert _field_id(child) != parent_field_id
    assert _committed_field_id(tmp_path, "self") == parent_field_id

    # The child holds no intention of its own and must not spend the parent's.
    child_gate = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "session_id": "child-session",
            "hook_event_name": "PreToolUse",
            "tool_name": _OWNER_TOOL,
            "tool_input": _OWNER_INPUT,
            "tool_use_id": "child-write",
        },
    )
    assert child_gate["hookSpecificOutput"]["permissionDecision"] == "deny"

    parent_gate = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "session_id": "parent-session",
            "hook_event_name": "PreToolUse",
            "tool_name": _OWNER_TOOL,
            "tool_input": _OWNER_INPUT,
            "tool_use_id": "parent-write",
        },
    )
    assert parent_gate["hookSpecificOutput"]["permissionDecision"] == "allow", (
        parent_gate
    )


def test_forcing_one_shared_owner_takes_the_parents_committed_slot(
    tmp_path,
) -> None:
    """Same script, except the child is told to act as "self".

    `_owner_id` honours an explicit `owner_id` in the payload, which reproduces
    the pre-fix behaviour without editing the live hook: the child's tick
    occupies the one COMMITTED slot and the parent's field is no longer the
    committed one. Keeps the test above from passing vacuously.
    """
    parent_field_id = _start_parent_and_register(tmp_path)

    child = _run_hook(
        tmp_path,
        "user-prompt-submit",
        {
            "session_id": "child-session",
            "owner_id": "self",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "look something up for the parent",
        },
    )
    child_field_id = _field_id(child)

    assert child_field_id != parent_field_id
    assert _committed_field_id(tmp_path, "self") == child_field_id


class _RecordingProducer:
    """Just enough of a TickProducer to see what person_id a tick is opened with."""

    def __init__(self) -> None:
        self.begin_kwargs: dict = {}

    def resolve_owner_id(self, _session_id):
        return "self"

    def begin_tick(self, _kind, _owner_id, **kwargs):
        self.begin_kwargs = kwargs
        return type("Field", (), {"tick_id": "tick-1"})()

    def compete_and_commit(self, _tick_id):
        return type("Outcome", (), {"field": None})()


def _record_user_prompt(monkeypatch, payload: dict) -> dict:
    from individual_kernel_mcp import hook_cli

    monkeypatch.setattr(hook_cli, "_surface_context", lambda _field: "")
    producer = _RecordingProducer()
    hook_cli.user_prompt_submit(payload, producer)
    return producer.begin_kwargs


def test_user_prompt_person_id_defaults_to_neutral_companion(monkeypatch) -> None:
    monkeypatch.delenv("COMPANION_ID", raising=False)
    kwargs = _record_user_prompt(monkeypatch, {"prompt": "hi", "session_id": "s"})
    assert kwargs["person_id"] == "companion"


def test_user_prompt_person_id_honours_companion_id_env(monkeypatch) -> None:
    monkeypatch.setenv("COMPANION_ID", "kouta")
    kwargs = _record_user_prompt(monkeypatch, {"prompt": "hi", "session_id": "s"})
    assert kwargs["person_id"] == "kouta"

    # An explicit person_id in the payload still wins over the environment.
    kwargs = _record_user_prompt(
        monkeypatch, {"prompt": "hi", "session_id": "s", "person_id": "guest"}
    )
    assert kwargs["person_id"] == "guest"


# --- #137: the gate fires from the committed .claude/settings.json as soon as
# uv is on PATH, but the way out of a deny (propose_field_action) is an MCP tool
# that only exists once scripts/setup.sh has written .mcp.json. Before setup the
# deny must say so; and unreadable stdin must not pass for an internal tool.

_WRITE_PAYLOAD = {
    "hook_event_name": "PreToolUse",
    "tool_name": "Write",
    "tool_input": {"file_path": "a.txt", "content": "x"},
    "tool_use_id": "write-137",
}


def _project_env(project_dir: Path) -> dict[str, str]:
    return {"CLAUDE_PROJECT_DIR": str(project_dir)}


def _write_mcp_json(project_dir: Path, servers: dict) -> None:
    (project_dir / ".mcp.json").write_text(
        json.dumps({"mcpServers": servers}), encoding="utf-8"
    )


def test_pre_tool_deny_points_at_setup_when_kernel_is_not_configured(
    tmp_path: Path,
) -> None:
    """The reporter's exact script: clone, uv on PATH, no .mcp.json."""
    project = tmp_path / "project"
    project.mkdir()
    env = _project_env(project)
    _run_hook(
        tmp_path,
        "session-start",
        {"session_id": "x", "source": "startup"},
        env_overrides=env,
    )
    _run_hook(
        tmp_path,
        "user-prompt-submit",
        {"session_id": "x", "prompt": "hi"},
        env_overrides=env,
    )
    result = _run_hook(tmp_path, "pre-tool-use", _WRITE_PAYLOAD, env_overrides=env)

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    reason = output["permissionDecisionReason"]
    assert reason.startswith("individual-kernel MCP server is not configured")
    assert "scripts/setup.sh" in reason
    assert str(project) in reason
    # The runtime's own verdict is kept, so nothing about the gate is hidden.
    assert reason.endswith(
        "Original reason: no matching pending intention; call propose_field_action first"
    )


def test_pre_tool_deny_keeps_original_reason_when_kernel_is_configured(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_mcp_json(
        project,
        {
            "individual-kernel": {
                "command": "uv",
                "args": ["run", "--package", "individual-kernel-mcp", "individual-kernel-mcp"],
            }
        },
    )
    result = _run_hook(
        tmp_path, "pre-tool-use", _WRITE_PAYLOAD, env_overrides=_project_env(project)
    )

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"] == (
        "no COMMITTED field; refresh the field before outward action"
    )


def test_pre_tool_allow_is_unaffected_by_missing_kernel_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    result = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},
        },
        env_overrides=_project_env(project),
    )

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "allow"
    assert output["permissionDecisionReason"] == (
        "internal/read-only tool; outward action gate not required"
    )


def test_kernel_server_configured_reads_only_the_project_mcp_json(tmp_path: Path) -> None:
    assert kernel_server_configured(tmp_path) is False

    _write_mcp_json(tmp_path, {"memory": {"command": "uv"}})
    assert kernel_server_configured(tmp_path) is False

    _write_mcp_json(tmp_path, {"individual-kernel": {"command": "uv"}})
    assert kernel_server_configured(tmp_path) is True

    (tmp_path / ".mcp.json").write_text("{not json", encoding="utf-8")
    assert kernel_server_configured(tmp_path) is None


def test_kernel_server_configured_honours_claude_project_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_mcp_json(project, {"individual-kernel": {"command": "uv"}})
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
    monkeypatch.chdir(tmp_path)

    assert kernel_server_configured() is True

    monkeypatch.delenv("CLAUDE_PROJECT_DIR")
    assert kernel_server_configured() is False


@pytest.mark.parametrize("raw_input", ["", "not json", "[1, 2]", "   \n"])
def test_pre_tool_use_fails_closed_on_unreadable_stdin(tmp_path: Path, raw_input: str) -> None:
    """A pipe that loses stdin used to read as allow; it must read as a deny."""
    result = _run_hook(tmp_path, "pre-tool-use", None, raw_input=raw_input)

    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"] == UNREADABLE_INPUT_REASON


def test_other_hooks_treat_unreadable_stdin_as_an_empty_payload(tmp_path: Path) -> None:
    result = _run_hook(tmp_path, "post-tool-batch", None, raw_input="not json")

    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PostToolBatch"
    assert output["additionalContext"] == "No committed field is available."

    assert _run_hook(tmp_path, "stop-failure", None, raw_input="") == {}


def test_pre_tool_gate_matches_non_ascii_tool_input(tmp_path) -> None:
    """#152: a correctly registered intention with Japanese input must match."""
    tool_name = "Write"
    tool_input = {"file_path": "/tmp/efpf-メモ", "content": "こんにちは、あ"}
    _run_hook(
        tmp_path,
        "user-prompt-submit",
        {
            "session_id": "session-152",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "日本語を書く",
        },
    )
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
            goal="日本語のファイルを書く",
        )
    )
    db.close()

    result = _run_hook(
        tmp_path,
        "pre-tool-use",
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": "write-152",
        },
    )

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "allow", output


def test_read_input_decodes_utf8_despite_cp932_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The byte-layer read must not care what encoding the wrapper claims."""
    payload = {"tool_name": "Write", "tool_input": {"content": "こんにちは、あ"}}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    wrapper = io.TextIOWrapper(
        io.BufferedReader(io.BytesIO(raw)),
        encoding="cp932",
        errors="surrogateescape",
    )
    monkeypatch.setattr(sys, "stdin", wrapper)

    value = _read_input()

    assert value is not None
    assert value["tool_input"]["content"] == "こんにちは、あ"


def test_emit_writes_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    wrapper = io.TextIOWrapper(buffer, encoding="cp932")
    monkeypatch.setattr(sys, "stdout", wrapper)

    _emit({"reason": "プロジェクト"})

    assert buffer.getvalue() == '{"reason":"プロジェクト"}\n'.encode()
