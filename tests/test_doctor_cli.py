from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import doctor
from scripts.doctor import CheckResult, CheckStatus


def test_json_output_has_stable_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        doctor,
        "run_doctor",
        lambda *_args, **_kwargs: [
            CheckResult(CheckStatus.OK, "python", "3.13.5"),
            CheckResult(
                CheckStatus.WARN,
                "tts:playback",
                "no player",
                "Install mpv.",
            ),
        ],
    )

    exit_code = doctor.main(["--json"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report == {
        "schema_version": 1,
        "platform": doctor.platform.system().lower(),
        "summary": {"ok": 1, "warn": 1, "error": 0},
        "checks": [
            {
                "status": "ok",
                "subject": "python",
                "detail": "3.13.5",
                "remediation": None,
            },
            {
                "status": "warn",
                "subject": "tts:playback",
                "detail": "no player",
                "remediation": "Install mpv.",
            },
        ],
    }


def test_json_output_returns_one_when_any_check_is_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        doctor,
        "run_doctor",
        lambda *_args, **_kwargs: [
            CheckResult(CheckStatus.ERROR, "config:file", "missing"),
        ],
    )

    assert doctor.main(["--json"]) == 1
    assert json.loads(capsys.readouterr().out)["summary"]["error"] == 1


def test_live_checks_delegate_to_the_isolated_mcp_probe(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "memory": {
                        "command": "uv",
                        "args": [
                            "run",
                            "--package",
                            "memory-mcp",
                            "memory-mcp",
                        ],
                        "env": {"MEMORY_EMBEDDING_MODEL": "fixture/model"},
                    }
                }
            }
        )
    )
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "ok": True,
                    "server": "memory",
                    "tool_count": 31,
                    "remember_roundtrip": True,
                }
            ),
            "",
        )

    results = doctor.run_live_checks(
        tmp_path,
        config_path,
        tmp_path / "state",
        runner=runner,
    )

    assert len(results) == 1
    assert results[0].status is CheckStatus.OK
    assert "31 tools" in results[0].detail
    assert "--remember-roundtrip" in calls[0]


def _repo_with_hooks(tmp_path: Path) -> Path:
    """A checkout as git delivers it: the hook settings, no .mcp.json."""
    root = tmp_path / "repo"
    (root / ".claude").mkdir(parents=True)
    (root / ".claude" / "settings.json").write_text(
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"type": "command"}]}]}}),
        encoding="utf-8",
    )
    return root


def test_hook_gate_check_names_the_deny_when_mcp_json_is_missing(
    tmp_path: Path,
) -> None:
    root = _repo_with_hooks(tmp_path)
    config_path = root / ".mcp.json"

    result = doctor.check_hook_gate(root, None, config_path)

    assert result is not None
    assert result.status is CheckStatus.ERROR
    assert result.subject == "hooks:gate"
    assert "active as soon as uv is on PATH" in result.detail
    assert "denied" in result.detail
    assert "setup.sh" in (result.remediation or "")


def test_hook_gate_check_passes_when_individual_kernel_is_configured(
    tmp_path: Path,
) -> None:
    root = _repo_with_hooks(tmp_path)
    config = {"mcpServers": {"individual-kernel": {"command": "uv"}}}

    result = doctor.check_hook_gate(root, config, root / ".mcp.json")

    assert result is not None
    assert result.status is CheckStatus.OK

    missing_kernel = doctor.check_hook_gate(
        root, {"mcpServers": {"memory": {}}}, root / ".mcp.json"
    )
    assert missing_kernel is not None
    assert missing_kernel.status is CheckStatus.ERROR


def test_hook_gate_check_is_silent_without_a_committed_pre_tool_use_hook(
    tmp_path: Path,
) -> None:
    assert doctor.check_hook_gate(tmp_path, None, tmp_path / ".mcp.json") is None
