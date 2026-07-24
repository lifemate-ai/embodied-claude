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
