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


def test_config_files_are_read_as_utf8_regardless_of_locale(tmp_path: Path) -> None:
    """Config files are UTF-8 on disk; the locale codec must not decide their contents (#149)."""
    mic_device = "マイク (Realtek Audio)"  # a Japanese Windows device name
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"cam": {"env": {"MIC_DEVICE": mic_device}}}}),
        encoding="utf-8",
    )
    (tmp_path / "socialPolicy.toml").write_text(
        '# quiet hours — keep the night calm\n[global]\ntimezone = "Asia/Tokyo"\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '# ワークスペース\n[project]\nname = "x"\ndependencies = ["memory-mcp"]\n',
        encoding="utf-8",
    )

    config, result = doctor._load_config(config_path)

    assert result.status is CheckStatus.OK
    assert config is not None
    assert config["mcpServers"]["cam"]["env"]["MIC_DEVICE"] == mic_device
    assert doctor._check_social_policy(tmp_path).status is CheckStatus.OK
    packages = doctor.check_workspace_packages(tmp_path, {"mcpServers": {"memory": {}}})
    assert [item.status for item in packages] == [CheckStatus.OK]


def test_undecodable_config_files_become_check_errors_not_tracebacks(tmp_path: Path) -> None:
    """A byte sequence that is not UTF-8 is reported, not raised (#149)."""
    bad = b"\xff\xfe not utf-8"
    (tmp_path / ".mcp.json").write_bytes(bad)
    (tmp_path / "socialPolicy.toml").write_bytes(bad)
    (tmp_path / "pyproject.toml").write_bytes(bad)

    config, result = doctor._load_config(tmp_path / ".mcp.json")
    assert config is None
    assert result.status is CheckStatus.ERROR
    assert doctor._check_social_policy(tmp_path).status is CheckStatus.ERROR
    packages = doctor.check_workspace_packages(tmp_path, {"mcpServers": {"memory": {}}})
    assert [item.status for item in packages] == [CheckStatus.ERROR]
def test_transcription_check_follows_installed_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """doctor must notice when the backend listen will use is not importable (#151)."""
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)

    def only_faster(name: str) -> bool:
        return name == "faster_whisper"

    auto = doctor.check_transcription_backend({}, {}, module_available=only_faster)
    assert auto.status is CheckStatus.OK
    assert "faster-whisper" in auto.detail

    explicit_missing = doctor.check_transcription_backend(
        {"env": {"TRANSCRIBE_BACKEND": "openai-whisper"}}, {}, module_available=only_faster
    )
    assert explicit_missing.status is CheckStatus.WARN
    assert "transcription-whisper" in explicit_missing.remediation

    nothing = doctor.check_transcription_backend({}, {}, module_available=lambda _n: False)
    assert nothing.status is CheckStatus.WARN

    unknown = doctor.check_transcription_backend(
        {}, {"TRANSCRIBE_BACKEND": "vosk"}, module_available=only_faster
    )
    assert unknown.status is CheckStatus.ERROR

    results = doctor.check_optional_dependencies(
        {"mcpServers": {"wifi-cam": {}}}, which=lambda _n: None, module_available=only_faster
    )
    assert [r.subject for r in results] == ["wifi-cam:ffmpeg", "wifi-cam:transcription"]
