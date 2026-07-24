from __future__ import annotations

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

from scripts import setup
from scripts.doctor import CheckResult, CheckStatus
from scripts.onboarding import CORE_SERVER_NAMES, FeatureSelection
from scripts.setup import execute_setup

ROOT = Path(__file__).parents[1]
SETUP = ROOT / "scripts" / "setup.py"


def test_core_sync_command_has_no_optional_extras() -> None:
    assert setup.build_sync_command(FeatureSelection()) == [
        "uv",
        "sync",
        "--locked",
        "--no-dev",
    ]


def test_sync_command_contains_each_selected_extra_once() -> None:
    selection = FeatureSelection(
        camera="tapo",
        transcription="faster",
        voice="elevenlabs",
        x_enabled=True,
    )

    assert setup.build_sync_command(selection) == [
        "uv",
        "sync",
        "--locked",
        "--no-dev",
        "--extra",
        "camera-tapo",
        "--extra",
        "transcription-faster",
        "--extra",
        "voice-elevenlabs",
        "--extra",
        "x",
    ]


def _make_fixture_workspace(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        """
[project]
name = "fixture"
version = "0.1.0"
dependencies = [
  "memory-mcp",
  "desire-system",
  "sociality-mcp",
  "individual-kernel-mcp",
]
"""
    )
    (root / "uv.lock").write_text("version = 1\n")


def _run_setup_cli(*arguments: str, environment: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(SETUP), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_setup_help_lists_stable_profile_options() -> None:
    result = _run_setup_cli("--help")

    assert result.returncode == 0
    assert "--profile" in result.stdout
    assert "--with-camera" in result.stdout
    assert "--with-transcription" in result.stdout
    assert "--with-voice" in result.stdout
    assert "--embedding-model" in result.stdout
    assert "--dry-run" in result.stdout


def test_noninteractive_core_dry_run_lists_only_core_servers() -> None:
    result = _run_setup_cli(
        "--profile",
        "core",
        "--non-interactive",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert tuple(config["mcpServers"]) == CORE_SERVER_NAMES
    assert not (ROOT / ".mcp.json").exists()


def test_optional_dry_run_redacts_secrets() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "TAPO_CAMERA_HOST": "192.0.2.10",
            "TAPO_USERNAME": "private-user",
            "TAPO_PASSWORD": "private-password",
        }
    )

    result = _run_setup_cli(
        "--profile",
        "core",
        "--with-camera",
        "tapo",
        "--non-interactive",
        "--dry-run",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "private-user" not in result.stdout
    assert "private-password" not in result.stdout
    assert result.stdout.count("<redacted>") == 2


def test_noninteractive_selection_names_every_missing_variable() -> None:
    environment = os.environ.copy()
    for key in (
        "XAI_API_KEY",
        "X_CONSUMER_KEY",
        "X_CONSUMER_SECRET",
        "X_ACCESS_TOKEN",
        "X_ACCESS_TOKEN_SECRET",
    ):
        environment.pop(key, None)

    result = _run_setup_cli(
        "--with-x",
        "--non-interactive",
        "--dry-run",
        environment=environment,
    )

    assert result.returncode == 2
    for key in (
        "XAI_API_KEY",
        "X_CONSUMER_KEY",
        "X_CONSUMER_SECRET",
        "X_ACCESS_TOKEN",
        "X_ACCESS_TOKEN_SECRET",
    ):
        assert key in result.stderr


def test_dry_run_calls_no_side_effects(tmp_path: Path) -> None:
    _make_fixture_workspace(tmp_path)
    output = StringIO()

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("dry-run attempted a side effect")

    result = execute_setup(
        FeatureSelection(),
        {},
        repo_root=tmp_path,
        home=tmp_path / "home",
        dry_run=True,
        force=False,
        skip_model_download=False,
        runner=unexpected_call,
        doctor=unexpected_call,
        output=output,
    )

    assert result == 0
    assert not (tmp_path / ".mcp.json").exists()
    assert not (tmp_path / "socialPolicy.toml").exists()
    assert set(json.loads(output.getvalue())["mcpServers"]) == set(CORE_SERVER_NAMES)


def test_real_orchestration_syncs_once_writes_config_and_runs_doctor(
    tmp_path: Path,
) -> None:
    _make_fixture_workspace(tmp_path)
    policy_source = tmp_path / "examples" / "configs" / "socialPolicy.example.toml"
    policy_source.parent.mkdir(parents=True)
    policy_source.write_text('version = 1\nname = "fixture"\n')
    commands: list[list[str]] = []
    doctor_calls: list[tuple[Path, Path, Path]] = []
    output = StringIO()

    def fake_runner(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_doctor(repo_root, config_path, home, **_kwargs):
        doctor_calls.append((repo_root, config_path, home))
        return [CheckResult(CheckStatus.OK, "fixture", "ready")]

    result = execute_setup(
        FeatureSelection(),
        {},
        repo_root=tmp_path,
        home=tmp_path / "home",
        dry_run=False,
        force=False,
        skip_model_download=True,
        runner=fake_runner,
        doctor=fake_doctor,
        output=output,
    )

    assert result == 0
    assert commands == [["uv", "sync", "--locked", "--no-dev"]]
    config = json.loads((tmp_path / ".mcp.json").read_text())
    assert set(config["mcpServers"]) == set(CORE_SERVER_NAMES)
    assert (tmp_path / "socialPolicy.toml").read_text() == policy_source.read_text()
    assert doctor_calls == [
        (tmp_path, tmp_path / ".mcp.json", tmp_path / "home"),
    ]
    assert "Run /mcp" in output.getvalue()


def test_setup_warms_the_selected_embedding_model(tmp_path: Path) -> None:
    _make_fixture_workspace(tmp_path)
    policy_source = tmp_path / "examples" / "configs" / "socialPolicy.example.toml"
    policy_source.parent.mkdir(parents=True)
    policy_source.write_text('version = 1\nname = "fixture"\n')
    calls: list[tuple[list[str], dict]] = []

    def fake_runner(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_doctor(*_args, **_kwargs):
        return [CheckResult(CheckStatus.OK, "fixture", "ready")]

    result = execute_setup(
        FeatureSelection(embedding_model="base"),
        {},
        repo_root=tmp_path,
        home=tmp_path / "home",
        dry_run=False,
        force=False,
        skip_model_download=False,
        runner=fake_runner,
        doctor=fake_doctor,
        output=StringIO(),
    )

    assert result == 0
    assert calls[0][0] == ["uv", "sync", "--locked", "--no-dev"]
    assert calls[1][0][:5] == [
        "uv",
        "run",
        "--package",
        "memory-mcp",
        "python",
    ]
    assert (
        calls[1][1]["env"]["MEMORY_EMBEDDING_MODEL"]
        == "intfloat/multilingual-e5-base"
    )
