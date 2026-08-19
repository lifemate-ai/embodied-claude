"""Four things headless startup used to lose without a word (#140).

Each of these fails at the configuration layer and looks fine at the execution
layer: the heartbeat completes, the log reads normally, and only the absence
is missing. Interactive use mostly catches them because a person is watching;
`claude -p` has nobody watching.
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import pytest

from scripts import doctor
from scripts.doctor import CheckResult, CheckStatus
from scripts.onboarding import CORE_SERVER_NAMES, FeatureSelection
from scripts.setup import execute_setup
from scripts.setup_io import ConfigConflictError, enable_headless_servers

ROOT = Path(__file__).parents[1]
SCRIPT = (ROOT / "autonomous-action.sample.sh").read_text(encoding="utf-8")


# --- 1. project MCP servers need approval before `claude -p` will load them ---


def test_enable_list_is_created_with_every_written_server(tmp_path: Path) -> None:
    settings = tmp_path / ".claude" / "settings.local.json"

    assert enable_headless_servers(settings, ["memory", "sociality"]) is True

    assert json.loads(settings.read_text(encoding="utf-8")) == {
        "enabledMcpjsonServers": ["memory", "sociality"]
    }


def test_enable_list_merges_without_clobbering_other_keys(tmp_path: Path) -> None:
    # autonomous-action.sh reads permissions.allow from this very file; losing
    # it would trade one silent startup loss for another.
    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["mcp__memory__recall"]},
                "enabledMcpjsonServers": ["individual-kernel", "stale-server"],
            }
        ),
        encoding="utf-8",
    )

    assert enable_headless_servers(settings, ["memory", "memory", "sociality"]) is True

    loaded = json.loads(settings.read_text(encoding="utf-8"))
    assert loaded["permissions"] == {"allow": ["mcp__memory__recall"]}
    # Replaced, not appended: editing .mcp.json resets approval, so the list
    # has to track exactly what was just written, duplicates and stale names
    # included.
    assert loaded["enabledMcpjsonServers"] == ["memory", "sociality"]


def test_enable_list_is_left_alone_when_already_current(tmp_path: Path) -> None:
    settings = tmp_path / "settings.local.json"
    settings.write_text('{"enabledMcpjsonServers": ["memory"]}\n', encoding="utf-8")
    before = settings.read_text(encoding="utf-8")

    assert enable_headless_servers(settings, ["memory"]) is False
    assert settings.read_text(encoding="utf-8") == before


def test_enable_list_refuses_to_overwrite_unreadable_settings(tmp_path: Path) -> None:
    settings = tmp_path / "settings.local.json"
    settings.write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigConflictError):
        enable_headless_servers(settings, ["memory"])


def _fixture_workspace(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0.1.0"\n'
        'dependencies = ["memory-mcp", "desire-system", "sociality-mcp", '
        '"individual-kernel-mcp"]\n'
    )
    (root / "uv.lock").write_text("version = 1\n")
    policy = root / "examples" / "configs" / "socialPolicy.example.toml"
    policy.parent.mkdir(parents=True)
    policy.write_text('version = 1\nname = "fixture"\n')


def test_setup_enables_exactly_the_servers_it_wrote(tmp_path: Path) -> None:
    _fixture_workspace(tmp_path)
    output = StringIO()

    result = execute_setup(
        FeatureSelection(),
        {},
        repo_root=tmp_path,
        home=tmp_path / "home",
        dry_run=False,
        force=False,
        skip_model_download=True,
        runner=lambda command, **_k: __import__("subprocess").CompletedProcess(
            command, 0, "", ""
        ),
        doctor=lambda *_a, **_k: [CheckResult(CheckStatus.OK, "fixture", "ready")],
        output=output,
    )

    assert result == 0
    written = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
    settings = json.loads(
        (tmp_path / ".claude" / "settings.local.json").read_text(encoding="utf-8")
    )
    assert settings["enabledMcpjsonServers"] == list(written) == list(CORE_SERVER_NAMES)
    assert "headless" in output.getvalue()


def test_dry_run_does_not_write_the_enable_list(tmp_path: Path) -> None:
    _fixture_workspace(tmp_path)

    execute_setup(
        FeatureSelection(),
        {},
        repo_root=tmp_path,
        home=tmp_path / "home",
        dry_run=True,
        force=False,
        skip_model_download=True,
        output=StringIO(),
    )

    assert not (tmp_path / ".claude").exists()


def test_settings_local_is_ignored_by_git() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".claude/settings.local.json" in ignored


def _config(*names: str) -> dict:
    return {"mcpServers": {name: {"command": "uv", "args": []} for name in names}}


def test_doctor_warns_per_server_missing_from_the_enable_list(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.local.json").write_text(
        '{"enabledMcpjsonServers": ["memory"]}', encoding="utf-8"
    )

    results = doctor.check_headless_approval(tmp_path, _config("memory", "sociality"))

    assert [(r.status, r.subject) for r in results] == [
        (CheckStatus.WARN, "headless:sociality")
    ]
    assert "not enabled for headless runs" in results[0].detail
    assert "`claude -p` will skip it silently" in results[0].detail
    assert "enabledMcpjsonServers" in results[0].remediation


def test_doctor_warns_for_every_server_when_the_settings_file_is_absent(
    tmp_path: Path,
) -> None:
    results = doctor.check_headless_approval(tmp_path, _config("memory", "sociality"))

    assert [r.subject for r in results] == ["headless:memory", "headless:sociality"]
    assert all(r.status is CheckStatus.WARN for r in results)


def test_doctor_is_satisfied_by_a_complete_list_or_enable_all(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    settings = tmp_path / ".claude" / "settings.local.json"

    settings.write_text('{"enabledMcpjsonServers": ["memory", "sociality"]}')
    [result] = doctor.check_headless_approval(tmp_path, _config("memory", "sociality"))
    assert result.status is CheckStatus.OK

    settings.write_text('{"enableAllProjectMcpServers": true}')
    [result] = doctor.check_headless_approval(tmp_path, _config("memory", "sociality"))
    assert result.status is CheckStatus.OK


# --- 2. memory HTTP recall port -------------------------------------------


def test_doctor_warns_when_the_recall_port_is_closed() -> None:
    result = doctor.check_memory_http_port({}, is_listening=lambda _h, _p: False)

    assert result.status is CheckStatus.WARN
    assert result.detail == (
        "memory HTTP recall port 18900 is not listening; individual-kernel ticks "
        "will carry no memory candidates"
    )


def test_doctor_probes_the_configured_recall_port() -> None:
    probed: list[tuple[str, int]] = []

    def listening(host: str, port: int) -> bool:
        probed.append((host, port))
        return True

    result = doctor.check_memory_http_port(
        {"MEMORY_HTTP_PORT": "18901"}, is_listening=listening
    )

    assert probed == [("127.0.0.1", 18901)]
    assert result.status is CheckStatus.OK


def test_doctor_really_connects_to_a_closed_port() -> None:
    import socket

    # Bind and release to find a port nothing is listening on right now.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    result = doctor.check_memory_http_port({"MEMORY_HTTP_PORT": str(port)})

    assert result.status is CheckStatus.WARN
    assert f"port {port} is not listening" in result.detail


# --- 3. @SOUL.md / @TODO.md / @ROUTINES.md --------------------------------


def test_script_checks_each_mentioned_file_before_building_the_prompt() -> None:
    check_at = SCRIPT.index("for PROMPT_FILE in SOUL.md TODO.md ROUTINES.md")
    prompt_at = SCRIPT.index('PROMPT="自律行動タイム')
    assert check_at < prompt_at

    for name in ("SOUL.md", "TODO.md", "ROUTINES.md"):
        assert f"@{name}" in SCRIPT
    assert re.search(r'WARN: \$PROMPT_FILE not found in \$SCRIPT_DIR', SCRIPT)
    assert ">&2" in SCRIPT[check_at:prompt_at]
    assert '>> "$LOG_FILE"' in SCRIPT[check_at:prompt_at]
    # Warn, do not abort: the heartbeat still runs on CLAUDE.md alone.
    assert "exit" not in SCRIPT[check_at:prompt_at]


def test_templates_and_doc_exist_for_the_three_files() -> None:
    for name in ("SOUL", "TODO", "ROUTINES"):
        assert (ROOT / "examples" / f"{name}.sample.md").is_file()
    doc = (ROOT / "docs" / "autonomous-files.md").read_text(encoding="utf-8")
    for name in ("SOUL.md", "TODO.md", "ROUTINES.md"):
        assert name in doc


def test_doctor_warns_for_missing_files_once_the_script_is_installed(
    tmp_path: Path,
) -> None:
    (tmp_path / "SOUL.md").write_text("# me\n")

    [before] = doctor.check_autonomous_files(tmp_path)
    assert before.status is CheckStatus.OK
    assert "TODO.md, ROUTINES.md absent" in before.detail

    (tmp_path / "autonomous-action.sh").write_text("#!/bin/bash\n")
    after = doctor.check_autonomous_files(tmp_path)
    assert [(r.status, r.subject) for r in after] == [
        (CheckStatus.WARN, "autonomous:TODO.md"),
        (CheckStatus.WARN, "autonomous:ROUTINES.md"),
    ]
    assert "@TODO.md reference" in after[0].detail

    (tmp_path / "TODO.md").write_text("- [ ]\n")
    (tmp_path / "ROUTINES.md").write_text("|\n")
    [present] = doctor.check_autonomous_files(tmp_path)
    assert present.status is CheckStatus.OK


# --- 4. .mcp.json.example no longer pins SOCIAL_DB_PATH --------------------


def test_examples_do_not_pin_state_overrides() -> None:
    for name in (".mcp.json.example", "autonomous-mcp.json.example"):
        config = json.loads((ROOT / name).read_text(encoding="utf-8"))
        for server in config["mcpServers"].values():
            env = server.get("env", {})
            assert "SOCIAL_DB_PATH" not in env, name
            assert "MEMORY_HTTP_PORT" not in env, name
