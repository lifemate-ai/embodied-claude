from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROBE = ROOT / "scripts" / "mcp_probe.py"
FIXTURE = ROOT / "tests" / "fixtures" / "mcp_fixture_server.py"


def _run_probe(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--name",
            "fixture",
            "--command",
            sys.executable,
            "--arg",
            str(FIXTURE),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_probe_initializes_server_and_lists_tools() -> None:
    result = _run_probe()

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "ok": True,
        "server": "fixture",
        "tool_count": 1,
        "remember_roundtrip": False,
    }


def test_probe_rejects_a_server_with_no_tools() -> None:
    result = _run_probe("--arg=--empty")
    report = json.loads(result.stdout)

    assert result.returncode == 1
    assert report["ok"] is False
    assert report["server"] == "fixture"
    assert "no tools" in report["error"]
