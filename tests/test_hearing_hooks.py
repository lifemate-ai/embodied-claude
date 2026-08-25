"""The hearing hooks are plain bash + embedded Python and have to survive Git Bash
on Windows (#139). These tests read the scripts as text and pin the three
portability fixes so they cannot quietly regress on a POSIX-only machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
HOOKS = ROOT / ".claude" / "hooks"
SCRIPTS = ("hearing-hook.sh", "hearing-stop-hook.sh")


def _read(name: str) -> str:
    return (HOOKS / name).read_text(encoding="utf-8")


def _python_heredocs(script: str) -> list[str]:
    """Return the bodies of every ``<<'PYEOF' ... PYEOF`` block in the script."""
    blocks = re.findall(r"<<'PYEOF'.*?\n(.*?)\nPYEOF", script, flags=re.DOTALL)
    assert blocks, "expected at least one embedded Python heredoc"
    return blocks


@pytest.mark.parametrize("name", SCRIPTS)
def test_embedded_python_never_hardcodes_tmp(name: str) -> None:
    """Python's Path("/tmp/...") resolves to the current drive on Windows, while the
    shell's /tmp is the MSYS temp dir. The location has to come from the shell via
    HEARING_DIR, with tempfile.gettempdir() as the stand-alone fallback."""
    script = _read(name)
    for body in _python_heredocs(script):
        assert '"/tmp/' not in body, name
        assert "'/tmp/" not in body, name
        assert 'os.environ.get("HEARING_DIR") or tempfile.gettempdir()' in body, name
    # Shell side resolves the directory exactly once and exports it for Python.
    assert 'HEARING_DIR="${HEARING_DIR:-${TMPDIR:-/tmp}}"' in script, name
    assert "export HEARING_DIR" in script, name
    # No shell-level literal either (PID/offset/timing/context files all live under it).
    for line in script.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert not re.search(r'(?<![\w{:-])/tmp/', line), (name, line)


@pytest.mark.parametrize("name", SCRIPTS)
def test_interpreter_is_probed_not_assumed(name: str) -> None:
    """`python3` may be the Microsoft Store alias (exits 49 without running anything).
    The hook must honour HEARING_PYTHON, otherwise try python3 then python and keep
    the first that actually executes `import sys`; none -> exit 0 quietly."""
    script = _read(name)
    assert "find_python()" in script, name
    assert "HEARING_PYTHON" in script, name
    assert "for candidate in python3 python; do" in script, name
    assert '"$candidate" -c "import sys"' in script, name
    assert 'PY="$(find_python)" || exit 0' in script, name
    # Every remaining interpreter call goes through $PY, never a bare python3.
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "for candidate in" in stripped:
            continue
        assert not re.match(r'^\w+=\$\(python3? ', stripped), (name, line)
        assert not stripped.startswith("python3 -"), (name, line)


@pytest.mark.parametrize("name", SCRIPTS)
def test_daemon_liveness_falls_back_to_tasklist(name: str) -> None:
    """MSYS `kill -0` only sees MSYS processes; a daemon started as a native Windows
    process looks dead. pid_alive() asks tasklist when kill -0 fails."""
    script = _read(name)
    assert "pid_alive()" in script, name
    assert 'kill -0 "$1" 2>/dev/null && return 0' in script, name
    assert 'tasklist /FI "PID eq $1"' in script, name
    # Git Bash rewrites "/FI" into a Windows path unless conversion is disabled.
    assert "MSYS_NO_PATHCONV=1" in script, name
    assert 'pid_alive "$PID"' in script, name
    assert 'kill -0 "$PID"' not in script, name


def test_stop_hook_separates_library_dir_from_buffer_dir() -> None:
    """HEARING_DIR is where the buffer lives; the hearing library handed to
    `uv run --directory` is HEARING_LIB_DIR. The two used to share one name."""
    script = _read("hearing-stop-hook.sh")
    assert '--directory "$HEARING_LIB_DIR"' in script
    assert '--directory "$HEARING_DIR"' not in script


def test_stop_hook_documents_its_timeout_budget() -> None:
    """A Stop hook registered with the core hooks' timeout of 10 is killed mid-wait
    and never extends a turn, silently. The header and the docs both say >= 25 s."""
    script = _read("hearing-stop-hook.sh")
    header = "\n".join(line for line in script.splitlines()[:40] if line.startswith("#"))
    assert "25" in header
    doc = (ROOT / "docs" / "hearing-hooks.md").read_text(encoding="utf-8")
    assert '"timeout": 30' in doc
    assert "HEARING_PYTHON" in doc
    assert "HEARING_DIR" in doc
    assert "bash.exe" in doc
