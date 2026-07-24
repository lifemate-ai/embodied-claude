from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_posix_launchers_delegate_with_exec_and_all_arguments() -> None:
    for name, target in (
        ("setup.sh", "scripts/setup.py"),
        ("doctor.sh", "scripts/doctor.py"),
    ):
        script = (ROOT / "scripts" / name).read_text()
        assert "command -v uv" in script
        assert "exec uv run --no-project --python 3.13 python" in script
        assert target in script
        assert '"$@"' in script


def test_windows_launchers_delegate_and_preserve_exit_status() -> None:
    for name, target in (
        ("setup.cmd", r"scripts\setup.py"),
        ("doctor.cmd", r"scripts\doctor.py"),
    ):
        script = (ROOT / "scripts" / name).read_text()
        assert 'cd /d "%~dp0.."' in script
        assert "where uv" in script
        assert "uv run --no-project --python 3.13 python" in script
        assert target in script
        assert "%*" in script
        assert "exit /b %errorlevel%" in script
