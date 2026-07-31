@echo off
setlocal
cd /d "%~dp0.."

rem Every argument goes straight to scripts\setup.py, which owns the option
rem surface; run with --help for the full list. `setup.cmd --all` configures
rem every server, filling absent credentials with obviously fake values.

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ 1>&2
  exit /b 127
)

uv run --no-project --python 3.13 python scripts\setup.py %*
exit /b %errorlevel%
