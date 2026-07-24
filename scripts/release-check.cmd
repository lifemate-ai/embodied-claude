@echo off
setlocal
cd /d "%~dp0.."

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ 1>&2
  exit /b 127
)

uv run --no-project --python 3.13 python scripts\release_check.py %*
exit /b %errorlevel%
