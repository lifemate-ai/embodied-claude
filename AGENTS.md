# Repository Guidelines

## Overview
This repository contains Python MCP servers that provide Claude with embodied capabilities:
vision, pan/tilt control, hearing, voice output, and long-term memory.

## Canonical And Mirror
- Canonical repository: `embodied-claude/`
- Local migration mirror: `../embodied-claude-main/`
- Mirror policy: canonical is source-of-truth and mirror is sync-only.

## Project Structure
- `usb-webcam-mcp/`: USB camera capture server.
- `wifi-cam-mcp/`: Wi-Fi PTZ camera control + audio capture server.
- `elevenlabs-t2s-mcp/`: ElevenLabs text-to-speech server.
- `memory-mcp/`: Long-term memory server (ChromaDB).
- `system-temperature-mcp/`: temperature and current time server.
- `installer/`: GUI installer.
- `.github/workflows/ci.yml`: CI quality gates.
- `.mcp.json.example`: MCP local config template.

## Build And Test
Run commands from each package directory:
- `uv sync --extra dev`
- `uv run ruff check .`
- `uv run pytest -q`

Package smoke checks:
- `cd usb-webcam-mcp && uv run python -c "import usb_webcam_mcp.server"`
- `cd system-temperature-mcp && uv run python -c "import system_temperature_mcp.server"`
- `cd installer && uv run python -c "import installer.main"`

## Tool Naming Policy
- Canonical wifi tools: `see`, `look_left`, `look_right`, `look_up`, `look_down`, `look_around`
- Deprecated aliases are still accepted: `camera_capture`, `camera_pan_left`, `camera_pan_right`, `camera_tilt_up`, `camera_tilt_down`, `camera_look_around`
- When alias names are used, the server should return a deprecation notice.

## Style
- Python 3.10+ baseline (`system-temperature-mcp` is 3.12+).
- Ruff line length: 100.
- Async-first design for MCP handlers.

## Security
- Do not commit `.env` or real credentials.
- `.mcp.json` is local-only and must stay untracked.
- Commit and maintain `.mcp.json.example` only.

## Supported Platforms
- Official support: Linux + WSL2.
- macOS: best effort.
