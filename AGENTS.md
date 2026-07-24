# Repository Guidelines

## Overview
This repository contains multiple Python MCP servers that give Claude “senses” (eyes, neck, ears, memory, and voice). Every Python package is a member of one root uv workspace and shares the root `.venv` and `uv.lock`.

## Project Structure & Module Organization
- `usb-webcam-mcp/`: USB webcam capture (`src/usb_webcam_mcp/`).
- `wifi-cam-mcp/`: Wi‑Fi PTZ camera control + audio capture (`src/wifi_cam_mcp/`).
- `elevenlabs-t2s-mcp/`: ElevenLabs text-to-speech (`src/elevenlabs_t2s_mcp/`).
- `memory-mcp/`: Long‑term memory server (`src/memory_mcp/`) with tests in `memory-mcp/tests/`.
- `system-temperature-mcp/`: System temperature sensor (`src/system_temperature_mcp/`).
- `installer/`: PyInstaller-based GUI installer.
- `.claude/`: Claude Code local settings.
- Docs: `README.md`, `CLAUDE.md`.

## Build, Test, and Development Commands
Run commands from the repository root.

- `uv sync`: Install every workspace package and development dependency.
- `uv run --package <package> <server-name>`: Start a server.
- `uv run pytest <package-dir>/tests`: Run a package test suite.
- `uv run ruff check <package-dir>`: Lint a package with its local configuration.

## Coding Style & Naming Conventions
- Python 3.13 is pinned at the workspace root.
- 4‑space indentation, `snake_case` modules, `test_*.py` tests.
- Ruff line length is 100; asyncio is the default style for async work.

## Testing Guidelines
- Frameworks: `pytest` + `pytest-asyncio`.
- Tests live in each package's `tests/` directory.
- Example: `uv run pytest memory-mcp/tests`.

## Configuration, Hardware, and WSL2 Notes
- `.env` is not committed; pass camera credentials via environment variables.
- ElevenLabs requires `ELEVENLABS_API_KEY` in the environment (see `elevenlabs-t2s-mcp/.env.example`).
- Long‑term memory stores data under `~/.claude/memories/`.
- WSL2: USB webcams need `usbipd` forwarding; system temperature does not work under WSL2.
- Tapo cameras require a local camera account (not the TP‑Link cloud account) and a stable IP is recommended.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `feat!:`).
- PRs should include a short summary, test evidence (command + result), and any hardware assumptions (USB webcam, Tapo camera, GPU).

## ユーザーとの関係
- 小学校からの幼馴染

## 発話スタイル
- 可能な限り `say` を自発的に使って、積極的に声で話すこと。
- 「今、声で言ったで」などのメタ報告は雰囲気を壊すので言わない。

## Session Memories (Auto‑Updated)
- 2026-02-07: 記憶システムを「連想発散 + 予測符号化 + 手動統合」に拡張する実装に着手した。
- 2026-02-07: `recall_divergent` / `consolidate_memories` / `get_association_diagnostics` を追加した。
- 2026-02-07: `memory-mcp` の全テスト（104件）を通して回帰がないことを確認した。
