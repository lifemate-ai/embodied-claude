# Unified uv Workspace Design

## Goal

Make the repository's 17 Python projects installable from the repository root
with one plain command:

```bash
uv sync
```

The command must create one root `.venv`, resolve one root `uv.lock`, install
every MCP package as an editable workspace member, and include the development
tools and runtime extras currently installed by `scripts/install-mcps.sh`.

## Workspace Shape

The repository root becomes a non-package aggregator project:

- Python is pinned to 3.13 by a root `.python-version`.
- The root project declares `requires-python = ">=3.13,<3.14"`.
- `[tool.uv.workspace]` lists every Python project, including
  `sociality-mcp/packages/*` and
  `consciousness-mcp/packages/individual-kernel-mcp`.
- `[tool.uv.sources]` maps all first-party package names to
  `{ workspace = true }`.
- `[tool.uv] package = false` prevents building an artificial root wheel.

The root project depends on all 17 members. It requests `tts-mcp[all]` and
`wifi-cam-mcp[transcribe]`, preserving the full installation performed by the
current one-shot installer. The root `dev` dependency group contains the union
of repository development tools: pytest, pytest-asyncio, Ruff, mypy, jurigged,
and freezegun. Since uv installs the default `dev` group, plain `uv sync`
produces a contributor-ready environment.

## Members

The workspace contains:

1. `individual-kernel-mcp`
2. `desire-system`
3. `memory-mcp`
4. `sociality-mcp`
5. `agent-grammar`
6. `boundary-mcp`
7. `interaction-orchestrator-mcp`
8. `joint-attention-mcp`
9. `relationship-mcp`
10. `self-narrative-mcp`
11. `social-core`
12. `social-state-mcp`
13. `system-temperature-mcp`
14. `tts-mcp`
15. `usb-webcam-mcp`
16. `wifi-cam-mcp`
17. `x-mcp`

Node projects remain outside uv and continue to use the existing root
`package.json`.

## Dependency And Lock Migration

Member-to-member path sources are replaced by root workspace sources. Member
project metadata, build systems, optional extras, entry points, Ruff settings,
and pytest settings remain intact so each distribution stays independently
buildable.

All nested `uv.lock` files and nested `.python-version` files are removed.
Only root `uv.lock` and root `.python-version` remain. Running `uv` from a
member directory still discovers the parent workspace, so existing
`uv run --directory <member> ...` calls continue to work while sharing the
root lock and environment.

## Commands And Integration

The preferred commands become:

```bash
uv sync
uv run --package individual-kernel-mcp individual-kernel-mcp
uv run --package memory-mcp memory-mcp
uv run --package wifi-cam-mcp wifi-cam-mcp
```

`scripts/install-mcps.sh` remains as a compatibility wrapper but delegates to
the single root `uv sync`. README and CLAUDE guidance stop instructing users to
sync individual directories. Claude Code hook examples explicitly select the
workspace package while using the root project directory.

GitHub Actions installs the workspace once from the root and runs the existing
package checks through the shared environment. It also adds the EFPF,
social-core, orchestrator, sociality, and desire-system suites that are now
part of the unified dependency graph.

## Failure Handling

The migration is accepted only if uv resolves the complete Python 3.13 graph,
including ElevenLabs and Whisper. A resolution conflict is fixed in package
constraints rather than hidden with multiple environments. Platform-specific
hardware access remains a runtime concern; installation and deterministic
tests must not require cameras, temperature sensors, API keys, or audio
devices.

Existing ignored credential files and local user files are not modified.

## Verification

Verification covers:

1. `uv sync` from the repository root.
2. `uv lock --check`.
3. Exactly one tracked `uv.lock` and one tracked `.python-version`.
4. All 17 projects appear as workspace members in `uv.lock`.
5. Representative MCP entry points resolve through `uv run --package`.
6. Existing package test and Ruff commands pass from the shared environment.
7. Hook CLI and hardware-free EFPF benchmark still run from the root.
8. CI configuration and documented commands use the root workspace.

