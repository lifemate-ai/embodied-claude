# Friendly Onboarding and Setup Design

## Purpose

Embodied Claude currently explains its capabilities before it gives a new user
a reliable path to a first working experience. The root README mixes the
minimum setup, hardware-specific configuration, optional cloud services,
complete tool inventories, and advanced autonomous operation. Copying the full
`.mcp.json.example` also enables servers whose credentials or hardware are not
configured.

The new onboarding must let a user start a useful, hardware-free runtime first,
then add only the capabilities they actually have.

## Goals

1. A user with `git`, `uv`, and Claude Code can create a working Core
   configuration from the repository root with one setup command.
2. Core requires no hardware and no API keys.
3. Optional integrations are included in `.mcp.json` only after the user
   selects and configures them.
4. Setup is safe to rerun and does not silently overwrite credentials.
5. A doctor command distinguishes blocking errors from optional capability
   warnings and gives exact remediation commands.
6. English and Japanese root READMEs lead with the same purpose-first journey.
7. macOS, Linux, WSL2, and Windows-native users have explicit commands and
   bounded platform claims.

## Non-Goals

- Installing Claude Code, hardware drivers, VOICEVOX, ffmpeg, mpv, or vendor
  camera applications automatically.
- Discovering camera credentials or cloud API secrets.
- Proving that physical hardware works when it is not connected.
- Replacing the full `.mcp.json.example` reference.
- Changing MCP APIs, EFPF semantics, database schemas, or the unified uv
  workspace.

## Supported Journeys

### Core: the default first success

Core configures these MCP servers:

- `memory`
- `desire-system`
- `sociality`
- `individual-kernel`

They use their existing defaults under `~/.claude/` and require neither
hardware nor third-party credentials. Setup also creates `socialPolicy.toml`
from `examples/configs/socialPolicy.example.toml` when the file is absent.

The default memory model for setup-generated configurations is
`intfloat/multilingual-e5-small`, reducing the first model download. Users can
select the higher-quality `intfloat/multilingual-e5-base` model explicitly.
Setup preloads the selected model unless `--skip-model-download` is supplied so
the first `remember` call does not appear to hang.

The documented success check is:

1. Start Claude Code in the repository.
2. Run `/mcp` and confirm the four Core servers are connected.
3. Ask Claude to remember a short setup fact.
4. Ask Claude to recall that fact.

### Optional capabilities

The wizard can add:

| Feature | MCP server | Required input or dependency |
|---|---|---|
| USB camera | `usb-webcam` | Connected camera; WSL2 may need `usbipd` |
| Tapo camera and local/camera audio | `wifi-cam` | Host, local camera username, password; ffmpeg for audio |
| Local VOICEVOX speech | `tts` | VOICEVOX URL; mpv or ffplay for local playback |
| ElevenLabs speech | `tts` | API key; optional voice ID; mpv or ffplay for local playback |
| X integration | `x-mcp` | xAI and X API credentials |
| Host temperature and time | `system-temperature` | Platform sensor support; Windows uses LibreHardwareMonitor |

Unselected features are absent from the generated `.mcp.json`. The full
reference file remains available for manual or advanced configuration.

## Command-Line Interface

### POSIX entrypoint

```bash
./scripts/setup.sh
```

`setup.sh` is a thin adapter. It verifies that `uv` exists and delegates to the
cross-platform Python CLI:

```bash
uv run --no-project --python 3.13 python scripts/setup.py
```

### Windows-native entrypoint

PowerShell users run the same Python CLI directly:

```powershell
uv run --no-project --python 3.13 python scripts/setup.py
```

No Bash environment is required on Windows.

### Stable options

```text
--profile core
--with-camera usb|tapo
--with-voice voicevox|elevenlabs
--with-x
--with-system-temperature
--embedding-model small|base
--skip-model-download
--non-interactive
--dry-run
--force
```

With no arguments, setup runs an interactive wizard. `--profile core` selects
the Core server set and is also the default for `--non-interactive`.

In non-interactive mode, optional integrations read their existing canonical
environment variable names. Missing required values are errors that name every
missing variable. Core never needs environment input.

The supported environment variables are:

- Tapo: `TAPO_CAMERA_HOST`, `TAPO_USERNAME`, `TAPO_PASSWORD`
- VOICEVOX: `VOICEVOX_URL` (defaults to `http://localhost:50021`)
- ElevenLabs: `ELEVENLABS_API_KEY`; optional `ELEVENLABS_VOICE_ID`
- X: `XAI_API_KEY`, `X_CONSUMER_KEY`, `X_CONSUMER_SECRET`,
  `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`

`--dry-run` performs no workspace sync, model download, policy copy, config
write, backup, or permission change. It prints a redacted summary of the
configuration that would be produced.

## Components

### `scripts/onboarding.py`

Pure, standard-library domain logic:

- canonical server definitions and argument lists
- feature selection models
- environment validation
- `.mcp.json` construction
- placeholder and secret detection
- redaction
- safe config comparison

It has no prompts, subprocess calls, or file writes. Tests import it directly.

### `scripts/setup_io.py`

Explicit filesystem operations shared by setup and tests:

- semantic comparison and overwrite planning
- timestamped backup selection
- atomic `.mcp.json` writes with private POSIX permissions
- copy-if-absent handling for `socialPolicy.toml`

It contains no prompts or subprocess calls. Dry-run paths never invoke it.

### `scripts/setup.py`

CLI orchestration:

1. Parse arguments.
2. Prompt for optional capabilities when interactive.
3. Collect secrets with `getpass.getpass`.
4. Build and validate the proposed config.
5. If dry-run, print a redacted summary and stop.
6. Run `uv sync --locked`.
7. Protect or back up an existing `.mcp.json`.
8. Write `.mcp.json` atomically and use mode `0600` on POSIX.
9. Copy the example social policy only when the destination is absent.
10. Preload the selected memory model unless skipped.
11. Run the doctor.
12. Print the exact Claude Code first-success steps.

### `scripts/setup.sh`

The Bash adapter:

- uses `set -euo pipefail`
- resolves the repository root from its own location
- prints an actionable uv installation command when `uv` is missing
- forwards every argument without reinterpretation

### `scripts/doctor.py`

The doctor performs read-only checks and emits one of three statuses:

- `[ok]`: requirement is satisfied
- `[warn]`: an optional selected capability may be unavailable
- `[error]`: Core cannot start or the generated configuration is invalid

It checks:

- Python is exactly within the root `>=3.13,<3.14` requirement.
- root `pyproject.toml` and `uv.lock` exist.
- `uv lock --check` succeeds.
- `.mcp.json` exists, is valid JSON, validates known setup-managed server
  shapes, and has no placeholder credential values.
- every configured Python package and import target is present in the synced
  workspace.
- `socialPolicy.toml` is valid TOML when present.
- selected camera/audio integrations have their relevant `ffmpeg`, `mpv`, or
  `ffplay` executables.
- state directories under `~/.claude` already exist and are writable, or their
  nearest existing parents are writable.

Unknown/custom MCP server entries produce a warning instead of an error. The
doctor does not create state directories while checking them. Warnings do not
make the doctor fail. Any error produces a nonzero exit status. Every warning
and error includes a concrete next command or documentation link.

## Configuration Safety

- `.mcp.json` remains ignored by Git.
- Setup never writes placeholder secrets.
- Interactive secrets are read without terminal echo and never printed back.
- Dry-run output replaces every secret with `<redacted>`.
- If `.mcp.json` does not exist, setup writes it atomically.
- If the existing file is semantically equivalent JSON to the generated config,
  rerunning is successful and leaves it unchanged.
- If it differs, setup refuses by default.
- `--force` creates `.mcp.json.backup-YYYYMMDD-HHMMSS` before replacing it.
- `.gitignore` covers setup-created backup names without ignoring
  `.mcp.json.example`.
- `socialPolicy.toml` is never overwritten.

## Documentation Architecture

### Root `README.md` and `README-ja.md`

The root READMEs use this order:

1. What experience the project creates.
2. A three-command Core quick start.
3. A visible success checklist.
4. A capability chooser with exact setup flags.
5. Platform support matrix and prerequisites.
6. A concise architecture explanation.
7. Links to detailed setup and component documentation.
8. Advanced/research context, attribution, and license.

The package catalog is retained as a compact reference after onboarding. Large
tool inventories and hardware walkthroughs no longer interrupt the first-run
path.

Both READMEs:

- use `lifemate-ai/embodied-claude` for clone, badge, and current links
- state that the unified workspace uses Python 3.13 and one root `.venv`
- do not tell users to create per-package `.env` files when `.mcp.json` is the
  canonical credential source
- distinguish supported POSIX/WSL setup from partial Windows-native hardware
  support
- include the exact doctor and non-interactive commands

### `docs/setup.md`

Detailed task-oriented recipes live in one setup guide:

- Core interactive and non-interactive setup
- selecting small versus base memory model
- USB and Tapo camera setup links
- VOICEVOX and ElevenLabs configuration
- X credentials
- Windows-native command and current limitations
- existing-config migration and backup behavior
- doctor output and troubleshooting

Package READMEs remain authoritative for hardware-specific details.

## Error Handling

- Unknown CLI values exit with status `2` and argparse usage.
- Missing `uv` exits before any write and shows the official install command.
- Failed `uv sync --locked` stops setup before config replacement.
- Invalid or missing optional credentials stop before sync or writes.
- Failed model preload is an error unless the user explicitly used
  `--skip-model-download`.
- Doctor errors make setup exit nonzero after printing the complete report.
- Optional executable or physical-hardware checks are warnings, not errors.

## Testing

### Unit tests

`tests/test_onboarding.py` covers:

- exact Core server set
- each optional feature's server and environment mapping
- optional credentials cannot be omitted in non-interactive mode
- no placeholder is written
- secrets are redacted
- existing identical config is idempotent
- differing config requires `--force`
- forced replacement creates a backup
- dry-run creates no files and runs no subprocesses
- generated JSON round-trips
- doctor reports invalid JSON and placeholders as errors
- missing optional executables are warnings

### Shell and CLI smoke tests

- `bash -n scripts/setup.sh`
- setup dry-run from a directory outside the repository
- non-interactive Core setup in a temporary repository copy/home
- doctor against the generated Core config
- `uv lock --check`

### Documentation tests

Root tests assert:

- both quick starts use the supported setup entrypoints
- clone and badge URLs use the current organization
- first-success steps exist in both languages
- copied full `.mcp.json.example` is not presented as the default path
- per-package `uv sync` and `.env` instructions are absent from root onboarding
- platform claims and Windows commands agree

### Regression verification

The existing root workspace tests, package tests, Ruff checks, and GitHub CI
remain required.

## Completion Criteria

The work is complete only when:

1. A clean temporary HOME can run non-interactive Core setup and produce a
   validated `.mcp.json` without hardware or secrets.
2. The generated Core config contains exactly the four Core MCP servers.
3. Rerunning setup preserves identical config, refuses a different config, and
   backs it up only with `--force`.
4. Dry-run is side-effect free.
5. Doctor returns success for Core and actionable results for selected optional
   capabilities.
6. English and Japanese READMEs expose the same quick-start journey.
7. The full local test/lint suite and GitHub CI pass.
