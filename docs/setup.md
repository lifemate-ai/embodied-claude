# Setup Guide

This guide covers the generated project-local `.mcp.json`. It is the canonical
credential source for the guided setup. You do not need to run `uv sync` inside
individual package directories or create package-local `.env` files.

## Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Claude Code
- Network access for the first workspace sync and memory model download

The workspace requires Python 3.13. `uv` can install and select it
automatically.

## Core Setup

Core works without hardware or third-party API keys.

### Linux, macOS, and WSL2

```bash
git clone https://github.com/lifemate-ai/embodied-claude.git
cd embodied-claude
./scripts/setup.sh --profile core --non-interactive
```

### Windows 11 native PowerShell

```powershell
git clone https://github.com/lifemate-ai/embodied-claude.git
cd embodied-claude
scripts\setup.cmd --profile core --non-interactive
```

WSL2 is not required. The Windows launcher runs the same Python setup service
and writes the same portable Claude Code hook configuration as POSIX setup.

Setup performs one locked Core workspace sync, writes the Core `.mcp.json`,
approves those servers for headless runs in `.claude/settings.local.json`,
creates `socialPolicy.toml` only when absent, warms the selected memory model,
and runs the doctor.

Core contains:

| Server | Purpose |
|---|---|
| `memory` | Persistent recall and associations |
| `desire-system` | Bounded needs and homeostatic state |
| `sociality` | People, relationships, boundaries, and interaction context |
| `individual-kernel` | Enacted field runtime, action gate, and diagnostics |

## Guided Chooser

Run without arguments:

```bash
./scripts/setup.sh
```

On Windows, use `scripts\setup.cmd`.

The chooser asks which camera, voice engine, X integration, host sensor, and
memory model are available. It asks for credentials only after the
corresponding capability is selected. Secret input is not echoed.

The stable options are:

```text
--profile core
--with-camera usb|tapo
--with-transcription whisper|faster
--with-voice voicevox|elevenlabs
--with-x
--with-system-temperature
--embedding-model small|base
--skip-model-download
--non-interactive
--dry-run
--force
```

Selections compose. For example:

```bash
./scripts/setup.sh \
  --with-camera tapo \
  --with-voice voicevox \
  --with-system-temperature \
  --non-interactive
```

## Memory Model

The generated Core config defaults to:

```text
intfloat/multilingual-e5-small
```

It downloads faster and needs less disk/RAM. Select the larger existing default
for better retrieval quality:

```bash
./scripts/setup.sh \
  --profile core \
  --embedding-model base \
  --non-interactive
```

Setup warms the model before the first Claude Code session. To postpone that
download:

```bash
./scripts/setup.sh \
  --profile core \
  --skip-model-download \
  --non-interactive
```

The first memory operation will download the model instead.

## Optional Capabilities

Non-interactive setup reads credentials from the current environment and writes
only selected values to the ignored, mode-`0600` `.mcp.json`.

### USB camera

No credential is required:

```bash
./scripts/setup.sh --with-camera usb --non-interactive
```

On WSL2, forward the USB device with `usbipd` before starting Claude Code. On
other platforms, the camera must be visible to OpenCV.

### Tapo camera

Create a local camera account in the Tapo application. This is not the TP-Link
cloud account.

Required environment:

| Variable | Meaning |
|---|---|
| `TAPO_CAMERA_HOST` | Camera IP address or hostname |
| `TAPO_USERNAME` | Local camera username |
| `TAPO_PASSWORD` | Local camera password |

POSIX:

```bash
export TAPO_CAMERA_HOST=192.168.1.100
export TAPO_USERNAME='camera-user'
export TAPO_PASSWORD='camera-password'
./scripts/setup.sh --with-camera tapo --non-interactive
```

PowerShell:

```powershell
$env:TAPO_CAMERA_HOST = "192.168.1.100"
$env:TAPO_USERNAME = "camera-user"
$env:TAPO_PASSWORD = "camera-password"
scripts\setup.cmd --with-camera tapo --non-interactive
```

`ffmpeg` is optional for still images/PTZ and required for camera audio.
Transcription models are not installed with the camera alone; select exactly
one backend when needed:

```text
--with-transcription whisper|faster
```

This option requires `--with-camera tapo`. See
[`wifi-cam-mcp/README.md`](../wifi-cam-mcp/README.md) and
[`wifi-cam-mcp/README_WinNative.md`](../wifi-cam-mcp/README_WinNative.md).

### VOICEVOX

Start a VOICEVOX engine first. Setup uses
`http://localhost:50021` unless `VOICEVOX_URL` is set:

```bash
export VOICEVOX_URL=http://localhost:50021
./scripts/setup.sh --with-voice voicevox --non-interactive
```

Local playback needs `mpv` or `ffplay`. A missing player is a doctor warning,
not a Core failure.

### ElevenLabs

`ELEVENLABS_API_KEY` is required. `ELEVENLABS_VOICE_ID` is optional because
`tts-mcp` has a default voice.

```bash
export ELEVENLABS_API_KEY='...'
export ELEVENLABS_VOICE_ID='...'  # optional
./scripts/setup.sh --with-voice elevenlabs --non-interactive
```

PowerShell uses `$env:ELEVENLABS_API_KEY` and
`$env:ELEVENLABS_VOICE_ID`.

### X search and posting

The generated X server exposes both xAI-backed search and authenticated posting,
so non-interactive setup requires all five values:

| Variable | Source |
|---|---|
| `XAI_API_KEY` | xAI Console |
| `X_CONSUMER_KEY` | X Developer Portal |
| `X_CONSUMER_SECRET` | X Developer Portal |
| `X_ACCESS_TOKEN` | X Developer Portal |
| `X_ACCESS_TOKEN_SECRET` | X Developer Portal |

```bash
export XAI_API_KEY='...'
export X_CONSUMER_KEY='...'
export X_CONSUMER_SECRET='...'
export X_ACCESS_TOKEN='...'
export X_ACCESS_TOKEN_SECRET='...'
./scripts/setup.sh --with-x --non-interactive
```

### Host temperature and time

```bash
./scripts/setup.sh --with-system-temperature --non-interactive
```

Linux uses available `/sys`/hwmon sensors. Windows native temperature support
uses the LibreHardwareMonitor bridge documented in
[`system-temperature-mcp/README_WinNative.md`](../system-temperature-mcp/README_WinNative.md).
WSL2 normally cannot see Windows host temperature sensors.

## Preview Without Side Effects

Use `--dry-run` to inspect a redacted generated config:

```bash
./scripts/setup.sh \
  --profile core \
  --with-camera tapo \
  --non-interactive \
  --dry-run
```

Dry-run does not:

- run `uv sync`
- download the embedding model
- write or back up `.mcp.json`
- create `socialPolicy.toml`
- create state directories
- change permissions

Credentials are rendered as `<redacted>`.

## Existing `.mcp.json`

Setup has three outcomes:

1. No file: create `.mcp.json` atomically.
2. Semantically equivalent JSON: keep the original bytes unchanged.
3. Different or invalid JSON: stop and require explicit `--force`.

Forced replacement:

```bash
./scripts/setup.sh --profile core --non-interactive --force
```

The previous bytes are saved first as:

```text
.mcp.json.backup-YYYYMMDD-HHMMSS
```

If that name exists, setup adds a numeric suffix. Backup files are ignored by
Git.

Setup does not merge unknown custom MCP servers into the generated profile.
Keep a manual copy or re-add custom entries after generation. The doctor
reports unknown entries as warnings and does not modify them.

### `env` blocks override the inherited environment

A value in a server's `env` block wins over whatever the parent process
exported, so only put values there that you want pinned. Setup writes
`SOCIAL_DB_PATH` and `MEMORY_HTTP_PORT` only when they are set in the
environment it runs in; otherwise the servers fall back to their code defaults
(`~/.claude/sociality/social.db` and `18900`) and a later
`SOCIAL_DB_PATH=... claude` still takes effect. Copying the defaults into
`.mcp.json` by hand would silently pin them.

## Headless (`claude -p`) and MCP approval

Claude Code loads project `.mcp.json` servers only after they have been
approved once, and `claude -p` has no way to ask. Unapproved servers are skipped
without a message, so an autonomous heartbeat can start with no memory, no
sociality and no kernel and log a normal-looking run. Editing `.mcp.json`
resets the approval, so adding a camera later puts every server back to
pending.

Setup therefore also writes the approval, as the list Claude Code reads from
the repository:

```json
// .claude/settings.local.json
{ "enabledMcpjsonServers": ["memory", "desire-system", "sociality", "individual-kernel"] }
```

Other keys in that file (for example `permissions.allow`, which
`autonomous-action.sh` reads) are kept; only the list is replaced, and it is
set to exactly the servers setup just generated. The file is ignored by Git.

If you edit `.mcp.json` by hand, re-run setup or add the new server name to
`enabledMcpjsonServers` yourself. The doctor warns for every `.mcp.json`
server missing from the list:

```text
[warn] headless:memory: memory is in .mcp.json but not enabled for headless runs; `claude -p` will skip it silently
```

`claude mcp list` showing `Pending approval` is the same condition seen from
the other side.

## Doctor

Run static checks:

```bash
./scripts/doctor.sh
```

Windows:

```powershell
scripts\doctor.cmd
```

Inspect another file:

```bash
uv run python scripts/doctor.py --config /path/to/.mcp.json
```

Run real MCP startup checks against isolated temporary state:

```bash
./scripts/doctor.sh --live
```

Windows:

```powershell
scripts\doctor.cmd --live
```

Use `--json` with either mode for a stable machine-readable report.

Statuses:

| Status | Meaning |
|---|---|
| `[ok]` | The requirement is satisfied |
| `[warn]` | An optional selected capability may be unavailable |
| `[error]` | Core or a selected configuration cannot start correctly |

Static doctor is read-only. It does not create state directories, start MCP
servers, or connect to hardware. Its one network action is a TCP connect to
the memory HTTP recall port on localhost. It verifies:

- Python 3.13
- current `uv.lock`
- valid known MCP command shapes
- selected required environment values
- root workspace package declarations
- `socialPolicy.toml`
- writable existing state parents
- optional `ffmpeg`, `mpv`, or `ffplay`
- every `.mcp.json` server is enabled for headless runs (see above)
- memory HTTP recall port (`MEMORY_HTTP_PORT`, default `18900`) is listening.
  `individual-kernel` pulls memory candidates into each tick over it and
  commits the field without them when nothing answers. memory-mcp binds the
  port when Claude Code starts it, so this is a warning until the first
  session is running.
- `SOUL.md`, `TODO.md`, `ROUTINES.md` for the autonomous prompt (a warning
  once `autonomous-action.sh` is installed; see
  [`docs/autonomous-files.md`](autonomous-files.md))

Unknown custom MCP entries are warnings.

`--live` additionally starts each selected MCP over stdio, performs protocol
initialization, lists tools, and makes one memory write to an isolated temporary
database. Selected hardware integrations may contact their configured devices.
The temporary diagnostic state is deleted afterward.

## First Success

After setup:

```bash
claude
```

Then:

1. Run `/mcp`.
2. Confirm `memory`, `desire-system`, `sociality`, and `individual-kernel`.
3. Ask Claude to remember a short fact.
4. Ask for that fact in a later turn.

Claude Code loads project MCP configuration at startup. Restart it after
changing `.mcp.json`.

## Troubleshooting

### `uv` is missing

POSIX:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal and rerun setup.

### A Core server is disconnected

1. Restart Claude Code after setup.
2. Run `./scripts/doctor.sh --live`, or `scripts\doctor.cmd --live` on Windows.
3. Run `uv sync --locked` if the workspace or lock check fails.
4. Inspect `/mcp` for the server's stderr.

### Setup refuses an existing config

This is intentional. Inspect the redacted proposal with `--dry-run`, preserve
custom entries, then use `--force` only when replacing the file is intended.

### The first memory call downloads a model

Setup was run with `--skip-model-download`, or the selected model cache was
removed. Rerun setup without that option.

### Camera or audio is unavailable

Core can still operate. Run the doctor, then follow the package-specific
hardware guide linked above. Setup does not install device drivers or vendor
software.
