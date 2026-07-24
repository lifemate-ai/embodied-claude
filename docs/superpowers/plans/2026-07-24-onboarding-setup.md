# Friendly Onboarding and Setup Implementation Plan

> **For Codex:** Execute this plan task by task. Use test-driven development:
> add each failing test first, confirm the expected failure, implement the
> smallest complete behavior, and rerun the focused test before continuing.

**Goal:** Give new users one safe, cross-platform setup path that creates a
hardware-free Core configuration by default, offers explicit optional
capabilities, diagnoses startup blockers, and leads the READMEs with a concrete
first-success journey.

**Architecture:** Keep configuration generation and validation in a
standard-library-only `scripts.onboarding` module. Thin CLI adapters call that
module, run workspace commands, and perform atomic filesystem changes. The
doctor shares the same server definitions so generation and validation cannot
drift. Root tests cover the pure domain behavior, subprocess-facing CLI
behavior, and documentation contract.

**Tech Stack:** Python 3.13 standard library, `argparse`, `dataclasses`,
`json`, `pathlib`, `subprocess`, `tomllib`, pytest, Ruff, uv.

---

## Task 1: Pin the onboarding contract with failing tests

**Files:**

- Create: `scripts/__init__.py`
- Create: `tests/test_onboarding.py`

### Step 1: Add the pure configuration expectations

Add tests that import these intended public names:

```python
from scripts.onboarding import (
    FeatureSelection,
    build_mcp_config,
    configs_equivalent,
    missing_environment,
    redact_config,
)
```

Cover:

- `FeatureSelection()` generates exactly `memory`, `desire-system`,
  `sociality`, and `individual-kernel`.
- Every generated server uses
  `uv run --package <package> <entrypoint>`.
- Core injects `MEMORY_EMBEDDING_MODEL=intfloat/multilingual-e5-small`.
- `embedding_model="base"` selects `intfloat/multilingual-e5-base`.
- USB and system-temperature selections add only their respective servers.
- Tapo, VOICEVOX, ElevenLabs, and X selections copy only canonical environment
  keys.
- ElevenLabs requires only `ELEVENLABS_API_KEY`; its voice ID is optional.
- Unknown placeholder-like credential values are rejected.
- Redaction removes secrets without changing non-secret values.
- JSON objects with different key order/formatting compare equivalent.

### Step 2: Keep the tests deterministic

Use injected environment mappings. The profile-generation tests must not need
hardware, credentials, network access, or filesystem writes.

### Step 3: Run the tests and confirm the intended failure

Run:

```bash
uv run pytest tests/test_onboarding.py -q
```

Expected: collection fails because `scripts.onboarding` does not exist yet.

### Step 4: Keep the red checkpoint local

Do not commit the intentionally failing state. Continue directly to Task 2 and
commit the tests together with the first green implementation.

## Task 2: Implement canonical profile generation

**Files:**

- Create: `scripts/onboarding.py`
- Test: `tests/test_onboarding.py`

### Step 1: Define typed selections and server specifications

Implement immutable data models:

```python
@dataclass(frozen=True)
class FeatureSelection:
    profile: str = "core"
    camera: str | None = None
    voice: str | None = None
    x_enabled: bool = False
    system_temperature: bool = False
    embedding_model: str = "small"


@dataclass(frozen=True)
class ServerSpec:
    name: str
    package: str
    entrypoint: str
    required_environment: tuple[str, ...] = ()
    optional_environment: tuple[str, ...] = ()
```

Keep a single registry for all setup-managed servers. Map voice choices to the
same `tts` server with different environment validation.

### Step 2: Generate the selected `.mcp.json`

Implement:

```python
def build_mcp_config(
    selection: FeatureSelection,
    environment: Mapping[str, str],
) -> dict[str, object]:
    ...
```

Requirements:

- deterministic server order
- no unselected server entries
- no placeholder values
- Core independent of ambient optional credentials
- POSIX-neutral `uv` command
- existing individual-kernel compatibility environment retained
- only selected secrets copied into the generated config

### Step 3: Implement environment diagnostics and redaction

Implement:

```python
def missing_environment(...) -> tuple[str, ...]: ...
def redact_config(config: Mapping[str, object]) -> dict[str, object]: ...
def configs_equivalent(left: object, right: object) -> bool: ...
```

Treat keys containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD` as secrets.
Treat Tapo username and ElevenLabs voice ID as private configuration too.

### Step 4: Run focused tests

Run:

```bash
uv run pytest tests/test_onboarding.py -q -k \
  "core or optional or environment or redaction or equivalent"
uv run ruff check scripts/onboarding.py tests/test_onboarding.py
```

Expected: all selected tests and Ruff pass.

### Step 5: Commit

```bash
git add scripts/__init__.py scripts/onboarding.py \
  tests/test_onboarding.py
git commit -m "feat: add onboarding profile generation"
```

## Task 3: Implement safe filesystem operations

**Files:**

- Modify: `scripts/onboarding.py`
- Modify: `tests/test_onboarding.py`
- Modify: `.gitignore`

### Step 1: Add failing filesystem safety tests

Test intended helpers for:

- a new config write
- idempotent semantic-equivalent config
- refusal to replace a different config without `force=True`
- timestamped backup before forced replacement
- atomic destination mode `0600` on POSIX
- social policy copy only when absent

Use `tmp_path` and an injected clock. Run the focused tests and confirm they
fail because the filesystem helpers do not exist.

### Step 2: Implement explicit config decisions

Represent the decision before writing:

```python
class ConfigAction(StrEnum):
    CREATE = "create"
    KEEP = "keep"
    REPLACE = "replace"


@dataclass(frozen=True)
class ConfigPlan:
    action: ConfigAction
    destination: Path
    backup: Path | None = None
```

`plan_config_write()` must parse an existing file and:

- return `KEEP` for semantically equivalent JSON
- raise a user-facing conflict error for invalid/different JSON without force
- choose a unique timestamped backup path when forced

### Step 3: Implement atomic writes and policy copy

Implement:

- same-directory temporary file
- `json.dump(..., indent=2)` plus final newline
- flush and `os.fsync`
- `os.replace`
- `chmod(0o600)` on POSIX
- backup via `shutil.copy2` before replacement
- policy copy only when the target is absent

Do not write or create anything from dry-run code paths.

### Step 4: Ignore generated backups

Add:

```gitignore
.mcp.json.backup-*
```

Keep `.mcp.json.example` tracked.

### Step 5: Run focused tests

Run:

```bash
uv run pytest tests/test_onboarding.py -q -k \
  "write or backup or force or policy or equivalent"
uv run ruff check scripts/onboarding.py tests/test_onboarding.py
```

### Step 6: Commit

```bash
git add scripts/onboarding.py tests/test_onboarding.py .gitignore
git commit -m "feat: protect generated MCP configuration"
```

## Task 4: Implement the read-only doctor

**Files:**

- Create: `scripts/doctor.py`
- Modify: `scripts/onboarding.py`
- Modify: `tests/test_onboarding.py`

### Step 1: Add failing doctor tests

Cover:

- nearest-existing-parent writability without directory creation
- unknown custom MCP entries reported as warnings
- malformed known setup-managed entries reported as errors
- missing optional playback/hardware tools reported as warnings
- missing Core package/entrypoint reported as errors

Inject command lookup and subprocess runner functions. Run
`uv run pytest tests/test_onboarding.py -q -k doctor` and confirm the expected
missing-symbol failures.

### Step 2: Add diagnostic result types

Implement:

```python
class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    subject: str
    detail: str
    remediation: str | None = None
```

Pure checker functions receive a repository root, home path, parsed config,
`shutil.which`-style lookup function, and subprocess runner.

### Step 3: Validate the shared setup-managed shapes

For each known entry validate:

- `command == "uv"`
- exact package and entrypoint arguments
- required selected environment values
- no placeholder values
- Core packages are present in the root workspace

Unknown custom server entries produce `[warn]`, never `[error]`.

### Step 4: Add platform and dependency checks

Check:

- Python `3.13.x`
- root `pyproject.toml` and `uv.lock`
- `uv lock --check`
- `socialPolicy.toml` TOML validity when present
- nearest writable parent for required state paths without creating paths
- `ffmpeg`, `mpv`/`ffplay`, and supported hardware conditions as warnings

Do not import or start MCP servers in the doctor. Validate package presence from
workspace metadata and entrypoint metadata so the command stays side-effect
free.

### Step 5: Build the CLI

Support:

```bash
uv run python scripts/doctor.py
uv run python scripts/doctor.py --config /path/to/.mcp.json
```

Print `[ok]`, `[warn]`, and `[error]` lines. Exit `1` iff at least one error is
present, otherwise `0`.

### Step 6: Run focused tests

Run:

```bash
uv run pytest tests/test_onboarding.py -q -k doctor
uv run ruff check scripts/doctor.py scripts/onboarding.py tests/test_onboarding.py
```

### Step 7: Commit

```bash
git add scripts/doctor.py scripts/onboarding.py tests/test_onboarding.py
git commit -m "feat: add onboarding doctor"
```

## Task 5: Implement the setup CLI and POSIX wrapper

**Files:**

- Create: `scripts/setup.py`
- Create: `scripts/setup.sh`
- Modify: `tests/test_onboarding.py`
- Modify: `tests/test_uv_workspace.py`
- Modify: `scripts/install-mcps.sh`

### Step 1: Add CLI subprocess tests before implementation

Run safe CLI paths with `sys.executable`. Exercise mutating orchestration through
functions with injected repository/home paths and subprocess runners rather
than adding test-only CLI flags. Cover:

- `--help`
- Core `--non-interactive --dry-run`
- dry-run redacts selected secrets
- missing selected environment values exit `2` and name every variable
- mutually exclusive camera/voice choices are rejected by argparse
- existing different `.mcp.json` refuses without `--force`
- `--force` backs up then replaces
- dry-run does not call sync, model warmup, doctor, or write files

Avoid invoking `uv sync` or downloading a model in unit tests.

### Step 2: Implement argument parsing and wizard

Stable flags:

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

With no arguments, ask concise yes/no and choice questions. Read secrets using
`getpass.getpass`. Keep all prompts in `setup.py`; domain logic stays pure.

### Step 3: Implement orchestration

Execution order:

1. Resolve and validate the repository root.
2. Build and validate the proposal.
3. Print and exit for dry-run.
4. Run `uv sync --locked`.
5. Plan and safely write `.mcp.json`.
6. Copy `socialPolicy.toml` if absent.
7. Warm the selected memory embedding model unless skipped.
8. Run `scripts/doctor.py`.
9. Print the exact `/mcp`, remember, and recall success steps.

Use argument lists for subprocesses; never `shell=True`.

### Step 4: Add the Bash adapter

`scripts/setup.sh` must:

- use `set -euo pipefail`
- resolve its own repository root
- fail with the official `uv` install URL when `uv` is missing
- execute:

```bash
uv run --no-project --python 3.13 python scripts/setup.py "$@"
```

Mark it executable.

### Step 5: Keep the old installer as a compatibility alias

Change `scripts/install-mcps.sh` to explain that setup is the recommended
first-run path while preserving its current sync-and-warm behavior for existing
automation. Do not add a second workspace loop.

### Step 6: Run focused tests

Run:

```bash
uv run pytest tests/test_onboarding.py tests/test_uv_workspace.py -q
uv run ruff check scripts/setup.py scripts/doctor.py scripts/onboarding.py tests
```

### Step 7: Commit

```bash
git add scripts/setup.py scripts/setup.sh scripts/install-mcps.sh \
  tests/test_onboarding.py tests/test_uv_workspace.py
git commit -m "feat: add guided workspace setup"
```

## Task 6: Replace capability-first root documentation

**Files:**

- Rewrite: `README.md`
- Rewrite: `README-ja.md`
- Create: `docs/setup.md`
- Modify: `.mcp.json.example`
- Modify: `tests/test_uv_workspace.py`

### Step 1: Add documentation contract tests

Assert both root READMEs contain:

- current `lifemate-ai/embodied-claude` clone URL
- Core setup commands
- first-success checklist
- exact optional setup flags
- doctor command
- Python 3.13 and one root `.venv`
- platform support table
- strict wording that hardware and cloud features are optional

Assert they do not contain:

- `kmizu/embodied-claude`
- instructions to create package-local `.env` files
- claims that Windows native is wholly unsupported

### Step 2: Rewrite the root README opening

Use this order:

1. one paragraph explaining the experience
2. prerequisites
3. three-command Core quick start
4. four-item success checklist
5. optional capability chooser
6. platform support
7. concise causal/runtime architecture
8. package reference
9. research claim limitations, attribution, and license

Keep detailed feature inventories below the first-success path.

### Step 3: Mirror the journey in Japanese

Keep command blocks and option names identical to English. Translate the user
journey naturally rather than maintaining divergent sections.

### Step 4: Write the detailed setup guide

Document:

- interactive and non-interactive Core
- selected environment variables
- memory model choice
- each optional capability
- Windows PowerShell command
- backup/force behavior
- doctor meanings and remediation
- migration from manually copied `.mcp.json`

Link to package READMEs for hardware-specific procedures.

### Step 5: Make the full example clearly advanced

Retain `.mcp.json.example`, remove obsolete `garmin-health`, update comments or
adjacent documentation so users do not copy it as the default path, and keep
its workspace commands aligned with the registry.

### Step 6: Run docs tests and scans

Run:

```bash
uv run pytest tests/test_uv_workspace.py tests/test_onboarding.py -q
rg -n "kmizu/embodied-claude|cd .* && uv sync|package-local .*\\.env" \
  README.md README-ja.md docs/setup.md
git diff --check
```

The `rg` command should return no obsolete onboarding references.

### Step 7: Commit

```bash
git add README.md README-ja.md docs/setup.md .mcp.json.example \
  tests/test_uv_workspace.py
git commit -m "docs: lead with a working core setup"
```

## Task 7: Prove clean-environment behavior

**Files:**

- Modify as needed: `scripts/setup.py`
- Modify as needed: `scripts/doctor.py`
- Modify as needed: `tests/test_onboarding.py`
- Create: `tests/fixtures/onboarding/` only if static sample configs improve
  readability

### Step 1: Run a no-side-effect dry run

From the repository root:

```bash
tmp_home="$(mktemp -d)"
HOME="$tmp_home" ./scripts/setup.sh \
  --profile core \
  --non-interactive \
  --dry-run
test ! -e "$tmp_home/.claude"
test ! -e .mcp.json
```

Expected: four Core servers are listed, no paths are created, no config is
written.

### Step 2: Run a real setup in an isolated repository copy

Use a temporary copy or worktree so the production `.mcp.json` is untouched:

```bash
tmp_repo="$(mktemp -d)"
git archive HEAD | tar -x -C "$tmp_repo"
cd "$tmp_repo"
HOME="$(mktemp -d)" uv run --no-project --python 3.13 \
  python scripts/setup.py \
  --profile core \
  --non-interactive \
  --skip-model-download
```

Expected:

- one root sync
- `.mcp.json` with exactly four Core servers
- `socialPolicy.toml` created
- doctor exits zero
- rerunning succeeds without rewriting the semantically equivalent config

### Step 3: Exercise refusal and forced backup

Add a custom/different `.mcp.json`, rerun without `--force`, and verify refusal.
Then rerun with `--force` and verify exactly one timestamped backup contains the
old content.

### Step 4: Exercise optional dry runs

Run redacted dry-runs for Tapo, VOICEVOX, ElevenLabs, and X with fixture
environment values. Confirm no secret appears in stdout or stderr.

### Step 5: Fix any failures test-first

For each defect, add or tighten a unit test before modifying implementation.

### Step 6: Commit smoke fixes

```bash
git add scripts tests
git commit -m "test: cover clean onboarding smoke paths"
```

Skip the commit if no files changed.

## Task 8: Run the complete verification matrix

### Step 1: Run root lint and tests

```bash
uv lock --check
uv run ruff check scripts tests
uv run pytest tests -q
```

### Step 2: Run the repository CI commands

```bash
uv run ruff check memory-mcp
uv run ruff check tts-mcp
uv run ruff check wifi-cam-mcp
uv run ruff check desire-system
uv run ruff check system-temperature-mcp
uv run ruff check consciousness-mcp/packages/individual-kernel-mcp
uv run ruff check sociality-mcp
uv run mypy memory-mcp/src/memory_mcp/ --ignore-missing-imports
```

Then run every pytest command listed in `.github/workflows/ci.yml`.

### Step 3: Verify the final diff

```bash
git status --short
git diff origin/main...HEAD --check
git diff --stat origin/main...HEAD
```

Inspect every changed file for secrets and generated local paths:

```bash
git diff origin/main...HEAD | rg \
  "API_KEY|PASSWORD|ACCESS_TOKEN|/home/|/tmp/embodied-claude"
```

Only documented variable names and test fixtures may match.

## Task 9: Publish and verify the pull request

### Step 1: Push the branch

```bash
git push -u origin docs/onboarding
```

### Step 2: Open a draft pull request against `main`

The PR body must include:

- purpose-first onboarding summary
- generated Core server list
- config preservation/backup behavior
- Windows and POSIX entrypoints
- doctor behavior
- exact test/lint/smoke evidence
- explicit note that hardware integrations were validated deterministically,
  not against connected hardware

### Step 3: Inspect GitHub checks

Use `gh pr checks --watch` or the GitHub connector. If a check fails, inspect
the exact log, reproduce locally, add a regression test, fix, push, and wait for
green.

### Step 4: Report the usable path

Give the user:

- PR URL
- Core setup command
- doctor command
- tests/lint results
- any remaining platform/hardware limitation
