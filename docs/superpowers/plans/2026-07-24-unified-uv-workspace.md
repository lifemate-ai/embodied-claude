# Unified uv Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one root `uv sync` install all 17 Python projects, full runtime
extras, and contributor tools into one root environment governed by one lock.

**Architecture:** The repository root becomes a non-package aggregator project
and uv workspace. All first-party dependencies resolve with
`{ workspace = true }`; the root depends on every member and owns the only
`.python-version`, `.venv`, and `uv.lock`.

**Tech Stack:** uv workspaces, PEP 621 `pyproject.toml`, Python 3.13, pytest,
Ruff, GitHub Actions, Bash.

## Global Constraints

- Include all 17 Python projects listed in the approved design.
- Use Python `>=3.13,<3.14` and a root `.python-version` containing `3.13`.
- Plain `uv sync` must install every member, dev tools, `tts-mcp[all]`, and
  `wifi-cam-mcp[transcribe]`.
- Keep member build systems, entry points, extras, Ruff, and pytest settings.
- Keep existing member-directory `uv run --directory` callers working through workspace
  discovery.
- Leave Node projects, credentials, ignored user files, and hardware behavior
  unchanged.
- Store exactly one tracked `uv.lock` and one tracked `.python-version`, both at
  the repository root.

---

### Task 1: Root Workspace And Single Lock

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `tests/test_uv_workspace.py`
- Create: `uv.lock` with `uv lock`
- Modify: every member `pyproject.toml` that currently has `[tool.uv.sources]`
- Delete: all nested `uv.lock` files
- Delete: all nested `.python-version` files

**Interfaces:**
- Consumes: the 17 existing PEP 621 project names and their dependency metadata.
- Produces: root workspace commands such as `uv sync`,
  `uv run --package memory-mcp memory-mcp`, and
  one shared editable environment.

- [ ] **Step 1: Write the failing workspace structure test**

Create `tests/test_uv_workspace.py`:

```python
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
MEMBERS = {
    "consciousness-mcp/packages/individual-kernel-mcp": "individual-kernel-mcp",
    "desire-system": "desire-system",
    "memory-mcp": "memory-mcp",
    "sociality-mcp": "sociality-mcp",
    "sociality-mcp/packages/agent-grammar": "agent-grammar",
    "sociality-mcp/packages/boundary-mcp": "boundary-mcp",
    "sociality-mcp/packages/interaction-orchestrator-mcp":
        "interaction-orchestrator-mcp",
    "sociality-mcp/packages/joint-attention-mcp": "joint-attention-mcp",
    "sociality-mcp/packages/relationship-mcp": "relationship-mcp",
    "sociality-mcp/packages/self-narrative-mcp": "self-narrative-mcp",
    "sociality-mcp/packages/social-core": "social-core",
    "sociality-mcp/packages/social-state-mcp": "social-state-mcp",
    "system-temperature-mcp": "system-temperature-mcp",
    "tts-mcp": "tts-mcp",
    "usb-webcam-mcp": "usb-webcam-mcp",
    "wifi-cam-mcp": "wifi-cam-mcp",
    "x-mcp": "x-mcp",
}


def _root_config() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_root_declares_every_python_project() -> None:
    config = _root_config()
    assert set(config["tool"]["uv"]["workspace"]["members"]) == set(MEMBERS)
    assert config["project"]["requires-python"] == ">=3.13,<3.14"
    assert (ROOT / ".python-version").read_text().strip() == "3.13"


def test_root_depends_on_every_workspace_member() -> None:
    config = _root_config()
    dependency_names = {
        value.split("[", 1)[0].split("=", 1)[0]
        for value in config["project"]["dependencies"]
    }
    assert dependency_names == set(MEMBERS.values())
    assert set(config["tool"]["uv"]["sources"]) == set(MEMBERS.values())
    assert all(
        source == {"workspace": True}
        for source in config["tool"]["uv"]["sources"].values()
    )


def test_only_root_lock_and_python_pin_remain() -> None:
    nested_locks = [
        path.relative_to(ROOT)
        for path in ROOT.glob("**/uv.lock")
        if path.parent != ROOT and "tmp" not in path.parts
    ]
    nested_pins = [
        path.relative_to(ROOT)
        for path in ROOT.glob("**/.python-version")
        if path.parent != ROOT and "tmp" not in path.parts
    ]
    assert nested_locks == []
    assert nested_pins == []
```

- [ ] **Step 2: Run the structure test and verify RED**

Run:

```bash
uv run --project memory-mcp pytest tests/test_uv_workspace.py -q
```

Expected: FAIL because root `pyproject.toml` and root `.python-version` do not
exist and nested locks remain.

- [ ] **Step 3: Create the root aggregator manifest**

Create `pyproject.toml` with:

```toml
[project]
name = "embodied-claude-workspace"
version = "0.1.0"
description = "Unified development and runtime environment for embodied-claude MCP servers."
requires-python = ">=3.13,<3.14"
dependencies = [
    "agent-grammar",
    "boundary-mcp",
    "desire-system",
    "individual-kernel-mcp",
    "interaction-orchestrator-mcp",
    "joint-attention-mcp",
    "memory-mcp",
    "relationship-mcp",
    "self-narrative-mcp",
    "social-core",
    "social-state-mcp",
    "sociality-mcp",
    "system-temperature-mcp",
    "tts-mcp[all]",
    "usb-webcam-mcp",
    "wifi-cam-mcp[transcribe]",
    "x-mcp",
]

[dependency-groups]
dev = [
    "freezegun>=1.5.0",
    "jurigged>=0.6.0",
    "mypy>=1.19.1",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
]

[tool.uv]
package = false

[tool.uv.workspace]
members = [
    "consciousness-mcp/packages/individual-kernel-mcp",
    "desire-system",
    "memory-mcp",
    "sociality-mcp",
    "sociality-mcp/packages/agent-grammar",
    "sociality-mcp/packages/boundary-mcp",
    "sociality-mcp/packages/interaction-orchestrator-mcp",
    "sociality-mcp/packages/joint-attention-mcp",
    "sociality-mcp/packages/relationship-mcp",
    "sociality-mcp/packages/self-narrative-mcp",
    "sociality-mcp/packages/social-core",
    "sociality-mcp/packages/social-state-mcp",
    "system-temperature-mcp",
    "tts-mcp",
    "usb-webcam-mcp",
    "wifi-cam-mcp",
    "x-mcp",
]

[tool.uv.sources]
agent-grammar = { workspace = true }
boundary-mcp = { workspace = true }
desire-system = { workspace = true }
individual-kernel-mcp = { workspace = true }
interaction-orchestrator-mcp = { workspace = true }
joint-attention-mcp = { workspace = true }
memory-mcp = { workspace = true }
relationship-mcp = { workspace = true }
self-narrative-mcp = { workspace = true }
social-core = { workspace = true }
social-state-mcp = { workspace = true }
sociality-mcp = { workspace = true }
system-temperature-mcp = { workspace = true }
tts-mcp = { workspace = true }
usb-webcam-mcp = { workspace = true }
wifi-cam-mcp = { workspace = true }
x-mcp = { workspace = true }
```

Create `.python-version` with exactly:

```text
3.13
```

- [ ] **Step 4: Remove redundant member source tables and state**

Remove each member `[tool.uv.sources]` table because root workspace sources
apply to every member. Delete all 17 nested `uv.lock` files and these pins:

```text
desire-system/.python-version
memory-mcp/.python-version
system-temperature-mcp/.python-version
usb-webcam-mcp/.python-version
x-mcp/.python-version
```

- [ ] **Step 5: Resolve and sync the full workspace**

Run:

```bash
uv lock
uv sync
```

Expected: one root `uv.lock`, one root `.venv`, all 17 editable members, dev
tools, ElevenLabs, and Whisper resolve on Python 3.13.

- [ ] **Step 6: Run the structure test and verify GREEN**

Run:

```bash
uv run pytest tests/test_uv_workspace.py -q
uv lock --check
```

Expected: `3 passed` and lock check succeeds.

- [ ] **Step 7: Commit the workspace graph**

```bash
git add pyproject.toml .python-version uv.lock tests/test_uv_workspace.py \
  consciousness-mcp desire-system memory-mcp sociality-mcp \
  system-temperature-mcp tts-mcp usb-webcam-mcp wifi-cam-mcp x-mcp
git commit -m "build: unify Python projects in uv workspace"
```

### Task 2: Runtime And Installer Integration

**Files:**
- Modify: `tests/test_uv_workspace.py`
- Modify: `scripts/install-mcps.sh`
- Modify: `.mcp.json.example`
- Modify: `.claude/settings.json`
- Modify: `.claude/settings.example.json`

**Interfaces:**
- Consumes: root workspace and package names from Task 1.
- Produces: compatibility installer and project-owned runtime commands that
  always target the root environment.

- [ ] **Step 1: Add failing command integration assertions**

Append:

```python
import json


def test_mcp_example_uses_workspace_packages() -> None:
    config = json.loads((ROOT / ".mcp.json.example").read_text())
    expected = {
        "usb-webcam": ("usb-webcam-mcp", "usb-webcam-mcp"),
        "wifi-cam": ("wifi-cam-mcp", "wifi-cam-mcp"),
        "desire-system": ("desire-system", "desire-system"),
        "memory": ("memory-mcp", "memory-mcp"),
        "system-temperature": (
            "system-temperature-mcp",
            "system-temperature-mcp",
        ),
        "tts": ("tts-mcp", "tts-mcp"),
        "x-mcp": ("x-mcp", "x-mcp"),
        "sociality": ("sociality-mcp", "sociality-mcp"),
        "individual-kernel": (
            "individual-kernel-mcp",
            "individual-kernel-mcp",
        ),
    }
    for server_name, (package, entry_point) in expected.items():
        assert config["mcpServers"][server_name]["args"] == [
            "run",
            "--package",
            package,
            entry_point,
        ]


def test_hooks_select_workspace_package() -> None:
    for filename in (".claude/settings.json", ".claude/settings.example.json"):
        text = (ROOT / filename).read_text()
        assert '--directory \\"$CLAUDE_PROJECT_DIR\\"' in text
        assert "--package individual-kernel-mcp" in text
        assert "consciousness-mcp/packages/individual-kernel-mcp" not in text
```

- [ ] **Step 2: Run the integration assertions and verify RED**

```bash
uv run pytest tests/test_uv_workspace.py -q
```

Expected: FAIL because MCP and hook commands still select member directories.

- [ ] **Step 3: Collapse the installer**

Replace the per-directory loop in `scripts/install-mcps.sh` with one root:

```bash
echo "==> embodied-claude workspace (uv sync)"
uv sync
```

Keep `--dev` accepted as a deprecated no-op because root dev dependencies are
installed by default. Keep the memory E5 warm-up, but run it with:

```bash
uv run --package memory-mcp python -c '
from memory_mcp.config import MemoryConfig
from memory_mcp.embedding import E5EmbeddingFunction
model = MemoryConfig.from_env().embedding_model
print(f"  warming {model}")
E5EmbeddingFunction(model)._load_model()
print("  done")
'
```

- [ ] **Step 4: Point MCP and hooks at workspace packages**

For every Python MCP in `.mcp.json.example`, use the exact package and entry
point mapping asserted in Step 1. For every EFPF hook command in both settings
files, replace the member-directory prefix with
`uv run --directory "$CLAUDE_PROJECT_DIR" --package individual-kernel-mcp`
and retain its existing `efpf-hook` subcommand.

- [ ] **Step 5: Verify runtime integration**

```bash
uv run pytest tests/test_uv_workspace.py -q
shellcheck scripts/install-mcps.sh || true
printf '{}\n' | uv run --package individual-kernel-mcp efpf-hook diagnostics | jq .
```

Expected: workspace assertions pass and hook diagnostics return JSON. A missing
optional `shellcheck` binary may be reported but is not a failure.

- [ ] **Step 6: Commit runtime integration**

```bash
git add tests/test_uv_workspace.py scripts/install-mcps.sh \
  .mcp.json.example .claude/settings.json .claude/settings.example.json
git commit -m "build: route MCP runtimes through root workspace"
```

### Task 3: CI And Documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `consciousness-mcp/README.md`
- Modify: `consciousness-mcp/packages/individual-kernel-mcp/README.md`

**Interfaces:**
- Consumes: root sync and package-selection commands.
- Produces: one documented setup path and CI verification against the shared
  lock/environment.

- [ ] **Step 1: Replace CI's per-directory synchronization**

Use a root workspace job with Python 3.13:

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.13"
- uses: astral-sh/setup-uv@v5
- name: Install workspace
  run: uv sync --locked
```

Run package suites with explicit paths through the shared environment:

```yaml
- run: uv run ruff check memory-mcp
- run: uv run mypy memory-mcp/src/memory_mcp --ignore-missing-imports
- run: uv run pytest memory-mcp/tests -v
- run: uv run ruff check tts-mcp
- run: uv run pytest tts-mcp/tests -v
- run: uv run ruff check wifi-cam-mcp
```

Add the deterministic EFPF/sociality/desire checks using their existing package
directories and configs:

```yaml
- run: uv run pytest consciousness-mcp/packages/individual-kernel-mcp/tests -q
- run: uv run pytest sociality-mcp/packages/social-core/tests -q
- run: uv run pytest sociality-mcp/packages/interaction-orchestrator-mcp/tests -q
- run: uv run pytest sociality-mcp/tests -q
- run: uv run pytest desire-system/tests -q
```

- [ ] **Step 2: Rewrite setup documentation**

Make root setup start with:

```bash
uv sync
```

Document:

```bash
uv run --package memory-mcp memory-mcp
uv run --package individual-kernel-mcp individual-kernel-mcp
uv run --package wifi-cam-mcp wifi-cam-mcp
uv run --package individual-kernel-mcp pytest \
  consciousness-mcp/packages/individual-kernel-mcp/tests
```

Remove guidance that asks users to sync individual member directories. Retain
hardware, environment-variable, secret, and strict-runtime caveats.

- [ ] **Step 3: Verify docs and workflow syntax**

```bash
rg -n "cd [^&]+&& uv sync|runs `uv sync` in each|sync.*individual" \
  README.md CLAUDE.md consciousness-mcp scripts .github
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: no obsolete per-project sync instructions. YAML parses successfully
using PyYAML already present in the workspace graph.

- [ ] **Step 4: Commit CI and docs**

```bash
git add .github/workflows/ci.yml README.md CLAUDE.md consciousness-mcp
git commit -m "docs: document unified uv workflow"
```

### Task 4: Full Shared-Environment Verification

**Files:**
- Modify only files required to fix verified workspace regressions.

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: evidence that one sync supports the repository's existing runtime
  and deterministic test surface.

- [ ] **Step 1: Verify lock, environment, and member graph**

```bash
uv sync --locked
uv lock --check
uv tree --depth 1
```

Expected: success with all 17 first-party projects in one graph.

- [ ] **Step 2: Verify representative entry points**

```bash
uv run --package individual-kernel-mcp efpf-hook --help
uv run python -c \
  "import individual_kernel_mcp, memory_mcp, tts_mcp, wifi_cam_mcp"
```

- [ ] **Step 3: Run all relevant tests**

```bash
uv run pytest consciousness-mcp/packages/individual-kernel-mcp/tests -q
uv run pytest sociality-mcp/packages/social-core/tests -q
uv run pytest sociality-mcp/packages/interaction-orchestrator-mcp/tests -q
uv run pytest sociality-mcp/tests -q
uv run pytest desire-system/tests -q
uv run pytest memory-mcp/tests -q
uv run pytest tts-mcp/tests -q
```

Expected: all suites pass without hardware or API keys.

- [ ] **Step 4: Run package-aware lint and type checks**

```bash
uv run ruff check consciousness-mcp/packages/individual-kernel-mcp
uv run ruff check sociality-mcp/packages/social-core
uv run ruff check sociality-mcp/packages/interaction-orchestrator-mcp
uv run ruff check sociality-mcp
uv run ruff check desire-system
uv run ruff check memory-mcp
uv run ruff check tts-mcp
uv run ruff check wifi-cam-mcp
uv run mypy memory-mcp/src/memory_mcp --ignore-missing-imports
```

Expected: all commands pass.

- [ ] **Step 5: Verify EFPF root workflows**

```bash
uv run --package individual-kernel-mcp python \
  benchmarks/phenomenal_candidate/run.py \
  --output-dir /tmp/efpf-workspace-benchmark
printf '{}\n' |
  SOCIAL_DB_PATH=/tmp/efpf-workspace-hook.db \
  uv run --package individual-kernel-mcp efpf-hook session-start |
  jq -e '.hookSpecificOutput.hookEventName == "SessionStart"'
```

Expected: benchmark emits `FieldIntegrityReport`; hook JSON assertion succeeds.

- [ ] **Step 6: Inspect final scope and commit fixes**

```bash
git diff --check
git status --short
git add pyproject.toml .python-version uv.lock tests scripts \
  .mcp.json.example .claude/settings.json .claude/settings.example.json \
  .github/workflows/ci.yml README.md CLAUDE.md consciousness-mcp \
  desire-system memory-mcp sociality-mcp system-temperature-mcp \
  tts-mcp usb-webcam-mcp wifi-cam-mcp x-mcp
git commit -m "test: verify unified uv workspace"
```

Do not stage `.claude/settings.local.json`, backup files, note drafts, `tmp/`,
or any user-owned untracked file.
