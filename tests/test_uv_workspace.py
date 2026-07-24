from __future__ import annotations

import json
import re
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
    "sociality-mcp/packages/interaction-orchestrator-mcp": (
        "interaction-orchestrator-mcp"
    ),
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


def test_transcription_extra_requires_python_313_compatible_numba() -> None:
    config = tomllib.loads((ROOT / "wifi-cam-mcp" / "pyproject.toml").read_text())
    assert "numba>=0.63.1" in config["project"]["optional-dependencies"]["transcribe"]


def test_installer_performs_one_workspace_sync() -> None:
    script = (ROOT / "scripts" / "install-mcps.sh").read_text()
    sync_commands = re.findall(r"^[ \t]*uv sync[ \t]*$", script, flags=re.MULTILINE)
    assert sync_commands == ["uv sync"]
    assert "MCP_DIRS=" not in script
    assert "uv run --package memory-mcp python -c" in script


def test_setup_uses_one_cross_platform_workspace_entrypoint() -> None:
    wrapper = (ROOT / "scripts" / "setup.sh").read_text()
    setup = (ROOT / "scripts" / "setup.py").read_text()
    gitignore = (ROOT / ".gitignore").read_text()

    assert "uv run --no-project --python 3.13" in wrapper
    assert 'python scripts/setup.py "$@"' in wrapper
    assert "https://astral.sh/uv/install.sh" in wrapper
    assert '["uv", "sync", "--locked"]' in setup
    assert "MCP_DIRS=" not in setup
    assert ".mcp.json.backup-*" in gitignore


def test_mcp_example_runs_python_servers_from_workspace_packages() -> None:
    config = json.loads((ROOT / ".mcp.json.example").read_text())
    expected = {
        "usb-webcam": ("usb-webcam-mcp", "usb-webcam-mcp"),
        "wifi-cam": ("wifi-cam-mcp", "wifi-cam-mcp"),
        "desire-system": ("desire-system", "desire-system"),
        "memory": ("memory-mcp", "memory-mcp"),
        "system-temperature": ("system-temperature-mcp", "system-temperature-mcp"),
        "tts": ("tts-mcp", "tts-mcp"),
        "x-mcp": ("x-mcp", "x-mcp"),
        "sociality": ("sociality-mcp", "sociality-mcp"),
        "individual-kernel": ("individual-kernel-mcp", "individual-kernel-mcp"),
    }

    servers = config["mcpServers"]
    for server_name, (package, entrypoint) in expected.items():
        assert servers[server_name]["command"] == "uv"
        assert servers[server_name]["args"] == [
            "run",
            "--package",
            package,
            entrypoint,
        ]


def test_efpf_hooks_run_from_the_root_workspace() -> None:
    prefix = (
        'uv run --directory "$CLAUDE_PROJECT_DIR" '
        "--package individual-kernel-mcp efpf-hook "
    )
    for settings_path in (
        ROOT / ".claude" / "settings.json",
        ROOT / ".claude" / "settings.example.json",
    ):
        config = json.loads(settings_path.read_text())
        commands = [
            hook["command"]
            for entries in config["hooks"].values()
            for entry in entries
            for hook in entry["hooks"]
            if "efpf-hook" in hook["command"]
        ]
        assert commands
        assert all(command.startswith(prefix) for command in commands)


def test_ci_uses_the_locked_root_workspace() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'python-version: "3.13"' in workflow
    assert workflow.count("run: uv sync --locked") == 1
    assert "working-directory:" not in workflow
    assert "uv lock --check" in workflow
    assert (
        'uv run --project "$GITHUB_WORKSPACE" --directory memory-mcp pytest -q'
    ) in workflow
    assert (
        'uv run --project "$GITHUB_WORKSPACE" --directory '
        "consciousness-mcp/packages/individual-kernel-mcp pytest -q"
    ) in workflow
    assert (
        'uv run --project "$GITHUB_WORKSPACE" --directory '
        "sociality-mcp/packages/agent-grammar pytest -q"
    ) in workflow


def test_primary_docs_describe_the_single_workspace() -> None:
    readme = (ROOT / "README.md").read_text()
    readme_ja = (ROOT / "README-ja.md").read_text()
    claude = (ROOT / "CLAUDE.md").read_text()
    consciousness = (ROOT / "consciousness-mcp" / "README.md").read_text()
    benchmark = (
        ROOT / "benchmarks" / "phenomenal_candidate" / "README.md"
    ).read_text()
    kernel = (
        ROOT / "consciousness-mcp" / "packages" / "individual-kernel-mcp" / "README.md"
    ).read_text()

    assert "Python 3.13" in readme
    assert "Python 3.13" in readme_ja
    assert "single root `.venv`" in readme
    assert "単一の root `.venv`" in readme_ja
    assert "uv sync --extra dev" not in claude
    assert "--package individual-kernel-mcp" in consciousness
    assert "--package individual-kernel-mcp" in kernel
    assert "python benchmarks/phenomenal_candidate/run.py" in consciousness
    assert "python benchmarks/phenomenal_candidate/run.py" in benchmark


def test_docs_bound_platform_and_global_memory_launch() -> None:
    readme = (ROOT / "README.md").read_text()
    readme_ja = (ROOT / "README-ja.md").read_text()
    memory = (ROOT / "memory-mcp" / "README.md").read_text()

    assert "macOS (Apple Silicon)" in readme
    assert "macOS（Apple Silicon）" in readme_ja
    assert (
        '"--directory", "/path/to/embodied-claude", '
        '"--package", "memory-mcp"'
    ) in memory


def test_primary_docs_lead_with_the_guided_core_setup() -> None:
    readme = (ROOT / "README.md").read_text()
    readme_ja = (ROOT / "README-ja.md").read_text()
    setup_guide = (ROOT / "docs" / "setup.md").read_text()

    for document in (readme, readme_ja):
        assert "lifemate-ai/embodied-claude" in document
        assert "--profile core --non-interactive" in document
        assert "/mcp" in document
        assert "--with-camera" in document
        assert "--with-voice" in document
        assert "--with-x" in document
        assert "--with-system-temperature" in document
        assert "scripts/doctor.py" in document
        assert "docs/setup.md" in document
        assert "kmizu/embodied-claude" not in document
        assert "cp .env.example .env" not in document

    assert "Windows native" in readme
    assert "Windows ネイティブ" in readme_ja
    assert "TAPO_CAMERA_HOST" in setup_guide
    assert "ELEVENLABS_API_KEY" in setup_guide
    assert "X_ACCESS_TOKEN_SECRET" in setup_guide
    assert ".mcp.json.backup-" in setup_guide
