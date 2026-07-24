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
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_root_declares_every_python_project() -> None:
    config = _root_config()
    assert set(config["tool"]["uv"]["workspace"]["members"]) == set(MEMBERS)
    assert config["project"]["requires-python"] == ">=3.13,<3.14"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.13"


def test_root_dependencies_install_only_the_core_runtime() -> None:
    config = _root_config()
    dependency_names = {
        value.split("[", 1)[0].split("=", 1)[0]
        for value in config["project"]["dependencies"]
    }
    assert dependency_names == {
        "desire-system",
        "individual-kernel-mcp",
        "memory-mcp",
        "sociality-mcp",
    }
    assert set(config["tool"]["uv"]["sources"]) == set(MEMBERS.values())
    assert all(
        source == {"workspace": True}
        for source in config["tool"]["uv"]["sources"].values()
    )


def test_root_extras_install_optional_capabilities_explicitly() -> None:
    extras = _root_config()["project"]["optional-dependencies"]

    assert extras["camera-usb"] == ["usb-webcam-mcp"]
    assert extras["camera-tapo"] == ["wifi-cam-mcp"]
    assert extras["transcription-whisper"] == ["wifi-cam-mcp[transcribe]"]
    assert extras["transcription-faster"] == ["wifi-cam-mcp[transcribe-faster]"]
    assert extras["voice-voicevox"] == ["tts-mcp"]
    assert extras["voice-elevenlabs"] == ["tts-mcp[elevenlabs]"]
    assert extras["x"] == ["x-mcp"]
    assert extras["system-temperature"] == ["system-temperature-mcp"]
    assert all("transcrib" not in item for item in extras["camera-tapo"])


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
    config = tomllib.loads((ROOT / "wifi-cam-mcp" / "pyproject.toml").read_text(encoding="utf-8"))
    assert "numba>=0.63.1" in config["project"]["optional-dependencies"]["transcribe"]


def test_installer_performs_one_workspace_sync() -> None:
    script = (ROOT / "scripts" / "install-mcps.sh").read_text(encoding="utf-8")
    sync_commands = re.findall(r"^[ \t]*uv sync.*$", script, flags=re.MULTILINE)
    assert sync_commands == ["uv sync --locked --all-extras --group dev"]
    assert "MCP_DIRS=" not in script
    assert "uv run --package memory-mcp python -c" in script


def test_setup_uses_one_cross_platform_workspace_entrypoint() -> None:
    wrapper = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    setup = (ROOT / "scripts" / "setup.py").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "uv run --no-project --python 3.13" in wrapper
    assert 'python scripts/setup.py "$@"' in wrapper
    assert "https://astral.sh/uv/install.sh" in wrapper
    assert '["uv", "sync", "--locked", "--no-dev"]' in setup
    assert "MCP_DIRS=" not in setup
    assert ".mcp.json.backup-*" in gitignore


def test_mcp_example_runs_python_servers_from_workspace_packages() -> None:
    config = json.loads((ROOT / ".mcp.json.example").read_text(encoding="utf-8"))
    expected = {
        "desire-system": ("desire-system", "desire-system"),
        "memory": ("memory-mcp", "memory-mcp"),
        "sociality": ("sociality-mcp", "sociality-mcp"),
        "individual-kernel": ("individual-kernel-mcp", "individual-kernel-mcp"),
    }

    servers = config["mcpServers"]
    assert set(servers) == set(expected)
    for server_name, (package, entrypoint) in expected.items():
        assert servers[server_name]["command"] == "uv"
        assert servers[server_name]["args"] == [
            "run",
            "--package",
            package,
            entrypoint,
        ]


def test_efpf_hooks_run_from_the_root_workspace() -> None:
    for settings_path in (
        ROOT / ".claude" / "settings.json",
        ROOT / ".claude" / "settings.example.json",
    ):
        config = json.loads(settings_path.read_text(encoding="utf-8"))
        hooks = [
            hook
            for entries in config["hooks"].values()
            for entry in entries
            for hook in entry["hooks"]
            if "efpf-hook" in hook["command"]
            or "efpf-hook" in hook.get("args", [])
        ]
        assert hooks
        for hook in hooks:
            assert hook["command"] == "uv"
            assert hook["args"][:5] == [
                "run",
                "--directory",
                "${CLAUDE_PROJECT_DIR}",
                "--package",
                "individual-kernel-mcp",
            ]
            assert hook["args"][5] == "efpf-hook"
            assert "$CLAUDE_PROJECT_DIR" not in json.dumps(hook)


def test_core_hook_settings_do_not_require_posix_shell_scripts() -> None:
    config = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    hooks = [
        hook
        for entries in config["hooks"].values()
        for entry in entries
        for hook in entry["hooks"]
    ]

    assert all(".sh" not in json.dumps(hook) for hook in hooks)


def test_ci_uses_the_locked_root_workspace() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.13"' in workflow
    assert "run: uv sync --locked --all-extras --group dev" in workflow
    assert "run: uv sync --locked --group dev" in workflow
    assert "run: uv run ruff check ." in workflow
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


def test_ci_release_gate_covers_linux_macos_and_windows() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "ubuntu-latest" in workflow
    assert "macos-latest" in workflow
    assert "windows-latest" in workflow
    assert "./scripts/setup.sh" in workflow
    assert "./scripts/doctor.sh" in workflow
    assert r"scripts\setup.cmd" in workflow
    assert r"scripts\doctor.cmd" in workflow
    assert "test_embedding_warmup.py" in workflow
    assert "--live" in workflow


def test_primary_docs_describe_the_single_workspace() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_ja = (ROOT / "README-ja.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    consciousness = (ROOT / "consciousness-mcp" / "README.md").read_text(encoding="utf-8")
    benchmark = (
        ROOT / "benchmarks" / "phenomenal_candidate" / "README.md"
    ).read_text(encoding="utf-8")
    kernel = (
        ROOT / "consciousness-mcp" / "packages" / "individual-kernel-mcp" / "README.md"
    ).read_text(encoding="utf-8")

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
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_ja = (ROOT / "README-ja.md").read_text(encoding="utf-8")
    memory = (ROOT / "memory-mcp" / "README.md").read_text(encoding="utf-8")

    assert "macOS (Apple Silicon)" in readme
    assert "macOS（Apple Silicon）" in readme_ja
    assert (
        '"--directory", "/path/to/embodied-claude", '
        '"--package", "memory-mcp"'
    ) in memory


def test_primary_docs_lead_with_the_guided_core_setup() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_ja = (ROOT / "README-ja.md").read_text(encoding="utf-8")
    setup_guide = (ROOT / "docs" / "setup.md").read_text(encoding="utf-8")

    for document in (readme, readme_ja):
        assert "lifemate-ai/embodied-claude" in document
        assert "--profile core --non-interactive" in document
        assert "/mcp" in document
        assert "--with-camera" in document
        assert "--with-voice" in document
        assert "--with-x" in document
        assert "--with-system-temperature" in document
        assert "docs/setup.md" in document
        assert "kmizu/embodied-claude" not in document
        assert "cp .env.example .env" not in document

    assert "Windows native" in readme
    assert "Windows ネイティブ" in readme_ja
    assert "TAPO_CAMERA_HOST" in setup_guide
    assert "ELEVENLABS_API_KEY" in setup_guide
    assert "X_ACCESS_TOKEN_SECRET" in setup_guide
    assert ".mcp.json.backup-" in setup_guide


def test_primary_docs_make_windows_and_live_diagnostics_first_class() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_ja = (ROOT / "README-ja.md").read_text(encoding="utf-8")
    setup_guide = (ROOT / "docs" / "setup.md").read_text(encoding="utf-8")

    for document in (readme, readme_ja, setup_guide):
        assert r"scripts\setup.cmd" in document
        assert r"scripts\doctor.cmd --live" in document
    assert "--with-transcription whisper|faster" in document
    assert "Windows 11" in readme
    assert "Windows 11" in setup_guide
    assert "WSL2 is not required" in setup_guide
