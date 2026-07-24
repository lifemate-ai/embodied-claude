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
