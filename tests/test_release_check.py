from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RELEASE_VERSION = "0.5.0"
RELEASE_TAG = f"v{RELEASE_VERSION}"


def _workspace_pyprojects() -> list[Path]:
    workspace = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return [ROOT / "pyproject.toml", *(
        member / "pyproject.toml"
        for pattern in workspace["tool"]["uv"]["workspace"]["members"]
        for member in ROOT.glob(pattern)
    )]


def test_all_workspace_projects_share_the_release_version() -> None:
    versions = {
        str(path.relative_to(ROOT)): tomllib.loads(path.read_text(encoding="utf-8"))["project"][
            "version"
        ]
        for path in _workspace_pyprojects()
    }

    assert versions
    assert set(versions.values()) == {RELEASE_VERSION}, versions


def test_release_verifier_ignores_unrelated_projects_but_checks_members(tmp_path: Path) -> None:
    from scripts.release_check import ReleaseCheckError, validate_workspace_version

    project = '[project]\nversion = "1"\n'
    (tmp_path / "pyproject.toml").write_text(
        project + '[tool.uv.workspace]\nmembers = ["member"]\n', encoding="utf-8"
    )
    for name in ("member", "unrelated"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pyproject.toml").write_text(project, encoding="utf-8")
    (tmp_path / "unrelated/pyproject.toml").write_text("invalid TOML!", encoding="utf-8")
    validate_workspace_version("1", tmp_path)
    (tmp_path / "member/pyproject.toml").write_text(project.replace('"1"', '"2"'), encoding="utf-8")
    with pytest.raises(ReleaseCheckError, match="member"):
        validate_workspace_version("1", tmp_path)


def test_runtime_server_defaults_share_the_release_version() -> None:
    runtime_configs = (
        ROOT / "memory-mcp" / "src" / "memory_mcp" / "config.py",
        ROOT / "tts-mcp" / "src" / "tts_mcp" / "config.py",
        ROOT / "wifi-cam-mcp" / "src" / "wifi_cam_mcp" / "config.py",
    )

    for path in runtime_configs:
        source = path.read_text(encoding="utf-8")
        assert f'version: str = "{RELEASE_VERSION}"' in source
        assert f'MCP_SERVER_VERSION", "{RELEASE_VERSION}"' in source


def test_changelog_contains_the_dated_release() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "## [Unreleased]" in changelog
    assert f"## [{RELEASE_VERSION}] - 2026-11-23" in changelog
    assert "phenomenal-consciousness candidate architecture" in changelog
    assert "proof of phenomenal consciousness" not in changelog


def test_release_verifier_accepts_only_the_exact_version_tag() -> None:
    release_check = importlib.import_module("scripts.release_check")

    release_check.validate_tag(RELEASE_TAG, RELEASE_VERSION)
    with pytest.raises(release_check.ReleaseCheckError, match=RELEASE_TAG):
        release_check.validate_tag("v0.3", RELEASE_VERSION)


def test_release_workflow_publishes_but_never_creates_tags() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "tags:" in workflow
    assert "v*" in workflow
    assert "contents: write" in workflow
    assert "uv sync --locked --all-extras --group dev" in workflow
    assert "scripts/release_check.py" in workflow
    assert "gh release create" in workflow
    assert "git tag" not in workflow
