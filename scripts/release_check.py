#!/usr/bin/env python3
"""Validate a workspace release before publishing it."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseCheckError(RuntimeError):
    """A release invariant was not satisfied."""


def workspace_versions(repo_root: Path = REPO_ROOT) -> dict[Path, str]:
    """Return every workspace project version keyed by relative pyproject path."""

    versions: dict[Path, str] = {}
    for path in sorted(repo_root.rglob("pyproject.toml")):
        if ".venv" in path.parts:
            continue
        relative = path.relative_to(repo_root)
        try:
            version = tomllib.loads(path.read_text())["project"]["version"]
        except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
            raise ReleaseCheckError(f"cannot read project version from {relative}: {error}")
        versions[relative] = str(version)
    if not versions:
        raise ReleaseCheckError("no workspace pyproject.toml files were found")
    return versions


def validate_workspace_version(
    expected_version: str,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Require all workspace packages to use one exact release version."""

    mismatches = {
        path: version
        for path, version in workspace_versions(repo_root).items()
        if version != expected_version
    }
    if mismatches:
        detail = ", ".join(f"{path}={version}" for path, version in mismatches.items())
        raise ReleaseCheckError(
            f"workspace projects must all be {expected_version}; found {detail}"
        )


def validate_tag(tag: str, version: str) -> None:
    """Require the canonical v-prefixed tag for a release version."""

    expected = f"v{version}"
    if tag != expected:
        raise ReleaseCheckError(f"release tag must be exactly {expected}; found {tag!r}")


def validate_changelog(version: str, repo_root: Path = REPO_ROOT) -> None:
    """Require the release version to have a dated changelog heading."""

    try:
        changelog = (repo_root / "CHANGELOG.md").read_text()
    except OSError as error:
        raise ReleaseCheckError(f"cannot read CHANGELOG.md: {error}")
    if f"## [{version}] - " not in changelog:
        raise ReleaseCheckError(f"CHANGELOG.md has no dated section for {version}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default=os.environ.get("GITHUB_REF_NAME"),
        help="Release tag (default: GITHUB_REF_NAME)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root_version = workspace_versions()[Path("pyproject.toml")]
    try:
        if not args.tag:
            raise ReleaseCheckError("--tag is required outside a tag workflow")
        validate_tag(args.tag, root_version)
        validate_workspace_version(root_version)
        validate_changelog(root_version)
    except ReleaseCheckError as error:
        print(f"release check failed: {error}", file=sys.stderr)
        return 1
    print(f"release check passed: {args.tag} ({len(workspace_versions())} projects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
