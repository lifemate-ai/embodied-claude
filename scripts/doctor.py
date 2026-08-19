#!/usr/bin/env python3
"""Read-only diagnostics for an embodied-claude workspace configuration."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.onboarding import (
    CORE_SERVER_NAMES,
    ELEVENLABS_REQUIRED_ENVIRONMENT,
    SERVER_SPECS,
    TAPO_REQUIRED_ENVIRONMENT,
    X_REQUIRED_ENVIRONMENT,
    is_placeholder_value,
)


class CheckStatus(StrEnum):
    """Severity of one doctor check."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class CheckResult:
    """One actionable diagnostic result."""

    status: CheckStatus
    subject: str
    detail: str
    remediation: str | None = None


def check_state_path(path: Path) -> CheckResult:
    """Check future state-path writability without creating any directories."""

    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    writable = candidate.exists() and os.access(candidate, os.W_OK)
    if writable:
        return CheckResult(
            CheckStatus.OK,
            f"state:{path}",
            f"nearest existing parent is writable: {candidate}",
        )
    return CheckResult(
        CheckStatus.ERROR,
        f"state:{path}",
        f"nearest existing parent is not writable: {candidate}",
        f"Grant write access to {candidate} or choose a writable HOME.",
    )


def _server_shape(name: str) -> tuple[str, list[str]]:
    spec = SERVER_SPECS[name]
    return "uv", ["run", "--package", spec.package, spec.entrypoint]


def _placeholder_keys(server: Mapping[str, Any]) -> tuple[str, ...]:
    environment = server.get("env", {})
    if not isinstance(environment, Mapping):
        return ("env",)
    return tuple(
        str(key)
        for key, value in environment.items()
        if isinstance(value, str) and is_placeholder_value(value)
    )


def _required_server_environment(
    name: str,
    server: Mapping[str, Any],
) -> tuple[str, ...]:
    if name == "memory":
        return ("MEMORY_EMBEDDING_MODEL",)
    if name == "wifi-cam":
        return TAPO_REQUIRED_ENVIRONMENT
    if name == "x-mcp":
        return X_REQUIRED_ENVIRONMENT
    if name != "tts":
        return ()

    environment = server.get("env", {})
    if not isinstance(environment, Mapping):
        return ("env",)
    engine = environment.get("TTS_DEFAULT_ENGINE")
    if engine == "voicevox":
        return ("VOICEVOX_URL",)
    if engine == "elevenlabs":
        return ELEVENLABS_REQUIRED_ENVIRONMENT
    return ("TTS_DEFAULT_ENGINE",)


def validate_mcp_config(config: Mapping[str, Any]) -> list[CheckResult]:
    """Validate setup-managed MCP entries while tolerating custom servers."""

    servers = config.get("mcpServers")
    if not isinstance(servers, Mapping):
        return [
            CheckResult(
                CheckStatus.ERROR,
                "config:mcpServers",
                "mcpServers must be a JSON object.",
                "Regenerate the config with ./scripts/setup.sh --force.",
            )
        ]

    results: list[CheckResult] = []
    for core_name in CORE_SERVER_NAMES:
        if core_name not in servers:
            results.append(
                CheckResult(
                    CheckStatus.ERROR,
                    f"server:{core_name}",
                    "Core server is missing from the configuration.",
                    "Run ./scripts/setup.sh --profile core --force.",
                )
            )

    for raw_name, raw_server in servers.items():
        name = str(raw_name)
        if name not in SERVER_SPECS:
            results.append(
                CheckResult(
                    CheckStatus.WARN,
                    f"server:{name}",
                    "Custom MCP server is not managed or modified by setup.",
                    "Validate this custom entry manually.",
                )
            )
            continue
        if not isinstance(raw_server, Mapping):
            results.append(
                CheckResult(
                    CheckStatus.ERROR,
                    f"server:{name}",
                    "Known server entry must be a JSON object.",
                    "Regenerate the config with ./scripts/setup.sh --force.",
                )
            )
            continue

        expected_command, expected_args = _server_shape(name)
        problems: list[str] = []
        if raw_server.get("command") != expected_command:
            problems.append(f"command must be {expected_command!r}")
        if raw_server.get("args") != expected_args:
            problems.append(f"args must be {expected_args!r}")
        placeholders = _placeholder_keys(raw_server)
        if placeholders:
            problems.append(f"placeholder values remain in: {', '.join(placeholders)}")
        environment = raw_server.get("env", {})
        required = _required_server_environment(name, raw_server)
        missing = (
            required
            if not isinstance(environment, Mapping)
            else tuple(key for key in required if not environment.get(key))
        )
        if missing:
            problems.append(f"required environment values are missing: {', '.join(missing)}")

        if problems:
            results.append(
                CheckResult(
                    CheckStatus.ERROR,
                    f"server:{name}",
                    "; ".join(problems),
                    "Regenerate this entry with ./scripts/setup.sh --force.",
                )
            )
        else:
            results.append(
                CheckResult(
                    CheckStatus.OK,
                    f"server:{name}",
                    "workspace command shape is valid",
                )
            )
    return results


def _dependency_name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0]


def check_workspace_packages(
    repo_root: Path,
    config: Mapping[str, Any],
) -> list[CheckResult]:
    """Check that configured workspace package names are root dependencies."""

    pyproject_path = repo_root / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject_path.read_text())
        requirements = project["project"]["dependencies"]
        optional = project["project"].get("optional-dependencies", {})
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        return [
            CheckResult(
                CheckStatus.ERROR,
                "workspace:pyproject",
                f"cannot read root project dependencies: {error}",
                "Restore pyproject.toml and run uv sync --locked.",
            )
        ]

    declared_by: dict[str, str] = {
        _dependency_name(str(item)): "Core dependencies"
        for item in requirements
    }
    if isinstance(optional, Mapping):
        for extra, extra_requirements in optional.items():
            if not isinstance(extra_requirements, list):
                continue
            for item in extra_requirements:
                declared_by.setdefault(
                    _dependency_name(str(item)),
                    f"the {extra} extra",
                )
    servers = config.get("mcpServers", {})
    configured = servers if isinstance(servers, Mapping) else {}
    results: list[CheckResult] = []
    checked_packages: set[str] = set()
    for name in configured:
        spec = SERVER_SPECS.get(str(name))
        if spec is None or spec.package in checked_packages:
            continue
        checked_packages.add(spec.package)
        declaration = declared_by.get(spec.package)
        if declaration is not None:
            results.append(
                CheckResult(
                    CheckStatus.OK,
                    f"package:{spec.package}",
                    f"declared in {declaration}",
                )
            )
        else:
            results.append(
                CheckResult(
                    CheckStatus.ERROR,
                    f"package:{spec.package}",
                    "configured server package is missing from root dependencies",
                    "Restore the workspace declaration, then run uv sync --locked.",
                )
            )
    return results


def check_optional_dependencies(
    config: Mapping[str, Any],
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> list[CheckResult]:
    """Check non-blocking executables used by selected optional capabilities."""

    servers = config.get("mcpServers", {})
    configured = servers if isinstance(servers, Mapping) else {}
    results: list[CheckResult] = []

    if "wifi-cam" in configured:
        if which("ffmpeg"):
            results.append(
                CheckResult(CheckStatus.OK, "wifi-cam:ffmpeg", "ffmpeg is available")
            )
        else:
            results.append(
                CheckResult(
                    CheckStatus.WARN,
                    "wifi-cam:ffmpeg",
                    "camera audio and transcription need ffmpeg",
                    "Install ffmpeg, then rerun uv run python scripts/doctor.py.",
                )
            )

    if "tts" in configured:
        player = next((name for name in ("mpv", "ffplay") if which(name)), None)
        if player:
            results.append(
                CheckResult(CheckStatus.OK, "tts:playback", f"{player} is available")
            )
        else:
            results.append(
                CheckResult(
                    CheckStatus.WARN,
                    "tts:playback",
                    "local playback needs mpv or ffplay",
                    "Install mpv (recommended) or ffmpeg/ffplay.",
                )
            )
    return results


def _load_config(path: Path) -> tuple[dict[str, Any] | None, CheckResult]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None, CheckResult(
            CheckStatus.ERROR,
            "config:file",
            f"configuration does not exist: {path}",
            "Run ./scripts/setup.sh.",
        )
    except (OSError, json.JSONDecodeError) as error:
        return None, CheckResult(
            CheckStatus.ERROR,
            "config:file",
            f"configuration is not readable JSON: {error}",
            "Fix the file or run ./scripts/setup.sh --force.",
        )
    if not isinstance(value, dict):
        return None, CheckResult(
            CheckStatus.ERROR,
            "config:file",
            "configuration root must be a JSON object",
            "Run ./scripts/setup.sh --force.",
        )
    return value, CheckResult(CheckStatus.OK, "config:file", f"loaded {path}")


def check_hook_gate(
    repo_root: Path,
    config: Mapping[str, Any] | None,
    config_path: Path,
) -> CheckResult | None:
    """Connect the committed hook gate to the config that makes it passable.

    `.claude/settings.json` is in git and fires `efpf-hook pre-tool-use` as soon
    as `uv` is on PATH; `.mcp.json` is gitignored and only `scripts/setup.sh`
    writes it. A checkout that has the first without the second denies every
    outward tool action and has no way to register an intention (#137). The
    missing-config check already says "run setup"; this one says why a session
    opened at the repository root cannot write until then. Returns None when
    the repository ships no PreToolUse hook, so the check is silent elsewhere.
    """

    settings_path = repo_root / ".claude" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    hooks = settings.get("hooks") if isinstance(settings, Mapping) else None
    if not isinstance(hooks, Mapping) or "PreToolUse" not in hooks:
        return None

    servers = config.get("mcpServers") if isinstance(config, Mapping) else None
    if isinstance(servers, Mapping) and "individual-kernel" in servers:
        return CheckResult(
            CheckStatus.OK,
            "hooks:gate",
            "committed PreToolUse hook is backed by the configured individual-kernel server",
        )
    return CheckResult(
        CheckStatus.ERROR,
        "hooks:gate",
        "the committed .claude/settings.json hooks are active as soon as uv is on PATH; "
        f"until {config_path} registers individual-kernel, outward tool actions "
        "(Write, Edit, mutating Bash) are denied in any Claude Code session opened at "
        "this repository root",
        "Run ./scripts/setup.sh (scripts\\setup.cmd on Windows), then restart the session.",
    )


def _check_python() -> CheckResult:
    if sys.version_info[:2] == (3, 13):
        return CheckResult(CheckStatus.OK, "python", sys.version.split()[0])
    return CheckResult(
        CheckStatus.ERROR,
        "python",
        f"Python 3.13 is required; found {sys.version.split()[0]}",
        "Run through uv with --python 3.13.",
    )


def _check_lock(
    repo_root: Path,
    *,
    which: Callable[[str], str | None],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> CheckResult:
    if not which("uv"):
        return CheckResult(
            CheckStatus.ERROR,
            "workspace:lock",
            "uv is not installed or not on PATH",
            "Install uv from https://docs.astral.sh/uv/getting-started/installation/.",
        )
    result = runner(
        ["uv", "lock", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return CheckResult(CheckStatus.OK, "workspace:lock", "uv.lock is current")
    detail = result.stderr.strip() or result.stdout.strip() or "uv lock --check failed"
    return CheckResult(
        CheckStatus.ERROR,
        "workspace:lock",
        detail,
        "Run uv lock, review the change, then run uv sync --locked.",
    )


def _check_social_policy(repo_root: Path) -> CheckResult:
    policy_path = repo_root / "socialPolicy.toml"
    if not policy_path.exists():
        return CheckResult(
            CheckStatus.WARN,
            "social-policy",
            "socialPolicy.toml is absent; runtime defaults will be used",
            "Run ./scripts/setup.sh to create the example policy.",
        )
    try:
        tomllib.loads(policy_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        return CheckResult(
            CheckStatus.ERROR,
            "social-policy",
            f"socialPolicy.toml is invalid: {error}",
            "Fix the TOML or restore examples/configs/socialPolicy.example.toml.",
        )
    return CheckResult(CheckStatus.OK, "social-policy", "socialPolicy.toml is valid")


def run_doctor(
    repo_root: Path,
    config_path: Path,
    home: Path,
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[CheckResult]:
    """Run all read-only checks."""

    results = [
        _check_python(),
        _check_lock(repo_root, which=which, runner=runner),
        _check_social_policy(repo_root),
        check_state_path(home / ".claude" / "memories"),
        check_state_path(home / ".claude" / "sociality" / "social.db"),
    ]
    config, config_result = _load_config(config_path)
    results.append(config_result)
    gate_result = check_hook_gate(repo_root, config, config_path)
    if gate_result is not None:
        results.append(gate_result)
    if config is not None:
        results.extend(validate_mcp_config(config))
        results.extend(check_workspace_packages(repo_root, config))
        results.extend(check_optional_dependencies(config, which=which))
    return results


def run_live_checks(
    repo_root: Path,
    config_path: Path,
    state_root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[CheckResult]:
    """Initialize configured MCP servers in isolated state directories."""

    config, config_result = _load_config(config_path)
    if config is None:
        return [config_result]
    servers = config.get("mcpServers", {})
    if not isinstance(servers, Mapping):
        return [
            CheckResult(
                CheckStatus.ERROR,
                "live:mcpServers",
                "mcpServers must be a JSON object.",
                "Fix the config before running doctor --live.",
            )
        ]

    results: list[CheckResult] = []
    probe = repo_root / "scripts" / "mcp_probe.py"
    for raw_name in servers:
        name = str(raw_name)
        if name not in SERVER_SPECS:
            continue
        command = [
            "uv",
            "run",
            "--directory",
            str(repo_root),
            "--package",
            "individual-kernel-mcp",
            "python",
            str(probe),
            "--config",
            str(config_path),
            "--server",
            name,
            "--cwd",
            str(repo_root),
            "--state-dir",
            str(state_root / name),
        ]
        if name == "memory":
            command.append("--remember-roundtrip")
        completed = runner(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report = {
                "ok": False,
                "error": completed.stderr.strip() or "probe returned invalid JSON",
            }
        if completed.returncode == 0 and report.get("ok") is True:
            tool_count = int(report.get("tool_count", 0))
            detail = f"initialized and exposed {tool_count} tools"
            if report.get("remember_roundtrip"):
                detail += "; isolated remember round-trip passed"
            results.append(CheckResult(CheckStatus.OK, f"live:{name}", detail))
        else:
            detail = str(report.get("error") or "MCP live probe failed")
            results.append(
                CheckResult(
                    CheckStatus.ERROR,
                    f"live:{name}",
                    detail,
                    f"Run scripts/doctor.py --live --config {config_path} for details.",
                )
            )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="MCP config to inspect (default: <repo>/.mcp.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a stable machine-readable report",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Start configured MCP servers against isolated temporary state",
    )
    return parser


def _json_report(results: Sequence[CheckResult]) -> dict[str, Any]:
    summary = {
        status.value: sum(result.status is status for result in results)
        for status in CheckStatus
    }
    return {
        "schema_version": 1,
        "platform": platform.system().lower(),
        "summary": summary,
        "checks": [
            {
                "status": result.status.value,
                "subject": result.subject,
                "detail": result.detail,
                "remediation": result.remediation,
            }
            for result in results
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or repo_root / ".mcp.json"
    results = run_doctor(repo_root, config_path, Path.home())
    if args.live:
        with tempfile.TemporaryDirectory(prefix="embodied-claude-doctor-") as temp:
            results.extend(
                run_live_checks(
                    repo_root,
                    config_path,
                    Path(temp),
                )
            )
    if args.json:
        json.dump(_json_report(results), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for result in results:
            print(f"[{result.status}] {result.subject}: {result.detail}")
            if result.remediation:
                print(f"  -> {result.remediation}")
    return int(any(result.status is CheckStatus.ERROR for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
