#!/usr/bin/env python3
"""Configure a useful embodied-claude workspace without enabling unused MCPs."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from io import TextIOBase
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.doctor import CheckResult, CheckStatus, run_doctor
from scripts.onboarding import (
    FeatureSelection,
    all_tools_selection,
    build_mcp_config,
    fill_handson_environment,
    missing_environment,
    redact_config,
)
from scripts.setup_io import (
    ConfigAction,
    ConfigConflictError,
    apply_config_plan,
    copy_policy_if_missing,
    enable_headless_servers,
    plan_config_write,
)

_MODEL_WARMUP = """
from memory_mcp.config import MemoryConfig
from memory_mcp.embedding import E5EmbeddingFunction

model = MemoryConfig.from_env().embedding_model
print(f"  warming {model}")
E5EmbeddingFunction(model)._load_model()
print("  done")
""".strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core",), default="core")
    parser.add_argument("--with-camera", choices=("usb", "tapo"))
    parser.add_argument(
        "--with-transcription",
        choices=("whisper", "faster"),
        help="Transcribe Tapo audio; requires --with-camera tapo",
    )
    parser.add_argument("--with-voice", choices=("voicevox", "elevenlabs"))
    parser.add_argument("--with-x", action="store_true")
    parser.add_argument("--with-system-temperature", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Enable every server, filling absent credentials with obviously "
            "fake values. For demos and hands-on sessions, not for real use"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        choices=("small", "base"),
        default="small",
    )
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _prompt_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    rendered = "/".join(choices)
    while True:
        value = input(f"{prompt} [{rendered}] ({default}): ").strip().lower()
        selected = value or default
        if selected in choices:
            return selected
        print(f"Choose one of: {rendered}", file=sys.stderr)


def _prompt_bool(prompt: str, *, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.", file=sys.stderr)


def _interactive_selection() -> FeatureSelection:
    print("Choose only the capabilities available on this machine.")
    camera = _prompt_choice("Camera", ("none", "usb", "tapo"), "none")
    transcription = (
        _prompt_choice(
            "Tapo audio transcription",
            ("none", "whisper", "faster"),
            "none",
        )
        if camera == "tapo"
        else "none"
    )
    voice = _prompt_choice(
        "Voice",
        ("none", "voicevox", "elevenlabs"),
        "none",
    )
    return FeatureSelection(
        camera=None if camera == "none" else camera,
        transcription=(
            None if transcription == "none" else transcription
        ),
        voice=None if voice == "none" else voice,
        x_enabled=_prompt_bool("Enable X search and posting"),
        system_temperature=_prompt_bool("Enable host temperature and time"),
        embedding_model=_prompt_choice(
            "Memory embedding model",
            ("small", "base"),
            "small",
        ),
    )


def _required_input(
    environment: dict[str, str],
    key: str,
    prompt: str,
    *,
    secret: bool,
) -> None:
    if environment.get(key, "").strip():
        return
    reader = getpass.getpass if secret else input
    environment[key] = reader(f"{prompt}: ").strip()


def _interactive_environment(
    selection: FeatureSelection,
    source: Mapping[str, str],
) -> dict[str, str]:
    environment = dict(source)
    if selection.camera == "tapo":
        _required_input(
            environment,
            "TAPO_CAMERA_HOST",
            "Tapo camera host or IP",
            secret=False,
        )
        _required_input(
            environment,
            "TAPO_USERNAME",
            "Tapo local camera username",
            secret=False,
        )
        _required_input(
            environment,
            "TAPO_PASSWORD",
            "Tapo local camera password",
            secret=True,
        )
    if selection.voice == "voicevox":
        default = environment.get("VOICEVOX_URL", "http://localhost:50021")
        entered = input(f"VOICEVOX URL ({default}): ").strip()
        environment["VOICEVOX_URL"] = entered or default
    elif selection.voice == "elevenlabs":
        _required_input(
            environment,
            "ELEVENLABS_API_KEY",
            "ElevenLabs API key",
            secret=True,
        )
        if not environment.get("ELEVENLABS_VOICE_ID", "").strip():
            voice_id = input("ElevenLabs voice ID (optional): ").strip()
            if voice_id:
                environment["ELEVENLABS_VOICE_ID"] = voice_id
    if selection.x_enabled:
        for key, label in (
            ("XAI_API_KEY", "xAI API key"),
            ("X_CONSUMER_KEY", "X consumer key"),
            ("X_CONSUMER_SECRET", "X consumer secret"),
            ("X_ACCESS_TOKEN", "X access token"),
            ("X_ACCESS_TOKEN_SECRET", "X access token secret"),
        ):
            _required_input(environment, key, label, secret=True)
    return environment


def _validate_workspace(repo_root: Path) -> None:
    missing = [
        name
        for name in ("pyproject.toml", "uv.lock")
        if not (repo_root / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"Not an embodied-claude workspace; missing: {', '.join(missing)}"
        )


def _print_doctor_results(
    results: Sequence[CheckResult],
    output: TextIOBase,
) -> bool:
    for result in results:
        print(f"[{result.status}] {result.subject}: {result.detail}", file=output)
        if result.remediation:
            print(f"  -> {result.remediation}", file=output)
    return any(result.status is CheckStatus.ERROR for result in results)


def build_sync_command(selection: FeatureSelection) -> list[str]:
    """Build one locked uv sync for exactly the selected runtime profile."""
    command = ["uv", "sync", "--locked", "--no-dev"]
    for extra in selection.uv_extras():
        command.extend(("--extra", extra))
    return command


def execute_setup(
    selection: FeatureSelection,
    environment: Mapping[str, str],
    *,
    repo_root: Path,
    home: Path,
    dry_run: bool,
    force: bool,
    skip_model_download: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    doctor: Callable[..., list[CheckResult]] = run_doctor,
    output: TextIOBase = sys.stdout,
) -> int:
    """Execute setup with injectable side-effect boundaries for tests."""

    _validate_workspace(repo_root)
    config = build_mcp_config(selection, environment)
    if dry_run:
        json.dump(redact_config(config), output, indent=2)
        output.write("\n")
        return 0

    print("==> syncing the locked workspace", file=output)
    runner(build_sync_command(selection), cwd=repo_root, check=True)

    destination = repo_root / ".mcp.json"
    plan = plan_config_write(destination, config, force=force)
    apply_config_plan(plan, config)
    if plan.action is ConfigAction.KEEP:
        print("==> keeping equivalent .mcp.json", file=output)
    elif plan.action is ConfigAction.CREATE:
        print("==> created .mcp.json", file=output)
    else:
        print(f"==> backed up old config to {plan.backup.name}", file=output)
        print("==> replaced .mcp.json", file=output)

    # Without this, `claude -p` (the autonomous heartbeat) starts with none of
    # the servers just written and says nothing about it (#140).
    settings_path = repo_root / ".claude" / "settings.local.json"
    server_names = list(config["mcpServers"])
    if enable_headless_servers(settings_path, server_names):
        print(
            "==> approved "
            + ", ".join(server_names)
            + " for headless runs in .claude/settings.local.json",
            file=output,
        )
    else:
        print("==> keeping headless approval in .claude/settings.local.json", file=output)

    policy_source = repo_root / "examples" / "configs" / "socialPolicy.example.toml"
    if not policy_source.is_file():
        raise ValueError(f"Missing social policy example: {policy_source}")
    if copy_policy_if_missing(policy_source, repo_root / "socialPolicy.toml"):
        print("==> created socialPolicy.toml", file=output)
    else:
        print("==> keeping existing socialPolicy.toml", file=output)

    if not skip_model_download:
        print("==> warming the memory embedding model", file=output)
        warm_environment = os.environ.copy()
        warm_environment["MEMORY_EMBEDDING_MODEL"] = config["mcpServers"]["memory"][
            "env"
        ]["MEMORY_EMBEDDING_MODEL"]
        runner(
            [
                "uv",
                "run",
                "--package",
                "memory-mcp",
                "python",
                "-c",
                _MODEL_WARMUP,
            ],
            cwd=repo_root,
            env=warm_environment,
            check=True,
        )

    print("==> checking the generated setup", file=output)
    results = doctor(
        repo_root,
        destination,
        home,
        runner=runner,
    )
    if _print_doctor_results(results, output):
        return 1

    print(file=output)
    print("Setup is ready:", file=output)
    print("1. Start Claude Code in this repository.", file=output)
    print("2. Run /mcp and confirm the four Core servers are connected.", file=output)
    print("3. Ask Claude to remember a short setup fact.", file=output)
    print("4. Ask Claude to recall that fact.", file=output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(raw_arguments)
    interactive = not raw_arguments

    if interactive:
        selection = _interactive_selection()
        environment = _interactive_environment(selection, os.environ)
    elif args.all:
        # --all overrides the individual --with-* flags rather than merging
        # with them: "everything" has no coherent reading that also honours a
        # narrower choice made on the same command line.
        selection = all_tools_selection(args.embedding_model)
        environment = fill_handson_environment(os.environ)
    else:
        selection = FeatureSelection(
            profile=args.profile,
            camera=args.with_camera,
            transcription=args.with_transcription,
            voice=args.with_voice,
            x_enabled=args.with_x,
            system_temperature=args.with_system_temperature,
            embedding_model=args.embedding_model,
        )
        environment = dict(os.environ)
    missing = missing_environment(selection, environment)
    if missing:
        parser.error(
            "selected capabilities require environment variables: "
            + ", ".join(missing)
        )

    try:
        return execute_setup(
            selection,
            environment,
            repo_root=Path(__file__).resolve().parents[1],
            home=Path.home(),
            dry_run=args.dry_run,
            force=args.force,
            skip_model_download=args.skip_model_download,
        )
    except (ConfigConflictError, ValueError) as error:
        print(f"setup error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(
            f"setup command failed with exit code {error.returncode}: {error.cmd}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
