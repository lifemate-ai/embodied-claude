"""Explicit filesystem operations for the onboarding setup CLI."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from scripts.onboarding import configs_equivalent


class ConfigConflictError(RuntimeError):
    """Raised when setup would replace an existing config without consent."""


class ConfigAction(StrEnum):
    """Filesystem action selected for a generated MCP configuration."""

    CREATE = "create"
    KEEP = "keep"
    REPLACE = "replace"


@dataclass(frozen=True)
class ConfigPlan:
    """A reviewed config write decision."""

    action: ConfigAction
    destination: Path
    backup: Path | None = None


def _unique_backup_path(destination: Path, now: datetime) -> Path:
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    base = destination.with_name(f"{destination.name}.backup-{timestamp}")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def plan_config_write(
    destination: Path,
    config: Mapping[str, Any],
    *,
    force: bool = False,
    now: datetime | None = None,
) -> ConfigPlan:
    """Inspect an existing config and decide how setup may proceed."""

    if not destination.exists():
        return ConfigPlan(ConfigAction.CREATE, destination)

    try:
        existing = json.loads(destination.read_text())
    except (OSError, json.JSONDecodeError) as error:
        if not force:
            raise ConfigConflictError(
                f"{destination} is not valid JSON; fix it or rerun with --force"
            ) from error
    else:
        if configs_equivalent(existing, config):
            return ConfigPlan(ConfigAction.KEEP, destination)
        if not force:
            raise ConfigConflictError(
                f"{destination} differs from the proposed config; rerun with --force "
                "to back it up and replace it"
            )

    timestamp = now or datetime.now(UTC)
    return ConfigPlan(
        ConfigAction.REPLACE,
        destination,
        _unique_backup_path(destination, timestamp),
    )


def apply_config_plan(plan: ConfigPlan, config: Mapping[str, Any]) -> None:
    """Apply an approved config plan using a same-directory atomic replace."""

    if plan.action is ConfigAction.KEEP:
        return

    destination = plan.destination
    if plan.action is ConfigAction.REPLACE:
        if plan.backup is None:
            raise ValueError("A replacement config plan requires a backup path")
        shutil.copy2(destination, plan.backup)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f"{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        if os.name == "posix":
            temporary_path.chmod(0o600)
        os.replace(temporary_path, destination)
        temporary_path = None
        if os.name == "posix":
            destination.chmod(0o600)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


HEADLESS_ENABLE_KEY = "enabledMcpjsonServers"


def enable_headless_servers(settings_path: Path, server_names: Sequence[str]) -> bool:
    """Approve project MCP servers for `claude -p` by name.

    Claude Code loads `.mcp.json` servers only after they are approved, and a
    headless run has no way to ask. It skips them silently instead, and every
    edit of `.mcp.json` resets the approval (#140). The list in
    `.claude/settings.local.json` is what makes the approval survive, so it is
    rewritten to exactly the servers setup just generated: other keys in the
    file are left alone, and the list is replaced rather than appended to.

    Returns whether the file changed.
    """

    settings: dict[str, Any] = {}
    if settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigConflictError(
                f"{settings_path} is not valid JSON; fix it before rerunning setup"
            ) from error
        if not isinstance(loaded, dict):
            raise ConfigConflictError(
                f"{settings_path} must contain a JSON object at the top level"
            )
        settings = loaded

    wanted = list(dict.fromkeys(str(name) for name in server_names))
    if settings.get(HEADLESS_ENABLE_KEY) == wanted:
        return False
    settings[HEADLESS_ENABLE_KEY] = wanted
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


def copy_policy_if_missing(source: Path, destination: Path) -> bool:
    """Copy the example social policy without replacing user configuration."""

    if destination.exists():
        return False
    shutil.copy2(source, destination)
    return True
