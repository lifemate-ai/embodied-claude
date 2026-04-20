"""Read-only access to the desire-system snapshot file.

The orchestrator does not own desire computation; it only surfaces the latest
snapshot that the desire-system daemon has already written. Keeping this as a
file-read keeps the orchestrator free of a runtime dependency on desire-system.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def default_desires_path() -> Path:
    return Path(
        os.getenv(
            "DESIRES_PATH",
            str(Path.home() / ".claude" / "desires.json"),
        )
    ).expanduser()


def load_desire_snapshot(path: Path | None = None) -> dict[str, Any] | None:
    """Return the current desire snapshot as a plain dict, or ``None``."""

    target = path or default_desires_path()
    if not target.exists():
        return None
    try:
        with target.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data
