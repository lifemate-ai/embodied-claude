"""Behavior configuration loader -- reads mcpBehavior.toml at project root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_TOML_PATH = Path(
    os.getenv("MCP_BEHAVIOR_TOML", "")
    or str(Path(__file__).resolve().parents[3] / "mcpBehavior.toml")
)


def load_behavior(section: str) -> dict[str, Any]:
    """Load a section from mcpBehavior.toml.

    Returns empty dict if file doesn't exist or section is missing.
    Reads the file on every call (no caching) so changes are picked up immediately.
    """
    if not _TOML_PATH.is_file():
        return {}
    try:
        with _TOML_PATH.open("rb") as f:
            data = tomllib.load(f)
        return dict(data.get(section, {}))
    except Exception:
        return {}


def get_behavior(section: str, key: str, default: Any = None) -> Any:
    """Get a single value from mcpBehavior.toml."""
    return load_behavior(section).get(key, default)
