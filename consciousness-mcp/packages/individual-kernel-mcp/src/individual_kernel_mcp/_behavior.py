"""Behavior configuration loader -- reads mcpBehavior.toml at project root.

This package sits one level deeper than the other MCP packages
(consciousness-mcp/packages/individual-kernel-mcp/src/individual_kernel_mcp),
so the repo root is parents[5] instead of parents[3].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef,unused-ignore]

_TOML_PATH = Path(
    os.getenv("MCP_BEHAVIOR_TOML", "")
    or str(Path(__file__).resolve().parents[5] / "mcpBehavior.toml")
)


def load_behavior(section: str) -> dict[str, Any]:
    """Load a section from mcpBehavior.toml.

    Returns empty dict if file doesn't exist or section is missing.
    Reads the file on every call (no caching) so changes are picked up immediately.
    """
    toml_path = Path(os.getenv("MCP_BEHAVIOR_TOML", "") or str(_TOML_PATH))
    if not toml_path.is_file():
        return {}
    try:
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
        return dict(data.get(section, {}))
    except Exception:
        return {}


def get_behavior(section: str, key: str, default: Any = None) -> Any:
    """Get a single value from mcpBehavior.toml."""
    return load_behavior(section).get(key, default)
