"""Who the agent is and who it lives with, read from the environment.

The sociality packages used to carry one person's identifier as a literal
default (``person_id="kouta"``) and one agent's name in an output string. Those
are facts about one deployment, not about the code, so they are read from the
environment here with neutral defaults. The values are read at call time rather
than import time so a test (or a long-lived process whose environment changes)
sees the current setting without a module reload.

``COMPANION_ID`` is the machine identifier used as ``person_id`` in the social
substrate. ``COMPANION_NAME`` is the display name (the same variable
``desire-system`` and ``memory-mcp`` use). ``SELF_NAME`` is the agent's own
name, as introduced for ``desire-system`` in #134.
"""

from __future__ import annotations

import os

DEFAULT_COMPANION_ID = "companion"
DEFAULT_COMPANION_NAME = "あなた"
DEFAULT_SELF_NAME = "This agent"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "")
    return value.strip() or default


def companion_id() -> str:
    """Machine identifier of the primary companion (``COMPANION_ID``)."""
    return _env("COMPANION_ID", DEFAULT_COMPANION_ID)


def companion_name() -> str:
    """Display name of the primary companion (``COMPANION_NAME``)."""
    return _env("COMPANION_NAME", DEFAULT_COMPANION_NAME)


def self_name() -> str:
    """The agent's own name (``SELF_NAME``)."""
    return _env("SELF_NAME", DEFAULT_SELF_NAME)
