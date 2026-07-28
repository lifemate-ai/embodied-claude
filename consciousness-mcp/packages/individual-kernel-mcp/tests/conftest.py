"""Shared fixtures for individual-kernel-mcp tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from social_core.db import SocialDB

# Baseline the whole suite runs against. Without this, every test reads the
# repository's mcpBehavior.toml, so changing a shipped default silently changes
# what unrelated tests assert. Tests that care about a flag set it themselves.
BASELINE_BEHAVIOR = """[individual-kernel]
generative_field_model = true
generative_rollout_horizon = 2
allostatic_valence = false
valence_coupling = false
"""


@pytest.fixture(autouse=True)
def baseline_behavior(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path_factory.mktemp("behavior") / "mcpBehavior.toml"
    path.write_text(BASELINE_BEHAVIOR, encoding="utf-8")
    monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(path))


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "social.db"


@pytest.fixture
def social_db(temp_db_path: Path) -> Iterator[SocialDB]:
    db = SocialDB(temp_db_path)
    db.connect()
    try:
        yield db
    finally:
        db.close()
