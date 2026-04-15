"""Fixtures for self-narrative-mcp tests."""

from pathlib import Path

import pytest

from self_narrative_mcp.store import SelfNarrativeStore


@pytest.fixture
def store(tmp_path: Path) -> SelfNarrativeStore:
    narrative_store = SelfNarrativeStore(tmp_path / "social.db")
    yield narrative_store
    narrative_store.close()
