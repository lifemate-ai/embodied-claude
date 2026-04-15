"""Fixtures for relationship-mcp tests."""

from pathlib import Path

import pytest

from relationship_mcp.store import RelationshipStore


@pytest.fixture
def store(tmp_path: Path) -> RelationshipStore:
    relationship_store = RelationshipStore(tmp_path / "social.db")
    yield relationship_store
    relationship_store.close()
