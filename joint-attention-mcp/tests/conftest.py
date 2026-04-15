"""Fixtures for joint-attention-mcp tests."""

from pathlib import Path

import pytest

from joint_attention_mcp.store import JointAttentionStore


@pytest.fixture
def store(tmp_path: Path) -> JointAttentionStore:
    joint_store = JointAttentionStore(tmp_path / "social.db")
    yield joint_store
    joint_store.close()
