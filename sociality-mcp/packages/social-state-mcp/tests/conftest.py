"""Fixtures for social-state-mcp tests."""

from pathlib import Path

import pytest

from social_state_mcp.store import SocialStateStore


@pytest.fixture
def store(tmp_path: Path) -> SocialStateStore:
    social_store = SocialStateStore(
        tmp_path / "social.db",
        quiet_hours_windows=["00:00-07:00"],
        policy_timezone="Asia/Tokyo",
    )
    yield social_store
    social_store.close()
