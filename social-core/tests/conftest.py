"""Fixtures for social-core tests."""

from pathlib import Path

import pytest

from social_core.db import SocialDB


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "social.db"


@pytest.fixture
def social_db(temp_db_path: Path) -> SocialDB:
    db = SocialDB(temp_db_path)
    db.connect()
    yield db
    db.close()
