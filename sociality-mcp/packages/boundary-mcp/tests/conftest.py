"""Fixtures for boundary-mcp tests."""

from pathlib import Path

import pytest

from boundary_mcp.store import BoundaryStore


@pytest.fixture
def policy_path(tmp_path: Path) -> Path:
    path = tmp_path / "socialPolicy.toml"
    path.write_text(
        """
[global]
timezone = "Asia/Tokyo"
quiet_hours = ["00:00-07:00"]
max_nudges_per_hour = 2

[[posting_rules]]
channel = "x"
require_face_consent = true
require_review_if_person_present = true

[[person_rules]]
person_id = "kouta"
avoid_actions = ["camera_speaker_after_midnight"]
preferred_nudge_style = "brief_gentle"
""".strip(),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def store(tmp_path: Path, policy_path: Path) -> BoundaryStore:
    boundary_store = BoundaryStore(tmp_path / "social.db", policy_path=policy_path)
    yield boundary_store
    boundary_store.close()
