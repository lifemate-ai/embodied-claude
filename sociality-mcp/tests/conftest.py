"""Fixtures for sociality-mcp tests."""

from pathlib import Path

import pytest

from sociality_mcp import server


@pytest.fixture(autouse=True)
def sociality_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy_path = tmp_path / "socialPolicy.toml"
    policy_path.write_text(
        """
[global]
quiet_hours = ["00:00-07:00"]
max_nudges_per_hour = 2

[[posting_rules]]
channel = "x"
require_face_consent = true
require_review_if_person_present = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOCIAL_DB_PATH", str(tmp_path / "social.db"))
    monkeypatch.setenv("SOCIAL_POLICY_PATH", str(policy_path))
    server.reset_store_cache()
    yield
    server.reset_store_cache()
