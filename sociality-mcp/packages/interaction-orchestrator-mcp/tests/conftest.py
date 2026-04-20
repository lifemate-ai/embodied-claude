"""Fixtures for interaction-orchestrator-mcp tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from boundary_mcp.store import BoundaryStore
from joint_attention_mcp.store import JointAttentionStore
from relationship_mcp.store import RelationshipStore
from self_narrative_mcp.store import SelfNarrativeStore
from social_core import SocialDB
from social_state_mcp.store import SocialStateStore

from interaction_orchestrator_mcp.store import InteractionOrchestratorStore


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
def db(tmp_path: Path) -> SocialDB:
    social_db = SocialDB(tmp_path / "social.db")
    yield social_db
    social_db.close()


@pytest.fixture
def stores(db: SocialDB, policy_path: Path):
    """Provide a bundle of substrate stores sharing one SocialDB."""

    social_state = SocialStateStore(
        db=db, quiet_hours_windows=["00:00-07:00"], policy_timezone="Asia/Tokyo"
    )
    relationship = RelationshipStore(db=db)
    joint_attention = JointAttentionStore(db=db)
    boundary = BoundaryStore(db=db, policy_path=policy_path)
    self_narrative = SelfNarrativeStore(db=db)
    orchestrator = InteractionOrchestratorStore(db=db)
    return {
        "social_state": social_state,
        "relationship": relationship,
        "joint_attention": joint_attention,
        "boundary": boundary,
        "self_narrative": self_narrative,
        "orchestrator": orchestrator,
    }
