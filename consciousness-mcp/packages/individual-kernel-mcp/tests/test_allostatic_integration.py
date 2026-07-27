"""The tick runtime's body state on either side of the allostatic switch.

With the flag off the legacy arithmetic must be reproduced exactly; with it on
the body state must stop being pinned by a desire file that nobody is updating.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.enacted_field import EnactedField, TriggerKind
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)

# The live desire file has not been rewritten since this instant.
STALE = "2026-05-18T02:04:32+00:00"


@pytest.fixture
def behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _configure(*, allostatic: bool) -> None:
        path = tmp_path / "mcpBehavior.toml"
        path.write_text(
            "[individual-kernel]\n"
            f"allostatic_valence = {'true' if allostatic else 'false'}\n"
            "generative_field_model = false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(path))

    return _configure


def _producer(social_db: SocialDB, tmp_path: Path) -> TickProducer:
    interoception = tmp_path / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 20.0}}), encoding="utf-8")
    desires = tmp_path / "desires.json"
    desires.write_text(
        json.dumps(
            {
                "updated_at": STALE,
                "desires": {"identity_coherence": 0.4},
                "discomforts": {"identity_coherence": 0.5},
                "dominant": "identity_coherence",
            }
        ),
        encoding="utf-8",
    )
    return TickProducer(
        social_db, interoception_path=interoception, desires_path=desires
    )


def _commit(producer: TickProducer) -> EnactedField:
    opened = producer.begin_tick(TriggerKind.USER_PROMPT)
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref="desire:identity_coherence",
            content_summary="need identity_coherence",
            source=CandidateSource.DESIRE,
            source_mode=SourceMode.INFERRED,
            modality="internal",
            precision=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
            continuity_with_previous=1.0,
            controllability=1.0,
        )
    )
    return producer.compete_and_commit(opened.tick_id).field


class TestFlagOff:
    def test_legacy_valence_is_reproduced_exactly(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        behavior(allostatic=False)
        field = _commit(_producer(social_db, tmp_path))
        # 0.65 * 0.0 + 0.35 * (-0.6 * 0.5)
        assert field.interoception.valence == pytest.approx(-0.105)

    def test_legacy_controllability_is_reproduced_exactly(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        behavior(allostatic=False)
        field = _commit(_producer(social_db, tmp_path))
        # 0.45 + 0.25 * (1 - 0.5)
        assert field.interoception.controllability == pytest.approx(0.575)

    def test_legacy_need_vector_is_the_recorded_discomforts(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        behavior(allostatic=False)
        field = _commit(_producer(social_db, tmp_path))
        assert field.interoception.need_vector == {"identity_coherence": 0.5}


class TestFlagOn:
    def test_stale_desire_file_no_longer_pins_the_need(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        """The recorded discomfort was 0.5 and the file has not moved since May."""
        behavior(allostatic=True)
        field = _commit(_producer(social_db, tmp_path))
        need = field.interoception.need_vector["identity_coherence"]
        assert need != pytest.approx(0.5)

    def test_valence_stays_in_range(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        behavior(allostatic=True)
        field = _commit(_producer(social_db, tmp_path))
        assert -1.0 <= field.interoception.valence <= 1.0

    def test_controllability_is_no_longer_derived_from_discomfort(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        behavior(allostatic=True)
        field = _commit(_producer(social_db, tmp_path))
        assert field.interoception.controllability != pytest.approx(0.575)

    def test_controllability_follows_a_measured_ownership_score(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        """Once an action has actually been owned, control reflects that score."""
        behavior(allostatic=True)
        producer = _producer(social_db, tmp_path)
        first = _commit(producer)
        producer.fields.update(
            first.model_copy(
                update={
                    "agency_state": first.agency_state.model_copy(
                        update={
                            "registered_intention": True,
                            "ownership_score": 0.82,
                        }
                    )
                }
            )
        )
        second = _commit(producer)
        assert second.interoception.controllability == pytest.approx(0.82)

    def test_body_state_is_not_the_legacy_value(
        self, social_db: SocialDB, tmp_path: Path, behavior
    ) -> None:
        behavior(allostatic=True)
        field = _commit(_producer(social_db, tmp_path))
        assert field.interoception.valence != pytest.approx(-0.105)
