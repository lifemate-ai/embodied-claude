"""Closing a body action: the reafference, not the return code, sets causal fit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.agency import ActionProposal, AgencyStore
from individual_kernel_mcp.body_contingency import (
    BodyContingencyStore,
    BodyObservation,
    BodyPose,
)
from individual_kernel_mcp.enacted_field import EnactedField, TriggerKind
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)


def _producer(social_db: SocialDB, tmp_path: Path) -> TickProducer:
    interoception = tmp_path / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 20.0}}), encoding="utf-8")
    desires = tmp_path / "desires.json"
    desires.write_text(json.dumps({"desires": {}}), encoding="utf-8")
    return TickProducer(
        social_db, interoception_path=interoception, desires_path=desires
    )


def _commit(producer: TickProducer) -> EnactedField:
    opened = producer.begin_tick(TriggerKind.USER_PROMPT)
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref="desire:look_outside",
            content_summary="need look_outside",
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


def _propose(
    agency: AgencyStore, field: EnactedField, *, tool_name: str = "look_left"
) -> str:
    record = agency.propose(
        ActionProposal(
            field_id=field.field_id,
            tool_name=tool_name,
            tool_input={"degrees": 30},
            goal="look toward the window",
            confidence=0.8,
            # Stated so the timing term is actually scored rather than falling
            # back to the neutral 0.5 an unknown expectation earns.
            expected_latency_ms=800,
        )
    )
    return record.action_id


def _observation(pan_after: float) -> BodyObservation:
    return BodyObservation(
        before=BodyPose(pan=0.0, tilt=0.0),
        after=BodyPose(pan=pan_after, tilt=0.0),
        observed_latency_ms=800,
    )


class TestCausalFitFollowsTheBody:
    def test_matched_movement_beats_movement_nobody_commanded(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        agency = AgencyStore(social_db)

        matched_id = _propose(agency, _commit(producer))
        _, matched = agency.close(
            action_id=matched_id,
            actual_result_summary="panned left",
            success=True,
            latency_ms=800,
            body_observation=_observation(-30.0),
        )

        # Same successful tool call, but the body went the other way.
        inverted_id = _propose(agency, _commit(producer))
        _, inverted = agency.close(
            action_id=inverted_id,
            actual_result_summary="panned left",
            success=True,
            latency_ms=800,
            body_observation=_observation(30.0),
        )

        assert matched.exclusive_causal_fit > 0.9
        assert inverted.exclusive_causal_fit <= 0.1
        assert inverted.ownership_score < matched.ownership_score

    def test_success_alone_no_longer_grants_full_causal_fit(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        agency = AgencyStore(social_db)
        action_id = _propose(agency, _commit(producer))

        # The call returned success but the body never moved.
        _, assessment = agency.close(
            action_id=action_id,
            actual_result_summary="panned left",
            success=True,
            latency_ms=800,
            body_observation=_observation(0.0),
        )
        assert assessment.exclusive_causal_fit <= 0.15


class TestNoBodyChannel:
    def test_non_body_tool_keeps_the_previous_scoring(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        agency = AgencyStore(social_db)
        action_id = _propose(agency, _commit(producer), tool_name="Bash")
        _, assessment = agency.close(
            action_id=action_id,
            actual_result_summary="ok",
            success=True,
            latency_ms=800,
        )
        assert assessment.exclusive_causal_fit == pytest.approx(1.0)

    def test_failed_non_body_tool_keeps_the_previous_penalty(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        agency = AgencyStore(social_db)
        action_id = _propose(agency, _commit(producer), tool_name="Bash")
        _, assessment = agency.close(
            action_id=action_id,
            actual_result_summary="failed",
            success=False,
            latency_ms=800,
        )
        assert assessment.exclusive_causal_fit == pytest.approx(0.65)


class TestLedger:
    def test_every_close_leaves_one_verdict(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        agency = AgencyStore(social_db)
        action_id = _propose(agency, _commit(producer))
        agency.close(
            action_id=action_id,
            actual_result_summary="panned left",
            success=True,
            latency_ms=800,
            body_observation=_observation(-30.0),
        )
        stored = BodyContingencyStore(social_db).for_action(action_id)
        assert stored is not None
        assert stored["verdict"] == "self_caused"
        assert json.loads(stored["commanded_delta_json"])["pan"] == -30.0

    def test_closing_twice_does_not_duplicate_the_verdict(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        producer = _producer(social_db, tmp_path)
        agency = AgencyStore(social_db)
        action_id = _propose(agency, _commit(producer))
        for _ in range(2):
            agency.close(
                action_id=action_id,
                actual_result_summary="panned left",
                success=True,
                latency_ms=800,
                body_observation=_observation(-30.0),
            )
        rows = BodyContingencyStore(social_db).recent(limit=50)
        assert len([row for row in rows if row["action_id"] == action_id]) == 1
