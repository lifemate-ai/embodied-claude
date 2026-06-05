"""Tests for the Epistemic Effect fields on ResponsePlan.

ResponsePlan must carry observations and inferences as EpistemicClaim lists,
so downstream consumers (and Kokone herself) can keep observed facts and
inferred guesses structurally separate. The plan-of-record is
/home/mizushima/.claude/plans/jazzy-wishing-starfish.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from social_core.models import EpistemicClaim

from interaction_orchestrator_mcp.schemas import (
    BoundaryHint,
    InitiativeHint,
    MemoryUseHint,
    ResponsePlan,
    ToneHint,
)


def _minimal_plan_fields() -> dict:
    """Construct the smallest valid ResponsePlan field set."""
    return dict(
        primary_move="answer_directly",
        why_this_move="user asked a direct question",
        tone=ToneHint(),
        memory_use=MemoryUseHint(),
        initiative=InitiativeHint(),
        boundary=BoundaryHint(),
    )


def test_response_plan_has_empty_epistemic_lists_by_default() -> None:
    plan = ResponsePlan(**_minimal_plan_fields())
    assert plan.observations == []
    assert plan.inferences == []


def test_response_plan_accepts_observations() -> None:
    obs = EpistemicClaim(
        content="本棚が画面右側に見える",
        evidence_type="observed",
        source="wifi_cam.see",
    )
    plan = ResponsePlan(**_minimal_plan_fields(), observations=[obs])
    assert len(plan.observations) == 1
    assert plan.observations[0].evidence_type == "observed"


def test_response_plan_accepts_inferences() -> None:
    inf = EpistemicClaim(
        content="コウタは集中しているように見える",
        evidence_type="inferred",
        derived_from=["obs_001"],
        confidence=0.7,
    )
    plan = ResponsePlan(**_minimal_plan_fields(), inferences=[inf])
    assert len(plan.inferences) == 1
    assert plan.inferences[0].evidence_type == "inferred"


def test_response_plan_rejects_observed_in_inferences_field() -> None:
    """Inferences field must reject claims tagged observed — guard against mix-up."""
    obs = EpistemicClaim(
        content="本棚が見える",
        evidence_type="observed",
        source="wifi_cam.see",
    )
    with pytest.raises(ValidationError):
        ResponsePlan(**_minimal_plan_fields(), inferences=[obs])


def test_response_plan_rejects_inferred_in_observations_field() -> None:
    """Observations field must reject claims tagged anything other than observed."""
    inf = EpistemicClaim(
        content="集中しているように見える",
        evidence_type="inferred",
    )
    with pytest.raises(ValidationError):
        ResponsePlan(**_minimal_plan_fields(), observations=[inf])


def test_response_plan_serializes_epistemic_lists() -> None:
    """model_dump must surface observations / inferences for MCP transport."""
    obs = EpistemicClaim(
        content="眼鏡をかけた人物が中央に座っている",
        evidence_type="observed",
        source="wifi_cam.see",
    )
    inf = EpistemicClaim(
        content="作業に集中している",
        evidence_type="inferred",
        derived_from=["obs_glasses"],
        confidence=0.6,
    )
    plan = ResponsePlan(
        **_minimal_plan_fields(),
        observations=[obs],
        inferences=[inf],
    )
    dumped = plan.model_dump(mode="json")
    assert "observations" in dumped
    assert "inferences" in dumped
    assert dumped["observations"][0]["evidence_type"] == "observed"
    assert dumped["inferences"][0]["evidence_type"] == "inferred"
