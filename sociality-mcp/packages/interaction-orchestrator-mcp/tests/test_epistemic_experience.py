"""Tests for the Epistemic Effect field on RecordAgentExperienceInput.

When the agent records an experience, it should be able to declare HOW it
came to know what it's recording — observed via a tool, inferred from
prior observations, recalled from memory, heard from a person, or assumed.

evidence_type is optional for backward compat; existing call sites continue
to work. The plan-of-record is /home/mizushima/.claude/plans/jazzy-wishing-starfish.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from interaction_orchestrator_mcp.schemas import RecordAgentExperienceInput


def _minimal_fields() -> dict:
    return dict(
        kind="agent_observation",
        summary="本棚が画面右側に見えた",
    )


def test_record_experience_evidence_type_is_optional() -> None:
    """Pre-existing call sites without evidence_type must keep working."""
    payload = RecordAgentExperienceInput(**_minimal_fields())
    assert payload.evidence_type is None


def test_record_experience_accepts_observed() -> None:
    payload = RecordAgentExperienceInput(
        **_minimal_fields(),
        evidence_type="observed",
    )
    assert payload.evidence_type == "observed"


def test_record_experience_accepts_each_evidence_type() -> None:
    for kind in ("observed", "inferred", "remembered", "heard", "assumed"):
        payload = RecordAgentExperienceInput(
            **_minimal_fields(),
            evidence_type=kind,
        )
        assert payload.evidence_type == kind


def test_record_experience_rejects_unknown_evidence_type() -> None:
    with pytest.raises(ValidationError):
        RecordAgentExperienceInput(
            **_minimal_fields(),
            evidence_type="guessed",  # type: ignore[arg-type]
        )


def test_record_experience_serializes_evidence_type() -> None:
    payload = RecordAgentExperienceInput(
        **_minimal_fields(),
        evidence_type="inferred",
    )
    dumped = payload.model_dump(mode="json")
    assert dumped["evidence_type"] == "inferred"
