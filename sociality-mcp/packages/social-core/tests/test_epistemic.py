"""Tests for EvidenceType and EpistemicClaim primitives.

These are the 5 epistemic categories from the Agent Grammar design:
observed / inferred / remembered / heard / assumed.

The plan-of-record is /home/mizushima/.claude/plans/jazzy-wishing-starfish.md.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from social_core.models import EVIDENCE_TYPES, EpistemicClaim


def test_evidence_types_are_defined() -> None:
    assert set(EVIDENCE_TYPES) == {
        "observed",
        "inferred",
        "remembered",
        "heard",
        "assumed",
    }


def test_epistemic_claim_observed_minimal() -> None:
    claim = EpistemicClaim(
        content="本棚が画面右側に見える",
        evidence_type="observed",
        source="wifi_cam.see",
    )
    assert claim.content == "本棚が画面右側に見える"
    assert claim.evidence_type == "observed"
    assert claim.source == "wifi_cam.see"
    assert claim.confidence == 1.0
    assert claim.derived_from == []


def test_epistemic_claim_inferred_from_observations() -> None:
    claim = EpistemicClaim(
        content="コウタは集中しているように見える",
        evidence_type="inferred",
        derived_from=["obs_001", "obs_002"],
        confidence=0.7,
    )
    assert claim.evidence_type == "inferred"
    assert claim.derived_from == ["obs_001", "obs_002"]
    assert claim.confidence == 0.7


def test_epistemic_claim_remembered_with_memory_id() -> None:
    claim = EpistemicClaim(
        content="前にコウタは本棚の前に座っていた",
        evidence_type="remembered",
        source="memory_abc123",
    )
    assert claim.evidence_type == "remembered"
    assert claim.source == "memory_abc123"


def test_epistemic_claim_heard_from_person() -> None:
    claim = EpistemicClaim(
        content="上原さんが体調を崩したらしい",
        evidence_type="heard",
        source="person_uehara",
        confidence=0.8,
    )
    assert claim.evidence_type == "heard"
    assert claim.source == "person_uehara"


def test_epistemic_claim_assumed_low_confidence() -> None:
    claim = EpistemicClaim(
        content="深夜帯はコウタは作業中であることが多い",
        evidence_type="assumed",
        confidence=0.3,
    )
    assert claim.evidence_type == "assumed"
    assert claim.confidence == 0.3


def test_epistemic_claim_rejects_unknown_evidence_type() -> None:
    with pytest.raises(ValidationError):
        EpistemicClaim(
            content="どっかから来た話",
            evidence_type="guessed",  # type: ignore[arg-type]
        )


def test_epistemic_claim_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        EpistemicClaim(content="", evidence_type="observed")


def test_epistemic_claim_confidence_upper_bound() -> None:
    with pytest.raises(ValidationError):
        EpistemicClaim(
            content="test",
            evidence_type="observed",
            confidence=1.5,
        )


def test_epistemic_claim_confidence_lower_bound() -> None:
    with pytest.raises(ValidationError):
        EpistemicClaim(
            content="test",
            evidence_type="observed",
            confidence=-0.1,
        )


def test_epistemic_claim_rejects_extra_fields() -> None:
    """extra='forbid' must be honored for forward-compat safety."""
    with pytest.raises(ValidationError):
        EpistemicClaim(
            content="test",
            evidence_type="observed",
            unknown_field="oops",  # type: ignore[call-arg]
        )
