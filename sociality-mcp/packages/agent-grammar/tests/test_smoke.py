"""Smoke tests for the agent-grammar package skeleton."""

from __future__ import annotations

import pytest

from agent_grammar import EpistemicClaim, EvidenceType, derive, observe
from agent_grammar.grammar import TURN_GRAMMAR_DRAFT


def test_public_reexports_are_available() -> None:
    assert EpistemicClaim is not None
    assert EvidenceType is not None


def test_observe_builds_observed_claim() -> None:
    claim = observe("本棚が見える", source="wifi_cam.see")
    assert claim.evidence_type == "observed"
    assert claim.source == "wifi_cam.see"
    assert claim.confidence == 1.0


def test_derive_builds_inferred_claim_from_observations() -> None:
    obs = observe("眼鏡をかけた人が中央にいる", source="wifi_cam.see")
    inf = derive("作業に集中している", from_claims=[obs])
    assert inf.evidence_type == "inferred"
    assert inf.derived_from == ["wifi_cam.see"]
    assert inf.confidence == pytest.approx(0.6)


def test_derive_falls_back_to_content_when_no_source() -> None:
    upstream = EpistemicClaim(
        content="室内が暗い",
        evidence_type="observed",
    )
    inf = derive("夜である可能性が高い", from_claims=[upstream])
    assert inf.derived_from == ["室内が暗い"]


def test_turn_grammar_draft_mentions_core_rules() -> None:
    """The draft grammar lists the rule names downstream code will refer to."""
    for rule in ("Turn", "Observe", "Interpret", "Respond", "BoundaryCheck"):
        assert rule in TURN_GRAMMAR_DRAFT
