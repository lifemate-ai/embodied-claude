"""Affect changes which candidate wins, and changes nothing about permission.

The first class is the point of the whole change: with affect neutral, valence
is inert by construction, and with it moved the same two bids resolve
differently.
"""

from __future__ import annotations

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.frame import ConsciousFrameInput, TickFrameStore
from individual_kernel_mcp.valence_coupling import AffectState
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
    WorkspaceEngine,
)


def _needy(tick_id: str) -> WorkspaceCandidate:
    """A pressing need with nothing new to learn from it."""
    return WorkspaceCandidate(
        tick_id=tick_id,
        kind=CandidateKind.GOAL,
        content_ref="desire:identity_coherence",
        content_summary="need identity_coherence",
        source=CandidateSource.DESIRE,
        source_mode=SourceMode.INFERRED,
        modality="internal",
        precision=0.7,
        need_relevance=1.0,
        expected_information_gain=0.0,
    )


def _novel(tick_id: str) -> WorkspaceCandidate:
    """Something worth looking into that no need is pushing."""
    return WorkspaceCandidate(
        tick_id=tick_id,
        kind=CandidateKind.PERCEPT,
        content_ref="event:something-new",
        content_summary="an unexplored thing",
        source=CandidateSource.EXPLICIT,
        source_mode=SourceMode.LIVE,
        modality="external",
        precision=0.7,
        need_relevance=0.0,
        expected_information_gain=1.0,
    )


def _open_tick(db: SocialDB) -> str:
    """Candidates carry a foreign key to a real tick, so make one."""
    return TickFrameStore(db).record(ConsciousFrameInput()).tick_id


def _winner(db: SocialDB, affect: AffectState) -> str:
    tick_id = _open_tick(db)
    engine = WorkspaceEngine(db, affect=affect)
    engine.add_candidate(_needy(tick_id))
    engine.add_candidate(_novel(tick_id))
    return engine.compete(tick_id).winner.content_ref


class TestAffectDecidesBetweenBids:
    def test_bad_affect_favours_the_need(self, social_db: SocialDB) -> None:
        won = _winner(social_db, AffectState(valence=-0.9, arousal=0.2))
        assert won == "desire:identity_coherence"

    def test_good_affect_favours_the_novelty(self, social_db: SocialDB) -> None:
        won = _winner(social_db, AffectState(valence=0.9, arousal=0.2))
        assert won == "event:something-new"

    def test_neutral_affect_matches_an_engine_with_no_affect_at_all(
        self, social_db: SocialDB
    ) -> None:
        neutral = _winner(social_db, AffectState.neutral())
        bare_tick = _open_tick(social_db)
        bare = WorkspaceEngine(social_db)
        bare.add_candidate(_needy(bare_tick))
        bare.add_candidate(_novel(bare_tick))
        assert neutral == bare.compete(bare_tick).winner.content_ref


class TestScoresStayAuditable:
    def test_stored_score_matches_the_recorded_order(self, social_db: SocialDB) -> None:
        """Scores are frozen at insert, so the trace must reproduce the ranking."""
        tick_id = _open_tick(social_db)
        engine = WorkspaceEngine(
            social_db, affect=AffectState(valence=-0.9, arousal=0.2)
        )
        engine.add_candidate(_needy(tick_id))
        engine.add_candidate(_novel(tick_id))
        result = engine.compete(tick_id)
        ranked = sorted(result.scores.items(), key=lambda kv: (-kv[1], kv[0]))
        assert result.winner.candidate_id == ranked[0][0]

    def test_applied_weights_are_reported(self, social_db: SocialDB) -> None:
        engine = WorkspaceEngine(
            social_db, affect=AffectState(valence=-0.6, arousal=0.4)
        )
        trace = engine.affect_trace()
        assert trace["valence"] == pytest.approx(-0.6)
        assert trace["need"] > trace["base_need"]
        assert trace["information"] < trace["base_information"]
