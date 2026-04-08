"""Tests for metacognition module."""

import json
from pathlib import Path

import pytest

from memory_mcp.metacognition import (
    HypothesisStatus,
    MetacognitionTracker,
)


@pytest.fixture
def tracker(tmp_path: Path) -> MetacognitionTracker:
    return MetacognitionTracker(tmp_path / "metacognition.json")


class TestHypothesize:
    def test_creates_hypothesis(self, tracker: MetacognitionTracker) -> None:
        h = tracker.hypothesize(
            hypothesis="The bug is in camera.py",
            context="Camera tilt inverted",
            approach="Swap tilt signs",
        )
        assert h.id == "hyp_1"
        assert h.status == HypothesisStatus.ACTIVE
        assert h.hypothesis == "The bug is in camera.py"
        assert h.rejection_count == 0

    def test_increments_id(self, tracker: MetacognitionTracker) -> None:
        h1 = tracker.hypothesize("h1", "ctx", "a1")
        h2 = tracker.hypothesize("h2", "ctx", "a2")
        assert h1.id == "hyp_1"
        assert h2.id == "hyp_2"

    def test_supersedes_previous_active_in_same_context(
        self, tracker: MetacognitionTracker
    ) -> None:
        h1 = tracker.hypothesize("first", "ctx", "a1")
        h2 = tracker.hypothesize("second", "ctx", "a2")

        status = tracker.get_status()
        active = status["active_hypotheses"]
        assert len(active) == 1
        assert active[0]["id"] == h2.id

    def test_different_contexts_stay_independent(
        self, tracker: MetacognitionTracker
    ) -> None:
        tracker.hypothesize("h1", "context_a", "a1")
        tracker.hypothesize("h2", "context_b", "a2")

        status = tracker.get_status()
        active = status["active_hypotheses"]
        assert len(active) == 2

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        state_path = tmp_path / "metacognition.json"
        t1 = MetacognitionTracker(state_path)
        t1.hypothesize("persisted", "ctx", "approach")

        t2 = MetacognitionTracker(state_path)
        status = t2.get_status()
        assert len(status["active_hypotheses"]) == 1
        assert status["active_hypotheses"][0]["hypothesis"] == "persisted"


class TestVerify:
    def test_confirm_hypothesis(self, tracker: MetacognitionTracker) -> None:
        h = tracker.hypothesize("correct guess", "ctx", "approach")
        result = tracker.verify(h.id, "It worked!", True)
        assert result.status == HypothesisStatus.CONFIRMED
        assert result.outcome == "It worked!"

    def test_reject_hypothesis(self, tracker: MetacognitionTracker) -> None:
        h = tracker.hypothesize("wrong guess", "ctx", "approach")
        result = tracker.verify(h.id, "Did not work", False)
        assert result.status == HypothesisStatus.REJECTED

    def test_unknown_id_returns_none(self, tracker: MetacognitionTracker) -> None:
        result = tracker.verify("nonexistent", "outcome", True)
        assert result is None

    def test_rejection_count_increments(self, tracker: MetacognitionTracker) -> None:
        h1 = tracker.hypothesize("guess1", "ctx", "a1")
        tracker.verify(h1.id, "failed", False)

        h2 = tracker.hypothesize("guess2", "ctx", "a2")
        assert h2.rejection_count == 1

        tracker.verify(h2.id, "failed again", False)

        h3 = tracker.hypothesize("guess3", "ctx", "a3")
        assert h3.rejection_count == 2


class TestGetStatus:
    def test_empty_tracker(self, tracker: MetacognitionTracker) -> None:
        status = tracker.get_status()
        assert status["active_hypotheses"] == []
        assert status["recent_rejections"] == []
        assert status["warnings"] == []
        assert status["stats"]["total"] == 0

    def test_warning_after_threshold_rejections(
        self, tracker: MetacognitionTracker
    ) -> None:
        h1 = tracker.hypothesize("g1", "stuck_context", "a1")
        tracker.verify(h1.id, "fail", False)
        h2 = tracker.hypothesize("g2", "stuck_context", "a2")
        tracker.verify(h2.id, "fail", False)
        tracker.hypothesize("g3", "stuck_context", "a3")  # active

        status = tracker.get_status()
        assert len(status["warnings"]) == 1
        assert "stuck_context" in status["warnings"][0]
        assert "2回棄却" in status["warnings"][0]

    def test_no_warning_without_active_hypothesis(
        self, tracker: MetacognitionTracker
    ) -> None:
        h1 = tracker.hypothesize("g1", "ctx", "a1")
        tracker.verify(h1.id, "fail", False)
        h2 = tracker.hypothesize("g2", "ctx", "a2")
        tracker.verify(h2.id, "fail", False)
        # No active hypothesis remaining

        status = tracker.get_status()
        assert status["warnings"] == []

    def test_confirmation_rate(self, tracker: MetacognitionTracker) -> None:
        h1 = tracker.hypothesize("g1", "c1", "a1")
        tracker.verify(h1.id, "ok", True)
        h2 = tracker.hypothesize("g2", "c2", "a2")
        tracker.verify(h2.id, "fail", False)
        h3 = tracker.hypothesize("g3", "c3", "a3")
        tracker.verify(h3.id, "ok", True)

        status = tracker.get_status()
        assert status["stats"]["confirmation_rate"] == "67%"


class TestContextHistory:
    def test_returns_only_matching_context(
        self, tracker: MetacognitionTracker
    ) -> None:
        tracker.hypothesize("h1", "context_a", "a1")
        tracker.hypothesize("h2", "context_b", "a2")
        tracker.hypothesize("h3", "context_a", "a3")

        history = tracker.get_context_history("context_a")
        assert len(history) == 2
        assert all(h.context == "context_a" for h in history)

    def test_empty_context(self, tracker: MetacognitionTracker) -> None:
        history = tracker.get_context_history("nonexistent")
        assert history == []
