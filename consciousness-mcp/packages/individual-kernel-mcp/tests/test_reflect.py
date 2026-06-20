"""Tests for reflect_attention_schema — on-demand attention reflection (Phase 2.4).

Produces an AttentionReflection summarizing the tracker's recent buffer
(and optionally extra history pulled from SQLite). ON-DEMAND ONLY — not
per-tick. Used by introspect() (Phase 2.5) and by /consolidate skill.
"""

from __future__ import annotations

import pytest

from individual_kernel_mcp.attention_schema import (
    AttentionSchemaTracker,
    Modality,
)
from individual_kernel_mcp.reflect import (
    AttentionReflection,
    ModalityDwell,
    reflect_attention_schema,
)


class TestEmptyReflection:
    def test_empty_tracker_returns_empty(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        r = reflect_attention_schema(tracker)
        assert isinstance(r, AttentionReflection)
        assert r.window_size == 0
        assert r.current is None
        assert r.modality_distribution == {}
        assert r.focus_changes == 0
        assert r.dominant_focal is None


class TestCurrent:
    def test_current_is_most_recent_buffered(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="old",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:00+00:00",
        )
        tracker.record(
            focal_target_ref="new",
            modality="visual",
            intensity=0.7,
            ts="2026-06-21T03:00:10+00:00",
        )
        r = reflect_attention_schema(tracker)
        assert r.current is not None
        assert r.current.focal_target_ref == "new"


class TestWindowSize:
    def test_window_matches_buffer(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        for i in range(5):
            tracker.record(
                focal_target_ref=f"t_{i}",
                modality="visual",
                intensity=0.5,
                ts=f"2026-06-21T03:00:0{i}+00:00",
            )
        r = reflect_attention_schema(tracker)
        assert r.window_size == 5


class TestModalityDistribution:
    def test_sums_to_one(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="a",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:00+00:00",
        )
        tracker.record(
            focal_target_ref="b",
            modality="auditory",
            intensity=0.5,
            ts="2026-06-21T03:00:01+00:00",
        )
        tracker.record(
            focal_target_ref="c",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:02+00:00",
        )
        r = reflect_attention_schema(tracker)
        total = sum(r.modality_distribution.values())
        assert total == pytest.approx(1.0)
        assert r.modality_distribution["visual"] == pytest.approx(2 / 3)
        assert r.modality_distribution["auditory"] == pytest.approx(1 / 3)


class TestDominantFocal:
    def test_dominant_focal_is_most_frequent(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        for i in range(3):
            tracker.record(
                focal_target_ref="popular",
                modality="visual",
                intensity=0.5,
                ts=f"2026-06-21T03:00:0{i}+00:00",
            )
        tracker.record(
            focal_target_ref="rare",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:03+00:00",
        )
        r = reflect_attention_schema(tracker)
        assert r.dominant_focal == "popular"

    def test_no_dominant_when_all_none(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref=None,
            modality="internal",
            intensity=0.5,
            ts="2026-06-21T03:00:00+00:00",
        )
        tracker.record(
            focal_target_ref=None,
            modality="internal",
            intensity=0.5,
            ts="2026-06-21T03:00:01+00:00",
        )
        r = reflect_attention_schema(tracker)
        assert r.dominant_focal is None


class TestFocusChanges:
    def test_zero_changes_for_constant_focal(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        for i in range(5):
            tracker.record(
                focal_target_ref="stable",
                modality="visual",
                intensity=0.5,
                ts=f"2026-06-21T03:00:0{i}+00:00",
            )
        r = reflect_attention_schema(tracker)
        assert r.focus_changes == 0

    def test_change_count_matches_transitions(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        # a → b → b → c = 2 changes
        sequence = ["a", "b", "b", "c"]
        for i, focal in enumerate(sequence):
            tracker.record(
                focal_target_ref=focal,
                modality="visual",
                intensity=0.5,
                ts=f"2026-06-21T03:00:0{i}+00:00",
            )
        r = reflect_attention_schema(tracker)
        assert r.focus_changes == 2


class TestModalityDwells:
    def test_aggregated_per_modality(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="a",
            modality="visual",
            intensity=0.5,
            dwell_seconds=5.0,
            ts="2026-06-21T03:00:00+00:00",
        )
        tracker.record(
            focal_target_ref="a",
            modality="visual",
            intensity=0.5,
            dwell_seconds=15.0,
            ts="2026-06-21T03:00:10+00:00",
        )
        tracker.record(
            focal_target_ref="b",
            modality="auditory",
            intensity=0.5,
            dwell_seconds=3.0,
            ts="2026-06-21T03:00:20+00:00",
        )
        r = reflect_attention_schema(tracker)
        by_modality: dict[Modality, ModalityDwell] = {
            md.modality: md for md in r.modality_dwells
        }
        assert by_modality["visual"].average_dwell_seconds == pytest.approx(10.0)
        assert by_modality["visual"].max_dwell_seconds == pytest.approx(15.0)
        assert by_modality["visual"].count == 2
        assert by_modality["auditory"].count == 1


class TestExtraHistory:
    def test_extra_history_pulls_from_db(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        # Persist some history
        for i in range(3):
            tracker.record(
                focal_target_ref="old",
                modality="visual",
                intensity=0.5,
                ts=f"2026-06-20T0{i}:00:00+00:00",
            )
        tracker.flush()
        # New activity in current buffer
        tracker.record(
            focal_target_ref="new",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:00+00:00",
        )
        r = reflect_attention_schema(tracker, extra_history=10)
        assert r.window_size == 4

    def test_no_extra_history_by_default(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="persisted",
            modality="visual",
            intensity=0.5,
            ts="2026-06-20T03:00:00+00:00",
        )
        tracker.flush()
        # buffer is now empty; default reflect ignores SQLite
        r = reflect_attention_schema(tracker)
        assert r.window_size == 0
