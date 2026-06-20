"""Tests for AttentionSchema + AttentionSchemaTracker (Phase 2.4).

The attention schema is a low-dimensional model of the pattern of winning
(AST / Theory B R4). It is NOT the same as workspace.py — workspace picks
winners; the schema models the shape of attention itself for control and
introspection (Phase 2.5 will surface it via introspect()).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from individual_kernel_mcp.attention_schema import (
    AttentionSchema,
    AttentionSchemaTracker,
    Modality,
)
from individual_kernel_mcp.frame import (
    ConsciousFrameInput,
    TickFrameStore,
)


class TestAttentionSchemaModel:
    def test_minimal_construction(self) -> None:
        s = AttentionSchema(
            schema_id="sch_001",
            ts="2026-06-21T03:00:00+00:00",
            owner_id="self",
            modality="visual",
            intensity=0.5,
        )
        assert s.owner_id == "self"
        assert s.modality == "visual"
        assert s.focal_target_ref is None
        assert s.dwell_seconds == 0.0

    def test_intensity_bounded_0_to_1(self) -> None:
        with pytest.raises(ValidationError):
            AttentionSchema(
                schema_id="x",
                ts="2026-06-21T03:00:00+00:00",
                owner_id="self",
                modality="visual",
                intensity=1.5,
            )
        with pytest.raises(ValidationError):
            AttentionSchema(
                schema_id="x",
                ts="2026-06-21T03:00:00+00:00",
                owner_id="self",
                modality="visual",
                intensity=-0.1,
            )

    def test_rejects_unknown_modality(self) -> None:
        with pytest.raises(ValidationError):
            AttentionSchema(
                schema_id="x",
                ts="2026-06-21T03:00:00+00:00",
                owner_id="self",
                modality="olfactory",  # type: ignore[arg-type]
                intensity=0.5,
            )

    def test_all_documented_modalities_accepted(self) -> None:
        modalities: tuple[Modality, ...] = ("visual", "auditory", "internal", "social")
        for m in modalities:
            s = AttentionSchema(
                schema_id="x",
                ts="2026-06-21T03:00:00+00:00",
                owner_id="self",
                modality=m,
                intensity=0.5,
            )
            assert s.modality == m


class TestTrackerRingBuffer:
    def test_default_capacity_is_60(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        assert tracker.capacity == 60

    def test_buffer_initially_empty(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        assert tracker.buffer == ()

    def test_record_appends_to_buffer(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(focal_target_ref="wifi_cam.see", modality="visual", intensity=0.8)
        assert len(tracker.buffer) == 1
        assert tracker.buffer[0].focal_target_ref == "wifi_cam.see"

    def test_buffer_caps_at_capacity(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db, capacity=5)
        for i in range(10):
            tracker.record(
                focal_target_ref=f"target_{i}", modality="visual", intensity=0.5
            )
        assert len(tracker.buffer) == 5
        # Newest 5 retained
        assert tracker.buffer[-1].focal_target_ref == "target_9"
        assert tracker.buffer[0].focal_target_ref == "target_5"

    def test_owner_id_propagates(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db, owner_id="kouta")
        s = tracker.record(
            focal_target_ref="wifi_cam.see", modality="visual", intensity=0.5
        )
        assert s.owner_id == "kouta"


class TestUpdateFromFrame:
    def test_modality_inferred_from_camera_focal(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        frames = TickFrameStore(social_db)
        frame = frames.record(
            ConsciousFrameInput(
                attention_target_ref="wifi_cam.see", ignited=True
            )
        )
        s = tracker.update_from_frame(frame)
        assert s.modality == "visual"

    def test_modality_inferred_from_listen_focal(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        frames = TickFrameStore(social_db)
        frame = frames.record(
            ConsciousFrameInput(
                attention_target_ref="wifi_cam.listen", ignited=True
            )
        )
        s = tracker.update_from_frame(frame)
        assert s.modality == "auditory"

    def test_modality_inferred_social_for_person_target(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        frames = TickFrameStore(social_db)
        frame = frames.record(
            ConsciousFrameInput(
                attention_target_ref="person:kouta", ignited=True
            )
        )
        s = tracker.update_from_frame(frame)
        assert s.modality == "social"

    def test_modality_internal_when_no_focal(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        frames = TickFrameStore(social_db)
        frame = frames.record(
            ConsciousFrameInput(attention_target_ref=None, ignited=True)
        )
        s = tracker.update_from_frame(frame)
        assert s.modality == "internal"

    def test_ignited_frame_yields_higher_intensity(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        frames = TickFrameStore(social_db)
        ignited = frames.record(
            ConsciousFrameInput(attention_target_ref="wifi_cam.see", ignited=True)
        )
        subliminal = frames.record(
            ConsciousFrameInput(attention_target_ref="wifi_cam.see", ignited=False)
        )
        s_ig = tracker.update_from_frame(ignited)
        s_sub = tracker.update_from_frame(subliminal)
        assert s_ig.intensity > s_sub.intensity

    def test_source_tick_id_tracked(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        frames = TickFrameStore(social_db)
        frame = frames.record(
            ConsciousFrameInput(attention_target_ref="wifi_cam.see", ignited=True)
        )
        s = tracker.update_from_frame(frame)
        assert s.source_tick_id == frame.tick_id


class TestDwell:
    def test_first_focal_has_zero_dwell(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        s = tracker.record(
            focal_target_ref="wifi_cam.see",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:00+00:00",
        )
        assert s.dwell_seconds == 0.0

    def test_same_focal_accumulates_dwell(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="wifi_cam.see",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:00+00:00",
        )
        # 10 seconds later, same focal
        s = tracker.update_dwell_from_ts(
            focal_target_ref="wifi_cam.see",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:10+00:00",
        )
        assert s.dwell_seconds == pytest.approx(10.0)

    def test_different_focal_resets_dwell(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="wifi_cam.see",
            modality="visual",
            intensity=0.5,
            ts="2026-06-21T03:00:00+00:00",
            dwell_seconds=20.0,
        )
        s = tracker.update_dwell_from_ts(
            focal_target_ref="person:kouta",
            modality="social",
            intensity=0.5,
            ts="2026-06-21T03:00:10+00:00",
        )
        assert s.dwell_seconds == 0.0


class TestFlush:
    def test_flush_persists_all_buffered(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(
            focal_target_ref="wifi_cam.see", modality="visual", intensity=0.5
        )
        tracker.record(
            focal_target_ref="person:kouta", modality="social", intensity=0.7
        )
        count = tracker.flush()
        assert count == 2
        rows = social_db.fetchall("SELECT modality FROM attention_schemas")
        assert {r["modality"] for r in rows} == {"visual", "social"}

    def test_flush_clears_buffer(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        tracker.record(focal_target_ref="x", modality="visual", intensity=0.5)
        tracker.flush()
        assert tracker.buffer == ()

    def test_flush_empty_buffer_returns_zero(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        assert tracker.flush() == 0

    def test_flush_persists_source_tick_id(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db)
        frames = TickFrameStore(social_db)
        frame = frames.record(
            ConsciousFrameInput(attention_target_ref="wifi_cam.see", ignited=True)
        )
        tracker.update_from_frame(frame)
        tracker.flush()
        row = social_db.fetchone(
            "SELECT source_tick_id FROM attention_schemas LIMIT 1"
        )
        assert row["source_tick_id"] == frame.tick_id

    def test_flush_owner_id_persisted(self, social_db) -> None:
        tracker = AttentionSchemaTracker(db=social_db, owner_id="kouta")
        tracker.record(
            focal_target_ref="wifi_cam.see", modality="visual", intensity=0.5
        )
        tracker.flush()
        row = social_db.fetchone("SELECT owner_id FROM attention_schemas LIMIT 1")
        assert row["owner_id"] == "kouta"
