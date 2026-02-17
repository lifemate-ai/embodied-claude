"""Tests for desire_updater."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from desire_updater import (
    DesireState,
    calculate_desire_level,
    compute_desires,
    get_latest_memory_timestamp,
    load_desires,
    save_desires,
)


class TestCalculateDesireLevel:
    def test_no_prior_satisfaction_returns_max(self):
        assert calculate_desire_level(None, 1.0) == 1.0

    def test_just_satisfied_returns_zero(self):
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        last = now  # 今まさに満たされた
        assert calculate_desire_level(last, 1.0, now) == 0.0

    def test_half_elapsed_returns_half(self):
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=0.5)
        assert calculate_desire_level(last, 1.0, now) == pytest.approx(0.5, abs=0.01)

    def test_full_elapsed_returns_one(self):
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=1.0)
        assert calculate_desire_level(last, 1.0, now) == 1.0

    def test_over_elapsed_capped_at_one(self):
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=5.0)
        assert calculate_desire_level(last, 1.0, now) == 1.0

    def test_custom_threshold(self):
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=1.0)
        # 2時間が閾値なら1時間経過で0.5
        assert calculate_desire_level(last, 2.0, now) == pytest.approx(0.5, abs=0.01)

    def test_naive_datetime_handled(self):
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        last = datetime(2026, 2, 18, 11, 0, 0)  # naive
        # should not raise
        result = calculate_desire_level(last, 1.0, now)
        assert result == 1.0


class TestGetLatestMemoryTimestamp:
    def _make_collection(self, docs, metas):
        coll = MagicMock()
        coll.get.return_value = {"documents": docs, "metadatas": metas}
        return coll

    def test_returns_none_when_no_match(self):
        coll = self._make_collection(
            ["今日は晴れです", "部屋が暑い"],
            [
                {"timestamp": "2026-02-18T10:00:00"},
                {"timestamp": "2026-02-18T11:00:00"},
            ],
        )
        result = get_latest_memory_timestamp(coll, ["外を見た", "空を見た"])
        assert result is None

    def test_returns_latest_matching_timestamp(self):
        coll = self._make_collection(
            ["外を見た、空が青い", "今日は雨", "空を見た、曇ってる"],
            [
                {"timestamp": "2026-02-18T08:00:00"},
                {"timestamp": "2026-02-18T09:00:00"},
                {"timestamp": "2026-02-18T10:00:00"},  # 最新
            ],
        )
        result = get_latest_memory_timestamp(coll, ["外を見た", "空を見た"])
        assert result is not None
        assert result.hour == 10

    def test_handles_missing_timestamp_in_meta(self):
        coll = self._make_collection(
            ["外を見た"],
            [{"timestamp": ""}],  # タイムスタンプなし
        )
        result = get_latest_memory_timestamp(coll, ["外を見た"])
        assert result is None

    def test_handles_collection_error(self):
        coll = MagicMock()
        coll.get.side_effect = Exception("DB error")
        result = get_latest_memory_timestamp(coll, ["外を見た"])
        assert result is None


class TestComputeDesires:
    def test_all_desires_computed(self):
        coll = MagicMock()
        coll.get.return_value = {"documents": [], "metadatas": []}
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)

        state = compute_desires(coll, now)

        assert set(state.desires.keys()) == {
            "look_outside",
            "browse_curiosity",
            "miss_companion",
            "observe_room",
        }

    def test_all_max_when_no_memories(self):
        coll = MagicMock()
        coll.get.return_value = {"documents": [], "metadatas": []}
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)

        state = compute_desires(coll, now)

        for level in state.desires.values():
            assert level == 1.0

    def test_dominant_is_highest_desire(self):
        # look_outside だけ最近満たされてる
        coll = MagicMock()
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        recent = (now - timedelta(minutes=5)).isoformat()

        coll.get.return_value = {
            "documents": ["外を見た、空が青い"],
            "metadatas": [{"timestamp": recent}],
        }

        state = compute_desires(coll, now)

        # look_outside だけ低いはずなので dominant は他のどれか
        assert state.dominant != "look_outside"
        assert state.desires["look_outside"] < state.desires["miss_companion"]

    def test_desires_values_in_range(self):
        coll = MagicMock()
        coll.get.return_value = {"documents": [], "metadatas": []}
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)

        state = compute_desires(coll, now)

        for level in state.desires.values():
            assert 0.0 <= level <= 1.0


class TestSaveAndLoadDesires:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-02-18T12:00:00+00:00",
                desires={"look_outside": 0.8, "miss_companion": 0.5},
                dominant="look_outside",
            )
            save_desires(state, path)

            loaded = load_desires(path)
            assert loaded is not None
            assert loaded.dominant == "look_outside"
            assert loaded.desires["look_outside"] == pytest.approx(0.8)
            assert loaded.desires["miss_companion"] == pytest.approx(0.5)

    def test_load_missing_file_returns_none(self):
        result = load_desires(Path("/nonexistent/path/desires.json"))
        assert result is None

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "desires.json"
            state = DesireState(
                updated_at="2026-02-18T12:00:00",
                desires={"observe_room": 1.0},
                dominant="observe_room",
            )
            save_desires(state, path)
            assert path.exists()

    def test_saved_json_is_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "desires.json"
            state = DesireState(
                updated_at="2026-02-18T12:00:00",
                desires={"look_outside": 0.6},
                dominant="look_outside",
            )
            save_desires(state, path)

            with open(path) as f:
                data = json.load(f)
            assert data["dominant"] == "look_outside"
            assert data["desires"]["look_outside"] == pytest.approx(0.6)
