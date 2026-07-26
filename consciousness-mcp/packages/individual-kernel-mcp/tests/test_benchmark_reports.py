"""Benchmark emits both mandated reports with disciplined vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from individual_kernel_mcp.benchmark import run_candidate_benchmark
from individual_kernel_mcp.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss,
)

_FORBIDDEN_PHRASES = (
    "consciousness probability",
    "is conscious",
    "sentience score",
)


@pytest.fixture(autouse=True)
def _generative_flag_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml = tmp_path / "behavior-on.toml"
    toml.write_text(
        "[individual-kernel]\n"
        "generative_field_model = true\n"
        "generative_rollout_horizon = 2\n"
    )
    monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(toml))


class TestPureMetrics:
    def test_brier_hand_computed(self) -> None:
        assert brier_score([(0.8, True), (0.4, False)]) == pytest.approx(
            ((0.8 - 1.0) ** 2 + 0.4**2) / 2
        )

    def test_log_loss_clips_extremes(self) -> None:
        value = log_loss([(0.0, True)])
        assert value > 0.0
        assert value < 10.0

    def test_ece_perfect_calibration_is_zero(self) -> None:
        pairs = [(0.5, True), (0.5, False)]
        ece, bins = expected_calibration_error(pairs)
        assert ece == pytest.approx(0.0)
        assert bins[0]["count"] == 2.0


class TestBenchmarkReports:
    def test_benchmark_writes_both_reports(self, tmp_path: Path) -> None:
        output = tmp_path / "out"
        result = run_candidate_benchmark(
            output_dir=output, db_path=tmp_path / "field.db"
        )
        indicator = (output / "indicator-profile.md").read_text(encoding="utf-8")
        calibration = (output / "prediction-calibration.md").read_text(
            encoding="utf-8"
        )
        combined = (indicator + calibration).lower()
        for phrase in _FORBIDDEN_PHRASES:
            assert phrase not in combined
        assert "not proof of phenomenology" in indicator
        assert "not proof of phenomenology" in calibration
        assert "n_resolved" in calibration
        assert result["calibration"]["n_resolved"] >= 1
        assert result["calibration"]["reliable"] is False
        assert any(
            "Provisional" in note
            for note in result["calibration"]["uncertainty_notes"]
        )
        detail = result["profile"]["prediction_calibration_detail"]
        assert detail["source"] in {"legacy_binary", "generative"}
        payload = json.loads(
            (output / "prediction-calibration.json").read_text(encoding="utf-8")
        )
        assert payload["baseline_uniform_brier"] >= 0.0
