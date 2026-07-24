from __future__ import annotations

import json

from individual_kernel_mcp.benchmark import run_candidate_benchmark


def test_candidate_benchmark_writes_inspectable_indicator_outputs(tmp_path):
    result = run_candidate_benchmark(
        output_dir=tmp_path / "results",
        db_path=tmp_path / "benchmark.db",
    )

    json_path = tmp_path / "results" / "indicator-profile.json"
    markdown_path = tmp_path / "results" / "indicator-profile.md"
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert result["profile"]["profile_name"] == "FieldIntegrityReport"
    assert stored == result
    assert set(result["ablations"]) == {
        "focal_clamp",
        "hor_feedback",
        "reality",
        "valence",
    }
    assert result["ablations"]["reality"]["mean_score_delta"] > 0.0
    assert result["ablations"]["valence"]["mean_score_delta"] > 0.0
    assert result["profile"]["valence_coupling"] > 0.0
    assert result["dry_run"]["committed_field_summary"] == (
        "calibrated camera room is visible"
    )
    assert result["dry_run"]["next_field_id"]
    assert any(
        ref.startswith("outcome:")
        for ref in result["dry_run"]["next_field_peripheral_refs"]
    )
    assert result["dry_run"]["outcome_mismatch"]
    assert "consciousness probability" not in markdown.lower()
    assert "Field Integrity Report" in markdown
