"""Generative MCP tool surface: rollout, queries, calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from individual_kernel_mcp import server
from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOCIAL_DB_PATH", str(tmp_path / "social.db"))
    toml = tmp_path / "behavior-on.toml"
    toml.write_text(
        "[individual-kernel]\n"
        "generative_field_model = true\n"
        "generative_rollout_horizon = 2\n"
    )
    monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(toml))
    server.reset_store_cache()
    yield
    server.reset_store_cache()


def _commit_field() -> str:
    stores = server._stores()
    opened = stores.tick_producer.begin_tick(TriggerKind.USER_PROMPT)
    stores.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref="desire:identity_coherence",
            content_summary="focus desire",
            source=CandidateSource.EXPLICIT,
            source_mode=SourceMode.INFERRED,
            precision=1.0,
            prediction_error=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
            expected_information_gain=1.0,
            continuity_with_previous=1.0,
            controllability=1.0,
            social_relevance=1.0,
        )
    )
    return stores.tick_producer.compete_and_commit(opened.tick_id).field.field_id


class TestGenerativeTools:
    def test_rollout_without_field_reports_error(self) -> None:
        result = server.rollout_protention()
        assert result == {"error": "no committed field for owner"}

    def test_rollout_returns_forecast_and_steps(self) -> None:
        field_id = _commit_field()
        result = server.rollout_protention(tool_name="Write", horizon=2)
        assert result["field_id"] == field_id
        assert result["action_kind"] == "tool:Write"
        assert len(result["steps"]) == 2
        assert result["forecast"]["predictions"]

    def test_query_tools_return_lists(self) -> None:
        _commit_field()
        assert isinstance(server.query_imagined_trajectories(), list)
        assert isinstance(server.query_experienced_transitions(), list)

    def test_calibration_report_shape(self) -> None:
        report = server.get_generative_model_calibration()
        assert report["n_resolved"] == 0
        assert report["reliable"] is False
        assert "brier" in report
