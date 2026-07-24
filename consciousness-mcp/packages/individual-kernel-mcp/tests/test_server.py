"""Smoke + integration tests for the MCP server surface.

Phase 2.1 exposed 3 tools. This file extends to the Phase 2.3-2.5 surface
that comes with the integration PR: 11 new tools wrapping TickFrameStore,
AttentionSchemaTracker, HORStore + composer.
"""

from __future__ import annotations

import pytest

from individual_kernel_mcp import server
from individual_kernel_mcp.attention_schema import AttentionSchemaTracker
from individual_kernel_mcp.counterfactual import CounterfactualStore
from individual_kernel_mcp.frame import TickFrameStore
from individual_kernel_mcp.hor import HORStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Isolate every test from the real ~/.claude/social.db.

    Sets SOCIAL_DB_PATH to a tmp_path-derived sqlite file AND clears the
    module-level _stores() cache before AND after each test so the
    SocialDB connection points at the temp file.
    """
    db_path = tmp_path / "social.db"
    monkeypatch.setenv("SOCIAL_DB_PATH", str(db_path))
    server.reset_store_cache()
    yield
    server.reset_store_cache()


class TestPhase21Tools:
    def test_phase21_tools_still_registered(self) -> None:
        assert callable(server.record_counterfactual)
        assert callable(server.query_counterfactuals)
        assert callable(server.sleep_consolidate)


class TestPhase23to25ToolsRegistered:
    def test_all_phase_2_3_to_2_5_tool_names_are_registered_on_server_module(
        self,
    ) -> None:
        """Every new wrapper from the integration PR is importable from server."""
        names = [
            "record_tick_frame",
            "get_tick_frame",
            "query_tick_frames",
            "record_attention_schema",
            "update_attention_from_frame",
            "flush_attention_schemas",
            "summarize_attention_schema",
            "record_hor",
            "get_hor",
            "query_hors",
            "compose_introspection_report",
        ]
        for name in names:
            assert hasattr(server, name), f"missing tool: {name}"
            assert callable(getattr(server, name)), f"not callable: {name}"

    def test_enacted_field_tools_are_registered(self) -> None:
        names = [
            "begin_subjective_tick",
            "add_workspace_candidate",
            "commit_subjective_field",
            "get_current_subjective_field",
            "get_subjective_field",
            "query_subjective_fields",
            "propose_field_action",
            "get_pending_intention",
            "close_field_action",
            "close_subjective_tick",
            "get_field_diagnostics",
            "run_field_ablation",
            "pause_field_runtime",
            "resume_field_runtime",
        ]
        for name in names:
            assert callable(getattr(server, name))


class TestStoresDataclass:
    def test_stores_dataclass_exposes_phase_2_3_to_2_5_stores(self) -> None:
        """IndividualKernelStores carries tick/attention/hor stores."""
        stores = server._stores()
        assert isinstance(stores.tick_frames, TickFrameStore)
        assert isinstance(stores.attention_tracker, AttentionSchemaTracker)
        assert isinstance(stores.hor_store, HORStore)
        assert isinstance(stores.counterfactual, CounterfactualStore)
        assert stores.attention_tracker.owner_id == "self"

    def test_stores_share_a_single_socialdb(self) -> None:
        stores = server._stores()
        assert stores.tick_frames._db is stores.db  # noqa: SLF001
        assert stores.attention_tracker.db is stores.db
        assert stores.hor_store._db is stores.db  # noqa: SLF001


class TestTickFrameRoundTrip:
    def test_record_get_query_round_trip(self) -> None:
        result = server.record_tick_frame(
            {
                "person_id": "kouta",
                "ignited": True,
                "attention_target_ref": "wifi_cam.see",
                "dominant_desire": "browse_curiosity",
                "winning_memory_ids": ["mem_a", "mem_b"],
                "prediction_error": {"extero": 0.7, "intero": 0.2, "mnemonic": 0.4},
            }
        )
        tick_id = result["tick_id"]
        assert tick_id.startswith("tick_")

        fetched = server.get_tick_frame(tick_id)
        assert fetched is not None
        assert fetched["attention_target_ref"] == "wifi_cam.see"
        assert fetched["winning_memory_ids"] == ["mem_a", "mem_b"]

        ignited = server.query_tick_frames(ignited_only=True)
        assert len(ignited) == 1
        assert ignited[0]["tick_id"] == tick_id

    def test_get_missing_tick_returns_none(self) -> None:
        assert server.get_tick_frame("tick_nonexistent_xyz") is None


class TestAttentionSchemaRoundTrip:
    def test_record_flush_summarize_round_trip(self) -> None:
        server.record_attention_schema(
            focal_target_ref="wifi_cam.see",
            modality="visual",
            intensity=0.7,
        )
        server.record_attention_schema(
            focal_target_ref="wifi_cam.listen",
            modality="auditory",
            intensity=0.5,
        )
        flush_result = server.flush_attention_schemas()
        assert flush_result["count"] == 2
        summary = server.summarize_attention_schema()
        # Buffer is empty after flush; reflection without extra_history is empty.
        assert summary["window_size"] == 0

        summary_with_history = server.summarize_attention_schema(extra_history=10)
        assert summary_with_history["window_size"] == 2
        modalities = set(summary_with_history["modality_distribution"].keys())
        assert {"visual", "auditory"} <= modalities

    def test_update_attention_from_frame_uses_recorded_frame(self) -> None:
        frame = server.record_tick_frame(
            {
                "attention_target_ref": "wifi_cam.see",
                "ignited": True,
            }
        )
        schema = server.update_attention_from_frame(tick_id=frame["tick_id"])
        assert schema["modality"] == "visual"
        assert schema["source_tick_id"] == frame["tick_id"]
        # Buffer holds it now.
        flush_result = server.flush_attention_schemas()
        assert flush_result["count"] == 1

    def test_update_attention_from_missing_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="no tick_frame"):
            server.update_attention_from_frame(tick_id="tick_nonexistent")


class TestHORRoundTrip:
    def test_record_get_query_round_trip(self) -> None:
        result = server.record_hor(
            {
                "target_kind": "none",
                "asserted_mode": "attending",
                "asserted_content": "I am attending the workspace",
                "source": "reflection",
            }
        )
        hor_id = result["hor_id"]
        assert hor_id.startswith("hor_")
        fetched = server.get_hor(hor_id)
        assert fetched is not None
        assert fetched["asserted_mode"] == "attending"

        results = server.query_hors(asserted_mode="attending")
        assert len(results) == 1
        assert results[0]["hor_id"] == hor_id

    def test_get_missing_hor_returns_none(self) -> None:
        assert server.get_hor("hor_nonexistent_xyz") is None


class TestComposeIntrospectionReport:
    def test_integrates_hor_attention_counterfactual(self) -> None:
        opened = server.begin_subjective_tick(
            trigger="user_prompt",
            user_text="attend to the silence",
        )
        server.commit_subjective_field(opened["tick_id"])
        # Seed a counterfactual (recent)
        server.record_counterfactual(
            {
                "rejected_action": "speak_loudly",
                "source": "boundary_deny",
            }
        )

        report = server.compose_introspection_report(window_hours=24)
        assert report["canonical_statement"]  # non-empty
        assert report["current_hor"] is not None
        assert report["current_hor"]["source"] == "schema_readout"
        assert report["recent_counterfactual_count"] == 1
        assert report["hor_validation"]["well_formed"] is True
        assert report["field_hor_consistent"] is True

    def test_empty_state_returns_report(self) -> None:
        report = server.compose_introspection_report()
        assert report["current_hor"] is None
        assert report["recent_counterfactual_count"] == 0
        assert report["canonical_statement"] == (
            "Current state unavailable: no committed field."
        )


class TestBoundaryPolicyDocstring:
    def test_server_module_docstring_declares_runtime_boundary_gating(self) -> None:
        doc = server.__doc__ or ""
        assert "boundary-mcp" in doc
        assert "ActionBottleneck" in doc


class TestFieldDiagnostics:
    def test_reports_action_trace_consistency_and_prediction_mismatch(self) -> None:
        opened = server.begin_subjective_tick(
            trigger="user_prompt",
            user_text="write the fixture",
        )
        field = server.commit_subjective_field(opened["tick_id"])["field"]
        intention = server.propose_field_action(
            field_id=field["field_id"],
            tool_name="Write",
            tool_input={"file_path": "/tmp/fixture", "content": "ok"},
            predicted_effects={},
            goal="write fixture",
        )
        gate = server._stores().field_runtime.gate_tool(
            tool_name="Write",
            tool_input={"file_path": "/tmp/fixture", "content": "ok"},
        )
        assert gate.allow
        server.close_field_action(
            action_id=intention["action_id"],
            actual_result_summary="fixture written",
            success=True,
        )

        diagnostics = server.get_field_diagnostics()

        assert diagnostics["action_trace_consistency"] is True
        assert diagnostics["recent_action_outcomes"][0]["mismatch_vector"]
        assert diagnostics["runtime_safety"] == {
            "mass_copy_enabled": False,
            "high_frequency_spawning_enabled": False,
            "reversible_ablation": True,
            "pause_resume_available": True,
        }


class TestServerToolsHonorIsolatedDB:
    def test_server_tools_do_not_write_to_real_home_db(self, tmp_path) -> None:
        """The isolated_db fixture redirects writes; sanity check it works."""

        # The isolated_db fixture set SOCIAL_DB_PATH to tmp_path/social.db.
        # Recording should hit that file, not the real one.
        server.record_tick_frame({"attention_target_ref": "x", "ignited": False})
        assert (tmp_path / "social.db").exists() or any(
            tmp_path.rglob("social.db")
        )


def test_main_is_importable() -> None:
    assert callable(server.main)
