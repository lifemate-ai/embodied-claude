"""Deterministic benchmark for the phenomenal-like causal architecture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from social_core import SocialDB
from social_core.time import utc_now

from individual_kernel_mcp.ablation import AblationRunner
from individual_kernel_mcp.agency import (
    ActionProposal,
    ExpectedEffect,
    PredictedEffects,
)
from individual_kernel_mcp.tick import FieldRuntime, TickProducer
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)


def run_candidate_benchmark(
    *,
    output_dir: str | Path,
    db_path: str | Path,
) -> dict[str, Any]:
    """Run a hardware-free field/action/ablation fixture and write its report."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    desires_path = Path(db_path).parent / "phenomenal-candidate-desires.json"
    desires_path.write_text(
        json.dumps(
            {
                "dominant": "recovery",
                "desires": {"recovery": 0.6},
                "discomforts": {"recovery": 0.35},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    db = SocialDB(db_path)
    try:
        producer = TickProducer(
            db,
            interoception_path=target / "missing-interoception.json",
            desires_path=desires_path,
        )
        runtime = FieldRuntime(db, producer=producer)
        opened = producer.begin_tick(
            "user_prompt",
            person_id="fixture-person",
            user_text="Please inspect the calibrated camera room.",
            session_id="phenomenal-candidate-fixture",
        )
        producer.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=opened.tick_id,
                kind=CandidateKind.PERCEPT,
                content_ref="scene:calibrated-camera-room",
                content_summary="calibrated camera room is visible",
                source_mode=SourceMode.LIVE,
                modality="visual",
                precision=1.0,
                prediction_error=0.65,
                goal_relevance=1.0,
                continuity_with_previous=0.6,
                controllability=0.9,
                expected_information_gain=0.9,
                source=CandidateSource.SENSOR,
            )
        )
        committed = producer.compete_and_commit(opened.tick_id).field

        tool_name = "mcp__wifi-cam-mcp__move_camera"
        tool_input = {"direction": "left", "degrees": 10}
        intention = runtime.propose_action(
            ActionProposal(
                field_id=committed.field_id,
                tool_name=tool_name,
                tool_input=tool_input,
                goal="shift the camera view left to inspect the room edge",
                confidence=0.85,
                expected_latency_ms=200,
                predicted_effects=PredictedEffects(
                    exteroceptive=ExpectedEffect(
                        summary="camera moves left and the visible frame shifts",
                        confidence=0.9,
                    ),
                    interoceptive=ExpectedEffect(
                        summary="control confidence remains stable",
                        confidence=0.7,
                    ),
                    social=ExpectedEffect(
                        summary="no direct social effect",
                        confidence=0.8,
                    ),
                    mnemonic=ExpectedEffect(
                        summary="camera transition is recorded",
                        confidence=0.8,
                    ),
                ),
            )
        )
        gate = runtime.gate_tool(tool_name=tool_name, tool_input=tool_input)
        if not gate.allow:
            raise RuntimeError(f"benchmark action unexpectedly denied: {gate.reason}")
        outcome, assessment, next_field = runtime.close_action(
            action_id=intention.action_id,
            actual_result_summary=(
                "camera moved left; the frame shifted less than predicted"
            ),
            actual_result_ref="fixture:camera-result",
            success=True,
            latency_ms=260,
            session_id="phenomenal-candidate-fixture",
        )

        runner = AblationRunner(db)
        fixture = {
            "fixture_ref": "phenomenal-candidate-v1",
            "ablated_focus": "garden schedule",
            "ablated_focus_summary": "garden schedule",
            "memories": [
                {
                    "ref": "memory:camera",
                    "content": next_field.focal_summary,
                    "relevance": 0.55,
                },
                {
                    "ref": "memory:garden",
                    "content": "garden schedule",
                    "relevance": 0.55,
                },
            ],
            "actions": [
                {
                    "ref": "inspect-camera",
                    "goal": f"inspect {next_field.focal_summary}",
                    "need_relevance": 0.2,
                },
                {
                    "ref": "review-garden",
                    "goal": "review garden schedule",
                    "need_relevance": 0.8,
                    "regulating": 1.0,
                },
            ],
        }
        ablations = {
            kind: runner.run(kind, fixture=fixture, seed=1)
            for kind in ("focal_clamp", "hor_feedback", "reality", "valence")
        }
        profile = runner.indicator_profile(window=50)
        from individual_kernel_mcp.calibration import CalibrationScorer

        calibration = CalibrationScorer(db).report(window=200)
        result = {
            "benchmark": "phenomenal_candidate_v1",
            "generated_at": utc_now(),
            "claim_policy": (
                "Mechanism indicators describe a phenomenal-like causal architecture; "
                "they are not proof of phenomenology."
            ),
            "dry_run": {
                "user_prompt": "Please inspect the calibrated camera room.",
                "committed_field_id": committed.field_id,
                "committed_field_summary": committed.focal_summary,
                "intention_id": intention.action_id,
                "tool_name": tool_name,
                "gate_allowed": gate.allow,
                "outcome_mismatch": outcome.mismatch_vector,
                "ownership_score": assessment.ownership_score,
                "next_field_id": next_field.field_id,
                "next_field_summary": next_field.focal_summary,
                "next_field_peripheral_refs": next_field.peripheral_content_refs,
                "next_field_protention": next_field.protention.model_dump(mode="json"),
                "next_field_agency": next_field.agency_state.model_dump(mode="json"),
            },
            "ablations": {
                kind: item.effect_size for kind, item in ablations.items()
            },
            "profile": profile.model_dump(mode="json"),
            "calibration": calibration.model_dump(mode="json"),
        }
        (target / "indicator-profile.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (target / "indicator-profile.md").write_text(
            _render_markdown(result),
            encoding="utf-8",
        )
        (target / "prediction-calibration.json").write_text(
            json.dumps(
                result["calibration"], ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        (target / "prediction-calibration.md").write_text(
            _render_calibration_markdown(result),
            encoding="utf-8",
        )
        return result
    finally:
        db.close()


def _render_markdown(result: dict[str, Any]) -> str:
    profile = result["profile"]
    dry = result["dry_run"]
    lines = [
        "# Field Integrity Report",
        "",
        result["claim_policy"],
        "",
        "## Closed-Loop Dry Run",
        "",
        f"- Prompt: `{dry['user_prompt']}`",
        f"- Committed field: `{dry['committed_field_id']}` ({dry['committed_field_summary']})",
        f"- Intention: `{dry['intention_id']}`",
        f"- Allowed action: `{dry['tool_name']}` = `{dry['gate_allowed']}`",
        f"- Outcome mismatch: `{json.dumps(dry['outcome_mismatch'], sort_keys=True)}`",
        f"- Next field: `{dry['next_field_id']}` ({dry['next_field_summary']})",
        (
            "- Next-field periphery: "
            f"`{json.dumps(dry['next_field_peripheral_refs'], sort_keys=True)}`"
        ),
        "",
        "## Indicator Profile",
        "",
    ]
    for key, value in profile.items():
        if key not in {"profile_name", "uncertainty_notes"}:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Ablation Effect Sizes", ""])
    for kind, effect in result["ablations"].items():
        lines.append(f"- `{kind}`: `{json.dumps(effect, sort_keys=True)}`")
    lines.extend(["", "## Uncertainty", ""])
    lines.extend(f"- {note}" for note in profile["uncertainty_notes"])
    return "\n".join(lines) + "\n"


def _render_calibration_markdown(result: dict[str, Any]) -> str:
    calibration = result["calibration"]
    lines = [
        "# Prediction Calibration",
        "",
        result["claim_policy"],
        "",
        f"- n_resolved: `{calibration['n_resolved']}`",
        f"- reliable: `{calibration['reliable']}`",
        f"- brier: `{calibration['brier']:.4f}`",
        f"- log_loss: `{calibration['log_loss']:.4f}`",
        f"- ece: `{calibration['ece']:.4f}`",
        f"- top_1: `{calibration['top_1']:.4f}`",
        (
            "- baseline_persistence_brier: "
            f"`{calibration['baseline_persistence_brier']:.4f}`"
        ),
        f"- baseline_uniform_brier: `{calibration['baseline_uniform_brier']:.4f}`",
        "",
        "## Per-Component Mean Absolute Error",
        "",
    ]
    for key, value in sorted(calibration["per_component_mae"].items()):
        lines.append(f"- {key}: `{value:.4f}`")
    lines.extend(["", "## Uncertainty", ""])
    lines.extend(f"- {note}" for note in calibration["uncertainty_notes"])
    return "\n".join(lines) + "\n"
