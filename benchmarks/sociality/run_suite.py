#!/usr/bin/env python3
"""Small deterministic benchmark harness for the sociality MCP family."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in (
    "social-core",
    "social-state-mcp",
    "relationship-mcp",
    "joint-attention-mcp",
    "boundary-mcp",
):
    sys.path.insert(0, str(ROOT / package / "src"))

from boundary_mcp.store import BoundaryStore  # noqa: E402
from joint_attention_mcp.store import JointAttentionStore  # noqa: E402
from relationship_mcp.store import RelationshipStore  # noqa: E402
from social_core import SocialEventCreate  # noqa: E402
from social_state_mcp.store import SocialStateStore  # noqa: E402


def main(fixtures_dir: str) -> None:
    directory = Path(fixtures_dir)
    outputs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "social.db"
        social_state = SocialStateStore(db_path)
        relationship = RelationshipStore(db_path)
        joint_attention = JointAttentionStore(db_path)
        boundary = BoundaryStore(db_path, policy_path=ROOT / "examples" / "configs" / "socialPolicy.example.toml")

        for fixture_path in sorted(directory.glob("*.json")):
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            scenario = fixture["scenario"]
            for event in fixture.get("events", []):
                social_state.events.ingest(SocialEventCreate.model_validate(event))
            if "scene" in fixture:
                joint_attention.ingest_scene_parse(fixture["scene"])
            if scenario == "focused_work_low_energy":
                outputs.append(
                    {
                        "scenario": scenario,
                        "state": social_state.get_social_state(window_seconds=900, person_id="kouta").model_dump(mode="json"),
                    }
                )
            elif scenario == "direct_question":
                outputs.append(
                    {
                        "scenario": scenario,
                        "state": social_state.get_social_state(window_seconds=900, person_id="kouta").model_dump(mode="json"),
                    }
                )
            elif scenario == "two_mugs":
                outputs.append(
                    {
                        "scenario": scenario,
                        "resolution": joint_attention.resolve_reference(expression="the blue mug", person_id="kouta").model_dump(mode="json"),
                    }
                )
            elif scenario == "late_night_post":
                outputs.append(
                    {
                        "scenario": scenario,
                        "decision": boundary.evaluate_action(
                            action_type="post_tweet",
                            channel="x",
                            person_id="kouta",
                            context=fixture["context"],
                            payload_preview=fixture["payload_preview"],
                        ).model_dump(mode="json"),
                    }
                )
            elif scenario == "commitment_persistence":
                relationship.upsert_person(person_id="kouta", canonical_name="水島宏太", aliases=["こうちゃん"], role="companion")
                relationship.create_commitment(
                    person_id="kouta",
                    text="remind about dentist tomorrow morning",
                    due_at="2026-04-16T08:00:00+09:00",
                    source="conversation",
                )
                outputs.append(
                    {
                        "scenario": scenario,
                        "person_model": relationship.get_person_model(person_id="kouta").model_dump(mode="json"),
                    }
                )
            elif scenario == "relationship_continuity":
                relationship.upsert_person(person_id="kouta", canonical_name="水島宏太", aliases=["こうちゃん"], role="companion")
                relationship.ingest_interaction(
                    person_id="kouta",
                    channel="voice",
                    direction="human_to_ai",
                    text=fixture["text"],
                    ts=fixture["ts"],
                )
                outputs.append(
                    {
                        "scenario": scenario,
                        "suggestions": [item.model_dump(mode="json") for item in relationship.suggest_followup(person_id="kouta", context="evening_checkin")],
                    }
                )

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python benchmarks/sociality/run_suite.py benchmarks/sociality/fixtures")
    main(sys.argv[1])
