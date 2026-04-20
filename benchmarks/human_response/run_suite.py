"""Run the human-response benchmark suite against the interaction orchestrator.

Usage (from the sociality-mcp venv)::

    uv run --directory sociality-mcp python \\
        ../benchmarks/human_response/run_suite.py

The suite loads every fixture under ``fixtures/``, replays its setup
(ingest events, write agent experiences / interpretation shifts, fake a
desire snapshot), then calls ``compose_interaction_context`` and
``plan_response``. Each fixture carries rule-based expectations tagged by
scoring dimension; the suite aggregates into a :class:`SuiteScore` and
fails when the floors in §17 are violated.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = BENCH_DIR / "fixtures"

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from scoring import (  # noqa: E402  (sys.path manipulated above)
    AssertionResult,
    FixtureScore,
    SuiteScore,
    evaluate_rule,
    load_fixtures,
)


def _build_stores(tmp_path: Path, policy_timezone: str = "Asia/Tokyo"):
    # Defer MCP imports until the sociality venv is active.
    from boundary_mcp.store import BoundaryStore
    from interaction_orchestrator_mcp.store import InteractionOrchestratorStore
    from joint_attention_mcp.store import JointAttentionStore
    from relationship_mcp.store import RelationshipStore
    from self_narrative_mcp.store import SelfNarrativeStore
    from social_core import SocialDB
    from social_state_mcp.store import SocialStateStore

    policy_path = tmp_path / "socialPolicy.toml"
    policy_path.write_text(
        f"""
[global]
timezone = "{policy_timezone}"
quiet_hours = ["00:00-07:00"]
max_nudges_per_hour = 2

[[person_rules]]
person_id = "kouta"
avoid_actions = ["camera_speaker_after_midnight"]
preferred_nudge_style = "brief_gentle"
""".strip(),
        encoding="utf-8",
    )

    db = SocialDB(tmp_path / "social.db")
    return {
        "db": db,
        "social_state": SocialStateStore(
            db=db,
            quiet_hours_windows=["00:00-07:00"],
            policy_timezone=policy_timezone,
        ),
        "relationship": RelationshipStore(db=db),
        "joint_attention": JointAttentionStore(db=db),
        "boundary": BoundaryStore(db=db, policy_path=policy_path),
        "self_narrative": SelfNarrativeStore(db=db),
        "orchestrator": InteractionOrchestratorStore(db=db),
        "policy_timezone": policy_timezone,
    }


def _apply_setup(stores: dict[str, Any], setup: dict[str, Any]) -> None:
    from interaction_orchestrator_mcp.schemas import (
        RecordAgentExperienceInput,
        RecordInterpretationShiftInput,
    )

    for event in setup.get("events", []):
        stores["social_state"].ingest_social_event(event)
    for person in setup.get("persons", []):
        stores["relationship"].upsert_person(**person)
    for interaction in setup.get("interactions", []):
        stores["relationship"].ingest_interaction(**interaction)
    for boundary in setup.get("boundaries", []):
        stores["relationship"].record_boundary(**boundary)
    for commitment in setup.get("commitments", []):
        stores["relationship"].create_commitment(**commitment)
    for experience in setup.get("agent_experiences", []):
        stores["orchestrator"].record_agent_experience(
            RecordAgentExperienceInput.model_validate(experience)
        )
    for shift in setup.get("interpretation_shifts", []):
        stores["orchestrator"].record_interpretation_shift(
            RecordInterpretationShiftInput.model_validate(shift)
        )


def _seed_desires(tmp_path: Path, desires: dict[str, Any] | None) -> None:
    if not desires:
        os.environ.pop("DESIRES_PATH", None)
        return
    path = tmp_path / "desires.json"
    path.write_text(json.dumps(desires), encoding="utf-8")
    os.environ["DESIRES_PATH"] = str(path)


def _run_fixture(fixture: dict[str, Any], tmp_path: Path) -> FixtureScore:
    from interaction_orchestrator_mcp.compose import compose_interaction_context
    from interaction_orchestrator_mcp.plan import plan_response
    from interaction_orchestrator_mcp.schemas import (
        ComposeInteractionContextInput,
        PlanResponseInput,
    )

    stores = _build_stores(tmp_path)
    try:
        setup = fixture.get("setup", {})
        _apply_setup(stores, setup)
        _seed_desires(tmp_path, setup.get("desires"))

        inp = fixture.get("input", {})
        ctx = compose_interaction_context(
            ComposeInteractionContextInput(
                person_id=inp.get("person_id"),
                channel=inp.get("channel", "chat"),
                user_text=inp.get("user_text"),
                autonomous_trigger=inp.get("autonomous_trigger"),
                include_private=True,
                max_chars=3000,
            ),
            social_state_store=stores["social_state"],
            relationship_store=stores["relationship"],
            joint_attention_store=stores["joint_attention"],
            boundary_store=stores["boundary"],
            self_narrative_store=stores["self_narrative"],
            orchestrator_store=stores["orchestrator"],
            policy_timezone=stores["policy_timezone"],
        )
        plan = plan_response(
            PlanResponseInput(interaction_context=ctx, user_text=inp.get("user_text"))
        )
        bundle = {"ctx": ctx.model_dump(mode="json"), "plan": plan.model_dump(mode="json")}
    finally:
        stores["db"].close()

    score = FixtureScore(
        fixture_id=fixture["id"], description=fixture.get("description", "")
    )
    for rule in fixture.get("expected", []):
        dimension = rule.get("dimension")
        if dimension not in {
            "context_specificity",
            "relationship_continuity",
            "bounded_initiative",
            "boundary_respect",
            "memory_selectivity",
            "self_correction",
            "non_genericness",
            "technical_fit",
            "no_confabulation",
        }:
            continue
        ok, detail = evaluate_rule(rule, bundle)
        score.results.append(
            AssertionResult(
                dimension=dimension,
                rule=f"{rule.get('op')} {rule.get('path')}",
                passed=ok,
                detail=detail if not ok else "",
            )
        )
    return score


def run_suite(fixture_dir: Path = FIXTURE_DIR) -> SuiteScore:
    fixtures = load_fixtures(fixture_dir)
    suite = SuiteScore()
    for fixture in fixtures:
        with tempfile.TemporaryDirectory() as tmp:
            suite.fixtures.append(_run_fixture(fixture, Path(tmp)))
    return suite


def _print_report(suite: SuiteScore) -> int:
    for fixture in suite.fixtures:
        per = fixture.per_dimension()
        status = "PASS" if not fixture.failures() else "FAIL"
        print(f"[{status}] {fixture.fixture_id}: {fixture.description}")
        for dim, score in per.items():
            print(f"    {dim}: {score:.2f}")
        for failure in fixture.failures():
            print(f"    x {failure.dimension} :: {failure.rule} — {failure.detail}")
    per = suite.per_dimension_mean()
    print("\n== Suite aggregate ==")
    for dim, score in per.items():
        print(f"  {dim}: {score:.2f}")
    print(f"  average: {suite.average():.2f}")
    passed, reasons = suite.passes()
    if passed:
        print("SUITE PASS")
        return 0
    print("SUITE FAIL:")
    for reason in reasons:
        print(f"  - {reason}")
    return 1


if __name__ == "__main__":
    suite = run_suite()
    raise SystemExit(_print_report(suite))
