from __future__ import annotations

from boundary_mcp.store import BoundaryStore

from individual_kernel_mcp.agency import (
    ActionProposal,
    ExpectedEffect,
    PredictedEffects,
)
from individual_kernel_mcp.boundary_adapter import BoundaryPolicyAdapter
from individual_kernel_mcp.hook_cli import pre_tool_use
from individual_kernel_mcp.tick import FieldRuntime, TickProducer


def test_existing_boundary_policy_denies_hook_runtime_action(
    social_db, tmp_path
) -> None:
    policy_path = tmp_path / "socialPolicy.toml"
    policy_path.write_text(
        """
[global]
timezone = "UTC"
quiet_hours = []
max_nudges_per_hour = 10

[[person_rules]]
person_id = "kouta"
avoid_actions = ["say"]
""".strip(),
        encoding="utf-8",
    )
    producer = TickProducer(
        social_db,
        interoception_path=tmp_path / "interoception.json",
        desires_path=tmp_path / "desires.json",
    )
    opened = producer.begin_tick("user_prompt", person_id="kouta", user_text="hello")
    field = producer.compete_and_commit(opened.tick_id).field
    runtime = FieldRuntime(
        social_db,
        producer=producer,
        boundary_evaluator=BoundaryPolicyAdapter(
            BoundaryStore(db=social_db, policy_path=policy_path)
        ),
    )
    intention = runtime.propose_action(
        ActionProposal(
            field_id=field.field_id,
            tool_name="mcp__tts__say",
            tool_input={"text": "hello"},
            predicted_effects=PredictedEffects(
                social=ExpectedEffect(summary="person hears hello")
            ),
            goal="speak",
        )
    )

    result = pre_tool_use(
        {
            "tool_name": "mcp__tts__say",
            "tool_input": {"text": "hello"},
        },
        runtime,
    )

    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "person-specific rule avoids say" in output["permissionDecisionReason"]
    assert producer.agency.get(intention.action_id).status == "DENIED"
