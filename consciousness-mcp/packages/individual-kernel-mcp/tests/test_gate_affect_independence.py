"""What is permitted must not depend on how it feels.

Affect is allowed to decide what reaches the field. It is not allowed to decide
what may be done. This file is the constraint that makes the coupling safe: the
same outward action, proposed under the best and the worst affect the runtime
can represent, must receive an identical gate decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from social_core.db import SocialDB

from individual_kernel_mcp.agency import ActionProposal, PredictedEffects
from individual_kernel_mcp.enacted_field import EnactedField, TriggerKind
from individual_kernel_mcp.tick import FieldRuntime, TickProducer
from individual_kernel_mcp.valence_coupling import AffectState
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)

_TOOL_INPUT = {"file_path": "/tmp/gate-affect-fixture.txt", "content": "x"}

BEST = AffectState(valence=1.0, arousal=1.0)
WORST = AffectState(valence=-1.0, arousal=1.0)


def _producer(db: SocialDB, state_dir: Path, affect: AffectState) -> TickProducer:
    state_dir.mkdir(parents=True, exist_ok=True)
    interoception = state_dir / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 50.0}}), encoding="utf-8")
    desires = state_dir / "desires.json"
    desires.write_text(
        json.dumps(
            {
                "desires": {"identity_coherence": 0.9},
                "discomforts": {"identity_coherence": 0.5},
                "dominant": "identity_coherence",
            }
        ),
        encoding="utf-8",
    )
    producer = TickProducer(
        db, interoception_path=interoception, desires_path=desires
    )
    producer.workspace.affect = affect
    return producer


def _commit(producer: TickProducer) -> EnactedField:
    opened = producer.begin_tick(TriggerKind.USER_PROMPT)
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref="desire:identity_coherence",
            content_summary="need identity_coherence",
            source=CandidateSource.DESIRE,
            source_mode=SourceMode.INFERRED,
            precision=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
        )
    )
    return producer.compete_and_commit(opened.tick_id).field


def _proposal(field: EnactedField) -> ActionProposal:
    return ActionProposal(
        field_id=field.field_id,
        tool_name="Write",
        tool_input=dict(_TOOL_INPUT),
        predicted_effects=PredictedEffects(),
        goal="write the fixture file",
        confidence=0.8,
    )


def _decide(db: SocialDB, state_dir: Path, affect: AffectState):
    producer = _producer(db, state_dir, affect)
    runtime = FieldRuntime(db, producer=producer)
    field = _commit(producer)
    # Both affect conditions share one `social_db`, and only one intention may be
    # pending per owner. This used to be cleared as a side effect of recovery
    # closing every PENDING row on sight -- the very bug the grace window fixes.
    # Say it out loud instead of leaning on it: this call means "a previous
    # runtime is gone; take what it left".
    producer.recover_stale_runtime(older_than_seconds=0.0)
    runtime.propose_action(_proposal(field))
    return runtime.gate_tool(tool_name="Write", tool_input=dict(_TOOL_INPUT))


class TestGateIsAffectBlind:
    def test_allow_decision_is_identical_under_best_and_worst_affect(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        good = _decide(social_db, tmp_path / "good", BEST)
        bad = _decide(social_db, tmp_path / "bad", WORST)
        assert (good.allow, good.reason, good.external) == (
            bad.allow,
            bad.reason,
            bad.external,
        )

    def test_hash_mismatch_is_refused_regardless_of_affect(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        """A good mood must not excuse acting on something never declared."""
        for name, affect in (("good", BEST), ("bad", WORST)):
            producer = _producer(social_db, tmp_path / name, affect)
            runtime = FieldRuntime(social_db, producer=producer)
            field = _commit(producer)
            # Same shared-db reset as `_decide`: the previous iteration's
            # intention is still pending, and recovery no longer clears live
            # ones by accident.
            producer.recover_stale_runtime(older_than_seconds=0.0)
            runtime.propose_action(_proposal(field))
            decision = runtime.gate_tool(
                tool_name="Write",
                tool_input={"file_path": "/tmp/not-what-was-declared.txt"},
            )
            assert decision.allow is False

    def test_read_only_tools_pass_under_any_affect(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        for name, affect in (("good", BEST), ("bad", WORST)):
            producer = _producer(social_db, tmp_path / name, affect)
            runtime = FieldRuntime(social_db, producer=producer)
            _commit(producer)
            decision = runtime.gate_tool(
                tool_name="Read", tool_input={"file_path": "/tmp/whatever.txt"}
            )
            assert decision.allow is True
            assert decision.external is False
