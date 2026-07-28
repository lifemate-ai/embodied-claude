"""One rule decides what comes next, and everything that stores it agrees.

`predicted_next_focus` is assigned twice while a tick commits: once from the
protention, then again from the attention schema. Both derived it from the same
competition by the same rule -- the runner-up, or the winner when nothing was
rejected -- so the two agreed, and these tests passed before the rule was moved
into one place. They exist to keep it that way: if the derivation is ever split
again, the object, the trace and the stored surface stop matching here first.

The value round-trips through the epistemic trace. `enacted_fields` has no
column of that name, and `EnactedFieldStore.update` does not write one, so the
trace is the only home it has.
"""

from __future__ import annotations

import json
from pathlib import Path

from social_core.db import SocialDB

from individual_kernel_mcp.enacted_field import EnactedFieldStore, TriggerKind
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)


def _producer(social_db: SocialDB, tmp_path: Path) -> TickProducer:
    interoception = tmp_path / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 50.0}}), encoding="utf-8")
    desires = tmp_path / "desires.json"
    desires.write_text(
        json.dumps({"desires": {"identity_coherence": 0.9}, "dominant": "identity_coherence"}),
        encoding="utf-8",
    )
    return TickProducer(
        social_db, interoception_path=interoception, desires_path=desires
    )


def _commit(producer: TickProducer, focus_ref: str = "desire:identity_coherence"):
    opened = producer.begin_tick(TriggerKind.USER_PROMPT)
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref=focus_ref,
            content_summary=f"focus {focus_ref}",
            source=CandidateSource.DESIRE,
            source_mode=SourceMode.INFERRED,
            precision=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
            continuity_with_previous=1.0,
            controllability=1.0,
        )
    )
    return producer.compete_and_commit(opened.tick_id).field


class TestOneCanonicalPrediction:
    def test_a_reloaded_field_predicts_what_the_committed_one_did(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        committed = _commit(_producer(social_db, tmp_path))
        reloaded = EnactedFieldStore(social_db).get(committed.field_id)

        assert reloaded is not None
        assert reloaded.predicted_next_focus == committed.predicted_next_focus

    def test_the_trace_agrees_with_the_field(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        # The trace is diagnostics for the same state, so it may not carry a
        # second, different answer under the same name.
        committed = _commit(_producer(social_db, tmp_path))
        reloaded = EnactedFieldStore(social_db).get(committed.field_id)

        assert reloaded is not None
        assert (
            reloaded.epistemic_trace.get("predicted_next_focus")
            == reloaded.predicted_next_focus
        )

    def test_the_stored_surface_agrees_with_the_reloaded_field(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        # The surface is what the agent reads. It is rendered before the row is
        # written, so a prediction that does not survive persistence would make
        # the stored string describe a state the stored row does not have.
        committed = _commit(_producer(social_db, tmp_path))
        reloaded = EnactedFieldStore(social_db).get(committed.field_id)

        assert reloaded is not None
        assert reloaded.predicted_next_focus is not None
        assert f'predicted_next="{reloaded.predicted_next_focus}"' in (
            reloaded.phenomenal_surface
        )

    def test_the_attention_schema_predicts_what_the_field_predicts(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        # The two used to be computed independently. They are now one call, and
        # the schema is the record a reader consults for the same question.
        producer = _producer(social_db, tmp_path)
        committed = _commit(producer)

        assert committed.attention_schema_ref is not None
        row = social_db.fetchone(
            "SELECT predicted_next_focus FROM attention_schemas WHERE schema_id = ?",
            (committed.attention_schema_ref,),
        )

        assert row is not None
        assert row["predicted_next_focus"] == committed.predicted_next_focus
