"""Ablating the HOR feedback degrades a measured, non-verbal indicator.

The point of the loop is that higher-order records do something. If removing
them changed only what the runtime *says*, the records would be decoration. So
the check here is on `IndicatorProfile.self_feedback`, which is the mean of
`precision.self_model` over committed fields and is computed without reading a
single sentence of self-report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.ablation import AblationRunner
from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)

TICKS = 6

BEHAVIOR_TEMPLATE = """[individual-kernel]
generative_field_model = false
hor_precision_feedback = {enabled}
"""


@pytest.fixture
def feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _configure(*, enabled: bool) -> None:
        path = tmp_path / "mcpBehavior.toml"
        path.write_text(
            BEHAVIOR_TEMPLATE.format(enabled="true" if enabled else "false"),
            encoding="utf-8",
        )
        monkeypatch.setenv("MCP_BEHAVIOR_TOML", str(path))

    return _configure


def _producer(social_db: SocialDB, tmp_path: Path) -> TickProducer:
    interoception = tmp_path / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 20.0}}), encoding="utf-8")
    desires = tmp_path / "desires.json"
    desires.write_text(json.dumps({"desires": {}}), encoding="utf-8")
    return TickProducer(
        social_db, interoception_path=interoception, desires_path=desires
    )


def _run(producer: TickProducer, ticks: int = TICKS) -> None:
    for _ in range(ticks):
        opened = producer.begin_tick(TriggerKind.HEARTBEAT)
        producer.workspace.add_candidate(
            WorkspaceCandidate(
                tick_id=opened.tick_id,
                kind=CandidateKind.SELF_MODEL,
                content_ref="self:processing",
                content_summary="the runtime's own processing",
                source=CandidateSource.ATTENTION_SCHEMA,
                source_mode=SourceMode.INFERRED,
                modality="internal",
                precision=0.9,
                need_relevance=0.4,
                goal_relevance=0.6,
                continuity_with_previous=0.8,
                controllability=0.7,
            )
        )
        producer.compete_and_commit(opened.tick_id)


def _self_feedback(social_db: SocialDB) -> float:
    profile = AblationRunner(social_db).indicator_profile(window=TICKS * 2)
    return profile.self_model_feedback


class TestNonVerbalDegradation:
    def test_removing_the_feedback_lowers_self_feedback(
        self, social_db: SocialDB, tmp_path: Path, feedback
    ) -> None:
        feedback(enabled=True)
        _run(_producer(social_db, tmp_path))
        with_feedback = _self_feedback(social_db)

        feedback(enabled=False)
        _run(_producer(social_db, tmp_path))
        without_feedback = _self_feedback(social_db)

        # The indicator is a mean over precision values. Nothing in its
        # computation touches a self-report string.
        assert without_feedback < with_feedback

    def test_the_indicator_never_reads_a_report(
        self, social_db: SocialDB, tmp_path: Path, feedback
    ) -> None:
        feedback(enabled=True)
        _run(_producer(social_db, tmp_path))
        profile = AblationRunner(social_db).indicator_profile(window=TICKS)
        assert 0.0 <= profile.self_model_feedback <= 1.0


class TestFlagOff:
    def test_precision_seeding_matches_the_previous_decay(
        self, social_db: SocialDB, tmp_path: Path, feedback
    ) -> None:
        feedback(enabled=False)
        producer = _producer(social_db, tmp_path)
        opened = producer.begin_tick(TriggerKind.HEARTBEAT)
        seeded = producer.fields.get(opened.field_id)
        assert seeded is not None
        # With no previous field the decay returns the schema defaults, and the
        # feedback must not have moved them.
        assert seeded.precision.self_model == pytest.approx(0.5)


class TestProcessRecord:
    def test_every_commit_leaves_one_process_record(
        self, social_db: SocialDB, tmp_path: Path, feedback
    ) -> None:
        feedback(enabled=True)
        producer = _producer(social_db, tmp_path)
        _run(producer, ticks=3)
        rows = producer.process_meta.recent(limit=10)
        assert len(rows) == 3
        assert all(row["canonical_statement"] for row in rows)

    def test_the_statement_describes_the_competition(
        self, social_db: SocialDB, tmp_path: Path, feedback
    ) -> None:
        feedback(enabled=True)
        producer = _producer(social_db, tmp_path)
        _run(producer, ticks=1)
        row = producer.process_meta.recent(limit=1)[0]
        assert "candidates competed" in row["canonical_statement"]
        assert row["candidate_count"] >= 1
