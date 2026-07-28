"""The same content, arriving by three routes, must not leave the same trace.

If being told a contingency trained it as strongly as undergoing it, the
recorded provenance would be cosmetic. These tests run identical information
down each route from identical history and check what diverges.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from social_core.db import SocialDB

from individual_kernel_mcp.enacted_field import TriggerKind
from individual_kernel_mcp.experienced_transition import (
    ExperiencedTransition,
    ExperiencedTransitionStore,
)
from individual_kernel_mcp.fork import (
    KNOWLEDGE_SOURCES,
    fork_history,
    info_content_hash,
)
from individual_kernel_mcp.generative_model import CountBasedGenerativeFieldModel
from individual_kernel_mcp.tick import TickProducer
from individual_kernel_mcp.workspace import (
    CandidateKind,
    CandidateSource,
    SourceMode,
    WorkspaceCandidate,
)

CONTENT = "panning left brings the window into view"
SIGNATURE = "desire|user_prompt|look_outside|neu|low|tool:look_left"


def _transition(
    *, route: str, field_id: str = "f", tick_id: str = "t"
) -> ExperiencedTransition:
    return ExperiencedTransition(
        next_field_id=field_id,
        next_tick_id=tick_id,
        context_signature=SIGNATURE,
        action_kind="tool:look_left",
        outcome_bucket="ok/short/percept/=",
        source_mode=SourceMode.LIVE,
        knowledge_source=route,
        info_content_hash=info_content_hash(CONTENT),
        mean_prediction_error=0.2,
        valence_before=0.0,
        valence_after=0.1,
        valence_change=0.1,
        arousal_before=0.2,
        arousal_after=0.2,
        agency_confidence=0.8,
        ownership_confidence=0.8,
    )


def _observations(db: SocialDB) -> float:
    row = db.fetchone(
        "SELECT COALESCE(SUM(observation_count), 0.0) AS total "
        "FROM generative_transition_stats",
        (),
    )
    return float(row["total"]) if row else 0.0


@contextmanager
def _runtime_recording_paused() -> Iterator[None]:
    """Commit a field without the prediction loop claiming it.

    Every commit that has a predecessor is recorded by the runtime as
    `experienced`, and the store is idempotent per field, so a test that wants
    to choose the route has to commit its field with that recording switched
    off. The behaviour flag already allows it: it is re-read on every call.
    """
    path = Path(os.environ["MCP_BEHAVIOR_TOML"])
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace(
            "generative_field_model = true", "generative_field_model = false"
        ),
        encoding="utf-8",
    )
    try:
        yield
    finally:
        path.write_text(original, encoding="utf-8")


def _committed_field(db: SocialDB, tmp_path: Path) -> tuple[str, str]:
    """Commit a real tick and return its (field_id, tick_id).

    Transitions carry foreign keys into the field and frame tables, so a
    synthetic id cannot be stored at all. Running against real committed
    history is also what makes the comparison a fair one.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    interoception = tmp_path / "interoception.json"
    interoception.write_text(json.dumps({"now": {"arousal": 20.0}}), encoding="utf-8")
    desires = tmp_path / "desires.json"
    desires.write_text(json.dumps({"desires": {}}), encoding="utf-8")
    producer = TickProducer(db, interoception_path=interoception, desires_path=desires)
    opened = producer.begin_tick(TriggerKind.USER_PROMPT)
    producer.workspace.add_candidate(
        WorkspaceCandidate(
            tick_id=opened.tick_id,
            kind=CandidateKind.GOAL,
            content_ref="desire:look_outside",
            content_summary="need look_outside",
            source=CandidateSource.DESIRE,
            source_mode=SourceMode.INFERRED,
            modality="internal",
            precision=1.0,
            need_relevance=1.0,
            goal_relevance=1.0,
            continuity_with_previous=1.0,
            controllability=1.0,
        )
    )
    field = producer.compete_and_commit(opened.tick_id).field
    return field.field_id, field.tick_id


def _ingest(db: SocialDB, tmp_path: Path, *, route: str) -> bool:
    """Store one transition and offer it to the model, as the runtime would.

    The store step matters: `update` guards on the stored row's `applied_at`,
    so a transition that was never recorded can never be learned from either.
    """
    with _runtime_recording_paused():
        field_id, tick_id = _committed_field(db, tmp_path)
    transition = _transition(route=route, field_id=field_id, tick_id=tick_id)
    ExperiencedTransitionStore(db).record(transition)
    return CountBasedGenerativeFieldModel(db).update(transition)


class TestRoutesDiverge:
    def test_experience_trains_the_model(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        before = _observations(social_db)
        assert _ingest(social_db, tmp_path, route="experienced")
        assert _observations(social_db) > before

    @pytest.mark.parametrize("route", ["told", "imagined", "replayed"])
    def test_other_routes_do_not_train_the_model(
        self, social_db: SocialDB, tmp_path: Path, route: str
    ) -> None:
        before = _observations(social_db)
        assert not _ingest(social_db, tmp_path, route=route)
        assert _observations(social_db) == before

    def test_identical_content_diverges_only_by_route(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        told = _transition(route="told")
        lived = _transition(route="experienced")
        # Same information by construction, so anything downstream that differs
        # is attributable to the route.
        assert told.info_content_hash == lived.info_content_hash
        assert told.context_signature == lived.context_signature

        assert _ingest(social_db, tmp_path, route="told") is False
        assert _ingest(social_db, tmp_path, route="experienced") is True


class TestRouteValidation:
    @pytest.mark.parametrize("route", list(KNOWLEDGE_SOURCES))
    def test_every_declared_route_is_accepted(self, route: str) -> None:
        assert _transition(route=route).knowledge_source == route

    def test_an_undeclared_route_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _transition(route="dreamt")

    def test_imagined_content_still_cannot_claim_a_live_source_mode(self) -> None:
        # The pre-existing invariant is untouched: whatever the route, the
        # source mode may not be imagined or remembered.
        with pytest.raises(ValueError):
            ExperiencedTransition(
                next_field_id="f",
                next_tick_id="t",
                context_signature=SIGNATURE,
                action_kind="tool:look_left",
                outcome_bucket="ok/short/percept/=",
                source_mode=SourceMode.IMAGINED,
                knowledge_source="imagined",
                mean_prediction_error=0.2,
                valence_before=0.0,
                valence_after=0.0,
                valence_change=0.0,
                arousal_before=0.2,
                arousal_after=0.2,
                agency_confidence=0.5,
                ownership_confidence=0.5,
            )


class TestFork:
    def test_a_fork_starts_from_the_same_history(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        _ingest(social_db, tmp_path, route="experienced")
        seeded = _observations(social_db)
        assert seeded > 0.0
        with fork_history(social_db, tmp_path / "forks", label="told") as arm:
            assert _observations(arm.db) == seeded

    def test_writing_in_one_arm_leaves_the_source_untouched(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        before = _observations(social_db)
        with fork_history(social_db, tmp_path / "forks", label="lived") as arm:
            _ingest(arm.db, tmp_path / "arm", route="experienced")
            assert _observations(arm.db) > before
        assert _observations(social_db) == before

    def test_two_arms_diverge_by_route_alone(
        self, social_db: SocialDB, tmp_path: Path
    ) -> None:
        _ingest(social_db, tmp_path / "seed", route="experienced")
        seeded = _observations(social_db)
        assert seeded > 0.0
        directory = tmp_path / "forks"
        with (
            fork_history(social_db, directory, label="lived") as lived,
            fork_history(social_db, directory, label="told") as told,
        ):
            assert lived.fork_id != told.fork_id
            # Both arms demonstrably start from the same non-empty history.
            assert _observations(lived.db) == seeded
            assert _observations(told.db) == seeded
            _ingest(lived.db, tmp_path / "a", route="experienced")
            _ingest(told.db, tmp_path / "b", route="told")
            # Identical content, identical starting history, different route.
            assert _observations(lived.db) > seeded
            assert _observations(told.db) == seeded


class TestContentHash:
    def test_same_content_hashes_alike(self) -> None:
        assert info_content_hash(CONTENT) == info_content_hash(CONTENT)

    def test_different_content_hashes_differently(self) -> None:
        assert info_content_hash(CONTENT) != info_content_hash(CONTENT + ".")
