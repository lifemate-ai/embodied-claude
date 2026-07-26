"""Coordinator for the generative prediction loop.

The only module the tick runtime imports from the generative layer. It wires
the count model, trajectory store, and experienced-transition store into four
moments of the action cycle:

- `imagine_for_intention`: at propose time, persist a ProtentionDistribution
  (action branch + no-action branch) and mark the modal action trajectory
  intended.
- `mark_enacted` / `mark_denied_trajectory`: at gate time, advance or return
  the linked trajectory. The boundary decision itself never reads this module.
- `record_transition_and_update`: at commit time, write one
  ExperiencedTransition (prediction errors are computed against the model
  BEFORE it learns) and count it into the stats exactly once.
- `reconcile_trajectories`: after the microtick commit, settle the enacted
  trajectory into observed / partially_observed / contradicted.

Everything is flag-guarded by `generative_enabled()` (mcpBehavior.toml,
re-read per call) and returns None / no-ops when disabled.
"""

from __future__ import annotations

import secrets

from social_core.db import SocialDB

from ._behavior import get_behavior
from .agency import ActionOutcome, AgencyStore, IntentionRecord
from .enacted_field import EnactedField
from .experienced_transition import ExperiencedTransition, ExperiencedTransitionStore
from .generative_model import (
    MAX_HORIZON,
    NO_ACTION,
    CountBasedGenerativeFieldModel,
    action_kind_of,
    focus_kind_of,
    outcome_bucket_of,
    parse_outcome_bucket,
    valence_delta_facet_of,
)
from .trajectory import (
    ImaginedTrajectory,
    ProtentionDistribution,
    TrajectoryStatus,
    TrajectoryStore,
    normalized_entropy,
)
from .workspace import SourceMode

OBSERVED_ERROR_THRESHOLD = 0.35
PARTIAL_ERROR_THRESHOLD = 0.70


def generative_enabled() -> bool:
    return bool(get_behavior("individual-kernel", "generative_field_model", True))


def configured_horizon() -> int:
    try:
        value = int(get_behavior("individual-kernel", "generative_rollout_horizon", 2))
    except (TypeError, ValueError):
        value = 2
    return max(1, min(value, MAX_HORIZON))


def _combined_uncertainty(step_uncertainties: list[float]) -> float:
    remaining_confidence = 1.0
    for value in step_uncertainties:
        remaining_confidence *= 1.0 - min(1.0, max(0.0, value))
    return min(1.0, max(0.0, 1.0 - remaining_confidence))


class FieldPredictionLoop:
    def __init__(
        self,
        db: SocialDB,
        *,
        agency: AgencyStore | None = None,
        owner_id: str = "self",
    ) -> None:
        self._db = db
        self.owner_id = owner_id
        self.agency = agency or AgencyStore(db)
        self.trajectories = TrajectoryStore(db)
        self.transitions = ExperiencedTransitionStore(db)
        self.model = CountBasedGenerativeFieldModel(db, owner_id=owner_id)

    def imagine_for_intention(
        self, field: EnactedField, intention: IntentionRecord
    ) -> ProtentionDistribution | None:
        if not generative_enabled():
            return None
        horizon = configured_horizon()
        action_kind = action_kind_of(intention.tool_name)
        signature = self.model.infer(field, action_kind=action_kind)
        forecast = self.model.forecast(signature)
        confidence = min(max(intention.confidence, 0.05), 0.95)

        top = forecast.predictions[:2]
        top_mass = sum(p.probability for p in top)
        distribution_id = f"prot_{secrets.token_urlsafe(12)}"
        trajectories: list[ImaginedTrajectory] = []
        for prediction in top:
            steps = self.model.rollout(
                field,
                action_kind=action_kind,
                horizon=horizon,
                first_bucket=prediction.bucket,
            )
            trajectories.append(
                ImaginedTrajectory(
                    distribution_id=distribution_id,
                    owner_id=self.owner_id,
                    field_id=field.field_id,
                    tick_id=field.tick_id,
                    action_ref=intention.action_id,
                    action_kind=action_kind,
                    context_signature=signature.key(),
                    horizon=horizon,
                    steps=steps,
                    probability=confidence * (prediction.probability / top_mass),
                    uncertainty=_combined_uncertainty([s.uncertainty for s in steps]),
                )
            )

        no_action_signature = self.model.infer(field, action_kind=NO_ACTION)
        no_action_steps = self.model.rollout(
            field, action_kind=NO_ACTION, horizon=horizon
        )
        trajectories.append(
            ImaginedTrajectory(
                distribution_id=distribution_id,
                owner_id=self.owner_id,
                field_id=field.field_id,
                tick_id=field.tick_id,
                action_ref=None,
                action_kind=NO_ACTION,
                context_signature=no_action_signature.key(),
                horizon=horizon,
                steps=no_action_steps,
                probability=1.0 - confidence,
                uncertainty=_combined_uncertainty(
                    [s.uncertainty for s in no_action_steps]
                ),
            )
        )

        distribution = ProtentionDistribution(
            distribution_id=distribution_id,
            owner_id=self.owner_id,
            field_id=field.field_id,
            tick_id=field.tick_id,
            action_ref=intention.action_id,
            trajectories=trajectories,
            entropy=normalized_entropy([t.probability for t in trajectories]),
        )
        self.trajectories.create_distribution(distribution)
        self.trajectories.transition(
            trajectories[0].trajectory_id,
            TrajectoryStatus.INTENDED,
            reason=f"linked to intention {intention.action_id}",
        )
        return distribution

    def mark_enacted(self, action_id: str) -> None:
        if not generative_enabled():
            return
        trajectory = self.trajectories.find_active_for_action(action_id)
        if trajectory is not None and trajectory.status is TrajectoryStatus.INTENDED:
            self.trajectories.transition(
                trajectory.trajectory_id,
                TrajectoryStatus.ENACTED,
                reason="tool gate allowed",
            )

    def mark_denied_trajectory(self, action_id: str) -> None:
        if not generative_enabled():
            return
        trajectory = self.trajectories.find_active_for_action(action_id)
        if trajectory is not None and trajectory.status is TrajectoryStatus.INTENDED:
            self.trajectories.transition(
                trajectory.trajectory_id,
                TrajectoryStatus.IMAGINED,
                reason="boundary denied",
            )

    def record_transition_and_update(
        self,
        *,
        previous: EnactedField | None,
        committed: EnactedField,
        action_id: str | None,
    ) -> ExperiencedTransition | None:
        if not generative_enabled():
            return None
        if previous is None:
            return None

        intention = self.agency.get(action_id) if action_id else None
        outcome: ActionOutcome | None = (
            self.agency.outcome_for_action(action_id) if action_id else None
        )
        action_kind = (
            action_kind_of(intention.tool_name) if intention is not None else NO_ACTION
        )
        signature = self.model.infer(previous, action_kind=action_kind)

        valence_before = previous.interoception.valence
        valence_after = committed.interoception.valence
        valence_delta = valence_after - valence_before
        next_focus_kind = focus_kind_of(committed.focal_content_ref)
        success = outcome.success if outcome is not None else None
        latency_ms = outcome.latency_ms if outcome is not None else None
        bucket = outcome_bucket_of(
            success=success,
            latency_ms=latency_ms,
            next_focus_kind=next_focus_kind,
            valence_delta=valence_delta,
        )
        probability_of_observed, _, _ = self.model.probability_of(signature, bucket)

        trajectory = (
            self.trajectories.find_active_for_action(action_id) if action_id else None
        )
        if trajectory is not None and trajectory.status is TrajectoryStatus.INTENDED:
            # Still INTENDED at commit time means the gate never allowed it
            # (an allowed action would be ENACTED here), so the imagined
            # branch was never executed.
            trajectory = self.trajectories.transition(
                trajectory.trajectory_id,
                TrajectoryStatus.IMAGINED,
                reason="intention expired without execution",
            )

        errors: dict[str, float] = {}
        if outcome is not None:
            for channel, value in outcome.mismatch_vector.items():
                errors[channel] = min(1.0, max(0.0, float(value)))
        if trajectory is not None and trajectory.steps:
            predicted = parse_outcome_bucket(trajectory.steps[0].outcome_bucket)
            errors["focus_kind_error"] = (
                0.0 if predicted["next_focus_kind"] == next_focus_kind else 1.0
            )
            errors["valence_delta_error"] = (
                0.0
                if predicted["valence_delta"] == valence_delta_facet_of(valence_delta)
                else 1.0
            )
        mean_error = (
            sum(errors.values()) / len(errors)
            if errors
            else min(1.0, max(0.0, 1.0 - probability_of_observed))
        )
        errors["probability_of_observed"] = probability_of_observed

        if committed.reality_mode is SourceMode.LIVE:
            source_mode = SourceMode.LIVE
        elif committed.reality_mode is SourceMode.MIXED:
            source_mode = SourceMode.MIXED
        else:
            source_mode = SourceMode.INFERRED

        transition = ExperiencedTransition(
            owner_id=committed.owner_id,
            previous_field_id=previous.field_id,
            next_field_id=committed.field_id,
            previous_tick_id=previous.tick_id,
            next_tick_id=committed.tick_id,
            action_ref=action_id,
            outcome_ref=outcome.outcome_id if outcome is not None else None,
            intended_trajectory_id=(
                trajectory.trajectory_id
                if trajectory is not None
                and trajectory.status
                in {TrajectoryStatus.INTENDED, TrajectoryStatus.ENACTED}
                else None
            ),
            distribution_id=(
                trajectory.distribution_id if trajectory is not None else None
            ),
            context_signature=signature.key(),
            action_kind=action_kind,
            outcome_bucket=bucket,
            observed_external={
                "focal_content_ref": committed.focal_content_ref,
                "focus_kind": next_focus_kind,
                "trigger_kind": committed.trigger_kind.value,
                "reality_score": committed.reality_score,
            },
            observed_internal={
                "valence": valence_after,
                "arousal": committed.interoception.arousal,
                "uncertainty": committed.interoception.uncertainty,
                "controllability": committed.interoception.controllability,
            },
            observed_social={"person_id": committed.person_id},
            prediction_errors=errors,
            mean_prediction_error=min(1.0, max(0.0, mean_error)),
            valence_before=valence_before,
            valence_after=valence_after,
            valence_change=valence_delta,
            arousal_before=previous.interoception.arousal,
            arousal_after=committed.interoception.arousal,
            agency_confidence=(
                min(
                    1.0,
                    max(
                        0.0,
                        1.0
                        - (
                            sum(outcome.mismatch_vector.values())
                            / len(outcome.mismatch_vector)
                        ),
                    ),
                )
                if outcome is not None and outcome.mismatch_vector
                else 0.5
            ),
            ownership_confidence=(
                outcome.ownership_score
                if outcome is not None
                else previous.agency_state.ownership_score
            ),
            success=success,
            latency_ms=latency_ms,
            expected_latency_ms=(
                intention.expected_latency_ms if intention is not None else None
            ),
            source_mode=source_mode,
        )
        stored, created = self.transitions.record(transition)
        if created:
            self.model.update(stored)
        return stored

    def reconcile_trajectories(self, *, action_id: str, next_field_id: str) -> None:
        if not generative_enabled():
            return
        trajectory = self.trajectories.find_active_for_action(action_id)
        if trajectory is None or trajectory.status is not TrajectoryStatus.ENACTED:
            return
        transition = self.transitions.get_by_next_field(next_field_id)
        if transition is None:
            return
        error = transition.mean_prediction_error
        predicted_success = parse_outcome_bucket(trajectory.steps[0].outcome_bucket)[
            "success"
        ]
        actual_success = parse_outcome_bucket(transition.outcome_bucket)["success"]
        if actual_success == "fail" and predicted_success != "fail":
            target = TrajectoryStatus.CONTRADICTED
        elif error < OBSERVED_ERROR_THRESHOLD:
            target = TrajectoryStatus.OBSERVED
        elif error <= PARTIAL_ERROR_THRESHOLD:
            target = TrajectoryStatus.PARTIALLY_OBSERVED
        else:
            target = TrajectoryStatus.CONTRADICTED
        self.trajectories.transition(
            trajectory.trajectory_id,
            target,
            reason=f"mean prediction error {error:.3f}",
        )
