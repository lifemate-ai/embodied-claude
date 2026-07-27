"""Affective modulation of workspace competition.

Until now valence was computed every tick and then read by nothing: it appeared
in the field surface but changed no decision. This module gives it one narrow,
auditable path of influence — it scales the weights that decide which candidate
reaches the committed field.

The shape follows two well-attested effects. Negative affect narrows attention
toward pressing needs; positive affect widens it toward exploration. So a bad
mood raises the weight on need relevance and lowers the weight on expected
information gain, and a good mood raises information gain alone — feeling well
is a reason to look further afield, not a reason to want more.

What is deliberately *not* modulated:

- Ignition and conflict thresholds. Whether a field ignites at all is structural;
  letting mood move it would make the runtime's own reportability mood-dependent.
- The boundary gate. What is permitted must never depend on how it feels. The
  gate reads none of this, and a test asserts that its decisions are identical
  under any affect.

Arousal is accepted and recorded but does not yet modulate anything; the
inverted-U it is supposed to produce needs calibration data this runtime has not
collected. It is in the signature so that adding it later is not a schema change.

Everything here is a pure function of its arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

from individual_kernel_mcp.workspace import CompetitionWeights

# How far affect may move a weight, as a fraction of its base value. Kept small:
# this is a bias on an existing competition, not a new controller.
EXPLORATION_GAIN = 0.5
NEED_URGENCY_GAIN = 0.5


@dataclass(frozen=True)
class AffectState:
    """The affective read used for modulation, validated at construction."""

    valence: float
    arousal: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.valence <= 1.0:
            raise ValueError(f"valence out of range: {self.valence}")
        if not 0.0 <= self.arousal <= 1.0:
            raise ValueError(f"arousal out of range: {self.arousal}")

    @classmethod
    def neutral(cls) -> "AffectState":
        return cls(valence=0.0, arousal=0.0)

    @property
    def is_neutral(self) -> bool:
        return self.valence == 0.0


def modulate_weights(
    base: CompetitionWeights,
    affect: AffectState,
) -> CompetitionWeights:
    """Scale competition weights by affect, leaving thresholds untouched.

    At neutral valence the returned weights are equal to ``base``, so enabling
    this coupling changes nothing until affect actually moves.
    """

    if affect.is_neutral:
        return base

    # Widening: exploration rises with good affect and falls with bad.
    information = base.information * (1.0 + EXPLORATION_GAIN * affect.valence)
    # Narrowing: needs press harder when affect is bad, and are not suppressed
    # when it is good. Comfort should not make a real need quieter.
    urgency = max(0.0, -affect.valence)
    need = base.need * (1.0 + NEED_URGENCY_GAIN * urgency)

    return base.model_copy(
        update={
            "information": max(0.0, information),
            "need": max(0.0, need),
        }
    )


def modulation_trace(base: CompetitionWeights, affect: AffectState) -> dict[str, float]:
    """The record written into the tick's epistemic trace.

    Weights are frozen into each candidate's score at insert time, so the only
    way to audit a past competition is to know which weights were in force. This
    is that record.
    """

    applied = modulate_weights(base, affect)
    return {
        "valence": affect.valence,
        "arousal": affect.arousal,
        "need": applied.need,
        "information": applied.information,
        "base_need": base.need,
        "base_information": base.information,
    }
