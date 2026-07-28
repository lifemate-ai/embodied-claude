# Valence Coupling Design

Status: shipped 2026-07-27 on the `feat/valence-coupling` branch, behind
`[individual-kernel] valence_coupling`, which defaults to **on** (verified live 2026-07-28). Scope: which
candidate wins workspace competition. Ignition thresholds, the `<current_field>`
surface, and the boundary gate are unchanged.

Claim policy: this couples one functional variable to another. Nothing in it is
a phenomenal claim.

## The defect

PR-2 made valence a real quantity that can move in both directions. It still
changed nothing: `CompetitionWeights` had no affective term, so the body state
was computed every tick, written into the field surface, and read by no
decision. An affect that influences nothing is a display, not a state.

## What affect now does

```
information = base.information * (1 + 0.5 * valence)
need        = base.need        * (1 + 0.5 * max(0, -valence))
```

Negative affect raises the weight on need relevance and lowers the weight on
expected information gain: attention narrows onto what is pressing. Positive
affect raises information gain alone: feeling well is a reason to look further
afield, not a reason to want more. At exactly zero valence the returned weights
are the base weights, so enabling the coupling changes nothing until affect
actually moves.

The demonstration is two bids that differ only in what they offer -- a pressing
need with nothing to learn, and something novel that no need is pushing. Under
valence -0.9 the need wins; under +0.9 the novelty wins; at 0 the result equals
an engine constructed with no affect at all.

## What affect is not allowed to do

**The boundary gate.** What is permitted must never depend on how it feels.
`tests/test_gate_affect_independence.py` asserts that the same outward action
receives an identical `ToolGateDecision` under the best and worst affect the
runtime can represent, that a hash mismatch is refused under both, and that
read-only tools pass under both. A good mood must not excuse acting on something
never declared.

**Ignition and conflict thresholds.** Whether a field ignites at all is
structural. Letting mood move that threshold would make the runtime's own
reportability mood-dependent, which would corrupt every downstream measurement
that reads `ignited`.

**Arousal.** Accepted and recorded, but it modulates nothing yet. The
inverted-U it is supposed to produce needs calibration data this runtime has not
collected, and assuming a curve without data is worse than leaving the term at
zero. It sits in the signature so adding it later is not a schema change.

## Where the modulation is applied

Scores are computed in `add_candidate` and frozen into the row; `compete` ranks
by the stored score. So affect is applied at insert time, and `begin_tick` sets
`workspace.affect` from the body state before any candidate for that tick is
added.

Re-scoring at competition time was rejected. It would leave the stored score
disagreeing with the ranking that actually happened, and a past competition
could no longer be reproduced from what was recorded. `WorkspaceEngine.affect_trace()`
reports the applied weights for exactly this reason: frozen scores can only be
audited against the weights that produced them.

## Module layout

`valence_coupling` imports `CompetitionWeights` from `workspace`, so `workspace`
defers its import of `valence_coupling` into the three methods that need it and
declares the type under `TYPE_CHECKING`. Splitting `CompetitionWeights` into a
third module would also work; it was not worth indirecting an import path used
throughout the package for a two-symbol cycle.

## Two flags, deliberately separate

`allostatic_valence` decides how the body state is computed. `valence_coupling`
decides whether that state may influence what reaches the field. Either can be
enabled without the other, which is what makes the intervention experiment
possible: run the same workload with coupling off and on, and the difference is
attributable to the coupling alone.

## What later phases take from here

- Memory recall and encoding are the next targets: mood-congruent retrieval is
  well attested, and `AffectState` is already the right input for it.
- Learning rate modulation belongs with the generative model, not here; the
  model owns its own update path.
- When `AllostaticVariable`'s separated appetitive and aversive terms are
  threaded through instead of the scalar, weights can be modulated by *why*
  affect is where it is, not only by where it is.
