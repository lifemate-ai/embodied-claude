# Allostatic Valence Design

Status: shipped 2026-07-27 on the `feat/allostatic-valence` branch, behind
`[individual-kernel] allostatic_valence`, which defaults to **off**. Scope: the
body state only. Action selection, workspace competition, the `<current_field>`
surface, and the boundary gate are unchanged.

Claim policy: this layer computes a functional affect variable. Nothing in it is
a phenomenal claim.

## Two defects in the previous body state

**Valence could not be positive.** It was an EMA whose target was
`-0.6 * max(discomfort)`, so the upper bound was exactly zero. A run of
well-predicted, successful actions moved it toward 0 and no further; nothing
could register as going well.

**The needs had stopped moving.** Desire levels came from a file written by an
external updater on a 5-minute timer. That timer had been dead since
2026-05-13, so every level was frozen at its last written value, which in turn
pinned valence at -0.30 for two months. The failure was invisible: the crontab
entry existed, and both of its failure stages (a missing `uv` on cron's PATH,
then a working directory that no longer existed after the repository moved)
produced either logs nobody read or no logs at all.

A third, smaller defect: `controllability` was `0.45 + 0.25 * (1 - discomfort)`,
a restatement of discomfort rather than a measure of whether actions were
actually landing.

## Desire projection: the kernel owns the clock

Levels grow linearly, `level = elapsed / satisfaction_hours`, so a recorded
level implies when the need was last met, and the same ramp can be evaluated at
any later instant. `project_desires` re-derives every level for *now* from the
recorded snapshot plus elapsed time.

The external writer is therefore demoted from clock to corrector: when it runs,
it supplies a fresh, memory-grounded reading; when it does not, the kernel keeps
moving on its own. A stalled writer costs accuracy, not motion.

Edge cases pinned by tests: a level already at 1.0 stays saturated (it implies
no last-satisfied instant); a 70-day stall saturates instead of overflowing;
backwards clock skew never lowers a level below what was recorded; an
unparsable timestamp is treated as "just written" rather than extrapolated from
a guess.

Set points are adjusted for the time of day: social and outward-looking needs
settle overnight, and `identity_coherence` does not, because it is wanted
equally at every hour. Its set point is 0.9 while the others sit at 0.2-0.3,
so it is a need to *stay* coherent rather than one that accumulates by neglect.

## Valence composition

```
appetitive = max(0, expected_valence_delta) * measured_controllability
aversive   = max(0, -expected_valence_delta)
           + 0.55 * mean_discomfort
           + 0.25 * uncertainty
           + 0.20 * unresolved_prediction_error
valence    = clamp(appetitive - aversive, -1, 1)
```

Expected improvement is scaled by measured control: an improvement nobody can
bring about is not good news. Expected worsening is *not* discounted that way,
since harm one cannot avert is still aversive.

`expected_valence_delta` comes from the count model shipped in PR #108:

```
E[dvalence | context] = sum over outcome buckets of
                        P(bucket | context) * mean valence change in that bucket
```

This is the point where the previous phase pays off. The transitions the model
learned from experience become the estimate of whether the current situation is
going anywhere good. With no history the forecast falls back to the uniform
prior, whose buckets carry small symmetric deltas, so the estimate sits near
zero rather than inventing an expectation.

`controllability` is now the measured `ownership_score` of the previous field,
used only once an intention has actually been registered and closed; before
that there is nothing measured, and the neutral midpoint stands in.

## Integration

`TickProducer._read_interoception` branches on the flag. The legacy arithmetic
is preserved verbatim as `_legacy_interoception` and is reproduced bit-for-bit
when the flag is off (asserted against the exact constants -0.105, 0.575, and
the recorded discomfort map).

The allostatic branch is wrapped so a prediction or storage fault falls back to
the legacy path instead of blocking a tick: a body state must always be
producible. This tolerance has a cost that showed up during development, and is
worth recording. A `TypeError` inside the new branch (`utc_now()` returns an ISO
string, not a `datetime`) was swallowed silently, and the flag simply appeared
not to work. The fallback is right for production and blinding during
development; when this branch misbehaves, remove the `except` before theorising.

No migration is required. Every input already existed:
`generative_transition_stats` for expected improvement, `agency_state` for
measured control, `experienced_transitions` for unresolved error, and the desire
file for needs.

## Why the flag defaults to off

The generative layer in PR #108 shipped enabled because it only *added* rows.
This change rewrites a value the whole runtime reads, including workspace
salience and the negative-exposure counter. It ships off so the live smoke can
be run deliberately: enable it, take a few ticks, compare the reported body
state against the same period under the legacy rule, then decide.

## What later phases take from here

- PR-3 (valence coupling) reads `AllostaticVariable`'s separated appetitive and
  aversive terms rather than the scalar, so competition weights and recall can
  be modulated by *why* affect is where it is.
- `as_snapshot()` is shaped for `experienced_transitions.allostatic_snapshot_json`,
  which already exists from migration 009.
- The organism daemon (PR-7) can retire the external desire writer entirely; the
  projection already covers the interval between writes.
