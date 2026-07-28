# Body Agency Design

Status: shipped 2026-07-28 on the `feat/body-contingency` branch. Scope: the
`exclusive_causal_fit` term of the agency assessment, and a new ledger of
reafference verdicts. Action selection, workspace competition, the
`<current_field>` surface, and the boundary gate are unchanged.

Claim policy: this compares one measured quantity against another. Nothing in
it is a phenomenal claim.

## The defect

`AgencyStore.close` set `exclusive_causal_fit = 1.0 if success else 0.65`. The
term is supposed to answer "was this change caused by me and not by something
else", and it was answering "did the tool call return without raising". Those
are different questions, and the difference is exactly the interesting case: if
someone else turns the camera, or the camera drifts, or the motor stalls while
the API still reports success, the old rule awards full causal credit.

The term carries weight 0.15 in the ownership score, so the error was small but
systematic, and it made `ownership_score` unusable as evidence about the body.

## The reafference test

For a commanded body change, compare the observed change against it:

```
direction = (cos(commanded, observed) + 1) / 2
magnitude = 1 while |observed/commanded - 1| <= 0.35, falling to 0 at 2x
timing    = exp(-|observed_latency - expected_latency| / (2 * expected))
score     = 0.5*direction + 0.3*magnitude + 0.2*timing
```

Direction is weighted highest because moving the *wrong way* is stronger
evidence of an external cause than moving the right way by the wrong amount.

The score is then capped by a verdict, so a favourable component cannot rescue
an unfavourable classification:

| verdict | when | ceiling |
|---|---|---|
| `self_caused` | commanded and observed agree in direction | none |
| `inverted` | observed opposes the command | 0.10 |
| `externally_caused` | the body moved with nothing commanded | 0.10 |
| `unresponsive` | commanded, but the body did not move | 0.15 |
| `no_change` | nothing commanded and nothing moved | 0.50 (no evidence) |
| `unverified` | no observation, or a tool that does not move the body | falls back |

Movement below 1.0 degree counts as no movement: that is the encoder noise floor
on the PTZ hardware, and treating jitter as motion would manufacture
`externally_caused` verdicts out of a stationary camera.

## What stays the same without a body channel

Most actions are not body actions. `commanded_delta_from_tool` returns None for
anything outside the camera tool set, which yields `unverified`, which restores
the old `1.0 / 0.65` heuristic exactly. Two tests assert this, so the change is
additive for every non-camera action rather than a silent re-scoring of the
whole history.

## Where observations come from

The kernel does not import the camera package. `AgencyStore.close` takes an
optional `BodyObservation` (a before pose, an after pose, and a latency), and
the caller supplies it. `source` distinguishes `measured` from `declared` so a
later audit can discount poses that were asserted rather than read back from
`get_hw_position()`.

This keeps the dependency pointing the right way and leaves the verdict
reproducible from the stored row alone.

## Ledger

Migration 010 adds `body_contingencies`: one row per action at most, enforced by
a partial unique index on `action_id`. Closing an action twice returns the
existing row rather than writing a second one, because the outcome it justified
was also written once. A storage fault while recording is swallowed: the ledger
is an audit trail, not a precondition for closing an action, and a failed insert
must not strand a pending intention.

## What was deliberately left out

**Replacing the mismatch tokenizer with embedding cosine.** The roadmap called
for this here. Hooks are a fresh subprocess per call, so an embedding model
would be loaded on every tool invocation; the target for hook latency is under
100ms and a transformer load is two orders of magnitude past that. The
CJK-aware token overlap from PR-1 stays until an embedding can live in a
resident process, which is what the PR-7 daemon provides.

**Arousal-weighted timing.** The timing term uses the stated expectation only.
An unknown expectation scores a neutral 0.5 rather than guessing.

## What later phases take from here

The ledger is the substrate for a self/other discrimination measurement: the
rate of `externally_caused` verdicts over a window is a direct, non-verbal
indicator, and it belongs in the indicator profile once enough rows exist to
make the rate meaningful.
