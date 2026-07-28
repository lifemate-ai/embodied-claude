# Higher-Order Feedback and Its Ablation

Status: shipped 2026-07-28 on the `feat/process-meta-hor` branch, behind
`[individual-kernel] hor_precision_feedback`, which defaults to **off** pending
a live check. Scope: the precision a tick starts with, and a new ledger of
process-level self representations. Action selection, workspace competition,
the `<current_field>` surface, and the boundary gate are unchanged.

Claim policy: this couples one recorded functional variable to another. Nothing
in it is a phenomenal claim.

## The defect

The runtime ends each tick by recording a higher-order representation -- most
often `attending`, asserting what it was attending to. Those rows went into
`hor_records` and no code read them back. A HOR could be present, absent, or
wrong and every other quantity in the system came out identical.

That makes the record decoration. It is also the thing that would make an
ablation experiment vacuous: removing something inert cannot degrade anything.

## What the feedback does

`_seed_precision` decays the previous tick's precision exactly as before, then
adds a bias derived from the last three higher-order records:

```
contribution = 0.15 * confidence      per record
channel      = seeing -> extero, feeling -> intero, remembering -> mnemonic,
               attending / intending / wanting -> self_model
cap          = 0.20                   per channel, however many records
```

The cap is the point of the design: repeating an assertion cannot substitute
for evidence. Twenty confident `attending` records raise `self_model` by the
same 0.20 that two do.

`attending`, `intending` and `wanting` all land on `self_model` because they are
about the runtime's own processing rather than about a sense.

## The ablation, on a non-verbal indicator

The requirement this design has to meet is that removing the feedback degrades
something that is *not* a self-report. `IndicatorProfile.self_model_feedback`
qualifies: it is the mean of `precision.self_model` over recent committed
fields, computed without reading a single sentence the runtime produced.

`tests/test_hor_ablation.py::TestNonVerbalDegradation` runs the same six-tick
workload twice, once with the flag on and once off, and asserts the indicator
is strictly lower in the ablated run. It passes. That is the whole claim: the
higher-order record has a measurable downstream, and the measurement does not
go through language.

The magnitude is small by construction -- the cap is 0.20 on one channel -- so
this is a demonstration that the coupling exists, not that it dominates.

## ProcessMetaRepresentation

Migration 011 adds `process_meta_representations`: one row per committed tick,
recording how the field was produced rather than what it was about. Candidate
count, competition margin and entropy, whether it ignited, whether it was
contested, whether an intention was registered, and the precision bias the HORs
contributed.

It carries a `canonical_statement()` -- a sentence like "11 candidates competed
and it settled; the choice was contested by a margin of 0.195". The statement
lives next to the numbers it summarises so the two cannot drift apart. This is
the object a report about one's own processing would have to be a report *of*;
without it, such a report has no referent inside the system.

The write is wrapped: a storage fault must not roll back a field that has
already been committed.

## The live measurement, and why the flag now ships on

Run 2026-07-28 against the live history. Two arms, each a fresh copy of
`~/.claude/sociality/social.db` taken through the sqlite backup API, differing
only in `hor_precision_feedback`. Same candidate, same interoception and desire
inputs, one committed tick each.

| channel | off | on | delta |
| --- | --- | --- | --- |
| extero | 0.307692 | 0.307692 | +0.000000 |
| intero | 0.307692 | 0.307692 | +0.000000 |
| mnemonic | 0.307692 | 0.307692 | +0.000000 |
| self_model | 0.538208 | 0.668208 | **+0.130000** |
| social | 0.307692 | 0.307692 | +0.000000 |

The winner was `desire:identity_coherence` in both arms.

Three things make this the measurement the flag was waiting for. The live
history's higher-order records all assert modes that map to `self_model`, so
exactly one channel moves and the other four are bit-identical. The bump is
0.130, below the 0.20 cap, so the value is what the records actually say rather
than a saturated constant. And the selection is unchanged, which is expected
rather than lucky: competition scores read the candidate's own precision, not
the field's `PrecisionState`.

The blast radius is therefore the `self_model` channel, its own carry-forward
(`0.65 * previous + 0.25 * attention_intensity`, so a per-tick bias of 0.130
settles around +0.24 rather than accumulating), and `IndicatorProfile`
`self_model_feedback`, which is a mean over that channel. The boundary gate
reads none of it. In short, the flag changes a measured quantity and nothing
that acts, which is exactly what it was designed to do.

## Why the flag shipped off until then

The two flags shipped in the previous PR were switched on only after a live
check showed what they did to the running system. The same standard applies
here: this one changes the precision every subsequent tick starts from, and the
live desire and HOR history is nothing like the fixture. It goes on after the
same kind of before-and-after measurement, not before.

## What later phases take from here

Report independence is currently a stub. With process records available, it
becomes measurable properly: hold the process record fixed, vary the surface,
and check that non-verbal indicators do not move (surface-blind invariance);
then vary the process record and check the report follows (surface fidelity).
That needs the fork machinery from PR-6 to run both arms against identical
history.
