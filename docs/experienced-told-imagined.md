# Experienced, Told, Imagined

Status: shipped 2026-07-28 on the `feat/experienced-told-imagined` branch. Scope:
which route a transition arrived by, and which routes may train the sensorimotor
model. Action selection, workspace competition, the `<current_field>` surface,
and the boundary gate are unchanged.

Claim policy: this measures how records differ by provenance. Nothing in it is a
phenomenal claim.

## The question

Does the runtime distinguish knowing because it happened from knowing because it
was told and from knowing because it was imagined?

Asking cannot settle it. A system with no such distinction can still produce the
sentence "I know that because I did it" -- the sentence is cheap, and its
presence is evidence about the surface rather than about the machinery. The
question is only settled by running identical information down each route from
identical history and looking at what actually diverges.

## Four routes, one gate

`ExperiencedTransition.knowledge_source` takes one of four values, declared in
`fork.py` so that neither the schema nor the model has to import the other:

```
experienced   underwent it
told          someone reported it
imagined      rolled it out internally
replayed      reconstructed during consolidation
```

Recording is permissive: all four are storable, and an undeclared route is
rejected outright. Learning is not. `CountBasedGenerativeFieldModel.update`
returns `False` for anything but `experienced` before it touches a count:

```python
if transition.knowledge_source != LEARNABLE_KNOWLEDGE_SOURCE:
    return False
```

So hearsay and imagination can be remembered in full and still fail to move a
sensorimotor contingency. That asymmetry -- rememberable, not learnable -- is the
whole content of the distinction as implemented.

The route is orthogonal to `source_mode`, which was already enforced: a
transition may never carry `imagined` or `remembered` as its source mode,
whatever its route. An imagined scenario is recorded as an imagined *route* over
an inferred *mode*; it never gets promoted into a live perception.

## Identity of content

A divergence is only attributable to the route if the payload was the same.
`info_content_hash` is a sha256 over the content itself, computed independently
of how it arrived, and the two arms of an experiment assert equality on it
before they diverge. Without that the comparison would silently become a
comparison of two different pieces of information.

## What was measured

`test_experienced_told_imagined.py`, 16 tests:

- one `experienced` transition raises the model's total observation count;
  `told`, `imagined` and `replayed` each leave it exactly unchanged
- the same content with the same context signature returns `True` by the lived
  route and `False` by the told route, in the same database, one after the other
- two forks of one seeded history start at an equal, non-zero observation count,
  and after ingesting identical content only the lived arm moves
- every declared route is storable; `dreamt` raises
- imagined content still cannot claim a live source mode

The pre-existing test asserting `experienced`-only recording was replaced rather
than deleted: the invariant it protected has moved from the schema to the model,
and the new test says so.

## Limits

The divergence demonstrated here is the learning gate, which is a single
conditional. It shows the routes are not interchangeable; it does not show the
rest of the system treats them differently. Recall weighting, confidence in
reports, and whether a told contingency decays differently from a lived one are
all untouched -- each would be its own measurement.

There is also no ingestion API yet. `ingest_told_report` and
`ingest_imagined_scenario` remain to be built; today a route is set by whoever
constructs the transition, which is enough for the experiment and not enough for
live use.

See also `docs/fork-divergence.md` for how the two arms are kept apart.
