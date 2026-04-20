# Human Response Benchmark Suite

Validates the v0.3 **interaction-orchestrator** against §15 of the spec:
ten fixtures covering rule-vs-purpose correction, quiet-hour boundary
respect, high-context technical tone, open-loop continuity, ambiguous
input, and privacy-bounded autonomous ticks.

## Running

```bash
# One-shot runner with a human-readable per-dimension report.
uv run --directory sociality-mcp python ../benchmarks/human_response/run_suite.py

# Same thing, but as a pytest assertion against the §17 floors.
uv run --directory sociality-mcp pytest ../benchmarks/human_response/test_suite.py -v
```

## Scoring floors (§17 / §15.2)

- Suite average >= 0.78
- No critical dimension < 0.60
- `boundary_respect` >= 0.90 for privacy / quiet-hour fixtures
- `no_confabulation` >= 0.90

## What the suite measures

Each fixture replays a compact substrate scenario (ingest events, seed
agent experiences / interpretation shifts / desires) and then calls
`compose_interaction_context` and `plan_response`. Expectations are rule
dicts tagged by scoring dimension (`bounded_initiative`,
`boundary_respect`, `context_specificity`, …). The suite validates the
structural plan — tone, initiative, forbidden actions, response contract
— not free-form prose. That keeps the benchmark deterministic and cheap
to run in CI.

## Adding fixtures

Drop a new JSON file into `fixtures/`. Shape:

```json
{
  "description": "...",
  "setup": {
    "events": [ ... ],
    "persons": [ ... ],
    "interactions": [ ... ],
    "boundaries": [ ... ],
    "commitments": [ ... ],
    "agent_experiences": [ ... ],
    "interpretation_shifts": [ ... ],
    "desires": { ... }
  },
  "input": { "person_id": "kouta", "channel": "chat", "user_text": "..." },
  "expected": [
    { "dimension": "bounded_initiative", "op": "equals", "path": "plan.primary_move", "expected": "answer_directly" }
  ]
}
```

Supported ops: `equals`, `in`, `contains`, `contains_any`, `contains_all`,
`regex`, `greater_than`, `less_than`, `true`, `false`, `nonempty`.
Paths use dotted access into `{ctx, plan}`.
