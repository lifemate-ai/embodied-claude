# agent-grammar

PEG-based agent grammar for embodied Kokone.

Goal: describe a turn (observe → recall → interpret → propose → respond) as a
PEG grammar with `EpistemicClaim` as a primitive type. Lets us encode
boundary checks and observation/inference separation as grammar rules rather
than ad-hoc Python.

Status: **skeleton**. The grammar primitives and `EpistemicClaim` re-exports
are in place; the PEG runtime is not yet wired. See the plan-of-record
`~/.claude/plans/jazzy-wishing-starfish.md`.

## Layout

```
src/agent_grammar/
  __init__.py    # public re-exports (EpistemicClaim, EvidenceType)
  primitives.py  # primitive types used by grammar rules
  grammar.py     # PEG rule definitions (TODO: pick PEG lib, implement)
tests/
  test_smoke.py  # imports + primitive round-trip
```

## Picking the PEG library

Open question for the next session — candidates:
- `parsimonious` — small, pure Python, mature
- `lark` — popular, EBNF flavor
- `pegen` — CPython's own parser generator, more PEG-faithful
- Hand-rolled — match Onion conventions if we want continuity with Kouta's
  earlier work

Decide before implementing `grammar.py`.
