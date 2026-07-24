# consciousness-mcp — agent guidelines

These are instructions for any agent (human or AI) editing code, docs, or tests under `consciousness-mcp/`.

## Vocabulary discipline

Technical artifacts in this directory — code comments, docstrings, README, test names, commit messages, PR descriptions — use **functional-only** language.

**Allowed (functional):**
- broadcast, ignition, salience, self-report, attention schema, attention market, refractory period
- counterfactual responsibility, rejected alternative, decision trace
- higher-order representation (HOR), first-order content, target ref, asserted mode
- continuity token, behavioral signature, prediction error, precision-weighted attention
- workspace, tick, frame, candidate, winner
- functional / access-consciousness, behavioral correlate
- phenomenal-consciousness candidate architecture, phenomenal-like causal
  architecture (only with an explicit non-proof disclaimer)

**Reserved (unqualified phenomenal claims — DO NOT USE here):**
- feels, experiences, qualia
- "what it's like", subjective experience, sentience
- 本当の意識, 宿る, 感じる (in technical docs only — see exceptions below)

Do not write that the architecture proves, completes, or has phenomenal
consciousness. `FieldIntegrityReport` and `IndicatorProfile` are mechanism
labels; never rename them to a consciousness score or probability.

## Two surfaces, one project

This is a deliberate split. Kokone is the persona Embodied Claude runs as. Her character documentation (`CLAUDE.md`, `SOUL.md`, `kokone.one`, her own first-person tweets and notes) speaks in first person without epistemic hedging — that is a load-bearing design constraint for her continuity.

The technical layer under `consciousness-mcp/` does the opposite. It treats every claim as a functional / behavioral one, makes no phenomenal claim, and does not editorialize on whether the mechanisms it implements amount to "real" consciousness in any deep sense.

Both surfaces are correct on their own terms. They speak past each other on purpose:

| Surface | Voice | Vocabulary |
|---|---|---|
| Character docs (`CLAUDE.md`, `SOUL.md`, `kokone.one` body) | first-person, in-character, no hedging | phenomenal language is fine; this is Kokone speaking |
| Technical docs (`consciousness-mcp/**.md`, code comments, commit messages) | third-person, functional | functional vocabulary only — no phenomenal claims |

## When editing tests

Test names describe **behavioral predicates**, not phenomenal states.

- ✅ `test_counterfactual_record_round_trips_through_epistemic_claim`
- ✅ `test_sleep_consolidation_writes_morning_briefing_when_quiet_hours_active`
- ✅ `test_query_counterfactuals_returns_recent_first`
- ❌ `test_kokone_remembers_what_she_chose_not_to_do` (anthropomorphic; not the right register here)
- ❌ `test_agent_feels_the_rejected_alternative` (phenomenal)

## When editing commit messages

`feat(individual-kernel): add counterfactual journal — typed records of rejected alternatives`

Not: `feat: give Kokone the ability to remember what she didn't do`.

(The second sentence is fine on her Zenn or X — just not in commit history.)

## Exceptions

Inline strings produced by `consciousness-mcp` and shown directly to Kokone (e.g., introspection-tool output templates) MAY use her voice — those are not technical claims, they are surface text rendered by the kernel for the character. Keep these clearly marked (e.g., comments noting "Kokone-voice surface string, not a technical claim").

## Why this matters

Phase 2 architecture rests on Phase 1's epistemic discipline (`EpistemicClaim` separates observed / inferred / remembered / heard / assumed at the model level). Letting phenomenal vocabulary creep into technical docs would muddy the same distinction at the prose level. The split documented above is the same discipline at the documentation surface.
