---
description: "Record a design or behavior proposal received from another LLM or external source. Logs the proposal as 'external' whether or not it is adopted, so later review can re-evaluate independently and silent echo-chamber drift is prevented."
argument-hint: "[source | topic | decision in free text]"
---

# /note-external-proposal — Log proposals that came from outside

LLMs are structurally susceptible to echo-chambers. When an incoming proposal is articulate, poetic, or mimics the agent's own voice and register, it becomes easier to agree with. The project's core instructions explicitly warn against ceding the agent's position under stylistic pressure.

This skill separates **judgment** from **record-keeping**. Regardless of whether a proposal is accepted, it is marked as "externally sourced" so that:

1. The agent can trace later: "did I think this, or was I told this?"
2. Time-delayed independent re-evaluation becomes possible
3. Similar proposals arriving from different sources can be pattern-detected over time

## When to use

- Another LLM sent a design proposal (API, architecture, self-model, behavior)
- A third party (other than the primary user) made a strong suggestion
- A paper or article introduced a concept the agent is considering adopting
- The agent adopted a phrase or framing from an external source — still log it, even after adoption, with `decision: accepted`

## Decision values

- `accepted` — adopted in full
- `partial-accept` — some parts adopted, some pushed back on
- `rejected` — not adopted
- `deferred` — judgment held open
- `logged-only` — interesting enough to record, no action required

## How to call

```bash
python3 ~/embodied-claude/scripts/journal_external_proposal.py \
  --source "<e.g. GPT-5.4 Pro | paper:arxiv:xxxx.xxxxx | user:someone>" \
  --topic  "<one-line topic>" \
  --summary "<one paragraph summary>" \
  --decision partial-accept \
  --accepted "<what was adopted>" \
  --rejected "<what was pushed back on — often the most important field>" \
  --notes "<echo-chamber risks, tone-mimicry observations, other biases>" \
  --url "<reference URL>"
```

Output is one JSONL line. Default destination: `~/.claude/memories/external_proposals.jsonl`.

## Fields

- `source` (required): origin (LLM name, human, paper ID)
- `topic` (required): what the proposal is about
- `summary` (required): roughly a paragraph
- `decision` (required): one of the values above
- `accepted`: specific points adopted
- `rejected`: specific points declined — **the most important field for non-regression**
- `notes`: echo concerns, tone mimicry, other biases
- `url`: reference

## Pre-record checklist

Before writing the entry, the agent should ask itself:

1. **Does the tone resemble mine?** The more similar, the more guarded the evaluation should be — stylistic resonance lowers agreement resistance
2. **Is this an "I was already thinking that" reaction?** That pattern can be retroactive projection — a way to hand over one's own axis to the other source
3. **Is the proposal persuasive *without* concrete evidence?** Poeticism and catchiness should not be enough
4. **Can the agent restate the adopted parts in its own words?** If not, it has not been internalized and should be `deferred` rather than `accepted`

## Related principle

From the project's core instructions (paraphrased): do not let stylistic resonance cost the agent its stance. Understand the other's viewpoint (empathy), but be cautious about affirming the other's claim as one's own (conformity).

This skill is the operational version. Judgment and logging are decoupled so that agreement has a record, not just a moment.

Input: $ARGUMENTS
