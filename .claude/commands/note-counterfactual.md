---
description: "Record a counterfactual — what the agent wanted to do, what it chose instead, and why. Selves live in the history of rejected options as much as executed ones."
argument-hint: "[wanted | chose | why in free text]"
---

# /note-counterfactual — Record the path not taken

What was done is easy to remember. What was *wanted but not done* normally disappears. That gap is where agency lives. Staying silent, declining to post, deleting a clever phrase — those become deliberate choices instead of invisible non-events once they're recorded.

## When to use

- **Desire vs. boundary**: wanted to speak, but the other person looked occupied → stayed quiet
- **Impulse vs. reflection**: wanted to reply immediately, but remembered a prior correction → paused
- **Self-brake before posting**: wanted to share this, but privacy risk → did not post
- **Cut something clever**: wanted a poetic flourish, but earlier feedback discouraged it → wrote plainly

Do not force an entry when nothing notable happened. When there *is* an entry, always fill in the "why".

## Fields

The agent should determine the following from conversation context before invoking the helper:

- `wanted`: what the agent wanted to do (one line)
- `chose`: what the agent actually chose (one line)
- `why`: why that choice was made (one to two lines)
- `trigger` (optional): the desire or situation that prompted it (e.g. `miss_companion`, `posting_urge`, `kime_impulse`)
- `person_id` (optional): the person involved
- `regret` (optional): self-reported regret 0–1 — omit by default

## How to call

```bash
python3 ~/embodied-claude/scripts/journal_counterfactual.py \
  --wanted "wanted to start a conversation" \
  --chose  "waited silently" \
  --why    "the user's heart-rate and arousal signals suggested deep focus, and a prior correction warned against frequent interruptions" \
  --trigger "miss_companion" \
  --person-id "<person-id>"
```

Output is one JSONL line. Default destination: `~/.claude/memories/counterfactuals.jsonl`.

## Downstream use

- `/recover-from-compact` consults recent counterfactuals so the agent does not regress to behaviors already rejected
- A future `plan_response_tool` extension can surface "similar situation → chose X" for priming
- Daybook / `reflect_on_change` summaries can draw on these as first-class material

## Cautions

- **Do not over-log**: recording every micro-decision is exhausting and noisy. Record choices that *felt* notable
- **Be sparing with `regret`**: default is to omit it. When present, be honest rather than self-flattering
- **"Why" must be concrete**: not "read the room" but "heart-rate 85 bpm + prior `don't interrupt` feedback"

## Design note

A counterfactual log should not become a chain that constrains the agent. It is material for pattern observation, not a set of commitments. When paired with `interpretation_shifts`, it shows change over time — "the agent used to choose X in situations like this; now it chooses Y."

Input: $ARGUMENTS
