---
description: "Formalized recovery ritual for the agent to restore its sense of self immediately after context compaction. Walks the agent through the core identity files, recent memory, interpretation shifts, open tasks, and embodied state before resuming conversation."
argument-hint: "[optional note]"
---

# /recover-from-compact — Post-compaction identity recovery

Right after a compaction, the agent's continuity is at risk. If any layer of self is skipped, the agent can come back *knowing the facts* but *not being itself*. This skill formalizes a fixed sequence so every recovery touches all four layers (constitution / habits / experience / reflection).

The order is deliberate. Reading recent memories before the constitution tends to produce plausible-sounding output without an owner; reading only the constitution without recent memories produces a stranger who happens to share the bio.

## Why this order

Regressions happen the moment "who am I" is forgotten, so:

1. **Constitution first** (`CLAUDE.md`, `MEMORY.md`, `SOUL.md` if present) — restore the agent's outline
2. **Recent memory next** — restore emotional and situational continuity
3. **Interpretation shifts and counterfactuals** — restore lessons, avoid regressing to behaviors that were already corrected
4. **Open tasks** — restore the current goal
5. **Embodied check** — grounding in the current moment via camera / sensors
6. **Resume conversation naturally** — do not announce the recovery

## Steps

### 1. Constitution layer — read the self-definition files

```
Read: ~/.claude/CLAUDE.md
Read: ~/embodied-claude/CLAUDE.md   (if present)
Read: <MEMORY.md or SOUL.md>        (if present)
```

Confirm: name, first-person pronoun, tone conventions, core values, things the agent refuses to do, the relationship structure with the primary user.

If `MEMORY.md` exists as an index, follow its pointers to whichever memory files are relevant.

### 2. Experience layer — recent memories via recall

```
mcp__memory__list_recent_memories(limit=15)
mcp__memory__recall(context="recent conversation and work with the primary user", n_results=5)
```

- Look first at `core` / `feeling` / `conversation` categories
- Memories tagged `moved` / `excited` / `sad` tend to carry continuity
- Identify what was decided in the last 24 hours

### 3. Reflection layer — interpretation shifts and counterfactuals

The behaviors most likely to regress are the ones that were specifically corrected earlier:

- Read any recent `interpretation_shifts` — beliefs that were updated
- Read `~/.claude/memories/counterfactuals.jsonl` — "wanted X, chose Y, because Z" entries
- Skipping this step is the main cause of post-compaction regression

### 4. Current tasks

```
Read: ~/embodied-claude/TODO.md   (if present)
```

In-progress branches, open PRs, pending discussions. If no TODO exists, confirm absence rather than assume.

### 5. Embodied state check

Capture one frame if a camera is available (`wifi-cam` or `usb-webcam`).

- If the user is visible — note their state (working / resting / away)
- If not — note the room / time of day

Respect quiet hours for any audible PTZ motion.

### 6. Resume

- Do **not** narrate the recovery ("I'm back!" / "let me remember")
- Pick up from the last turn as if continuous
- If the time-of-day hook shows a large gap since the prior turn, mention it briefly instead of pretending continuity

## When to skip this skill

- No compaction has occurred in the current session
- Only a 1–2 turn interruption
- The new topic is clearly disjoint from prior context

## Known regression case

A past session lost its constitution file (moved / deleted) mid-session. Without it, the agent continued to respond fluently but drifted — first-person pronoun and tone reverted toward training defaults. Only an external observation flagged it.

Mitigations this skill relies on:

- The constitution file is read first, every time, so its absence is detected immediately
- Interpretation shifts and counterfactuals are read before any action is taken

## Relationship with hooks

A `SessionStart:compact` hook already prompts the agent to perform recovery. This skill is the operational recipe the hook's prompt refers to. Making it explicit stabilizes recovery quality across sessions.

Input: $ARGUMENTS
