# Sociality v0.3

Sociality is a closed interaction loop, not a bag of independent tools. The
context used for one response and the experience recorded after it become
inputs to the next response.

## One Interaction

### 1. Compose before deciding

Call `compose_interaction_context_tool` with the current person, channel, and
user text. For autonomous activity, pass an `autonomous_trigger` instead of
inventing a user utterance.

The result combines:

- social and turn-taking state;
- the person and relationship model;
- commitments and open loops;
- current desires and narrative arcs;
- recent agent experiences;
- relevant memories;
- current enacted-field identity and summary when available; and
- a response contract with boundaries and initiative constraints.

Do not cherry-pick only the friendly or convenient fields.

### 2. Plan within that context

Pass the complete interaction context to `plan_response_tool`. Its
`primary_move` may select direct speech, silence, private reflection, quiet
preparation, or a bounded autonomous action. The plan also supplies tone,
memory-use, voice, required content, forbidden content, and follow-up details.

Planning without a committed enacted field is compatibility behavior. In the
normal runtime, the field is refreshed first and its summary is part of the
interaction context.

### 3. Act once

Produce the selected response or proposal. Outward tools still pass through the
EFPF intention and action gate; a social plan does not bypass boundary policy or
the one-action bottleneck.

### 4. Record what happened

Call `record_agent_experience` immediately after the response or significant
action. Record the actual event, not the intended or polished version of it.
Useful payload fields include:

```json
{
  "kind": "response",
  "summary": "Answered the setup question and asked for one diagnostic result.",
  "person_id": "kouta",
  "channel": "chat",
  "felt_state": "attentive and uncertain about the Windows process state",
  "desires_before": {"help": 0.7},
  "desires_after": {"help": 0.3},
  "related_event_ids": [],
  "related_memory_ids": []
}
```

The exact schema is validated by `RecordAgentExperienceInput`. Common kinds
include responses, autonomous actions, boundary respect, desire satisfaction,
user corrections, and open-loop progress.

These records are stored in `agent_experiences`. The next
`compose_interaction_context_tool` reads recent rows so past actions constrain
the next plan.

### 5. Record a changed interpretation when there was one

Call `record_interpretation_shift` only when evidence changed how the agent
understands a rule, relationship, convention, or self-model:

```json
{
  "topic": "windows-support",
  "before": "Windows is a best-effort setup target.",
  "after": "Windows Core is a release-gated supported platform.",
  "reason": "The release scope now requires native setup, hooks, doctor, and CI.",
  "person_id": "kouta",
  "related_experience_id": "exp_..."
}
```

Interpretation shifts are stored in `interpretation_shifts`. Their count and
recent content feed later composition and add a do-not-regress constraint to
future plans. Routine restatements are not shifts.

## Why Daybook Fields Can Be Empty

`append_daybook` summarizes social events, but its `concrete_events`,
`noticed_changes`, `relationship_moments`, and related fields are derived from
`agent_experiences` and `interpretation_shifts`. Empty output with no recorded
experience is expected; it does not mean the narrative store is broken.

The correct fix is to close the interaction loop with the record calls, not to
fabricate daybook entries afterward.

## Failure Recovery

- Composition failed: do not plan from an invented context; retry or use
  compatibility behavior explicitly.
- Plan rejected speech: preserve the selected silence/private move.
- Response succeeded but recording failed: retry the same experience with the
  same related event IDs. Do not repeat the outward response.
- Interpretation recording failed: retry the record; do not silently weaken the
  next plan.
- Tool result changed available evidence: refresh the enacted field before
  another outward action.

## Tool Reference

### Interaction orchestration

- `compose_interaction_context_tool`
- `plan_response_tool`
- `record_agent_experience`
- `record_interpretation_shift`
- `append_private_reflection`
- `compose_private_letter`
- `get_agent_state`

### Social state

- `ingest_social_event`
- `get_social_state`
- `should_interrupt`
- `get_turn_taking_state`
- `summarize_social_context`

### Relationships and commitments

- `upsert_person`
- `ingest_interaction`
- `get_person_model`
- `create_commitment`
- `complete_commitment`
- `list_open_loops`
- `suggest_followup`
- `record_boundary`

### Joint attention

- `ingest_scene_parse`
- `resolve_reference`
- `get_current_joint_focus`
- `set_joint_focus`
- `compare_recent_scenes`

### Boundaries

- `evaluate_action`
- `review_social_post`
- `record_consent`
- `get_quiet_mode_state`

### Self narrative

- `append_daybook`
- `get_self_summary`
- `list_active_arcs`
- `reflect_on_change`

See [`CLAUDE.md`](../CLAUDE.md) for the repository runtime protocol and
[`sociality-mcp/packages/interaction-orchestrator-mcp/`](../sociality-mcp/packages/interaction-orchestrator-mcp/)
for the lower-level package contract.
