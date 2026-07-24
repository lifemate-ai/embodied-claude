# sociality-mcp

`sociality-mcp` is the unified MCP facade for embodied-claude's social middle
layer. It composes the current relationship, turn-taking, memory, desire,
boundary, and enacted-field context into one response plan, then records what
the agent actually did so the next interaction can depend on it.

## The v0.3 Interaction Loop

Use the orchestrator in this order:

```text
compose_interaction_context_tool
-> plan_response_tool
-> act or respond
-> record_agent_experience
-> record_interpretation_shift (only when an interpretation changed)
```

1. Call `compose_interaction_context_tool` before a response or autonomous
   action. Pass its complete result to `plan_response_tool`.
2. Condition the response or action on the returned primary move, boundary,
   tone, memory-use, and initiative constraints.
3. Call `record_agent_experience` immediately after the response or significant
   action. This is the recurrent update, not optional telemetry.
4. Call `record_interpretation_shift` when the agent changed how it understands
   a rule, relationship signal, convention, or self-model. Do not call it after
   every ordinary response.
5. Create or complete commitments and ingest social events when the interaction
   actually changes those stores.

The next composition surfaces recent experiences and accumulated
interpretation shifts. `append_daybook` also reads those records. If experience
recording is skipped, an otherwise valid daybook may have empty
`concrete_events`, `noticed_changes`, or `relationship_moments`.

For payload examples, recovery behavior, and the relationship between the
interaction loop and the current enacted field, read the
[Sociality v0.3 guide](../docs/sociality.md).

## Setup

From the repository root, use the guided setup:

```bash
./scripts/setup.sh --profile core --non-interactive
```

The setup creates `socialPolicy.toml` only when it is absent and configures the
shared SQLite database at `~/.claude/sociality/social.db`. Override the database
with `SOCIAL_DB_PATH` or the policy with `SOCIAL_POLICY_PATH`.

To run only this server during development:

```bash
uv run --package sociality-mcp sociality-mcp
```

## Tool Families

- Interaction orchestration: compose, plan, experience, interpretation shifts,
  private reflection, and agent state.
- Social state: events, turn-taking, interruption, and context summaries.
- Relationships: people, interactions, commitments, open loops, and follow-up.
- Joint attention: scene ingestion, reference resolution, and shared focus.
- Boundaries: action evaluation, post review, consent, and quiet mode.
- Self narrative: daybook, self summary, and active arcs.

The complete surface and payload contracts are in
[`docs/sociality.md`](../docs/sociality.md) and the package-level READMEs under
[`packages/`](./packages/).
