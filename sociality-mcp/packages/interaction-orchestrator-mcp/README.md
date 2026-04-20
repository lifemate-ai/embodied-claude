# interaction-orchestrator-mcp

Human Response Orchestrator for Embodied Claude / Kokone v0.3.

This package is consumed by the top-level `sociality-mcp` façade and is not
run as a standalone MCP server. It exposes the coordination layer that turns
substrate signals (social state, relationship, memory, desire, narrative arcs,
joint attention) into stable, context-aware social moves and records the
resulting experiences back into the shared SQLite store.

## Tools surfaced via sociality-mcp

- `compose_interaction_context` — assemble a compact, prompt-ready context
  bundle for the next response or autonomous action.
- `plan_response` — pick a social move (answer_directly, stay_silent,
  quietly_prepare, …) with an explicit tone/memory/initiative/boundary plan.
- `record_agent_experience` — persist what the agent just did as an
  experience, not merely a log line.
- `record_interpretation_shift` — remember moments where the agent updated
  how it interprets a rule, a relationship, or a self-model.
- `append_private_reflection` — write a private note without nudging anyone.
- `compose_private_letter` — store a composed letter that may later be
  shared.
- `get_agent_state` — return a short self-state summary (desires, recent
  experiences, active arcs).
