# social-state-mcp

Compact, confidence-bearing social state inference over the shared social event store.

Tools:

- `ingest_social_event`
- `get_social_state`
- `should_interrupt`
- `get_turn_taking_state`
- `summarize_social_context`

Example MCP config:

```json
{
  "mcpServers": {
    "social-state": {
      "command": "uv",
      "args": ["run", "social-state-mcp"],
      "cwd": "/path/to/embodied-claude/social-state-mcp"
    }
  }
}
```

Recommended orchestration:

1. Before speaking: `get_social_state`, then `should_interrupt`
2. After hearing/seeing something: `ingest_social_event`
3. During a live conversation: `get_turn_taking_state`
