# relationship-mcp

Compact person models, commitments, open loops, rituals, and boundaries on top of the shared
social event store.

Tools:

- `upsert_person`
- `ingest_interaction`
- `get_person_model`
- `create_commitment`
- `complete_commitment`
- `list_open_loops`
- `suggest_followup`
- `record_boundary`

`suggest_followup` returns a ready-made `text` in plain Japanese by default. Set
`RELATIONSHIP_TONE=kansai` for the original Kansai phrasing (unknown values fall
back to `neutral`). Every suggestion also carries `intent` and `topic`, independent of
the tone, so the agent can phrase the follow-up in its own voice.

Example MCP config:

```json
{
  "mcpServers": {
    "relationship": {
      "command": "uv",
      "args": ["run", "relationship-mcp"],
      "cwd": "/path/to/embodied-claude/relationship-mcp"
    }
  }
}
```
