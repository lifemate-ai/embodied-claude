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
