# self-narrative-mcp

Compact autobiographical daybooks and identity arcs built from already-ingested social events.

Tools:

- `append_daybook`
- `get_self_summary`
- `list_active_arcs`
- `reflect_on_change`

Example MCP config:

```json
{
  "mcpServers": {
    "self-narrative": {
      "command": "uv",
      "args": ["run", "self-narrative-mcp"],
      "cwd": "/path/to/embodied-claude/self-narrative-mcp"
    }
  }
}
```
