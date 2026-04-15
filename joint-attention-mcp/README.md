# joint-attention-mcp

Structured scene grounding for deictic language such as "that mug", "the blue mug", and
"what I was just looking at".

Tools:

- `ingest_scene_parse`
- `resolve_reference`
- `get_current_joint_focus`
- `set_joint_focus`
- `compare_recent_scenes`

Example MCP config:

```json
{
  "mcpServers": {
    "joint-attention": {
      "command": "uv",
      "args": ["run", "joint-attention-mcp"],
      "cwd": "/path/to/embodied-claude/joint-attention-mcp"
    }
  }
}
```
