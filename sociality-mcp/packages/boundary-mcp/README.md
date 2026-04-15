# boundary-mcp

Action gating for quiet hours, privacy, per-person boundaries, nudge saturation, and social post
review.

Tools:

- `evaluate_action`
- `review_social_post`
- `record_consent`
- `get_quiet_mode_state`

Policy file:

- `socialPolicy.toml`
- override with `SOCIAL_POLICY_PATH`

Example MCP config:

```json
{
  "mcpServers": {
    "boundary": {
      "command": "uv",
      "args": ["run", "boundary-mcp"],
      "cwd": "/path/to/embodied-claude/boundary-mcp"
    }
  }
}
```
