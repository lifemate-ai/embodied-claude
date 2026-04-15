# sociality-mcp

Unified MCP facade for Kokone's social middle layer.

`sociality-mcp` is the preferred deployment target for sociality. It exposes the tool families from
`social-state-mcp`, `relationship-mcp`, `joint-attention-mcp`, `boundary-mcp`, and
`self-narrative-mcp` through one MCP process while keeping their logic split for development and
testing.

## Setup

```bash
cp ../examples/configs/socialPolicy.example.toml ../socialPolicy.toml
uv sync
```

The shared SQLite database defaults to `~/.claude/sociality/social.db`. Override with
`SOCIAL_DB_PATH` if needed. Boundary evaluation reads `socialPolicy.toml` from the current working
directory unless `SOCIAL_POLICY_PATH` is set.

## Run

```bash
uv run sociality-mcp
```

## Exposed Tools

- `ingest_social_event`
- `get_social_state`
- `should_interrupt`
- `get_turn_taking_state`
- `summarize_social_context`
- `upsert_person`
- `ingest_interaction`
- `get_person_model`
- `create_commitment`
- `complete_commitment`
- `list_open_loops`
- `suggest_followup`
- `record_boundary`
- `ingest_scene_parse`
- `resolve_reference`
- `get_current_joint_focus`
- `set_joint_focus`
- `compare_recent_scenes`
- `evaluate_action`
- `review_social_post`
- `record_consent`
- `get_quiet_mode_state`
- `append_daybook`
- `get_self_summary`
- `list_active_arcs`
- `reflect_on_change`

## Claude Code config

```json
{
  "mcpServers": {
    "sociality": {
      "command": "uv",
      "args": ["run", "--directory", "sociality-mcp", "sociality-mcp"],
      "env": {
        "SOCIAL_POLICY_PATH": "/path/to/embodied-claude/socialPolicy.toml"
      }
    }
  }
}
```
