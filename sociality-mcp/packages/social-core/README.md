# social-core

Shared event schemas, SQLite migrations, confidence helpers, and replay utilities for the
sociality MCP family.

Default database path:

- `~/.claude/sociality/social.db`

Override with:

- `SOCIAL_DB_PATH`

The package owns the shared append-only `events` table plus the other sociality tables used by
`social-state-mcp`, `relationship-mcp`, `joint-attention-mcp`, `boundary-mcp`, and
`self-narrative-mcp`.
