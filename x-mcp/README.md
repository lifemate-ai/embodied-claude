# grok-mcp

MCP server for searching X (Twitter) in real-time via xAI Grok live search.

## Setup

```bash
cp .env.example .env
# Fill in your XAI_API_KEY
uv sync
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_x(query)` | Search X posts by keyword / hashtag |
| `get_user_tweets(username)` | Get recent tweets from a user |
| `get_mentions(username)` | Get recent mentions of a user |
| `get_trending_topic(topic)` | Summarize what's being said about a topic |

## Claude Code integration

Add to `~/.claude/settings.json`:

```json
"mcpServers": {
  "grok-mcp": {
    "command": "uv",
    "args": ["run", "--project", "/path/to/grok-mcp", "python", "/path/to/grok-mcp/src/server.py"],
    "env": {}
  }
}
```
