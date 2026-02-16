# Canonical And Mirror Policy

## Scope
- Canonical source-of-truth: `embodied-claude/`
- Local migration mirror: `../embodied-claude-main/`

## Rules
1. All implementation changes are authored in canonical first.
2. Mirror updates are one-way sync from canonical.
3. Local-only files are excluded from drift checks:
   - `.git/`
   - `.venv/`
   - `__pycache__/`
   - `.pytest_cache/`
   - `.mcp.json`

## Sync Workflow
1. Edit and validate in canonical.
2. Sync changed files to mirror.
3. Run `scripts/check-mirror-sync.sh`.
4. Resolve drift before merge.

## Why
- Keeps contributor focus on one authoritative codebase.
- Preserves a migration mirror without hidden divergence.
- Makes CI and release behavior deterministic.
