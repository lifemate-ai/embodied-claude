# individual-kernel-mcp

The individual kernel is the implementation package for the EFPF
phenomenal-like causal architecture. See
[`../../README.md`](../../README.md) for the architecture, claim policy,
runtime modes, welfare interlocks, and limitations.

## MCP Surface

The automatic path is Claude Code hooks and heartbeat integration. These MCP
tools are the inspection, explicit-intention, and experiment surface:

| Tool | Role |
|---|---|
| `begin_subjective_tick` | open a typed tick for an explicit experiment |
| `add_workspace_candidate` | add an inspected/debug candidate |
| `commit_subjective_field` | compete and atomically commit one field |
| `get_current_subjective_field` | read the current compact and typed field |
| `get_subjective_field` / `query_subjective_fields` | inspect field history |
| `propose_field_action` | register exact tool/hash, goal, and predicted effects |
| `get_pending_intention` | inspect the one pending intention |
| `close_field_action` | record result, mismatch, ownership, and next microtick |
| `close_subjective_tick` | close a committed episode |
| `pause_field_runtime` / `resume_field_runtime` | welfare/runtime interlock |
| `get_field_diagnostics` | field, HOR, action, consistency, and exposure diagnostics |
| `run_field_ablation` | reversible focal/HOR/valence/reality intervention |

Legacy counterfactual, sleep, frame, attention-schema, HOR, and introspection
tools remain available. `compose_introspection_report` reads the latest
committed field; without one it returns `unknown / no committed field`.

## Action Gate

For an outward action, the caller must:

1. Have one current `COMMITTED` field.
2. Call `propose_field_action` with the exact MCP tool name and input.
3. Pass `PreToolUse`, which matches the normalized input hash and boundary.
4. Execute no more than one outward action for that field/tick.
5. Let `PostToolUse` or `PostToolUseFailure` close the intention.
6. Use the resulting tool-result microtick before another outward action.

Internal reads do not consume the outward-action slot. Perception or recall
results make the current field stale and require one batched refresh before an
outward action.

## Setup And Verification

```bash
uv sync
uv run pytest consciousness-mcp/packages/individual-kernel-mcp/tests
uv run ruff check consciousness-mcp/packages/individual-kernel-mcp
```

Start the MCP server:

```bash
uv run --package individual-kernel-mcp individual-kernel-mcp
```

Run hook diagnostics:

```bash
printf '%s\n' '{"session_id":"smoke","hook_event_name":"SessionStart"}' |
  SOCIAL_DB_PATH=/tmp/efpf-smoke.db uv run --package individual-kernel-mcp efpf-hook session-start | jq .
```

## Hook Smoke Tests

Create a field from a real `UserPromptSubmit` payload:

```bash
printf '%s\n' \
  '{"session_id":"smoke","cwd":"/tmp","permission_mode":"default","hook_event_name":"UserPromptSubmit","prompt":"Inspect the room."}' |
  SOCIAL_DB_PATH=/tmp/efpf-smoke.db uv run --package individual-kernel-mcp efpf-hook user-prompt-submit | jq .
```

Verify that an outward tool without an intention is denied:

```bash
printf '%s\n' \
  '{"session_id":"smoke","hook_event_name":"PreToolUse","tool_name":"mcp__tts__say","tool_input":{"text":"hello"},"tool_use_id":"tool-1"}' |
  SOCIAL_DB_PATH=/tmp/efpf-smoke.db uv run --package individual-kernel-mcp efpf-hook pre-tool-use | jq .
```

Use the `get_current_subjective_field` and `propose_field_action` MCP tools,
then replay the same `PreToolUse` JSON. It is allowed only when the tool name
and normalized input hash exactly match. A different text is denied, and a
second call for the same tick is deferred by `ActionBottleneck`.

Close the loop with a real `PostToolUse` payload:

```bash
printf '%s\n' \
  '{"session_id":"smoke","hook_event_name":"PostToolUse","tool_name":"mcp__tts__say","tool_input":{"text":"hello"},"tool_use_id":"tool-1","tool_response":{"ok":true,"summary":"audio played"},"duration_ms":120}' |
  SOCIAL_DB_PATH=/tmp/efpf-smoke.db uv run --package individual-kernel-mcp efpf-hook post-tool-use | jq .
```

## Ablation

With a field committed, call:

```text
run_field_ablation(kind="focal_clamp", fixture={...}, seed=1)
run_field_ablation(kind="hor_feedback", fixture={...}, seed=1)
run_field_ablation(kind="valence", fixture={...}, seed=1)
run_field_ablation(kind="reality", fixture={...}, seed=1)
```

Every run stores the baseline, intervention result, effect sizes, and a
reversible field snapshot in `field_ablation_runs`.
