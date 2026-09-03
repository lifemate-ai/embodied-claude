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

How `evaluate_action` reads the policy:

- `[[privacy_zones]]` match when `context.zone` (or `context.zone_name`) equals the zone `name`,
  or `context.camera_preset` is in its `camera_presets`. An action in the zone's
  `deny_actions` is denied. High urgency still lets in-room actions such as `speak_loud`
  through with `allow_with_override`; publishing actions (`post_image`, `post_text`,
  `post_tweet`, ...) stay denied, because an emergency does not create consent to publish.
  A call that passes neither key matches no zone.
- `[[posting_rules]]` with `require_review_if_person_present = true` denies a post while a
  person is present (`context.scene_contains_face`, `context.person_present`, or
  `payload_preview.person_mentions`) until the caller has run `review_social_post` and
  passes `context.reviewed = true`. `require_face_consent` keeps its previous meaning.
- `[[person_rules]]` `preferred_nudge_style` is returned as `nudge_style` on every result
  for that person so the caller can shape the nudge or reply.

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
