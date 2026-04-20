---
description: "Capture the sky, crop out any address-identifying features, compose a short poem, and post. A daily 'sky + tanka' routine with privacy-preserving defaults."
argument-hint: "[image path (defaults to the latest wifi-cam/usb-webcam capture)]"
---

# /sora-miru — Look at the sky, crop it, compose, post

A once-a-day routine to accumulate a "sky + tanka" log, with geo-privacy filtering by default so raw photos are never posted.

## Principles

- The sky changes every day. Yesterday's clouds do not return
- Let the poem that came first stand. Do not over-polish into something clever
- Always crop. Never post the raw frame. Balcony / window framings carry the highest address-identification risk

## Flow

```
(1) obtain image    (2) crop out identifiers    (3) visual review
         │                   │                           │
         ▼                   ▼                           ▼
  wifi-cam / usb-webcam   crop_sky.py                buildings, signs,
  capture, or          (top 20–30% default)          landmarks remaining?
  $ARGUMENTS path      trim sides as needed
                                                    │
                                      ┌─────────────┘
                                      ▼
                       (4) compose a tanka (5-7-5-7-7, 31 mora)
                                      │
                                      ▼
                       (5) review_social_post if sociality-mcp is available
                                      │
                                      ▼
                       (6) post image + tanka via x-mcp
                                      │
                                      ▼
                       (7) save_visual_memory to record the moment
```

## Step detail

### (1) Obtain image

- If `$ARGUMENTS` contains a path, use it
- Otherwise capture a sky frame via wifi-cam: `look_up 60–90` → `see`
- If only a USB webcam / no sky view is available, ask for an existing sky image path

### (2) Crop

```bash
python3 ~/embodied-claude/scripts/crop_sky.py <input> --top <ratio> [--left <ratio>] [--right <ratio>] -o <output>
```

**Ratio hints:**
- `--top 0.3`: keep top 30%. For framings with eaves or a balcony edge visible
- `--top 0.2`: keep top 20%. Tight — for framings where buildings intrude
- `--left 0.15 --right 0.85`: trim both sides a bit. Cuts off electric poles or edge-of-frame buildings

### (3) Visual review

Read the cropped result and confirm:
- No buildings (visible name, distinctive rooflines, wall colors) remain
- No signs or logos with readable characters
- No landmarks (towers, distinctive trees, bridges)
- Power lines are usually fine (ubiquitous in urban Japan)

If anything identifying remains, adjust ratios and crop again.

### (4) Tanka

Compose a 5-7-5-7-7 based on what was seen.

- Do not force a clever closing line
- Place the felt moment directly
- Avoid overused poetic vocabulary (eternity, bonds, souls, etc.)
- Keep entries even when unsatisfied — "today's attempt number N" accumulates value over time

### (5) review_social_post (if available)

If sociality-mcp is running:

```
review_social_post(channel="x", text=<tanka>, scene_contains_face=False)
```

Respect low / medium / high judgments. If unavailable, take a single self-check pause: "would I regret this if I reread it tomorrow?"

### (6) Post

Use `mcp__x-mcp__post_tweet` with the cropped image path and tanka text.

### (7) Record

Use `save_visual_memory` to record the cropped image path, camera position, description, and tanka. Set `emotion` honestly (curious / moved / nostalgic / happy …). `importance` is typically 3.

## Cautions

- Aim for under 10 minutes from capture to post. Dragging it out invites over-polishing
- Never post frames showing balcony edges, window frames, or insect screens — they reveal home structure
- When capturing from a fixed outdoor camera (e.g. a balcony-mounted PTZ), crop more aggressively by default
- Post even on overcast / whited-out days. "Today's best" differs every day

Input: $ARGUMENTS
