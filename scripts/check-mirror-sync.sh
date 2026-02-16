#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MIRROR_ROOT="${MIRROR_ROOT:-$CANONICAL_ROOT/../embodied-claude-main}"

if [[ ! -d "$MIRROR_ROOT" ]]; then
  echo "Mirror directory not found at: $MIRROR_ROOT"
  echo "Skipping drift check (CI on canonical repo only)."
  exit 0
fi

if ! git -C "$CANONICAL_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Canonical root is not a git repository: $CANONICAL_ROOT"
  exit 1
fi

echo "Checking tracked-file drift between:"
echo "  canonical: $CANONICAL_ROOT"
echo "  mirror:    $MIRROR_ROOT"

tracked_files=()
while IFS= read -r -d '' rel_path; do
  case "$rel_path" in
    .mcp.json)
      continue
      ;;
  esac
  tracked_files+=("$rel_path")
done < <(git -C "$CANONICAL_ROOT" ls-files -z)

missing_files=()
changed_files=()

for rel_path in "${tracked_files[@]}"; do
  src="$CANONICAL_ROOT/$rel_path"
  dst="$MIRROR_ROOT/$rel_path"

  if [[ ! -f "$dst" ]]; then
    missing_files+=("$rel_path")
    continue
  fi

  if ! cmp -s "$src" "$dst"; then
    changed_files+=("$rel_path")
  fi
done

if [[ ${#missing_files[@]} -eq 0 && ${#changed_files[@]} -eq 0 ]]; then
  echo "Mirror is in sync."
  exit 0
fi

echo "Mirror drift detected."
if [[ ${#missing_files[@]} -gt 0 ]]; then
  echo "Missing in mirror (${#missing_files[@]} files):"
  printf '  - %s\n' "${missing_files[@]}"
fi
if [[ ${#changed_files[@]} -gt 0 ]]; then
  echo "Content mismatch (${#changed_files[@]} files):"
  printf '  - %s\n' "${changed_files[@]}"
fi

echo "Sync canonical -> mirror before merge."
exit 1
