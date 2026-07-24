# Releasing embodied-claude

Releases are cut from `main`. The tag is the deliberate human-controlled
boundary; GitHub Actions validates and publishes it but never creates it.

## Preflight

Run the same workspace and release checks used by CI:

```bash
uv sync --locked --all-extras --group dev
uv run pytest
uv run ruff check .
./scripts/release-check.sh --tag v0.3.0
```

On Windows:

```bat
uv sync --locked --all-extras --group dev
uv run pytest
uv run ruff check .
scripts\release-check.cmd --tag v0.3.0
```

Every `pyproject.toml`, `uv.lock`, the dated `CHANGELOG.md` section, and the tag
must contain the same version.

## Publish

After the release PR is green and merged:

```bash
git switch main
git pull --ff-only
git tag -a v0.3.0 -m "embodied-claude v0.3.0"
git push origin v0.3.0
```

The `Release` workflow verifies the tag and publishes the corresponding GitHub
Release. Confirm that the workflow succeeds and that the release is visible
before announcing it.

Do not describe this release as proving phenomenal consciousness. The supported
term is `phenomenal-consciousness candidate architecture`.
