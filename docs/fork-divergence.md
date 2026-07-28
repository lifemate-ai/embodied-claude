# Forking the History

Status: shipped 2026-07-28 on the `feat/experienced-told-imagined` branch. Scope:
running two arms from one history without either contaminating the other. No
runtime path forks anything yet; this is experiment machinery.

Claim policy: this is an isolation mechanism for comparisons. Nothing in it is a
phenomenal claim.

## Why an isolated copy at all

A provenance experiment has one requirement that is easy to state and easy to
get wrong: the two arms must start from the *same* history and must not be able
to see each other's writes. If the told arm can observe what the lived arm
learned, the comparison measures the plumbing.

## What was rejected

**Namespacing by `owner_id`.** The obvious cheap option: give each arm its own
owner and keep one database. It does not hold. Several readers in this package
query without an owner filter, so an in-database fork would leak across arms --
and it would leak quietly, producing a plausible number rather than an error.

**Copying the database file.** The first implementation did `shutil.copy2` on
the `.db`. It was wrong, and the tests caught it: a fork of a seeded history
came back with zero observations. `SocialDB.connect` sets
`PRAGMA journal_mode=WAL`, so recent commits are still in the `-wal` sidecar and
a copy of the main file alone starts the arm from an older, emptier state. The
failure mode is the dangerous one: no exception, just an arm that quietly
branched from the wrong point.

## What it does

`fork_history` opens a fresh `SocialDB` at a new path and fills it via SQLite's
backup API:

```python
self.db = SocialDB(self.path)
source.connect().backup(self.db.connect())
```

The backup takes a consistent snapshot under SQLite's own locking, WAL content
included, and the source is only ever read -- an experiment cannot write back
into the history it branched from. The handle is a context manager, so an arm
closes with its `with` block.

Each fork gets a `fork_id`, and `imagined_trajectories`,
`protention_distributions` and `experienced_transitions` all carry a nullable
`fork_id` column reserved for labelling rows by arm.

## What was measured

In `test_experienced_told_imagined.py::TestFork`:

- a fork of a seeded history reports the same observation count as its source
- writing in an arm leaves the source's count exactly unchanged
- two arms of one seeded history start equal, and after identical content only
  the arm that received it by the lived route moves

The second of those is what the file copy would have passed while being broken:
an empty arm also leaves the source untouched. The first is what caught it.

## A note for tests that fork

The prediction loop records a transition for every commit that has a
predecessor, as `experienced`, and the store is idempotent per field. A test
that wants to choose the route has to commit its field with that recording
switched off, which the `generative_field_model` behaviour flag already allows
because it is re-read on every call. `_runtime_recording_paused` in the test
module does exactly that and nothing else.

## Limits

A fork is a whole-database snapshot, so it is O(database size) per arm. That is
fine for fixtures and for occasional experiments against live history; it is not
a mechanism for forking on every tick.

Nothing labels rows with `fork_id` yet. The column exists so that an arm's
writes can be identified after a merge-back, and no merge-back is implemented.

See also `docs/experienced-told-imagined.md` for what the arms are comparing.
