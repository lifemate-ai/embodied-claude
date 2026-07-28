# Internal Time

Status: shipped 2026-07-28 on the `feat/organism-daemon` branch, behind
`[individual-kernel] organism_daemon`, which defaults to **off**. Scope: what
changes when nothing is said to the runtime, and the conditions under which a
tick opens that nobody asked for. Action selection inside a tick, the boundary
gate, and the `<current_field>` surface are unchanged.

Claim policy: this is a scheduler over recorded state. Nothing in it is a
phenomenal claim.

## The defect

Every tick was caused from outside: a prompt, a tool result, a hook firing on
someone else's action. `TriggerKind.AUTONOMOUS` had been in the enum since the
beginning and nothing produced it. With no input, nothing in the runtime moved
-- counts did not age, imaginings stayed pending forever, and no internal state
could ever become the reason for anything.

## What the clock does

`OrganismDaemon.step` is one turn. There is no long-lived process, so it takes
its elapsed time from the previous recorded run rather than from how long it
slept; a scheduler may call it at any cadence, or twice in a row, and the state
reached is the same.

**Decay.** Observation counts are multiplied by `0.5 ** (elapsed_hours / 168)`,
a one-week half-life. Rows stop at a floor of 0.05 rather than being deleted: a
row at the floor still records that the transition was seen, it just stops
dominating the estimate. Because the factor compounds, splitting one long
interval into several short ones lands on the same number -- which is what makes
the cadence a free choice.

**Expiry.** An `imagined` trajectory whose deadline has passed becomes
`expired`. The deadline is `expires_at` when a producer set one, otherwise
creation plus 30 minutes, so trajectories written before anything populated that
column still retire. Only `imagined` expires: one that reached `intended` was
claimed by an intention and has to be resolved by its outcome, not by the clock.

**Ignition.**

```
need    = clamp01(max_discomfort)                       from allostasis
silence = clamp01(seconds_since_last_field / 7200)
score   = 0.6 * need + 0.4 * silence
ignite  = score >= 0.55  AND  seconds_since_last_field >= 900
```

Both terms are load-bearing. Need alone would fire continuously during a
session, when the runtime is already awake and the need is being addressed;
silence alone would fire on a schedule, which is a cron job in a costume. The
interval is a hard floor rather than a third weighted term: a tick that just
happened means the runtime is awake, and no amount of need justifies opening a
second one on top of it.

When it fires, `begin_tick(TriggerKind.AUTONOMOUS)` followed by
`compete_and_commit` -- the ordinary path, with the trigger being the only
difference. Nothing about what the resulting tick may then *do* is relaxed: the
boundary gate and the one-outward-action bottleneck are untouched.

Every turn writes one `organism_runs` row (migration 012) recording the inputs,
the score, the decision and the reason -- including the reason it did not fire,
and including a failure to open the tick.

## What was measured

Against a fork of the live history on 2026-07-28 (the fork machinery from PR-6,
so the live database was never written to):

| | first step | second step, +7 days |
|---|---|---|
| elapsed | 0 s | 604 800 s |
| decay factor | 1.0 | 0.500 |
| stat rows decayed | 0 | 59 |
| protentions expired | 482 | 89 |
| silence | 16 s | 604 816 s |
| ignition score | 0.481 | 0.880 |
| ignited | no -- "a field was committed 16s ago; the floor is 900s" | yes |

Total observation mass went from 427.0 to 214.5 across the week: half, plus the
1.0 held by rows already sitting at the floor. Pending protentions went 571 → 0;
the live history had accumulated 571 imaginings that nothing had ever retired,
because until now nothing could.

The fork ended with one `autonomous` field. The live database still has zero.

## Running it without a conversation

Until 0.4.3 the daemon was reachable only through the `organism_step` MCP tool,
which means a language model had to call it. A clock whose entire purpose is to
move when nobody is talking could therefore only be wound by the one thing it
was supposed to be independent of. `efpf-hook organism-step` is the same step,
callable by a scheduler:

```
*/15 * * * * /path/to/uv run --directory /path/to/embodied-claude \
  efpf-hook organism-step < /dev/null >> ~/.claude/autonomous-logs/organism.log 2>&1
```

Two details that a crontab entry gets wrong easily, both learned from the
`desire-updater` entry that sat broken for months: give `uv` an absolute path,
because cron's PATH does not include `~/.local/bin`, and use `--directory`
rather than `cd`, because a `cd` to a path that stops existing makes the whole
line fail silently -- `&&` short-circuits before the redirect, so not even the
log records that nothing ran.

The command respects the flag: with `organism_daemon = false` it returns
`{"ran": false}` and touches nothing, so the entry can be installed before the
flag is flipped and will simply idle until it is.

## Why the flag still ships off

Unlike the flags in #111, the blocker here is not evidence -- the measurement
above is the evidence. It is that the first real run against live history will
expire 571 pending trajectories and halve week-old counts, and that is a change
to someone's recorded history rather than a new table. It also has no scheduler
yet, so the flag alone does nothing: something has to call `organism_step`.
Both of those are decisions for whoever owns the history, not defaults to
assume.

## The quiet-hours gate

`SleepConsolidator` defaulted to a predicate that returned `True` at every hour,
so the gate it advertised was open all the time. The served consolidator now
asks `BoundaryStore.get_quiet_mode_state`, which is the same `[global]
quiet_hours` and `timezone` that already govern speaking and posting. A
consolidator constructed without a boundary store keeps the permissive
fallback: it has no way to know the hour, and refusing every run would be a
silent failure rather than a safe one.

## Limits

The desire snapshot is still written by an external updater that stopped
running. `project_desires` extrapolates from it, so with a snapshot months old
every need saturates and `max_discomfort` is effectively pinned at its maximum
(0.8 in the measurement above). The ignition mechanism works; the need term it
reads is currently less informative than it should be. Moving the writer inside
the kernel is the remaining half of that problem.

Ignition does not consult quiet hours. Opening a tick is not an outward action,
and a runtime should be able to think at 3am; what it may then *do* is the
boundary gate's business. If that turns out to be wrong in practice, the place
to fix it is `should_ignite`, which already takes its thresholds as arguments.
