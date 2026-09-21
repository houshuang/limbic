# `limbic.cerebellum.cost_log` — cost per useful change, not cost per call

One SQLite ledger, one row per provider call. `cached_call` and both built-in
transports write to it without being asked; the parts you add by hand are the
price table and the *outcome*.

Default path `~/.local/share/limbic/llm_costs.db`, overridden by `$COST_LOG_DB`.
`cost_log` is a module-level singleton that resolves that path at **import**
time — see the warning at the end before you write a test.

## Ten lines

```python
from limbic.cerebellum.cost_log import cost_for, cost_log, price_for

print(price_for("gpt-5.4-mini"))                              # (0.75, 4.5) USD / 1M tokens
print(round(cost_for("gpt-5.4-mini", 10_000, 2_000, 8_000), 6))   # 0.0111

rec = cost_log.log(project="demo", model="gpt-5.4-mini", purpose="code_spans",
                   prompt_tokens=10_000, completion_tokens=2_000, cached_tokens=8_000)
cost_log.set_packet_id(rec.id, "pkt-0001")
cost_log.record_outcome(rec.id, "applied", detail="2 fields written")

print(cost_log.multi_group_summary(by=["project", "purpose", "outcome"]))
# [{'project': 'demo', 'purpose': 'code_spans', 'outcome': 'applied', 'calls': 1,
#   'cost_usd': 0.0111, 'cache_hit_rate': 0.0, 'applied_count': 1,
#   'cost_per_applied': 0.0111, ...}]
```

`cost_per_applied` is the number the ledger was rebuilt to produce. Without an
outcome you can compute cost per call and never cost per useful change, and the
audit's central inversion — 12.3k lines of machinery around **0** writes, while
a plain deterministic join next door produced 2,336 of 2,342 proposals — is
invisible in a cost-per-call report and obvious in this one.

## `outcome`, and why it is set afterwards

A call's cost is known when it returns; whether it was *worth* anything is known
later, once the caller has tried to use the result. So `outcome` is a second
write against the row id:

```python
cost_log.record_outcome(call_id, "applied")   # applied | no_op | rejected | held | error
```

`record_outcome` and `set_packet_id` both return `False` rather than raising
when the row does not exist — including when `call_id` is `None` because the
original ledger write failed. That is deliberate (a billed response must never
be lost to bookkeeping, see [`docs/calls.md`](calls.md)), but it does mean an
outcome can be silently dropped for a call whose row never landed. Check the
return value if you are counting.

`packet_id` ties one packet's whole history together — dry run, real call,
post-truncation halves — so a batch can be replanned against what it already
cost. `run_packets` sets it for you.

## Prices

| function | |
|---|---|
| `price_for(model, *, strict=True)` | `(input, output)` USD per 1M tokens |
| `cached_input_price_for(model, *, strict=True)` | the provider's cached-input rate |
| `cost_for(model, prompt_tokens, completion_tokens, cached_tokens=0, *, strict=True)` | total USD |
| `compute_cost(model, prompt, completion, cached=0)` | same, returns `None` for an unknown model |

`strict=True` (the default) raises `UnknownModelPriceError` for an unpriced
model. That is the point of it: the silent-$0 path let one project carry a price
table 5–7× too low for months, and nothing in a report looked wrong.
`strict=False` opts back into best-effort — which is what the built-in
transports use, so an unpriced model logs a visible $0 with a warning rather
than failing a call that has already been paid for.

`cost_for` bills `cached_tokens` at the provider's cached rate rather than the
full input rate. OpenAI and Gemini list prices were read 2026-09-21. **Anthropic
is deliberately absent from the cached table**: its cache has a write surcharge
that a single cached-token count cannot express, so pricing it from one number
would be wrong in a way that looks right. Rows already in the ledger are never
repriced.

`cost_log.log(ts=...)` accepts an explicit timestamp so a backfilled call keeps
its own date instead of the date you noticed it was missing.

## Reading it back

```bash
python -m limbic.cerebellum.cost_log report --by project,purpose,outcome --since 7d
python -m limbic.cerebellum.cost_log report --group-by model --days 30
python -m limbic.cerebellum.cost_log dashboard        # local web view
```

In Python: `query(...)`, `summary(group_by=...)`, `multi_group_summary(by=[...])`,
`total(days=...)`. `merge_from(path)` folds another machine's ledger in;
`sync_from_remote(host, remote_db)` fetches one over ssh first.

## What the ledger cannot see: `cerebellum.forensics`

Every row above covers a call made *through* limbic. A human or a coordinator
driving Claude Code or Codex interactively is invisible to it — and that is
where most of the spend actually is. `forensics` is a read-only scan over those
session JSONL files.

```python
from limbic.cerebellum.forensics import parse_since, scan_claude_sessions

for s in scan_claude_sessions(since=parse_since("1d")):
    if not s.subagents:
        continue
    main, side = s.totals("main"), s.totals("sidechain")
    print(f"{s.cwd}  main={sum(main.values()):,}  subagents={sum(side.values()):,}"
          f" ({len(s.subagents)})")
    for sub in s.subagents[:1]:
        print(f"    entrance fee {sub.first_turn_context:,}  {sub.model}")
```

Real output from one day of this machine's sessions:

```
/Users/stian/src/research  main=19,848,168  subagents=179,619,595 (22)
    entrance fee 61,449  claude-sonnet-5
/Users/stian/src/nrk       main=11,820,737  subagents=2,157,058 (1)
    entrance fee 19,238  claude-opus-5
```

That shape is the finding. In the first session, delegated work cost **9× the
main thread**, and each of the 22 subagents paid tens of thousands of tokens to
establish its context before doing any work at all — the audit measured a 49.8K
median entrance fee. A ledger that only sees `cached_call` rows reports none of
this.

`totals(which)` takes `"main"`, `"sidechain"` or `"all"` and returns a dict of
`input_tokens`, `cache_creation_tokens`, `cache_read_tokens`, `output_tokens`
and `requests`. `scan_codex_sessions` is the Codex equivalent;
`attrib_session(path)` breaks a single session down by category; the CLI is
`python -m limbic.cerebellum.forensics {claude,codex} --since 30d`.

Three counting rules are baked in, each silently wrong if skipped:

1. **Never read a session's cumulative token counter.** A forked Codex subagent
   starts already carrying its parent's total, so the last cumulative value
   overcounts — a first pass at this reported 2.66B and 6.7B input tokens for
   two projects against corrected figures of 0.85B and 1.06B. A *resumed*
   session errs the other way, resetting its counter while the replayed history
   still costs real input tokens. Both are fixed by summing deltas over events
   whose total actually changed.
2. **Claude Code JSONL repeats one `message.id`** across the lines of a
   streaming response. Summing `message.usage` without deduping by id multiplies
   the count by however many chunks it took.
3. **Claude Code subagent transcripts are separate files**, under
   `<session-id>/subagents/agent-*.jsonl` — not `isSidechain` lines inside the
   parent transcript. A scan that reads only the main file reports zero subagent
   tokens for a session that spawned a dozen.

A full scan is not cheap: 680 sessions took about 100 s here. Pass `since=`.

`scan_claude_sessions` and `scan_codex_sessions` default to
`~/.claude/projects` and `~/.codex/sessions`; pass `root=` for anywhere else.
The `known_projects` argument on `scan_codex_session(s)`, `project_from_cwd`
and `project_from_mentioned_paths` defaults to a hard-coded list of this
author's own project names — pass your own.

## Writing a test against any of this

`cost_log` resolves `$COST_LOG_DB` when the module is first imported, so a
fixture that sets the variable has already lost. `tests/conftest.py` does two
things, in this order, and a consumer's test suite should copy both:

1. Set `COST_LOG_DB` and `LIMBIC_CALL_CACHE_DB` at **conftest module import**,
   before pytest imports any test module.
2. Wrap `sqlite3.connect` to raise on any path under `~/.local/share/limbic`.

Describing the rule in a docstring is what was there before. On 20 Sep 2026 two
agents polluted the real ledger by running the suite. The guard is load-bearing,
so it has its own tests in `tests/test_ledger_isolation.py`.
