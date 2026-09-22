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

## Subscription calls: `billing_mode` and `notional_cost_usd`

A Codex call under a ChatGPT plan, or a `claude -p` call under Max, burns real
tokens and spends no money. Both facts matter, and folding them into one number
loses one of them. So a row says which kind it is:

| `billing_mode` | `cost_usd` | `notional_cost_usd` |
|---|---|---|
| `billed` (default) | the money | `NULL` |
| `subscription` | `0`, enforced | what the same tokens would cost on the API |

```python
cost_log.log(project="hvaskjer", model="gpt-5.5", purpose="enrich",
             prompt_tokens=14_169, completion_tokens=5, cached_tokens=4_480,
             cost_usd=0.0, billing_mode="subscription",
             notional_cost_usd=cost_for("gpt-5.5", 14_169, 5, 4_480))
```

`log()` refuses a subscription row with a non-zero `cost_usd`. That is the
guard the whole split exists for: every reader that predates the column — this
module's `total()`, an older limbic on another host, an ad-hoc
`SELECT SUM(cost_usd)` — keeps returning real spend without knowing anything
changed. To see the other figure you have to name it, and therefore label it:
`total_notional()`, `notional_cost_usd` in both summaries,
`notional_cost_per_applied` in `multi_group_summary`, and its own section in the
dashboard. A model with no known price logs its tokens with
`notional_cost_usd = NULL` rather than an invented $0.

Rows written before this column existed read as `billed` with no notional
figure, which is what they were.

`merge_from` copies only the columns both databases have, so a host still on an
older limbic merges fine and picks up local defaults for the rest.

### Codex

`cerebellum.codex_cli` writes these rows for you. Both `codex_json` and
`codex_research` pass `--json`, parse the per-turn usage out of Codex's event
stream, and log one row per attempt — including failed and timed-out attempts,
which is where an agent loop's worst spending hides:

```python
codex_research(mission, project="hvaskjer", purpose="enrich", scratch_dir=run)
```

`project` falls back to `$LIMBIC_CODEX_PROJECT` and then to the enclosing git
repo's name; it lands as `"unattributed"` if neither exists, rather than failing
a call that has already burned its tokens. `cost_log=False` per call, or
`LIMBIC_CODEX_COST_LOG=0`, turns capture off and drops the `--json` flag with
it. A ledger that is locked or unwritable is logged and ignored — the model's
answer is already in hand.

`metadata.prompt_transport` records how the prompt reached `codex exec`:
`"argv"` for the ordinary case, `"stdin"` once it passes `PROMPT_ARGV_LIMIT`
and has to be piped in to stay under the kernel's per-argument cap.

One assumption worth knowing: `input_tokens` is read as *including*
`cached_input_tokens`, and `output_tokens` as including reasoning tokens, which
is the OpenAI Responses convention that `cost_for` already expects. If Codex
ever changes that, every notional figure moves.

## Reading it back

```bash
python -m limbic.cerebellum.cost_log report --by project,purpose,outcome --since 7d
python -m limbic.cerebellum.cost_log report --group-by model --days 30
python -m limbic.cerebellum.cost_log dashboard        # local web view
```

In Python: `query(...)`, `summary(group_by=...)`, `multi_group_summary(by=[...])`,
`total(days=...)`, `total_notional(days=...)`. Both summaries accept
`billing_mode` as a group column. `merge_from(path)` folds another machine's ledger in;
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
