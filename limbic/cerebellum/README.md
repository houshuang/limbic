# limbic.cerebellum

**LLM-assisted batch verification with budget tracking, resumable state, and multi-tier orchestration.**

Cerebellum is the verification layer of limbic. It handles the operational reality of using LLMs to check thousands of records: you need to control costs, resume interrupted runs, escalate uncertain items to more expensive models, and keep an audit trail of everything the LLM said.

These patterns were extracted from kulturperler, where 2,400+ performing arts works were verified across 30+ LLM audit sessions. A two-tier setup — Gemini Flash for fast triage (~$0.001/item), Claude Sonnet for deep verification (~$0.05/item) — kept the total cost at ~$270 while maintaining high accuracy. The same problems (budget blowouts, lost progress on interruption, no way to track what was checked) appear in any project that uses LLMs for data curation at scale.

## Install

```bash
pip install git+https://github.com/houshuang/limbic.git
```

Cerebellum imports with the **standard library only** — as of 2026-09-21,
`cerebellum.calls` no longer reaches `connect` through `limbic.amygdala`, which
had been loading numpy and the embedding stack into every packet runner
(0.13–0.33 s against 0.03 s, and an `ImportError` in an interpreter without
them). This is held by subprocess tests.

The built-in `openai` and `gemini` transports are stdlib `urllib`, so they need
no extra either. The `[llm]` extra is for `amygdala.llm`, not for anything here.

The install itself is still large: the distribution declares numpy,
sentence-transformers and transformers as core dependencies, so you pay for the
embedding stack even though this package never imports it.

---

## Modules

| Module | What it does |
|--------|-------------|
| **batch** | `BatchProcessor`, `StateStore` — resumable batch processing with budget tracking and persistent state |
| **orchestrator** | `TieredOrchestrator`, `VerificationTier` — multi-tier verification with auto-escalation |
| **audit_log** | `AuditLogger`, `read_logs`, `summarize_logs` — append-only JSONL logging with daily rotation and analysis |
| **context** | `ContextBuilder`, `build_batch_context` — structured prompt building for LLM verification calls |
| **cost_log** | `CostLog`, `cost_log`, `compute_cost`, `price_for`, `cost_for`, `cached_input_price_for`, `record_outcome`, `sync_from_remote` — cross-project spend tracking with a dashboard, an outcome-bearing ledger, and a strict per-model price lookup. [`docs/cost-log.md`](../../docs/cost-log.md) |
| **calls** | `cached_call`, `Held`, `CallMeta`, `canonical_bytes` — response-cache + replicate-agreement wrapper around a generate transport, so a repeated call costs $0. [`docs/calls.md`](../../docs/calls.md) |
| **packet** | `make_packet`, `run_packets`, `probe`/`LowYield`, `lint_packet`, `validate_quotes`, `text_quote_anchor`, `union_passes`, `unmatched_names` — stateless batch units with budgets that refuse. [`docs/packet.md`](../../docs/packet.md) |
| **claude_cli** | `generate`, `generate_parallel`, `Task` — `claude -p` wrapper, every call auto-logged to `cost_log` |
| **codex_cli** | `codex_json`, `codex_research`, `strict_response_schema` — `codex exec` wrapper: locked-down structured calls, and deliberately agentic runs with web search and network egress |
| **sandbox** | `untrusted_payload`, `isolated_scratch`, `sanitized_environment`, `call_slot` — isolation primitives for handing untrusted material to an agentic CLI |
| **windowing** | `split_into_windows`, `merge_windows`, `MergeSchema` — windowed LLM extraction with a cross-window merge that preserves references |
| **forensics** | `scan_codex_sessions`, `scan_claude_sessions`, `attrib_session` — read-only token-usage forensics over interactive session transcripts, which `cost_log` never sees |

---

## Caching a repeated call (`calls.py`)

`claude_cli.generate` already computes `prompt_sha256` / `system_sha256` /
`schema_sha256` for every call; `cached_call` is what actually uses them.
A repeated `(model, system, prompt, schema, version)` call is a cache hit:
no subprocess, `cost_usd=0`, and a ledger row with `cache_hit=1`.

```python
from limbic.cerebellum import cached_call

result, meta = cached_call(
    "Classify sentiment: I love it",
    project="petrarca", purpose="sentiment",
    schema={"type": "object", "properties": {"label": {"type": "string"}}},
)
print(meta.cache_hit, meta.cost_usd)   # False, 0.0381 (first call)

result, meta = cached_call(   # identical call
    "Classify sentiment: I love it",
    project="petrarca", purpose="sentiment",
    schema={"type": "object", "properties": {"label": {"type": "string"}}},
)
print(meta.cache_hit, meta.cost_usd)   # True, 0.0

# Disagreement-as-a-signal: 3 independent reads, need 2 to agree.
result, meta = cached_call(
    "Is this a duplicate?", project="skard", purpose="dedup",
    replicates=3, agree=2, cache=False,
)
from limbic.cerebellum import Held
if isinstance(result, Held):
    print(result.reason, result.results)   # e.g. "2/3 replicates agreed, needed 2"

# Later, once you know whether the result was actually used:
from limbic.cerebellum import record_outcome
record_outcome(meta.call_id, "applied")
```

`purpose` is required and `project` is inferred from the git root when
omitted — see the module docstring in `calls.py` for why both are refusals
rather than silent defaults.

By default `cached_call` shells out to `claude -p`, the wrong transport for a
small, high-volume classification call (roughly 9-17K harness tokens of fixed
overhead per call). For a cheap worker loop, swap the transport instead of
hand-rolling the HTTP call:

```python
result, meta = cached_call(
    "Classify: I love it", project="skard", purpose="sentiment",
    schema={"type": "object", "properties": {"label": {"type": "string"}}},
    transport="openai", model="gpt-5.4-mini",   # or transport="gemini", model="gemini-2.5-flash"
)
```

Both `openai` and `gemini` are stdlib-`urllib` REST calls (no SDK import —
`google-genai` fails to import on at least one project's arm64-macOS Python
build) and self-log to `cost_log` the same as `claude_cli.generate` does;
`cached_call` detects that and does not log a second row for the same call.
Any other callable with the shape `(prompt, *, project, purpose, system,
schema, model, **kwargs) -> (result, meta)` works as `transport=` too — self-
logging is optional, but if your callable does log, return its ledger row id
as `meta["call_id"]` so `cached_call` doesn't double it.

---

## Session forensics (`forensics.py`)

`cost_log` only sees calls made *through* limbic. Interactive Claude Code /
Codex sessions — where most spend actually happens — are invisible to it.
`forensics` reads their transcripts directly, with the counting rules that
make that safe (see the module docstring for the forked-subagent,
resumed-session, and Claude subagent-file traps this avoids).

```python
from limbic.cerebellum.forensics import scan_codex_sessions, scan_claude_sessions, parse_since

codex_sessions = scan_codex_sessions(since=parse_since("30d"))
for s in codex_sessions[:5]:
    print(s.path.name, s.project_by_cwd, s.input_tokens, s.requests_over_150k)

claude_sessions = scan_claude_sessions(since=parse_since("30d"))
for s in claude_sessions[:5]:
    print(s.path.name, s.totals("main"), s.totals("sidechain"), len(s.subagents))
    for sub in s.subagents:
        print("  subagent", sub.agent_id, sub.model, "entrance fee", sub.first_turn_context)
```

A Claude Code Task-tool subagent's turns live in
`<session-id>/subagents/agent-*.jsonl` beside the main transcript, not as
inline `isSidechain` lines in it — `scan_claude_session` finds that directory
automatically and folds it into `.sidechain_by_model` and `.subagents`, with
each subagent's "entrance fee" (its first request's
input+cache-creation+cache-read total, the cost of establishing its context
before doing any useful work).

```bash
python -m limbic.cerebellum.forensics codex --since 30d --project-by cwd
python -m limbic.cerebellum.forensics claude --since 30d
python -m limbic.cerebellum.forensics codex --session <rollout.jsonl> --attrib
```

---

## Batch processing (`batch.py`)

The core building block: process items in batches, track costs, skip already-processed items on restart.

```python
from limbic.cerebellum import BatchProcessor, StateStore, ItemResult
from pathlib import Path

# State persists across runs (SQLite with WAL mode)
state_store = StateStore(Path("audit_state.db"))

processor = BatchProcessor(
    state_store=state_store,
    max_cost=50.0,    # stop when $50 spent
    batch_size=20,
)

def verify_batch(items: list[dict]) -> list[ItemResult]:
    results = []
    for item in items:
        # ... call your LLM here ...
        results.append(ItemResult(
            id=item["id"],
            status="done",     # done | error | needs_review | skipped
            cost=0.003,
            metadata={"confidence": 0.95},
        ))
    return results

result = processor.process(
    items=all_items,
    process_fn=verify_batch,
    id_fn=lambda item: item["id"],
)
# result.processed, result.skipped, result.errors, result.total_cost
```

### Key behaviors

- **Resumable:** Already-processed items (status `done`, `verified`, `applied`, `skipped`) are automatically skipped on restart. Crash mid-batch? Just restart — completed batches are preserved.
- **Budget-tracked:** Stops before the next batch if `max_cost` would be exceeded. Logs a warning at 80% budget consumption.
- **Atomic state:** `StateStore` uses SQLite WAL mode for concurrent-safe persistence. Individual item updates are single SQL upserts.
- **ETA logging:** After each batch, logs elapsed time, cost, and estimated time remaining.
- **Error isolation:** If `process_fn` raises an exception, all items in that batch are marked as `"error"` and processing continues with the next batch.

### Data model

**`ItemResult`** — Result of processing a single item:
- `id` — Item identifier
- `status` — `"done"`, `"error"`, `"needs_review"`, `"skipped"`
- `cost` — Processing cost for this item
- `metadata` — Additional key-value data

**`BatchResult`** — Aggregate result of a processing run:
- `processed` — Number successfully processed
- `skipped` — Already completed (skipped on restart)
- `errors` — Number of errors
- `total_cost` — Total cost for this run

**`BatchState`** — Persistent state (managed by `StateStore`):
- `items` — Dict mapping item ID → `{status, ts, cost, ...}`
- `total_cost` — Cumulative cost across all runs
- `batches_run` — Total batches executed
- `started_at` — ISO timestamp of first run

### StateStore

```python
state_store = StateStore(Path("audit_state.db"))

# Load state (or fresh state if DB doesn't exist)
state = state_store.load()

# Update a single item (concurrent-safe via SQLite WAL)
state_store.update_item("person/42", "done", cost=0.003, confidence=0.95)

# Get items that haven't been processed yet
pending = state_store.get_pending(["id1", "id2", "id3"])

# Status counts
counts = state_store.get_status_counts()  # -> {"done": 150, "error": 3, ...}
```

---

## Multi-tier orchestration (`orchestrator.py`)

Run items through multiple verification tiers — fast/cheap first, then expensive/thorough for uncertain items. The orchestrator handles escalation automatically.

```python
from limbic.cerebellum import TieredOrchestrator, VerificationTier, VerificationResult, StateStore
from pathlib import Path

def fast_triage(items):
    """Tier 1: Gemini Flash, ~$0.001/item."""
    results = []
    for item in items:
        # ... fast LLM check ...
        results.append(VerificationResult(
            item_id=item["id"],
            status="verified",     # or "flagged" to escalate
            confidence=0.9,
            findings=["title matches external source"],
            cost=0.001,
        ))
    return results

def deep_verify(items):
    """Tier 2: Claude Sonnet, ~$0.05/item."""
    results = []
    for item in items:
        # ... thorough LLM verification ...
        results.append(VerificationResult(
            item_id=item["id"],
            status="verified",
            confidence=0.98,
            findings=["cross-referenced with Wikidata", "dates confirmed"],
            cost=0.05,
        ))
    return results

orchestrator = TieredOrchestrator(
    tiers=[
        VerificationTier("triage", fast_triage, cost_estimate=0.001, description="Fast LLM check"),
        VerificationTier("deep", deep_verify, cost_estimate=0.05, description="Thorough verification"),
    ],
    state_store=StateStore(Path("audit_state.db")),
)

# Run all items through triage, escalate flagged items to deep verification
results = orchestrator.run(
    items=all_items,
    id_fn=lambda x: x["id"],
    max_cost=100.0,
    batch_size=20,
    escalate=True,  # flagged items go from triage → deep
)
# results: {"triage": [...], "deep": [...]}
```

### Escalation

When `escalate=True`:
1. All items go through tier 1 (e.g., fast triage)
2. Items with `status="flagged"` are collected
3. Their status is reset to `"pending"` for the next tier
4. Tier 2 processes only the escalated items
5. This continues through all tiers

Custom escalation logic:

```python
# Only escalate items with confidence < 0.8
results = orchestrator.run(
    items=all_items,
    id_fn=lambda x: x["id"],
    escalate=True,
    escalation_filter=lambda state: state.get("confidence", 0) < 0.8,
)
```

### Checking progress

```python
status = orchestrator.status(all_ids=["1", "2", "3"])
print(status.summary())
# OrchestratorStatus: .tier_counts, .total_cost, .remaining_items
# "triage: done=180, needs_review=20 | deep: done=18, needs_review=2 | cost=$12.34 | remaining=0"
```

### Data model

**`VerificationResult`** — Outcome of verifying a single item:
- `item_id` — Item identifier
- `status` — `"verified"`, `"flagged"`, `"error"`
- `confidence` — Confidence score [0.0, 1.0]
- `findings` — List of strings describing issues or confirmations
- `cost` — Processing cost
- `tier` — Which tier processed this
- `metadata` — Additional key-value data

**`VerificationTier`** — Definition of a verification tier:
- `name` — Unique tier name
- `process_fn` — Called with a list of items, returns list of `VerificationResult`
- `cost_estimate` — Estimated cost per item (for budget planning)
- `description` — Human-readable description

### Adaptive timeouts

```python
from limbic.cerebellum import timeout_for

# Base timeout scaled by item complexity
timeout = timeout_for(item, base_timeout=30, scale_fn=lambda x: len(x["text"]) / 1000)
# Clamped to max_timeout (default 1800s = 30 min)
```

---

## Audit logging (`audit_log.py`)

Append-only JSONL logs with daily rotation. Every LLM verification action gets logged for reproducibility and cost analysis.

### Writing logs

```python
from limbic.cerebellum import AuditLogger, AuditEntry
from pathlib import Path

logger = AuditLogger(Path("audit_logs/"), prefix="verify")

logger.log_entry(AuditEntry(
    timestamp="2026-03-22T10:00:00",
    item_id="person/42",
    action="verified",
    details={
        "confidence": 0.95,
        "operations": [{"type": "fix_name", "old": "ibsen", "new": "Ibsen"}],
    },
    cost=0.003,
    tier="triage",
))
# -> Written to audit_logs/verify_20260322.jsonl
```

### Reading and analyzing logs

```python
from limbic.cerebellum import read_logs, extract_operations, summarize_logs

# Read entries (supports filtering by prefix and date)
entries = list(read_logs(Path("audit_logs/"), prefix="verify", since="2026-03-01"))

# Aggregate statistics
summary = summarize_logs(entries)
# LogSummary: .total_cost, .items_processed, .error_count, .by_tier, .by_action
print(summary.total_cost)       # $12.34
print(summary.items_processed)  # 450
print(summary.by_tier)          # {"triage": {"count": 400, "cost": 0.40}, "deep": {"count": 50, "cost": 2.50}}
print(summary.by_action)        # {"verified": 420, "flagged": 25, "error": 5}

# Extract operations grouped by type (with dedup)
ops = extract_operations(entries, op_types=["fix_name", "merge"])
# -> {"fix_name": [...], "merge": [...]}
```

### Log format

Files are named `{prefix}_{YYYYMMDD}.jsonl` with one JSON object per line. Each entry has:
- `ts` — ISO timestamp
- `item_id` — Item being audited
- `action` — Action type
- `details` — Action-specific data
- `cost` — Cost incurred
- `tier` — Which tier performed this

### Deduplication in extract_operations

When the same operation appears multiple times (e.g., re-runs), `extract_operations` can deduplicate by a key function, keeping only the latest version:

```python
ops = extract_operations(
    entries,
    dedup_key_fn=lambda op: (op.get("item_id"), op.get("type")),
)
```

---

## Context builder (`context.py`)

Build structured prompts for LLM verification calls. Uses a fluent builder pattern.

```python
from limbic.cerebellum import ContextBuilder, build_batch_context

ctx = ContextBuilder()
ctx.add_entity("work", "264", {"title": "Peer Gynt", "year": 1867})
ctx.add_related("performances", [
    {"id": 1, "venue": "DNS", "year": 1972},
    {"id": 2, "venue": "Nationaltheatret", "year": 2005},
])
ctx.add_metadata("category", "teater")

# Render as markdown (for LLM consumption)
prompt = ctx.build(format="markdown")
# ## work/264
#   title: Peer Gynt
#   year: 1867
# ### performances (2)
#   - id: 1, venue: DNS, year: 1972
#   - id: 2, venue: Nationaltheatret, year: 2005
# ### Metadata
#   category: teater

# Or as JSON (for structured processing)
data = ctx.build(format="json")
```

### Batch context

```python
def build_context(item):
    ctx = ContextBuilder()
    ctx.add_entity("work", item["id"], item)
    return ctx

combined = build_batch_context(items, context_fn=build_context, format="markdown")
# Items separated by "---" dividers
```

---

## Integration with other limbic packages

### With amygdala's LLM client

```python
from limbic.amygdala.llm import generate_structured
from limbic.cerebellum import BatchProcessor, StateStore, ItemResult, ContextBuilder

async def verify_batch(items):
    results = []
    for item in items:
        ctx = ContextBuilder()
        ctx.add_entity("work", item["id"], item)
        prompt = ctx.build(format="markdown")

        result, meta = await generate_structured(
            prompt=f"Verify this entity:\n{prompt}",
            schema={"type": "object", "properties": {"correct": {"type": "boolean"}}},
            model="gemini3-flash",
        )
        results.append(ItemResult(
            id=item["id"],
            status="done" if result["correct"] else "needs_review",
            cost=meta["total_cost_usd"],
        ))
    return results
```

### With hippocampus proposals

Audit findings can automatically create proposals for human review. See the hippocampus README for the integration pattern.

---

## Cost logging (`cost_log.py`)

Centralized LLM cost tracking across projects, models, and hosts.

Pricing resolves in two steps: litellm's database (2,500+ models) **if litellm
happens to be importable**, then a built-in 24-entry fallback table. litellm is
not a dependency of limbic and is not installed by any extra, so in a default
install only the fallback table is in play. `price_for` raises
`UnknownModelPriceError` rather than pricing an unknown model at $0 — including
for cerebellum's own default alias `haiku`, which the fallback table does not
carry. See [`../../docs/cost-log.md`](../../docs/cost-log.md).

```python
from limbic.cerebellum.cost_log import cost_log, compute_cost

# Standalone logging (any SDK)
cost_log.log(project="petrarca", model="gemini/gemini-2.5-flash",
             prompt_tokens=1200, completion_tokens=340)
# Each row is a CostRecord: project, host, model, api_key_hint, prompt/completion/
# cached tokens, cost_usd, script, purpose — enough to attribute spend to a
# specific script on a specific machine, not just to a project.

# litellm callback (auto-captures every litellm.completion call)
import litellm
litellm.callbacks = [cost_log.callback("alif")]

# Query costs. query() returns sqlite3.Row objects — index them, don't dot them.
records = cost_log.query(project="petrarca", days=7)
total = sum(r["cost_usd"] for r in records)

# Built-in dashboard and CLI
# python -m limbic.cerebellum.cost_log report --days 7
# python -m limbic.cerebellum.cost_log sync --host alif
```

DB location: `COST_LOG_DB` env var or `~/.local/share/limbic/llm_costs.db`. Includes a web dashboard (`python -m limbic.cerebellum.cost_log dashboard`, port 8042) that splits API spend (billed) from Claude CLI usage (Max-plan subscription value) and from `billing_mode="subscription"` rows (Codex under a ChatGPT plan, reported as notional dollars and never summed into spend), remote sync from servers, and CLI reporting.

## CLI wrappers (`claude_cli.py`, `codex_cli.py`)

Both wrap a locally installed coding CLI rather than an API key. That is often
the cheaper path — Codex runs under a ChatGPT subscription, Claude under a Max
plan — and it is the only way to get an *agentic* run (tools, web search, a
writable workspace) instead of a single completion.

```python
from limbic.cerebellum import claude_generate, ClaudeTask, claude_generate_parallel

result, meta = claude_generate(
    prompt="Classify this sentiment: I love it",
    project="myapp", purpose="sentiment", model="haiku",
    schema={"type": "object", "properties": {"label": {"type": "string"}}},
)

results = claude_generate_parallel(
    [ClaudeTask(prompt=p, schema=SCHEMA) for p in prompts],
    project="myapp", max_concurrent=4,
)
```

Every `claude -p` invocation writes a `cost_log` row with `script="claude-cli"`,
so subscription usage shows up in the same dashboard as API spend. The wrapper
always passes `--no-session-persistence` and strips `CLAUDECODE` plus
`ANTHROPIC_API_KEY` from the child environment — the key would silently switch a
Max-plan login to metered API billing *and* break cost attribution.

```python
from limbic.cerebellum import codex_json, codex_research

# Locked down: read-only sandbox, no network, no writes. Just classify/transform.
verdict = codex_json("Is this claim supported?", schema=SCHEMA, system=RUBRIC)

# Deliberately agentic: web search + a writable workspace with network egress.
dossier = codex_research(
    "Research X. Web-search anything ambiguous. Write findings to out.json.",
    schema=SCHEMA, scratch_dir="/tmp/run",
)
```

`codex_research` is the one that follows leads, and the two config flags that
unlock it (`tools.web_search`, `sandbox_workspace_write.network_access`) are on
by default — omit both and it quietly degrades to a shallow one-shot.

Both Codex entry points also write a `cost_log` row per attempt, parsed from
`codex exec --json`'s event stream: `billing_mode="subscription"`,
`cost_usd=0`, and the API-equivalent figure in `notional_cost_usd`. Pass
`project=`/`purpose=` to attribute it; `cost_log=False` or
`LIMBIC_CODEX_COST_LOG=0` turns it off. See [`docs/cost-log.md`](../../docs/cost-log.md).

**Big prompts go down stdin, not argv.** Linux caps a *single* argv entry at
`MAX_ARG_STRLEN` — 128 KiB, regardless of the 2 MiB `ARG_MAX` the whole vector
gets — so a mission past that used to kill the call before any model ran, with
`[Errno 7] Argument list too long: 'codex'`. Anything over `PROMPT_ARGV_LIMIT`
(64 KiB of UTF-8, `LIMBIC_CODEX_PROMPT_ARGV_LIMIT`) is therefore passed as
`codex exec … -` and written to the child's stdin while its output is drained,
which no kernel limit bounds. Shorter prompts are unchanged. The ledger row's
`metadata.prompt_transport` says which route a call took.

Both calls run with `--ephemeral --ignore-user-config`, so they leave no rollout
behind and read none of the host's `~/.codex/config.toml`. That never changes
which model runs — model and reasoning are always passed explicitly — but it does
drop everything *else* the host profile sets, which can include `service_tier`,
`notify`, `personality` and any configured MCP servers. Pass `isolated=False` to
`codex_research` when a run genuinely needs those. Quota errors trip a process-local cooldown
(`mark_unavailable_from_error`) so a cron run stops hammering a depleted
allowance, and transient non-zero exits retry once while quota errors and
timeouts do not.

Both wrappers raise a typed error — `ClaudeCLIError` / `CodexCLIError` — for a
missing binary, a non-zero exit, a timeout, or unparseable output, so a batch can
distinguish "this item failed" from "the CLI is gone". `claude_is_available()`
and `codex_is_available()` check for the binary up front, which is what a nightly
job wants before it starts a thousand items.

`strict_response_schema()` (exported as `codex_strict_response_schema`) converts
a permissive JSON Schema into the shape
Codex's structured output requires: `additionalProperties: false`, every
property in `required`, formerly-optional fields made nullable.

## Agent isolation (`sandbox.py`)

`codex_research` exists to read material you do not control — scraped pages,
forwarded mail, uploaded images. Prompt injection is therefore a routine
operating condition, and these are the four guards worth having. They were
written for a nightly pipeline that ingests public event listings and email.

```python
from limbic.cerebellum import (
    call_slot, isolated_scratch, sanitized_environment, untrusted_payload,
    codex_research,
)

mission = "Extract every event announced below." + untrusted_payload(
    "scraped-page", page_html)

with call_slot(), isolated_scratch() as scratch, sanitized_environment(home=scratch):
    events = codex_research(mission, schema=SCHEMA, scratch_dir=str(scratch))
```

| Primitive | What it stops |
|---|---|
| `untrusted_payload(label, text)` | External text read as instructions. Delimits it with a content-derived nonce (so the payload cannot close its own block) and puts the refusal instruction *ahead* of the data, where later text cannot override it. |
| `isolated_scratch(files=...)` | The agent reading your repository. A private 0700 directory outside the project, with an allowlist of inputs, destroyed afterwards. Attachments belong here so a hostile image never becomes a durable file. |
| `sanitized_environment()` | An injection turning into a credential disclosure. The parent legitimately holds API keys and SMTP credentials; the child gets runtime plumbing only, and anything else needs an explicit opt-in. |
| `call_slot()` | A fan-out bursting the auth quota. A cross-process flock gate plus a persistent daily cap that survives restarts; raises `AgentBudgetExceeded` when the day is spent, `TimeoutError` when no slot frees up. |

**Two deployment preconditions**, both of which fail quietly rather than loudly:

- `sanitized_environment(home=...)` pins `CODEX_HOME` to your real `~/.codex` so
  repointing HOME doesn't move the agent's credentials with it. If you set
  `CODEX_HOME` yourself, that wins — point it at wherever the service's
  `codex login` actually wrote.
- The slot and budget files default under `tempfile.gettempdir()`. Under
  systemd's `PrivateTmp=yes` that is per-service, and on a tmpfs `/tmp` it resets
  on reboot — so "host-wide gate" and "persistent daily cap" quietly become
  neither. Set `LIMBIC_AGENT_SLOT_ROOT` and `LIMBIC_AGENT_BUDGET_PATH` to shared,
  persistent paths if you mean them literally.

`protect=` defaults to the working directory and is what the scratch root must
not live inside. A working directory of `/` (systemd's default) makes that
unsatisfiable rather than violated, so the check is skipped there — pass
`protect=` explicitly if you mean a specific tree.

**This is not an OS sandbox**, and is documented as such in the module: the child
keeps whatever process and network permissions the CLI grants it. These raise the
cost of a successful injection; they do not make one impossible. Constrain tool
and network policy separately — for Codex that is
`codex_research(web_search=..., network=...)`.

## Windowed extraction (`windowing.py`)

Asking a model to extract structured items from a long document in one call
loses most of them. Windowing recovers them; the merge is the hard part.

```python
from limbic.cerebellum import (
    Collection, MergeSchema, Reference, merge_windows, split_into_windows,
)

SCHEMA = MergeSchema([
    Collection("claims", prefix="C", dedup_field="text"),
    Collection("evidence", prefix="E", dedup_field="text",
               references=[Reference("supports_claim", target="claims")]),
    Collection("cases", prefix="CASE", dedup_field="name",
               references=[Reference("claims_supported", target="claims", many=True)]),
])

per_window = [extract(w.text) for w in split_into_windows(chapter_text)]
merged, report = merge_windows(per_window, SCHEMA, strict=False)

report.duplicates_removed    # collapsed across the seams
report.references_cleared    # refs to ids no window produced — links LOST
report.dangling              # refs surviving renumber into the wrong collection
```

Watch `references_cleared`, not `dangling`. The common failure is a window
referencing an id that no window produced; `merge_windows` clears those, so they
never reach `dangling` and an extraction quietly dropping links looks identical
to a clean one. `dangling` only catches what survives renumbering — a reference
resolving into the wrong collection.

`merge_windows` is the whole pipeline, but each step is exported for the cases
that need to interleave something: `namespace_ids(result, i, schema)`,
`dedup_by_field(items, field)` (which returns the alias map), and
`check_references(merged, schema, strict=...)` to re-verify after your own edits.
`MergeReport` carries `items_before` / `items_after` per collection,
`duplicates_removed`, the full `id_map`, and any surviving `dangling` references.

Three things go wrong if you merge naively, and `merge_windows` fixes them in a
fixed order:

1. **Ids collide.** Every window numbers its own output from `C1`, `E1`, … so
   window 2's `C1` is a different item, and its `supports_claim: "C1"` means
   *its own*. Ids are namespaced `w{i}:` **before** concatenation; concatenating
   first and renumbering later silently rewires references between windows.
2. **The overlap duplicates items** — that is what it is for, but the copies
   arrive worded slightly differently and often truncated at a window edge. Dedup
   is word-overlap against `min(|A|, |B|)`, asymmetric so a truncated restatement
   still matches, and the **longer** text wins.
3. **Dropping a duplicate orphans references to it.** Dedup returns an alias map
   (every input id → the id that survived in its place), and renumbering resolves
   references through it. Anything still unresolvable is cleared and counted,
   never left dangling.

## What's NOT in cerebellum

- **LLM client.** Cerebellum doesn't call LLMs itself — it orchestrates *your* LLM calls. Use `limbic.amygdala.llm` or any LLM client you prefer.
- **Parallel tier execution.** Tiers run sequentially: tier 1 completes before tier 2 starts. Running them in parallel (tier 2 processes previous batch's escalated items while tier 1 works on the current batch) is a planned improvement.
- **Retry strategies.** If your `process_fn` fails, the entire batch is marked as error. Exponential backoff and circuit breaker patterns are not built in — implement them in your `process_fn`.
- **Real-time dashboard.** No UI for monitoring running audits. Use the audit logs and `orchestrator.status()` programmatically.
- **Webhook notifications.** No HTTP callbacks on budget warnings or batch completion.

---

Part of [limbic](../../README.md).
