# limbic.cerebellum

**LLM-assisted batch verification with budget tracking, resumable state, and multi-tier orchestration.**

Cerebellum is the verification layer of limbic. It handles the operational reality of using LLMs to check thousands of records: you need to control costs, resume interrupted runs, escalate uncertain items to more expensive models, and keep an audit trail of everything the LLM said.

These patterns were extracted from kulturperler, where 2,400+ performing arts works were verified across 30+ LLM audit sessions. A two-tier setup — Gemini Flash for fast triage (~$0.001/item), Claude Sonnet for deep verification (~$0.05/item) — kept the total cost at ~$270 while maintaining high accuracy. The same problems (budget blowouts, lost progress on interruption, no way to track what was checked) appear in any project that uses LLMs for data curation at scale.

## Install

```bash
pip install limbic
# Cerebellum has no extra dependencies beyond stdlib.
# But you'll probably want the LLM client too:
pip install "limbic[llm]"
```

---

## Modules

| Module | What it does |
|--------|-------------|
| **batch** | `BatchProcessor`, `StateStore` — resumable batch processing with budget tracking and persistent state |
| **orchestrator** | `TieredOrchestrator`, `VerificationTier` — multi-tier verification with auto-escalation |
| **audit_log** | `AuditLogger`, `read_logs`, `summarize_logs` — append-only JSONL logging with daily rotation and analysis |
| **context** | `ContextBuilder`, `build_batch_context` — structured prompt building for LLM verification calls |

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

## What's NOT in cerebellum

- **LLM client.** Cerebellum doesn't call LLMs itself — it orchestrates *your* LLM calls. Use `limbic.amygdala.llm` or any LLM client you prefer.
- **Parallel tier execution.** Tiers run sequentially: tier 1 completes before tier 2 starts. Running them in parallel (tier 2 processes previous batch's escalated items while tier 1 works on the current batch) is a planned improvement.
- **Retry strategies.** If your `process_fn` fails, the entire batch is marked as error. Exponential backoff and circuit breaker patterns are not built in — implement them in your `process_fn`.
- **Real-time dashboard.** No UI for monitoring running audits. Use the audit logs and `orchestrator.status()` programmatically.
- **Webhook notifications.** No HTTP callbacks on budget warnings or batch completion.
