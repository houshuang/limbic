# `limbic.cerebellum.calls` — one call, one cache entry, one ledger row

`cached_call` is the single entry point every provider call goes through. It
caches on the exact inputs, writes one `cost_log` row per call including
failures, and refuses the two attribution mistakes that made the 20 Sep 2026
`llm-pipeline-audit` unable to answer "what did this stage cost".

What the audit found, which is the whole reason this module exists: 17% of one
project's hashed calls (about $178 notional) and 76% of one extraction stage
repeated an identical prompt+system+model and paid again each time — the hashes
were already being computed, nothing read them back. `purpose` was empty on
~40% of ledger rows, and 25,642 rows were attributed to the literal project
`"limbic"` by an env-var default in `amygdala.llm.generate_structured`.

## Ten lines

```python
from limbic.cerebellum.calls import cached_call

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}}

def fake(prompt, *, project, purpose, system, schema, model, **kw):
    return {"label": "positive"}, {"cost": 0.0012, "model": model}

result, meta = cached_call("Classify: I love it", project="demo",
                           purpose="sentiment", schema=SCHEMA, transport=fake)
print(result, meta.cache_hit, meta.cost_usd)   # {'label': 'positive'} False 0.0012

again, meta2 = cached_call("Classify: I love it", project="demo",
                           purpose="sentiment", schema=SCHEMA, transport=fake)
print(again, meta2.cache_hit, meta2.cost_usd)  # {'label': 'positive'} True 0.0
```

A hit costs $0 and still writes a ledger row, with `cache_hit=1` and the
original call's cost in `metadata.original_cost_usd` — so a cache is visible as
saved money rather than as a hole in the record.

`transport=` is `"claude_cli"` (default), `"openai"`, `"gemini"`, or any
callable with the signature above. Passing a fake, as here, is how you test a
pipeline without spending anything.

## Two arguments that are not optional

- **`purpose` is required.** No default, `ValueError` if empty. A row with no
  purpose cannot be traced back to a pipeline stage.
- **`project` is inferred, never defaulted.** Empty means "read the enclosing
  git repo's directory name"; outside a git repo it raises rather than guess.
  The silent fixed default is what produced the 25,642 unattributable rows.

## The cache key

Derived from `(model, system, prompt, schema, version)` — each hashed, then
hashed together. `version` is yours to bump when the prompt means something new
but its bytes did not change enough to matter.

| argument | effect |
|---|---|
| `cache=True` | default: read and write |
| `cache=False` | bypass entirely |
| `cache="refresh"` | force a fresh call, overwrite the entry |
| `ttl_days=n` | entry expires after `n` days; `None` (default) never expires |
| `cache_key="…"` | replace the derived key with your own |
| `cache_db_path=` | default `~/.local/share/limbic/llm_cache.db`, or `$LIMBIC_CALL_CACHE_DB` |

The SQLite write lock is never held across the model call: read, release, call,
write. Three separate connections, by construction.

## `request=` — when the caller owns the bytes

Pass a fully built provider body instead of `prompt`/`system`/`schema`. The
exact bytes are posted, the response cache is keyed on those bytes, and the
result is the provider's raw response dict rather than extracted text.

```python
from limbic.cerebellum.calls import cached_call, canonical_bytes

body = {"model": "gpt-5.4-mini", "input": "Code this page", "max_output_tokens": 256}

def fake(prompt, *, project, purpose, system, schema, model, request=None, **kw):
    assert request == canonical_bytes(body)     # what you built is what is posted
    return {"id": "resp_abc", "output": []}, {"cost": 0.02, "model": model}

raw, meta = cached_call(request=body, project="demo", purpose="code_spans",
                        transport=fake)
print(raw["id"], meta.request_sha256[:12], meta.model)   # resp_abc 6a80881658… gpt-5.4-mini
```

A dict is serialised once by `canonical_bytes` (sorted keys, no whitespace,
UTF-8 without ASCII escaping); `bytes` are taken as they are. `model` is read
out of the body. `meta.request_sha256` is set and the three prompt/system/schema
hashes are empty strings.

This mode exists because of a specific migration failure. skard rejected
`cached_call` in its first form because the transport rebuilt the request body,
which would have changed the hash its paid artefacts were addressed by —
stored responses would no longer have been findable. With `request=`, a
consumer that already addresses its purchases by a request hash can adopt the
ledger and the transports without invalidating anything it has bought. After
the change, request bytes were identical on 850 of 850 stored passes.

It also preserves provider-side prompt caching, which keys off a byte-identical
prefix: rebuilding the body is exactly what breaks it.

Constraints: `request=` cannot be combined with `prompt`/`system`/`schema`
(ValueError), and cannot be combined with `replicates` — give each replicate its
own `cache_key` instead.

## Replicates: disagreement is a signal, not an error

```python
from limbic.cerebellum.calls import Held, cached_call

answers = iter([{"dup": True}, {"dup": False}, {"dup": True}])

def flaky(prompt, *, project, purpose, system, schema, model, **kw):
    return next(answers), {"cost": 0.001, "model": model}

result, meta = cached_call("Same pair?", project="demo", purpose="dedup",
                           transport=flaky, replicates=3, agree=3)
print(isinstance(result, Held), result.reason)   # True  2/3 replicates agreed, needed 3
print(round(meta.cost_usd, 4))                   # 0.003 — all three were billed
```

`replicates > 1` disables the cache and makes that many independent calls.
`agree` defaults to `replicates` (unanimous). Below it you get a `Held` in the
result position — not an exception — carrying `reason`, `results` and `metas`,
so the caller can route it to a human. Results are compared as canonical JSON,
so dicts compare structurally.

Why hold rather than take the majority: the audit measured three independent
reads of the same input agreeing only 62.8% of the time, while *exact agreement
between two reads* lifted precision from 0.71 to 0.97. Agreement is a far better
signal than any single call's confidence score, and the disagreements are the
items worth a person's attention.

## `CallMeta`

| field | |
|---|---|
| `call_id` | ledger row id — `None` if the row could not be written (see below) |
| `cache_hit` | |
| `cost_usd` | this call, or the sum across replicates |
| `model` | as resolved by the transport, which may differ from what you asked for |
| `cache_key` | |
| `prompt_sha256`, `system_sha256`, `schema_sha256` | empty when `request=` was used |
| `request_sha256` | set only when `request=` was used |
| `replicate_metas` | per-replicate transport metadata, or `None` |
| `raw` | transport-native metadata: `response_id`, `duration_s`, token counts, … |

## A bookkeeping failure never loses a billed response

Once the provider's response is in hand it has been paid for, so nothing in the
accounting path is allowed to take it down with it. The ledger write in both
built-in transports, in `_log_call`, on a cache hit, and the response-cache
write all warn through `logging` instead of raising. You get the response; you
get `CallMeta.call_id is None` when the row could not be written.

That is recoverable rather than lost: the provider's response id is in
`meta.raw["response_id"]` and is enough to backfill the row. The invariant came
out of skard's adoption, whose runner tests it directly.

## Built-in transports

Both `openai` and `gemini` are stdlib `urllib` only — no provider SDK. (The
`google-genai` SDK is deliberately avoided: it crashes on import on arm64-macOS
Python builds in at least one environment here, and a REST call is all this
needs.)

Both self-log to `cost_log` and return `call_id` in their metadata, which
`_log_call` checks so that `cached_call` does not write a second row carrying
the same cost — without that check, every reported total for a self-logging
transport would be doubled. A bespoke callable that does not self-log has no
`call_id`, so `cached_call` logs on its behalf. Either way the caller gets a
real row id usable with `record_outcome`.

Both price the call through `cost_for(..., strict=False)`, so an unpriced model
logs a visible $0 with a warning rather than failing the call itself. On a
failed request both write an `outcome="error"` row before re-raising
`TransportError`.

The OpenAI transport records the response's own `id` in the ledger row and the
cache entry. OpenAI retains a response for 30 days by default, so a result lost
locally — crashed before the caller persisted it — can be fetched back with
`GET /v1/responses/{id}` instead of re-bought. Gemini's `generateContent` is
stateless with no fetch-by-id endpoint, so its `responseId` is recorded for
provenance only.

## Gotchas

- **`model="haiku"` is the default**, which suits the `claude_cli` transport and
  almost certainly not yours. Set it explicitly.
- **A schema that varies per call defeats provider prompt caching.** The schema
  is rendered *ahead* of the input in the cached prefix, so a per-item `enum` of
  that item's own candidate IDs changes the prefix every call: measured 0%
  cached input over six calls with an otherwise identical 5,962-token prefix,
  against 58% for the same batch with a fixed slot enum. Use
  `hippocampus.resolve.slot_enum` and validate the returned id yourself.
- **Tests must not touch the real ledger.** `cost_log` is a module-level
  singleton that resolves its path at *import* time, so setting `COST_LOG_DB`
  in a fixture is already too late. `tests/conftest.py` sets it at module import
  and wraps `sqlite3.connect` to refuse anything under `~/.local/share/limbic`.
  Copy that pattern; two agents polluted the production ledger on 20 Sep 2026
  before it existed.

Cost accounting for these rows is [`docs/cost-log.md`](cost-log.md). Running a
whole batch through `cached_call` with budgets and a yield probe is
[`docs/packet.md`](packet.md).
