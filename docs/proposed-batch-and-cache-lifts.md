# Proposed: narrow cache keys, and partial-retry batch validation

Two mechanisms found in a consumer's nightly pipeline that look general. Neither
is built here yet; this is the design, and what each would cost.

## 1. Declaring which fields the cache key depends on

`cached_call` already has both hooks: `version=` (folded into `_cache_key`
alongside model, system, prompt and schema hashes) and `cache_key=` (replaces
the derived key outright). So the missing piece is smaller than it first looks —
not a key redesign, just a helper for the projection:

```python
def content_key(payload, fields, *, normalise_whitespace=True) -> str: ...
```

Hash a whitespace-normalised, sorted-key JSON projection of *only* the named
fields, and pass the result as `prompt=` or `cache_key=`. The point is what it
leaves out: a timestamp, a signed image URL or a coordinate that moves between
runs busts a cache whose answer could not have changed. The observed version
stamp additionally folds in reasoning effort and the tool contract, which
`_cache_key` does not cover — those belong in `version=`, not in a new column.

**Constraint:** `_cache_key`'s payload must not change. Existing paid caches in
consumers are addressed by today's key, and a reshuffle silently re-buys every
stored answer. A test should pin the key for a fixed call and fail if it moves.

## 2. `validated_records` — partial success as a first-class result

Structured-output verification for a *batch*, returning
`(accepted_by_key, still_missing, errors)`:

- an answer whose identity is not in the asked-for set never enters the cache;
- **two answers for one key are rejected**, rather than arbitrarily keeping one —
  a real hallucination signal that `resolve.unslot()` and `packet.validate_result`
  both currently have no name for;
- a key with no answer comes back as *missing*, not as an error, so the caller
  retries exactly the failures instead of re-buying the whole batch.

`resolve.unslot()` treats an unknown id as a hard problem, and
`packet.validate_result` checks schema, id membership and citation spans but not
duplicates. The natural home is `cerebellum.packet`, beside `validate_result`,
generic over the identity and per-record validator callables. Worth doing when a
second consumer needs it; one caller does not yet justify the surface.
