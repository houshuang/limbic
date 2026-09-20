# `limbic.cerebellum.packet` — the unit of work is a call, not an agent

A packet is a frozen, hashed, self-contained request: a byte-identical static
prefix, a variable body, a fixed schema. No conversation, no tools, no growing
context.

The measurement behind the module: one traced 25-page packet cost **3.8M
tokens** as a tool-using subagent (36 tool calls, context 27K → 146K, five
throw-away parsers, hand-repaired JSON brackets) against **≈40K** as one
stateless structured-output call. That is 95×, on the same model, for the same
work. Across the same project, 1,283M agent tokens produced curated JSON
7,000× smaller than the tokens spent on it.

## Ten lines

```python
from limbic.cerebellum.packet import lint_packet, make_packet, probe, run_packets

packets = [make_packet(PREFIX, body, SCHEMA, prompt_version="v1") for body in bodies]
print(lint_packet(packets))                                     # caching and schema warnings
report = probe(packets, n=50, yield_fn=lambda r: len(r["items"]),  # 50 items BEFORE machinery
               project="skard", purpose="code_spans", transport="openai",
               execute=True, min_yield=0.2)                     # raises LowYield if it is not worth it
result = run_packets(packets, project="skard", purpose="code_spans", transport="openai",
                     model="gpt-5.6-luna", max_calls=200, max_tokens=2_000_000,
                     split=halve, execute=True)
```

`execute=False` is the default everywhere. A dry run prices the batch and
returns what it *would* send — the cheapest thing you can do before committing.

## The order that matters

1. **`probe(n=50)` first.** The audit's clearest inversion: 12.3k lines of
   sealed machinery, 26 prompt versions and 13 per-packet test files around a
   model stream that produced **0 writes** — while the campaign next to it got
   2,336 of 2,342 proposals from a plain deterministic join (384 model calls
   produced 6). Nobody had run a 50-item yield probe. `yield_fn` counts
   *actionable* outputs, not items returned; `min_yield` makes the probe refuse
   rather than report.
2. **`lint_packet` before you spend.** Its warnings are each a bill someone
   already paid.
3. **`run_packets` with budgets that refuse.** The run stops at `max_calls`, or
   when the next call's estimated tokens would cross `max_tokens`, and reports
   what is left rather than finishing over budget.

## The caching rules the lint enforces

- **The schema must be byte-identical across the batch.** It is rendered
  *ahead* of the input in the provider's cache prefix, so a per-item `enum` of
  that item's own candidate IDs changes the prefix every call: measured **0%
  cached input** over six calls with an otherwise identical 5,962-token prefix.
  The same batch with a fixed slot enum measured **58%**. Use
  `hippocampus.resolve.slot_enum`.
- **A shared prefix under ~1,024 tokens should not carry a cache key.** In one
  35k-episode campaign, 20 unique records per packet left the shared prefix
  under the minimum, 92.5% of input was billed as cache *writes*, and caching
  **cost 9.7% more** than not caching.
- **Derivable fields are not data.** 46% of each packet in that campaign was an
  `evidence_fields` list identical on every record. The lint flags a body field
  that is byte-identical across the batch (move it to the prefix, pay once) and
  a list that repeats one value.
- **Past ~25 items per call a model drops items silently.** Cut the packet
  rather than raising the output cap.

## Truncation: split once, never re-ask

Re-asking the same packet spends the same tokens on the same overflow. Pass
`split=` — a callable returning smaller packets — and `run_packets` calls it
**at most once** per packet, queues the halves, and marks the parent
`rejected: truncated output` in the ledger. Without a split hook, truncation is
a recorded failure, not a retry loop.

## Two narrow passes beat one wide call

`union_passes(results_by_pass, key)` merges passes instead of choosing between
them. Measured: folding name-accounting into the coding call **lost 66 known
entities**. The candidate pass knows the entities the KB holds; the name pass
knows the unknown ones; neither is a superset. Deduplication is across passes,
not within one — a pass that legitimately coded the same span twice keeps both.

## Recall you can check without a model

`unmatched_names(text, known_labels, corpus_lowercase_vocab)` returns
name-like strings no candidate accounts for. A candidate lookup can only offer
entities the KB already holds, so a person it has never heard of is invisible:
a corpus run printed "Harald Hårfagre" and the coding returned neither an
assertion nor a new-entity proposal, because nothing put the name in front of
the model.

Build the vocabulary with `corpus_lowercase_words(texts)` — words the corpus
itself writes in lower case *somewhere* are common nouns. This is a far better
filter than a hand-written stopword list: "Dessuten", "Barna" and "Stoffet" all
appear lower-cased elsewhere in the same documents, while "Hårfagre" and
"Hernes" never do. Only single-word candidates are filtered this way, so a
two-word name is never lost to it.

Hand the result to the model as a list it must account for one way or the
other. That accounting is also your deterministic recall check on the run.

## Evidence: `validate_quotes`

Whitespace is collapsed on both sides and **nothing else**. No normalising of
spelling, no expanding of abbreviations, no repairing of OCR: if the page says
"av av", the quote must say "av av". An exact-substring check is what makes
invented evidence unrepresentable rather than discouraged.

## Ledger

Every call lands exactly one row, including failures, carrying `packet_id`,
`purpose`, `outcome` and (on the OpenAI transport) the provider's
`response_id` — a response is retained for 30 days, so a result lost locally
can be fetched back instead of re-bought. `outcome_fn` closes the loop:
without an outcome you can compute cost per call, never cost per useful change.

```bash
python -m limbic.cerebellum.cost_log report --by project,purpose,outcome --since 7d
```

## Paid artefacts

A packet's `input_sha256` covers the prefix hash, so editing the instructions
later cannot silently invalidate a batch that was already bought — it produces
a different packet id instead. Before replanning, check whether every packet of
a plan already has a cached response, and require an explicit
`--discard-paid`-shaped flag to throw one away.
