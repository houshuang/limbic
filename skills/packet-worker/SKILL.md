---
name: packet-worker
description: "Apply a codebook, extract structured records, or classify across N documents without spending agent tokens on the work itself. Use when the task is 'code/extract/classify/enrich these documents' at a scale above a handful — building a stateless packet runner with deterministic candidate retrieval, fixed slot enums, exact-quote validation, hard budgets and a yield probe. Triggers on 'apply this codebook', 'extract claims from', 'classify these', 'enrich these records', 'run this over the corpus', 'batch campaign'."
---

# Packet worker

**Never spawn agents as coders.** One traced 25-page packet cost 3.8M tokens as
a tool-using subagent and ≈40K as one stateless structured-output call — 95×,
same model, same work. Your job is the codebook, the packet boundaries, the
adjudication of holds and the sampled QA. The coding itself is an API call.

Everything here is `limbic.cerebellum.packet` and `limbic.hippocampus.resolve`;
read `docs/packet.md` and `docs/resolve.md` before writing a runner.

## Order of work — do not reorder

1. **Retrieve deterministically first.** Build a candidate index over the local
   KB (`resolve.build_index`), then `resolve.text_candidates` per span. If a
   plain join already produces the records, stop: in one campaign 2,336 of
   2,342 proposals came from the join and 384 model calls produced 6.
2. **Probe 50 items before building anything.**
   `probe(packets, n=50, yield_fn=…, min_yield=…, execute=True)`. `yield_fn`
   counts *actionable* outputs, not items returned. 12.3k lines of sealed
   machinery, 26 prompt versions and 13 per-packet test files were built around
   a stream that produced 0 writes because nobody did this. Low yield means a
   deterministic join, not a better prompt.
3. **Then** write the runner: `make_packet` → `lint_packet` → `run_packets`.
4. Validate, union the passes, propose. Apply through
   `hippocampus.apply.apply_proposal`, never by writing the record yourself.

## Packet rules

- **Fixed slot enums, never per-item enums.** A per-item enum of that item's
  candidate IDs changes the JSON schema every call, and the schema sits ahead
  of the input in the provider's cache prefix: measured 0% cached input; a
  fixed `c01..cNN` enum on the same batch measured 58%. Use
  `resolve.slot_enum(cards, n_slots)` and `resolve.unslot` to map back. It also
  removes the last place the model handles an identifier of yours — asking a
  model to echo your hashes made valid answers fail.
- **"None of these" is always in the enum**, and it is a wanted answer. Scoring
  coverage without it produced 130 fabricated biographies.
- **Run `lint_packet(packets)` and act on every line.** It flags a varying
  schema, a shared prefix under the ~1,024-token provider cache minimum (below
  it, caching billed 92.5% of one campaign's input as cache writes and cost
  9.7% *more*), derivable body fields (46% of one packet was an identical
  `evidence_fields` list) and packets past ~25 items, where models start
  dropping items silently.
- **Freeze the prefix with the packet.** `input_sha256` covers the prefix hash;
  editing a shared mutable prefix file later silently invalidated a batch that
  had already been paid for.

## Budgets are refusals

`run_packets(max_calls=…, max_tokens=…)` stops and reports what is left rather
than finishing over budget. `execute=False` is the default: dry-run first,
always — it prices the batch for free.

**Truncation: split once, never re-ask.** Pass `split=`; re-asking the same
packet spends the same tokens on the same overflow.

## Separate passes for separate jobs

Do not fold a second job into the coding call. Adding name-accounting to one
coding call **lost 66 known entities**. Run it as its own pass and merge with
`union_passes` — the candidate pass knows the entities the KB holds, the name
pass knows the unknown ones, and neither is a superset.

## Two checks that catch what the model cannot self-report

- **Exact quotes.** `validate_quotes(items, pages)` — whitespace collapse and
  nothing else. No spelling normalisation, no OCR repair. If the page says
  "av av", the quote says "av av".
- **A deterministic recall scan.** `unmatched_names(text, known_labels,
  corpus_lowercase_words(corpus))` surfaces name-like strings no candidate
  covers, and the model must account for each one — code it or dismiss it with
  a reason. A candidate lookup can only offer entities the KB already holds; a
  corpus run printed "Harald Hårfagre" and returned nothing because nothing put
  the name in front of the model. Use the corpus's own lower-case vocabulary as
  the common-noun filter: a word the corpus writes lower-case somewhere
  ("Dessuten", "Stoffet") is a common noun; "Hårfagre" never appears that way.

## Report what is NOT known

Every batch ends with a coverage block that states the denominator, the held
items and *why*, the unaccounted names, and what the run could not see at all
(unquotable sources, truncated candidate lists, pages excluded for poor OCR).
A coverage number without a residual is a completeness claim you did not check.

## Rerun discipline

- **Print the headline number before and after, every time.** A rerun that does
  not state the delta is how a five-month-old link corruption stayed invisible
  while every structural check passed.
- **Disagreement holds, it never tie-breaks.** `replicates=2, agree=2`: three
  reads of one input agreed only 62.8% of the time, but exact agreement between
  two lifted precision 0.71 → 0.97. A confidence score of 0.88 means nothing.
- **Protect paid artefacts.** Before replanning, check whether every packet of
  the plan already has a cached response, and require an explicit
  `--discard-paid` flag to throw one away. Store the provider `response_id`
  (retained 30 days) so a locally lost result is refetched, not re-bought.
- **Every call is ledgered**, failures included, with `purpose`, `packet_id`
  and an `outcome`. Without `outcome` you can compute cost per call and never
  cost per useful change: `report --by project,purpose,outcome`.
