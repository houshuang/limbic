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
read `docs/packet.md` and `docs/resolve.md` before writing a runner, and
`docs/refuse.md` + `docs/audit.md` before writing an apply.

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
4. Validate, union the passes, propose. Run the QA loop below. Apply through
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

## QA loop — anything that can be a guard is a guard

A rule you wrote into the prompt is not enforced by having been written there.
**Any instruction telling the model not to do X is also a check for X you have
not written yet** — "don't invent a year", "don't describe your own citation",
"don't answer with the item number" were all in one prompt, and all three
happened. Every item below is a call, not a habit; the skill only says when.

**On every output that is prose** (a rendering, a definition, a summary — any
pass where there is no schema to fail):

- `packet.slot_echo_refusal(text, slot_ids, source=…)` — the output is not a
  sentence at all. Ten records reached a published graph defined as `i01`.
- `packet.meta_leak_refusals(text, phrases=…)` — it describes your evidence
  instead of the subject. Curate `phrases` from what your audit found; the
  shipped default is English and deliberately small.
- `packet.rendering_fidelity_refusals(source, rendering, exonyms=…, fold=…)` —
  it added a year or a name the source does not contain.

**Before any write, in the dry run and the real run, from the same line:**

- `refuse.schema_refusals(records, schema, only=…)`, with the semantic guards
  (`temporal_plausibility_refusals`, `dates_disagree`, the prose refusals
  above) passed as `extra_checks`. `only` filters records, never runs — a
  check that runs on `--execute` only is how a dry run reports a clean apply.
- `refuse.expect(changed=N)` around the decision loop; write after the block.
  Decide inside, commit outside, and a miscount raises before the write.

**Any batch that decided something about the world** — as opposed to rendering
or wording it — gets **one independent audit pass**, demote-only:

    from limbic.hippocampus.audit import apply_audit
    decisions, report = apply_audit(decisions, audit, key_fields=(...), hold_sections=(...))

Call it; do not hand-roll the fold-in. Four Kulturbase campaigns on 21 Sep 2026
each hand-transcribed the fold-in instead; replaying their raw auditor output
through `apply_audit` reproduced every final verdict (48/48, 18/18, 8/8, 74/74),
so the hand work bought nothing — and the one campaign that wrote its own code
drifted the verdict vocabulary away from `right`/`wrong`/`cannot_tell` with
nothing to catch it. Brief the auditor per `docs/blind-audit.md` (different
model family, blind view, full set when cheap, that vocabulary, non-`right`
rows only) **after reconciliation**. An auditor that may promote is a second
proposer: you only hear from it when it agrees.

**A per-item audit cannot see set-level invariants** (one authority id per
entity, every entity reachable). Before approval, simulate the whole batch
against the project's own validator; do not let the commit hook be the first
thing that runs it.

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
- **Protect paid artefacts.** Still prose, and marked as such: before
  replanning, check whether every packet of the plan already has a cached
  response, and require an explicit `--discard-paid` flag to throw one away.
  Store the provider `response_id` (retained 30 days) so a locally lost result
  is refetched, not re-bought. Design for the missing guard:
  `limbic/docs/proposed-paid-artefact-registry.md`.
- **Every call is ledgered**, failures included, with `purpose`, `packet_id`
  and an `outcome`. Without `outcome` you can compute cost per call and never
  cost per useful change: `report --by project,purpose,outcome`.
