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
report = probe(packets, n=50, yield_fn=lambda r: len(r["items"]),  # 50 packets BEFORE machinery
               project="skard", purpose="code_spans", transport="openai",
               execute=True, min_yield=0.2)                     # raises LowYield if it is not worth it
result = run_packets(packets, project="skard", purpose="code_spans", transport="openai",
                     model="gpt-5.6-luna", max_calls=200, max_tokens=2_000_000,
                     split=halve, execute=True)
```

`execute=False` is the default on `run_packets`. A dry run prices the batch and
returns what it *would* send — the cheapest thing you can do before committing.
(`probe` has no `execute` of its own; it passes yours through. See below.)

## The order that matters

1. **`probe(n=50)` first.** The audit's clearest inversion: 12.3k lines of
   sealed machinery, 26 prompt versions and 13 per-packet test files around a
   model stream that produced **0 writes** — while the campaign next to it got
   2,336 of 2,342 proposals from a plain deterministic join (384 model calls
   produced 6). Nobody had run a yield probe. `yield_fn` counts *actionable*
   outputs, not items returned; `min_yield` makes the probe refuse rather than
   report.

   `n` is a number of **packets**, stratified across the batch — not a number
   of items. And a probe has to actually spend: it forwards `**run_kwargs` to
   `run_packets`, where `execute=False` is the default, so a dry-run probe
   returns zero results, computes a 0.00% yield, and raises `LowYield` every
   time. Pass `execute=True` to a probe you mean.
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
  a list that repeats one value. Both warnings have a floor, so a small constant
  is not nagged about: the constant-across-batch check needs a canonical value
  over **200 characters**, and the repeated-value check a list of **more than
  three** elements. A five-element identical `evidence_fields` therefore trips
  the second warning and not the first.
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

## Evidence: `validate_quotes`, and an anchor that survives a reflow

`validate_quotes(items, pages)` is an exact substring check. Whitespace is
collapsed on both sides and **nothing else** — no normalising of spelling, no
expanding of abbreviations, no repairing of OCR: if the page says "av av", the
quote must say "av av". Case is significant. An exact-substring check is what
makes invented evidence unrepresentable rather than discouraged.

It answers "is this quote really on that page", and nothing else. When the page
text can change under you — a re-extraction, a different OCR pass — you want the
quote to still be *locatable*, which is `text_quote_anchor`:

```python
from limbic.cerebellum.packet import reanchor_quote, text_quote_anchor

anchor = text_quote_anchor("The page says av av here.", "says av av", "p1")
# {'type','page_id','exact','prefix','suffix','start','end','occurrence_index',
#  'page_text_sha256','selector_sha256','span_sha256','extraction_version_id'}
found = reanchor_quote({"p1": "The page says av av here."}, "says av av")
```

A TextQuoteSelector-style record: the exact string, its prefix and suffix
context (48 characters each by default), which occurrence it is, character
offsets, and hashes of the page text, the selector and the span.
`reanchor_quote(pages, exact)` finds it again in new text and returns
`(page_id, anchor)` or `None`.

Two differences from `validate_quotes` worth knowing: `text_quote_anchor`
matches **case-insensitively** (`text_quote_anchor("Some Page here", "some
page", "p1")["exact"]` is `"Some Page"`), and an **empty quote raises
`ValueError`** rather than matching at offset 0. That second one is a fix, not a
preference: three items had validated as evidence on an empty string. Use
`unresolved_text_quote_anchor(page_text, expected_exact, page_id)` to record a
quote you could *not* anchor, which is a held item rather than a match.

## Ledger

Each call aims at exactly one row, failures included, carrying `packet_id`,
`purpose`, `outcome` and (on the OpenAI transport) the provider's `response_id`
— a response is retained for 30 days, so a result lost locally can be fetched
back instead of re-bought. `outcome_fn` closes the loop: without an outcome you
can compute cost per call, never cost per useful change.

The row is not a guarantee, and deliberately so. A billed response must never be
lost to a locked or unwritable ledger, so those writes warn instead of raising;
when one fails, `CallMeta.call_id` is `None` and the later
`set_packet_id` / `record_outcome` calls return `False` rather than attaching to
anything. The response id in the transport metadata is what you backfill from.
See [`docs/calls.md`](calls.md).

```bash
python -m limbic.cerebellum.cost_log report --by project,purpose,outcome --since 7d
```

Input tokens the provider served from its own cache are billed at the cached
rate by `cost_for`, so a well-cached batch reports as cheaper rather than as
mysteriously the same. [`docs/cost-log.md`](cost-log.md) has the price table
rules.

## Paid artefacts

A packet's `input_sha256` covers the prefix hash, so editing the instructions
later cannot silently invalidate a batch that was already bought — it produces
a different packet id instead. Before replanning, check whether every packet of
a plan already has a cached response, and require an explicit
`--discard-paid`-shaped flag to throw one away.

If your paid artefacts are addressed by a hash of the *provider request body*
rather than by a packet id, build the body yourself and hand it over verbatim
with `cached_call(request=...)`: those exact bytes are posted and the response
cache is keyed on them, so adopting the ledger does not re-key anything you have
already bought. This is what let one consumer migrate with request bytes
identical on 850 of 850 stored passes. See [`docs/calls.md`](calls.md).

## Output refusals: what to do when the output is prose

A packet that renders or rewrites text — translate this definition, say this in
the other register — produces prose, and prose has no schema. Three defects
have shipped from such stages, and three different checks caught them, so they
stay three functions. One `lint_output()` would force every caller to accept
all three sets of assumptions to get any one of them.

The rule the whole episode teaches: **any instruction you write into the prompt
telling the model not to do X is also a check for X that you have not written
yet.** "Don't invent a year." "Don't describe your own citation." "Don't answer
with the item number." All three were in the prompt. All three happened.

| Function | Refuses |
|---|---|
| `slot_echo_refusal(text, slot_ids, *, source=None)` | output that is not prose at all: the item's own slot id, no space, no lowercase word, a fraction of its source. First reason only — once the answer is the item number, nothing else about it is worth reporting |
| `meta_leak_refusals(text, *, phrases=…)` | text that describes the pipeline's evidence rather than the subject ("identified only as the source"), and text whose whole content is that the thing has its name — anchored end to end, so "A ballad titled Terje Vigen describes a sailor's ordeal" stands |
| `rendering_fidelity_refusals(source, rendering, *, exonyms=…, fold=…, parts=…)` | a rendering that added a fact its source does not contain: a year not in the source, a capitalised name in neither the source nor the caller's known names, or more than `max_ratio` times the length |

```python
def refusals(english: str, rendering: str, labels: list[str]) -> list[str]:
    first = slot_echo_refusal(rendering, source=english)
    if first:
        return [first]
    return [
        *meta_leak_refusals(rendering, phrases=MY_CORPUS_PHRASES),
        *rendering_fidelity_refusals(
            english, rendering, known_names=labels, exonyms=MY_EXONYMS,
            fold=my_name_key, parts=lambda w: re.split(r"[-–—]", w)[:1],
        ),
    ]
```

The shipped defect this comes from: one packet answered every item with its own
slot id, and ten records went into a published graph *defined as* `i01`.
Nothing between the model and the store looked at whether the answer was a
sentence.

**Words stay in your project, shape comes from here.** Everything
language-shaped is injectable and almost nothing language-shaped ships: the
exonym set (the target language's own forms for names the source gives in its
own — Danmark for Denmark), `fold` (the comparison key — a language that
inflects names needs its own), `parts` (how a compound splits), and
`exempt_years` (for a language that spells a century as a four-digit number,
like bokmål's "1100-tallet"). `DEFAULT_META_PHRASES` and `DEFAULT_VACUOUS` are the only defaults, both
English-only and small on purpose: a phrase list is a corpus's vocabulary,
and one lifted from another corpus refuses good text and misses the bad.
Curate yours from what an audit actually finds.

**Fit check.** skard's 519 stored bokmål renderings, run through its own
`nb_refusals` and through the composition above: **identical verdict on
519/519**, 88 refused on each side, including all ten `i01`…`i10` slot echoes.
The refusal *wording* differs on all 88 (limbic says "output", skard says
"rendering"); the verdicts do not. Parameterised to get there: the phrase
regexes, the exonym set, `fold=name_key` (Norwegian name morphology),
`parts` (leading element of a hyphenated compound) and `exempt_years` (the
century form). One bug was found by the fit check and fixed: known names must
be tokenised like the source, not folded whole, or the label "Det gamle
Hellas" fails to make "Hellas" a known name.
