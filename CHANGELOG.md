# Changelog

All notable changes to the limbic monorepo (formerly amygdala) are documented here. Dates are in CET.

---

## 2026-09-23 -- GPT-6 Luna/Sol and Claude Opus 5.5 become the default aliases

Prices from https://developers.openai.com/api/docs/pricing and
https://platform.claude.com/docs/en/models/overview, both read 2026-09-23.

### Changed

- `luna` now maps to `gpt-6-luna` ($0.10 / $0.50 per M, cached input $0.01),
  half or less of `gpt-5.6-luna` ($0.20 / $1.20). `sol` maps to `gpt-6-sol`
  ($2 / $10, against $4 / $20). `opus` maps to `claude-opus-5-5` ($4 / $20,
  against $5 / $25). There is no GPT-6 Terra and no Claude Sonnet 5.5, so
  `terra` and `sonnet` are unchanged.
- The OpenAI HTTP transport in `cerebellum.calls` defaults to `gpt-6-luna`
  instead of `gpt-5.4-mini`. Callers that relied on the default get cache
  misses once, because the model is part of the cache key.

### Added

- `astra` (`gpt-6-astra`, $10 / $50). The previous generation stays
  addressable as `luna56`, `sol56` and `opus5`.
- GPT-6 and Opus 5.5 rows in `cost_log`'s fallback and cached-input price tables.

### Fixed

- `generate_structured` on any Claude 5 alias (`opus`, `sonnet`, `fable`)
  failed with a 400: it prefilled the assistant turn with `{`, which those
  models reject. It now asks for JSON through Anthropic structured outputs
  (`output_config.format`), closes every object in the schema with
  `additionalProperties: false` as that API requires, and reads the text
  block rather than `content[0]`, which can be a thinking block.

---

## 2026-09-22 -- Closing the gaps a fit check found in `apply_audit`

A fit check of `hippocampus.audit.apply_audit` against four real Kulturbase
blind-audit campaigns (21 Sep 2026) found `apply_audit` itself was not
missing anything those campaigns needed — replaying their raw auditor output
through it reproduced every hand-computed verdict exactly (48/48, 18/18,
8/8, 74/74) — but found three things around it that every campaign had
either reimplemented by hand or was missing outright.

### Added

- **`hippocampus.audit.blind_view`** — strips the researcher's own verdict
  (`my_verdict`, `score`, `tier`, `disposition`, ...) from records before
  they reach the auditor, and reports which fields it actually found and
  removed. Every one of the four campaigns built this by hand.
- **`hippocampus.audit.bucket_by_verdict`** — turns a raw `{id, verdict,
  reason}` auditor response into `apply_audit`-ready sections, and raises
  `AuditError` on any verdict outside an explicit `vocabulary=` (default
  `right`/`wrong`/`cannot_tell`). One campaign's raw output had drifted to
  `same_work`/`different`/`unsure`; nothing caught it at the time, and this
  is the guard that would have.
- **`hippocampus.audit.check_audit_coverage`**, and `apply_audit`'s new
  `sent_keys=` argument — reports which of the ids actually sent to the
  auditor never appear in any section of its response, the direction
  `unknown_ids` did not cover. Folded into `apply_audit`'s report under
  `coverage` when `sent_keys` is given; omitted otherwise.
- `apply_audit`'s docstring now says explicitly that group invariants (one
  canonical id per entity, no reintroduced duplicate) are out of scope by
  design — a per-key audit cannot see them, and they are the calling
  project's validator's job, not this function's.

[`docs/audit.md`](docs/audit.md), [`docs/blind-audit.md`](docs/blind-audit.md)

## 2026-09-22 -- A prompt too big for argv

### Fixed

- **`cerebellum.codex_cli` no longer hands an oversized prompt to `execve`.**
  Linux caps a *single* argv entry at `MAX_ARG_STRLEN` — 128 KiB — whatever the
  2 MiB `ARG_MAX` the whole vector gets, so a mission past that died before any
  model ran: `[Errno 7] Argument list too long: 'codex'`, surfacing as
  `CodexCLIError("codex CLI subprocess error: …")`. On alif that took out
  hvaskjer's podcast and Kulturbase adjudication on every nightly run from
  19 Sep — 64 failures, both scripts silently falling back to "preserving
  existing matches", four days with nothing adjudicated.

  A prompt over `PROMPT_ARGV_LIMIT` (64 KiB of UTF-8, overridable with
  `LIMBIC_CODEX_PROMPT_ARGV_LIMIT`) now goes to `codex exec … -`, which reads
  its instructions from stdin — written from a thread started *after* the
  stdout/stderr drains, so a prompt larger than the pipe buffer cannot deadlock
  against the child's own output, and a child that never reads stdin parks that
  thread rather than the `wait()` the timeout and process-group kill depend on.
  Shorter prompts keep the argv path unchanged. The verification call that
  proves it: 152,974 bytes, answered in 6s. `metadata.prompt_transport` on the
  ledger row records which route each call took.

---

## 2026-09-21 -- Guards that refuse, lifted with a fit check each

A read-only survey of every QA mechanism skard and Kulturbase built against
every incident they had found one clean division: **mechanisms that refuse a
specific bad state worked; mechanisms that describe a state never refused
anything.** This release lifts seven refusals and leaves the describing behind.

Three of three functions lifted on 20 Sep were rejected by their own source
project for semantic mismatch, so every function below was fit-checked against
the project's real stored data before being committed — numbers in each doc,
and one of the checks found a real bug in the lift.

### Added

- **`hippocampus.audit.apply_audit`** — folds an independent audit into a
  settled decision set, demote-only by construction: the only disposition it
  writes is the hold value, it never appends a record, and it re-checks that
  invariant over its own output. Corrections must pass the caller's validator;
  unknown ids are reported, never dropped; `source_errors` are kept as findings
  and never applied, because a fact that is wrong in the *source* was
  reproduced faithfully. Refuses a decision set that has no disposition field —
  an audit folds in after reconciliation, not before.
  **Fit check** vs skard's `demote_on_audit` over 2,925 real decisions and its
  real `independent-audit.json`: 86 demoted on both sides, identical held sets
  (symmetric difference 0), 0 promotions, 424/424 definition corrections
  identical. limbic additionally reported 1 unknown id and 2 already-held rows
  that skard passed over silently. [`docs/audit.md`](docs/audit.md)
- **`hippocampus.refuse.schema_refusals`** — validates exactly the records an
  apply would write, with no argument that can express "skip on a dry run":
  `only` filters records, not runs. The defect was a check that ran on
  `--execute` only. Includes a stdlib-only JSON Schema 2020-12 subset
  (`json_schema_refusals`) that **raises** on an unimplemented keyword rather
  than passing the record, so limbic still takes no schema dependency;
  `jsonschema_backed()` swaps in the real library where a project has it.
  **Fit check**: skard's `evidence-spine.schema.json` over 682 live concepts —
  0 refusals, matching the gate they pass, full keyword coverage so nothing was
  skipped; four one-field mutations refused 200/200 each.
- **`cerebellum.packet.slot_echo_refusal` / `meta_leak_refusals` /
  `rendering_fidelity_refusals`** — three functions, not one, because three
  different checks caught three different defects. Covers the shipped case
  where one packet answered every item with its own slot id and ten records
  entered a published graph defined as `i01`. Everything language-shaped is
  injectable; only a small English phrase list ships.
  **Fit check** vs skard's `nb_refusals` over its 519 stored bokmål renderings:
  **identical verdict on 519/519**, 88 refused each side, all ten `i01`…`i10`
  echoes caught. Parameterised to get there: phrase regexes, exonym set,
  `fold=name_key`, compound `parts`, and century `exempt_years`. The check
  caught a real bug in the lift — known names must be tokenised the way the
  source is, or the label "Det gamle Hellas" fails to make "Hellas" known.
- **`hippocampus.refuse.temporal_plausibility_refusals`** — refuses a
  machine-derived death inside living memory and an exact-year life over a
  century. `living_year` is a required argument and ships as no constant: it is
  a statement about now. **Fit check** vs skard's `extent_refusals`: identical
  on 682/682 concepts.
- **`hippocampus.refuse.dates_disagree`** *(new — no reference implementation
  existed)* — refuses a record whose prose contradicts its own structured
  dates, the `peter-kolbjornsen` case (prose 1687–1737, extent 1683–1738) that
  neither project's guards could see because one read definitions and the other
  read extents. Conservative by measurement, not by taste: matching any year
  range fired on 74 of 580 real pairs and was mostly reporting that a reign is
  not a life, so the default fires only on a *fenced* range or a birth/death
  pair. At that setting, 4 of 580 flagged — 2 real, 2 false positives of one
  kind (a monument carrying the commemorated person's dates).
- **`hippocampus.refuse.expect` / `declared_count`** *(new)* — a step declares
  how many records it will change and raises on the way out if it did not.
  "A commit promising 6 changes touched 3,963 files." This closes
  **checklist item 2.7**, which said **CODE** while no code existed in any
  project; that row now names the function.
- **[`docs/blind-audit.md`](docs/blind-audit.md)** — the brief for the second
  reader that produces an audit file: different model family, blind to earlier
  reads *and* earlier audits, the full set rather than a sample when it is
  cheap (a sample almost never contains both members of a duplicate pair), a
  fixed `right`/`wrong`/`cannot_tell` vocabulary with `cannot_tell`
  encouraged, non-`ok` rows only, corrections supplied rather than described.

### Not built, deliberately

- **Paid-artefact registry** — design note in
  [`docs/proposed-paid-artefact-registry.md`](docs/proposed-paid-artefact-registry.md).
  Its acceptance test is a fit check against a plan/packet/run/cache directory
  layout, and that layout is the part that varies most between projects. Build
  it when a second project has a tree to check against.
- **`apply_batch` + rollback** — still deferred; see
  [`docs/proposed-batch-and-cache-lifts.md`](docs/proposed-batch-and-cache-lifts.md).

---

## 2026-09-21 -- The library meets its first consumers

On 20 Sep no consumer had migrated to the functions lifted the day before. That
turned out to be a finding rather than a to-do: skard rejected three of three
candidate migrations because the extracted functions did not fit the code they
were extracted from. The transport rebuilt the request body, so stored paid
responses would no longer have been addressable; `fold` mapped æ→a where skard
maps æ→ae (505 of 3,908 strings differ); `validate_quotes` is a substring check
where skard stores anchors. Everything below closed one of those gaps or one
found by Kulturbase's `kb_resolve`. Both now import limbic: skard for every
provider call (request bytes identical on 850 of 850 stored passes), Kulturbase
for its folding primitives (gold-set rows identical).

### Added

- **A Codex cost adapter.** `codex_cli.py` used to say, in its own module
  docstring, that Codex calls "are not written to `cost_log` — there is no
  Codex cost adapter". An audit of one consumer found 1.28 billion tokens in
  eight days, all of it through this transport and none of it in any ledger.
  Both `codex_json` and `codex_research` now pass `--json`, parse the per-turn
  usage out of Codex's event stream (`parse_usage`), and write one row per
  attempt — including failed and timed-out attempts, where an agent loop's
  worst spending hides. New `project=`/`purpose=`/`packet_id=` attribute the
  row; `project` falls back to `$LIMBIC_CODEX_PROJECT`, then the git root, then
  a conspicuous `"unattributed"` rather than failing a call whose tokens are
  already spent. `cost_log=False` or `LIMBIC_CODEX_COST_LOG=0` turns it off.
  Verified against the real shape from codex-cli 0.153.4 on 2026-09-21:
  `{"type":"turn.completed","usage":{"input_tokens":…,"cached_input_tokens":…,
  "output_tokens":…,"reasoning_output_tokens":…}}`, recorded as a test fixture.
  In `--json` mode the raw event stream is never returned as an answer: if
  `--output-last-message` is empty and no recognised event carried the agent's
  message — an older CLI with an event shape we have not seen — the call fails
  exactly as an empty result always did, rather than handing back a transcript
  that nothing downstream could tell from a real answer.
- **`billing_mode` and `notional_cost_usd` on the ledger.** A subscription call
  burns real tokens and spends no money; one number cannot hold both. A
  `subscription` row carries `cost_usd = 0` — `log()` refuses it otherwise —
  and the API-equivalent estimate in `notional_cost_usd`. Every reader that
  predates the column therefore keeps returning real spend, and a reader that
  wants the other figure has to name it and so has to label it:
  `total_notional()`, `notional_cost_usd` in both summaries,
  `notional_cost_per_applied`, and its own dashboard section. An unpriced model
  logs its tokens with a NULL notional rather than an invented $0.
- **The `cached_call` cache key is pinned by a test.** It addresses answers
  consumers have already paid for, so a reordering or a new field in
  `_cache_key`'s payload silently re-buys every stored response. A failure
  there is now a migration, not a test to update.
- **`docs/proposed-batch-and-cache-lifts.md`,** designs for two mechanisms a
  consumer has and this library does not: a narrow content-hash projection for
  cache keys (the hooks — `version=`, `cache_key=` — already exist; the helper
  does not), and batch structured-output validation that rejects duplicate
  answers per key and returns the missing ones for retry rather than erroring
  the batch. Neither is built.
- **`cached_call(request=, cache_key=)`.** A fully built provider body is posted
  as its exact bytes through the openai/gemini transports, the response cache
  is keyed on those bytes (or on the caller's own key), the raw response comes
  back, and the ledger row is still written — `outcome=error` on a failed
  request. A consumer whose paid artefacts are addressed by a request hash can
  adopt the ledger and transports without invalidating what it has bought.
- **`cost_for` and `cached_input_price_for`.** Cached input tokens are billed at
  the provider's cached rate (OpenAI and Gemini list prices read 2026-09-21;
  Anthropic left out, since its cache has a write surcharge one token count
  cannot express). `compute_cost` and both HTTP transports use them; `price_for`
  keeps its two-tuple and strict semantics. Rows already in the ledger are not
  repriced.
- **`CostLog.log(ts=)`,** so a backfilled call keeps its own date.
- **`fold(profile="ascii")`.** `[0-9a-z ]` keys, `ae` for the ligature,
  underscore as a token break — identical to skard's fold on 74,707 real
  strings. The default names profile is unchanged.
- **`packet.text_quote_anchor` and `reanchor_quote`.** A TextQuoteSelector-style
  anchor (exact, prefix, suffix, occurrence, offsets, selector/span/page
  hashes) beside `validate_quotes`, which is unchanged. An empty quote is an
  unresolved anchor, not a match at offset 0 — three skard items that had
  validated that way are now held.
- `hippocampus.resolve.invert_name` and `strip_parenthetical` are public, so a
  consumer keeping its own index can share the name surfaces.

### Changed

- **`hippocampus.resolve` and `limbic.cerebellum` import with the standard
  library only.** `connect` moved to `limbic._sqlite` (re-exported from
  `amygdala.index` unchanged), the hippocampus package resolves its exports on
  first access, and `cerebellum.calls` no longer reaches `connect` through
  `limbic.amygdala` — that one import loaded the embedding stack and numpy into
  every packet runner (0.13–0.33 s against 0.03 s, and an ImportError in an
  interpreter without them). Both are held by subprocess tests.

### Fixed

- **`merge_from` copies the columns both ledgers have.** It was
  `INSERT ... SELECT *`, which fails on the column count the moment a host runs
  an older limbic than the machine merging its rows — stranding every remote
  row over one new field. Columns the remote lacks take local defaults.
- **A folded key only meets a key from the same spelling table.** "Bø" expands
  to "boe", which is also what the different name "Bøe" drops to, so the folded
  layer linked them at 0.97. Each indexed key records the table that produced
  it (drop, expand, or both when they agree). An index built before the column
  existed reads as "both" until its kind is rebuilt. The token layers
  (`token_overlap`, `text_candidates`) still compare tokens across tables; that
  needs spelling-tagged tokens and is open.
- **A bookkeeping failure never loses a billed response.** Once the provider's
  response is in hand it has been paid for: the ledger write in both
  transports, in `_log_call`, on a cache hit, and the response-cache write now
  warn instead of raising. `CallMeta.call_id` is `None` when the row could not
  be written; the provider response id in the metadata is enough to backfill
  it. Found by skard's adoption, whose runner tests exactly this invariant.
- `gemini-2.5-flash-lite` fallback price corrected from 0.15 / 0.60 to the
  listed 0.10 / 0.40 per 1M tokens (pricing page read 2026-09-21).

### Still missing for consumers

A public usage parser for a raw Responses payload (skard keeps its own
`usage_of`), and post-call fields on a ledger row.

---

## 2026-09-20 -- Packets, entity resolution, and the write boundary

Wave 2 of the `llm-pipeline-audit` response. The audit found limbic is adopted
only where it is one import and one call — `hippocampus.ProposalStore`,
extracted *from* Kulturbase, has zero consumers including Kulturbase — so
everything here is a plain function usable in ten lines. No base classes to
subclass, no registries, no config objects.

Three things proven in real projects the same day, lifted into the library:
skard's stateless packet runner (measured 95x cheaper than the same work inside
a tool-using subagent), Kulturbase's `kb_resolve` (recall@1 0.93, false-match
rate 0.000 on 200 negatives), and the preimage-checked apply that governance
found to be the only proposal mechanism that ever refused a bad write.

### Added

- **`hippocampus.resolve`.** Entity resolution over a SQLite sidecar index.
  `fold(text, lang)` with two Nordic spellings — the character map runs *before*
  NFKD, or "Zauberflöte" can never produce "zauberfloete"; `name_keys` covering
  inversion, particles and the Norwegian genitive; `build_index(conn_or_path,
  rows, kind)`; `candidates(index, query, k, kind, hints)` returning compact
  cards with a match type and score; `text_candidates` for literal presence in
  a passage; `slot_enum(cards, n_slots)` / `unslot(items, slot_map)`. Hints
  rerank and never filter. The near-miss fuzzy layer is **off by default** and
  capped below the confidence threshold when on: folding and token-set matching
  recovered only 5% of one campaign's 5,194 held unmatched names, so fuzziness
  is rarely the recall gap, and it is the layer that manufactures
  plausible-looking wrong answers.
- **`cerebellum.packet`.** `make_packet(static_prefix, body, schema, *,
  prompt_version)` returns a frozen dict whose `input_sha256` covers the prefix
  hash — editing a shared mutable prefix file later silently invalidated a
  batch that had already been paid for. `run_packets(...)` on top of
  `cached_call`: dry-run by default, budgets that refuse rather than finish,
  every call ledgered (failures included) with `packet_id` and `outcome`, and
  truncation answered by a split-once hook rather than by re-asking.
  `probe(packets, n=50, yield_fn=..., min_yield=...)` raises `LowYield` — the
  cheapest fix for the governance-to-yield inversion (12.3k lines of machinery
  around 0 writes, while a plain join next door produced 2,336 of 2,342
  proposals). `lint_packet` warns when the schema varies across the batch
  (measured 0% cached input; a fixed slot enum on the same batch measured 58%),
  when the shared prefix is under the ~1,024-token provider cache minimum
  (below it, caching cost 9.7% *more*), when a body field is derivable, and
  when a packet is past ~25 items. Plus `validate_quotes` (exact substring,
  whitespace collapse only), `union_passes` (adding name-accounting to a coding
  call lost 66 known entities — run separate passes and merge), and
  `unmatched_names` / `corpus_lowercase_words`, the deterministic recall scan
  whose common-noun filter is the corpus's own lower-case vocabulary.
- **`hippocampus.apply.apply_proposal`.** Field whitelist, exact preimage check
  where `MISSING` (absent key) is distinct from `None` (explicit null),
  validators, atomic write, and a receipt emitted on every attempt including
  refusals. Never partial. Validators: `enum_member`, `regex`,
  `wikidata_exists`, and `wikidata_type_is(expected)` — the one existence
  checks miss, which is not a small gap: of 901 work QIDs audited in one
  catalogue, **198 pointed at something that was not that work**, and every one
  passed an existence check (*Et dukkehjem* resolved to Ramon Llull).
- **`CostLog.set_packet_id`,** giving the reserved `packet_id` column a writer,
  so one packet's whole history — dry run, real call, post-truncation halves —
  is answerable before replanning a batch.
- **Skills** `packet-worker`, `thin-worker-brief` and `coordinator-hygiene`,
  plus `docs/new-data-project-checklist.md` (the tiered Tier 0/1/2 checklist,
  each item tagged CODE with its enforcing function or PROSE, with a "when NOT
  to adopt" line per tier). Module docs in `docs/resolve.md`, `docs/packet.md`
  and `docs/apply.md`.

### Changed

- `ProposalStore`'s docstring now points at `apply_proposal`: the store is the
  filing cabinet, not the lock — it has no preimage check, so an `approved`
  status in it is a string in a file.

### Fixed

- **`tests/conftest.py` now refuses production databases.** Two agents polluted
  the real cost ledger (`~/.local/share/limbic/llm_costs.db`) by running the
  suite on 20 Sep 2026: `cost_log` is a module-level singleton that resolves
  its path at *import* time, so a fixture setting `COST_LOG_DB` is already too
  late. The environment variables are now set at conftest module import, every
  test gets its own ledger and cache file, and `sqlite3.connect` raises on any
  path under `~/.local/share/limbic`. Tested in
  `tests/test_ledger_isolation.py` — the guard is load-bearing, so it has its
  own tests.

---

## 2026-09-20 -- Response cache, outcome-bearing ledger, and session forensics

The 20 Sep 2026 `llm-pipeline-audit` found that consumers only adopt limbic
where it is one import and one call, and that the pieces that would actually
save money were missing: no response cache despite the hashes already being
computed, no `outcome` field so cost per useful change couldn't be measured,
a silent `project="limbic"` default, and no visibility into interactive
agent-session spend (which the ledger never sees). This release closes those
gaps as plain functions, not new frameworks.

### Added
- **`cerebellum.calls.cached_call`.** A thin wrapper around `claude_cli.generate`
  (or any callable with the same shape) that caches by
  `sha256(model, system, prompt, schema, version)` in a small SQLite store
  (`~/.local/share/limbic/llm_cache.db` by default, via `amygdala.connect`).
  A hit costs $0 and a `cache_hit=1` ledger row instead of a subprocess call.
  `cache=True|False|"refresh"`, `ttl_days`, and `replicates=n, agree=k` for
  independent-read agreement (returns a `Held` result on disagreement rather
  than trusting a single confidence score — the audit found exact agreement
  between two reads lifted precision 0.71 -> 0.97, while three-way agreement
  was only 62.8%). `purpose` is a required argument (empty on ~40% of rows
  per the audit); an empty `project` is inferred from the git root rather
  than silently defaulting, which is what produced 25,642 unattributable
  "limbic" rows in `amygdala.llm.generate_structured`. Never holds the
  SQLite write lock across the model call (read -> release -> call -> write).
- **`cost_log` ledger gained `cache_hit`, `outcome`, and `packet_id` columns**,
  added via an idempotent `ALTER TABLE` migration guarded by
  `PRAGMA table_info` (safe against an existing database with the old
  schema). `CostLog.record_outcome(call_id, outcome, detail="")` sets
  `outcome` (applied/no_op/rejected/held/error) after the fact, once the
  caller knows whether the result was used — the field the ledger needed to
  answer "cost per useful change" instead of only "cost per call".
- **`cost_log.price_for(model)`.** Raises `UnknownModelPriceError` for an
  unpriced model instead of the silent-$0 path that let Otak carry a price
  table 5-7x too low for months. `strict=False` opts back into the old
  best-effort behaviour.
- **`python -m limbic.cerebellum.cost_log report --by project,purpose,outcome --since 30d`.**
  A multi-column breakdown (`CostLog.multi_group_summary`) with cache-hit
  rate and `cost_per_applied` per group, alongside the existing single-column
  `--group-by` report.
- **`limbic.cerebellum.forensics`**, ported from the audit's prototype
  scripts (`~/src/research/llm-pipeline-audit/data/`). `cost_log` only sees
  calls made *through* limbic; this is the read-only forensic layer over the
  interactive Claude Code / Codex session transcripts where most spend
  actually happens. Two counting rules made structural: never trust a
  session's cumulative token counter (a forked Codex subagent inherits its
  parent's cumulative count; a resumed session can reset it) — sum
  `last_token_usage` deltas instead; and dedupe Claude JSONL by
  `message.id` before summing `usage`, since a streamed response repeats the
  same id across several lines. `python -m limbic.cerebellum.forensics
  codex|claude [--since 30d] [--project-by cwd|paths] [--session FILE --attrib]`.

### Changed
- `CostLog.log()` gained optional `cache_hit`, `outcome`, `packet_id`
  keyword arguments (all default to the previous behaviour; existing callers
  are unaffected).

---

## 2026-09-20 -- HTTP transports, fixed double-logging, and subagent forensics

Review follow-up on the response cache above. `claude -p` adds roughly 9-17K
harness tokens per call, the wrong transport for the cheap, high-volume
workers (Kulturbase's Luna campaigns, skard's packet runner) that POST
directly to a provider API and never reach the ledger. Fixing that surfaced a
real cost-doubling bug in `cached_call` itself.

### Added
- **`cached_call(transport="openai"|"gemini")`.** Built-in HTTP transports —
  stdlib `urllib` only, no `openai` or `google-genai` SDK dependency
  (`google-genai` fails to import on at least one project's arm64-macOS
  Python build). OpenAI via the Responses API with strict `json_schema`
  structured output; Gemini via REST `generateContent` with a local
  null-union schema stripper. Both self-log to `cost_log` through
  `price_for()`, so an unpriced model logs a visible $0 with a warning
  instead of failing the call.
- **`limbic.cerebellum.forensics` now finds Claude Code subagents.** A
  Task-tool subagent's turns are not inline `isSidechain: true` lines in the
  parent's own transcript — they live in
  `<session-id>/subagents/agent-*.jsonl` beside it. The scanner missed this
  entirely (reported 0 subagent tokens on real sessions with a dozen
  subagents each) until fixed to scan that directory and fold it into
  `sidechain_by_model`. Also reports each subagent's "entrance fee" (its
  first request's input+cache-creation+cache-read total — a real 30-day scan
  came in at a 51.5K median, matching the audit's 49.8K).

### Fixed
- **`cached_call` was double-logging every cache miss with the `claude_cli`
  transport**, since `claude_cli.generate()` already writes its own
  `cost_log` row and `cached_call` then wrote a second one carrying the same
  cost — silently doubling every reported total for the default transport.
  `claude_cli.generate()` (and both new HTTP transports) now return a
  `call_id` in their metadata; `cached_call` logs only when a transport
  didn't already (`calls._log_call`). Caught by a test that actually counted
  ledger rows for the real transport, which the first round's tests didn't.
- **`limbic.cerebellum.__init__` no longer re-exports the `cost_log`
  singleton.** `from .cost_log import ..., cost_log, ...` rebinds the
  *package* attribute `limbic.cerebellum.cost_log` from the submodule to
  that instance, so `import limbic.cerebellum.cost_log` silently returned the
  instance rather than the module. No consumer was affected (alif, petrarca,
  otak, dragoman, nrk all use the fully-qualified
  `from limbic.cerebellum.cost_log import cost_log`, which resolves via
  `sys.modules` and was never shadowed) — confirmed by grep before removing
  the re-export.
- A test-only artifact: an earlier version of the `cached_call` test suite
  exercised the real `claude_cli` transport without patching
  `claude_cli.cost_log`, which briefly wrote fake rows into the real
  production ledger (`~/.local/share/limbic/llm_costs.db`). Deleted by id;
  the fixture now patches both bindings.

---

## 2026-09-16 -- Bounded parallel fan-out in amygdala.llm

### Added
- **`generate_parallel` / `generate_parallel_sync` and `LLMTask`.** Ported from
  the `llm_providers.py` that otak and hirsch-atlas each carry a byte-identical
  copy of — the one capability those copies had that `amygdala.llm` did not, and
  therefore the blocker on retiring them. Results come back in input order; a
  task that fails returns `(None, {"error", "tag"})` rather than taking the batch
  with it, because a 300-item fan-out should not lose the 299 that worked.
  `max_concurrent` is the only backpressure there is.
- **`Retry-After` is honoured** when a provider sends one, in place of guessing
  with exponential backoff. The server knows when it will be ready.

---

## 2026-09-16 -- Agentic call isolation, windowed extraction, and a documentation audit

### Fixed
- **`codex_research` ran without `--ephemeral` / `--ignore-user-config`.** The
  agentic entry point — the one that reads untrusted web pages with network
  egress — was the one *without* the isolation flags that the locked-down
  `codex_json` already passed. A downstream pipeline (koigen/hvaskjer) had been
  rebinding `codex_cli._run` at runtime to inject them rather than fork limbic.
  Now `isolated=True` is the default here too; pass `isolated=False` when a run
  genuinely needs the host profile (a locally configured MCP server, say).
- **A timed-out `codex exec` left its children running.**
  `subprocess.run(timeout=...)` reaps only the direct child. `_run` now starts
  its own process group and SIGTERM/SIGKILLs the tree. Verified with a forked
  grandchild that kept writing after the parent was reaped under the old code.
- **Captured output was unbounded.** An agentic run could stream until the parent
  ran out of memory. stdout/stderr are now drained concurrently through a
  bounded tail (`LIMBIC_CODEX_OUTPUT_LIMIT`, 2 MB default) that keeps the *end* —
  where the fatal error is — and reports how much it dropped.
- **Both CLI wrappers snapshotted `os.environ` at import.** Any env change a
  caller made afterwards was invisible to the subprocess, including scrubbing
  secrets before handing an agent hostile text — exactly the case the scrub
  exists for. `codex_cli._codex_env()` and `claude_cli._claude_env()` now read
  the environment per call.
- **CI had been red on `main` since 2026-09-04.** `pip install -e ".[dev]"`
  pulled in neither the `llm` nor the `temporal` extra, so `test_llm.py` failed
  with `ModuleNotFoundError` while most of `test_temporal.py` silently skipped.
  It passed locally only because a developer venv accumulates every extra.
  `dev` now self-references `[hippocampus,llm,temporal]`, and the two provider
  test classes use the `importorskip` guard the rest of the suite already had.

### Added
- **`limbic.cerebellum.sandbox`** — isolation primitives for agentic calls,
  ported from the koigen/hvaskjer nightly where they ran against real scraped
  and emailed input:
  - `untrusted_payload()` delimits external material with a content-derived
    nonce and puts the refusal instruction ahead of the data;
  - `isolated_scratch()` gives the agent a private 0700 workspace outside the
    project, with an allowlist of inputs, destroyed afterwards;
  - `sanitized_environment()` allowlists the child's environment so an injection
    cannot become a credential disclosure;
  - `call_slot()` bounds concurrency across processes and enforces a persistent
    daily call cap.

  Documented as **not** an OS sandbox: the child keeps whatever the CLI grants it.
- **`limbic.cerebellum.windowing`** — windowed LLM extraction and a merge that
  preserves cross-references. Whole-chapter extraction loses most of a text; on
  the Hirsch corpus 6K windows with 1K overlap produced 80–150 claims per chapter
  against 20–30, and that structural change beat every prompt variation.
  `merge_windows` enforces the order that makes the merge safe: namespace ids
  before concatenating, dedup before renumbering, rewrite references as part of
  renumbering. Generalized from the otak/hirsch-atlas copies, which had already
  started to diverge:
  - the hardcoded collections become a declared `MergeSchema`, validated at
    construction;
  - the two dedup variants collapse into one that keeps the **longer** text;
  - word-overlap tokenizing was ASCII-only (`[a-z]+`), so every Norwegian or
    non-Latin item compared as the empty set and nothing deduplicated;
  - `split_into_windows` validates its parameters and cannot fail to advance.

### Fixed in review

Found by a Fable review of the branch before merge; all four reproduced first.

- **`_run` overran the timeout it had just enforced.** A descendant with its own
  session (`setsid`) survives `killpg` and still holds the pipe, so its reader
  thread is parked in `read()` holding the buffer lock that `close()` needs —
  a 1s timeout returned after 6.1s. Streams whose reader is still alive are now
  left to the daemon thread, and the reader joins share one budget instead of
  one each. Same case now returns in 2.1s.
- **One undecodable byte became a 900s false timeout.** `UnicodeDecodeError` is
  a `ValueError`, which the drain thread caught and returned on; the child then
  stalled on a full pipe and the caller saw a timeout, which does not retry.
  `Popen` now uses `errors="replace"`, matching what `_finish` already did.
- **`sandbox` refused to run from a working directory of `/`.** `protect`
  defaults to the cwd, and systemd's default `WorkingDirectory` is `/`, so every
  scratch path was "inside" the protected tree — the check is unsatisfiable
  there rather than violated, and is now skipped. hvaskjer protected a fixed
  repo root; the cwd default was introduced by the port.
- **`sanitized_environment(home=...)` logged Codex out.** Codex resolves
  `~/.codex/auth.json` from `$HOME`, so repointing HOME moved its credentials.
  hvaskjer only worked because its deploy exports `CODEX_HOME`; that precondition
  did not survive the port, and the documented recipe was broken as written.
  `CODEX_HOME` is now pinned to the original HOME unless the operator set it.
- **`MergeReport.dangling` was nearly vacuous.** `_renumber` clears every
  unresolvable reference before `check_references` runs, so the common failure —
  a window referencing an id no window produced — showed `dangling == []`. Added
  `references_cleared`, which is the counter that actually means "links lost".
- **`tests/test_claude_cli.py` wrote to the real cost database.** Its
  `reload(cc)` (there to refresh the old `_ENV` snapshot, now removed) also
  reset the module's `cost_log` past the `tmp_cost_log` fixture, leaving 15
  `project='testproj'` rows in `~/.local/share/limbic/llm_costs.db`. Predates
  this branch; removing the now-pointless reload fixes it.

Also documented two behaviours that are correct but were understated:
`--ignore-user-config` drops the whole host profile (`service_tier`, `notify`,
`personality`, MCP servers), not just "nothing about which model runs"; and the
slot/budget defaults under `tempfile.gettempdir()` are per-service under
systemd's `PrivateTmp=yes` and reset on a tmpfs `/tmp`, so "host-wide" and
"persistent" need the env overrides to be literally true.

### Documentation
- README claimed **325 tests across three packages**; there were 530 across four.
  Now 606, with an accurate per-package table.
- `limbic.drive` shipped on 2026-09-13 with no CHANGELOG entry, no mention in
  `CLAUDE.md`, and no place in the architecture diagram or package tables. Added,
  along with its Python API (`validate_plan`, `check_calibrations`).
- Documented seven modules that had code and tests but no prose anywhere —
  roughly 2,950 lines and 164 tests: `hippocampus.wikidata_resolve`,
  `amygdala.wikidata`, `amygdala.temporal`, `amygdala.retrieval_eval`,
  `amygdala.serendipity`, `cerebellum.claude_cli`, `cerebellum.codex_cli`.
  Every one had arrived as an upstream from a consumer project, where the
  explanation stayed behind in that project's own writeup.
- Recorded findings from production corpora in *Design decisions*, including two
  **negative results kept deliberately**: LLM reranking cascades never beat the
  free cross-encoder (the bottleneck is first-stage recall, not ranking), and
  agentic file-reading leads on quality but erodes with scale at ~1000× the cost.
  Also the e5-vs-MiniLM encoder trade (an aggregate win hiding a Norwegian
  regression) and the two-axis serendipity judging that made link output usable.
- Architecture diagram and all four module tables brought back in line with the
  actual tree.

---

## 2026-09-13 -- Drive: calibration-first planning policy

### Added
- **`limbic.drive`**, a fourth package: deterministic policy checks for a Drive
  direction card. The model still supplies the judgment — what the user means,
  which precedent matters, what a representative pilot is — while `validate_plan`
  makes the expensive mistakes mechanically difficult. A v0 plan cannot spawn
  workers, spend model calls, or authorize a batch before the user has
  experienced one small pilot.
- `validate_plan(plan)` returns *every* violation rather than the first, so a
  plan gets one round of correction instead of one per rule.
- `check_calibrations()` replays three bundled historical cases — drawn from the
  NRK apps, the Otak/Hirsch investigation, and the Codex/Claude workflow research
  — so a policy change that would have re-allowed a past mistake fails here
  instead of in a live session.
- CLI: `python -m limbic.drive calibrate` and
  `python -m limbic.drive validate <plan.json>`.
- `skills/drive/` ships the shared Codex/Claude planning skill itself.

---

## 2026-09-04 -- EDTF wildcard spellings and qualifier flags in temporal

### Fixed
- **`parse_date` rejected 2012-draft EDTF wildcards.** The draft spelled
  unspecified digits lowercase (`19uu`, `196x`); ratified EDTF (ISO 8601-2:2019)
  uses uppercase `X`, and the `edtf` package only accepts that. Archive data
  predates the change often enough that both have to work, so date-shaped tokens
  are normalized before parsing. `test_edtf_uncertain_decade_parses_when_available`
  had been failing on this.
- **`DateRange.uncertain` was never set by anything.** The EDTF path now maps the
  qualifier suffix onto the precision flags: `?` uncertain, `~` approximate,
  `%` both — matching how the regex path already sets `approximate` for "circa".

## 2026-09-04 -- Current-generation model support in amygdala.llm

### Added
- **GPT-5.6 tiers** in `MODELS`: `luna` (gpt-5.6-luna), `terra` (gpt-5.6-terra),
  `sol` (gpt-5.6-sol), plus `gpt55`, `gpt54-mini`, `gpt54-nano`.
- **Gemini 3.x**: `gemini38-flash`, `gemini35-flash`, `gemini35-flash-lite`,
  `gemini31-pro`, `gemini31-flash-lite`.
- **Claude 5**: `opus` (claude-opus-5), `fable` (claude-fable-5-1); `sonnet` now
  maps to `claude-sonnet-5`.
- `### LLM client` section in README listing every key, wire id, and price.
- `tests/test_llm.py` covering the registry, per-provider call shaping, Gemini
  token accounting, and the fallback path.

### Fixed
- **OpenAI structured output was broken for every model.** `_call_openai` sent
  `response_format={"type": "json_object"}` without mentioning JSON in the
  messages, which the API rejects with a 400 unless the caller's own prompt
  happened to contain the word. The schema is now appended to the system prompt,
  which both satisfies that requirement and tells the model what shape to return
  (previously OpenAI models were asked for JSON with no schema at all).
- **Gemini thinking tokens were not costed.** `_call_gemini` reported only
  `candidates_token_count`, but Google bills thinking tokens as output. A 5-token
  answer with 136 thinking tokens was under-reported ~28x. `output_tokens` now
  includes `thoughts_token_count`.
- Stale prices in `MODELS` and `cost_log._FALLBACK_PRICES` (e.g. gemini-3-flash
  was listed at 0.10/0.40, actually 0.50/3.00; `claude-sonnet-4` is retired on the
  first-party API; `claude-haiku-4-5-20241022` was never a valid id).
- `cost_log._fallback_cost` only stripped a `gemini/` prefix, so `openai/…` and
  `anthropic/…` models never matched the table. It now ignores any provider prefix.

## 2026-06-18 -- Serendipity link-finder

### Added
- **`limbic.amygdala.serendipity`** — surfaces useful but *non-obvious* document
  links (the objective relevance/precision can't capture). `serendipity_pairs`
  ranks pairs in an inverted-U similarity "sweet spot" (related but not
  duplicate/unrelated), with a cross-facet bonus (source/era/domain) so links you
  wouldn't manually make score highest. `abc_bridges` does Swanson ABC bridging
  (transitive A–C links via a shared intermediate B). Bands are embedding-space
  dependent — calibrate to the model / use whitening. Tests in
  `tests/test_serendipity.py`.

## 2026-06-18 -- Retrieval evaluation harness

### Added
- **`limbic.amygdala.retrieval_eval`** — the IR-evaluation loop amygdala lacked.
  `calibrate` validated an LLM *judge* against humans, but nothing scored
  *retrieval*. New module adds:
  - Graded metrics: `ndcg`, `recall`, `mrr`, `average_precision` (2^g-1 gain,
    `rel_threshold` binarisation for recall/MRR/MAP).
  - `pool(runs, depth)` — union of competing methods' top-`depth` results so no
    method is penalised for surfacing a good doc the others missed.
  - `judge_pool(...)` — backend-agnostic graded judging (inject any
    `judge_fn(query, doc) -> grade`: LLM, Codex, human, heuristic), incremental/
    resumable so enlarging the pool only judges new pairs.
  - `make_llm_judge(...)` — default judge over `limbic.amygdala.llm`.
  - `score(runs, qrels, strata=...)` + `format_report(...)` — per-method and
    per-stratum (e.g. query-category) breakdown.
  - Composes with `calibrate.validate_llm_judge` to check the judge against a
    human-labelled subset. Tests in `tests/test_retrieval_eval.py`.

## 2026-04-17 -- Cost dashboard surfaces CLI subscription value

### Changed
- **Whole-dollar formatting** in the cost dashboard. `fmt()` and the daily-chart y-axis now use `Math.round` instead of `toFixed(4)` / `toFixed(2)`, with locale comma separators (e.g. `$13` not `$13.8257`).
- **CLI section now shows cost.** Previously CLI tables hid `cost_usd` (rendering `-` in the recent-calls panel and omitting the column entirely from the by-project / by-model panels). The data was already logged but invisible — making heavy CLI users (e.g. alif at ~$272/wk subscription value) appear as "0 usage". CLI totals add a "Subscription Value" card; CLI by-project / by-model add a "Sub Value" column and sort by cost.
- CLI by-project / by-model SQL `ORDER BY` switched from `total_tokens` to `cost_usd` so the ordering matches the new primary column.

### Added
- `cli.cost_usd` field in the `/api/summary` response (sum of `cost_usd` across `script='claude-cli'` rows).
- `cost_usd` field in each row of `cli.by_project` and `cli.by_model`.

## 2026-03-26 -- Search improvements from claude-chat-search integration

### Fixed
- **FTS5 query sanitization bug** — unquoted tokens let reserved words (AND, OR, NOT, NEAR) act as FTS5 operators, producing wrong results. Tokens are now quoted in both `FTS5Index._sanitize_query()` and `Index._sanitize_query()`.

### Added
- **FTS5 auto-sync triggers** on the `Index` class — replaces 30-line manual `_sync_fts_for()` with 3 SQLite triggers that keep `chunks_fts` in sync on INSERT/DELETE/UPDATE. Triggers fire within the same transaction, so FTS stays consistent even on crash.
- **`Index.grep(pattern)`** — exact substring search via SQL LIKE. For file paths, error messages, and code patterns that FTS5 tokenization mangles.
- **`dedup_by(results, key_fn)`** — utility to keep only the top-scoring result per group. Useful for session deduplication and similar patterns.
- **`Index.rebuild_fts()`** — public method for one-time FTS rebuild on databases created before triggers existed.
- 15 new tests: FTS5 sanitization with reserved words and unicode, trigger lifecycle, grep, dedup_by.

### Changed
- `Index._sync_fts_for()` replaced by `rebuild_fts()` — no longer called automatically (triggers handle it).

## 2026-03-23 -- Knowledge map experiments and default propagator switch

### Changed
- **Default propagator switched to "bayesian"** in `init_beliefs()`. Comprehensive experiment across 5 topologies × 50 trials shows Bayesian propagator reaches 80% accuracy in 7.2 questions vs 8.8 for heuristic (18% fewer questions overall, 42% fewer on chains).
- `adjust_for_calibration()` now discounts all beliefs above 0.5 (not just unassessed), accepts optional `graph` parameter for re-propagation.

### Added
- `next_probe_batch(n)`: diversity-aware batch probe selection. Uses sequential greedy with simulated outcomes to avoid redundant probes (e.g., won't pick 3 siblings of the same parent). Batch(5) reaches 80% accuracy in 1 round.
- Comprehensive experiment: `experiments/exp_knowledge_map_matrix.py` — tests propagator × strategy × topology × noise × calibration × batch selection.

### Fixed
- Dampening test updated for Bayesian propagator behavior (CPD model gives slightly different grandchild beliefs than heuristic's multiplicative dampening).

## 2026-03-23 -- Bayesian CPD parameter optimization

### Changed
- Optimized Bayesian propagation CPD parameters via 180-config grid sweep:
  - `_CPD_HIGH`: 0.85 → 0.90 (P(known | all prereqs known))
  - `_CPD_LOW`: 0.15 → 0.05 (P(known | any prereq unknown))
  - Bayesian accuracy: 69.5% → 70.4% (+0.9%)
- Extracted hardcoded CPD values into module-level constants (`_CPD_HIGH`, `_CPD_LOW`, `_EVIDENCE_THRESHOLD`)

## 2026-03-23 -- Kulturperler migration to limbic

### Changed
- Updated kulturperler DR-arkivet scripts to use limbic instead of custom implementations:
  - `import_drdk_batch.py` and `enrich_from_drdk.py`: replaced custom state management (load_state/save_state/update_production_state with fcntl locking) with `limbic.cerebellum.StateStore`; replaced custom JSONL logging (log_event/get_log_path) with `limbic.cerebellum.AuditLogger`; replaced bare `sqlite3.connect()` with `limbic.amygdala.connect()`
  - `fetch_drdk_catalog.py`: replaced `sqlite3.connect()` with `limbic.amygdala.connect()`
  - `cleanup_utils.py`: added `get_db()`, `get_state_store()`, `get_audit_logger()` helpers that wrap limbic for use by downstream scripts

### Fixed
- Documentation: `StateStore` examples incorrectly referenced SQLite (`.db` extension, "WAL mode") — it actually uses JSON files with atomic temp-file writes and `fcntl.flock()`. Fixed in both README.md and cerebellum/README.md.

## 2026-03-22 -- Calibration metrics and confidence-based pair classification

### Added
- `limbic.amygdala.calibrate` module: Cohen's kappa, `validate_llm_judge()` (Bootstrap Validation Protocol), `intra_rater_reliability()` for measuring LLM judge consistency
- `classify_pairs_with_confidence()` in cluster.py: confidence-calibrated pair classification using cosine + NLI cascade, with per-label precision/recall/F1
- `format_for_eval_harness()` in cluster.py: format classification results for evaluation
- 42 new tests for calibrate and cluster modules

## 2026-03-22 -- Limbic monorepo restructure

### Added
- `limbic.hippocampus`: proposal system (`Proposal`, `Change`, `ProposalStore`), cascade merges (`ReferenceGraph`, `apply_merge`, `apply_delete`), entity deduplication with composable veto gates (`VetoMatcher`, `ExclusionList`), data validation framework (`Validator`, `Rule`), YAML-backed entity store with file locking (`YAMLStore`). 54 tests.
- `limbic.cerebellum`: resumable batch processor with budget tracking (`BatchProcessor`, `StateStore`), multi-tier verification orchestrator with auto-escalation (`TieredOrchestrator`, `VerificationTier`), JSONL audit logger with daily rotation (`AuditLogger`), LLM context builder (`ContextBuilder`). 33 tests.
- `limbic.__init__` top-level package with docstring documenting sub-packages
- Backwards-compatible import shims during migration (re-exports from `amygdala.*` to `limbic.amygdala.*`)

### Changed
- Restructured from single-package `amygdala` to `limbic` monorepo with three sub-packages
- Package name in pyproject.toml changed from `amygdala` to `limbic`
- All source code moved from `amygdala/` to `limbic/amygdala/`
- Added `hippocampus` optional dependency group (pyyaml)

### Removed
- Backwards-compatible `amygdala/` shims removed after all consumers migrated

## 2026-03-22 -- Document similarity module

### Added
- `limbic.amygdala.document_similarity`: document-level thematic matching using weighted multi-field embeddings
- `Document`, `SimilarityPair`, `find_similar_documents()`, `embed_documents()`, `document_similarity_matrix()`
- 94% accuracy on human-rated pairs, AUROC=0.930 on 300-pair dataset, Spearman rho=0.818
- Weighted multi-field strategy (0.5x summary + 0.5x claims) outperforms concatenation (94% vs 89%)
- Calibrated thresholds for four use cases: feed ranking, balanced, high confidence, near-duplicate
- Design rationale document (`experiments/document_similarity_design.md`) and calibration data (`experiments/calibration_document_similarity.md`)
- 15 tests for document similarity

## 2026-03-20 -- Initial public release

### Added
- `amygdala.embed`: Sentence embedding with 3 whitening modes (Soft-ZCA, All-but-the-top, PCA), Matryoshka truncation, text genericization, persistent SQLite cache. Default model: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim).
- `amygdala.search`: Numpy brute-force vector search, SQLite FTS5 with porter stemming and query sanitization, hybrid RRF fusion, cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)
- `amygdala.novelty`: Multi-signal novelty scoring (global + topic-local + centroid specificity + temporal decay), NLI cross-encoder cascade for contradiction detection, `classify_pairs()` with cosine+NLI pipeline
- `amygdala.cluster`: Greedy centroid clustering (batch + incremental), complete linkage, pairwise cosine matrix, `extract_pairs()` for cross-group deduplication
- `amygdala.cache`: Persistent SQLite-backed embedding cache (83-452x speedup)
- `amygdala.index`: SQLite document/chunk storage with `connect()` helper (WAL, busy timeout, cache)
- `amygdala.knowledge_map`: Adaptive knowledge probing via Shannon entropy maximization and Bayesian belief propagation, overclaiming detection with foil concepts, KST fringe computation
- `amygdala.knowledge_map_gen`: LLM-powered knowledge graph generation from topic descriptions
- `amygdala.llm`: Multi-provider LLM client (Gemini, Anthropic, OpenAI) with structured output, retry, and cost tracking
- README with full API documentation and benchmark results
- 21 experiment scripts with results (model comparison, whitening sweep, novelty optimization, fusion comparison, clustering, NLI evaluation, genericization, Karpathy loop, reranking, temporal decay, domain whitening, Soft-ZCA, similarity graph, task embeddings, large corpus novelty, cross-lingual retrieval, query expansion, incremental clustering, NFCorpus eval, persistent cache, All-but-the-top)
- Eval scripts for STS-B, SciFact, QQP, Norwegian PAWS-X
- Research documents: advanced algorithms, assessment datasets, tutorial platform graphs
- CI via GitHub Actions

## 2026-03-20 -- Knowledge map module

### Added
- `knowledge_map.py`: KnowledgeGraph, BeliefState, init_beliefs, next_probe (entropy-maximizing), update_beliefs (Bayesian propagation), coverage_report, is_converged, calibrate_beliefs, knowledge_fringes
- `knowledge_map_gen.py`: LLM-powered graph generation from domain descriptions
- `knowledge_map_simulation.py`: Monte Carlo simulation for convergence testing
- Research documents on advanced algorithms and assessment datasets

## 2026-03-19 -- Experiments 15-21, incremental clustering, persistent cache

### Added
- Experiments 15-21: large corpus novelty (27K claims), cross-lingual retrieval, query expansion (PRF), incremental clustering, NFCorpus eval, persistent embedding cache, All-but-the-top whitening
- `IncrementalCentroidCluster`: streaming-compatible clustering matching batch quality at threshold >= 0.85, 1.8x faster
- `PersistentEmbeddingCache`: SQLite-backed cache with 83-452x speedup on warm hits
- All-but-the-top whitening mode: simpler math, matches Soft-ZCA performance (+27.4% NN-gap)

### Key findings
- PRF query expansion hurts search quality (-1.2% to -7.2%) -- rejected
- Cross-lingual MRR=1.0 for Norwegian-to-English retrieval without translation
- Incremental clustering has zero order sensitivity at threshold >= 0.85

## 2026-03-19 -- CI, development workflow, experiments 13-14

### Added
- GitHub Actions CI running all tests on every PR
- Experiments 13-14: similarity graph layer (BFS surfaces 64% items vector misses), task-specific embeddings (not worth it -- search and novelty anti-correlated at -0.953)

## 2026-03-19 -- Pair extraction and classification

### Added
- `extract_pairs()`: cross-group pair extraction from similarity matrices
- `classify_pairs()`: cosine + NLI cascade for pair classification
- `complete_linkage_cluster()`: stricter clustering variant
- Petrarca calibration thresholds document

### Fixed
- Build backend configuration in pyproject.toml

## 2026-03-19 -- Amygdala v2: experiment-driven optimization

### Changed
- Default embedding model changed from `all-MiniLM-L6-v2` to `paraphrase-multilingual-MiniLM-L12-v2` (better accuracy, Norwegian support, faster)
- Whitening changed from default-on to opt-in (hurts diverse corpora, helps domain-specific)

### Added
- Matryoshka truncation support (`truncate_dim=`)
- Centroid-distance specificity in novelty scoring (+17% separation)
- Adaptive top-K for novelty (K=1 at <=50 items, K=10 at 1000+)
- NLI classification functions (`nli_classify`, `nli_classify_batch`) for contradiction detection
- FTS5 query sanitization (fixed SciFact from 0.003 to 0.638 nDCG)
- Cross-encoder reranking (`rerank()`)
- Text genericization (strip numbers, dates, URLs before embedding)
- Temporal decay for novelty scoring
- 12 experiment scripts with full results
- Eval scripts for STS-B, SciFact, QQP, Norwegian PAWS-X
- RESEARCH.md documenting all experiment findings
