# Changelog

All notable changes to the limbic monorepo (formerly amygdala) are documented here. Dates are in CET.

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
