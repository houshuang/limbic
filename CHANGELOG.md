# Changelog

All notable changes to the limbic monorepo (formerly amygdala) are documented here. Dates are in CET.

---

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
