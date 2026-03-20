# Amygdala Experiment Plan

**Created**: 2026-03-19
**Goal**: Systematically optimize amygdala's algorithms using the Karpathy Loop pattern (modify → measure → keep/discard), with eval data from political proposals, knowledge system, annotation system, and conversation search tool.

## Language Strategy

Most content is English. Some Norwegian (political proposals dataset, Norwegian education claims).
- **Primary**: Best English embedding model
- **Fallback**: Detect Norwegian text → either use a multilingual model or auto-translate before embedding
- The system already translates to English pre-embedding, so Norwegian handling is secondary

## Eval Datasets

| Dataset | Location | Size | Ground Truth? | Used By |
|---------|----------|------|---------------|---------|
| Calibration variants | `local calibration data | 120 claims (30 para/ext/contra/unrel) | Yes — LLM-generated variants with human-verifiable labels | Exp 1, 2, 3 |
| Retrieval benchmark v2 | `local calibration data | 15 queries with 3-method answers + quality scores | Semi — quality scored by LLM judge | Exp 4 |
| Chat search query log | `~/.conversation search tool/index.db` → `search_log` table | 108 real queries | Yes — real user queries (relevance is implicit) | Exp 4 |
| political proposals dataset | `local eval data → `llm_proposals` | 8,766 proposals with topics, specificity labels | **Mixed** — topics are LLM-assigned, specificity is LLM-classified then heuristically recalibrated. Use topic labels as soft ground truth only. | Exp 5 |
| Claims DB | `$AMYGDALA_EVAL_DB` | 27K claims with evidence levels, argument links | **Mixed** — argument links are LLM-classified. Use with caution. | Future |
| Real opposes pairs | `local calibration data (pareto data) | 100 labeled pairs (50 supports + 50 opposes) | **Mixed** — labels from knowledge graph, themselves LLM-assigned | Exp 6 |

## Experiments

### Exp 1: Embedding Model Comparison
**Status**: Running
**Hypothesis**: Newer models outperform all-MiniLM-L6-v2 on discrimination tasks.
**Models** (all locally cached):
- `all-MiniLM-L6-v2` (384d, 22M params) — current baseline
- `nomic-embed-text-v1.5` (768d, 137M, Matryoshka → 384/256/128) — top accuracy in class
- `all-mpnet-base-v2` (768d, 110M) — strongest general-purpose SBERT model
- `paraphrase-multilingual-MiniLM-L12-v2` (384d, 118M) — multilingual, for Norwegian fallback testing

**Metrics**:
- Mean cosine: related pairs vs unrelated pairs (discrimination gap)
- Per-variant-type accuracy at optimal threshold
- Embedding latency (ms per text)
- Norwegian text quality (embed a few Norwegian claim pairs, check similarity)

**Method**: Embed all 120 calibration claims + their source claims with each model. Compute pairwise cosines. Find optimal thresholds. Compare.

**Matryoshka test**: For nomic, also test at 384d and 256d truncation to see quality vs speed tradeoff.

### Exp 2: Whitening Dimension Sweep
**Status**: Running
**Hypothesis**: whiten_dims=256 may not be optimal.
**RESEARCH.md question**: "What's the optimal whitening dimension for 384-dim MiniLM?"

**Method**:
1. Sample 5,000 texts from domain claims DB
2. For each whiten_dims in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]:
   a. Fit whitening on the 5K sample
   b. Embed + whiten the 120 calibration claims
   c. Measure: mean pairwise cosine, discrimination gap, novelty F1
3. If Exp 1 identifies a better model, repeat with that model's native dimension

**Expected**: Sweet spot around 128-256 where discrimination peaks.

### Exp 3: Novelty Scoring Improvements
**Status**: Running
**Sub-experiments**:

**3a: Top-K sweep** — Vary K in "mean of top K similarities" from 1 to 20.
Previous experiments found K=7 optimal. Does this hold on calibration data?

**3b: Centroid-distance specificity** — Compute corpus centroid. For each claim:
`specificity = 1 - cosine(embedding, centroid)`. Multiply into novelty score.

**3c: Global/local weight sweep** — Current: 0.4 global + 0.6 local.
Sweep: (0.2/0.8), (0.3/0.7), (0.4/0.6), (0.5/0.5), (0.6/0.4), (0.7/0.3).

**Metric**: Novelty F1 on calibration set (paraphrase→low novelty, unrelated→high novelty).

### Exp 4: Convex Combination vs RRF
**Status**: Running
**Hypothesis**: Convex combination outperforms RRF with ~40 labeled queries (per ACM TOIS paper).

**Method**:
1. Build a test corpus from amygdala test data (20 documents from test_novelty.py)
2. Create 20+ query-relevance pairs
3. Get separate ranked lists from VectorIndex and FTS5Index
4. Compare: RRF (k=60) vs convex combination at α = 0.0, 0.1, ..., 1.0
5. Measure: MRR, NDCG@10, Recall@10

**Note**: The retrieval benchmark v2 data uses the knowledge system's search infrastructure, not amygdala's directly. We build a self-contained test within amygdala.

### Exp 5: Clustering Algorithm Comparison (FUTURE)
**Status**: Not started
**Data quality note**: Topic labels in the eval set are LLM-assigned (not human-verified). Use as soft ground truth — good enough for comparing algorithms against each other, not for absolute quality claims. The experiment log and design docs in political proposals should be checked for details on how labels were generated and any known issues.

### Exp 6: NLI Cross-Encoder Cascade (FUTURE)
**Status**: Not started

### Exp 7: Genericization Quality (FUTURE)
**Status**: Not started

### Exp 8: Full Karpathy Loop (FUTURE)
**Status**: Not started — runs after Exp 1-4 narrow the parameter ranges.

## Results Log

### Exp 1 Results (2026-03-19)
**Status**: Complete. See `results/exp1_results.json`.

| Model | Dim | Discrim Gap | Accuracy | Norwegian | Latency |
|-------|-----|------------|----------|-----------|---------|
| MiniLM-L6 | 384 | **0.730** | 75% | 0.16 | 0.40s |
| Nomic-v1.5 | 768 | 0.285 | 73% | 0.49 | 1.21s |
| Nomic-v1.5 | 384 | 0.288 | 73% | 0.46 | 1.21s |
| Nomic-v1.5 | 256 | 0.287 | 74% | 0.47 | 1.21s |
| MPNet-base | 768 | 0.714 | 75% | 0.19 | 0.70s |
| **Multilingual-MiniLM** | **384** | **0.728** | **80%** | **0.84** | **0.30s** |

**Findings**:
1. Multilingual-MiniLM wins overall: best accuracy (80%), best Norwegian (0.84), fastest (0.30s)
2. Nomic has worst discrimination — high anisotropy (unrelated at 0.54). Needs whitening or Matryoshka truncation
3. MiniLM-L6 and MPNet are English-only (Norwegian cosine 0.16-0.19)
4. Nomic's Matryoshka truncation (768→256d) barely affects quality

**Action**: Test nomic WITH whitening. Test multilingual-MiniLM as primary model with English-only fallback for speed-critical paths.

### Exp 2 Results (2026-03-19)
**Status**: Complete. See `results/exp2_results.json`.

| Dims | Mean Pairwise | Related | Unrelated | Gap | Accuracy |
|------|--------------|---------|-----------|-----|----------|
| raw  | 0.077 | 0.748 | -0.001 | **0.748** | 75% |
| 32   | 0.289 | 0.837 | 0.247 | 0.590 | **75.8%** |
| 128  | 0.101 | 0.733 | 0.072 | 0.661 | 74.2% |
| 256 (current) | 0.051 | 0.695 | 0.020 | 0.675 | 74.2% |
| 384  | 0.044 | 0.670 | 0.012 | 0.658 | 75% |

**Findings**:
1. **Raw (no whitening) has BEST discrimination gap** on this diverse dataset
2. Whitening raises the unrelated floor without proportionally raising related — net negative for diverse corpora
3. A prior experiment showed 0.80 mean pairwise because corpus was domain-homogeneous; this calibration set is diverse
4. Whitening helps within-domain corpora, hurts diverse ones

**Action**: Make whitening optional. Default off for diverse corpora, on for domain-specific. Also need to test whitening on nomic (which DOES have high anisotropy).

### Exp 3 Results (2026-03-19)
**Status**: Complete. See `results/exp3_results.json`.

**3a: Top-K sweep** (all F1=1.0 — separation is the key metric):
| K | Para novelty | Unrelated novelty | Separation |
|---|-------------|------------------|-----------|
| 1 | 0.271 | 0.878 | **0.607** |
| 3 | 0.536 | 0.899 | 0.363 |
| 7 | 0.667 | 0.925 | 0.258 |
| 10 (current) | 0.712 | 0.938 | 0.226 |

K=1 best for small corpus (30 sources). Previous experiments found K=7 best for 5,436 claims. K should scale with corpus size.

**3b: Centroid specificity improves separation by 17%**: 0.607 → 0.710. Validated.

**3c: Global/local weights**: Skipped — only 1 natural cluster in 30 source claims. Need larger corpus.

**Action**: Implement adaptive K based on corpus size. Add centroid-distance specificity weighting.

### Exp 4 Results (2026-03-19) — REVISED
**Status**: Complete. See `results/exp4_results.json` (rerun with 148 docs, 45 queries).

With a strong embedding model, fusion method is irrelevant — vector-only MRR=0.933.

**Key finding — RRF beats convex under embedding degradation:**

| Noise | Vec MRR | Convex (best α) | RRF (best k) | Winner |
|-------|---------|-----------------|--------------|--------|
| 0.0 | 0.933 | 0.933 | 0.933 | Tie |
| 0.1 | 0.760 | 0.760 | 0.771 | RRF |
| 0.2 | 0.227 | 0.228 | 0.377 | **RRF** |
| 0.5 | 0.043 | 0.045 | 0.178 | **RRF (4x)** |

RRF uses ranks (robust to noise), convex combination uses scores (amplifies noise). **Keep RRF as default.**

## Synthesis and Next Steps

### Key Takeaways from Exp 1-4

1. **Model choice matters more than post-processing.** Multilingual-MiniLM-L12-v2 beats MiniLM-L6-v2 on accuracy, speed, AND Norwegian — without any whitening.

2. **Whitening is situational.** Helps domain-homogeneous corpora, hurts diverse ones. Should be opt-in, not default. Nomic needs it most (0.54 unrelated baseline).

3. **Centroid specificity is validated** (+17% separation). Should be added to novelty.py.

4. **Novelty top-K should be adaptive** — K=1 for small index, K=5-10 for large. Could use K = max(1, min(10, len(index) // 100)).

5. **Keep RRF.** Robust to embedding quality degradation (4x better MRR than convex at noise=0.5).

### Code Changes Implemented (2026-03-19)

1. ✅ Default model → `paraphrase-multilingual-MiniLM-L12-v2`
2. ✅ Whitening is opt-in (must pass `whiten_dims=`)
3. ✅ Matryoshka truncation support (`truncate_dim=`)
4. ✅ Centroid-distance specificity in novelty_score() (30% blend)
5. ✅ Adaptive top-K in novelty scoring
6. ✅ All 48 tests passing

### Human-Annotated Eval (Running)

| Eval | Dataset | Size | Ground Truth | Status | Key Result |
|------|---------|------|-------------|--------|------------|
| STS-B | Sentence similarity | 1,379 pairs | Human 0-5 | **Done** | Multilingual-MiniLM Spearman=0.844, beats old default by +2.4pts |
| SciFact | Claim retrieval | 5K docs, 300 queries | Expert relevance | **Done** | nDCG@10=0.484 (vector), FTS5 useless on scientific terms |
| QQP | Duplicate detection | 10K pairwise + 2K corpus | Human labels | **Done** | AUC=0.860 (multilingual), MiniLM-L6 slightly better on English QQP |

### STS-B Results (Human-Annotated)

| Model | Variant | Spearman (↑) | Pearson | Dim |
|-------|---------|-------------|---------|-----|
| **Multilingual-MiniLM** | **raw** | **0.8441** | 0.8342 | 384 |
| MPNet-base | raw | 0.8342 | 0.8404 | 768 |
| MiniLM-L6 (old default) | raw | 0.8203 | 0.8274 | 384 |
| Multilingual-MiniLM | whitened-128d | 0.8367 | 0.8305 | 128 |
| MPNet-base | whitened-128d | 0.8372 | 0.8400 | 128 |

Key: Raw beats whitened on diverse data. 128d whitening > 256d whitening (less noise).

### SciFact Results (Expert-Annotated, with FTS5 query sanitization)

| Method | nDCG@10 | MRR@10 | Recall@10 |
|--------|---------|--------|-----------|
| Vector | 0.484 | 0.455 | 0.616 |
| **FTS5** | **0.638** | **0.606** | 0.766 |
| **Hybrid** | 0.619 | 0.570 | **0.796** |

FTS5 dominates on nDCG when queries are properly sanitized (scientific claims share exact terms with abstracts). Hybrid gets best recall (+3pp over FTS5). First run had 299/300 empty FTS5 results due to unsanitized queries — **fixed: added `_sanitize_query()` to `FTS5Index`**.

### Remaining Experiments

- Exp 5: Clustering on 20 Newsgroups (human topic labels)
- Exp 6: NLI cross-encoder cascade
- Exp 1b: Nomic + whitening combination
- Norwegian eval: NbAiLab/norwegian-paws-x
