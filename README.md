# Limbic

**Data curation toolkit: embeddings, search, proposals, and AI-assisted verification.**

## Drive: choose the first move before the swarm

`skills/drive` is a shared Codex/Claude planning skill for open-ended voice dumps
such as “research this” and “improve this.” It retrieves the nearest local
precedents, proposes one representative pilot, and stops for human judgment. The
v0 policy deliberately permits no worker calls or project mutation.

```bash
python -m limbic.drive calibrate
python -m limbic.drive validate /path/to/drive-plan.json
```

The three bundled calibration cases capture costly failure modes from the NRK
apps, the Otak/Hirsch investigation, and the Codex/Claude workflow research.
`calibrate` replays all of them; a policy change that would have re-allowed a
past mistake fails there rather than in a live session.

The same checks are available as a library, so a host that builds plans itself
can gate them without shelling out:

```python
from limbic.drive import validate_plan, check_calibrations, SCHEMA_VERSION

violations = validate_plan(plan)      # [] means the plan is allowed to run
if violations:
    raise ValueError(violations)

failures = [c for c in check_calibrations() if c.errors]
for f in failures:
    print(f.case_id, f.errors)     # CalibrationResult
```

`load_calibration_cases()` returns the bundled cases as plain dicts if you want
to extend the set or inspect what a case actually asserts. `SCHEMA_VERSION`
identifies the plan shape `validate_plan` expects, so a host that stores plans
can tell an old card from a current one.

`validate_plan` returns *every* violation rather than the first, so a plan gets
one round of correction instead of one per rule.

Limbic grew out of the same problems appearing across multiple projects:

- **otak / alif** — a 67K-node claims-first knowledge system where new annotations needed novelty detection ("is this claim already captured?"), clustering for dedup, and cosine+NLI cascade to tell paraphrases from contradictions
- **petrarca** — a news curation pipeline that needed document-level similarity matching, calibrated thresholds for "related" vs "near-duplicate," and hybrid search across multilingual content
- **kulturperler** — a Nordic performing arts archive (10,000+ entities) where deduplicating persons required fuzzy matching with veto gates, merging records meant cascade-relinking all performances and credits, and LLM verification of 2,400+ works needed budget control across 30+ audit sessions (~$270 total)
- **conversation search** — hybrid RRF search over chat history, where the FTS5 query sanitization and cross-encoder reranking patterns were first validated
- **reading/annotation tools** — novelty scoring and `classify_pairs` to detect when new annotations overlap with existing knowledge

The same patterns kept recurring: deduplicating entities by fuzzy name, merging records with cascading references, tracking what an LLM had verified, staying within API budgets, searching across languages. Limbic is the generalized result: three packages that handle the full pipeline from **finding patterns** in data to **managing the changes** to **verifying correctness**, plus a fourth (`limbic.drive`) that decides what to spend effort on before any of it starts.

## Three packages, one pipeline

```
limbic.amygdala          limbic.hippocampus          limbic.cerebellum
 finds patterns            manages changes             verifies correctness
 ─────────────           ──────────────────          ─────────────────────
 Embedding               Proposals                   Batch processing
 Vector search            (modify/merge/delete        (resumable, budget-
 Hybrid search             with lifecycle)              tracked, persistent)
 Query expansion         Cascade merges              Multi-tier orchestrator
 Multi-list RRF           (relink all references       (triage -> deep verify
 Novelty detection         when merging entities)       with auto-escalation)
 Clustering              Deduplication               Audit logging
 Document similarity      (veto-gate filtering)       (JSONL with analysis)
 Knowledge mapping       Validation                  Context builder
 LLM client               (composable rules)           (for LLM prompts)
 Calibration metrics     YAML store                  Cost logging
 Temporal reasoning        (file-locked atomic)        (cross-project, dashboard)
 Wikidata client          Wikidata resolver           Claude / Codex CLI
 Retrieval eval            (deterministic scoring)      (structured + agentic)
 Serendipity links                                    Agent isolation
 SQLite helpers                                       Windowed extraction
```

Plus **`limbic.drive`** off to the side: a planning policy that turns an
open-ended request into one bounded pilot before any of the above runs.

| Package | Purpose | Core dependency |
|---------|---------|-----------------|
| **limbic.amygdala** | Find patterns: embed, search, deduplicate, score novelty | numpy, sentence-transformers |
| **limbic.hippocampus** | Manage changes: proposals with review lifecycle, cascade merges, validation | pyyaml |
| **limbic.cerebellum** | Verify correctness: LLM-assisted batch audits with budget control, cross-project cost logging, CLI wrappers, agent isolation | (none beyond stdlib; litellm optional for cost computation) |
| **limbic.drive** | Decide what to do first: validate a plan against a calibration-first policy before spending anything | (none beyond stdlib) |

Each of the three core packages has its own detailed README in its directory.

## Is this for you?

**Good fit:**
- You have 1K–100K short texts (claims, findings, notes, entity records) and need search, deduplication, or novelty detection
- You maintain a dataset where entities reference each other and need to merge duplicates without breaking links
- You want LLM-assisted data curation with budget control, resumable batches, and audit trails
- Your corpus is **domain-focused** (e.g., all about one field) where off-the-shelf embeddings struggle to differentiate similar items
- You need **multilingual** support (especially English + Norwegian, but any language pair that sentence-transformers supports)
- You want numpy-based search without the operational overhead of a vector database
- You want hybrid search (vector + full-text) with a single `pip install`

**Not a good fit:**
- You need billion-scale vector search (use FAISS, Milvus, or Qdrant)
- You need document-level RAG with chunking strategies (use LlamaIndex or LangChain)
- You only need basic `sentence-transformers` — limbic.amygdala adds value through whitening, novelty, clustering, and hybrid search on top of it

## Install

```bash
pip install -e .

# With YAML-backed proposals and data store:
pip install -e ".[hippocampus]"

# With LLM support (for knowledge_map generation, novelty NLI cascade):
pip install -e ".[llm]"

# Everything for development:
pip install -e ".[dev,llm,hippocampus]"
```

**Requirements:** Python >= 3.11, numpy, sentence-transformers. No vector database needed.

---

## limbic.amygdala

**Embedding, search, novelty detection, and clustering for knowledge-dense text corpora.** Optimized for collections of claims, research findings, notes, and annotations — not generic documents. See [limbic/amygdala/README.md](limbic/amygdala/README.md) for full documentation.

### What's inside

| Module | What it does | Key numbers |
|--------|-------------|-------------|
| **embed** | Sentence embedding with 3 whitening modes, Matryoshka truncation, genericization, persistent cache | 83–452x speedup with SQLite cache; +32% nearest-neighbor separation with Soft-ZCA whitening |
| **search** | Numpy vector search, SQLite FTS5, hybrid RRF fusion, cross-encoder reranking, multi-list RRF with contribution tracing, LLM query expansion (lex/vec/hyde) | RRF 4x more robust than convex fusion; reranking +16% nDCG on medical; query expansion 3-5x score improvement |
| **novelty** | Multi-signal novelty scoring: global + topic-local + centroid specificity + temporal decay + NLI cascade | +17% novel/known separation with centroid specificity; NLI fixes 94% of high-cosine contradictions |
| **cluster** | Greedy centroid clustering (batch + incremental), complete linkage, pairwise cosine, confidence-calibrated pair classification | Incremental matches batch quality at threshold >= 0.85, 1.8x faster; order-sensitive at lower thresholds |
| **document_similarity** | Document-level thematic similarity using weighted multi-field embeddings | 94% accuracy on human-rated pairs; AUROC=0.930 on 300-pair dataset; rho=0.818 |
| **calibrate** | Cohen's kappa, LLM judge validation (Bootstrap Validation Protocol), intra-rater reliability | Validates LLM judges against human gold labels |
| **cache** | Persistent SQLite-backed embedding cache | 20K texts: 48s cold → 585ms warm |
| **index** | SQLite document/chunk storage with hybrid search | Single-file, zero-config, FTS5 built in |
| **knowledge_map** | Adaptive knowledge probing via EIG selection with Bayesian belief propagation, batch probing, KST fringes | Converges in 5–8 questions on 20-node graphs; Bayesian propagator 42% faster than heuristic on chains |
| **llm** | Multi-provider LLM client (Gemini, Anthropic, OpenAI) with structured output and retry | Auto-fallback, cost tracking, async + sync |
| **temporal** | Uncertain-date parsing ("940s", "circa 942", "4th century BC", EDTF) into integer year ranges, Allen interval relations, soft plausibility scoring | Indexes as two ints; `edtf` extra optional |
| **wikidata** | Cache-backed, rate-limited Wikidata client: search, get, batched get_many, SPARQL | 30-day payload cache, 5 req/s token bucket, maxlag-aware |
| **retrieval_eval** | Pooled-judgment IR evaluation: pool -> LLM-judge -> nDCG / Recall / MRR / MAP, with strata | Answers "which retrieval knob actually wins?" |
| **serendipity** | Non-obvious link finding: inverted-U similarity band, cross-facet bonus, Swanson ABC bridging | 70% of surfaced links rated surprising *and* useful |

### Quick start

```python
from limbic.amygdala import EmbeddingModel, VectorIndex, HybridSearch, FTS5Index, novelty_score

# Embed text (multilingual model handles English, Norwegian, and 50+ languages)
model = EmbeddingModel()
vecs = model.embed_batch(["Education improves democratic participation",
                          "Schools need more funding for special education",
                          "Quantum entanglement in superconductors"])

# Vector search
index = VectorIndex()
index.add(["claim1", "claim2", "claim3"], vecs)
results = index.search(model.embed("democracy and education"), limit=2)

# Novelty scoring -- is this claim new to the corpus?
score = novelty_score(model.embed("Teachers need better training"), index)
# 0.0 = duplicate, 1.0 = completely novel

# Hybrid search (vector + full-text via SQLite FTS5)
fts = FTS5Index()
for i, text in enumerate(["Education improves democratic participation",
                          "Schools need more funding"]):
    fts.add(f"claim{i+1}", text)
hybrid = HybridSearch(vector_index=index, fts_index=fts)
results = hybrid.search(model.embed("school funding"), "school funding", limit=5)
```

### Recipe: Analyze a corpus of responses

A common task: you have 50–500 texts (policy responses, reviews, survey answers) and want to find shared arguments, unique insights, and contradictions. This pipeline chains embedding, whitening, clustering, and novelty detection:

```python
from limbic.amygdala import (
    EmbeddingModel, VectorIndex, greedy_centroid_cluster,
    batch_novelty, pairwise_cosine, extract_pairs, classify_pairs,
)

# 1. Embed with domain-appropriate settings
#    genericize=True strips numbers/dates that poison similarity
#    whiten_epsilon=0.1 spreads the narrow embedding cone (essential for domain corpora)
model = EmbeddingModel(genericize=True, whiten_epsilon=0.1, cache_path="cache.db")
texts = [claim["text"] for claim in claims]
model.fit_whitening(texts)
vecs = model.embed_batch(texts)

# 2. Cluster to find shared arguments (0.85 threshold after whitening)
clusters = greedy_centroid_cluster(vecs, threshold=0.85)
# Each cluster = group of claims making ~the same argument
# Count distinct sources per cluster → "how many respondents say this?"

# 3. Score novelty per claim
index = VectorIndex()
index.add([str(i) for i in range(len(vecs))], vecs)
scores = batch_novelty(vecs, index)
# 0.0 = everyone says this, 1.0 = only this source says it
# Aggregate per source to rank "who brings the most novel arguments?"

# 4. Find contradictions (cosine can't distinguish agree vs disagree)
pairs = extract_pairs(pairwise_cosine(vecs), threshold=0.72)
classified = classify_pairs(texts, pairs)
# Returns KNOWN (paraphrase), NEW (contradiction), EXTENDS (elaboration)
```

See also the [entity dedup recipe](#deduplication-with-veto-gates) in limbic.hippocampus and the [batch verification recipe](#quick-start-batch-processing) in limbic.cerebellum.

### Embedding and whitening

The default model is `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). Chosen over `all-MiniLM-L6-v2` based on experiments:

| Metric | Multilingual-MiniLM-L12 | MiniLM-L6 |
|--------|------------------------|-----------|
| Classification accuracy | **80%** | 75% |
| Norwegian cross-lingual quality | **0.84** | 0.16 |
| Contradiction separation | **0.15 gap** | ~0 |
| Speed (150 texts) | **0.30s** | 0.35s |
| STS-B Spearman | **0.844** | 0.822 |

#### Whitening for domain-specific corpora

Off-the-shelf embeddings put everything in a narrow cone — unrelated texts in the same domain score 0.7+ cosine similarity, making it hard to distinguish "similar" from "identical." Whitening spreads the distribution:

```python
from limbic.amygdala import EmbeddingModel

# Soft-ZCA whitening (recommended for domain-focused corpora)
model = EmbeddingModel(whiten_epsilon=0.1)
model.fit_whitening(corpus_texts)  # compute whitening transform
vec = model.embed("now whitened")  # still 384-dim, much better separation

# Before whitening: mean pairwise cosine ~0.80
# After whitening:  mean pairwise cosine ~0.24
# Result: +32% nearest-neighbor separation gap
```

Three whitening modes, all opt-in:

| Mode | Code | Effect | When to use |
|------|------|--------|-------------|
| **Soft-ZCA** | `EmbeddingModel(whiten_epsilon=0.1)` | +32% NN-gap, preserves all dims | Domain-focused corpora (recommended) |
| **All-but-the-top** | `EmbeddingModel(whiten_abt=1)` | +27% NN-gap, simpler math | When you don't want to tune epsilon |
| **PCA** | `EmbeddingModel(whiten_dims=128)` | +24% NN-gap, reduces dims | When you need dimensionality reduction |

**Rule of thumb:** If your corpus is domain-focused (all about education, all about medicine, all about politics), **always whiten**. Without whitening, clustering thresholds below ~0.90 produce excessive false positives — a 0.70 threshold on raw domain embeddings groups nearly everything together. After whitening, 0.85 is a reliable starting threshold. Validated on 27K education claims (otak) and 6.5K political proposals (MDG).

**Don't whiten diverse corpora.** On mixed-domain data, raw embeddings already separate well. Whitening helps when your entire corpus is about one field and everything looks the same to the model. The Karpathy-loop experiment (120 configs) confirmed: **current defaults are rank 1/120** — whitening is the biggest anti-pattern on diverse data.

#### Genericization (pre-embedding text normalization)

Strips numbers, dates, currencies, and URLs before embedding. Use when your texts contain variable specifics around the same argument:

- **Policy/finance:** "allocate 50M" vs "allocate 200M" → same argument, different amount
- **Municipal/regional:** "Oslo kommune" vs "Bergen kommune" → same proposal, different place
- **Legal/regulatory:** "§ 2-3" vs "§ 1-1" → same type of reference, different section

```python
model = EmbeddingModel(genericize=True)
# +14% accuracy on number/date-heavy text, no effect on proper nouns (Experiment 7)
```

Skip when proper nouns carry real semantic signal (e.g., comparing claims *about* different people). Combine with whitening for domain-focused corpora:

```python
model = EmbeddingModel(genericize=True, whiten_epsilon=0.1)
```

#### Other embedding features

```python
from limbic.amygdala import EmbeddingModel

# Matryoshka truncation (reduce dimensions for speed/storage)
model = EmbeddingModel(truncate_dim=256)

# Persistent embedding cache (survives restarts)
model = EmbeddingModel(cache_path="embeddings.db")
# 20K texts: 48s cold -> 585ms warm (83x speedup)
# ~2.2 KB per 384-dim entry, stores pre-whitening vectors
```

### Search

Three search modes that compose together:

```python
from limbic.amygdala import VectorIndex, FTS5Index, HybridSearch, rerank

# Pure vector search -- brute-force cosine, faster than ANN at <100K vectors
vi = VectorIndex()
vi.add(ids, embeddings)
results = vi.search(query_vec, limit=10)
results = vi.search(query_vec, limit=10, filter_ids={"id1", "id2"})  # filtered

# Pure full-text search -- SQLite FTS5 with porter stemming
fts = FTS5Index("index.db")  # or ":memory:"
fts.add("doc1", "some text content", metadata={"source": "arxiv"})
results = fts.search("text content", limit=10)

# Hybrid search -- Reciprocal Rank Fusion combines both
hybrid = HybridSearch(vector_index=vi, fts_index=fts)
results = hybrid.search(query_vec, "query text", limit=10)

# Cross-encoder reranking (requires results with content)
reranked = rerank("query text", results)  # uses ms-marco-MiniLM-L-6-v2
```

**Why RRF over convex fusion?** Tested on 148 documents with 45 queries. RRF is 4x more robust when embedding quality degrades (common in domain-specific corpora). It's also parameter-free.

**Why brute-force over ANN?** At <100K vectors, numpy matrix multiply is faster than index-building overhead. No need for FAISS, Annoy, or HNSWlib until you're well past 100K.

#### Search benchmarks

| Dataset | Vector nDCG@10 | FTS5 nDCG@10 | Hybrid | Hybrid + rerank |
|---------|---------------|-------------|--------|-----------------|
| SciFact (5K docs, 300 queries) | 0.484 | 0.638 | 0.674 | **0.641** |
| NFCorpus (3.6K docs) | 0.235 | 0.126 | 0.286 | **0.333** |

FTS5 dominates on scientific text (exact terminology matters); vector dominates on medical queries (semantic matching matters). Reranking helps on NFCorpus (+16%) but slightly hurts on SciFact (-5%), likely because scientific terminology already gives exact matches high FTS5 scores.

### Multi-list RRF and query expansion

For advanced search scenarios, limbic provides LLM-powered query expansion and multi-list fusion with full contribution tracing:

```python
from limbic.amygdala import (
    EmbeddingModel, VectorIndex, FTS5Index,
    expand_query, multi_list_rrf, expanded_hybrid_search, strong_signal,
)

# --- Multi-list RRF with contribution tracing ---
# Fuse any number of ranked lists (from different search strategies)
fused = multi_list_rrf(
    [vec_results, fts_results, reranked_results],
    ["vector", "fts", "reranked"],
)
for r in fused[:3]:
    print(f"{r.id}: {r.score:.4f}")
    for t in r.traces:
        print(f"  {t.list_label}: rank {t.rank} → +{t.contribution:.4f}")
# Top-rank bonuses (QMD-style): +0.05 for rank 1, +0.02 for ranks 2-3
# Each fused hit is a TracedResult carrying RRFContribution entries, so a
# surprising ranking is attributable to the list that produced it rather than
# being an opaque blended score.

# --- LLM query expansion ---
# Generates lex (keyword variants), vec (semantic rephrases), hyde (hypothetical docs)
expanded = expand_query(
    "database lock problems",
    domain_context="The corpus contains SQLite WAL mode discussions",
)
# [ExpandedQuery(type="lex", query="deadlock contention WAL"),
#  ExpandedQuery(type="vec", query="concurrent write failures in SQLite"),
#  ExpandedQuery(type="hyde", query="When multiple writers attempt..."), ...]

# --- One-call expanded hybrid search ---
# Combines expand_query + multi_list_rrf in a single call
model = EmbeddingModel()
results = expanded_hybrid_search(
    "effect of digital tools on learning",
    vector_index=vi,
    fts_index=fts,
    embed_fn=model.embed,
    domain_context="Nordic education research",
)
# Each result has full traces showing which sub-query contributed

# --- Skip expansion when not needed ---
top_scores = [r.score for r in first_pass_results[:2]]
if strong_signal(top_scores, threshold=0.82, gap=0.12):
    # Top result is strong and clearly separated — skip expensive LLM expansion
    pass
```

**Why query expansion?** A single query misses vocabulary the user doesn't think of. Lex variants find different keywords; vec variants capture different framings; hyde variants bridge the query-document vocabulary gap by generating hypothetical answers. Multi-list RRF fuses all results without manual weight tuning.

### Novelty detection

Novelty scoring answers: **"Is this text saying something new relative to what I already have?"**

```python
from limbic.amygdala import VectorIndex, novelty_score, batch_novelty, nli_classify

# Basic novelty -- 0.0 = exact duplicate, 1.0 = completely novel
score = novelty_score(query_vec, index)

# With topic-local context (higher weight to same-category neighbors)
score = novelty_score(query_vec, index, category_ids={"id1", "id2", "id3"})

# With centroid specificity (generic claims near corpus center get dampened)
# +17% separation on diverse data
score = novelty_score(query_vec, index, use_centroid_specificity=True)

# With temporal decay (older items contribute less to "already known")
# Half-life ~35 days at lambda=0.02
ages = {"id1": 0.0, "id2": 30.0, "id3": 90.0}  # age in days
score = novelty_score(query_vec, index, timestamps=ages, decay_lambda=0.02)

# corpus_centroid(vectors) gives the mean direction the specificity signal
# dampens against; compute it once and reuse it across a batch.

# NLI cascade -- cosine can't tell paraphrases from contradictions
# (both score ~0.73). NLI cross-encoder resolves this:
result = nli_classify("Education improves outcomes",
                      "Education has no effect on outcomes")
# -> {"label": "contradiction", "contradiction": 0.92, ...}
```

#### The cosine similarity problem

Cosine similarity **cannot distinguish agreement from disagreement**. Two claims that say opposite things about the same topic often have *higher* cosine similarity than two unrelated claims. This is well-documented in the literature but rarely addressed in embedding libraries.

The `classify_pairs()` function implements a cosine + NLI cascade:
- **Below threshold** (e.g., 0.72): cosine-confident NEW (skip NLI)
- **Above threshold**: NLI cross-encoder runs to catch high-cosine contradictions
  - Entailment → KNOWN, Contradiction → NEW, Neutral → EXTENDS

This catches the case that matters most: claims that cosine says are similar but actually contradict each other. 94% accuracy at ~13ms per pair.

#### Performance at scale

Tested on a 27K-claim knowledge base:
- `novelty_score()`: **1.1ms per call**
- `batch_novelty()`: **3.6ms per claim** (brute-force bottleneck)
- Adaptive K scales with index size: K=1 at <=50 items, K=10 at 1000+

### Clustering

Two strategies optimized for deduplication, not topic discovery:

```python
from limbic.amygdala import greedy_centroid_cluster, IncrementalCentroidCluster, pairwise_cosine, extract_pairs

# Batch clustering -- good when you have all vectors upfront
clusters = greedy_centroid_cluster(embeddings, threshold=0.85)
# Returns list of clusters (each a list of indices). Singletons excluded.

# Incremental clustering -- for streaming/continuous ingestion
# Close to batch quality at threshold >= 0.85, 1.8x faster (order-sensitive)
clusterer = IncrementalCentroidCluster(threshold=0.85)
for i, vec in enumerate(vecs):
    cluster_id = clusterer.add(i, vec)
clusters = clusterer.get_clusters(min_size=2)

# Pairwise similarity + pair extraction
sim_matrix = pairwise_cosine(embeddings)
pairs = extract_pairs(sim_matrix, threshold=0.7)
# Cross-group pairs only (e.g., cross-document dedup):
pairs = extract_pairs(sim_matrix, threshold=0.7,
                      groups=["doc1", "doc1", "doc2", "doc2"],
                      cross_group_only=True)
```

#### Choosing a threshold

The right threshold depends on whether you've whitened your embeddings:

| Corpus state | Threshold | Rationale |
|---|---|---|
| Raw embeddings, diverse corpus | 0.70–0.75 | Embeddings already spread well across domains |
| Raw embeddings, domain-focused | 0.90+ | Narrow similarity cone — lower thresholds group everything |
| **Whitened embeddings** (recommended) | **0.85** | Whitening restores meaningful spread |

If your largest cluster has 50+ members, your threshold is too low or you need whitening. Always validate a sample of cluster pairs with an LLM before using results downstream (~$0.001/pair with Gemini Flash).

**Why greedy centroid over union-find?** Union-find causes transitive chaining — at threshold 0.85, it produces clusters of 1,500+ items. Greedy centroid caps naturally at ~50. Discovered this empirically when clustering 27K claims.

**Why not HDBSCAN?** Both tested. Similar V-measure (~0.55) on 20 Newsgroups. Both are designed for dedup, not topic discovery. Greedy centroid is simpler, needs no hyperparameter tuning, and works incrementally.

### Document similarity

Find thematically similar documents in a corpus using weighted multi-field embeddings:

```python
from limbic.amygdala import Document, find_similar_documents

docs = [
    Document(id="art1", texts={"summary": "Sicily's history spans Greek, Roman, and Norman periods.", "claims": "Greeks founded Syracuse in 734 BC."}),
    Document(id="art2", texts={"summary": "Sicilian Baroque architecture defines the island's cultural identity.", "claims": "Sicilian Baroque is a UNESCO World Heritage style."}),
    Document(id="art3", texts={"summary": "Python asyncio provides concurrent I/O execution.", "claims": "Event loops manage coroutine scheduling."}),
]

# Weighted multi-field embedding (best strategy: 94% accuracy, rho=0.818)
pairs = find_similar_documents(
    docs,
    text_fields={"summary": 0.5, "claims": 0.5},
    threshold=0.52,  # calibrated for 80% precision, 78% recall
)
# -> [SimilarityPair(id_a="art1", id_b="art2", score=0.74, field_scores={"summary": 0.78, "claims": 0.65})]
```

**Why weighted multi-field?** Embedding summary and claims separately then combining with equal weights (0.5/0.5) outperforms concatenating them into one text (94% vs 89% accuracy). Concatenation lets the longer text dominate; weighted combination preserves the distinct signal geometry of each representation. Developed and calibrated for petrarca's news article similarity matching.

Calibrated thresholds from 300 LLM-rated + 18 human-rated article pairs:

| Use case | Threshold | Precision | Recall | F1 |
|----------|-----------|-----------|--------|-----|
| Feed ranking (recall-focused) | 0.49 | 71% | 82% | 76% |
| Briefing card (balanced) | 0.52 | 80% | 78% | 79% |
| High confidence | 0.55 | 91% | 75% | 82% |
| Near-duplicate detection | 0.64 | 96% | 73% | 83% |

### Knowledge mapping

Adaptive knowledge probing: efficiently map what someone knows about a topic using information theory.

```python
from limbic.amygdala.knowledge_map import (
    KnowledgeGraph, init_beliefs, next_probe, next_probe_batch,
    update_beliefs, coverage_report, knowledge_fringes,
)

# Define a knowledge graph (or generate one with LLM -- see below)
graph = KnowledgeGraph(nodes=[
    {"id": "crdt", "title": "CRDTs", "level": 1, "description": "Conflict-free replicated data types"},
    {"id": "lamport", "title": "Lamport clocks", "level": 2, "prerequisites": ["crdt"]},
    {"id": "mirror", "title": "Mirror protocol", "level": 3, "prerequisites": ["crdt", "lamport"]},
])

# Initialize — Bayesian propagation by default (best accuracy)
state = init_beliefs(graph)

# Get next question -- maximizes expected information gain
probe = next_probe(graph, state)
# -> {"node_id": "crdt", "question_type": "recognition", "information_gain": 1.2, ...}

# Or get a batch of diverse, high-value questions at once
probes = next_probe_batch(graph, state, n=3)

# User responds with familiarity level
update_beliefs(graph, state, "crdt", "solid")
# Bayesian propagation: knowing CRDTs well -> prerequisites likely known too

# Check coverage
report = coverage_report(graph, state)
# -> {"known": [...], "unknown": [...], "uncertain": [...], "coverage_pct": 33.3}

# Find learning frontier (KST fringes)
fringes = knowledge_fringes(graph, state)
# -> {"outer_fringe": ["lamport"], ...}  -- ready to learn next
```

Features:
- **Expected Information Gain** probe selection (simulates all possible answers)
- **Batch probe selection** via `next_probe_batch(n)` — diversity-aware, avoids redundant probes
- **Bayesian belief propagation** (Pearl's forward-backward, 0.16ms) — 42% faster convergence than heuristic on chains. Also provides implicit overclaiming defense via constraint propagation.
- **Overclaiming detection** via foil concepts (signal detection theory)
- **KST inner/outer fringe** computation for learning path recommendations
- **LLM-powered graph generation** from domain descriptions or document outlines
- **DAG validation**: rejects cycles and duplicate node IDs at construction

```python
# Generate a knowledge graph from a topic description
from limbic.amygdala.knowledge_map_gen import graph_from_description
graph = await graph_from_description("Conflict-free replicated data types")
# -> 15-50 nodes with prerequisites, obscurity levels, descriptions
```

### LLM client

`limbic.amygdala.llm` is a thin multi-provider client: `generate` for text,
`generate_structured` for JSON (plus `_sync` variants). Models are addressed by
short key, not wire id — `generate_structured(..., model="luna")` — and only keys
present in `MODELS` are accepted.

| Key | Wire id | $/M in | $/M out |
|---|---|---|---|
| `gemini38-flash` | gemini-3.8-flash | 0.75 | 3.75 |
| `gemini35-flash` | gemini-3.5-flash | 1.50 | 9.00 |
| `gemini35-flash-lite` | gemini-3.5-flash-lite | 0.30 | 2.50 |
| `gemini31-pro` | gemini-3.1-pro-preview | 2.00 | 12.00 |
| `gemini31-flash-lite` | gemini-3.1-flash-lite | 0.25 | 1.50 |
| `gemini3-flash` *(default)* | gemini-3-flash-preview | 0.50 | 3.00 |
| `gemini25-flash` | gemini-2.5-flash | 0.30 | 2.50 |
| `gemini25-pro` | gemini-2.5-pro | 1.25 | 10.00 |
| `fable` | claude-fable-5-1 | 10.00 | 50.00 |
| `opus` | claude-opus-5 | 5.00 | 25.00 |
| `sonnet` | claude-sonnet-5 | 2.00 | 10.00 |
| `haiku` | claude-haiku-4-5-20251001 | 1.00 | 5.00 |
| `sol` | gpt-5.6-sol | 4.00 | 20.00 |
| `terra` | gpt-5.6-terra | 2.00 | 12.00 |
| `luna` | gpt-5.6-luna | 0.20 | 1.20 |
| `gpt55` | gpt-5.5 | 5.00 | 30.00 |
| `gpt54-mini` | gpt-5.4-mini | 0.75 | 4.50 |
| `gpt54-nano` | gpt-5.4-nano | 0.20 | 1.25 |
| `gpt41-mini` | gpt-4.1-mini | 0.40 | 1.60 |
| `gpt41-nano` | gpt-4.1-nano | 0.10 | 0.40 |

Keys are read from `GEMINI_KEY`/`GOOGLE_API_KEY`, `ANTHROPIC_KEY`/`ANTHROPIC_API_KEY`,
`OPENAI_KEY`/`OPENAI_API_KEY`.

```python
from limbic.amygdala.llm import generate_structured_sync

schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
result, meta = generate_structured_sync("Is this claim supported?", schema, model="luna")
print(meta["total_cost_usd"], meta["model"])
```

Notes:

- **Reasoning/thinking tokens are billed as output** on both Gemini and the
  GPT-5.x tiers, and `meta["output_tokens"]` includes them. A 5-token answer from
  `gemini38-flash` can carry 100+ thinking tokens; pass `thinking_budget=0` to
  suppress it on models that allow it (`gemini31-pro` requires thinking).
- **A reasoning model can spend its whole `max_tokens` budget thinking** and
  return an empty string. `FALLBACK` retargets to a second model when the response
  won't parse as JSON; raise `max_tokens` if you see this often.
- Structured calls send the schema to every provider, so the model is told what
  shape to return rather than just "return JSON".

### Temporal reasoning

Historical and archival data rarely carries a clean date. `temporal` parses the
expressions it *does* carry into integer year ranges, then answers interval
questions over them.

```python
from limbic.amygdala import parse_date, overlaps, during, plausibility_score

parse_date("940s")              # DateRange(start=940, end=949)
parse_date("circa 942")         # DateRange(start=942, end=942, approximate=True)
parse_date("4th century BC")    # DateRange(start=-400, end=-301)
parse_date("942-996")           # DateRange(start=942, end=996)
parse_date("196X")              # DateRange(start=1960, end=1969)   (EDTF, needs the extra)

# Allen interval relations: before, after, during, overlaps, meets, equals
overlaps(parse_date("940s"), parse_date("942-996"))   # True — was X alive when Y happened?

# Soft consistency instead of a hard filter: how plausible is this candidate
# date given the surrounding context? 1.0 inside, decaying outside.
plausibility_score(parse_date("1015"), context=parse_date("990-1030"))   # 1.0
plausibility_score(parse_date("1450"), context=parse_date("990-1030"))   # 0.015
```

Two integer years is coarser than full EDTF, but it indexes trivially (two
columns, a BETWEEN) and covers every query entity resolution actually asks. The
precision flags (`approximate`, `uncertain`) are carried alongside rather than
folded into the range, so "c. 942" and "942?" stay distinguishable — and "circa
942" stays a *point* with a flag rather than silently widening into a decade you
never asserted. Widen it yourself if your domain wants that; `plausibility_score`
already decays softly outside the range, which covers most of the reason to.

Full EDTF strings are supported when the optional package is installed
(`pip install "limbic[temporal]"`). Both the ratified uppercase spelling (`196X`)
and the 2012-draft lowercase one (`196x`, `19uu`) parse, because archive data
predates the change.

### Wikidata client

```python
from limbic.amygdala import WikidataClient

wd = WikidataClient(user_agent="myproject/1.0 (you@example.com)",
                    cache_db_path="wikidata_cache.db")

candidates = wd.search("Rollo")          # ranked Candidate list (API-popularity biased)
entity = wd.get("Q57285")                # labels, aliases, descriptions, claims
entities = wd.get_many(["Q1", "Q2"])     # batched, up to 50 QIDs per HTTP call
rows = wd.sparql("SELECT ?x WHERE { ... }")
```

Every response goes through `PayloadCache` (30-day default TTL), so a re-run of
an enrichment pass costs nothing. Requests are rate-limited by an in-process
token bucket at Wikidata's published 5 req/s, and `maxlag` is honoured — a
`MaxlagError` means the server asked you to back off, not that the data is
missing. `WikidataNotFound` is the distinct "this QID doesn't exist" case.

A `user_agent` is required, not optional: Wikidata blocks anonymous bulk
clients, and the failure is a silent throttle rather than an error.

Errors are typed so a caller can tell them apart: `WikidataError` is the base,
`MaxlagError` means the server asked you to back off (retry later, the data is
fine), and `WikidataNotFound` means the QID genuinely does not exist. Label and
alias lookups walk `DEFAULT_LANGS` in order, so a missing English label falls
back rather than coming back empty.

For turning a *mention* into a QID rather than fetching a known one, see
[`hippocampus.wikidata_resolve`](#wikidata-entity-resolution).

### Retrieval evaluation

`calibrate` validates an LLM judge against humans; `retrieval_eval` validates
*retrieval*. It is the standard pooled-judgment IR loop, so "is hybrid better
than vector here?" gets a number instead of an anecdote.

```python
from limbic.amygdala import retrieval_eval as rev

# runs  : {query_id: {method_name: [doc_id, ...ranked best-first]}}
# qrels : {query_id: {doc_id: grade}}   graded 0-3
pooled = rev.pool(runs, depth=10)               # union of every method's top-10
qrels = rev.judge_pool(pooled, queries, doc_text=get_text,
                       judge_fn=rev.make_llm_judge())
scores = rev.score(runs, qrels, k_ndcg=10, k_recall=20, rel_threshold=2,
                   strata=query_category)       # per-category breakdown
print(rev.format_report(scores))
```

Pooling is what keeps the comparison fair: judging only one method's results
scores that method against its own definition of relevant. `judge_pool` accepts
`existing` qrels so adding a method re-judges only the newly pooled documents.

The pool is still only as wide as the methods in it — see
[Design decisions](#design-decisions-with-evidence) for what happened when a
genuinely different method was added late.

### Serendipity: non-obvious links

Retrieval optimises precision. This optimises *surprise* — pairs related enough
to be meaningful but far enough apart that you would not have connected them.

```python
from limbic.amygdala import serendipity as ser

pairs = ser.serendipity_pairs(
    ids, embeddings,
    metas=metas, facet_key=lambda m: m["source_type"],
    band=(0.55, 0.82),      # the inverted-U sweet spot — CALIBRATE THIS
    facet_bonus=0.15,       # crossing a source/era boundary is more surprising
    top=50,
)
# [{"a": ..., "b": ..., "sim": 0.63, "score": 0.91}, ...]

# Swanson ABC bridging: A and C aren't similar, but both relate strongly to B
bridges = ser.abc_bridges(ids, embeddings, low=0.4, high=0.7)
```

The band is embedding-space dependent and **must** be recalibrated per model: a
raw multilingual encoder compresses everything into a high, narrow range, while
whitening spreads the unrelated floor down. Two measured settings:

| Corpus | Embeddings | Band that worked |
|---|---|---|
| Whitened, domain-focused | Soft-ZCA, `whiten_epsilon=0.1` | `(0.55, 0.82)` (the default) |
| Raw multilingual MiniLM, multi-domain personal corpus | no whitening | `(0.42, 0.74)` |

Geometry only proposes candidates. Whether a link is *worth* anything is a
separate judgment — scoring candidates on surprise and usefulness separately
(0–3 each) rather than one blended score is what made the output usable; see
[Design decisions](#design-decisions-with-evidence).

### SQLite connection helper

```python
from limbic.amygdala import connect

conn = connect("my.db")  # or connect("my.db", readonly=True)
```

Applies all best practices automatically: WAL journal mode, 30s busy timeout, NORMAL synchronous, 64MB page cache, foreign key enforcement. Use this for any project that touches SQLite.

### Cross-lingual support

The multilingual model achieves **MRR=1.0** on Norwegian-to-English retrieval out of the box. No translation step needed — embed Norwegian and English text into the same space and search across languages natively.

```python
from limbic.amygdala import EmbeddingModel

model = EmbeddingModel()
v_no = model.embed("Utdanning er viktig for demokratiet")
v_en = model.embed("Education is important for democracy")
similarity = float(v_no @ v_en)  # -> 0.86
```

---

## limbic.hippocampus

**Proposal-based data change management with cascade merges, deduplication, and validation.** For datasets where entities reference each other and changes need human review before application. See [limbic/hippocampus/README.md](limbic/hippocampus/README.md) for full documentation.

### Quick start

```python
from limbic.hippocampus import ProposalStore, Proposal

# Set up a proposal store (creates pending/approved/applied/rejected directories)
store = ProposalStore("data/proposals")

# Create a modify proposal
store.create_modify(
    "person/42",
    field_changes={"name": "Henrik Ibsen", "birth_year": "1828"},
    title="Fix Ibsen birth year",
    reasoning="Was incorrectly listed as 1829",
    current_state={"name": "Henrik Ibsen", "birth_year": 1829},
)

# Create a merge proposal (source into target)
store.create_merge(
    "person/99", "person/42",
    title="Merge duplicate Ibsen",
    reasoning="Same person, different records from two import batches",
)

# Create a delete proposal
store.create_delete(
    "work/879",
    title="Remove orphaned work",
    reasoning="No performances reference this work",
)

# Lifecycle: pending -> approved -> applied (or rejected)
proposals = store.list_pending()
store.approve(proposals[0].id)
applied = store.list_approved()
store.mark_applied(applied[0].id)
```

### Cascade merges

When merging duplicate entities, all references must be relinked. The cascade module handles this declaratively:

```python
from limbic.hippocampus import ReferenceSpec, ReferenceGraph, apply_merge

# Declare how entity types reference each other
graph = ReferenceGraph([
    ReferenceSpec("performance", "work_id", "work"),
    ReferenceSpec("performance", "credits", "person", is_array=True, sub_field="person_id"),
    ReferenceSpec("work", "playwrights", "person", is_array=True),
    ReferenceSpec("episode", "performance_id", "performance"),
])

# Merge person/99 into person/42 -- automatically relinks all
# performances, works, and episodes that referenced person/99
changes = apply_merge(
    graph,
    source_id="99", target_id="42", entity_type="person",
    data_loader=my_loader, data_writer=my_writer, data_deleter=my_deleter,
)
# changes: ["Relinked performance/301.credits: 99 -> 42", "Deleted person/99"]
```

### Deduplication with veto gates

Candidate duplicate pairs pass through a chain of veto gates. Any gate can reject a pair:

```python
from limbic.hippocampus import VetoMatcher, CandidatePair, ExclusionList
from limbic.hippocampus import exact_field, initial_match, no_conflict, gender_check

matcher = VetoMatcher(
    gates=[
        initial_match("name"),           # first letter must match
        exact_field("birth_year"),       # if both have birth_year, must agree
        no_conflict("wikidata_id"),      # conflicting external IDs = not same person
        gender_check("name", male_names={"erik", "hans"}, female_names={"anna", "grete"}),
    ],
    exclusions=ExclusionList(),
)

pair = CandidatePair(
    id_a="42", id_b="99",
    fields_a={"name": "Henrik Ibsen", "birth_year": 1828},
    fields_b={"name": "Henrik J. Ibsen", "birth_year": 1828},
    score=0.95,
)
result = matcher.check_pair(pair)
# result.accepted = True, result.reason = "passed all gates"
```

Built-in gates: `exact_field`, `initial_match`, `no_conflict`, `gender_check`, `reference_ratio`. Custom gates are a single function returning `(accepted: bool, reason: str)`.

### Validation

Composable rules that check entities and produce errors or warnings:

```python
from limbic.hippocampus import Validator, required_field, valid_values, reference_exists, no_orphans, conditional_required

validator = Validator([
    required_field("work", "title"),
    valid_values("work", "category", {"teater", "opera", "konsert", "film"}),
    reference_exists("performance", "work_id", "work"),
    no_orphans("person", [("work", "playwrights"), ("performance", "credits", "person_id")]),
    conditional_required("work", lambda d: d.get("category") == "opera", "composers",
                         condition_label="category is opera"),
])

result = validator.validate(entities)
print(result.summary())  # "3 errors, 1 warnings"
```

### YAML store

File-locked, atomic YAML storage with typed entity access:

```python
from limbic.hippocampus import YAMLStore

store = YAMLStore("data/", schema={
    "person": "persons",
    "work": "plays",
    "performance": "performances",
})

data = store.load("person", "42")        # -> dict or None
store.save("person", "42", data)         # atomic write with advisory lock
store.delete("person", "99")             # -> True if existed
ids = store.all_ids("person")            # -> {"42", "43", ...}
for pid, pdata in store.iter_type("person"):
    pass  # iterate all persons
store.backup("person", "42")             # timestamped backup
```

---

### Wikidata entity resolution

`amygdala.wikidata` fetches a QID you already know. This decides *which* QID a
mention means — deterministically, with an audit record, before any LLM is
involved.

```python
from limbic.amygdala import WikidataClient
from limbic.hippocampus import WikidataResolver, validate_chosen_qid

resolver = WikidataResolver(
    WikidataClient(user_agent="myproject/1.0 (you@example.com)"),
    embedder=model,                       # optional: context similarity heuristic
    existing_kb_lookup=lookup_in_my_kb,   # optional: prefer entities you already have
)

res = resolver.resolve("Rollo", context_text="...Viking ruler of Normandy...",
                       type_hint="person", date_hint=parse_date("860-930"),
                       already_resolved={"William Longsword": "Q313659"})

res.status       # "resolved" | "ambiguous" | "not_found"
res.chosen_qid   # "Q57285" when resolved, None when ambiguous
res.confidence
res.candidates   # every ScoredCandidate with its per-heuristic breakdown
res.reasoning
```

Five heuristics are scored independently and combined by `DEFAULT_WEIGHTS`, each
contributing to `ScoredCandidate.scores` so a resolution is inspectable rather
than a bare QID:

| Heuristic | Weight | What it uses |
|---|---|---|
| `coherence` | 0.30 | Does the candidate's family/role claims (P22 father, P25 mother, P26 spouse, P40 child, P39 position, P108 employer) point at QIDs already resolved in this batch? The strongest signal, because it is the one an unrelated same-named entity cannot fake. |
| `type` | 0.25 | Does `P31 instance of` match the `type_hint`, per the `TYPE_HINT_P31` allowlist? |
| `description` | 0.20 | Cosine similarity between the candidate's description and `context_text`, via the `embedder` you passed. |
| `date` | 0.15 | `amygdala.temporal.plausibility_score` of the candidate's P569/P570 (or P571/P576) dates against your `date_hint`. |
| `rank` | 0.10 | The search API's position. A weak prior on purpose — Wikidata's rank is popularity-biased. |

Pass `already_resolved={mention: qid}` to feed the coherence heuristic what the
rest of the batch has already settled; resolving a cast list in one pass is
markedly more accurate than resolving each name cold.

Two design choices matter in practice:

- **Type mismatch is a soft penalty (~0.3), not a filter.** Wikidata's class
  hierarchy is deep and any hand-written P31 allowlist is shallow; hard-filtering
  discards correct answers whose `instance of` is three subclasses away from what
  you listed.
- **Ambiguity is a status, not a guess.** Below `absolute_threshold`, or when the
  runner-up is within `margin_ratio`, it returns `status="ambiguous"` with the
  candidates ranked — the point at which handing the shortlist to an LLM is cheap
  and safe. Guessing at that point is what produces confidently wrong links.

Three module constants make the heuristics inspectable and overridable:
`TYPE_HINT_P31` (the `instance of` allowlist per type hint), `COHERENCE_PROPERTIES`
(the properties compared for context coherence), and `DEFAULT_WEIGHTS` (how the
five scores combine). Pass `weights=` to reweight for a corpus where, say, dates
are reliable and context is thin.

`validate_chosen_qid(candidates, chosen_qid)` closes the loop after LLM
disambiguation: it verifies the model picked from the candidate set rather than
inventing a plausible-looking QID.

## limbic.cerebellum

**LLM-assisted batch verification with budget tracking, resumable state, and multi-tier orchestration.** For when you need an LLM to verify thousands of records but want to control costs and resume interrupted runs. See [limbic/cerebellum/README.md](limbic/cerebellum/README.md) for full documentation.

### Quick start: batch processing

```python
from limbic.cerebellum import BatchProcessor, StateStore, ItemResult
from pathlib import Path

# State persists across runs (SQLite with WAL mode)
state_store = StateStore(Path("audit_state.db"))

processor = BatchProcessor(
    state_store=state_store,
    max_cost=50.0,    # stop when $50 spent
    batch_size=20,
)

def verify_batch(items: list[dict]) -> list[ItemResult]:
    results = []
    for item in items:
        # ... call your LLM here ...
        results.append(ItemResult(
            id=item["id"],
            status="done",     # done | error | needs_review | skipped
            cost=0.003,
            metadata={"confidence": 0.95},
        ))
    return results

result = processor.process(
    items=all_items,
    process_fn=verify_batch,
    id_fn=lambda item: item["id"],
)
# result.processed, result.skipped, result.errors, result.total_cost
```

Features:
- **Resumable**: already-processed items are skipped on restart
- **Budget-tracked**: stops at `max_cost`, warns at 80%
- **Atomic state**: SQLite WAL mode for concurrent-safe persistence
- **ETA logging**: per-batch cost and time-remaining estimates

### Multi-tier orchestration

Run items through triage (cheap/fast) then deep verification (expensive/thorough), with automatic escalation:

```python
from limbic.cerebellum import TieredOrchestrator, VerificationTier, VerificationResult, StateStore
from pathlib import Path

def fast_triage(items):
    """Tier 1: Gemini Flash, ~$0.001/item."""
    results = []
    for item in items:
        results.append(VerificationResult(
            item_id=item["id"],
            status="verified",     # or "flagged" to escalate
            confidence=0.9,
            findings=["title matches external source"],
            cost=0.001,
        ))
    return results

def deep_verify(items):
    """Tier 2: Claude Sonnet, ~$0.05/item."""
    results = []
    for item in items:
        results.append(VerificationResult(
            item_id=item["id"],
            status="verified",
            confidence=0.98,
            findings=["cross-referenced with Wikidata", "dates confirmed"],
            cost=0.05,
        ))
    return results

orchestrator = TieredOrchestrator(
    tiers=[
        VerificationTier("triage", fast_triage, cost_estimate=0.001, description="Fast LLM check"),
        VerificationTier("deep", deep_verify, cost_estimate=0.05, description="Thorough verification"),
    ],
    state_store=StateStore(Path("audit_state.db")),
)

results = orchestrator.run(
    items=all_items,
    id_fn=lambda x: x["id"],
    max_cost=100.0,
    batch_size=20,
    escalate=True,
)

status = orchestrator.status(all_ids=["1", "2", "3"])
print(status.summary())
# OrchestratorStatus: .tier_counts, .total_cost, .remaining_items
```

### Audit logging

Append-only JSONL logs with daily rotation, extraction, and analysis:

```python
from limbic.cerebellum import AuditLogger, AuditEntry, read_logs, extract_operations, summarize_logs
from pathlib import Path

# Write audit entries
logger = AuditLogger(Path("audit_logs/"), prefix="verify")
logger.log_entry(AuditEntry(
    timestamp="2026-03-22T10:00:00",
    item_id="person/42",
    action="verified",
    details={"confidence": 0.95, "operations": [{"type": "fix_name", "old": "ibsen", "new": "Ibsen"}]},
    cost=0.003,
    tier="triage",
))

# Read and analyze
entries = list(read_logs(Path("audit_logs/"), prefix="verify", since="2026-03-01"))
summary = summarize_logs(entries)
# LogSummary: .total_cost, .items_processed, .error_count, .by_tier, .by_action

# Extract operations grouped by type (with dedup)
ops = extract_operations(entries, op_types=["fix_name", "merge"])
```

### Cost logging

Centralized LLM cost tracking across projects, models, and hosts. Uses litellm's pricing data (2,500+ models) for automatic cost computation:

```python
from limbic.cerebellum.cost_log import cost_log, compute_cost

# Standalone logging (any SDK)
cost_log.log(project="petrarca", model="gemini/gemini-2.5-flash",
             prompt_tokens=1200, completion_tokens=340)
# Each row is a CostRecord: project, host, model, api_key_hint, prompt/completion/
# cached tokens, cost_usd, script, purpose — enough to attribute spend to a
# specific script on a specific machine, not just to a project.

# litellm callback (auto-captures every litellm.completion call)
import litellm
litellm.callbacks = [cost_log.callback("alif")]

# Query costs
records = cost_log.query(project="petrarca", days=7)
total = sum(r.cost_usd for r in records)

# Built-in dashboard and CLI
# python -m limbic.cerebellum.cost_log report --days 7
# python -m limbic.cerebellum.cost_log sync --host alif
```

DB location: `COST_LOG_DB` env var or `~/.local/share/limbic/llm_costs.db`. Includes a web dashboard (`python -m limbic.cerebellum.cost_log dashboard`, port 8042) that splits API spend (billed) from Claude CLI usage (Max-plan subscription value), remote sync from servers, and CLI reporting.

### Context builder

Build structured prompts for LLM verification calls:

```python
from limbic.cerebellum import ContextBuilder, build_batch_context

ctx = ContextBuilder()
ctx.add_entity("work", "264", {"title": "Peer Gynt", "year": 1867})
ctx.add_related("performances", [{"id": 1, "venue": "DNS", "year": 1972}])
ctx.add_metadata("category", "teater")
prompt = ctx.build(format="markdown")

# Batch context for multiple items
combined = build_batch_context(items, context_fn=my_context_builder, format="markdown")
```

---

### CLI wrappers: Claude and Codex

Both wrap a locally installed coding CLI rather than an API key. That is often
the cheaper path — Codex runs under a ChatGPT subscription, Claude under a Max
plan — and it is the only way to get an *agentic* run (tools, web search, a
writable workspace) instead of a single completion.

```python
from limbic.cerebellum import claude_generate, ClaudeTask, claude_generate_parallel

result, meta = claude_generate(
    prompt="Classify this sentiment: I love it",
    project="myapp", purpose="sentiment", model="haiku",
    schema={"type": "object", "properties": {"label": {"type": "string"}}},
)

results = claude_generate_parallel(
    [ClaudeTask(prompt=p, schema=SCHEMA) for p in prompts],
    project="myapp", max_concurrent=4,
)
```

Every `claude -p` invocation writes a `cost_log` row with `script="claude-cli"`,
so subscription usage shows up in the same dashboard as API spend. The wrapper
always passes `--no-session-persistence` and strips `CLAUDECODE` plus
`ANTHROPIC_API_KEY` from the child environment — the key would silently switch a
Max-plan login to metered API billing *and* break cost attribution.

```python
from limbic.cerebellum import codex_json, codex_research

# Locked down: read-only sandbox, no network, no writes. Just classify/transform.
verdict = codex_json("Is this claim supported?", schema=SCHEMA, system=RUBRIC)

# Deliberately agentic: web search + a writable workspace with network egress.
dossier = codex_research(
    "Research X. Web-search anything ambiguous. Write findings to out.json.",
    schema=SCHEMA, scratch_dir="/tmp/run",
)
```

`codex_research` is the one that follows leads, and the two config flags that
unlock it (`tools.web_search`, `sandbox_workspace_write.network_access`) are on
by default — omit both and it quietly degrades to a shallow one-shot.

Both calls run with `--ephemeral --ignore-user-config`, so they leave no rollout
behind and read none of the host's `~/.codex` settings; pass `isolated=False` to
`codex_research` if a run genuinely needs the host profile (a locally configured
MCP server, say). Quota errors trip a process-local cooldown
(`mark_unavailable_from_error`) so a cron run stops hammering a depleted
allowance, and transient non-zero exits retry once while quota errors and
timeouts do not.

Both wrappers raise a typed error — `ClaudeCLIError` / `CodexCLIError` — for a
missing binary, a non-zero exit, a timeout, or unparseable output, so a batch can
distinguish "this item failed" from "the CLI is gone". `claude_is_available()`
and `codex_is_available()` check for the binary up front, which is what a nightly
job wants before it starts a thousand items.

`strict_response_schema()` (exported as `codex_strict_response_schema`) converts
a permissive JSON Schema into the shape
Codex's structured output requires: `additionalProperties: false`, every
property in `required`, formerly-optional fields made nullable.

### Agent isolation (`sandbox.py`)

`codex_research` exists to read material you do not control — scraped pages,
forwarded mail, uploaded images. Prompt injection is therefore a routine
operating condition, and these are the four guards worth having. They were
written for a nightly pipeline that ingests public event listings and email.

```python
from limbic.cerebellum import (
    call_slot, isolated_scratch, sanitized_environment, untrusted_payload,
    codex_research,
)

mission = "Extract every event announced below." + untrusted_payload(
    "scraped-page", page_html)

with call_slot(), isolated_scratch() as scratch, sanitized_environment(home=scratch):
    events = codex_research(mission, schema=SCHEMA, scratch_dir=str(scratch))
```

| Primitive | What it stops |
|---|---|
| `untrusted_payload(label, text)` | External text read as instructions. Delimits it with a content-derived nonce (so the payload cannot close its own block) and puts the refusal instruction *ahead* of the data, where later text cannot override it. |
| `isolated_scratch(files=...)` | The agent reading your repository. A private 0700 directory outside the project, with an allowlist of inputs, destroyed afterwards. Attachments belong here so a hostile image never becomes a durable file. |
| `sanitized_environment()` | An injection turning into a credential disclosure. The parent legitimately holds API keys and SMTP credentials; the child gets runtime plumbing only, and anything else needs an explicit opt-in. |
| `call_slot()` | A fan-out bursting the auth quota. A cross-process flock gate plus a persistent daily cap that survives restarts; raises `AgentBudgetExceeded` when the day is spent, `TimeoutError` when no slot frees up. |

**This is not an OS sandbox**, and is documented as such in the module: the child
keeps whatever process and network permissions the CLI grants it. These raise the
cost of a successful injection; they do not make one impossible. Constrain tool
and network policy separately — for Codex that is
`codex_research(web_search=..., network=...)`.

### Windowed extraction (`windowing.py`)

Asking a model to extract structured items from a long document in one call
loses most of them. Windowing recovers them; the merge is the hard part.

```python
from limbic.cerebellum import (
    Collection, MergeSchema, Reference, merge_windows, split_into_windows,
)

SCHEMA = MergeSchema([
    Collection("claims", prefix="C", dedup_field="text"),
    Collection("evidence", prefix="E", dedup_field="text",
               references=[Reference("supports_claim", target="claims")]),
    Collection("cases", prefix="CASE", dedup_field="name",
               references=[Reference("claims_supported", target="claims", many=True)]),
])

per_window = [extract(w.text) for w in split_into_windows(chapter_text)]
merged, report = merge_windows(per_window, SCHEMA, strict=False)

report.duplicates_removed    # collapsed across the seams
report.dangling              # references that still don't resolve
```

`merge_windows` is the whole pipeline, but each step is exported for the cases
that need to interleave something: `namespace_ids(result, i, schema)`,
`dedup_by_field(items, field)` (which returns the alias map), and
`check_references(merged, schema, strict=...)` to re-verify after your own edits.
`MergeReport` carries `items_before` / `items_after` per collection,
`duplicates_removed`, the full `id_map`, and any surviving `dangling` references.

Three things go wrong if you merge naively, and `merge_windows` fixes them in a
fixed order:

1. **Ids collide.** Every window numbers its own output from `C1`, `E1`, … so
   window 2's `C1` is a different item, and its `supports_claim: "C1"` means
   *its own*. Ids are namespaced `w{i}:` **before** concatenation; concatenating
   first and renumbering later silently rewires references between windows.
2. **The overlap duplicates items** — that is what it is for, but the copies
   arrive worded slightly differently and often truncated at a window edge. Dedup
   is word-overlap against `min(|A|, |B|)`, asymmetric so a truncated restatement
   still matches, and the **longer** text wins.
3. **Dropping a duplicate orphans references to it.** Dedup returns an alias map
   (every input id → the id that survived in its place), and renumbering resolves
   references through it. Anything still unresolvable is cleared and counted,
   never left dangling.

## Design decisions (with evidence)

Every significant design choice in limbic.amygdala was tested in controlled experiments. 23 experiments total, each with a specific hypothesis, dataset, and quantitative result:

| # | Question | Finding | Dataset |
|---|----------|---------|---------|
| 1 | Best embedding model? | Multilingual-MiniLM-L12 wins on all metrics | 150 calibration pairs |
| 2 | Does whitening help? | **Situational.** Helps domain-specific (+32%), hurts diverse (-3%) | STS-B, QQP, calibration |
| 3 | Optimal novelty K? | Adaptive: K=1 for <=50 items, K=10 for 1000+ | Calibration set |
| 4 | RRF vs convex fusion? | RRF 4x better under embedding degradation | 148 docs, 45 queries |
| 5 | Clustering method? | All methods ~0.55 V-measure. Greedy centroid simplest. | 20 Newsgroups |
| 6 | NLI for contradictions? | 94% accuracy on high-cosine contradictions | SICK (4,906 pairs) |
| 7 | Text genericization? | +14% on numbers/dates, 0% proper nouns, -6% URLs | 50 claim pairs |
| 8 | Are defaults optimal? | **Yes.** Rank 1/120 in grid search. | 120 configs, 3 datasets |
| 9 | Cross-encoder reranking? | +16% on NFCorpus, -5% on SciFact (dataset-dependent) | 5K + 3.6K docs |
| 10 | Temporal decay? | +9.3% Spearman at lambda=0.02 (half-life ~35 days) | Time-ordered calibration |
| 11 | Whitening on domain data? | +34.5% gap at 64d, +24% at 128d | 27K education claims |
| 12 | Soft-ZCA vs PCA? | Soft-ZCA strictly better (+32% vs +24%) | Domain calibration |
| 13 | Similarity graph layer? | Graph BFS surfaces 64% items vector misses | 27K claims |
| 14 | Task-specific LoRA? | Not worth it. Search-novelty correlation -0.953. | Multi-task eval |
| 15 | Novelty at 27K scale? | 1.1ms/call. Works fine. | 27K domain claims |
| 16 | Cross-lingual retrieval? | MRR=1.0 Norwegian-to-English. Translation unnecessary. | Bilingual claim set |
| 17 | PRF query expansion? | **Hurts** (-1.2% to -7.2%). Don't do it. | SciFact |
| 18 | Incremental clustering? | Identical to batch at >=0.85. 1.8x faster. | Synthetic + real |
| 19 | NFCorpus search? | Hybrid+rerank best (0.333 nDCG). | 3.6K medical docs |
| 20 | Persistent cache? | 83-452x speedup. Lossless. | 20K embeddings |
| 21 | All-but-the-top? | Matches Soft-ZCA (+27.4%), simpler math | Domain calibration |
| 22 | Document-level similarity? | Weighted 0.5×summary + 0.5×claims: **94% acc, rho=0.818**. Beats single-field (89%), concatenation (89%), LLM judge (78%), topic Jaccard (50%). AUROC=0.930 on 300 pairs. | 18 human + 300 LLM + 50 synthetic pairs |
| 23 | Knowledge map: best propagator × strategy? | **Bayesian + EIG** best overall (avg 7.2 Q→80%). Bayesian 42% faster than heuristic on chains. Post-hoc foil calibration doesn't help; Bayesian constraint propagation is the primary overclaiming defense. Batch probing maintains efficiency (5 Qs in 1 round = same as sequential). | 5 topologies × 50 trials |

Experiment code is in the `experiments/` directory if you want to reproduce or extend them.

### Findings from production corpora

The numbered experiments above are controlled and synthetic-adjacent. These came
out of evaluating limbic against real corpora, and two of them are **negative
results kept deliberately** — they are the expensive things not to build.

**Cross-encoder rerank is the cheap winner; LLM reranking is not worth its cost.**
A pooled-judgment eval (38 queries, 877 graded judgments, scaled 789 → 4,668
documents) compared three increasingly expensive LLM rerankers against the free
cross-encoder: snippet reranking, a wide hybrid+FTS union, and reading the *full
text* of the top 25. All three plateaued at ~0.54 nDCG@10 — tied with
`rerank()` — while `hybrid_rerank` reached the best cheap recall at 0.606.
Reading full documents instead of snippets bought nothing.

The reason generalises: **the bottleneck is first-stage recall, not ranking.** No
reranker can reorder a document that is not in its candidate list. Spend the
effort on retrieval, not on re-reading what retrieval already found.

**Index retrieval is scale-invariant; agentic file-reading is not.** In the same
eval, an agent with grep over the raw files led on quality (0.741 nDCG vs 0.521
for hybrid) but at roughly 1000× the cost and latency — and its advantage eroded
with corpus size (recall 0.77 → 0.66 at 6× the documents) while
`hybrid_rerank` stayed flat (0.532 → 0.534). Agentic retrieval is for deep,
small-set, high-value tasks; the index is for everything else.

**A better first-stage encoder is the lever that does work.** Swapping the
default multilingual MiniLM for `multilingual-e5-base` lifted the best cheap
method by +0.046 nDCG and +0.08 Recall@20 at zero marginal cost — the first cheap
method to close on the agentic result. But e5 *regressed* on Norwegian (bilingual
nDCG 0.442 vs MiniLM's 0.569), which is why MiniLM remains the default: it was
chosen for cross-lingual strength (experiment 16). Treat an encoder swap as a
per-corpus decision, and measure the bilingual case separately — an aggregate
win can hide a language-specific regression.

**Score serendipity on two axes, not one.** Judging candidate links on
*surprising* (0–3) and *useful* (0–3) separately yielded 28/40 links scoring ≥2
on both, with the judge correctly demoting the obvious pairs. A single blended
"interestingness" score cannot distinguish "obvious and useful" from "surprising
and pointless", which are the two failure modes that make a link feed unusable.

**Pool against a genuinely different method.** Adding the agentic method to the
judgment pool *lowered* every index method's score, as it should have: pooling
only similar methods grades them against their own shared blind spots. If every
method in your pool shares an architecture, the absolute numbers are optimistic.


## Common pitfalls

**"My clusters are huge (50+ members)"**
Your threshold is too low, or you're using raw embeddings on domain-focused text. Whiten first (`whiten_epsilon=0.1`), then cluster at 0.85.

**"Everything scores 0.7+ similarity"**
Domain-focused corpus without whitening. The narrow embedding cone compresses all scores into a band. Use `EmbeddingModel(whiten_epsilon=0.1)` and `fit_whitening(corpus)`.

**"Novelty scores are all 0.3–0.5 with no spread"**
Same cause as above — whitening spreads the distribution so novelty scores become meaningful. Also consider `use_centroid_specificity=True` for an additional +17% separation.

**"Cosine says two opposite claims are highly similar"**
Expected behavior — cosine measures *topical* similarity, not agreement. Two claims about the same topic that say opposite things will score high. Use `classify_pairs()` or `nli_classify()` to distinguish agree/disagree/extend.

**"NLI says 'contradiction' on paraphrases"**
Cross-encoder is noisy below 0.72 cosine. The default `classify_pairs()` cascade only runs NLI on high-cosine pairs to avoid this. Don't lower the threshold.

**"I'm getting different clusters on the same data"**
`IncrementalCentroidCluster` is order-sensitive at thresholds below 0.85 (Experiment 18). Use `greedy_centroid_cluster()` (batch mode) for reproducible results, or raise the threshold.

## Architecture

```
limbic/
  amygdala/                         hippocampus/                cerebellum/
  ───────────                       ──────────────              ────────────
  embed.py -> cache.py              proposals.py                batch.py
    |                                (Proposal, Change,          (BatchProcessor,
  search.py -> VectorIndex,          ProposalStore)               StateStore,
               FTS5Index,                                         ItemResult)
               HybridSearch,        cascade.py                      |
               rerank,               (ReferenceGraph,            orchestrator.py
               multi_list_rrf,        apply_merge,               (TieredOrchestrator,
               expand_query,          apply_delete)               VerificationTier)
               type_diversity_cap,                                  |
               expanded_hybrid_     dedup.py                    audit_log.py
                 search              (VetoMatcher,               (AuditLogger,
    |                                 VetoGate,                   read_logs,
  novelty.py -> VectorIndex           ExclusionList)              extract_operations)
    |                                                               |
  cluster.py (numpy only)           validate.py                 context.py
    |                                (Validator, Rule,           (ContextBuilder,
  document_similarity.py              composable checks)          build_batch_context)
    |                                                               |
  index.py -> search + connect()    store.py                    cost_log.py
    |                                (YAMLStore, file-locked)    (CostLog, cost_log,
  calibrate.py                          |                          compute_cost,
    |                               wikidata_resolve.py            dashboard, sync)
  knowledge_map.py (pure algo)       (WikidataResolver,              |
    |                                 five weighted             claude_cli.py
  knowledge_map_gen.py -> llm.py      heuristics, audited        (generate,
    |                                 Resolution)                 generate_parallel)
  llm.py (Gemini/Anthropic/OpenAI)                                  |
    |                                                           codex_cli.py
  temporal.py (DateRange,                                        (codex_json,
    parse_date, Allen relations)                                  codex_research)
    |                                                               |
  wikidata.py -> cache.py                                       sandbox.py
    (WikidataClient, TokenBucket)                                (untrusted_payload,
    |                                                             isolated_scratch,
  retrieval_eval.py                                               sanitized_environment,
    (pool -> judge -> nDCG)                                       call_slot)
    |                                                               |
  serendipity.py                                                windowing.py
    (inverted-U band,                                            (split_into_windows,
     Swanson ABC bridges)                                         merge_windows)

  drive/
  ───────
  policy.py -> calibration_cases.json
   (validate_plan, check_calibrations)
```

Design principles:
- **No external services.** Everything runs locally. SQLite for persistence, numpy for vectors, YAML for hippocampus entities.
- **Opt-in complexity.** Basic usage needs only numpy + sentence-transformers. YAML support, LLM features, and orchestration are all opt-in via extras.
- **Storage-agnostic.** Cascade merges, validation, and batch processing use callback functions — bring your own storage backend.
- **Numpy arrays everywhere.** All embedding operations return `np.ndarray` for interop.
- **Two-tier caching.** In-memory LRU (fast path) + optional SQLite persistent cache.

## How the packages compose

The three core packages are independent but designed to work together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Curation Pipeline                       │
│                                                                     │
│  1. FIND PATTERNS (amygdala)                                        │
│     embed entities → cluster → find duplicate candidates            │
│     score novelty → detect new items vs existing                    │
│     hybrid search → retrieve relevant context                      │
│                          │                                          │
│  2. MANAGE CHANGES (hippocampus)                                    │
│     veto-gate filter candidate pairs → create merge proposals       │
│     cascade merge accepted pairs → relink all references            │
│     validate dataset → catch broken refs, missing fields            │
│                          │                                          │
│  3. VERIFY CORRECTNESS (cerebellum)                                 │
│     batch-process entities through LLM → triage + deep verify       │
│     track budget → resume on restart → audit log everything         │
│     flagged items → create proposals for human review               │
└─────────────────────────────────────────────────────────────────────┘
```

`limbic.drive` sits *before* step 1 rather than inside it: given an open-ended
request, it validates that the plan starts with one small calibration pilot
instead of a fan-out. See [Drive](#drive-choose-the-first-move-before-the-swarm).

## Tests

606 tests:

```bash
pip install -e ".[dev]"      # pulls the llm/temporal/hippocampus extras too
python -m pytest tests/ -v
```

| Package | Tests |
|---------|-------|
| limbic.amygdala | 361 |
| limbic.hippocampus | 106 |
| limbic.cerebellum | 134 |
| limbic.drive | 5 |

Ten of those hit the live Wikidata API (`test_wikidata_live.py`,
`test_wikidata_resolve_live.py`); deselect them for an offline run. `[dev]`
deliberately installs every optional extra — when it did not, CI ran a quietly
smaller suite than a developer's venv and drifted red without anyone noticing.

CI runs on every PR via GitHub Actions.

## Used in production

Limbic powers search, data curation, and knowledge management in several systems:

- **otak / alif** — a **67K-node claims-first knowledge system** using embedding, novelty detection, hybrid search, clustering (canonical finding synthesis), and cosine+NLI cascade for deduplication. Podcast fact-checking showed that structured search changes 31% of verdicts vs. flat embedding search alone.
- **petrarca** — a **news curation pipeline** using document similarity to find related articles, calibrated thresholds for feed ranking vs near-duplicate detection, and hybrid search across multilingual content.
- **kulturperler** — a **Nordic performing arts archive** (10,000+ entities) using proposals for all data changes, cascade merges for deduplicating persons/works, tiered LLM verification of 2,400+ works across 30+ audit sessions, veto-gate dedup of fuzzy-matched person names. Total audit cost: ~$270. The DR-arkivet import scripts use `StateStore` and `AuditLogger` for resumable batch imports with JSONL audit trails, and `connect()` for all SQLite access.
- A **reading and annotation system** using novelty scoring and `classify_pairs` to detect when new annotations overlap with existing knowledge.
- **[claude-chat-search](https://github.com/houshuang/claude-chat-search)** — hybrid RRF search over Claude Code chat history with optional LLM query expansion via `expand_query` and `multi_list_rrf`.
- **hvaskjer** — an unsupervised nightly culture-listings pipeline that feeds scraped pages and forwarded email to `codex_research`. The isolation primitives in `cerebellum.sandbox` come from it, and it is the reason `codex_research` is `isolated` by default.
- **otak / hirsch-atlas** — book-length argument extraction (10 books, 101 chapters, ~10k claims) using `cerebellum.windowing` for the sliding-window extraction and cross-window merge.
- **a personal 20-year corpus** (blog, notes, talks, transcripts, tweets) — the pooled-judgment evaluation in [Design decisions](#design-decisions-with-evidence) ran here, using `retrieval_eval` and `serendipity`.

## License

MIT
