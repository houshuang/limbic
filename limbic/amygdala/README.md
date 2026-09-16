# limbic.amygdala

**Embedding, search, novelty detection, clustering, and knowledge mapping for knowledge-dense text corpora.**

Amygdala is the pattern-finding layer of limbic. It grew out of recurring needs across several projects — a claims-first knowledge system (otak/alif), a news curation pipeline (petrarca), a Nordic performing arts archive (kulturperler), and others. The same problems kept appearing: embedding domain-specific text where off-the-shelf models couldn't separate similar from identical, searching across languages, detecting novelty, deduplicating entities. Through 22 controlled experiments, these solutions were generalized into a single toolkit that now powers systems processing 1K–67K text items.

It's optimized for **short knowledge-dense texts** — claims, research findings, annotations, entity descriptions — not generic documents. If your corpus is "things people wrote about a specific domain," amygdala will serve you well. If you need document-level RAG with chunking, look at LlamaIndex or LangChain.

## Install

```bash
# Core (embedding, search, novelty, clustering)
pip install limbic

# With LLM support (knowledge map generation, multi-provider LLM client)
pip install "limbic[llm]"
```

**Requirements:** Python >= 3.11, numpy, sentence-transformers. No vector database needed.

---

## Modules

| Module | What it does | Key numbers |
|--------|-------------|-------------|
| **embed** | Sentence embedding with 3 whitening modes, Matryoshka truncation, genericization, persistent cache | 83–452x speedup with SQLite cache; +32% nearest-neighbor separation with Soft-ZCA whitening |
| **search** | Numpy vector search, SQLite FTS5, hybrid RRF fusion, cross-encoder reranking, multi-list RRF with contribution tracing, LLM query expansion, type-diversity capping | +32.5% nDCG with reranking; RRF 4x more robust than convex fusion under embedding degradation |
| **novelty** | Multi-signal novelty scoring: global + topic-local + centroid specificity + temporal decay + NLI cascade | +17% novel/known separation with centroid specificity; NLI fixes 94% of high-cosine contradictions |
| **cluster** | Greedy centroid clustering (batch + incremental), complete linkage, pairwise cosine, confidence-calibrated pair classification | Incremental matches batch quality at threshold >= 0.85, 1.8x faster; order-sensitive at lower thresholds |
| **document_similarity** | Document-level thematic similarity using weighted multi-field embeddings | 94% accuracy on human-rated pairs; AUROC=0.930 on 300-pair dataset; Spearman rho=0.818 |
| **cache** | Persistent SQLite-backed embedding cache | 20K texts: 48s cold → 585ms warm |
| **index** | SQLite document/chunk storage with hybrid search, grep, `connect()` helper | Single-file, zero-config, FTS5 auto-synced via triggers |
| **calibrate** | Cohen's kappa, LLM judge validation (Bootstrap Validation Protocol), intra-rater reliability | Validates LLM judges against human gold labels |
| **knowledge_map** | Adaptive knowledge probing via EIG selection with Bayesian belief propagation, batch probing, KST fringes | Converges in 5–8 questions on 20-node graphs; Bayesian propagator 42% faster than heuristic on chains |
| **knowledge_map_gen** | LLM-powered knowledge graph generation from topic descriptions | Generates 15–50 node prerequisite DAGs |
| **llm** | Multi-provider LLM client (Gemini, Anthropic, OpenAI) with structured output and retry | Auto-fallback, cost tracking, async + sync |
| **temporal** | Uncertain-date parsing ("940s", "circa 942", "4th century BC", EDTF) into integer year ranges, Allen interval relations, soft plausibility scoring | Indexes as two ints; `edtf` extra optional |
| **wikidata** | Cache-backed, rate-limited Wikidata client: search, get, batched get_many, SPARQL | 30-day payload cache, 5 req/s token bucket, maxlag-aware |
| **retrieval_eval** | Pooled-judgment IR evaluation: pool -> LLM-judge -> nDCG / Recall / MRR / MAP, with strata | Answers "which retrieval knob actually wins?" |
| **serendipity** | Non-obvious link finding: inverted-U similarity band, cross-facet bonus, Swanson ABC bridging | 70% of surfaced links rated surprising *and* useful |

---

## Recipe: analyze a corpus of responses

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

See also the [entity dedup recipe](../hippocampus/README.md#deduplication-deduppy) in limbic.hippocampus and the [batch verification recipe](../cerebellum/README.md#batch-processing-batchpy) in limbic.cerebellum.

## Embedding (`embed.py`)

The `EmbeddingModel` class wraps sentence-transformers with features designed for domain-specific corpora.

### Default model

`paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, 118M parameters). Chosen over `all-MiniLM-L6-v2` through Experiment 1, which tested four models on discrimination gap, accuracy, Norwegian quality, and latency:

| Metric | Multilingual-MiniLM-L12 | MiniLM-L6 | MPNet-base | Nomic v1.5 |
|--------|------------------------|-----------|------------|------------|
| Classification accuracy | **80%** | 75% | 75% | 73% |
| Norwegian cross-lingual quality | **0.84** | 0.16 | 0.19 | 0.49 |
| Discrimination gap | **0.728** | 0.730 | 0.714 | 0.285 |
| Speed (150 texts) | **0.30s** | 0.40s | 0.70s | 1.21s |
| STS-B Spearman | **0.844** | 0.820 | 0.834 | — |

### Basic usage

```python
from limbic.amygdala import EmbeddingModel

model = EmbeddingModel()
vec = model.embed("Education improves democratic participation")  # -> np.ndarray (384,)
vecs = model.embed_batch(["claim 1", "claim 2", "claim 3"])       # -> np.ndarray (3, 384)
```

### Whitening for domain-specific corpora

Off-the-shelf embeddings put everything in a narrow cone — unrelated texts in the same domain score 0.7+ cosine similarity, making it hard to distinguish "similar" from "identical." Whitening spreads the distribution.

**When to whiten:** Your corpus is domain-focused (all about education, all about medicine, etc.) and raw embeddings don't separate well. **In practice, this means almost always for single-domain corpora** — without whitening, clustering thresholds below ~0.90 produce excessive false positives, and novelty scores compress into a meaningless 0.3–0.5 band. Confirmed across Experiments 2, 11, 12, and the Karpathy loop (Experiment 8: 120 configurations, current defaults rank 1/120). Validated on 27K education claims and 6.5K political proposals.

**When NOT to whiten:** Your corpus is diverse (mixed domains). Whitening raises the unrelated floor without proportionally raising related — net negative. Experiment 2 confirmed this: raw embeddings have the best discrimination gap on diverse data.

Three whitening modes, all opt-in:

| Mode | Code | Effect | When to use |
|------|------|--------|-------------|
| **Soft-ZCA** | `EmbeddingModel(whiten_epsilon=0.1)` | +32% NN-gap, preserves all dims | Domain-focused corpora (recommended) |
| **All-but-the-top** | `EmbeddingModel(whiten_abt=1)` | +27% NN-gap, simpler math | When you don't want to tune epsilon |
| **PCA** | `EmbeddingModel(whiten_dims=128)` | +24% NN-gap, reduces dims | When you need dimensionality reduction |

```python
# Soft-ZCA whitening (recommended for domain-focused corpora)
model = EmbeddingModel(whiten_epsilon=0.1)
model.fit_whitening(corpus_texts)  # compute whitening transform from your corpus
vec = model.embed("now whitened")  # still 384-dim, much better separation

# Before whitening: mean pairwise cosine ~0.80
# After whitening:  mean pairwise cosine ~0.24
```

### Genericization (pre-embedding text normalization)

Strips numbers, dates, currencies, and URLs before embedding. Use when texts contain variable specifics around the same argument (e.g., "allocate 50M" vs "allocate 200M", "Oslo kommune" vs "Bergen kommune", "§ 2-3" vs "§ 1-1"). +14% accuracy on number/date-heavy text, no effect on proper nouns (Experiment 7).

```python
model = EmbeddingModel(genericize=True)

# Combine with whitening for domain-focused corpora (recommended):
model = EmbeddingModel(genericize=True, whiten_epsilon=0.1)
```

Skip when proper nouns carry real semantic signal (e.g., comparing claims *about* different people).

### Other embedding features

```python
# Matryoshka truncation (reduce dimensions for speed/storage)
model = EmbeddingModel(truncate_dim=256)

# Persistent embedding cache (survives restarts)
model = EmbeddingModel(cache_path="embeddings.db")
# 20K texts: 48s cold -> 585ms warm (83x speedup)
# ~2.2 KB per 384-dim entry, stores pre-whitening vectors
```

### Cross-lingual support

The multilingual model achieves **MRR=1.0** on Norwegian-to-English retrieval out of the box (Experiment 16). No translation step needed:

```python
model = EmbeddingModel()
v_no = model.embed("Utdanning er viktig for demokratiet")
v_en = model.embed("Education is important for democracy")
similarity = float(v_no @ v_en)  # -> 0.86
```

---

## Search (`search.py`)

Three search modes that compose together. All return `Result(id, score)` namedtuples.

```python
from limbic.amygdala import VectorIndex, FTS5Index, HybridSearch, rerank

# Pure vector search — brute-force cosine, faster than ANN at <100K vectors
vi = VectorIndex()
vi.add(ids, embeddings)
results = vi.search(query_vec, limit=10)
results = vi.search(query_vec, limit=10, filter_ids={"id1", "id2"})  # filtered

# Pure full-text search — SQLite FTS5 with porter stemming
fts = FTS5Index("index.db")  # or ":memory:"
fts.add("doc1", "some text content", metadata={"source": "arxiv"})
results = fts.search("text content", limit=10)

# Hybrid search — Reciprocal Rank Fusion combines both
hybrid = HybridSearch(vector_index=vi, fts_index=fts)
results = hybrid.search(query_vec, "query text", limit=10)

# Cross-encoder reranking — +32.5% nDCG on top of any search
reranked = rerank("query text", results)  # uses ms-marco-MiniLM-L-6-v2

# Group-by deduplication — keep best result per group
from limbic.amygdala import dedup_by
deduped = dedup_by(results, key_fn=lambda r: r.metadata["session_id"])
```

### Design decisions

**Why RRF over convex fusion?** Experiment 4 tested on 148 documents with 45 queries. RRF is 4x more robust when embedding quality degrades (common in domain-specific corpora). It's also parameter-free — convex combination requires tuning α.

**Why brute-force over ANN?** At <100K vectors, numpy matrix multiply is faster than index-building overhead. No need for FAISS, Annoy, or HNSWlib until you're well past 100K.

**FTS5 query sanitization:** The first run on SciFact returned 299/300 empty results from FTS5 because special characters in scientific queries broke the parser. `FTS5Index` now auto-sanitizes queries (extracting unicode word tokens and quoting each to prevent reserved words like AND/OR/NOT/NEAR from being interpreted as operators). This fix alone moved FTS5 nDCG from 0.003 to 0.638.

### Benchmarks

| Dataset | Vector nDCG@10 | FTS5 nDCG@10 | Hybrid | Hybrid + rerank |
|---------|---------------|-------------|--------|-----------------|
| SciFact (5K docs, 300 queries) | 0.484 | 0.638 | 0.674 | **0.641** |
| NFCorpus (3.6K docs) | 0.235 | 0.126 | 0.286 | **0.333** |

FTS5 dominates on scientific text (exact terminology matters); vector dominates on medical queries (semantic matching matters). Reranking helps on NFCorpus (+16%) but slightly hurts on SciFact (-5%), likely because scientific terminology already gives exact matches high FTS5 scores.

---

### Search benchmarks

| Dataset | Vector nDCG@10 | FTS5 nDCG@10 | Hybrid | Hybrid + rerank |
|---------|---------------|-------------|--------|-----------------|
| SciFact (5K docs, 300 queries) | 0.484 | 0.638 | 0.674 | **0.641** |
| NFCorpus (3.6K docs) | 0.235 | 0.126 | 0.286 | **0.333** |

FTS5 dominates on scientific text (exact terminology matters); vector dominates on medical queries (semantic matching matters). Reranking helps on NFCorpus (+16%) but slightly hurts on SciFact (-5%), likely because scientific terminology already gives exact matches high FTS5 scores.

## Multi-list RRF and query expansion (`search.py`)

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

## Result diversity (`search.py`)

```python
from limbic.amygdala import type_diversity_cap, dedup_by

# No single facet may take more than 60% of the top-k
balanced = type_diversity_cap(
    results, key_fn=lambda r: r.metadata["source_type"], max_fraction=0.6, k=10,
)

# Collapse near-identical hits (e.g. several chunks of the same document)
unique = dedup_by(results, key_fn=lambda r: r.metadata["doc_id"])
```

Relevance ranking has no opinion about balance, so one prolix source can take
every slot in an answer that should span several. `type_diversity_cap` caps each
group at `ceil(max_fraction * k)`, defers the over-cap items, and backfills from
them only if the result would otherwise come up short — so it costs nothing when
the ranking is already balanced. Relative order within a group is preserved, and
it assumes the input is already sorted by score descending.


## Novelty detection (`novelty.py`)

Answers: **"Is this text saying something new relative to what I already have?"** Returns a float from 0.0 (exact duplicate) to 1.0 (completely novel).

```python
from limbic.amygdala import VectorIndex, novelty_score, batch_novelty, nli_classify

# Basic novelty
score = novelty_score(query_vec, index)

# With topic-local context (higher weight to same-category neighbors)
score = novelty_score(query_vec, index, category_ids={"id1", "id2", "id3"})

# With centroid specificity (generic claims near corpus center get dampened)
# +17% separation on diverse data (Experiment 3b)
score = novelty_score(query_vec, index, use_centroid_specificity=True)

# With temporal decay (older items contribute less to "already known")
# Half-life ~35 days at lambda=0.02 (Experiment 10)
ages = {"id1": 0.0, "id2": 30.0, "id3": 90.0}  # age in days
score = novelty_score(query_vec, index, timestamps=ages, decay_lambda=0.02)

# Batch scoring (3.6ms per claim at 27K scale)
scores = batch_novelty(query_vecs, index)
```

### The cosine similarity problem and NLI cascade

Cosine similarity **cannot distinguish agreement from disagreement**. Two claims that say opposite things about the same topic often have *higher* cosine similarity than two unrelated claims. Experiment 6 (on SICK, 4,906 pairs) confirmed this is a real problem.

The `classify_pairs()` function implements a cosine + NLI cascade:
- **Above threshold** (e.g., 0.88): cosine-confident KNOWN
- **Below threshold** (e.g., 0.72): cosine-confident NEW
- **In between**: NLI cross-encoder decides (entailment/contradiction/neutral)

```python
from limbic.amygdala import nli_classify, classify_pairs

# Single pair — ~13ms
result = nli_classify("Education improves outcomes", "Education has no effect")
# -> {"label": "contradiction", "contradiction": 0.92, ...}

# Batch classification with cosine + NLI cascade
# texts: list of (text_a, text_b) pairs; scores: cosine similarities
pairs_result = classify_pairs(texts, scores, known_threshold=0.88, extends_threshold=0.72)
```

94% accuracy on high-cosine contradictions. Fixes cases where cosine alone would classify contradictions as duplicates.

### Adaptive top-K

The number of neighbors considered scales with index size (Experiment 3):
- K=1 for ≤50 items (small corpus, single nearest neighbor is most informative)
- K=10 for 1000+ items (smooths over local density variation)
- Formula: `K = max(1, min(10, len(index) // 100))`

### Performance at scale

Tested on a 27K-claim knowledge base (Experiment 15):
- `novelty_score()`: **1.1ms per call**
- `batch_novelty()`: **3.6ms per claim**

---


`corpus_centroid(vectors)` returns the mean direction that the
centroid-specificity signal dampens against — compute it once and reuse it
across a batch rather than letting each call re-derive it.

## Clustering (`cluster.py`)

Two strategies optimized for **deduplication**, not topic discovery.

```python
from limbic.amygdala import greedy_centroid_cluster, IncrementalCentroidCluster, pairwise_cosine, extract_pairs

# Batch clustering — good when you have all vectors upfront
clusters = greedy_centroid_cluster(embeddings, threshold=0.85)
# Returns list of clusters (each a list of indices). Singletons excluded.

# Incremental clustering — for streaming/continuous ingestion
# Identical quality to batch at threshold >= 0.85, 1.8x faster (Experiment 18)
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

### Confidence-calibrated pair classification

```python
from limbic.amygdala import classify_pairs_with_confidence, format_for_eval_harness

# Classify pairs with confidence-based labels and per-label metrics
# pairs: list of (idx_a, idx_b, cosine_score) from extract_pairs()
result = classify_pairs_with_confidence(pairs, texts,
                                         confident_threshold=0.75, reject_threshold=0.30)

# Format for evaluation harness
eval_data = format_for_eval_harness(result)
```

### Choosing a threshold

The right threshold depends on whether you've whitened your embeddings:

| Corpus state | Threshold | Rationale |
|---|---|---|
| Raw embeddings, diverse corpus | 0.70–0.75 | Embeddings already spread well across domains |
| Raw embeddings, domain-focused | 0.90+ | Narrow similarity cone — lower thresholds group everything |
| **Whitened embeddings** (recommended) | **0.85** | Whitening restores meaningful spread |

If your largest cluster has 50+ members, your threshold is too low or you need whitening. Always LLM-validate a sample of cluster pairs before using results downstream.

### Design decisions

**Why greedy centroid over union-find?** Union-find causes transitive chaining — at threshold 0.85, it produces clusters of 1,500+ items through "friend of a friend" effects. Greedy centroid caps naturally at ~50. Discovered empirically when clustering 27K claims.

**Why not HDBSCAN?** Experiment 5 tested both on 20 Newsgroups with human topic labels. Similar V-measure (~0.55). Greedy centroid is simpler, needs no hyperparameter tuning, and works incrementally.

**Incremental clustering properties** (Experiment 18):
- Order-sensitive: insertion order can change cluster assignments, especially at lower thresholds. Use batch `greedy_centroid_cluster` when determinism matters.
- Matches batch quality closely at threshold ≥ 0.85 (similar ARI, similar cluster count)
- 1.8x faster (single-pass vs. pairwise comparison)

---

## Document similarity (`document_similarity.py`)

Find thematically similar documents using weighted multi-field embeddings.

```python
from limbic.amygdala import Document, find_similar_documents, document_similarity_matrix

docs = [
    Document(id="art1", texts={"summary": "Sicily's history...", "claims": "Greeks founded Syracuse..."}),
    Document(id="art2", texts={"summary": "Sicilian Baroque...", "claims": "Sicilian Baroque is UNESCO..."}),
]

# Weighted multi-field (best strategy: 94% accuracy, rho=0.818)
pairs = find_similar_documents(
    docs,
    text_fields={"summary": 0.5, "claims": 0.5},
    threshold=0.52,  # calibrated for 80% precision, 78% recall
)
# -> [SimilarityPair(id_a="art1", id_b="art2", score=0.74, field_scores={...})]

# Full similarity matrix
ids, matrix = document_similarity_matrix(docs, text_fields={"summary": 0.5, "claims": 0.5})
```

### Calibrated thresholds

From 300 LLM-rated + 18 human-rated article pairs:

| Use case | Threshold | Precision | Recall | F1 |
|----------|-----------|-----------|--------|-----|
| Feed ranking (recall-focused) | 0.49 | 71% | 82% | 76% |
| Balanced | 0.52 | 80% | 78% | 79% |
| High confidence | 0.55 | 91% | 75% | 82% |
| Near-duplicate detection | 0.64 | 96% | 73% | 83% |

### What didn't work

Tested and rejected approaches:
- **Topic tag Jaccard:** 50% accuracy — useless
- **LLM-as-judge:** 78% accuracy, systematically over-rates within-domain similarity
- **Two-stage embed-then-LLM pipeline:** doesn't beat embedding alone
- **Max-sim claim matching:** 72% — individual claims too narrow for document-level overlap

---

## Knowledge mapping (`knowledge_map.py`)

Adaptive knowledge probing: efficiently map what someone knows about a topic using information theory. Expected Information Gain probe selection with Bayesian belief propagation through prerequisite DAGs.

```python
from limbic.amygdala.knowledge_map import (
    KnowledgeGraph, init_beliefs, next_probe, next_probe_batch,
    update_beliefs, coverage_report, knowledge_fringes
)

# Define a knowledge graph
graph = KnowledgeGraph(nodes=[
    {"id": "crdt", "title": "CRDTs", "level": 1, "description": "..."},
    {"id": "lamport", "title": "Lamport clocks", "level": 2, "prerequisites": ["crdt"]},
    {"id": "mirror", "title": "Mirror protocol", "level": 3, "prerequisites": ["crdt", "lamport"]},
])

# Initialize — Bayesian propagation by default (best accuracy)
state = init_beliefs(graph)

# Get next question — maximizes expected information gain
probe = next_probe(graph, state)
# -> {"node_id": "crdt", "question_type": "recognition", "information_gain": 1.2}

# Or get a batch of diverse, high-value questions at once
probes = next_probe_batch(graph, state, n=3)
# Avoids redundant probes (e.g., won't pick 3 siblings of the same parent)

# Update after user response — propagates through prerequisite DAG
update_beliefs(graph, state, "crdt", "solid")

# Check coverage and learning frontier
report = coverage_report(graph, state)  # known/unknown/uncertain lists
fringes = knowledge_fringes(graph, state)  # outer_fringe = ready to learn next
```

### Propagation backends

| Backend | Accuracy (K=5) | Q→80% on chains | Latency | Dependencies |
|---------|---------------|-----------------|---------|-------------|
| `"bayesian"` (default) | 69-74% | 7 questions | 0.16ms | none |
| `"heuristic"` | 65-69% | 12 questions | 0.08ms | none |

Both backends are zero-dependency. The Bayesian backend implements Pearl's
forward-backward belief propagation with noisy-AND CPDs — exact on trees/chains,
approximate on dense DAGs. Validated across 5 graph topologies × 50 trials
(see `experiments/exp_knowledge_map_matrix.py`).

The Bayesian propagator also provides implicit overclaiming defense through
constraint propagation: if someone claims to know a node but its children are
unknown, the backward pass adjusts the belief downward.

### Features

- **Expected Information Gain** probe selection (simulates all possible answers)
- **Batch probe selection** via `next_probe_batch(n)` — diversity-aware sequential greedy
- **Bayesian belief propagation** through prerequisite DAG (Pearl's forward-backward)
- **Overclaiming detection** via foil concepts (signal detection theory)
- **KST inner/outer fringe** computation for learning path recommendations
- **Noisy observation mode** for BKT-style updates (treats self-report as noisy signal)
- **Convergence** in 5–8 questions on 20-node graphs (verified via Monte Carlo simulation)

### LLM-powered graph generation

```python
from limbic.amygdala.knowledge_map_gen import graph_from_description
graph = await graph_from_description("Conflict-free replicated data types")
# -> 15-50 nodes with prerequisites, obscurity levels, descriptions
```

---

## Calibration (`calibrate.py`)

Utilities for measuring agreement between raters (human vs. LLM, or LLM vs. LLM).

```python
from limbic.amygdala import cohens_kappa, validate_llm_judge, intra_rater_reliability

# Cohen's kappa — inter-rater agreement
kappa = cohens_kappa(["A", "B", "A"], ["A", "B", "B"])

# Bootstrap Validation Protocol — validate an LLM judge against gold labels
result = validate_llm_judge(gold_labels, llm_labels)
# result["kappa"], result["recommendation"], result["per_label"] (precision/recall/F1)

# Intra-rater reliability — is the LLM consistent with itself?
consistency = intra_rater_reliability(pass1_labels, pass2_labels)
# consistency["kappa"], consistency["quality"]
```

---

## LLM client (`llm.py`)

Multi-provider async LLM client with structured output, retry, auto-fallback, and cost tracking.

Supported providers and models:
- **Gemini**: gemini3-flash, gemini25-flash, gemini25-pro
- **Anthropic**: sonnet (Claude Sonnet 4), haiku (Claude Haiku 4.5)
- **OpenAI**: gpt41-mini, gpt41-nano

```python
from limbic.amygdala.llm import generate, generate_structured

# Simple generation
text = await generate("What is the capital of France?")

# Structured output with JSON schema
result, meta = await generate_structured(
    prompt="Classify this text",
    schema={"type": "object", "properties": {"label": {"type": "string"}}},
)
# meta includes cost, tokens, latency
```

Features: automatic retry with exponential backoff on 429/500/503, fallback chains (gemini3-flash → gemini25-flash), cost calculation per call.

---

## SQLite connection helper (`index.py`)

```python
from limbic.amygdala import connect

conn = connect("my.db")  # or connect("my.db", readonly=True)
```

Applies all best practices automatically: WAL journal mode, 30s busy timeout, NORMAL synchronous, 64MB page cache, foreign key enforcement. Use this for any project that touches SQLite.

### Index: document/chunk storage

```python
from limbic.amygdala.index import Index

idx = Index("my_index.db")
idx.add_document("file.md", chunks=[{"content": "text"}])
results = idx.search("query", embedding_model=model)  # hybrid search
results = idx.grep("/path/to/file")  # exact substring search
idx.rebuild_fts()  # one-time FTS rebuild for pre-trigger databases
```

The `Index` class uses SQLite triggers to keep FTS5 in sync automatically — no manual sync needed.

---

## Persistent embedding cache (`cache.py`)

```python
from limbic.amygdala import PersistentEmbeddingCache

cache = PersistentEmbeddingCache("embeddings.db")
# EmbeddingModel uses this automatically when cache_path= is set
```

SQLite-backed, keyed by text hash. Stores pre-whitening vectors so the same cache works across whitening configurations. ~2.2 KB per 384-dim entry.

---

## Temporal reasoning (`temporal.py`)

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

## Wikidata client (`wikidata.py`)

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
`TokenBucket` (thread-safe) at Wikidata's published 5 req/s, and `maxlag` is honoured — a
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
[`hippocampus.wikidata_resolve`](../hippocampus/README.md#wikidata-entity-resolution-wikidata_resolvepy).

## Retrieval evaluation (`retrieval_eval.py`)

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
[Design decisions](../../docs/EVIDENCE.md) for what happened when a
genuinely different method was added late.

## Serendipity (`serendipity.py`)

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
[Design decisions](../../docs/EVIDENCE.md).

## What's NOT in amygdala

- **Billion-scale vector search.** Use FAISS, Milvus, or Qdrant for that. Amygdala's brute-force numpy works up to ~100K vectors.
- **Document chunking / RAG pipelines.** Use LlamaIndex or LangChain. Amygdala embeds individual texts, not multi-page documents.
- **Fine-tuning.** Experiment 14 showed that task-specific embeddings aren't worth it for this use case (search and novelty are anti-correlated at -0.953 — optimizing one hurts the other).
- **Query expansion.** Experiment 17 showed PRF query expansion hurts search quality (-1.2% to -7.2% across metrics). Rejected.

## Full API reference

See the module docstrings in each `.py` file and the [main limbic README](../../README.md) for usage examples.

---

Part of [limbic](../../README.md).
