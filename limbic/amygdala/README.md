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
| **search** | Numpy vector search, SQLite FTS5, hybrid RRF fusion, cross-encoder reranking | +32.5% nDCG with reranking; RRF 4x more robust than convex fusion under embedding degradation |
| **novelty** | Multi-signal novelty scoring: global + topic-local + centroid specificity + temporal decay + NLI cascade | +17% novel/known separation with centroid specificity; NLI fixes 94% of high-cosine contradictions |
| **cluster** | Greedy centroid clustering (batch + incremental), complete linkage, pairwise cosine, confidence-calibrated pair classification | Incremental matches batch quality at threshold >= 0.85, 1.8x faster; order-sensitive at lower thresholds |
| **document_similarity** | Document-level thematic similarity using weighted multi-field embeddings | 94% accuracy on human-rated pairs; AUROC=0.930 on 300-pair dataset; Spearman rho=0.818 |
| **cache** | Persistent SQLite-backed embedding cache | 20K texts: 48s cold → 585ms warm |
| **index** | SQLite document/chunk storage with hybrid search, `connect()` helper | Single-file, zero-config, FTS5 built in |
| **calibrate** | Cohen's kappa, LLM judge validation (Bootstrap Validation Protocol), intra-rater reliability | Validates LLM judges against human gold labels |
| **knowledge_map** | Adaptive knowledge probing via entropy maximization, with heuristic or exact Bayesian propagation | Converges in 8–12 questions on 30-node graphs |
| **knowledge_map_gen** | LLM-powered knowledge graph generation from topic descriptions | Generates 15–50 node prerequisite DAGs |
| **llm** | Multi-provider LLM client (Gemini, Anthropic, OpenAI) with structured output and retry | Auto-fallback, cost tracking, async + sync |

---

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

**When to whiten:** Your corpus is domain-focused (all about education, all about medicine, etc.) and raw embeddings don't separate well. Confirmed across Experiments 2, 11, 12, and the Karpathy loop (Experiment 8: 120 configurations, current defaults rank 1/120).

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

### Other embedding features

```python
# Matryoshka truncation (reduce dimensions for speed/storage)
model = EmbeddingModel(truncate_dim=256)

# Text genericization (strip numbers, dates, URLs before embedding)
# Prevents "2024" and "$1.5M" from dominating similarity
model = EmbeddingModel(genericize=True)
# +14% accuracy on number/date-heavy text, no effect on proper nouns (Experiment 7)

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
```

### Design decisions

**Why RRF over convex fusion?** Experiment 4 tested on 148 documents with 45 queries. RRF is 4x more robust when embedding quality degrades (common in domain-specific corpora). It's also parameter-free — convex combination requires tuning α.

**Why brute-force over ANN?** At <100K vectors, numpy matrix multiply is faster than index-building overhead. No need for FAISS, Annoy, or HNSWlib until you're well past 100K.

**FTS5 query sanitization:** The first run on SciFact returned 299/300 empty results from FTS5 because special characters in scientific queries broke the parser. `FTS5Index` now auto-sanitizes queries. This fix alone moved FTS5 nDCG from 0.003 to 0.638.

### Benchmarks

| Dataset | Vector nDCG@10 | FTS5 nDCG@10 | Hybrid | Hybrid + rerank |
|---------|---------------|-------------|--------|-----------------|
| SciFact (5K docs, 300 queries) | 0.484 | 0.638 | 0.674 | **0.641** |
| NFCorpus (3.6K docs) | 0.235 | 0.126 | 0.286 | **0.333** |

FTS5 dominates on scientific text (exact terminology matters); vector dominates on medical queries (semantic matching matters). Reranking helps on NFCorpus (+16%) but slightly hurts on SciFact (-5%), likely because scientific terminology already gives exact matches high FTS5 scores.

---

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

Adaptive knowledge probing: efficiently map what someone knows about a topic using information theory. Entropy-maximizing probe selection with two propagation backends.

```python
from limbic.amygdala.knowledge_map import (
    KnowledgeGraph, init_beliefs, next_probe,
    update_beliefs, coverage_report, knowledge_fringes
)

# Define a knowledge graph
graph = KnowledgeGraph(nodes=[
    {"id": "crdt", "title": "CRDTs", "level": 1, "description": "..."},
    {"id": "lamport", "title": "Lamport clocks", "level": 2, "prerequisites": ["crdt"]},
    {"id": "mirror", "title": "Mirror protocol", "level": 3, "prerequisites": ["crdt", "lamport"]},
])

# Heuristic propagation (default, zero dependencies)
state = init_beliefs(graph)

# Exact belief propagation (+1-5% accuracy, zero dependencies)
state = init_beliefs(graph, propagator="bayesian")

# Get next question — maximizes expected information gain
probe = next_probe(graph, state)
# -> {"node_id": "crdt", "question_type": "recognition", "information_gain": 1.2}

# Update after user response — propagates through prerequisite DAG
update_beliefs(graph, state, "crdt", "solid")

# Check coverage and learning frontier
report = coverage_report(graph, state)  # known/unknown/uncertain lists
fringes = knowledge_fringes(graph, state)  # outer_fringe = ready to learn next
```

### Propagation backends

| Backend | Accuracy (K=5) | Latency | Dependencies |
|---------|---------------|---------|-------------|
| `"heuristic"` | 65-69% | 0.08ms | none |
| `"bayesian"` | 69-74% | 0.16ms | none |

Both backends are zero-dependency (numpy only). The heuristic uses bidirectional
rule-based propagation with global sweeps. The Bayesian backend implements
Pearl's forward-backward belief propagation with noisy-AND CPDs — exact on
trees/chains, approximate on dense DAGs with v-structures.

### Features

- **Expected Information Gain** probe selection (simulates all possible answers)
- **Bidirectional belief propagation** through prerequisite DAG (heuristic or exact Bayesian)
- **Overclaiming detection** via foil concepts (signal detection theory)
- **KST inner/outer fringe** computation for learning path recommendations
- **Convergence** in 8–12 questions on 30-node graphs (verified via Monte Carlo simulation)

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

---

## Persistent embedding cache (`cache.py`)

```python
from limbic.amygdala import PersistentEmbeddingCache

cache = PersistentEmbeddingCache("embeddings.db")
# EmbeddingModel uses this automatically when cache_path= is set
```

SQLite-backed, keyed by text hash. Stores pre-whitening vectors so the same cache works across whitening configurations. ~2.2 KB per 384-dim entry.

---

## What's NOT in amygdala

- **Billion-scale vector search.** Use FAISS, Milvus, or Qdrant for that. Amygdala's brute-force numpy works up to ~100K vectors.
- **Document chunking / RAG pipelines.** Use LlamaIndex or LangChain. Amygdala embeds individual texts, not multi-page documents.
- **Fine-tuning.** Experiment 14 showed that task-specific embeddings aren't worth it for this use case (search and novelty are anti-correlated at -0.953 — optimizing one hurts the other).
- **Query expansion.** Experiment 17 showed PRF query expansion hurts search quality (-1.2% to -7.2% across metrics). Rejected.

## Full API reference

See the module docstrings in each `.py` file and the [main limbic README](../README.md) for usage examples.
