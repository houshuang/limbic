# Amygdala

**Embedding, search, novelty detection, and clustering for knowledge-dense text corpora.** Optimized for collections of claims, research findings, notes, and annotations — not generic documents.

If you're building a personal knowledge system, research tool, claim extractor, or any application that works with **thousands to tens of thousands of short, semantically similar texts**, amygdala provides the primitives you need without pulling in a vector database or a framework.

## Is this for you?

**Good fit:**
- You have 1K–100K short texts (claims, findings, notes, annotations) and need search, deduplication, or novelty detection
- Your corpus is **domain-focused** (e.g., all about education, all about biology) where off-the-shelf embeddings struggle to differentiate similar items
- You need **multilingual** support (especially English + Norwegian, but any language pair that `sentence-transformers` supports)
- You want numpy-based search without the operational overhead of a vector database
- You want hybrid search (vector + full-text) with a single `pip install`

**Not a good fit:**
- You need billion-scale vector search (use FAISS, Milvus, or Qdrant)
- You need document-level RAG with chunking strategies (use LlamaIndex or LangChain)
- You only need basic `sentence-transformers` — amygdala adds value through whitening, novelty, clustering, and hybrid search on top of it

## What's inside

| Module | What it does | Key numbers |
|--------|-------------|-------------|
| **embed** | Sentence embedding with 3 whitening modes, Matryoshka truncation, genericization, persistent cache | 83–452x speedup with SQLite cache; +32% nearest-neighbor separation with Soft-ZCA whitening |
| **search** | Numpy vector search, SQLite FTS5, hybrid RRF fusion, cross-encoder reranking | +32.5% nDCG with reranking; RRF 4x more robust than convex fusion under embedding degradation |
| **novelty** | Multi-signal novelty scoring: global + topic-local + centroid specificity + temporal decay + NLI cascade | +17% novel/known separation with centroid specificity; NLI fixes 94% of high-cosine contradictions |
| **cluster** | Greedy centroid clustering (batch + incremental), complete linkage, pairwise cosine | Incremental matches batch quality at threshold ≥ 0.85, 1.8x faster, zero order sensitivity |
| **cache** | Persistent SQLite-backed embedding cache | 20K texts: 48s cold → 585ms warm |
| **index** | SQLite document/chunk storage with hybrid search | Single-file, zero-config, FTS5 built in |
| **knowledge_map** | Adaptive knowledge probing via Shannon entropy maximization and Bayesian belief propagation | Converges in 8–12 questions on 30-node graphs |
| **llm** | Multi-provider LLM client (Gemini, Anthropic, OpenAI) with structured output and retry | Auto-fallback, cost tracking, async + sync |

## Install

```bash
pip install -e .

# With LLM support (for knowledge_map generation, novelty NLI cascade):
pip install -e ".[llm]"
```

**Requirements:** Python ≥ 3.11, numpy, sentence-transformers. No vector database needed.

## Quick start

```python
from amygdala import EmbeddingModel, VectorIndex, HybridSearch, FTS5Index, novelty_score

# Embed text (multilingual model handles English, Norwegian, and 50+ languages)
model = EmbeddingModel()
vecs = model.embed_batch(["Education improves democratic participation",
                          "Schools need more funding for special education",
                          "Quantum entanglement in superconductors"])

# Vector search
index = VectorIndex()
index.add(["claim1", "claim2", "claim3"], vecs)
results = index.search(model.embed("democracy and education"), limit=2)

# Novelty scoring — is this claim new to the corpus?
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

## Embedding & whitening

The default model is `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). Chosen over `all-MiniLM-L6-v2` based on our experiments:

| Metric | Multilingual-MiniLM-L12 | MiniLM-L6 |
|--------|------------------------|-----------|
| Classification accuracy | **80%** | 75% |
| Norwegian cross-lingual quality | **0.84** | 0.16 |
| Contradiction separation | **0.15 gap** | ~0 |
| Speed (150 texts) | **0.30s** | 0.35s |
| STS-B Spearman | **0.844** | 0.822 |

### Whitening for domain-specific corpora

Off-the-shelf embeddings put everything in a narrow cone — unrelated texts in the same domain score 0.7+ cosine similarity, making it hard to distinguish "similar" from "identical." Whitening spreads the distribution:

```python
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

**Don't whiten diverse corpora.** On mixed-domain data, raw embeddings already separate well. Whitening helps when your entire corpus is about one field and everything looks the same to the model. Our Karpathy-loop experiment (120 configs) confirmed: **current defaults are rank 1/120** — whitening is the biggest anti-pattern on diverse data.

### Other embedding features

```python
# Matryoshka truncation (reduce dimensions for speed/storage)
model = EmbeddingModel(truncate_dim=256)

# Text genericization (strip numbers, dates, URLs before embedding)
# Prevents "2024" and "$1.5M" from dominating similarity
model = EmbeddingModel(genericize=True)
# +14% accuracy on number/date-heavy text, no effect on proper nouns

# Persistent embedding cache (survives restarts)
model = EmbeddingModel(cache_path="embeddings.db")
# 20K texts: 48s cold → 585ms warm (83x speedup)
# ~2.2 KB per 384-dim entry, stores pre-whitening vectors
```

## Search

Three search modes that compose together:

```python
from amygdala import VectorIndex, FTS5Index, HybridSearch, rerank

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

**Why RRF over convex fusion?** We tested both on 148 documents with 45 queries. RRF is 4x more robust when embedding quality degrades (common in domain-specific corpora). It's also parameter-free.

**Why brute-force over ANN?** At <100K vectors, numpy matrix multiply is faster than index-building overhead. No need for FAISS, Annoy, or HNSWlib until you're well past 100K.

### Search benchmarks

| Dataset | Vector nDCG@10 | FTS5 nDCG@10 | Hybrid | Hybrid + rerank |
|---------|---------------|-------------|--------|-----------------|
| SciFact (5K docs, 300 queries) | 0.484 | 0.638 | 0.674 | **0.641** |
| NFCorpus (3.6K docs) | 0.235 | 0.126 | 0.286 | **0.333** |

FTS5 dominates on scientific text (exact terminology matters); vector dominates on medical queries (semantic matching matters). Hybrid + rerank wins on both.

## Novelty detection

Novelty scoring answers: **"Is this text saying something new relative to what I already have?"**

```python
from amygdala import VectorIndex, novelty_score, batch_novelty, nli_classify

# Basic novelty — 0.0 = exact duplicate, 1.0 = completely novel
score = novelty_score(query_vec, index)

# With topic-local context (higher weight to same-category neighbors)
score = novelty_score(query_vec, index, category_ids={"id1", "id2", "id3"})

# With centroid specificity (generic claims near corpus center get dampened)
# +17% separation on diverse data
score = novelty_score(query_vec, index, use_centroid_specificity=True)

# With temporal decay (older items contribute less to "already known")
# Half-life ~35 days at λ=0.02
ages = {"id1": 0.0, "id2": 30.0, "id3": 90.0}  # age in days
score = novelty_score(query_vec, index, timestamps=ages, decay_lambda=0.02)

# NLI cascade — cosine can't tell paraphrases from contradictions
# (both score ~0.73). NLI cross-encoder resolves this:
result = nli_classify("Education improves outcomes",
                      "Education has no effect on outcomes")
# → {"label": "contradiction", "contradiction": 0.92, ...}
```

### The cosine similarity problem

Cosine similarity **cannot distinguish agreement from disagreement**. Two claims that say opposite things about the same topic often have *higher* cosine similarity than two unrelated claims. This is well-documented in the literature but rarely addressed in embedding libraries.

Amygdala's `classify_pairs()` function implements a cosine + NLI cascade:
- **Above threshold** (e.g., 0.88): cosine-confident KNOWN
- **Below threshold** (e.g., 0.72): cosine-confident NEW
- **In between**: NLI cross-encoder decides (entailment/contradiction/neutral)

This gives 94% accuracy on high-cosine contradictions at ~13ms per pair.

### Performance at scale

Tested on a 27K-claim knowledge base:
- `novelty_score()`: **1.1ms per call**
- `batch_novelty()`: **3.6ms per claim** (brute-force bottleneck)
- Adaptive K scales with index size: K=1 at ≤50 items, K=10 at 1000+

## Clustering

Two strategies optimized for deduplication, not topic discovery:

```python
from amygdala import greedy_centroid_cluster, IncrementalCentroidCluster, pairwise_cosine, extract_pairs

# Batch clustering — good when you have all vectors upfront
clusters = greedy_centroid_cluster(embeddings, threshold=0.85)
# Returns list of clusters (each a list of indices). Singletons excluded.

# Incremental clustering — for streaming/continuous ingestion
# Identical quality to batch at threshold ≥ 0.85, 1.8x faster, zero order sensitivity
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

**Why greedy centroid over union-find?** Union-find causes transitive chaining — at threshold 0.85, it produces clusters of 1,500+ items. Greedy centroid caps naturally at ~50. Discovered this empirically when clustering 27K claims in a knowledge system.

**Why not HDBSCAN?** We tested both. Similar V-measure (~0.55) on 20 Newsgroups. Both are designed for dedup, not topic discovery. Greedy centroid is simpler, needs no hyperparameter tuning, and works incrementally.

## Knowledge mapping

Adaptive knowledge probing: efficiently map what someone knows about a topic using information theory.

```python
from amygdala.knowledge_map import KnowledgeGraph, init_beliefs, next_probe, update_beliefs, coverage_report, knowledge_fringes

# Define a knowledge graph (or generate one with LLM — see below)
graph = KnowledgeGraph(nodes=[
    {"id": "crdt", "title": "CRDTs", "level": 1, "description": "Conflict-free replicated data types"},
    {"id": "lamport", "title": "Lamport clocks", "level": 2, "prerequisites": ["crdt"]},
    {"id": "mirror", "title": "Mirror protocol", "level": 3, "prerequisites": ["crdt", "lamport"]},
])

# Initialize belief state (priors based on concept obscurity)
state = init_beliefs(graph)

# Get next question — maximizes expected information gain (Shannon entropy)
probe = next_probe(graph, state)
# → {"node_id": "crdt", "question_type": "recognition", "information_gain": 1.2, ...}

# User responds with familiarity level
update_beliefs(graph, state, "crdt", "solid")
# Automatically propagates: knowing CRDTs well → prerequisites likely known too

# Check coverage
report = coverage_report(graph, state)
# → {"known": [...], "unknown": [...], "uncertain": [...], "coverage_pct": 33.3}

# Find learning frontier
fringes = knowledge_fringes(graph, state)
# → {"outer_fringe": ["lamport"], ...}  — ready to learn next
```

Features:
- **Expected Information Gain** probe selection (simulates all possible answers)
- **Multi-hop belief propagation** through prerequisite DAG (with dampening)
- **Overclaiming detection** via foil concepts (signal detection theory)
- **KST inner/outer fringe** computation for learning path recommendations
- **LLM-powered graph generation** from domain descriptions or document outlines

```python
# Generate a knowledge graph from a topic description
from amygdala.knowledge_map_gen import graph_from_description
graph = await graph_from_description("Conflict-free replicated data types")
# → 15-50 nodes with prerequisites, obscurity levels, descriptions
```

## SQLite connection helper

```python
from amygdala import connect

conn = connect("my.db")  # or connect("my.db", readonly=True)
```

Applies all best practices automatically: WAL journal mode, 30s busy timeout, NORMAL synchronous, 64MB page cache, foreign key enforcement. Use this for any project that touches SQLite.

## Cross-lingual support

The multilingual model achieves **MRR=1.0** on Norwegian→English retrieval out of the box. No translation step needed — embed Norwegian and English text into the same space and search across languages natively.

```python
model = EmbeddingModel()
v_no = model.embed("Utdanning er viktig for demokratiet")
v_en = model.embed("Education is important for democracy")
similarity = float(v_no @ v_en)  # → 0.86
```

## Design decisions (with evidence)

Every significant design choice was tested in controlled experiments. 21 experiments total, each with a specific hypothesis, dataset, and quantitative result:

| # | Question | Finding | Dataset |
|---|----------|---------|---------|
| 1 | Best embedding model? | Multilingual-MiniLM-L12 wins on all metrics | 150 calibration pairs |
| 2 | Does whitening help? | **Situational.** Helps domain-specific (+32%), hurts diverse (-3%) | STS-B, QQP, calibration |
| 3 | Optimal novelty K? | Adaptive: K=1 for ≤50 items, K=10 for 1000+ | Calibration set |
| 4 | RRF vs convex fusion? | RRF 4x better under embedding degradation | 148 docs, 45 queries |
| 5 | Clustering method? | All methods ~0.55 V-measure. Greedy centroid simplest. | 20 Newsgroups |
| 6 | NLI for contradictions? | 94% accuracy on high-cosine contradictions | SICK (4,906 pairs) |
| 7 | Text genericization? | +14% on numbers/dates, 0% proper nouns, -6% URLs | 50 claim pairs |
| 8 | Are defaults optimal? | **Yes.** Rank 1/120 in grid search. | 120 configs, 3 datasets |
| 9 | Cross-encoder reranking? | +32.5% nDCG@10 on SciFact | 5K docs, 300 queries |
| 10 | Temporal decay? | +9.3% Spearman at λ=0.02 (half-life ~35 days) | Time-ordered calibration |
| 11 | Whitening on domain data? | +34.5% gap at 64d, +24% at 128d | 27K education claims |
| 12 | Soft-ZCA vs PCA? | Soft-ZCA strictly better (+32% vs +24%) | Domain calibration |
| 13 | Similarity graph layer? | Graph BFS surfaces 64% items vector misses | 27K claims |
| 14 | Task-specific LoRA? | Not worth it. Search↔novelty correlation -0.953. | Multi-task eval |
| 15 | Novelty at 27K scale? | 1.1ms/call. Works fine. | 27K domain claims |
| 16 | Cross-lingual retrieval? | MRR=1.0 Norwegian→English. Translation unnecessary. | Bilingual claim set |
| 17 | PRF query expansion? | **Hurts** (-1.2% to -7.2%). Don't do it. | SciFact |
| 18 | Incremental clustering? | Identical to batch at ≥0.85. 1.8x faster. | Synthetic + real |
| 19 | NFCorpus search? | Hybrid+rerank best (0.333 nDCG). | 3.6K medical docs |
| 20 | Persistent cache? | 83–452x speedup. Lossless. | 20K embeddings |
| 21 | All-but-the-top? | Matches Soft-ZCA (+27.4%), simpler math | Domain calibration |

Experiment code is in the `experiments/` directory if you want to reproduce or extend them.

## Architecture

Nine modules, minimal cross-dependencies:

```
embed.py ──→ cache.py (optional persistent cache)
   ↓
search.py ──→ VectorIndex, FTS5Index, HybridSearch, rerank
   ↓
novelty.py ──→ uses VectorIndex for neighbor lookup
   │
cluster.py ──→ standalone (numpy only)
   │
index.py ──→ uses search.py (VectorIndex, FTS5) + connect() helper
   │
knowledge_map.py ──→ pure algorithm (no ML, no IO)
   │
knowledge_map_gen.py ──→ uses llm.py for graph generation
   │
llm.py ──→ standalone (Gemini/Anthropic/OpenAI)
```

Design principles:
- **No external services.** Everything runs locally. SQLite for persistence, numpy for vectors.
- **Opt-in complexity.** Basic usage needs only numpy + sentence-transformers. Whitening, NLI, reranking, and LLM features are all opt-in.
- **Numpy arrays everywhere.** All embedding operations return `np.ndarray` for interop.
- **Two-tier caching.** In-memory LRU (fast path) + optional SQLite persistent cache.

## Tests

136 tests covering all modules:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

CI runs on every PR via GitHub Actions.

## Used in production

Amygdala powers search and knowledge management in several systems:

- A **67K-node claims-first knowledge system** — embedding, novelty detection, hybrid search, clustering (canonical finding synthesis), and cosine+NLI cascade for deduplication. Podcast fact-checking found that structured search changes 31% of verdicts vs. flat embedding search alone.
- A **reading and annotation system** — novelty scoring and `classify_pairs` to detect when new annotations overlap with existing knowledge.
- A **conversation search tool** — hybrid RRF search over chat history.

## License

MIT
