# Limbic

**Data curation toolkit: embeddings, search, proposals, and AI-assisted verification.**

Limbic grew out of the same problems appearing across multiple projects:

- **otak / alif** — a 67K-node claims-first knowledge system where new annotations needed novelty detection ("is this claim already captured?"), clustering for dedup, and cosine+NLI cascade to tell paraphrases from contradictions
- **petrarca** — a news curation pipeline that needed document-level similarity matching, calibrated thresholds for "related" vs "near-duplicate," and hybrid search across multilingual content
- **kulturperler** — a Nordic performing arts archive (10,000+ entities) where deduplicating persons required fuzzy matching with veto gates, merging records meant cascade-relinking all performances and credits, and LLM verification of 2,400+ works needed budget control across 30+ audit sessions (~$270 total)
- **conversation search** — hybrid RRF search over chat history, where the FTS5 query sanitization and cross-encoder reranking patterns were first validated
- **reading/annotation tools** — novelty scoring and `classify_pairs` to detect when new annotations overlap with existing knowledge

The same patterns kept recurring: deduplicating entities by fuzzy name, merging records with cascading references, tracking what an LLM had verified, staying within API budgets, searching across languages. Limbic is the generalized result: three packages that handle the full pipeline from **finding patterns** in data to **managing the changes** to **verifying correctness**.

## Three packages, one pipeline

```
limbic.amygdala          limbic.hippocampus          limbic.cerebellum
 finds patterns            manages changes             verifies correctness
 ─────────────           ──────────────────          ─────────────────────
 Embedding               Proposals                   Batch processing
 Vector search            (modify/merge/delete        (resumable, budget-
 Hybrid search             with lifecycle)              tracked, persistent)
 Novelty detection       Cascade merges              Multi-tier orchestrator
 Clustering               (relink all references       (triage -> deep verify
 Document similarity       when merging entities)       with auto-escalation)
 Knowledge mapping       Deduplication               Audit logging
 LLM client               (veto-gate filtering)       (JSONL with analysis)
 Calibration metrics     Validation                  Context builder
 SQLite helpers            (composable rules)           (for LLM prompts)
                         YAML store
                          (file-locked atomic)
```

| Package | Purpose | Core dependency |
|---------|---------|-----------------|
| **limbic.amygdala** | Find patterns: embed, search, deduplicate, score novelty | numpy, sentence-transformers |
| **limbic.hippocampus** | Manage changes: proposals with review lifecycle, cascade merges, validation | pyyaml |
| **limbic.cerebellum** | Verify correctness: LLM-assisted batch audits with budget control | (none beyond stdlib) |

Each package has its own detailed README in its directory.

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
| **search** | Numpy vector search, SQLite FTS5, hybrid RRF fusion, cross-encoder reranking | +32.5% nDCG with reranking; RRF 4x more robust than convex fusion under embedding degradation |
| **novelty** | Multi-signal novelty scoring: global + topic-local + centroid specificity + temporal decay + NLI cascade | +17% novel/known separation with centroid specificity; NLI fixes 94% of high-cosine contradictions |
| **cluster** | Greedy centroid clustering (batch + incremental), complete linkage, pairwise cosine, confidence-calibrated pair classification | Incremental matches batch quality at threshold >= 0.85, 1.8x faster; order-sensitive at lower thresholds |
| **document_similarity** | Document-level thematic similarity using weighted multi-field embeddings | 94% accuracy on human-rated pairs; AUROC=0.930 on 300-pair dataset; rho=0.818 |
| **calibrate** | Cohen's kappa, LLM judge validation (Bootstrap Validation Protocol), intra-rater reliability | Validates LLM judges against human gold labels |
| **cache** | Persistent SQLite-backed embedding cache | 20K texts: 48s cold → 585ms warm |
| **index** | SQLite document/chunk storage with hybrid search | Single-file, zero-config, FTS5 built in |
| **knowledge_map** | Adaptive knowledge probing via entropy maximization with heuristic or exact belief propagation (zero deps) | Converges in 8–12 questions on 30-node graphs |
| **llm** | Multi-provider LLM client (Gemini, Anthropic, OpenAI) with structured output and retry | Auto-fallback, cost tracking, async + sync |

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

**Don't whiten diverse corpora.** On mixed-domain data, raw embeddings already separate well. Whitening helps when your entire corpus is about one field and everything looks the same to the model. The Karpathy-loop experiment (120 configs) confirmed: **current defaults are rank 1/120** — whitening is the biggest anti-pattern on diverse data.

#### Other embedding features

```python
from limbic.amygdala import EmbeddingModel

# Matryoshka truncation (reduce dimensions for speed/storage)
model = EmbeddingModel(truncate_dim=256)

# Text genericization (strip numbers, dates, URLs before embedding)
# Prevents "2024" and "$1.5M" from dominating similarity
model = EmbeddingModel(genericize=True)
# +14% accuracy on number/date-heavy text, no effect on proper nouns

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

# Cross-encoder reranking -- +32.5% nDCG@10 on SciFact
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

# NLI cascade -- cosine can't tell paraphrases from contradictions
# (both score ~0.73). NLI cross-encoder resolves this:
result = nli_classify("Education improves outcomes",
                      "Education has no effect on outcomes")
# -> {"label": "contradiction", "contradiction": 0.92, ...}
```

#### The cosine similarity problem

Cosine similarity **cannot distinguish agreement from disagreement**. Two claims that say opposite things about the same topic often have *higher* cosine similarity than two unrelated claims. This is well-documented in the literature but rarely addressed in embedding libraries.

The `classify_pairs()` function implements a cosine + NLI cascade:
- **Above threshold** (e.g., 0.88): cosine-confident KNOWN
- **Below threshold** (e.g., 0.72): cosine-confident NEW
- **In between**: NLI cross-encoder decides (entailment/contradiction/neutral)

This gives 94% accuracy on high-cosine contradictions at ~13ms per pair.

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

**Why greedy centroid over union-find?** Union-find causes transitive chaining — at threshold 0.85, it produces clusters of 1,500+ items. Greedy centroid caps naturally at ~50. Discovered this empirically when clustering 27K claims in alif.

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
from limbic.amygdala.knowledge_map import KnowledgeGraph, init_beliefs, next_probe, update_beliefs, coverage_report, knowledge_fringes

# Define a knowledge graph (or generate one with LLM -- see below)
graph = KnowledgeGraph(nodes=[
    {"id": "crdt", "title": "CRDTs", "level": 1, "description": "Conflict-free replicated data types"},
    {"id": "lamport", "title": "Lamport clocks", "level": 2, "prerequisites": ["crdt"]},
    {"id": "mirror", "title": "Mirror protocol", "level": 3, "prerequisites": ["crdt", "lamport"]},
])

# Initialize belief state (priors based on concept obscurity)
state = init_beliefs(graph)

# Get next question -- maximizes expected information gain (Shannon entropy)
probe = next_probe(graph, state)
# -> {"node_id": "crdt", "question_type": "recognition", "information_gain": 1.2, ...}

# User responds with familiarity level
update_beliefs(graph, state, "crdt", "solid")
# Automatically propagates: knowing CRDTs well -> prerequisites likely known too

# Check coverage
report = coverage_report(graph, state)
# -> {"known": [...], "unknown": [...], "uncertain": [...], "coverage_pct": 33.3}

# Find learning frontier
fringes = knowledge_fringes(graph, state)
# -> {"outer_fringe": ["lamport"], ...}  -- ready to learn next
```

Features:
- **Expected Information Gain** probe selection (simulates all possible answers)
- **Multi-hop belief propagation** through prerequisite DAG (with dampening)
- **Overclaiming detection** via foil concepts (signal detection theory)
- **KST inner/outer fringe** computation for learning path recommendations
- **LLM-powered graph generation** from domain descriptions or document outlines

```python
# Generate a knowledge graph from a topic description
from limbic.amygdala.knowledge_map_gen import graph_from_description
graph = await graph_from_description("Conflict-free replicated data types")
# -> 15-50 nodes with prerequisites, obscurity levels, descriptions
```

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
    no_orphans("person", [("work", "playwrights"), ("performance", "credits")]),
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
# summary.total_cost, summary.items_processed, summary.by_tier, summary.by_action

# Extract operations grouped by type (with dedup)
ops = extract_operations(entries, op_types=["fix_name", "merge"])
```

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

## Design decisions (with evidence)

Every significant design choice in limbic.amygdala was tested in controlled experiments. 22 experiments total, each with a specific hypothesis, dataset, and quantitative result:

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
| 9 | Cross-encoder reranking? | +32.5% nDCG@10 on SciFact | 5K docs, 300 queries |
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

Experiment code is in the `experiments/` directory if you want to reproduce or extend them.

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
               rerank                (ReferenceGraph,            orchestrator.py
    |                                 apply_merge,               (TieredOrchestrator,
  novelty.py -> VectorIndex           apply_delete)               VerificationTier)
    |                                                               |
  cluster.py (numpy only)           dedup.py                    audit_log.py
    |                                (VetoMatcher,               (AuditLogger,
  document_similarity.py             VetoGate,                   read_logs,
    |                                 ExclusionList)              extract_operations)
  index.py -> search + connect()                                    |
    |                               validate.py                 context.py
  calibrate.py                       (Validator, Rule,           (ContextBuilder,
    |                                 composable checks)          build_batch_context)
  knowledge_map.py (pure algo)
    |                               store.py
  knowledge_map_gen.py -> llm.py     (YAMLStore, file-locked)
    |
  llm.py (Gemini/Anthropic/OpenAI)
```

Design principles:
- **No external services.** Everything runs locally. SQLite for persistence, numpy for vectors, YAML for hippocampus entities.
- **Opt-in complexity.** Basic usage needs only numpy + sentence-transformers. YAML support, LLM features, and orchestration are all opt-in via extras.
- **Storage-agnostic.** Cascade merges, validation, and batch processing use callback functions — bring your own storage backend.
- **Numpy arrays everywhere.** All embedding operations return `np.ndarray` for interop.
- **Two-tier caching.** In-memory LRU (fast path) + optional SQLite persistent cache.

## How the packages compose

The three packages are independent but designed to work together:

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

## Tests

290 tests covering all three packages:

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

| Package | Tests |
|---------|-------|
| limbic.amygdala | 147 |
| limbic.hippocampus | 54 |
| limbic.cerebellum | 33 |

CI runs on every PR via GitHub Actions.

## Used in production

Limbic powers search, data curation, and knowledge management in several systems:

- **otak / alif** — a **67K-node claims-first knowledge system** using embedding, novelty detection, hybrid search, clustering (canonical finding synthesis), and cosine+NLI cascade for deduplication. Podcast fact-checking showed that structured search changes 31% of verdicts vs. flat embedding search alone.
- **petrarca** — a **news curation pipeline** using document similarity to find related articles, calibrated thresholds for feed ranking vs near-duplicate detection, and hybrid search across multilingual content.
- **kulturperler** — a **Nordic performing arts archive** (10,000+ entities) using proposals for all data changes, cascade merges for deduplicating persons/works, tiered LLM verification of 2,400+ works across 30+ audit sessions, veto-gate dedup of fuzzy-matched person names. Total audit cost: ~$270.
- A **reading and annotation system** using novelty scoring and `classify_pairs` to detect when new annotations overlap with existing knowledge.
- A **conversation search tool** using hybrid RRF search over chat history.

## License

MIT
