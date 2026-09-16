# Limbic — AI Agent Guide

**Data curation toolkit: embeddings, search, proposals, and AI-assisted verification.**

Three core packages: `limbic.amygdala` (find patterns), `limbic.hippocampus` (manage changes), `limbic.cerebellum` (verify correctness) — plus `limbic.drive`, a planning policy that runs before any of them.

## Rules of Thumb

### Always whiten domain-focused corpora

If the corpus is about one domain (education, medicine, politics, performing arts), **always use whitening**. Without it, raw embeddings compress into a narrow similarity band (0.7–0.9) and downstream clustering/novelty/search all degrade.

```python
model = EmbeddingModel(whiten_epsilon=0.1)
model.fit_whitening(corpus_texts)
```

Skip whitening only when your corpus spans many unrelated domains.

### Always genericize number-heavy text

If texts contain variable amounts, dates, section references, or place names around the same argument, use `genericize=True`. This prevents "allocate 50M" and "allocate 200M" from being treated as different arguments.

```python
model = EmbeddingModel(genericize=True, whiten_epsilon=0.1)
```

### Clustering thresholds depend on whitening

| Corpus state | Threshold | Why |
|---|---|---|
| Raw embeddings, diverse corpus | 0.70–0.75 | Embeddings already spread |
| Raw embeddings, domain-focused | 0.90+ | Narrow cone, everything looks similar |
| **Whitened, homogeneous text** (extracted claims) | **0.85** | Very similar surface form |
| **Whitened, diverse authorship** (op-eds, responses) | **0.70–0.75** | Different writers phrase same argument differently |

If your largest cluster has 50+ members, your threshold is too low or you need whitening. Start at 0.75 post-whitening, then sweep [0.70, 0.75, 0.80, 0.85] on your data. Validated on 27K education claims (0.85), 6.5K political proposals (0.75–0.80), and 1.7K op-ed claims (0.75).

### Always validate thresholds before shipping

An initial threshold (even 0.85) can produce false positives in your specific domain. LLM-validate a sample of 50–100 pairs at your chosen threshold before using results downstream. Gemini Flash validation costs ~$0.001/pair.

## Common Pipelines

### Analyze a corpus of responses (policy, reviews, surveys, consultation)

```python
from limbic.amygdala import (
    EmbeddingModel, VectorIndex, greedy_centroid_cluster,
    batch_novelty, pairwise_cosine, extract_pairs, classify_pairs,
)

# 1. Embed with domain-appropriate settings
model = EmbeddingModel(
    genericize=True,          # strip numbers/dates that poison similarity
    whiten_epsilon=0.1,       # domain-focused → always whiten
    cache_path="cache.db",    # avoid re-embedding on reruns
)
texts = [claim["text"] for claim in claims]
model.fit_whitening(texts)
vecs = model.embed_batch(texts)

# 2. Cluster to find shared arguments
clusters = greedy_centroid_cluster(vecs, threshold=0.85)
# Each cluster = group of claims making the same argument
# Count sources per cluster to find "most common arguments"

# 3. Score novelty per claim
index = VectorIndex()
index.add([str(i) for i in range(len(vecs))], vecs)
scores = batch_novelty(vecs, index)
# 0.0 = everyone says this, 1.0 = only this source says it

# 4. Detect contradictions within clusters
pairs = extract_pairs(pairwise_cosine(vecs), threshold=0.72)
classified = classify_pairs(texts, pairs)
# Returns KNOWN (paraphrase), NEW (contradiction), EXTENDS (elaboration)

# 5. Aggregate per source
source_novelty = {}
for claim, score in zip(claims, scores):
    source_novelty.setdefault(claim["source"], []).append(score)
# Rank sources by mean novelty to find "who brings fresh arguments"
```

### Deduplicate entities with merge proposals

```python
from limbic.amygdala import EmbeddingModel, VectorIndex, pairwise_cosine, extract_pairs
from limbic.hippocampus import VetoMatcher, ProposalStore, ReferenceGraph, apply_merge
from limbic.hippocampus import exact_field, initial_match, no_conflict

# 1. Find candidate pairs via embedding similarity
model = EmbeddingModel()
vecs = model.embed_batch([e["name"] for e in entities])
pairs = extract_pairs(pairwise_cosine(vecs), threshold=0.80)

# 2. Filter through veto gates
matcher = VetoMatcher(gates=[
    initial_match("name"),
    exact_field("birth_year"),
    no_conflict("external_id"),
])
confirmed = [p for p in pairs if matcher.check_pair(p).accepted]

# 3. Create merge proposals for review
store = ProposalStore("proposals/")
for a, b, score in confirmed:
    store.create_merge(a, b, title=f"Merge {a} into {b}", reasoning=f"Similarity {score:.2f}")

# 4. After approval, cascade-merge with automatic relinking
graph = ReferenceGraph([...])  # declare entity reference structure
for proposal in store.list_approved():
    apply_merge(graph, proposal.source_id, proposal.target_id, ...)
```

### LLM-verified batch processing with budget control

```python
from limbic.cerebellum import BatchProcessor, StateStore, TieredOrchestrator, VerificationTier

# Tier 1: cheap triage. Tier 2: expensive deep check. Auto-escalate flagged items.
orchestrator = TieredOrchestrator(
    tiers=[
        VerificationTier("triage", triage_fn, cost_estimate=0.001),
        VerificationTier("deep", verify_fn, cost_estimate=0.05),
    ],
    state_store=StateStore(Path("state.db")),
)
results = orchestrator.run(items, id_fn=lambda x: x["id"], max_cost=50.0, escalate=True)
```

## Common Pitfalls

**"My clusters are huge (50+ members)"**
Your threshold is too low, or you're using raw embeddings on domain-focused text. Whiten first (`whiten_epsilon=0.1`), then cluster at 0.85.

**"Everything scores 0.7+ similarity"**
Domain-focused corpus without whitening. The narrow embedding cone compresses all scores. Use `EmbeddingModel(whiten_epsilon=0.1)` and `fit_whitening(corpus)`.

**"Novelty scores are all 0.3–0.5 with no spread"**
Same cause — whitening spreads the distribution so novelty scores become meaningful.

**"NLI says 'contradiction' on obvious paraphrases"**
Cross-encoder is noisy below 0.72 cosine. The default `classify_pairs()` cascade only runs NLI on high-cosine pairs to avoid this. Don't lower the threshold.

**"My reranker isn't helping"**
Probably not the reranker. A pooled IR eval across three increasingly expensive
LLM rerankers (snippets, wide union, full text of top-25) found all of them tied
with the free cross-encoder at ~0.54 nDCG: **the bottleneck is first-stage
recall, not ranking**, and no reranker can reorder a document that isn't in its
candidate list. Widen retrieval or change the encoder instead. Measure with
`limbic.amygdala.retrieval_eval`.

**"Should I swap to a better embedding model?"**
Measure the languages separately. `multilingual-e5-base` beat the MiniLM default
by +0.046 nDCG / +0.08 Recall@20 overall — while regressing on Norwegian (0.442
vs 0.569). An aggregate win routinely hides a per-language regression, and MiniLM
is the default precisely for its cross-lingual strength. E5-style encoders also
need `query: ` / `passage: ` prefixes, which `EmbeddingModel` does not add for you.

**"Cosine says two opposite claims are highly similar"**
This is expected — cosine measures *topical* similarity, not agreement. Two claims about the same topic that say opposite things will score high. Use `classify_pairs()` or `nli_classify()` to distinguish agree/disagree.

## Key API Notes

- `EmbeddingModel.embed()` returns `np.ndarray` (1D). `embed_batch()` returns 2D.
- `fit_whitening()` accepts either a list of strings or a 2D numpy array.
- `VectorIndex.search()` returns `list[Result]` with `.id` and `.score`.
- `greedy_centroid_cluster()` returns `list[list[int]]` — each inner list is indices into the input array. Singletons are excluded.
- `batch_novelty()` returns `list[float]` in same order as input vectors.
- `novelty_score()` returns a single float. Use `batch_novelty()` for bulk.
- `classify_pairs()` expects `list[tuple[int, int]]` indices into a texts list.
- All functions are synchronous. Async LLM calls available via `limbic.amygdala.llm`.

## Package Overview

| Package | Import | Purpose |
|---|---|---|
| `limbic.amygdala` | `from limbic.amygdala import EmbeddingModel, VectorIndex, ...` | Embedding, search, novelty, clustering, calibration |
| `limbic.hippocampus` | `from limbic.hippocampus import ProposalStore, ...` | Change proposals, cascade merges, dedup, validation |
| `limbic.cerebellum` | `from limbic.cerebellum import BatchProcessor, ...` | LLM batch verification, budget tracking, audit logs, Claude/Codex CLI wrappers, agent isolation, windowed extraction |
| `limbic.drive` | `from limbic.drive import validate_plan` | Calibration-first plan policy: refuse a plan that fans out before one pilot |

See README.md for full API documentation with benchmarks and experiment evidence.
