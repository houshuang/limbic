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

If your largest cluster has 50+ members, your threshold is too low or you need whitening. If you get *zero* clusters, it is too high — whitening can leave a corpus whose maximum pairwise similarity is below 0.85. Start at 0.75 post-whitening, then sweep [0.70, 0.75, 0.80, 0.85] on your data. Validated on 27K education claims (0.85), 6.5K political proposals (0.75–0.80), and 1.7K op-ed claims (0.75).

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
from limbic.amygdala import EmbeddingModel, pairwise_cosine, extract_pairs
from limbic.hippocampus import (
    CandidatePair, ProposalStore, VetoMatcher, exact_field, initial_match,
)

# 1. Find candidate pairs via embedding similarity.
#    extract_pairs returns (i, j, score) INDICES into the input array.
model = EmbeddingModel()
vecs = model.embed_batch([e["name"] for e in entities])
pairs = extract_pairs(pairwise_cosine(vecs), threshold=0.80)

# 2. Filter through veto gates. check_pair takes a CandidatePair carrying both
#    records, not the index tuple — the gates read fields, not embeddings.
matcher = VetoMatcher(gates=[initial_match("name"), exact_field("birth_year")])
store = ProposalStore("proposals/")
for i, j, score in pairs:
    a, b = entities[i], entities[j]
    result = matcher.check_pair(CandidatePair(
        id_a=a["id"], id_b=b["id"], fields_a=a, fields_b=b, score=score))
    if not result.accepted:
        print(result.reason)   # "exact_birth_year: birth_year differs: 1828 vs 1808"
        continue
    # 3. File a merge proposal. Refs are "type/id", not bare ids.
    store.create_merge(f"person/{a['id']}", f"person/{b['id']}",
                       title=f"Merge {a['name']} into {b['name']}",
                       reasoning=f"Similarity {score:.2f}")
```

A filed proposal is `status="pending"` in a YAML file. `ProposalStore` has **no
preimage check**, so approving one is a string change in a file — to actually
write, go through `hippocampus.apply.apply_proposal`, which refuses a stale or
out-of-whitelist write. For merges that must relink every reference, build a
`ReferenceGraph([ReferenceSpec(...)])` and call `apply_merge(graph, source_id,
target_id, entity_type, data_loader, data_writer, data_deleter)`.

### Apply a codebook / extract / classify over N documents

Do not spawn agents as coders: one traced packet cost 3.8M tokens as a
tool-using subagent and ≈40K as one stateless call. Probe the yield before
building anything — a campaign with 12.3k lines of machinery produced 0 writes
while a plain join next door produced 2,336 of 2,342 proposals.

```python
from limbic.hippocampus.resolve import build_index, text_candidates, slot_enum, unslot
from limbic.cerebellum.packet import make_packet, lint_packet, probe, run_packets

index = build_index("kb.idx", rows, kind="person")
cards = text_candidates(index, page_text, k=40)          # code retrieves, model picks
fragment, slot_map = slot_enum(cards, n_slots=40)        # fixed slots: schema identical per batch
packets = [make_packet(PREFIX, body, SCHEMA, prompt_version="v1") for body in bodies]
print(lint_packet(packets))                              # caching + derivable-field warnings
probe(packets, n=50, yield_fn=lambda r: len(r["items"]), min_yield=0.2,
      project="p", purpose="code", execute=True)         # raises LowYield: do not scale
run_packets(packets, project="p", purpose="code", max_calls=200,
            max_tokens=2_000_000, split=halve, execute=True)
```

Then write through the boundary, never directly:

```python
from limbic.hippocampus.apply import MISSING, apply_proposal, wikidata_type_is

apply_proposal(path, {"wikidata_id": qid}, preimage={"wikidata_id": MISSING},
               allowed_fields={"wikidata_id"}, validators=[wikidata_type_is("work")],
               receipt=Path("receipts.jsonl"))
```

`probe(n=50)` samples 50 **packets**, not 50 items, and needs `execute=True` —
a dry-run probe has nothing to measure and raises `LowYield` every time.

See `docs/new-data-project-checklist.md` before starting a new data project,
then `docs/packet.md`, `docs/resolve.md`, `docs/apply.md`, `docs/calls.md`
(the cache, `request=`, replicate agreement) and `docs/cost-log.md` (the
ledger, outcomes, forensics).

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

**"I get zero clusters"**
The mirror image, and more common after whitening than people expect. Whitening
moves the whole similarity distribution down, so an inherited threshold can sit
above your corpus's *maximum* pairwise similarity and match nothing. Check it:
`pairwise_cosine(vecs)` off-diagonal max. Start at 0.75 post-whitening and sweep.

**"Every novelty score is exactly 0.0"**
You scored vectors that are in the index, so each item is its own nearest
neighbour. Below 51 items the adaptive top-k is 1, which makes this exact rather
than approximate — every score is `0.0`. Above 51 the scores are quietly
depressed instead. Score held-out items against an index of what you already
had, or pass an explicit `top_k`.

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
- `extract_pairs()` returns `list[tuple[int, int, float]]` — **indices**, not ids.
  `VetoMatcher.check_pair()` takes a `CandidatePair` carrying both records.
- `apply_proposal()` with `writer=` does **not** also update an in-memory
  mapping; persist what the writer receives.
- `cached_call()` requires `purpose=`, and infers `project=` from the git root
  rather than defaulting.
- All functions are synchronous. Async LLM calls available via `limbic.amygdala.llm`.

## Package Overview

| Package | Import | Purpose |
|---|---|---|
| `limbic.amygdala` | `from limbic.amygdala import EmbeddingModel, VectorIndex, ...` | Embedding, search, novelty, clustering, calibration |
| `limbic.hippocampus` | `from limbic.hippocampus import apply_proposal, candidates, ...` | Entity resolution, the preimage-checked write boundary, cascade merges, dedup, validation |
| `limbic.cerebellum` | `from limbic.cerebellum import make_packet, run_packets, ...` | Stateless packets, response cache, cost ledger, LLM batch verification, Claude/Codex CLI wrappers, agent isolation, windowed extraction |
| `limbic.drive` | `from limbic.drive import validate_plan` | Calibration-first plan policy: refuse a plan that fans out before one pilot |

See README.md for full API documentation with benchmarks and experiment evidence.
