# Limbic

**Data curation toolkit: embeddings, search, proposals, and AI-assisted verification.**

Limbic is four Python packages for collections of short, knowledge-dense text —
claims, findings, notes, entity records — where the hard questions are *"is this
already in here?"*, *"are these two the same thing?"*, and *"did the model get
this right, and what did it cost?"*

```bash
pip install git+https://github.com/houshuang/limbic.git
```

**`limbic.amygdala` finds patterns.** Embedding with whitening, vector + FTS5
hybrid search with RRF fusion and cross-encoder reranking, novelty scoring,
clustering, document similarity, knowledge mapping, temporal reasoning, Wikidata
lookup, retrieval evaluation. It is numpy and SQLite; there is no vector
database and no service to run. Its defaults come out of 23 controlled
experiments, [written up with their numbers](docs/EVIDENCE.md).

**`limbic.hippocampus` manages changes.** Deterministic entity resolution
against a local index, a preimage-checked write boundary that refuses a stale or
out-of-scope write, cascade merges that relink every reference, veto-gate dedup,
composable validation, a file-locked YAML proposal store. It is
storage-agnostic: merges, validation and dedup all take callbacks, so you bring
your own backend.

**`limbic.cerebellum` runs and accounts for the model calls.** One cached,
ledgered entry point for every provider call; stateless packets with budgets
that refuse; a yield probe that raises before you build machinery around a dead
idea; resumable batch processing with tiered escalation; Claude/Codex CLI
wrappers with filesystem isolation; forensics over interactive agent sessions.

**`limbic.drive` decides what to do first** — it gates a plan against a
calibration-first policy before it spends anything. Pure policy, no I/O.

The first three form a pipeline (find → change → verify); the fourth runs before
any of it. Everything is local — SQLite for persistence, numpy for vectors, no
external services.

| Package | API reference |
|---|---|
| `limbic.amygdala` | [→](limbic/amygdala/README.md) |
| `limbic.hippocampus` | [→](limbic/hippocampus/README.md) |
| `limbic.cerebellum` | [→](limbic/cerebellum/README.md) |
| `limbic.drive` | [→](limbic/drive/README.md) |

## Start here if you care about LLM pipelines

This is the part of the library with the most recent thinking in it, and the
part that reads least like a search toolkit. Most of it was lifted out of
working projects in September 2026 after an audit of what the pipelines around
here were actually doing with their money.

Read [**docs/new-data-project-checklist.md**](docs/new-data-project-checklist.md)
first — a tiered checklist where every item is tagged with the function that
enforces it, or marked as prose if nothing does. Then:

| Doc | What it covers |
|---|---|
| [`docs/calls.md`](docs/calls.md) | `cached_call`: response cache, `request=` verbatim bytes, replicate agreement and `Held`, `CallMeta`, and why a bookkeeping failure never loses a billed response |
| [`docs/cost-log.md`](docs/cost-log.md) | the ledger — `outcome`, `cache_hit`, `packet_id`, `billing_mode` and notional dollars for subscription calls, the price functions, and `cerebellum.forensics` for the interactive spend a ledger never sees |
| [`docs/packet.md`](docs/packet.md) | `cerebellum.packet`: the unit of work is a call, not an agent. Probes, budgets, prompt-cache rules, truncation, quote anchors |
| [`docs/resolve.md`](docs/resolve.md) | `hippocampus.resolve`: retrieve candidates in code so a hallucinated identifier is unrepresentable |
| [`docs/apply.md`](docs/apply.md) | `hippocampus.apply`: the model proposes, code writes |
| [`docs/refuse.md`](docs/refuse.md) | `hippocampus.refuse`: guards that run before the write — schema refusals in dry *and* real runs, implausible life dates, prose that contradicts its own structured fields, and a declared changed-count bound |
| [`docs/audit.md`](docs/audit.md) | `hippocampus.audit`: folding an independent audit in demote-only — it can take a decision back, never make one |
| [`docs/blind-audit.md`](docs/blind-audit.md) | how to brief the second reader that produces that audit: different model family, full set, fixed verdict vocabulary, `cannot_tell` encouraged |
| [`docs/proposed-batch-and-cache-lifts.md`](docs/proposed-batch-and-cache-lifts.md) | not built yet: narrow cache-key projections, and batch validation that returns partial success for retry |

Three findings from that work, if you want to know whether this is worth your
time before reading further:

- One traced 25-page unit of work cost **3.8M tokens** as a tool-using subagent
  and **≈40K** as one stateless structured-output call — same model, same work.
- A campaign with 12.3k lines of machinery, 26 prompt versions and 13 test files
  produced **0 writes**, while a plain deterministic join next door produced
  2,336 of 2,342 proposals. Nobody had run a yield probe first. `probe()` now
  raises rather than reports.
- Of 901 Wikidata QIDs audited in one catalogue, **198 pointed at something that
  was not that work**, and every one passed an existence check. *Et dukkehjem*
  resolved to Ramon Llull. Hence `wikidata_type_is`, not just `wikidata_exists`.

## Quick start

```python
from limbic.amygdala import (
    EmbeddingModel, VectorIndex, greedy_centroid_cluster, batch_novelty,
)

# Single-domain corpus → whiten, or everything lands in a narrow 0.7-0.9 band
model = EmbeddingModel(genericize=True, whiten_epsilon=0.1, cache_path="cache.db")
model.fit_whitening(texts)
vecs = model.embed_batch(texts)

# Find the shared arguments. Sweep the threshold on YOUR data — see below.
clusters = greedy_centroid_cluster(vecs, threshold=0.75)

# Score how novel each new item is against what you already had
index = VectorIndex()
index.add(existing_ids, existing_vecs)      # NOT the items you're scoring
scores = batch_novelty(new_vecs, index)     # 0.0 = already known, 1.0 = new
```

Three things decide most outcomes, and two of them bite quietly:

- **Whiten a single-domain corpus.** Raw embeddings compress into a narrow
  0.7–0.9 band and clustering, novelty and search all degrade together.
- **Sweep the clustering threshold; never inherit one.** Whitening moves the
  whole distribution, so a threshold that worked elsewhere can return *zero*
  clusters here — in one 40-text corpus, whitening left a maximum pairwise
  similarity of 0.845, so a 0.85 threshold matched nothing at all. Start at 0.75
  and sweep [0.70, 0.75, 0.80, 0.85].
- **Don't score items against an index that contains them.** Each item is then
  its own nearest neighbour. Below 51 items the adaptive top-k is 1, so every
  score comes back exactly `0.0`; above it the scores are merely depressed. Hold
  the items out, or pass an explicit `top_k`.

Full API docs live in each package's README, linked in the table above.

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

## Where it came from

Limbic grew out of the same problems appearing across multiple projects:

- **otak / alif** — a 67K-node claims-first knowledge system where new annotations needed novelty detection ("is this claim already captured?"), clustering for dedup, and cosine+NLI cascade to tell paraphrases from contradictions
- **petrarca** — a news curation pipeline that needed document-level similarity matching, calibrated thresholds for "related" vs "near-duplicate," and hybrid search across multilingual content
- **kulturperler** — a Nordic performing arts archive (10,000+ entities) where deduplicating persons required fuzzy matching with veto gates, merging records meant cascade-relinking all performances and credits, and LLM verification of 2,400+ works needed budget control across 30+ audit sessions (~$270 total)
- **conversation search** — hybrid RRF search over chat history, where the FTS5 query sanitization and cross-encoder reranking patterns were first validated
- **reading/annotation tools** — novelty scoring and `classify_pairs` to detect when new annotations overlap with existing knowledge

The same patterns kept recurring: deduplicating entities by fuzzy name, merging records with cascading references, tracking what an LLM had verified, staying within API budgets, searching across languages. Limbic is the generalized result: three packages that handle the full pipeline from **finding patterns** in data to **managing the changes** to **verifying correctness**, plus a fourth (`limbic.drive`) that decides what to spend effort on before any of it starts.

## Install

```bash
L=git+https://github.com/houshuang/limbic.git
pip install $L                           # core: numpy, sentence-transformers, transformers
pip install "limbic[hippocampus] @ $L"   # + pyyaml, for the YAML proposal store
pip install "limbic[llm] @ $L"           # + google-genai, anthropic, openai
pip install "limbic[temporal] @ $L"      # + edtf, for full EDTF dates
```

From a clone, `pip install -e ".[dev]"` installs every extra plus pytest.

The built-in `openai` and `gemini` transports in `cerebellum.calls` use stdlib
`urllib` and need **none** of the `[llm]` extra — that extra is for
`amygdala.llm`. Requires Python 3.11+.

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
instead of a fan-out. See [`limbic.drive`](limbic/drive/README.md).

## Design decisions

Every significant choice in `limbic.amygdala` was tested in a controlled
experiment — 23 of them, each with a hypothesis, a dataset and a number. A few
that change how you use it:

| Question | Finding |
|---|---|
| Does whitening help? | **Situational.** +32% on domain-specific, −3% on diverse |
| RRF vs convex fusion? | RRF 4× more robust under embedding degradation |
| Cross-encoder reranking? | +16% nDCG on NFCorpus, −5% on SciFact — dataset-dependent |
| PRF query expansion? | **Hurts** (−1.2% to −7.2%). Don't |
| Cross-lingual retrieval? | MRR=1.0 Norwegian→English. Translation unnecessary |
| Task-specific LoRA? | Not worth it. Search/novelty correlation −0.953 |

**[Full table of all 23 experiments and the production-corpus findings →](docs/EVIDENCE.md)**,
including the two results worth knowing before you build something: LLM
reranking cascades never beat the free cross-encoder (the bottleneck is
first-stage recall, not ranking), and agentic file-reading wins on quality but
erodes with scale at ~1000× the cost.

## Common pitfalls

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

Four packages, no external services. See each package README for its module
layout; the short version:

- **amygdala** is numpy and SQLite. `embed` → `cache`, `search` → `VectorIndex` /
  `FTS5Index` / `HybridSearch`, then `novelty`, `cluster`, `document_similarity`
  on top. `llm`, `wikidata` and `temporal` are the optional-dependency edges.
- **hippocampus** is storage-agnostic by design: cascade merges, validation and
  dedup all take callbacks, so you bring your own backend. `store` is the
  file-locked YAML one if you don't have one.
- **cerebellum** wraps the expensive, failure-prone parts: `calls` is the one
  cached, ledgered entry point every provider call goes through; `packet` puts
  budgets and a yield probe on top of it; `cost_log` is the ledger and
  `forensics` covers the sessions it cannot see; `batch` and `orchestrator` for
  resumability and tiered escalation; `claude_cli`/`codex_cli`/`sandbox` for
  driving an agent CLI safely; `windowing` for sliding-window extraction.
- **drive** is pure policy — no I/O, no model calls.

Design principles: no external services; storage-agnostic callbacks; numpy
arrays everywhere in amygdala; two-tier caching (in-memory LRU + optional
SQLite).

**The pipeline layer imports with the standard library only.**
`cerebellum.calls`, `cerebellum.packet`, `cerebellum.cost_log` and
`hippocampus.resolve` load in ~0.35 s with numpy, torch and
sentence-transformers blocked — `connect` lives in `limbic._sqlite` so a packet
runner does not drag the embedding stack in behind it. That was worth 0.13–0.33 s
against 0.03 s per process, and an `ImportError` in an interpreter without them.

The install does not yet reflect that: `pip install limbic` still pulls numpy,
sentence-transformers and transformers (and therefore torch) as hard
dependencies, so using only the pipeline layer costs a large install you never
import. Splitting the embedding stack into an extra is the obvious next
packaging change and has not been made.

## Tests

872 tests:

```bash
pip install -e ".[dev]"      # pulls the llm/temporal/hippocampus extras too
python -m pytest tests/ -v
```

| Package | Tests |
|---------|-------|
| limbic.amygdala | 371 |
| limbic.hippocampus | 194 |
| limbic.cerebellum | 302 |
| limbic.drive | 5 |

Ten of those hit the live Wikidata API (`test_wikidata_live.py`,
`test_wikidata_resolve_live.py`); deselect them for an offline run. `[dev]`
deliberately installs every optional extra — when it did not, CI ran a quietly
smaller suite than a developer's venv and drifted red without anyone noticing.

CI runs on every PR via GitHub Actions.

## Who actually uses this

Worth being precise about, because a library extracted from working projects can
look more adopted than it is. An audit on 20 September 2026 found that limbic
gets adopted exactly where it is **one import and one call**, and not otherwise.

**Adopted, verified against real data:**

- **skard** routes every provider call through `cached_call`. Its first attempt
  at migrating was rejected: the transport rebuilt the request body, which would
  have changed the hash its paid artefacts were addressed by. After `request=`
  was added, request bytes were identical on 850 of 850 stored passes.
- **Kulturbase**'s `kb_resolve` takes its folding primitives from
  `hippocampus.resolve`, which is why `fold(profile="ascii")` exists — its own
  fold maps æ→ae where the default maps æ→a, and 505 of 3,908 strings differed.
  Gold-set rows came out identical after the switch.
- The older search and curation surface is in day-to-day use across the projects
  listed below.

**Not adopted anywhere, including by the project it was extracted from:**

- **`hippocampus.ProposalStore` has zero consumers.** It was lifted out of
  Kulturbase, and Kulturbase does not import it back. It is a filing cabinet
  without a lock — no preimage check, so an `approved` status in it is a string
  in a file. `hippocampus.apply.apply_proposal` is the mechanism that actually
  refuses bad writes, and it is a function precisely because of this.
- Three of three candidate migrations were rejected on 20 September, each
  because an extracted function did not fit the code it was extracted from.
  Every one of those gaps is closed in the 21 September changelog entry; none of
  them was visible from inside this repo.

**Known gaps**, stated so you do not discover them the hard way:

- **Token-layer spelling collisions are open.** The folded layer now refuses to
  match keys from different spelling tables (Bø vs Bøe), but `token_overlap` and
  `text_candidates` still compare tokens across tables. That needs
  spelling-tagged tokens.
- **There is no public usage parser for a raw Responses payload.** With
  `cached_call(request=...)` you get the provider's raw response dict back and
  parse it yourself; skard keeps its own `usage_of`. Post-call fields on a
  ledger row are also missing.
- An index built before the `spelling` column existed reads as `"both"` until
  that kind is rebuilt, so it keeps the old matching behaviour silently.

## Where it is used

- **otak / alif** — a **67K-node claims-first knowledge system** using embedding, novelty detection, hybrid search, clustering (canonical finding synthesis), and cosine+NLI cascade for deduplication. Podcast fact-checking showed that structured search changes 31% of verdicts vs. flat embedding search alone.
- **petrarca** — a **news curation pipeline** using document similarity to find related articles, calibrated thresholds for feed ranking vs near-duplicate detection, and hybrid search across multilingual content.
- **kulturperler** — a **Nordic performing arts archive** (10,000+ entities) using proposals for all data changes, cascade merges for deduplicating persons/works, tiered LLM verification of 2,400+ works across 30+ audit sessions, veto-gate dedup of fuzzy-matched person names. Total audit cost: ~$270. The DR-arkivet import scripts use `StateStore` and `AuditLogger` for resumable batch imports with JSONL audit trails, and `connect()` for all SQLite access.
- A **reading and annotation system** using novelty scoring and `classify_pairs` to detect when new annotations overlap with existing knowledge.
- **[claude-chat-search](https://github.com/houshuang/claude-chat-search)** — hybrid RRF search over Claude Code chat history with optional LLM query expansion via `expand_query` and `multi_list_rrf`.
- **hvaskjer** — an unsupervised nightly culture-listings pipeline that feeds scraped pages and forwarded email to `codex_research`. The isolation primitives in `cerebellum.sandbox` come from it, and it is the reason `codex_research` is `isolated` by default.
- **otak / hirsch-atlas** — book-length argument extraction (10 books, 101 chapters, ~10k claims) using `cerebellum.windowing` for the sliding-window extraction and cross-window merge.
- **a personal 20-year corpus** (blog, notes, talks, transcripts, tweets) — the pooled-judgment evaluation in [Design decisions](docs/EVIDENCE.md) ran here, using `retrieval_eval` and `serendipity`.

## License

MIT
