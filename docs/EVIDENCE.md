# Design decisions (with evidence)

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

## Findings from production corpora

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

---

Part of [limbic](../README.md).
