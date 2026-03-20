# Web Research: State of the Art for Amygdala

**Date**: 2026-03-19

## 1. Embedding Models

**all-MiniLM-L6-v2 is definitively outdated** (2021, 5-8% below modern alternatives).

### Top Recommendations

| Model | Params | Dims | Matryoshka | Key Advantage |
|-------|--------|------|------------|---------------|
| ModernBERT-embed-base (Nomic) | ~150M | 768 | To 256d | SOTA for size class, RoPE, Flash Attn, 8K context |
| nomic-embed-text-v1.5 | 137M | 768 | To 64d | Open-source, reproducible, already cached locally |
| snowflake-arctic-embed-s | 33M | 384 | No | Drop-in replacement, same footprint |
| bge-small-en-v1.5 | 33M | 384 | No | Strong accuracy at small size |

**Multilingual (for Norwegian fallback)**:
| Model | Params | Dims | Notes |
|-------|--------|------|-------|
| jina-embeddings-v3 | 570M | 1024 | Task-specific LoRA adapters, strong multilingual |
| Qwen3-Embedding-0.6B | 600M | 1024 | 100+ languages including Norwegian |
| snowflake-arctic-embed-m-v2.0 | — | — | Designed for multilingual retrieval |

**Key insight: Matryoshka makes PCA whitening unnecessary.** MRL-trained models produce valid embeddings at any truncation level (768→256), trained end-to-end. This is strictly superior to post-hoc PCA whitening.

## 2. PCA Whitening vs Alternatives

**PCA whitening is no longer best practice** when using Matryoshka models.

If keeping whitening (for non-Matryoshka models):
1. **Soft-ZCA whitening** (ESANN 2025): Adds epsilon regularization. `W = U(Λ + εI)^{-1/2} U^T`, ε=0.01-0.1. Better than PCA whitening.
2. **All-but-the-top** (Mu & Viswanath 2018): Just remove top 1-3 principal components. Simpler, nearly as effective.
3. **BERT-flow**: Obsolete for modern contrastive-trained models.

Papers:
- "Isotropy Matters: Soft-ZCA Whitening" (ESANN 2025, arxiv 2411.17538)
- "Whitening Sentence Representations" (Su et al., AAAI 2022)
- "All-but-the-Top" (Mu & Viswanath, ICLR 2018)

## 3. Novelty Detection

Beyond cosine similarity:

| Signal | Method | Effort |
|--------|--------|--------|
| Distributional (current) | 1 - mean(top-K cosine) | Done |
| Temporal | weight neighbors by recency: `exp(-λ * age_days)` | Easy |
| Isolation Forest | sklearn IsolationForest on embeddings | Easy |
| NLI-based | Cross-encoder to distinguish entailment from contradiction | Medium |
| Information-theoretic | GMM density estimation, log-likelihood | Medium |
| Lexical | TF-IDF novelty (rare terms) | Easy |

**Key finding**: "AI-based novelty detection in crowdsourced idea spaces" (2023) — the choice of embedding model matters more than the choice of novelty algorithm.

Papers:
- "Novelty Detection: A Perspective from NLP" (Computational Linguistics, 2022)
- "AI-based novelty detection in crowdsourced idea spaces" (Innovation, 2023)
- "Isolation Forest in Novelty Detection" (arxiv 2505.08489)

## 4. Clustering

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| Greedy centroid (current) | Guaranteed tight | O(N²), order-dependent | Small/medium, guaranteed tightness |
| HDBSCAN | No K needed, handles density | 30-70% outlier rate on short text | Exploratory, accepts noise |
| Leiden community detection | Fast, sparse graph, scales | Looser clusters | Large datasets |
| BIRCH | Incremental `partial_fit()` | Less tight | Online/streaming |

**BERTopic insight**: UMAP(n_components=5) before HDBSCAN dramatically reduces outlier rate.

## 5. Hybrid Search

**Convex combination outperforms RRF** (Bruch et al., ACM TOIS 2023):
- Only needs ~40 labeled queries to tune α
- RRF discards score magnitudes (uses only ranks)
- CC generalizes better out-of-domain

**Cross-encoder reranking** adds 5-15% relevance. `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, 3 lines of code).

**Not worth it at amygdala's scale**: SPLADE, ColBERT, ANN indexes (these pay off at >100K docs).

## 6. Ideas to Borrow from Other Libraries

| From | Idea | Value for Amygdala |
|------|------|-------------------|
| txtai | Similarity graph layer on embeddings | Graph-based knowledge exploration |
| LlamaIndex | Sentence-window retrieval | Better embedding quality + context |
| Weaviate | Auto-embed on insert | Convenience for Index class |
| jina-embeddings-v3 | Task-specific LoRA adapters | Different "views" for search vs clustering |

## Summary: Prioritized Recommendations

### Highest Impact, Easiest
1. Switch to ModernBERT-embed-base or nomic-embed-text-v1.5 with Matryoshka truncation to 256d
2. Add cross-encoder reranking (3 lines)
3. Switch RRF → convex combination

### Medium Effort, High Impact
4. NLI-based novelty refinement
5. Temporal decay in novelty scoring
6. Incremental clustering mode
7. Autoresearch loop for parameter optimization

### Longer-Term
8. Similarity graph layer
9. Multiple embedding backends with Matryoshka-aware config
10. Isolation Forest / GMM novelty modes
