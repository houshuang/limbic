"""Experiment 19: NFCorpus Search Quality Evaluation

Evaluates amygdala's search methods (vector, FTS5, hybrid RRF, cross-encoder reranking)
on NFCorpus, a medical domain retrieval benchmark from BEIR.

NFCorpus: 3,633 medical documents, 323 test queries, graded relevance (0/1/2).
Medical domain tests hybrid search where exact terminology (drug names, conditions)
matters alongside semantic understanding.

Metrics: nDCG@10 (graded), MRR@10 (binary: rel>=1), Recall@10 (binary: rel>=1).
"""

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex, FTS5Index, HybridSearch, rerank

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
K = 10


# ── Metrics ───────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    relevances = [qrel.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if qrel.get(doc_id, 0.0) >= 1:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    relevant = {did for did, score in qrel.items() if score >= 1}
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return found / len(relevant)


# ── FTS5 query sanitization ───────────────────────────────────────────

def sanitize_fts5_query(text: str) -> str:
    words = re.findall(r'[a-zA-Z]{2,}', text)
    if not words:
        return text
    return " OR ".join(f'"{w}"' for w in words)


# ── Data loading ──────────────────────────────────────────────────────

def load_nfcorpus():
    from datasets import load_dataset

    print("Loading NFCorpus from HuggingFace...")
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    queries_ds = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels")["test"]

    corpus = {}
    for row in corpus_ds:
        text = row["title"] + " " + row["text"] if row["title"] else row["text"]
        corpus[row["_id"]] = text

    queries = {}
    for row in queries_ds:
        queries[row["_id"]] = row["text"]

    # Build qrels: query_id -> {doc_id: relevance}
    qrels = {}
    for row in qrels_ds:
        qid = row["query-id"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][row["corpus-id"]] = row["score"]

    test_query_ids = sorted(qrels.keys())
    test_queries = {qid: queries[qid] for qid in test_query_ids if qid in queries}

    # Relevance stats
    all_scores = [s for qr in qrels.values() for s in qr.values()]
    score_dist = defaultdict(int)
    for s in all_scores:
        score_dist[s] += 1

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Test queries: {len(test_queries)}")
    print(f"  Relevance judgments: {len(all_scores)} (score dist: {dict(sorted(score_dist.items()))})")
    avg_rels = len(all_scores) / len(test_queries) if test_queries else 0
    print(f"  Avg judgments per query: {avg_rels:.1f}")
    return corpus, test_queries, qrels


# ── Evaluation helpers ────────────────────────────────────────────────

def evaluate_method(query_ids, retrieved_per_query, qrels):
    metrics = {"ndcg": [], "mrr": [], "recall": []}
    for qid, retrieved in zip(query_ids, retrieved_per_query):
        metrics["ndcg"].append(ndcg_at_k(retrieved, qrels[qid], K))
        metrics["mrr"].append(mrr_at_k(retrieved, qrels[qid], K))
        metrics["recall"].append(recall_at_k(retrieved, qrels[qid], K))
    return metrics


def summarize_metrics(metrics):
    return {
        "nDCG@10": float(np.mean(metrics["ndcg"])),
        "MRR@10": float(np.mean(metrics["mrr"])),
        "Recall@10": float(np.mean(metrics["recall"])),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    corpus, queries, qrels = load_nfcorpus()

    corpus_ids = sorted(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    # Embed corpus
    print(f"\nEmbedding {len(corpus_texts)} documents with {MODEL_NAME}...")
    model = EmbeddingModel(model_name=MODEL_NAME)
    t0 = time.perf_counter()
    corpus_embeddings = model.embed_batch(corpus_texts)
    embed_time = time.perf_counter() - t0
    print(f"  Done in {embed_time:.1f}s ({len(corpus_texts)/embed_time:.0f} docs/s)")
    print(f"  Embedding shape: {corpus_embeddings.shape}")

    # Build vector index
    print("\nBuilding VectorIndex...")
    vi = VectorIndex()
    vi.add(corpus_ids, corpus_embeddings)
    print(f"  {vi.size} vectors indexed")

    # Build FTS5 index
    print("Building FTS5Index...")
    fts = FTS5Index()
    fts.add_batch([{"id": cid, "content": corpus[cid]} for cid in corpus_ids])
    print(f"  {fts.size} documents indexed")

    # Build hybrid search
    hybrid = HybridSearch(vector_index=vi, fts_index=fts)

    # Embed queries
    query_ids = sorted(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    print(f"\nEmbedding {len(query_texts)} queries...")
    query_embeddings = model.embed_batch(query_texts)

    # Corpus content lookup (for cross-encoder reranking)
    corpus_content = {cid: corpus[cid] for cid in corpus_ids}

    def attach_content(results):
        """Ensure all results have content for cross-encoder reranking."""
        for r in results:
            if not r.content:
                r.content = corpus_content.get(r.id, "")
        return results

    methods = {}

    # ── Vector-only ──────────────────────────────────────────────────
    print("\nEvaluating vector-only search...")
    t0 = time.perf_counter()
    vec_retrieved = []
    vec_results_full = []
    for i, qid in enumerate(query_ids):
        results = vi.search(query_embeddings[i], limit=K)
        vec_results_full.append(results)
        vec_retrieved.append([r.id for r in results])
    vec_latency = (time.perf_counter() - t0) / len(query_ids)
    vec_metrics = evaluate_method(query_ids, vec_retrieved, qrels)
    methods["vector"] = {**summarize_metrics(vec_metrics), "avg_query_ms": vec_latency * 1000}

    # ── FTS5-only ────────────────────────────────────────────────────
    print("Evaluating FTS5-only search...")
    t0 = time.perf_counter()
    fts_retrieved = []
    fts_failures = 0
    for i, qid in enumerate(query_ids):
        safe_query = sanitize_fts5_query(queries[qid])
        results = fts.search(safe_query, limit=K)
        if not results:
            fts_failures += 1
        fts_retrieved.append([r.id for r in results])
    fts_latency = (time.perf_counter() - t0) / len(query_ids)
    fts_metrics = evaluate_method(query_ids, fts_retrieved, qrels)
    methods["fts5"] = {**summarize_metrics(fts_metrics), "avg_query_ms": fts_latency * 1000, "empty_results": fts_failures}

    # ── Hybrid RRF ───────────────────────────────────────────────────
    print("Evaluating hybrid RRF search...")
    t0 = time.perf_counter()
    hyb_retrieved = []
    hyb_results_full = []
    for i, qid in enumerate(query_ids):
        safe_query = sanitize_fts5_query(queries[qid])
        results = hybrid.search(query_embeddings[i], safe_query, limit=K)
        hyb_results_full.append(results)
        hyb_retrieved.append([r.id for r in results])
    hyb_latency = (time.perf_counter() - t0) / len(query_ids)
    hyb_metrics = evaluate_method(query_ids, hyb_retrieved, qrels)
    methods["hybrid"] = {**summarize_metrics(hyb_metrics), "avg_query_ms": hyb_latency * 1000}

    # ── Vector + cross-encoder reranking ─────────────────────────────
    print("Evaluating vector + cross-encoder reranking...")
    # Retrieve more candidates for reranking
    t0 = time.perf_counter()
    vec_rerank_retrieved = []
    for i, qid in enumerate(query_ids):
        candidates = vi.search(query_embeddings[i], limit=50)
        attach_content(candidates)
        reranked = rerank(queries[qid], candidates, top_n=50)
        vec_rerank_retrieved.append([r.id for r in reranked[:K]])
    vec_rerank_latency = (time.perf_counter() - t0) / len(query_ids)
    vec_rerank_metrics = evaluate_method(query_ids, vec_rerank_retrieved, qrels)
    methods["vector+rerank"] = {**summarize_metrics(vec_rerank_metrics), "avg_query_ms": vec_rerank_latency * 1000}

    # ── Hybrid + cross-encoder reranking ─────────────────────────────
    print("Evaluating hybrid + cross-encoder reranking...")
    t0 = time.perf_counter()
    hyb_rerank_retrieved = []
    for i, qid in enumerate(query_ids):
        safe_query = sanitize_fts5_query(queries[qid])
        candidates = hybrid.search(query_embeddings[i], safe_query, limit=50)
        attach_content(candidates)
        reranked = rerank(queries[qid], candidates, top_n=50)
        hyb_rerank_retrieved.append([r.id for r in reranked[:K]])
    hyb_rerank_latency = (time.perf_counter() - t0) / len(query_ids)
    hyb_rerank_metrics = evaluate_method(query_ids, hyb_rerank_retrieved, qrels)
    methods["hybrid+rerank"] = {**summarize_metrics(hyb_rerank_metrics), "avg_query_ms": hyb_rerank_latency * 1000}

    # ── Per-query analysis: vector vs FTS5 ───────────────────────────
    print("\nAnalyzing per-query differences (vector vs FTS5)...")

    vec_wins = []
    fts_wins = []
    ties = 0
    for i, qid in enumerate(query_ids):
        v = vec_metrics["ndcg"][i]
        f = fts_metrics["ndcg"][i]
        delta = v - f
        if delta > 0.01:
            vec_wins.append((qid, queries[qid], v, f, delta))
        elif delta < -0.01:
            fts_wins.append((qid, queries[qid], v, f, delta))
        else:
            ties += 1

    vec_wins.sort(key=lambda x: -x[4])
    fts_wins.sort(key=lambda x: x[4])

    # Characterize differences
    def avg_query_length(items):
        if not items:
            return 0
        return np.mean([len(q.split()) for _, q, *_ in items])

    def avg_unique_terms(items):
        if not items:
            return 0
        return np.mean([len(set(q.lower().split())) for _, q, *_ in items])

    # ── Results table ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Experiment 19: NFCorpus Search Quality Evaluation")
    print(f"Model: {MODEL_NAME}  |  Corpus: {len(corpus)} docs  |  Queries: {len(queries)}")
    print(f"Relevance: graded (0/1/2)  |  nDCG uses graded, MRR/Recall use binary (>=1)")
    print("=" * 80)

    header = f"{'Method':<18s} {'nDCG@10':>9s} {'MRR@10':>9s} {'Recall@10':>11s} {'ms/query':>10s}"
    print(header)
    print("-" * 80)
    for name, m in methods.items():
        row = f"{name:<18s} {m['nDCG@10']:>9.4f} {m['MRR@10']:>9.4f} {m['Recall@10']:>11.4f} {m['avg_query_ms']:>10.2f}"
        print(row)

    print("-" * 80)
    best_method = max(methods, key=lambda m: methods[m]["nDCG@10"])
    print(f"Best nDCG@10: {best_method} ({methods[best_method]['nDCG@10']:.4f})")

    if fts_failures > 0:
        print(f"\nNote: FTS5 returned empty results for {fts_failures}/{len(queries)} queries")

    # ── Vector vs FTS5 analysis ──────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("Per-query nDCG@10 comparison: Vector vs FTS5")
    print(f"  Vector wins: {len(vec_wins)}  |  FTS5 wins: {len(fts_wins)}  |  Ties (<0.01): {ties}")

    print(f"\n  Avg query length (words):")
    print(f"    Vector-wins queries: {avg_query_length(vec_wins):.1f}")
    print(f"    FTS5-wins queries:   {avg_query_length(fts_wins):.1f}")
    print(f"  Avg unique terms:")
    print(f"    Vector-wins queries: {avg_unique_terms(vec_wins):.1f}")
    print(f"    FTS5-wins queries:   {avg_unique_terms(fts_wins):.1f}")

    print(f"\n  Top 5 queries where VECTOR beats FTS5:")
    for qid, query, v, f, delta in vec_wins[:5]:
        print(f"    [{qid}] nDCG: vec={v:.3f} fts={f:.3f} (+{delta:.3f})")
        print(f"      Query: {query[:100]}")

    print(f"\n  Top 5 queries where FTS5 beats VECTOR:")
    for qid, query, v, f, delta in fts_wins[:5]:
        print(f"    [{qid}] nDCG: vec={v:.3f} fts={f:.3f} ({delta:.3f})")
        print(f"      Query: {query[:100]}")

    # ── Reranking lift analysis ──────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("Reranking lift:")
    vec_base = methods["vector"]["nDCG@10"]
    vec_rr = methods["vector+rerank"]["nDCG@10"]
    hyb_base = methods["hybrid"]["nDCG@10"]
    hyb_rr = methods["hybrid+rerank"]["nDCG@10"]
    print(f"  Vector: {vec_base:.4f} -> {vec_rr:.4f} ({(vec_rr-vec_base)/vec_base*100:+.1f}%)")
    print(f"  Hybrid: {hyb_base:.4f} -> {hyb_rr:.4f} ({(hyb_rr-hyb_base)/hyb_base*100:+.1f}%)")

    # ── Save results ──────────────────────────────────────────────────
    analysis = {
        "vector_vs_fts5": {
            "vector_wins": len(vec_wins),
            "fts5_wins": len(fts_wins),
            "ties": ties,
            "avg_query_len_vector_wins": round(avg_query_length(vec_wins), 1),
            "avg_query_len_fts5_wins": round(avg_query_length(fts_wins), 1),
            "top5_vector_wins": [
                {"qid": qid, "query": query[:200], "vec_ndcg": round(v, 4), "fts_ndcg": round(f, 4)}
                for qid, query, v, f, _ in vec_wins[:5]
            ],
            "top5_fts5_wins": [
                {"qid": qid, "query": query[:200], "vec_ndcg": round(v, 4), "fts_ndcg": round(f, 4)}
                for qid, query, v, f, _ in fts_wins[:5]
            ],
        },
        "reranking_lift": {
            "vector_base_ndcg": round(vec_base, 4),
            "vector_reranked_ndcg": round(vec_rr, 4),
            "vector_lift_pct": round((vec_rr - vec_base) / vec_base * 100, 1),
            "hybrid_base_ndcg": round(hyb_base, 4),
            "hybrid_reranked_ndcg": round(hyb_rr, 4),
            "hybrid_lift_pct": round((hyb_rr - hyb_base) / hyb_base * 100, 1),
        },
    }

    results = {
        "experiment": 19,
        "benchmark": "NFCorpus (BEIR)",
        "model": MODEL_NAME,
        "corpus_size": len(corpus),
        "num_queries": len(queries),
        "k": K,
        "embedding_dim": int(corpus_embeddings.shape[1]),
        "embed_time_s": round(embed_time, 1),
        "relevance_scale": "0/1/2 (graded; binary threshold >= 1 for MRR/Recall)",
        "methods": methods,
        "analysis": analysis,
    }
    out_path = RESULTS_DIR / "exp19_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
