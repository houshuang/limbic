"""SciFact Search Quality Evaluation

Evaluates amygdala's search methods (vector, FTS5, hybrid RRF) on SciFact,
an expert-annotated scientific claim retrieval benchmark from BEIR.

SciFact: 5,183 scientific abstracts, 300 test queries (claims), binary relevance.
This directly tests the core use case: retrieving relevant documents
for claim-style queries.

Metrics: nDCG@10, MRR@10, Recall@10 (standard BEIR metrics).
"""

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex, FTS5Index, HybridSearch

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
K = 10


# ── Metrics ───────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def ndcg_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    """Normalized DCG at k."""
    relevances = [qrel.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    """Mean Reciprocal Rank at k."""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if qrel.get(doc_id, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    """Recall at k: fraction of relevant docs found in top-k."""
    relevant = {did for did, score in qrel.items() if score > 0}
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return found / len(relevant)


# ── FTS5 query sanitization ───────────────────────────────────────────

def sanitize_fts5_query(text: str) -> str:
    """Convert natural language text to a safe FTS5 query.

    FTS5 MATCH treats hyphens, quotes, parentheses etc. as operators.
    We extract alphanumeric words and join them with spaces (implicit OR
    in FTS5 when using column match).
    """
    words = re.findall(r'[a-zA-Z]{2,}', text)
    if not words:
        return text
    # Quote each word and join with OR for broader matching
    return " OR ".join(f'"{w}"' for w in words)


# ── Data loading ──────────────────────────────────────────────────────

def load_scifact():
    """Load SciFact corpus, queries, and test qrels from HuggingFace."""
    from datasets import load_dataset

    print("Loading SciFact from HuggingFace...")
    corpus_ds = load_dataset("mteb/scifact", "corpus")["corpus"]
    queries_ds = load_dataset("mteb/scifact", "queries")["queries"]
    qrels_ds = load_dataset("mteb/scifact", "default")["test"]

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

    # Only keep queries that have test qrels
    test_query_ids = sorted(qrels.keys())
    test_queries = {qid: queries[qid] for qid in test_query_ids if qid in queries}

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Test queries: {len(test_queries)} (with {sum(len(v) for v in qrels.values())} relevance judgments)")
    return corpus, test_queries, qrels


# ── Main ──────────────────────────────────────────────────────────────

def main():
    corpus, queries, qrels = load_scifact()

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

    # ── Evaluate all three methods ────────────────────────────────────
    methods = {}

    # Vector-only
    print("\nEvaluating vector-only search...")
    t0 = time.perf_counter()
    vec_metrics = {"ndcg": [], "mrr": [], "recall": []}
    for i, qid in enumerate(query_ids):
        results = vi.search(query_embeddings[i], limit=K)
        retrieved = [r.id for r in results]
        vec_metrics["ndcg"].append(ndcg_at_k(retrieved, qrels[qid], K))
        vec_metrics["mrr"].append(mrr_at_k(retrieved, qrels[qid], K))
        vec_metrics["recall"].append(recall_at_k(retrieved, qrels[qid], K))
    vec_latency = (time.perf_counter() - t0) / len(query_ids)
    methods["vector"] = {
        "nDCG@10": float(np.mean(vec_metrics["ndcg"])),
        "MRR@10": float(np.mean(vec_metrics["mrr"])),
        "Recall@10": float(np.mean(vec_metrics["recall"])),
        "avg_query_ms": vec_latency * 1000,
    }

    # FTS5-only (with sanitized queries)
    print("Evaluating FTS5-only search...")
    t0 = time.perf_counter()
    fts_metrics = {"ndcg": [], "mrr": [], "recall": []}
    fts_failures = 0
    for i, qid in enumerate(query_ids):
        safe_query = sanitize_fts5_query(queries[qid])
        results = fts.search(safe_query, limit=K)
        if not results:
            fts_failures += 1
        retrieved = [r.id for r in results]
        fts_metrics["ndcg"].append(ndcg_at_k(retrieved, qrels[qid], K))
        fts_metrics["mrr"].append(mrr_at_k(retrieved, qrels[qid], K))
        fts_metrics["recall"].append(recall_at_k(retrieved, qrels[qid], K))
    fts_latency = (time.perf_counter() - t0) / len(query_ids)
    methods["fts5"] = {
        "nDCG@10": float(np.mean(fts_metrics["ndcg"])),
        "MRR@10": float(np.mean(fts_metrics["mrr"])),
        "Recall@10": float(np.mean(fts_metrics["recall"])),
        "avg_query_ms": fts_latency * 1000,
        "empty_results": fts_failures,
    }

    # Hybrid RRF (with sanitized FTS5 queries)
    print("Evaluating hybrid RRF search...")
    t0 = time.perf_counter()
    hyb_metrics = {"ndcg": [], "mrr": [], "recall": []}
    for i, qid in enumerate(query_ids):
        safe_query = sanitize_fts5_query(queries[qid])
        results = hybrid.search(query_embeddings[i], safe_query, limit=K)
        retrieved = [r.id for r in results]
        hyb_metrics["ndcg"].append(ndcg_at_k(retrieved, qrels[qid], K))
        hyb_metrics["mrr"].append(mrr_at_k(retrieved, qrels[qid], K))
        hyb_metrics["recall"].append(recall_at_k(retrieved, qrels[qid], K))
    hyb_latency = (time.perf_counter() - t0) / len(query_ids)
    methods["hybrid"] = {
        "nDCG@10": float(np.mean(hyb_metrics["ndcg"])),
        "MRR@10": float(np.mean(hyb_metrics["mrr"])),
        "Recall@10": float(np.mean(hyb_metrics["recall"])),
        "avg_query_ms": hyb_latency * 1000,
    }

    # ── Per-query comparison: where hybrid beats vector ───────────────
    hybrid_wins = 0
    vector_wins = 0
    ties = 0
    for i in range(len(query_ids)):
        h = hyb_metrics["ndcg"][i]
        v = vec_metrics["ndcg"][i]
        if h > v + 1e-9:
            hybrid_wins += 1
        elif v > h + 1e-9:
            vector_wins += 1
        else:
            ties += 1

    # ── Results table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SciFact Search Quality Evaluation")
    print(f"Model: {MODEL_NAME}  |  Corpus: {len(corpus)} docs  |  Queries: {len(queries)}")
    print("=" * 70)

    header = f"{'Method':<12s} {'nDCG@10':>9s} {'MRR@10':>9s} {'Recall@10':>11s} {'ms/query':>10s}"
    print(header)
    print("-" * 70)
    for name, m in methods.items():
        row = f"{name:<12s} {m['nDCG@10']:>9.4f} {m['MRR@10']:>9.4f} {m['Recall@10']:>11.4f} {m['avg_query_ms']:>10.2f}"
        print(row)

    print("-" * 70)
    best_method = max(methods, key=lambda m: methods[m]["nDCG@10"])
    print(f"Best nDCG@10: {best_method} ({methods[best_method]['nDCG@10']:.4f})")

    print(f"\nPer-query nDCG@10 comparison (hybrid vs vector):")
    print(f"  Hybrid wins: {hybrid_wins}  |  Vector wins: {vector_wins}  |  Ties: {ties}")

    if methods["fts5"].get("empty_results", 0) > 0:
        print(f"\nNote: FTS5 returned empty results for {methods['fts5']['empty_results']}/{len(queries)} queries")
        print("  (FTS5 match syntax can fail on queries with special characters)")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "benchmark": "SciFact (BEIR)",
        "model": MODEL_NAME,
        "corpus_size": len(corpus),
        "num_queries": len(queries),
        "k": K,
        "embedding_dim": int(corpus_embeddings.shape[1]),
        "embed_time_s": round(embed_time, 1),
        "methods": methods,
        "per_query_comparison": {
            "hybrid_wins": hybrid_wins,
            "vector_wins": vector_wins,
            "ties": ties,
        },
    }
    out_path = RESULTS_DIR / "eval_scifact_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
