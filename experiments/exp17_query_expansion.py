"""Experiment 17: Pseudo-Relevance Feedback (PRF) Query Expansion

On SciFact, vector search nDCG@10=0.484 while FTS5 gets 0.638.
PRF is a cheap technique: embed query -> take top-K results -> average
their embeddings with the original query -> re-search. This could close
the gap between vector and FTS5.

Configurations tested:
  a. Baseline vector search (no expansion)
  b. PRF n_feedback=3, alpha=0.7 (Rocchio classic)
  c. PRF n_feedback=5, alpha=0.7
  d. PRF n_feedback=3, alpha=0.5
  e. PRF n_feedback=3, alpha=0.9
  f. Best PRF + hybrid search (PRF-expanded vector + FTS5 via RRF)

Also analyzes per-query wins/losses to understand when PRF helps or hurts.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex, FTS5Index, HybridSearch

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
K = 10


# -- Metrics (same as eval_scifact_search.py) ---------------------------------

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
        if qrel.get(doc_id, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    relevant = {did for did, score in qrel.items() if score > 0}
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return found / len(relevant)


# -- Data loading --------------------------------------------------------------

def load_scifact():
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

    qrels = {}
    for row in qrels_ds:
        qid = row["query-id"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][row["corpus-id"]] = row["score"]

    test_query_ids = sorted(qrels.keys())
    test_queries = {qid: queries[qid] for qid in test_query_ids if qid in queries}

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Test queries: {len(test_queries)} (with {sum(len(v) for v in qrels.values())} relevance judgments)")
    return corpus, test_queries, qrels


# -- PRF query expansion ------------------------------------------------------

def expand_query(query_vec: np.ndarray, corpus_embeddings: np.ndarray,
                 id_to_idx: dict[str, int], index: VectorIndex,
                 n_feedback: int = 3, alpha: float = 0.7) -> np.ndarray:
    """Pseudo-relevance feedback: blend original query with top-N result embeddings."""
    results = index.search(query_vec, limit=n_feedback)
    if not results:
        return query_vec
    feedback_indices = [id_to_idx[r.id] for r in results]
    feedback_vecs = corpus_embeddings[feedback_indices]
    feedback_centroid = np.mean(feedback_vecs, axis=0)
    expanded = alpha * query_vec + (1 - alpha) * feedback_centroid
    norm = np.linalg.norm(expanded)
    if norm > 0:
        expanded = expanded / norm
    return expanded


# -- Evaluation helper ---------------------------------------------------------

def evaluate_search(query_ids, retrieve_fn, qrels):
    """Run retrieval for all queries and compute metrics."""
    metrics = {"ndcg": [], "mrr": [], "recall": []}
    for qid in query_ids:
        retrieved = retrieve_fn(qid)
        metrics["ndcg"].append(ndcg_at_k(retrieved, qrels[qid], K))
        metrics["mrr"].append(mrr_at_k(retrieved, qrels[qid], K))
        metrics["recall"].append(recall_at_k(retrieved, qrels[qid], K))
    return {
        "nDCG@10": float(np.mean(metrics["ndcg"])),
        "MRR@10": float(np.mean(metrics["mrr"])),
        "Recall@10": float(np.mean(metrics["recall"])),
    }, metrics


# -- Main ----------------------------------------------------------------------

def main():
    corpus, queries, qrels = load_scifact()

    corpus_ids = sorted(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    # Embed corpus
    print(f"\nEmbedding {len(corpus_texts)} documents with {MODEL_NAME}...")
    model = EmbeddingModel(model_name=MODEL_NAME)
    t0 = time.perf_counter()
    corpus_embeddings = model.embed_batch(corpus_texts)
    embed_time = time.perf_counter() - t0
    print(f"  Done in {embed_time:.1f}s ({len(corpus_texts)/embed_time:.0f} docs/s)")

    # Build vector index
    print("\nBuilding VectorIndex...")
    vi = VectorIndex()
    vi.add(corpus_ids, corpus_embeddings)

    # Build FTS5 index
    print("Building FTS5Index...")
    fts = FTS5Index()
    fts.add_batch([{"id": cid, "content": corpus[cid]} for cid in corpus_ids])

    # Embed queries
    query_ids = sorted(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    print(f"\nEmbedding {len(query_texts)} queries...")
    query_embeddings = model.embed_batch(query_texts)
    qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}

    # -- Baseline: vector search -----------------------------------------------
    print("\n--- Evaluating baseline vector search ---")
    t0 = time.perf_counter()
    baseline_agg, baseline_per = evaluate_search(
        query_ids,
        lambda qid: [r.id for r in vi.search(query_embeddings[qid_to_idx[qid]], limit=K)],
        qrels
    )
    baseline_ms = (time.perf_counter() - t0) / len(query_ids) * 1000
    baseline_agg["avg_query_ms"] = baseline_ms

    # -- Baseline: FTS5 search -------------------------------------------------
    # Pass raw query text; FTS5Index._sanitize_query handles it internally.
    print("--- Evaluating baseline FTS5 search ---")
    t0 = time.perf_counter()
    fts_agg, fts_per = evaluate_search(
        query_ids,
        lambda qid: [r.id for r in fts.search(queries[qid], limit=K)],
        qrels
    )
    fts_ms = (time.perf_counter() - t0) / len(query_ids) * 1000
    fts_agg["avg_query_ms"] = fts_ms

    # -- PRF configurations ----------------------------------------------------
    prf_configs = [
        ("PRF(3,0.7)", 3, 0.7),
        ("PRF(5,0.7)", 5, 0.7),
        ("PRF(3,0.5)", 3, 0.5),
        ("PRF(3,0.9)", 3, 0.9),
    ]

    prf_results = {}
    prf_per_query = {}

    for name, n_fb, alpha in prf_configs:
        print(f"--- Evaluating {name} ---")
        t0 = time.perf_counter()

        def make_retrieve(n_fb=n_fb, alpha=alpha):
            def retrieve(qid):
                qvec = query_embeddings[qid_to_idx[qid]]
                expanded = expand_query(qvec, corpus_embeddings, id_to_idx, vi,
                                        n_feedback=n_fb, alpha=alpha)
                return [r.id for r in vi.search(expanded, limit=K)]
            return retrieve

        agg, per = evaluate_search(query_ids, make_retrieve(), qrels)
        prf_ms = (time.perf_counter() - t0) / len(query_ids) * 1000
        agg["avg_query_ms"] = prf_ms
        agg["n_feedback"] = n_fb
        agg["alpha"] = alpha
        prf_results[name] = agg
        prf_per_query[name] = per

    # -- Find best PRF config for hybrid test ----------------------------------
    best_prf_name = max(prf_results, key=lambda n: prf_results[n]["nDCG@10"])
    best_n_fb = prf_results[best_prf_name]["n_feedback"]
    best_alpha = prf_results[best_prf_name]["alpha"]

    # -- Baseline hybrid (no PRF) for comparison -------------------------------
    print("--- Evaluating baseline hybrid (no PRF) ---")
    hybrid = HybridSearch(vector_index=vi, fts_index=fts)
    t0 = time.perf_counter()
    hybrid_base_agg, hybrid_base_per = evaluate_search(
        query_ids,
        lambda qid: [r.id for r in hybrid.search(
            query_embeddings[qid_to_idx[qid]],
            queries[qid], limit=K)],
        qrels
    )
    hybrid_base_ms = (time.perf_counter() - t0) / len(query_ids) * 1000
    hybrid_base_agg["avg_query_ms"] = hybrid_base_ms

    # -- PRF + Hybrid: expand vector part, keep FTS5, RRF ----------------------
    print(f"--- Evaluating PRF({best_n_fb},{best_alpha}) + Hybrid ---")

    # Build a new VectorIndex that we can search with expanded queries
    # We reuse the same index but expand each query before passing to hybrid
    t0 = time.perf_counter()

    def prf_hybrid_retrieve(qid):
        qvec = query_embeddings[qid_to_idx[qid]]
        expanded = expand_query(qvec, corpus_embeddings, id_to_idx, vi,
                                n_feedback=best_n_fb, alpha=best_alpha)
        return [r.id for r in hybrid.search(expanded, queries[qid], limit=K)]

    prf_hybrid_agg, prf_hybrid_per = evaluate_search(
        query_ids, prf_hybrid_retrieve, qrels
    )
    prf_hybrid_ms = (time.perf_counter() - t0) / len(query_ids) * 1000
    prf_hybrid_agg["avg_query_ms"] = prf_hybrid_ms

    # -- Per-query analysis: where does PRF help/hurt? -------------------------
    best_prf_per = prf_per_query[best_prf_name]
    helped = []
    hurt = []
    for i, qid in enumerate(query_ids):
        base_ndcg = baseline_per["ndcg"][i]
        prf_ndcg = best_prf_per["ndcg"][i]
        delta = prf_ndcg - base_ndcg
        if abs(delta) > 1e-9:
            entry = {
                "qid": qid,
                "query": queries[qid][:100],
                "baseline_ndcg": round(base_ndcg, 4),
                "prf_ndcg": round(prf_ndcg, 4),
                "delta": round(delta, 4),
            }
            if delta > 0:
                helped.append(entry)
            else:
                hurt.append(entry)

    helped.sort(key=lambda x: x["delta"], reverse=True)
    hurt.sort(key=lambda x: x["delta"])

    # -- Print results ---------------------------------------------------------
    all_methods = {
        "vector_baseline": baseline_agg,
        "fts5_baseline": fts_agg,
    }
    all_methods.update(prf_results)
    all_methods["hybrid_baseline"] = hybrid_base_agg
    all_methods[f"hybrid+{best_prf_name}"] = prf_hybrid_agg

    print("\n" + "=" * 80)
    print("Experiment 17: Pseudo-Relevance Feedback (PRF) Query Expansion")
    print(f"Model: {MODEL_NAME}  |  Corpus: {len(corpus)} docs  |  Queries: {len(queries)}")
    print("=" * 80)

    header = f"{'Method':<22s} {'nDCG@10':>9s} {'MRR@10':>9s} {'Recall@10':>11s} {'ms/query':>10s}"
    print(header)
    print("-" * 80)
    for name, m in all_methods.items():
        row = f"{name:<22s} {m['nDCG@10']:>9.4f} {m['MRR@10']:>9.4f} {m['Recall@10']:>11.4f} {m['avg_query_ms']:>10.2f}"
        print(row)

    print("-" * 80)

    # Highlight deltas from baseline
    print(f"\nDeltas from vector baseline (nDCG@10 = {baseline_agg['nDCG@10']:.4f}):")
    for name, m in all_methods.items():
        if name == "vector_baseline" or name == "fts5_baseline":
            continue
        delta = m["nDCG@10"] - baseline_agg["nDCG@10"]
        pct = delta / baseline_agg["nDCG@10"] * 100
        print(f"  {name:<22s}: {delta:+.4f} ({pct:+.1f}%)")

    fts_gap = fts_agg["nDCG@10"] - baseline_agg["nDCG@10"]
    best_prf_delta = prf_results[best_prf_name]["nDCG@10"] - baseline_agg["nDCG@10"]
    gap_closed = best_prf_delta / fts_gap * 100 if fts_gap > 0 else 0

    print(f"\nVector-FTS5 gap: {fts_gap:.4f}")
    print(f"Best PRF improvement: {best_prf_delta:+.4f} ({best_prf_name})")
    print(f"Gap closed by PRF: {gap_closed:.1f}%")

    # Per-query analysis
    n_helped = len(helped)
    n_hurt = len(hurt)
    n_unchanged = len(query_ids) - n_helped - n_hurt
    print(f"\nPer-query analysis ({best_prf_name} vs baseline):")
    print(f"  Helped: {n_helped}  |  Hurt: {n_hurt}  |  Unchanged: {n_unchanged}")

    if helped:
        print(f"\n  Top 5 queries where PRF helped most:")
        for e in helped[:5]:
            print(f"    {e['qid']}: {e['baseline_ndcg']:.4f} -> {e['prf_ndcg']:.4f} ({e['delta']:+.4f}) | {e['query']}")

    if hurt:
        print(f"\n  Top 5 queries where PRF hurt most:")
        for e in hurt[:5]:
            print(f"    {e['qid']}: {e['baseline_ndcg']:.4f} -> {e['prf_ndcg']:.4f} ({e['delta']:+.4f}) | {e['query']}")

    # -- Save results ----------------------------------------------------------
    results = {
        "experiment": "exp17_query_expansion",
        "description": "Pseudo-relevance feedback (PRF) query expansion on SciFact",
        "model": MODEL_NAME,
        "corpus_size": len(corpus),
        "num_queries": len(queries),
        "k": K,
        "methods": all_methods,
        "best_prf_config": best_prf_name,
        "vector_fts5_gap": round(fts_gap, 4),
        "best_prf_improvement": round(best_prf_delta, 4),
        "gap_closed_pct": round(gap_closed, 1),
        "per_query_analysis": {
            "helped": n_helped,
            "hurt": n_hurt,
            "unchanged": n_unchanged,
            "top_helped": helped[:10],
            "top_hurt": hurt[:10],
        },
    }
    out_path = RESULTS_DIR / "exp17_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
