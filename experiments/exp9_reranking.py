"""Experiment 9: Cross-encoder reranking benchmark on SciFact.

Measures the relevance boost from applying cross-encoder reranking on top of
vector, FTS5, and hybrid search results. Uses the same SciFact setup as
eval_scifact_search.py for direct comparison.

Protocol:
  1. Build vector, FTS5, and hybrid indices (same as baseline eval)
  2. For each query, retrieve top-50 candidates from each method
  3. Apply cross-encoder reranking, take top-10
  4. Measure nDCG@10, MRR@10, Recall@10 for all 6 variants
  5. Also measure reranking latency

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, ~5ms/pair)

Note: The amygdala rerank() function loads the model per call. For benchmarking
300 queries x 3 methods, we load the model once and score directly, then
validate equivalence with rerank() on a sample.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, VectorIndex, FTS5Index, HybridSearch
from amygdala.search import Result, rerank

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K = 10
FETCH_K = 50  # retrieve more candidates for reranking


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
        if qrel.get(doc_id, 0.0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], qrel: dict[str, float], k: int) -> float:
    relevant = {did for did, score in qrel.items() if score > 0}
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return found / len(relevant)


# ── Data loading (reused from eval_scifact_search.py) ────────────────

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


# ── Reranking helper ─────────────────────────────────────────────────

def rerank_with_ce(ce, query: str, results: list[Result]) -> list[Result]:
    """Rerank results using a pre-loaded CrossEncoder model."""
    if not results:
        return results
    pairs = [(query, r.content) for r in results]
    scores = ce.predict(pairs)
    reranked = []
    for r, score in zip(results, scores):
        reranked.append(Result(id=r.id, score=float(score), content=r.content,
                               metadata=r.metadata, source=r.source))
    reranked.sort(key=lambda r: r.score, reverse=True)
    return reranked


def populate_content(results: list[Result], corpus: dict[str, str]) -> list[Result]:
    """Ensure all results have content populated (vector search omits it)."""
    return [Result(id=r.id, score=r.score, content=corpus.get(r.id, r.content),
                   metadata=r.metadata, source=r.source) for r in results]


# ── Eval helper ──────────────────────────────────────────────────────

def evaluate_method(query_ids, all_retrieved, qrels):
    """Compute aggregate metrics for a set of retrieved results."""
    metrics = {"ndcg": [], "mrr": [], "recall": []}
    for qid, retrieved in zip(query_ids, all_retrieved):
        doc_ids = [r.id for r in retrieved[:K]]
        metrics["ndcg"].append(ndcg_at_k(doc_ids, qrels[qid], K))
        metrics["mrr"].append(mrr_at_k(doc_ids, qrels[qid], K))
        metrics["recall"].append(recall_at_k(doc_ids, qrels[qid], K))
    return {
        "nDCG@10": float(np.mean(metrics["ndcg"])),
        "MRR@10": float(np.mean(metrics["mrr"])),
        "Recall@10": float(np.mean(metrics["recall"])),
    }


# ── Main ─────────────────────────────────────────────────────────────

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
    print(f"  Done in {embed_time:.1f}s")

    # Build indices
    print("\nBuilding VectorIndex...")
    vi = VectorIndex()
    vi.add(corpus_ids, corpus_embeddings)

    print("Building FTS5Index...")
    fts = FTS5Index()
    fts.add_batch([{"id": cid, "content": corpus[cid]} for cid in corpus_ids])

    hybrid = HybridSearch(vector_index=vi, fts_index=fts)

    # Embed queries
    query_ids = sorted(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    print(f"Embedding {len(query_texts)} queries...")
    query_embeddings = model.embed_batch(query_texts)

    # Load cross-encoder once
    print(f"\nLoading cross-encoder: {CE_MODEL_NAME}...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CE_MODEL_NAME)
    print("  Cross-encoder ready")

    # ── Retrieve candidates ──────────────────────────────────────────

    print(f"\nRetrieving top-{FETCH_K} candidates per query per method...")

    # Vector search (top-50, populate content from corpus)
    vec_results_all = []
    for i, qid in enumerate(query_ids):
        results = vi.search(query_embeddings[i], limit=FETCH_K)
        results = populate_content(results, corpus)
        vec_results_all.append(results)

    # FTS5 search (top-50, already has content)
    # Pass raw query text — FTS5Index.search() sanitizes internally
    fts_results_all = []
    for i, qid in enumerate(query_ids):
        results = fts.search(queries[qid], limit=FETCH_K)
        fts_results_all.append(results)

    # Hybrid search (top-50, populate content for any missing)
    # HybridSearch passes query_text to FTS5Index.search() which sanitizes
    hyb_results_all = []
    for i, qid in enumerate(query_ids):
        results = hybrid.search(query_embeddings[i], queries[qid], limit=FETCH_K)
        results = populate_content(results, corpus)
        hyb_results_all.append(results)

    # ── Baseline metrics (top-10 from unreranked results) ────────────

    print("\nComputing baseline metrics (top-10 unreranked)...")
    methods = {}
    methods["vector"] = evaluate_method(query_ids, vec_results_all, qrels)
    methods["fts5"] = evaluate_method(query_ids, fts_results_all, qrels)
    methods["hybrid"] = evaluate_method(query_ids, hyb_results_all, qrels)

    # ── Reranked metrics ─────────────────────────────────────────────

    print("Reranking vector results...")
    t0 = time.perf_counter()
    vec_reranked_all = []
    for i, qid in enumerate(query_ids):
        reranked = rerank_with_ce(ce, queries[qid], vec_results_all[i])
        vec_reranked_all.append(reranked)
    vec_rerank_time = time.perf_counter() - t0
    vec_rerank_ms = vec_rerank_time / len(query_ids) * 1000

    print("Reranking FTS5 results...")
    t0 = time.perf_counter()
    fts_reranked_all = []
    for i, qid in enumerate(query_ids):
        reranked = rerank_with_ce(ce, queries[qid], fts_results_all[i])
        fts_reranked_all.append(reranked)
    fts_rerank_time = time.perf_counter() - t0
    fts_rerank_ms = fts_rerank_time / len(query_ids) * 1000

    print("Reranking hybrid results...")
    t0 = time.perf_counter()
    hyb_reranked_all = []
    for i, qid in enumerate(query_ids):
        reranked = rerank_with_ce(ce, queries[qid], hyb_results_all[i])
        hyb_reranked_all.append(reranked)
    hyb_rerank_time = time.perf_counter() - t0
    hyb_rerank_ms = hyb_rerank_time / len(query_ids) * 1000

    methods["vector+rerank"] = evaluate_method(query_ids, vec_reranked_all, qrels)
    methods["vector+rerank"]["rerank_ms_per_query"] = round(vec_rerank_ms, 1)

    methods["fts5+rerank"] = evaluate_method(query_ids, fts_reranked_all, qrels)
    methods["fts5+rerank"]["rerank_ms_per_query"] = round(fts_rerank_ms, 1)

    methods["hybrid+rerank"] = evaluate_method(query_ids, hyb_reranked_all, qrels)
    methods["hybrid+rerank"]["rerank_ms_per_query"] = round(hyb_rerank_ms, 1)

    # ── Validate against amygdala's rerank() function ────────────────

    print("\nValidating amygdala rerank() on first query...")
    sample_reranked = rerank(queries[query_ids[0]], vec_results_all[0],
                             model_name=CE_MODEL_NAME)
    sample_ids_func = [r.id for r in sample_reranked[:K]]
    sample_ids_direct = [r.id for r in vec_reranked_all[0][:K]]
    if sample_ids_func == sample_ids_direct:
        print("  rerank() matches direct CE scoring")
    else:
        print(f"  WARNING: rerank() differs from direct scoring")
        print(f"    rerank():  {sample_ids_func[:5]}...")
        print(f"    direct CE: {sample_ids_direct[:5]}...")

    # ── Per-query analysis: where does reranking help most? ──────────

    vec_ndcg_base = [ndcg_at_k([r.id for r in vec_results_all[i][:K]], qrels[qid], K)
                     for i, qid in enumerate(query_ids)]
    vec_ndcg_rr = [ndcg_at_k([r.id for r in vec_reranked_all[i][:K]], qrels[qid], K)
                   for i, qid in enumerate(query_ids)]
    hyb_ndcg_base = [ndcg_at_k([r.id for r in hyb_results_all[i][:K]], qrels[qid], K)
                     for i, qid in enumerate(query_ids)]
    hyb_ndcg_rr = [ndcg_at_k([r.id for r in hyb_reranked_all[i][:K]], qrels[qid], K)
                   for i, qid in enumerate(query_ids)]

    vec_improvements = sum(1 for b, r in zip(vec_ndcg_base, vec_ndcg_rr) if r > b + 1e-9)
    vec_degradations = sum(1 for b, r in zip(vec_ndcg_base, vec_ndcg_rr) if b > r + 1e-9)
    hyb_improvements = sum(1 for b, r in zip(hyb_ndcg_base, hyb_ndcg_rr) if r > b + 1e-9)
    hyb_degradations = sum(1 for b, r in zip(hyb_ndcg_base, hyb_ndcg_rr) if b > r + 1e-9)

    # Find best overall method
    best_method = max(methods, key=lambda m: methods[m]["nDCG@10"])

    # ── Results table ────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("Experiment 9: Cross-Encoder Reranking Benchmark (SciFact)")
    print(f"Bi-encoder: {MODEL_NAME}  |  Cross-encoder: {CE_MODEL_NAME}")
    print(f"Corpus: {len(corpus)} docs  |  Queries: {len(queries)}  |  Fetch: top-{FETCH_K} -> rerank -> top-{K}")
    print("=" * 80)

    header = f"{'Method':<18s} {'nDCG@10':>9s} {'MRR@10':>9s} {'Recall@10':>11s} {'Rerank ms':>11s}"
    print(header)
    print("-" * 80)

    for name in ["vector", "vector+rerank", "fts5", "fts5+rerank", "hybrid", "hybrid+rerank"]:
        m = methods[name]
        rerank_ms = m.get("rerank_ms_per_query", "")
        rerank_str = f"{rerank_ms:>11.1f}" if rerank_ms != "" else f"{'—':>11s}"
        row = f"{name:<18s} {m['nDCG@10']:>9.4f} {m['MRR@10']:>9.4f} {m['Recall@10']:>11.4f} {rerank_str}"
        print(row)

        # Print delta for reranked methods
        if "+rerank" in name:
            base_name = name.replace("+rerank", "")
            base = methods[base_name]
            delta_ndcg = m["nDCG@10"] - base["nDCG@10"]
            delta_mrr = m["MRR@10"] - base["MRR@10"]
            delta_recall = m["Recall@10"] - base["Recall@10"]
            sign = lambda x: "+" if x >= 0 else ""
            print(f"  {'delta':<16s} {sign(delta_ndcg)}{delta_ndcg:>8.4f} {sign(delta_mrr)}{delta_mrr:>8.4f} {sign(delta_recall)}{delta_recall:>10.4f}")

    print("-" * 80)
    print(f"Best nDCG@10: {best_method} ({methods[best_method]['nDCG@10']:.4f})")

    print(f"\nPer-query nDCG@10 impact of reranking:")
    print(f"  Vector:  {vec_improvements} improved, {vec_degradations} degraded, "
          f"{len(query_ids) - vec_improvements - vec_degradations} unchanged")
    print(f"  Hybrid:  {hyb_improvements} improved, {hyb_degradations} degraded, "
          f"{len(query_ids) - hyb_improvements - hyb_degradations} unchanged")

    # ── Save results ─────────────────────────────────────────────────

    results = {
        "experiment": "exp9_reranking",
        "benchmark": "SciFact (BEIR)",
        "bi_encoder": MODEL_NAME,
        "cross_encoder": CE_MODEL_NAME,
        "corpus_size": len(corpus),
        "num_queries": len(queries),
        "k": K,
        "fetch_k": FETCH_K,
        "methods": methods,
        "per_query_impact": {
            "vector_rerank": {
                "improved": vec_improvements,
                "degraded": vec_degradations,
                "unchanged": len(query_ids) - vec_improvements - vec_degradations,
            },
            "hybrid_rerank": {
                "improved": hyb_improvements,
                "degraded": hyb_degradations,
                "unchanged": len(query_ids) - hyb_improvements - hyb_degradations,
            },
        },
        "deltas": {
            "vector_rerank_nDCG": round(methods["vector+rerank"]["nDCG@10"] - methods["vector"]["nDCG@10"], 4),
            "fts5_rerank_nDCG": round(methods["fts5+rerank"]["nDCG@10"] - methods["fts5"]["nDCG@10"], 4),
            "hybrid_rerank_nDCG": round(methods["hybrid+rerank"]["nDCG@10"] - methods["hybrid"]["nDCG@10"], 4),
        },
    }

    out_path = RESULTS_DIR / "exp9_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
