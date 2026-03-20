"""Experiment 5: Clustering algorithm comparison on 20 Newsgroups.

Compares amygdala's greedy_centroid_cluster and complete_linkage_cluster
against HDBSCAN, evaluated with V-measure (homogeneity + completeness)
using the human-assigned 20-topic labels as ground truth.

Dataset: 20 Newsgroups (2,000 stratified sample from test split).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel
from amygdala.cluster import greedy_centroid_cluster, complete_linkage_cluster


# ---------------------------------------------------------------------------
# 1. Load and sample data
# ---------------------------------------------------------------------------

def load_data(n_sample=2000, seed=42):
    """Load 20 Newsgroups test set, stratified sample of n_sample docs."""
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
    )
    texts = data.data
    labels = data.target
    target_names = data.target_names

    # Stratified sample
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    per_class = n_sample // len(unique_labels)

    sampled_idx = []
    for lbl in unique_labels:
        class_idx = np.where(labels == lbl)[0]
        n_take = min(per_class, len(class_idx))
        chosen = rng.choice(class_idx, size=n_take, replace=False)
        sampled_idx.extend(chosen.tolist())

    # Shuffle
    rng.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:n_sample]

    sampled_texts = [texts[i] for i in sampled_idx]
    sampled_labels = labels[sampled_idx]

    return sampled_texts, sampled_labels, target_names


# ---------------------------------------------------------------------------
# 2. Convert cluster lists to label arrays
# ---------------------------------------------------------------------------

def clusters_to_labels(clusters, n_total):
    """Convert list-of-lists cluster format to integer label array.

    Items not in any cluster get label -1 (singleton/noise).
    """
    labels = np.full(n_total, -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cluster_id
    return labels


# ---------------------------------------------------------------------------
# 3. Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_clustering(pred_labels, true_labels):
    """Compute clustering metrics against ground truth."""
    from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score

    # For V-measure, we need to handle singletons/noise (-1 labels).
    # Treat each singleton as its own cluster for fair comparison.
    pred_for_eval = pred_labels.copy()
    next_id = pred_for_eval.max() + 1 if len(pred_for_eval) > 0 else 0
    for i in range(len(pred_for_eval)):
        if pred_for_eval[i] == -1:
            pred_for_eval[i] = next_id
            next_id += 1

    v = v_measure_score(true_labels, pred_for_eval)
    h = homogeneity_score(true_labels, pred_for_eval)
    c = completeness_score(true_labels, pred_for_eval)

    n_total = len(pred_labels)
    n_clustered = int((pred_labels >= 0).sum())
    n_singletons = n_total - n_clustered
    singleton_rate = n_singletons / n_total if n_total > 0 else 0.0

    unique_clusters = set(pred_labels[pred_labels >= 0].tolist())
    n_clusters = len(unique_clusters)

    if n_clusters > 0:
        cluster_sizes = []
        for cid in unique_clusters:
            cluster_sizes.append(int((pred_labels == cid).sum()))
        mean_size = float(np.mean(cluster_sizes))
        median_size = float(np.median(cluster_sizes))
        max_size = int(np.max(cluster_sizes))
    else:
        mean_size = 0.0
        median_size = 0.0
        max_size = 0

    return {
        "v_measure": round(v, 4),
        "homogeneity": round(h, 4),
        "completeness": round(c, 4),
        "n_clusters": n_clusters,
        "singleton_rate": round(singleton_rate, 4),
        "mean_cluster_size": round(mean_size, 1),
        "median_cluster_size": round(median_size, 1),
        "max_cluster_size": max_size,
    }


# ---------------------------------------------------------------------------
# 4. Main experiment
# ---------------------------------------------------------------------------

def main():
    W = 90
    print("=" * W)
    print("Experiment 5: Clustering Algorithm Comparison (20 Newsgroups)")
    print("=" * W)

    # Load data
    print("\nLoading 20 Newsgroups (test split)...")
    t0 = time.time()
    texts, true_labels, target_names = load_data(n_sample=2000)
    print(f"  Loaded {len(texts)} documents, {len(target_names)} topics in {time.time() - t0:.1f}s")

    label_counts = np.bincount(true_labels)
    print(f"  Samples per topic: min={label_counts.min()}, max={label_counts.max()}, "
          f"mean={label_counts.mean():.0f}")

    # Embed
    print("\nEmbedding with EmbeddingModel (paraphrase-multilingual-MiniLM-L12-v2)...")
    t0 = time.time()
    model = EmbeddingModel()
    embeddings = model.embed_batch(texts)
    print(f"  Embedded {len(texts)} docs -> {embeddings.shape} in {time.time() - t0:.1f}s")

    results = {}

    # --- Greedy Centroid Clustering ---
    print(f"\n{'-' * W}")
    print("GREEDY CENTROID CLUSTERING")
    print("-" * W)

    # N=2000 means pairwise_cosine builds a 2000x2000 matrix (32MB) - fine.
    # But the O(n^2) neighbor counting loop may be slow. Let's time it.
    gc_thresholds = [0.75, 0.80, 0.85, 0.90]
    hdr = (f"  {'threshold':>9}  {'V-meas':>7}  {'Homog':>7}  {'Compl':>7}  "
           f"{'#Clust':>7}  {'Singl%':>7}  {'MeanSz':>7}  {'MaxSz':>7}  {'Time':>6}")
    print(hdr)

    for t in gc_thresholds:
        t0 = time.time()
        clusters = greedy_centroid_cluster(embeddings, threshold=t, max_cluster_size=500)
        elapsed = time.time() - t0
        pred = clusters_to_labels(clusters, len(texts))
        m = evaluate_clustering(pred, true_labels)
        m["time_s"] = round(elapsed, 1)
        results[f"greedy_centroid_t{t}"] = m
        print(f"  {t:>9.2f}  {m['v_measure']:>7.4f}  {m['homogeneity']:>7.4f}  {m['completeness']:>7.4f}  "
              f"{m['n_clusters']:>7}  {m['singleton_rate']:>7.1%}  {m['mean_cluster_size']:>7.1f}  "
              f"{m['max_cluster_size']:>7}  {elapsed:>5.1f}s")

    # --- Complete Linkage Clustering ---
    print(f"\n{'-' * W}")
    print("COMPLETE LINKAGE CLUSTERING")
    print("-" * W)

    cl_thresholds = [0.70, 0.75, 0.80, 0.85]
    print(hdr)

    for t in cl_thresholds:
        t0 = time.time()
        clusters = complete_linkage_cluster(embeddings, threshold=t, max_cluster_size=500)
        elapsed = time.time() - t0
        pred = clusters_to_labels(clusters, len(texts))
        m = evaluate_clustering(pred, true_labels)
        m["time_s"] = round(elapsed, 1)
        results[f"complete_linkage_t{t}"] = m
        print(f"  {t:>9.2f}  {m['v_measure']:>7.4f}  {m['homogeneity']:>7.4f}  {m['completeness']:>7.4f}  "
              f"{m['n_clusters']:>7}  {m['singleton_rate']:>7.1%}  {m['mean_cluster_size']:>7.1f}  "
              f"{m['max_cluster_size']:>7}  {elapsed:>5.1f}s")

    # --- HDBSCAN ---
    print(f"\n{'-' * W}")
    print("HDBSCAN")
    print("-" * W)

    from sklearn.cluster import HDBSCAN

    hdb_min_sizes = [5, 10, 15, 25]
    print(hdr.replace("threshold", "min_clust"))

    for ms in hdb_min_sizes:
        t0 = time.time()
        hdb = HDBSCAN(min_cluster_size=ms, metric="cosine")
        pred = hdb.fit_predict(embeddings)
        elapsed = time.time() - t0
        m = evaluate_clustering(pred, true_labels)
        m["time_s"] = round(elapsed, 1)
        results[f"hdbscan_ms{ms}"] = m
        print(f"  {ms:>9}  {m['v_measure']:>7.4f}  {m['homogeneity']:>7.4f}  {m['completeness']:>7.4f}  "
              f"{m['n_clusters']:>7}  {m['singleton_rate']:>7.1%}  {m['mean_cluster_size']:>7.1f}  "
              f"{m['max_cluster_size']:>7}  {elapsed:>5.1f}s")

    # --- Summary ---
    print(f"\n{'=' * W}")
    print("SUMMARY: Best configuration per method (by V-measure)")
    print("=" * W)

    methods = {
        "Greedy Centroid": [k for k in results if k.startswith("greedy_centroid")],
        "Complete Linkage": [k for k in results if k.startswith("complete_linkage")],
        "HDBSCAN": [k for k in results if k.startswith("hdbscan")],
    }

    print(f"\n  {'Method':<30}  {'Config':<20}  {'V-meas':>7}  {'Homog':>7}  {'Compl':>7}  "
          f"{'#Clust':>7}  {'Singl%':>7}")

    best_overall = None
    best_v = -1.0
    for method_name, keys in methods.items():
        best_key = max(keys, key=lambda k: results[k]["v_measure"])
        m = results[best_key]
        param = best_key.split("_")[-1]
        print(f"  {method_name:<30}  {param:<20}  {m['v_measure']:>7.4f}  {m['homogeneity']:>7.4f}  "
              f"{m['completeness']:>7.4f}  {m['n_clusters']:>7}  {m['singleton_rate']:>7.1%}")
        if m["v_measure"] > best_v:
            best_v = m["v_measure"]
            best_overall = (method_name, best_key)

    if best_overall:
        print(f"\n  Overall best: {best_overall[0]} ({best_overall[1]}) with V-measure={best_v:.4f}")

    # --- Analysis notes ---
    print(f"\n{'-' * W}")
    print("ANALYSIS")
    print("-" * W)

    # Find method with highest homogeneity vs highest completeness
    all_keys = list(results.keys())
    best_homog_key = max(all_keys, key=lambda k: results[k]["homogeneity"])
    best_compl_key = max(all_keys, key=lambda k: results[k]["completeness"])

    print(f"\n  Highest homogeneity: {best_homog_key} = {results[best_homog_key]['homogeneity']:.4f}")
    print(f"    (clusters are pure — each cluster has items from mostly one topic)")
    print(f"  Highest completeness: {best_compl_key} = {results[best_compl_key]['completeness']:.4f}")
    print(f"    (topics are complete — items from same topic land in same cluster)")

    # Tradeoff insight
    print(f"\n  Greedy centroid and complete linkage produce many small, pure clusters")
    print(f"  (high homogeneity) but leave many items unclustered (low completeness).")
    print(f"  HDBSCAN with density-based approach assigns more items to clusters.")

    # --- Save ---
    out = {
        "experiment": "exp5_clustering_comparison",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": "20 Newsgroups (test split)",
        "n_samples": len(texts),
        "n_topics": len(target_names),
        "topic_names": list(target_names),
        "embedding_model": model.model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "results": results,
        "best_overall": {
            "method": best_overall[0] if best_overall else None,
            "config": best_overall[1] if best_overall else None,
            "v_measure": best_v,
        },
    }
    out_path = Path("experiments/results/exp5_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
