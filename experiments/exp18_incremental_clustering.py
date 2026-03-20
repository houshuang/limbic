import os
"""Experiment 18: Incremental clustering vs batch greedy centroid clustering.

Amygdala's greedy_centroid_cluster requires all vectors upfront and builds an
O(N^2) similarity matrix. Otak has 60K+ claims and ingests continuously.
An incremental variant would let consumers add items one-at-a-time without
re-clustering everything.

Key questions:
1. Is incremental clustering quality comparable to batch?
2. How sensitive is it to insertion order?
3. Does it scale to 60K+ claims?

Methods compared:
- Batch: greedy_centroid_cluster (existing)
- Incremental (original order): add one-by-one in dataset order
- Incremental (shuffled): add in random order, repeated 5 times

Datasets:
- 20 Newsgroups (2K sample, same as exp5)
- Otak claims (~5K sample from claims database)
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel
from amygdala.cluster import greedy_centroid_cluster


# ---------------------------------------------------------------------------
# Incremental Centroid Clustering
# ---------------------------------------------------------------------------

class IncrementalCentroidCluster:
    """Assign items to nearest cluster centroid, or create new cluster."""

    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.centroids = []       # list of centroid vectors (normalized)
        self.clusters = []        # list of lists of (id, vec)
        self._centroid_arr = None # cached numpy array for batch dot products

    def add(self, id, vec):
        """Assign to nearest cluster if above threshold, else create new.

        Returns the cluster index assigned to.
        """
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if not self.centroids:
            self._new_cluster(id, vec)
            return 0

        # Compute similarities to all centroids
        if self._centroid_arr is None or len(self._centroid_arr) != len(self.centroids):
            self._centroid_arr = np.array(self.centroids)
        sims = self._centroid_arr @ vec

        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            self._add_to_cluster(best_idx, id, vec)
            return best_idx
        else:
            self._new_cluster(id, vec)
            return len(self.clusters) - 1

    def _new_cluster(self, id, vec):
        self.centroids.append(vec.copy())
        self.clusters.append([(id, vec)])
        self._centroid_arr = None  # invalidate cache

    def _add_to_cluster(self, idx, id, vec):
        self.clusters[idx].append((id, vec))
        # Update centroid as running mean (re-normalized)
        n = len(self.clusters[idx])
        old_centroid = self.centroids[idx]
        new_centroid = old_centroid * ((n - 1) / n) + vec * (1 / n)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        self.centroids[idx] = new_centroid
        self._centroid_arr = None  # invalidate cache

    def get_clusters_as_id_lists(self):
        """Return clusters as list of lists of IDs (batch format)."""
        return [[id for id, _ in cluster] for cluster in self.clusters]

    def get_clusters_multi(self, min_size=2):
        """Return only clusters with >= min_size members."""
        return [
            [id for id, _ in cluster]
            for cluster in self.clusters
            if len(cluster) >= min_size
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clusters_to_labels(clusters, n_total):
    """Convert list-of-lists cluster format to integer label array."""
    labels = np.full(n_total, -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cluster_id
    return labels


def co_cluster_pairs(clusters):
    """Return set of (min(i,j), max(i,j)) pairs that are co-clustered."""
    pairs = set()
    for cluster in clusters:
        for a_idx in range(len(cluster)):
            for b_idx in range(a_idx + 1, len(cluster)):
                i, j = cluster[a_idx], cluster[b_idx]
                pairs.add((min(i, j), max(i, j)))
    return pairs


def pair_agreement(pairs_a, pairs_b):
    """What fraction of pairs in A are also in B (precision-style)."""
    if not pairs_a:
        return 1.0
    return len(pairs_a & pairs_b) / len(pairs_a)


def evaluate_clustering(pred_labels, true_labels):
    """Compute V-measure against ground truth."""
    from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score

    pred_for_eval = pred_labels.copy()
    next_id = pred_for_eval.max() + 1 if len(pred_for_eval) > 0 else 0
    for i in range(len(pred_for_eval)):
        if pred_for_eval[i] == -1:
            pred_for_eval[i] = next_id
            next_id += 1

    return {
        "v_measure": round(float(v_measure_score(true_labels, pred_for_eval)), 4),
        "homogeneity": round(float(homogeneity_score(true_labels, pred_for_eval)), 4),
        "completeness": round(float(completeness_score(true_labels, pred_for_eval)), 4),
    }


def cluster_stats(clusters, n_total):
    """Compute cluster count and size distribution stats."""
    n_clusters = len(clusters)
    if n_clusters == 0:
        return {"n_clusters": 0, "n_clustered": 0, "singleton_rate": 1.0}

    sizes = [len(c) for c in clusters]
    n_clustered = sum(sizes)

    return {
        "n_clusters": n_clusters,
        "n_clustered": n_clustered,
        "n_singletons": n_total - n_clustered,
        "singleton_rate": round((n_total - n_clustered) / n_total, 4),
        "mean_size": round(float(np.mean(sizes)), 1),
        "median_size": round(float(np.median(sizes)), 1),
        "max_size": int(np.max(sizes)),
        "min_size": int(np.min(sizes)),
        "size_p90": round(float(np.percentile(sizes, 90)), 1),
    }


# ---------------------------------------------------------------------------
# 1. Load 20 Newsgroups
# ---------------------------------------------------------------------------

def load_newsgroups(n_sample=2000, seed=42):
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
    )
    texts = data.data
    labels = data.target

    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    per_class = n_sample // len(unique_labels)

    sampled_idx = []
    for lbl in unique_labels:
        class_idx = np.where(labels == lbl)[0]
        n_take = min(per_class, len(class_idx))
        chosen = rng.choice(class_idx, size=n_take, replace=False)
        sampled_idx.extend(chosen.tolist())

    rng.shuffle(sampled_idx)
    sampled_idx = sampled_idx[:n_sample]

    sampled_texts = [texts[i] for i in sampled_idx]
    sampled_labels = labels[sampled_idx]

    return sampled_texts, sampled_labels, data.target_names


# ---------------------------------------------------------------------------
# 2. Load domain claims
# ---------------------------------------------------------------------------

def load_domain_claims(db_path, n_sample=5000, seed=42):
    """Load claim texts from claims database."""
    CLAIM_TYPE_ID = "ef498d42-63f3-4dc2-baa3-a5fad9e7a72a"
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    # Get distinct claim node IDs and their names (latest version)
    rows = conn.execute("""
        SELECT n.id, n.name
        FROM nodes n
        JOIN idx_types t ON n.id = t.node_id
        WHERE t.type_id = ?
          AND n.deleted_at IS NULL
        GROUP BY n.id
        HAVING n.version = MAX(n.version)
    """, (CLAIM_TYPE_ID,)).fetchall()
    conn.close()

    # Sample
    rng = np.random.RandomState(seed)
    if len(rows) > n_sample:
        indices = rng.choice(len(rows), size=n_sample, replace=False)
        rows = [rows[i] for i in indices]

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return ids, texts


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    W = 95
    print("=" * W)
    print("Experiment 18: Incremental Clustering vs Batch Greedy Centroid")
    print("=" * W)

    results = {}
    threshold = 0.85

    # -----------------------------------------------------------------------
    # Part A: 20 Newsgroups comparison
    # -----------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("PART A: 20 Newsgroups (2K documents, threshold={:.2f})".format(threshold))
    print("=" * W)

    print("\nLoading 20 Newsgroups...")
    t0 = time.time()
    texts, true_labels, target_names = load_newsgroups(n_sample=2000)
    print(f"  Loaded {len(texts)} documents in {time.time() - t0:.1f}s")

    print("\nEmbedding...")
    t0 = time.time()
    model = EmbeddingModel()
    embeddings = model.embed_batch(texts)
    print(f"  Embedded -> {embeddings.shape} in {time.time() - t0:.1f}s")

    n = len(embeddings)

    # --- A1: Batch clustering ---
    print(f"\n{'-' * W}")
    print("A1: Batch greedy centroid cluster")
    print("-" * W)

    t0 = time.time()
    batch_clusters = greedy_centroid_cluster(embeddings, threshold=threshold, max_cluster_size=500)
    batch_time = time.time() - t0

    batch_labels = clusters_to_labels(batch_clusters, n)
    batch_eval = evaluate_clustering(batch_labels, true_labels)
    batch_stats = cluster_stats(batch_clusters, n)
    batch_pairs = co_cluster_pairs(batch_clusters)

    print(f"  Time: {batch_time:.2f}s")
    print(f"  Clusters: {batch_stats['n_clusters']}, Singletons: {batch_stats['n_singletons']} ({batch_stats['singleton_rate']:.1%})")
    print(f"  Cluster sizes: mean={batch_stats['mean_size']}, median={batch_stats['median_size']}, max={batch_stats['max_size']}")
    print(f"  V-measure: {batch_eval['v_measure']:.4f} (H={batch_eval['homogeneity']:.4f}, C={batch_eval['completeness']:.4f})")
    print(f"  Co-clustered pairs: {len(batch_pairs)}")

    results["newsgroups_batch"] = {
        **batch_eval, **batch_stats,
        "time_s": round(batch_time, 2),
        "n_co_cluster_pairs": len(batch_pairs),
    }

    # --- A2: Incremental (original order) ---
    print(f"\n{'-' * W}")
    print("A2: Incremental clustering (original order)")
    print("-" * W)

    t0 = time.time()
    inc = IncrementalCentroidCluster(threshold=threshold)
    for i in range(n):
        inc.add(i, embeddings[i])
    inc_time = time.time() - t0

    inc_clusters_all = inc.get_clusters_as_id_lists()
    inc_clusters_multi = inc.get_clusters_multi(min_size=2)
    inc_labels = clusters_to_labels(inc_clusters_multi, n)
    inc_eval = evaluate_clustering(inc_labels, true_labels)
    inc_stats = cluster_stats(inc_clusters_multi, n)
    inc_pairs = co_cluster_pairs(inc_clusters_multi)

    # Agreement with batch
    agreement_batch_in_inc = pair_agreement(batch_pairs, inc_pairs)
    agreement_inc_in_batch = pair_agreement(inc_pairs, batch_pairs)

    print(f"  Time: {inc_time:.2f}s (speedup: {batch_time / inc_time:.1f}x)")
    print(f"  Total clusters (incl singletons): {len(inc_clusters_all)}")
    print(f"  Multi-member clusters: {inc_stats['n_clusters']}")
    print(f"  Singletons: {inc_stats.get('n_singletons', 'N/A')} ({inc_stats['singleton_rate']:.1%})")
    print(f"  Cluster sizes: mean={inc_stats['mean_size']}, median={inc_stats['median_size']}, max={inc_stats['max_size']}")
    print(f"  V-measure: {inc_eval['v_measure']:.4f} (H={inc_eval['homogeneity']:.4f}, C={inc_eval['completeness']:.4f})")
    print(f"  Co-clustered pairs: {len(inc_pairs)}")
    print(f"  Agreement: {agreement_batch_in_inc:.1%} of batch pairs in incremental, "
          f"{agreement_inc_in_batch:.1%} of incremental pairs in batch")

    results["newsgroups_incremental_original"] = {
        **inc_eval, **inc_stats,
        "time_s": round(inc_time, 2),
        "total_clusters_incl_singletons": len(inc_clusters_all),
        "n_co_cluster_pairs": len(inc_pairs),
        "agreement_batch_in_inc": round(agreement_batch_in_inc, 4),
        "agreement_inc_in_batch": round(agreement_inc_in_batch, 4),
    }

    # --- A3: Incremental (shuffled, 5 runs) ---
    print(f"\n{'-' * W}")
    print("A3: Incremental clustering (5 random shuffles)")
    print("-" * W)

    shuffle_v_measures = []
    shuffle_cluster_counts = []
    shuffle_agreements_batch = []
    shuffle_agreements_inc = []
    shuffle_times = []

    for run in range(5):
        rng = np.random.RandomState(seed=run * 7 + 13)
        order = rng.permutation(n)

        t0 = time.time()
        inc_s = IncrementalCentroidCluster(threshold=threshold)
        for idx in order:
            inc_s.add(int(idx), embeddings[idx])
        s_time = time.time() - t0
        shuffle_times.append(s_time)

        s_clusters = inc_s.get_clusters_multi(min_size=2)
        s_labels = clusters_to_labels(s_clusters, n)
        s_eval = evaluate_clustering(s_labels, true_labels)
        s_stats = cluster_stats(s_clusters, n)
        s_pairs = co_cluster_pairs(s_clusters)

        agr_b = pair_agreement(batch_pairs, s_pairs)
        agr_i = pair_agreement(s_pairs, batch_pairs)

        shuffle_v_measures.append(s_eval["v_measure"])
        shuffle_cluster_counts.append(s_stats["n_clusters"])
        shuffle_agreements_batch.append(agr_b)
        shuffle_agreements_inc.append(agr_i)

        print(f"  Run {run + 1}: V={s_eval['v_measure']:.4f}, "
              f"clusters={s_stats['n_clusters']}, "
              f"agree(batch->inc)={agr_b:.1%}, "
              f"agree(inc->batch)={agr_i:.1%}, "
              f"time={s_time:.2f}s")

    v_arr = np.array(shuffle_v_measures)
    c_arr = np.array(shuffle_cluster_counts)
    ab_arr = np.array(shuffle_agreements_batch)
    ai_arr = np.array(shuffle_agreements_inc)

    print(f"\n  V-measure: mean={v_arr.mean():.4f}, std={v_arr.std():.4f}, "
          f"range=[{v_arr.min():.4f}, {v_arr.max():.4f}]")
    print(f"  Cluster count: mean={c_arr.mean():.0f}, std={c_arr.std():.0f}, "
          f"range=[{c_arr.min()}, {c_arr.max()}]")
    print(f"  Agreement (batch->inc): mean={ab_arr.mean():.1%}, std={ab_arr.std():.4f}")
    print(f"  Agreement (inc->batch): mean={ai_arr.mean():.1%}, std={ai_arr.std():.4f}")

    results["newsgroups_incremental_shuffled"] = {
        "v_measure_mean": round(float(v_arr.mean()), 4),
        "v_measure_std": round(float(v_arr.std()), 4),
        "v_measure_min": round(float(v_arr.min()), 4),
        "v_measure_max": round(float(v_arr.max()), 4),
        "n_clusters_mean": round(float(c_arr.mean()), 1),
        "n_clusters_std": round(float(c_arr.std()), 1),
        "n_clusters_min": int(c_arr.min()),
        "n_clusters_max": int(c_arr.max()),
        "agreement_batch_in_inc_mean": round(float(ab_arr.mean()), 4),
        "agreement_batch_in_inc_std": round(float(ab_arr.std()), 4),
        "agreement_inc_in_batch_mean": round(float(ai_arr.mean()), 4),
        "agreement_inc_in_batch_std": round(float(ai_arr.std()), 4),
        "time_mean_s": round(float(np.mean(shuffle_times)), 2),
        "runs": 5,
    }

    # --- A4: Compare batch vs incremental delta ---
    print(f"\n{'-' * W}")
    print("A4: Batch vs Incremental Summary (20 Newsgroups)")
    print("-" * W)

    v_delta = v_arr.mean() - batch_eval["v_measure"]
    print(f"  Batch V-measure:          {batch_eval['v_measure']:.4f}")
    print(f"  Incremental V-measure:    {v_arr.mean():.4f} +/- {v_arr.std():.4f}")
    print(f"  Delta:                    {v_delta:+.4f}")
    print(f"  Batch time:               {batch_time:.2f}s")
    print(f"  Incremental time (mean):  {np.mean(shuffle_times):.2f}s")
    print(f"  Speedup:                  {batch_time / np.mean(shuffle_times):.1f}x")

    # -----------------------------------------------------------------------
    # Part B: Otak scale test
    # -----------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("PART B: Otak Claims Scale Test")
    print("=" * W)

    eval_db = os.environ.get("AMYGDALA_EVAL_DB", "eval_claims.db")
    db_path_obj = Path(eval_db)

    if db_path_obj.exists():
        print("\nLoading claims from claims database...")
        t0 = time.time()
        claim_ids, claim_texts = load_domain_claims(eval_db, n_sample=5000)
        print(f"  Loaded {len(claim_texts)} claims in {time.time() - t0:.1f}s")

        # Filter out empty/very short claims
        valid = [(cid, txt) for cid, txt in zip(claim_ids, claim_texts) if txt and len(txt.strip()) > 10]
        claim_ids = [v[0] for v in valid]
        claim_texts = [v[1] for v in valid]
        print(f"  After filtering: {len(claim_texts)} claims")

        # Embed
        print("\nEmbedding claims...")
        t0 = time.time()
        claim_embeddings = model.embed_batch(claim_texts)
        embed_time = time.time() - t0
        print(f"  Embedded -> {claim_embeddings.shape} in {embed_time:.1f}s")

        n_claims = len(claim_embeddings)

        # Incremental clustering
        print(f"\n{'-' * W}")
        print(f"B1: Incremental clustering on {n_claims} claims (threshold={threshold})")
        print("-" * W)

        t0 = time.time()
        inc_domain = IncrementalCentroidCluster(threshold=threshold)
        for i in range(n_claims):
            inc_domain.add(i, claim_embeddings[i])
        domain_inc_time = time.time() - t0

        domain_clusters_all = inc_domain.get_clusters_as_id_lists()
        domain_clusters_multi = inc_domain.get_clusters_multi(min_size=2)
        domain_stats = cluster_stats(domain_clusters_multi, n_claims)

        print(f"  Time: {domain_inc_time:.2f}s")
        print(f"  Total clusters (incl singletons): {len(domain_clusters_all)}")
        print(f"  Multi-member clusters: {domain_stats['n_clusters']}")
        if domain_stats['n_clusters'] > 0:
            print(f"  Clustered items: {domain_stats['n_clustered']} ({1 - domain_stats['singleton_rate']:.1%})")
            print(f"  Cluster sizes: mean={domain_stats['mean_size']}, median={domain_stats['median_size']}, "
                  f"max={domain_stats['max_size']}, p90={domain_stats['size_p90']}")

        # Size distribution histogram
        if domain_stats['n_clusters'] > 0:
            sizes = [len(c) for c in domain_clusters_multi]
            bins = [2, 3, 5, 10, 20, 50, 100, float("inf")]
            print(f"\n  Cluster size distribution:")
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                count = sum(1 for s in sizes if lo <= s < hi)
                label = f"  {int(lo)}-{int(hi)-1}" if hi != float("inf") else f"  {int(lo)}+"
                bar = "#" * min(count, 60)
                print(f"    {label:>8}: {count:>5}  {bar}")

        # Show sample clusters
        if domain_clusters_multi:
            print(f"\n  Sample clusters (top 5 by size):")
            sorted_clusters = sorted(domain_clusters_multi, key=len, reverse=True)
            for ci, cluster in enumerate(sorted_clusters[:5]):
                print(f"\n    Cluster {ci + 1} ({len(cluster)} members):")
                for member_id in cluster[:4]:
                    text = claim_texts[member_id][:100]
                    print(f"      - {text}")
                if len(cluster) > 4:
                    print(f"      ... and {len(cluster) - 4} more")

        # Test order sensitivity on domain data
        print(f"\n{'-' * W}")
        print("B2: Order sensitivity on domain claims (3 shuffles)")
        print("-" * W)

        domain_shuffle_counts = []
        domain_shuffle_times = []

        for run in range(3):
            rng = np.random.RandomState(seed=run * 11 + 7)
            order = rng.permutation(n_claims)

            t0 = time.time()
            inc_s = IncrementalCentroidCluster(threshold=threshold)
            for idx in order:
                inc_s.add(int(idx), claim_embeddings[idx])
            s_time = time.time() - t0
            domain_shuffle_times.append(s_time)

            s_clusters = inc_s.get_clusters_multi(min_size=2)
            s_count = len(s_clusters)
            domain_shuffle_counts.append(s_count)

            s_clustered = sum(len(c) for c in s_clusters)
            print(f"  Run {run + 1}: {s_count} clusters, "
                  f"{s_clustered} items clustered ({s_clustered/n_claims:.1%}), "
                  f"time={s_time:.2f}s")

        c_arr_domain = np.array(domain_shuffle_counts)
        print(f"\n  Cluster count: mean={c_arr_domain.mean():.0f}, std={c_arr_domain.std():.0f}, "
              f"range=[{c_arr_domain.min()}, {c_arr_domain.max()}]")
        print(f"  Coefficient of variation: {c_arr_domain.std() / c_arr_domain.mean():.1%}")

        results["domain_incremental"] = {
            **domain_stats,
            "n_claims": n_claims,
            "embed_time_s": round(embed_time, 2),
            "cluster_time_s": round(domain_inc_time, 2),
            "total_clusters_incl_singletons": len(domain_clusters_all),
        }
        results["domain_order_sensitivity"] = {
            "runs": 3,
            "n_clusters_mean": round(float(c_arr_domain.mean()), 1),
            "n_clusters_std": round(float(c_arr_domain.std()), 1),
            "n_clusters_min": int(c_arr_domain.min()),
            "n_clusters_max": int(c_arr_domain.max()),
            "coefficient_of_variation": round(float(c_arr_domain.std() / c_arr_domain.mean()), 4),
            "time_mean_s": round(float(np.mean(domain_shuffle_times)), 2),
        }
    else:
        print(f"\n  Eval DB not found at {eval_db}, skipping Part B.")

    # -----------------------------------------------------------------------
    # Summary & Conclusions
    # -----------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("CONCLUSIONS")
    print("=" * W)

    print(f"""
  1. QUALITY: Incremental clustering {'matches' if abs(v_delta) < 0.02 else 'differs from'} batch quality.
     Batch V-measure: {batch_eval['v_measure']:.4f}
     Incremental V-measure: {v_arr.mean():.4f} +/- {v_arr.std():.4f} (delta={v_delta:+.4f})

  2. SPEED: Incremental is {batch_time / np.mean(shuffle_times):.1f}x {'faster' if batch_time > np.mean(shuffle_times) else 'slower'} than batch.
     Batch: {batch_time:.2f}s, Incremental: {np.mean(shuffle_times):.2f}s (N={n})
     Incremental is O(N*K) where K = number of clusters, vs O(N^2) for batch.

  3. ORDER SENSITIVITY: V-measure std={v_arr.std():.4f}, cluster count std={c_arr.std():.0f}.
     {'Low' if v_arr.std() < 0.01 else 'Moderate' if v_arr.std() < 0.03 else 'High'} order sensitivity.

  4. AGREEMENT WITH BATCH:
     {ab_arr.mean():.1%} of batch pairs preserved in incremental.
     {ai_arr.mean():.1%} of incremental pairs also in batch.

  5. RECOMMENDATION: Incremental clustering is {'viable' if v_arr.std() < 0.03 and ab_arr.mean() > 0.5 else 'marginal' if ab_arr.mean() > 0.3 else 'not recommended'} for continuous ingestion pipelines.
     It avoids the O(N^2) similarity matrix, enabling 60K+ claim clustering.
""")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out = {
        "experiment": "exp18_incremental_clustering",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "threshold": threshold,
        "dataset_newsgroups": {
            "n_samples": n,
            "embedding_model": model.model_name,
            "embedding_dim": int(embeddings.shape[1]),
        },
        "results": results,
    }

    out_path = Path("experiments/results/exp18_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
