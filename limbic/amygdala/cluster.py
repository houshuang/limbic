"""Clustering and pairwise similarity utilities.

Includes:
- pairwise_cosine: cosine similarity matrix
- extract_pairs: find similar pairs above threshold, with optional group filtering
- greedy_centroid_cluster / complete_linkage_cluster: two clustering strategies
- classify_pairs_with_confidence: triage pairs into confident/uncertain buckets
- format_for_eval_harness: convert uncertain pairs to eval harness format
"""

import numpy as np


def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, dim) array. Will be L2-normalized internally.

    Returns:
        (N, N) similarity matrix with values in [-1, 1].
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    return normed @ normed.T


def extract_pairs(
    similarity: np.ndarray,
    threshold: float,
    groups: list | None = None,
    cross_group_only: bool = False,
) -> list[tuple[int, int, float]]:
    """Find all pairs with similarity >= threshold.

    Args:
        similarity: (N, N) pre-computed similarity matrix.
        threshold: Minimum cosine similarity to include.
        groups: Optional list of group labels (length N). When provided with
            cross_group_only=True, only returns pairs from different groups.
            Useful for cross-article or cross-document pair extraction.
        cross_group_only: If True and groups is provided, skip same-group pairs.

    Returns:
        List of (i, j, score) tuples sorted by score descending.
    """
    n = similarity.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if cross_group_only and groups is not None and groups[i] == groups[j]:
                continue
            score = float(similarity[i, j])
            if score >= threshold:
                pairs.append((i, j, score))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def greedy_centroid_cluster(
    embeddings: np.ndarray,
    threshold: float = 0.85,
    max_cluster_size: int = 50,
) -> list[list[int]]:
    """Greedy centroid clustering on cosine similarity.

    Each cluster member must be >= threshold similar to the centroid.
    Nodes with the most neighbors are selected as centroids first.

    Args:
        embeddings: (N, dim) array.
        threshold: Minimum cosine similarity to centroid.
        max_cluster_size: Maximum members per cluster.

    Returns:
        List of clusters, each a list of integer indices into embeddings.
        Singletons (items with no neighbors) are excluded.
    """
    n = len(embeddings)
    if n < 2:
        return []

    sim_matrix = pairwise_cosine(embeddings)

    # Count neighbors per node
    neighbor_count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                neighbor_count[i] += 1
                neighbor_count[j] += 1

    # Process nodes by decreasing degree
    order = np.argsort(-neighbor_count)
    assigned = set()
    clusters = []

    for centroid_idx in order:
        centroid_idx = int(centroid_idx)
        if centroid_idx in assigned:
            continue
        if neighbor_count[centroid_idx] == 0:
            continue

        cluster = [centroid_idx]
        assigned.add(centroid_idx)

        for other in range(n):
            if other in assigned:
                continue
            if len(cluster) >= max_cluster_size:
                break
            if sim_matrix[centroid_idx, other] >= threshold:
                cluster.append(other)
                assigned.add(other)

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


class IncrementalCentroidCluster:
    """Online clustering: assign each item to the nearest centroid or start a new cluster.

    Unlike greedy_centroid_cluster (which needs all vectors upfront and builds
    an O(N^2) similarity matrix), this processes items one-at-a-time in O(N*K)
    where K is the number of clusters. Quality is identical to batch at
    threshold >= 0.85, with effectively zero order sensitivity. (Exp 18)

    Usage:
        clusterer = IncrementalCentroidCluster(threshold=0.85)
        for i, vec in enumerate(embeddings):
            cluster_id = clusterer.add(i, vec)
        clusters = clusterer.get_clusters(min_size=2)
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._centroids: list[np.ndarray] = []
        self._clusters: list[list] = []
        self._centroid_arr: np.ndarray | None = None

    def add(self, id, vec: np.ndarray) -> int:
        """Assign to nearest cluster if above threshold, else create new.

        Returns the cluster index assigned to.
        """
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if not self._centroids:
            return self._new_cluster(id, vec)

        if self._centroid_arr is None or len(self._centroid_arr) != len(self._centroids):
            self._centroid_arr = np.array(self._centroids)
        sims = self._centroid_arr @ vec

        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            self._add_to_cluster(best_idx, id, vec)
            return best_idx
        return self._new_cluster(id, vec)

    def _new_cluster(self, id, vec: np.ndarray) -> int:
        self._centroids.append(vec.copy())
        self._clusters.append([id])
        self._centroid_arr = None
        return len(self._clusters) - 1

    def _add_to_cluster(self, idx: int, id, vec: np.ndarray):
        self._clusters[idx].append(id)
        n = len(self._clusters[idx])
        new_centroid = self._centroids[idx] * ((n - 1) / n) + vec * (1 / n)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        self._centroids[idx] = new_centroid
        self._centroid_arr = None

    def get_clusters(self, min_size: int = 2) -> list[list]:
        """Return clusters with at least min_size members."""
        return [c for c in self._clusters if len(c) >= min_size]

    @property
    def n_clusters(self) -> int:
        return len(self._clusters)

    @property
    def centroids(self) -> list[np.ndarray]:
        return list(self._centroids)


def complete_linkage_cluster(
    embeddings: np.ndarray,
    threshold: float = 0.78,
    max_cluster_size: int = 50,
) -> list[list[int]]:
    """Complete-linkage greedy clustering on cosine similarity.

    Stricter than centroid clustering: every member must be >= threshold
    similar to ALL other members, not just the centroid. Useful for
    deduplication tasks where false merges are costly.

    Args:
        embeddings: (N, dim) array.
        threshold: Minimum cosine similarity between ALL pairs in a cluster.
        max_cluster_size: Maximum members per cluster.

    Returns:
        List of clusters. Singletons excluded.
    """
    n = len(embeddings)
    if n < 2:
        return []

    sim_matrix = pairwise_cosine(embeddings)
    assigned = set()
    clusters = []

    # Sort by number of high-similarity neighbors (descending)
    neighbor_counts = (sim_matrix >= threshold).sum(axis=1) - 1  # exclude self
    order = np.argsort(-neighbor_counts)

    for seed in order:
        seed = int(seed)
        if seed in assigned or neighbor_counts[seed] <= 0:
            continue

        cluster = [seed]
        assigned.add(seed)

        # Try to add each unassigned node
        for candidate in range(n):
            if candidate in assigned or len(cluster) >= max_cluster_size:
                continue
            # Must be >= threshold to ALL current members
            if all(sim_matrix[candidate, m] >= threshold for m in cluster):
                cluster.append(candidate)
                assigned.add(candidate)

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def classify_pairs_with_confidence(
    pairs: list[tuple[int, int, float]],
    texts: list[str],
    labels: list[str] | None = None,
    confident_threshold: float = 0.75,
    reject_threshold: float = 0.30,
) -> tuple[list[dict], list[dict]]:
    """Classify pairs and separate confident from uncertain results.

    Uses cosine similarity score to triage pairs into three zones:
    - score >= confident_threshold: auto-classified as "duplicate"
    - score < reject_threshold: auto-classified as "different"
    - score in between: marked as uncertain, needing human review

    This is designed to feed into a calibration-eval harness where a
    human reviewer resolves the uncertain pairs, producing labeled data
    that can be used to tune thresholds.

    Args:
        pairs: List of (i, j, score) tuples from extract_pairs().
        texts: The original text list (indexed by i, j).
        labels: Relationship labels to classify into. Defaults to
            ["duplicate", "different"]. Only the first two are used
            for confident/rejected; uncertain items get None.
        confident_threshold: Pairs with score >= this are auto-accepted
            as the first label (default "duplicate").
        reject_threshold: Pairs with score < this are auto-classified
            as the second label (default "different").

    Returns:
        (confident, uncertain) tuple.
        confident: items with clear classifications. Each dict has keys:
            i, j, score, classification, confidence.
        uncertain: items needing human review. Each dict has keys:
            i, j, score, classification (None), confidence, reason.
    """
    if labels is None:
        labels = ["duplicate", "different"]
    if len(labels) < 2:
        raise ValueError("labels must have at least 2 entries")

    accept_label = labels[0]
    reject_label = labels[1]

    confident: list[dict] = []
    uncertain: list[dict] = []

    for i, j, score in pairs:
        if score >= confident_threshold:
            confident.append({
                "i": i,
                "j": j,
                "score": score,
                "classification": accept_label,
                "confidence": score,
            })
        elif score < reject_threshold:
            confident.append({
                "i": i,
                "j": j,
                "score": score,
                "classification": reject_label,
                "confidence": 1.0 - score,
            })
        else:
            # Ambiguous zone: confidence is low, proportional to distance
            # from the midpoint of the ambiguous range
            midpoint = (confident_threshold + reject_threshold) / 2
            distance_from_mid = abs(score - midpoint)
            range_half = (confident_threshold - reject_threshold) / 2
            amb_confidence = distance_from_mid / range_half if range_half > 0 else 0.0

            uncertain.append({
                "i": i,
                "j": j,
                "score": score,
                "classification": None,
                "confidence": amb_confidence,
                "reason": (
                    f"Cosine similarity {score:.3f} is in the ambiguous zone "
                    f"[{reject_threshold:.2f}, {confident_threshold:.2f}). "
                    f"Too similar to auto-reject, too dissimilar to auto-accept."
                ),
            })

    return confident, uncertain


def format_for_eval_harness(
    uncertain: list[dict],
    texts: list[str],
) -> list[dict]:
    """Convert uncertain pairs into the eval harness data format.

    Produces a list of dicts suitable for a human-evaluation or
    LLM-as-judge calibration harness, where each item presents
    two texts for comparison.

    Args:
        uncertain: List of uncertain pair dicts from
            classify_pairs_with_confidence().
        texts: The original text list (indexed by i, j in each pair).

    Returns:
        List of dicts with keys: id, content, output, meta, output_label.
    """
    results = []
    for item in uncertain:
        i = item["i"]
        j = item["j"]
        score = item["score"]
        results.append({
            "id": f"pair_{i}_{j}",
            "content": texts[i],
            "output": texts[j],
            "meta": f"cosine={score:.3f}",
            "output_label": "Compared text",
        })
    return results
