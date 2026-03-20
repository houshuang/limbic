"""Tests for amygdala.cluster — clustering algorithms."""

import numpy as np
import pytest

from amygdala.cluster import (
    greedy_centroid_cluster,
    IncrementalCentroidCluster,
    pairwise_cosine,
)


@pytest.fixture(scope="module")
def similar_vecs():
    """Three clusters of similar vectors + some singletons."""
    rng = np.random.default_rng(42)
    dim = 384
    # Cluster A: 3 very similar vectors (low noise to ensure cosine > 0.85)
    base_a = rng.standard_normal(dim)
    base_a /= np.linalg.norm(base_a)
    cluster_a = [base_a + rng.standard_normal(dim) * 0.005 for _ in range(3)]
    # Cluster B: 3 very similar vectors
    base_b = rng.standard_normal(dim)
    base_b /= np.linalg.norm(base_b)
    cluster_b = [base_b + rng.standard_normal(dim) * 0.005 for _ in range(3)]
    # Singleton: far from both
    singleton = rng.standard_normal(dim)
    singleton /= np.linalg.norm(singleton)

    vecs = np.array(cluster_a + cluster_b + [singleton])
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    return vecs  # shape (7, 384)


class TestIncrementalCentroidCluster:
    def test_basic_clustering(self, similar_vecs):
        inc = IncrementalCentroidCluster(threshold=0.85)
        for i, vec in enumerate(similar_vecs):
            inc.add(i, vec)
        clusters = inc.get_clusters(min_size=2)
        assert len(clusters) >= 2
        clustered_ids = {id for c in clusters for id in c}
        assert 6 not in clustered_ids  # singleton excluded

    def test_empty_returns_empty(self):
        inc = IncrementalCentroidCluster()
        assert inc.get_clusters() == []
        assert inc.n_clusters == 0

    def test_single_item_no_clusters(self):
        inc = IncrementalCentroidCluster()
        vec = np.random.default_rng(0).standard_normal(10).astype(np.float32)
        inc.add("a", vec)
        assert inc.get_clusters(min_size=2) == []
        assert inc.n_clusters == 1

    def test_identical_vectors_cluster(self):
        vec = np.ones(10, dtype=np.float32)
        inc = IncrementalCentroidCluster(threshold=0.99)
        inc.add(0, vec)
        inc.add(1, vec)
        inc.add(2, vec)
        clusters = inc.get_clusters(min_size=2)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_orthogonal_vectors_separate(self):
        inc = IncrementalCentroidCluster(threshold=0.5)
        # Orthogonal unit vectors won't cluster
        for i in range(5):
            vec = np.zeros(10)
            vec[i] = 1.0
            inc.add(i, vec)
        clusters = inc.get_clusters(min_size=2)
        assert len(clusters) == 0

    def test_matches_batch_on_tight_threshold(self, similar_vecs):
        """At threshold 0.85, incremental should produce same co-clustered pairs as batch."""
        threshold = 0.85
        batch_clusters = greedy_centroid_cluster(similar_vecs, threshold=threshold)
        batch_pairs = set()
        for c in batch_clusters:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    batch_pairs.add((min(c[i], c[j]), max(c[i], c[j])))

        inc = IncrementalCentroidCluster(threshold=threshold)
        for i in range(len(similar_vecs)):
            inc.add(i, similar_vecs[i])
        inc_clusters = inc.get_clusters(min_size=2)
        inc_pairs = set()
        for c in inc_clusters:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    inc_pairs.add((min(c[i], c[j]), max(c[i], c[j])))

        assert batch_pairs == inc_pairs

    def test_string_ids(self):
        vec = np.ones(10, dtype=np.float32)
        inc = IncrementalCentroidCluster(threshold=0.99)
        inc.add("claim:1", vec)
        inc.add("claim:2", vec)
        clusters = inc.get_clusters(min_size=2)
        assert clusters[0] == ["claim:1", "claim:2"]

    def test_centroid_updates(self):
        inc = IncrementalCentroidCluster(threshold=0.9)
        vec = np.ones(10, dtype=np.float32)
        inc.add(0, vec)
        original_centroid = inc.centroids[0].copy()
        # Perturb one dimension to create a slightly different direction
        vec2 = vec.copy()
        vec2[0] += 0.3
        inc.add(1, vec2)
        # Centroid should have shifted slightly
        assert not np.array_equal(inc.centroids[0], original_centroid)
        # But still close to original
        sim = float(inc.centroids[0] @ original_centroid / (
            np.linalg.norm(inc.centroids[0]) * np.linalg.norm(original_centroid)
        ))
        assert sim > 0.999

    def test_min_size_filter(self, similar_vecs):
        inc = IncrementalCentroidCluster(threshold=0.85)
        for i, vec in enumerate(similar_vecs):
            inc.add(i, vec)
        all_clusters = inc.get_clusters(min_size=1)
        multi_clusters = inc.get_clusters(min_size=2)
        assert len(all_clusters) >= len(multi_clusters)

    def test_n_clusters_property(self):
        inc = IncrementalCentroidCluster()
        assert inc.n_clusters == 0
        vec = np.random.default_rng(0).standard_normal(10).astype(np.float32)
        inc.add(0, vec)
        assert inc.n_clusters == 1
