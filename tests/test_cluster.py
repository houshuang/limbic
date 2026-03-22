"""Tests for amygdala.cluster — clustering algorithms."""

import numpy as np
import pytest

from limbic.amygdala.cluster import (
    classify_pairs_with_confidence,
    format_for_eval_harness,
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


class TestClassifyPairsWithConfidence:
    """Tests for classify_pairs_with_confidence."""

    @pytest.fixture
    def sample_texts(self):
        return [
            "Deep learning improves NLP tasks",
            "Neural networks enhance language processing",
            "Cats are popular pets worldwide",
            "The stock market rose 2% today",
            "Machine learning transforms text analysis",
        ]

    def test_high_score_classified_as_duplicate(self, sample_texts):
        pairs = [(0, 1, 0.92)]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(confident) == 1
        assert len(uncertain) == 0
        assert confident[0]["classification"] == "duplicate"
        assert confident[0]["confidence"] == 0.92

    def test_low_score_classified_as_different(self, sample_texts):
        pairs = [(0, 3, 0.15)]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(confident) == 1
        assert len(uncertain) == 0
        assert confident[0]["classification"] == "different"
        assert confident[0]["confidence"] == pytest.approx(0.85)

    def test_mid_score_is_uncertain(self, sample_texts):
        pairs = [(0, 4, 0.55)]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(confident) == 0
        assert len(uncertain) == 1
        item = uncertain[0]
        assert item["classification"] is None
        assert item["i"] == 0
        assert item["j"] == 4
        assert item["score"] == 0.55
        assert "reason" in item
        assert "ambiguous zone" in item["reason"]

    def test_mixed_pairs(self, sample_texts):
        pairs = [
            (0, 1, 0.92),  # high -> duplicate
            (0, 4, 0.55),  # mid -> uncertain
            (0, 3, 0.15),  # low -> different
            (1, 4, 0.80),  # high -> duplicate
            (2, 3, 0.40),  # mid -> uncertain
        ]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(confident) == 3
        assert len(uncertain) == 2
        classifications = [c["classification"] for c in confident]
        assert classifications.count("duplicate") == 2
        assert classifications.count("different") == 1

    def test_custom_labels(self, sample_texts):
        pairs = [(0, 1, 0.90), (0, 3, 0.10)]
        confident, uncertain = classify_pairs_with_confidence(
            pairs, sample_texts, labels=["same_claim", "unrelated"]
        )
        assert confident[0]["classification"] == "same_claim"
        assert confident[1]["classification"] == "unrelated"

    def test_custom_thresholds(self, sample_texts):
        pairs = [(0, 1, 0.55)]
        # With default thresholds (0.75/0.30), 0.55 is uncertain
        _, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(uncertain) == 1

        # With lower confident threshold, 0.55 becomes confident
        confident, uncertain = classify_pairs_with_confidence(
            pairs, sample_texts, confident_threshold=0.50
        )
        assert len(confident) == 1
        assert len(uncertain) == 0
        assert confident[0]["classification"] == "duplicate"

    def test_empty_pairs(self, sample_texts):
        confident, uncertain = classify_pairs_with_confidence([], sample_texts)
        assert confident == []
        assert uncertain == []

    def test_boundary_at_confident_threshold(self, sample_texts):
        # Exactly at confident_threshold should be classified as duplicate
        pairs = [(0, 1, 0.75)]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(confident) == 1
        assert confident[0]["classification"] == "duplicate"

    def test_boundary_at_reject_threshold(self, sample_texts):
        # Exactly at reject_threshold should be uncertain (not rejected)
        pairs = [(0, 1, 0.30)]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(uncertain) == 1
        assert uncertain[0]["classification"] is None

    def test_just_below_reject_threshold(self, sample_texts):
        pairs = [(0, 1, 0.29)]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(confident) == 1
        assert confident[0]["classification"] == "different"

    def test_labels_too_few_raises(self, sample_texts):
        with pytest.raises(ValueError, match="at least 2"):
            classify_pairs_with_confidence(
                [(0, 1, 0.5)], sample_texts, labels=["only_one"]
            )

    def test_uncertain_confidence_at_midpoint_is_zero(self, sample_texts):
        # Midpoint of [0.30, 0.75) is 0.525
        pairs = [(0, 1, 0.525)]
        _, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert len(uncertain) == 1
        assert uncertain[0]["confidence"] == pytest.approx(0.0, abs=1e-6)

    def test_uncertain_confidence_near_edges(self, sample_texts):
        # Near confident_threshold: should have high confidence
        pairs = [(0, 1, 0.74)]
        _, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert uncertain[0]["confidence"] > 0.9

        # Near reject_threshold: should also have high confidence
        pairs = [(0, 1, 0.31)]
        _, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        assert uncertain[0]["confidence"] > 0.9


class TestFormatForEvalHarness:
    """Tests for format_for_eval_harness."""

    @pytest.fixture
    def sample_texts(self):
        return [
            "Deep learning improves NLP tasks",
            "Neural networks enhance language processing",
            "Cats are popular pets worldwide",
            "The stock market rose 2% today",
            "Machine learning transforms text analysis",
        ]

    def test_basic_format(self, sample_texts):
        uncertain = [
            {"i": 0, "j": 4, "score": 0.55, "classification": None,
             "confidence": 0.1, "reason": "ambiguous"},
        ]
        result = format_for_eval_harness(uncertain, sample_texts)
        assert len(result) == 1
        item = result[0]
        assert item["id"] == "pair_0_4"
        assert item["content"] == sample_texts[0]
        assert item["output"] == sample_texts[4]
        assert item["meta"] == "cosine=0.550"
        assert item["output_label"] == "Compared text"

    def test_multiple_pairs(self, sample_texts):
        uncertain = [
            {"i": 0, "j": 4, "score": 0.55, "classification": None,
             "confidence": 0.1, "reason": "ambiguous"},
            {"i": 2, "j": 3, "score": 0.40, "classification": None,
             "confidence": 0.3, "reason": "ambiguous"},
        ]
        result = format_for_eval_harness(uncertain, sample_texts)
        assert len(result) == 2
        assert result[0]["id"] == "pair_0_4"
        assert result[1]["id"] == "pair_2_3"
        assert result[1]["content"] == sample_texts[2]
        assert result[1]["output"] == sample_texts[3]

    def test_empty_uncertain(self, sample_texts):
        result = format_for_eval_harness([], sample_texts)
        assert result == []

    def test_score_formatting_precision(self, sample_texts):
        uncertain = [
            {"i": 0, "j": 1, "score": 0.123456789, "classification": None,
             "confidence": 0.0, "reason": "ambiguous"},
        ]
        result = format_for_eval_harness(uncertain, sample_texts)
        assert result[0]["meta"] == "cosine=0.123"

    def test_roundtrip_with_classify(self, sample_texts):
        """End-to-end: extract_pairs -> classify -> format."""
        pairs = [
            (0, 1, 0.92),
            (0, 4, 0.55),
            (0, 3, 0.15),
            (2, 3, 0.40),
        ]
        confident, uncertain = classify_pairs_with_confidence(pairs, sample_texts)
        harness_items = format_for_eval_harness(uncertain, sample_texts)
        # Two uncertain pairs: (0,4) at 0.55 and (2,3) at 0.40
        assert len(harness_items) == 2
        ids = {item["id"] for item in harness_items}
        assert "pair_0_4" in ids
        assert "pair_2_3" in ids
