"""Tests for amygdala.novelty — novelty scoring with adaptive K and centroid specificity."""

import numpy as np
import pytest

from amygdala.embed import EmbeddingModel
from amygdala.search import VectorIndex
from amygdala.novelty import novelty_score, batch_novelty, _adaptive_k, corpus_centroid


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


@pytest.fixture(scope="module")
def ml_index(model):
    """An index of ML-related texts."""
    texts = [
        "Deep learning for image classification",
        "Convolutional neural networks for object detection",
        "Recurrent neural networks for sequence modeling",
        "Transformer architectures for natural language processing",
        "Generative adversarial networks for image synthesis",
        "Reinforcement learning for robot control",
        "Transfer learning for domain adaptation",
        "Federated learning for privacy preservation",
        "Graph neural networks for molecular property prediction",
        "Self-supervised learning from unlabeled data",
    ]
    vecs = model.embed_batch(texts)
    vi = VectorIndex()
    vi.add([str(i) for i in range(len(texts))], vecs)
    return vi


class TestAdaptiveK:
    def test_small_index_uses_k1(self):
        assert _adaptive_k(10, None) == 1
        assert _adaptive_k(50, None) == 1

    def test_medium_index_uses_k3(self):
        assert _adaptive_k(100, None) == 3

    def test_large_index_uses_k10(self):
        assert _adaptive_k(5000, None) == 10

    def test_explicit_k_overrides(self):
        assert _adaptive_k(10, 7) == 7
        assert _adaptive_k(10000, 3) == 3


class TestNoveltyScore:
    def test_identical_text_is_not_novel(self, model, ml_index):
        vec = model.embed("Deep learning for image classification")
        score = novelty_score(vec, ml_index)
        assert score < 0.5

    def test_similar_text_low_novelty(self, model, ml_index):
        vec = model.embed("Neural networks for visual recognition tasks")
        score = novelty_score(vec, ml_index)
        assert score < 0.7

    def test_unrelated_text_high_novelty(self, model, ml_index):
        vec = model.embed("Norwegian education policy and curriculum reform in primary schools")
        score = novelty_score(vec, ml_index)
        assert score > 0.3

    def test_novelty_range(self, model, ml_index):
        vec = model.embed("Something about anything")
        score = novelty_score(vec, ml_index)
        assert 0.0 <= score <= 1.0

    def test_empty_index(self, model):
        vi = VectorIndex()
        vec = model.embed("anything at all")
        score = novelty_score(vec, vi)
        assert score == 1.0


class TestCentroidSpecificity:
    def test_generic_text_lower_score(self, model, ml_index):
        """Generic text near corpus centroid should score lower than specific text."""
        generic = model.embed("Learning and computation")
        specific = model.embed("Norwegian fjord ecosystem biodiversity assessment")
        score_generic = novelty_score(generic, ml_index)
        score_specific = novelty_score(specific, ml_index)
        assert score_specific > score_generic

    def test_centroid_specificity_can_be_toggled(self, model, ml_index):
        vec = model.embed("Some text")
        score_with = novelty_score(vec, ml_index, use_centroid_specificity=True)
        score_without = novelty_score(vec, ml_index, use_centroid_specificity=False)
        assert score_with != score_without

    def test_precomputed_centroid(self, model, ml_index):
        centroid = corpus_centroid(ml_index)
        assert centroid is not None
        vec = model.embed("Test text")
        score = novelty_score(vec, ml_index, centroid=centroid)
        assert 0.0 <= score <= 1.0


class TestSpecificityWeighting:
    def test_measurable_gets_full_weight(self, model, ml_index):
        vec = model.embed("Climate change causes ocean temperature rise of 2 degrees")
        score_measurable = novelty_score(vec, ml_index, specificity="measurable")
        score_vague = novelty_score(vec, ml_index, specificity="vague")
        assert score_measurable > score_vague

    def test_vague_reduces_score(self, model, ml_index):
        vec = model.embed("Things are changing in the world")
        score_none = novelty_score(vec, ml_index)
        score_vague = novelty_score(vec, ml_index, specificity="vague")
        assert score_vague < score_none


class TestCategoryNovelty:
    def test_within_category_novelty(self, model):
        texts_a = [
            "Machine learning for image classification",
            "Deep learning for object detection",
            "Neural networks for visual recognition",
        ]
        texts_b = [
            "Norwegian education policy reform",
            "Student assessment in primary school",
            "Teacher training in Scandinavia",
        ]

        all_texts = texts_a + texts_b
        vecs = model.embed_batch(all_texts)
        vi = VectorIndex()
        ids = [str(i) for i in range(len(all_texts))]
        vi.add(ids, vecs)

        query = model.embed("Curriculum reform and student outcomes")
        ml_ids = {"0", "1", "2"}
        edu_ids = {"3", "4", "5"}

        score_in_ml = novelty_score(query, vi, category_ids=ml_ids)
        score_in_edu = novelty_score(query, vi, category_ids=edu_ids)
        assert score_in_ml > score_in_edu


class TestTemporalDecay:
    def test_decay_makes_old_items_less_relevant(self, model, ml_index):
        """With K>1, an old close match contributes less → novelty shifts."""
        vec = model.embed("Deep learning for image classification")
        # Close match (item "0") is recent, others recent too
        recent_ts = {str(i): 0.0 for i in range(10)}
        score_recent = novelty_score(vec, ml_index, top_k=5,
                                     timestamps=recent_ts, decay_lambda=0.1)

        # Close match (item "0") is old, others are recent
        mixed_ts = {str(i): 0.0 for i in range(10)}
        mixed_ts["0"] = 200.0  # the closest match is very old
        score_old_match = novelty_score(vec, ml_index, top_k=5,
                                        timestamps=mixed_ts, decay_lambda=0.1)

        # When the close match is old, it's down-weighted → novelty increases
        assert score_old_match > score_recent

    def test_zero_lambda_equals_no_decay(self, model, ml_index):
        """decay_lambda=0 should produce identical results to no timestamps."""
        vec = model.embed("Neural networks for visual recognition")
        score_no_ts = novelty_score(vec, ml_index)
        ts = {str(i): 50.0 for i in range(10)}
        score_zero_lambda = novelty_score(vec, ml_index, timestamps=ts, decay_lambda=0.0)
        assert abs(score_no_ts - score_zero_lambda) < 1e-6

    def test_no_timestamps_equals_no_decay(self, model, ml_index):
        """Missing timestamps dict should produce same results as no decay."""
        vec = model.embed("Reinforcement learning for robotics")
        score_none = novelty_score(vec, ml_index)
        score_with_lambda = novelty_score(vec, ml_index, decay_lambda=0.05)
        assert abs(score_none - score_with_lambda) < 1e-6

    def test_decay_with_batch(self, model, ml_index):
        """batch_novelty should pass through temporal decay."""
        texts = ["Deep learning for images", "Norwegian education policy"]
        vecs = model.embed_batch(texts)
        ts = {str(i): 100.0 for i in range(10)}

        batch_scores = batch_novelty(vecs, ml_index, timestamps=ts, decay_lambda=0.02)
        individual_scores = [
            novelty_score(vecs[i], ml_index, timestamps=ts, decay_lambda=0.02)
            for i in range(2)
        ]
        for bs, is_ in zip(batch_scores, individual_scores):
            assert abs(bs - is_) < 1e-6


class TestBatchNovelty:
    def test_batch_matches_individual(self, model, ml_index):
        texts = [
            "Deep learning for image classification",
            "Norwegian education policy reform",
        ]
        vecs = model.embed_batch(texts)

        batch_scores = batch_novelty(vecs, ml_index)
        individual_scores = [novelty_score(vecs[i], ml_index) for i in range(2)]

        assert len(batch_scores) == 2
        for bs, is_ in zip(batch_scores, individual_scores):
            assert abs(bs - is_) < 1e-6
