"""Tests for amygdala.embed — embedding, whitening, genericization, caching."""

import os
import tempfile
import time
import numpy as np
import pytest

from amygdala.embed import EmbeddingModel, DEFAULT_MODEL


@pytest.fixture(scope="module")
def model():
    """Shared model instance (loading is slow, do it once)."""
    return EmbeddingModel()


@pytest.fixture(scope="module")
def genericize_model():
    return EmbeddingModel(genericize=True)


class TestBasicEmbedding:
    def test_embed_returns_correct_dim(self, model):
        vec = model.embed("The cat sat on the mat.")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_is_normalized(self, model):
        vec = model.embed("The cat sat on the mat.")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch(self, model):
        texts = ["Hello world", "Goodbye world", "Machine learning is great"]
        vecs = model.embed_batch(texts)
        assert vecs.shape == (3, 384)

    def test_similar_texts_have_high_similarity(self, model):
        v1 = model.embed("The cat sat on the mat.")
        v2 = model.embed("A cat was sitting on a mat.")
        v3 = model.embed("Quantum entanglement in superconductors")
        sim_close = float(v1 @ v2)
        sim_far = float(v1 @ v3)
        assert sim_close > sim_far
        assert sim_close > 0.5

    def test_empty_batch(self, model):
        vecs = model.embed_batch([])
        assert vecs.shape[0] == 0

    def test_default_model_is_multilingual(self):
        assert "multilingual" in DEFAULT_MODEL

    def test_cross_lingual_similarity(self, model):
        """Norwegian and English expressing the same idea should be similar."""
        v_no = model.embed("Utdanning er viktig for demokratiet")
        v_en = model.embed("Education is important for democracy")
        sim = float(v_no @ v_en)
        assert sim > 0.7


class TestWhitening:
    def test_whitening_changes_dimension(self):
        corpus = [
            "Education policy in Norway",
            "Student achievement and learning outcomes",
            "Teacher training programs",
            "Curriculum reform in Scandinavian countries",
            "Assessment methods in primary school",
            "Digital learning tools for classrooms",
            "Special education and inclusion",
            "Higher education funding models",
            "Vocational training effectiveness",
            "Early childhood education research",
        ]
        model_w = EmbeddingModel(whiten_dims=128)
        model_w.fit_whitening(corpus)
        vec = model_w.embed("New text about education")
        assert vec.shape == (128,)

    def test_whitening_requires_whiten_dims(self):
        model = EmbeddingModel()  # no whiten_dims or whiten_epsilon
        with pytest.raises(ValueError, match="whiten_epsilon"):
            model.fit_whitening(["text1", "text2"])

    def test_whitening_spreads_distribution(self):
        """After whitening, similarity distribution should be wider."""
        corpus = [
            f"Policy proposal about education topic {i} with details"
            for i in range(50)
        ]
        model_raw = EmbeddingModel()
        raw_vecs = model_raw.embed_batch(corpus)
        raw_sim = raw_vecs @ raw_vecs.T
        raw_offdiag = raw_sim[np.triu_indices(len(corpus), k=1)]

        model_w = EmbeddingModel(whiten_dims=256)
        model_w.fit_whitening(corpus)
        w_vecs = model_w.embed_batch(corpus)
        w_sim = w_vecs @ w_vecs.T
        w_offdiag = w_sim[np.triu_indices(len(corpus), k=1)]

        assert w_offdiag.std() > raw_offdiag.std() * 0.5 or w_offdiag.mean() < raw_offdiag.mean()

    def test_whitening_params_reusable(self):
        corpus = ["text " + str(i) for i in range(20)]
        m1 = EmbeddingModel(whiten_dims=128)
        params = m1.fit_whitening(corpus)

        m2 = EmbeddingModel(whiten_dims=128)
        m2.set_whitening(params)

        v1 = m1.embed("test text")
        v2 = m2.embed("test text")
        assert np.allclose(v1, v2, atol=1e-5)


class TestSoftZCA:
    def test_soft_zca_preserves_dimensions(self):
        corpus = [
            "Education policy in Norway",
            "Student achievement and learning outcomes",
            "Teacher training programs",
            "Curriculum reform in Scandinavian countries",
            "Assessment methods in primary school",
            "Digital learning tools for classrooms",
            "Special education and inclusion",
            "Higher education funding models",
            "Vocational training effectiveness",
            "Early childhood education research",
        ]
        model_zca = EmbeddingModel(whiten_epsilon=0.1)
        model_zca.fit_whitening(corpus)
        vec = model_zca.embed("New text about education")
        assert vec.shape == (384,)  # Soft-ZCA preserves all dims

    def test_soft_zca_improves_isotropy(self):
        """After Soft-ZCA, mean pairwise cosine should drop (better isotropy)."""
        corpus = [
            f"Education research finding number {i} about student learning"
            for i in range(50)
        ]
        model_raw = EmbeddingModel()
        raw_vecs = model_raw.embed_batch(corpus)
        raw_sim = raw_vecs @ raw_vecs.T
        raw_mpc = raw_sim[np.triu_indices(len(corpus), k=1)].mean()

        model_zca = EmbeddingModel(whiten_epsilon=0.1)
        model_zca.fit_whitening(corpus)
        zca_vecs = model_zca.embed_batch(corpus)
        zca_sim = zca_vecs @ zca_vecs.T
        zca_mpc = zca_sim[np.triu_indices(len(corpus), k=1)].mean()

        assert zca_mpc < raw_mpc

    def test_soft_zca_params_reusable(self):
        corpus = ["education text " + str(i) for i in range(20)]
        m1 = EmbeddingModel(whiten_epsilon=0.1)
        params = m1.fit_whitening(corpus)

        m2 = EmbeddingModel(whiten_epsilon=0.1)
        m2.set_whitening(params)

        v1 = m1.embed("test text")
        v2 = m2.embed("test text")
        assert np.allclose(v1, v2, atol=1e-5)

    def test_soft_zca_requires_epsilon(self):
        model = EmbeddingModel()  # no whiten_epsilon or whiten_dims
        with pytest.raises(ValueError, match="whiten_epsilon"):
            model.fit_whitening(["text1", "text2"])

    def test_epsilon_controls_aggressiveness(self):
        """Larger epsilon = less aggressive = closer to raw (higher pairwise sims)."""
        corpus = [
            f"Education topic {i} with various details about learning"
            for i in range(50)
        ]
        m_aggressive = EmbeddingModel(whiten_epsilon=0.001)
        m_aggressive.fit_whitening(corpus)
        vecs_agg = m_aggressive.embed_batch(corpus)
        sim_agg = vecs_agg @ vecs_agg.T
        mean_agg = sim_agg[np.triu_indices(len(corpus), k=1)].mean()

        m_mild = EmbeddingModel(whiten_epsilon=10.0)
        m_mild.fit_whitening(corpus)
        vecs_mild = m_mild.embed_batch(corpus)
        sim_mild = vecs_mild @ vecs_mild.T
        mean_mild = sim_mild[np.triu_indices(len(corpus), k=1)].mean()

        # Very large ε ≈ identity, so pairwise sims stay high (closer to raw)
        # Very small ε ≈ full ZCA, sims spread out more (lower mean)
        assert abs(mean_agg) < abs(mean_mild) or mean_agg < mean_mild


class TestTruncation:
    def test_truncate_dim(self):
        model = EmbeddingModel(truncate_dim=128)
        vec = model.embed("test truncation")
        assert vec.shape == (128,)
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01

    def test_dim_property_with_truncation(self):
        model = EmbeddingModel(truncate_dim=256)
        assert model.dim == 256


class TestGenericization:
    def test_strips_urls(self, genericize_model):
        model = genericize_model
        v1 = model.embed("Check https://example.com/page for details about education")
        v2 = model.embed("Check for details about education")
        sim = float(v1 @ v2)
        assert sim > 0.9

    def test_strips_numbers(self, genericize_model):
        model = genericize_model
        v1 = model.embed("The program served 1,500 students in 2023")
        v2 = model.embed("The program served students in")
        sim = float(v1 @ v2)
        assert sim > 0.8

    def test_genericization_flag(self):
        model_no_gen = EmbeddingModel(genericize=False)
        model_gen = EmbeddingModel(genericize=True)
        text = "In 2024, 75% of the 500 students passed"
        v_raw = model_no_gen.embed(text)
        v_gen = model_gen.embed(text)
        sim = float(v_raw @ v_gen)
        assert sim < 0.99


class TestCaching:
    def test_cache_hit_is_fast(self, model):
        text = "This is a unique test text for caching"
        model.embed(text)
        t0 = time.time()
        model.embed(text)
        hot_time = time.time() - t0
        assert hot_time < 0.01

    def test_cache_eviction(self):
        model = EmbeddingModel(cache_size=3)
        model.embed("text A")
        model.embed("text B")
        model.embed("text C")
        model.embed("text D")  # evicts "text A"
        model.embed("text A")  # re-embed, should work

    def test_batch_populates_cache(self, model):
        texts = ["batch cache test alpha", "batch cache test beta"]
        model.embed_batch(texts)
        t0 = time.time()
        model.embed("batch cache test alpha")
        assert time.time() - t0 < 0.01


class TestAllButTop:
    def test_abt_preserves_dimensions(self):
        corpus = [
            "Education policy in Norway",
            "Student achievement and learning outcomes",
            "Teacher training programs",
            "Curriculum reform in Scandinavian countries",
            "Assessment methods in primary school",
            "Digital learning tools for classrooms",
            "Special education and inclusion",
            "Higher education funding models",
            "Vocational training effectiveness",
            "Early childhood education research",
        ]
        model_abt = EmbeddingModel(whiten_abt=1)
        model_abt.fit_whitening(corpus)
        vec = model_abt.embed("New text about education")
        assert vec.shape == (384,)

    def test_abt_improves_isotropy(self):
        """After ABT, mean pairwise cosine should drop (better isotropy)."""
        corpus = [
            f"Education research finding number {i} about student learning"
            for i in range(50)
        ]
        model_raw = EmbeddingModel()
        raw_vecs = model_raw.embed_batch(corpus)
        raw_sim = raw_vecs @ raw_vecs.T
        raw_mpc = raw_sim[np.triu_indices(len(corpus), k=1)].mean()

        model_abt = EmbeddingModel(whiten_abt=1)
        model_abt.fit_whitening(corpus)
        abt_vecs = model_abt.embed_batch(corpus)
        abt_sim = abt_vecs @ abt_vecs.T
        abt_mpc = abt_sim[np.triu_indices(len(corpus), k=1)].mean()

        assert abt_mpc < raw_mpc

    def test_abt_params_reusable(self):
        corpus = ["education text " + str(i) for i in range(20)]
        m1 = EmbeddingModel(whiten_abt=2)
        params = m1.fit_whitening(corpus)

        m2 = EmbeddingModel(whiten_abt=2)
        m2.set_whitening(params)

        v1 = m1.embed("test text")
        v2 = m2.embed("test text")
        assert np.allclose(v1, v2, atol=1e-5)

    def test_abt_d_controls_components_removed(self):
        """Higher D removes more components, changing the embedding more."""
        corpus = [f"Topic {i} about education and learning" for i in range(30)]

        m1 = EmbeddingModel(whiten_abt=1)
        m1.fit_whitening(corpus)
        v1 = m1.embed("test text")

        m3 = EmbeddingModel(whiten_abt=3)
        m3.fit_whitening(corpus)
        v3 = m3.embed("test text")

        # D=1 and D=3 should produce different embeddings
        assert not np.allclose(v1, v3, atol=0.01)


class TestPersistentCache:
    def test_cache_roundtrip(self, model):
        """Embeddings cached to disk should match originals exactly."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        try:
            m = EmbeddingModel(cache_path=cache_path)
            v1 = m.embed("persistent cache test text")
            # Clear in-memory cache to force persistent cache read
            m._cache.clear()
            v2 = m.embed("persistent cache test text")
            assert np.allclose(v1, v2, atol=1e-7)
        finally:
            os.unlink(cache_path)

    def test_cache_batch_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        try:
            m = EmbeddingModel(cache_path=cache_path)
            texts = ["cache batch A", "cache batch B", "cache batch C"]
            vecs1 = m.embed_batch(texts)
            m._cache.clear()
            vecs2 = m.embed_batch(texts)
            assert np.allclose(vecs1, vecs2, atol=1e-7)
        finally:
            os.unlink(cache_path)

    def test_cache_survives_new_instance(self):
        """Cache persists across EmbeddingModel instances."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        try:
            m1 = EmbeddingModel(cache_path=cache_path)
            v1 = m1.embed("cross-instance cache test")
            del m1
            m2 = EmbeddingModel(cache_path=cache_path)
            m2._cache.clear()
            v2 = m2.embed("cross-instance cache test")
            assert np.allclose(v1, v2, atol=1e-7)
        finally:
            os.unlink(cache_path)

    def test_cache_partial_hits(self):
        """Batch with some cached and some uncached texts."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name
        try:
            m = EmbeddingModel(cache_path=cache_path)
            m.embed("already cached text")
            m._cache.clear()
            vecs = m.embed_batch(["already cached text", "brand new text"])
            assert vecs.shape == (2, 384)
        finally:
            os.unlink(cache_path)
