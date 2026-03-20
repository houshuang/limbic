"""Tests for amygdala.search — vector, FTS5, and hybrid search."""

import numpy as np
import pytest

from amygdala.embed import EmbeddingModel
from amygdala.search import VectorIndex, FTS5Index, HybridSearch, Result


TEST_DOCS = [
    {"id": "1", "content": "Machine learning algorithms for image classification"},
    {"id": "2", "content": "Deep neural networks and convolutional architectures"},
    {"id": "3", "content": "Natural language processing with transformers"},
    {"id": "4", "content": "Reinforcement learning for game playing agents"},
    {"id": "5", "content": "Computer vision applications in autonomous driving"},
    {"id": "6", "content": "Transfer learning and domain adaptation techniques"},
    {"id": "7", "content": "Norwegian education policy and curriculum reform"},
    {"id": "8", "content": "Student assessment methods in primary school"},
    {"id": "9", "content": "Teacher training programs in Scandinavia"},
    {"id": "10", "content": "Digital learning tools for classroom instruction"},
    {"id": "11", "content": "Special education and inclusive practices"},
    {"id": "12", "content": "Early childhood development and learning"},
    {"id": "13", "content": "Climate change impact on global ecosystems"},
    {"id": "14", "content": "Renewable energy sources and sustainability"},
    {"id": "15", "content": "Ocean acidification and marine biodiversity"},
    {"id": "16", "content": "Deforestation and tropical rainforest conservation"},
    {"id": "17", "content": "Genetic engineering and CRISPR gene editing"},
    {"id": "18", "content": "Vaccine development and immunology research"},
    {"id": "19", "content": "Protein folding prediction with AlphaFold"},
    {"id": "20", "content": "Quantum computing and cryptographic applications"},
]


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


@pytest.fixture(scope="module")
def embedded_docs(model):
    texts = [d["content"] for d in TEST_DOCS]
    vecs = model.embed_batch(texts)
    return vecs


class TestVectorIndex:
    def test_add_and_search(self, model, embedded_docs):
        vi = VectorIndex()
        ids = [d["id"] for d in TEST_DOCS]
        vi.add(ids, embedded_docs)
        assert vi.size == 20

        query = model.embed("image recognition and neural networks")
        results = vi.search(query, limit=5)

        assert len(results) == 5
        assert all(isinstance(r, Result) for r in results)
        assert results[0].score >= results[1].score  # sorted descending

        # Top results should be ML-related
        top_ids = {r.id for r in results}
        ml_ids = {"1", "2", "5", "6"}
        assert len(top_ids & ml_ids) >= 2

    def test_filter_ids(self, model, embedded_docs):
        vi = VectorIndex()
        ids = [d["id"] for d in TEST_DOCS]
        vi.add(ids, embedded_docs)

        query = model.embed("machine learning")
        education_ids = {"7", "8", "9", "10", "11", "12"}
        results = vi.search(query, limit=5, filter_ids=education_ids)

        assert all(r.id in education_ids for r in results)

    def test_empty_index(self, model):
        vi = VectorIndex()
        query = model.embed("anything")
        results = vi.search(query, limit=5)
        assert results == []

    def test_source_field(self, model, embedded_docs):
        vi = VectorIndex()
        vi.add([d["id"] for d in TEST_DOCS], embedded_docs)
        query = model.embed("test")
        results = vi.search(query, limit=1)
        assert results[0].source == "vector"


class TestFTS5Index:
    def test_add_and_search(self):
        fts = FTS5Index()
        for doc in TEST_DOCS:
            fts.add(id=doc["id"], content=doc["content"])
        assert fts.size == 20

        results = fts.search("machine learning", limit=5)
        assert len(results) > 0
        assert results[0].id == "1"  # exact match

    def test_keyword_matching(self):
        fts = FTS5Index()
        for doc in TEST_DOCS:
            fts.add(id=doc["id"], content=doc["content"])

        results = fts.search("Norwegian education", limit=5)
        assert any(r.id == "7" for r in results)

    def test_batch_add(self):
        fts = FTS5Index()
        fts.add_batch(TEST_DOCS)
        assert fts.size == 20

        results = fts.search("transformer", limit=3)
        assert any(r.id == "3" for r in results)

    def test_metadata(self):
        fts = FTS5Index()
        fts.add(id="m1", content="test document", metadata={"source": "arxiv", "year": 2024})
        results = fts.search("test", limit=1)
        assert results[0].metadata["source"] == "arxiv"

    def test_no_results(self):
        fts = FTS5Index()
        fts.add(id="1", content="hello world")
        results = fts.search("xyzzyx", limit=5)
        assert results == []

    def test_source_field(self):
        fts = FTS5Index()
        fts.add(id="1", content="test content")
        results = fts.search("test", limit=1)
        assert results[0].source == "fts"


class TestHybridSearch:
    def test_rrf_merges_both(self, model, embedded_docs):
        vi = VectorIndex()
        ids = [d["id"] for d in TEST_DOCS]
        vi.add(ids, embedded_docs)

        fts = FTS5Index()
        fts.add_batch(TEST_DOCS)

        hybrid = HybridSearch(vector_index=vi, fts_index=fts)

        query_vec = model.embed("neural network image classification")
        results = hybrid.search(query_vec, "neural network image classification", limit=5)

        assert len(results) == 5
        assert results[0].source == "hybrid"
        # Top result should be ML-related
        assert results[0].id in {"1", "2", "5"}

    def test_hybrid_boosts_keyword_match(self, model, embedded_docs):
        """A document matching both semantically and by keyword should rank higher."""
        vi = VectorIndex()
        ids = [d["id"] for d in TEST_DOCS]
        vi.add(ids, embedded_docs)

        fts = FTS5Index()
        fts.add_batch(TEST_DOCS)

        hybrid = HybridSearch(vector_index=vi, fts_index=fts)

        # "CRISPR" is a unique keyword in doc 17, plus semantic similarity
        query_vec = model.embed("CRISPR gene editing technology")
        results = hybrid.search(query_vec, "CRISPR gene editing", limit=5)

        assert results[0].id == "17"

    def test_filter_ids_in_hybrid(self, model, embedded_docs):
        vi = VectorIndex()
        ids = [d["id"] for d in TEST_DOCS]
        vi.add(ids, embedded_docs)

        fts = FTS5Index()
        fts.add_batch(TEST_DOCS)

        hybrid = HybridSearch(vector_index=vi, fts_index=fts)

        env_ids = {"13", "14", "15", "16"}
        query_vec = model.embed("environmental science")
        results = hybrid.search(query_vec, "environmental science", limit=5, filter_ids=env_ids)

        assert all(r.id in env_ids for r in results)
