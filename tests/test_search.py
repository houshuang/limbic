"""Tests for amygdala.search — vector, FTS5, hybrid search, multi-list RRF, and query expansion."""

import numpy as np
import pytest

from limbic.amygdala.embed import EmbeddingModel
from limbic.amygdala.index import Index
from limbic.amygdala.search import (
    VectorIndex, FTS5Index, HybridSearch, Result, dedup_by,
    multi_list_rrf, TracedResult, RRFContribution, strong_signal,
)


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


class TestFTS5Sanitization:
    def test_reserved_word_not(self):
        """FTS5 reserved word NOT should be quoted, not interpreted as operator."""
        fts = FTS5Index()
        fts.add(id="1", content="do NOT use this pattern in production")
        fts.add(id="2", content="a safe pattern for production use")
        results = fts.search("do NOT use this", limit=5)
        assert any(r.id == "1" for r in results)

    def test_reserved_word_and_or(self):
        fts = FTS5Index()
        fts.add(id="1", content="cats AND dogs OR birds")
        results = fts.search("cats AND dogs", limit=5)
        assert any(r.id == "1" for r in results)

    def test_reserved_word_near(self):
        fts = FTS5Index()
        fts.add(id="1", content="the NEAR miss was scary")
        results = fts.search("NEAR miss", limit=5)
        assert any(r.id == "1" for r in results)

    def test_unicode_norwegian(self):
        fts = FTS5Index()
        fts.add(id="1", content="norsk utdanningspolitikk og læreplanreform")
        results = fts.search("læreplanreform", limit=5)
        assert len(results) > 0
        assert results[0].id == "1"

    def test_unicode_mixed(self):
        fts = FTS5Index()
        fts.add(id="1", content="Ångström measurements in über precise experiments")
        results = fts.search("Ångström über", limit=5)
        assert any(r.id == "1" for r in results)


class TestIndexTriggers:
    def test_insert_syncs_fts(self):
        idx = Index()
        idx.add_document("doc1", [{"content": "hello world"}], mtime=1.0)
        results = idx._fts_search("hello", limit=5)
        assert len(results) == 1

    def test_delete_cleans_fts(self):
        idx = Index()
        idx.add_document("doc1", [{"content": "unique_token_xyz"}], mtime=1.0)
        assert len(idx._fts_search("unique_token_xyz")) == 1
        # Re-adding with different content replaces (DELETE old + INSERT new)
        idx.add_document("doc1", [{"content": "completely different"}], mtime=2.0)
        assert len(idx._fts_search("unique_token_xyz")) == 0
        assert len(idx._fts_search("completely different")) == 1

    def test_claims_sync_fts(self):
        idx = Index()
        idx.add_claims([{"id": "c1", "content": "claim about education"}])
        results = idx._fts_search("education")
        assert len(results) == 1


class TestGrep:
    def test_exact_substring(self):
        idx = Index()
        idx.add_document("doc1", [{"content": "file at /Users/stian/src/limbic/index.py"}], mtime=1.0)
        idx.add_document("doc2", [{"content": "some other content entirely"}], mtime=1.0)
        results = idx.grep("/Users/stian/src/limbic")
        assert len(results) == 1
        assert results[0].source == "grep"

    def test_code_pattern(self):
        idx = Index()
        idx.add_document("doc1", [{"content": "def _sync_fts_for(self, doc_path):"}], mtime=1.0)
        idx.add_document("doc2", [{"content": "def search(self, query):"}], mtime=1.0)
        results = idx.grep("_sync_fts_for")
        assert len(results) == 1

    def test_empty_pattern(self):
        idx = Index()
        idx.add_document("doc1", [{"content": "anything"}], mtime=1.0)
        assert idx.grep("") == []
        assert idx.grep("   ") == []

    def test_collection_filter(self):
        idx = Index()
        idx.add_document("doc1", [{"content": "error: connection refused"}], collection="logs", mtime=1.0)
        idx.add_document("doc2", [{"content": "error: connection timeout"}], collection="other", mtime=1.0)
        results = idx.grep("error: connection", collection="logs")
        assert len(results) == 1


class TestDedupBy:
    def test_keeps_first_per_group(self):
        results = [
            Result(id="1", score=0.9, metadata={"session": "A"}),
            Result(id="2", score=0.8, metadata={"session": "B"}),
            Result(id="3", score=0.7, metadata={"session": "A"}),
            Result(id="4", score=0.6, metadata={"session": "C"}),
        ]
        deduped = dedup_by(results, key_fn=lambda r: r.metadata["session"])
        assert len(deduped) == 3
        ids = [r.id for r in deduped]
        assert "1" in ids  # kept (first from A)
        assert "3" not in ids  # dropped (second from A)

    def test_empty_input(self):
        assert dedup_by([], key_fn=lambda r: r.id) == []

    def test_all_unique(self):
        results = [
            Result(id="1", score=0.9, metadata={"g": "A"}),
            Result(id="2", score=0.8, metadata={"g": "B"}),
        ]
        deduped = dedup_by(results, key_fn=lambda r: r.metadata["g"])
        assert len(deduped) == 2


class TestMultiListRRF:
    def test_basic_fusion(self):
        results = multi_list_rrf(
            ranked_lists=[
                [{"id": "a"}, {"id": "b"}],
                [{"id": "b"}, {"id": "c"}],
            ],
            list_labels=["vec", "fts"],
        )
        assert all(isinstance(r, TracedResult) for r in results)
        ids = [r.id for r in results]
        assert "a" in ids and "b" in ids and "c" in ids
        # "b" appears in both lists, should be top
        assert results[0].id == "b"

    def test_top_rank_bonus(self):
        """Items at rank 1 get +0.05 bonus."""
        results = multi_list_rrf(
            ranked_lists=[
                [{"id": "a"}, {"id": "b"}],
            ],
            list_labels=["vec"],
        )
        k = 60
        a_expected = 1 / (k + 1) + 0.05  # rank 1 bonus
        b_expected = 1 / (k + 2) + 0.02  # rank 2-3 bonus
        assert abs(results[0].score - a_expected) < 1e-10
        assert abs(results[1].score - b_expected) < 1e-10

    def test_no_bonus(self):
        results = multi_list_rrf(
            ranked_lists=[[{"id": "a"}]],
            list_labels=["vec"],
            top_rank_bonus=False,
        )
        k = 60
        assert abs(results[0].score - 1 / (k + 1)) < 1e-10

    def test_traces(self):
        results = multi_list_rrf(
            ranked_lists=[
                [{"id": "a"}, {"id": "b"}],
                [{"id": "b"}, {"id": "a"}],
            ],
            list_labels=["vec", "fts"],
        )
        b = next(r for r in results if r.id == "b")
        assert len(b.traces) == 2
        labels = {t.list_label for t in b.traces}
        assert labels == {"vec", "fts"}

    def test_custom_id_fn(self):
        results = multi_list_rrf(
            ranked_lists=[
                [{"chunk_id": 10}, {"chunk_id": 20}],
            ],
            list_labels=["vec"],
            id_fn=lambda x: x["chunk_id"],
        )
        assert results[0].id == 10

    def test_many_lists(self):
        """7 lists (original + 5 expanded) should produce higher scores."""
        single = multi_list_rrf(
            ranked_lists=[[{"id": "a"}]],
            list_labels=["vec"],
        )
        many = multi_list_rrf(
            ranked_lists=[[{"id": "a"}] for _ in range(7)],
            list_labels=[f"list_{i}" for i in range(7)],
        )
        assert many[0].score > single[0].score
        assert len(many[0].traces) == 7

    def test_empty_lists(self):
        results = multi_list_rrf([], [])
        assert results == []


class TestStrongSignal:
    def test_clear_winner(self):
        assert strong_signal([0.90, 0.70]) is True

    def test_gap_too_small(self):
        assert strong_signal([0.85, 0.80]) is False

    def test_top_too_low(self):
        assert strong_signal([0.60, 0.40]) is False

    def test_single_result(self):
        assert strong_signal([0.95]) is False

    def test_custom_thresholds(self):
        assert strong_signal([0.70, 0.50], threshold=0.65, gap=0.10) is True

    def test_exact_boundary(self):
        assert strong_signal([0.82, 0.70], threshold=0.82, gap=0.12) is True
        assert strong_signal([0.82, 0.71], threshold=0.82, gap=0.12) is False
