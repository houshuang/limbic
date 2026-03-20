"""Amygdala — embedding, search, novelty, clustering, and knowledge mapping primitives."""

from .embed import EmbeddingModel
from .search import VectorIndex, FTS5Index, HybridSearch, Result, rerank
from .novelty import novelty_score, batch_novelty, corpus_centroid, nli_classify, nli_classify_batch, classify_pairs
from .cluster import greedy_centroid_cluster, complete_linkage_cluster, IncrementalCentroidCluster, pairwise_cosine, extract_pairs
from .cache import PersistentEmbeddingCache
from .index import connect
from .knowledge_map import KnowledgeGraph, BeliefState, init_beliefs, next_probe, update_beliefs, coverage_report, is_converged, calibrate_beliefs, adjust_for_calibration, knowledge_fringes

__all__ = [
    "EmbeddingModel",
    "VectorIndex",
    "FTS5Index",
    "HybridSearch",
    "Result",
    "rerank",
    "novelty_score",
    "batch_novelty",
    "corpus_centroid",
    "nli_classify",
    "nli_classify_batch",
    "classify_pairs",
    "greedy_centroid_cluster",
    "complete_linkage_cluster",
    "IncrementalCentroidCluster",
    "pairwise_cosine",
    "extract_pairs",
    "PersistentEmbeddingCache",
    "connect",
    "KnowledgeGraph",
    "BeliefState",
    "init_beliefs",
    "next_probe",
    "update_beliefs",
    "coverage_report",
    "is_converged",
    "calibrate_beliefs",
    "adjust_for_calibration",
    "knowledge_fringes",
]
