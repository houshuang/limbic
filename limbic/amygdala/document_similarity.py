"""Document-level similarity using weighted multi-field embeddings.

Finds thematically similar document pairs in a corpus by embedding text fields
and computing pairwise cosine similarity. Supports weighted combination of
multiple text fields (e.g., summary + claims) which outperforms single-field
embedding in calibration experiments.

Calibration results (Petrarca, 2026-03-21):
  - Weighted 0.5×summary + 0.5×claims: 94% accuracy, Spearman ρ=0.818 (18 human pairs)
  - Summary only baseline:             89% accuracy, Spearman ρ=0.654
  - Validated on 300 LLM-rated pairs:  AUROC=0.930, best F1=0.83 at threshold=0.64
  - Synthetic benchmark (50 pairs):    Spearman ρ=0.895

Recommended thresholds (from 300-pair calibration):
  - Feed ranking (recall-focused):     0.49 (P=71%, R=82%, F1=76%)
  - Briefing card (balanced):          0.52 (P=80%, R=78%, F1=79%)
  - High confidence:                   0.55 (P=91%, R=75%, F1=82%)
  - Near-duplicate detection:          0.64 (P=96%, R=73%, F1=83%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .embed import EmbeddingModel
from .cluster import pairwise_cosine


@dataclass
class Document:
    """A document with an ID and named text fields to embed.

    Example:
        Document(
            id="article_123",
            texts={"summary": "Article about...", "claims": "Claim 1. Claim 2."},
            metadata={"title": "My Article", "topics": ["ai", "agents"]},
        )
    """
    id: str
    texts: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityPair:
    """A pair of similar documents with score and optional metadata."""
    id_a: str
    id_b: str
    score: float
    field_scores: dict[str, float] = field(default_factory=dict)


def find_similar_documents(
    documents: list[Document],
    *,
    # Text field configuration
    text_fields: dict[str, float] | str = "summary",
    model: EmbeddingModel | None = None,

    # Thresholds
    threshold: float = 0.52,
    max_pairs: int | None = None,
) -> list[SimilarityPair]:
    """Find similar document pairs using weighted multi-field embeddings.

    Embeds one or more text fields per document, combines with weights,
    and returns pairs above the cosine similarity threshold.

    Args:
        documents: List of Document objects with text fields to embed.
        text_fields: Either a single field name (str) or a dict mapping
            field names to weights. Example: {"summary": 0.5, "claims": 0.5}
            Documents missing a field use the first available field as fallback.
        model: EmbeddingModel to use. Creates default if None.
        threshold: Minimum cosine similarity to return.
        max_pairs: Maximum number of pairs to return (top-N by score).

    Returns:
        List of SimilarityPair sorted by score descending.
    """
    if not documents or len(documents) < 2:
        return []

    # Normalize text_fields to dict
    if isinstance(text_fields, str):
        field_weights = {text_fields: 1.0}
    else:
        field_weights = dict(text_fields)

    if not field_weights:
        raise ValueError("text_fields must specify at least one field")

    if model is None:
        model = EmbeddingModel()

    # Build embeddings per field, then combine
    ids, combined = _embed_weighted(documents, field_weights, model)
    if len(ids) < 2:
        return []

    # Compute similarities
    sim_matrix = pairwise_cosine(combined)

    # Also compute per-field scores if multiple fields
    field_matrices = {}
    if len(field_weights) > 1:
        for fname in field_weights:
            field_ids, field_embs = _embed_single_field(documents, fname, model)
            if len(field_ids) == len(ids):
                field_matrices[fname] = pairwise_cosine(field_embs)

    # Extract pairs
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            score = float(sim_matrix[i][j])
            if score >= threshold:
                field_scores = {}
                for fname, fmat in field_matrices.items():
                    field_scores[fname] = round(float(fmat[i][j]), 4)

                pairs.append(SimilarityPair(
                    id_a=ids[i],
                    id_b=ids[j],
                    score=round(score, 4),
                    field_scores=field_scores,
                ))

    pairs.sort(key=lambda p: p.score, reverse=True)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    return pairs


def embed_documents(
    documents: list[Document],
    *,
    text_fields: dict[str, float] | str = "summary",
    model: EmbeddingModel | None = None,
) -> tuple[list[str], np.ndarray]:
    """Embed documents using weighted multi-field strategy.

    Returns:
        (ids, embeddings) where ids[i] corresponds to embeddings[i].
        Embeddings are L2-normalized.
    """
    if isinstance(text_fields, str):
        field_weights = {text_fields: 1.0}
    else:
        field_weights = text_fields

    if model is None:
        model = EmbeddingModel()

    return _embed_weighted(documents, field_weights, model)


def document_similarity_matrix(
    documents: list[Document],
    *,
    text_fields: dict[str, float] | str = "summary",
    model: EmbeddingModel | None = None,
) -> tuple[list[str], np.ndarray]:
    """Compute full pairwise similarity matrix for documents.

    Returns:
        (ids, sim_matrix) where sim_matrix[i][j] is cosine similarity
        between documents ids[i] and ids[j].
    """
    ids, embeddings = embed_documents(
        documents, text_fields=text_fields, model=model,
    )
    return ids, pairwise_cosine(embeddings)


# --- Internal helpers ---

def _normalize(embs: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embs / norms


def _embed_single_field(
    documents: list[Document],
    field_name: str,
    model: EmbeddingModel,
) -> tuple[list[str], np.ndarray]:
    """Embed a single text field from documents."""
    ids = []
    texts = []
    for doc in documents:
        text = doc.texts.get(field_name, "")
        if not text:
            # Fallback: use first available text field
            for v in doc.texts.values():
                if v:
                    text = v
                    break
        if text:
            ids.append(doc.id)
            texts.append(text)

    if not texts:
        return [], np.empty((0, model.dim), dtype=np.float32)

    embeddings = model.embed_batch(texts)
    return ids, _normalize(embeddings)


def _embed_weighted(
    documents: list[Document],
    field_weights: dict[str, float],
    model: EmbeddingModel,
) -> tuple[list[str], np.ndarray]:
    """Embed documents using weighted combination of multiple fields.

    Each field is embedded separately, normalized, then combined with weights.
    The combined vector is re-normalized.
    """
    if len(field_weights) == 1:
        fname = next(iter(field_weights))
        return _embed_single_field(documents, fname, model)

    # Embed each field
    field_results: dict[str, tuple[list[str], np.ndarray]] = {}
    for fname in field_weights:
        ids, embs = _embed_single_field(documents, fname, model)
        field_results[fname] = (ids, embs)

    # Use the first field's IDs as reference (all should match since fallback fills gaps)
    ref_ids = field_results[next(iter(field_weights))][0]

    # Weighted combination
    combined = np.zeros((len(ref_ids), model.dim), dtype=np.float32)
    total_weight = sum(field_weights.values())
    if total_weight <= 0:
        raise ValueError("text_fields weights must sum to a positive value")

    for fname, weight in field_weights.items():
        _, embs = field_results[fname]
        if embs.shape[0] == len(ref_ids):
            combined += (weight / total_weight) * embs

    return ref_ids, _normalize(combined)
