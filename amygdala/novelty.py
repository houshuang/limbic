"""Novelty scoring and relationship classification.

Signals for novelty_score():
1. Global novelty: 1 - mean(top-K similarities across entire index)
2. Topic-local novelty: 1 - mean(top-K similarities within same category)
3. Centroid specificity: distance from corpus centroid (opt-in, helps diverse corpora)
4. Categorical specificity: measurable > concrete > vague
5. NLI classification: distinguish entailment (KNOWN) from contradiction (NEW)
6. Temporal decay: weight older neighbors less (opt-in, for time-ordered corpora)

classify_pairs() implements the cosine + NLI cascade pattern:
- Cosine handles easy cases (above known_threshold = KNOWN, below extends = NEW)
- NLI cross-encoder resolves the ambiguous middle zone

Key findings (experiments 2026-03-19):
- Centroid specificity: +17% separation on diverse data, -3% AUC on homogeneous (QQP)
- Adaptive K: K=1 for ≤50 items, K=10 for 1000+
- Cosine can't distinguish contradictions — NLI cross-encoder fixes this
- Temporal decay: +9.3% Spearman at λ=0.02 (half-life ~35 days). (Exp 10)
"""

import numpy as np

from .search import VectorIndex

SPECIFICITY_WEIGHTS = {
    "measurable": 1.0,
    "concrete": 0.8,
    "vague": 0.3,
}

DEFAULT_GLOBAL_WEIGHT = 0.4
DEFAULT_LOCAL_WEIGHT = 0.6
SPECIFICITY_BLEND = 0.3


def _adaptive_k(index_size: int, top_k: int | None) -> int:
    """Choose K based on index size. Small index → small K."""
    if top_k is not None:
        return top_k
    if index_size <= 50:
        return 1
    if index_size <= 200:
        return 3
    if index_size <= 1000:
        return 5
    return 10


def corpus_centroid(index: VectorIndex) -> np.ndarray | None:
    """Compute mean embedding of all items in the index."""
    if index._matrix is None or index.size == 0:
        return None
    return index._matrix.mean(axis=0)


def _decay_weighted_mean(sims: list[float], ids: list[str],
                         timestamps: dict[str, float] | None,
                         decay_lambda: float) -> float:
    """Compute weighted mean similarity with exponential temporal decay.

    Items with higher age (in days) contribute less to the "already known" signal.
    """
    if not timestamps or decay_lambda <= 0:
        return float(np.mean(sims))
    weights = [np.exp(-decay_lambda * timestamps.get(id_, 0.0)) for id_ in ids]
    total_w = sum(weights)
    if total_w < 1e-8:
        return float(np.mean(sims))
    return float(sum(s * w for s, w in zip(sims, weights)) / total_w)


def novelty_score(
    query_embedding: np.ndarray,
    index: VectorIndex,
    top_k: int | None = None,
    category_ids: set[str] | None = None,
    specificity: str | None = None,
    centroid: np.ndarray | None = None,
    use_centroid_specificity: bool = False,
    timestamps: dict[str, float] | None = None,
    decay_lambda: float = 0.0,
) -> float:
    """Compute novelty score for a single embedding against an index.

    Args:
        query_embedding: The embedding to score.
        index: VectorIndex containing the reference corpus.
        top_k: Number of nearest neighbors. None = adaptive based on index size.
        category_ids: IDs within the same category (for local novelty).
        specificity: "measurable", "concrete", or "vague" for categorical weighting.
        centroid: Pre-computed corpus centroid (avoids recomputing per call).
        use_centroid_specificity: Blend in centroid-distance specificity signal.
            Helps on topically diverse corpora (+17% separation), hurts on
            homogeneous corpora (QQP: AUC drops 3%). Default off.
        timestamps: Dict mapping item ID → age in days. Used with decay_lambda
            for temporal decay weighting. Older items contribute less.
        decay_lambda: Exponential decay rate. 0 = no decay (default).
            0.02 = half-life ~35 days (optimal in Exp 10).

    Returns:
        Novelty score in [0, 1]. Higher = more novel.
    """
    if index.size == 0:
        return 1.0

    k = _adaptive_k(index.size, top_k)

    # Global novelty
    global_results = index.search(query_embedding, limit=k)
    if not global_results:
        return 1.0
    global_sims = [r.score for r in global_results]
    global_ids = [r.id for r in global_results]
    global_novelty = 1.0 - _decay_weighted_mean(
        global_sims, global_ids, timestamps, decay_lambda)

    # Topic-local novelty
    if category_ids is not None and len(category_ids) > 0:
        local_results = index.search(query_embedding, limit=k, filter_ids=category_ids)
        if local_results:
            local_sims = [r.score for r in local_results]
            local_ids = [r.id for r in local_results]
            local_novelty = 1.0 - _decay_weighted_mean(
                local_sims, local_ids, timestamps, decay_lambda)
        else:
            local_novelty = 1.0
        combined = DEFAULT_GLOBAL_WEIGHT * global_novelty + DEFAULT_LOCAL_WEIGHT * local_novelty
    else:
        combined = global_novelty

    # Centroid-distance specificity
    if use_centroid_specificity:
        if centroid is None:
            centroid = corpus_centroid(index)
        if centroid is not None:
            query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-8)
            centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-8)
            centroid_sim = float(query_norm @ centroid_norm)
            centroid_spec = 1.0 - centroid_sim
            combined = (1 - SPECIFICITY_BLEND) * combined + SPECIFICITY_BLEND * centroid_spec

    # Categorical specificity weighting
    if specificity and specificity in SPECIFICITY_WEIGHTS:
        combined *= SPECIFICITY_WEIGHTS[specificity]

    return float(np.clip(combined, 0.0, 1.0))


def batch_novelty(
    embeddings: np.ndarray,
    index: VectorIndex,
    top_k: int | None = None,
    categories: list[set[str] | None] | None = None,
    specificities: list[str | None] | None = None,
    use_centroid_specificity: bool = False,
    timestamps: dict[str, float] | None = None,
    decay_lambda: float = 0.0,
) -> list[float]:
    """Compute novelty scores for a batch of embeddings.

    Pre-computes centroid once for efficiency.
    """
    n = len(embeddings)
    if categories is None:
        categories = [None] * n
    if specificities is None:
        specificities = [None] * n

    centroid = corpus_centroid(index) if use_centroid_specificity else None

    scores = []
    for i in range(n):
        score = novelty_score(
            embeddings[i],
            index,
            top_k=top_k,
            category_ids=categories[i],
            specificity=specificities[i],
            centroid=centroid,
            use_centroid_specificity=use_centroid_specificity,
            timestamps=timestamps,
            decay_lambda=decay_lambda,
        )
        scores.append(score)
    return scores


_nli_cache: dict[str, "CrossEncoder"] = {}


def _get_cross_encoder(model_name: str):
    """Cache CrossEncoder models — loading takes ~2s, reuse is instant."""
    if model_name not in _nli_cache:
        from sentence_transformers import CrossEncoder
        _nli_cache[model_name] = CrossEncoder(model_name)
    return _nli_cache[model_name]


def nli_classify(text_a: str, text_b: str,
                 model_name: str = "cross-encoder/nli-deberta-v3-base",
                 ) -> dict:
    """Classify the relationship between two texts using NLI cross-encoder.

    Cosine similarity can't distinguish paraphrases from contradictions
    (both score ~0.73). NLI models solve this: 99.2% accuracy on 3-class NLI
    for synthetic pairs. On real academic claims, accuracy drops to 53-57%
    (inferential contradictions are hard), but free and fast (13ms/pair).

    Args:
        text_a: First text (premise).
        text_b: Second text (hypothesis).
        model_name: NLI cross-encoder model.

    Returns:
        Dict with 'label' (entailment/contradiction/neutral) and per-class scores.
    """
    ce = _get_cross_encoder(model_name)
    scores = ce.predict([(text_a, text_b)])[0]
    labels = ["contradiction", "entailment", "neutral"]
    label_scores = {l: float(s) for l, s in zip(labels, scores)}
    best_label = max(label_scores, key=label_scores.get)
    return {"label": best_label, **label_scores}


def nli_classify_batch(pairs: list[tuple[str, str]],
                       model_name: str = "cross-encoder/nli-deberta-v3-base",
                       ) -> list[dict]:
    """Classify relationship for multiple text pairs. Batched for efficiency."""
    if not pairs:
        return []
    ce = _get_cross_encoder(model_name)
    all_scores = ce.predict(pairs)
    labels = ["contradiction", "entailment", "neutral"]
    results = []
    for scores in all_scores:
        label_scores = {l: float(s) for l, s in zip(labels, scores)}
        best_label = max(label_scores, key=label_scores.get)
        results.append({"label": best_label, **label_scores})
    return results


_NLI_TO_CLASSIFICATION = {
    "entailment": "KNOWN",
    "neutral": "EXTENDS",
    "contradiction": "NEW",
}


def classify_pairs(
    texts: list[tuple[str, str]],
    scores: list[float],
    known_threshold: float,
    extends_threshold: float,
) -> list[dict]:
    """Classify text pairs as KNOWN/EXTENDS/NEW using cosine + NLI cascade.

    Cosine similarity is fast but can't distinguish paraphrases from
    contradictions. This function uses cosine for easy cases and NLI
    cross-encoder for the ambiguous middle zone:

        score >= known_threshold  → KNOWN  (cosine confident)
        score <  extends_threshold → NEW   (cosine confident)
        otherwise                 → NLI decides (entailment→KNOWN,
                                    contradiction→NEW, neutral→EXTENDS)

    Calibration methodology: sample ~15 pairs per cosine band, run
    nli_classify_batch(), find the entailment cliff (→ known_threshold)
    and contradiction spike (→ extends_threshold). See
    amygdala/experiments/calibration_consumer project_thresholds.md for an example.

    Args:
        texts: List of (text_a, text_b) pairs.
        scores: Cosine similarity score for each pair (same length as texts).
        known_threshold: Above this, classify as KNOWN without NLI.
        extends_threshold: Below this, classify as NEW without NLI.

    Returns:
        List of dicts with 'classification' (KNOWN/EXTENDS/NEW),
        'score' (cosine), and 'nli_label' (str if NLI was used, else None).
    """
    assert len(texts) == len(scores)

    results: list[dict] = [None] * len(texts)  # type: ignore[list-item]

    # Pass 1: cosine handles easy cases
    ambiguous_indices = []
    for i, score in enumerate(scores):
        if score >= known_threshold:
            results[i] = {"classification": "KNOWN", "score": score, "nli_label": None}
        elif score < extends_threshold:
            results[i] = {"classification": "NEW", "score": score, "nli_label": None}
        else:
            ambiguous_indices.append(i)

    # Pass 2: NLI resolves the ambiguous zone in one batch
    if ambiguous_indices:
        ambiguous_texts = [texts[i] for i in ambiguous_indices]
        nli_results = nli_classify_batch(ambiguous_texts)
        for idx, nli in zip(ambiguous_indices, nli_results):
            classification = _NLI_TO_CLASSIFICATION.get(nli["label"], "EXTENDS")
            results[idx] = {
                "classification": classification,
                "score": scores[idx],
                "nli_label": nli["label"],
            }

    return results
