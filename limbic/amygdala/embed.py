"""Embedding with optional whitening, Matryoshka truncation, genericization, and caching.

Default model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim).
Chosen over all-MiniLM-L6-v2 based on experiments (2026-03-19):
- 80% vs 75% classification accuracy on calibration set
- 0.84 vs 0.16 Norwegian cross-lingual quality
- Fastest of all tested models (0.30s for 150 texts)
- Uniquely separates contradictions from paraphrases (0.15 gap)

Whitening is opt-in. It helps domain-homogeneous corpora (all claims about
education) but hurts diverse corpora. Three modes:
- Soft-ZCA (recommended): EmbeddingModel(whiten_epsilon=0.1) — regularized,
  preserves all dimensions. +32% NN-separation on domain data. (Exp 12)
- All-but-the-top: EmbeddingModel(whiten_abt=1) — removes top D principal
  components. Matches Soft-ZCA quality (+27%) with simpler math. (Exp 21)
- PCA (legacy): EmbeddingModel(whiten_dims=128) — truncates dimensions.
  +24% discrimination gap at 128d. (Exp 11)
"""

import re
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WhiteningParams:
    """Stored whitening transform: mean + projection matrix."""
    mean: np.ndarray
    W: np.ndarray  # shape: (original_dim, whiten_dims)


DEFAULT_PATTERNS = [
    (re.compile(r'https?://\S+'), ''),
    (re.compile(r'\S+@\S+\.\S+'), ''),
    (re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'), ''),
    (re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b'), ''),
    (re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', re.IGNORECASE), ''),
    (re.compile(r'\b20[0-9]{2}(?:\s*[-–]\s*20[0-9]{2})?\b'), ''),
    (re.compile(r'\b19[0-9]{2}(?:\s*[-–]\s*19[0-9]{2})?\b'), ''),
    (re.compile(r'\d[\d,.]*\s*%'), ''),
    (re.compile(r'[$€£]\s*\d[\d,.]*(?:\s*(?:million|billion|trillion|[MBT]))?', re.IGNORECASE), ''),
    (re.compile(r'\b\d[\d,.]*\b'), ''),
    (re.compile(r'\s{2,}'), ' '),
]

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingModel:
    """Sentence embedding with optional whitening, Matryoshka truncation, and genericization.

    Usage:
        model = EmbeddingModel()                      # multilingual, 384-dim
        model = EmbeddingModel(truncate_dim=256)       # Matryoshka truncation
        model = EmbeddingModel(whiten_epsilon=0.1)     # Soft-ZCA whitening (recommended)
        model = EmbeddingModel(whiten_abt=1)           # All-but-the-top whitening
        model = EmbeddingModel(whiten_dims=128)        # PCA whitening (legacy)
        model = EmbeddingModel(cache_path="emb.db")    # persistent SQLite cache
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        whiten_dims: int | None = None,
        whiten_epsilon: float | None = None,
        whiten_abt: int | None = None,
        truncate_dim: int | None = None,
        genericize: bool = False,
        genericize_patterns: list | None = None,
        cache_size: int = 4096,
        cache_path: str | None = None,
    ):
        self.model_name = model_name
        self.whiten_dims = whiten_dims
        self.whiten_epsilon = whiten_epsilon
        self.whiten_abt = whiten_abt
        self.truncate_dim = truncate_dim
        self.genericize = genericize
        self.patterns = genericize_patterns or DEFAULT_PATTERNS
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._model = None
        self._whitening: WhiteningParams | None = None
        self._persistent_cache = None
        if cache_path is not None:
            from .cache import PersistentEmbeddingCache
            self._persistent_cache = PersistentEmbeddingCache(str(cache_path), model_name)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, truncate_dim=self.truncate_dim)
        return self._model

    def _genericize(self, text: str) -> str:
        """Strip numbers, dates, URLs, etc. to focus on semantic content."""
        result = text
        for pattern, replacement in self.patterns:
            result = pattern.sub(replacement, result)
        return result.strip()

    def _prepare_text(self, text: str) -> str:
        if self.genericize:
            return self._genericize(text)
        return text

    def _apply_whitening(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA whitening: center, project, renormalize."""
        if self._whitening is None:
            return embeddings
        centered = embeddings - self._whitening.mean
        whitened = centered @ self._whitening.W
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        return whitened / np.maximum(norms, 1e-8)

    def fit_whitening(self, texts: list[str]) -> WhiteningParams:
        """Compute whitening parameters from a corpus.

        Opt-in: helps domain-homogeneous corpora (all claims about one field).
        Hurts diverse corpora where unrelated items already have low similarity.

        Uses Soft-ZCA when whiten_epsilon is set (recommended), PCA when
        whiten_dims is set (legacy).
        """
        if self.whiten_epsilon is None and self.whiten_dims is None and self.whiten_abt is None:
            raise ValueError(
                "Set whiten_epsilon, whiten_dims, or whiten_abt to use whitening: "
                "EmbeddingModel(whiten_epsilon=0.1) or EmbeddingModel(whiten_dims=256) "
                "or EmbeddingModel(whiten_abt=1)"
            )
        raw = self._raw_embed_batch(texts)
        mean = raw.mean(axis=0)
        centered = raw - mean
        n = len(texts)
        cov = centered.T @ centered / n
        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        if self.whiten_abt is not None:
            # All-but-the-top: remove top D principal components
            # W = I - U_D @ U_D^T where U_D are top D eigenvectors of covariance
            D = min(self.whiten_abt, len(S))
            W = np.eye(len(mean)) - U[:, :D] @ U[:, :D].T
        elif self.whiten_epsilon is not None:
            # Soft-ZCA: W = U @ diag(1/sqrt(S + ε)) @ U^T
            # Preserves all dimensions, regularized by ε
            D_inv = np.diag(1.0 / np.sqrt(S + self.whiten_epsilon))
            W = U @ D_inv @ U.T
        else:
            # PCA whitening (legacy): truncate to k dimensions
            k = min(self.whiten_dims, len(S))
            W = U[:, :k] @ np.diag(1.0 / np.sqrt(S[:k] + 1e-8))

        self._whitening = WhiteningParams(mean=mean, W=W)
        self._cache.clear()
        return self._whitening

    def set_whitening(self, params: WhiteningParams):
        """Load pre-computed whitening parameters."""
        self._whitening = params
        self._cache.clear()

    def _raw_embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed without whitening or caching. Returns L2-normalized vectors."""
        model = self._load_model()
        prepared = [self._prepare_text(t) for t in texts]
        return model.encode(
            prepared,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        ).astype(np.float32)

    def _get_raw_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get raw embeddings, using persistent cache when available."""
        if self._persistent_cache is None:
            return self._raw_embed_batch(texts)
        prepared = [self._prepare_text(t) for t in texts]
        cached, miss_indices = self._persistent_cache.get_batch(prepared)
        if not miss_indices:
            return np.vstack(cached)
        miss_texts = [texts[i] for i in miss_indices]
        new_raws = self._raw_embed_batch(miss_texts)
        miss_prepared = [prepared[i] for i in miss_indices]
        self._persistent_cache.put_batch(miss_prepared, new_raws)
        for k, i in enumerate(miss_indices):
            cached[i] = new_raws[k]
        return np.vstack(cached)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns 1-D array."""
        prepared = self._prepare_text(text)
        if prepared in self._cache:
            self._cache.move_to_end(prepared)
            return self._cache[prepared]

        raw = self._get_raw_embeddings([text])
        vec = self._apply_whitening(raw)[0]

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[prepared] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (N, dim) array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        for i, text in enumerate(texts):
            prepared = self._prepare_text(text)
            if prepared in self._cache:
                self._cache.move_to_end(prepared)
                results[i] = self._cache[prepared]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            raw = self._get_raw_embeddings(uncached_texts)
            whitened = self._apply_whitening(raw)
            for j, idx in enumerate(uncached_indices):
                vec = whitened[j]
                results[idx] = vec
                prepared = self._prepare_text(uncached_texts[j])
                if len(self._cache) >= self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[prepared] = vec

        return np.vstack(results)

    @property
    def dim(self) -> int:
        """Output embedding dimension."""
        if self._whitening is not None:
            return self._whitening.W.shape[1]
        if self.whiten_epsilon is not None:
            # Soft-ZCA preserves all dims, but whitening not yet fitted
            pass
        if self.truncate_dim is not None:
            return self.truncate_dim
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
