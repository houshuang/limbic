"""Unified search: numpy vector, SQLite FTS5, hybrid RRF, and cross-encoder reranking.

Usage:
    vi = VectorIndex()
    vi.add(ids, embeddings)
    results = vi.search(query_embedding, limit=10)

    fts = FTS5Index(db_path)
    fts.add(id="doc1", content="some text", metadata={"source": "arxiv"})
    results = fts.search("some text", limit=10)

    hybrid = HybridSearch(vector_index=vi, fts_index=fts)
    results = hybrid.search(query_embedding, query_text, limit=10)

    # Cross-encoder reranking (5-15% relevance boost on top of any search)
    reranked = rerank(query_text, results, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    """A single search result."""
    id: str
    score: float
    content: str = ""
    metadata: dict = field(default_factory=dict)
    source: str = ""  # "vector", "fts", "hybrid"


class VectorIndex:
    """In-memory numpy matrix for brute-force cosine similarity search."""

    def __init__(self):
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids: list[str], embeddings: np.ndarray):
        """Add embeddings to the index. Will be L2-normalized internally."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"ids length ({len(ids)}) must match embeddings rows ({embeddings.shape[0]})"
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms
        if self._matrix is None:
            self._matrix = normalized
            self._ids = list(ids)
        else:
            self._matrix = np.vstack([self._matrix, normalized])
            self._ids.extend(ids)

    def search(self, query_embedding: np.ndarray, limit: int = 10,
               filter_ids: set[str] | None = None) -> list[Result]:
        """Search by cosine similarity. Returns Results sorted by score descending."""
        if self._matrix is None or not self._ids:
            return []
        query = query_embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        scores = self._matrix @ query
        if filter_ids is not None:
            mask = np.array([id_ in filter_ids for id_ in self._ids])
            scores = np.where(mask, scores, -2.0)
        if limit >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, limit)[:limit]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [Result(id=self._ids[i], score=float(scores[i]), source="vector")
                for i in top_indices if scores[i] > -2.0][:limit]

    @property
    def size(self) -> int:
        return len(self._ids)


class FTS5Index:
    """SQLite FTS5 full-text search index."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS fts_docs (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT DEFAULT '{}');
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(content, content_rowid='rowid', tokenize='porter unicode61');
        """)

    def add(self, id: str, content: str, metadata: dict | None = None):
        meta_str = json.dumps(metadata or {})
        self.conn.execute("INSERT OR REPLACE INTO fts_docs (id, content, metadata) VALUES (?,?,?)", (id, content, meta_str))
        rowid = self.conn.execute("SELECT rowid FROM fts_docs WHERE id = ?", (id,)).fetchone()[0]
        self.conn.execute("INSERT OR REPLACE INTO fts_index (rowid, content) VALUES (?,?)", (rowid, content))
        self.conn.commit()

    def add_batch(self, docs: list[dict]):
        """Add multiple docs. Each needs 'id', 'content', optional 'metadata'."""
        for d in docs:
            self.conn.execute("INSERT OR REPLACE INTO fts_docs (id,content,metadata) VALUES (?,?,?)",
                              (d["id"], d["content"], json.dumps(d.get("metadata", {}))))
        self.conn.commit()
        # Sync FTS incrementally instead of full rebuild
        for d in docs:
            rowid = self.conn.execute(
                "SELECT rowid FROM fts_docs WHERE id = ?", (d["id"],)
            ).fetchone()[0]
            self.conn.execute(
                "INSERT OR REPLACE INTO fts_index (rowid, content) VALUES (?,?)",
                (rowid, d["content"])
            )
        self.conn.commit()

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize a query for FTS5 MATCH syntax.

        Raw text with hyphens, parentheses, numbers, or special chars
        causes FTS5 OperationalError. Keep unicode word characters (handles
        Norwegian æøå, German ü, etc.) and join with OR.
        """
        tokens = re.findall(r'[\w]+', query, re.UNICODE)
        tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
        return " OR ".join(tokens) if tokens else ""

    def search(self, query: str, limit: int = 10) -> list[Result]:
        sanitized = self._sanitize_query(query)
        if not sanitized:
            return []
        try:
            rows = self.conn.execute("""
                SELECT d.id, d.content, d.metadata, rank FROM fts_index fi
                JOIN fts_docs d ON d.rowid = fi.rowid WHERE fts_index MATCH ? ORDER BY rank LIMIT ?
            """, (sanitized, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [Result(id=r["id"], score=-r["rank"], content=r["content"],
                       metadata=json.loads(r["metadata"]), source="fts") for r in rows]

    @property
    def size(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM fts_docs").fetchone()[0]


class HybridSearch:
    """Combines vector + FTS5 search with Reciprocal Rank Fusion."""

    def __init__(self, vector_index: VectorIndex, fts_index: FTS5Index, rrf_k: int = 60):
        self.vector_index = vector_index
        self.fts_index = fts_index
        self.rrf_k = rrf_k

    def search(self, query_embedding: np.ndarray, query_text: str,
               limit: int = 10, filter_ids: set[str] | None = None) -> list[Result]:
        fetch = limit * 5
        vec_results = self.vector_index.search(query_embedding, limit=fetch, filter_ids=filter_ids)
        fts_results = self.fts_index.search(query_text, limit=fetch)
        if filter_ids is not None:
            fts_results = [r for r in fts_results if r.id in filter_ids]

        scores: dict[str, float] = {}
        content_map: dict[str, str] = {}
        metadata_map: dict[str, dict] = {}
        for rank, r in enumerate(vec_results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, r in enumerate(fts_results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            content_map[r.id] = r.content
            metadata_map[r.id] = r.metadata

        top_ids = sorted(scores, key=scores.get, reverse=True)[:limit]

        # Hydrate content for vector-only hits from FTS backing store
        missing = [id_ for id_ in top_ids if id_ not in content_map]
        if missing:
            ph = ",".join("?" * len(missing))
            try:
                rows = self.fts_index.conn.execute(
                    f"SELECT id, content, metadata FROM fts_docs WHERE id IN ({ph})",
                    missing,
                ).fetchall()
                for r in rows:
                    content_map[r["id"]] = r["content"]
                    metadata_map[r["id"]] = json.loads(r["metadata"])
            except Exception:
                pass

        return [Result(id=id_, score=scores[id_], content=content_map.get(id_, ""),
                       metadata=metadata_map.get(id_, {}), source="hybrid")
                for id_ in top_ids]


_rerank_cache: dict[str, "CrossEncoder"] = {}


def rerank(query: str, results: list[Result],
           model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
           top_n: int | None = None) -> list[Result]:
    """Re-score results with a cross-encoder for 5-15% relevance boost.

    The cross-encoder sees (query, document) pairs and produces a relevance
    score that's more accurate than bi-encoder cosine similarity, at the
    cost of being O(N) per query instead of O(1).

    Args:
        query: The search query text.
        results: Results from any search method (vector, FTS5, hybrid).
        model_name: Cross-encoder model. Default is tiny (22M params).
        top_n: Only rerank the top N results (None = all).

    Returns:
        Results re-sorted by cross-encoder score.
    """
    if not results:
        return results
    empty = [r for r in results if not r.content]
    if empty:
        raise ValueError(
            f"rerank() requires results with content, but {len(empty)}/{len(results)} "
            f"have empty content (e.g. from VectorIndex). Hydrate content before reranking."
        )
    if model_name not in _rerank_cache:
        from sentence_transformers import CrossEncoder
        _rerank_cache[model_name] = CrossEncoder(model_name)
    ce = _rerank_cache[model_name]
    to_rerank = results[:top_n] if top_n else results
    pairs = [(query, r.content) for r in to_rerank]
    scores = ce.predict(pairs)
    reranked = []
    for r, score in zip(to_rerank, scores):
        reranked.append(Result(id=r.id, score=float(score), content=r.content,
                               metadata=r.metadata, source=r.source))
    reranked.sort(key=lambda r: r.score, reverse=True)
    if top_n and top_n < len(results):
        reranked.extend(results[top_n:])
    return reranked
