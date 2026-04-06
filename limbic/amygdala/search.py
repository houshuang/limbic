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

    # LLM query expansion (lex/vec/hyde variants) — 3-5x score improvement
    expanded = expand_query("database lock problems")
    # [{"type": "lex", "query": "deadlock contention"}, {"type": "hyde", "query": "..."}, ...]

    # Multi-list RRF with contribution tracing
    fused, traces = multi_list_rrf([vec_results, fts_results, ...], ["vec", "fts", ...])
"""

import json
import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


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
        return " OR ".join(f'"{t}"' for t in tokens) if tokens else ""

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


def dedup_by(results: list[Result], key_fn: Callable[[Result], str]) -> list[Result]:
    """Keep only the first (highest-scored) result per group defined by key_fn.

    Assumes results are already sorted by score descending.
    """
    seen: dict[str, Result] = {}
    for r in results:
        key = key_fn(r)
        if key not in seen:
            seen[key] = r
    return list(seen.values())


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


# ---------------------------------------------------------------------------
# Multi-list RRF with top-rank bonuses and contribution tracing
# ---------------------------------------------------------------------------

@dataclass
class RRFContribution:
    """One contribution to a result's RRF score."""
    list_label: str
    rank: int       # 1-indexed
    contribution: float


@dataclass
class TracedResult:
    """Search result with RRF contribution traces."""
    id: str
    score: float
    traces: list[RRFContribution] = field(default_factory=list)


def multi_list_rrf(
    ranked_lists: list[list],
    list_labels: list[str],
    k: int = 60,
    id_fn: Callable | None = None,
    top_rank_bonus: bool = True,
) -> list[TracedResult]:
    """Reciprocal Rank Fusion across N ranked lists with contribution tracing.

    Accepts lists of any type — use id_fn to extract the ID from each item.
    Defaults to treating items as dicts with an "id" key.

    Top-rank bonuses (QMD-style): +0.05 for rank 1, +0.02 for ranks 2-3.

    Args:
        ranked_lists: N lists, each in ranked order (best first).
        list_labels: Human-readable label per list (e.g. "vec:original", "fts:hyde").
        k: RRF constant (default 60).
        id_fn: Extract ID from an item. Defaults to item["id"] or item.id.
        top_rank_bonus: Add bonuses for top-ranked items.

    Returns:
        List of TracedResult sorted by descending score.
    """
    if id_fn is None:
        def id_fn(item):
            if isinstance(item, dict):
                return item.get("id") or item.get("chunk_id")
            return getattr(item, "id", None)

    scores: dict = {}
    traces: dict = {}

    for list_idx, ranked in enumerate(ranked_lists):
        label = list_labels[list_idx] if list_idx < len(list_labels) else f"list_{list_idx}"
        for rank, item in enumerate(ranked):
            item_id = id_fn(item)
            if item_id is None:
                continue
            contribution = 1.0 / (k + rank + 1)
            scores[item_id] = scores.get(item_id, 0.0) + contribution
            traces.setdefault(item_id, []).append(
                RRFContribution(list_label=label, rank=rank + 1, contribution=contribution)
            )

    if top_rank_bonus:
        for item_id, contribs in traces.items():
            top_rank = min(c.rank for c in contribs)
            if top_rank == 1:
                scores[item_id] += 0.05
            elif top_rank <= 3:
                scores[item_id] += 0.02

    return sorted(
        [TracedResult(id=item_id, score=score, traces=traces.get(item_id, []))
         for item_id, score in scores.items()],
        key=lambda r: r.score,
        reverse=True,
    )


# ---------------------------------------------------------------------------
# LLM query expansion (lex / vec / hyde variants)
# ---------------------------------------------------------------------------

_EXPAND_PROMPT = """Expand this search query into 5 sub-queries for a hybrid search engine.

Query: {query}

Output format (5 lines, no extra text):
lex: keywords variant 1 (different technical terms)
lex: keywords variant 2 (alternate phrasing)
vec: semantic rephrase of the intent
hyde: hypothetical document excerpt that would answer the query
hyde: another hypothetical excerpt, different angle"""

_EXPAND_LINE_RE = re.compile(r'^(?:\d+\.\s*)?(lex|vec|hyde):\s*(.+)$', re.IGNORECASE)


@dataclass
class ExpandedQuery:
    """A single expanded sub-query."""
    type: str   # "lex", "vec", or "hyde"
    query: str


def expand_query(
    query: str,
    model: str = "gemini3-flash",
    fallback_model: str = "gemini25-flash",
    domain_context: str | None = None,
) -> list[ExpandedQuery]:
    """Expand a search query into lex/vec/hyde sub-queries using an LLM.

    Returns a list of ExpandedQuery. Falls back to fallback_model if primary fails.
    Empty list on total failure (never raises).

    Query types:
      - lex: keyword variants for full-text/BM25 search (different vocabulary)
      - vec: semantic rephrase for vector/embedding search (different framing)
      - hyde: hypothetical document excerpts (bridges query-document vocab gap)

    Args:
        query: The user's search query.
        model: Primary LLM model for expansion.
        fallback_model: Fallback if primary fails.
        domain_context: Optional domain hint appended to the prompt, e.g.
            "The corpus contains Nordic education research claims using
            academic terminology (didactics, Bildung, tilpasset opplæring)."
            This steers lex/hyde variants toward domain-specific vocabulary.
    """
    from .llm import generate_sync

    prompt = _EXPAND_PROMPT.format(query=query)
    if domain_context:
        prompt += f"\n\nDomain context: {domain_context}"
    system = "Output exactly 5 lines. No thinking, no explanation."

    for m in [model, fallback_model]:
        try:
            raw = generate_sync(prompt=prompt, system_prompt=system, model=m)
            queries = []
            for line in raw.strip().splitlines():
                match = _EXPAND_LINE_RE.match(line.strip())
                if match:
                    queries.append(ExpandedQuery(
                        type=match.group(1).lower(),
                        query=match.group(2).strip(),
                    ))
            if queries:
                return queries
        except Exception as e:
            log.debug("expand_query with %s failed: %s", m, e)
            continue

    return []


def expanded_hybrid_search(
    query: str,
    vector_index: "VectorIndex",
    fts_index: "FTS5Index",
    embed_fn: Callable[[str], np.ndarray],
    limit: int = 10,
    filter_ids: set[str] | None = None,
    domain_context: str | None = None,
) -> list[TracedResult]:
    """One-call expanded hybrid search — the easy integration path.

    Runs expand_query, feeds sub-queries through vector+FTS backends,
    and fuses everything with multi_list_rrf. Returns TracedResults
    with full contribution tracing.

    Args:
        query: User's natural-language query.
        vector_index: A VectorIndex instance.
        fts_index: An FTS5Index instance.
        embed_fn: Function that embeds a string → numpy array (e.g. model.embed).
        limit: Number of results.
        filter_ids: Optional set of IDs to restrict results to.
        domain_context: Domain hint for query expansion (see expand_query).

    Returns:
        List of TracedResult sorted by descending score.

    Example::

        from limbic.amygdala import EmbeddingModel, expanded_hybrid_search

        model = EmbeddingModel()
        results = expanded_hybrid_search(
            "effect of digital tools on learning",
            vector_index=vi,
            fts_index=fts,
            embed_fn=model.embed,
            domain_context="Nordic education research with Scandinavian terminology",
        )
    """
    fetch = limit * 5

    # First-pass: original query through both backends
    query_vec = embed_fn(query)
    vec_results = vector_index.search(query_vec, limit=fetch, filter_ids=filter_ids)
    fts_results = fts_index.search(query, limit=fetch)
    if filter_ids is not None:
        fts_results = [r for r in fts_results if r.id in filter_ids]

    ranked_lists = [vec_results, fts_results]
    labels = ["vec:original", "fts:original"]

    # Expand and run sub-queries
    expanded = expand_query(query, domain_context=domain_context)
    for eq in expanded:
        if eq.type == "lex":
            fts_hits = fts_index.search(eq.query, limit=fetch)
            if filter_ids is not None:
                fts_hits = [r for r in fts_hits if r.id in filter_ids]
            ranked_lists.append(fts_hits)
            labels.append(f"fts:{eq.query[:30]}")
        elif eq.type in ("vec", "hyde"):
            vec_hits = vector_index.search(
                embed_fn(eq.query), limit=fetch, filter_ids=filter_ids,
            )
            ranked_lists.append(vec_hits)
            labels.append(f"vec:{eq.type}:{eq.query[:25]}")

    return multi_list_rrf(ranked_lists, labels)[:limit]


def strong_signal(
    top_scores: list[float],
    threshold: float = 0.82,
    gap: float = 0.12,
) -> bool:
    """Check if first-pass search already has a confident top result.

    Use this to skip expensive LLM query expansion when the top result
    is strong AND clearly separated from the runner-up.

    Args:
        top_scores: Scores of at least the top 2 results (descending).
        threshold: Minimum score for the top result.
        gap: Minimum gap between top and second result.

    Returns:
        True if expansion can safely be skipped.
    """
    if len(top_scores) < 2:
        return False
    return top_scores[0] >= threshold and (top_scores[0] - top_scores[1]) >= gap
