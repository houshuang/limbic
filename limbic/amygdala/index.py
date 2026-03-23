"""SQLite-based chunk/claim storage with incremental updates.

Usage:
    from amygdala.index import Index
    idx = Index("my_index.db")
    idx.add_document("file.md", chunks=[{"content": "text", "metadata": {}}])
    results = idx.search("query", embedding_model=model, limit=10)

    # Standalone connection with best practices:
    from amygdala import connect
    conn = connect("my.db")
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from .search import VectorIndex, Result


def connect(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with best-practice PRAGMAs.

    Applies: WAL journal mode, 30s busy timeout, NORMAL synchronous (with WAL),
    64MB page cache, foreign key enforcement.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory.
        readonly: If True, open with uri=True and ?mode=ro for read-only access.

    Returns:
        Configured sqlite3.Connection with row_factory=sqlite3.Row.
    """
    path_str = str(db_path)
    if readonly and path_str != ":memory:":
        uri = f"file:{path_str}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
    else:
        conn = sqlite3.connect(path_str, timeout=30)
    conn.row_factory = sqlite3.Row
    if path_str != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    path TEXT PRIMARY KEY, mtime REAL NOT NULL,
    collection TEXT DEFAULT 'default', metadata TEXT DEFAULT '{}', indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_path TEXT NOT NULL REFERENCES documents(path),
    content TEXT NOT NULL, metadata TEXT DEFAULT '{}',
    collection TEXT DEFAULT 'default', embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_path);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, content_rowid='rowid', tokenize='porter unicode61'
);
"""


class Index:
    """SQLite-backed document/chunk storage with hybrid search."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = connect(db_path)
        self.conn.executescript(SCHEMA)
        self._vi_cache: dict[str | None, VectorIndex] = {}
        self._vi_dirty = True

    def add_document(self, path: str, chunks: list[dict], collection: str = "default",
                     metadata: dict | None = None, mtime: float | None = None):
        """Add/replace a document. Chunks need 'content', optional 'metadata'/'embedding'."""
        if mtime is None:
            mtime = os.path.getmtime(path) if os.path.exists(path) else time.time()
        existing = self.conn.execute("SELECT mtime FROM documents WHERE path = ?", (path,)).fetchone()
        if existing and existing["mtime"] == mtime:
            return
        self.conn.execute("DELETE FROM chunks WHERE doc_path = ?", (path,))
        self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))
        self.conn.execute(
            "INSERT INTO documents (path,mtime,collection,metadata,indexed_at) VALUES (?,?,?,?,?)",
            (path, mtime, collection, json.dumps(metadata or {}), time.time()))
        for chunk in chunks:
            emb = chunk.get("embedding")
            blob = emb.tobytes() if isinstance(emb, np.ndarray) else None
            self.conn.execute(
                "INSERT INTO chunks (doc_path,content,metadata,collection,embedding) VALUES (?,?,?,?,?)",
                (path, chunk["content"], json.dumps(chunk.get("metadata", {})), collection, blob))
        self.conn.commit()
        self._sync_fts_for(path)
        self._vi_dirty = True

    def add_claims(self, claims: list[dict], collection: str = "claims"):
        """Add claims (idempotent). Each needs 'id', 'content', optional 'metadata'/'embedding'."""
        doc_paths = []
        for c in claims:
            dp = f"claim:{c['id']}"
            doc_paths.append(dp)
            now = time.time()
            self.conn.execute("DELETE FROM chunks WHERE doc_path = ?", (dp,))
            self.conn.execute(
                "INSERT OR REPLACE INTO documents (path,mtime,collection,metadata,indexed_at) VALUES (?,?,?,?,?)",
                (dp, now, collection, json.dumps(c.get("metadata", {})), now))
            emb = c.get("embedding")
            blob = emb.tobytes() if isinstance(emb, np.ndarray) else None
            self.conn.execute(
                "INSERT INTO chunks (doc_path,content,metadata,collection,embedding) VALUES (?,?,?,?,?)",
                (dp, c["content"], json.dumps(c.get("metadata", {})), collection, blob))
        self.conn.commit()
        for dp in doc_paths:
            self._sync_fts_for(dp)
        self._vi_dirty = True

    def _sync_fts_for(self, doc_path: str):
        """Incrementally sync FTS index for chunks belonging to doc_path."""
        # Get current chunk rowids for this doc
        chunk_rows = self.conn.execute(
            "SELECT rowid as rid, content FROM chunks WHERE doc_path = ?", (doc_path,)
        ).fetchall()
        new_rowids = {r["rid"] for r in chunk_rows}
        # Get existing FTS rowids for this doc's chunks
        if new_rowids:
            ph = ",".join("?" * len(new_rowids))
            existing = self.conn.execute(
                f"SELECT rowid FROM chunks_fts WHERE rowid IN ({ph})", list(new_rowids)
            ).fetchall()
            existing_rowids = {r[0] for r in existing}
        else:
            existing_rowids = set()
        # Delete stale FTS entries (chunks that were removed)
        old_rowids = self.conn.execute(
            "SELECT f.rowid FROM chunks_fts f LEFT JOIN chunks c ON f.rowid = c.rowid WHERE c.rowid IS NULL"
        ).fetchall()
        for r in old_rowids:
            self.conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (r[0],))
        # Insert new FTS entries
        for r in chunk_rows:
            if r["rid"] not in existing_rowids:
                self.conn.execute(
                    "INSERT OR REPLACE INTO chunks_fts (rowid, content) VALUES (?,?)",
                    (r["rid"], r["content"])
                )
        self.conn.commit()

    def _build_vector_index(self, collection: str | None = None) -> VectorIndex:
        if not self._vi_dirty and collection in self._vi_cache:
            return self._vi_cache[collection]
        vi = VectorIndex()
        q = "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        params: list = []
        if collection:
            q += " AND collection = ?"
            params.append(collection)
        rows = self.conn.execute(q, params).fetchall()
        if rows:
            ids = [str(r["id"]) for r in rows]
            vecs = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32).copy() for r in rows])
            vi.add(ids, vecs)
        self._vi_cache[collection] = vi
        self._vi_dirty = False
        return vi

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize query for FTS5 MATCH syntax."""
        import re
        tokens = re.findall(r'[\w]+', query, re.UNICODE)
        tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
        return " OR ".join(tokens) if tokens else ""

    def _fts_search(self, query: str, limit: int = 10, collection: str | None = None) -> list[Result]:
        query = self._sanitize_query(query)
        if not query:
            return []
        try:
            q = "SELECT c.id, c.content, c.metadata, rank FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid WHERE chunks_fts MATCH ?"
            params: list = [query]
            if collection:
                q += " AND c.collection = ?"
                params.append(collection)
            q += " ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = self.conn.execute(q, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [Result(id=str(r["id"]), score=-r["rank"], content=r["content"],
                       metadata=json.loads(r["metadata"]), source="fts") for r in rows]

    def search(self, query: str, embedding_model=None, limit: int = 10,
               collection: str | None = None, hybrid: bool = True) -> list[Result]:
        """Search. Uses hybrid (vector+FTS5+RRF) if embedding_model provided, else FTS5 only."""
        if not (hybrid and embedding_model):
            return self._fts_search(query, limit, collection)
        query_vec = embedding_model.embed(query)
        vi = self._build_vector_index(collection)
        vec_results = vi.search(query_vec, limit=limit * 5)
        fts_results = self._fts_search(query, limit * 5, collection)
        rrf_k = 60
        scores: dict[str, float] = {}
        for rank, r in enumerate(vec_results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, r in enumerate(fts_results):
            scores[str(r.id)] = scores.get(str(r.id), 0.0) + 1.0 / (rrf_k + rank + 1)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        return self._hydrate_results(sorted_ids, scores)

    def _hydrate_results(self, ids: list[str], scores: dict[str, float]) -> list[Result]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, content, metadata FROM chunks WHERE id IN ({ph})", [int(i) for i in ids]).fetchall()
        rm = {str(r["id"]): r for r in rows}
        return [Result(id=i, score=scores.get(i, 0.0), content=rm[i]["content"],
                       metadata=json.loads(rm[i]["metadata"]), source="hybrid")
                for i in ids if i in rm]

    def get_stats(self) -> dict:
        return {
            "chunks": self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "documents": self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "collections": [r[0] for r in self.conn.execute("SELECT DISTINCT collection FROM chunks").fetchall()],
            "embeddings": self.conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0],
        }

    def needs_reindex(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        existing = self.conn.execute("SELECT mtime FROM documents WHERE path = ?", (path,)).fetchone()
        return existing is None or existing["mtime"] != os.path.getmtime(path)
