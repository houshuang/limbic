"""Persistent SQLite-backed embedding cache.

Stores raw (pre-whitening) embeddings keyed by text hash and model name.
Use with EmbeddingModel(cache_path="embeddings.db") for warm startup.

At 20K embeddings: cold embed ~48s, warm cache lookup ~585ms (83x speedup).
Storage overhead: ~2.2 KB per 384-dim float32 embedding. (Exp 20)
"""

import hashlib
import time

import numpy as np

from .index import connect


class PersistentEmbeddingCache:
    """SQLite-backed persistent embedding cache for raw embeddings.

    Usage:
        cache = PersistentEmbeddingCache("cache.db", "paraphrase-multilingual-MiniLM-L12-v2")
        cache.put("some text", embedding_vector)
        vec = cache.get("some text")  # returns np.ndarray or None

    Or use via EmbeddingModel(cache_path="cache.db") for automatic integration.
    """

    def __init__(self, db_path: str, model_name: str):
        self.model_name = model_name
        self.conn = connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (text_hash, model_name)
            )
        """)
        self.conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model_name = ?",
            (self._hash(text), self.model_name),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32).copy()

    def put(self, text: str, embedding: np.ndarray):
        self.conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, model_name, embedding, created_at) "
            "VALUES (?, ?, ?, ?)",
            (self._hash(text), self.model_name, embedding.astype(np.float32).tobytes(), time.time()),
        )
        self.conn.commit()

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Return (results, miss_indices). results[i] is np.ndarray or None."""
        results: list[np.ndarray | None] = [None] * len(texts)
        hashes = [self._hash(t) for t in texts]
        hash_to_indices: dict[str, list[int]] = {}
        for i, h in enumerate(hashes):
            hash_to_indices.setdefault(h, []).append(i)

        unique_hashes = list(hash_to_indices.keys())
        found: dict[str, bytes] = {}
        for start in range(0, len(unique_hashes), 500):
            chunk = unique_hashes[start:start + 500]
            ph = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT text_hash, embedding FROM embedding_cache "
                f"WHERE text_hash IN ({ph}) AND model_name = ?",
                chunk + [self.model_name],
            ).fetchall()
            for row in rows:
                found[row["text_hash"]] = row["embedding"]

        miss_indices: list[int] = []
        for h, indices in hash_to_indices.items():
            if h in found:
                vec = np.frombuffer(found[h], dtype=np.float32).copy()
                for i in indices:
                    results[i] = vec
            else:
                miss_indices.extend(indices)
        return results, sorted(miss_indices)

    def put_batch(self, texts: list[str], embeddings: np.ndarray):
        now = time.time()
        rows = [
            (self._hash(t), self.model_name, embeddings[i].astype(np.float32).tobytes(), now)
            for i, t in enumerate(texts)
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, model_name, embedding, created_at) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM embedding_cache WHERE model_name = ?", (self.model_name,))
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM embedding_cache WHERE model_name = ?",
            (self.model_name,),
        ).fetchone()[0]
