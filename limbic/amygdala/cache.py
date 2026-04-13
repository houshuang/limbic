"""Persistent SQLite-backed caches.

Two cache classes share this module because they share the same SQLite-backed
pattern (connection via `amygdala.connect`, idempotent schema, simple key
lookup) but store different payload types:

- `PersistentEmbeddingCache` — numpy float32 vectors keyed by (text_hash, model).
- `PayloadCache` — JSON payloads keyed by (key, source) with TTL semantics,
  intended for external-API responses (Wikidata, GeoNames, Pleiades, VIAF, …).

Both accept ":memory:" or a file path and rely on `amygdala.connect` to apply
the project's standard PRAGMAs (WAL, busy_timeout=30s, etc.).
"""

import hashlib
import json
import time
from typing import Callable, Any

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

    def __init__(self, db_path: str, model_name: str, truncate_dim: int | None = None):
        self.model_name = self._build_key(model_name, truncate_dim)
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
    def _build_key(model_name: str, truncate_dim: int | None) -> str:
        if truncate_dim is not None:
            return f"{model_name}:dim={truncate_dim}"
        return model_name

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


class PayloadCache:
    """Generic SQLite-backed cache for external API payloads (JSON) with TTL.

    Intended for wrapping external services where responses are JSON-able and
    change infrequently: Wikidata entities, GeoNames place lookups, Pleiades
    gazetteer payloads, VIAF authority records, and so on.

    Each cache instance is scoped to a `source` string (e.g., "wikidata_get",
    "wikidata_search", "geonames"). Multiple sources can share one SQLite file
    without key collisions; clearing/invalidating is per-source.

    Usage:
        cache = PayloadCache("external_cache.db", source="wikidata_get")
        def fetch():
            return requests.get(...).json()
        payload = cache.get_or_fetch("Q42", fetch)  # cache miss → fetch + store
        payload = cache.get_or_fetch("Q42", fetch)  # hit → no HTTP

    Concurrency: WAL mode (configured by `amygdala.connect`) allows multiple
    reader connections and a single writer. Opening multiple `PayloadCache`
    instances against the same db_path is safe — writes will queue via the
    30-second busy_timeout. Each instance owns its connection; call `close()`
    or use the context-manager form when done.

    Lifecycle:
        with PayloadCache("cache.db", source="wikidata_get") as cache:
            payload = cache.get_or_fetch("Q42", fetch_fn)
    """

    DEFAULT_TTL_SECONDS = 30 * 86400  # 30 days

    def __init__(self, db_path: str, source: str, default_ttl_seconds: int | None = None):
        if not source:
            raise ValueError("PayloadCache requires a non-empty source tag")
        self.source = source
        self.default_ttl = default_ttl_seconds if default_ttl_seconds is not None else self.DEFAULT_TTL_SECONDS
        self.conn = connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS payload_cache (
                key TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL,
                fetched_at INTEGER NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                PRIMARY KEY (key, source)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_payload_cache_source ON payload_cache(source)"
        )
        self.conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Batch lookup. Returns only the fresh (non-expired) hits.

        Missing or expired keys are simply absent from the returned dict — the
        caller can diff against the input list to find keys that still need
        fetching. Chunks queries in groups of 500 to respect SQLite's
        parameter limit.
        """
        if not keys:
            return {}
        now = int(time.time())
        out: dict[str, Any] = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT key, payload, fetched_at, ttl_seconds FROM payload_cache "
                f"WHERE source = ? AND key IN ({placeholders})",
                [self.source, *chunk],
            ).fetchall()
            for row in rows:
                if now - row["fetched_at"] <= row["ttl_seconds"]:
                    out[row["key"]] = json.loads(row["payload"])
        return out

    def put_many(self, items: dict[str, Any], ttl_seconds: int | None = None) -> None:
        """Batch insert. All entries get the same TTL."""
        if not items:
            return
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = int(time.time())
        rows = [
            (k, self.source, json.dumps(v), now, ttl)
            for k, v in items.items()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO payload_cache "
            "(key, source, payload, fetched_at, ttl_seconds) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None  # type: ignore[assignment]

    def __enter__(self) -> "PayloadCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get(self, key: str, allow_stale: bool = False) -> Any | None:
        """Return cached payload if present and fresh, else None.

        Set allow_stale=True to return expired entries (e.g., for fallback when
        the upstream API is unavailable).
        """
        row = self.conn.execute(
            "SELECT payload, fetched_at, ttl_seconds FROM payload_cache "
            "WHERE key = ? AND source = ?",
            (key, self.source),
        ).fetchone()
        if row is None:
            return None
        if not allow_stale:
            age = int(time.time()) - row["fetched_at"]
            if age > row["ttl_seconds"]:
                return None
        return json.loads(row["payload"])

    def put(self, key: str, payload: Any, ttl_seconds: int | None = None) -> None:
        """Store payload with TTL. Overwrites any existing entry for (key, source)."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self.conn.execute(
            "INSERT OR REPLACE INTO payload_cache "
            "(key, source, payload, fetched_at, ttl_seconds) VALUES (?, ?, ?, ?, ?)",
            (key, self.source, json.dumps(payload), int(time.time()), ttl),
        )
        self.conn.commit()

    def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl_seconds: int | None = None,
    ) -> Any:
        """Return cached payload or call fetch_fn, store result, return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        payload = fetch_fn()
        self.put(key, payload, ttl_seconds)
        return payload

    def delete(self, key: str) -> None:
        self.conn.execute(
            "DELETE FROM payload_cache WHERE key = ? AND source = ?",
            (key, self.source),
        )
        self.conn.commit()

    def clear_source(self) -> None:
        """Remove all entries for this cache's source. Leaves other sources alone."""
        self.conn.execute("DELETE FROM payload_cache WHERE source = ?", (self.source,))
        self.conn.commit()

    def invalidate_before(self, cutoff_ts: int) -> int:
        """Remove entries fetched before cutoff_ts (unix seconds). Returns rowcount."""
        cur = self.conn.execute(
            "DELETE FROM payload_cache WHERE source = ? AND fetched_at < ?",
            (self.source, cutoff_ts),
        )
        self.conn.commit()
        return cur.rowcount

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM payload_cache WHERE source = ?", (self.source,)
        ).fetchone()[0]

    def stats(self) -> dict:
        now = int(time.time())
        total = self.count()
        fresh = self.conn.execute(
            "SELECT COUNT(*) FROM payload_cache "
            "WHERE source = ? AND ? - fetched_at <= ttl_seconds",
            (self.source, now),
        ).fetchone()[0]
        return {
            "source": self.source,
            "total": total,
            "fresh": fresh,
            "expired": total - fresh,
        }
