"""Experiment 20: SQLite-backed persistent embedding cache.

Amygdala's EmbeddingModel has an in-memory LRU cache. For large corpora
(a 60K claims corpus), re-embedding on every process restart costs ~30-60s.
A SQLite-backed persistent cache would make warm startup instant.

This experiment:
1. Implements a PersistentEmbeddingCache prototype backed by SQLite
2. Benchmarks cold embed, cache write, warm lookup at 1K/5K/10K/20K scale
3. Measures per-embedding latency, DB file size, memory comparison
4. Validates cache correctness (np.allclose)
5. Tests partial cache hit scenario (80% cached, 20% cold)

Key question: what's the speedup from persistent cache, and is the
SQLite overhead acceptable?
"""

import gc
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from amygdala import EmbeddingModel, connect

OTAK_DB = Path(os.environ.get("AMYGDALA_EVAL_DB", "eval_claims.db"))
RESULTS_PATH = Path("experiments/results/exp20_results.json")

CORPUS_SIZES = [1_000, 5_000, 10_000, 20_000]


# ---------------------------------------------------------------------------
# PersistentEmbeddingCache prototype
# ---------------------------------------------------------------------------

class PersistentEmbeddingCache:
    """SQLite-backed persistent embedding cache.

    Stores embeddings keyed by SHA-256 hash of the input text, scoped to
    a specific model name and dimension so different model configs don't
    collide.
    """

    def __init__(self, db_path: str | Path, model_name: str, dim: int):
        self.db_path = str(db_path)
        self.model_name = model_name
        self.dim = dim
        self.conn = connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (text_hash, model_name, dim)
            )
        """)
        self.conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model_name = ? AND dim = ?",
            (self._hash(text), self.model_name, self.dim),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32).copy()

    def put(self, text: str, embedding: np.ndarray):
        self.conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, model_name, dim, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (self._hash(text), self.model_name, self.dim, embedding.tobytes(), time.time()),
        )

    def put_batch(self, texts: list[str], embeddings: np.ndarray):
        now = time.time()
        rows = [
            (self._hash(t), self.model_name, self.dim, embeddings[i].tobytes(), now)
            for i, t in enumerate(texts)
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, model_name, dim, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Return cached embeddings and indices of cache misses."""
        results: list[np.ndarray | None] = [None] * len(texts)
        miss_indices: list[int] = []
        hashes = [self._hash(t) for t in texts]

        # Batch lookup using IN clause (chunked to avoid SQLite variable limits)
        chunk_size = 500
        hash_to_indices: dict[str, list[int]] = {}
        for i, h in enumerate(hashes):
            hash_to_indices.setdefault(h, []).append(i)

        unique_hashes = list(hash_to_indices.keys())
        found_hashes: dict[str, bytes] = {}

        for start in range(0, len(unique_hashes), chunk_size):
            chunk = unique_hashes[start : start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT text_hash, embedding FROM embedding_cache "
                f"WHERE text_hash IN ({placeholders}) AND model_name = ? AND dim = ?",
                chunk + [self.model_name, self.dim],
            ).fetchall()
            for row in rows:
                found_hashes[row["text_hash"]] = row["embedding"]

        for h, indices in hash_to_indices.items():
            if h in found_hashes:
                vec = np.frombuffer(found_hashes[h], dtype=np.float32).copy()
                for i in indices:
                    results[i] = vec
            else:
                for i in indices:
                    miss_indices.append(i)

        return results, sorted(miss_indices)

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM embedding_cache WHERE model_name = ? AND dim = ?",
            (self.model_name, self.dim),
        ).fetchone()[0]

    def close(self):
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_claims(n: int) -> list[str]:
    """Load n distinct claims from claims database."""
    conn = sqlite3.connect(str(OTAK_DB))
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT n.name FROM idx_knowledge_item_claim_type k "
        "JOIN nodes n ON n.id = k.node_id "
        "WHERE n.deleted_at IS NULL AND n.name IS NOT NULL "
        "AND length(n.name) > 20 AND length(n.name) < 500 "
        "ORDER BY RANDOM() LIMIT ?",
        (n,),
    )
    texts = [row[0] for row in cur.fetchall()]
    conn.close()
    return texts


def get_process_memory_mb() -> float:
    """Get current process RSS in MB (macOS/Linux)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes
    except Exception:
        return 0.0


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_corpus_size(model: EmbeddingModel, texts: list[str], label: str) -> dict:
    """Benchmark cold embed, cache write, warm lookup for a given corpus."""
    n = len(texts)
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label} ({n:,} texts)")
    print(f"{'='*60}")

    # 1. Cold embedding (no cache involved)
    model._cache.clear()
    gc.collect()
    t0 = time.perf_counter()
    embeddings = model.embed_batch(texts)
    cold_time = time.perf_counter() - t0
    print(f"  Cold embed:       {format_time(cold_time)} ({n / cold_time:.0f} texts/sec)")

    # 2. Cache write
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        cache_path = f.name

    cache = PersistentEmbeddingCache(cache_path, model.model_name, embeddings.shape[1])
    t0 = time.perf_counter()
    cache.put_batch(texts, embeddings)
    write_time = time.perf_counter() - t0
    print(f"  Cache write:      {format_time(write_time)} ({n / write_time:.0f} texts/sec)")

    # File size
    cache.close()
    db_size_bytes = os.path.getsize(cache_path)
    db_size_mb = db_size_bytes / (1024 * 1024)
    bytes_per_entry = db_size_bytes / n if n > 0 else 0
    print(f"  DB file size:     {db_size_mb:.2f} MB ({bytes_per_entry:.0f} bytes/entry)")

    # 3. Warm lookup (fresh cache object to simulate process restart)
    cache2 = PersistentEmbeddingCache(cache_path, model.model_name, embeddings.shape[1])

    # Single-item lookup latency
    sample_indices = np.random.choice(n, min(200, n), replace=False)
    t0 = time.perf_counter()
    for idx in sample_indices:
        cache2.get(texts[idx])
    single_time = time.perf_counter() - t0
    per_item_us = (single_time / len(sample_indices)) * 1_000_000
    print(f"  Single lookup:    {per_item_us:.1f}us/item")

    # Batch lookup
    t0 = time.perf_counter()
    cached_results, misses = cache2.get_batch(texts)
    batch_time = time.perf_counter() - t0
    per_item_batch_us = (batch_time / n) * 1_000_000
    print(f"  Batch lookup:     {format_time(batch_time)} ({per_item_batch_us:.1f}us/item, {n / batch_time:.0f} texts/sec)")
    print(f"  Cache misses:     {len(misses)}")

    # Speedup
    speedup = cold_time / batch_time if batch_time > 0 else float("inf")
    print(f"  Speedup vs cold:  {speedup:.1f}x")

    cache2.close()
    os.unlink(cache_path)

    return {
        "n": n,
        "cold_embed_sec": round(cold_time, 4),
        "cache_write_sec": round(write_time, 4),
        "batch_lookup_sec": round(batch_time, 4),
        "single_lookup_us": round(per_item_us, 1),
        "batch_lookup_us_per_item": round(per_item_batch_us, 1),
        "speedup_vs_cold": round(speedup, 1),
        "db_size_mb": round(db_size_mb, 2),
        "bytes_per_entry": round(bytes_per_entry, 0),
        "texts_per_sec_cold": round(n / cold_time, 0),
        "texts_per_sec_warm": round(n / batch_time, 0),
    }


def test_correctness(model: EmbeddingModel, texts_100: list[str]) -> dict:
    """Verify cached embeddings exactly match fresh embeddings."""
    print(f"\n{'='*60}")
    print(f"  Correctness test (100 texts)")
    print(f"{'='*60}")

    texts = texts_100[:100]

    # Embed fresh
    model._cache.clear()
    embeddings = model.embed_batch(texts)

    # Store in cache
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        cache_path = f.name
    cache = PersistentEmbeddingCache(cache_path, model.model_name, embeddings.shape[1])
    cache.put_batch(texts, embeddings)
    cache.close()

    # Retrieve from fresh cache (simulates restart)
    cache2 = PersistentEmbeddingCache(cache_path, model.model_name, embeddings.shape[1])
    cached_results, misses = cache2.get_batch(texts)

    assert len(misses) == 0, f"Expected 0 misses, got {len(misses)}"
    cached_matrix = np.vstack(cached_results)
    all_close = np.allclose(embeddings, cached_matrix, atol=1e-7)
    max_diff = np.max(np.abs(embeddings - cached_matrix))

    print(f"  np.allclose:      {all_close}")
    print(f"  Max abs diff:     {max_diff:.2e}")
    print(f"  Shape original:   {embeddings.shape}")
    print(f"  Shape cached:     {cached_matrix.shape}")

    # Also test single get
    for i in range(10):
        single = cache2.get(texts[i])
        assert single is not None, f"Single get returned None for index {i}"
        assert np.allclose(single, embeddings[i], atol=1e-7), f"Mismatch at index {i}"
    print(f"  Single get:       10/10 match")

    cache2.close()
    os.unlink(cache_path)

    return {
        "all_close": bool(all_close),
        "max_abs_diff": float(max_diff),
        "misses": len(misses),
    }


def test_partial_cache(model: EmbeddingModel, texts_1k: list[str]) -> dict:
    """Test partial cache scenario: 80% cached, 20% cold."""
    print(f"\n{'='*60}")
    print(f"  Partial cache test (80% hit rate)")
    print(f"{'='*60}")

    n = len(texts_1k)
    split = int(n * 0.8)
    cached_texts = texts_1k[:split]
    all_texts = texts_1k

    # Pre-embed the cached portion
    model._cache.clear()
    cached_embeddings = model.embed_batch(cached_texts)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        cache_path = f.name
    cache = PersistentEmbeddingCache(cache_path, model.model_name, cached_embeddings.shape[1])
    cache.put_batch(cached_texts, cached_embeddings)

    # Now simulate: embed_batch_with_cache for all texts
    t0 = time.perf_counter()
    results, miss_indices = cache.get_batch(all_texts)
    lookup_time = time.perf_counter() - t0

    # Embed the misses
    miss_texts = [all_texts[i] for i in miss_indices]
    model._cache.clear()
    t1 = time.perf_counter()
    miss_embeddings = model.embed_batch(miss_texts)
    embed_time = time.perf_counter() - t1

    # Store the newly embedded texts
    t2 = time.perf_counter()
    cache.put_batch(miss_texts, miss_embeddings)
    store_time = time.perf_counter() - t2

    partial_total = lookup_time + embed_time + store_time

    # Compare to full cold embed
    model._cache.clear()
    t3 = time.perf_counter()
    _ = model.embed_batch(all_texts)
    full_cold_time = time.perf_counter() - t3

    speedup = full_cold_time / partial_total if partial_total > 0 else float("inf")

    print(f"  Total texts:      {n}")
    print(f"  Cache hits:       {split} ({split/n*100:.0f}%)")
    print(f"  Cache misses:     {len(miss_indices)} ({len(miss_indices)/n*100:.0f}%)")
    print(f"  Lookup time:      {format_time(lookup_time)}")
    print(f"  Miss embed time:  {format_time(embed_time)}")
    print(f"  Store new time:   {format_time(store_time)}")
    print(f"  Total partial:    {format_time(partial_total)}")
    print(f"  Full cold embed:  {format_time(full_cold_time)}")
    print(f"  Speedup:          {speedup:.1f}x")

    cache.close()
    os.unlink(cache_path)

    return {
        "n": n,
        "cache_hits": split,
        "cache_misses": len(miss_indices),
        "lookup_sec": round(lookup_time, 4),
        "miss_embed_sec": round(embed_time, 4),
        "store_new_sec": round(store_time, 4),
        "partial_total_sec": round(partial_total, 4),
        "full_cold_sec": round(full_cold_time, 4),
        "speedup": round(speedup, 1),
    }


def test_memory_comparison(model: EmbeddingModel, texts_10k: list[str]) -> dict:
    """Compare memory usage: in-memory dict vs SQLite cache."""
    print(f"\n{'='*60}")
    print(f"  Memory comparison (10K embeddings)")
    print(f"{'='*60}")

    n = len(texts_10k)
    dim = model.dim

    # Theoretical in-memory size: n * dim * 4 bytes (float32) + dict overhead
    embedding_bytes = n * dim * 4
    # Python dict overhead: ~100 bytes per entry (key hash, pointer, etc.)
    dict_overhead = n * 100
    # Key storage: average text length * n
    avg_text_len = sum(len(t) for t in texts_10k) / n
    key_bytes = int(avg_text_len * n)
    in_memory_est_mb = (embedding_bytes + dict_overhead + key_bytes) / (1024 * 1024)

    # SQLite cache size on disk
    model._cache.clear()
    embeddings = model.embed_batch(texts_10k)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        cache_path = f.name
    cache = PersistentEmbeddingCache(cache_path, model.model_name, dim)
    cache.put_batch(texts_10k, embeddings)
    cache.close()

    db_size_mb = os.path.getsize(cache_path) / (1024 * 1024)

    # Pure embedding data size
    raw_data_mb = embedding_bytes / (1024 * 1024)

    print(f"  Embeddings:       {n:,} x {dim}d = {raw_data_mb:.1f} MB raw")
    print(f"  In-memory dict:   ~{in_memory_est_mb:.1f} MB (estimated)")
    print(f"  SQLite on disk:   {db_size_mb:.1f} MB")
    print(f"  SQLite overhead:  {(db_size_mb / raw_data_mb - 1) * 100:.0f}% vs raw data")
    print(f"  Avg text length:  {avg_text_len:.0f} chars")

    os.unlink(cache_path)

    return {
        "n": n,
        "dim": dim,
        "raw_data_mb": round(raw_data_mb, 2),
        "in_memory_est_mb": round(in_memory_est_mb, 2),
        "sqlite_db_mb": round(db_size_mb, 2),
        "sqlite_overhead_pct": round((db_size_mb / raw_data_mb - 1) * 100, 1),
        "avg_text_len": round(avg_text_len, 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Experiment 20: SQLite-backed Persistent Embedding Cache")
    print("=" * 60)

    # Load model once
    model = EmbeddingModel()
    _ = model.embed("warmup")  # trigger model load
    dim = model.dim
    print(f"Model: {model.model_name}, dim={dim}")

    # Load max corpus
    max_n = max(CORPUS_SIZES)
    print(f"\nLoading {max_n:,} claims from claims database...")
    all_claims = load_claims(max_n)
    actual_n = len(all_claims)
    print(f"Loaded {actual_n:,} distinct claims")

    results = {
        "experiment": "exp20_embedding_cache",
        "model": model.model_name,
        "dim": dim,
        "available_claims": actual_n,
    }

    # --- Correctness test ---
    results["correctness"] = test_correctness(model, all_claims[:100])

    # --- Benchmark at each corpus size ---
    results["benchmarks"] = {}
    for size in CORPUS_SIZES:
        if size > actual_n:
            print(f"\nSkipping {size:,} (only {actual_n:,} available)")
            continue
        texts = all_claims[:size]
        label = f"{size // 1000}K"
        results["benchmarks"][label] = benchmark_corpus_size(model, texts, label)

    # --- Partial cache test ---
    partial_n = min(5000, actual_n)
    results["partial_cache"] = test_partial_cache(model, all_claims[:partial_n])

    # --- Memory comparison ---
    mem_n = min(10000, actual_n)
    results["memory"] = test_memory_comparison(model, all_claims[:mem_n])

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Corpus size benchmarks:")
    print(f"  {'Size':>6s} | {'Cold':>8s} | {'Write':>8s} | {'Warm':>8s} | {'Speedup':>8s} | {'DB Size':>8s} | {'us/item':>8s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for label, b in results["benchmarks"].items():
        print(
            f"  {label:>6s} | "
            f"{format_time(b['cold_embed_sec']):>8s} | "
            f"{format_time(b['cache_write_sec']):>8s} | "
            f"{format_time(b['batch_lookup_sec']):>8s} | "
            f"{b['speedup_vs_cold']:>7.1f}x | "
            f"{b['db_size_mb']:>6.2f}MB | "
            f"{b['batch_lookup_us_per_item']:>7.1f}"
        )

    p = results["partial_cache"]
    print(f"\n  Partial cache (80% hit): {p['speedup']:.1f}x speedup")

    m = results["memory"]
    print(f"\n  Memory (10K embeddings):")
    print(f"    Raw data:       {m['raw_data_mb']:.1f} MB")
    print(f"    In-memory dict: ~{m['in_memory_est_mb']:.1f} MB")
    print(f"    SQLite on disk: {m['sqlite_db_mb']:.1f} MB (+{m['sqlite_overhead_pct']:.0f}% overhead)")

    c = results["correctness"]
    print(f"\n  Correctness: {'PASS' if c['all_close'] else 'FAIL'} (max diff: {c['max_abs_diff']:.2e})")

    # Key finding
    if results["benchmarks"]:
        largest = list(results["benchmarks"].values())[-1]
        print(f"\n  KEY FINDING:")
        print(f"  At {largest['n']:,} texts, warm cache lookup is {largest['speedup_vs_cold']:.0f}x faster than cold embedding.")
        print(f"  Batch lookup: {largest['batch_lookup_us_per_item']:.1f}us/item vs {largest['cold_embed_sec']/largest['n']*1e6:.0f}us/item cold.")
        print(f"  SQLite overhead is ~{m['sqlite_overhead_pct']:.0f}% over raw embedding data.")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
