"""Tests for PayloadCache — generic external-API payload cache with TTL."""

import time

import pytest

from limbic.amygdala import PayloadCache


def test_put_get_roundtrip(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="wikidata_get")
    cache.put("Q42", {"label": "Douglas Adams", "dob": "1952-03-11"})
    assert cache.get("Q42") == {"label": "Douglas Adams", "dob": "1952-03-11"}


def test_miss_returns_none(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="wikidata_get")
    assert cache.get("Q999999999") is None


def test_get_or_fetch_miss_calls_fn(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="geonames")
    calls = []

    def fetcher():
        calls.append(1)
        return {"name": "Palermo", "country": "IT"}

    result = cache.get_or_fetch("palermo", fetcher)
    assert result == {"name": "Palermo", "country": "IT"}
    assert calls == [1]

    # Second call is a cache hit; fetcher should not run
    result2 = cache.get_or_fetch("palermo", fetcher)
    assert result2 == result
    assert calls == [1]  # unchanged


def test_ttl_expiry(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test", default_ttl_seconds=1)
    cache.put("k", {"v": 1})
    assert cache.get("k") == {"v": 1}

    # Forge expiry by rewriting fetched_at
    cache.conn.execute(
        "UPDATE payload_cache SET fetched_at = fetched_at - 3600 WHERE key = ?",
        ("k",),
    )
    cache.conn.commit()

    assert cache.get("k") is None
    # allow_stale=True still returns it
    assert cache.get("k", allow_stale=True) == {"v": 1}


def test_custom_ttl_overrides_default(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test", default_ttl_seconds=3600)
    cache.put("k", {"v": 1}, ttl_seconds=1)
    row = cache.conn.execute(
        "SELECT ttl_seconds FROM payload_cache WHERE key = ?", ("k",)
    ).fetchone()
    assert row["ttl_seconds"] == 1


def test_source_isolation(tmp_path):
    db = str(tmp_path / "c.db")
    wiki = PayloadCache(db, source="wikidata_get")
    geo = PayloadCache(db, source="geonames")

    wiki.put("same_key", {"from": "wikidata"})
    geo.put("same_key", {"from": "geonames"})

    assert wiki.get("same_key") == {"from": "wikidata"}
    assert geo.get("same_key") == {"from": "geonames"}

    # clear_source only affects one
    wiki.clear_source()
    assert wiki.get("same_key") is None
    assert geo.get("same_key") == {"from": "geonames"}


def test_delete(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    cache.put("a", {"x": 1})
    cache.put("b", {"x": 2})
    cache.delete("a")
    assert cache.get("a") is None
    assert cache.get("b") == {"x": 2}


def test_invalidate_before(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    cache.put("old", {"v": 1})
    cache.put("new", {"v": 2})
    # Age the "old" entry
    cache.conn.execute(
        "UPDATE payload_cache SET fetched_at = fetched_at - 86400 WHERE key = ?",
        ("old",),
    )
    cache.conn.commit()
    cutoff = int(time.time()) - 3600
    removed = cache.invalidate_before(cutoff)
    assert removed == 1
    assert cache.get("old", allow_stale=True) is None
    assert cache.get("new") == {"v": 2}


def test_stats(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test", default_ttl_seconds=1)
    cache.put("fresh", {"v": 1})
    cache.put("expired", {"v": 2})
    cache.conn.execute(
        "UPDATE payload_cache SET fetched_at = fetched_at - 3600 WHERE key = ?",
        ("expired",),
    )
    cache.conn.commit()
    s = cache.stats()
    assert s["source"] == "test"
    assert s["total"] == 2
    assert s["fresh"] == 1
    assert s["expired"] == 1


def test_empty_source_rejected(tmp_path):
    with pytest.raises(ValueError):
        PayloadCache(str(tmp_path / "c.db"), source="")


def test_payload_can_be_list(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    cache.put("candidates", [{"qid": "Q42"}, {"qid": "Q1"}])
    assert cache.get("candidates") == [{"qid": "Q42"}, {"qid": "Q1"}]


def test_put_overwrites(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    cache.put("k", {"v": 1})
    cache.put("k", {"v": 2})
    assert cache.get("k") == {"v": 2}
    assert cache.count() == 1


def test_get_many_returns_only_fresh(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="wikidata_get")
    cache.put("Q42", {"label": "Douglas Adams"})
    cache.put("Q1", {"label": "Universe"})
    cache.put("Q5", {"label": "human"}, ttl_seconds=1)
    # Age Q5 past its TTL
    cache.conn.execute(
        "UPDATE payload_cache SET fetched_at = fetched_at - 3600 WHERE key = ?",
        ("Q5",),
    )
    cache.conn.commit()

    result = cache.get_many(["Q42", "Q1", "Q5", "Q999"])
    assert set(result.keys()) == {"Q42", "Q1"}
    assert result["Q42"] == {"label": "Douglas Adams"}
    assert result["Q1"] == {"label": "Universe"}


def test_get_many_empty_input(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    assert cache.get_many([]) == {}


def test_get_many_chunks_large_input(tmp_path):
    """Verify the 500-chunk batching doesn't lose data at the boundary."""
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    # Seed 1200 entries (> 2 chunks of 500)
    items = {f"k{i}": {"n": i} for i in range(1200)}
    cache.put_many(items)
    result = cache.get_many(list(items.keys()))
    assert len(result) == 1200
    assert result["k0"] == {"n": 0}
    assert result["k599"] == {"n": 599}  # spans first chunk boundary
    assert result["k1000"] == {"n": 1000}  # spans second


def test_put_many(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    cache.put_many({"a": {"v": 1}, "b": {"v": 2}, "c": {"v": 3}})
    assert cache.count() == 3
    assert cache.get("a") == {"v": 1}
    assert cache.get("c") == {"v": 3}


def test_put_many_empty_is_noop(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    cache.put_many({})  # must not raise
    assert cache.count() == 0


def test_context_manager_closes_connection(tmp_path):
    db = str(tmp_path / "c.db")
    with PayloadCache(db, source="test") as cache:
        cache.put("k", {"v": 1})
        assert cache.get("k") == {"v": 1}
    assert cache.conn is None


def test_close_is_idempotent(tmp_path):
    cache = PayloadCache(str(tmp_path / "c.db"), source="test")
    cache.put("k", {"v": 1})
    cache.close()
    cache.close()  # second call must not raise
    assert cache.conn is None
