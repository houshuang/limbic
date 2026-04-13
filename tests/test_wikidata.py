"""Unit tests for WikidataClient — mocked HTTP via the fetcher injection point.

Live/integration tests that hit real Wikidata live in test_wikidata_live.py
and are gated by the LIMBIC_LIVE_WIKIDATA=1 environment variable.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace
from typing import Callable

import pytest

from limbic.amygdala import (
    Candidate,
    Entity,
    MaxlagError,
    PayloadCache,
    TokenBucket,
    WikidataClient,
    WikidataError,
)


# --- fake fetcher infrastructure ---


class MockFetcher:
    """Records requests and returns scripted responses.

    `responses` is a list of (url_predicate, response) pairs. `response` may
    be bytes (returned as body), an Exception (raised), or a callable taking
    the Request and returning bytes.

    If no predicate matches, raises AssertionError with the URL — so forgetting
    to script a response fails loudly.
    """

    def __init__(self, responses: list[tuple[Callable[[str], bool], object]]):
        self.responses = responses
        self.calls: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request, timeout: float) -> bytes:
        self.calls.append(req)
        url = req.full_url
        for predicate, response in self.responses:
            if predicate(url):
                if isinstance(response, Exception):
                    raise response
                if callable(response):
                    return response(req)
                return response
        raise AssertionError(f"No mock response for URL: {url}")


def matches(substring: str) -> Callable[[str], bool]:
    return lambda url: substring in url


def json_bytes(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")


# --- shared fixtures ---


VALID_UA = "TestApp/0.1 (mailto:test@example.com)"


@pytest.fixture
def no_rate_limit_client_factory(tmp_path):
    """Returns a factory: `make(fetcher, **overrides) -> WikidataClient`.

    Uses a very fast rate limiter (1000 req/s) so tests don't sleep, and a
    real PayloadCache in a tmp file.
    """
    def make(fetcher: MockFetcher, **overrides) -> WikidataClient:
        return WikidataClient(
            user_agent=VALID_UA,
            cache_db_path=str(tmp_path / f"wd_{id(fetcher)}.db"),
            rate_per_sec=1000.0,
            fetcher=fetcher,
            **overrides,
        )
    return make


# --- UA validation ---


def test_user_agent_required():
    with pytest.raises(ValueError, match="user_agent"):
        WikidataClient(user_agent="", fetcher=MockFetcher([]))


def test_user_agent_must_have_contact_info():
    with pytest.raises(ValueError, match="contact info"):
        WikidataClient(user_agent="Petrarca/0.1", fetcher=MockFetcher([]))


def test_user_agent_with_parens_accepted():
    # Parens indicate a contact/details section per UA convention
    WikidataClient(user_agent="Petrarca/0.1 (mailto:x@y)", fetcher=MockFetcher([]))


# --- search ---


def test_search_parses_candidates(no_rate_limit_client_factory):
    fetcher = MockFetcher([
        (matches("wbsearchentities"), json_bytes({
            "search": [
                {
                    "id": "Q214867",
                    "label": "Rollo",
                    "description": "Viking leader, first Duke of Normandy",
                    "match": {"text": "Rollo", "type": "label"},
                    "aliases": ["Hrólfr"],
                },
                {
                    "id": "Q82692",
                    "label": "Rollo",
                    "description": "DC Comics character",
                    "match": {"text": "Rollo", "type": "label"},
                },
            ],
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    results = client.search("Rollo", limit=10)
    assert len(results) == 2
    assert results[0] == Candidate(
        qid="Q214867",
        label="Rollo",
        description="Viking leader, first Duke of Normandy",
        match_text="Rollo",
        match_type="label",
        rank=0,
        aliases=["Hrólfr"],
    )
    assert results[1].qid == "Q82692"
    assert results[1].rank == 1


def test_search_empty_name_returns_empty(no_rate_limit_client_factory):
    fetcher = MockFetcher([])
    client = no_rate_limit_client_factory(fetcher)
    assert client.search("") == []
    assert fetcher.calls == []  # no HTTP call for empty name


def test_search_caches_results(no_rate_limit_client_factory):
    payload = json_bytes({"search": [{"id": "Q42", "label": "Douglas Adams"}]})
    fetcher = MockFetcher([(matches("wbsearchentities"), payload)])
    client = no_rate_limit_client_factory(fetcher)
    r1 = client.search("Douglas Adams")
    r2 = client.search("Douglas Adams")
    assert r1 == r2
    assert len(fetcher.calls) == 1  # second call served from cache


def test_search_cache_key_respects_language(no_rate_limit_client_factory):
    responses = [
        (lambda u: "language=en" in u, json_bytes({
            "search": [{"id": "Q90", "label": "Paris", "description": "capital of France"}]
        })),
        (lambda u: "language=no" in u, json_bytes({
            "search": [{"id": "Q90", "label": "Paris", "description": "Frankrikes hovedstad"}]
        })),
    ]
    fetcher = MockFetcher(responses)
    client = no_rate_limit_client_factory(fetcher)
    client.search("Paris", language="en")
    client.search("Paris", language="no")
    assert len(fetcher.calls) == 2  # different languages → separate fetches


# --- get / get_many ---


def _entity_stub(qid: str, label_en: str, desc_en: str = "", claims: dict | None = None) -> dict:
    return {
        qid: {
            "id": qid,
            "labels": {"en": {"language": "en", "value": label_en}},
            "descriptions": {"en": {"language": "en", "value": desc_en}},
            "aliases": {},
            "claims": claims or {},
            "modified": "2026-04-13T00:00:00Z",
        }
    }


def test_get_parses_entity(no_rate_limit_client_factory):
    fetcher = MockFetcher([
        (matches("wbgetentities"), json_bytes({
            "entities": _entity_stub("Q214867", "Rollo", "first Duke of Normandy"),
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    entity = client.get("Q214867")
    assert entity is not None
    assert entity.qid == "Q214867"
    assert entity.label("en") == "Rollo"
    assert entity.description("en") == "first Duke of Normandy"


def test_get_missing_returns_none(no_rate_limit_client_factory):
    fetcher = MockFetcher([
        (matches("wbgetentities"), json_bytes({
            "entities": {"Q99999999": {"id": "Q99999999", "missing": ""}},
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    assert client.get("Q99999999") is None


def test_get_many_batches_at_50(no_rate_limit_client_factory):
    qids = [f"Q{i}" for i in range(1, 76)]  # 75 QIDs → 2 batches
    batches_seen: list[list[str]] = []

    def respond(req: urllib.request.Request) -> bytes:
        parsed = urllib.parse.urlparse(req.full_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        batch = params["ids"].split("|")
        batches_seen.append(batch)
        entities = {}
        for q in batch:
            entities[q] = {"id": q, "labels": {"en": {"language": "en", "value": q}}}
        return json_bytes({"entities": entities})

    fetcher = MockFetcher([(matches("wbgetentities"), respond)])
    client = no_rate_limit_client_factory(fetcher)
    result = client.get_many(qids)
    assert len(result) == 75
    assert len(batches_seen) == 2
    assert len(batches_seen[0]) == 50
    assert len(batches_seen[1]) == 25


def test_get_many_uses_cache_on_second_call(no_rate_limit_client_factory):
    fetcher = MockFetcher([
        (matches("wbgetentities"), json_bytes({
            "entities": _entity_stub("Q42", "Douglas Adams"),
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    client.get_many(["Q42"])
    client.get_many(["Q42"])
    assert len(fetcher.calls) == 1


def test_get_many_partial_cache_hit(no_rate_limit_client_factory):
    """If some QIDs are cached and others aren't, only the missing ones go over the wire."""
    fetcher_calls: list[list[str]] = []

    def respond(req):
        parsed = urllib.parse.urlparse(req.full_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        ids = params["ids"].split("|")
        fetcher_calls.append(ids)
        entities = {q: {"id": q, "labels": {"en": {"language": "en", "value": q}}} for q in ids}
        return json_bytes({"entities": entities})

    fetcher = MockFetcher([(matches("wbgetentities"), respond)])
    client = no_rate_limit_client_factory(fetcher)
    client.get_many(["Q1", "Q2"])
    client.get_many(["Q1", "Q2", "Q3", "Q4"])
    # First call fetched Q1+Q2. Second call should only fetch Q3+Q4.
    assert fetcher_calls[0] == ["Q1", "Q2"]
    assert fetcher_calls[1] == ["Q3", "Q4"]


def test_get_follows_redirect(no_rate_limit_client_factory):
    """Requesting a QID that's been redirected: entity returns under resolved QID with redirect_from set."""
    fetcher = MockFetcher([
        (matches("wbgetentities"), json_bytes({
            "redirects": [{"from": "Q111", "to": "Q222"}],
            "entities": {"Q222": {
                "id": "Q222",
                "labels": {"en": {"language": "en", "value": "Merged target"}},
            }},
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    entity = client.get("Q111")
    assert entity is not None
    assert entity.qid == "Q222"
    assert entity.redirect_from == "Q111"


def test_get_many_empty_returns_empty(no_rate_limit_client_factory):
    fetcher = MockFetcher([])
    client = no_rate_limit_client_factory(fetcher)
    assert client.get_many([]) == {}
    assert fetcher.calls == []


def test_get_many_preserves_input_order(no_rate_limit_client_factory):
    """Output dict iteration order must match the input list, regardless of
    whether each QID was a cache hit or a fresh fetch."""
    fetcher = MockFetcher([
        (matches("wbgetentities"), json_bytes({
            "entities": {
                "Q1": {"id": "Q1", "labels": {"en": {"language": "en", "value": "A"}}},
                "Q2": {"id": "Q2", "labels": {"en": {"language": "en", "value": "B"}}},
                "Q3": {"id": "Q3", "labels": {"en": {"language": "en", "value": "C"}}},
            },
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    # Warm Q2 into the cache only
    client.get_many(["Q2"])
    # Now request all three in an order that interleaves hit (Q2) and miss (Q1, Q3)
    result = client.get_many(["Q3", "Q2", "Q1"])
    assert list(result.keys()) == ["Q3", "Q2", "Q1"]


# --- retries & errors ---


def test_retry_on_http_429(no_rate_limit_client_factory):
    attempts = [0]

    def respond(req):
        attempts[0] += 1
        if attempts[0] == 1:
            raise urllib.error.HTTPError(
                url=req.full_url, code=429, msg="Too Many Requests",
                hdrs=SimpleNamespace(get=lambda k: None), fp=None,  # type: ignore[arg-type]
            )
        return json_bytes({"search": [{"id": "Q42", "label": "ok"}]})

    fetcher = MockFetcher([(matches("wbsearchentities"), respond)])
    client = no_rate_limit_client_factory(fetcher, max_retries=3)
    # Patch sleep to avoid slow test
    import limbic.amygdala.wikidata as wd_mod
    orig_sleep = wd_mod.time.sleep
    wd_mod.time.sleep = lambda _s: None
    try:
        results = client.search("whatever")
    finally:
        wd_mod.time.sleep = orig_sleep
    assert len(results) == 1
    assert attempts[0] == 2


def test_retry_respects_retry_after(no_rate_limit_client_factory):
    attempts = [0]
    slept: list[float] = []

    def respond(req):
        attempts[0] += 1
        if attempts[0] == 1:
            hdrs = SimpleNamespace(get=lambda k: "7" if k == "Retry-After" else None)
            raise urllib.error.HTTPError(
                url=req.full_url, code=503, msg="Service Unavailable",
                hdrs=hdrs, fp=None,  # type: ignore[arg-type]
            )
        return json_bytes({"search": []})

    fetcher = MockFetcher([(matches("wbsearchentities"), respond)])
    client = no_rate_limit_client_factory(fetcher, max_retries=3)
    import limbic.amygdala.wikidata as wd_mod
    orig_sleep = wd_mod.time.sleep
    wd_mod.time.sleep = lambda s: slept.append(s)
    try:
        client.search("x")
    finally:
        wd_mod.time.sleep = orig_sleep
    # Either the retry-after wait of 7 was used, or the bucket-sleep (close to zero
    # at 1000 req/s). The retry sleep must be present.
    assert 7.0 in slept or any(abs(s - 7.0) < 0.01 for s in slept)


def test_retry_on_maxlag_error(no_rate_limit_client_factory):
    attempts = [0]

    def respond(req):
        attempts[0] += 1
        if attempts[0] == 1:
            return json_bytes({"error": {"code": "maxlag", "info": "replication lag"}})
        return json_bytes({"search": [{"id": "Q42", "label": "ok"}]})

    fetcher = MockFetcher([(matches("wbsearchentities"), respond)])
    client = no_rate_limit_client_factory(fetcher, max_retries=3)
    import limbic.amygdala.wikidata as wd_mod
    orig_sleep = wd_mod.time.sleep
    wd_mod.time.sleep = lambda _s: None
    try:
        results = client.search("x")
    finally:
        wd_mod.time.sleep = orig_sleep
    assert len(results) == 1
    assert attempts[0] == 2


def test_maxlag_wait_reflects_parsed_lag(no_rate_limit_client_factory):
    """When the maxlag info string contains '6.8 seconds lagged', the retry
    wait should be at least 6.8 + buffer, not the 5-second default."""
    attempts = [0]
    slept: list[float] = []

    def respond(req):
        attempts[0] += 1
        if attempts[0] == 1:
            return json_bytes({
                "error": {
                    "code": "maxlag",
                    "info": "Waiting for wdqs1014: 6.8 seconds lagged.",
                }
            })
        return json_bytes({"search": []})

    fetcher = MockFetcher([(matches("wbsearchentities"), respond)])
    client = no_rate_limit_client_factory(fetcher, max_retries=3)
    import limbic.amygdala.wikidata as wd_mod
    orig_sleep = wd_mod.time.sleep
    wd_mod.time.sleep = lambda s: slept.append(s)
    try:
        client.search("x")
    finally:
        wd_mod.time.sleep = orig_sleep
    # Expect a sleep ≥ 6.8 + 2 = 8.8 (plus up to 2s jitter), capped at 30
    assert any(s >= 8.8 for s in slept), (
        f"expected a maxlag sleep ≥ 8.8s based on parsed lag; got {slept}"
    )


def test_parse_maxlag_seconds_variants():
    from limbic.amygdala.wikidata import _parse_maxlag_seconds
    assert _parse_maxlag_seconds("Waiting for wdqs1014: 6.8 seconds lagged.") == 6.8
    assert _parse_maxlag_seconds("6 seconds lagged") == 6.0
    assert _parse_maxlag_seconds("12.345 seconds lagged more text") == 12.345
    assert _parse_maxlag_seconds("replication lag") is None
    assert _parse_maxlag_seconds("") is None
    assert _parse_maxlag_seconds(None) is None  # type: ignore[arg-type]


def test_ssl_error_fails_fast_no_retry(tmp_path):
    """SSL errors are permanent; don't waste the retry budget."""
    import ssl as _ssl
    attempts = [0]

    def respond(req):
        attempts[0] += 1
        raise urllib.error.URLError(_ssl.SSLError("certificate verify failed"))

    fetcher = MockFetcher([(matches("wbsearchentities"), respond)])
    client = WikidataClient(
        user_agent=VALID_UA,
        cache_db_path=str(tmp_path / "c.db"),
        rate_per_sec=1000.0,
        max_retries=5,
        fetcher=fetcher,
    )
    with pytest.raises(WikidataError, match="SSL error"):
        client.search("anything")
    assert attempts[0] == 1  # no retry


def test_non_retryable_error_raises(no_rate_limit_client_factory):
    def respond(req):
        raise urllib.error.HTTPError(
            url=req.full_url, code=400, msg="Bad Request",
            hdrs=SimpleNamespace(get=lambda k: None), fp=None,  # type: ignore[arg-type]
        )

    fetcher = MockFetcher([(matches("wbsearchentities"), respond)])
    client = no_rate_limit_client_factory(fetcher, max_retries=3)
    with pytest.raises(WikidataError, match="HTTP 400"):
        client.search("x")


def test_max_retries_exceeded(no_rate_limit_client_factory):
    def respond(req):
        raise urllib.error.HTTPError(
            url=req.full_url, code=503, msg="Service Unavailable",
            hdrs=SimpleNamespace(get=lambda k: None), fp=None,  # type: ignore[arg-type]
        )

    fetcher = MockFetcher([(matches("wbsearchentities"), respond)])
    client = no_rate_limit_client_factory(fetcher, max_retries=2)
    import limbic.amygdala.wikidata as wd_mod
    orig_sleep = wd_mod.time.sleep
    wd_mod.time.sleep = lambda _s: None
    try:
        with pytest.raises(WikidataError, match="max retries"):
            client.search("x")
    finally:
        wd_mod.time.sleep = orig_sleep


def test_api_error_other_than_maxlag_raises(no_rate_limit_client_factory):
    fetcher = MockFetcher([
        (matches("wbsearchentities"), json_bytes({"error": {"code": "badvalue", "info": "bad"}})),
    ])
    client = no_rate_limit_client_factory(fetcher, max_retries=1)
    with pytest.raises(WikidataError, match="badvalue"):
        client.search("x")


# --- Entity helpers ---


def test_entity_claim_qids():
    entity = Entity(
        qid="Q42",
        claims={
            "P22": [{  # father
                "mainsnak": {
                    "datatype": "wikibase-item",
                    "datavalue": {"value": {"id": "Q100"}},
                },
                "rank": "normal",
            }],
            "P40": [  # children
                {
                    "mainsnak": {
                        "datatype": "wikibase-item",
                        "datavalue": {"value": {"id": "Q200"}},
                    },
                    "rank": "normal",
                },
                {
                    "mainsnak": {
                        "datatype": "wikibase-item",
                        "datavalue": {"value": {"id": "Q201"}},
                    },
                    "rank": "deprecated",
                },
            ],
        },
    )
    assert entity.claim_qids("P22") == ["Q100"]
    assert entity.claim_qids("P40") == ["Q200"]  # deprecated skipped
    assert entity.claim_qids("P99") == []


def test_entity_external_ids():
    entity = Entity(
        qid="Q42",
        claims={
            "P214": [{
                "mainsnak": {
                    "datatype": "external-id",
                    "datavalue": {"value": "113230702"},
                },
                "rank": "normal",
            }],
            "P1566": [{
                "mainsnak": {
                    "datatype": "external-id",
                    "datavalue": {"value": "2988507"},
                },
                "rank": "normal",
            }],
        },
    )
    ids = entity.external_ids(["P214", "P1566", "P227"])
    assert ids == {"P214": "113230702", "P1566": "2988507"}


def test_entity_label_description_accessors():
    entity = Entity(
        qid="Q42",
        labels={"en": "Douglas Adams", "de": "Douglas Adams"},
        descriptions={"en": "British writer", "de": "britischer Schriftsteller"},
    )
    assert entity.label("en") == "Douglas Adams"
    assert entity.label("de") == "Douglas Adams"
    assert entity.label("nonexistent") is None
    assert entity.description("de") == "britischer Schriftsteller"


# --- TokenBucket ---


def test_token_bucket_blocks_when_exhausted():
    bucket = TokenBucket(rate_per_sec=10.0, capacity=2)
    start = time.monotonic()
    bucket.take()  # 2 → 1
    bucket.take()  # 1 → 0
    bucket.take()  # must wait ~0.1s for refill
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08, f"expected ≥0.08s wait, got {elapsed:.3f}s"


def test_token_bucket_refills_over_time():
    bucket = TokenBucket(rate_per_sec=100.0, capacity=1)
    bucket.take()
    time.sleep(0.05)  # earn 5 tokens at 100/s but cap is 1
    # Next take should not block appreciably
    start = time.monotonic()
    bucket.take()
    elapsed = time.monotonic() - start
    assert elapsed < 0.02


def test_token_bucket_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=0)
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=-1)


# --- SPARQL ---


def test_sparql_returns_bindings(no_rate_limit_client_factory):
    fetcher = MockFetcher([
        (matches("sparql"), json_bytes({
            "head": {"vars": ["item", "itemLabel"]},
            "results": {"bindings": [
                {
                    "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q214867"},
                    "itemLabel": {"type": "literal", "value": "Rollo"},
                }
            ]},
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    results = client.sparql("SELECT ?item ?itemLabel WHERE { ... }")
    assert len(results) == 1
    assert results[0]["itemLabel"]["value"] == "Rollo"


def test_sparql_cached(no_rate_limit_client_factory):
    fetcher = MockFetcher([
        (matches("sparql"), json_bytes({
            "results": {"bindings": [{"x": {"value": "1"}}]},
        })),
    ])
    client = no_rate_limit_client_factory(fetcher)
    q = "SELECT ?x WHERE { }"
    client.sparql(q)
    client.sparql(q)
    assert len(fetcher.calls) == 1


# --- Cache disabling ---


def test_no_cache_path_still_works(tmp_path):
    """Constructing without cache_db_path just disables caching."""
    fetcher = MockFetcher([
        (matches("wbsearchentities"), json_bytes({"search": []})),
    ])
    client = WikidataClient(
        user_agent=VALID_UA,
        cache_db_path=None,
        rate_per_sec=1000.0,
        fetcher=fetcher,
    )
    assert client.cache_get is None
    assert client.cache_search is None
    client.search("x")
    client.search("x")
    # Without cache, both calls hit the fetcher
    assert len(fetcher.calls) == 2


def test_explicit_cache_overrides_db_path(tmp_path):
    """Passing cache_search explicitly wins over cache_db_path."""
    explicit_cache = PayloadCache(str(tmp_path / "explicit.db"), source="custom")
    fetcher = MockFetcher([(matches("wbsearchentities"), json_bytes({"search": []}))])
    client = WikidataClient(
        user_agent=VALID_UA,
        cache_db_path=str(tmp_path / "default.db"),
        cache_search=explicit_cache,
        rate_per_sec=1000.0,
        fetcher=fetcher,
    )
    assert client.cache_search is explicit_cache
    assert client.cache_get is not explicit_cache  # cache_get still from db_path
