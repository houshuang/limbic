"""Thin Wikidata API client, cache-backed, rate-limited.

Provides three operations against the Wikidata REST + SPARQL endpoints:

- `search(name)`    — `wbsearchentities`, ranked candidates (API-popularity-biased).
- `get(qid)`        — `wbgetentities`, full entity (labels, aliases, descriptions, claims).
- `get_many(qids)`  — batched `wbgetentities` (up to 50 QIDs per HTTP call).
- `sparql(query)`   — POST to `query.wikidata.org/sparql`, returns bindings list.

All responses are cached via `PayloadCache` (30-day default TTL). Rate limited
via an in-process token bucket (Wikidata policy: ≤5 req/s). The `maxlag=5`
parameter is sent with every API request so the server can politely throttle
us during replication lag. Retries with exponential-backoff jitter on 429 and
5xx; single back-off on maxlag violations.

`User-Agent` is mandatory per Wikidata's UA policy — a bare "Python-urllib/X"
agent can be blocked. Construct with a descriptive string including contact:

    client = WikidataClient(
        cache_db_path="wikidata_cache.db",
        user_agent="Petrarca/0.1 (mailto:stian@example.com) limbic/0.1",
    )

Hallucination defense: this client never *generates* QIDs. It returns only
what the API returned. Resolvers that call an LLM to disambiguate candidates
must treat any QID the LLM emits as invalid unless it appears in the candidate
set returned by `search` or `get_many`.

Thread safety: a `WikidataClient` is not safe to share across threads. The
underlying `PayloadCache` holds a `sqlite3.Connection` which defaults to
`check_same_thread=True`; cross-thread use will raise. If you need
concurrency, give each worker its own client (file-backed caches coexist
cleanly via SQLite WAL).

Testing: the constructor accepts a `fetcher` callable for HTTP injection
(signature: `(urllib.request.Request, float timeout) -> bytes`). Unit tests
pass a mock fetcher; live smoke tests leave it at its default (`urlopen`).
"""

from __future__ import annotations

import json
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .cache import PayloadCache


# Default multilingual set. Matches the user's reading languages; expand/shrink
# per caller via the `languages` parameter on get/get_many.
DEFAULT_LANGS: tuple[str, ...] = (
    "en", "no", "nb", "nn", "sv", "da", "it", "de", "fr", "es", "zh", "id",
)

# Wikidata API limits + policy
WBGETENTITIES_MAX_IDS = 50        # per-call batch limit for `ids=Q1|Q2|...`
DEFAULT_RATE_PER_SEC = 5.0        # ≤5 req/s per Wikidata UA policy
DEFAULT_MAXLAG_SECONDS = 5        # server-side lag threshold
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 5  # enough to ride out ~30-40s of replication lag

# Cache source tags — separate sources keep `clear_source` and
# `invalidate_before` scoped per operation type.
CACHE_SOURCE_GET = "wikidata_get"
CACHE_SOURCE_SEARCH = "wikidata_search"
CACHE_SOURCE_SPARQL = "wikidata_sparql"

API_BASE_DEFAULT = "https://www.wikidata.org/w/api.php"
SPARQL_BASE_DEFAULT = "https://query.wikidata.org/sparql"


class WikidataError(Exception):
    """Generic Wikidata API error."""


class MaxlagError(WikidataError):
    """Wikidata server reported replication lag above our threshold."""


class WikidataNotFound(WikidataError):
    """A specific QID returned as missing or does not exist."""


# --- dataclasses ---


@dataclass
class Candidate:
    """A single candidate from `wbsearchentities`.

    `rank` is the position in the API's returned list (0 = first). Wikidata
    ranks by popularity/sitelinks — "Paris" returns the city before the
    Trojan prince. Treat rank as a weak prior at best; rerank by context.
    """
    qid: str
    label: str
    description: str = ""
    match_text: str = ""
    match_type: str = ""       # "label" | "alias" | "description" | ""
    rank: int = 0
    aliases: list[str] = field(default_factory=list)


@dataclass
class Entity:
    """A Wikidata entity payload from `wbgetentities`."""
    qid: str
    labels: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    claims: dict[str, list[dict]] = field(default_factory=dict)
    redirect_from: str | None = None   # populated when the requested QID was redirected
    modified: str | None = None        # last-modified timestamp from API

    def label(self, lang: str = "en") -> str | None:
        return self.labels.get(lang)

    def description(self, lang: str = "en") -> str | None:
        return self.descriptions.get(lang)

    def claim_qids(self, property_id: str) -> list[str]:
        """QIDs referenced by `wikibase-item` claims of this property.

        Returns [] if the property isn't present or has no wikibase-item
        values. Skips claims with deprecated rank.
        """
        out: list[str] = []
        for stmt in self.claims.get(property_id, []):
            if stmt.get("rank") == "deprecated":
                continue
            mainsnak = stmt.get("mainsnak") or {}
            if mainsnak.get("datatype") != "wikibase-item":
                continue
            value = (mainsnak.get("datavalue") or {}).get("value") or {}
            if isinstance(value, dict) and "id" in value:
                out.append(value["id"])
        return out

    def external_ids(self, properties: Iterable[str]) -> dict[str, str]:
        """Extract `external-id` claims (P214 VIAF, P1566 GeoNames, …).

        Returns `{property_id: value}` for single-valued properties. If a
        property has multiple values, returns the first non-deprecated one;
        callers needing the full list can walk `self.claims` directly.
        """
        out: dict[str, str] = {}
        for p in properties:
            for stmt in self.claims.get(p, []):
                if stmt.get("rank") == "deprecated":
                    continue
                mainsnak = stmt.get("mainsnak") or {}
                if mainsnak.get("datatype") != "external-id":
                    continue
                val = (mainsnak.get("datavalue") or {}).get("value")
                if isinstance(val, str):
                    out[p] = val
                    break
        return out


# --- token bucket ---


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Capacity tokens refill at `rate_per_sec`. `take(n)` blocks (via
    `time.sleep`) until n tokens are available. Designed for polite
    external-API throttling; not intended for high-throughput workloads.
    """

    def __init__(self, rate_per_sec: float, capacity: int | None = None):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity if capacity is not None else max(1, int(rate_per_sec)))
        self.tokens = self.capacity
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def take(self, n: float = 1.0) -> None:
        if n <= 0:
            return
        with self.lock:
            now = time.monotonic()
            # Refill based on elapsed time since last take.
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return
            deficit = n - self.tokens
            wait = deficit / self.rate
        time.sleep(wait)
        with self.lock:
            # Consume tokens earned by the wait.
            self.last = time.monotonic()
            self.tokens = 0.0


# --- client ---


Fetcher = Callable[[urllib.request.Request, float], bytes]


class WikidataClient:
    """Cache-backed, rate-limited Wikidata client.

    Parameters
    ----------
    cache_db_path:
        SQLite file for all three internal PayloadCaches. Pass `":memory:"`
        for an ephemeral cache or `None` to disable caching entirely.
    user_agent:
        Required. Must include contact info per Wikidata UA policy.
    cache_get, cache_search, cache_sparql:
        Explicit PayloadCache overrides. Override when you need fine-grained
        control (separate TTLs, different files per source). Mutually
        exclusive with cache_db_path for that source; explicit wins.
    rate_per_sec:
        Token-bucket refill rate. Default 5 req/s per Wikidata policy.
    fetcher:
        HTTP transport override. Default uses `urllib.request.urlopen`. Tests
        inject a fake fetcher with signature `(Request, timeout) -> bytes`.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        cache_db_path: str | None = None,
        cache_get: PayloadCache | None = None,
        cache_search: PayloadCache | None = None,
        cache_sparql: PayloadCache | None = None,
        rate_per_sec: float = DEFAULT_RATE_PER_SEC,
        api_base: str = API_BASE_DEFAULT,
        sparql_base: str = SPARQL_BASE_DEFAULT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        maxlag_seconds: int = DEFAULT_MAXLAG_SECONDS,
        ssl_context: ssl.SSLContext | None = None,
        fetcher: Fetcher | None = None,
    ):
        if not user_agent or "(" not in user_agent:
            raise ValueError(
                "user_agent must include contact info per Wikidata UA policy, "
                "e.g. 'MyApp/1.0 (mailto:you@example.com)'. See "
                "https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy"
            )
        self.user_agent = user_agent
        self.api_base = api_base
        self.sparql_base = sparql_base
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.maxlag = maxlag_seconds
        self.bucket = TokenBucket(rate_per_sec)
        self.ssl_context = ssl_context or _default_ssl_context()
        self._fetch = fetcher or self._default_fetch

        # Wire caches: explicit overrides win, else derive from cache_db_path,
        # else no caching. Per-source so invalidate/clear stay scoped.
        self.cache_get = cache_get
        self.cache_search = cache_search
        self.cache_sparql = cache_sparql
        if cache_db_path is not None:
            if self.cache_get is None:
                self.cache_get = PayloadCache(cache_db_path, source=CACHE_SOURCE_GET)
            if self.cache_search is None:
                self.cache_search = PayloadCache(cache_db_path, source=CACHE_SOURCE_SEARCH)
            if self.cache_sparql is None:
                self.cache_sparql = PayloadCache(cache_db_path, source=CACHE_SOURCE_SPARQL)

    # ---------- public API ----------

    def search(
        self,
        name: str,
        *,
        limit: int = 10,
        language: str = "en",
    ) -> list[Candidate]:
        """Search Wikidata for entities whose label/alias matches `name`.

        Wraps `wbsearchentities`. Results are ranked by the API (popularity
        /sitelink-biased) — your resolver must rerank by context.

        `language` is a single language tag. Wikidata's search endpoint is
        single-language per call; for multilingual resolution, call multiple
        times and merge. `limit` is passed through to the API, which caps
        it at 50 — higher values are silently truncated server-side.
        """
        if not name:
            return []
        key = f"{language}|{name.lower()}|limit={limit}"

        def do() -> dict:
            params = {
                "action": "wbsearchentities",
                "search": name,
                "language": language,
                "limit": str(limit),
                "format": "json",
                "maxlag": str(self.maxlag),
            }
            return self._request_json_get(self.api_base, params)

        payload = self._cached_or_fetch(self.cache_search, key, do)
        return _parse_search_results(payload)

    def get(
        self,
        qid: str,
        *,
        languages: Iterable[str] | None = None,
    ) -> Entity | None:
        """Fetch a single entity by QID. Returns None if missing."""
        out = self.get_many([qid], languages=languages)
        return out.get(qid)

    def get_many(
        self,
        qids: list[str],
        *,
        languages: Iterable[str] | None = None,
    ) -> dict[str, Entity]:
        """Fetch multiple entities in 50-QID batches.

        Returns `{requested_qid: Entity}` preserving the order of `qids`.
        Missing entities are simply absent from the output. If a QID was
        redirected, `Entity.redirect_from` contains the original requested
        QID and `Entity.qid` is the canonical target.
        """
        if not qids:
            return {}
        langs = list(languages) if languages is not None else list(DEFAULT_LANGS)
        langs_key = ",".join(langs)

        # Phase 1: cache lookup
        cache_hits: dict[str, Entity] = {}
        if self.cache_get is not None:
            cache_keys = {q: f"{q}|langs={langs_key}" for q in qids}
            hits = self.cache_get.get_many(list(cache_keys.values()))
            for q, k in cache_keys.items():
                if k in hits:
                    cache_hits[q] = _entity_from_dict(hits[k])

        missing = [q for q in qids if q not in cache_hits]

        # Phase 2: fetch missing in batches of WBGETENTITIES_MAX_IDS
        fetched: dict[str, Entity] = {}
        for start in range(0, len(missing), WBGETENTITIES_MAX_IDS):
            batch = missing[start:start + WBGETENTITIES_MAX_IDS]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "languages": "|".join(langs),
                "format": "json",
                "maxlag": str(self.maxlag),
            }
            payload = self._request_json_get(self.api_base, params)

            # Redirects: {"from": "Q123", "to": "Q456"}. Requesting Q123
            # returns the entity under Q456.
            redirects: dict[str, str] = {}
            for redir in payload.get("redirects") or []:
                src = redir.get("from")
                dst = redir.get("to")
                if src and dst:
                    redirects[src] = dst

            entities = payload.get("entities") or {}
            for requested_qid in batch:
                resolved_qid = redirects.get(requested_qid, requested_qid)
                ent_data = entities.get(resolved_qid)
                if ent_data is None or "missing" in ent_data:
                    continue
                entity = _entity_from_api(ent_data)
                if requested_qid in redirects:
                    entity.redirect_from = requested_qid
                if self.cache_get is not None:
                    self.cache_get.put(
                        f"{requested_qid}|langs={langs_key}",
                        _entity_to_dict(entity),
                    )
                fetched[requested_qid] = entity

        # Phase 3: reassemble in input order
        out: dict[str, Entity] = {}
        for q in qids:
            if q in cache_hits:
                out[q] = cache_hits[q]
            elif q in fetched:
                out[q] = fetched[q]
        return out

    def sparql(self, query: str) -> list[dict]:
        """Execute a SPARQL query at `query.wikidata.org`.

        Returns the `results.bindings` list directly — each binding is a dict
        of variable name → `{"type": ..., "value": ...}`. Callers typically
        project out just the `.value`.
        """
        key = query.strip()

        def do() -> dict:
            body = urllib.parse.urlencode({"query": query}).encode("utf-8")
            req_factory = lambda: urllib.request.Request(
                self.sparql_base,
                data=body,
                method="POST",
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/sparql-results+json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            return self._request_json_with_retry(req_factory)

        payload = self._cached_or_fetch(self.cache_sparql, key, do)
        return ((payload.get("results") or {}).get("bindings") or [])

    # ---------- internals ----------

    def _cached_or_fetch(
        self,
        cache: PayloadCache | None,
        key: str,
        fetch_fn: Callable[[], dict],
    ) -> dict:
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                return cached
        payload = fetch_fn()
        if cache is not None:
            cache.put(key, payload)
        return payload

    def _request_json_get(self, url: str, params: dict) -> dict:
        """GET + parse JSON + check for API error, with retries on all retryable
        failure modes (HTTP 429/5xx, URL errors, and API-level maxlag)."""
        def build_request() -> urllib.request.Request:
            full_url = f"{url}?{urllib.parse.urlencode(params)}"
            return urllib.request.Request(
                full_url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            )
        return self._request_json_with_retry(build_request)

    def _request_json_with_retry(
        self, request_factory: Callable[[], urllib.request.Request]
    ) -> dict:
        """Run (rate-limit, fetch, parse, check-error) with retries.

        Retries on HTTP 429/5xx, network errors, and API-level `maxlag`.
        Non-retryable API errors (anything else) raise WikidataError immediately.
        Non-retryable HTTP errors (4xx other than 429) also raise immediately.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self.bucket.take()
            try:
                raw = self._fetch(request_factory(), self.timeout)
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code in (429, 500, 502, 503, 504):
                    wait = _parse_retry_after(getattr(e, "headers", None))
                    if wait is None:
                        wait = _backoff_seconds(attempt)
                    time.sleep(min(wait, 30.0))
                    continue
                raise WikidataError(f"HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                # SSL errors are never transient — fail fast with a clear
                # message instead of wasting retries. Common cause: missing
                # CA bundle on the Python install; pass `ssl_context` to
                # WikidataClient to fix (e.g. ssl.create_default_context(
                # cafile=certifi.where())).
                if isinstance(e.reason, ssl.SSLError):
                    raise WikidataError(
                        f"SSL error fetching {request_factory().full_url}: {e.reason}. "
                        "If your Python install lacks a CA bundle, pass ssl_context=... "
                        "to WikidataClient (e.g. certifi.where())."
                    ) from e
                last_exc = e
                time.sleep(_backoff_seconds(attempt))
                continue

            payload = json.loads(raw)
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"] or {}
                code = err.get("code", "")
                info = err.get("info", "")
                if code == "maxlag":
                    # Retryable: server lag will settle. Parse the reported
                    # lag from the info string and wait at least that long —
                    # retrying sooner would just get rejected again.
                    last_exc = MaxlagError(info or "maxlag")
                    lag = _parse_maxlag_seconds(info)
                    wait = max(5.0, (lag or 0.0) + 2.0) + random.uniform(0, 2.0)
                    time.sleep(min(wait, 30.0))
                    continue
                raise WikidataError(f"{code}: {info}")
            return payload
        raise WikidataError(
            f"max retries ({self.max_retries}) exceeded"
        ) from last_exc

    def _default_fetch(self, req: urllib.request.Request, timeout: float) -> bytes:
        with urllib.request.urlopen(req, timeout=timeout, context=self.ssl_context) as resp:
            return resp.read()


# --- helpers ---


def _parse_search_results(payload: dict) -> list[Candidate]:
    out: list[Candidate] = []
    for i, s in enumerate(payload.get("search") or []):
        match = s.get("match") or {}
        out.append(
            Candidate(
                qid=s.get("id", ""),
                label=s.get("label", ""),
                description=s.get("description", ""),
                match_text=match.get("text", ""),
                match_type=match.get("type", ""),
                rank=i,
                aliases=list(s.get("aliases") or []),
            )
        )
    return out


def _entity_from_api(data: dict) -> Entity:
    qid = data.get("id", "")
    labels = {
        lang: (v.get("value") or "")
        for lang, v in (data.get("labels") or {}).items()
    }
    descriptions = {
        lang: (v.get("value") or "")
        for lang, v in (data.get("descriptions") or {}).items()
    }
    aliases = {
        lang: [a.get("value", "") for a in lst]
        for lang, lst in (data.get("aliases") or {}).items()
    }
    claims = data.get("claims") or {}
    return Entity(
        qid=qid,
        labels=labels,
        aliases=aliases,
        descriptions=descriptions,
        claims=claims,
        modified=data.get("modified"),
    )


def _entity_to_dict(entity: Entity) -> dict:
    return {
        "qid": entity.qid,
        "labels": entity.labels,
        "aliases": entity.aliases,
        "descriptions": entity.descriptions,
        "claims": entity.claims,
        "redirect_from": entity.redirect_from,
        "modified": entity.modified,
    }


def _entity_from_dict(d: dict) -> Entity:
    return Entity(
        qid=d.get("qid", ""),
        labels=d.get("labels") or {},
        aliases=d.get("aliases") or {},
        descriptions=d.get("descriptions") or {},
        claims=d.get("claims") or {},
        redirect_from=d.get("redirect_from"),
        modified=d.get("modified"),
    )


_MAXLAG_SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*seconds?\s*lagged", re.IGNORECASE)


def _parse_maxlag_seconds(info: str) -> float | None:
    """Extract the lag value (seconds) from a Wikidata maxlag error message.

    Wikidata's maxlag errors look like: "Waiting for wdqs1014: 6.8 seconds
    lagged." Parsing the number lets the retry wait out the *actual* lag
    rather than a fixed guess.
    """
    if not info:
        return None
    m = _MAXLAG_SECONDS_RE.search(info)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _default_ssl_context() -> ssl.SSLContext | None:
    """Return an SSL context using certifi's CA bundle when available.

    Python installs on macOS and elsewhere sometimes ship without a populated
    system CA store, causing urllib to fail with `CERTIFICATE_VERIFY_FAILED`.
    If `certifi` is installed we use its bundle; otherwise we return None,
    which means `urlopen` falls back to its platform default (works on most
    systems).
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff (1.5^attempt) plus up to an equal jitter."""
    base = 1.5 ** attempt
    return base + random.uniform(0, base)


def _parse_retry_after(headers) -> float | None:
    """Extract Retry-After seconds from response headers.

    Only integer/float seconds are parsed. HTTP-date format is ignored
    (rare in practice for Wikidata's 429/5xx responses); caller falls back
    to exponential backoff.
    """
    if headers is None:
        return None
    try:
        val = headers.get("Retry-After") if hasattr(headers, "get") else None
    except Exception:
        return None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
