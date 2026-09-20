"""cached_call — a response cache plus replicate-agreement wrapper around the
existing generate transports (`claude_cli.generate` today; any callable with
the same shape can be plugged in), so a repeated (model, system, prompt,
schema, version) call costs zero the second time and every call still lands
exactly one row in `cost_log`.

Why this exists: `claude_cli` already computes `prompt_sha256` / `system_sha256`
/ `schema_sha256` for every call but nothing reads them back. The 20 Sep 2026
llm-pipeline-audit found 17% of Alif's hashed calls (about $178 notional) and
76% of one Petrarca extraction stage repeated an identical prompt+system+model
with a fresh spend each time — the hash existed, the cache did not. It also
found `purpose` empty on ~40% of ledger rows and 25,642 rows silently
attributed to the literal project "limbic" (an env-var default in
`amygdala.llm.generate_structured`). This module makes both mistakes
structurally harder: `purpose` is a required argument, and an empty `project`
is inferred from the git root rather than defaulted.

Usage — cache a repeated classification call:

    from limbic.cerebellum.calls import cached_call

    result, meta = cached_call(
        "Classify sentiment: I love it",
        project="petrarca", purpose="sentiment",
        schema={"type": "object", "properties": {"label": {"type": "string"}}},
    )
    # A second call with the same prompt/system/schema/model/version is a
    # cache hit: no subprocess, cost_usd=0, and a ledger row with cache_hit=1.

Usage — replicate-agreement instead of trusting a single confidence score
(the audit: three independent reads of the same input agreed only 62.8% of
the time, but exact agreement between two reads lifted precision 0.71 -> 0.97):

    result, meta = cached_call(
        "Is this pair a duplicate?", project="skard", purpose="dedup",
        replicates=3, agree=2, cache=False,
    )
    if isinstance(result, Held):
        ...  # route to a human; result.results holds what each replicate said
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from limbic.amygdala import connect

from .claude_cli import generate as _claude_cli_generate
from .cost_log import cost_log, price_for

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS call_cache (
    cache_key   TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    purpose     TEXT NOT NULL,
    model       TEXT NOT NULL,
    transport   TEXT NOT NULL,
    version     TEXT DEFAULT '',
    result_json TEXT NOT NULL,
    meta_json   TEXT DEFAULT '{}',
    created_at  REAL NOT NULL,
    expires_at  REAL,
    hit_count   INTEGER DEFAULT 0,
    last_hit_at REAL
);
CREATE INDEX IF NOT EXISTS idx_call_cache_project ON call_cache(project);
"""


def _default_cache_db_path() -> Path:
    env = os.environ.get("LIMBIC_CALL_CACHE_DB")
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "limbic" / "llm_cache.db"


def _open(cache_db_path: str | Path | None):
    conn = connect(str(cache_db_path or _default_cache_db_path()))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


@dataclass
class Held:
    """Returned in place of a result when `replicates` disagree past `agree`.

    Not an error — the audit's headline finding made concrete: disagreement
    among independent reads of the same input is a cheap, real hold signal,
    worth surfacing rather than silently keeping the first answer or trusting
    a single model's confidence score.
    """

    reason: str
    results: list[Any]
    metas: list[dict]


@dataclass
class CallMeta:
    """Metadata returned alongside every `cached_call` result."""

    call_id: str
    cache_hit: bool
    cost_usd: float
    model: str
    cache_key: str
    prompt_sha256: str
    system_sha256: str
    schema_sha256: str
    replicate_metas: list[dict] | None = None
    raw: dict = field(default_factory=dict)  # transport-native metadata (session_id, turns, ...)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(*, model: str, system: str, prompt: str, schema: dict | None, version: str) -> str:
    schema_json = json.dumps(schema, sort_keys=True) if schema else ""
    payload = {
        "model": model,
        "system_sha256": _hash(system or ""),
        "prompt_sha256": _hash(prompt),
        "schema_sha256": _hash(schema_json) if schema_json else "",
        "version": version or "",
    }
    return _hash(json.dumps(payload, sort_keys=True))


def _infer_project() -> str:
    """Derive a project tag from the enclosing git repo's directory name.

    Only used when the caller passes an empty `project`. Silently defaulting
    to a fixed string (as `amygdala.llm.generate_structured` does with
    "limbic") is the exact bug this refuses to repeat — it produced 25,642
    unattributable ledger rows. Raises rather than guess further when no git
    root can be found.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception as e:
        raise ValueError(
            "project is required and could not be inferred (git not "
            f"available: {e}); pass project= explicitly"
        ) from e
    top = proc.stdout.strip()
    if proc.returncode != 0 or not top:
        raise ValueError(
            "project is required and could not be inferred (cwd is not "
            "inside a git repository); pass project= explicitly"
        )
    return Path(top).name


def _extract_cost(meta: dict) -> float:
    for key in ("cost", "cost_usd", "total_cost_usd"):
        if key in meta and meta[key] is not None:
            return float(meta[key])
    return 0.0


def _log_call(
    raw_meta: dict, *, project: str, purpose: str, model: str, cost_usd: float,
    cache_hit: bool, metadata: dict,
) -> str:
    """Log a cost_log row for a transport call, unless the transport already
    logged one itself.

    Every self-logging transport in this codebase (`claude_cli.generate`, and
    the built-in `openai`/`gemini` transports below) returns a `call_id` key
    in its metadata for exactly this reason: without this check, `cached_call`
    would write a *second* row carrying the same cost, doubling every reported
    total for the default transport. A transport that doesn't self-log (a
    bespoke callable, a test fake) has no `call_id`, so it falls through to
    `cached_call` logging on its behalf as before. Either way, the caller gets
    back a real ledger row id usable with `record_outcome`.
    """
    existing = raw_meta.get("call_id")
    if existing:
        return existing
    record = cost_log.log(
        project=project, model=model, purpose=purpose, script="cached_call",
        cost_usd=cost_usd, cache_hit=cache_hit, metadata=metadata,
    )
    return record.id


# ---------------------------------------------------------------------------
# Built-in HTTP transports: OpenAI Responses API, Gemini REST.
#
# Both are stdlib-`urllib` only (no `openai`/`google-genai` SDK dependency).
# `google-genai` is deliberately avoided here: it crashes on import on
# arm64-macOS Python builds in at least one project's environment (see
# `reference_polaris_api_keys`), and a REST call is all `cached_call` needs.
# Both self-log to `cost_log` (matching `claude_cli.generate`'s convention)
# and return `call_id` in their metadata so `cached_call` doesn't log a
# second row for the same call — see `_log_call`. This matters because these
# are exactly the transports the cheap, high-volume workers (Kulturbase's
# Luna campaigns, skard's packet runner) call directly today, bypassing the
# ledger entirely; giving them a one-line `cached_call(transport="openai")`
# swap is the point.
# ---------------------------------------------------------------------------


class TransportError(RuntimeError):
    """Raised by the built-in `openai`/`gemini` transports on a failed request
    or a response that doesn't contain the expected output."""


def _http_post_json(url: str, payload: dict, *, headers: dict, timeout: int) -> dict:
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise TransportError(f"HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise TransportError(f"request failed: {e.reason}") from e


def _extract_openai_text(data: dict) -> str:
    # The SDK's convenience `output_text` property isn't guaranteed on the
    # raw REST payload, so fall back to walking `output` message content.
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        return data["output_text"]
    chunks = []
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") in ("output_text", "text") and part.get("text"):
                chunks.append(part["text"])
    if not chunks:
        raise TransportError(f"no output text in Responses API payload: {json.dumps(data)[:300]}")
    return "".join(chunks)


def _openai_generate(
    prompt: str, *, project: str, purpose: str, system: str = "", schema: dict | None = None,
    model: str = "gpt-5.4-mini", max_output_tokens: int = 4096, timeout: int = 120, **_ignored: Any,
) -> tuple[Any, dict]:
    """Built-in transport: OpenAI Responses API via stdlib `urllib`.

    Structured output via a strict `json_schema` text format. Self-logs to
    `cost_log` via `price_for` (so an unpriced model logs a visible $0 with a
    warning rather than the call itself failing). The response's own `id` is
    stored in both the ledger row and (via the returned metadata) the cache
    entry, as `response_id` — a response is retained by OpenAI for 30 days by
    default, so a result lost locally (crashed before the caller persisted
    it) can be fetched back with `GET /v1/responses/{response_id}` instead of
    being re-bought.

    Caching caveat, measured on today's skard pilot: the Responses API's
    automatic prompt caching keys off an exact byte-identical *prefix*, and
    `instructions` + the `text.format` schema come ahead of `input` in that
    prefix. A schema that varies per call — e.g. a per-item `enum` of that
    item's own candidate IDs — changes the prefix on every call and defeats
    caching entirely (measured: 0% cached input). Keep `schema` byte-identical
    across a batch (a generic `id: string` field, not a per-call enum) and
    validate the returned id against the allowed set yourself afterward
    (slot indirection) rather than encoding it in the schema; the same batch
    with an identical schema measured 58% cached input.
    """
    if not project:
        raise ValueError("project is required (used for cost_log attribution)")
    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise TransportError("OPENAI_KEY or OPENAI_API_KEY is not set")

    payload: dict[str, Any] = {"model": model, "input": prompt, "max_output_tokens": max_output_tokens}
    if system:
        payload["instructions"] = system
    if schema:
        payload["text"] = {"format": {"type": "json_schema", "name": "response", "strict": True, "schema": schema}}

    t0 = time.time()
    try:
        data = _http_post_json(
            "https://api.openai.com/v1/responses", payload,
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
        )
    except TransportError as e:
        cost_log.log(project=project, model=model, purpose=purpose, script="cached_call.openai",
                     cost_usd=0.0, metadata={"failed": True, "error": str(e)[:500]})
        raise
    duration_s = time.time() - t0

    text = _extract_openai_text(data)
    result = json.loads(text) if schema else text

    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    inp_price, out_price = price_for(model, strict=False)
    cost_usd = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000

    response_id = data.get("id")
    record = cost_log.log(
        project=project, model=model, purpose=purpose, script="cached_call.openai",
        prompt_tokens=input_tokens, completion_tokens=output_tokens, cached_tokens=cached_tokens,
        cost_usd=cost_usd, metadata={"duration_s": round(duration_s, 2), "response_id": response_id},
    )
    return result, {
        "cost": cost_usd, "model": model, "call_id": record.id, "duration_s": duration_s,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens,
        "response_id": response_id,
    }


def _strip_gemini_schema(schema: Any) -> Any:
    """Gemini's REST schema doesn't accept a `"type": [..., "null"]` union
    (the JSON-Schema style `oas3plus`/pydantic emits) — drop "null" and keep
    the first remaining type. Small local copy of `amygdala.llm`'s private
    helper of the same shape, to avoid depending on another module's
    underscore-prefixed internal."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, list):
            out[k] = next((t for t in v if t != "null"), "string")
        elif isinstance(v, dict):
            out[k] = _strip_gemini_schema(v)
        elif isinstance(v, list):
            out[k] = [_strip_gemini_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


def _extract_gemini_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as e:
        raise TransportError(f"no text in Gemini payload: {json.dumps(data)[:300]}") from e


def _gemini_generate(
    prompt: str, *, project: str, purpose: str, system: str = "", schema: dict | None = None,
    model: str = "gemini-2.5-flash", max_output_tokens: int = 8192, timeout: int = 120, **_ignored: Any,
) -> tuple[Any, dict]:
    """Built-in transport: Gemini REST via stdlib `urllib` — deliberately not
    the `google-genai` SDK (see module-section docstring above). Self-logs to
    `cost_log` via `price_for`. Unlike the OpenAI Responses API, Gemini's
    `generateContent` is stateless — there's no `GET`-by-id endpoint — so its
    `responseId` (when present) is recorded for provenance only, not as a
    "fetch it back" capability."""
    if not project:
        raise ValueError("project is required (used for cost_log attribution)")
    api_key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise TransportError("GEMINI_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY is not set")

    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if schema:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = _strip_gemini_schema(schema)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    t0 = time.time()
    try:
        data = _http_post_json(url, payload, headers={}, timeout=timeout)
    except TransportError as e:
        cost_log.log(project=project, model=model, purpose=purpose, script="cached_call.gemini",
                     cost_usd=0.0, metadata={"failed": True, "error": str(e)[:500]})
        raise
    duration_s = time.time() - t0

    text = _extract_gemini_text(data)
    result = json.loads(text) if schema else text

    usage = data.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    cached_tokens = usage.get("cachedContentTokenCount", 0)
    inp_price, out_price = price_for(model, strict=False)
    cost_usd = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000

    response_id = data.get("responseId")
    record = cost_log.log(
        project=project, model=model, purpose=purpose, script="cached_call.gemini",
        prompt_tokens=input_tokens, completion_tokens=output_tokens, cached_tokens=cached_tokens,
        cost_usd=cost_usd, metadata={"duration_s": round(duration_s, 2), "response_id": response_id},
    )
    return result, {
        "cost": cost_usd, "model": model, "call_id": record.id, "duration_s": duration_s,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens,
        "response_id": response_id,
    }


_DEFAULT_TRANSPORTS: dict[str, Callable[..., tuple[Any, dict]]] = {
    "claude_cli": _claude_cli_generate,
    "openai": _openai_generate,
    "gemini": _gemini_generate,
}


def _resolve_transport(transport: str | Callable[..., tuple[Any, dict]]) -> Callable[..., tuple[Any, dict]]:
    if callable(transport):
        return transport
    try:
        return _DEFAULT_TRANSPORTS[transport]
    except KeyError:
        raise ValueError(
            f"unknown transport {transport!r}; use one of "
            f"{sorted(_DEFAULT_TRANSPORTS)} or pass a callable with the "
            "signature (prompt, *, project, purpose, system, schema, model, "
            "**kwargs) -> (result, meta)"
        ) from None


def cached_call(
    prompt: str,
    *,
    project: str = "",
    purpose: str,
    system: str = "",
    schema: dict | None = None,
    model: str = "haiku",
    version: str = "",
    transport: str | Callable[..., tuple[Any, dict]] = "claude_cli",
    cache: bool | Literal["refresh"] = True,
    ttl_days: int | None = None,
    replicates: int = 1,
    agree: int | None = None,
    cache_db_path: str | Path | None = None,
    **transport_kwargs: Any,
) -> tuple[Any | Held, CallMeta]:
    """Call an LLM transport once, cached by (model, system, prompt, schema, version).

    Args:
        project: Attribution tag for cost_log rows. If empty, inferred from
            the enclosing git repo's directory name; raises if that fails.
        purpose: Required. Task label (e.g. "extract_claims"). A ledger row
            with no purpose cannot be traced back to a pipeline stage.
        version: Caller-supplied prompt/schema version. Bump it to invalidate
            old cache entries without touching model/system/prompt/schema.
        transport: "claude_cli" (default), or any callable with the signature
            `(prompt, *, project, purpose, system, schema, model, **kwargs)
            -> (result, meta)` — pass a fake one in tests to avoid real calls.
        cache: True (default) to read/write the cache, False to bypass it
            entirely, or "refresh" to force a fresh call and overwrite the
            cached entry.
        ttl_days: Cache entry lifetime. None (default) never expires.
        replicates: When > 1, disables the cache and makes `replicates`
            independent calls instead of one. Use with `agree`.
        agree: Minimum number of replicates that must produce the same
            (JSON-canonicalised) result for it to be returned; otherwise a
            `Held` is returned instead. Defaults to `replicates` (unanimous).

    Returns:
        `(result, CallMeta)` — or `(Held(...), CallMeta)` when replicates
        disagree.

    Raises:
        ValueError: `purpose` is empty, `project` is empty and can't be
            inferred, or `agree > replicates`.
    """
    if not purpose:
        raise ValueError(
            "purpose is required — the 20 Sep 2026 audit found it empty on "
            "~40% of ledger rows, which made spend untraceable to a stage"
        )
    project = project or _infer_project()
    fn = _resolve_transport(transport)

    prompt_sha = _hash(prompt)
    system_sha = _hash(system) if system else ""
    schema_sha = _hash(json.dumps(schema, sort_keys=True)) if schema else ""
    key = _cache_key(model=model, system=system, prompt=prompt, schema=schema, version=version)

    if replicates and replicates > 1:
        return _replicated_call(
            fn, prompt=prompt, project=project, purpose=purpose, system=system,
            schema=schema, model=model, replicates=replicates, agree=agree,
            key=key, prompt_sha=prompt_sha, system_sha=system_sha, schema_sha=schema_sha,
            **transport_kwargs,
        )

    # --- 1. Read: check the cache, then release the connection. ---
    if cache and cache != "refresh":
        conn = _open(cache_db_path)
        row = conn.execute(
            "SELECT result_json, meta_json, expires_at FROM call_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is not None and (row["expires_at"] is None or row["expires_at"] > time.time()):
            conn.execute(
                "UPDATE call_cache SET hit_count = hit_count + 1, last_hit_at = ? "
                "WHERE cache_key = ?",
                (time.time(), key),
            )
            conn.commit()
            conn.close()
            result = json.loads(row["result_json"])
            stored_meta = json.loads(row["meta_json"])
            record = cost_log.log(
                project=project, model=model, purpose=purpose, script="cached_call",
                cost_usd=0.0, cache_hit=True,
                metadata={
                    "cache_key": key, "prompt_sha256": prompt_sha,
                    "system_sha256": system_sha, "schema_sha256": schema_sha,
                    "original_cost_usd": stored_meta.get("cost_usd", 0.0),
                },
            )
            return result, CallMeta(
                call_id=record.id, cache_hit=True, cost_usd=0.0, model=model,
                cache_key=key, prompt_sha256=prompt_sha, system_sha256=system_sha,
                schema_sha256=schema_sha, raw=stored_meta,
            )
        conn.close()

    # --- 2. Call: no DB connection held during the (slow) model call. ---
    result, raw_meta = fn(
        prompt, project=project, purpose=purpose, system=system, schema=schema,
        model=model, **transport_kwargs,
    )
    cost_usd = _extract_cost(raw_meta)
    resolved_model = raw_meta.get("model", model)

    call_id = _log_call(
        raw_meta, project=project, purpose=purpose, model=resolved_model,
        cost_usd=cost_usd, cache_hit=False,
        metadata={
            "cache_key": key, "prompt_sha256": prompt_sha,
            "system_sha256": system_sha, "schema_sha256": schema_sha,
            "transport_meta": raw_meta,
        },
    )

    # --- 3. Write: short, separate connection. ---
    if cache:
        conn = _open(cache_db_path)
        expires_at = (time.time() + ttl_days * 86400) if ttl_days else None
        transport_name = transport if isinstance(transport, str) else getattr(transport, "__name__", "callable")
        conn.execute(
            "INSERT OR REPLACE INTO call_cache "
            "(cache_key, project, purpose, model, transport, version, "
            " result_json, meta_json, created_at, expires_at, hit_count, last_hit_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)",
            (
                key, project, purpose, resolved_model, transport_name, version,
                json.dumps(result), json.dumps({"cost_usd": cost_usd, **raw_meta}),
                time.time(), expires_at,
            ),
        )
        conn.commit()
        conn.close()

    return result, CallMeta(
        call_id=call_id, cache_hit=False, cost_usd=cost_usd, model=resolved_model,
        cache_key=key, prompt_sha256=prompt_sha, system_sha256=system_sha,
        schema_sha256=schema_sha, raw=raw_meta,
    )


def _replicated_call(
    fn: Callable[..., tuple[Any, dict]],
    *,
    prompt: str,
    project: str,
    purpose: str,
    system: str,
    schema: dict | None,
    model: str,
    replicates: int,
    agree: int | None,
    key: str,
    prompt_sha: str,
    system_sha: str,
    schema_sha: str,
    **transport_kwargs: Any,
) -> tuple[Any, CallMeta]:
    if agree is None:
        agree = replicates
    if agree > replicates:
        raise ValueError(f"agree ({agree}) cannot exceed replicates ({replicates})")

    results: list[Any] = []
    metas: list[dict] = []
    call_ids: list[str] = []
    for i in range(replicates):
        result, raw_meta = fn(
            prompt, project=project, purpose=purpose, system=system, schema=schema,
            model=model, **transport_kwargs,
        )
        cost_usd = _extract_cost(raw_meta)
        call_id = _log_call(
            raw_meta, project=project, purpose=purpose, model=raw_meta.get("model", model),
            cost_usd=cost_usd, cache_hit=False,
            metadata={
                "cache_key": key, "replicate_index": i,
                "prompt_sha256": prompt_sha, "system_sha256": system_sha,
                "schema_sha256": schema_sha, "transport_meta": raw_meta,
            },
        )
        results.append(result)
        metas.append(raw_meta)
        call_ids.append(call_id)

    # Compare on canonical JSON so dict-vs-dict and str-vs-str both compare
    # structurally rather than by object identity.
    canon = [r if isinstance(r, str) else json.dumps(r, sort_keys=True) for r in results]
    counts = Counter(canon)
    winner, n = counts.most_common(1)[0]
    total_cost = sum(_extract_cost(m) for m in metas)
    call_meta = CallMeta(
        call_id=call_ids[0], cache_hit=False, cost_usd=total_cost, model=model,
        cache_key=key, prompt_sha256=prompt_sha, system_sha256=system_sha,
        schema_sha256=schema_sha, replicate_metas=metas,
    )
    if n >= agree:
        return results[canon.index(winner)], call_meta
    return Held(
        reason=f"{n}/{replicates} replicates agreed, needed {agree}",
        results=results, metas=metas,
    ), call_meta
