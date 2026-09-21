"""Stateless work packets: the unit of work that is a *call*, not an agent.

The single largest finding of the 20 Sep 2026 llm-pipeline-audit: a cheap model
inside a tool-using harness is not a cheap call. One traced 25-page packet cost
**3.8M tokens** as a Luna subagent (36 tool calls, context 27K -> 146K, five
throw-away parsers, hand-repaired JSON) against **~40K** as one stateless
structured-output call — 95x. Across one project, 1,283M agent tokens produced
a graph whose curated JSON was 7,000x smaller than the tokens spent on it.

A packet is frozen, hashed and self-contained: a byte-identical static prefix,
a variable body, a fixed schema. No conversation, no tools, no growing context.

Usage — build packets, check they are worth building machinery for, then run:

    from limbic.cerebellum.packet import make_packet, probe, run_packets

    packets = [make_packet(PREFIX, body, SCHEMA, prompt_version="v1")
               for body in bodies]
    print(lint_packet(packets))                      # cache and schema warnings
    report = probe(packets, n=50, yield_fn=len, project="skard",
                   purpose="code_spans", execute=True)
    if report["yield_rate"] < 0.2:
        ...  # a deterministic join probably does this job; do not scale
    result = run_packets(packets, project="skard", purpose="code_spans",
                         max_calls=200, max_tokens=2_000_000, execute=True)

`execute=False` is the default everywhere: a dry run prices the batch and
returns what it *would* send, which is the cheapest thing you can do before
committing to a campaign.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from .calls import Held, cached_call
from .cost_log import cost_log, price_for

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_META_PHRASES",
    "DEFAULT_VACUOUS",
    "SLOT_ID",
    "LowYield",
    "Packet",
    "corpus_lowercase_words",
    "estimate_tokens",
    "lint_packet",
    "make_packet",
    "meta_leak_refusals",
    "probe",
    "reanchor_quote",
    "rendering_fidelity_refusals",
    "run_packets",
    "slot_echo_refusal",
    "text_quote_anchor",
    "union_passes",
    "unmatched_names",
    "unresolved_text_quote_anchor",
    "validate_quotes",
]

# Provider prefix caching only engages above roughly a thousand tokens. Below
# it a cache key is pure overhead — and worse: in one 35k-episode campaign 20
# unique records per packet left the shared prefix under this minimum, so 92.5%
# of input was billed as cache *writes* and caching **cost 9.7% more**.
MIN_CACHE_PREFIX_TOKENS = 1_024

# chars/4 under-counts real corpora. Estimating over the whole serialised
# request and scaling bounded both pilot calls from above (9,265 -> 9,821 vs
# 9,389 actual; 8,953 -> 9,490 vs 9,379 actual). A budget must never be crossed
# by a call the estimate cleared.
ESTIMATE_SAFETY_FACTOR = 1.06

# Past roughly this many items per call a model starts dropping items
# silently rather than reporting that it did.
MAX_EXPECTED_ITEMS = 25

_WHITESPACE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic token estimate for budget arithmetic."""
    return int(len(text) / 4 * ESTIMATE_SAFETY_FACTOR) + 1


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Packet(dict):
    """A frozen work unit. Mutating it after construction is refused.

    `input_sha256` covers the prefix hash, the body and the schema, so a packet
    that was paid for can be recognised later. A shared, mutable per-batch
    prefix file is the trap this closes: editing the instructions afterwards
    silently invalidated a batch that had already been bought.
    """

    _frozen = False

    def _freeze(self) -> "Packet":
        self._frozen = True
        return self

    def _refuse(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            f"packet {self.get('packet_id')} is frozen; its input_sha256 "
            "identifies exactly these bytes. Build a new packet instead.")

    def __setitem__(self, key: Any, value: Any) -> None:
        if self._frozen:
            self._refuse()
        super().__setitem__(key, value)

    __delitem__ = _refuse  # type: ignore[assignment]
    clear = _refuse  # type: ignore[assignment]
    pop = _refuse  # type: ignore[assignment]
    popitem = _refuse  # type: ignore[assignment]
    setdefault = _refuse  # type: ignore[assignment]
    update = _refuse  # type: ignore[assignment]


def make_packet(
    static_prefix: str,
    body: Mapping[str, Any] | str,
    schema: Mapping[str, Any] | None,
    *,
    prompt_version: str,
    packet_id: str | None = None,
    max_output_tokens: int | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Packet:
    """Freeze one call's input and hash it.

    `static_prefix` must be byte-identical across the batch (that is what buys
    the prefix cache and what `lint_packet` checks). `body` is everything that
    varies — pages, candidate cards, slots. `schema` must also be identical
    across the batch: it is rendered *ahead* of the input in the provider's
    cache prefix, so a per-item `enum` of that item's own candidate IDs defeats
    caching entirely (measured 0% cached input; the same batch with a fixed
    slot enum measured 58%). Use `hippocampus.resolve.slot_enum` instead.
    """
    body_text = body if isinstance(body, str) else _canonical(body)
    prefix_sha = _sha(static_prefix)
    schema_sha = _sha(_canonical(schema)) if schema else ""
    packet = Packet({
        "prompt_version": prompt_version,
        "static_prefix": static_prefix,
        "static_prefix_sha256": prefix_sha,
        "static_prefix_tokens": estimate_tokens(static_prefix),
        "body": dict(body) if isinstance(body, Mapping) else body,
        "body_text": body_text,
        "schema": dict(schema) if schema else None,
        "schema_sha256": schema_sha,
        "max_output_tokens": max_output_tokens,
        "meta": dict(meta or {}),
    })
    packet["input_sha256"] = _sha(_canonical({
        "prompt_version": prompt_version,
        "static_prefix_sha256": prefix_sha,
        "body_text": body_text,
        "schema_sha256": schema_sha,
    }))
    packet["packet_id"] = packet_id or packet["input_sha256"][:16]
    packet["estimated_input_tokens"] = (
        packet["static_prefix_tokens"] + estimate_tokens(body_text)
        + estimate_tokens(_canonical(schema) if schema else ""))
    return packet._freeze()


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

def lint_packet(packet: Packet | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[str]:
    """Warnings about a packet, or about a batch given several.

    Cross-packet checks (a schema or prefix that varies) need the batch, so
    pass the list. Each warning names the measured failure behind it, because
    every one of them cost real money in a shipped campaign.
    """
    if isinstance(packet, Mapping):
        packets: list[Mapping[str, Any]] = [packet]
    else:
        packets = list(packet)
    if not packets:
        return []
    warnings: list[str] = []

    prefix_hashes = {p.get("static_prefix_sha256") for p in packets}
    schema_hashes = {p.get("schema_sha256") for p in packets}
    if len(packets) > 1 and len(schema_hashes) > 1:
        warnings.append(
            f"the response schema varies across the batch ({len(schema_hashes)} "
            "distinct schemas). The schema is rendered ahead of the input in "
            "the provider's cache prefix, so a per-packet enum defeats caching "
            "entirely (measured 0% cached input; a fixed slot enum on the same "
            "batch measured 58%). Use resolve.slot_enum and map slots back.")
    if len(packets) > 1 and len(prefix_hashes) > 1:
        warnings.append(
            f"the static prefix varies across the batch ({len(prefix_hashes)} "
            "distinct prefixes); nothing will be cached. Move anything that "
            "varies into the body.")

    smallest = min(int(p.get("static_prefix_tokens") or 0) for p in packets)
    if smallest < MIN_CACHE_PREFIX_TOKENS:
        warnings.append(
            f"the shared prefix is ~{smallest} tokens, under the provider "
            f"cache minimum of {MIN_CACHE_PREFIX_TOKENS}. Do not set a prompt "
            "cache key: below the minimum, caching billed 92.5% of one "
            "campaign's input as cache writes and cost 9.7% more than not "
            "caching. Either enlarge the shared prefix or drop the key.")

    warnings.extend(_derivable_field_warnings(packets))

    biggest = max(_expected_items(p) for p in packets)
    if biggest > MAX_EXPECTED_ITEMS:
        warnings.append(
            f"a packet offers {biggest} items' worth of work; past "
            f"~{MAX_EXPECTED_ITEMS} a call starts dropping items silently. "
            "Cut the packet rather than raising the output cap.")
    return warnings


def _expected_items(packet: Mapping[str, Any]) -> int:
    body = packet.get("body")
    if not isinstance(body, Mapping):
        return 0
    return max((len(v) for v in body.values() if isinstance(v, (list, tuple))), default=0)


def _derivable_field_warnings(packets: Sequence[Mapping[str, Any]]) -> list[str]:
    """Body fields that carry no information and should not be paid for.

    46% of each packet in one campaign was an `evidence_fields` list that was
    identical on every record and derivable from the schema.
    """
    warnings: list[str] = []
    bodies = [p.get("body") for p in packets if isinstance(p.get("body"), Mapping)]
    if not bodies:
        return warnings

    if len(bodies) > 1:
        constant = [
            key for key in bodies[0]
            if all(key in b and _canonical(b[key]) == _canonical(bodies[0][key]) for b in bodies)
            and len(_canonical(bodies[0][key])) > 200
        ]
        for key in constant:
            warnings.append(
                f"body field {key!r} is byte-identical in every packet "
                f"(~{estimate_tokens(_canonical(bodies[0][key]))} tokens each). "
                "Move it into the static prefix, where it is paid for once and "
                "then cached, instead of once per call.")

    for key, value in bodies[0].items():
        if isinstance(value, list) and len(value) > 3:
            rendered = {_canonical(v) for v in value}
            if len(rendered) == 1:
                warnings.append(
                    f"body field {key!r} repeats one identical value "
                    f"{len(value)} times; it is derivable, not data.")
    return warnings


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _usage_tokens(meta: Any) -> int:
    raw = getattr(meta, "raw", None) or {}
    total = 0
    for key in ("input_tokens", "output_tokens"):
        value = raw.get(key)
        if isinstance(value, int):
            total += value
    return total


def _default_is_truncated(result: Any, meta: Any, packet: Mapping[str, Any]) -> bool:
    raw = getattr(meta, "raw", None) or {}
    if raw.get("status") == "incomplete":
        return True
    if (raw.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
        return True
    cap = packet.get("max_output_tokens")
    produced = raw.get("output_tokens")
    return bool(cap and isinstance(produced, int) and produced >= cap)


def run_packets(
    packets: Iterable[Packet | Mapping[str, Any]],
    *,
    purpose: str,
    project: str = "",
    model: str = "haiku",
    transport: str | Callable[..., tuple[Any, dict]] = "openai",
    max_calls: int = 50,
    max_tokens: int = 1_000_000,
    replicates: int = 1,
    agree: int | None = None,
    execute: bool = False,
    split: Callable[[Mapping[str, Any]], Sequence[Packet]] | None = None,
    is_truncated: Callable[[Any, Any, Mapping[str, Any]], bool] | None = None,
    outcome_fn: Callable[[Any], str | None] | None = None,
    cache: bool | str = True,
    **transport_kwargs: Any,
) -> dict[str, Any]:
    """Send a batch of frozen packets as stateless calls, under hard budgets.

    `execute=False` (the default) prices the batch and returns what it *would*
    send without calling anything — run that first, always.

    Budgets are refusals, not advice: the run stops at `max_calls` or when the
    next call's *estimated* tokens would cross `max_tokens`, and reports what
    is left rather than finishing the job over budget.

    Truncation is answered by `split` — a callable returning smaller packets —
    **once** per packet, never by re-asking. Re-asking the same packet spends
    the same tokens on the same overflow.

    Every call lands exactly one ledger row, including failures, carrying the
    packet id and (for the OpenAI transport) the provider's `response_id`, so a
    result lost locally can be fetched back instead of re-bought.
    """
    packets = list(packets)
    is_truncated = is_truncated or _default_is_truncated
    estimated = sum(int(p.get("estimated_input_tokens") or 0) for p in packets) * replicates
    inp_price, out_price = price_for(model, strict=False)
    report: dict[str, Any] = {
        "packets": len(packets),
        "planned_calls": len(packets) * replicates,
        "estimated_input_tokens": estimated,
        "estimated_cost_usd": round(estimated * inp_price / 1_000_000, 4),
        "model": model,
        "warnings": lint_packet(packets),
        "executed": bool(execute),
        "calls": 0,
        "tokens": 0,
        "cost_usd": 0.0,
        "results": [],
        "failures": [],
        "held": [],
        "split": [],
        "stopped": None,
    }
    if not execute:
        report["stopped"] = "dry run (execute=False)"
        return report

    queue = list(packets)
    split_done: set[str] = set()
    while queue:
        packet = queue.pop(0)
        estimate = int(packet.get("estimated_input_tokens") or 0)
        if report["calls"] + replicates > max_calls:
            report["stopped"] = f"max_calls={max_calls} reached"
            queue.insert(0, packet)
            break
        if report["tokens"] + estimate * replicates > max_tokens:
            report["stopped"] = f"max_tokens={max_tokens} would be exceeded"
            queue.insert(0, packet)
            break

        kwargs = dict(transport_kwargs)
        if packet.get("max_output_tokens"):
            kwargs.setdefault("max_output_tokens", packet["max_output_tokens"])
        try:
            result, meta = cached_call(
                packet["body_text"],
                project=project, purpose=purpose, system=packet["static_prefix"],
                schema=packet.get("schema"), model=model,
                version=f"{packet['prompt_version']}:{packet['input_sha256'][:16]}",
                transport=transport, cache=cache,
                replicates=replicates, agree=agree, **kwargs,
            )
        except Exception as error:  # a failed call is still a paid call
            record = cost_log.log(
                project=project or "unknown", model=model, purpose=purpose,
                script="run_packets", cost_usd=0.0, outcome="error",
                packet_id=packet["packet_id"],
                metadata={"failed": True, "error": str(error)[:500],
                          "input_sha256": packet["input_sha256"]},
            )
            report["calls"] += replicates
            report["tokens"] += estimate * replicates
            report["failures"].append({
                "packet_id": packet["packet_id"], "error": str(error)[:500],
                "call_id": record.id})
            continue

        report["calls"] += replicates
        report["tokens"] += _usage_tokens(meta) or estimate * replicates
        report["cost_usd"] = round(report["cost_usd"] + (meta.cost_usd or 0.0), 6)
        cost_log.set_packet_id(meta.call_id, packet["packet_id"])

        if isinstance(result, Held):
            cost_log.record_outcome(meta.call_id, "held", result.reason)
            report["held"].append({"packet_id": packet["packet_id"],
                                   "reason": result.reason,
                                   "call_id": meta.call_id})
            continue

        if is_truncated(result, meta, packet):
            children = list(split(packet)) if (split and packet["packet_id"] not in split_done) else []
            cost_log.record_outcome(meta.call_id, "rejected", "truncated output")
            split_done.add(packet["packet_id"])
            report["split"].append({"packet_id": packet["packet_id"],
                                    "into": [c["packet_id"] for c in children]})
            if children:
                queue = list(children) + queue
            else:
                report["failures"].append({
                    "packet_id": packet["packet_id"],
                    "error": "output truncated and no split hook produced smaller packets",
                    "call_id": meta.call_id})
            continue

        if outcome_fn is not None:
            outcome = outcome_fn(result)
            if outcome:
                cost_log.record_outcome(meta.call_id, outcome)
        report["results"].append({
            "packet_id": packet["packet_id"], "call_id": meta.call_id,
            "cache_hit": meta.cache_hit, "cost_usd": meta.cost_usd,
            "result": result})
    else:
        report["stopped"] = "all packets sent"
    report["remaining"] = len(queue)
    return report


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class LowYield(RuntimeError):
    """Raised by `probe(min_yield=...)` when a batch is not worth building for."""


def probe(
    packets: Sequence[Packet | Mapping[str, Any]],
    n: int = 50,
    *,
    yield_fn: Callable[[Any], int],
    stratify_by: Callable[[Mapping[str, Any]], Any] | None = None,
    min_yield: float | None = None,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Run a stratified `n` packets and report what they are worth.

    **The first call of any campaign.** The audit's clearest inversion:
    12.3k lines of sealed machinery, 26 prompt versions and 13 per-packet test
    files were built around a model stream that produced **0 writes**, and in
    the campaign next to it 2,336 of 2,342 proposals came from a plain
    deterministic join — 384 model calls produced 6. Nobody had run a 50-item
    yield probe first.

    `yield_fn(result) -> int` counts *actionable* outputs from one result, not
    items returned. A low yield usually means a deterministic join does this
    job, not that the prompt needs another version.
    """
    sample = _stratified_sample(packets, n, stratify_by)
    run_kwargs.setdefault("max_calls", len(sample) * int(run_kwargs.get("replicates", 1) or 1))
    run_kwargs.setdefault("outcome_fn",
                          lambda result: "applied" if yield_fn(result) else "no_op")
    report = run_packets(sample, **run_kwargs)

    actionable = [yield_fn(r["result"]) for r in report["results"]]
    calls = report["calls"] or len(sample)
    produced = sum(actionable)
    with_output = sum(1 for a in actionable if a > 0)
    out = {
        "sampled": len(sample),
        "calls": calls,
        "actionable_outputs": produced,
        "calls_with_output": with_output,
        "yield_rate": round(with_output / calls, 4) if calls else 0.0,
        "outputs_per_call": round(produced / calls, 3) if calls else 0.0,
        "cost_usd": report["cost_usd"],
        "cost_per_actionable": (round(report["cost_usd"] / produced, 6) if produced else None),
        "held": len(report["held"]),
        "disagreement_rate": (round(len(report["held"]) / calls, 4) if calls else 0.0),
        "failures": len(report["failures"]),
        "executed": report["executed"],
        "warnings": report["warnings"],
        "projected_cost_usd": (
            round(report["cost_usd"] / calls * len(packets), 2) if calls and report["cost_usd"] else None),
    }
    if min_yield is not None and out["yield_rate"] < min_yield:
        raise LowYield(
            f"yield {out['yield_rate']:.2%} is below the {min_yield:.2%} floor "
            f"on {out['sampled']} packets ({produced} actionable outputs, "
            f"${report['cost_usd']:.4f}). Do not build campaign machinery on "
            "this stream — check whether a deterministic join does the job.")
    return out


def _stratified_sample(
    packets: Sequence[Mapping[str, Any]], n: int,
    stratify_by: Callable[[Mapping[str, Any]], Any] | None,
) -> list[Mapping[str, Any]]:
    """Deterministic, evenly spread sample — no RNG, so a probe is repeatable."""
    if n >= len(packets):
        return list(packets)
    if stratify_by is None:
        step = len(packets) / n
        return [packets[int(i * step)] for i in range(n)]
    strata: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for packet in packets:
        strata[stratify_by(packet)].append(packet)
    out: list[Mapping[str, Any]] = []
    index = 0
    while len(out) < n:
        added = False
        for key in sorted(strata, key=str):
            group = strata[key]
            if index < len(group) and len(out) < n:
                out.append(group[index])
                added = True
        if not added:
            break
        index += 1
    return out


# ---------------------------------------------------------------------------
# Validation and merging
# ---------------------------------------------------------------------------

def validate_quotes(
    items: Iterable[Mapping[str, Any]],
    pages: Mapping[str, str],
    *,
    quote_field: str = "quote",
    page_field: str = "page_ref",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only items whose quote is a literal substring of the cited page.

    Whitespace is collapsed on both sides and nothing else. No normalising of
    spelling, no expanding of abbreviations, no repairing of OCR: if the page
    says "av av", the quote must say "av av". This is what makes invented
    evidence unrepresentable rather than merely discouraged, and it is the
    check that a production repo already had and never wired into its runner.

    Returns `(valid, problems)`.
    """
    valid: list[dict[str, Any]] = []
    problems: list[str] = []
    collapsed = {ref: _WHITESPACE.sub(" ", text) for ref, text in pages.items()}
    for position, item in enumerate(items):
        ref = item.get(page_field)
        if ref not in collapsed:
            problems.append(f"item {position}: {page_field}={ref!r} is not a page of this packet")
            continue
        quote = item.get(quote_field) or ""
        if not quote:
            problems.append(f"item {position}: empty {quote_field}")
            continue
        if _WHITESPACE.sub(" ", quote).strip() not in collapsed[ref]:
            problems.append(
                f"item {position}: {quote_field} is not on page {ref}: {quote[:80]!r}")
            continue
        valid.append(dict(item))
    return valid, problems


def _normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def text_quote_anchor(
    page_text: str,
    exact: str,
    page_id: str,
    *,
    occurrence_index: int = 0,
    context_chars: int = 48,
) -> dict[str, Any]:
    """Anchor a quote to a page as a W3C-style TextQuoteSelector.

    Where `validate_quotes` answers "is it there", this returns *where*: the
    span as the page spells it, `prefix`/`suffix` context, which occurrence,
    and hashes a later build can compare. Whitespace is collapsed and trimmed
    on both sides and the match ignores case; nothing else is normalised.
    `start`/`end` index the whitespace-collapsed page and are descriptive
    only — they stay out of `selector_sha256`, so a correction elsewhere on
    the page does not revoke a review of an unchanged span, while
    `page_text_sha256` still records which extraction was read.

    Raises `ValueError` when the quote is empty or that occurrence is absent.
    """
    normalized_page = _normalize_text(page_text)
    normalized_exact = _normalize_text(exact)
    if not normalized_exact:
        raise ValueError(f"{page_id}: empty quote")
    matches = list(re.finditer(re.escape(normalized_exact), normalized_page, re.IGNORECASE))
    if occurrence_index < 0 or occurrence_index >= len(matches):
        raise ValueError(
            f"{page_id}: exact text occurrence {occurrence_index} not found: {normalized_exact!r}"
        )
    match = matches[occurrence_index]
    selector = {
        "type": "TextQuoteSelector",
        "page_id": page_id,
        "exact": match.group(0),
        "prefix": normalized_page[max(0, match.start() - context_chars):match.start()],
        "suffix": normalized_page[match.end():match.end() + context_chars],
        "occurrence_index": occurrence_index,
    }
    page_sha = _sha(normalized_page)
    return {
        **selector,
        "start": match.start(),
        "end": match.end(),
        "selector_sha256": _sha(_canonical(selector)),
        "span_sha256": _sha(_normalize_text(match.group(0))),
        "page_text_sha256": page_sha,
        "extraction_version_id": f"sha256:{page_sha}",
    }


def unresolved_text_quote_anchor(page_text: str, expected_exact: str, page_id: str) -> dict[str, Any]:
    """Describe a quote that no longer resolves, without pretending it matched."""
    normalized_page = _normalize_text(page_text)
    selector = {
        "type": "UnresolvedTextQuoteSelector",
        "page_id": page_id,
        "expected_exact": _normalize_text(expected_exact),
    }
    page_sha = _sha(normalized_page)
    return {
        **selector,
        "selector_sha256": _sha(_canonical(selector)),
        "page_text_sha256": page_sha,
        "extraction_version_id": f"sha256:{page_sha}",
    }


def reanchor_quote(
    pages: Mapping[str, str],
    exact: str,
    *,
    cited: str | None = None,
    context_chars: int = 48,
) -> tuple[str, dict[str, Any]] | None:
    """Find the one page, other than `cited`, that carries this quote.

    For a quote that failed to anchor where it was cited: on exactly one other
    page it is a slot slip and `(page_id, anchor)` comes back; on two it is a
    real ambiguity and on none it is unsupported, and both return None rather
    than pick. Pass only the pages an item may legitimately cite.
    """
    if not _normalize_text(exact or ""):
        return None
    found: list[tuple[str, dict[str, Any]]] = []
    for page_id, text in pages.items():
        if page_id == cited:
            continue
        try:
            found.append((page_id, text_quote_anchor(text, exact, page_id, context_chars=context_chars)))
        except ValueError:
            continue
    return found[0] if len(found) == 1 else None


def union_passes(
    results_by_pass: Mapping[str, Iterable[Mapping[str, Any]]],
    key: Callable[[Mapping[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Union several passes over the same input, deduplicated by `key`.

    Passes are merged, not chosen between. Measured: folding name-accounting
    into the coding call *lost 66 known entities* — the candidate pass knows
    the entities the KB already holds and the name pass knows the unknown ones,
    and neither is a superset of the other. Two narrow calls beat one wide one
    whenever the jobs are different jobs.

    Deduplication is across passes, not within one: a pass that legitimately
    returned two codings of the same span keeps both.
    """
    merged: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for pass_name in results_by_pass:
        added: set[Any] = set()
        for item in results_by_pass[pass_name]:
            item_key = key(item)
            if item_key in seen:
                continue
            added.add(item_key)
            merged.append({**item, "_pass": pass_name})
        seen |= added
    return merged


# ---------------------------------------------------------------------------
# Unmatched names — the deterministic recall cross-check
# ---------------------------------------------------------------------------

# A capitalised word, including the Nordic letters and accented forms.
_WORD = r"[A-ZÆØÅÄÖÉÜ][a-zæøåäöéèüïá'’-]{1,}"
# Particles that sit inside a name. "og"/"and" are excluded on purpose: they
# join two separate names rather than extending one.
_NAME_PARTICLE = r"(?:av|von|van|de|den|der|du|la|le|til|of|paa|på)"
# Names never span a line break: joining across one produced "Norsk\nFelles",
# which is two headings, not a person.
_NAME_RE = re.compile(rf"{_WORD}(?:[ \t]+(?:{_NAME_PARTICLE}[ \t]+)?{_WORD})*")
# A sentence can end with punctuation and then a dash, bullet or quote before
# the next word; ". — Dessuten" read as mid-sentence looks like a name.
_SENTENCE_START = re.compile(r"(?:^|[.!?:;»”\"][\s\-–—•*»“\"']*|\n[\s\-–—•*]*)$")
_MIN_NAME_CHARS = 4
# Above this share of mid-sentence capitalised words the source uses an
# orthography that capitalises every noun, where a single capital carries no
# signal at all.
_NOUN_CAPITALISATION_RATE = 0.15


def _simple_fold(text: str) -> str:
    out = unicodedata.normalize("NFKD", text.casefold())
    out = "".join(ch for ch in out if not unicodedata.combining(ch))
    out = out.replace("ø", "o").replace("æ", "a").replace("å", "a")
    return _WHITESPACE.sub(" ", re.sub(r"[^\w\s]", " ", out, flags=re.UNICODE)).strip()


def corpus_lowercase_words(texts: Iterable[str], *, minimum: int = 3) -> set[str]:
    """Words the corpus itself writes in lower case somewhere.

    A far better common-noun filter than any hand-written stopword list:
    "Dessuten", "Barna" and "Stoffet" all appear lower-cased elsewhere in the
    same documents, while "Hårfagre" and "Hernes" never do. Build it once per
    corpus and pass it to `unmatched_names`.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        for word in re.findall(r"(?<![\w])[a-zæøåäöéèü][a-zæøåäöéèü'’-]{2,}", text):
            folded = _simple_fold(word)
            if folded:
                counts[folded] += 1
    return {word for word, count in counts.items() if count >= minimum}


def _noun_capitalising(text: str) -> bool:
    words = re.findall(r"(?<=[a-zæøå,] )([A-Za-zÆØÅæøå][a-zæøåé]{2,})", text)
    if len(words) < 20:
        return False
    return sum(1 for w in words if w[0].isupper()) / len(words) > _NOUN_CAPITALISATION_RATE


def unmatched_names(
    text: str,
    known_labels: Iterable[str],
    corpus_lowercase_vocab: Iterable[str] = (),
    *,
    stopwords: Iterable[str] = (),
) -> list[str]:
    """Name-like strings in `text` that none of `known_labels` accounts for.

    A candidate lookup can only offer entities the KB already holds, so a
    person the KB has never heard of is invisible to it. Measured: a corpus run
    printed "Harald Hårfagre" and the coding returned neither an assertion nor
    a "new entity", because nothing put the name in front of the model. For any
    census whose question is "who is named", silently dropping unknown names is
    the worst failure there is.

    Deliberately generous: a false positive costs one line of disposition, a
    false negative costs a missing person. Hand the result to the model as a
    list it must account for one way or the other — that accounting is also
    your deterministic recall check on the run.
    """
    surfaces: set[str] = set()
    known_words: set[str] = set()
    for label in known_labels:
        folded = _simple_fold(str(label))
        if folded:
            surfaces.add(folded)
            known_words.update(folded.split())
    common = {_simple_fold(w) for w in corpus_lowercase_vocab}
    stop = {_simple_fold(w) for w in stopwords}
    singles_are_noise = _noun_capitalising(text)

    found: list[str] = []
    for match in _NAME_RE.finditer(text):
        raw = match.group(0).strip(" -–—'’")
        folded = _simple_fold(raw)
        if not folded:
            continue
        words = folded.split()
        # A short word inside a name is normal ("Et Dukkehjem"), so the test is
        # that at least one word is substantial, not that every one is.
        if max(len(w) for w in words) < _MIN_NAME_CHARS:
            continue
        if len(words) == 1:
            start = match.start()
            if singles_are_noise or _SENTENCE_START.search(text[max(0, start - 10):start]):
                continue
            if words[0] in common:
                continue
        if all(w in stop or w.rstrip("s") in stop for w in words):
            continue
        if folded in surfaces:
            continue
        # Drop a fragment of something already offered: "Ibsen" when "Henrik
        # Ibsen" is already a candidate.
        if all(w in known_words for w in words):
            continue
        if raw not in found:
            found.append(raw)
    return found


# ---------------------------------------------------------------------------
# Output refusals
#
# A packet that renders or rewrites text — translate this definition, say this
# in the other register — produces prose, and prose has no schema. Three
# distinct defects have shipped from such stages, and three different checks
# caught them. They stay three functions: a single `lint_output()` would force
# every caller to accept all three sets of assumptions to get any one of them.
#
# The rule the whole episode teaches: any instruction you write into the prompt
# telling the model not to do X — don't invent a year, don't describe your own
# citation, don't answer with the item number — is also a check for X you have
# not written yet.
# ---------------------------------------------------------------------------

# «i01», «r7», «q03» — a slot id from the packet the model was answering. One
# packet answered every item with its own slot id and ten records went into a
# published graph defined as "i01".
SLOT_ID = re.compile(r"[^\W\d_]{1,2}\d{1,3}[.,]?", re.UNICODE)
# Any word of two letters or more. Prose contains one that is lowercase; an
# identifier and a shouted heading do not. Case is checked in Python rather
# than in the class, so the pattern stays usable for any alphabet.
WORD = re.compile(r"(?<![^\s])[^\W\d_]{2,}")
MIN_RENDERING_CHARS = 12

# Phrases that describe the evidence the pipeline happened to read rather than
# the thing being described. Shipped as English only, and small: a phrase list
# is a corpus's vocabulary, and one lifted from another corpus refuses good
# text and misses the bad. Curate your own from what an audit actually finds
# and pass it as `phrases`.
DEFAULT_META_PHRASES = re.compile(
    r"as stated in the (label|prompt|text|passage)"
    r"|in the (label|cue|quote|given citation|supplied citation|supplied text)"
    r"|the (supplied|given) (citation|quote|source|passage|text)"
    r"|identified (only )?as the (source|author)"
    r"|identified (here|in the (passage|quote|text|label|citation|account))"
    r"|the cited (account|passage|source|text)"
    r"|(according to|based on) the (provided|supplied|given) ",
    re.I,
)
# «A person named Kristoffer Visted.» — a definition whose whole content is
# that the thing has the name it has. Anchored end to end on purpose: matching
# the opening formula alone refuses «A ballad titled Terje Vigen describes a
# sailor's ordeal in a storm», which is a real definition that happens to start
# the same way. Only a label may follow the formula, and then nothing else.
# A word character that is neither a digit, an underscore, nor a lowercase
# letter — i.e. a capital, in any alphabet, without enumerating one.
_UPPER = r"[^\W\d_a-zß-öø-ÿ]"
_LABEL_WORD = _UPPER + r"[\w’'-]*"
DEFAULT_VACUOUS = re.compile(
    r"^(?:[Aa]n?|[Tt]he)\s+(?:\w+\s+){1,3}(?:named|titled|called)\s+"
    rf"[«\"']?{_LABEL_WORD}"
    rf"(?:[\s,-]+(?:av|von|de|van|der|den|of|the|di|du|la|le)?\s*{_LABEL_WORD})*"
    r"[»\"']?[\s.!?]*$"
)

_YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")
# A word that is not sentence-initial; the capital is checked in Python so the
# pattern does not have to enumerate an alphabet's uppercase letters.
_CAPITALISED = re.compile(r"(?<![.!?]\s)(?<!^)\b([^\W\d_][\w’'-]{2,})")


def _fold_default(word: str) -> str:
    """Casefold and strip diacritics. Any language-specific rule is the caller's."""

    stripped = unicodedata.normalize("NFD", word.casefold())
    return "".join(c for c in stripped if not unicodedata.combining(c))


def slot_echo_refusal(
    text: str | None,
    slot_ids: Iterable[str] = (),
    *,
    source: str | None = None,
    min_chars: int = MIN_RENDERING_CHARS,
    min_ratio: float = 0.4,
    slot_pattern: "re.Pattern[str]" = SLOT_ID,
    word_pattern: "re.Pattern[str]" = WORD,
) -> str | None:
    """Why this output is not prose at all, or None.

    A cascade, first reason only: once the answer is the item number, nothing
    else about it is worth reporting. The checks are deliberately the cheapest
    ones that would have caught the real defect — a sentence has a space in
    it, has a lowercase word in it, is not an identifier, and is not a small
    fraction of what it was asked to render.

    `slot_ids` are this packet's own item ids, matched exactly; `slot_pattern`
    additionally catches the shape of an id from a packet you did not pass in.
    `source` enables the short-side length ratio and may be omitted for a
    generation task that has no source text.
    """

    stripped = (text or "").strip()
    if not stripped:
        return "empty output"
    if stripped in set(slot_ids) or slot_pattern.fullmatch(stripped):
        return f"output is a slot id, not a sentence: {stripped}"
    if " " not in stripped or len(stripped) < min_chars:
        return f"output is too short to be a sentence: {stripped[:40]}"
    if not any(m.group(0).islower() for m in word_pattern.finditer(stripped)):
        return f"output has no lowercase word in it: {stripped[:40]}"
    if source and len(stripped) < min_ratio * len(source):
        return f"output is under {min_ratio:g} of its source: {stripped[:40]}"
    return None


def meta_leak_refusals(
    text: str | None,
    *,
    phrases: "re.Pattern[str]" = DEFAULT_META_PHRASES,
    vacuous: "re.Pattern[str] | None" = DEFAULT_VACUOUS,
) -> list[str]:
    """Refuse text that describes the pipeline's own evidence instead of the subject.

    "…identified only as the source", "as stated in the label". Every phrase
    in the default list was found by a blind audit *inside an already applied
    definition*: the prompt told the model not to write them, the model wrote
    them anyway, and nothing between the model and the store disagreed.

    Both defaults are a starting point, not a shipped answer. `phrases` is per
    corpus and per language, and the right way to build one is to read what an
    audit found and add exactly that. `vacuous` is anchored end to end so only
    a definition that is *nothing but* the formula is refused — "A ballad
    titled Terje Vigen describes a sailor's ordeal" says something and stands.
    A corpus whose definitions run to several sentences will want its own.
    """

    stripped = (text or "").strip()
    out: list[str] = []
    if stripped and phrases.search(stripped):
        out.append("text describes the evidence rather than the subject")
    if stripped and vacuous is not None and vacuous.match(stripped):
        out.append("text says only that the subject has its name")
    return out


def rendering_fidelity_refusals(
    source: str,
    rendering: str | None,
    *,
    known_names: Iterable[str] = (),
    exonyms: Iterable[str] = (),
    max_ratio: float = 2.0,
    stem: int = 4,
    year_pattern: "re.Pattern[str]" = _YEAR,
    exempt_years: Callable[[str, str], Iterable[str]] | None = None,
    capitalised_pattern: "re.Pattern[str]" = _CAPITALISED,
    fold: Callable[[str], str] = _fold_default,
    parts: Callable[[str], Sequence[str]] = lambda word: (word,),
) -> list[str]:
    """Refuse a rendering that added a fact its source does not contain.

    Rendering into another language or register is the one task where the
    model has no licence to know anything: every year and every name in the
    output has to be in the input. A rendering stage invented plausible dates
    and plausible people, and both read as competent prose.

    Not a cascade — every refusal here is separately true, so all are
    returned. Three checks: the output is more than `max_ratio` times its
    source; it states a year the source does not; it names a capitalised thing
    that is in neither the source nor `known_names` nor `exonyms`.

    Everything language-shaped is injectable and nothing language-shaped ships:
    `exonyms` (the target language's own forms for names the source gives in
    its own — Danmark for Denmark), `fold` (the comparison key: a language
    that inflects names needs its own), `parts` (how a compound splits, for a
    language that compounds: Norwegian's "Perth-traktaten" is the treaty of
    Perth, not a new name), and `exempt_years` (a `(source, rendering) ->
    years` callable, for a language that spells a century as a four-digit
    number — bokmål's "1100-tallet" for "the twelfth century").

    A name counts as known when it shares a `stem`-character prefix with a
    known name, in both directions, after folding.
    """

    text = (rendering or "").strip()
    if not text:
        return ["empty rendering"]
    out: list[str] = []
    if len(text) > max_ratio * max(len(source), 1):
        out.append(f"rendering is more than {max_ratio:g} times the length of its source")

    exempt = set(exempt_years(source, text)) if exempt_years else set()
    invented_years = sorted(set(year_pattern.findall(text)) - set(year_pattern.findall(source)) - exempt)
    if invented_years:
        out.append(f"year(s) not in the source: {', '.join(invented_years)}")

    # Tokenised, not folded whole: a label is "Det gamle Hellas" and the name
    # the rendering may use is "Hellas". Folding the label as one string hides
    # every word in it.
    known = {
        fold(word)
        for text in (source, *known_names, *exonyms)
        for word in re.findall(r"[\w’']+", text)
    }
    known.discard("")
    invented_names = sorted(
        {word for word in capitalised_pattern.findall(text)
         if word[:1].isupper()
         and not _is_known_name(word, known, stem=stem, fold=fold, parts=parts)}
    )
    if invented_names:
        out.append(f"name(s) not in the source: {', '.join(invented_names)}")
    return out


def _is_known_name(
    word: str,
    known: set[str],
    *,
    stem: int,
    fold: Callable[[str], str],
    parts: Callable[[str], Sequence[str]],
) -> bool:
    """Whether every part of a capitalised word matches a known name by stem."""

    for part in parts(word):
        key = fold(part)
        if not key:
            continue
        if not any(
            key.startswith(other[: min(stem, len(other))])
            or other.startswith(key[: min(stem, len(key))])
            for other in known
            if len(other) >= 3
        ):
            return False
    return True
