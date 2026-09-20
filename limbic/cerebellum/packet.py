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
    "LowYield",
    "Packet",
    "corpus_lowercase_words",
    "estimate_tokens",
    "lint_packet",
    "make_packet",
    "probe",
    "run_packets",
    "union_passes",
    "unmatched_names",
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
