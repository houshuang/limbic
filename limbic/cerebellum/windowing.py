"""Windowed LLM extraction — split long text, extract per window, merge safely.

Asking a model to extract structured items from a whole chapter at once loses
most of them: attention spreads, the output budget binds, and the tail of the
document gets a sentence where the head got a paragraph. Splitting the text into
overlapping windows and extracting from each recovers them. On the Hirsch corpus
(10 books, 101 chapters, ~10k claims) 6,000-character windows with 1,000
characters of overlap produced **80–150 claims per chapter against 20–30** for
whole-chapter extraction, and that structural change beat every prompt variation
tried against it.

The cost is a merge problem, and it is the part that is easy to get wrong:

1. **Ids collide.** Every window numbers its own output from ``C1``, ``E1``, …
   so window 2's ``C1`` is a different item from window 1's — and window 2's
   ``supports_claim: "C1"`` means *its own*. Concatenating first and renumbering
   later silently rewires references between windows.
2. **The overlap duplicates items.** That is the point of the overlap, but the
   duplicates arrive worded slightly differently and often truncated at a window
   edge.
3. **Dropping a duplicate orphans references to it.** Anything that pointed at
   the dropped copy has to be repointed at the survivor, not left dangling.

:func:`merge_windows` does all three in order: namespace, dedup with an alias
map, then renumber every id and rewrite every reference through that map. You
declare the shape of your extraction once as a :class:`MergeSchema`; nothing here
knows about claims or evidence specifically.

Ported from the otak/hirsch-atlas extraction pipeline, where the same ~200 lines
had already been copied into a second repository and started to drift.

Usage::

    from limbic.cerebellum.windowing import (
        Collection, MergeSchema, Reference, merge_windows, split_into_windows,
    )

    SCHEMA = MergeSchema([
        Collection("claims", prefix="C", dedup_field="text"),
        Collection("evidence", prefix="E", dedup_field="text",
                   references=[Reference("supports_claim", target="claims")]),
        Collection("cases", prefix="CASE", dedup_field="name",
                   references=[Reference("claims_supported", target="claims", many=True)]),
    ])

    per_window = [extract(w.text) for w in split_into_windows(chapter_text)]
    merged, report = merge_windows(per_window, SCHEMA)
    print(report.duplicates_removed, report.dangling)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple

log = logging.getLogger(__name__)

__all__ = [
    "Window",
    "split_into_windows",
    "Reference",
    "Collection",
    "MergeSchema",
    "MergeReport",
    "namespace_ids",
    "dedup_by_field",
    "merge_windows",
    "check_references",
]

DEFAULT_WINDOW_SIZE = 6000
DEFAULT_OVERLAP = 1000
DEFAULT_DEDUP_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class Window(NamedTuple):
    """One slice of the source text and where it started, for provenance."""

    text: str
    start: int


def split_into_windows(
    text: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    *,
    boundary: str = "\n\n",
    snap: int = 500,
) -> list[Window]:
    """Split ``text`` into overlapping windows that end at paragraph boundaries.

    Cutting mid-paragraph costs items on both sides of the cut — the model sees
    half an argument twice and extracts neither well — so each window is nudged
    to the nearest ``boundary`` within ``snap`` characters of the target end.
    ``overlap`` then gives the next window enough lead-in to re-read whatever was
    straddling the seam; the duplicates that produces are what
    :func:`merge_windows` is for.

    Text shorter than ``window_size`` comes back as a single window.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 <= overlap < window_size:
        raise ValueError("overlap must be >= 0 and < window_size")
    if len(text) <= window_size:
        return [Window(text, 0)]

    windows: list[Window] = []
    start = 0
    while start < len(text):
        end = start + window_size
        if end >= len(text):
            windows.append(Window(text[start:], start))
            break
        search_start = max(end - snap, start)
        break_at = text.rfind(boundary, search_start, end + snap)
        if break_at > search_start:
            end = break_at + len(boundary)
        windows.append(Window(text[start:end], start))
        # The boundary snap can pull `end` back far enough that subtracting the
        # overlap would not advance; never go backwards.
        start = max(end - overlap, start + 1)
    return windows


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """A field on one collection's items pointing at another collection's ids."""

    field: str
    target: str
    many: bool = False


@dataclass(frozen=True)
class Collection:
    """One list in the extraction result.

    ``prefix`` is the final id prefix (``"C"`` → ``C1``, ``C2``, …).
    ``dedup_field`` names the text field compared when removing duplicates from
    the overlap; leave it ``None`` to keep every item.
    """

    name: str
    prefix: str
    dedup_field: str | None = None
    references: tuple[Reference, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "references", tuple(self.references))


@dataclass(frozen=True)
class MergeSchema:
    """The collections a windowed extraction produces, and how they cross-reference."""

    collections: tuple[Collection, ...]

    def __init__(self, collections: Iterable[Collection]):
        object.__setattr__(self, "collections", tuple(collections))
        names = [c.name for c in self.collections]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate collection names: {names}")
        known = set(names)
        for coll in self.collections:
            for ref in coll.references:
                if ref.target not in known:
                    raise ValueError(
                        f"{coll.name}.{ref.field} targets unknown collection {ref.target!r}")

    def __iter__(self):
        return iter(self.collections)


@dataclass
class MergeReport:
    """What the merge did, for logging and tests."""

    items_before: dict[str, int] = field(default_factory=dict)
    items_after: dict[str, int] = field(default_factory=dict)
    id_map: dict[str, str] = field(default_factory=dict)
    references_checked: int = 0
    # Refs that pointed at an id no window produced. _renumber clears these, so
    # they never reach `dangling` — without the count, losing a link and losing
    # nothing look identical in the report.
    references_cleared: int = 0
    dangling: list[str] = field(default_factory=list)

    @property
    def duplicates_removed(self) -> int:
        return sum(self.items_before.values()) - sum(self.items_after.values())


# ---------------------------------------------------------------------------
# Merge steps
# ---------------------------------------------------------------------------


def namespace_ids(result: dict, window_index: int, schema: MergeSchema) -> dict:
    """Prefix every id and reference in one window's result with ``w{i}:``.

    Window ids are local: window 2's ``C1`` is not window 1's ``C1``, and window
    2's ``supports_claim: "C1"`` means its own. Namespacing before concatenation
    is what keeps that true — without it the merge silently repoints references
    across windows. Also stamps ``source_window`` for later provenance.

    Mutates and returns ``result``.
    """
    def tag(ref):
        if isinstance(ref, str) and ref.strip():
            return f"w{window_index}:{ref.strip()}"
        return ref

    for coll in schema:
        for item in result.get(coll.name) or []:
            if not isinstance(item, dict):
                continue
            if item.get("id"):
                item["id"] = tag(item["id"])
            item["source_window"] = window_index

    for coll in schema:
        for item in result.get(coll.name) or []:
            if not isinstance(item, dict):
                continue
            for ref in coll.references:
                val = item.get(ref.field)
                if ref.many:
                    if isinstance(val, list):
                        item[ref.field] = [tag(v) for v in val]
                elif val:
                    item[ref.field] = tag(val)
    return result


def _word_set(text: str) -> set[str]:
    """Lowercased word set. Unicode-aware, so æøå and non-Latin scripts count."""
    return set(re.findall(r"\w+", (text or "").casefold(), flags=re.UNICODE))


def dedup_by_field(
    items: list[dict],
    dedup_field: str,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> tuple[list[dict], dict[str, str]]:
    """Drop items whose text overlaps an earlier one by more than ``threshold``.

    Overlap is ``|A ∩ B| / min(|A|, |B|)`` over word sets — asymmetric on
    purpose, so a truncated restatement from a window edge still matches the full
    one. The **longer** text wins, because the short copy is usually the one that
    got cut off.

    Returns ``(kept, alias)`` where ``alias`` maps *every* input id to the id
    that survived in its place — including ids that were kept and then displaced
    by a longer duplicate. :func:`merge_windows` needs that to repoint references
    at survivors instead of leaving them dangling.
    """
    if not items:
        return items, {}
    kept: list[dict] = []
    slot_of: dict[str, int] = {}
    for item in items:
        words = _word_set(item.get(dedup_field, ""))
        duplicate_of = None
        for i, existing in enumerate(kept):
            existing_words = _word_set(existing.get(dedup_field, ""))
            smaller = min(len(words), len(existing_words))
            if not smaller:
                continue
            if len(words & existing_words) / smaller > threshold:
                duplicate_of = i
                break
        if duplicate_of is None:
            kept.append(item)
            if item.get("id"):
                slot_of[item["id"]] = len(kept) - 1
            continue
        if len(item.get(dedup_field, "")) > len(kept[duplicate_of].get(dedup_field, "")):
            kept[duplicate_of] = item
        if item.get("id"):
            slot_of[item["id"]] = duplicate_of
    alias = {old: kept[i].get("id") for old, i in slot_of.items()}
    return kept, alias


def _renumber(result: dict, schema: MergeSchema, alias: dict) -> tuple[dict, int]:
    """Assign final sequential ids and rewrite references through ``alias``."""
    id_map: dict[str, str] = {}
    for coll in schema:
        for i, item in enumerate(result.get(coll.name) or [], 1):
            old, new = item.get("id"), f"{coll.prefix}{i}"
            if old:
                id_map[old] = new
            item["id"] = new

    def resolve(ref):
        if not isinstance(ref, str) or not ref.strip():
            return None
        target = ref.strip()
        for _ in range(4):  # alias chains are one deep; the bound guards cycles
            if target in id_map:
                return id_map[target]
            nxt = alias.get(target)
            if not nxt or nxt == target:
                break
            target = nxt
        return id_map.get(target)

    unresolved = 0
    for coll in schema:
        for item in result.get(coll.name) or []:
            for ref in coll.references:
                val = item.get(ref.field)
                if ref.many:
                    if not isinstance(val, list):
                        continue
                    new_vals = []
                    for v in val:
                        resolved = resolve(v)
                        if resolved is None:
                            unresolved += 1
                            log.warning("Dropping dangling %s.%s reference %r on %s",
                                        coll.name, ref.field, v, item.get("id"))
                        else:
                            new_vals.append(resolved)
                    item[ref.field] = new_vals
                elif val:
                    resolved = resolve(val)
                    if resolved is None:
                        unresolved += 1
                        log.warning("Clearing dangling %s.%s reference %r on %s",
                                    coll.name, ref.field, val, item.get("id"))
                        item[ref.field] = None
                    else:
                        item[ref.field] = resolved
    return id_map, unresolved


def check_references(result: dict, schema: MergeSchema, *, strict: bool = False) -> tuple[int, list[str]]:
    """Verify every reference resolves to an id present in the same result.

    Returns ``(references_checked, dangling)``. Raises ``ValueError`` when
    ``strict`` — otherwise logs, so a production run stays observable rather than
    dying on one bad reference.
    """
    ids_by_coll = {
        coll.name: {it["id"] for it in (result.get(coll.name) or []) if it.get("id")}
        for coll in schema
    }
    dangling: list[str] = []
    checked = 0
    for coll in schema:
        for item in result.get(coll.name) or []:
            for ref in coll.references:
                val = item.get(ref.field)
                refs = (val if isinstance(val, list) else []) if ref.many else ([val] if val else [])
                for r in refs:
                    checked += 1
                    if r not in ids_by_coll.get(ref.target, set()):
                        dangling.append(f"{coll.name}[{item.get('id')}].{ref.field} -> {r!r}")
    if dangling:
        msg = (f"Referential integrity: {len(dangling)} dangling reference(s) "
               f"of {checked} checked: {dangling[:10]}")
        if strict:
            raise ValueError(msg)
        log.error(msg)
    return checked, dangling


def merge_windows(
    results: Iterable[dict],
    schema: MergeSchema,
    *,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    strict: bool = False,
) -> tuple[dict, MergeReport]:
    """Namespace, concatenate, deduplicate and renumber per-window extractions.

    The order is not negotiable: ids must be namespaced *before* concatenation
    (or references get rewired between windows), dedup must happen *before*
    renumbering (or the alias map has nothing to point at), and references are
    rewritten as part of renumbering (or dropping a duplicate orphans them).

    Returns ``(merged, report)``. ``strict`` makes a surviving dangling reference
    raise instead of log — use it in tests.

    Note which counter to watch: ``report.references_cleared`` is the common
    failure (a window referenced an id that no window produced, so the link is
    gone), while ``report.dangling`` catches only what survives renumbering — a
    reference resolving into the wrong collection. An extraction dropping links
    shows up in the first, not the second.
    """
    merged: dict[str, list] = {coll.name: [] for coll in schema}
    for index, result in enumerate(results):
        namespace_ids(result, index, schema)
        for coll in schema:
            merged[coll.name].extend(result.get(coll.name) or [])

    report = MergeReport(items_before={k: len(v) for k, v in merged.items()})

    alias: dict[str, str] = {}
    for coll in schema:
        if coll.dedup_field is None:
            continue
        merged[coll.name], coll_alias = dedup_by_field(
            merged[coll.name], coll.dedup_field, dedup_threshold)
        alias.update(coll_alias)

    report.items_after = {k: len(v) for k, v in merged.items()}
    report.id_map, report.references_cleared = _renumber(merged, schema, alias)
    if report.references_cleared:
        log.warning("merge_windows: %d unresolvable cross-reference(s) cleared",
                    report.references_cleared)
    report.references_checked, report.dangling = check_references(
        merged, schema, strict=strict)
    return merged, report
