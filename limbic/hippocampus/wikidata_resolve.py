"""Deterministic Wikidata entity resolver.

Takes a mention plus context signals and scores Wikidata candidate QIDs
across five heuristics, emitting a `Resolution` audit record with either a
confident `chosen_qid` or `status="ambiguous"` for downstream LLM
disambiguation (planned for PR 4 of the entity-resolution rollout).

Heuristics (each returns a score in [0, 1]):

1. **type_score** — does the candidate's `P31 instance-of` match the
   caller's `type_hint` ("person" / "place" / "event" / "work")? Uses a
   hardcoded P31 allowlist per hint. Matches → 1.0. Non-match → small
   soft penalty (~0.3), not a hard filter — Wikidata's class hierarchy is
   deep and our allowlist is shallow.

2. **date_score** — if the caller supplies a `date_hint DateRange` and the
   candidate has P569/P570 or P571/P576 date claims, use
   `amygdala.temporal.plausibility_score` (DELICATE-style exponential decay,
   never hard-filters). Missing dates on either side → neutral 1.0.

3. **description_score** — cosine similarity between the candidate's
   English description and the caller's `context_text`, via whatever
   `EmbeddingModel` the resolver was constructed with. Missing either side →
   neutral 0.5.

4. **coherence_score** — bonus when the candidate's family/relationship
   claims reference QIDs already resolved in this batch. Walks P22 (father),
   P25 (mother), P26 (spouse), P40 (child), P39 (position held), P108
   (employer), P131 (located in), P17 (country). Hit on any edge → 1.0.
   No hits → neutral 0.3 (no signal, not a penalty).

5. **rank_score** — decays with the API's search-result position
   (popularity-biased). Weak prior (weight ≪ others) — the plan explicitly
   warns against trusting Wikidata's rank.

The weighted sum produces `total`. Decision rules:
- If `existing_kb_lookup(mention, type_hint)` returns a QID → short-circuit
  with `status="kb_hit"` (no API calls).
- If candidates are empty → `status="no_match"`.
- If top.total >= ABSOLUTE_THRESHOLD and top.total >= MARGIN_RATIO * second.total
  → `status="resolved"`, `chosen_qid=top.qid`.
- Else → `status="ambiguous"`, `chosen_qid=None`, top-K candidates preserved
  for LLM disambiguation downstream.

QID-in-candidate-set validator (`validate_chosen_qid`) is provided for
callers who receive LLM-emitted QIDs — LLMs confidently generate plausible
Q-numbers that don't exist or don't match the candidate set. Any chosen QID
that isn't in the resolved candidate set must be rejected. See
`memory/project_hallucinated_qids_incident.md` in petrarca for the
documented incident that motivated this rule.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from ..amygdala.temporal import DateRange, plausibility_score
from ..amygdala.wikidata import Candidate, Entity, WikidataClient


# --- type-hint → P31 allowlist ---
#
# Shallow hardcoded sets covering the most common Wikidata instance-of values
# per type. Deeper subClassOf chains (P279) are intentionally not walked here
# — that adds SPARQL overhead and diminishing returns for most mentions the
# resolver sees. PR 3/4 can extend if coverage gaps appear.

TYPE_HINT_P31: dict[str, set[str]] = {
    "person": {
        "Q5",            # human
    },
    "place": {
        "Q515",          # city
        "Q3957",         # town
        "Q6256",         # country
        "Q35657",        # US state
        "Q486972",       # human settlement
        "Q1549591",      # big city
        "Q10864048",     # first-level administrative country subdivision
        "Q188509",       # suburb
        "Q8502",         # mountain
        "Q23397",        # lake
        "Q4022",         # river
        "Q3957",         # town
        "Q5107",         # continent
    },
    "event": {
        "Q1190554",      # occurrence
        "Q178561",       # battle
        "Q198",          # war
        "Q179057",       # explosion
        "Q40231",        # election
        "Q1128324",      # siege
        "Q2334719",      # legal case
        "Q13418847",     # historical event
    },
    "work": {
        "Q571",          # book
        "Q7725634",      # literary work
        "Q47461344",     # written work
        "Q17537576",     # creative work
        "Q386724",       # work
        "Q11424",        # film
        "Q482994",       # album
        "Q7366",         # song
        "Q3305213",      # painting
        "Q1004",         # comics
    },
}


# --- Wikidata claim properties used for coherence walks ---
COHERENCE_PROPERTIES = (
    "P22",   # father
    "P25",   # mother
    "P26",   # spouse
    "P40",   # child
    "P39",   # position held
    "P108",  # employer
    "P131",  # located in
    "P17",   # country
    "P463",  # member of
)

# Wikidata claim properties used to extract a candidate's date range.
DATE_PROPERTIES_PERSON = ("P569", "P570")    # born, died
DATE_PROPERTIES_ENTITY = ("P571", "P576")    # inception, dissolved

# Scoring configuration. Weights sum to 1.0.
#
# Coherence (0.30) weighted higher than description (0.20) because a direct
# P22/P25/etc. edge to an already-resolved anchor is harder evidence than
# semantic similarity between two short strings. Description embedding is
# still substantial but noisier: Wikidata's English descriptions are often
# very short ("American politician", "Italian footballer") so embedding
# cosine there is a lower-information signal than a graph edge.
#
# Type (0.25) is moderate because our P31 allowlist is shallow — matches are
# strong evidence, non-matches are softly penalised (not excluded).
#
# Date (0.15) is low because the plausibility_score is soft by design (per
# DELICATE) and many mentions lack a date hint entirely.
#
# Rank (0.10) is a weak prior — the plan explicitly warns that Wikidata's
# popularity-biased ranking shouldn't drive selection.
DEFAULT_WEIGHTS: dict[str, float] = {
    "type": 0.25,
    "date": 0.15,
    "description": 0.20,
    "coherence": 0.30,
    "rank": 0.10,
}

ABSOLUTE_THRESHOLD = 0.55   # top candidate must exceed this to resolve
MARGIN_RATIO = 1.25         # top must beat second-best by this ratio
TOP_K_EMBED = 5             # embed descriptions for top-K candidates only
DATE_DECAY_YEARS = 100.0    # plausibility_score scale


# --- dataclasses ---


@dataclass
class ScoredCandidate:
    """A candidate with per-heuristic scores and total."""
    qid: str
    label: str
    description: str
    rank: int
    scores: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    dates: DateRange | None = None
    aliases: list[str] = field(default_factory=list)
    external_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class Resolution:
    """Audit record for a single resolver call.

    The resolver's output contract. `status` classifies the outcome; `chosen_qid`
    is populated only when `status="resolved"` or `status="kb_hit"`. Candidates
    are sorted descending by `total`.
    """
    mention: str
    context_text: str
    type_hint: str | None
    date_hint: DateRange | None
    status: str                # "resolved" | "ambiguous" | "no_match" | "kb_hit"
    chosen_qid: str | None
    confidence: float          # top candidate's total, or 1.0 for kb_hit
    candidates: list[ScoredCandidate]
    reasoning: str
    resolver_version: str = "0.1"
    created_at: float = field(default_factory=time.time)


# --- embedder protocol ---


class Embedder(Protocol):
    def embed(self, text: str): ...  # returns np.ndarray


# --- resolver ---


class WikidataResolver:
    """Deterministic Wikidata entity resolver.

    Parameters
    ----------
    client:
        WikidataClient for search + candidate lookup.
    embedder:
        Embedding model exposing `embed(text) -> np.ndarray`. Optional — if
        None, `description_score` is skipped (neutral). Inject a fake in
        tests.
    existing_kb_lookup:
        Callable `(mention, type_hint) -> QID or None`. If provided and it
        returns a QID, the resolver short-circuits with `status="kb_hit"`
        (no API calls). Use this to plug in Petrarca's `shared_entities`.
    weights:
        Per-heuristic weight override. Must sum to ~1.0. Default is tuned
        for historical/literary mentions; tune per your corpus.
    search_language:
        Language for the initial wbsearchentities call. Default "en".
    search_limit:
        How many candidates to retrieve per search. Default 10.
    """

    def __init__(
        self,
        client: WikidataClient,
        *,
        embedder: Embedder | None = None,
        existing_kb_lookup: Callable[[str, str | None], str | None] | None = None,
        weights: dict[str, float] | None = None,
        search_language: str = "en",
        search_limit: int = 10,
        absolute_threshold: float = ABSOLUTE_THRESHOLD,
        margin_ratio: float = MARGIN_RATIO,
    ):
        self.client = client
        self.embedder = embedder
        self.existing_kb_lookup = existing_kb_lookup
        self.weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        self.search_language = search_language
        self.search_limit = search_limit
        self.absolute_threshold = absolute_threshold
        self.margin_ratio = margin_ratio

    # ---------- public API ----------

    def resolve(
        self,
        mention: str,
        *,
        context_text: str = "",
        type_hint: str | None = None,
        date_hint: DateRange | None = None,
        already_resolved: dict[str, str] | None = None,
    ) -> Resolution:
        """Resolve a single mention to a Wikidata QID, or flag as ambiguous."""
        already_resolved = already_resolved or {}

        # Step 1: existing-KB short-circuit.
        if self.existing_kb_lookup is not None:
            hit = self.existing_kb_lookup(mention, type_hint)
            if hit:
                return Resolution(
                    mention=mention,
                    context_text=context_text,
                    type_hint=type_hint,
                    date_hint=date_hint,
                    status="kb_hit",
                    chosen_qid=hit,
                    confidence=1.0,
                    candidates=[],
                    reasoning=f"Existing KB hit for '{mention}' → {hit}.",
                )

        # Step 2: candidate retrieval.
        raw_candidates = self.client.search(
            mention, limit=self.search_limit, language=self.search_language
        )
        if not raw_candidates:
            return Resolution(
                mention=mention,
                context_text=context_text,
                type_hint=type_hint,
                date_hint=date_hint,
                status="no_match",
                chosen_qid=None,
                confidence=0.0,
                candidates=[],
                reasoning=f"No Wikidata candidates for '{mention}'.",
            )

        # Step 3: enrich via wbgetentities (single batch call).
        qids = [c.qid for c in raw_candidates if c.qid]
        entity_map = self.client.get_many(qids) if qids else {}

        # Step 4: score each candidate.
        scored: list[ScoredCandidate] = []
        for cand in raw_candidates:
            entity = entity_map.get(cand.qid)
            sc = self._score_candidate(
                cand, entity, context_text, type_hint, date_hint, already_resolved
            )
            scored.append(sc)

        # Top-K embedding pass (runs only if embedder available).
        # Embedding is the most expensive heuristic; we apply it to the top
        # candidates by the cheap signals first, then re-sort.
        if self.embedder is not None and context_text:
            top_by_cheap = sorted(scored, key=lambda s: s.total, reverse=True)[:TOP_K_EMBED]
            self._apply_description_scores(top_by_cheap, context_text)
            # Recompute totals for the rescored candidates
            for sc in top_by_cheap:
                sc.total = self._combine_scores(sc.scores)

        scored.sort(key=lambda s: s.total, reverse=True)

        # Step 5: decision.
        return self._decide(mention, context_text, type_hint, date_hint, scored)

    def resolve_all(
        self,
        mentions: list[dict[str, Any]],
    ) -> list[Resolution]:
        """Dependency-order batch resolution.

        `mentions` is a list of dicts with keys `mention, context_text,
        type_hint, date_hint`. Resolves in two passes:

        1. First pass with empty `already_resolved` — surfaces singletons
           (candidates unambiguous enough to resolve without anchors).
        2. Second pass for any ambiguous mentions, now seeded with the first
           pass's resolved QIDs as `already_resolved` anchors. Coherence
           signal can now disambiguate.

        Order within each pass is input order.
        """
        # Pass 1: no anchors
        resolutions_p1: list[Resolution] = []
        anchors: dict[str, str] = {}
        for m in mentions:
            res = self.resolve(
                m["mention"],
                context_text=m.get("context_text", ""),
                type_hint=m.get("type_hint"),
                date_hint=m.get("date_hint"),
                already_resolved={},
            )
            resolutions_p1.append(res)
            if res.status in ("resolved", "kb_hit") and res.chosen_qid:
                anchors[m["mention"]] = res.chosen_qid

        # If no mentions were ambiguous, we're done.
        ambiguous_idx = [i for i, r in enumerate(resolutions_p1) if r.status == "ambiguous"]
        if not ambiguous_idx or not anchors:
            return resolutions_p1

        # Pass 2: retry ambiguous mentions with anchors
        final = list(resolutions_p1)
        for i in ambiguous_idx:
            m = mentions[i]
            res = self.resolve(
                m["mention"],
                context_text=m.get("context_text", ""),
                type_hint=m.get("type_hint"),
                date_hint=m.get("date_hint"),
                already_resolved=anchors,
            )
            final[i] = res
        return final

    # ---------- scoring ----------

    def _score_candidate(
        self,
        cand: Candidate,
        entity: Entity | None,
        context_text: str,
        type_hint: str | None,
        date_hint: DateRange | None,
        already_resolved: dict[str, str],
    ) -> ScoredCandidate:
        dates = _extract_dates(entity) if entity else None
        external_ids = _extract_external_ids(entity) if entity else {}

        scores: dict[str, float] = {
            "type": _type_score(entity, type_hint),
            "date": _date_score(dates, date_hint),
            "coherence": _coherence_score(entity, already_resolved),
            "rank": _rank_score(cand.rank),
            # description is filled in by _apply_description_scores when an
            # embedder is available. Default to neutral.
            "description": 0.5,
        }
        sc = ScoredCandidate(
            qid=cand.qid,
            label=cand.label,
            description=cand.description,
            rank=cand.rank,
            scores=scores,
            dates=dates,
            aliases=cand.aliases,
            external_ids=external_ids,
        )
        sc.total = self._combine_scores(scores)
        return sc

    def _apply_description_scores(
        self, candidates: list[ScoredCandidate], context_text: str
    ) -> None:
        """Fill in the `description` score for candidates via embedding cosine.

        Runs the embedder once per unique text. Mutates each candidate's
        `scores["description"]`. Caller is responsible for recomputing `total`.
        """
        if self.embedder is None:
            return
        context_vec = self.embedder.embed(context_text)
        for sc in candidates:
            if not sc.description:
                continue
            try:
                desc_vec = self.embedder.embed(sc.description)
            except Exception:
                # Non-fatal: leave the neutral 0.5 default.
                continue
            sc.scores["description"] = _cosine_unit(context_vec, desc_vec)

    def _combine_scores(self, scores: dict[str, float]) -> float:
        total = 0.0
        for key, weight in self.weights.items():
            total += weight * scores.get(key, 0.0)
        return total

    def _decide(
        self,
        mention: str,
        context_text: str,
        type_hint: str | None,
        date_hint: DateRange | None,
        scored: list[ScoredCandidate],
    ) -> Resolution:
        if not scored:
            return Resolution(
                mention=mention, context_text=context_text, type_hint=type_hint,
                date_hint=date_hint, status="no_match",
                chosen_qid=None, confidence=0.0, candidates=[],
                reasoning=f"No candidates survived scoring for '{mention}'.",
            )

        top = scored[0]
        second_total = scored[1].total if len(scored) > 1 else 0.0

        passes_abs = top.total >= self.absolute_threshold
        passes_margin = (
            second_total == 0.0 or top.total >= self.margin_ratio * second_total
        )

        if passes_abs and passes_margin:
            reasoning = (
                f"Resolved '{mention}' → {top.qid} ({top.label}). "
                f"top total={top.total:.3f} "
                f"(type={top.scores.get('type', 0):.2f}, "
                f"date={top.scores.get('date', 0):.2f}, "
                f"desc={top.scores.get('description', 0):.2f}, "
                f"coherence={top.scores.get('coherence', 0):.2f}, "
                f"rank={top.scores.get('rank', 0):.2f}); "
                f"second={second_total:.3f}."
            )
            return Resolution(
                mention=mention, context_text=context_text, type_hint=type_hint,
                date_hint=date_hint, status="resolved",
                chosen_qid=top.qid, confidence=top.total, candidates=scored,
                reasoning=reasoning,
            )

        reason_parts = []
        if not passes_abs:
            reason_parts.append(
                f"top total {top.total:.3f} below threshold {self.absolute_threshold:.2f}"
            )
        if not passes_margin:
            reason_parts.append(
                f"margin {top.total:.3f}/{second_total:.3f} "
                f"({top.total / max(second_total, 1e-9):.2f}x) "
                f"below required {self.margin_ratio}x"
            )
        reasoning = (
            f"Ambiguous for '{mention}': "
            + "; ".join(reason_parts)
            + f". Top 3: "
            + ", ".join(f"{c.qid}({c.total:.2f})" for c in scored[:3])
            + "."
        )
        return Resolution(
            mention=mention, context_text=context_text, type_hint=type_hint,
            date_hint=date_hint, status="ambiguous",
            chosen_qid=None, confidence=top.total, candidates=scored,
            reasoning=reasoning,
        )


# --- candidate-membership validator ---


def validate_chosen_qid(
    candidates: Iterable[ScoredCandidate], chosen_qid: str | None
) -> bool:
    """True iff `chosen_qid` is in the candidate set (or is None/empty).

    Use this to guard against LLM QID hallucination — if a downstream
    disambiguation step returns a chosen QID, it must be a member of the
    candidate set the resolver produced. A QID that "sounds right" but
    isn't in the set is always invalid.

    Documented incident: see memory/project_hallucinated_qids_incident.md
    in petrarca (2026-04-13) — 5 of 7 QIDs typed from memory for a test
    fixture turned out to reference completely unrelated entities.
    """
    if not chosen_qid:
        return True
    qids = {c.qid for c in candidates}
    return chosen_qid in qids


# --- scoring helpers (pure functions, independently testable) ---


def _type_score(entity: Entity | None, type_hint: str | None) -> float:
    """1.0 if candidate's P31 is in the allowlist for type_hint; 0.3 otherwise.

    A hint of None → neutral 0.5 (no signal). No entity → neutral 0.5.
    """
    if type_hint is None or entity is None:
        return 0.5
    allowlist = TYPE_HINT_P31.get(type_hint)
    if not allowlist:
        return 0.5
    instance_of = set(entity.claim_qids("P31"))
    if instance_of & allowlist:
        return 1.0
    # Soft penalty: we don't walk the full subClassOf chain, so we may be
    # wrong. Keep a floor so this signal doesn't erase good candidates.
    return 0.3


def _extract_dates(entity: Entity | None) -> DateRange | None:
    """Build a DateRange from the entity's time claims, if any.

    Prefers P569/P570 (birth/death) for persons; falls back to P571/P576
    (inception/dissolved) for other entities. Returns None when no usable
    dates are found.
    """
    if entity is None:
        return None

    start = _extract_claim_year(entity, "P569") or _extract_claim_year(entity, "P571")
    end = _extract_claim_year(entity, "P570") or _extract_claim_year(entity, "P576")
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    assert start is not None and end is not None
    return DateRange(start=min(start, end), end=max(start, end))


def _extract_claim_year(entity: Entity, property_id: str) -> int | None:
    """Parse the first year from a time-typed claim."""
    for stmt in entity.claims.get(property_id, []):
        if stmt.get("rank") == "deprecated":
            continue
        mainsnak = stmt.get("mainsnak") or {}
        if mainsnak.get("datatype") != "time":
            continue
        value = (mainsnak.get("datavalue") or {}).get("value") or {}
        time_str = value.get("time") or ""
        # Wikidata format: "+0942-00-00T00:00:00Z" or "-0044-00-00T00:00:00Z"
        if not time_str:
            continue
        try:
            sign = -1 if time_str.startswith("-") else 1
            year_part = time_str.lstrip("+-").split("-")[0]
            return sign * int(year_part)
        except (ValueError, IndexError):
            continue
    return None


def _extract_external_ids(entity: Entity) -> dict[str, str]:
    """Pull the common external-ID P-properties in one go.

    Matches the "cheap path" from the entity-resolution plan: fetch at
    resolution time so downstream enrichment becomes cache lookup rather
    than re-fetching.
    """
    return entity.external_ids(
        ("P214", "P227", "P244", "P1566", "P1584", "P1667", "P434", "P245", "P268")
    )


def _date_score(candidate_dates: DateRange | None, date_hint: DateRange | None) -> float:
    """Soft temporal plausibility. Neutral 1.0 if either side is missing.

    Neutral 1.0 (not 0.5) on missing data because absence of date info
    shouldn't down-rank a candidate — only a *conflict* should.
    """
    if candidate_dates is None or date_hint is None:
        return 1.0
    return plausibility_score(candidate_dates, date_hint, penalty_scale_years=DATE_DECAY_YEARS)


def _coherence_score(
    entity: Entity | None, already_resolved: dict[str, str]
) -> float:
    """Bonus if candidate's relationship claims reference an already-resolved QID."""
    if entity is None or not already_resolved:
        return 0.3
    anchor_qids = set(already_resolved.values())
    for prop in COHERENCE_PROPERTIES:
        for qid in entity.claim_qids(prop):
            if qid in anchor_qids:
                return 1.0
    return 0.3


def _rank_score(rank: int) -> float:
    """Decays with API rank. Rank 0 → 1.0; rank 5 → ~0.6; rank 10 → ~0.36.

    Very weak prior by design (weight is small in `DEFAULT_WEIGHTS`) — the
    plan warns that Wikidata ranks are popularity-biased and shouldn't drive
    selection.
    """
    return math.exp(-rank / 10.0)


def _cosine_unit(a, b) -> float:
    """Cosine similarity mapped to [0, 1] (0 → 0.5, 1 → 1.0, -1 → 0.0).

    Accepts numpy arrays; imports numpy lazily to keep this module useful
    even when the embedder isn't wired in.
    """
    import numpy as np

    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a_arr))
    nb = float(np.linalg.norm(b_arr))
    if na == 0.0 or nb == 0.0:
        return 0.5
    cos = float(np.dot(a_arr, b_arr) / (na * nb))
    # Map [-1, 1] → [0, 1] so it composes cleanly with other [0, 1] scores.
    return (cos + 1.0) / 2.0
