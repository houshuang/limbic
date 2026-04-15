"""Unit tests for the deterministic Wikidata resolver.

These mock the WikidataClient entirely — they don't hit the network.
Live tests against real Wikidata live in test_wikidata_resolve_live.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from limbic.amygdala.temporal import DateRange
from limbic.amygdala.wikidata import Candidate, Entity
from limbic.hippocampus import (
    Resolution,
    ScoredCandidate,
    WikidataResolver,
    validate_chosen_qid,
)
from limbic.hippocampus.wikidata_resolve import (
    _any_person_candidate,
    _coherence_score,
    _cosine_unit,
    _date_score,
    _extract_dates,
    _rank_score,
    _regnal_variants,
    _type_score,
)


# --- fakes ---


class FakeWikidataClient:
    """Minimal stand-in for WikidataClient. Scripted search + get_many."""

    def __init__(
        self,
        *,
        search_results: dict[str, list[Candidate]] | None = None,
        entities: dict[str, Entity] | None = None,
    ):
        self.search_results = search_results or {}
        self.entities = entities or {}
        self.search_calls: list[str] = []
        self.get_many_calls: list[list[str]] = []

    def search(self, name: str, *, limit: int = 10, language: str = "en") -> list[Candidate]:
        self.search_calls.append(name)
        return self.search_results.get(name, [])

    def get_many(self, qids: list[str], *, languages=None) -> dict[str, Entity]:
        self.get_many_calls.append(list(qids))
        return {q: self.entities[q] for q in qids if q in self.entities}


class FakeEmbedder:
    """Deterministic embedder: returns a vector determined by a text-to-vec map.

    Any text without an override returns a seeded random vector (stable).
    """

    def __init__(self, overrides: dict[str, list[float]] | None = None, dim: int = 8):
        self.overrides = overrides or {}
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        if text in self.overrides:
            v = np.asarray(self.overrides[text], dtype=np.float32)
            return v
        # Stable hash-based vector so identical texts produce identical vecs.
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        return rng.normal(0, 1, size=self.dim).astype(np.float32)


# --- helpers ---


def _person_entity(qid: str, *, born: int | None = None, died: int | None = None,
                   p31: list[str] | None = None,
                   p22: list[str] | None = None,
                   p40: list[str] | None = None) -> Entity:
    """Quickly build a person Entity with common Wikidata claim shapes."""
    claims: dict[str, list[dict]] = {}
    if p31 is not None:
        claims["P31"] = [{
            "mainsnak": {"datatype": "wikibase-item",
                          "datavalue": {"value": {"id": pid}}},
            "rank": "normal",
        } for pid in p31]
    if born is not None:
        claims["P569"] = [{
            "mainsnak": {"datatype": "time",
                          "datavalue": {"value": {"time": f"{'+' if born >= 0 else '-'}{abs(born):04d}-00-00T00:00:00Z"}}},
            "rank": "normal",
        }]
    if died is not None:
        claims["P570"] = [{
            "mainsnak": {"datatype": "time",
                          "datavalue": {"value": {"time": f"{'+' if died >= 0 else '-'}{abs(died):04d}-00-00T00:00:00Z"}}},
            "rank": "normal",
        }]
    if p22 is not None:
        claims["P22"] = [{
            "mainsnak": {"datatype": "wikibase-item",
                          "datavalue": {"value": {"id": pid}}},
            "rank": "normal",
        } for pid in p22]
    if p40 is not None:
        claims["P40"] = [{
            "mainsnak": {"datatype": "wikibase-item",
                          "datavalue": {"value": {"id": pid}}},
            "rank": "normal",
        } for pid in p40]
    return Entity(qid=qid, claims=claims)


# --- heuristic unit tests ---


def test_type_score_hits_allowlist():
    entity = _person_entity("Q1", p31=["Q5"])
    assert _type_score(entity, "person") == 1.0


def test_type_score_miss_soft_penalty():
    # Q5 is human; a person hint on a place entity should penalise but not zero out.
    entity = _person_entity("Q1", p31=["Q515"])  # city
    assert _type_score(entity, "person") == 0.3


def test_type_score_no_hint_is_neutral():
    entity = _person_entity("Q1", p31=["Q5"])
    assert _type_score(entity, None) == 0.5


def test_type_score_no_entity_is_neutral():
    assert _type_score(None, "person") == 0.5


def test_extract_dates_from_birth_death():
    entity = _person_entity("Q1", born=942, died=996)
    dr = _extract_dates(entity)
    assert dr == DateRange(start=942, end=996)


def test_extract_dates_negative_years():
    entity = _person_entity("Q1", born=-100, died=-44)
    dr = _extract_dates(entity)
    assert dr is not None
    assert dr.start == -100 and dr.end == -44


def test_extract_dates_none_when_absent():
    entity = _person_entity("Q1", p31=["Q5"])
    assert _extract_dates(entity) is None


def test_date_score_missing_is_neutral():
    assert _date_score(None, DateRange(start=900, end=1000)) == 1.0
    assert _date_score(DateRange(start=900, end=1000), None) == 1.0


def test_date_score_overlap_is_full():
    cand = DateRange(start=942, end=996)
    ctx = DateRange(start=900, end=1050)
    assert _date_score(cand, ctx) == 1.0


def test_date_score_decays_with_gap():
    strauss = DateRange(start=1864, end=1949)
    viking_age = DateRange(start=900, end=1050)
    score = _date_score(strauss, viking_age)
    assert 0 < score < 0.1  # ~800-year gap, 100-year scale → very small


def test_coherence_score_hits_anchor():
    # Candidate has P22 (father) = Q300, which is an already-resolved anchor
    entity = _person_entity("Q1", p22=["Q300"])
    assert _coherence_score(entity, {"Rollo": "Q300"}) == 1.0


def test_coherence_score_no_anchor_is_neutral():
    entity = _person_entity("Q1", p22=["Q999"])
    assert _coherence_score(entity, {"Rollo": "Q300"}) == 0.3


def test_coherence_score_empty_anchors_is_neutral():
    entity = _person_entity("Q1", p22=["Q300"])
    assert _coherence_score(entity, {}) == 0.3


def test_rank_score_decays():
    assert _rank_score(0) == 1.0
    assert _rank_score(1) < 1.0
    assert _rank_score(10) < _rank_score(5)


def test_cosine_unit_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine_unit(v, v) == pytest.approx(1.0)


def test_cosine_unit_orthogonal_is_half():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_unit(a, b) == pytest.approx(0.5)


def test_cosine_unit_opposite_is_zero():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_unit(a, b) == pytest.approx(0.0)


def test_cosine_unit_zero_norm_is_neutral():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_unit(a, b) == 0.5


# --- validator ---


def test_validate_chosen_qid_in_set():
    cands = [
        ScoredCandidate(qid="Q1", label="A", description="", rank=0),
        ScoredCandidate(qid="Q2", label="B", description="", rank=1),
    ]
    assert validate_chosen_qid(cands, "Q1")
    assert validate_chosen_qid(cands, "Q2")


def test_validate_chosen_qid_not_in_set():
    cands = [ScoredCandidate(qid="Q1", label="A", description="", rank=0)]
    assert not validate_chosen_qid(cands, "Q999")


def test_validate_chosen_qid_none_or_empty_passes():
    cands = [ScoredCandidate(qid="Q1", label="A", description="", rank=0)]
    assert validate_chosen_qid(cands, None)
    assert validate_chosen_qid(cands, "")


# --- resolver integration with fakes ---


def _rollo_style_fixture() -> FakeWikidataClient:
    """A small fixture modeled on the Rollo/Normans transcript.

    - Rollo search returns Q273773 (Viking Count of Rouen) + a distractor.
    - Richard search returns Q333359 (Richard I of Normandy, grandson of Rollo)
      and Q12345 (Richard Strauss, 19th-century composer) — the classic
      disambiguation test case.
    """
    return FakeWikidataClient(
        search_results={
            "Rollo": [
                Candidate(qid="Q273773", label="Rollo",
                          description="10th-century Viking and Count of Rouen", rank=0),
                Candidate(qid="Q7361286", label="Rollo Duke of Normandy",
                          description="play by Fletcher et al.", rank=1),
            ],
            "Richard I": [
                # Popularity-biased API puts Strauss first!
                Candidate(qid="Q12345", label="Richard Strauss",
                          description="German composer (1864-1949)", rank=0),
                Candidate(qid="Q333359", label="Richard I of Normandy",
                          description="10th-century duke of Normandy", rank=1),
            ],
        },
        entities={
            "Q273773": _person_entity("Q273773", born=846, died=930, p31=["Q5"], p40=["Q315838"]),
            "Q7361286": Entity(qid="Q7361286", claims={
                "P31": [{
                    "mainsnak": {"datatype": "wikibase-item",
                                  "datavalue": {"value": {"id": "Q7725634"}}},  # literary work
                    "rank": "normal",
                }],
            }),
            # Richard Strauss: modern composer, human
            "Q12345": _person_entity("Q12345", born=1864, died=1949, p31=["Q5"]),
            # Richard I of Normandy: human, grandson of Rollo (P22 → William Longsword → P22 → Rollo)
            # For PR 2 we only check direct P22/P25 etc. — so put Rollo as father directly (test shortcut)
            "Q333359": _person_entity("Q333359", born=942, died=996, p31=["Q5"], p22=["Q273773"]),
        },
    )


def test_resolver_no_match():
    client = FakeWikidataClient(search_results={"Nobody": []})
    resolver = WikidataResolver(client=client)
    res = resolver.resolve("Nobody")
    assert res.status == "no_match"
    assert res.chosen_qid is None
    assert res.candidates == []


def test_resolver_kb_hit_shortcircuits_api():
    def kb(mention, type_hint):
        if mention == "Paris":
            return "Q90"
        return None
    client = FakeWikidataClient()
    resolver = WikidataResolver(client=client, existing_kb_lookup=kb)
    res = resolver.resolve("Paris", type_hint="place")
    assert res.status == "kb_hit"
    assert res.chosen_qid == "Q90"
    assert res.confidence == 1.0
    assert client.search_calls == []  # no API call


def test_resolver_picks_correct_rollo():
    client = _rollo_style_fixture()
    resolver = WikidataResolver(client=client)
    res = resolver.resolve(
        "Rollo",
        context_text="Viking who arrived in France and negotiated for Normandy",
        type_hint="person",
        date_hint=DateRange(start=850, end=950),
    )
    assert res.status == "resolved"
    assert res.chosen_qid == "Q273773"


def test_resolver_picks_correct_richard_with_context():
    """The disambiguation test: Richard I (Normandy) should beat Richard Strauss
    when the date_hint is Viking-era.
    """
    client = _rollo_style_fixture()
    resolver = WikidataResolver(client=client)
    res = resolver.resolve(
        "Richard I",
        context_text="Viking-era Norman duke, son of William Longsword",
        type_hint="person",
        date_hint=DateRange(start=900, end=1000),
    )
    assert res.chosen_qid == "Q333359", (
        f"Expected Richard I of Normandy (Q333359), got {res.chosen_qid}. "
        f"Reasoning: {res.reasoning}"
    )


def test_resolver_coherence_disambiguates_richard():
    """Without date hint, coherence via Rollo-as-anchor should still pick Normandy."""
    client = _rollo_style_fixture()
    resolver = WikidataResolver(client=client)
    res = resolver.resolve(
        "Richard I",
        context_text="",
        type_hint="person",
        already_resolved={"Rollo": "Q273773"},  # Rollo pre-resolved
    )
    # Richard I of Normandy's P22 = Q273773 (in our fixture); Strauss has none.
    # That should provide enough coherence to pick the right one.
    assert res.chosen_qid == "Q333359"


def test_resolver_ambiguous_without_context():
    """Richard I with no hints: date-neutral, type-neutral, no anchors.
    Scores are close → ambiguous."""
    client = _rollo_style_fixture()
    resolver = WikidataResolver(client=client)
    res = resolver.resolve("Richard I")
    # With no signals, both persons score similarly; the rank prior lightly
    # favors whoever's first. The assertion here is that status is ambiguous.
    assert res.status == "ambiguous"
    qids = [c.qid for c in res.candidates]
    assert "Q333359" in qids and "Q12345" in qids


def test_resolver_reasoning_includes_signal_breakdown():
    client = _rollo_style_fixture()
    resolver = WikidataResolver(client=client)
    res = resolver.resolve(
        "Rollo",
        context_text="Viking raider",
        type_hint="person",
        date_hint=DateRange(start=850, end=950),
    )
    assert "type=" in res.reasoning
    assert "date=" in res.reasoning
    assert "coherence=" in res.reasoning


def test_resolve_all_dep_order_anchors_second_pass():
    """First pass resolves Rollo (no ambiguity). Second pass resolves Richard I
    using Rollo as an anchor → coherence kicks in, right Richard is picked."""
    client = _rollo_style_fixture()
    resolver = WikidataResolver(client=client)
    mentions = [
        {"mention": "Rollo", "type_hint": "person",
         "context_text": "Viking founder of Normandy",
         "date_hint": DateRange(start=850, end=950)},
        {"mention": "Richard I", "type_hint": "person",
         "context_text": ""},  # no date hint → Pass 1 ambiguous
    ]
    resolutions = resolver.resolve_all(mentions)
    assert len(resolutions) == 2
    assert resolutions[0].chosen_qid == "Q273773"
    # Pass 2: with Rollo as anchor, Richard I of Normandy's P22 match → resolved
    assert resolutions[1].chosen_qid == "Q333359", resolutions[1].reasoning


def test_resolve_all_preserves_input_order():
    client = _rollo_style_fixture()
    resolver = WikidataResolver(client=client)
    mentions = [
        {"mention": "Richard I", "type_hint": "person",
         "context_text": "", "date_hint": None},
        {"mention": "Rollo", "type_hint": "person",
         "context_text": "Viking founder", "date_hint": DateRange(start=850, end=950)},
    ]
    resolutions = resolver.resolve_all(mentions)
    assert resolutions[0].mention == "Richard I"
    assert resolutions[1].mention == "Rollo"


def test_embedder_used_when_provided():
    """Description-context embedding should bump the score for the candidate
    whose description is similar to the context."""
    client = _rollo_style_fixture()
    # Override embeddings so Rollo's description matches the context strongly,
    # and the distractor's description doesn't.
    embedder = FakeEmbedder(overrides={
        "Viking Count Rouen 10th century": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "10th-century Viking and Count of Rouen": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "play by Fletcher et al.": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    resolver = WikidataResolver(client=client, embedder=embedder)
    res = resolver.resolve(
        "Rollo",
        context_text="Viking Count Rouen 10th century",
        type_hint="person",
    )
    # Find the scores for each candidate
    rollo_score = next(c for c in res.candidates if c.qid == "Q273773")
    distractor_score = next(c for c in res.candidates if c.qid == "Q7361286")
    assert rollo_score.scores["description"] > distractor_score.scores["description"]


def test_resolver_survives_candidate_without_entity():
    """If get_many doesn't return all entities, missing ones get neutral scores,
    not exceptions."""
    client = FakeWikidataClient(
        search_results={"X": [Candidate(qid="Q1", label="X", description="", rank=0)]},
        entities={},  # Q1 intentionally absent
    )
    resolver = WikidataResolver(client=client)
    res = resolver.resolve("X")
    # Top candidate with only rank=1.0 signal scored against neutral defaults;
    # likely ambiguous but must not crash.
    assert res.status in ("resolved", "ambiguous")
    assert res.candidates and res.candidates[0].qid == "Q1"


def test_resolver_weights_override():
    client = _rollo_style_fixture()
    # Turn off everything except the rank prior → whoever's rank 0 wins.
    resolver = WikidataResolver(
        client=client,
        weights={"type": 0, "date": 0, "description": 0, "coherence": 0, "rank": 1.0},
        absolute_threshold=0.9,  # rank 0 = 1.0, rank 1 = 0.905 — barely passes
        margin_ratio=1.0,
    )
    res = resolver.resolve("Richard I")
    # With only the rank prior, Strauss (rank 0 in our fixture) wins.
    assert res.chosen_qid == "Q12345"


# --- regnal-name variant retry (Bug 3: Karl XII of Sweden) ---


def test_regnal_variants_charles_of_sweden():
    # Roman numeral + "of <place>" → regnal shape; Karl ↔ Carl ↔ Charles class
    variants = _regnal_variants("Karl XII of Sweden")
    assert "Carl XII of Sweden" in variants
    assert "Charles XII of Sweden" in variants
    assert "Karl XII of Sweden" not in variants  # original excluded


def test_regnal_variants_requires_regnal_shape():
    # "Karl Schmidt" is a modern name, not regnal → no variants
    assert _regnal_variants("Karl Schmidt") == []


def test_regnal_variants_of_place_without_numeral():
    # "Odo of France" has "of <place>" but no numeral — still regnal-looking
    # (though "Odo" isn't in our variants table, so no results)
    variants = _regnal_variants("Louis of France")
    assert "Ludwig of France" in variants
    assert "Luis of France" in variants


def test_regnal_variants_unknown_first_name():
    # Regnal shape but first name not in any variant class
    assert _regnal_variants("Olaf III of Norway") != []  # in our table
    assert _regnal_variants("Thorfinn III of Orkney") == []  # not in table


def test_regnal_variants_single_token_mention():
    # Need at least 2 tokens to be regnal
    assert _regnal_variants("Karl") == []


def test_any_person_candidate_true():
    entity_map = {
        "Q1": _person_entity("Q1", p31=["Q5"]),  # human
        "Q2": Entity(qid="Q2"),
    }
    cands = [Candidate(qid="Q1", label="", rank=0), Candidate(qid="Q2", label="", rank=1)]
    assert _any_person_candidate(cands, entity_map) is True


def test_any_person_candidate_false():
    entity_map = {
        "Q1": _person_entity("Q1", p31=["Q3305213"]),  # painting
        "Q2": _person_entity("Q2", p31=["Q3305213"]),  # painting
    }
    cands = [Candidate(qid="Q1", label="", rank=0), Candidate(qid="Q2", label="", rank=1)]
    assert _any_person_candidate(cands, entity_map) is False


def test_resolver_retries_when_no_candidates():
    # Zero candidates for "Karl XII of Sweden" — retry with "Carl" / "Charles" variants
    client = FakeWikidataClient(
        search_results={
            "Karl XII of Sweden": [],
            "Carl XII of Sweden": [
                Candidate(qid="Q52934", label="Carl XII of Sweden",
                          description="King of Sweden 1697-1718", rank=0),
            ],
        },
        entities={
            "Q52934": _person_entity("Q52934", born=1682, died=1718, p31=["Q5"]),
        },
    )
    resolver = WikidataResolver(client=client)
    res = resolver.resolve(
        "Karl XII of Sweden",
        context_text="Swedish king, Great Northern War",
        type_hint="person",
        date_hint=DateRange(start=1680, end=1720),
    )
    assert res.chosen_qid == "Q52934", res.reasoning
    # Verify we tried both the original and at least one variant
    assert "Karl XII of Sweden" in client.search_calls
    assert any(v in client.search_calls for v in ("Carl XII of Sweden", "Charles XII of Sweden"))


def test_resolver_retries_when_only_paintings_returned():
    # The actual Karl XII failure mode: search returns only paintings of him,
    # not the person. Retry should surface the real person via spelling variant.
    client = FakeWikidataClient(
        search_results={
            "Karl XII of Sweden": [
                Candidate(qid="Q119811370", label="Karl XII of Sweden",
                          description="painting by Schröder", rank=0),
                Candidate(qid="Q106357900", label="Karl XII of Sweden",
                          description="painting by Ankarcrona", rank=1),
            ],
            "Carl XII of Sweden": [
                Candidate(qid="Q52934", label="Carl XII of Sweden",
                          description="King of Sweden 1697-1718", rank=0),
            ],
        },
        entities={
            # Paintings: P31 = Q3305213 (painting), NOT Q5 (human)
            "Q119811370": Entity(qid="Q119811370", claims={
                "P31": [{"mainsnak": {"datatype": "wikibase-item",
                                       "datavalue": {"value": {"id": "Q3305213"}}},
                         "rank": "normal"}],
            }),
            "Q106357900": Entity(qid="Q106357900", claims={
                "P31": [{"mainsnak": {"datatype": "wikibase-item",
                                       "datavalue": {"value": {"id": "Q3305213"}}},
                         "rank": "normal"}],
            }),
            "Q52934": _person_entity("Q52934", born=1682, died=1718, p31=["Q5"]),
        },
    )
    resolver = WikidataResolver(client=client)
    res = resolver.resolve(
        "Karl XII of Sweden",
        context_text="King of Sweden, fought Peter the Great",
        type_hint="person",
        date_hint=DateRange(start=1680, end=1720),
    )
    assert res.chosen_qid == "Q52934", (
        f"Expected Carl XII person Q52934, got {res.chosen_qid}. Reasoning: {res.reasoning}"
    )


def test_resolver_does_not_retry_for_non_regnal_mention():
    # "Nobody" doesn't match any regnal shape → no retries, falls through to no_match.
    client = FakeWikidataClient(search_results={"Nobody": []})
    resolver = WikidataResolver(client=client)
    res = resolver.resolve("Nobody", type_hint="person")
    assert res.status == "no_match"
    # Only one search call (the original); no variants were tried
    assert client.search_calls == ["Nobody"]


# --- weak-structural-match downgrade (Bug 4: Count Odo → Fire Brigade Museum) ---


def test_weak_structural_single_candidate_downgraded():
    """Count Odo was resolved to a Fire Brigade Museum because it was the
    only candidate scoring past threshold — even though type=0.30 (wrong type)
    and date=0.00 (big gap). With the fix, status should be ambiguous.
    """
    # Single candidate: museum, P31=Q3305213 (painting) — NOT Q5 (human).
    # Inception 1985, so date plausibility vs 885-886 hint is ~0.
    museum_entity = Entity(qid="Q67389525", claims={
        "P31": [{"mainsnak": {"datatype": "wikibase-item",
                               "datavalue": {"value": {"id": "Q3305213"}}},
                 "rank": "normal"}],
        "P571": [{"mainsnak": {"datatype": "time",
                                "datavalue": {"value": {"time": "+1985-00-00T00:00:00Z"}}},
                  "rank": "normal"}],
    })
    client = FakeWikidataClient(
        search_results={
            "Count Odo": [
                Candidate(qid="Q67389525", label="Count Ödön Széchenyi Fire Brigade Museum",
                          description="museum in Istanbul", rank=0),
            ],
        },
        entities={"Q67389525": museum_entity},
    )
    # Use weights that would have pushed the museum past threshold (description
    # + coherence + rank = non-negligible). To make the test deterministic,
    # set weights so total crosses 0.55 despite type=0.30, date<0.5.
    resolver = WikidataResolver(
        client=client,
        weights={"type": 0.20, "date": 0.10, "description": 0.20,
                 "coherence": 0.20, "rank": 0.30},
        absolute_threshold=0.4,  # lowered so single-candidate passes on raw signals
        margin_ratio=1.0,
    )
    res = resolver.resolve(
        "Count Odo",
        context_text="Viking siege of Paris 885-886",
        type_hint="person",
        date_hint=DateRange(start=880, end=900),
    )
    # Should be ambiguous, not resolved — the structural mismatch guards us.
    assert res.status == "ambiguous", (
        f"Expected ambiguous due to weak structural signals (type=0.30, date≈0.0), "
        f"got {res.status}. Reasoning: {res.reasoning}"
    )
    assert res.chosen_qid is None
    assert "weak structural match" in res.reasoning


def test_weak_structural_does_not_fire_without_type_hint():
    """When no type_hint is given, type_score=0.5 (neutral) — downgrade should
    not fire even if date_score is low, because we have no structural
    evidence of mismatch."""
    entity = Entity(qid="Q1", claims={
        "P571": [{"mainsnak": {"datatype": "time",
                                "datavalue": {"value": {"time": "+2000-00-00T00:00:00Z"}}},
                  "rank": "normal"}],
    })
    client = FakeWikidataClient(
        search_results={"X": [Candidate(qid="Q1", label="X",
                                         description="thing", rank=0)]},
        entities={"Q1": entity},
    )
    resolver = WikidataResolver(client=client, absolute_threshold=0.4,
                                 margin_ratio=1.0)
    res = resolver.resolve("X", date_hint=DateRange(start=800, end=900))
    # Without type_hint, we don't downgrade — status depends on totals
    # Just verify the new rule didn't hijack the decision.
    assert res.status in ("resolved", "ambiguous")
    if res.status == "ambiguous":
        assert "weak structural match" not in res.reasoning


def test_weak_structural_does_not_fire_when_type_matches():
    """When type_score=1.0 (P31 hits), downgrade does not fire even if
    date_score is low."""
    entity = _person_entity("Q5000", born=1800, died=1870, p31=["Q5"])
    client = FakeWikidataClient(
        search_results={"Alice": [Candidate(qid="Q5000", label="Alice",
                                              description="modern person",
                                              rank=0)]},
        entities={"Q5000": entity},
    )
    resolver = WikidataResolver(client=client, absolute_threshold=0.5,
                                 margin_ratio=1.0)
    # Big date gap: candidate 1800-1870 vs hint 500-600 → date_score near 0
    res = resolver.resolve(
        "Alice",
        type_hint="person",
        date_hint=DateRange(start=500, end=600),
    )
    # type_score=1.0 (> 0.5) → weak_structural is False → still resolves if totals pass
    assert res.status in ("resolved", "ambiguous")
    if res.status == "ambiguous":
        assert "weak structural match" not in res.reasoning


def test_resolver_does_not_retry_when_person_already_found():
    # "Louis XIV" returns a valid human on first try → no variant retry needed.
    client = FakeWikidataClient(
        search_results={
            "Louis XIV": [
                Candidate(qid="Q7742", label="Louis XIV",
                          description="King of France 1643-1715", rank=0),
            ],
        },
        entities={
            "Q7742": _person_entity("Q7742", born=1638, died=1715, p31=["Q5"]),
        },
    )
    resolver = WikidataResolver(client=client)
    resolver.resolve("Louis XIV", type_hint="person",
                     date_hint=DateRange(start=1630, end=1720))
    # Only one search call — no variant retry because Q7742 is already a human
    assert client.search_calls == ["Louis XIV"]
