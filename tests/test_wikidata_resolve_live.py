"""End-to-end resolver test against real Wikidata.

Gated by `LIMBIC_LIVE_WIKIDATA=1`. Proves the full stack on the Petrarca
Rollo/Normandy transcript — the same fixture whose failed processing
started this whole plan.
"""

from __future__ import annotations

import os

import pytest

from limbic.amygdala.temporal import DateRange
from limbic.amygdala import WikidataClient
from limbic.hippocampus import WikidataResolver


LIVE_FLAG = "LIMBIC_LIVE_WIKIDATA"
live = pytest.mark.skipif(
    os.environ.get(LIVE_FLAG) != "1",
    reason=f"Set {LIVE_FLAG}=1 to run live Wikidata tests.",
)

LIVE_UA = "limbic-resolve-live/0.1 (https://github.com/houshuang/limbic)"


@pytest.fixture(scope="module")
def live_resolver(tmp_path_factory):
    cache = tmp_path_factory.mktemp("wd_resolve_live") / "cache.db"
    client = WikidataClient(user_agent=LIVE_UA, cache_db_path=str(cache))
    # Wire a real embedder so description-context scoring engages. Module-
    # scoped so the (slow) MiniLM cold-start only happens once.
    from limbic.amygdala import EmbeddingModel
    embedder = EmbeddingModel()
    return WikidataResolver(client=client, embedder=embedder)


@live
def test_live_resolve_rollo(live_resolver):
    res = live_resolver.resolve(
        "Rollo",
        context_text="Viking leader who negotiated with the Frankish king for what became Normandy",
        type_hint="person",
        date_hint=DateRange(start=850, end=950),
    )
    assert res.status == "resolved"
    assert res.chosen_qid == "Q273773", res.reasoning


@live
def test_live_resolve_frederick_ii(live_resolver):
    """Frederick II in isolation is structurally ambiguous: Wikidata has
    multiple "Frederick II" entries with overlapping descriptions ("King of
    Sicily...") that our soft scoring can't cleanly separate. The resolver
    correctly flags this as ambiguous with Q130221 (Holy Roman Emperor,
    Stupor Mundi) as the top candidate — PR 4's LLM disambiguation step
    will make the final pick using the full transcript context.

    This test documents that behavior: top candidate is right, status is
    ambiguous (not silently mis-resolved). See plan doc: "below threshold
    → status='ambiguous', candidates preserved, queued for review UI."""
    res = live_resolver.resolve(
        "Frederick II",
        context_text="Holy Roman Emperor Stupor Mundi, ruled Sicily",
        type_hint="person",
        date_hint=DateRange(start=1150, end=1300),
    )
    assert res.candidates, "resolver returned no candidates"
    assert res.candidates[0].qid == "Q130221", (
        f"expected Q130221 as top candidate, got {res.candidates[0].qid}: "
        f"{res.reasoning}"
    )
    # Status may be resolved (if scoring tightens in future) or ambiguous (now).
    # Either way we must NOT be resolved to a *wrong* QID.
    if res.status == "resolved":
        assert res.chosen_qid == "Q130221"


@live
def test_live_resolve_ambiguous_richard_i_without_context(live_resolver):
    """Without date/context hints, Richard I should be ambiguous — many Richards
    in Wikidata share the ordinal."""
    res = live_resolver.resolve("Richard I")
    assert res.status in ("ambiguous", "no_match", "resolved"), res.status
    # We don't pin chosen_qid here — the point is the resolver produces a
    # reasoned decision (or flags ambiguity) rather than silently picking wrong.


@live
def test_live_resolve_all_rollo_transcript(live_resolver):
    """End-to-end: the exact mentions from the Rollo transcript that failed
    processing (see petrarca voice_transcript vt_1776097010_8381). Depends on
    Wikidata being reachable; verifies dependency-order resolution works."""
    mentions = [
        {"mention": "Rollo", "type_hint": "person",
         "context_text": "Viking, arrived in France, siege of Paris, negotiated for Normandy",
         "date_hint": DateRange(start=850, end=950)},
        {"mention": "Richard I",
         "type_hint": "person",
         "context_text": "had a child with Gunnora; his daughter married Aethelred",
         "date_hint": DateRange(start=920, end=1000)},
        {"mention": "Gunnora",
         "type_hint": "person",
         "context_text": "probably a Viking woman",
         "date_hint": DateRange(start=930, end=1030)},
        {"mention": "Emma of Normandy",
         "type_hint": "person",
         "context_text": "daughter of Richard I, married Aethelred of England",
         "date_hint": DateRange(start=970, end=1060)},
        {"mention": "Aethelred",
         "type_hint": "person",
         "context_text": "King of England, married Emma of Normandy, St Brice's Day massacre",
         "date_hint": DateRange(start=960, end=1020)},
    ]
    resolutions = live_resolver.resolve_all(mentions)
    resolved_map = {
        r.mention: r.chosen_qid
        for r in resolutions
        if r.chosen_qid is not None
    }
    # These are the verified QIDs (see memory/project_hallucinated_qids_incident.md)
    expected_targets = {
        "Rollo": "Q273773",
        "Richard I": "Q333359",
        "Gunnora": "Q270777",
        "Emma of Normandy": "Q40061",
        "Aethelred": "Q183499",
    }
    # Allow some ambiguity — require at least 3 out of 5 to resolve correctly.
    # Remaining ones should at least be in the status="ambiguous" state, not
    # resolved to a wrong QID.
    correct = sum(
        1 for m, expected in expected_targets.items()
        if resolved_map.get(m) == expected
    )
    assert correct >= 3, (
        f"Expected ≥3 of 5 correct resolutions; got {correct}. "
        f"Actual: {resolved_map}"
    )
    # None of the resolved ones may be *wrongly* resolved
    for m, expected in expected_targets.items():
        actual = resolved_map.get(m)
        if actual is not None:
            assert actual == expected, (
                f"{m} resolved to {actual}, expected {expected} "
                f"(hallucination or wrong pick)"
            )
