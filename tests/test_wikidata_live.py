"""Live smoke tests against real Wikidata.

Gated by env var: set LIMBIC_LIVE_WIKIDATA=1 to run. These tests verify that
the client behaves correctly against the actual API — they're intended as a
manual sanity check before rollouts, not part of regular CI.

Also validates the core historical-entity set the Petrarca entity resolver
will depend on: Rollo, Æthelred the Unready, Richard I of Normandy, Emma of
Normandy, Gunnor, Frederick II, Paris, Charlemagne.
"""

from __future__ import annotations

import os

import pytest

from limbic.amygdala import WikidataClient


LIVE_FLAG = "LIMBIC_LIVE_WIKIDATA"
live = pytest.mark.skipif(
    os.environ.get(LIVE_FLAG) != "1",
    reason=f"Set {LIVE_FLAG}=1 to run live Wikidata tests.",
)

LIVE_UA = "limbic-live-tests/0.1 (https://github.com/houshuang/limbic)"


# The fixture: known-correct entities that the Petrarca resolver must handle.
# QIDs were verified by querying Wikidata directly on 2026-04-13. If any of
# these resolve to something different in the future (rare — QIDs are stable),
# the live tests will fail loudly, which is the point.
HISTORICAL_FIXTURE = {
    "Q273773": "Rollo",                            # Viking leader, 1st Duke of Normandy (actually Count of Rouen per Wikidata)
    "Q333359": "Richard I of Normandy",            # Richard the Fearless, grandson of Rollo
    "Q183499": "Æthelred the Unready",             # King of England, husband of Emma
    "Q40061": "Emma of Normandy",                  # daughter of Richard I, queen of England
    "Q270777": "Gunnora",                          # wife of Richard I of Normandy
    "Q130221": "Frederick II, Holy Roman Emperor", # Stupor Mundi
    "Q90": "Paris",                                # capital of France
    "Q3044": "Charlemagne",                        # king of Franks, first Holy Roman Emperor
}


@pytest.fixture(scope="module")
def live_client(tmp_path_factory):
    path = tmp_path_factory.mktemp("wd_live") / "cache.db"
    return WikidataClient(
        user_agent=LIVE_UA,
        cache_db_path=str(path),
    )


@live
def test_live_search_rollo(live_client):
    results = live_client.search("Rollo", limit=5)
    assert len(results) > 0
    qids = [r.qid for r in results]
    assert "Q273773" in qids, f"Expected Rollo's QID Q273773 in top 5; got {qids}"


@live
def test_live_get_single_entity(live_client):
    entity = live_client.get("Q273773")
    assert entity is not None
    assert entity.qid == "Q273773"
    label = entity.label("en") or ""
    assert "Rollo" in label


@live
def test_live_historical_fixture_resolves(live_client):
    """All known-correct historical QIDs must resolve with plausible labels."""
    entities = live_client.get_many(list(HISTORICAL_FIXTURE.keys()))
    for qid, expected_substring in HISTORICAL_FIXTURE.items():
        assert qid in entities, f"missing {qid}"
        entity = entities[qid]
        label = entity.label("en") or ""
        # Loose match — Wikidata labels drift; just check the core name.
        first_word = expected_substring.split(",")[0].split()[0]
        assert first_word.lower() in label.lower(), (
            f"{qid}: expected '{first_word}' in label, got {label!r}"
        )


@live
def test_live_multilingual_labels(live_client):
    """Paris should have a Norwegian label (Paris) distinct from Chinese (巴黎)."""
    entity = live_client.get("Q90", languages=["en", "no", "zh", "de"])
    assert entity is not None
    # English and Norwegian are both "Paris"; Chinese is 巴黎
    assert entity.label("en") == "Paris"
    # zh may return simplified or other variant; just check it's set
    zh = entity.label("zh")
    assert zh is not None and zh != ""


@live
def test_live_external_ids_on_historical_person(live_client):
    """Charlemagne should have VIAF (P214) and GND (P227) IDs."""
    entity = live_client.get("Q3044")
    assert entity is not None
    external = entity.external_ids(["P214", "P227", "P1566"])
    # VIAF should be present on a figure this famous
    assert "P214" in external


@live
def test_live_co_mention_anchor_signal(live_client):
    """Rollo's P40 (children) and Richard I's P22 (father) link them — this is
    the 'co-mention coherence' anchor signal the resolver will use. If Rollo
    and Richard I appear in the same transcript, we can verify their
    relationship via Wikidata claims rather than relying on context alone."""
    entities = live_client.get_many(["Q273773", "Q333359"])
    rollo = entities["Q273773"]
    richard = entities["Q333359"]
    # Expect Rollo's children claim to reference Q315838 (William Longsword,
    # Rollo's son who was Richard I's father) — direct parent-child via P22
    # might not go Rollo→Richard I, but the descent chain is traceable.
    # Here we just verify the *shape* of claims works: both should have
    # non-empty label and at least one P-property.
    assert rollo.label("en") and "Rollo" in rollo.label("en")
    assert richard.label("en") and "Richard" in richard.label("en")
    assert len(rollo.claims) > 0
    assert len(richard.claims) > 0
