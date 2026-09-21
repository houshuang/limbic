"""Tests for limbic.hippocampus.resolve."""

from __future__ import annotations

import pytest

from limbic.hippocampus.resolve import (
    Card, build_index, candidates, fold, name_keys, open_index, slot_enum,
    text_candidates, unslot,
)

PERSONS = [
    {"id": "1", "name": "Bjørnstjerne Bjørnson", "birth_year": 1832,
     "death_year": 1910, "summary": "Writer, Nobel laureate", "rank": 40},
    {"id": "2", "name": "Henrik Ibsen", "aliases": ["Ibsen, Henrik (1828-1906)"],
     "birth_year": 1828, "death_year": 1906, "summary": "Playwright", "rank": 90},
    {"id": "3", "name": "Arild Martinsen", "birth_year": 1950, "rank": 1},
    {"id": "4", "name": "Ludwig van Beethoven", "birth_year": 1770, "rank": 30},
    {"id": "5", "name": "Hansen, Sverre", "rank": 2},
    {"id": "6", "name": "Harald Hårfagre", "rank": 5},
]

WORKS = [
    {"id": "w1", "name": "Et dukkehjem", "aliases": ["A Doll's House"], "rank": 10},
    {"id": "w2", "name": "Die Zauberflöte", "aliases": ["Tryllefløyten"], "rank": 8},
    {"id": "w3", "name": "1984", "rank": 1},
]


@pytest.fixture
def index(tmp_path):
    idx = build_index(tmp_path / "kb.idx", PERSONS, kind="person")
    build_index(idx.conn, WORKS, kind="work")
    yield idx
    idx.close()


class TestFold:
    def test_drop_and_expand_spellings(self):
        assert fold("Bjørnson") == "bjornson"
        assert fold("Bjørnson", expand=True) == "bjoernson"
        assert fold("Næss", expand=True) == "naess"

    def test_map_runs_before_nfkd(self):
        """NFKD first would make the expanded spelling unreachable."""
        assert fold("Zauberflöte") == "zauberflote"
        assert fold("Zauberflöte", expand=True) == "zauberfloete"

    def test_separators_become_spaces(self):
        assert fold("Jean-Luc") == fold("Jean Luc") == "jean luc"

    def test_non_string_is_coerced_not_crashed(self):
        assert fold(1984) == "1984"

    def test_unknown_language_still_strips_decomposable_diacritics(self):
        assert fold("Café", lang="xx") == "cafe"


class TestFoldProfiles:
    def test_names_profile_is_the_default_and_unchanged(self):
        for text in ("Næss", "Bjørnson", "snake_case", "Zauberflöte"):
            assert fold(text) == fold(text, profile="names")
        assert fold("Næss") == "nass"
        assert fold("snake_case") == "snake_case"
        assert fold("Bjørnson", expand=True, profile="names") == "bjoernson"

    def test_ascii_profile(self):
        assert fold("Næss", profile="ascii") == "naess"
        assert fold("snake_case", profile="ascii") == "snake case"
        assert fold("Bjørnson, Bjørnstjerne", profile="ascii") == "bjornson bjornstjerne"
        assert fold("Zauberflöte", profile="ascii") == "zauberflote"
        assert fold("Œuvres de Łódź", profile="ascii") == "oeuvres de lodz"
        assert fold("Москва 1905", profile="ascii") == "1905"
        assert fold(None, profile="ascii") == ""
        assert fold(1984, profile="ascii") == "1984"

    def test_ascii_profile_ignores_lang_and_refuses_expand(self):
        assert fold("Zauberflöte", "de", profile="ascii") == "zauberflote"
        with pytest.raises(ValueError, match="single spelling"):
            fold("Næss", expand=True, profile="ascii")

    def test_unknown_profile(self):
        with pytest.raises(ValueError, match="unknown fold profile"):
            fold("x", profile="skard")


class TestNameKeys:
    def test_inversion(self):
        assert "henrik ibsen" in name_keys("Ibsen, Henrik")

    def test_generational_suffix_does_not_invert(self):
        assert "d y krogh" not in name_keys("Krogh, d.y.")

    def test_parenthetical_stripped(self):
        keys = name_keys("Ibsen, Henrik (1828-1906)")
        assert "henrik ibsen" in keys

    def test_particle_variant(self):
        keys = name_keys("Ludwig van Beethoven")
        assert "ludwig van beethoven" in keys
        assert "ludwig beethoven" in keys

    def test_genitive_stem(self):
        assert "ibsen hus" in name_keys("Ibsens hus")


class TestCandidates:
    def test_exact_beats_everything(self, index):
        cards = candidates(index, "Henrik Ibsen", kind="person")
        assert cards[0].id == "2"
        assert cards[0].match_type == "exact"
        assert cards[0].score == 1.0

    def test_dropped_diacritic_spelling_matches(self, index):
        assert candidates(index, "Bjornson", kind="person")[0].id == "1"

    def test_expanded_diacritic_spelling_matches(self, index):
        assert candidates(index, "Bjoernstjerne Bjoernson", kind="person")[0].id == "1"

    def test_inverted_name_matches(self, index):
        assert candidates(index, "Ibsen, Henrik", kind="person")[0].id == "2"

    def test_alias_matches_across_languages(self, index):
        assert candidates(index, "Tryllefløyten", kind="work")[0].id == "w2"

    def test_numeric_title_does_not_crash(self, index):
        assert candidates(index, "1984", kind="work")[0].id == "w3"

    def test_searches_every_kind_when_none_given(self, index):
        assert {c.kind for c in candidates(index, "Et dukkehjem")} == {"work"}

    def test_hints_rerank_but_never_filter(self, index):
        cards = candidates(index, "Henrik Ibsen", kind="person", hints={"birth_year": 1900})
        assert cards[0].id == "2"
        assert cards[0].score < 1.0
        assert any("birth_year" in n for n in cards[0].notes)

    def test_hint_agreement_marks_confident(self, index):
        cards = candidates(index, "Henrik Ibsen", kind="person", hints={"birth_year": 1828})
        assert "confident" in cards[0].notes

    def test_fuzzy_is_off_by_default(self, index):
        """A typo still reaches the deterministic layers; it just never scores
        as a near-miss, so nothing can auto-link on a one-character difference."""
        cards = candidates(index, "Aril Martinsen", kind="person")
        assert "fuzzy" not in {c.match_type for c in cards}
        assert all("below threshold" in c.notes for c in cards)

    def test_fuzzy_when_asked_is_capped_below_confidence(self, index):
        cards = candidates(index, "Aril Martinsen", kind="person", fuzzy=True)
        assert cards and cards[0].id == "3"
        assert cards[0].match_type == "fuzzy"
        assert cards[0].score <= 0.84
        assert "below threshold" in cards[0].notes

    def test_unknown_name_returns_nothing(self, index):
        assert candidates(index, "Zzyzx Qqqqq", kind="person") == []

    def test_empty_query_returns_nothing(self, index):
        assert candidates(index, "  ", kind="person") == []

    def test_rank_breaks_ties(self, index):
        """A lone surname returns the best-attested bearer, not the first id."""
        cards = candidates(index, "Hansen Sverre", kind="person")
        assert cards[0].id == "5"

    def test_card_line_is_compact(self, index):
        line = candidates(index, "Henrik Ibsen", kind="person")[0].line()
        assert "[2] Henrik Ibsen" in line
        assert "Playwright" in line


class TestTextCandidates:
    def test_full_name_in_passage(self, index):
        cards = text_candidates(index, "I 1879 skrev Henrik Ibsen Et dukkehjem.")
        assert {c.id for c in cards} == {"2", "w1"}

    def test_genitive_is_reached(self, index):
        cards = text_candidates(index, "Henrik Ibsens samtidsdramaer", kind="person")
        assert [c.id for c in cards] == ["2"]

    def test_partial_name_is_not_a_hit(self, index):
        assert text_candidates(index, "Ibsen alene", kind="person") == []

    def test_short_particle_does_not_block_a_work(self, index):
        """Index and scan must agree on the token minimum, or "Et dukkehjem"
        is unfindable in its own text."""
        assert [c.id for c in text_candidates(index, "les Et dukkehjem", kind="work")] == ["w1"]

    def test_empty_text(self, index):
        assert text_candidates(index, "", kind="person") == []


class TestSlotEnum:
    def test_enum_is_identical_regardless_of_candidate_count(self, index):
        a, _ = slot_enum(candidates(index, "Henrik Ibsen", kind="person"), 40)
        b, _ = slot_enum(candidates(index, "Bjornson", kind="person"), 40)
        assert a == b, "a per-packet enum is what measured 0% cached input"

    def test_slot_map_points_at_real_ids(self, index):
        cards = candidates(index, "Henrik Ibsen", kind="person")
        _, slot_map = slot_enum(cards, 40)
        assert slot_map["c01"] == "2"

    def test_none_is_always_representable(self, index):
        fragment, _ = slot_enum([], 5)
        assert fragment["enum"][0] == "none"
        assert len(fragment["enum"]) == 6

    def test_more_candidates_than_slots_is_refused(self, index):
        cards = [Card(kind="person", id=str(i), label=str(i), match_type="exact", score=1.0)
                 for i in range(5)]
        with pytest.raises(ValueError, match="exceed"):
            slot_enum(cards, 3)

    def test_width_follows_slot_count(self):
        fragment, _ = slot_enum([], 120)
        assert fragment["enum"][1] == "c001"
        assert fragment["enum"][-1] == "c120"


class TestUnslot:
    def test_maps_slots_back(self):
        items, problems = unslot([{"ref": "c02", "quote": "x"}], {"c01": "a", "c02": "b"})
        assert items[0]["ref_id"] == "b"
        assert problems == []

    def test_none_is_a_wanted_answer(self):
        items, problems = unslot([{"ref": "none"}], {"c01": "a"})
        assert items[0]["ref_id"] is None
        assert problems == []

    def test_unsupplied_slot_is_refused_not_guessed(self):
        items, problems = unslot([{"ref": "c09"}], {"c01": "a"})
        assert items == []
        assert "not a slot this packet supplied" in problems[0]


class TestIndex:
    def test_rebuild_one_kind_leaves_others_alone(self, tmp_path):
        idx = build_index(tmp_path / "kb.idx", PERSONS, kind="person")
        build_index(idx.conn, WORKS, kind="work")
        build_index(idx.conn, PERSONS[:1], kind="person")
        assert candidates(idx, "Et dukkehjem", kind="work")[0].id == "w1"
        assert candidates(idx, "Henrik Ibsen", kind="person") == []
        idx.close()

    def test_reopen_from_disk(self, tmp_path):
        build_index(tmp_path / "kb.idx", PERSONS, kind="person").close()
        with open_index(tmp_path / "kb.idx") as idx:
            assert candidates(idx, "Bjornson", kind="person")[0].id == "1"

    def test_index_is_a_sidecar_not_the_source(self, tmp_path):
        """The source database must be untouched by indexing."""
        source = tmp_path / "source.db"
        source.write_bytes(b"")
        build_index(tmp_path / "kb.idx", PERSONS, kind="person").close()
        assert source.read_bytes() == b""
        assert (tmp_path / "kb.idx").exists()


class TestPublicNameHelpers:
    def test_invert_name(self):
        from limbic.hippocampus.resolve import invert_name
        assert invert_name("Andre, Bjørn Tore") == "Bjørn Tore Andre"
        assert invert_name("Krogh, d.y.") is None
        assert invert_name("a, b, c") is None
        assert invert_name(1984) is None

    def test_strip_parenthetical(self):
        from limbic.hippocampus.resolve import strip_parenthetical
        assert strip_parenthetical("Ibsen, Henrik (1828-1906)") == "Ibsen, Henrik"
        assert strip_parenthetical("Henrik Ibsen") is None
        assert strip_parenthetical(1984) is None


def test_resolve_imports_without_heavy_dependencies():
    """A consumer without numpy or yaml installed must be able to import the
    resolver, and a CLI built on it must not pay their import time."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "import limbic.hippocampus.resolve\n"
        "from limbic.hippocampus import fold\n"
        "heavy = [m for m in ('numpy', 'yaml', 'limbic.amygdala') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_hippocampus_lazy_exports_are_complete():
    import limbic.hippocampus as hippocampus
    for name in hippocampus.__all__:
        assert getattr(hippocampus, name) is not None


class TestSpellingTables:
    """A query key and an indexed key must come from the same spelling table."""

    ROWS = [{"id": "1", "name": "Sigve Bøe"}, {"id": "2", "name": "Bjørnstjerne Bjørnson"},
            {"id": "3", "name": "Tor Åge Bringsværd"}]

    @pytest.fixture()
    def people(self):
        return build_index(":memory:", self.ROWS, "person")

    def test_expanded_query_does_not_meet_a_dropped_name(self, people):
        """"Bø" expands to "boe", which is what "Bøe" drops to: different names."""
        hits = candidates(people, "Sigve Bø", kind="person")
        assert all(c.match_type != "folded" for c in hits)
        assert all("confident" not in c.notes for c in hits)

    def test_dropped_query_does_not_meet_an_expanded_name(self):
        idx = build_index(":memory:", [{"id": "1", "name": "Sigve Bø"}], "person")
        assert all(c.match_type != "folded" for c in candidates(idx, "Sigve Bøe", kind="person"))

    @pytest.mark.parametrize("query,expected", [
        ("Sigve Boe", "1"), ("Bjoernstjerne Bjoernson", "2"), ("Bjornstjerne Bjornson", "2"),
        ("Bjørnson, Bjørnstjerne", "2"), ("Tor Aage Bringsvaerd", "3"), ("Tor Age Bringsvard", "3"),
    ])
    def test_either_plain_spelling_still_matches(self, people, query, expected):
        top = candidates(people, query, kind="person")[0]
        assert (top.id, top.match_type) == (expected, "folded")

    def test_transliterated_name_is_found_in_a_passage(self, people):
        assert [c.id for c in text_candidates(people, "Av Tor Aage BRINGSVÆRD.", kind="person")] == ["3"]

    def test_index_built_before_spellings_still_opens(self, tmp_path):
        import sqlite3
        path = tmp_path / "old.sqlite"
        build_index(path, self.ROWS, "person").close()
        conn = sqlite3.connect(path)
        conn.execute("ALTER TABLE resolve_name DROP COLUMN spelling")
        conn.commit(); conn.close()
        with open_index(path) as idx:
            assert candidates(idx, "Sigve Boe", kind="person")[0].id == "1"
