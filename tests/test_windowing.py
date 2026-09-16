"""Tests for limbic.cerebellum.windowing — windowed extraction and safe merge."""

from __future__ import annotations

import pytest

from limbic.cerebellum.windowing import (
    Collection,
    MergeSchema,
    Reference,
    check_references,
    dedup_by_field,
    merge_windows,
    namespace_ids,
    split_into_windows,
)


SCHEMA = MergeSchema([
    Collection("claims", prefix="C", dedup_field="text"),
    Collection("evidence", prefix="E", dedup_field="text",
               references=[Reference("supports_claim", target="claims")]),
    Collection("cases", prefix="CASE", dedup_field="name",
               references=[Reference("claims_supported", target="claims", many=True)]),
])


# ---------------------------------------------------------------------------
# split_into_windows
# ---------------------------------------------------------------------------


class TestSplitIntoWindows:
    def test_short_text_is_one_window(self):
        assert split_into_windows("short", window_size=100, overlap=10) == [("short", 0)]

    def test_windows_cover_the_whole_text(self):
        text = "".join(f"para {i}\n\n" for i in range(400))
        windows = split_into_windows(text, window_size=500, overlap=100)
        assert windows[0].start == 0
        assert windows[0].text.startswith("para 0")
        assert windows[-1].text.endswith(text[-20:])
        # Every window's recorded start must index the real text.
        for w in windows:
            assert text[w.start:w.start + len(w.text)] == w.text

    def test_windows_overlap(self):
        text = "".join(f"para {i}\n\n" for i in range(400))
        windows = split_into_windows(text, window_size=500, overlap=100)
        for prev, nxt in zip(windows, windows[1:]):
            assert nxt.start < prev.start + len(prev.text)

    def test_breaks_at_paragraph_boundaries(self):
        text = ("x" * 480) + "\n\n" + ("y" * 2000)
        windows = split_into_windows(text, window_size=500, overlap=100, snap=100)
        assert windows[0].text.endswith("\n\n")

    def test_unsplittable_text_still_terminates(self):
        """No paragraph boundary anywhere — must not loop or lose content."""
        text = "z" * 5000
        windows = split_into_windows(text, window_size=500, overlap=100)
        assert len(windows) > 1
        assert "".join(w.text for w in windows)  # completed at all
        assert windows[-1].text.endswith("z")

    @pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 200), (100, -1)])
    def test_invalid_parameters_are_refused(self, size, overlap):
        with pytest.raises(ValueError):
            split_into_windows("text" * 200, window_size=size, overlap=overlap)

    def test_pathological_overlap_still_advances(self):
        """A boundary snap can pull `end` back; start must never go backwards."""
        text = ("a\n\n" * 2000)
        windows = split_into_windows(text, window_size=120, overlap=119, snap=110)
        assert len(windows) < 10_000
        assert windows[-1].text.endswith(text[-3:])


# ---------------------------------------------------------------------------
# namespace_ids
# ---------------------------------------------------------------------------


class TestNamespaceIds:
    def test_ids_and_references_are_tagged(self):
        result = {
            "claims": [{"id": "C1", "text": "a"}],
            "evidence": [{"id": "E1", "text": "e", "supports_claim": "C1"}],
            "cases": [{"id": "CASE1", "name": "n", "claims_supported": ["C1"]}],
        }
        namespace_ids(result, 3, SCHEMA)
        assert result["claims"][0]["id"] == "w3:C1"
        assert result["evidence"][0]["supports_claim"] == "w3:C1"
        assert result["cases"][0]["claims_supported"] == ["w3:C1"]

    def test_source_window_is_stamped(self):
        result = {"claims": [{"id": "C1", "text": "a"}]}
        namespace_ids(result, 7, SCHEMA)
        assert result["claims"][0]["source_window"] == 7

    def test_missing_collections_and_empty_refs_are_tolerated(self):
        result = {"claims": [{"id": "C1", "text": "a"}],
                  "evidence": [{"id": "E1", "text": "e", "supports_claim": ""}]}
        namespace_ids(result, 0, SCHEMA)
        assert result["evidence"][0]["supports_claim"] == ""


# ---------------------------------------------------------------------------
# dedup_by_field
# ---------------------------------------------------------------------------


class TestDedupByField:
    def test_near_duplicates_collapse(self):
        items = [{"id": "a", "text": "the cat sat on the mat"},
                 {"id": "b", "text": "the cat sat on the mat today"}]
        kept, alias = dedup_by_field(items, "text")
        assert len(kept) == 1
        assert alias == {"a": "b", "b": "b"}

    def test_the_longer_text_wins(self):
        """A window edge truncates; the full restatement is the one to keep."""
        items = [{"id": "a", "text": "schools need a knowledge"},
                 {"id": "b", "text": "schools need a knowledge rich curriculum"}]
        kept, _ = dedup_by_field(items, "text")
        assert kept[0]["id"] == "b"

    def test_distinct_items_survive(self):
        items = [{"id": "a", "text": "reading comprehension needs background knowledge"},
                 {"id": "b", "text": "standardized testing distorts school incentives"}]
        kept, _ = dedup_by_field(items, "text")
        assert len(kept) == 2

    def test_alias_follows_a_displaced_survivor(self):
        """`a` is kept, then displaced by longer `b`; `a` must alias to `b`."""
        items = [{"id": "a", "text": "one two three"},
                 {"id": "b", "text": "one two three four five"}]
        _, alias = dedup_by_field(items, "text")
        assert alias["a"] == "b"

    def test_non_ascii_words_are_compared(self):
        """ASCII-only tokenizing treated every Norwegian sentence as empty."""
        items = [{"id": "a", "text": "skolen trenger et kunnskapsrikt læreplanverk"},
                 {"id": "b", "text": "skolen trenger et kunnskapsrikt læreplanverk nå"}]
        kept, _ = dedup_by_field(items, "text")
        assert len(kept) == 1

    def test_unrelated_non_ascii_items_are_not_collapsed(self):
        items = [{"id": "a", "text": "skolen trenger kunnskap"},
                 {"id": "b", "text": "været i Bergen er vått"}]
        kept, _ = dedup_by_field(items, "text")
        assert len(kept) == 2

    def test_empty_input(self):
        assert dedup_by_field([], "text") == ([], {})

    def test_empty_text_never_matches(self):
        items = [{"id": "a", "text": ""}, {"id": "b", "text": ""}]
        kept, _ = dedup_by_field(items, "text")
        assert len(kept) == 2


# ---------------------------------------------------------------------------
# merge_windows
# ---------------------------------------------------------------------------


def _window(text_a, ev_text, ev_ref):
    return {
        "claims": [{"id": "C1", "text": text_a}],
        "evidence": [{"id": "E1", "text": ev_text, "supports_claim": ev_ref}],
    }


class TestMergeWindows:
    def test_ids_are_sequential_and_prefixed(self):
        merged, _ = merge_windows(
            [_window("first claim here", "ev one", "C1"),
             _window("second unrelated claim", "ev two", "C1")],
            SCHEMA, strict=True)
        assert [c["id"] for c in merged["claims"]] == ["C1", "C2"]
        assert [e["id"] for e in merged["evidence"]] == ["E1", "E2"]

    def test_references_stay_with_their_own_window(self):
        """The bug namespacing exists to prevent: window 2's C1 is not window 1's."""
        merged, _ = merge_windows(
            [_window("first claim here", "ev one", "C1"),
             _window("second unrelated claim", "ev two", "C1")],
            SCHEMA, strict=True)
        by_text = {e["text"]: e["supports_claim"] for e in merged["evidence"]}
        claim_id = {c["text"]: c["id"] for c in merged["claims"]}
        assert by_text["ev one"] == claim_id["first claim here"]
        assert by_text["ev two"] == claim_id["second unrelated claim"]

    def test_overlap_duplicates_collapse_and_references_repoint(self):
        """A claim seen twice across the seam keeps both pieces of evidence."""
        merged, report = merge_windows(
            [_window("knowledge rich curriculum works", "ev one", "C1"),
             _window("knowledge rich curriculum works well", "ev two", "C1")],
            SCHEMA, strict=True)
        assert len(merged["claims"]) == 1
        assert report.duplicates_removed == 1
        assert {e["supports_claim"] for e in merged["evidence"]} == {"C1"}
        assert merged["claims"][0]["text"] == "knowledge rich curriculum works well"

    def test_list_references_are_rewritten(self):
        windows = [{
            "claims": [{"id": "C1", "text": "alpha beta gamma"},
                       {"id": "C2", "text": "delta epsilon zeta"}],
            "cases": [{"id": "CASE1", "name": "a case", "claims_supported": ["C1", "C2"]}],
        }]
        merged, _ = merge_windows(windows, SCHEMA, strict=True)
        assert merged["cases"][0]["claims_supported"] == ["C1", "C2"]

    def test_dangling_reference_is_cleared_not_left_pointing(self):
        windows = [{
            "claims": [{"id": "C1", "text": "alpha beta gamma"}],
            "evidence": [{"id": "E1", "text": "ev", "supports_claim": "C99"}],
        }]
        merged, report = merge_windows(windows, SCHEMA)
        assert merged["evidence"][0]["supports_claim"] is None
        assert report.dangling == []

    def test_strict_raises_on_a_surviving_dangling_reference(self):
        windows = [{
            "claims": [{"id": "C1", "text": "alpha beta gamma"}],
            "cases": [{"id": "CASE1", "name": "c", "claims_supported": ["C1"]}],
        }]
        merged, _ = merge_windows(windows, SCHEMA, strict=True)
        merged["claims"].clear()
        with pytest.raises(ValueError, match="dangling"):
            check_references(merged, SCHEMA, strict=True)

    def test_report_counts(self):
        _, report = merge_windows(
            [_window("knowledge rich curriculum works", "ev one", "C1"),
             _window("knowledge rich curriculum works well", "ev two", "C1")],
            SCHEMA, strict=True)
        assert report.items_before["claims"] == 2
        assert report.items_after["claims"] == 1
        assert report.references_checked == 2

    def test_empty_windows_are_tolerated(self):
        merged, report = merge_windows([{}, {"claims": []}], SCHEMA, strict=True)
        assert merged == {"claims": [], "evidence": [], "cases": []}
        assert report.duplicates_removed == 0

    def test_collection_without_dedup_field_keeps_everything(self):
        schema = MergeSchema([Collection("notes", prefix="N")])
        merged, _ = merge_windows(
            [{"notes": [{"id": "N1", "text": "same"}]},
             {"notes": [{"id": "N1", "text": "same"}]}], schema, strict=True)
        assert [n["id"] for n in merged["notes"]] == ["N1", "N2"]


class TestSchemaValidation:
    def test_duplicate_collection_names_are_refused(self):
        with pytest.raises(ValueError, match="duplicate collection"):
            MergeSchema([Collection("a", prefix="A"), Collection("a", prefix="B")])

    def test_reference_to_unknown_collection_is_refused(self):
        with pytest.raises(ValueError, match="unknown collection"):
            MergeSchema([Collection("a", prefix="A",
                                    references=[Reference("r", target="nope")])])
