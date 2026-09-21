"""Tests for limbic.cerebellum.packet."""

from __future__ import annotations

import pytest

from limbic.cerebellum.calls import Held
from limbic.cerebellum.cost_log import cost_log
from limbic.cerebellum.packet import (
    LowYield, Packet, corpus_lowercase_words, lint_packet, make_packet, probe,
    run_packets, union_passes, unmatched_names, validate_quotes,
)

SCHEMA = {"type": "object", "properties": {"items": {"type": "array"}}}
PREFIX = "You code one bounded span.\n" + ("instructions " * 600)


def packet(n: int, *, prefix: str = PREFIX, schema=SCHEMA) -> Packet:
    return make_packet(prefix, {"page": f"text {n}", "candidates": []}, schema,
                       prompt_version="v1")


def fake_transport(result, *, usage: dict | None = None, fail: bool = False):
    calls: list[dict] = []

    def transport(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        if fail:
            raise RuntimeError("provider exploded")
        meta = {"cost": 0.002, "model": kwargs.get("model"), **(usage or {})}
        return (result(len(calls)) if callable(result) else result), meta

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


class TestMakePacket:
    def test_identity_covers_the_prefix(self):
        """A prefix edited after the batch was bought must change the id."""
        a = make_packet("prefix A", {"x": 1}, SCHEMA, prompt_version="v1")
        b = make_packet("prefix B", {"x": 1}, SCHEMA, prompt_version="v1")
        assert a["input_sha256"] != b["input_sha256"]

    def test_identity_covers_the_schema(self):
        a = make_packet(PREFIX, {"x": 1}, SCHEMA, prompt_version="v1")
        b = make_packet(PREFIX, {"x": 1}, {"type": "string"}, prompt_version="v1")
        assert a["input_sha256"] != b["input_sha256"]

    def test_identity_covers_the_prompt_version(self):
        a = make_packet(PREFIX, {"x": 1}, SCHEMA, prompt_version="v1")
        b = make_packet(PREFIX, {"x": 1}, SCHEMA, prompt_version="v2")
        assert a["input_sha256"] != b["input_sha256"]

    def test_same_input_same_id(self):
        assert packet(1)["input_sha256"] == packet(1)["input_sha256"]

    def test_body_key_order_does_not_change_identity(self):
        a = make_packet(PREFIX, {"a": 1, "b": 2}, SCHEMA, prompt_version="v1")
        b = make_packet(PREFIX, {"b": 2, "a": 1}, SCHEMA, prompt_version="v1")
        assert a["input_sha256"] == b["input_sha256"]

    def test_mutation_is_refused(self):
        p = packet(1)
        with pytest.raises(TypeError, match="frozen"):
            p["body_text"] = "something else"
        with pytest.raises(TypeError, match="frozen"):
            p.update({"x": 1})

    def test_is_json_serialisable(self):
        import json
        assert json.loads(json.dumps(packet(1)))["prompt_version"] == "v1"


class TestLint:
    def test_varying_schema_is_flagged(self):
        packets = [packet(1), make_packet(PREFIX, {"page": "b"},
                                          {"enum": ["Q1", "Q2"]}, prompt_version="v1")]
        assert any("schema varies" in w for w in lint_packet(packets))

    def test_identical_schema_is_not_flagged(self):
        assert not any("schema varies" in w for w in lint_packet([packet(1), packet(2)]))

    def test_short_prefix_is_flagged(self):
        assert any("cache minimum" in w for w in lint_packet([packet(1, prefix="hi")]))

    def test_long_prefix_is_not_flagged(self):
        assert not any("cache minimum" in w for w in lint_packet([packet(1)]))

    def test_varying_prefix_is_flagged(self):
        packets = [packet(1), packet(2, prefix=PREFIX + " extra")]
        assert any("static prefix varies" in w for w in lint_packet(packets))

    def test_constant_body_field_is_flagged_as_derivable(self):
        shared = {"evidence_fields": ["field %d" % i for i in range(60)]}
        packets = [make_packet(PREFIX, {"page": f"p{i}", **shared}, SCHEMA,
                               prompt_version="v1") for i in range(3)]
        assert any("byte-identical in every packet" in w for w in lint_packet(packets))

    def test_repeated_value_in_one_body_is_flagged(self):
        p = make_packet(PREFIX, {"rows": [{"k": 1}] * 8}, SCHEMA, prompt_version="v1")
        assert any("derivable, not data" in w for w in lint_packet(p))

    def test_oversized_packet_is_flagged(self):
        p = make_packet(PREFIX, {"items": list(range(40))}, SCHEMA, prompt_version="v1")
        assert any("dropping items silently" in w for w in lint_packet(p))

    def test_accepts_a_single_packet_or_a_batch(self):
        assert isinstance(lint_packet(packet(1)), list)
        assert isinstance(lint_packet([packet(1)]), list)


class TestRunPackets:
    def test_dry_run_calls_nothing(self):
        transport = fake_transport({"items": []})
        report = run_packets([packet(1), packet(2)], purpose="t", project="p",
                             transport=transport)
        assert transport.calls == []
        assert report["executed"] is False
        assert report["planned_calls"] == 2
        assert report["estimated_input_tokens"] > 0

    def test_execute_sends_one_call_per_packet(self):
        transport = fake_transport({"items": [1]})
        report = run_packets([packet(1), packet(2)], purpose="t", project="p",
                             transport=transport, execute=True, cache=False)
        assert len(transport.calls) == 2
        assert len(report["results"]) == 2
        assert report["stopped"] == "all packets sent"

    def test_max_calls_refuses_rather_than_finishing(self):
        transport = fake_transport({"items": []})
        report = run_packets([packet(i) for i in range(5)], purpose="t", project="p",
                             transport=transport, execute=True, cache=False, max_calls=2)
        assert len(transport.calls) == 2
        assert "max_calls" in report["stopped"]
        assert report["remaining"] == 3

    def test_max_tokens_refuses_before_the_call(self):
        transport = fake_transport({"items": []})
        report = run_packets([packet(i) for i in range(5)], purpose="t", project="p",
                             transport=transport, execute=True, cache=False, max_tokens=1)
        assert transport.calls == []
        assert "max_tokens" in report["stopped"]
        assert report["remaining"] == 5

    def test_failure_is_ledgered_not_swallowed(self):
        transport = fake_transport(None, fail=True)
        report = run_packets([packet(1)], purpose="t", project="p",
                             transport=transport, execute=True, cache=False)
        assert report["failures"][0]["error"].startswith("provider exploded")
        rows = cost_log.query(project="p")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "error"
        assert rows[0]["packet_id"] == packet(1)["packet_id"]

    def test_success_row_carries_the_packet_id(self):
        transport = fake_transport({"items": [1]})
        run_packets([packet(1)], purpose="t", project="p", transport=transport,
                    execute=True, cache=False)
        rows = cost_log.query(project="p")
        assert rows[0]["packet_id"] == packet(1)["packet_id"]

    def test_truncation_splits_once_and_never_re_asks(self):
        transport = fake_transport({"items": []}, usage={"output_tokens": 100})
        p = make_packet(PREFIX, {"page": "x"}, SCHEMA, prompt_version="v1",
                        max_output_tokens=100)
        children = [packet(10), packet(11)]
        splits: list[str] = []

        def split(parent):
            splits.append(parent["packet_id"])
            return children

        report = run_packets([p], purpose="t", project="p", transport=transport,
                             execute=True, cache=False, split=split)
        # parent, then both halves; the parent is never asked a second time
        assert len(transport.calls) == 3
        assert splits == [p["packet_id"]]
        assert report["split"][0]["into"] == [c["packet_id"] for c in children]

    def test_truncation_without_a_split_hook_is_a_failure_not_a_retry(self):
        transport = fake_transport({"items": []}, usage={"output_tokens": 100})
        p = make_packet(PREFIX, {"page": "x"}, SCHEMA, prompt_version="v1",
                        max_output_tokens=100)
        report = run_packets([p], purpose="t", project="p", transport=transport,
                             execute=True, cache=False)
        assert len(transport.calls) == 1
        assert "truncated" in report["failures"][0]["error"]

    def test_disagreement_is_held_not_tie_broken(self):
        transport = fake_transport(lambda n: {"answer": n})
        report = run_packets([packet(1)], purpose="t", project="p",
                             transport=transport, execute=True, cache=False,
                             replicates=2, agree=2)
        assert report["held"] and not report["results"]
        rows = [r for r in cost_log.query(project="p") if r["outcome"] == "held"]
        assert rows

    def test_agreeing_replicates_return_the_result(self):
        transport = fake_transport({"answer": 1})
        report = run_packets([packet(1)], purpose="t", project="p",
                             transport=transport, execute=True, cache=False,
                             replicates=2, agree=2)
        assert report["results"][0]["result"] == {"answer": 1}

    def test_outcome_fn_closes_the_ledger_loop(self):
        transport = fake_transport({"items": []})
        run_packets([packet(1)], purpose="t", project="p", transport=transport,
                    execute=True, cache=False,
                    outcome_fn=lambda r: "applied" if r["items"] else "no_op")
        assert cost_log.query(project="p")[0]["outcome"] == "no_op"


class TestProbe:
    def test_dry_probe_reports_without_calling(self):
        transport = fake_transport({"items": [1]})
        report = probe([packet(i) for i in range(100)], n=10, yield_fn=lambda r: len(r["items"]),
                       purpose="t", project="p", transport=transport)
        assert transport.calls == []
        assert report["sampled"] == 10
        assert report["executed"] is False

    def test_sample_is_spread_and_deterministic(self):
        packets = [packet(i) for i in range(100)]
        transport = fake_transport({"items": []})
        a = probe(packets, n=5, yield_fn=len, purpose="t", project="p", transport=transport)
        b = probe(packets, n=5, yield_fn=len, purpose="t", project="p", transport=transport)
        assert a["sampled"] == b["sampled"] == 5

    def test_stratified_sample_covers_every_stratum(self):
        packets = [make_packet(PREFIX, {"page": str(i)}, SCHEMA, prompt_version="v1",
                               meta={"subject": "abc"[i % 3]}) for i in range(30)]
        transport = fake_transport({"items": []})
        captured: list[str] = []

        def yield_fn(_result):
            return 0

        probe(packets, n=3, yield_fn=yield_fn, stratify_by=lambda p: p["meta"]["subject"],
              purpose="t", project="p", transport=transport, execute=True, cache=False)
        for call in transport.calls:
            captured.append(call["prompt"])
        assert len(captured) == 3

    def test_yield_rate_and_cost_per_actionable(self):
        transport = fake_transport(lambda n: {"items": [1] if n % 2 else []})
        report = probe([packet(i) for i in range(4)], n=4,
                       yield_fn=lambda r: len(r["items"]), purpose="t", project="p",
                       transport=transport, execute=True, cache=False)
        assert report["calls"] == 4
        assert report["yield_rate"] == 0.5
        assert report["cost_per_actionable"] is not None

    def test_low_yield_refuses_rather_than_reporting(self):
        transport = fake_transport({"items": []})
        with pytest.raises(LowYield, match="deterministic join"):
            probe([packet(i) for i in range(4)], n=4, yield_fn=lambda r: len(r["items"]),
                  min_yield=0.2, purpose="t", project="p", transport=transport,
                  execute=True, cache=False)

    def test_probe_sets_ledger_outcomes(self):
        transport = fake_transport({"items": []})
        probe([packet(1)], n=1, yield_fn=lambda r: len(r["items"]), purpose="t",
              project="p", transport=transport, execute=True, cache=False)
        assert cost_log.query(project="p")[0]["outcome"] == "no_op"


class TestValidateQuotes:
    PAGES = {"p01": "Undervisningen skal gi elevane  innsikt i\nnorsk litteratur."}

    def test_exact_substring_passes(self):
        valid, problems = validate_quotes(
            [{"page_ref": "p01", "quote": "innsikt i norsk litteratur"}], self.PAGES)
        assert len(valid) == 1 and problems == []

    def test_whitespace_is_collapsed_on_both_sides(self):
        valid, _ = validate_quotes(
            [{"page_ref": "p01", "quote": "elevane   innsikt"}], self.PAGES)
        assert len(valid) == 1

    def test_paraphrase_is_refused(self):
        valid, problems = validate_quotes(
            [{"page_ref": "p01", "quote": "innsikt i norsk litteraturhistorie"}], self.PAGES)
        assert valid == [] and "is not on page p01" in problems[0]

    def test_spelling_is_not_repaired(self):
        valid, _ = validate_quotes(
            [{"page_ref": "p01", "quote": "innsikt i Norsk litteratur"}], self.PAGES)
        assert valid == []

    def test_unknown_page_is_refused(self):
        valid, problems = validate_quotes([{"page_ref": "p99", "quote": "x"}], self.PAGES)
        assert valid == [] and "not a page of this packet" in problems[0]

    def test_empty_quote_is_refused(self):
        valid, problems = validate_quotes([{"page_ref": "p01", "quote": ""}], self.PAGES)
        assert valid == [] and "empty quote" in problems[0]


class TestUnionPasses:
    def test_passes_are_merged_not_chosen_between(self):
        merged = union_passes(
            {"candidates": [{"id": "a"}, {"id": "b"}], "names": [{"id": "b"}, {"id": "c"}]},
            key=lambda i: i["id"])
        assert [i["id"] for i in merged] == ["a", "b", "c"]

    def test_first_pass_wins_and_provenance_is_kept(self):
        merged = union_passes(
            {"one": [{"id": "a", "v": 1}], "two": [{"id": "a", "v": 2}]},
            key=lambda i: i["id"])
        assert merged == [{"id": "a", "v": 1, "_pass": "one"}]

    def test_duplicates_within_one_pass_are_kept(self):
        merged = union_passes({"one": [{"id": "a"}, {"id": "a"}]}, key=lambda i: i["id"])
        assert len(merged) == 2


class TestUnmatchedNames:
    TEXT = ("Elevane skal lese Harald Hårfagre og Henrik Ibsen. "
            "Dessuten skal dei kjenne til Noreg.")

    def test_unknown_name_is_surfaced(self):
        found = unmatched_names(self.TEXT, known_labels=["Henrik Ibsen"])
        assert "Harald Hårfagre" in found

    def test_known_name_is_not_surfaced(self):
        assert "Henrik Ibsen" not in unmatched_names(self.TEXT, ["Henrik Ibsen"])

    def test_fragment_of_a_known_name_is_not_surfaced(self):
        assert "Ibsen" not in unmatched_names("Les Ibsen i dag.", ["Henrik Ibsen"])

    def test_sentence_initial_single_word_is_skipped(self):
        assert "Dessuten" not in unmatched_names(self.TEXT, [])

    def test_corpus_lowercase_vocab_filters_common_nouns(self):
        text = "Han las Stoffet nøye."
        assert "Stoffet" in unmatched_names(text, [])
        vocab = corpus_lowercase_words(["stoffet er stort", "stoffet igjen", "stoffet her"])
        assert "Stoffet" not in unmatched_names(text, [], vocab)

    def test_explicit_stopwords_drop_furniture(self):
        assert "Noreg" not in unmatched_names(self.TEXT, [], stopwords=["Noreg"])

    def test_names_never_span_a_line_break(self):
        found = unmatched_names("Norsk\nFelles mål her", [])
        assert "Norsk\nFelles" not in found

    def test_corpus_lowercase_words_needs_repetition(self):
        assert corpus_lowercase_words(["stoffet her"], minimum=3) == set()


# ---------------------------------------------------------------------------
# Quote anchoring
# ---------------------------------------------------------------------------

from limbic.cerebellum.packet import (  # noqa: E402
    reanchor_quote, text_quote_anchor, unresolved_text_quote_anchor,
)

_PAGE = "Eleven skal  lese\nIbsen og Bjørnson.  Eleven skal lese høyt,\tog eleven skal lese Ibsen igjen."


class TestTextQuoteAnchor:
    def test_selector_fields(self):
        anchor = text_quote_anchor(_PAGE, "lese   Ibsen og", "p12", context_chars=10)
        page = "Eleven skal lese Ibsen og Bjørnson. Eleven skal lese høyt, og eleven skal lese Ibsen igjen."
        assert anchor["type"] == "TextQuoteSelector"
        assert anchor["exact"] == "lese Ibsen og"
        assert page[anchor["start"]:anchor["end"]] == anchor["exact"]
        assert anchor["prefix"] == "even skal "
        assert anchor["suffix"] == " Bjørnson."
        assert anchor["occurrence_index"] == 0
        assert anchor["extraction_version_id"] == "sha256:" + anchor["page_text_sha256"]

    def test_match_ignores_case_and_returns_the_page_spelling(self):
        assert text_quote_anchor(_PAGE, "ELEVEN SKAL LESE høyt", "p12")["exact"] == "Eleven skal lese høyt"

    def test_occurrence_index(self):
        first = text_quote_anchor(_PAGE, "lese Ibsen", "p12")
        second = text_quote_anchor(_PAGE, "lese Ibsen", "p12", occurrence_index=1)
        assert second["start"] > first["start"]
        assert second["selector_sha256"] != first["selector_sha256"]
        assert second["span_sha256"] == first["span_sha256"]
        with pytest.raises(ValueError, match="occurrence 2 not found"):
            text_quote_anchor(_PAGE, "lese Ibsen", "p12", occurrence_index=2)

    def test_missing_and_empty_quotes_raise(self):
        with pytest.raises(ValueError, match="not found"):
            text_quote_anchor(_PAGE, "lese Hamsun", "p12")
        with pytest.raises(ValueError, match="empty quote"):
            text_quote_anchor(_PAGE, "  ", "p12")

    def test_selector_survives_an_edit_elsewhere_on_the_page(self):
        before = text_quote_anchor(_PAGE, "lese høyt", "p12", context_chars=8)
        after = text_quote_anchor("Rettet overskrift. " + _PAGE, "lese høyt", "p12", context_chars=8)
        assert after["selector_sha256"] == before["selector_sha256"]
        assert after["start"] != before["start"]
        assert after["page_text_sha256"] != before["page_text_sha256"]

    def test_hashes_match_the_reference_construction(self):
        import hashlib
        import json

        anchor = text_quote_anchor(_PAGE, "lese høyt", "p12")
        selector = {k: anchor[k] for k in ("type", "page_id", "exact", "prefix", "suffix", "occurrence_index")}
        canonical = json.dumps(selector, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert anchor["selector_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
        assert anchor["span_sha256"] == hashlib.sha256("lese høyt".encode()).hexdigest()

    def test_unresolved_anchor(self):
        anchor = unresolved_text_quote_anchor(_PAGE, "lese  Hamsun", "p12")
        assert anchor["type"] == "UnresolvedTextQuoteSelector"
        assert anchor["expected_exact"] == "lese Hamsun"
        assert "start" not in anchor and "exact" not in anchor
        assert anchor["page_text_sha256"] == text_quote_anchor(_PAGE, "lese", "p12")["page_text_sha256"]


class TestReanchorQuote:
    PAGES = {"p1": "Om Ibsen og hans tid.", "p2": "Bjørnson skrev Synnøve Solbakken.", "p3": "Mer om Ibsen."}

    def test_moves_to_the_single_other_page(self):
        page_id, anchor = reanchor_quote(self.PAGES, "Synnøve  Solbakken", cited="p1")
        assert page_id == "p2" and anchor["page_id"] == "p2" and anchor["exact"] == "Synnøve Solbakken"

    def test_ambiguous_absent_and_empty_return_none(self):
        assert reanchor_quote(self.PAGES, "Ibsen", cited="p2") is None
        assert reanchor_quote(self.PAGES, "Hamsun", cited="p1") is None
        assert reanchor_quote(self.PAGES, "", cited="p1") is None

    def test_the_cited_page_is_not_a_candidate(self):
        assert reanchor_quote(self.PAGES, "Synnøve", cited="p2") is None
