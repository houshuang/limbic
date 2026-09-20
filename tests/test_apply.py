"""Tests for limbic.hippocampus.apply."""

from __future__ import annotations

import json

import pytest

from limbic.hippocampus.apply import (
    MISSING, apply_proposal, enum_member, regex, wikidata_exists, wikidata_type_is,
)


@pytest.fixture
def record(tmp_path):
    path = tmp_path / "work.json"
    path.write_text(json.dumps({"id": "w1", "title": "Et dukkehjem", "year": 1879}))
    return path


class FakeEntity:
    def __init__(self, qid, p31, label="thing"):
        self.qid, self._p31, self._label = qid, p31, label

    def claim_qids(self, prop):
        return self._p31 if prop == "P31" else []

    def label(self, lang="en"):
        return self._label


class FakeWikidata:
    """No network in tests. `entities` maps QID -> FakeEntity or None."""

    def __init__(self, entities):
        self.entities = entities
        self.asked: list[str] = []

    def get(self, qid):
        self.asked.append(qid)
        return self.entities.get(qid)


class TestApplyProposal:
    def test_applies_and_writes_atomically(self, record):
        receipt = apply_proposal(record, {"year": 1880}, preimage={"year": 1879},
                                 allowed_fields={"year"})
        assert receipt["applied"] is True
        assert json.loads(record.read_text())["year"] == 1880
        assert receipt["sha256_before"] != receipt["sha256_after"]
        assert not list(record.parent.glob("*.part"))

    def test_field_outside_the_whitelist_is_refused(self, record):
        receipt = apply_proposal(record, {"title": "Nora"}, preimage={"title": "Et dukkehjem"},
                                 allowed_fields={"year"})
        assert receipt["applied"] is False
        assert "outside the whitelist" in receipt["reason"]
        assert json.loads(record.read_text())["title"] == "Et dukkehjem"

    def test_preimage_mismatch_is_refused(self, record):
        receipt = apply_proposal(record, {"year": 1880}, preimage={"year": 1867},
                                 allowed_fields={"year"})
        assert receipt["applied"] is False
        assert "preimage mismatch" in receipt["reason"]
        assert json.loads(record.read_text())["year"] == 1879

    def test_missing_preimage_entry_is_refused(self, record):
        receipt = apply_proposal(record, {"year": 1880}, preimage={},
                                 allowed_fields={"year"})
        assert "no preimage" in receipt["reason"]

    def test_absent_key_is_not_null(self, record):
        """A proposer that saw nothing must not overwrite an explicit null."""
        record.write_text(json.dumps({"id": "w1", "wikidata_id": None}))
        refused = apply_proposal(record, {"wikidata_id": "Q1"},
                                 preimage={"wikidata_id": MISSING},
                                 allowed_fields={"wikidata_id"})
        assert refused["applied"] is False
        assert "preimage mismatch" in refused["reason"]

    def test_absent_key_matches_an_absent_key(self, record):
        receipt = apply_proposal(record, {"wikidata_id": "Q1"},
                                 preimage={"wikidata_id": MISSING},
                                 allowed_fields={"wikidata_id"})
        assert receipt["applied"] is True

    def test_explicit_null_preimage_matches_an_explicit_null(self, record):
        record.write_text(json.dumps({"id": "w1", "wikidata_id": None}))
        receipt = apply_proposal(record, {"wikidata_id": "Q1"},
                                 preimage={"wikidata_id": None},
                                 allowed_fields={"wikidata_id"})
        assert receipt["applied"] is True

    def test_validator_failure_is_refused(self, record):
        receipt = apply_proposal(record, {"year": 999}, preimage={"year": 1879},
                                 allowed_fields={"year"},
                                 validators=[lambda f, v: "too early" if v < 1000 else None])
        assert "too early" in receipt["reason"]
        assert json.loads(record.read_text())["year"] == 1879

    def test_never_partial(self, record):
        """One refused field refuses the whole proposal."""
        receipt = apply_proposal(record, {"year": 1880, "title": "Nora"},
                                 preimage={"year": 1879, "title": "Et dukkehjem"},
                                 allowed_fields={"year"})
        assert receipt["applied"] is False
        assert json.loads(record.read_text()) == {"id": "w1", "title": "Et dukkehjem", "year": 1879}

    def test_empty_changes_is_refused(self, record):
        assert "no changes" in apply_proposal(record, {}, preimage={}, allowed_fields=set())["reason"]

    def test_receipt_is_written_for_a_refusal_too(self, record, tmp_path):
        log = tmp_path / "receipts" / "r.jsonl"
        apply_proposal(record, {"year": 1880}, preimage={"year": 1867},
                       allowed_fields={"year"}, receipt=log)
        apply_proposal(record, {"year": 1880}, preimage={"year": 1879},
                       allowed_fields={"year"}, receipt=log)
        lines = [json.loads(line) for line in log.read_text().splitlines()]
        assert [r["applied"] for r in lines] == [False, True]

    def test_mapping_target_with_a_writer(self):
        row = {"id": "p1", "name": "Ibsen"}
        written: list[dict] = []
        receipt = apply_proposal(row, {"name": "Henrik Ibsen"}, preimage={"name": "Ibsen"},
                                 allowed_fields={"name"}, writer=written.append)
        assert receipt["applied"] is True
        assert written == [{"id": "p1", "name": "Henrik Ibsen"}]
        assert row["name"] == "Ibsen", "the writer owns the write, not this function"

    def test_mapping_target_without_a_writer_mutates_in_place(self):
        row = {"id": "p1", "name": "Ibsen"}
        apply_proposal(row, {"name": "Henrik Ibsen"}, preimage={"name": "Ibsen"},
                       allowed_fields={"name"})
        assert row["name"] == "Henrik Ibsen"

    def test_yaml_round_trip(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml
        path = tmp_path / "p.yaml"
        path.write_text(yaml.safe_dump({"id": "p1", "born": 1828}))
        apply_proposal(path, {"born": 1829}, preimage={"born": 1828}, allowed_fields={"born"})
        assert yaml.safe_load(path.read_text())["born"] == 1829


class TestValidators:
    def test_enum_member(self):
        check = enum_member({"play", "novel"})
        assert check("kind", "play") is None
        assert "controlled vocabulary" in check("kind", "poem")

    def test_enum_member_scoped_to_named_fields(self):
        check = enum_member({"play"}, fields={"kind"})
        assert check("title", "anything") is None

    def test_enum_member_allows_null(self):
        assert enum_member({"play"})("kind", None) is None

    def test_regex(self):
        check = regex(r"Q[1-9]\d*")
        assert check("qid", "Q42") is None
        assert check("qid", "42") is not None

    def test_regex_requires_a_full_match(self):
        assert regex(r"Q\d+")("qid", "Q42x") is not None

    def test_wikidata_exists_accepts_a_live_qid(self):
        client = FakeWikidata({"Q1": FakeEntity("Q1", ["Q5"])})
        assert wikidata_exists(client=client)("qid", "Q1") is None

    def test_wikidata_exists_refuses_a_deleted_qid(self):
        client = FakeWikidata({})
        assert "does not exist" in wikidata_exists(client=client)("qid", "Q999")

    def test_wikidata_exists_refuses_a_non_qid(self):
        assert "is not a QID" in wikidata_exists(client=FakeWikidata({}))("qid", "not-a-qid")

    def test_wikidata_exists_allows_empty(self):
        client = FakeWikidata({})
        assert wikidata_exists(client=client)("qid", "") is None
        assert client.asked == []

    def test_type_check_catches_what_existence_misses(self):
        """198 of 901 work QIDs in one catalogue pointed at a non-work; every
        one passed an existence check. Et dukkehjem resolved to Ramon Llull."""
        llull = FakeEntity("Q170065", ["Q5"], label="Ramon Llull")
        client = FakeWikidata({"Q170065": llull})
        assert wikidata_exists(client=client)("wikidata_id", "Q170065") is None
        message = wikidata_type_is("work", client=client)("wikidata_id", "Q170065")
        assert "Ramon Llull" in message and "not a work" in message

    def test_type_check_accepts_the_right_kind(self):
        client = FakeWikidata({"Q1": FakeEntity("Q1", ["Q7725634"], "Et dukkehjem")})
        assert wikidata_type_is("work", client=client)("wikidata_id", "Q1") is None

    def test_type_check_accepts_an_explicit_qid_allowlist(self):
        client = FakeWikidata({"Q1": FakeEntity("Q1", ["Q11424"])})
        assert wikidata_type_is(["Q11424"], client=client)("wikidata_id", "Q1") is None

    def test_type_check_refuses_an_item_with_no_p31(self):
        client = FakeWikidata({"Q1": FakeEntity("Q1", [])})
        assert "no P31 claim" in wikidata_type_is("work", client=client)("wikidata_id", "Q1")

    def test_unknown_type_hint_fails_loudly_at_build_time(self):
        with pytest.raises(ValueError, match="no P31 allowlist"):
            wikidata_type_is("spaceship")

    def test_validators_compose_in_apply(self, record):
        client = FakeWikidata({"Q170065": FakeEntity("Q170065", ["Q5"], "Ramon Llull")})
        receipt = apply_proposal(
            record, {"wikidata_id": "Q170065"}, preimage={"wikidata_id": MISSING},
            allowed_fields={"wikidata_id"},
            validators=[wikidata_exists(client=client), wikidata_type_is("work", client=client)])
        assert receipt["applied"] is False
        assert "Ramon Llull" in receipt["reason"]
