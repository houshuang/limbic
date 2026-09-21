"""An audit demotes or does nothing. These tests exist to keep it that way."""

import pytest

from limbic.hippocampus.audit import AuditError, apply_audit


DECISIONS = [
    {"citation_key": "c1", "concept_id": "ada", "disposition": "new", "definition": "A mathematician."},
    {"citation_key": "c2", "concept_id": "ada", "disposition": "new", "definition": "A mathematician."},
    {"citation_key": "c3", "concept_id": "bob", "disposition": "variant_of", "definition": "A printer."},
    {"citation_key": "c4", "concept_id": "cyd", "disposition": "hold", "definition": ""},
]
KEYS = ("citation_key", "concept_id")


def test_a_hold_row_demotes_only_the_matching_decision():
    out, report = apply_audit(
        DECISIONS, {"citations": [{"citation_key": "c1", "note": "wrong person"}]},
        key_fields=KEYS, hold_sections=("citations",),
    )
    assert [d["disposition"] for d in out] == ["hold", "new", "variant_of", "hold"]
    assert out[0]["audit_note"] == "wrong person"
    assert out[0]["hold_reasons"] == ["independent_audit"]
    assert report["demoted"] == 1 and report["demoted_keys"] == ["c1"]


def test_a_concept_row_demotes_every_citation_of_that_concept():
    out, report = apply_audit(
        DECISIONS, {"concepts": [{"concept_id": "ada", "note": "duplicate of ada-lovelace"}]},
        key_fields=KEYS, hold_sections=("concepts",),
    )
    assert [d["disposition"] for d in out] == ["hold", "hold", "variant_of", "hold"]
    assert report["demoted"] == 2


def test_the_input_is_never_mutated():
    before = [dict(d) for d in DECISIONS]
    apply_audit(DECISIONS, {"citations": [{"citation_key": "c1", "note": "x"}]},
                key_fields=KEYS, hold_sections=("citations",))
    assert DECISIONS == before


def test_conflicting_hold_rows_for_one_id_stay_held():
    """Two audit rows disagree about c1 -- one holds it, the other tries to
    release it by asserting its own disposition. The row that arrives first
    wins the demotion and the release attempt is reported, never applied."""

    out, report = apply_audit(
        DECISIONS,
        {"citations": [
            {"citation_key": "c1", "note": "wrong person"},
            {"citation_key": "c1", "note": "actually right, unhold it", "disposition": "new"},
        ]},
        key_fields=KEYS, hold_sections=("citations",),
    )
    assert out[0]["disposition"] == "hold"
    assert out[0]["audit_note"] == "wrong person"
    assert report["demoted"] == 1
    assert report["already_held"] == [{"section": "citations", "key": "c1"}]


def test_an_audit_that_tries_to_promote_cannot():
    """The audit file's own words do not decide the disposition; the fold-in does."""

    out, _ = apply_audit(
        DECISIONS,
        {"citations": [{"citation_key": "c4", "note": "this is right, release it",
                        "disposition": "new"}]},
        key_fields=KEYS, hold_sections=("citations",),
    )
    assert out[3]["disposition"] == "hold"


def test_an_unnamed_section_is_reported_not_silently_dropped():
    _, report = apply_audit(
        DECISIONS, {"citations": [], "promotions": [{"citation_key": "c4"}]},
        key_fields=KEYS, hold_sections=("citations",),
    )
    assert report["ignored_sections"] == ["promotions"]


def test_an_unknown_id_is_reported():
    _, report = apply_audit(
        DECISIONS, {"citations": [{"citation_key": "nope", "note": "?"}]},
        key_fields=KEYS, hold_sections=("citations",),
    )
    assert report["unknown_ids"] == [{"section": "citations", "key": "nope"}]
    assert report["demoted"] == 0


def test_a_row_on_an_already_held_decision_is_reported_not_counted():
    _, report = apply_audit(
        DECISIONS, {"citations": [{"citation_key": "c4", "note": "still wrong"}]},
        key_fields=KEYS, hold_sections=("citations",),
    )
    assert report["demoted"] == 0
    assert report["already_held"] == [{"section": "citations", "key": "c4"}]


def test_a_correction_applies_when_the_validator_accepts_it():
    out, report = apply_audit(
        DECISIONS, {"definitions": {"ada": "The first programmer, 1815-1852."}},
        key_fields=KEYS, hold_sections=(),
        correction_sections=(("definitions", "definition"),),
        validator=lambda field, value, row: None,
    )
    assert out[0]["definition"] == "The first programmer, 1815-1852."
    assert len(report["corrected"]) == 2


def test_a_correction_the_validator_refuses_is_not_written():
    out, report = apply_audit(
        DECISIONS, {"definitions": {"ada": "i01"}},
        key_fields=KEYS, hold_sections=(),
        correction_sections=(("definitions", "definition"),),
        validator=lambda field, value, row: "output is a slot id" if len(value) < 12 else None,
    )
    assert out[0]["definition"] == "A mathematician."
    assert report["corrections_refused"] == [
        {"section": "definitions", "key": "ada", "field": "definition", "reason": "output is a slot id"}
    ]


def test_source_errors_are_kept_as_findings_and_never_applied():
    """A fact that is wrong in the source is a finding about the corpus."""

    errors = [{"concept_id": "bob", "cue": "the painter came to Stockholm",
               "note": "the textbook says so; he never left the Dutch Republic"}]
    out, report = apply_audit(
        DECISIONS, {"source_errors": errors},
        key_fields=KEYS, hold_sections=(), finding_sections=("source_errors",),
    )
    assert [d["disposition"] for d in out] == [d["disposition"] for d in DECISIONS]
    assert report["findings"]["source_errors"] == errors


@pytest.mark.parametrize("field", ["citation_key", "concept_id", "disposition"])
def test_a_correction_targeting_an_identifying_field_is_refused(field):
    """Re-keying is not a correction: it moves the decision onto another record."""

    with pytest.raises(AuditError, match="Re-key outside the audit"):
        apply_audit(
            DECISIONS, {"fixes": {"ada": "ada-lovelace"}},
            key_fields=KEYS, hold_sections=(),
            correction_sections=(("fixes", field),),
        )


def test_the_caller_can_protect_further_identity_fields():
    with pytest.raises(AuditError, match="Re-key outside the audit"):
        apply_audit(
            DECISIONS, {"fixes": {"ada": "Ada"}},
            key_fields=KEYS, hold_sections=(),
            correction_sections=(("fixes", "label"),),
            identity_fields=("label",),
        )


def test_the_refusal_does_not_depend_on_the_audit_carrying_such_a_row():
    """An empty section must not make a re-keying configuration look safe."""

    with pytest.raises(AuditError, match="Re-key outside the audit"):
        apply_audit(DECISIONS, {}, key_fields=KEYS, hold_sections=(),
                    correction_sections=(("fixes", "concept_id"),))


def test_an_unreconciled_decision_is_refused():
    with pytest.raises(AuditError, match="after reconciliation"):
        apply_audit([{"citation_key": "c1"}], {}, key_fields=KEYS)


def test_an_audit_row_with_no_usable_id_is_refused():
    with pytest.raises(AuditError, match="carries none of"):
        apply_audit(DECISIONS, {"citations": [{"note": "which one?"}]},
                    key_fields=KEYS, hold_sections=("citations",))


def test_every_disposition_is_unchanged_or_held():
    out, _ = apply_audit(
        DECISIONS,
        {"citations": [{"citation_key": k, "note": "n"} for k in ("c1", "c2", "c3", "c4")]},
        key_fields=KEYS, hold_sections=("citations",),
    )
    for old, new in zip(DECISIONS, out):
        assert new["disposition"] in (old["disposition"], "hold")
