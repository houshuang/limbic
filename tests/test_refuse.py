"""Pre-write guards: schema, plausibility, prose-vs-structure, declared count."""

import pytest

from limbic.hippocampus.refuse import (
    DeclaredCountError,
    SchemaSupportError,
    dates_disagree,
    declared_count,
    expect,
    json_schema_refusals,
    schema_refusals,
    temporal_plausibility_refusals,
)


SPINE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {
        "extent": {
            "type": "object",
            "required": ["start_year"],
            "properties": {
                "start_year": {"type": "integer", "minimum": -3000},
                "end_year": {"type": ["integer", "null"]},
                "precision": {"enum": ["exact-year", "circa", "century"]},
            },
            "additionalProperties": False,
        }
    },
    "type": "object",
    "required": ["id", "type", "definition"],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z0-9-]+$", "minLength": 2},
        "type": {"enum": ["person", "place", "work"]},
        "definition": {"type": "string", "minLength": 10},
        "temporal_extents": {"type": "array", "items": {"$ref": "#/$defs/extent"}, "minItems": 1},
        "identifiers": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
    "allOf": [
        {
            "if": {"properties": {"type": {"const": "person"}}, "required": ["type"]},
            "then": {"required": ["temporal_extents"]},
        }
    ],
}

GOOD = {
    "id": "ada-lovelace",
    "type": "person",
    "definition": "English mathematician who wrote the first published algorithm.",
    "temporal_extents": [{"start_year": 1815, "end_year": 1852, "precision": "exact-year"}],
    "identifiers": ["Q7259"],
}


# --- schema refusals -------------------------------------------------------

def test_a_valid_record_is_not_refused():
    assert schema_refusals([GOOD], SPINE) == []


def test_a_person_with_no_dates_is_refused():
    """The exact defect: a person with no dates and no identifier reached the store."""

    bad = {k: v for k, v in GOOD.items() if k not in ("temporal_extents", "identifiers")}
    assert schema_refusals([bad], SPINE) == [
        "ada-lovelace: 'temporal_extents' is a required property"
    ]


def test_refusals_are_prefixed_with_the_record_id():
    bad = {**GOOD, "id": "Ada Lovelace!"}
    assert schema_refusals([bad], SPINE)[0].startswith("Ada Lovelace!: id: ")


def test_a_record_with_no_id_still_reports():
    assert schema_refusals([{"type": "place"}], SPINE)[0].startswith("<no id>: ")


def test_only_limits_the_check_to_the_records_this_apply_writes():
    """A store already failing for an older reason is not blamed on this write."""

    stale = {"id": "old", "type": "nope"}
    assert schema_refusals([GOOD, stale], SPINE, only=lambda r: r["id"] == "ada-lovelace") == []
    assert schema_refusals([GOOD, stale], SPINE) != []


def test_extra_checks_come_back_through_the_same_call():
    refusals = schema_refusals(
        [GOOD], SPINE,
        extra_checks=[lambda r: ["definition names no century"] if "century" not in r["definition"] else []],
    )
    assert refusals == ["ada-lovelace: definition names no century"]


@pytest.mark.parametrize(
    "record, fragment",
    [
        ({**GOOD, "type": "creature"}, "is not one of"),
        ({**GOOD, "definition": "short"}, "is too short"),
        ({**GOOD, "id": "A"}, "does not match"),
        ({**GOOD, "temporal_extents": []}, "should be non-empty"),
        ({**GOOD, "temporal_extents": [{"end_year": 1852}]}, "is a required property"),
        ({**GOOD, "temporal_extents": [{"start_year": 1815, "x": 1}]}, "Additional properties"),
        ({**GOOD, "temporal_extents": [{"start_year": "1815"}]}, "is not of type"),
        ({**GOOD, "identifiers": ["Q1", "Q1"]}, "has non-unique elements"),
        ({**GOOD, "temporal_extents": [{"start_year": -9999}]}, "less than the minimum"),
    ],
)
def test_each_supported_keyword_refuses(record, fragment):
    refusals = schema_refusals([record], SPINE)
    assert any(fragment in r for r in refusals), refusals


def test_an_unsupported_keyword_raises_instead_of_passing_the_record():
    """A validator that skips the keyword that would have caught the defect is worse
    than none, because it reports a clean run."""

    with pytest.raises(SchemaSupportError, match="contentEncoding"):
        json_schema_refusals({"a": "x"}, {"properties": {"a": {"contentEncoding": "base64"}}})


def test_an_unrecognized_format_value_raises_instead_of_passing_silently():
    """`format` is in _SUPPORTED, so an unrecognized value (e.g. "uuid") slipped
    past the unsupported-keyword gate and validated everything as clean -- the
    exact vacuous pass this module exists to refuse."""

    with pytest.raises(SchemaSupportError, match="uuid"):
        json_schema_refusals({"a": "not-a-uuid"}, {"properties": {"a": {"format": "uuid"}}})


def test_a_bool_is_not_an_integer():
    assert json_schema_refusals({"n": True}, {"properties": {"n": {"type": "integer"}}})


def test_an_injected_validator_replaces_the_subset():
    assert schema_refusals([GOOD], SPINE, validate=lambda r, s: ["nope"]) == ["ada-lovelace: nope"]


# --- temporal plausibility -------------------------------------------------

def test_a_death_inside_living_memory_is_refused():
    out = temporal_plausibility_refusals(
        [{"start_year": 1940, "end_year": 2024, "status": "two-read-agreement"}],
        living_year=1990,
    )
    assert out == ["a death in 2024 is inside living memory and cannot be checked here"]


def test_an_exact_year_life_over_a_century_is_refused():
    out = temporal_plausibility_refusals(
        [{"start_year": 1650, "end_year": 1762, "precision": "exact-year"}], living_year=1990
    )
    assert out == ["an exact-year life of 112 years is two people or an error"]


def test_the_same_life_given_circa_is_not_refused():
    assert temporal_plausibility_refusals(
        [{"start_year": 1650, "end_year": 1762, "precision": "circa"}], living_year=1990
    ) == []


def test_an_authority_sourced_extent_is_left_alone():
    assert temporal_plausibility_refusals(
        [{"start_year": 1940, "end_year": 2024, "status": "wikidata"}],
        living_year=1990, statuses=("two-read-agreement",),
    ) == []


def test_living_year_has_no_default():
    with pytest.raises(TypeError):
        temporal_plausibility_refusals([])  # type: ignore[call-arg]


# --- prose vs structured dates ---------------------------------------------

def test_prose_contradicting_its_own_structured_dates_is_refused():
    """peter-kolbjornsen: the definition said 1687-1737, the record said 1683-1738."""

    out = dates_disagree(
        "Norwegian ironworks owner (1687–1737) who supplied the crown.",
        {"start_year": 1683, "end_year": 1738},
    )
    assert out == [
        "prose says the period begins 1687, the record says 1683",
        "prose says the period ends 1737, the record says 1738",
    ]


def test_agreeing_prose_is_not_refused():
    assert dates_disagree("Ironworks owner (1683–1738).", {"start_year": 1683, "end_year": 1738}) == []


def test_a_comma_fenced_range_is_a_lifespan_aside():
    """bartholomeus-deichman: "Norsk biskop av Oslo, 1671-1733" against 1671-1731."""

    assert dates_disagree(
        "Bishop of Oslo, 1671–1733, who argued for a Norwegian university.",
        {"start_year": 1671, "end_year": 1731},
    ) == ["prose says the period ends 1733, the record says 1731"]


def test_any_year_range_is_available_for_a_corpus_that_wants_it():
    from limbic.hippocampus.refuse import ANY_YEAR_RANGE

    assert dates_disagree(
        "President 1861–1865.", {"start_year": 1809, "end_year": 1865},
        range_pattern=ANY_YEAR_RANGE,
    ) == ["prose says the period begins 1861, the record says 1809"]


def test_a_birth_death_pair_is_read_as_a_range():
    assert dates_disagree("b. 1687, d. 1737", {"start_year": 1683, "end_year": 1738})


@pytest.mark.parametrize(
    "prose",
    [
        "He signed the constitution at Eidsvoll in 1814.",
        "A bishop of Nidaros with no dates given.",
        "Active in the 1700s.",
        # A reign is not a life: matching any range fired on 74 of 580 real
        # pairs, almost all of them this.
        "President of the United States 1861–1865, who led the Union.",
        "Bishop of Oslo from 1510 until 1522.",
    ],
)
def test_prose_without_a_fenced_lifespan_never_fires(prose):
    assert dates_disagree(prose, {"start_year": 1683, "end_year": 1738}) == []


def test_a_record_without_structured_dates_never_fires():
    assert dates_disagree("Lived 1687–1737.", {}) == []
    assert dates_disagree("Lived 1687–1737.", None) == []


def test_tolerance_allows_a_circa_year():
    assert dates_disagree("(c. 1684–1738)", {"start_year": 1683, "end_year": 1738}, tolerance=2) == []
    assert dates_disagree("(c. 1684–1738)", {"start_year": 1683, "end_year": 1738})


# --- declared count --------------------------------------------------------

def test_a_step_that_changes_more_than_it_declared_raises():
    """A commit promising six changes touched 3,963 files."""

    with pytest.raises(DeclaredCountError, match="declared 6 changes, counted 3963"):
        with expect(changed=6) as tally:
            for i in range(3963):
                tally.change(i)


def test_a_step_that_changes_what_it_declared_passes():
    with expect(changed=2) as tally:
        tally.change("a")
        tally.change("b")
    assert tally.keys == ["a", "b"]


def test_changes_are_deduplicated_by_key():
    with expect(changed=1) as tally:
        tally.change("a")
        tally.change("a")


def test_tolerance_widens_the_bound_and_still_has_one():
    with expect(changed=10, tolerance=2) as tally:
        for i in range(12):
            tally.change(i)
    with pytest.raises(DeclaredCountError, match="declared 8–12"):
        with expect(changed=10, tolerance=2) as tally:
            for i in range(13):
                tally.change(i)


def test_an_exception_inside_the_block_propagates_untouched():
    with pytest.raises(ValueError):
        with expect(changed=99) as tally:
            tally.change("a")
            raise ValueError("the step failed")


def test_override_is_explicit():
    declared_count(3963, 6, override=True)
    with pytest.raises(DeclaredCountError):
        declared_count(3963, 6)


def test_the_write_after_the_block_is_unreachable_on_a_mismatch():
    written = []
    with pytest.raises(DeclaredCountError):
        with expect(changed=1) as tally:
            tally.change("a")
            tally.change("b")
        written.append("committed")
    assert written == []
