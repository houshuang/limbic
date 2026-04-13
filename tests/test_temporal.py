"""Tests for temporal.py — date parsing + Allen interval relations + plausibility."""

import math

from limbic.amygdala.temporal import (
    DateRange,
    parse_date,
    before,
    after,
    during,
    overlaps,
    meets,
    equals,
    plausibility_score,
)


# --- parse_date ---


def test_plain_year():
    dr = parse_date("1095")
    assert dr == DateRange(start=1095, end=1095, original="1095")


def test_negative_year():
    dr = parse_date("-44")
    assert dr.start == -44 and dr.end == -44


def test_bc_suffix():
    dr = parse_date("44 BC")
    assert dr.start == -44 and dr.end == -44
    dr2 = parse_date("44 BCE")
    assert dr2.start == -44 and dr2.end == -44


def test_ad_suffix():
    dr = parse_date("1095 AD")
    assert dr.start == 1095 and dr.end == 1095


def test_range_dash():
    dr = parse_date("942-996")
    assert dr.start == 942 and dr.end == 996


def test_range_en_dash():
    dr = parse_date("942–996")
    assert dr.start == 942 and dr.end == 996


def test_range_slash():
    dr = parse_date("1095/1154")
    assert dr.start == 1095 and dr.end == 1154


def test_decade():
    dr = parse_date("940s")
    assert dr.start == 940 and dr.end == 949


def test_century_ad():
    # 10th century = 901 to 1000
    dr = parse_date("10th century")
    assert dr.start == 901 and dr.end == 1000


def test_century_ad_explicit():
    dr = parse_date("10th century AD")
    assert dr.start == 901 and dr.end == 1000


def test_century_bc():
    # 4th century BC = 400 BC to 301 BC
    dr = parse_date("4th century BC")
    assert dr.start == -400 and dr.end == -301


def test_century_bce():
    dr = parse_date("4th century BCE")
    assert dr.start == -400 and dr.end == -301


def test_circa_variants():
    # Note: includes "c.942" (no space) and "≈942" (unicode) per the regex allowing zero whitespace
    for s in ["circa 942", "c. 942", "c.942", "c 942", "ca 942", "ca. 942", "~942", "≈942"]:
        dr = parse_date(s)
        assert dr is not None, f"failed to parse {s!r}"
        assert dr.start == 942 and dr.end == 942
        assert dr.approximate is True


def test_invalid_returns_none():
    assert parse_date("not a date") is None
    assert parse_date("") is None
    assert parse_date(None) is None


def test_whitespace_tolerated():
    dr = parse_date("  1095  ")
    assert dr.start == 1095


def test_first_century_boundary():
    # 1st century AD = year 1 to 100
    dr = parse_date("1st century")
    assert dr.start == 1 and dr.end == 100


# --- DateRange helpers ---


def test_daterange_contains():
    dr = DateRange(start=940, end=949)
    assert dr.contains(942)
    assert dr.contains(940)
    assert dr.contains(949)
    assert not dr.contains(939)
    assert not dr.contains(950)


def test_daterange_midpoint_and_span():
    dr = DateRange(start=940, end=949)
    assert dr.midpoint() == 944.5
    assert dr.span_years() == 10


# --- Allen relations ---


def test_before():
    rollo = DateRange(start=846, end=930)  # approx Rollo's life
    richard_i = DateRange(start=942, end=996)  # Richard I of Normandy
    assert before(rollo, richard_i)
    assert not before(richard_i, rollo)


def test_after():
    rollo = DateRange(start=846, end=930)
    richard_i = DateRange(start=942, end=996)
    assert after(richard_i, rollo)


def test_during():
    reign = DateRange(start=1002, end=1016)  # Æthelred's last reign period
    event = DateRange(start=1002, end=1002)  # St Brice's Day massacre
    assert during(event, reign)
    assert not during(reign, event)


def test_during_inclusive_boundary():
    outer = DateRange(start=900, end=1000)
    boundary = DateRange(start=900, end=1000)
    assert during(boundary, outer)  # equal ranges are "during"


def test_overlaps():
    emma = DateRange(start=985, end=1052)  # Emma of Normandy approx
    aethelred = DateRange(start=968, end=1016)
    assert overlaps(emma, aethelred)
    assert overlaps(aethelred, emma)


def test_no_overlap():
    a = DateRange(start=900, end=950)
    b = DateRange(start=951, end=1000)
    assert not overlaps(a, b)


def test_meets():
    a = DateRange(start=900, end=950)
    b = DateRange(start=951, end=1000)
    assert meets(a, b)
    assert not meets(b, a)


def test_meets_with_gap():
    a = DateRange(start=900, end=950)
    b = DateRange(start=960, end=1000)
    assert not meets(a, b)


def test_equals():
    a = DateRange(start=942, end=996)
    b = DateRange(start=942, end=996, original="Richard I")
    assert equals(a, b)


# --- plausibility_score ---


def test_plausibility_full_score_for_overlap():
    candidate = DateRange(start=942, end=996)  # Richard I of Normandy
    context = DateRange(start=900, end=1050)   # Viking Age / early Normans
    assert plausibility_score(candidate, context) == 1.0


def test_plausibility_decays_with_gap():
    candidate = DateRange(start=1864, end=1949)  # Richard Strauss
    context = DateRange(start=900, end=1050)     # Viking Age
    # Gap ~ 814 years, penalty_scale 100 → exp(-8.14) ≈ very small
    score = plausibility_score(candidate, context, penalty_scale_years=100)
    assert 0 < score < 0.001


def test_plausibility_moderate_penalty_for_near_era():
    # Machiavelli (1469-1527) reading about Caesar (died -44)
    # Caesar as candidate for a Machiavelli-era context:
    caesar = DateRange(start=-100, end=-44)
    machiavelli_context = DateRange(start=1469, end=1527)
    # Gap = 1469 - (-44) = 1513 years. Soft penalty keeps it nonzero.
    score = plausibility_score(caesar, machiavelli_context, penalty_scale_years=500)
    # exp(-1513/500) ≈ 0.048 — not zero, preserves cross-era reference
    assert 0 < score < 0.1


def test_plausibility_one_over_e_at_penalty_scale():
    a = DateRange(start=0, end=0)
    b = DateRange(start=100, end=100)
    # Gap is 100, scale is 100, score should be ~1/e
    score = plausibility_score(a, b, penalty_scale_years=100)
    assert math.isclose(score, 1 / math.e, rel_tol=0.01)


def test_plausibility_rejects_nonpositive_scale():
    a = DateRange(start=0, end=0)
    b = DateRange(start=100, end=100)
    try:
        plausibility_score(a, b, penalty_scale_years=0)
        raise AssertionError("should have raised")
    except ValueError:
        pass


# --- EDTF integration (optional dep) ---


def test_edtf_uncertain_decade_parses_when_available():
    """EDTF '19uu' means 'some year in the 1900s'. Only runs if edtf is installed."""
    import pytest
    pytest.importorskip("edtf")
    dr = parse_date("19uu")
    assert dr is not None
    assert dr.start == 1900
    assert dr.end == 1999


def test_edtf_approximate_year_parses_when_available():
    """EDTF '1942?' is uncertain-year syntax. Regex doesn't match it; EDTF must."""
    import pytest
    pytest.importorskip("edtf")
    dr = parse_date("1942?")
    assert dr is not None
    assert dr.start == 1942


def test_daterange_dataclass_equality_differs_from_allen_equals():
    """Document the intentional asymmetry: Allen equals ignores metadata; == does not."""
    a = DateRange(start=942, end=996)
    b = DateRange(start=942, end=996, original="Richard I", approximate=True)
    assert equals(a, b)  # Allen semantics: same temporal extent
    assert a != b  # dataclass equality: different metadata
