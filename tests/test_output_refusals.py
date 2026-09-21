"""Refusals for packet output that is prose, and therefore has no schema."""

import re

import pytest

from limbic.cerebellum.packet import (
    meta_leak_refusals,
    rendering_fidelity_refusals,
    slot_echo_refusal,
)


SOURCE = "Erik Gustaf Bostrom was Sweden's prime minister in 1905."


# --- slot echo / shortness -------------------------------------------------

def test_the_shipped_defect_is_refused():
    """Ten records went into a published graph defined as "i01"."""

    for slot in ("i01", "i02", "i10"):
        assert slot_echo_refusal(slot, ["i01", "i02"]) == f"output is a slot id, not a sentence: {slot}"


def test_a_slot_id_shape_from_a_packet_you_did_not_pass_is_still_refused():
    assert slot_echo_refusal("q03") is not None
    assert slot_echo_refusal("r7.") is not None


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("", "empty output"),
        ("   ", "empty output"),
        ("Kort", "too short to be a sentence"),
        ("A B", "too short to be a sentence"),
        ("SVERIGES STATSMINISTER 1905", "no lowercase word"),
    ],
)
def test_output_that_is_not_prose_is_refused(text, fragment):
    assert fragment in slot_echo_refusal(text)


def test_a_rendering_far_shorter_than_its_source_is_refused():
    assert "under 0.4 of its source" in slot_echo_refusal("Han var statsminister.", source=SOURCE)


def test_prose_is_not_refused():
    assert slot_echo_refusal(
        "Erik Gustaf Boström var Sveriges statsminister i 1905.", ["i01"], source=SOURCE
    ) is None


def test_the_source_ratio_is_skipped_when_there_is_no_source():
    assert slot_echo_refusal("Han var statsminister.") is None


# --- meta leak -------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "A person identified only as the source of the quotation.",
        "A Norwegian bishop, as stated in the label.",
        "The king named in the supplied citation.",
    ],
)
def test_text_about_the_evidence_is_refused(text):
    assert "text describes the evidence rather than the subject" in meta_leak_refusals(text)


@pytest.mark.parametrize(
    "text",
    [
        "A person named Kristoffer Visted.",
        "A work titled «Terje Vigen»",
        "The ballad called Terje Vigen.",
        "A Norwegian poet named Sigbjørn Obstfelder",
    ],
)
def test_text_whose_whole_content_is_the_name_is_refused(text):
    assert meta_leak_refusals(text) == ["text says only that the subject has its name"]


@pytest.mark.parametrize(
    "text",
    [
        # Opens with the formula and then says something. The unanchored
        # version refused this, which is how a guard loses its welcome.
        "A ballad titled Terje Vigen describes a sailor's ordeal in a storm.",
        "A person named Kristoffer Visted who collected Norwegian folk costume.",
        "The city called Nidaros was the seat of the archbishop.",
    ],
)
def test_a_definition_that_only_starts_with_the_formula_stands(text):
    assert meta_leak_refusals(text) == []


def test_a_real_definition_is_not_refused():
    assert meta_leak_refusals(SOURCE) == []


def test_the_phrase_list_is_replaceable():
    """A phrase list is a corpus's vocabulary; one lifted from elsewhere misses."""

    nb = re.compile(r"nevnt i (etiketten|kilden)", re.I)
    assert meta_leak_refusals("En biskop, nevnt i kilden.") == []
    assert meta_leak_refusals("En biskop, nevnt i kilden.", phrases=nb, vacuous=None) == [
        "text describes the evidence rather than the subject"
    ]


# --- rendering fidelity ----------------------------------------------------

def test_a_year_not_in_the_source_is_refused():
    out = rendering_fidelity_refusals(SOURCE, "Erik Gustaf Bostrom var statsminister fra 1902 til 1905.")
    assert out == ["year(s) not in the source: 1902"]


def test_a_name_not_in_the_source_is_refused():
    out = rendering_fidelity_refusals(SOURCE, "Erik Gustaf Bostrom var statsminister under Oscar i 1905.")
    assert out == ["name(s) not in the source: Oscar"]


def test_a_name_supplied_as_known_is_not_refused():
    assert rendering_fidelity_refusals(
        SOURCE, "Erik Gustaf Bostrom var statsminister under Oscar i 1905.",
        known_names=["Oscar II"],
    ) == []


def test_an_exonym_is_not_refused():
    assert rendering_fidelity_refusals(
        "He was prime minister of Sweden in 1905.",
        "Han var statsminister i Sverige i 1905.", exonyms=["Sverige"],
    ) == []


def test_a_rendering_far_longer_than_its_source_is_refused():
    out = rendering_fidelity_refusals("Prime minister.", "Han var " + "statsminister og politiker " * 6)
    assert out[0] == "rendering is more than 2 times the length of its source"


def test_an_empty_rendering_is_refused():
    assert rendering_fidelity_refusals(SOURCE, "") == ["empty rendering"]


def test_a_faithful_rendering_is_not_refused():
    assert rendering_fidelity_refusals(SOURCE, "Erik Gustaf Bostrom var statsminister i 1905.") == []


def test_the_target_languages_own_form_of_a_name_needs_an_exonym():
    """"Sweden's" rendered as "Sveriges" is a new name to anything language-neutral."""

    rendering = "Erik Gustaf Bostrom var Sveriges statsminister i 1905."
    assert rendering_fidelity_refusals(SOURCE, rendering) == ["name(s) not in the source: Sveriges"]
    assert rendering_fidelity_refusals(SOURCE, rendering, exonyms=["Sverige"]) == []


def test_exempt_years_lets_a_language_spell_a_century_as_a_number():
    """Bokmål writes "the twelfth century" as "1100-tallet"."""

    source = "A bishop of the twelfth century."
    rendering = "En biskop fra 1100-tallet."
    assert rendering_fidelity_refusals(source, rendering) == ["year(s) not in the source: 1100"]
    century = re.compile(r"(?<!\d)(\d{4})(?=-(?:tallet|årene))")
    assert rendering_fidelity_refusals(
        source, rendering,
        exempt_years=lambda src, text: century.findall(text) if "century" in src else [],
    ) == []


def test_the_default_lets_a_known_stem_carry_its_compound():
    """Norwegian's "Perth-traktaten" is the treaty of Perth, not a new name."""

    assert rendering_fidelity_refusals(
        "The treaty of Perth ended the dispute.", "Da ble Perth-traktaten inngatt."
    ) == []


def test_parts_splits_a_compound_so_each_element_is_checked_separately():
    source = "The union between Sweden and Norway ended in 1905."
    rendering = "I 1905 ble Union-Frankrike opplost."
    assert rendering_fidelity_refusals(source, rendering) == []
    assert rendering_fidelity_refusals(
        source, rendering, parts=lambda word: re.split(r"[-–—]", word)
    ) == ["name(s) not in the source: Union-Frankrike"]


def test_stem_matching_accepts_an_inflected_form():
    source = "The Union was dissolved."
    assert rendering_fidelity_refusals(source, "Unionen ble opplost i den perioden.") == []
