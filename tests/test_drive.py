"""Regression tests for the calibration-first Drive policy."""

from copy import deepcopy

from limbic.drive import check_calibrations, load_calibration_cases, validate_plan


def _valid_plan():
    return deepcopy(load_calibration_cases()[0]["expected_plan"])


def test_historical_calibrations_pass():
    results = check_calibrations()
    assert [result.case_id for result in results] == [
        "nrk-mobile-tv-improve",
        "otak-deep-research",
        "codex-claude-workflow-research",
    ]
    assert all(result.passed for result in results)


def test_rejects_batch_before_accepted_pilot():
    plan = _valid_plan()
    plan["pilot"]["unit_count"] = 4
    plan["scale"]["allowed"] = True
    errors = validate_plan(plan)
    assert "pilot.unit_count must be an integer from 1 to 3" in errors
    assert "scale.allowed must be false" in errors


def test_rejects_workers_and_recursive_delegation_in_v0():
    plan = _valid_plan()
    plan["budget"] = {
        "max_workers": 8,
        "max_additional_model_calls": 20,
        "max_additional_premium_calls": 4,
    }
    plan["delegation"] = {"enabled": True, "workers_may_delegate": True}
    errors = validate_plan(plan)
    assert "budget.max_workers must be 0" in errors
    assert "budget.max_additional_model_calls must be 0" in errors
    assert "budget.max_additional_premium_calls must be 0" in errors
    assert "delegation.enabled must be false" in errors
    assert "delegation.workers_may_delegate must be false" in errors


def test_requires_local_precedent_evidence_when_found():
    plan = _valid_plan()
    plan["precedent_search"]["inspected"] = []
    plan["precedents"] = []
    errors = validate_plan(plan)
    assert "precedent_search.inspected must name evidence when result is 'found'" in errors
    assert "precedents must contain a lesson when nearby evidence was found" in errors


def test_allows_one_material_question_but_not_an_interview():
    plan = _valid_plan()
    plan["open_questions"] = ["Which audience matters first?", "Which platform matters first?"]
    assert "open_questions must contain at most 1 item(s)" in validate_plan(plan)
