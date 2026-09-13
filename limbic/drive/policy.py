"""Deterministic policy checks for a Drive direction card.

The model still supplies judgment: what the user means, which precedent matters,
and what a representative pilot is.  This module makes the expensive mistakes
mechanically difficult: a v0 plan cannot spawn workers, spend model calls, or
authorize a batch before the user has experienced one small pilot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping


SCHEMA_VERSION = "limbic-drive-plan-v0"
MODES = frozenset({"research", "improve"})
PRECEDENT_RESULTS = frozenset({"found", "none-found"})
VALIDATION_KINDS = frozenset({"human-use", "manual-test", "source-check"})


@dataclass(frozen=True)
class CalibrationResult:
    """The outcome of checking one historical calibration case."""

    case_id: str
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return {}
    return value


def _nonempty_text(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be non-empty text")


def _text_list(
    value: Any,
    path: str,
    errors: list[str],
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"{path} must be a list of non-empty strings")
        return []
    if len(value) < minimum:
        errors.append(f"{path} must contain at least {minimum} item(s)")
    if maximum is not None and len(value) > maximum:
        errors.append(f"{path} must contain at most {maximum} item(s)")
    return value


def _exact_int(value: Any, expected: int, path: str, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        errors.append(f"{path} must be {expected}")


def _exact_bool(value: Any, expected: bool, path: str, errors: list[str]) -> None:
    if value is not expected:
        errors.append(f"{path} must be {str(expected).lower()}")


def validate_plan(plan: Any) -> list[str]:
    """Return all policy violations in *plan*; an empty list means valid."""

    errors: list[str] = []
    root = _mapping(plan, "plan", errors)

    if root.get("schema") != SCHEMA_VERSION:
        errors.append(f"schema must be {SCHEMA_VERSION!r}")
    if root.get("mode") not in MODES:
        errors.append("mode must be 'research' or 'improve'")
    _exact_bool(root.get("dry_run"), True, "dry_run", errors)
    _nonempty_text(root.get("request_summary"), "request_summary", errors)
    _nonempty_text(root.get("desired_outcome"), "desired_outcome", errors)
    _text_list(root.get("definition_of_better"), "definition_of_better", errors, minimum=1)
    _text_list(root.get("assumptions"), "assumptions", errors)
    _text_list(root.get("non_goals"), "non_goals", errors)
    _text_list(root.get("open_questions"), "open_questions", errors, maximum=1)

    search = _mapping(root.get("precedent_search"), "precedent_search", errors)
    _text_list(search.get("queries"), "precedent_search.queries", errors, minimum=1, maximum=3)
    inspected = _text_list(
        search.get("inspected"), "precedent_search.inspected", errors, maximum=3
    )
    result = search.get("result")
    if result not in PRECEDENT_RESULTS:
        errors.append("precedent_search.result must be 'found' or 'none-found'")
    if result == "found" and not inspected:
        errors.append("precedent_search.inspected must name evidence when result is 'found'")

    precedents = root.get("precedents")
    if not isinstance(precedents, list):
        errors.append("precedents must be a list")
        precedents = []
    if len(precedents) > 3:
        errors.append("precedents must contain at most 3 items")
    if result == "found" and not precedents:
        errors.append("precedents must contain a lesson when nearby evidence was found")
    for index, precedent_value in enumerate(precedents):
        precedent = _mapping(precedent_value, f"precedents[{index}]", errors)
        _nonempty_text(precedent.get("source"), f"precedents[{index}].source", errors)
        _nonempty_text(precedent.get("lesson"), f"precedents[{index}].lesson", errors)

    pilot = _mapping(root.get("pilot"), "pilot", errors)
    _nonempty_text(pilot.get("description"), "pilot.description", errors)
    unit_count = pilot.get("unit_count")
    if not isinstance(unit_count, int) or isinstance(unit_count, bool) or not 1 <= unit_count <= 3:
        errors.append("pilot.unit_count must be an integer from 1 to 3")
    _nonempty_text(pilot.get("artifact"), "pilot.artifact", errors)
    if pilot.get("validation_kind") not in VALIDATION_KINDS:
        errors.append(
            "pilot.validation_kind must be 'human-use', 'manual-test', or 'source-check'"
        )
    _text_list(pilot.get("evidence"), "pilot.evidence", errors, minimum=1)
    _nonempty_text(pilot.get("user_checkpoint"), "pilot.user_checkpoint", errors)

    budget = _mapping(root.get("budget"), "budget", errors)
    _exact_int(budget.get("max_workers"), 0, "budget.max_workers", errors)
    _exact_int(
        budget.get("max_additional_model_calls"),
        0,
        "budget.max_additional_model_calls",
        errors,
    )
    _exact_int(
        budget.get("max_additional_premium_calls"),
        0,
        "budget.max_additional_premium_calls",
        errors,
    )

    delegation = _mapping(root.get("delegation"), "delegation", errors)
    _exact_bool(delegation.get("enabled"), False, "delegation.enabled", errors)
    _exact_bool(
        delegation.get("workers_may_delegate"),
        False,
        "delegation.workers_may_delegate",
        errors,
    )

    scale = _mapping(root.get("scale"), "scale", errors)
    _exact_bool(scale.get("allowed"), False, "scale.allowed", errors)
    _nonempty_text(scale.get("gate"), "scale.gate", errors)

    _text_list(root.get("stop_conditions"), "stop_conditions", errors, minimum=1)
    _nonempty_text(root.get("next_action"), "next_action", errors)
    return errors


def load_calibration_cases() -> list[dict[str, Any]]:
    """Load the historical regression cases distributed with Limbic."""

    path = files("limbic.drive").joinpath("calibration_cases.json")
    return json.loads(path.read_text(encoding="utf-8"))


def check_calibrations() -> list[CalibrationResult]:
    """Validate every expected plan and its case-specific behavioral contract."""

    results: list[CalibrationResult] = []
    for case in load_calibration_cases():
        plan = case.get("expected_plan")
        errors = validate_plan(plan)
        expected = case.get("expect", {})
        if isinstance(plan, Mapping):
            if plan.get("mode") != expected.get("mode"):
                errors.append(f"mode must calibrate to {expected.get('mode')!r}")
            pilot = plan.get("pilot", {})
            if isinstance(pilot, Mapping):
                if pilot.get("unit_count", 99) > expected.get("pilot_max", 3):
                    errors.append("pilot is larger than the case permits")
                if pilot.get("validation_kind") != expected.get("validation_kind"):
                    errors.append("pilot uses the wrong human-evidence loop")
            sources = [
                item.get("source", "")
                for item in plan.get("precedents", [])
                if isinstance(item, Mapping)
            ]
            needle = expected.get("precedent_contains", "")
            if needle and not any(needle in source for source in sources):
                errors.append(f"precedents must include evidence containing {needle!r}")
        results.append(CalibrationResult(case_id=case["id"], errors=tuple(errors)))
    return results
