"""Composable data validation framework.

Validation rules are composable functions that check entities and produce
errors or warnings. Rules are grouped into a Validator that runs them all
against a dataset and returns a ValidationResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Aggregated validation output."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no errors (warnings are acceptable)."""
        return len(self.errors) == 0

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def summary(self) -> str:
        """One-line summary string."""
        return f"{len(self.errors)} errors, {len(self.warnings)} warnings"


# RuleCheckFn: (entity_type, entity_id, entity_data, all_entities) -> list[str]
# where all_entities is dict[entity_type -> dict[entity_id -> data]]
RuleCheckFn = Callable[[str, str, dict[str, Any], dict[str, dict[str, Any]]], list[str]]


@dataclass
class Rule:
    """A single validation rule."""
    name: str
    check_fn: RuleCheckFn
    entity_type: str | None = None  # None means applies to all types
    severity: str = "error"  # error | warning


class Validator:
    """Runs a set of rules against a complete dataset."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = list(rules)

    def validate(self, entities: dict[str, dict[str, Any]]) -> ValidationResult:
        """Validate all entities. entities is {type: {id: data}}."""
        result = ValidationResult()
        for rule in self.rules:
            for etype, etype_entities in entities.items():
                if rule.entity_type is not None and rule.entity_type != etype:
                    continue
                for eid, edata in etype_entities.items():
                    messages = rule.check_fn(etype, eid, edata, entities)
                    for msg in messages:
                        full_msg = f"{etype}/{eid}: {msg}"
                        if rule.severity == "warning":
                            result.warnings.append(full_msg)
                        else:
                            result.errors.append(full_msg)
        return result


# ---------------------------------------------------------------------------
# Built-in rule constructors
# ---------------------------------------------------------------------------

def required_field(entity_type: str, field_name: str, *, severity: str = "error") -> Rule:
    """Field must be present and non-empty."""
    def check(etype: str, eid: str, data: dict, all_entities: dict) -> list[str]:
        val = data.get(field_name)
        if val is None or val == "":
            return [f"missing required field '{field_name}'"]
        return []
    return Rule(name=f"required_{entity_type}_{field_name}", check_fn=check,
                entity_type=entity_type, severity=severity)


def valid_values(entity_type: str, field_name: str, allowed: set[str], *, severity: str = "error") -> Rule:
    """Field value must be in the allowed set (if present)."""
    def check(etype: str, eid: str, data: dict, all_entities: dict) -> list[str]:
        val = data.get(field_name)
        if val is not None and val not in allowed:
            return [f"'{field_name}' value '{val}' not in {sorted(allowed)}"]
        return []
    return Rule(name=f"valid_{entity_type}_{field_name}", check_fn=check,
                entity_type=entity_type, severity=severity)


def reference_exists(
    source_type: str,
    field_name: str,
    target_type: str,
    *,
    sub_field: str | None = None,
    severity: str = "error",
) -> Rule:
    """Referenced entity must exist in the dataset.

    Handles scalar, flat list, and nested dict-list fields:
      - reference_exists('work', 'author_id', 'person') — scalar
      - reference_exists('work', 'author_ids', 'person') — flat list
      - reference_exists('perf', 'credits', 'person', sub_field='person_id') — nested
    """
    def check(etype: str, eid: str, data: dict, all_entities: dict) -> list[str]:
        ref_val = data.get(field_name)
        if ref_val is None:
            return []
        target_entities = all_entities.get(target_type, {})
        if isinstance(ref_val, list):
            if sub_field is not None:
                # Nested dict-list: credits[].person_id
                ref_ids = [
                    str(item.get(sub_field))
                    for item in ref_val
                    if isinstance(item, dict) and item.get(sub_field) is not None
                ]
            else:
                ref_ids = [str(r) for r in ref_val]
            missing = [r for r in ref_ids if r not in target_entities]
            if missing:
                return [f"{field_name} references {target_type}/{m} which does not exist"
                        for m in missing]
            return []
        if str(ref_val) not in target_entities:
            return [f"{field_name} references {target_type}/{ref_val} which does not exist"]
        return []
    return Rule(name=f"ref_{source_type}_{field_name}", check_fn=check,
                entity_type=source_type, severity=severity)


def no_orphans(
    entity_type: str,
    referenced_by: list[tuple],
    *,
    severity: str = "warning",
) -> Rule:
    """Entity must be referenced by at least one entity of the given types.

    referenced_by is a list of tuples:
      - (source_type, field_name) for scalar or flat-list references
      - (source_type, field_name, sub_field) for nested dict-list references
        e.g. ('performance', 'credits', 'person_id') matches credits[].person_id
    """
    def _refs_match(ref_val, eid_str: str, sub_field: str | None) -> bool:
        if ref_val is None:
            return False
        if sub_field is not None:
            if isinstance(ref_val, list):
                return any(
                    isinstance(item, dict) and str(item.get(sub_field)) == eid_str
                    for item in ref_val
                )
            return False
        if isinstance(ref_val, list):
            return any(str(v) == eid_str for v in ref_val)
        return str(ref_val) == eid_str

    def check(etype: str, eid: str, data: dict, all_entities: dict) -> list[str]:
        eid_str = str(eid)
        for ref_tuple in referenced_by:
            src_type, src_field = ref_tuple[0], ref_tuple[1]
            sub_field = ref_tuple[2] if len(ref_tuple) > 2 else None
            src_entities = all_entities.get(src_type, {})
            for src_data in src_entities.values():
                if _refs_match(src_data.get(src_field), eid_str, sub_field):
                    return []
        return [f"orphan: not referenced by any {', '.join(t[0] for t in referenced_by)}"]
    return Rule(name=f"orphan_{entity_type}", check_fn=check,
                entity_type=entity_type, severity=severity)


def conditional_required(
    entity_type: str,
    condition_fn: Callable[[dict[str, Any]], bool],
    field_name: str,
    *,
    condition_label: str = "condition met",
    severity: str = "error",
) -> Rule:
    """Field required only when condition_fn(data) returns True."""
    def check(etype: str, eid: str, data: dict, all_entities: dict) -> list[str]:
        if condition_fn(data):
            val = data.get(field_name)
            if val is None or val == "":
                return [f"missing '{field_name}' (required when {condition_label})"]
        return []
    return Rule(name=f"cond_req_{entity_type}_{field_name}", check_fn=check,
                entity_type=entity_type, severity=severity)
