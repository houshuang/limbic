"""An independent audit can take a decision back. It can never make one.

On 21 September 2026 a second model, of a different family, read ~1,300
dry-apply decisions blind and wrote its findings to a file. 391 of them became
holds. Nothing was promoted, nothing was added, and the pipeline's own
reconciliation ran first and untouched — the audit folded in afterwards, over
decisions that had already been settled.

That asymmetry is the whole mechanism. An auditor that can promote is a second
proposer with a nicer name: its agreement stops being evidence, because you
only ever hear from it when it agrees. An auditor that can only demote is
evidence, because every row it writes costs you something.

`apply_audit` is the fold-in. It is demote-only by construction: the only
disposition it ever writes is the hold value, and it never appends a record.
A correction (a definition rewritten, a date fixed) is the one thing it may
change in place, and only if the caller's validator accepts it — an auditor
writing prose is still a model writing prose.

Usage:

    from limbic.hippocampus.audit import apply_audit

    decisions, report = apply_audit(
        decisions,                                  # AFTER reconciliation
        json.loads(Path("independent-audit.json").read_text()),
        key_fields=("citation_key", "concept_id"),
        hold_sections=("citations", "concepts"),
        correction_sections=(("definitions", "definition"),),
        validator=lambda field, value, row: (
            None if len(value) > 20 else "correction is too short to be a definition"
        ),
    )
    print(report["demoted"], report["unknown_ids"], report["source_errors"])

An id in the audit that matches no decision is reported, never dropped. It
means the auditor read something this batch is not writing — a stale pack, a
renamed key, an auditor briefed on the wrong set — and every one of those is a
thing you want to see, not a row to skip past.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "AuditError",
    "apply_audit",
]

# (field_name, value, row) -> reason to refuse, or None to accept.
CorrectionValidator = Callable[[str, Any, Mapping[str, Any]], "str | None"]


class AuditError(Exception):
    """The audit cannot be folded in at all, as opposed to a row being refused."""


def _rows(section: Any, key_fields: Sequence[str]) -> Iterator[tuple[str, Any]]:
    """Every (id, payload) in an audit section, whichever shape it has.

    Two shapes are in the wild and both are reasonable: a list of rows each
    carrying its own id field, and a mapping from id to payload. Supporting
    one of them would just move the adapter into every caller.
    """

    if isinstance(section, Mapping):
        for key, value in section.items():
            yield str(key), value
        return
    if isinstance(section, (str, bytes)) or not isinstance(section, Iterable):
        raise AuditError(f"audit section is neither a list of rows nor a mapping: {type(section)!r}")
    for row in section:
        if not isinstance(row, Mapping):
            raise AuditError(f"audit row is not a mapping: {row!r}")
        for field in key_fields:
            if row.get(field):
                yield str(row[field]), row
                break
        else:
            raise AuditError(
                f"audit row carries none of {list(key_fields)}: {sorted(row)!r}"
            )


def _note(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        note = payload.get("note")
        return str(note) if note is not None else None
    return None


def _decision_key(decision: Mapping[str, Any], key_fields: Sequence[str]) -> list[str]:
    return [str(decision[f]) for f in key_fields if decision.get(f)]


def apply_audit(
    decisions: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    key_fields: Sequence[str] = ("id",),
    hold_sections: Sequence[str] = ("holds",),
    correction_sections: Sequence[tuple[str, str]] = (),
    finding_sections: Sequence[str] = ("source_errors",),
    disposition_field: str = "disposition",
    hold_value: str = "hold",
    validator: CorrectionValidator | None = None,
    note_field: str = "audit_note",
    reason_field: str = "hold_reasons",
    reason: str = "independent_audit",
    clear_fields: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fold an independent audit into a settled decision set, demote-only.

    `decisions` must already be reconciled: every row carries
    `disposition_field`, and a row that does not is a refusal, not a default.
    An audit read before reconciliation audits a draft, and the reconciliation
    that follows it silently overwrites what the audit decided.

    `hold_sections` name the audit sections whose rows demote a matching
    decision to `hold_value`. `correction_sections` are `(section, field)`
    pairs: the section's payload replaces `field` on the matching decision, if
    `validator` accepts it. `finding_sections` are kept in the report and never
    applied — `source_errors` is the canonical one: a wrong fact that is wrong
    *in the source*, which the pipeline reproduced faithfully. That is a
    finding about the corpus, not a defect in this batch, and folding it in as
    a correction would silently edit the source out from under the evidence.

    Returns `(decisions, report)`. The decisions are deep copies; the input is
    never mutated. Every returned disposition is either the one it came in
    with or `hold_value` — asserted before returning, because that guarantee is
    the entire reason this function exists.

    Sections present in `audit` that the caller did not name are reported under
    `ignored_sections`, so an auditor that invented a `promotions` section is
    visible rather than quietly dropped.
    """

    out = [copy.deepcopy(dict(d)) for d in decisions]
    for index, decision in enumerate(out):
        if disposition_field not in decision:
            raise AuditError(
                f"decision {index} has no {disposition_field!r}: an audit folds in after "
                "reconciliation, over decisions that have already been settled"
            )

    before = [d[disposition_field] for d in out]
    by_key: dict[str, list[dict[str, Any]]] = {}
    for decision in out:
        for key in _decision_key(decision, key_fields):
            by_key.setdefault(key, []).append(decision)

    report: dict[str, Any] = {
        "auditor": audit.get("auditor"),
        "date": audit.get("date"),
        "demoted": 0,
        "demoted_keys": [],
        "corrected": [],
        "corrections_refused": [],
        "unknown_ids": [],
        "already_held": [],
        "findings": {},
        "ignored_sections": [],
    }

    named = {*hold_sections, *(s for s, _ in correction_sections), *finding_sections}
    metadata = {"schema", "auditor", "date", "scope", "measured", "full_results", "version"}
    report["ignored_sections"] = sorted(k for k in audit if k not in named and k not in metadata)

    for section_name in hold_sections:
        for key, payload in _rows(audit.get(section_name) or [], key_fields):
            matched = by_key.get(key)
            if not matched:
                report["unknown_ids"].append({"section": section_name, "key": key})
                continue
            for decision in matched:
                if decision[disposition_field] == hold_value:
                    report["already_held"].append({"section": section_name, "key": key})
                    continue
                decision[disposition_field] = hold_value
                decision[note_field] = _note(payload)
                decision[reason_field] = [*(decision.get(reason_field) or []), reason]
                for field in clear_fields:
                    decision[field] = [] if isinstance(decision.get(field), list) else None
                report["demoted"] += 1
                report["demoted_keys"].append(key)

    for section_name, field in correction_sections:
        for key, payload in _rows(audit.get(section_name) or {}, key_fields):
            matched = by_key.get(key)
            if not matched:
                report["unknown_ids"].append({"section": section_name, "key": key})
                continue
            value = payload.get(field, payload) if isinstance(payload, Mapping) else payload
            refusal = validator(field, value, payload if isinstance(payload, Mapping) else {}) if validator else None
            if refusal:
                report["corrections_refused"].append(
                    {"section": section_name, "key": key, "field": field, "reason": refusal}
                )
                continue
            for decision in matched:
                report["corrected"].append(
                    {"key": key, "field": field, "from": decision.get(field), "to": value}
                )
                decision[field] = value

    for section_name in finding_sections:
        section = audit.get(section_name)
        if section:
            report["findings"][section_name] = copy.deepcopy(section)

    report["unknown_ids"].sort(key=lambda row: (row["section"], row["key"]))
    report["demoted_keys"] = sorted(set(report["demoted_keys"]))

    for old, decision in zip(before, out):
        new = decision[disposition_field]
        if new not in (old, hold_value):
            raise AuditError(
                f"audit moved a decision from {old!r} to {new!r}: an audit demotes or does nothing"
            )
    if len(out) != len(decisions):
        raise AuditError("audit changed the size of the decision set")
    return out, report
