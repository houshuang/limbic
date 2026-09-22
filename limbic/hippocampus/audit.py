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
    "blind_view",
    "bucket_by_verdict",
    "check_audit_coverage",
]

# (field_name, value, row) -> reason to refuse, or None to accept.
CorrectionValidator = Callable[[str, Any, Mapping[str, Any]], "str | None"]

# Fields the four Kulturbase campaigns of 21 September 2026 each stripped by
# hand before a record reached the auditor: the researcher's own verdict, in
# whatever name a given batch used for it, plus the confidence signals
# blind-audit.md's brief already says to withhold ("the pipeline's
# confidence, its checks_passed, its held/not-held status, or a previous
# auditor's notes"). Extend per campaign; a field this list does not name
# simply is not stripped.
DEFAULT_HIDDEN_FIELDS: tuple[str, ...] = (
    "disposition", "verdict", "my_verdict", "auditor_verdict",
    "final", "final_action", "final_reason",
    "score", "tier", "confidence", "checks_passed",
    "audit_note", "hold_reasons",
)

# blind-audit.md's brief template: right / wrong / cannot_tell, and nothing
# else. A campaign whose auditor needs a different vocabulary passes its own.
DEFAULT_VOCABULARY: tuple[str, ...] = ("right", "wrong", "cannot_tell")

# Sections `apply_audit` and `check_audit_coverage` never treat as rows to
# process: bookkeeping about the audit run itself, not about a record.
_METADATA_KEYS = {"schema", "auditor", "date", "scope", "measured", "full_results", "version"}


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


def blind_view(
    items: Sequence[Mapping[str, Any]],
    hide: Sequence[str] = DEFAULT_HIDDEN_FIELDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Strip the researcher's own verdict from records before they reach the auditor.

    blind-audit.md's brief exists because "a reader shown the answer grades
    the answer" -- but every one of four Kulturbase campaigns that ran a
    blind audit on 21 September 2026 built that stripped view by hand, with
    its own list of fields to drop. This is the shared version, so campaign
    five does not write it again.

    Returns `(view, hidden)`: the records with `hide` removed, in the same
    order and never mutated in place, and the sorted list of field names that
    were actually present on at least one record -- for the audit record, so
    a brief or a report can state what was withheld rather than leaving it
    implicit.

    `hide` names fields to drop if present; a field a record does not carry
    is simply not there, not an error. The default covers the shapes seen
    across those four campaigns (`my_verdict`, `tier`, `score`, ...) -- pass
    a narrower or wider tuple when a campaign's own fields differ.
    """

    hide = tuple(hide)
    hidden_seen: set[str] = set()
    view = []
    for item in items:
        row = dict(item)
        for field in hide:
            if field in row:
                hidden_seen.add(field)
                del row[field]
        view.append(row)
    return view, sorted(hidden_seen)


def bucket_by_verdict(
    rows: Iterable[Mapping[str, Any]],
    key_field: str,
    *,
    id_field: str = "id",
    verdict_field: str = "verdict",
    note_field: str = "reason",
    vocabulary: Sequence[str] = DEFAULT_VOCABULARY,
    right_value: str = "right",
) -> dict[str, list[dict[str, Any]]]:
    """Turn a raw `{id, verdict, reason}` auditor response into sections ready
    for `apply_audit`'s `hold_sections`, validating the vocabulary on the way.

    This is the step blind-audit.md's brief asks every campaign to do before
    the fold-in: bucket by verdict, keep only the non-`right` rows, and check
    that every verdict is one the brief actually offered. Nobody had written
    it, so c02 hand-rolled its own bucketing and quietly drifted the verdict
    vocabulary to `same_work`/`different`/`unsure` -- nothing caught it, and
    it shipped as "do not ship" only because someone happened to notice.

    A row whose `verdict_field` is not in `vocabulary` raises `AuditError`
    naming every offending row: an unbriefed vocabulary means either the
    auditor was not given this brief or drifted mid-run, and folding those
    rows in under an assumed meaning would be worse than refusing the whole
    response. A campaign whose auditor genuinely uses another vocabulary
    passes its own `vocabulary=` (and matching `right_value=`) here -- that
    mapping happens before the fold-in, never inside `apply_audit`, which
    stays agnostic to what a hold_section's name means.

    Rows whose verdict equals `right_value` are dropped: blind-audit.md's
    brief asks the auditor never to return them, and if one arrives anyway
    this is where it is discarded rather than mistakenly folded in as a hold.
    """

    sections: dict[str, list[dict[str, Any]]] = {}
    bad: list[tuple[Any, Any]] = []
    for row in rows:
        verdict = row.get(verdict_field)
        if verdict not in vocabulary:
            bad.append((row.get(id_field), verdict))
            continue
        if verdict == right_value:
            continue
        sections.setdefault(verdict, []).append(
            {key_field: row.get(id_field), "note": row.get(note_field)}
        )
    if bad:
        raise AuditError(
            f"{len(bad)} audit row(s) used a verdict outside {list(vocabulary)!r}: "
            f"{bad[:5]!r} (id, verdict). A campaign whose auditor needs a different "
            "vocabulary maps it onto this one, or passes vocabulary= here, before "
            "folding in -- a vocabulary the brief never offered was not validated by "
            "anyone."
        )
    return sections


def check_audit_coverage(
    sent_keys: Iterable[str],
    audit: Mapping[str, Any],
    *,
    key_fields: Sequence[str] = ("id",),
) -> dict[str, Any]:
    """Report which of the ids sent to the auditor never appear anywhere in
    its response.

    `apply_audit` already reports the opposite direction: a returned id that
    matches no decision (`unknown_ids`). This is the direction c02's own
    `reconcile.py` checked and nothing in the library did: an id that was
    sent and never came back at all, in any section. Missing is not the same
    as `right` -- a silent auditor row is blind-audit.md's convention for
    agreement, but a row dropped by a truncated batch or a briefed-on-the-
    wrong-set auditor looks identical from the data alone. This function does
    not guess which one happened; it only counts and surfaces the gap, so the
    caller decides rather than silently reading absence as agreement.

    Every section actually present in `audit` (other than the run's own
    metadata -- `auditor`, `date`, `scope`, ...) counts as accounting for an
    id, not only the ones the caller passed as `hold_sections`: a correction
    or a finding is still the auditor having said something about that id.
    """

    sent = {str(key) for key in sent_keys}
    seen: set[str] = set()
    for section_name, section in audit.items():
        if section_name in _METADATA_KEYS or not section:
            continue
        try:
            for key, _ in _rows(section, key_fields):
                seen.add(key)
        except AuditError:
            continue
    return {
        "sent": len(sent),
        "returned": len(sent & seen),
        "missing": sorted(sent - seen),
    }


def apply_audit(
    decisions: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    key_fields: Sequence[str] = ("id",),
    hold_sections: Sequence[str] = ("holds",),
    correction_sections: Sequence[tuple[str, str]] = (),
    finding_sections: Sequence[str] = ("source_errors",),
    identity_fields: Sequence[str] | None = None,
    disposition_field: str = "disposition",
    hold_value: str = "hold",
    validator: CorrectionValidator | None = None,
    note_field: str = "audit_note",
    reason_field: str = "hold_reasons",
    reason: str = "independent_audit",
    clear_fields: Sequence[str] = (),
    sent_keys: Iterable[str] | None = None,
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

    A correction may not target a field that identifies the record. `key_fields`
    and `disposition_field` are protected always, and `identity_fields` names
    any further ones (a `label`, a `source_id`). Re-keying is not a correction:
    it moves the decision onto a different record, so everything already
    demoted, matched or reported about it silently means something else. A
    caller who really wants to re-key does it outside the audit, where it reads
    as the migration it is.

    `sent_keys`, when given, is every key that was actually sent to the
    auditor (a `blind_view` call's own keys, typically). The report then
    carries `coverage`: which of those keys never showed up in *any* section
    of `audit`, not only `hold_sections` — a correction or a finding still
    counts as the auditor having judged that id. A missing key is left
    exactly as it was, never demoted on the strength of an absence; this only
    counts and surfaces the gap; see `check_audit_coverage` for what "showed
    up" means precisely. Omit `sent_keys` and `coverage` is left out of the
    report entirely, rather than reported empty, so a caller that checks
    `"coverage" in report` can tell whether this ran at all.

    This function is one key at a time, on purpose: it never looks across
    decisions to ask whether the *set* it is about to return is internally
    consistent (one canonical id per entity, no orphaned reference, no
    reintroduced duplicate). Those are group invariants, properties of the
    union of accepted decisions rather than of any single row, and they
    belong to the project's own validator, run once over the full post-fold-in
    state before the write — not here, and not by widening what one row's
    audit is allowed to see.
    """

    protected = set(identity_fields) if identity_fields is not None else set(key_fields)
    protected |= {*key_fields, disposition_field}
    for section_name, field in correction_sections:
        if field in protected:
            raise AuditError(
                f"correction section {section_name!r} targets {field!r}, which identifies "
                "the record or decides it: an audit corrects what a record says, never "
                "which record it is. Re-key outside the audit."
            )

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
    report["ignored_sections"] = sorted(k for k in audit if k not in named and k not in _METADATA_KEYS)

    if sent_keys is not None:
        report["coverage"] = check_audit_coverage(sent_keys, audit, key_fields=key_fields)

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
