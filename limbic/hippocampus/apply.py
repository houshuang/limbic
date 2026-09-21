"""The write boundary: the model proposes, this function writes.

The 20 Sep 2026 governance review traced every mechanism Kulturbase built
against every incident it had, and found one clean rule: **mechanisms that
refuse a specific bad state worked; mechanisms that describe a state never
refused anything.** Exact-preimage restore on apply is the strongest item in
that set — one further recovery the day after it shipped, none since.
`ProposalStore`, extracted from the same repo, has zero consumers and no
preimage check: it is the filing cabinet without the lock.

So this is ~90 lines of function, not a store, a directory layout or a status
lifecycle. It refuses on: a field outside the whitelist, an on-disk value that
is not what the proposer saw, or any validator that objects. It writes
atomically and emits its own receipt, so "sealed mutation record" stops being
a separate 191-line script.

Usage:

    from limbic.hippocampus.apply import MISSING, apply_proposal, wikidata_type_is

    receipt = apply_proposal(
        "data/works/et-dukkehjem.json",
        {"wikidata_id": "Q1194978"},
        preimage={"wikidata_id": MISSING},      # the proposer saw no such key
        allowed_fields={"wikidata_id", "year_written"},
        validators=[wikidata_type_is("work")],
        receipt=Path("receipts.jsonl"),
    )
    if not receipt["applied"]:
        print(receipt["reason"])

`MISSING` is not `None`. An absent key, an explicit null and a compiled
default are three different facts, and collapsing them is how a "no change"
proposal silently overwrites a value someone else wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Callable, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "MISSING",
    "apply_proposal",
    "enum_member",
    "regex",
    "wikidata_exists",
    "wikidata_type_is",
]

Validator = Callable[[str, Any], "str | None"]


class _Missing:
    """Absent key. Distinct from None, which is an explicit null."""

    _instance: "_Missing | None" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional extra: limbic[hippocampus]

        return yaml.safe_load(text) or {}
    return json.loads(text)


def _dump_atomic(path: Path, record: Mapping[str, Any]) -> None:
    """Write through a temp file in the same directory, then rename.

    A partial write is the failure mode this exists for: the 28 Aug incident
    cluster was interrupted applies whose preimages were no longer restorable.
    """
    if path.suffix in (".yaml", ".yml"):
        import yaml

        text = yaml.safe_dump(dict(record), allow_unicode=True, sort_keys=False)
    else:
        text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def apply_proposal(
    path_or_obj: str | Path | MutableMapping[str, Any],
    changes: Mapping[str, Any],
    *,
    preimage: Mapping[str, Any],
    allowed_fields: AbstractSet[str],
    validators: Sequence[Validator] = (),
    writer: Callable[[Mapping[str, Any]], None] | None = None,
    receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Apply `changes` to a record, or refuse and say why. Never partial.

    Args:
        path_or_obj: the record — a path to a JSON/YAML file, or a mutable
            mapping (a DB row you loaded yourself), in which case pass
            `writer` to persist it.
        changes: field -> new value.
        preimage: field -> the value the proposer saw, or `MISSING` for a key
            it saw as absent. Every changed field must appear here; a proposal
            with no preimage is exactly the unguarded write this refuses.
        allowed_fields: any other key in `changes` is refused. Prose written
            into a typed identifier field at confidence 0.88 with five sources
            is a real incident, not a hypothetical.
        validators: `(field, value) -> message | None`. A message refuses.
        writer: called with the whole updated record instead of writing a file
            or updating the mapping in place; the caller persists what it gets.
        receipt: path to append one JSON line per attempt, applied or not.

    Returns:
        A receipt dict: `applied`, `reason`, `target`, `fields`,
        `sha256_before`, `sha256_after` and `ts`. The receipt is returned even
        when refused, and written to `receipt` either way — a refusal you
        cannot count is a refusal you will argue about later.
    """
    is_path = isinstance(path_or_obj, (str, Path))
    path = Path(path_or_obj) if is_path else None
    if is_path:
        record: MutableMapping[str, Any] = _load(path)  # type: ignore[arg-type]
    else:
        record = path_or_obj  # type: ignore[assignment]

    target = str(path) if is_path else str(record.get("id", "<mapping>"))
    before_hash = _sha(record)
    result: dict[str, Any] = {
        "applied": False,
        "reason": "",
        "target": target,
        "fields": sorted(changes),
        "sha256_before": before_hash,
        "sha256_after": before_hash,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }

    reason = _refusal(record, changes, preimage=preimage,
                      allowed_fields=allowed_fields, validators=validators)
    if reason:
        result["reason"] = reason
        _append_receipt(receipt, result)
        return result

    updated = {**record, **changes}
    if writer is not None:
        writer(updated)
    elif is_path:
        _dump_atomic(path, updated)  # type: ignore[arg-type]
    else:
        record.update(changes)
    result["applied"] = True
    result["sha256_after"] = _sha(updated)
    _append_receipt(receipt, result)
    return result


def _refusal(
    record: Mapping[str, Any],
    changes: Mapping[str, Any],
    *,
    preimage: Mapping[str, Any],
    allowed_fields: AbstractSet[str],
    validators: Sequence[Validator],
) -> str:
    """The first reason to refuse, or "" to proceed. Checks are cheap-first."""
    if not changes:
        return "no changes proposed"
    outside = sorted(set(changes) - set(allowed_fields))
    if outside:
        return f"field(s) outside the whitelist: {', '.join(outside)}"
    for field_name in changes:
        if field_name not in preimage:
            return (f"no preimage for {field_name!r}; a proposal that does not "
                    "record what it saw cannot be checked against what is there")
        expected = preimage[field_name]
        actual = record[field_name] if field_name in record else MISSING
        if isinstance(expected, _Missing) != isinstance(actual, _Missing) or (
                not isinstance(expected, _Missing) and actual != expected):
            return (f"preimage mismatch on {field_name!r}: proposer saw "
                    f"{expected!r}, record holds {actual!r}")
    for field_name, value in changes.items():
        for validator in validators:
            message = validator(field_name, value)
            if message:
                return f"{field_name}: {message}"
    return ""


def _append_receipt(receipt: str | Path | None, result: Mapping[str, Any]) -> None:
    if receipt is None:
        return
    path = Path(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Validators people actually need
# ---------------------------------------------------------------------------

def enum_member(allowed: Iterable[Any], *, fields: AbstractSet[str] | None = None) -> Validator:
    """Refuse a value outside a controlled vocabulary."""
    allowed_set = set(allowed)

    def check(field_name: str, value: Any) -> str | None:
        if fields is not None and field_name not in fields:
            return None
        if value is None:
            return None
        if value not in allowed_set:
            return f"{value!r} is outside the controlled vocabulary"
        return None

    return check


def regex(pattern: str, *, fields: AbstractSet[str] | None = None) -> Validator:
    """Refuse a value that does not fully match `pattern`."""
    compiled = re.compile(pattern)

    def check(field_name: str, value: Any) -> str | None:
        if fields is not None and field_name not in fields:
            return None
        if value is None:
            return None
        if not isinstance(value, str) or not compiled.fullmatch(value):
            return f"{value!r} does not match {pattern!r}"
        return None

    return check


_QID = re.compile(r"Q[1-9]\d*")


def _client(client: Any):
    if client is not None:
        return client
    from limbic.amygdala.wikidata import WikidataClient

    return WikidataClient()


def wikidata_exists(*, client: Any = None, fields: AbstractSet[str] | None = None) -> Validator:
    """Refuse a QID that is malformed, deleted, or a redirect to nothing.

    Necessary and nowhere near sufficient — see `wikidata_type_is`.
    """
    def check(field_name: str, value: Any) -> str | None:
        if fields is not None and field_name not in fields:
            return None
        if value is None or value == "":
            return None
        if not isinstance(value, str) or not _QID.fullmatch(value):
            return f"{value!r} is not a QID"
        entity = _client(client).get(value)
        if entity is None:
            return f"{value} does not exist on Wikidata (deleted or never existed)"
        return None

    return check


def wikidata_type_is(
    expected: str | Iterable[str],
    *,
    client: Any = None,
    fields: AbstractSet[str] | None = None,
    property_id: str = "P31",
) -> Validator:
    """Refuse a QID whose `instance of` is not the kind of thing you asked for.

    This is the check an existence test cannot do, and the gap is not small:
    of 901 work QIDs in one catalogue audited on 20 Sep 2026, **198 pointed at
    something that was not that work** — other works, non-works, deleted items.
    Every one of them passed an existence check. *Et dukkehjem* resolved to
    Ramon Llull.

    `expected` is either a `hippocampus.wikidata_resolve.TYPE_HINT_P31` key
    ("person", "place", "work", ...) or an explicit iterable of allowed QIDs.
    Subclass chains are not walked, so an over-narrow allowlist refuses a
    legitimate value: widen the allowlist rather than dropping the check.
    """
    if isinstance(expected, str):
        from .wikidata_resolve import TYPE_HINT_P31

        allowed = set(TYPE_HINT_P31.get(expected) or ())
        if not allowed:
            raise ValueError(
                f"no P31 allowlist for type hint {expected!r}; pass an explicit "
                "iterable of QIDs instead of a name limbic does not know")
        label = expected
    else:
        allowed = {str(q) for q in expected}
        label = "/".join(sorted(allowed))

    def check(field_name: str, value: Any) -> str | None:
        if fields is not None and field_name not in fields:
            return None
        if value is None or value == "":
            return None
        if not isinstance(value, str) or not _QID.fullmatch(value):
            return f"{value!r} is not a QID"
        entity = _client(client).get(value)
        if entity is None:
            return f"{value} does not exist on Wikidata"
        instance_of = set(entity.claim_qids(property_id))
        if not instance_of:
            return f"{value} ({entity.label() or '?'}) has no {property_id} claim"
        if not (instance_of & allowed):
            return (f"{value} ({entity.label() or '?'}) is {property_id}="
                    f"{sorted(instance_of)}, not a {label}")
        return None

    return check
