"""Guards that refuse a bad state *before* the write, not at the gate after it.

Four defects, one shape. Each was found late, by something downstream, after
the bad value had already been written and rebuilt from:

* An apply wrote records the project's own schema refuses — a person with no
  dates and no identifier — and it only surfaced at the gate, after the graph,
  the census and the site had been rebuilt from them. The check existed; the
  apply ran it on `--execute` only, so the dry run reported success.
* Two blind reads agreed that a man alive in 2026 had died in 2024, and that
  another had lived 112 years to the exact year. Both are unverifiable at best
  and two conflated people at worst, and nothing said so.
* A concept's prose definition said 1687–1737 while its own structured dates
  said 1683–1738. The definition guard read definitions, the extent guard read
  extents, and neither ever read one against the other.
* A commit promising six changes touched 3,963 files, because nothing compared
  what the step said it would do with what it did.

So: validate exactly what an apply would write, in the dry run and the real
run, with the same call; refuse a date no one can check; read prose against the
structured field it contradicts; and make a step declare its own blast radius
before it is allowed to commit.

Usage:

    from limbic.hippocampus.refuse import expect, schema_refusals

    refusals = schema_refusals(records, schema, only=lambda r: r["id"] in touched)
    if refusals:
        raise SystemExit("\\n".join(refusals))       # in dry AND real runs

    with expect(changed=6) as tally:
        for record in records:
            if needs_change(record):
                tally.change(record["id"])
    write(tally.keys)                               # unreachable if the count is off

No third-party dependency: the JSON Schema subset below is the one real
schemas in this ecosystem actually use. Pass `validate=` to swap in the real
`jsonschema` library (or anything else) when a schema outgrows it.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "DeclaredCountError",
    "SchemaSupportError",
    "Tally",
    "ANY_YEAR_RANGE",
    "LIFESPAN_RANGE",
    "dates_disagree",
    "declared_count",
    "expect",
    "json_schema_refusals",
    "jsonschema_backed",
    "schema_refusals",
    "temporal_plausibility_refusals",
]


# ---------------------------------------------------------------------------
# Pre-write schema refusal
# ---------------------------------------------------------------------------

class SchemaSupportError(Exception):
    """The schema uses a keyword this subset does not implement.

    Raised rather than ignored. A validator that silently skips the keyword
    that would have caught the defect is worse than no validator, because it
    reports a clean run.
    """


_SUPPORTED = {
    "$anchor", "$comment", "$defs", "$id", "$ref", "$schema",
    "additionalProperties", "allOf", "anyOf", "const", "default", "deprecated",
    "description", "else", "enum", "examples", "exclusiveMaximum",
    "exclusiveMinimum", "format", "if", "items", "maxItems", "maxLength",
    "maximum", "minItems", "minLength", "minimum", "multipleOf", "not",
    "oneOf", "pattern", "properties", "required", "then", "title", "type",
    "uniqueItems",
}

_TYPES: dict[str, Any] = {
    "object": Mapping,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}

_FORMATS = {
    "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "date-time": re.compile(r"^\d{4}-\d{2}-\d{2}[Tt ].+$"),
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "uri": re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:"),
}


def _is_type(value: Any, name: str) -> bool:
    expected = _TYPES.get(name)
    if expected is None:
        raise SchemaSupportError(f"unknown JSON Schema type {name!r}")
    if name in ("number", "integer") and isinstance(value, bool):
        return False  # a bool is an int in Python and is not a number in JSON Schema
    if name == "number" and isinstance(value, int):
        return True
    return isinstance(value, expected)


def _resolve(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise SchemaSupportError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            raise SchemaSupportError(f"$ref {ref!r} does not resolve")
        node = node[part]
    if not isinstance(node, Mapping):
        raise SchemaSupportError(f"$ref {ref!r} does not point at a schema")
    return node


def _valid(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    return not _errors(value, schema, root, "")


def _errors(
    value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str
) -> list[str]:
    """Every way `value` fails `schema`, as `path: message`.

    Messages deliberately echo the phrasing of the `jsonschema` library, so a
    project that swaps this subset for the real one does not have to re-learn
    its own error strings.
    """

    if isinstance(schema, bool):
        return [] if schema else f"{path}: {value!r} is not allowed".lstrip(": ").splitlines()
    unknown = set(schema) - _SUPPORTED
    if unknown:
        raise SchemaSupportError(
            f"schema at {path or '<root>'} uses unsupported keyword(s) "
            f"{sorted(unknown)}: pass validate=jsonschema_backed(schema) instead"
        )

    def at(message: str) -> str:
        return f"{path}: {message}" if path else message

    out: list[str] = []
    if "$ref" in schema:
        out += _errors(value, _resolve(schema["$ref"], root), root, path)

    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(value, name) for name in names):
            shown = names[0] if len(names) == 1 else names
            out.append(at(f"{value!r} is not of type {shown!r}"))
            return out  # every other keyword would repeat the same one fact

    if "enum" in schema and value not in schema["enum"]:
        out.append(at(f"{value!r} is not one of {schema['enum']!r}"))
    if "const" in schema and value != schema["const"]:
        out.append(at(f"{schema['const']!r} was expected"))

    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append(at(f"{value!r} does not match {schema['pattern']!r}"))
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(at(f"{value!r} is too short"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(at(f"{value!r} is too long"))
        fmt = schema.get("format")
        if fmt in _FORMATS and not _FORMATS[fmt].match(value):
            out.append(at(f"{value!r} is not a {fmt!r}"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(at(f"{value!r} is less than the minimum of {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            out.append(at(f"{value!r} is greater than the maximum of {schema['maximum']}"))
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            out.append(
                at(f"{value!r} is less than or equal to the exclusive minimum of "
                   f"{schema['exclusiveMinimum']}")
            )
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            out.append(
                at(f"{value!r} is greater than or equal to the exclusive maximum of "
                   f"{schema['exclusiveMaximum']}")
            )
        if "multipleOf" in schema and value % schema["multipleOf"] != 0:
            out.append(at(f"{value!r} is not a multiple of {schema['multipleOf']}"))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(at(f"{value!r} should be non-empty" if schema["minItems"] == 1
                          else f"{value!r} is too short"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(at(f"{value!r} is too long"))
        if schema.get("uniqueItems"):
            seen: list[Any] = []
            for item in value:
                if item in seen:
                    out.append(at(f"{value!r} has non-unique elements"))
                    break
                seen.append(item)
        if "items" in schema:
            for index, item in enumerate(value):
                out += _errors(item, schema["items"], root, f"{path}[{index}]" if path else f"[{index}]")

    if isinstance(value, Mapping):
        for name in schema.get("required", ()):
            if name not in value:
                out.append(at(f"{name!r} is a required property"))
        properties = schema.get("properties", {})
        for name, sub in properties.items():
            if name in value:
                out += _errors(value[name], sub, root, f"{path}.{name}" if path else name)
        extra = schema.get("additionalProperties")
        if extra is False:
            for name in value:
                if name not in properties:
                    out.append(at(f"Additional properties are not allowed ({name!r} was unexpected)"))
        elif isinstance(extra, Mapping):
            for name, item in value.items():
                if name not in properties:
                    out += _errors(item, extra, root, f"{path}.{name}" if path else name)

    for sub in schema.get("allOf", ()):
        out += _errors(value, sub, root, path)
    if "anyOf" in schema and not any(_valid(value, sub, root) for sub in schema["anyOf"]):
        out.append(at(f"{value!r} is not valid under any of the given schemas"))
    if "oneOf" in schema:
        matched = sum(1 for sub in schema["oneOf"] if _valid(value, sub, root))
        if matched == 0:
            out.append(at(f"{value!r} is not valid under any of the given schemas"))
        elif matched > 1:
            out.append(at(f"{value!r} is valid under each of the given schemas"))
    if "not" in schema and _valid(value, schema["not"], root):
        out.append(at(f"{value!r} should not be valid under {schema['not']!r}"))
    if "if" in schema:
        branch = "then" if _valid(value, schema["if"], root) else "else"
        if branch in schema:
            out += _errors(value, schema[branch], root, path)
    return out


def json_schema_refusals(
    record: Mapping[str, Any], schema: Mapping[str, Any]
) -> list[str]:
    """Every way `record` fails `schema`, using the stdlib-only subset.

    Raises `SchemaSupportError` on a keyword it does not implement, rather
    than passing the record. Supported: type, enum, const, required,
    properties, additionalProperties, items, pattern, min/maxLength,
    min/maxItems, uniqueItems, minimum/maximum (incl. exclusive), multipleOf,
    format (date, date-time, email, uri), allOf/anyOf/oneOf/not, if/then/else
    and local `$ref`.
    """

    root = schema
    return _errors(record, schema, root, "")


def jsonschema_backed(
    schema: Mapping[str, Any], *, format_checker: bool = True
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], list[str]]:
    """A `validate=` callable backed by the real `jsonschema`, if it is installed.

    Raises `ImportError` when it is not, rather than degrading quietly: a
    caller asking for this asked for it on purpose.
    """

    from jsonschema import Draft202012Validator, FormatChecker  # noqa: PLC0415

    validator = Draft202012Validator(
        dict(schema), format_checker=FormatChecker() if format_checker else None
    )

    def validate(record: Mapping[str, Any], _schema: Mapping[str, Any]) -> list[str]:
        errors = sorted(validator.iter_errors(dict(record)), key=lambda e: list(e.absolute_path))
        return [error.message for error in errors]

    return validate


def schema_refusals(
    records: Iterable[Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    only: Callable[[Mapping[str, Any]], bool] | None = None,
    id_field: str = "id",
    extra_checks: Sequence[Callable[[Mapping[str, Any]], Sequence[str]]] = (),
    validate: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[str]] | None = None,
) -> list[str]:
    """Every way the records this apply would write fail the schema it must meet.

    Call this in the dry run and the real run, from the same place, with the
    same arguments. The defect this exists for is a check that ran only on
    `--execute`: the dry run reported a clean apply and the real one wrote
    records the gate then refused, after everything downstream had been rebuilt.

    `only` selects the records this apply actually touches, so a store already
    failing for some older reason is not blamed on this write. It is a
    predicate over records — not over whether the run is real — on purpose:
    there is no argument to this function that can express "skip on a dry run".

    `extra_checks` are `(record) -> [reason]` callables run alongside the
    schema, so semantic refusals (`temporal_plausibility_refusals`,
    `dates_disagree`, a meta-leak phrase list) come back through one call.

    Returns `"<id>: <reason>"` strings, empty when the write is safe.
    """

    check = validate or json_schema_refusals
    out: list[str] = []
    for record in records:
        if only is not None and not only(record):
            continue
        name = record.get(id_field) or "<no id>"
        for message in check(record, schema):
            out.append(f"{name}: {message}")
        for extra in extra_checks:
            for message in extra(record):
                out.append(f"{name}: {message}")
    return out


# ---------------------------------------------------------------------------
# Temporal plausibility
# ---------------------------------------------------------------------------

def temporal_plausibility_refusals(
    extents: Iterable[Mapping[str, Any]] | None,
    *,
    living_year: int,
    max_exact_age: int = 100,
    statuses: Collection[str] | None = None,
    exact_precision: str = "exact-year",
    start_field: str = "start_year",
    end_field: str = "end_year",
    status_field: str = "status",
    precision_field: str = "precision",
) -> list[str]:
    """Refuse a life the pipeline agreed on with itself and cannot check.

    A death inside living memory is the one date a corpus of older sources
    cannot settle: two blind reads agreed that a man alive in 2026 died in
    2024, and agreement is not evidence when both reads are guessing. A life
    longer than a century claimed *to the exact year* is two people or an
    error; claimed circa, it may be a tradition, and that is what "circa" is
    for — so the age check fires only at `exact_precision`.

    `living_year` is a required argument and ships as no constant anywhere: it
    goes stale by definition, being a statement about now. Pass the year
    beyond which your sources stop being able to tell you someone died.

    `statuses` limits the guard to machine-derived extents (the ones your own
    pipeline agreed on). Leave it `None` to check every extent; pass the
    statuses your pipeline writes to leave authority-sourced dates alone,
    since those were checked by something that can check them. Filter to the
    record types this applies to — people — before calling.
    """

    out: list[str] = []
    for extent in extents or ():
        if statuses is not None and extent.get(status_field) not in statuses:
            continue
        start, end = extent.get(start_field), extent.get(end_field)
        if end is not None and end >= living_year:
            out.append(
                f"a death in {end} is inside living memory and cannot be checked here"
            )
        if (
            start is not None
            and end is not None
            and end - start > max_exact_age
            and extent.get(precision_field) == exact_precision
        ):
            out.append(
                f"an exact-year life of {end - start} years is two people or an error"
            )
    return out


# A year, not part of a longer number. Not anchored to a century form: that is
# the caller's language's problem (Norwegian spells the 1100s "1100-tallet").
YEAR = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
# «(1687–1737)», «, 1671–1733,» — a range fenced by a bracket or commas, which
# in a definition is an aside about the subject itself. Measured on 580 real
# (definition, extent) pairs: any-range matching fired on 74 of them and was
# mostly right about the years and wrong about what they were — «president
# 1861–1865» against a life of 1809–1865 is a guard nobody keeps switched on.
# Fencing is what separates the aside from the office.
LIFESPAN_RANGE = re.compile(
    r"[(\[,]\s*(?:c|ca|circa)?\.?\s*(\d{3,4})\s*[–—−-]\s*(?:c|ca|circa)?\.?\s*(\d{3,4})\s*[)\],.;]"
)
# Any range at all, including «reigned from 1492 to 1503». Available for a
# corpus whose definitions only ever state lifespans; noisy anywhere else.
ANY_YEAR_RANGE = re.compile(
    r"(?<!\d)(\d{3,4})\s*(?:–|—|-|−|\bto\b|\btil\b)\s*(\d{3,4})(?!\d)"
)
# «b. 1687 … d. 1737», «født i 1687 … død i 1737». Markers, in either order.
BIRTH_MARKER = re.compile(
    r"\b(?:b|born|f|født|fodt)\.?\s*(?:i|in)?\s*(?:c|ca)?\.?\s*(\d{3,4})(?!\d)", re.I
)
DEATH_MARKER = re.compile(
    r"\b(?:d|died|død|dod)\.?\s*(?:i|in)?\s*(?:c|ca)?\.?\s*(\d{3,4})(?!\d)", re.I
)


def dates_disagree(
    prose: str | None,
    extent: Mapping[str, Any] | None,
    *,
    start_field: str = "start_year",
    end_field: str = "end_year",
    tolerance: int = 0,
    range_pattern: re.Pattern[str] = LIFESPAN_RANGE,
    birth_pattern: re.Pattern[str] = BIRTH_MARKER,
    death_pattern: re.Pattern[str] = DEATH_MARKER,
) -> list[str]:
    """Refuse a record whose own prose contradicts its own structured dates.

    `peter-kolbjornsen` carried a definition reading 1687–1737 and a corrected
    `temporal_extents` reading 1683–1738, and shipped. The meta-leak guard
    read definitions; the plausibility guard read extents; neither ever read
    one against the other, so a record could disagree with itself indefinitely.

    Deliberately conservative. It fires only on a *fenced* range —
    `(1687–1737)`, `, 1671–1733,` — or an explicit birth/death pair
    (`b. 1687 … d. 1737`). A definition that mentions 1814 once is not making
    a claim about a lifespan; neither is "president 1861–1865" next to a life
    of 1809–1865. Matching any range at all fired on 74 of 580 real
    (definition, extent) pairs and was mostly reporting that a reign is not a
    life, which is a guard people switch off. Pass
    `range_pattern=ANY_YEAR_RANGE` for a corpus whose definitions only ever
    state lifespans.

    `tolerance` allows a stated year to differ by that many years without
    refusing, for corpora where a year is routinely given circa. Returns one
    reason per disagreeing endpoint.
    """

    text = (prose or "").strip()
    if not text or not extent:
        return []
    start, end = extent.get(start_field), extent.get(end_field)
    if start is None and end is None:
        return []

    pairs: list[tuple[int, int]] = [
        (int(a), int(b)) for a, b in range_pattern.findall(text) if int(b) >= int(a)
    ]
    births = [int(y) for y in birth_pattern.findall(text)]
    deaths = [int(y) for y in death_pattern.findall(text)]
    if births and deaths:
        pairs.append((births[0], deaths[0]))
    if not pairs:
        return []

    out: list[str] = []
    for stated_start, stated_end in pairs:
        if start is not None and abs(stated_start - start) > tolerance:
            out.append(
                f"prose says the period begins {stated_start}, the record says {start}"
            )
        if end is not None and abs(stated_end - end) > tolerance:
            out.append(
                f"prose says the period ends {stated_end}, the record says {end}"
            )
    return out


# ---------------------------------------------------------------------------
# Declared-count guard
# ---------------------------------------------------------------------------

class DeclaredCountError(Exception):
    """A step changed a different number of things than it said it would."""


def declared_count(
    actual: int,
    declared: int,
    *,
    tolerance: int = 0,
    override: bool = False,
    label: str = "changes",
) -> None:
    """Raise unless `actual` is within `tolerance` of `declared`.

    A commit promising six changes touched 3,963 files. A batch described as
    "a few corrections" rewrote a whole collection. In both cases the number
    was known before the write and compared with nothing.

    `override=True` is the deliberate escape hatch, and it is a keyword you
    have to type: a step that legitimately cannot know its own count says so
    at the call site, where a reviewer reads it.
    """

    if override:
        return
    low, high = declared - tolerance, declared + tolerance
    if not low <= actual <= high:
        bound = f"{declared}" if tolerance == 0 else f"{low}–{high}"
        raise DeclaredCountError(
            f"declared {bound} {label}, counted {actual}: refusing to write"
        )


class Tally:
    """The things a step is about to change, counted as it decides on them."""

    def __init__(self) -> None:
        self._keys: list[Any] = []
        self._seen: set[Any] = set()

    def change(self, key: Any = None) -> None:
        """Record one intended change, deduplicating by `key` when given."""

        if key is None:
            self._keys.append(object())
            return
        if key in self._seen:
            return
        self._seen.add(key)
        self._keys.append(key)

    @property
    def keys(self) -> list[Any]:
        return list(self._keys)

    @property
    def count(self) -> int:
        return len(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


@contextmanager
def expect(
    *,
    changed: int,
    tolerance: int = 0,
    override: bool = False,
    label: str = "changes",
) -> Iterator[Tally]:
    """Declare how many records a step will change; raise on the way out if it did not.

    Decide inside the block, write after it. The check runs on exit, so a
    mismatch raises *before* the write, which is the only moment at which the
    number is still cheap to be wrong about:

        with expect(changed=6) as tally:
            for record in records:
                if needs_change(record):
                    tally.change(record["id"])
        commit(tally.keys)          # never reached when the count is off

    An exception raised inside the block propagates untouched: a step that
    failed has no count to answer for.
    """

    tally = Tally()
    yield tally
    declared_count(tally.count, changed, tolerance=tolerance, override=override, label=label)
