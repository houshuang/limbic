# `hippocampus.refuse` — guards that run before the write

Four guards, one shape: each refuses a specific bad state at the moment it is
still cheap to be wrong about, rather than describing it afterwards.

| Function | Refuses |
|---|---|
| `schema_refusals(records, schema, *, only=…)` | records this apply would write that the schema rejects |
| `temporal_plausibility_refusals(extents, *, living_year=…)` | a death no source can confirm, a century-long life claimed to the year |
| `dates_disagree(prose, extent)` | a record whose prose contradicts its own structured dates |
| `expect(changed=N)` / `declared_count(...)` | a step that changed a different number of things than it declared |

## `schema_refusals`

```python
refusals = schema_refusals(records, schema, only=lambda r: r["id"] in touched)
if refusals:
    raise SystemExit("\n".join(refusals))
```

**Call it from the same place in the dry run and the real run.** The defect
this exists for is a check that ran on `--execute` only: the dry run reported a
clean apply, the real one wrote records the gate then refused, and by the time
the gate said so the graph, the census and the site had been rebuilt from them.

There is deliberately no argument that can express "skip on a dry run". `only`
filters *records*, not runs — it selects the ones this apply actually touches,
so a store already failing for some older reason is not blamed on this write.

`extra_checks` are `(record) -> [reason]` callables run alongside the schema,
so the semantic guards below come back through the same call:

```python
schema_refusals(
    records, schema,
    only=lambda r: r["id"] in touched,
    extra_checks=[
        lambda r: temporal_plausibility_refusals(
            r.get("temporal_extents"), living_year=1990,
            statuses=("two-read-agreement",)) if r["type"] == "person" else [],
        lambda r: dates_disagree(r.get("definition"), (r.get("temporal_extents") or [{}])[0]),
        lambda r: meta_leak_refusals(r.get("definition"), phrases=MY_CORPUS_PHRASES),
    ],
)
```

### The schema validator

`json_schema_refusals` is a stdlib-only subset of JSON Schema 2020-12 —
`type`, `enum`, `const`, `required`, `properties`, `additionalProperties`,
`items`, `pattern`, `min`/`maxLength`, `min`/`maxItems`, `uniqueItems`,
`minimum`/`maximum` (incl. exclusive), `multipleOf`, `format` (date, date-time,
email, uri), `allOf`/`anyOf`/`oneOf`/`not`, `if`/`then`/`else`, and local
`$ref`. Its messages echo the `jsonschema` library's phrasing.

It **raises `SchemaSupportError`** on a keyword it does not implement, rather
than passing the record. A validator that silently skips the keyword that would
have caught the defect is worse than no validator, because it reports a clean
run.

limbic takes no dependency for this. `validate=` accepts any
`(record, schema) -> [message]` callable, and `jsonschema_backed(schema)`
returns one built on the real library if a project has it installed.

**Fit check.** Against skard's `evidence-spine.schema.json` and its 682 live
graph concepts: 0 refusals, matching the gate those concepts pass today; the
schema's full keyword set is supported, so nothing was skipped. Four
one-field mutations (drop `id`, drop `type`, an out-of-enum type, an id with
spaces) were refused on 200/200 concepts each. Exact message-text equality with
`jsonschema` could not be measured — it is not installed on this machine and
skard's own `concept_schema_refusals` raises without it.

## `temporal_plausibility_refusals`

```python
temporal_plausibility_refusals(
    extents, living_year=1990, max_exact_age=100,
    statuses=("two-read-agreement", "independent-audit-corrected"),
)
```

Two blind reads agreed that a man alive in 2026 had died in 2024. A death
inside living memory is the one date a corpus of older sources cannot settle,
and agreement is not evidence when both reads are guessing. A life longer than
a century claimed **to the exact year** is two people or an error; claimed
circa, it may be a tradition, and that is what "circa" is for — so the age
check fires only at `exact_precision`.

`living_year` is a required argument and ships as no constant anywhere. It is a
statement about now and goes stale by definition.

`statuses` limits the guard to the extents your own pipeline agreed on, leaving
authority-sourced dates alone — those were checked by something that can check
them. Filter to people before calling; a work published in 2024 is fine.

**Fit check.** Against skard's `extent_refusals` over its 682 live concepts:
identical output on 682/682, 0 refusals on both sides. What stayed in the
project: the `node_type == "person"` filter (a caller-side filter here) and the
status vocabulary.

## `dates_disagree`

```python
dates_disagree("Bishop of Oslo, 1671–1733, who…", {"start_year": 1671, "end_year": 1731})
# ["prose says the period ends 1733, the record says 1731"]
```

`peter-kolbjornsen` carried a definition reading 1687–1737 and structured
dates reading 1683–1738, and shipped. The meta-leak guard read definitions; the
plausibility guard read extents; neither ever read one against the other, so a
record could disagree with itself indefinitely.

**This one is new — no reference implementation existed in either project.**

It fires only on a *fenced* range — `(1687–1737)`, `, 1671–1733,` — or an
explicit birth/death pair (`b. 1687 … d. 1737`). That is not taste, it is the
measurement: matching any year range at all fired on **74 of 580** real
(definition, extent) pairs, and almost every one was reporting that a reign is
not a life — "president 1861–1865" beside a life of 1809–1865. A guard at that
rate is a guard people switch off. Fencing separates the biographical aside
from the office.

**Fit check** (new function, so: what it finds on real data). 580
(definition, extent) pairs from skard's live graph, 4 flagged:

| Record | Verdict |
|---|---|
| `peter-kolbjornsen` | **real** — the motivating case, 1687–1737 vs 1683–1738 |
| `bartholomeus-deichman` | **real** — 1671–1733 in prose vs 1671–1731 stored |
| `krohgstotten` | false positive — the parenthetical is the *commemorated* person's dates, not the monument's |
| `aleksios-i-komnenos` | false positive of the same kind |

2 real finds and 2 false positives in 580 is a rate worth reading; `tolerance`
and `range_pattern` are there to tune it, and `ANY_YEAR_RANGE` restores the
loose behaviour for a corpus whose definitions only ever state lifespans.

## `expect` / `declared_count`

```python
with expect(changed=6) as tally:
    for record in records:
        if needs_change(record):
            tally.change(record["id"])
commit(tally.keys)          # never reached when the count is off
```

A commit promising six changes touched 3,963 files. A complaint about two links
produced a mass category removal. In both cases the expected number was known
before the write and compared with nothing.

Decide inside the block, write after it: the check runs on exit, so a mismatch
raises *before* the write. An exception raised inside the block propagates
untouched — a step that failed has no count to answer for. `tolerance` widens
the bound and still leaves one; `override=True` is the deliberate escape hatch,
and it is a keyword you have to type at the call site where a reviewer reads it.

This closes checklist item 2.7, which until now was prose in three documents
and code in none.
