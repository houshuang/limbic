# `limbic.hippocampus.apply` — the model proposes, code writes

One function at the write boundary. It refuses on a field outside the
whitelist, a changed field with no entry in the preimage, an on-disk value that
is not what the proposer saw, an empty set of changes, or any validator that
objects. It writes atomically and emits its own receipt.

The 20 Sep 2026 governance review traced every mechanism one project built
against every incident it had and found a single clean rule: **mechanisms that
refuse a specific bad state worked; mechanisms that describe a state never
refused anything.** Exact-preimage restore on apply is the strongest item in
that set — one further recovery the day after it shipped, none since.
`ProposalStore`, extracted from the same repo, has zero consumers and no
preimage check. It is the filing cabinet without the lock, which is why this is
a function and not a store.

## Ten lines

```python
from pathlib import Path
from limbic.hippocampus.apply import MISSING, apply_proposal, wikidata_type_is

receipt = apply_proposal(
    "data/works/et-dukkehjem.json",
    {"wikidata_id": "Q1194978"},
    preimage={"wikidata_id": MISSING},          # the proposer saw no such key
    allowed_fields={"wikidata_id", "year_written"},
    validators=[wikidata_type_is("work")],
    receipt=Path("data/receipts.jsonl"),
)
if not receipt["applied"]:
    print(receipt["reason"])                    # refused, and the record is untouched
```

Targets can also be a mutable mapping — a DB row you loaded yourself. YAML files
round-trip when `pyyaml` is installed.

Three destinations, and exactly one of them runs:

| | what happens on success |
|---|---|
| `writer=fn` | `fn(merged_dict)` is called. **Your mapping is not touched** — the merged record only reaches you through `fn`. |
| a path, no `writer` | atomic write through a temp file in the same directory |
| a mapping, no `writer` | the mapping is updated in place |

The first row is the one that surprises people: passing both a mapping and a
`writer` does not also update the mapping, so a caller that reads the row back
afterwards sees the old values. Persist what `writer` receives.

## `MISSING` is not `None`

An absent key, an explicit null and a compiled default are three different
facts. Collapsing them is how a "no change" proposal silently overwrites a
value someone else wrote in between. `preimage={"wikidata_id": MISSING}`
asserts the proposer saw no such key; if the record now holds an explicit
`null`, that is a mismatch and the apply is refused.

## Never partial

One refused field refuses the whole proposal. The 28 Aug incident cluster was
interrupted applies whose preimages were no longer restorable; the write goes
through a temp file in the same directory and an `os.replace`, so a crash
leaves the old record, never half a new one.

## The receipt

Returned and written on **every** attempt, applied or refused:

```json
{"applied": false, "reason": "preimage mismatch on 'year': proposer saw 1867, record holds 1879",
 "target": "data/works/et-dukkehjem.json", "fields": ["year"],
 "sha256_before": "…", "sha256_after": "…", "ts": "2026-09-20T…Z"}
```

A refusal you cannot count is a refusal you will argue about later. This is
also why "sealed mutation record" stops being a separate 191-line script.

## Validators

| Validator | Refuses |
|---|---|
| `enum_member(allowed, *, fields=None)` | a value outside a controlled vocabulary |
| `regex(pattern, *, fields=None)` | a value that does not *fully* match |
| `wikidata_exists(*, client=None, fields=None)` | a malformed or deleted QID |
| `wikidata_type_is(expected, *, client=None, fields=None, property_id="P31")` | a QID that exists but is the wrong kind of thing |

Every argument after the first is keyword-only. `fields=` restricts a validator
to named fields; left `None` it sees every field in the proposal.

Signature: `(field, value) -> message | None`. Any message refuses. `None`
values pass every built-in validator, because null is a legal answer and
scoring coverage without it is how 130 fabricated bios reached production.

### Existence is not enough

Of 901 work QIDs audited in one catalogue on 20 Sep 2026, **198 pointed at
something that was not that work** — other works, non-works, deleted items.
Every one passed an existence check. *Et dukkehjem* resolved to Ramon Llull.

```python
wikidata_exists(client=c)("wikidata_id", "Q170065")     # -> None: it exists
wikidata_type_is("work", client=c)("wikidata_id", "Q170065")
# -> "Q170065 (Ramon Llull) is P31=['Q5'], not a work"
```

`expected` is a key of `hippocampus.wikidata_resolve.TYPE_HINT_P31` — exactly
`person`, `place`, `event` and `work` — or an
explicit iterable of QIDs. Subclass chains are **not** walked, so an
over-narrow allowlist refuses a legitimate value — widen the allowlist rather
than dropping the check. Pass `client=` to inject a fake in tests; the
default constructs a `WikidataClient` lazily, so no network is touched unless a
QID is actually validated.
