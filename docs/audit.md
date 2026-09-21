# `hippocampus.audit` — an audit demotes, or does nothing

`apply_audit(decisions, audit, ...) -> (decisions, report)`

An independent audit can take a decision back. It can never make one. That
asymmetry is the whole mechanism, and everything below is bookkeeping around
it.

An auditor that may promote is a second proposer with a nicer name: you only
hear from it when it agrees, so its agreement stops being evidence. An auditor
that may only demote is evidence, because every row it writes costs you
something. So `apply_audit` is demote-only *by construction* — the only
disposition it ever writes is the hold value, it never appends a record, and it
re-checks that invariant over its own output before returning. An audit file
that says `"disposition": "new"` on a held row changes nothing.

## Where it goes in the pipeline

**After reconciliation, over decisions that have already been settled.** An
audit read before reconciliation audits a draft, and the reconciliation that
follows overwrites what the audit decided without noticing. `apply_audit`
refuses a decision that carries no disposition field for exactly this reason:
an unsettled row is not something an audit can act on.

```
decide() → replicate agreement → reconcile → apply_audit() → schema_refusals() → write
```

## The audit file

Sections are named by the caller, so an existing project's file shape works
unchanged. Two shapes are accepted for every section: a list of rows each
carrying an id field, or a mapping from id to payload.

```json
{
  "auditor": "a model of a different family, briefed blind, 21 Sep 2026",
  "date": "2026-09-21",
  "scope": "all 1,342 new-concept decisions; full set, not a sample",
  "citations": [
    {"citation_key": "544c22ce", "note": "the Uppsala king of Harald's time, not Erik Segersäll"}
  ],
  "concepts": [
    {"concept_id": "kristina-sigurdsdatter", "note": "same woman as kristin-sigurdsdatter"}
  ],
  "definitions": {
    "margrethe-munthe": "Norwegian writer of children's songs and verse, 1860–1931."
  },
  "source_errors": [
    {"concept_id": "rembrandt",
     "citation_cue": "maleren Rembrandt kom også til Stockholm",
     "note": "the 1930s textbook says so; he never left the Dutch Republic. The read
              copied the source faithfully, so this is a source error reproduced,
              not a model hallucination."}
  ]
}
```

```python
decisions, report = apply_audit(
    decisions, audit,
    key_fields=("citation_key", "concept_id"),
    hold_sections=("citations", "concepts"),
    correction_sections=(("definitions", "definition"),),
    finding_sections=("source_errors",),
    validator=lambda field, value, row: slot_echo_refusal(value),
)
```

### `source_errors` is a finding, never a correction

A fact that is wrong *in the source* is not a defect in this batch. The
pipeline reproduced it faithfully, which is what a pipeline reading a 1930s
textbook is supposed to do. Folding it in as a correction would silently edit
the source out from under the evidence, and the next person to read the record
would find a claim the cited page does not make. It stays in the report, as a
finding about the corpus.

### Corrections pass the caller's validator

An auditor writing a replacement definition is still a model writing prose, and
prose is where the shipped `i01` defect came from. Every correction goes
through `validator(field, value, row)`; anything it refuses is reported under
`corrections_refused` and never written. The natural validator is the output
refusal kit in [`packet.md`](packet.md).

## The report

| Key | Meaning |
|---|---|
| `demoted`, `demoted_keys` | how many decisions moved to hold, and which |
| `corrected` | `{key, field, from, to}` per applied correction |
| `corrections_refused` | corrections the validator rejected, with the reason |
| `unknown_ids` | audit rows matching no decision — **reported, never dropped** |
| `already_held` | audit rows on a decision that was already held |
| `findings` | the finding sections, verbatim |
| `ignored_sections` | sections present in the file that the caller did not name |

An unknown id means the auditor read something this batch is not writing: a
stale pack, a renamed key, an auditor briefed on the wrong set. Every one of
those is a thing you want to see. `ignored_sections` is the same instinct
pointed at the file itself — an auditor that invented a `promotions` section
shows up here rather than being silently discarded.

## Fit check

Run against skard's real `independent-audit.json` and its 2,925 stored
decisions, with the recorded audit holds reversed to reconstruct the pre-audit
input:

| | skard `demote_on_audit` | `apply_audit` |
|---|---|---|
| decisions demoted | 86 | 86 |
| held after | 1,518 | 1,518 |
| held sets | identical (symmetric difference 0) | |
| promotions | 0 | 0 |
| definition corrections | 424/424 referents identical | |

`apply_audit` additionally reported 1 unknown id and 2 already-held rows that
skard's version passed over in silence.

What stayed in the project: the section names, the key fields, and the cluster
bookkeeping skard does when a demotion empties a referent (removing the label,
dropping the evidence row, deleting the referent). That last one is a
project-specific consequence of a demotion, not part of the demotion, so the
library reports and the caller decides.

See [`blind-audit.md`](blind-audit.md) for how to brief the auditor that
produces this file.
