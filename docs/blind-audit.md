# The blind audit — briefing the second reader

    from limbic.hippocampus.audit import apply_audit
    decisions, report = apply_audit(decisions, audit, key_fields=(...), hold_sections=(...))

That call is the fold-in and the one non-negotiable step on this page.
Everything below is the brief that produces `audit` for it. Skipping the call
to hand-write the fold-in is not a shortcut: on 21 September 2026 four
Kulturbase campaigns each hand-transcribed their own fold-in instead of
calling it — replaying their raw auditor output through `apply_audit`
reproduced every one of their hand-computed verdicts exactly (48/48, 18/18,
8/8, 74/74), so the hand work bought nothing — and the one campaign that
wrote its own reconciliation code separately drifted the verdict vocabulary
away from `right`/`wrong`/`cannot_tell` below with nothing to catch it. Use
the brief template as given; change only the domain nouns.

One independent pass over a batch that **decided something about the world**,
as opposed to one that rendered or worded something. Its output folds in
through [`apply_audit`](audit.md) and can only move a decision to hold.

What follows is what worked on 21 September 2026, when a second model read a
batch of ~1,300 settled decisions and produced 86 demotions, 47 corrected
definitions, 6 duplicate pairs, 9 corrected date extents and 2 source errors —
including a duplicate concept pair, a regnal span written into a lifespan
field, and a definition whose own citation contradicted it. None of those were
caught by the replicate agreement that had already passed them.

## The brief

> You are reading records this pipeline has already decided on. You have not
> seen how it decided, and you should not try to reconstruct it.
>
> For each record below, give a verdict:
>
> - `right` — the record is correct as it stands.
> - `wrong` — the record is incorrect. Say why in one sentence.
> - `cannot_tell` — the evidence given does not settle it.
>
> `cannot_tell` is a real answer and I want it whenever it is true. A guess
> that happens to be right is worth nothing to me, because I cannot tell it
> apart from a guess that happens to be wrong.
>
> Return only the rows that are **not** `right`. A row you do not return is a
> row you are saying is correct.
>
> Where you can supply a correction, supply it in the same row. A correction is
> a claim you are making, so make it only where you are confident.
>
> If the record is a faithful reading of a source that is itself wrong, say so
> as a source error rather than a correction. Do not silently fix the source.

## The rules that made it work

**A different model family.** The same family shares the same failure modes,
so its agreement is nearly free and nearly worthless. Two reads from one family
agreed on both of the wrong life dates that this audit caught.

**Blind to the earlier reads, and to earlier audits.** Do not include the
pipeline's confidence, its `checks_passed`, its held/not-held status, or a
previous auditor's notes. A reader shown the answer grades the answer. If you
are auditing after a previous round, audit the current records, not the diff.
`blind_view(items)` strips the common shapes of that (`my_verdict`, `score`,
`tier`, `disposition`, ...) and hands back the stripped records plus which
fields it actually found and removed, for the record — four campaigns each
wrote this by hand on 21 September 2026; call it instead.

**The full set when it is cheap.** A 60-row stratified sample measures a rate;
it does not find the 37th duplicate. At one stateless call per batch of records
the full set is usually affordable, and the full set is what produced the
duplicate pairs — they are invisible in a sample, because a sample almost never
contains both members. Sample only when the full set genuinely is not
affordable, and then fix the size and the seed so the sample is reproducible.

**A fixed verdict vocabulary.** `right` / `wrong` / `cannot_tell`, and nothing
else. Free-text verdicts do not fold in, and an auditor allowed to write
"mostly right" will. `bucket_by_verdict(rows, key_field)` turns the raw
`{id, verdict, reason}` response into `apply_audit`-ready sections and raises
on any verdict outside that vocabulary — the guard that would have caught
21 September's drift to `same_work`/`different`/`unsure`, which nothing did
at the time. A campaign whose auditor genuinely needs another vocabulary maps
it onto this one, or passes its own `vocabulary=`, before the fold-in — never
inside `apply_audit`, which does not and should not know what a section name
means.

**Non-`ok` rows only.** Returning every row doubles the output for no
information and invites the model to pad. Silence is the `right` verdict, which
also means a row the auditor dropped by accident reads as approval — so check
the returned count against what the brief said you would get. `apply_audit`'s
`sent_keys=` makes that check concrete: pass the ids actually sent (a
`blind_view` call's own keys), and the report's `coverage` names every one
that never appeared in the response at all. A missing id there is not
touched — silence-as-right is still the convention — but it is now counted
and visible instead of assumed.

**Corrections supplied, not described.** "The date is wrong" is a demotion.
"The date is wrong, it is 1683–1738" is a demotion plus a correction, and the
second one is what lets `apply_audit` repair rather than only hold.

**Source errors kept separate.** See [`audit.md`](audit.md#source_errors-is-a-finding-never-a-correction).

## Fold-in

```python
view, hidden = blind_view(decisions)              # before the brief
raw = call_the_auditor(view)                       # {id, verdict, reason} rows
audit = bucket_by_verdict(raw, key_field="work_id")
decisions, report = apply_audit(
    decisions, audit, key_fields=("work_id",), hold_sections=("wrong", "cannot_tell"),
    sent_keys=[d["work_id"] for d in decisions],   # after reconciliation
)
refusals = schema_refusals(to_write, schema, only=lambda r: r["id"] in touched)
```

Demote-only, so the audit can never widen the batch. What the auditor was right
about becomes a hold; what it was wrong about costs you a record you could have
kept, which is the cheaper of the two mistakes and the reason the asymmetry is
worth its price.
