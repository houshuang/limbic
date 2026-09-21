# Not built: a paid-artefact registry

"Paid work was nearly lost four times to cleanup, replan and regenerate steps
that did not know what had been bought." That protection exists as working code
in exactly one project (skard `packet_runner.py`) and as *prose* in the
`packet-worker` skill, which is the shape this library exists to fix.

The mechanism is two questions and one refusal:

- **Is this plan already paid for?** Every packet in the plan has a cached
  response under `(input_sha256, model, replicate)`. skard's
  `plan_is_paid_for:2966` walks plan → packet file → cache file and answers
  false on the first miss. limbic already owns the addressing half of this:
  `cached_call` keys on the request bytes and `make_packet` hashes the input.
- **Is this packet still the one its run was bought for?** A packet file, its
  run record and its cached response are one artefact in three pieces; when
  they disagree, every downstream number is computed over text that was never
  sent. skard's `integrity_problem:3171` compares the packet's `input_sha256`
  with the run record's and names the mismatch.
- **The refusal:** any entry point that replans, regenerates or cleans up
  requires an explicit `discard_paid=True` before it may orphan a cached
  response. Absent the flag it raises and names what would have been lost.

Proposed surface: `cerebellum.packet.paid_status(plan, cache_dir, model) ->
PaidStatus` with `paid`, `missing`, `orphaned` and `mismatched` fields, plus
`require_discard_paid` as a decorator for replan entry points.

Why it is not built here: the acceptance test is a fit check against skard's
`data/adjudication/citation/*` tree, and the plan/packet/run/cache **directory
layout** is the part that varies most between projects — it is the one place
where "the shape goes to limbic" is least obviously true. Lifting it without
that fit check is how the previous three lifts were rejected. Build it when a
second project has a paid-artefact tree to fit-check against, and let the two
layouts decide the parameterisation.

Until then the prose stays in the skill, marked as prose.
