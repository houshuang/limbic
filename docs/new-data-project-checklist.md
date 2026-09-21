# New data project checklist

Hand this to any agent on day one. Three tiers; each item is tagged **CODE**
(with the limbic function that enforces it, where one exists) or **PROSE**
(a rule a human has to hold, because nothing can refuse it).

The shaping constraint is **governance-to-yield**: one project built 12.3k
lines of sealed machinery, 26 prompt versions and 13 per-packet test files
around $2.43 of model calls that produced **0 writes**. Every item below must
cost less than the failure it prevents, *at your project's size*. That is what
the "when NOT to adopt" line on each tier is for.

Derived from the 20 Sep 2026 `llm-pipeline-audit` (its `lanes/governance.md`
§3–5 — an internal document, not part of this repo),
which traced ten incidents across a year to the mechanism that did or did not
prevent a recurrence. The one general finding: **a mechanism that refuses a
specific bad state worked; a mechanism that describes a state never refused
anything.** So prefer CODE, and when a rule appears in three documents, make it
a guard or delete it.

---

## Tier 0 — solo exploratory project, day one

Seven non-negotiables. 0.1–0.5 cost under an hour combined.

| # | Rule | Failure it prevents | Enforcement |
|---|---|---|---|
| 0.1 | Canonical source is diffable text; DB, index, embeddings and site data are derived and **never committed** | 9 historical DB blobs >100 MiB forced a history rewrite; elsewhere a 9-line data fix became a 259 KB diff that `git show` then fed back into agent context | **CODE**: a `.gitignore` line plus a 3-line pre-commit hook refusing `*.db` and generated dirs |
| 0.2 | Stable content-derived IDs, never reassigned on reprocess | A reprocess reassigned IDs and orphaned 12,248 model-verified linkages, which now survive only in a stray JSON file; a window-local remap corrupted evidence→claim edges for five months | **CODE**: ID = hash of a declared key tuple; a test asserting IDs are identical across two clean builds |
| 0.3 | Raw input frozen once with URL, timestamp and hash | Otherwise you cannot tell later whether the model was wrong or the source changed | **CODE**: fetch helper writes `raw/<sha256>` plus a manifest; never refetch over it |
| 0.4 | `null` is a legal, rewarded answer, never penalised by a coverage metric | 130 fabricated biographies and 14 literal prompt templates in production | **CODE**: schema allows null (`resolve.slot_enum` always emits a `"none"` member); coverage never scored alone |
| 0.5 | Model proposes, code writes: field whitelist plus preimage check at the boundary | Prose written into a typed Wikidata ID field at confidence 0.88 with five sources | **CODE**: `hippocampus.apply.apply_proposal` |
| 0.6 | Link/match accuracy measured against a **shuffled random baseline** | 33% correct vs 22% random, undetected for five months while every structural check passed | **CODE**: a 30-line test that fails when accuracy is within noise of shuffled targets |
| 0.7 | Small frozen gold sample (50–200), stratified, hand-checked once, committed | "Zero broken references" reported while 10 of 10 sampled works were wrong | **CODE**: gold file plus a scored eval. If the project has merge history, mine the gold set from applied merges — no model calls needed |

**When NOT to adopt:** under ~1,000 records with no public surface, 0.6 and 0.7
can wait for the first measured suspicion. 0.1–0.5 always pay.

---

## Tier 1 — batch campaigns with cheap workers

| # | Rule | Failure it prevents | Enforcement |
|---|---|---|---|
| 1.1 | **Yield probe on 50 packets before building any machinery** | 12.3k lines and 26 prompt versions → 0 writes; next door, 2,336 of 2,342 proposals came from a plain join while 384 model calls produced 6 | **CODE**: `cerebellum.packet.probe(n=50, min_yield=…, execute=True)`, which raises rather than reports |
| 1.2 | Frozen, hashed packets (10–40K) given to **calls**, not agents | One traced packet: 3.8M tokens as an agent vs ≈40K stateless — 95× | **CODE**: `cerebellum.packet.make_packet` + `run_packets`. **PROSE**: never spawn agents as coders |
| 1.3 | Preimage-checked proposals; missing key ≠ explicit null | The 28 Aug partial-apply cluster, preimages unrestorable | **CODE**: `apply_proposal(preimage={"f": MISSING})` |
| 1.4 | Replicate-and-agree instead of a confidence score | Three reads of one input agreed only 62.8% of the time; two blind reads lifted precision 0.71 → 0.97 | **CODE**: `run_packets(replicates=2, agree=2)` — disagreement **holds**, it never tie-breaks |
| 1.5 | Every ledger row carries `purpose` and `outcome` | 153,554 rows, $4,144 notional, `purpose` empty on 40%, so spend could not be traced to a stage | **CODE**: `cached_call` requires `purpose`; `run_packets(outcome_fn=…)` and `cost_log.record_outcome` |
| 1.6 | Residual report per batch: what was held, why, how many, **and what is not known** | Coverage optimisation quietly trading precision for completeness | **CODE**: the `run_packets` report (`held`, `failures`, `split`, `remaining`) |
| 1.7 | Identifiers come from a code-retrieved candidate list offered as a fixed slot enum plus "none of these" | "The full list is too large, use LIKE queries" — recall outsourced to the model's guesses at spelling | **CODE**: `resolve.candidates` → `resolve.slot_enum` → `resolve.unslot` |
| 1.8 | Quotes validated as literal substrings of the frozen source | Invented evidence that reads perfectly | **CODE**: `packet.validate_quotes` (whitespace collapse only) |
| 1.9 | A deterministic cross-check on recall, independent of the model | A corpus run printed "Harald Hårfagre" and returned nothing, because nothing put the name in front of the model | **CODE**: `packet.unmatched_names` + `corpus_lowercase_words` |
| 1.10 | Incident → bounded repair → **guard** → retire the script | 374 write-once scripts; 218 one-off campaign controllers = 32% of all script lines | **CODE**: an age report flagging one-commit files with no inbound reference. Without the guard you bought cleanup, not improvement |

**When NOT to adopt:** 1.2–1.4 suit streams of ≥500 items; below that the
packet machinery costs more than redoing the batch by hand. **1.1 always
applies** — it is cheapest exactly when you are least sure.

---

## Tier 2 — many concurrent agents plus a public release

| # | Rule | Failure it prevents | Enforcement |
|---|---|---|---|
| 2.1 | Owned lanes with declared write paths; a hook refuses commits outside them | Index-lock contention: a concurrent session sealed a batch and lost its commit — data on disk, recorded nowhere | **CODE**: an `agent_guard`-shaped hook, ~60 lines. The clearest prevented recurrence in the set |
| 2.2 | One OS file lock per shared resource, holder identity written into the lock | Lock contention and two-publisher risk | **CODE**: non-blocking `flock`. Never remove a lock on the evidence of a quiet process list |
| 2.3 | One integrator per destination; researchers get a pinned snapshot SHA, not the tree | "Branches colliding, lots of dirty files on main" | **PROSE** for the role; **CODE** for the pin (a SHA in the assignment) |
| 2.4 | Release candidate bound to an exact HEAD, plus a cheap **proven-equivalent** path for presentation-only changes | One paragraph of copy triggered a 95 MB / 627,500-row import and published 72 unrelated commits | **CODE**: a read-only proof against the live marker's data identity. A proven equivalence, not a waiver |
| 2.5 | Publish phases journalled and reconcilable — a step can see that it already ran | A publication refused five times on operations that had already succeeded | **CODE**: phase journal plus a reconcile-only entry point |
| 2.6 | Status derived, not asserted: one command prints source HEAD / remote HEAD / live marker | "live" claimed from a passing build | **CODE**: one `status` command, and nothing else may say "live". Prose cannot do this |
| 2.7 | Declare the expected changed-entity count before writing; exceeding it stops the run | A complaint about two links produced mass category removal; a commit promising 6 changes touched 3,963 files | **CODE**: `refuse.expect(changed=N)` — decide inside the block, write after it, so a mismatch raises before the write. `override=True` is the explicit escape hatch. Until 21 Sep 2026 this row said CODE and no code existed anywhere |
| 2.8 | Contention machinery **stands down** when only one lane is active | A worktree cap waived by hand three times in a week, in a session where there was nobody else writing | **CODE**: detect single-operator mode rather than being hand-waived each time |

**When NOT to adopt:** with one operator running one agent, Tier 2 is pure cost
— with one exception. **2.7 pays at every size:** blast-radius bounds are
about the request being narrower than the action, which is independent of how
many agents are running.

---

## Always

- Declare the expected changed-entity count before the first write.
- Every incident produces a bounded repair **and** a guard or test **and** a
  residual report **and** a decision to promote or delete the script.
- A rule stated in three documents becomes a guard, or gets deleted. Repetition
  is a tax every worker pays per turn and a refusal none of them can rely on.
- Tests never touch a production ledger or database. Set the path in a
  conftest **at import time** — a singleton resolves its path when the module
  is imported, so a fixture is already too late — and make the default path
  raise. limbic's own `tests/conftest.py` is the worked example.
