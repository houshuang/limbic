---
name: coordinator-hygiene
description: "Keep a long-running coordinating thread from becoming the bill, and keep its destructive commands from becoming the incident. Use when running a multi-step or multi-agent project from one thread — deciding when to hand off, how to wait for a long job, when to run gates, or before any rm -rf / git checkout -- / reset / stash / replan. Triggers on 'continue', 'keep going', 'check on the agents', 'run the full validation', 'clean up the worktree', 'start over', 'replan'."
---

# Coordinator hygiene

The coordinating thread is usually the most expensive thing in the project.
Measured: one thread made **3,838 requests over three days at a median 148K
context with a median 204-token output**, through 38 compactions — 65% of a
project's entire spend, with tool output returned under 2% of the input bill.
Another burned **221M tokens on 11 user messages**, four of them "continue".

Cost = number of tool calls × standing context. Everything below attacks one of
those two factors.

## One thread per task; state lives in a file

Write the handoff file first, update it as you go, and start a new thread when
the task changes. `HANDOFF.md` holds: what is done, what is in flight, the
pinned revision, the next action. Hand off rather than continuing past ~80 tool
calls or ~100k tokens.

"Continue ambitiously" on a 150K-token thread is the single most expensive
instruction available, because every subsequent tool call replays that context.

## Never poll

Do not `wait`/`sleep`/re-check a subagent from a large context — polling cost
32.7M tokens in one thread. Either run the long command **in the foreground
with a timeout**, or check the artefact the job writes (a file, a ledger row, a
status line), once, when you have reason to think it is done. A background job
that reports on completion is free; a loop that asks is not.

## Read narrowly

- `rg` with line ranges, never `cat` of a 672 KB monolith. If lookups are
  JSON dumps, build the CLI: in one project `MATCH` appeared 3 times against
  1,174 calls that dumped or grepped the graph.
- `git show`/`git diff` **always** with `--stat` or a path. Committed generated
  site data turns a 9-line fix into a 259 KB diff that then re-enters context.
- **Never load an image at original resolution into a coordinator.** One call
  was 12 MB. Downscale, or send it to a worker.
- Process docs are not free either: in one repo 33% of patches were HANDOFF,
  LOG, WORKPLAN and INDEX churn — twice the patch text that reached the data.

## Gate once per batch, not per edit

The validators are cheap; calling them from a 148K-token context is not — 452
test runs, 302 rebuilds and 88 full gates were each a ≥120K-token round trip
for a median 4.4K characters of output. Run one quiet gate, **cached by tree
hash**, at the end of a batch and on the release candidate. It should print a
handful of lines and write the full report to a file, exiting 0/1/2.

## Before anything destructive

`rm -rf`, `git checkout --`, `git reset`, `git clean`, `git stash`, deleting a
lock, or replanning a batch: **list what it would destroy first, dotfiles
included, and show the list.** Then:

- Mark **paid artefacts** — cached model responses, sealed receipts, frozen
  packets — and require an explicit flag (`--discard-paid`) to throw one away.
  Money already spent is not scratch space.
- Never make another agent's checkout look clean. `git stash` is shared across
  linked worktrees. Remove a worktree with `git worktree remove`, never
  `rm -rf` (which leaves a dangling registration).
- Do not remove a lock on the evidence of a quiet process list.
- **Declare the expected changed-entity count before writing**, and stop if the
  run exceeds it. A complaint about two links once produced mass category
  removal; a commit promising 6 changes touched 3,963 files. Blast-radius
  bounds pay at every project size, including a solo one.

## Tests never touch production

Point the ledger and any production DB at a temp path **at conftest import
time** — a module-level singleton resolves its path when it is imported, so a
fixture is already too late — and make the production path *raise*. Two agents
polluted a real cost ledger in one day by running the suite. See limbic's
`tests/conftest.py`.

## Refuses beats describes

The general finding behind all of this: **mechanisms that refuse a specific bad
state worked; mechanisms that describe a state never refused anything.** A
reviewer-independence check that only asserted the field was a non-empty string
caught nothing in a year. 528 lines of release machinery were unreachable dead
code. Meanwhile preimage restore, write-path binding and a write budget each
have clean post-introduction records.

So: **a rule you have written in three documents is a rule that should be a
guard.** Convert it, or delete it. Saying a thing nine times does not make it
truer, and every worker pays to read it.

## Report honestly

State what was verified and with which command, what was skipped, and what is
not known. If a number changed, print it before and after. A residual report —
what was held and why — is part of the result, not an optional extra.
## Finishing a long project

Long autonomous projects do not end by themselves; the coordinator keeps finding
work. These rules make stopping a decision rather than an accident.

- **A metric must prove it tracks the outcome, or it is dropped.** Before a number
  becomes a gate, show that moving it changes what a user gets. Internal purity
  counts (strict completion, receipts, pins) are diagnostics, never release gates.
  The release gate is what the user sees: the deployed site, the acceptance pass.
- **Stop after two rounds with nothing new.** A lane that makes no measurable
  progress in two rounds stops and reports the cause by category. Resistant
  failures usually mean the task or the grader is wrong, not the worker.
- **Name what was not covered.** "Unverified" is a label, not a failure. A partial
  plane ships labelled partial; a skipped check is listed as skipped.
- **Consult the strongest model at three moments only:** before committing to a
  plan, on a repeated failure, before declaring done. Write this into the project
  guide; left to discretion it fires too late.
- **Refuses beats describes.** A rule written in three documents is either a guard
  that refuses a bad state, or noise every worker pays to read. Do not add a
  receipt, hash pin or protocol unless a check actually uses it to refuse.
- **Handoffs under 400 words.** What is done, what is in flight, pinned revision,
  next action. The coordinator sends longer ones back.
- **Ship, then iterate.** A verified artifact that is not deployed is a liability:
  it goes stale and the next change has to re-verify everything. Deploy at every
  verified checkpoint when the owner has authorised deployment.
- **Process state vs done state.** Decide early which artefacts exist only to
  support the process (caches, session evidence, lane worktrees, intermediate
  campaign outputs) and which the finished project keeps (sources, the graph,
  code, the activated export). Delete the former at the release that no longer
  needs them. A finished project should fit on the disk indefinitely.
- **Workers' excuses, and coordinators' ceremonies.** Keep a short table of the
  phrases that precede skipped or invented work in this project ("out of scope",
  "follow-up", "needs a receipt first", "the ruling requires") with the
  rebuttal and the evidence that would actually settle it.
- **The check is the whole thing.** Every loop step needs a check that can fail
  programmatically; "the model is confident" is not a check. If a step can be
  described without judge, decide or assess, it is code, not a model call.
- **Return the unit, not the batch.** A correction goes back as one item with
  verdict, reason, evidence and scope. Cap at three attempts; after that the plan
  is at fault, not the worker.
- **Forbid the early stops.** Unattended workers end turns by announcing the next
  step, offering to continue, listing non-blocking decisions or reporting a
  milestone. Name these in the brief as not-done.
- **Pin rules, not facts, across compaction.** A short rules file re-read after
  every reset; the spec is never agent-edited.
