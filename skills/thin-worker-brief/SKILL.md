---
name: thin-worker-brief
description: "Brief a subagent so it does not pay a 50K-token entrance fee to rediscover the project. Use when delegating to workers, reviewers or researchers — writing a WORKER.md, choosing a model tier for a delegated role, setting a tool-call budget, or deciding whether to delegate at all. Triggers on 'spawn a subagent', 'delegate this', 'write a worker brief', 'brief the reviewers', 'fan this out', 'which model for this agent'."
---

# Thin worker brief

Every subagent re-reads its whole entrance context on **each turn**. Measured
median entrance fee: **51.5K tokens**, before the worker does anything. A
50K-token instruction stack is therefore not documentation, it is a per-turn
tax — and in the project that had one, workers still made 1,516 hand-rolled
`sqlite3` calls and rediscovered the schema 332 times in a month.

So: decide whether to delegate at all, then make the brief small.

## Delegate only when the work is not a call

A subagent is the right tool for judgement, search across an unknown surface,
and review. It is the wrong tool for applying a codebook to N documents — that
is `packet-worker`, and the gap is 95×. Before spawning, ask whether a
stateless call with a fixed schema does the job.

## The brief: ≤ 2K tokens

A `WORKER.md` of four parts, and nothing else:

1. **Schema card.** The tables or record shape the worker touches, with the
   traps inline ("`normalized_name` is only `lower(name)`: no diacritic
   folding"; "`prf_id` is TEXT"; "a title can be numeric-looking, never assume
   a string"). One screen. This replaces the rediscovery loop.
2. **The lookup CLI, with four example invocations.** Not "query the
   database" — the exact command. A documented-but-broken CLI is worse than
   none: one crashed on an integer title, took 5 s and missed the obvious
   query, and was called twice in a month against 1,516 raw `sqlite3` calls.
   Verify yours runs before you ship the brief.
3. **The proposal template**, and the statement that the worker proposes and
   never writes.
4. **Fetch helpers and flags** the worker will otherwise reinvent (the
   no-metrics query param, the offline snapshot path, the rate limit).

**A named reading list, never "read the docs".** Name the file *and the
section*: `CODEBOOK.md §11 borderline rules`, not `CODEBOOK.md`. An unbounded
pointer is read in full, on every turn, by every worker.

State the **pinned revision** (a SHA, not "main") and the worker's **write
scope** as literal paths.

## Model tier by role

Measured on identical review work: Opus-class reviewers ran a median **10.1M
tokens**; Sonnet-class in the same role ran **0.7–1.3M**. Roughly 10×, for
review output that was not better.

- **Sonnet-class by default** for reviewers, researchers, summarisers,
  fan-out search, and first-pass extraction.
- **Frontier only for adjudication**: contested merges, codebook design,
  edition/witness questions, and the final call on a disagreement a cheaper
  model already surfaced. Frontier models earned their keep exactly there.
- A cheap model inside a tool-using harness is **not** a cheap call. Stateless
  versus agentic matters more than which model.
- Consider a different *vendor* rather than a bigger model for an independent
  check: a second lens catches things a second sample of the same family does
  not, and it draws on a separate quota.

## Isolation and budget

- **Its own worktree**, its own branch, its own generated-output directory.
  Disjoint files do not make a shared index or test environment private.
- **A tool-call budget in the prompt** ("≤ 40 tool calls; hand off rather than
  continuing past ~80 or 100k tokens of context"). Without a number, workers
  continue.
- **No worker spawns a worker.** Fan-out below fan-out is how one thread
  reached 221M tokens on 11 user messages.
- Give the worker the shared parallel-work guide and a concrete assignment in
  its *initial* prompt. It does not inherit newly edited memory or a startup
  hook.

## The handoff back: ≤ 400 words

Require this shape, and say so in the brief:

- path, branch, base and HEAD;
- what changed, as owned paths;
- **verification actually run**, with the command and its result — not
  "tests pass";
- what is *not* done, and what was deliberately left alone;
- whether the checkout is released, and who cleans it up.

Reject a handoff that reports a conclusion with no command behind it. A worker
that says "verified" without evidence has told you nothing, and the cost of
asking again is another entrance fee.
