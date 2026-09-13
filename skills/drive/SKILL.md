---
name: drive
description: "Turn a long, ambiguous voice dump such as 'research this' or 'improve this' into a bounded direction card grounded in nearby evidence. Use when the user wants an agent to drive an open-ended research, product, design, or coding effort; choose the right first move; avoid an expensive rabbit hole; or coordinate Codex and Claude without prematurely spawning a swarm. In v0 this is planning-only: it must not dispatch workers or modify the target project."
---

# Drive

Compile the request into the smallest pilot that can disprove a bad direction. Do
not act as an autonomous project manager yet. The output is a direction card for
the user to judge before execution begins.

## Workflow

1. Read the full request and the target project's instructions. Infer the desired
   user outcome, what "better" means, and the important non-goals. State useful
   assumptions instead of turning the opening into an interview.
2. Before web research, broad code inspection, or implementation, retrieve the
   nearest local analogs. Inspect at most three strong sources: recent project
   artifacts, repository history, and prior conversations. If
   `claude-chat-search` is available, use `cross` directly; do not re-index first.
   If no analog exists, record the queries and `none-found`.
3. Choose `research` or `improve`. Read only the matching mode guide:
   [research](references/research.md) or [improve](references/improve.md).
4. Propose one representative pilot of one to three units and the evidence that
   would tell the user whether it works. Prefer something the user can actually
   use, browse, inspect, or manually test.
5. Stop before execution. In v0, set workers, model calls, and premium calls to
   zero; disable delegation; and forbid scaling until the user accepts the pilot.
6. Draft a plan matching [the plan contract](references/plan-contract.md). Check
   it with `python -m limbic.drive validate PLAN.json` when Limbic is importable.
   If it is not, apply the contract manually and say the automated check was
   unavailable. Do not install dependencies merely to run this check.
7. Present the concise direction card in chat. Ask at most one question, and only
   when two plausible answers would lead to materially different pilots.

## Non-negotiable gates

- Retrieve before planning: local precedent precedes generic external advice.
- No batch larger than three until one complete representative unit has been
  experienced and accepted.
- No worker may spawn another worker. Future worker models must be explicit.
- Premium models are for convergence after uncertainty is visible, not for broad
  initial exploration.
- Stop on evidence: user acceptance, a failed usefulness test, a material change
  of direction, or a missing dependency that changes the plan.

## Direction card

Lead with the proposed first move. Then include: inferred outcome; definition of
better; nearest precedents and lessons; assumptions/non-goals; pilot artifact;
how it will be judged; zero-call v0 budget; stop/scale gate; and the next action
after approval. Keep it to roughly one page even when the voice dump is long.

For why these constraints exist, see [calibration lessons](references/calibration.md).
