---
name: campaign-orchestration
description: "Run a multi-lane data campaign (book ingestion, extraction, reading campaigns) with long-running Codex or Claude worker sessions, without the failure modes we already paid for: supervision eating the budget, workers stopping early, stop rules that misfire, rate limits killing sessions, timeouts killing long calls, drivers and branches going quiet unnoticed overnight (lane-audit.sh, drive-loop.sh). Use when launching, watching, resuming or post-morteming worker lanes; when writing campaign-wide worker rules; or when a lane 'stopped', 'stalled', 'wrote a partial handoff' or 'hit 429/capacity'. Triggers on 'campaign', 'launch lanes', 'run overnight', 'orchestrate', 'codex workers', 'why did it stop', 'lane stalled', 'resume the lane'."
---

# Campaign orchestration

## Rule zero: enforce, don't advise

Every failure below was already written down as advice when it recurred. Advice fails because the next agent must
remember it under task pressure. So each lesson must become a mechanism that refuses or stops the bad pattern:
`launch.sh` refuses to start a lane unless `watch.sh` runs; `watch.sh` runs `costguard.py` every cycle and forces a
handoff from any session over 150k context/turn, 120 turns, or 15M tokens/hour sustained over 30+ minutes; sessions start drivers and exit, and the
watcher (zero tokens) starts a fresh small session when a driver finishes. **Measure cost from the first check, not
progress only.** When you add a lesson, ask: what script now refuses this? If none, it will recur.

Measured repeat (23 Oct 2026): the night before, a polling supervisor spent 122M tokens; the next morning the
coordinator avoided polling but launched lanes as long-lived Codex sessions that waited on their own drivers inside
the session — 270M tokens in 3 hours, contexts of 100–400k tokens per turn, 1,100+ turns. Same failure, one layer down.

Read this before launching worker lanes, and add to it after every incident
(see «Learning loop»). Every rule below cost real time or tokens once.

## The shape that works

- **Coordinator is event-driven, never polling.** Launch lanes, then run one
  background watcher (`scripts/watch.sh`) that exits only when a lane writes
  DONE.md, its process dies, or its log is silent for an hour. The coordinator
  wakes on that exit, **restarts the watcher as its first command** (before
  reading results; a context compaction in between is how it gets forgotten:
  skard 25 Oct, three release lanes sat stopped 2 h), then acts. Measured failure (skard,
  22–23 Oct 2026): 122M supervisory tokens vs 0.9M generation tokens; scheduled
  heartbeats and accounting monitors did the spending.
- **One lane = one whole queue.** A lane owns its full queue and a measurable
  deliverable. No pilots-before-pilots, no yield gates (a 10-scan pilot and a
  30% continuation gate on a 500-book queue produced 6 books overnight). Pilot
  only a genuinely new method, and size the pilot to the decision.
- **The queue is the job.** Per-item failures are recorded and the queue goes
  on. Transient errors retry with backoff, then park and retry at the end. A
  defect the worker fixed never counts toward stopping. Stop only when every
  item has an outcome or nothing at all can progress. «Two failures of the
  same step → stop» made three of five lanes quit within an hour, one right
  after fixing the defect it had hit.
- **DONE.md before the queue is exhausted is a failure, not a handoff.** Codex
  workers love writing a tidy «partial handoff» and stopping. Say so in the
  brief, and resume them (below) rather than accepting it.
- **Inherited budgets are not owner requirements.** Token ceilings, call caps
  and protocols chosen by an earlier coordinator are historical; check what the
  owner actually said.

- **The release gate is never the first full test run of new data.** A lane
  that changes populations (assertions, findings, works) runs the full project
  gate on its tree rebased onto main before it hands off; merging on focused
  tests alone moved six count tests and a page-size budget into the release
  gate (skard 26 Oct: one 85-min gate lost, then a manual resume). Tests must
  not read host state (free disk, optional packages).
- **A lane's gate runs on its candidate, not its base.** "Gate PASS" measured
  before the lane's own bundle is applied proves nothing about that bundle.
  Identity merges are the usual culprit: retargeting records onto a surviving
  id creates duplicates that only the combined data shows (skard release 11:
  three exports lost one guard at a time). Give the project a dry run of the
  export's refusals that collects every refusal in one pass, and have both the
  lane and the release cycle run it before any export.

## Mechanics (scripts in `scripts/`)

- `launch.sh <campaign-dir> <lane> [model] [effort]` starts `codex exec` with
  the brief `<campaign-dir>/briefs/<lane>.md` under `nohup caffeinate -i`, log
  in `<campaign-dir>/<lane>/codex.log`, pid in `pid`. (No tmux on the Mac.)
- `resume.sh <campaign-dir> <lane> <message-file>` resumes the lane's **own**
  session (`codex exec resume <session-id> "<msg>"`), so the worker keeps its
  context. Use it to deliver rule changes too: a running session read the
  rules once at start and never sees later edits.
- `watch.sh <campaign-dir> <lanes…>` is the watcher; it auto-resumes a lane that
  died on `at capacity` / `429 Too Many Requests` after 60 s (5 min after three
  repeats), with no cap on attempts, using `<campaign-dir>/msg-capacity.md`.
- Session-level provider errors (`Selected model is at capacity`, HTTP 429)
  kill `codex exec` outright; they are transient, so resume, do not relaunch.
  Stian (26 Oct 2026): Codex often times out with 429 but a plain retry works.
  Treat a single 429 as routine, not as an account limit: retry the call (or
  resume the session) after a short wait before reducing workers or stopping
  the lane. Only a run of consecutive 429s means real throttling, and even that
  is temporary and per model (26 Oct: Sol and Astra 429 for hours while Luna
  answered): keep retrying with backoff and never stop a lane for it (Stian:
  «just keep retrying 429, it always works in the end»).
- **Account rate limit is shared.** Lanes × parallel workers per lane is the
  real concurrency. Six lanes at 6–8 workers each drew 429s within 90 minutes;
  cap at ~4 model workers per lane for six lanes.
- **An error share is a stop condition; count only answered calls.** Skard 27 Oct: the `claude` CLI rejected
  nearly every Sonnet/Opus call before the API (`exited 1`, duration_api_ms 0); the transport called it
  `permanent-error`, the harvest escalated packets that had no cheap read, and a "1,992-call" round produced 335
  labels while the lane sat idle for 80 minutes. Drivers stop and report when >10% of a pass errors; harvests refuse
  escalation unless both cheap reads exist; the coordinator's first check reads the error share, not the call count.
- **Timeouts must exceed the longest single call.** `sleepsafe.py --limit 540`
  killed every whole-section Astra read (exit 124) and the lane counted it as a
  failure. Use a limit above the worst call (e.g. 3000 s) for long reads.
- Codex model names on the ChatGPT plan (Oct 2026): `gpt-6-luna`,
  `gpt-6.1-sol`, `gpt-6-astra`. `gpt-6.1-luna` / `gpt-6.1-astra` → HTTP 400.
  A bare `codex exec` call costs ~8K tokens of system prompt.
- Script bugs already hit: `set -e` + a glob `ls` that matches nothing exits the
  script; a relative message path breaks after `cd` (resume sent an empty
  prompt and the session ended). Use absolute paths.

- **Drivers must start in their own session** (`subprocess.Popen(..., start_new_session=True)`); a `nohup … &`
  launched from inside `codex exec` dies when the session exits. Every driver writes `driver.pid`, or the watcher
  thinks the lane is idle and relaunches sessions in a loop (12 relaunches in 25 minutes, 23 Oct).

- **Measure cost per unit of output, not just per session.** The session guard can't see a driver that makes 250
  calls/hour but finishes 2 books/hour (24 Oct: ~50–250 calls per contents list vs ~13 two days earlier). Every driver's
  PROGRESS line must carry calls and units done, and the coordinator's first status check compares calls/unit with the
  last known good rate.

- **Pick the cheaper orientation before any per-item LLM pass.** Compute calls = items × reads ÷ batch for both
  topic-major (each topic verifies its retrieved passages) and passage-major (each passage is tagged against a
  shortlist of candidate topics) and choose the smaller; measure shortlist recall on gold with zero calls first.
  Escalate only disagreeing cells, packed together, never whole batches. Measured failure (skard, 25 Oct 2026):
  topic-major at depth 1500 cost ~117 calls per topic, each passage read ~8 times, 611 whole batches sent to Sol;
  662 topics projected to 100k+ calls versus ~3.5k passage-major.
- **A big export runs alone.** On a laptop, stop (not pause) every lane before a release export: paused processes keep
  their RAM, which the export's memory peak pushes into swap on the same disk (skard 25 Oct: swap 14 → 22 GB, disk
  33 → 4 GB free). Check free disk ≥ export size + 10 GB first; delete re-fetchable caches, never pointers or generations.

- **Check merges against data that contains them.** A fast post-merge check that renders against the *old* snapshot
  cannot see data regressions; skard 24–25 Oct met navigation and count failures from three data merges only at the
  release gate. After each data merge run the data-sensitive test modules on the merged tree, and make data lanes pass
  the full test stage before DONE.

## The keeper: nothing waits for a session to notice (30 Oct 2026)

Measured (skard, night of 29–30 Oct): the coordinator hit its context limit at 01:47, wrote a handover «for a successor
session», and nothing ran from 03:40 to 07:40, because nothing starts a successor. Watchers had the same flaw one level
down: `watch-forever.sh` dropped a lane after its first event and waited for a coordinator to re-add it. Six stale
watcher processes were running the next morning.

`scripts/lane-keeper.py` replaces watch.sh / watch-forever.sh. launchd runs `tick` every two minutes
(`~/Library/LaunchAgents/com.stian.lane-keeper.plist`); it spends no tokens and reads the repo's `.coordinator.json`
(`state`, `handover_glob`, `coordinator` caps).
- **Lanes continue by themselves.** A session that ends with HANDOFF.md and no live driver gets a fresh session; one that
  ended on 429/capacity is resumed after a backoff, with no cap; one that ended with no DONE/READY/HANDOFF is resumed twice
  with «continue or write the right file»; DRIVER-FAILED.md resumes the worker once to fix it; costguard runs every tick.
- **Drivers survive reboots.** drive-loop.sh records `driver.cmd` and writes `DRIVER-EXIT` when it ends; a drive-loop
  that vanished without one is restarted, unless its sleepsafe child still runs. Stop one with STOP, never kill alone.
  `--once` is for jobs that must not rerun on failure (release cycles); a driver-only folder's exit becomes an event.
- **Outcomes that need judgement are events** (READY-*, DONE.md, job exit, caps, an hour's silence) in
  `<state>/keeper/EVENTS`. `lane-keeper.py ack <lane-dir> "<what you did>"` (HANDLED) or RETIRED clears them.
- **A coordinator is started when none is live.** Live = a `lane-keeper.py watch` Monitor heartbeat, a keeper-started
  coordinator still running, or a registered coordinator transcript written in the last 20 minutes. Otherwise the keeper
  starts `claude -p` in the repo on the newest handover, at most one at a time, 10 minutes apart, 16 a day; if two
  successive runs leave the same events pending it stops and writes COORDINATOR-STUCK.md.
- An interactive coordinator arms `lane-keeper.py watch` under a Monitor (re-arm on expiry) instead of lane-audit --loop.
  A retiring coordinator writes `<state>/<id>-HANDOVER-<YYYYMMDD-HHMM>.md` and leaves the keeper armed.
- `codex exec resume <unknown id>` silently starts a new session with no brief (found testing the keeper): resume only
  ids whose session file exists under ~/.codex/sessions.

## Lane lifecycle: nothing may go quiet

Measured (skard, night of 28–29 Oct 2026): five lanes stopped overnight and the coordinator noticed none
of them until the owner asked in the morning. A textbook driver was dead for 3.5 h, an NLH driver for 6 h, and a
topic-tree branch had sat for 21 h with a failed gate and no DONE.md. None of these was a hard failure; each
was a wrapper or bookkeeping gap. Scripts:

- **`scripts/drive-loop.sh <state> <name> <limit> -- <cmd>`** wraps every cache-resuming driver. A sleepsafe
  time limit (exit 124/125) restarts immediately, because a limit is not an outcome; three real failures in a
  row write `DRIVER-FAILED.md` with the log tail. Bare `sleepsafe --limit 3000 -- driver` with nothing
  around it is how the NLH driver died at 00:05 and stayed dead.
- **`scripts/lane-audit.sh <repo> <campaign-dirs…> [--loop 60]`** is the coordinator's single status
  check, and its `--loop` form is the overnight monitor. It prints DRIVER-DEAD, DRIVER-QUIET, SESSION-ENDED,
  ORPHAN-BRANCH (a worktree ahead of main, idle for more than 6 h, with no READY/DONE) and UNWATCHED; it prints nothing
  when all is well. Run it at session start, after a reboot or compaction, before saying "status", and
  under a monitor whenever lanes run unattended. `watch.sh` only knows the lanes it was started with, and
  drivers started outside `launch.sh` are invisible to it.
- **A lane is closed only by a file.** DONE*.md, READY*.md, DRIVER-FINISHED* or `RETIRED` (one line: why)
  closes a lane, but only while nothing in it runs. A lane resumed after an earlier READY is open again.
  When the coordinator abandons a lane, it writes RETIRED; it never just leaves the lane.
- **A failed gate is a handoff, not an end.** A worker whose gate fails writes DONE.md saying so (what
  failed, the log path); otherwise the branch becomes an orphan that only lane-audit finds.
- **Atomic writes need a unique temp name.** `path.with_suffix('.tmp')` shared between worker threads let
  two writers of the same book corrupt a JSON file (a trailing `}`) and lose a rename, and the driver died on
  its next read. Use `f'{name}.{pid}.{thread}.tmp'`, then `replace()`. Readers of per-item JSON files
  should fail per item, not per run.
- **A lane's cache must be self-contained.** A freeze step asserted that every cache key existed in the
  lane worktree's cache; three lived only in the main checkout's cache (written by an earlier run there),
  and the lane crashed in the freeze step. When a lane inherits work, copy the keys it references into the
  lane cache up front (content-addressed, so it is safe), or look them up in both places.

## Threads: one orchestrator, or more chats?

The coordinator's context is the main cost. skard 23–29 Oct: two coordinator sessions averaged 357k and
459k tokens per call (peaks ~900k), 45% of all project tokens in a week, and exhausted the Claude allowance
twice. Every report pasted into the coordinator stays in its context for the rest of the session.

- **Exactly one coordinator per integration destination** (a repo's main, a release). It owns merge order and
  shared resources: the heavy-job slot, disk, preview ports, the release. Two chats writing one destination
  produced the FB-ID collisions and renumbering merges.
- **A separate chat** (one Stian talks to) pays off when Stian will steer the work interactively (design
  rounds, research questions, a storage/disk session), or when the work carries its own long context that
  would bloat the coordinator (the music thread, 27–28 Oct). Its cost is a ~50k-token entrance fee to learn
  the project, and coordination by message.
- **A background subagent** is right for a bounded task with a brief and a file result, with no
  conversation expected and the coordinator merging the result. **A Codex lane plus driver** is right for
  data/bug queues and anything over ~30 model calls.
- **Not worth a new chat:** a task under ~20 tool calls, or one that needs the coordinator's live state
  to decide. Do those inline.
- **Hand off rather than grow.** At ~200k context the coordinator writes a handover and a successor
  continues (enforced by `~/.agents/bin/coordinator-hook.py`). A fresh coordinator with a good handover
  is cheaper than a long-lived one.

## Messages between threads

- **A message is a pointer, not a payload:** path + one-line verdict ("READY: …/state/READY-1.md, 197/197
  pass"). Reports go in files. Subagent results over ~6,000 characters arrive truncated, and a subagent may be
  unable to write report files, so tell it up front to return under 4,000 characters or in labelled parts.
- **State lives in files, not messages.** An idle notification can predate your last instruction (skard
  29 Oct: a 06:04 «preflight running» arrived after a 06:20 order). Before acting on a message, check the
  file it refers to. A worker waits on a file or a lock, never on a message ("pause until told" stalls).
- **Every chat's lanes are visible to lane-audit**, so a peer chat registers its lane folders in the
  project's `.coordinator.json` campaigns list (or its own campaign dir) instead of keeping them private.
- **heavy.sh is one slot for heavy jobs only.** Small tests and network-bound drivers run outside it; a
  research driver wrapped in heavy.sh would have held the slot for hours (skard 29 Oct, ten jobs queued).

## Holds, idle notices and handovers (review of 23–29 Oct, skard-state/orchestration-review-20261029)

- **A hold names its pool and the file that releases it** (Claude quota / Codex account / disk / heavy
  slot), and the step that sets it removes it. A Claude-quota pause became «paused every lane» (28 Oct), and a
  PAUSE file from a finished export kept a harvest idle for 18 h (29 Oct). A held lock means wait, never stop.
- **Idle notices are not turns.** An idle notification that repeats a message just received gets no reply:
  one music thread spent 52 replies and ~33M tokens acknowledging echoes (27–28 Oct).
- **A handover bullet that says «remaining», «left» or «next» names its owner:** «→ lane X», «→ waits on
  <file>», or RETIRED. A Claude thread coordinator is a lane too: handing off is not done, and waiting on a
  message instead of a file stalled lanes for half a day (27 Oct).
- **One live session per lane:** launch.sh refuses while the lane's pid is alive (double sessions in two
  lanes, 28 Oct 22:15).
- **Workers read COMMON, not INCIDENTS:** INCIDENTS is the coordinator's log (126 KB × 323 handoffs).
- Codex `gpt-6.1-terra` is the default for research reads (Stian 28 Oct: «astra is too expensive»).

## Learning loop

Every campaign folder has `INCIDENTS.md`: one row per incident, written when
it happens — date, lane, symptom, cause, cost (time/tokens/items), fix, and
whether it is a repeat. A repeat class, or anything that cost more than an
hour, becomes a rule here (generic) or in the project's worker rules
(project-specific: data-source quirks, tool flags), in the same turn.
Project specifics stay in the project; this file keeps the generic lesson
with its measured cost.

## Project-specific lesson files

- Skard / Læreplanatlas: `~/src/skard-state/campaign-20261023/COMMON.md`
  (worker rules) and `INCIDENTS.md` (NB access quirks, IIIF tiles, topic
  extraction pitfalls).
