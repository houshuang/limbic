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
- **The model name picks the engine.** A model matching `opus|sonnet|haiku|fable|claude-*` starts
  `claude -p … --model <m> --permission-mode bypassPermissions --session-id <uuid> --output-format stream-json
  --verbose` instead, with `ANTHROPIC_API_KEY` unset (subscription, not API billing); the effort argument is ignored.
  launch.sh writes `session id: <uuid>` into codex.log first, so the same log, pid, `model` and `session-at` files
  serve both engines. To switch a lane, write `sonnet medium` (or `opus high`) into `<lane>/model`: the keeper's
  next launch uses it, and a resume of a lane whose last session is not a Claude session
  (`~/.claude/projects/*/<sid>.jsonl`) starts a fresh Claude session with brief + HANDOFF instead. A live session
  keeps its engine until it ends. Switching back to Codex has no such bridge: the keeper reports ENDED; relaunch.
- The keeper also watches for runaways each tick: ORPHAN (pid-1 processes under a lane root older than 30 min),
  STOPPED-BUT-RUNNING (a STOPped lane's driver alive 10 min later), STALE-PREVIEW (local.mjs and similar older than
  6 h with no live session ancestor) and RESOURCE-FLOOR (disk < 30 GB free or swap > 20 GB, hourly, with top processes
  and the largest `disk_watch` dirs). Reporting only, until `"reap_orphans": true` in .coordinator.json; config keys are
  in lane-keeper.py's docstring. Sessions, coordinator.pid and tracked lane pid / driver.pid trees are never reaped.
- costguard.py reads Codex token_count records only. Claude lanes are listed as skipped in COST.md and never
  capped. The keeper runs the campaign's own copy of costguard.py, so refresh that copy to get the skip row.
- A cost-capped lane can write `NEXT-COMMAND`: one command line, `bash <installed scripts>/drive-loop.sh [--once] <lane-state-dir> <job> <limit> -- <driver argv>`. This is an already authorised continuation. The keeper starts it detached instead of a handoff session and consumes the file after launch; use `--once` for release cycles. A handoff may prepare this command even when told to stop new work.
- Optional `.coordinator.json` `release_scope` names a `RELEASE-SCOPE` file (relative to the repo, or absolute): one commit or path to a lane READY file per line, with `#` comments. READY files resolve relative to the scope file; labelled `HEAD` or `commit` IDs identify deliverables (bases and measurements are ignored). When all are on main and no release cycle is alive, the keeper emits `RELEASE-READY` once per scope hash.
- `resume.sh <campaign-dir> <lane> <message-file>` resumes the lane's **own**
  session (`codex exec resume <session-id> "<msg>"` or `claude -p --resume <session-id> "<msg>"`), so the worker keeps its
  context. Use it to deliver rule changes too: a running session read the
  rules once at start and never sees later edits.
- `watch.sh <campaign-dir> <lanes…>` is the watcher; it auto-resumes a lane that
  died on `at capacity` / `429 Too Many Requests` after 60 s (5 min after three
  repeats), with no cap on attempts, using `<campaign-dir>/msg-capacity.md`.
- Session-level provider errors (`Selected model is at capacity`, HTTP 429; for Claude lanes `usage limit`,
  `hit your … limit`, `rate_limit_error`, `overloaded_error`, `API Error: 429/529`, a rejected `rate_limit_event`)
  kill the session outright; they are transient, so resume, do not relaunch.
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
- **A big export owns the heavy slots.** heavy.sh queues other heavy jobs while lanes continue non-heavy work;
  do not STOP lanes to free slots. Check free disk ≥ export size + 10 GB first; delete re-fetchable caches,
  never pointers or generations (skard 25 Oct: export and swap exhausted the disk).

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
  coordinator still running, or a registered coordinator with an assistant or user tool-result entry in the last
  20 minutes (read from a bounded transcript tail; metadata writes do not count), or an unexpired lease
  (`lane-keeper.py lease --owner NAME --minutes N`, `--release`): an executor or interactive session that holds
  integration takes a lease so the keeper starts no competing coordinator. Otherwise the keeper starts one (`engine`
  claude or codex) on the newest handover, at most one at a time, 10 minutes apart (also after a retire), 16 a day; if two
  successive runs leave the same actionable events pending it stops and writes COORDINATOR-STUCK.md.
- **Only actionable events start a coordinator.** HEAVY-WAIT, QUIET and reaped runaways (ORPHAN, STOPPED-BUT-RUNNING,
  STALE-PREVIEW) are listed for the next coordinator but never wake one; a lane silent for 3 h becomes STALLED, which
  does. Measured 1 Nov 00:01–02:32: a successor every ~2 min, each reading ~350k tokens to acknowledge a heavy-slot wait,
  used all 48 daily starts 28 minutes before a release cutoff.
- **Fixed times are events too.** `"schedule"` in .coordinator.json fires CUTOFF / RELEASE-DAY / DEADLINE events at local
  times (optionally on given weekdays or one date). The one night that met its goal (31 Oct–1 Nov) had a 03:00 scope
  freeze and 06:00/07:00 checks; no lane event says «it is 03:00».
- An interactive coordinator arms `lane-keeper.py watch` under a Monitor (re-arm on expiry) instead of lane-audit --loop.
- **Handing over: write the handover, then `lane-keeper.py retire [--config F] [--session ID]`.** A retiring coordinator
  writes `<state>/<id>-HANDOVER-<YYYYMMDD-HHMM>.md` and runs retire (ID defaults to `$CLAUDE_CODE_SESSION_ID`). Without it
  the successor waited ~20 minutes, because the retiring transcript kept it «live». Retire appends the session ID to `keeper/retired` and
  removes `~/.agents/coordinators/<ID>.json`; named retired IDs never count again, even if late transcript or registry
  writes follow the marker. Older anonymous signals stop counting, running watches exit, and the next
  tick starts a successor once `min_gap_minutes` has passed since the last start. Retire refuses (exit 2) when the newest
  handover exceeds 2,800 characters or is run-together text (more than 12 characters per word, or 60+ character tokens
  that are not paths): Codex met a «400 words» cap by deleting spaces (82 «words» in 2,028 characters, 1 Nov).
  `--force` is for a watchdog retiring a dead session.
- **Registration requires a real coordinator signal.** `scripts/coordinator-hook.py` is the versioned source for
  `~/.agents/bin/coordinator-hook.py`; install it alongside keeper updates. It registers only the session owning a
  fresh `watch.heartbeat` (session ID and PID, written by `watch`) or `coordinator.session` with a live headless PID.
  `tick`, `retire --session <other>`, `status`, `ack`, `launch.sh`, and handover text mentioning `watch` do not
  register the caller. All hooks refuse retired IDs, removing any late registration; `watch` removes its own
  heartbeat on exit. A heartbeat with a dead watch PID stops counting immediately; a stale heartbeat expires after
  three minutes. A quiet transcript then permits a successor. Blank legacy heartbeats retain the three-minute expiry.
- **Headless coordinator: log and takeover.** The keeper runs `claude -p … --session-id <uuid> --output-format stream-json
  --verbose`, so `keeper/coordinator-<stamp>.log` fills while it works (`tail -f`). The session id is in
  `keeper/coordinator.session` and in keeper.log (`COORDINATOR-SESSION <id> … claude --resume <id>`). Its prompt says to
  integrate in a worktree, run `git status` before touching the main checkout, never reset/checkout --/restore/stash/clean
  a checkout it did not create (one ran `git reset --hard` in skard's main checkout on 30 Oct), and to end with retire.
  To take over interactively: `kill -9 $(cat keeper/coordinator.pid)` (SIGKILL to the claude process only, so the
  detached jobs it started survive), then `cd <repo> && claude --resume <id>`, write
  `~/.agents/coordinators/<new-session-id>.json` with `{"repo": …, "transcript": <that session's .jsonl>}`, and run
  `lane-keeper.py watch` under a Monitor.
- `codex exec resume <unknown id>` silently starts a new session with no brief (found testing the keeper): resume only
  ids whose session file exists under ~/.codex/sessions.
- **alif (Linux) runs its own keeper**: systemd `skard-lane-keeper.timer` (every 2 min, root, `CODEX_HOME=/opt/alif/.codex`)
  ticks `/srv/agents/skard/limbic/…/lane-keeper.py` with `/srv/agents/skard/keeper.json`: campaign
  `/srv/agents/skard/campaign-research` (launch.sh is a symlink to scripts/launch.sh), code snapshot `/srv/agents/skard/code`,
  `max_starts_per_day: 0` (the Mac is the only coordinator), `detach: systemd-run --scope` with `KillMode=process` (else lanes die with the tick), batched `notify` mail ≤1/30 min.
  Linux has no file birth time, so launch.sh/resume.sh write `session-at`; drive-loop.sh is bash, sleepsafe from `$SLEEPSAFE`.
- The Mac keeper mirrors it through `remote_events` in `.coordinator.json` (`ssh alif-ts`, byte offset in state.json):
  events show as `alif:<lane>`; `lane-keeper.py ack alif-ts:/srv/agents/skard/campaign-research/<lane> "<note>"` writes
  HANDLED there, and the mirrored event clears once the alif keeper has cleared it. Update alif with `git -C /srv/agents/skard/limbic pull`.

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

## Running unattended: who coordinates (1 Nov 2026)

Compared over five skard nights (skard-state/orchestration-review-20261101/FINDINGS.md): four Claude-coordinated
nights produced no release by morning three times and one release after three cycles; the Codex-coordinated night
(31 Oct–1 Nov) shipped a release 12 minutes past its 07:00 goal with 23 lane integrations and no owner message after
23:48. Claude coordinators used 180–550M tokens a night; the Codex root 66M (99% cached). Part of that is the tooling the
Claude nights built; the shape is what carries over:
- **Codex is the default coordinator engine** (`"coordinator": {"engine": "codex"}`); Claude takes design lanes and
  interactive design rounds. Switch only when Codex has no credit.
- **The coordinator is short-lived and stateless.** It is started per batch of actionable events, reads the newest
  handover and the event list, acts, writes a bounded handover, retires. It never watches a job, so its context never
  grows; the keeper is the only thing that runs all night, and it spends no tokens.
- **One writer.** An integration or release executor takes a lease for its run and releases it at the end; supervisors
  read and never integrate. A backstop chat is not needed: the keeper's liveness, STALLED and schedule events are the
  backstop. Four overlapping authorities on 31 Oct–1 Nov spent most of their text agreeing who was in charge.
- **Deadlines go in the schedule**, on fixed release days; nights run deadline-free reading lanes on the server.
- **Every coordinator keeps INCIDENTS.md**: one dated entry per failure, in the turn it happens (Codex wrote 30
  STATUS files and one incident line that night).
- **Guards are config, not conversation.** A coordinator does not raise `max_starts_per_day` or other caps to keep
  itself going (16→24→48 on 31 Oct); a cap it hits is an incident.

## Budget, autonomy and the owner's view (1 Nov 2026)

The owner shares both subscriptions with other work, and the allowance varies. A campaign does not take what is
available; it takes its paced share and reports.
- **Usage is measured, not guessed.** Codex sessions log the account's weekly window (`rate_limits.primary`); headless
  Claude logs `rate_limit_event` (five-hour and seven-day). `scripts/budget.py` reads the newest snapshot.
- **Pacing line:** `min(100 − reserve, (100 − reserve) × elapsed share of the window + slack)` for Claude; Codex uses `"pacing": "ceiling"` (line = 100 − reserve), because its allowance refills unpredictably and spacing out wastes it (Stian, 1 Nov). Owner settings in
  `~/.agents/budget.json`: `budget.py reserve codex 40`, `budget.py mode claude off|low|normal|burn` (burn = use up to the
  reserve now; it ends at the next reset). Measured 1 Nov: one overnight campaign used 26% of the Codex week in nine hours.
- **Autonomy window:** `budget.py allow [hours]` (default 16). After it, the keeper starts no coordinator and no fresh lane
  session; running work continues. AUTONOMY-ENDING fires an hour before.
- **Enforcement points:** the keeper gates coordinator starts, fresh sessions, resumes, NEXT-COMMAND and driver restarts;
  repeating drive-loops wait while `~/.agents/budget/HOLD-<engine>` exists (`<lane>/engine` = codex | claude | none);
  `--once` jobs never wait. One machine computes and mirrors hold files to the others (`budget.mirror`).
- **The owner sees it without asking:** the dashboard below, keeper/STATUS.md and STATUS.html (every 5 min) and a mailed digest at
  `digest.times`; budget and autonomy events go out with the batched notify mail. All of it costs no tokens.
- **The owner acts from one page: `scripts/dashboard.py`** (1 Nov 2026), a live local dashboard kept alive by launchd
  (`scripts/install-dashboard.sh <repo>/.coordinator.json`, run from the limbic main checkout; label
  `com.<user>.keeper-dashboard`, logs in `<state>/keeper/dashboard.{out,err}`; `--uninstall` removes it). It shows lanes on
  the Mac and remote hosts (`dashboard.remote`, read over ssh every 5 min with a timeout and cached), READY, today's
  integrations, keeper health, budget and autonomy, QUEUE.md, schedule, INCIDENTS.md tail and the repo's own panels
  (`dashboard.feeds`: commands that print `{"panels": […]}`). STATUS.html stays as the fallback and links to it; the digest
  links to it too. Actions are a whitelist in `dashboard.actions` (builtins: queue-move, owner-priority, lane-pause,
  lane-resume, autonomy-renew; or a fixed argv); a request passes structured arguments only, needs the per-start token
  embedded in the page, same Origin and a 127.0.0.1/localhost Host, and is appended to `keeper/dashboard-audit.jsonl`.
  Owner priorities (P0–P3) go to `<state>/OWNER-PRIORITIES.json`; the coordinator prompt names that file when it exists.
  Pause is `lane-keeper.py pause <lane>` (STOP + PAUSED: the drive-loop ends after its current step; the runaway check
  reaps a step still running 10 min later); `resume` only undoes such a pause and restarts the drive-loop under the tick
  lock. A coordinator's STOP and `--once` jobs are refused.

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
- **Do not STOP lanes to free heavy slots.** heavy.sh queues jobs for the available slots; lanes keep doing non-heavy work. The keeper reports `HEAVY-WAIT` instead of stopping a lane or treating a queued job as quiet.
- **heavy.sh is exclusive only for the release export and cycle** (skard 30 Oct: one slot for everything queued OCR calls an hour behind gates on a 14-core Mac). Other jobs share slots and wait only on the export, memory or disk.

## Holds, idle notices and handovers (review of 23–29 Oct, skard-state/orchestration-review-20261029)

- **A hold names its pool and the file that releases it** (Claude quota / Codex account / disk / heavy
  slot), and the step that sets it removes it. A Claude-quota pause became «paused every lane» (28 Oct), and a
  PAUSE file from a finished export kept a harvest idle for 18 h (29 Oct). A held lock means wait, never stop.
- **Idle notices are not turns.** An idle notification that repeats a message just received gets no reply:
  one music thread spent 52 replies and ~33M tokens acknowledging echoes (27–28 Oct).
- **A handover bullet that says «remaining», «left» or «next» names its owner:** «→ lane X», «→ waits on
  <file>», or RETIRED. A Claude thread coordinator is a lane too: handing off is not done, and waiting on a
  message instead of a file stalled lanes for half a day (27 Oct).
- **One live session per lane:** launch.sh and resume.sh share a lane lock and refuse a live pid or a launch within the last 120 seconds; logs append (double sessions in two
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
