#!/usr/bin/env python3
"""Run the owner's-eye overseer: a headless Claude session that looks at how the campaign is going, asks the
questions the owner would ask, acts within the authority its prompt grants, and appends a short report.

    overseer.py run --config REPO/.coordinator.json      (launchd, hourly)
    overseer.py review --config REPO/.coordinator.json   (launchd, every 2 min)
    overseer.py status --config REPO/.coordinator.json

Config (in .coordinator.json): "overseer": {"prompt": PATH, "state": DIR, "report": PATH, "model": "opus",
"timeout_minutes": 40, "reviews": [{"request": PATH, "review": PATH}], "review_model": "fable"}. One run at a time (lock file); a run is skipped while the Claude budget refuses.
Coordinators react to events; the overseer looks at outcomes (books per hour, idle capacity, waste, quality)
because on 1 Nov 2026 the owner had to ask the same questions by hand all evening.

`review` answers a lane's REVIEW-REQUEST within minutes instead of at the next hourly run: on 2 Nov every factory fix
waited about an hour for review, and a lane that launched its own review from inside its sandbox got "Not logged in"
and nobody noticed. A failed review is reported in the overseer report and retried at most every 30 minutes.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(config):
    cfg = json.loads(Path(config).read_text())
    o = cfg.get('overseer') or {}
    for key in ('prompt', 'state', 'report'):
        if not o.get(key):
            raise SystemExit(f'overseer.{key} missing in {config}')
    return cfg, o


def budget_ok():
    r = subprocess.run([sys.executable, str(HERE / 'budget.py'), 'status'], capture_output=True, text=True, timeout=60)
    line = next((l for l in r.stdout.splitlines() if l.startswith('claude:')), '')
    return line.startswith('claude: OK'), line


def run(config):
    cfg, o = load(config)
    state = Path(o['state']); state.mkdir(parents=True, exist_ok=True)
    lock = open(state / 'run.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('overseer: previous run still active'); return 0
    ok, line = budget_ok()
    stamp = time.strftime('%Y%m%d-%H%M')
    if not ok:
        with open(o['report'], 'a') as f:
            f.write(f'\n## {time.strftime("%Y-%m-%d %H:%M")} · overseer skipped: {line}\n')
        return 0
    prompt = Path(o['prompt']).read_text().replace('{REPORT}', o['report']).replace('{NOW}', time.strftime('%Y-%m-%d %H:%M %Z'))
    log = state / f'run-{stamp}.log'
    env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY')}
    cmd = ['claude', '-p', prompt, '--model', o.get('model', 'opus'), '--permission-mode', 'bypassPermissions',
           '--output-format', 'stream-json', '--verbose']
    with open(log, 'w') as out:
        try:
            p = subprocess.run(cmd, cwd=cfg.get('repo') or str(Path(config).parent), stdin=subprocess.DEVNULL,
                               stdout=out, stderr=subprocess.STDOUT, env=env, timeout=o.get('timeout_minutes', 40) * 60)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    (state / 'last.json').write_text(json.dumps({'t': time.time(), 'exit': code, 'log': str(log)}) + '\n')
    if code != 0:
        with open(o['report'], 'a') as f:
            f.write(f'\n## {time.strftime("%Y-%m-%d %H:%M")} · overseer run failed ({code}); log {log}\n')
    return 0


REVIEW_PROMPT = """A code review was requested in {request}. It is {now}. Read the request, the previous review {review} \
if it exists, and the diff the request names. Review read-only: correctness bugs, data loss, wrong terminal states, \
rows that become unclaimable, double processing, bypassed holds, rate-limit and memory safety. Check that every earlier \
must-fix the request claims to fix is fixed with a test that fails on the old code. Run only the focused tests for \
the changed files and /usr/bin/python3 -m py_compile on changed Python files; never run whole-graph, native or full \
suites (they hold the shared disk for many minutes), and stop any single command after 5 minutes. Do not flag style, cosmetics, hypothetical future mutation or \
speculative performance. Be decisive: this review gates production. Do not edit code or commit. If {review} exists, \
copy it to {backup} first. Then write {review}: a title naming the reviewed commit, one paragraph on what you ran, \
`## Verdict` with exactly APPROVE, APPROVE after N must-fix, or REJECT, then `## Findings`, numbered, each marked \
«must fix» or «should fix» with file:line and the concrete fix. Under 900 words."""


def review_due(item, seen, now):
    request, review = Path(item['request']), Path(item['review'])
    if not request.exists():
        return False
    mtime = request.stat().st_mtime
    if review.exists() and review.stat().st_mtime >= mtime:
        return False
    last = seen.get(str(request)) or {}
    if last.get('mtime') != mtime:
        return True
    return last.get('failures', 0) < 3 and now - last.get('t', 0) >= 1800


def review(config):
    cfg, o = load(config)
    state = Path(o['state']); state.mkdir(parents=True, exist_ok=True)
    lock = open(state / 'review.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0
    ledger = state / 'reviews.json'
    seen = json.loads(ledger.read_text()) if ledger.exists() else {}
    due = [i for i in o.get('reviews', []) if review_due(i, seen, time.time())]
    if not due:
        return 0
    ok, line = budget_ok()
    if not ok:
        return 0
    item = due[0]
    request, target = Path(item['request']), Path(item['review'])
    mtime = request.stat().st_mtime
    stamp = time.strftime('%Y%m%d-%H%M')
    prompt = REVIEW_PROMPT.format(request=request, review=target, now=time.strftime('%Y-%m-%d %H:%M %Z'),
                                  backup=target.with_name(f'{target.stem}-before-{stamp}.md'))
    log = state / f'review-{stamp}.log'
    env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY')}
    cmd = ['claude', '-p', prompt, '--model', o.get('review_model', 'fable'), '--permission-mode', 'bypassPermissions',
           '--output-format', 'stream-json', '--verbose']
    with open(log, 'w') as out:
        try:
            code = subprocess.run(cmd, cwd=item.get('cwd') or cfg.get('repo') or str(Path(config).parent),
                                  stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
                                  timeout=o.get('timeout_minutes', 40) * 60).returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    written = target.exists() and target.stat().st_mtime >= mtime
    entry = seen.get(str(request)) or {}
    failures = 0 if written else (entry.get('failures', 0) + 1 if entry.get('mtime') == mtime else 1)
    seen[str(request)] = {'mtime': mtime, 't': time.time(), 'exit': code, 'log': str(log), 'failures': failures}
    ledger.write_text(json.dumps(seen, indent=1) + '\n')
    if not written:
        with open(o['report'], 'a') as f:
            f.write(f'\n## {time.strftime("%Y-%m-%d %H:%M")} · review of {request} failed ({code}, attempt {failures} of 3); '
                    f'log {log}\n')
    return 0


def status(config):
    _, o = load(config)
    last = Path(o['state']) / 'last.json'
    print(last.read_text() if last.exists() else 'no run yet')
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 3 or args[0] not in ('run', 'review', 'status') or args[1] != '--config':
        raise SystemExit(__doc__)
    sys.exit({'run': run, 'review': review, 'status': status}[args[0]](args[2]))
