#!/usr/bin/env python3
"""Run the owner's-eye overseer: a headless Claude session that looks at how the campaign is going, asks the
questions the owner would ask, acts within the authority its prompt grants, and appends a short report.

    overseer.py run --config REPO/.coordinator.json      (launchd, hourly)
    overseer.py status --config REPO/.coordinator.json

Config (in .coordinator.json): "overseer": {"prompt": PATH, "state": DIR, "report": PATH, "model": "opus",
"timeout_minutes": 40}. One run at a time (lock file); a run is skipped while the Claude budget refuses.
Coordinators react to events; the overseer looks at outcomes (books per hour, idle capacity, waste, quality)
because on 1 Nov 2026 the owner had to ask the same questions by hand all evening.
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


def status(config):
    _, o = load(config)
    last = Path(o['state']) / 'last.json'
    print(last.read_text() if last.exists() else 'no run yet')
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 3 or args[0] not in ('run', 'status') or args[1] != '--config':
        raise SystemExit(__doc__)
    sys.exit(run(args[2]) if args[0] == 'run' else status(args[2]))
