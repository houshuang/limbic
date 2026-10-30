#!/usr/bin/env python3
"""Enforced cost guard for campaign lanes (no model tokens).

For each lane session: current context per turn, tokens since launch, tokens/hour.
Writes COST.md. Exit status 3 names lanes over a hard cap so the watcher can force a handoff.
Caps: context > 150k tokens per turn, or > 120 turns in one session, or > 15M gross tokens/hour sustained over 30+ min.
"""
import json, glob, os, sys, time, re
C = '/Users/stian/src/skard-state/campaign-20261023'
CTX, TURNS, RATE = 150_000, 120, 15_000_000
def session_file(lane):
    logs = sorted(glob.glob(f'{C}/{lane}/codex*.log'), key=os.path.getmtime)
    for log in reversed(logs):
        m = re.search(r'session id: (\S+)', open(log, errors='ignore').read(4000))
        if m:
            f = glob.glob(os.path.expanduser(f'~/.codex/sessions/*/*/*/*{m.group(1)}.jsonl'))
            if f: return f[0], os.path.getmtime(log)
    return None, None
rows, over = [], []
for lane in sys.argv[1:]:
    f, started = session_file(lane)
    if not f: continue
    turns = 0; first = last = None; ctx = 0
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        p = d.get('payload', {})
        if p.get('type') == 'token_count' and p.get('info'):
            ts = d.get('timestamp', '')
            if ts and ts >= time.strftime('%Y-%m-%dT%H:%M', time.gmtime(started - 60)):
                turns += 1; u = p['info']['total_token_usage']['total_tokens']
                first = first if first is not None else u; last = u
                ctx = p['info'].get('last_token_usage', {}).get('input_tokens', 0)
    if first is None: continue
    hours = max((time.time() - started) / 3600, 0.05); used = last - first
    cap = int(open(f'{C}/{lane}/ctx-cap').read()) if os.path.exists(f'{C}/{lane}/ctx-cap') else CTX
    flag = ctx > cap or turns > TURNS or (hours >= 0.5 and used / hours > RATE)
    rows.append(f'| {lane} | {ctx:,} | {turns} | {used:,} | {used/hours:,.0f} | {"OVER" if flag else ""} |')
    if flag: over.append(lane)
open(f'{C}/COST.md', 'w').write(time.strftime('%H:%M') + ' current session segment\n\n| lane | ctx/turn | turns | tokens | tokens/h | cap |\n|---|---|---|---|---|---|\n' + '\n'.join(rows) + '\n')
print(' '.join(over)); sys.exit(3 if over else 0)
