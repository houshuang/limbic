#!/usr/bin/env python3
"""Token budget and autonomy window for campaign keepers. Spends no tokens.

Both subscriptions report their own usage: every Codex session logs the account's weekly window
(`rate_limits.primary`: used_percent, window_minutes, resets_at), and a headless Claude run logs
`rate_limit_event` with five_hour and seven_day utilization and resetsAt. The newest snapshot is the
account's state, whatever project produced it, so other work counts against the same line.

The pacing line lets a campaign spend at most its share of the time elapsed in the window:

    line = min(cap, cap * elapsed_fraction + slack),   cap = 100 - reserve

so early in the week it cannot burn the week, and near a reset it may use what would otherwise lapse,
never the reserve. Per engine, "pacing": "ceiling" drops the time term (line = cap): for an allowance that
refills unpredictably (Codex, Stian 1 Nov 2026), spacing out wastes capacity; only the reserve is kept. Modes: off (nothing), low (half the line), normal, burn (up to cap now; ends at the
reset that was current when it was set).

Autonomy: after `allow` the keeper may start coordinators and fresh lane sessions for autonomy_hours
(default 16); then it starts nothing new until the owner renews. Running lanes and drivers continue.

Policy file ~/.agents/budget.json (override with AGENTS_BUDGET). Hold files for drivers:
~/.agents/budget/HOLD-<engine> (written by the keeper each tick; drive-loop waits while one exists).

    budget.py status
    budget.py reserve codex|claude PCT
    budget.py mode codex|claude off|low|normal|burn
    budget.py pacing codex|claude ceiling|linear
    budget.py allow [HOURS|forever]  # renew autonomy (default autonomy_hours); forever = until `stop`
    budget.py stop                   # end autonomy now
"""
import glob, json, math, os, sys, time
from pathlib import Path

POLICY = Path(os.environ.get('AGENTS_BUDGET', Path.home() / '.agents/budget.json'))
HOLD_DIR = Path(os.environ.get('AGENTS_BUDGET_DIR', Path.home() / '.agents/budget'))
CODEX_HOME = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
ENGINES = ('codex', 'claude')
MODES = ('off', 'low', 'normal', 'burn')
DEFAULT = {
    'reserve': {'codex': 40, 'claude': 50},
    'mode': {'codex': 'normal', 'claude': 'normal'},
    'mode_until': {},
    'pacing': {'codex': 'ceiling', 'claude': 'linear'},
    'slack': 5,
    'autonomy_hours': 16,
    'autonomy_until': 0,
    'claude_logs': [str(Path.home() / 'src/*-state/keeper/coordinator-*.log'),
                    str(Path.home() / 'src/*-state/campaign-*/*/claude*.log')],
    'stale_hours': 12,
}
TAIL = 400_000


def now():
    return time.time()


def load(path=None):
    path = Path(path or POLICY)
    P = json.loads(json.dumps(DEFAULT))
    try:
        user = json.loads(path.read_text())
    except (OSError, ValueError):
        return P
    for k, v in user.items():
        P[k] = {**P[k], **v} if isinstance(P.get(k), dict) and isinstance(v, dict) else v
    return P


def save(P, path=None):
    path = Path(path or POLICY)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(P, indent=1) + '\n')
    tmp.replace(path)


def tail_lines(path, n=TAIL):
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - n))
            return fh.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return []


def codex_snapshot(home=None, files=40):
    """Newest rate_limits seen in any Codex session log under CODEX_HOME."""
    home = Path(home or CODEX_HOME)
    logs = sorted(glob.glob(str(home / 'sessions/*/*/*/*.jsonl')), key=lambda f: os.path.getmtime(f), reverse=True)[:files]
    best = None
    for f in logs:
        for line in reversed(tail_lines(f)):
            if '"rate_limits"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            R = (d.get('payload') or {}).get('rate_limits') or {}
            p = R.get('primary') or {}
            if p.get('used_percent') is None or not p.get('resets_at'):
                continue
            t = os.path.getmtime(f)
            ts = d.get('timestamp')
            if ts:
                try:
                    import datetime as dt
                    t = dt.datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                except ValueError:
                    pass
            snap = {'engine': 'codex', 'used': float(p['used_percent']), 'window_min': int(p.get('window_minutes') or 10080),
                    'resets_at': float(p['resets_at']), 't': t, 'source': f,
                    'blocked': bool(R.get('rate_limit_reached_type'))}
            if not best or snap['t'] > best['t']:
                best = snap
            break
        if best and best['t'] >= os.path.getmtime(f):
            break   # older files cannot hold a newer snapshot
    return best


def claude_snapshot(globs):
    """Newest rate_limit_event (unifiedWindows) in headless Claude stream-json logs."""
    files = sorted({f for g in globs for f in glob.glob(g)}, key=lambda f: os.path.getmtime(f), reverse=True)[:300]
    for f in files:
        lines = tail_lines(f)
        if not any('"unifiedWindows"' in l for l in lines):
            continue   # Codex-engine coordinator logs share the glob
        for line in reversed(lines):
            if '"unifiedWindows"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            W = (d.get('rate_limit_info') or {}).get('unifiedWindows') or {}
            week, five = W.get('seven_day') or {}, W.get('five_hour') or {}
            if week.get('utilization') is None:
                continue
            return {'engine': 'claude', 'used': float(week['utilization']) * 100, 'window_min': 10080,
                    'resets_at': float(week.get('resetsAt') or 0), 't': os.path.getmtime(f), 'source': f,
                    'blocked': week.get('status') == 'rejected', 'five_blocked': five.get('status') == 'rejected',
                    'five_used': float(five.get('utilization') or 0) * 100, 'five_resets_at': float(five.get('resetsAt') or 0)}
    return None


def snapshot(engine, P):
    return codex_snapshot() if engine == 'codex' else claude_snapshot(P['claude_logs'])


def pacing_line(S, reserve, slack, mode, t, pacing='linear'):
    cap = max(0.0, 100.0 - reserve)
    if mode == 'off':
        return 0.0
    if mode == 'burn':
        return cap
    if pacing == 'ceiling':
        return cap / 2 if mode == 'low' else cap
    window = S['window_min'] * 60
    if t >= S['resets_at']:     # a reset passed since the snapshot: a new window has barely begun
        frac = min(1.0, (t - S['resets_at']) / window)
    else:
        frac = min(1.0, max(0.0, 1 - (S['resets_at'] - t) / window))
    line = cap * frac * (0.5 if mode == 'low' else 1.0) + slack
    return min(cap, line)


def mode_of(P, engine, t):
    m = P['mode'].get(engine, 'normal')
    until = P.get('mode_until', {}).get(engine)
    if m == 'burn' and until and t >= until:
        return 'normal'
    return m


def gate(engine, P=None, t=None, snap=None):
    """(ok, reason, info). Unknown usage is allowed and reported, not blocked: no data is not overspend."""
    P = P or load(); t = t or now()
    S = snap if snap is not None else snapshot(engine, P)
    m = mode_of(P, engine, t)
    reserve = P['reserve'].get(engine, 50)
    info = {'engine': engine, 'mode': m, 'reserve': reserve}
    if m == 'off':
        return False, f'{engine} mode off', info
    if not S:
        return True, f'{engine}: no usage data seen', info
    used = S['used'] if t < S['resets_at'] else 0.0
    line = pacing_line(S, reserve, P['slack'], m, t, P.get('pacing', {}).get(engine, 'linear'))
    info.update(used=used, line=line, resets_at=S['resets_at'], age_h=(t - S['t']) / 3600)
    if S.get('blocked') and t < S['resets_at']:
        return False, f'{engine} limit reached; resets {time.strftime("%a %H:%M", time.localtime(S["resets_at"]))}', info
    if (S.get('five_blocked') or S.get('five_used', 0) >= 95) and t < S.get('five_resets_at', 0):
        return False, (f'{engine} five-hour window {S.get("five_used", 0):.0f}% used; resets '
                       f'{time.strftime("%H:%M", time.localtime(S["five_resets_at"]))}'), info
    if used > line:
        return False, (f'{engine} {used:.0f}% used, pacing line {line:.0f}% (reserve {reserve}%, '
                       f'resets {time.strftime("%a %H:%M", time.localtime(S["resets_at"]))})'), info
    return True, f'{engine} {used:.0f}% used of a {line:.0f}% line', info


def autonomy(P=None, t=None):
    P = P or load(); t = t or now()
    until = float(P.get('autonomy_until') or 0)
    if until < 0:
        return True, math.inf
    return t < until, until


def write_holds(results, hold_dir=None):
    """results: {engine: (ok, reason)}. A hold file exists exactly while the engine is held."""
    d = Path(hold_dir or HOLD_DIR); d.mkdir(parents=True, exist_ok=True)
    for engine, (ok, reason) in results.items():
        f = d / f'HOLD-{engine}'
        if ok:
            f.unlink(missing_ok=True)
        elif not f.exists() or f.read_text().strip() != reason:
            f.write_text(reason + '\n')


def fmt_time(t):
    if t == math.inf:
        return 'stopped by the owner (no end)'
    return time.strftime('%a %d %b %H:%M', time.localtime(t)) if t else '—'


def status_lines(P=None, t=None):
    P = P or load(); t = t or now()
    ok, until = autonomy(P, t)
    out = [f'autonomy: {"until " + fmt_time(until) if ok else "expired " + fmt_time(until) if until else "not granted"}'
           f' (renew: budget.py allow)']
    for e in ENGINES:
        g, reason, info = gate(e, P, t)
        age = f', data {info["age_h"]:.1f} h old' if 'age_h' in info else ''
        out.append(f'{e}: {"OK" if g else "HOLD"}: {reason}; mode {info["mode"]}, reserve {info["reserve"]}%{age}')
    return out


def main(argv):
    cmd = argv[0] if argv else 'status'
    P = load()
    if cmd == 'status':
        print('\n'.join(status_lines(P)))
        return 0
    if cmd == 'reserve' and len(argv) == 3 and argv[1] in ENGINES:
        P['reserve'][argv[1]] = max(0, min(100, int(argv[2])))
    elif cmd == 'mode' and len(argv) == 3 and argv[1] in ENGINES and argv[2] in MODES:
        P['mode'][argv[1]] = argv[2]
        P.setdefault('mode_until', {}).pop(argv[1], None)
        if argv[2] == 'burn':
            S = snapshot(argv[1], P)
            if S and S['resets_at'] > now():
                P['mode_until'][argv[1]] = S['resets_at']
    elif cmd == 'pacing' and len(argv) == 3 and argv[1] in ENGINES and argv[2] in ('ceiling', 'linear'):
        P.setdefault('pacing', {})[argv[1]] = argv[2]
    elif cmd == 'allow':
        if len(argv) > 1 and argv[1] == 'forever':
            P['autonomy_until'] = -1
        else:
            hours = float(argv[1].rstrip('h')) if len(argv) > 1 else float(P['autonomy_hours'])
            P['autonomy_until'] = now() + hours * 3600
    elif cmd == 'stop':
        P['autonomy_until'] = 0
    else:
        print(__doc__); return 2
    save(P)
    print('\n'.join(status_lines(P)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
