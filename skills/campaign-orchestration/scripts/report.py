"""Keeper status for the owner: keeper/STATUS.md and STATUS.html every tick, a mailed digest at configured times.

Built from the keeper's own state, lane folders and git; no model calls. Called by lane-keeper.py tick.
"""
import html, subprocess, sys, time
from pathlib import Path

import budget

STATUS_EVERY = 300   # seconds between rewrites; lane folders are cheap but not free


def _lk(K):
    return sys.modules[type(K).__module__]


def lanes(K):
    lk = _lk(K)
    rows = []
    for C, ds in K.lanes():
        for d in ds:
            drivers = lk.driver_dirs(d)
            session, driver = lk.alive(d / 'pid'), any(lk.alive(p / 'driver.pid') for p in drivers)
            close = K.closing(d)
            if (d / 'STOP').exists():
                state = 'stopped'
            elif session or driver:
                state = 'running' + (' (session)' if session else '') + (' (driver)' if driver else '')
            elif any(lk.mtime(p / 'DRIVER-FAILED.md') > K.acked(d) for p in drivers):
                state = 'DRIVER-FAILED'
            elif close:
                state = close.replace('.md', '')
                if state.startswith('READY') and lk.mtime(d / 'HANDLED') >= lk.mtime(d / close):
                    state = 'handled'
            else:
                state = 'idle'
            age = (lk.now() - max(lk.newest_in(p) for p in drivers)) / 60
            if state in ('DONE', 'RETIRED') and age > 24 * 60:
                continue
            note = K.st['lanes'].get(str(d), {}).get('deferred', '')
            rows.append({'lane': d.name, 'campaign': C.name, 'state': state, 'quiet_min': age, 'note': note})
    order = {'running': 0, 'stopped': 2, 'idle': 3}
    rows.sort(key=lambda r: (order.get(r['state'].split()[0], 1), r['quiet_min']))
    return rows


def main_log(K, hours=12):
    try:
        out = subprocess.run(['git', '-C', str(K.repo), 'log', 'main', f'--since={hours} hours ago', '--format=%ad %s',
                              '--date=format:%a %H:%M'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [l for l in out.splitlines() if l]


def collect(K):
    lk = _lk(K)
    info = set(K.cfg.get('coordinator', {}).get('info_kinds', lk.INFO_KINDS))
    events = list(K.st['events'].values())
    log = main_log(K)
    L = K.lease() if hasattr(K, 'lease') else None
    return {
        'time': lk.stamp(),
        'budget': budget.status_lines() if K.budget_on() else ['budget gating off'],
        'disk_gb': lk.disk_free_gb(str(K.repo)),
        'swap_gb': lk.swap_used_gb(),
        'coordinator': K.coordinator_live() or 'none live',
        'lease': f'{L["owner"]} until {lk.stamp(L["until"])}' if L else '',
        'decisions': [e for e in events if e.get('kind') not in info],
        'info': [e for e in events if e.get('kind') in info],
        'lanes': lanes(K),
        'commits': log,
        'integrations': [l for l in log if any(w in l for w in ('Integrate', 'Apply', 'Release', 'release'))],
    }


def text(S):
    out = [f'# Keeper status {S["time"]}', '']
    out += S['budget']
    out += [f'disk free {S["disk_gb"]:.0f} GB; swap {S["swap_gb"]:.1f} GB; coordinator: {S["coordinator"]}'
            + (f'; lease {S["lease"]}' if S['lease'] else ''), '']
    out.append(f'## Waiting on a decision ({len(S["decisions"])})')
    out += [f'- {e["kind"]} {e.get("name", "")}: {e["detail"]}' for e in S['decisions'][-20:]] or ['- nothing']
    run = [r for r in S['lanes'] if r['state'].startswith('running')]
    out += ['', f'## Lanes ({len(run)} running of {len(S["lanes"])})']
    for r in S['lanes']:
        q = f'{r["quiet_min"]:.0f} min' if r['quiet_min'] < 120 else f'{r["quiet_min"] / 60:.1f} h'
        out.append(f'- {r["lane"]}: {r["state"]}, last write {q} ago' + (f'; deferred: {r["note"]}' if r['note'] else ''))
    out += ['', f'## Main, last 12 h ({len(S["commits"])} commits)']
    out += [f'- {l}' for l in S['integrations'][:25]] or ['- no integrations']
    if S['info']:
        out += ['', f'{len(S["info"])} informational events (waits, quiet lanes, budget notices)']
    return '\n'.join(out) + '\n'


CSS = """:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e2dfd8;--ok:#2f7d4f;--hold:#b4441f;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--ok:#6cc08c;--hold:#f08a64;--card:#1f1f1c}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif;margin:0 auto;max-width:980px;padding:16px}
h1{font-size:22px;margin:4px 0 12px}h2{font-size:16px;margin:22px 0 8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.ok{color:var(--ok)}.hold{color:var(--hold)}.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:500;font-size:13px}code{font-size:13px}"""


def page(S, live=''):
    e = html.escape
    cards = ''.join(f'<div class="card {"hold" if ": HOLD" in b or "expired" in b or "not granted" in b else "ok"}">{e(b)}</div>'
                    for b in S['budget'])
    cards += (f'<div class="card">disk {S["disk_gb"]:.0f} GB free · swap {S["swap_gb"]:.1f} GB<br>'
              f'<span class="mut">coordinator: {e(S["coordinator"])}{" · lease " + e(S["lease"]) if S["lease"] else ""}</span></div>')
    dec = ''.join(f'<tr><td>{e(x["kind"])}</td><td>{e(x.get("name", ""))}</td><td>{e(x["detail"])}</td></tr>'
                  for x in S['decisions'][-30:]) or '<tr><td colspan=3 class="mut">nothing</td></tr>'
    def q(m): return f'{m:.0f} min' if m < 120 else f'{m / 60:.1f} h'
    ln = ''.join(f'<tr><td>{e(r["lane"])}</td><td>{e(r["state"])}</td><td>{q(r["quiet_min"])}</td><td class="hold">{e(r["note"])}</td></tr>'
                 for r in S['lanes'])
    mn = ''.join(f'<li>{e(l)}</li>' for l in S['integrations'][:30]) or '<li class="mut">no integrations</li>'
    run = sum(r['state'].startswith('running') for r in S['lanes'])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300"><title>Keeper status</title><style>{CSS}</style></head><body>
<h1>Keeper status <span class="mut">{e(S["time"])}</span>{f' <a href="{e(live)}" style="font-size:14px">live dashboard</a>' if live else ''}</h1>
<div class="cards">{cards}</div>
<h2>Waiting on a decision ({len(S["decisions"])})</h2><table><tr><th>Kind</th><th>Lane</th><th>Detail</th></tr>{dec}</table>
<h2>Lanes ({run} running of {len(S["lanes"])})</h2><table><tr><th>Lane</th><th>State</th><th>Last write</th><th>Deferred</th></tr>{ln}</table>
<h2>Main, last 12 h ({len(S["commits"])} commits)</h2><ul>{mn}</ul>
</body></html>"""


def dashboard_url(K):
    D = K.cfg.get('dashboard')
    return f'http://127.0.0.1:{D.get("port", 4170)}/' if isinstance(D, dict) else ''


def due_digest(K, t=None):
    """The configured HH:MM most recently passed today that has not been sent."""
    times = K.cfg.get('digest', {}).get('times', [])
    lt = time.localtime(t or time.time())
    today = time.strftime('%F', lt)
    sent = K.st.setdefault('digest', {})
    for hhmm in sorted(times, reverse=True):
        h, m = map(int, hhmm.split(':'))
        if (lt.tm_hour, lt.tm_min) >= (h, m):
            return None if sent.get(hhmm) == today else hhmm
    return None


def tick(K):
    lk = _lk(K)
    last = K.st.get('status_written', 0)
    slot = due_digest(K)
    if lk.now() - last < STATUS_EVERY and not slot:
        return
    S = collect(K)
    body = text(S)
    if not K.dry:
        (K.k / 'STATUS.md').write_text(body)
        (K.k / 'STATUS.html').write_text(page(S, dashboard_url(K)))
    K.st['status_written'] = lk.now()
    if not slot:
        return
    cmd = (K.cfg.get('notify') or {}).get('cmd')
    K.st['digest'][slot] = time.strftime('%F')
    if K.dry or not cmd:
        return
    subject = f'{K.cfg.get("name", K.repo.name)}: {len(S["decisions"])} decisions, ' \
              f'{sum(r["state"].startswith("running") for r in S["lanes"])} lanes running, {len(S["integrations"])} integrations'
    try:
        ok = subprocess.run([cmd, subject, body + (f'\nDashboard: {dashboard_url(K)}' if dashboard_url(K) else '') + f'\nFull page: {K.k / "STATUS.html"}\n'], capture_output=True, timeout=90).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    K.log(f'DIGEST {slot} {"sent" if ok else "failed"}')
