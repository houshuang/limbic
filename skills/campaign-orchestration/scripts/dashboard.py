#!/usr/bin/env python3
"""The owner's live keeper dashboard: one local page, always current, from which he can also act. Standard library only
(runs under /usr/bin/python3 3.9 and 3.14). Spends no model tokens.

    dashboard.py serve [--config .coordinator.json] [--port N]   # 127.0.0.1 only; launchd keeps it alive
    dashboard.py scan --dirs D… [--disk P…] [--keeper K]         # lane/disk JSON; sent over ssh for remote hosts

Reads the repo's .coordinator.json like lane-keeper.py, plus an optional "dashboard" block:

    "dashboard": {
      "port": 4170,
      "feeds":   [{"argv": ["python3", "scripts/dashboard_feed.py"], "cwd": "<repo>", "every": 60, "timeout": 40}],
      "remote":  [{"name": "alif", "ssh": "alif-ts", "campaigns": ["/srv/…/campaign-research"], "disk": ["/srv/media", "/"],
                   "keeper": "/srv/…/keeper", "every": 300}],
      "actions": {"queue-move": {"builtin": "queue-move"}, "owner-priority": {"builtin": "owner-priority"},
                  "lane-pause": {"builtin": "lane-pause"}, "lane-resume": {"builtin": "lane-resume"},
                  "autonomy-renew": {"builtin": "autonomy-renew", "label": "Renew autonomy"},
                  "preview-start": {"argv": ["bash", "scripts/preview.sh", "restart"], "cwd": "<repo>",
                                    "env": {"SKARD_PREVIEW_PORT": "4190"}, "detach": true, "label": "Start preview"}},
      "links":   [{"label": "Preview", "url": "http://127.0.0.1:4190/"}]
    }

Actions are only those named in "actions": a request names one and passes structured arguments (a lane name found by
the scan, a queue item id, P0–P3); nothing from the browser becomes a command line. Every request must carry the token
embedded in the page at this start and come from the page's own origin; Host must be 127.0.0.1/localhost:<port>.
Every action is appended to <state>/keeper/dashboard-audit.jsonl.

A feed prints {"panels": [panel…]}; a panel is {"id", "title", "pills": [[label, n]…], "rows": [[label, value, url?]…],
"groups": [{"title", "open": bool, "columns": [...], "rows": [{"id", "cells": [...], "url"}]}], "links": [[label, url]],
"actions": [action names], "prioritise": bool}. With "prioritise", rows with an id get an owner-priority selector;
choices are stored in <state>/OWNER-PRIORITIES.json, which coordinators read.
"""
from driver_state import driver_dirs, owns_process
import hashlib, hmac, html, json, os, re, secrets, shlex, shutil, subprocess, sys, threading, time
from pathlib import Path

SCRIPTS = Path(globals().get('__file__', '.')).resolve().parent   # '-' when sent over ssh
DEFAULT_CONFIG = Path.home() / 'src/skard/.coordinator.json'
DEFAULT_PORT = 4170
CLOSING = re.compile(r'^(DONE\.md|READY-[0-9][0-9a-z]*\.md|DRIVER-FINISHED\.json|RETIRED)$')
ITEM = re.compile(r'^- \[(.)\]')
MARKS = {' ': 'open', '~': 'running', '!': 'waiting', 'x': 'done'}
PRIORITIES = ('P0', 'P1', 'P2', 'P3')
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
LOCAL_EVERY, DISK_SAMPLE_EVERY, DISK_KEEP = 20, 600, 72 * 3600
SSH = ['ssh', '-o', 'ConnectTimeout=8', '-o', 'BatchMode=yes']


def now():
    return time.time()


def mtime(p):
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return 0


def read(p, default=''):
    try:
        return Path(p).read_text(errors='replace')
    except OSError:
        return default


def tail(p, n=4096):
    try:
        with open(p, 'rb') as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - n))
            return fh.read().decode(errors='replace')
    except OSError:
        return ''


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def ago(t):
    if not t:
        return '—'
    s = max(0, now() - t)
    if s < 90: return f'{s:.0f} s'
    if s < 5400: return f'{s / 60:.0f} min'
    if s < 48 * 3600: return f'{s / 3600:.1f} h'
    return f'{s / 86400:.0f} d'


# ---------------------------------------------------------------- lanes (also run on remote hosts over ssh)

def alive(pidfile):
    """A live driver must match its recorded owner, argv and cwd."""
    if Path(pidfile).name == "driver.pid" and not owns_process(Path(pidfile).parent, read(pidfile)):
        return False
    try:
        pid = int(Path(pidfile).read_text().split()[0])
        if pid <= 0:
            return False
        os.kill(pid, 0)
    except (OSError, ValueError, IndexError):
        return False
    out = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    try:
        started = time.mktime(time.strptime(' '.join(out.split()), '%a %b %d %H:%M:%S %Y'))
    except ValueError:
        return True
    return started <= mtime(pidfile) + 5


def newest_file(d):
    best = (0, '')
    try:
        for e in os.scandir(d):
            if e.is_file() and not e.name.startswith('.'):
                t = e.stat().st_mtime
                if t > best[0]:
                    best = (t, e.name)
    except OSError:
        pass
    return best


def progress_line(d):
    """The last non-empty line of driver.log, else the newest file's name."""
    logs = [d / 'processing-driver.log', *(p / 'driver.log' for p in driver_dirs(d))]
    for path in sorted(logs, key=mtime, reverse=True):
        lines = [ANSI.sub('', l).strip() for l in tail(path).splitlines()]
        lines = [l for l in lines if l]
        if lines:
            return lines[-1][:200]
    return newest_file(d)[1]


def scan_lane(d, host):
    drivers = driver_dirs(d)
    s_alive, d_alive = alive(d / 'pid'), any(alive(p / 'driver.pid') for p in drivers)
    closing = [f for f in os.listdir(d) if CLOSING.match(f)]
    close = max(closing, key=lambda f: mtime(d / f)) if closing else ''
    readies = [f for f in closing if f.startswith('READY-')]
    ready = max(readies, key=lambda f: mtime(d / f)) if readies else ''
    handled = mtime(d / 'HANDLED')
    stop, paused = (d / 'STOP').exists(), (d / 'PAUSED').exists()
    spec = {}
    if (d / 'driver.cmd').exists():
        try:
            spec = json.loads(read(d / 'driver.cmd'))
        except ValueError:
            spec = {}
    if s_alive or d_alive:
        state = 'pausing' if stop else 'running'
    elif (d / 'RETIRED').exists():
        state, close = 'RETIRED', 'RETIRED'
    elif stop:
        state = 'paused' if paused else 'stopped'
    elif any(mtime(p / 'DRIVER-FAILED.md') > handled for p in drivers):
        state = 'DRIVER-FAILED'
    elif close:
        state = close.split('.')[0].split('-')[0]
        if state == 'READY' and handled >= mtime(d / close):
            state = 'handled'
    elif (d / 'HANDOFF.md').exists():
        state = 'handoff'
    else:
        state = 'idle'
    last = max(newest_file(p)[0] for p in drivers)
    return {
        'lane': d.name, 'campaign': d.parent.name, 'host': host, 'state': state,
        'session': s_alive, 'driver': d_alive,
        'engine': read(d / 'engine').strip() or ('codex' if (d / 'codex.log').exists() else ''),
        'model': read(d / 'model').strip().splitlines()[0][:40] if read(d / 'model').strip() else '',
        'last': last, 'progress': progress_line(d),
        'close': close, 'close_t': mtime(d / close) if close else 0,
        'ready': ready if ready and mtime(d / ready) > handled else '',
        'handled': read(d / 'HANDLED').strip()[:200] if handled and handled >= mtime(d / close) else '',
        'once': '--once' in spec.get('args', []), 'has_driver': bool(spec),
    }


def scan(dirs, host='local'):
    rows = []
    for c in dirs:
        C = Path(c)
        if not C.is_dir():
            continue
        for d in sorted(C.iterdir()):
            if d.is_dir() and ((d / 'pid').exists() or any((p / 'driver.pid').exists() for p in driver_dirs(d))):
                try:
                    rows.append(scan_lane(d, host))
                except OSError:
                    continue
    return rows


def disks(paths):
    out = []
    for p in paths:
        try:
            u = shutil.disk_usage(p)
            out.append({'path': p, 'free_gb': u.free / 2**30, 'total_gb': u.total / 2**30})
        except OSError:
            out.append({'path': p, 'free_gb': None, 'total_gb': None})
    return out


def scan_main(argv):
    opt = {'--dirs': [], '--disk': [], '--keeper': []}
    cur = None
    for a in argv:
        if a in opt:
            cur = a
        elif cur:
            opt[cur].append(a)
    host = os.uname().nodename
    keeper = opt['--keeper'][0] if opt['--keeper'] else ''
    print(json.dumps({'host': host, 't': now(), 'lanes': scan(opt['--dirs'], host), 'disk': disks(opt['--disk']),
                      'keeper_heartbeat': mtime(Path(keeper) / 'heartbeat') if keeper else 0,
                      'load': os.getloadavg()[0] if hasattr(os, 'getloadavg') else None}))


def remote_source():
    """Ship the same ownership helper with the existing stdin scan on uninstalled hosts."""
    helper = (Path(__file__).resolve().parent / 'driver_state.py').read_text()
    bootstrap = ("import sys, types\n"
                 "_driver_state = types.ModuleType('driver_state')\n"
                 "sys.modules['driver_state'] = _driver_state\n"
                 "exec(" + repr(helper) + ", _driver_state.__dict__)\n")
    return bootstrap.encode() + Path(__file__).read_bytes()


# ---------------------------------------------------------------- queue (QUEUE.md)

def queue_blocks(text):
    """Split QUEUE.md into blocks; joining every block's lines gives the text back unchanged."""
    blocks, section = [], ''
    for l in text.splitlines(keepends=True):
        m = ITEM.match(l)
        if l.startswith('#'):
            section = l.lstrip('#').strip()
            blocks.append({'kind': 'head', 'lines': [l], 'section': section})
        elif m:
            blocks.append({'kind': 'item', 'lines': [l], 'section': section, 'mark': m[1]})
        elif blocks and blocks[-1]['kind'] == 'item' and l[:1] in (' ', '\t') and l.strip():
            blocks[-1]['lines'].append(l)
        else:
            blocks.append({'kind': 'line', 'lines': [l], 'section': section})
    seen = {}
    for b in blocks:
        if b['kind'] == 'item':
            h = hashlib.sha1(b['lines'][0].strip().encode()).hexdigest()[:10]
            seen[h] = seen.get(h, 0) + 1
            b['id'] = f'{h}-{seen[h]}'
    return blocks


def queue_rev(text):
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def queue_move(text, item, direction):
    """Move a not-done item up, down or to the top among the not-done items of its section; done items keep their
    lines in place. Returns the new text (a permutation of the old lines)."""
    if direction not in ('up', 'down', 'top'):
        raise ValueError('direction must be up, down or top')
    blocks = queue_blocks(text)
    idx = next((i for i, b in enumerate(blocks) if b.get('id') == item), None)
    if idx is None:
        raise KeyError('queue item not found; the file changed')
    b = blocks[idx]
    if b['mark'] == 'x':
        raise ValueError('done items do not move')
    slots = [i for i, x in enumerate(blocks) if x['kind'] == 'item' and x['section'] == b['section'] and x['mark'] != 'x']
    order = list(slots)
    k = order.index(idx)
    if direction == 'up' and k > 0:
        order[k - 1], order[k] = order[k], order[k - 1]
    elif direction == 'down' and k < len(order) - 1:
        order[k + 1], order[k] = order[k], order[k + 1]
    elif direction == 'top':
        order.insert(0, order.pop(k))
    new = list(blocks)
    for slot, src in zip(slots, order):
        new[slot] = blocks[src]
    out = ''.join(l for x in new for l in x['lines'])
    if sorted(out.splitlines(keepends=True)) != sorted(text.splitlines(keepends=True)):
        raise AssertionError('reorder would change lines')
    return out


def queue_view(text):
    sections, cur = [], None
    for b in queue_blocks(text):
        if b['kind'] == 'head' and b['lines'][0].startswith('## '):
            cur = {'title': b['section'], 'items': []}
            sections.append(cur)
        elif b['kind'] == 'item' and cur is not None:
            first = b['lines'][0]
            body = ''.join(b['lines'])[len(first) - len(first.lstrip()):]
            body = ITEM.sub('', body, count=1).strip()
            title = re.match(r'\*\*(.+?)\*\*', body)
            rest = re.sub(r'\s+', ' ', body[title.end():] if title else body).strip(' :')
            cur['items'].append({'id': b['id'], 'state': MARKS.get(b['mark'], b['mark']),
                                 'title': title[1] if title else rest[:120], 'detail': rest if title else rest[120:]})
    return [s for s in sections if s['items']]


# ---------------------------------------------------------------- owner priorities and audit

def priorities_path(cfg):
    return state_dir(cfg) / 'OWNER-PRIORITIES.json'


def load_priorities(path):
    try:
        P = json.loads(Path(path).read_text())
        return P if isinstance(P.get('items'), dict) else {'items': {}}
    except (OSError, ValueError, AttributeError):
        return {'items': {}}


def set_priority(path, item, priority):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}', item or ''):
        raise ValueError('bad item id')
    if priority not in PRIORITIES + ('',):
        raise ValueError('priority must be P0–P3 or empty')
    P = load_priorities(path)
    P['about'] = 'Owner priorities set from the keeper dashboard: P0 first. Coordinators read this; only the owner writes it.'
    if priority:
        P['items'][item] = {'priority': priority, 'set': time.strftime('%Y-%m-%d %H:%M')}
    else:
        P['items'].pop(item, None)
    P['updated'] = time.strftime('%Y-%m-%d %H:%M')
    atomic_write(path, json.dumps(P, indent=1, ensure_ascii=False, sort_keys=True) + '\n')
    return P


def audit(cfg, entry):
    k = state_dir(cfg) / 'keeper'
    with open(k / 'dashboard-audit.jsonl', 'a') as fh:
        fh.write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M:%S'), **entry}, ensure_ascii=False) + '\n')


def state_dir(cfg):
    repo = Path(cfg['repo'])
    return Path(cfg.get('state', repo.parent / f'{repo.name}-state'))


# ---------------------------------------------------------------- collection

class Collector:
    """Builds the snapshot in a background thread; requests only read it."""

    def __init__(self, cfg_path):
        self.cfg_path = Path(cfg_path).resolve()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.snap = {'t': 0}
        self.feeds, self.remote = {}, {}
        self.load_cfg()

    def load_cfg(self):
        self.cfg = json.loads(self.cfg_path.read_text())
        self.D = self.cfg.get('dashboard', {})
        self.state = state_dir(self.cfg)
        self.k = self.state / 'keeper'
        self.k.mkdir(parents=True, exist_ok=True)

    def keeper(self):
        import importlib.util
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        if 'lane_keeper' not in sys.modules:
            spec = importlib.util.spec_from_file_location('lane_keeper', SCRIPTS / 'lane-keeper.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['lane_keeper'] = mod
        return sys.modules['lane_keeper'].Keeper(self.cfg_path, dry=True)

    def run_json(self, argv, cwd, timeout, stdin=None):
        r = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, timeout=timeout,
                           env={**os.environ, 'PATH': os.environ.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'})
        if r.returncode:
            raise RuntimeError(f'exit {r.returncode}: {r.stderr.decode(errors="replace").strip()[-300:]}')
        return json.loads(r.stdout)

    def refresh_feeds(self, force=False):
        for i, F in enumerate(self.D.get('feeds', [])):
            cur = self.feeds.get(i, {})
            if not force and now() - cur.get('t', 0) < F.get('every', 60):
                continue
            try:
                data = self.run_json(F['argv'], F.get('cwd', self.cfg['repo']), F.get('timeout', 40))
                self.feeds[i] = {'t': now(), 'panels': data.get('panels', []), 'error': ''}
            except Exception as e:
                self.feeds[i] = {**cur, 't': now(), 'error': f'{e!r}'[:300], 'panels': cur.get('panels', [])}

    def refresh_remote(self, force=False):
        src = remote_source()
        for R in self.D.get('remote', []):
            name = R.get('name', R['ssh'])
            cur = self.remote.get(name, {})
            if not force and now() - cur.get('tried', 0) < R.get('every', 300):
                continue
            args = ['--dirs', *R.get('campaigns', []), '--disk', *R.get('disk', ['/'])]
            if R.get('keeper'):
                args += ['--keeper', R['keeper']]
            cmd = 'python3 - scan ' + ' '.join(shlex.quote(a) for a in args)
            try:
                data = self.run_json([*SSH, R['ssh'], cmd], None, R.get('timeout', 20), stdin=src)
                self.remote[name] = {'tried': now(), 'ok': now(), 'data': data, 'error': ''}
            except Exception as e:
                self.remote[name] = {**cur, 'tried': now(), 'error': f'{type(e).__name__}: {e}'[:200]}

    def disk_trend(self, free):
        f = self.k / 'dashboard-disk.json'
        try:
            S = json.loads(read(f, '[]'))
        except ValueError:
            S = []
        if not S or now() - S[-1][0] >= DISK_SAMPLE_EVERY:
            S = [s for s in S if now() - s[0] < DISK_KEEP] + [[now(), round(free, 2)]]
            atomic_write(f, json.dumps(S))
        def at(hours):
            old = [s for s in S if s[0] <= now() - hours * 3600 + DISK_SAMPLE_EVERY]
            return free - old[-1][1] if old else None
        return {'samples': [s for s in S if now() - s[0] < 24 * 3600], 'd1': at(1), 'd24': at(24)}

    def git_today(self):
        """Today's integrations, one row per batch (merge + integrate + apply of one lane), each described from its diff."""
        repo = self.cfg['repo']
        try:
            out = subprocess.run(['git', '-C', repo, 'log', 'main', '--first-parent', '--since=midnight',
                                  '--format=%h\t%ad\t%s', '--date=format:%H:%M'],
                                 capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        rows = [l.split('\t', 2) for l in out.splitlines() if l.count('\t') >= 2]
        rows = [r for r in rows if re.match(r'(Integrate|Release|Apply|Merge)', r[2])]
        groups = []
        for h, t, subj in rows:
            m = re.match(r'(?:Apply thin lane bundles|Integrate lane)\s+(\S+)', subj)
            lane = m[1] if m else ''
            if lane and groups and groups[-1]['lane'] == lane:
                groups[-1]['hashes'].append(h); groups[-1]['first'] = t
                continue
            groups.append({'lane': lane, 'hashes': [h], 'time': t, 'first': t, 'subject': subj})
        for g in groups:
            g.update(self.describe(tuple(g['hashes'])))
        return groups

    def _git(self, *args, timeout=20):
        try:
            return subprocess.run(['git', '-C', self.cfg['repo'], *args], capture_output=True, text=True,
                                  timeout=timeout).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ''

    def describe(self, hashes):
        cache = self.__dict__.setdefault('_described', {})
        if hashes in cache:
            return cache[hashes]
        files, body, fb = {}, [], set()
        for h in hashes:
            msg = self._git('show', '-s', '--format=%B', h)
            fb.update(re.findall(r'\bFB-\d+\b', msg))
            para = [l for l in msg.split('\n\n', 1)[-1].splitlines() if l.strip() and not re.match(r'[\w-]+-By:|Merge ', l)]
            if para and '\n\n' in msg:
                body.append(' '.join(para)[:300])
            for line in self._git('diff', '--numstat', '--find-renames', f'{h}^1', h).splitlines():
                a, d, path = (line.split('\t', 2) + ['', ''])[:3]
                st = files.setdefault(path, {'add': 0, 'del': 0, 'bin': False, 'at': h})
                if a == '-':
                    st['bin'] = True
                else:
                    st['add'] += int(a); st['del'] += int(d)
                st['at'] = h
        items, batches, records, code, tests, ledgers, other = [], [], {}, [], [], [], []
        for path, st in sorted(files.items()):
            name = path.rsplit('/', 1)[-1]
            if re.match(r'data/(lane-bundles|axes/lane-bundle-intake)/[0-9a-f]{64}\.json$', path):
                try:
                    b = json.loads(read(Path(self.cfg['repo']) / 'data/lane-bundles' / name)
                                   or self._git('show', f'{st["at"]}:data/lane-bundles/{name}'))
                    key = re.sub(r'-\d{4}-\d\d-\d\dT[\d:+]+$', '', str(b.get('key', '')))
                    entry = f'{key} ({b.get("kind", "?")}, {len(b.get("outputs", {}))} outputs)'
                    if entry not in batches:
                        batches.append(entry)
                except ValueError:
                    pass
            elif 'ledger' in name:
                ledgers.append(name)
            elif path.endswith('.json') and path.startswith('data/'):
                records.setdefault(path.rsplit('/', 1)[0], []).append((path, st))
            elif re.search(r'(^|/)tests?/|test_', path):
                tests.append(f'{name} +{st["add"]}')
            elif re.search(r'\.(py|mjs|js|ts|tsx|css|html|sh)$', path):
                code.append(f'{path} +{st["add"]}/−{st["del"]}')
            else:
                other.append(f'{path}' + ('' if st['bin'] else f' +{st["add"]}/−{st["del"]}'))
        summary = []
        for folder, xs in records.items():
            titles = []
            for path, st in xs[:6]:
                try:
                    r = json.loads(self._git('show', f'{st["at"]}:{path}'))
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get('title'):
                    extra = ', '.join(str(x) for x in (r.get('year'), f'{r["n_entries"]} entries' if r.get('n_entries') else None) if x)
                    titles.append(f'«{str(r["title"])[:70]}»' + (f' ({extra})' if extra else ''))
            more = f' and {len(xs) - len(titles)} more' if len(xs) > len(titles) else ''
            summary.append(f'{len(xs)} record{"s" if len(xs) != 1 else ""} in {folder}')
            items.append(f'{folder}: ' + ('; '.join(titles) + more if titles else f'{len(xs)} files'))
        if code:
            summary.append(f'code: {len(code)} file{"s" if len(code) != 1 else ""}' + (f', {len(tests)} test files' if tests else ''))
            items.append('code: ' + ', '.join(code[:8]) + (f' and {len(code) - 8} more' if len(code) > 8 else ''))
        if tests and not code:
            summary.append(f'{len(tests)} test files')
        if tests:
            items.append('tests: ' + ', '.join(tests[:8]))
        if other:
            items.append('other: ' + ', '.join(other[:8]) + (f' and {len(other) - 8} more' if len(other) > 8 else ''))
            if not summary:
                summary.append(', '.join(o.split(' ')[0].rsplit('/', 1)[-1] for o in other[:3]))
        if batches:
            items.insert(0, 'from ' + '; '.join(batches))
        if fb:
            summary.append(', '.join(sorted(fb, key=lambda x: int(x[3:]))))
        if ledgers and not summary:
            summary.append('bookkeeping only (' + ', '.join(sorted(set(ledgers))) + ')')
        if body:
            items.append(body[0])
        source = ', '.join(b.split(' (')[0] for b in batches)
        out = {'summary': ' · '.join(summary) or 'no file changes', 'items': items, 'source': source}
        cache[hashes] = out
        return out

    def incidents(self, n=8):
        text = tail(self.state / 'INCIDENTS.md', 24000)
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        out, i = [], len(paras) - 1
        while i >= 0 and len(out) < n:
            p = paras[i]
            if p.startswith('#') and out and '\n' not in p:
                out[-1] = p.lstrip('# ') + '\n' + out[-1]   # heading paragraph followed by its body
            else:
                out.append(p.lstrip('#- ').strip())
            i -= 1
        return out[:n]

    def collect(self):
        self.load_cfg()
        S = {'t': now(), 'errors': []}
        try:
            K = self.keeper()
            lk = sys.modules['lane_keeper']
            info = set(self.cfg.get('coordinator', {}).get('info_kinds', lk.INFO_KINDS))
            ev = list(K.st['events'].values())
            S['events'] = sorted([e for e in ev if e.get('kind') not in info], key=lambda e: e.get('t', 0))
            S['info_events'] = len(ev) - len(S['events'])
            S['coordinator'] = K.coordinator_live() or ''
            L = K.lease()
            S['lease'] = f'{L["owner"]} until {time.strftime("%H:%M", time.localtime(L["until"]))}' if L else ''
            S['swap_gb'] = lk.swap_used_gb()
            S['coordinator_starts'] = K.st['coordinator'].get('starts', 0) if K.st['coordinator'].get('day') == time.strftime('%F') else 0
            S['max_starts'] = self.cfg.get('coordinator', {}).get('max_starts_per_day', 16)
            S['engine'] = self.cfg.get('coordinator', {}).get('engine', 'claude')
            S['stuck'] = (self.k / 'COORDINATOR-STUCK.md').exists()
        except Exception as e:
            S['errors'].append(f'keeper: {e!r}'[:300])
        S['heartbeat'] = mtime(self.k / 'heartbeat')
        try:
            import budget
            S['budget'] = budget.status_lines() if self.cfg.get('budget', {}).get('enabled') else ['budget gating off']
            S['autonomy'] = budget.autonomy()
        except Exception as e:
            S['budget'], S['autonomy'] = [f'budget: {e!r}'[:200]], (False, 0)
        free = shutil.disk_usage(self.cfg['repo']).free / 2**30
        S['disk'] = {'free': free, 'floor': self.cfg.get('disk_floor_gb', 30), **self.disk_trend(free)}
        S['lanes'] = scan(self.cfg.get('campaigns', []), 'Mac')
        q = (self.cfg.get('capacity') or {}).get('queue')
        qtext = read(q) if q else ''
        S['queue'] = {'path': q, 'rev': queue_rev(qtext), 'sections': queue_view(qtext),
                      'target': (self.cfg.get('capacity') or {}).get('target_lanes')}
        S['integrated'] = self.git_today()
        S['incidents'] = self.incidents()
        S['schedule'] = self.cfg.get('schedule', [])
        S['priorities'] = load_priorities(priorities_path(self.cfg))['items']
        S['audit'] = [json.loads(l) for l in tail(self.k / 'dashboard-audit.jsonl', 6000).splitlines()[-6:] if l.startswith('{')]
        self.refresh_feeds()
        self.refresh_remote()
        S['feeds'] = [self.feeds[i] for i in sorted(self.feeds)]
        S['remote'] = dict(self.remote)
        with self.lock:
            self.snap = S

    def loop(self):
        while True:
            try:
                self.collect()
            except Exception as e:
                with self.lock:
                    self.snap = {**self.snap, 'fatal': f'{e!r}'[:300]}
            self.wake.wait(LOCAL_EVERY)
            self.wake.clear()

    def get(self):
        with self.lock:
            return self.snap


# ---------------------------------------------------------------- actions

class Refused(Exception):
    def __init__(self, status, detail, reload=False):
        super().__init__(detail)
        self.status, self.detail, self.reload = status, detail, reload


class Actions:
    def __init__(self, C):
        self.C = C
        self.running = {}

    def local_lane(self, name):
        for r in self.C.get().get('lanes', []):
            if r['lane'] == name and r['host'] == 'Mac':
                for c in self.C.cfg.get('campaigns', []):
                    if (Path(c) / name).is_dir():
                        return Path(c) / name, r
        raise Refused(404, f'no local lane named {name!r}')

    def keeper_cmd(self, *args):
        r = subprocess.run([sys.executable, str(SCRIPTS / 'lane-keeper.py'), *args, '--config', str(self.C.cfg_path)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise Refused(409, (r.stdout + r.stderr).strip()[-300:] or f'exit {r.returncode}')
        return r.stdout.strip()[-300:]

    def do(self, name, args):
        spec = self.C.D.get('actions', {}).get(name)
        if not isinstance(spec, dict):
            raise Refused(403, f'action {name!r} is not in dashboard.actions')
        if not isinstance(args, dict):
            raise Refused(400, 'args must be an object')
        kind = spec.get('builtin')
        if kind == 'queue-move':
            q = (self.C.cfg.get('capacity') or {}).get('queue')
            if not q:
                raise Refused(409, 'no capacity.queue configured')
            text = read(q)
            if args.get('rev') != queue_rev(text):
                raise Refused(409, 'the queue changed since the page loaded; reloaded', reload=False)
            try:
                new = queue_move(text, str(args.get('item')), str(args.get('dir')))
            except (KeyError, ValueError) as e:
                raise Refused(409, str(e))
            if new == text:
                return 'already there'
            atomic_write(q, new)
            return f'moved {args.get("dir")}'
        if kind == 'owner-priority':
            try:
                set_priority(priorities_path(self.C.cfg), str(args.get('id', '')), str(args.get('priority', '')))
            except ValueError as e:
                raise Refused(400, str(e))
            return f'{args.get("id")} → {args.get("priority") or "no owner priority"}'
        if kind in ('lane-pause', 'lane-resume'):
            d, row = self.local_lane(str(args.get('lane', '')))
            return self.keeper_cmd('pause' if kind == 'lane-pause' else 'resume', str(d))
        if kind == 'autonomy-renew':
            r = subprocess.run([sys.executable, str(SCRIPTS / 'budget.py'), 'allow', *([str(spec['hours'])] if spec.get('hours') else [])],
                               capture_output=True, text=True, timeout=30)
            if r.returncode:
                raise Refused(500, r.stderr.strip()[-300:])
            return r.stdout.strip()[-200:] or 'autonomy renewed'
        if kind:
            raise Refused(403, f'unknown builtin {kind!r}')
        if not isinstance(spec.get('argv'), list) or not all(isinstance(a, str) for a in spec['argv']):
            raise Refused(500, f'action {name!r} has no argv list')
        env = {**os.environ, 'PATH': os.environ.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin', **spec.get('env', {})}
        cwd = spec.get('cwd', self.C.cfg['repo'])
        if spec.get('detach'):
            p = self.running.get(name)
            if p and p.poll() is None:
                raise Refused(409, f'{name} is still running (pid {p.pid})')
            log = self.C.k / f'dashboard-{name}.log'
            with open(log, 'a') as fh:
                fh.write(f'\n{time.strftime("%F %T")} start {name}\n')
                self.running[name] = subprocess.Popen(spec['argv'], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=fh,
                                                      stderr=subprocess.STDOUT, start_new_session=True)
            return f'started (pid {self.running[name].pid}); log {log.name}'
        r = subprocess.run(spec['argv'], cwd=cwd, env=env, capture_output=True, text=True, timeout=spec.get('timeout', 120))
        if r.returncode:
            raise Refused(500, (r.stdout + r.stderr).strip()[-300:])
        return r.stdout.strip()[-300:] or 'done'

    def states(self):
        out = {}
        for name, p in self.running.items():
            out[name] = 'running' if p.poll() is None else f'exit {p.returncode}'
        return out


# ---------------------------------------------------------------- page

E = lambda s: html.escape(str(s), quote=True)

CSS = """:root{--bg:#f7f6f2;--fg:#1c1c1a;--mut:#73716b;--line:#e3e0d8;--card:#fff;--ok:#2d7a4c;--warn:#a8650c;--bad:#b23a1d;
--acc:#2c5d8f;--pill:#eceae3}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#e9e8e3;--mut:#9c9a93;--line:#2f2e2b;--card:#1c1c1a;--ok:#6cbf8a;
--warn:#e0a24a;--bad:#ef7d5e;--acc:#7fb0e0;--pill:#2a2a27}}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:13.5px/1.4 -apple-system,system-ui,sans-serif;margin:0;padding:12px 16px 40px}
h1{font-size:17px;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:0 0 6px;font-weight:600}
h3{font-size:13px;margin:8px 0 4px;font-weight:600}
header{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(300px,2fr);gap:12px}@media (max-width:1000px){.grid{grid-template-columns:1fr}}
section{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:9px 11px;margin-bottom:12px}
table{border-collapse:collapse;width:100%}td,th{padding:3px 5px;border-top:1px solid var(--line);vertical-align:top;text-align:left}
th{color:var(--mut);font-weight:500;font-size:12px;border-top:0}tr:first-child td{border-top:0}
.mut{color:var(--mut)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.pill{display:inline-block;background:var(--pill);border-radius:9px;padding:0 7px;font-size:12px;margin-right:3px;white-space:nowrap}
.pill.ok{background:color-mix(in srgb,var(--ok) 18%,transparent)}.pill.warn{background:color-mix(in srgb,var(--warn) 20%,transparent)}
.pill.bad{background:color-mix(in srgb,var(--bad) 20%,transparent)}
.prog{font:12px ui-monospace,Menlo,monospace;color:var(--mut);max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kv{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:2px 10px}.kv div{overflow-wrap:anywhere}.kv div:nth-child(odd){color:var(--mut)}
button,select{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;padding:1px 7px;cursor:pointer}
button:hover{border-color:var(--acc)}button.main{border-color:var(--acc);color:var(--acc)}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
details>summary{cursor:pointer;color:var(--mut);font-size:12px;margin:4px 0}
.q td:first-child{width:62px}.q .done td{color:var(--mut)}.q .t{font-weight:500}.q .d{color:var(--mut);font-size:12px}
.mv{white-space:nowrap;width:1%}.mv button{padding:0 5px}
.inc{font-size:12.5px;border-top:1px solid var(--line);padding:4px 0;white-space:pre-wrap}.inc:first-of-type{border-top:0}
#toast{position:fixed;bottom:12px;right:12px;background:var(--fg);color:var(--bg);padding:6px 11px;border-radius:6px;opacity:0;transition:opacity .2s;max-width:60vw}
#toast.on{opacity:.93}svg.spark{vertical-align:middle}.nw{white-space:nowrap}"""

JS = """const T=document.querySelector('meta[name=token]').content,B=document.querySelector('meta[name=boot]').content;
let busy=0;const toast=m=>{const t=document.getElementById('toast');t.textContent=m;t.classList.add('on');clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove('on'),4000)};
async function act(action,args){busy++;try{const r=await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':T},body:JSON.stringify({action,args})});
const j=await r.json().catch(()=>({detail:'HTTP '+r.status}));toast(j.detail||('HTTP '+r.status));if(j.reload)return location.reload()}finally{busy--}refresh(true)}
document.addEventListener('click',e=>{const b=e.target.closest('button[data-act]');if(!b)return;if(b.dataset.confirm&&!confirm(b.dataset.confirm))return;
act(b.dataset.act,JSON.parse(b.dataset.args||'{}'))});
document.addEventListener('change',e=>{const s=e.target.closest('select[data-act]');if(!s)return;act(s.dataset.act,{...JSON.parse(s.dataset.args||'{}'),priority:s.value})});
async function refresh(force){if(busy&&!force)return;if(!force&&document.activeElement&&document.activeElement.tagName==='SELECT')return;
try{const r=await fetch('/fragment');const j=await r.json();if(j.boot!==B)return location.reload();const m=document.getElementById('main');
const open=[...m.querySelectorAll('details[id]')].filter(d=>d.open).map(d=>d.id),shut=[...m.querySelectorAll('details[id]')].filter(d=>!d.open).map(d=>d.id);
m.innerHTML=j.html;open.forEach(i=>{const d=document.getElementById(i);if(d)d.open=true});shut.forEach(i=>{const d=document.getElementById(i);if(d)d.open=false});
document.title=j.title}catch(e){document.title='Keeper (server down)'}}
setInterval(refresh,15000);document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh()});"""


def button(action, label, args=None, confirm='', cls=''):
    return (f'<button class="{cls}" data-act="{E(action)}" data-args="{E(json.dumps(args or {}))}"'
            + (f' data-confirm="{E(confirm)}"' if confirm else '') + f'>{E(label)}</button>')


OWNER = re.compile(r'(?:→|->)\s*(?:\w+\s+){0,3}?(?:lane|driver)\s+([\w.:-]*\d[\w.:-]*)')


def queue_state(item, lanes):
    """An item marked [~] shows what its named lane is really doing, not just «running»."""
    st = item['state']
    if st != 'running':
        return st, {'open': 'warn', 'waiting': ''}.get(st, '')
    names = [n.rstrip('.').split(':')[-1] for n in OWNER.findall(item['title'] + ' ' + item['detail'])]
    found = [lanes[n] for n in names if n in lanes]
    if not names:
        return 'assigned', ''
    if any(r['state'] in ('running', 'pausing') for r in found):
        return 'running', 'ok'
    if not found:
        return 'lane not found', 'warn'
    return 'lane ' + found[0]['state'].lower(), 'warn'


def pill(label, n=None, cls=''):
    return f'<span class="pill {cls}">{E(label)}{" " + E(n) if n is not None else ""}</span>'


def spark(samples, w=150, h=26):
    if len(samples) < 2:
        return ''
    t0, t1 = samples[0][0], samples[-1][0]
    vs = [s[1] for s in samples]
    lo, hi = min(vs), max(vs)
    span = (hi - lo) or 1
    pts = ' '.join(f'{(s[0] - t0) / ((t1 - t0) or 1) * w:.1f},{h - 2 - (s[1] - lo) / span * (h - 4):.1f}' for s in samples)
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="free disk, last 24 h">'
            f'<polyline fill="none" stroke="currentColor" stroke-width="1.4" points="{pts}"/></svg>')


class Page:
    def __init__(self, C, A):
        self.C, self.A = C, A

    def allowed(self, name):
        return name in self.C.D.get('actions', {})

    def lane_rows(self, rows, actions=True):
        out = []
        for r in rows:
            cls = {'running': 'ok', 'pausing': 'warn', 'paused': 'warn', 'stopped': 'mut'}.get(r['state'], '')
            what = 'session+driver' if r['session'] and r['driver'] else 'session' if r['session'] else 'driver' if r['driver'] else ''
            ctl = ''
            if actions and r['host'] == 'Mac' and r['has_driver'] and not r['once']:
                if r['state'] == 'running' and r['driver'] and self.allowed('lane-pause'):
                    ctl = button('lane-pause', 'Pause', {'lane': r['lane']}, f'Pause {r["lane"]}? Its driver stops after the current step.')
                elif r['state'] == 'paused' and self.allowed('lane-resume'):
                    ctl = button('lane-resume', 'Resume', {'lane': r['lane']})
            out.append(f'<tr><td><b>{E(r["lane"])}</b><div class="prog" title="{E(r["progress"])}">{E(r["progress"])}</div></td>'
                       f'<td>{pill(r["state"], cls=cls)}<div class="mut">{E(what)}</div></td>'
                       f'<td>{E(r["engine"] or "—")}<div class="mut">{E(r["host"])}</div></td>'
                       f'<td class="mut">{ago(r["last"])}</td><td class="mv">{ctl}</td></tr>')
        return ''.join(out)

    def lanes(self, S):
        local = S.get('lanes', [])
        remote = [r for v in S.get('remote', {}).values() for r in (v.get('data') or {}).get('lanes', [])]
        for r in remote:
            r['host'] = next((k for k, v in S['remote'].items() if r in (v.get('data') or {}).get('lanes', [])), r['host'])
        allr = local + remote
        running = sorted([r for r in allr if r['state'] in ('running', 'pausing')], key=lambda r: -r['last'])
        ready = [r for r in allr if r['ready']]
        paused = [r for r in allr if r['state'] in ('paused', 'stopped', 'handoff', 'idle') and now() - r['last'] < 48 * 3600
                  or r['state'] == 'paused']
        closed = sorted([r for r in allr if r['state'] in ('DONE', 'RETIRED', 'DRIVER') and now() - r['close_t'] < 24 * 3600],
                        key=lambda r: -r['close_t'])
        head = '<tr><th>Lane</th><th>State</th><th>Engine</th><th>Last write</th><th></th></tr>'
        h = (f'<section><h2>Lanes {pill("running", len(running), "ok")}{pill("ready", len(ready), "warn" if ready else "")}'
             f'{pill("paused/idle", len(paused))}{pill("closed 24 h", len(closed))}</h2>'
             f'<table>{head}{self.lane_rows(running) or "<tr><td class=mut colspan=5>none running</td></tr>"}</table>')
        if ready:
            h += '<h3>Ready for integration</h3><table>' + ''.join(
                f'<tr><td><b>{E(r["lane"])}</b> <span class="mut">{E(r["host"])}</span></td><td>{E(r["ready"])}</td>'
                f'<td class="mut">{ago(r["close_t"] if r["close"] == r["ready"] else r["last"])}</td></tr>' for r in ready) + '</table>'
        if paused:
            h += (f'<details id="d-paused"><summary>Paused, stopped and idle ({len(paused)})</summary><table>{head}'
                  f'{self.lane_rows(paused)}</table></details>')
        if closed:
            h += ('<details id="d-closed"><summary>Done and retired, last 24 h (' + str(len(closed)) + ')</summary><table>' + ''.join(
                f'<tr><td><b>{E(r["lane"])}</b> <span class="mut">{E(r["host"])}</span></td><td>{pill(r["state"])}</td>'
                f'<td class="mut">{E(r["handled"][:120])}</td><td class="mut">{ago(r["close_t"])}</td></tr>' for r in closed)
                + '</table></details>')
        return h + '</section>'

    def waiting(self, S):
        ev = S.get('events', [])
        integ = S.get('integrated', [])
        h = f'<section><h2>Waiting on the coordinator {pill(len(ev), cls="warn" if ev else "")}</h2>'
        h += ('<table>' + ''.join(f'<tr><td>{pill(e["kind"])}</td><td><b>{E(e.get("name", ""))}</b> '
                                  f'<span class="mut">{E(e["detail"][:220])}</span></td><td class="mut">{ago(e.get("t"))}</td></tr>'
                                  for e in ev[-25:]) + '</table>') if ev else '<div class="mut">nothing</div>'
        h += f'<h3>Integrated today {pill(len(integ))}</h3>'
        def row(g):
            lane = re.sub(r'-20\d{6}(-\d{4})?', '', g['lane']) if g['lane'] else g['subject'][:60]
            items = list(g.get('items', []))
            if g.get('source'):
                items.append(f'integrated by {g["lane"]}')
                lane = g['source']
            when = g['time'] if g['first'] == g['time'] else f'{g["first"]}–{g["time"]}'
            det = ''.join(f'<li>{E(x)}</li>' for x in items)
            det = (f'<details><summary class="mut">{len(g["hashes"])} commit{"s" if len(g["hashes"]) > 1 else ""} '
                   f'<code>{E(" ".join(g["hashes"]))}</code></summary><ul>{det}</ul></details>')
            return (f'<tr><td class="mut">{E(when)}</td><td><b>{E(lane)}</b><div>{E(g.get("summary", ""))}</div>'
                    f'{det}</td></tr>')
        h += ('<table>' + ''.join(row(g) for g in integ[:8]) + '</table>') if integ else '<div class="mut">none yet</div>'
        if len(integ) > 8:
            h += (f'<details id="d-integ"><summary>earlier today ({len(integ) - 8})</summary><table>'
                  + ''.join(row(g) for g in integ[8:]) + '</table></details>')
        return h + '</section>'

    def health(self, S):
        d = S.get('disk', {})
        free = d.get('free') or 0
        dcls = 'bad' if free < d.get('floor', 30) else 'warn' if free < d.get('floor', 30) * 2 else 'ok'
        def delta(x): return '—' if x is None else f'{x:+.1f} GB'
        hb = now() - (S.get('heartbeat') or 0)
        swap = S.get('swap_gb')
        ok, until = S.get('autonomy') or (False, 0)
        rows = [
            ('Disk free', f'<span class="{dcls}">{free:.0f} GB</span> {spark(d.get("samples", []))} '
                          f'<span class="mut">1 h {delta(d.get("d1"))} · 24 h {delta(d.get("d24"))}</span>'),
            ('Swap', '—' if swap is None else f'<span class="{"warn" if swap > 10 else ""}">{swap:.1f} GB</span>'),
            ('Keeper', f'<span class="{"ok" if hb < 300 else "bad"}">tick {ago(S.get("heartbeat"))} ago</span>'),
            ('Coordinator', (f'<span class="ok">{E(S.get("coordinator"))}</span>' if S.get('coordinator') else '<span class="mut">none live</span>')
             + f' <span class="mut">{E(S.get("engine", ""))} · starts {S.get("coordinator_starts", 0)}/{S.get("max_starts", "")}</span>'
             + (' ' + pill('stuck', cls='bad') if S.get('stuck') else '') + (f' <span class="mut">until {E(S["lease"].split(" until ")[-1])}</span>' if S.get('lease') else '')),
            ('Autonomy', ('<span class="ok">no end</span>' if ok and until == float('inf') else
                          f'<span class="{"ok" if until - now() > 3600 else "warn"}">until {time.strftime("%a %H:%M", time.localtime(until))}</span>'
                          if ok else '<span class="bad">' + ('expired' if until else 'not granted') + '</span>')
             + (' ' + button('autonomy-renew', self.C.D['actions']['autonomy-renew'].get('label', 'Renew'), {}, 'Renew autonomy?')
                if self.allowed('autonomy-renew') else '')),
        ]
        for b in S.get('budget', []):
            if b.startswith('autonomy'):
                continue
            eng, _, rest = b.partition(': ')
            cls = 'bad' if 'HOLD' in rest[:6] else 'ok'
            rows.append((f'Budget {eng}', f'<span class="{cls}">{E(rest)}</span>'))
        for name, R in S.get('remote', {}).items():
            data = R.get('data') or {}
            dk = ' · '.join(f'{x["path"]} {x["free_gb"]:.0f} GB' for x in data.get('disk', []) if x.get('free_gb') is not None)
            hb2 = data.get('keeper_heartbeat')
            stale = f' <span class="bad">{E(R["error"])}</span>' if R.get('error') else ''
            rows.append((name, (f'{dk} · keeper tick {ago(hb2)} ago · <span class="mut">read {ago(R.get("ok"))} ago</span>' if data else '')
                         + stale))
        h = '<section><h2>System</h2><div class="kv">' + ''.join(f'<div>{E(k)}</div><div>{v}</div>' for k, v in rows) + '</div>'
        if S.get('errors'):
            h += ''.join(f'<div class="bad">{E(e)}</div>' for e in S['errors'])
        return h + '</section>'

    def schedule(self, S):
        sch = S.get('schedule', [])
        if not sch:
            return ''
        rows = ''.join(f'<tr><td>{E(x.get("at", ""))}</td><td>{E(x.get("date") or " ".join(x.get("days", [])) or "daily")}</td>'
                       f'<td>{pill(x.get("kind", "SCHEDULE"))}</td><td>{E(x.get("note", ""))}</td></tr>' for x in sch)
        return f'<section><h2>Schedule</h2><table>{rows}</table></section>'

    def panel(self, P, prios):
        h = f'<section><h2>{E(P.get("title", P.get("id", "")))} ' + ''.join(pill(l, n) for l, n in P.get('pills', [])) + '</h2>'
        if P.get('rows'):
            h += '<div class="kv">' + ''.join(
                f'<div>{E(r[0])}</div><div>' + (f'<a href="{E(r[2])}" target="_blank">{E(r[1])}</a>' if len(r) > 2 and r[2] else E(r[1]))
                + '</div>' for r in P['rows']) + '</div>'
        btns = [button(a, self.C.D['actions'][a].get('label', a), {}, self.C.D['actions'][a].get('confirm', ''))
                for a in P.get('actions', []) if self.allowed(a)]
        links = [f'<a href="{E(u)}" target="_blank">{E(l)}</a>' for l, u in P.get('links', [])]
        if btns or links:
            st = self.A.states()
            h += '<div style="margin-top:6px">' + ' '.join(btns + links) + ''.join(
                f' <span class="mut">{E(a)}: {E(st[a])}</span>' for a in P.get('actions', []) if a in st) + '</div>'
        for gi, G in enumerate(P.get('groups', [])):
            rows = G.get('rows', [])
            if P.get('prioritise'):
                rank = lambda r: PRIORITIES.index(prios[r['id']]['priority']) if r.get('id') in prios else 9
                rows = sorted(rows, key=rank)
            cols = G.get('columns', [])
            body = ''
            for r in rows:
                last = len(r.get('cells', [])) - 1
                cells = ''.join(f'<td{" class=nw" if i < last else ""}>{E(c)}</td>' for i, c in enumerate(r.get('cells', [])))
                if r.get('url'):
                    cells = f'<td><a href="{E(r["url"])}" target="_blank">{E(r["cells"][0])}</a></td>' + ''.join(
                        f'<td>{E(c)}</td>' for c in r['cells'][1:])
                sel = ''
                if P.get('prioritise') and r.get('id') and self.allowed('owner-priority'):
                    cur = prios.get(r['id'], {}).get('priority', '')
                    sel = (f'<td class="mv"><select data-act="owner-priority" data-args="{E(json.dumps({"id": r["id"]}))}">'
                           + ''.join(f'<option value="{p}"{" selected" if p == cur else ""}>{p or "—"}</option>' for p in ('',) + PRIORITIES)
                           + '</select></td>')
                body += f'<tr>{sel}{cells}</tr>'
            th = ('<th>Owner</th>' if P.get('prioritise') and self.allowed('owner-priority') else '') + ''.join(f'<th>{E(c)}</th>' for c in cols)
            table = f'<table><tr>{th}</tr>{body}</table>' if rows else '<div class="mut">none</div>'
            did = f'd-{P.get("id", "p")}-{gi}'
            h += (f'<details id="{E(did)}"{" open" if G.get("open") else ""}><summary>{E(G.get("title", ""))} ({len(rows)})</summary>'
                  f'{table}</details>')
        return h + '</section>'

    def feeds(self, S, prios):
        out = []
        for F in S.get('feeds', []):
            for P in F.get('panels', []):
                out.append((P.get('place', 'main'), self.panel(P, prios)))
            if F.get('error'):
                out.append(('side', f'<section><h2>Feed</h2><div class="bad">{E(F["error"])}</div></section>'))
        return out

    def queue(self, S):
        Q = S.get('queue', {})
        secs = Q.get('sections', [])
        if not secs:
            return ''
        counts = {}
        for s in secs:
            for it in s['items']:
                counts[it['state']] = counts.get(it['state'], 0) + 1
        can = self.allowed('queue-move')
        lanes = {r['lane']: r for r in S.get('lanes', [])}
        h = '<section class="q"><h2>Queue ' + ''.join(
            pill(k, counts[k], {'open': 'warn', 'running': 'ok'}.get(k, '')) for k in ('open', 'running', 'waiting', 'done') if k in counts) + '</h2>'
        for i, s in enumerate(secs):
            live = [it for it in s['items'] if it['state'] != 'done']
            done = [it for it in s['items'] if it['state'] == 'done']
            h += f'<h3>{E(s["title"])} <span class="mut">{len(live)} active · {len(done)} done</span></h3><table>'
            for j, it in enumerate(live):
                mv = ''
                if can and len(live) > 1:
                    a = {'item': it['id'], 'rev': Q['rev']}
                    mv = ((button('queue-move', '↑', {**a, 'dir': 'up'}) if j else '')
                          + (button('queue-move', '↓', {**a, 'dir': 'down'}) if j < len(live) - 1 else ''))
                label, cls = queue_state(it, lanes)
                h += (f'<tr><td>{pill(label, cls=cls)}</td><td><div class="t">{E(it["title"])}</div>'
                      f'<div class="d">{E(it["detail"][:260])}</div></td><td class="mv">{mv}</td></tr>')
            h += '</table>'
            if done:
                h += (f'<details id="d-q{i}"><summary>done ({len(done)})</summary><table>' + ''.join(
                    f'<tr class="done"><td>{pill("done")}</td><td>{E(it["title"])}</td></tr>' for it in done) + '</table></details>')
        return h + '</section>'

    def incidents(self, S):
        inc = S.get('incidents', [])
        if not inc:
            return ''
        h = f'<section><h2>Incidents {pill(len(inc))}</h2>'
        for p in inc:
            first, _, rest = p.partition('\n')
            h += f'<div class="inc"><b>{E(first[:220])}</b>' + (f'<div class="mut">{E(rest[:500])}</div>' if rest else '') + '</div>'
        return h + '</section>'

    def audit(self, S):
        a = S.get('audit', [])
        if not a:
            return ''
        return ('<details id="d-audit"><summary>Recent actions</summary><table>' + ''.join(
            f'<tr><td class="mut">{E(x.get("t", "")[11:16])}</td><td>{E(x.get("action"))}</td><td>{E(json.dumps(x.get("args", {}))[:90])}</td>'
            f'<td class="{"ok" if x.get("ok") else "bad"}">{E(str(x.get("detail", ""))[:120])}</td></tr>' for x in reversed(a))
                + '</table></details>')

    def title(self, S):
        run = sum(r['state'] in ('running', 'pausing') for r in S.get('lanes', []))
        return f'Keeper · {run} running · {len(S.get("events", []))} waiting'

    def fragment(self, S):
        if not S.get('t'):
            return '<section class="mut">collecting…</section>'
        prios = S.get('priorities', {})
        fe = self.feeds(S, prios)
        left = self.lanes(S) + self.waiting(S) + self.queue(S) + ''.join(h for p, h in fe if p == 'main')
        right = (self.health(S) + ''.join(h for p, h in fe if p == 'side') + self.schedule(S) + self.incidents(S) + self.audit(S))
        name = self.C.cfg.get('name', Path(self.C.cfg['repo']).name)
        top = (f'<header><h1>{E(name)} keeper</h1><span class="mut">{time.strftime("%a %d %b %H:%M:%S", time.localtime(S["t"]))}</span>'
               + (f'<span class="bad">{E(S["fatal"])}</span>' if S.get('fatal') else '') + '</header>')
        return top + f'<div class="grid"><div>{left}</div><div>{right}</div></div>'

    def full(self, S, token, boot):
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="token" content="{E(token)}"><meta name="boot" content="{E(boot)}"><title>{E(self.title(S))}</title><style>{CSS}</style></head>
<body><div id="main">{self.fragment(S)}</div><div id="toast"></div><script>{JS}</script></body></html>"""


# ---------------------------------------------------------------- server

def make_handler(C, A, P, port, token, boot):
    from http.server import BaseHTTPRequestHandler
    hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
    origins = {f'http://{h}' for h in hosts}

    class H(BaseHTTPRequestHandler):
        server_version = 'keeper-dashboard'

        def log_message(self, *a):
            pass

        def send(self, code, body, ctype='application/json; charset=utf-8'):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                                                        "img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def host_ok(self):
            return self.headers.get('Host', '') in hosts

        def do_GET(self):
            if not self.host_ok():
                return self.send(421, json.dumps({'detail': 'wrong host'}))
            S = C.get()
            path = self.path.split('?')[0]
            if path == '/':
                return self.send(200, P.full(S, token, boot), 'text/html; charset=utf-8')
            if path == '/fragment':
                return self.send(200, json.dumps({'boot': boot, 'html': P.fragment(S), 'title': P.title(S)}))
            if path == '/healthz':
                return self.send(200, json.dumps({'ok': True, 'snapshot_age': now() - S.get('t', 0)}))
            self.send(404, json.dumps({'detail': 'not found'}))

        def do_POST(self):
            name, args, status, detail, reload = '', {}, 200, '', False
            try:
                if self.path != '/action':
                    raise Refused(404, 'not found')
                check_request(self.headers, hosts, origins, token)
                n = int(self.headers.get('Content-Length') or 0)
                if n > 65536:
                    raise Refused(413, 'too large')
                try:
                    body = json.loads(self.rfile.read(n) or b'{}')
                except ValueError:
                    raise Refused(400, 'body must be JSON')
                name, args = str(body.get('action', '')), body.get('args', {})
                detail = A.do(name, args)
                ok = True
            except Refused as e:
                status, detail, ok, reload = e.status, e.detail, False, e.reload
            except Exception as e:
                status, detail, ok = 500, f'{e!r}'[:300], False
            try:
                audit(C.cfg, {'action': name, 'args': args, 'ok': ok, 'status': status, 'detail': detail})
            except OSError:
                pass
            C.wake.set()
            self.send(status, json.dumps({'detail': detail, 'ok': ok, 'reload': reload}))

    return H


def check_request(headers, hosts, origins, token):
    """Raises Refused unless the request comes from this page: right Host, same Origin, this start's token, JSON."""
    if headers.get('Host', '') not in hosts:
        raise Refused(421, 'wrong host')
    if headers.get('Origin', '') not in origins:
        raise Refused(403, 'cross-origin request refused')
    if headers.get('Sec-Fetch-Site', 'same-origin') != 'same-origin':
        raise Refused(403, 'cross-site request refused')
    if not hmac.compare_digest(headers.get('X-Dashboard-Token', ''), token):
        raise Refused(403, 'stale page: token from an earlier start; reloading', reload=True)
    if not headers.get('Content-Type', '').startswith('application/json'):
        raise Refused(415, 'JSON only')


def serve(cfg_path, port=None):
    from http.server import ThreadingHTTPServer
    C = Collector(cfg_path)
    port = int(port or C.D.get('port', DEFAULT_PORT))
    token, boot = secrets.token_urlsafe(24), secrets.token_hex(6)
    A = Actions(C)
    P = Page(C, A)
    threading.Thread(target=C.loop, daemon=True).start()
    httpd = ThreadingHTTPServer(('127.0.0.1', port), make_handler(C, A, P, port, token, boot))
    (C.k / 'dashboard.url').write_text(f'http://127.0.0.1:{port}/\n')
    print(f'{time.strftime("%F %T")} keeper dashboard on http://127.0.0.1:{port}/ (config {C.cfg_path})', flush=True)
    httpd.serve_forever()


def main(argv):
    mode = argv[0] if argv else 'serve'
    if mode == 'scan':
        return scan_main(argv[1:])
    cfg, port = DEFAULT_CONFIG, None
    if '--config' in argv:
        cfg = Path(argv[argv.index('--config') + 1])
    if '--port' in argv:
        port = argv[argv.index('--port') + 1]
    if mode == 'serve':
        return serve(cfg, port)
    sys.exit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
