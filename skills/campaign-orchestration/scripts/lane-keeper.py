#!/usr/bin/env python3
"""Keeps campaign lanes and their coordinator going without anyone watching. Spends no model tokens itself.

Run by launchd every two minutes (`tick`). Reads the repo's .coordinator.json:
    {"repo": …, "campaigns": [lane-state dirs], "state": <dir for keeper/ and handovers>,
     "handover_glob": "*HANDOVER*.md", "coordinator": {"max_starts_per_day": 16, "min_gap_minutes": 10}}

Per lane folder (one with pid or driver.pid), when its worker session is gone:
  - HANDOFF.md (or a HANDOFF.<n>.md newer than the session) and no live driver → fresh session (campaign launch.sh)
  - 429 / capacity in the log tail → resume the same session after a backoff, without a cap
  - ended with no DONE / READY / HANDOFF → resume it twice with msg-keeper-continue, then wake the coordinator
  - its drive-loop was killed (driver.cmd, no DRIVER-EXIT) → restart the drive-loop, no session needed
  - DRIVER-FAILED.md → resume the worker once to fix it, then wake the coordinator
While a session runs: costguard over cap → stop it and resume with msg-handoff (once per session).
Lane outcomes that need judgement (READY-*, DONE.md, a driver-only job's exit, caps, an hour's silence) become
events in <state>/keeper/EVENTS. A lane's events are cleared by `ack <lane-dir> <note>` (writes HANDLED) or RETIRED.

When events are pending and no coordinator is live, the keeper starts one: `claude -p` in the repo on the newest
handover. Live means: a `lane-keeper.py watch` heartbeat (an interactive coordinator's Monitor), a keeper-started
coordinator still running, or a registered coordinator with an assistant/tool-result entry in the last 20 minutes. `retire` (run by a
coordinator after it writes its handover) appends keeper/retired: named sessions never count again; older signals stop counting, so the
next tick starts a successor, without waiting out min_gap_minutes. The headless run's log is
keeper/coordinator-<stamp>.log (stream-json); its session id is in keeper/coordinator.session and keeper.log.

    lane-keeper.py tick   [--config F] [--dry-run]
    lane-keeper.py watch  [--config F]          # Monitor command for an interactive coordinator
    lane-keeper.py retire [--config F] [--session ID]   # ID defaults to $CLAUDE_CODE_SESSION_ID
    lane-keeper.py ack    <lane-dir | host:lane-dir> [note…]
    lane-keeper.py status [--config F]

Optional config keys:
  "remote_events": [{"ssh": "alif-ts", "path": "/srv/agents/skard/keeper/EVENTS", "prefix": "alif"}]
      each tick reads that keeper's new EVENTS lines over ssh and records them here, lane named <prefix>:<lane> and
      kept as host:<lane-dir>; they clear when the remote keeper clears them (ack host:<lane-dir> writes HANDLED there)
  "notify": {"cmd": "/srv/agents/bin/notify", "min_gap_minutes": 30}
      new events are mailed in one batch at most once per gap
  "detach": ["systemd-run", "--scope", "--quiet", "--collect"]
      prefix for every command the keeper starts, so lanes leave a systemd timer's cgroup instead of dying with it
Runaway checks (each tick; events are reported once per process, under the keeper dir — ack it to clear them):
  ORPHAN               a process reparented to pid 1, older than orphan_age_minutes (30), whose cwd or argv is under a
                       lane root: <repo>/.claude/worktrees, <repo>/.codex/worktrees, the campaign dirs, "worktree_roots"
  STOPPED-BUT-RUNNING  a lane with STOP whose driver.pid (or its sleepsafe) still runs 10 min after STOP
  STALE-PREVIEW        a preview server ("preview_patterns", default local.mjs / wrangler dev / http.server) in the repo,
                       older than preview_age_hours (6), with no live claude/codex/tmux/sshd/login ancestor
  RESOURCE-FLOOR       free disk < disk_floor_gb (30) or swap used > swap_ceiling_gb (20); at most once an hour, with the
                       top 5 processes by memory and the largest dirs one level under each "disk_watch" path
  "reap_orphans": true reaps the first three (SIGTERM, SIGKILL after reap_grace_seconds, 10) and logs each pid, its
                       footprint and argv. Never reaped: the keeper's own tree, claude/codex sessions and their trees,
                       coordinator.pid, and every lane's pid / driver.pid tree (except a STOPped lane's driver).
Linux: systemd timer instead of launchd; CODEX_HOME is honoured; caffeinate is used only where it exists.
"""
import datetime as dt
import fcntl, hashlib, json, os, re, shlex, shutil, signal, subprocess, sys, time, uuid
from pathlib import Path
from session_guard import last_session_id, tail_text

DEFAULT_CONFIG = Path.home() / 'src/skard/.coordinator.json'
SCRIPTS = Path(__file__).resolve().parent
REG = Path.home() / '.agents/coordinators'
CLOSING = re.compile(r'^(DONE\.md|READY-[0-9][0-9a-z]*\.md|DRIVER-FINISHED\.json|RETIRED)$')
TRANSIENT = re.compile(r'at capacity|429 Too Many|retry limit'
                       # Claude CLI: limit results and rejected rate_limit_events; "allowed" events are routine
                       r'|usage limit|hit your (?:\w+ )?limit|rate_limit_error|overloaded_error|API Error: (?:429|529)'
                       r'|"error":\s*"rate_limit"|"rate_limit_info":\s*\{"status":\s*"rejected"', re.I)
CLAUDE_MODEL = re.compile(r'^(opus|sonnet|haiku|fable|claude-)')
FRESH_LIMIT, CONTINUE_LIMIT, QUIET = 25, 2, 3600
TRANSCRIPT_TAIL = 1_000_000  # bounded I/O even for a multi-hundred-MB session
COORD_LIVE = 1200
RETIRE_GRACE = 60   # a retiring session still writes its last message after `retire`
_FNM = Path.home() / 'Library/Application Support/fnm/node-versions/v24.11.1/installation/bin'
PATH = ':'.join(dict.fromkeys([*([str(_FNM)] if _FNM.is_dir() else []), str(Path.home() / '.local/bin'), '/opt/homebrew/bin',
                               *os.environ.get('PATH', '').split(':'), '/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin']))
CODEX_SESSIONS = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')) / 'sessions'
CLAUDE_PROJECTS = Path(os.environ.get('CLAUDE_CONFIG_DIR', Path.home() / '.claude')) / 'projects'
CAFFEINATE = ['caffeinate', '-i'] if shutil.which('caffeinate') else []
SSH = ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes']

MSG_CONTINUE = """Lane keeper (automatic): your session ended without DONE.md, READY-<n>.md or HANDOFF.md in the lane
folder. If the queue still has work, continue it now from where you were. If a long driver should carry it, start it
detached under drive-loop.sh, write HANDOFF.md (state, queue position, the command that continues it) and exit; a fresh
session will pick up when the driver finishes. If the work is finished, write READY-<n>.md or DONE.md. If it is
blocked on something outside the lane, write DONE.md naming the blocker and the file that would release it."""

MSG_DRIVER_FAILED = """Lane keeper (automatic): your driver failed three runs in a row; DRIVER-FAILED.md in the lane folder
has the log tail. Fix the cause and restart the driver under drive-loop.sh, then continue the queue. If it cannot be
fixed inside the lane, write DONE.md saying what failed, what you tried, and the log path."""


def now():
    return time.time()


def stamp(t=None):
    return time.strftime('%F %T', time.localtime(t or now()))


def mtime(p):
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return 0


def read_text(p):
    try:
        return Path(p).read_text().strip()
    except OSError:
        return ''


def retired_ids(p):
    # Keep the existing marker format; append retirements so a later handover
    # cannot revive an older session whose registry file reappeared.
    return {line.split()[-1] for line in read_text(p).splitlines() if len(line.split()) == 3
            and line.split()[-1] != '-'}


def transcript_activity(p):
    """Last assistant/tool completion timestamp, never filesystem metadata writes.

    Read a bounded tail and discard any partial first/last JSON line. No valid
    work entry in that tail means there is no evidence this session is live.
    """
    try:
        with Path(p).open('rb') as fh:
            fh.seek(0, 2)
            start = max(0, fh.tell() - TRANSCRIPT_TAIL)
            fh.seek(start)
            if start:
                fh.readline(TRANSCRIPT_TAIL)  # incomplete first line
            lines = fh.read(TRANSCRIPT_TAIL).splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            content = (row.get('message') or {}).get('content', [])
            tool_result = row.get('type') == 'user' and isinstance(content, list) and any(
                isinstance(c, dict) and c.get('type') == 'tool_result' for c in content)
            if row.get('type') != 'assistant' and not tool_result:
                continue
            t = dt.datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            if t.tzinfo is not None:
                return t.timestamp()
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            continue
    return 0


def birth(p):
    """When the file was created. Linux has no st_birthtime: for codex.log, launch.sh and resume.sh write the session
    start beside it (session-at); without either the short-session and later-handoff checks stay off."""
    p = Path(p)
    try:
        return p.stat().st_birthtime
    except AttributeError:
        pass
    except OSError:
        return 0
    try:
        return float((p.parent / 'session-at').read_text().split()[0]) if p.name == 'codex.log' and p.exists() else 0
    except (OSError, ValueError, IndexError):
        return 0


def alive(pidfile, pid=None):
    """A pid file is written after its process starts; a process that started later holds a recycled pid."""
    try:
        pid = int(pid if pid is not None else Path(pidfile).read_text().split()[0])
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


def newest_in(d):
    try:
        return max((e.stat().st_mtime for e in os.scandir(d) if e.is_file()), default=0)
    except OSError:
        return 0


SESSION_NAMES = {'claude', 'codex'}
INTERACTIVE_NAMES = SESSION_NAMES | {'tmux', 'sshd', 'login', 'screen'}
PREVIEW_PATTERNS = [r'local\.mjs', r'wrangler\S*\s+(pages\s+)?dev', r'http\.server']


def etime_seconds(s):
    days, _, rest = s.strip().rpartition('-')
    parts = [int(p) for p in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def ps_table():
    out = subprocess.run(['ps', '-axo', 'pid=,ppid=,pgid=,etime=,rss=,command='], capture_output=True, text=True,
                         timeout=30).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6:
            pid, ppid, pgid, et, rss, cmd = parts
            rows[int(pid)] = {'pid': int(pid), 'ppid': int(ppid), 'pgid': int(pgid), 'age': etime_seconds(et),
                              'rss': int(rss), 'cmd': cmd}
    return rows


def proc_cwds(pids):
    if not pids:
        return {}
    if Path('/proc/self/cwd').exists():
        out = {}
        for p in pids:
            try: out[p] = os.readlink(f'/proc/{p}/cwd')
            except OSError: pass
        return out
    try:
        r = subprocess.run(['lsof', '-a', '-d', 'cwd', '-Fn', '-p', ','.join(map(str, pids))],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out, cur = {}, None
    for line in r.stdout.splitlines():
        if line.startswith('p'): cur = int(line[1:])
        elif line.startswith('n') and cur: out[cur] = line[1:]
    return out


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def swap_used_gb():
    try:
        if sys.platform == 'darwin':
            m = re.search(r'used = ([\d.]+)M', subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True,
                                                               text=True, timeout=10).stdout)
            return float(m[1]) / 1024 if m else None
        info = dict(l.split(':', 1) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l)
        return (int(info['SwapTotal'].split()[0]) - int(info['SwapFree'].split()[0])) / 2**20
    except (OSError, KeyError, ValueError, subprocess.TimeoutExpired):
        return None


def top_processes(table, n=5):
    """By footprint (top's MEM on macOS, RSS elsewhere), each with its full argv from ps."""
    rows = []
    if sys.platform == 'darwin' and shutil.which('top'):
        try:
            out = subprocess.run(['top', '-l', '1', '-o', 'mem', '-n', str(n), '-stats', 'pid,mem'],
                                 capture_output=True, text=True, timeout=30).stdout.splitlines()
            start = next(i for i, l in enumerate(out) if l.split()[:1] == ['PID']) + 1
            rows = [(int(l.split()[0]), l.split()[1]) for l in out[start:start + n] if l.split()]
        except (OSError, StopIteration, ValueError, IndexError, subprocess.TimeoutExpired):
            rows = []
    if not rows:
        rows = [(p, f'{r["rss"] // 1024}M') for p, r in sorted(table.items(), key=lambda x: -x[1]['rss'])[:n]]
    return ', '.join(f'{p} {mem} {table.get(p, {}).get("cmd", "?")[:120]}' for p, mem in rows)


def largest_dirs(paths, n=5, timeout=60):
    sizes = []
    for path in paths:
        try:
            out = subprocess.run(['du', '-sk', '-d', '1', str(path)], capture_output=True, text=True, timeout=timeout).stdout
        except subprocess.TimeoutExpired:
            sizes.append((-1, f'{path} (du timed out)')); continue
        except OSError:
            continue
        for line in out.splitlines():
            kb, _, d = line.partition('\t')
            if kb.isdigit() and Path(d) != Path(path):
                sizes.append((int(kb), d))
    top = sorted(sizes, reverse=True)[:n]
    return ', '.join(f'{kb / 2**20:.1f}G {d}' if kb >= 0 else d for kb, d in top) or 'none'


def descendants(table, roots):
    children = {}
    for p, r in table.items():
        children.setdefault(r['ppid'], []).append(p)
    seen, todo = set(), [r for r in roots if r in table]
    while todo:
        p = todo.pop()
        if p in seen: continue
        seen.add(p); todo.extend(children.get(p, []))
    return seen


def ancestors(table, pid):
    out = []
    while pid in table and table[pid]['ppid'] not in (0, pid) and len(out) < 64:
        pid = table[pid]['ppid']; out.append(pid)
    return out


def session_proc(cmd, names=SESSION_NAMES):
    return any(Path(t).name in names for t in cmd.split()[:2])


class Keeper:
    def __init__(self, cfg_path, dry=False):
        self.cfg_path = Path(cfg_path).resolve()
        self.cfg = json.loads(Path(cfg_path).read_text())
        self.repo = Path(self.cfg['repo'])
        self.state_dir = Path(self.cfg.get('state', self.repo.parent / f'{self.repo.name}-state'))
        self.k = self.state_dir / 'keeper'
        self.k.mkdir(parents=True, exist_ok=True)
        self.dry = dry
        self.sf = self.k / 'state.json'
        self.st = json.loads(self.sf.read_text()) if self.sf.exists() else {}
        for key in ('events', 'lanes', 'coordinator', 'remote', 'notify'):
            self.st.setdefault(key, {})

    # bookkeeping
    def log(self, msg):
        line = f'{stamp()} {"[dry] " if self.dry else ""}{msg}'
        print(line)
        if not self.dry:
            with open(self.k / 'keeper.log', 'a') as fh:
                fh.write(line + '\n')

    def event(self, d, kind, detail, key=None, name=None):
        key = key or f'{d}|{kind}'
        if key in self.st['events']:
            return
        name = name or Path(d).name
        self.st['events'][key] = {'lane': str(d), 'name': name, 'kind': kind, 'detail': detail, 't': now()}
        self.log(f'EVENT {kind} {name}: {detail}')
        line = f'{stamp()} {kind} {name} {detail} ({d})'
        if self.cfg.get('notify'):
            self.st['notify'].setdefault('queue', []).append(line)
        if not self.dry:
            with open(self.k / 'EVENTS', 'a') as fh:
                fh.write(line + '\n')

    def lane(self, d):
        return self.st['lanes'].setdefault(str(d), {})

    def save(self):
        if not self.dry:
            tmp = self.sf.with_name(f'state.json.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(self.st, indent=1))
            tmp.replace(self.sf)

    def run(self, argv, cwd, out=None):
        """Start a detached command (its own session, so it outlives the keeper)."""
        self.log('RUN ' + ' '.join(shlex.quote(a) for a in argv))
        if self.dry:
            return
        fh = open(out, 'a') if out else subprocess.DEVNULL
        process = subprocess.Popen([*self.cfg.get('detach', []), *argv], cwd=cwd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True, env={**os.environ, 'PATH': PATH, 'REPO': str(self.repo)})
        if out:
            fh.close()
        if self.cfg.get('detach'):
            time.sleep(1)   # systemd-run must leave the tick's cgroup before the tick ends

        return process

    # lanes
    @staticmethod
    def claude_lane(d):
        model = read_text(d / 'model').split()
        return bool(model and CLAUDE_MODEL.match(model[0]))

    def session_id(self, d):
        """Only an id with a session file of the lane's engine: `codex exec resume <unknown id>` silently starts a new,
        briefless session, and a Codex id cannot be resumed by claude after an engine switch."""
        sid = last_session_id(d / 'codex.log')
        if not sid:
            return None
        if Keeper.claude_lane(d):
            return sid if list(CLAUDE_PROJECTS.glob(f'*/{sid}.jsonl')) else None
        return sid if list(CODEX_SESSIONS.glob(f'*/*/*/*{sid}.jsonl')) else None

    def acked(self, d):
        return max(mtime(d / 'HANDLED'), mtime(d / 'RETIRED'), mtime(d / 'MERGED.md'))

    def closing(self, d):
        files = [f for f in os.listdir(d) if CLOSING.match(f)]
        return max(files, key=lambda f: mtime(d / f)) if files else None

    def handoff(self, d):
        if (d / 'HANDOFF.md').exists():
            return True
        started = float(read_text(d / 'session-at') or birth(d / 'codex.log'))
        later = sorted((p for p in d.glob('HANDOFF.*.md') if mtime(p) >= started and started),
                       key=mtime, reverse=True)
        if later and not self.dry:
            (d / 'HANDOFF.md').write_text(later[0].read_text())
        return bool(later)

    def launch(self, C, d, why):
        n = int((d / 'fresh').read_text()) if (d / 'fresh').exists() else 0
        if n >= FRESH_LIMIT:
            return self.event(d, 'FRESH-LIMIT', f'{n} fresh sessions')
        started = float(read_text(d / 'session-at') or birth(d / 'codex.log'))
        lived = mtime(d / 'codex.log') - started
        made_driver = mtime(d / 'driver.pid') >= started
        fresh_at = float((d / 'fresh-at').read_text() or 0) if (d / 'fresh-at').exists() else 0
        if (d / 'codex.log').exists() and lived < 300 and not made_driver and fresh_at and started >= fresh_at:
            return self.event(d, 'SHORT-SESSION', f'last fresh session lived {lived:.0f}s and started no driver')
        model = (d / 'model').read_text().split() if (d / 'model').exists() else []
        if not self.dry:
            (d / 'fresh').write_text(str(n + 1)); (d / 'fresh-at').write_text(str(int(now())))
        self.log(f'FRESH {d.name} ({why})')
        # Campaigns may carry a copied pre-guard launcher. Use the versioned
        # entrypoint so manual resumes and keeper launches share admission.
        self.run(['env', f'CAMPAIGN_DIR={C}', 'bash', str(SCRIPTS / 'launch.sh'), d.name, *model], self.repo, C / 'autoresume.log')

    def resume(self, C, d, msg_file, why):
        if not self.session_id(d):
            if Keeper.claude_lane(d):   # engine switched to Claude (or its session never started): brief + HANDOFF
                return self.launch(C, d, f'{why}; no Claude session to resume')
            return self.event(d, 'ENDED', f'{why}; no session id in codex.log to resume')
        self.log(f'RESUME {d.name} ({why})')
        self.run(['bash', str(SCRIPTS / 'resume.sh'), str(C), d.name, str(msg_file)], self.repo, C / 'autoresume.log')

    def msg(self, name, text):
        p = self.k / name
        if not p.exists() or p.read_text() != text:
            p.write_text(text)
        return p

    def costguard(self, C, lanes):
        g = C / 'costguard.py'
        if not lanes or not g.exists():
            return
        r = subprocess.run([sys.executable, str(g), *lanes], capture_output=True, text=True, timeout=120)
        for name in r.stdout.split():
            d = C / name
            pid = (d / 'pid').read_text().strip()
            if (d / 'forced-handoff').exists() and (d / 'forced-handoff').read_text().strip() == pid:
                continue
            self.log(f'COST-CAP {name}: stopping session {pid} for a handoff')
            if self.dry:
                continue
            for target in subprocess.run(['pgrep', '-P', pid], capture_output=True, text=True).stdout.split() + [pid]:
                try:
                    os.kill(int(target), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            time.sleep(3)
            if self.start_next_command(d):
                (d / 'forced-handoff').write_text(pid)
                continue
            self.run(['bash', str(SCRIPTS / 'resume.sh'), str(C), name, str(C / 'msg-handoff.md')], self.repo, C / 'autoresume.log')
            time.sleep(1)
            (d / 'forced-handoff').write_text((d / 'pid').read_text().strip())

    def start_next_command(self, d):
        """NEXT-COMMAND is one approved drive-loop argv, consumed after a detached start."""
        path = d / 'NEXT-COMMAND'
        if not path.exists(): return False
        if alive(d / 'driver.pid'): return True
        if (d / 'STOP').exists(): return False
        try:
            lines = path.read_text().strip().splitlines()
            if len(lines) != 1: raise ValueError('expected one command line')
            argv = shlex.split(lines[0])
            if argv and argv[0] in ('bash', '/bin/bash'): argv = argv[1:]
            if not argv or Path(argv[0]).resolve() != (SCRIPTS / 'drive-loop.sh').resolve():
                raise ValueError('expected the installed drive-loop.sh')
            args = argv[1:]
            if args and args[0] == '--once': args = args[1:]
            if len(args) < 5 or Path(args[0]).resolve() != d.resolve() or args[3] != '--' or int(args[2]) <= 0:
                raise ValueError('expected this lane, name, positive limit, --, driver argv')
        except (ValueError, OSError) as error:
            self.log(f'NEXT-COMMAND {d.name}: refused ({error})')
            return False
        process = self.run(['bash', *argv], self.repo, d / 'driver.log')
        if not self.dry:
            if not alive(d / 'driver.pid'):
                (d / 'driver.pid').write_text(str(process.pid) + '\n')
            path.unlink()
        return True

    def check_release_scope(self):
        """A scope contains commits or paths to READY files containing commit IDs."""
        configured = self.cfg.get('release_scope')
        if not configured: return
        path = Path(configured)
        if not path.is_absolute(): path = self.repo / path
        try: raw = path.read_bytes()
        except OSError as error:
            self.log(f'RELEASE-SCOPE: {error}'); return
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self.st.get('release_ready', []): return
        hold = read_text(self.state_dir / 'RELEASE-CYCLE-RUNNING')
        pid = re.search(r'^pid=(\d+)$', hold, re.M)
        if pid and alive(self.state_dir / 'RELEASE-CYCLE-RUNNING', pid=int(pid[1])): return
        items = []
        for line in raw.decode().splitlines():
            item = line.split('#', 1)[0].strip()
            if not item: continue
            if re.fullmatch(r'[0-9a-f]{7,40}', item):
                items.append(item); continue
            ready = path.parent / item
            if not ready.suffix: ready = ready.with_suffix('.md')
            text = read_text(ready)
            # Bases, incident commits and numeric measurements are not heads.
            commits = re.findall(r'\b(?:HEAD|commit)\s*[:=]?\s*`?([0-9a-f]{7,40})\b', text, re.I)
            if not commits: return
            items.extend(commits)
        if not items: return
        for commit in items:
            result = subprocess.run(['git', '-C', str(self.repo), 'merge-base', '--is-ancestor', commit, 'main'],
                                    capture_output=True)
            if result.returncode: return
        self.event(path, 'RELEASE-READY', f'all {len(items)} scope commits are on main; no release cycle running',
                   key=f'release|{digest}')
        self.st.setdefault('release_ready', []).append(digest)

    def waiting_heavy(self, d):
        for name in ('driver.log', 'codex.log'):
            lines = tail_text(d / name).splitlines()
            if lines and re.search(r'heavy\.sh:.*(?:waiting|lock|slots?)', lines[-1]): return True
        # Slot-count waits can be silent. A heavy wrapper sleeping under this
        # lane is queued; a wrapper with the actual job child is running work.
        result = subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True, text=True)
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split(None, 2)
            if len(fields)==3 and fields[0].isdigit() and fields[1].isdigit():
                rows.append((int(fields[0]), int(fields[1]), fields[2]))
        descendants = {int(read_text(d / name)) for name in ('pid', 'driver.pid') if read_text(d / name).isdigit()}
        while True:
            found = {pid for pid, parent, _ in rows if parent in descendants}
            if found <= descendants: break
            descendants.update(found)
        wrappers = {pid for pid, _, command in rows if pid in descendants and
                    re.search(r'(?:^|\s)(?:\S*/)?heavy\.sh(?:\s|$)', command)}
        return any(parent in wrappers and re.match(r'(?:\S*/)?sleep(?:\s|$)', command)
                   for _, parent, command in rows)

    def tend(self, C, d):
        L = self.lane(d)
        s_alive, d_alive = alive(d / 'pid'), alive(d / 'driver.pid')
        quiet = now() - newest_in(d)
        # A READY-<n> is a checkpoint for the integrator, not the end of the lane: report it even while work goes on.
        for f in os.listdir(d):
            if f.startswith('READY-') and CLOSING.match(f) and mtime(d / f) > self.acked(d):
                self.event(d, 'READY', f'{f} written {stamp(mtime(d / f))}', key=f'{d}|{f}|{int(mtime(d / f))}')
        if s_alive or d_alive:
            if self.waiting_heavy(d):
                self.event(d, 'HEAVY-WAIT', 'waiting on the heavy lock; lane remains running',
                           key=f'{d}|HEAVY-WAIT|{int(mtime(d / "driver.pid"))}')
                return
            if quiet > QUIET:
                bucket = int(newest_in(d))
                self.event(d, 'QUIET', f'{"session" if s_alive else "driver"} running, lane folder silent {quiet/60:.0f} min',
                           key=f'{d}|QUIET|{bucket}')
            return
        close = self.closing(d)
        if close and close.startswith('READY-') and self.handoff(d) and mtime(d / 'HANDOFF.md') > mtime(d / close):
            close = None   # checkpoint already reported above; a handoff written after it says the queue goes on
        if close:
            if mtime(d / close) > self.acked(d):
                self.event(d, close.split('.')[0].split('-')[0], f'{close} written {stamp(mtime(d / close))}',
                           key=f'{d}|{close}|{int(mtime(d / close))}')
            return
        if (d / 'STOP').exists():
            return
        if self.start_next_command(d): return
        cmd = d / 'driver.cmd'
        exit_note = (d / 'DRIVER-EXIT').read_text().strip() if (d / 'DRIVER-EXIT').exists() else None
        if (d / 'driver.pid').exists() and cmd.exists() and exit_note in (None, 'term') and not (d / 'DRIVER-FAILED.md').exists():
            spec = json.loads(cmd.read_text())
            name = [a for a in spec['args'] if a != '--once'][1]
            if subprocess.run(['pgrep', '-f', f'sleepsafe.py --name {name} '], capture_output=True).returncode == 0:
                return self.event(d, 'DRIVER-ORPHAN', f'drive-loop is gone but its driver {name} still runs', key=f'{d}|ORPHAN|{int(mtime(d / "driver.pid"))}')
            L['restarts'] = [t for t in L.get('restarts', []) if now() - t < 6 * 3600]
            if len(L['restarts']) >= 3:
                return self.event(d, 'DRIVER-DYING', 'drive-loop killed three times in six hours', key=f'{d}|DYING|{int(L["restarts"][0])}')
            L['restarts'].append(now())
            self.log(f'DRIVER-RESTART {d.name}: drive-loop was killed ({exit_note or "no exit recorded"})')
            return self.run(['bash', str(SCRIPTS / 'drive-loop.sh'), *spec['args']], spec['cwd'], d / 'driver.log')
        failed = d / 'DRIVER-FAILED.md'
        if failed.exists() and mtime(failed) > self.acked(d):
            if (d / 'pid').exists() and L.get('driver_failed') != int(mtime(failed)):
                L['driver_failed'] = int(mtime(failed))
                return self.resume(C, d, self.msg('msg-driver-failed.md', MSG_DRIVER_FAILED), 'driver failed')
            return self.event(d, 'DRIVER-FAILED', 'three failed runs; see DRIVER-FAILED.md', key=f'{d}|DF|{int(mtime(failed))}')
        if not (d / 'pid').exists():
            ex = d / 'DRIVER-EXIT'
            if ex.exists() and mtime(ex) > self.acked(d):
                code = ex.read_text().strip()
                self.event(d, 'JOB-FINISHED' if code == '0' else 'JOB-FAILED', f'drive-loop exit {code}', key=f'{d}|EXIT|{int(mtime(ex))}')
            return
        if self.handoff(d):
            return self.launch(C, d, 'handoff, no live driver')
        tail = tail_text(d / 'codex.log')
        if TRANSIENT.search(tail):
            streak = L.get('transient', 0)
            wait = 60 if streak < 3 else 300
            if now() - mtime(d / 'codex.log') < wait:
                return
            L['transient'] = streak + 1
            return self.resume(C, d, C / 'msg-capacity.md', f'429/capacity #{streak + 1}')
        L['transient'] = 0
        sid = self.session_id(d)
        if L.get('continue_sid') != sid:
            L['continue_sid'], L['continues'] = sid, 0
        if L.get('continues', 0) < CONTINUE_LIMIT:
            L['continues'] = L.get('continues', 0) + 1
            L['continued_at'] = now()
            return self.resume(C, d, self.msg('msg-keeper-continue.md', MSG_CONTINUE), f'ended without DONE/READY/HANDOFF, continue #{L["continues"]}')
        self.event(d, 'ENDED', f'session ended {CONTINUE_LIMIT + 1} times without DONE/READY/HANDOFF', key=f'{d}|ENDED|{sid}')

    def lanes(self):
        for c in self.cfg.get('campaigns', []):
            C = Path(c)
            if not C.is_dir():
                continue
            yield C, [d for d in sorted(C.iterdir()) if d.is_dir() and ((d / 'pid').exists() or (d / 'driver.pid').exists())]

    def clear_events(self, remote_pending):
        for key, e in list(self.st['events'].items()):
            if key.startswith('remote|'):
                host = e['lane'].split(':', 1)[0]
                if host in remote_pending and (e['lane'], e['kind']) not in remote_pending[host]:
                    del self.st['events'][key]
                continue
            d = Path(e['lane'])
            if not d.exists() or self.acked(d) > e['t']:
                del self.st['events'][key]

    # another machine's keeper
    def pull_remote(self):
        """New lines of each remote keeper's EVENTS become events here. Returns {host: {(host:lane, kind)}} of what the
        remote keeper still has pending, for hosts that answered; an unreachable host keeps its events as they are."""
        pending = {}
        for r in self.cfg.get('remote_events', []):
            host, path = r['ssh'], r['path']
            prefix = r.get('prefix', host.split('-')[0])
            R = self.st['remote'].setdefault(f'{host}:{path}', {'offset': 0})
            q = shlex.quote(path)
            state = shlex.quote(str(Path(path).parent / 'state.json'))
            beat = shlex.quote(str(Path(path).parent / 'heartbeat'))
            cmd = (f'f={q}; s=$(wc -c < "$f" 2>/dev/null || echo 0); echo "$s $(( $(date +%s) - $(stat -c %Y {beat} 2>/dev/null || echo 0) ))"; '
                   f'o={R["offset"]}; [ "$s" -lt "$o" ] && o=0; echo "$o"; tail -c +$((o + 1)) "$f" 2>/dev/null | head -c $((s - o)); '
                   f'echo; echo "--state--"; cat {state} 2>/dev/null')
            try:
                out = subprocess.run([*SSH, host, cmd], capture_output=True, text=True, timeout=40)
                if out.returncode != 0:
                    raise OSError(f'ssh exit {out.returncode}: {out.stderr.strip()[:200]}')
                head, rest = out.stdout.split('\n', 1)
                _, age = (int(x) for x in head.split())
                start, rest = rest.split('\n', 1)
                start_off = int(start)
                body, _, state_txt = rest.rpartition('\n--state--\n')
            except (OSError, ValueError, subprocess.TimeoutExpired) as e:
                if not R.get('failing'):
                    R['failing'] = True
                    self.log(f'REMOTE {host}: cannot read {path} ({e}); keeping its events, will retry each tick')
                continue
            if R.pop('failing', None):
                self.log(f'REMOTE {host}: reachable again')
            complete = body[:body.rfind('\n') + 1]   # a line still being written waits for the next tick
            for line in complete.splitlines():
                m = re.match(r'^(\S+ \S+) (\S+) (\S+) (.*?)(?: \((/[^)]*)\))?$', line)
                if not m:
                    continue
                t, kind, name, detail, lane = m.groups()
                self.event(f'{host}:{lane or name}', kind, f'{detail} [{t} on {host}]', key=f'remote|{host}|{line}',
                           name=f'{prefix}:{name}')
            R['offset'] = start_off + len(complete.encode())
            if age > 900:
                self.event(f'{host}:{Path(path).parent}', 'REMOTE-STALE', f'keeper heartbeat on {host} is {age // 60} min old',
                           key=f'remote|{host}|stale|{int(now() // 21600)}', name=f'{prefix}:keeper')
            try:
                rs = json.loads(state_txt) if state_txt.strip() else {}
            except ValueError:
                continue
            pending[host] = {(f'{host}:{e["lane"]}', e['kind']) for e in rs.get('events', {}).values()}
            if age > 900:
                pending[host].add((f'{host}:{Path(path).parent}', 'REMOTE-STALE'))
        return pending

    def notify(self):
        """At most one mail per gap, carrying every event since the last one. A failed send keeps the queue."""
        c, N = self.cfg.get('notify'), self.st['notify']
        if not c or not N.get('queue') or self.dry or not os.access(c['cmd'], os.X_OK):
            return
        if now() - N.get('last', 0) < c.get('min_gap_minutes', 30) * 60:
            return
        q = N['queue']
        subject = f'{self.cfg.get("name", self.repo.name)}/keeper: {len(q)} event{"s" if len(q) != 1 else ""}'
        body = 'Lane keeper events since the last mail:\n\n' + '\n'.join(q[-60:]) + f'\n\nAll events: {self.k / "EVENTS"}\n'
        try:
            ok = subprocess.run([c['cmd'], subject, body], capture_output=True, timeout=90).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        N['last'] = now()
        if ok:
            N['queue'] = []
        self.log(f'NOTIFY {"sent" if ok else "failed"}: {len(q)} events')

    # coordinator
    def coordinator_live(self):
        """Retired IDs never revive; other sessions need a watch, headless PID or recent work."""
        retired = mtime(self.k / 'retired')
        ids = retired_ids(self.k / 'retired')
        beat = mtime(self.k / 'watch.heartbeat')
        owner = read_text(self.k / 'watch.heartbeat').split()
        watch_alive = not owner or (len(owner) == 2 and alive(self.k / 'watch.heartbeat', owner[1]))
        if watch_alive and now() - beat < 180 and beat > retired and (not owner or owner[0] not in ids):
            return 'watch heartbeat'
        sid = read_text(self.k / 'coordinator.session')
        if sid not in ids and alive(self.k / 'coordinator.pid') and mtime(self.k / 'coordinator.pid') > retired:
            return 'keeper-started coordinator'
        for f in REG.glob('*.json') if REG.exists() else []:
            if f.stem in ids:
                continue
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if r.get('repo') != str(self.repo) or not r.get('transcript'):
                continue
            t = transcript_activity(r['transcript'])
            if now() - t < COORD_LIVE and (mtime(f) > retired or t > retired + RETIRE_GRACE):
                return f'session {f.stem[:8]} active'
        return None

    def retire(self, sid=None):
        """Publish retirement before removing registration: the writer must not recreate it."""
        sid = sid or os.environ.get('CLAUDE_CODE_SESSION_ID') or os.environ.get('CLAUDE_SESSION_ID')
        if sid and not re.fullmatch(r'[0-9A-Za-z-]+', sid):
            raise ValueError('invalid session id')
        with (self.k / 'retired').open('a') as fh:
            fh.write(f'{stamp()} {sid or "-"}\n')
        removed = ''
        if sid:
            f = REG / f'{sid}.json'
            try:
                if json.loads(f.read_text()).get('repo') == str(self.repo):
                    f.unlink(); removed = f'; removed {f}'
            except (OSError, ValueError):
                pass
        self.log(f'RETIRE coordinator {sid or "(no session id)"}{removed}')

    def handover(self):
        files = [p for p in self.state_dir.glob(self.cfg.get('handover_glob', '*HANDOVER*.md')) if p.is_file()]
        return max(files, key=mtime) if files else None

    def maybe_start_coordinator(self):
        pending = self.st['events']
        if not pending:
            return
        live = self.coordinator_live()
        if live:
            return
        c = self.cfg.get('coordinator', {})
        S = self.st['coordinator']
        today = time.strftime('%F')
        if S.get('day') != today:
            S['day'], S['starts'] = today, 0
        after_retire = mtime(self.k / 'retired') > S.get('last', 0)   # once: the start below moves 'last' past it
        if not after_retire and now() - S.get('last', 0) < c.get('min_gap_minutes', 10) * 60:
            return
        keys = sorted(pending)
        if keys == S.get('last_keys') and S.get('same', 0) >= 2:
            if not S.get('stuck_noted'):
                S['stuck_noted'] = True
                self.log('COORDINATOR-STUCK: two coordinators left the same events pending; waiting for a human or an ack')
                (self.k / 'COORDINATOR-STUCK.md').write_text(f'{stamp()}: these events survived two coordinator runs:\n\n'
                                                            + '\n'.join(f'- {pending[k]["kind"]} {pending[k]["lane"]}: {pending[k]["detail"]}' for k in keys) + '\n')
            return
        if S['starts'] >= c.get('max_starts_per_day', 16):
            return
        S['same'] = S.get('same', 0) + 1 if keys == S.get('last_keys') else 1
        S.update(last=now(), last_keys=keys, starts=S['starts'] + 1, stuck_noted=False)
        handover = self.handover()
        events = '\n'.join(f'- {e["kind"]} {e["lane"]}: {e["detail"]}' for e in pending.values())
        script = SCRIPTS / 'lane-keeper.py'
        out = self.k / f'coordinator-{time.strftime("%Y%m%d-%H%M")}.log'
        sid = str(uuid.uuid4())
        retire = f'python3 {script} retire --config {self.cfg_path} --session {sid}'
        prompt = f"""You are the integrator and coordinator for {self.repo}, started headless by lane-keeper at {stamp()} because \
events are pending and no coordinator was live. Read the newest handover first: {handover}. Then the repository rules.

Pending events (also in {self.k / 'EVENTS'}):
{events}

Handle each event: integrate READY checkpoints by the repository's integration rules, review DONE files (a DONE before \
the queue is exhausted is a failure: resume or relaunch the lane), fix or relaunch failed drivers. After handling a \
lane run `python3 {script} ack <lane-dir> "<one line: what you did>"`, or write RETIRED there with the reason.
Git safety: integrate in a worktree you create for it. If you must use the main checkout, run `git status` there \
first, and never run `git reset --hard`, `git checkout -- <path>`, `git restore`, `git stash` or `git clean` in a checkout \
you did not create: other sessions keep uncommitted work in them.
Do not watch long jobs from this session and do not arm monitors. Start long jobs detached under \
`{SCRIPTS / 'drive-loop.sh'}` in a folder under one of the campaign dirs (use --once for a job that must not rerun on \
failure, such as a release cycle); the keeper reports how they end and starts a coordinator then.
Before you end, write {self.state_dir}/keeper-HANDOVER-<YYYYMMDD-HHMM>.md (under 400 words): what you did, what runs, \
what each remaining item waits on. The next coordinator starts from it. Then run `{retire}` and end."""
        self.log(f'COORDINATOR-START ({len(pending)} events; handover {handover and handover.name}) → {out.name}')
        self.log(f'COORDINATOR-SESSION {sid}: log {out}; take over with kill -9 <coordinator.pid>, then '
                 f'cd {self.repo} && claude --resume {sid}')
        if self.dry:
            return
        (self.k / 'coordinator-prompt.txt').write_text(prompt)
        (self.k / 'coordinator.session').write_text(sid + '\n')
        p = subprocess.Popen(self.coordinator_cmd(prompt, sid),
                             cwd=self.repo, stdin=subprocess.DEVNULL, stdout=open(out, 'a'), stderr=subprocess.STDOUT,
                             start_new_session=True, env={**os.environ, 'PATH': PATH})
        (self.k / 'coordinator.pid').write_text(str(p.pid))

    @staticmethod
    def coordinator_cmd(prompt, sid):
        return [*CAFFEINATE, 'claude', '-p', prompt, '--permission-mode', 'bypassPermissions', '--session-id', sid,
                '--output-format', 'stream-json', '--verbose']

    # runaways
    def kill_pids(self, pids, why):
        if self.dry:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for p in pids:
                try: os.kill(p, sig)
                except (ProcessLookupError, PermissionError): pass
            if sig == signal.SIGKILL:
                return
            deadline = now() + self.cfg.get('reap_grace_seconds', 10)
            while now() < deadline:
                left = []
                for p in pids:
                    try: os.kill(p, 0); left.append(p)
                    except (ProcessLookupError, PermissionError): pass
                # a killed child stays a zombie until its (dead) parent's reaper collects it
                left = [p for p in left if 'Z' not in subprocess.run(['ps', '-o', 'stat=', '-p', str(p)],
                                                                     capture_output=True, text=True).stdout]
                if not left: return
                time.sleep(0.5)
            pids = left
            self.log(f'REAP {why}: SIGKILL {pids}')

    def reap(self, table, pid, protected, kind):
        targets = {pid} | descendants(table, [pid])
        if table[pid]['pgid'] == pid:
            targets |= {p for p, r in table.items() if r['pgid'] == pid}
        targets -= protected
        for p in sorted(targets):
            r = table[p]
            self.log(f'REAP {kind} {p} rss={r["rss"] // 1024}M age={r["age"] // 60}m {r["cmd"][:300]}')
        self.kill_pids(sorted(targets), kind)

    def runaway_event(self, seen, kind, pid, r, detail, reaping):
        key = f'{kind}|{pid}|{hashlib.sha1(r["cmd"].encode()).hexdigest()[:8]}'
        if key in seen:
            return False
        seen[key] = now()
        label = 'reaping ' if reaping else 'would reap ' if self.dry and self.cfg.get('reap_orphans') else ''
        self.event(self.k, kind, f'{label}pid {pid} rss={r["rss"] // 1024}M '
                   f'age={r["age"] // 60}m {detail} {r["cmd"][:200]}', key=f'runaway|{key}', name='machine')
        return True

    def runaways(self):
        c = self.cfg
        reap = bool(c.get('reap_orphans')) and not self.dry
        table = ps_table()
        seen = self.st.setdefault('runaways', {})
        for key in list(seen):
            kind, _, rest = key.partition('|')
            pid = rest.split('|')[0]
            if kind == 'resource' and now() - seen[key] > 86400 or kind != 'resource' and int(pid) not in table:
                del seen[key]
        sessions = [p for p, r in table.items() if session_proc(r['cmd'])]
        tracked = [self.k / 'coordinator.pid']
        lanes = [(C, d) for C, ds in self.lanes() for d in ds]
        for _, d in lanes:
            tracked += [d / 'pid', d / 'driver.pid']
        tracked_pids = [int(t) for t in map(read_text, tracked) if t.isdigit()]
        protected = {os.getpid(), *ancestors(table, os.getpid())} | descendants(table, [os.getpid(), *sessions, *tracked_pids])
        roots = [str(self.repo / '.claude/worktrees'), str(self.repo / '.codex/worktrees'),
                 *map(str, c.get('campaigns', [])), *map(str, c.get('worktree_roots', []))]
        roots = [r.rstrip('/') + '/' for r in roots]
        preview = re.compile('|'.join(c.get('preview_patterns', PREVIEW_PATTERNS)))
        old = [p for p, r in table.items() if p not in protected and r['ppid'] == 1
               and r['age'] >= c.get('orphan_age_minutes', 30) * 60]
        cwd = proc_cwds([p for p in old if not any(root in table[p]['cmd'] for root in roots)])

        def under(p, paths):
            where = cwd.get(p, '') + '/'
            return any(x in table[p]['cmd'] or where.startswith(x) for x in paths)

        # STOPPED-BUT-RUNNING
        for _, d in lanes:
            stop = d / 'STOP'
            if not stop.exists() or now() - mtime(stop) < 600 or not alive(d / 'driver.pid'):
                continue
            pid = int(read_text(d / 'driver.pid'))
            if pid not in table:
                continue
            group = [pid]
            spec = json.loads(read_text(d / 'driver.cmd') or '{}')
            name = [a for a in spec.get('args', []) if a != '--once'][1:2]
            if name:
                group += [p for p, r in table.items() if f'sleepsafe.py --name {name[0]} ' in r['cmd'] + ' ']
            tree = descendants(table, group)
            keep = protected - tree
            for p in group:
                if self.runaway_event(seen, 'STOPPED-BUT-RUNNING', p, table[p], f'lane {d.name} STOP {stamp(mtime(stop))}', reap) and reap:
                    self.reap(table, p, keep, 'STOPPED-BUT-RUNNING')
        # STALE-PREVIEW
        preview_roots = [str(self.repo).rstrip('/') + '/', *roots]
        for p, r in table.items():
            if p in protected or not preview.search(r['cmd']) or r['age'] < c.get('preview_age_hours', 6) * 3600:
                continue
            if not any(x in r['cmd'] for x in preview_roots):
                continue
            if any(session_proc(table[a]['cmd'], INTERACTIVE_NAMES) for a in ancestors(table, p) if a in table):
                continue
            if self.runaway_event(seen, 'STALE-PREVIEW', p, r, '', reap) and reap:
                self.reap(table, p, protected, 'STALE-PREVIEW')
        # ORPHAN
        for p in old:
            if preview.search(table[p]['cmd']) or not under(p, roots):
                continue
            if self.runaway_event(seen, 'ORPHAN', p, table[p], f'cwd={cwd.get(p, "?")}', reap) and reap:
                self.reap(table, p, protected, 'ORPHAN')
        # RESOURCE-FLOOR
        free, swap = disk_free_gb(self.repo), swap_used_gb()
        floor, ceiling = c.get('disk_floor_gb', 30), c.get('swap_ceiling_gb', 20)
        reasons = ([f'free disk {free:.1f} GB < {floor} GB'] if free < floor else []) + \
                  ([f'swap used {swap:.1f} GB > {ceiling} GB'] if swap is not None and swap > ceiling else [])
        key = f'resource|{time.strftime("%Y%m%d%H")}'
        if reasons and key not in seen:
            seen[key] = now()
            self.event(self.k, 'RESOURCE-FLOOR', '; '.join(reasons) + f'; top: {top_processes(table)}; largest: '
                       f'{largest_dirs(c.get("disk_watch", []))}', key=f'runaway|{key}', name='machine')

    def tick(self):
        lock = open(self.k / 'tick.lock', 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        (self.k / 'heartbeat').touch()
        for C, ds in self.lanes():
            (C / '.watcher-heartbeat').touch()   # launch.sh refuses without it: the keeper now runs the cost guard
            self.costguard(C, [d.name for d in ds if alive(d / 'pid')])
            for d in ds:
                try:
                    self.tend(C, d)
                except Exception as e:  # one lane's odd files must not stop the others
                    self.log(f'ERROR {d.name}: {e!r}')
        try:
            self.runaways()
        except Exception as e:
            self.log(f'ERROR runaways: {e!r}')
        self.check_release_scope()
        self.clear_events(self.pull_remote())
        self.maybe_start_coordinator()
        self.notify()
        self.save()


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args else 'status'
    cfg = DEFAULT_CONFIG
    if '--config' in args:
        i = args.index('--config'); cfg = Path(args[i + 1]); del args[i:i + 2]
    dry = '--dry-run' in args
    if mode == 'ack':
        host, sep, rpath = args[0].partition(':')
        if sep and rpath.startswith('/') and '/' not in host:
            note = shlex.quote(f'{stamp()} {" ".join(args[1:])}')
            r = subprocess.run([*SSH, host, f'd={shlex.quote(rpath)}; [ -d "$d" ] && printf "%s\\n" {note} > "$d/HANDLED"'])
            print(f'acked {host}:{Path(rpath).name}' if r.returncode == 0 else f'ack failed on {host} (exit {r.returncode})')
            sys.exit(r.returncode)
        d = Path(args[0]).resolve()
        (d / 'HANDLED').write_text(f'{stamp()} {" ".join(args[1:])}\n')
        print(f'acked {d.name}')
        return
    K = Keeper(cfg, dry)
    if mode == 'tick':
        K.tick()
    elif mode == 'watch':
        ev = K.k / 'EVENTS'
        ev.touch()
        pos = ev.stat().st_size
        started = now()
        sid = os.environ.get('CLAUDE_CODE_SESSION_ID') or os.environ.get('CLAUDE_SESSION_ID') or '-'
        beat = K.k / 'watch.heartbeat'
        owner = f'{sid} {os.getpid()}\n'
        try:
            while True:
                if sid in retired_ids(K.k / 'retired') or mtime(K.k / 'retired') > started:
                    print('retired: this watch stops so the keeper can start a successor'); return
                beat.write_text(owner)
                if ev.stat().st_size > pos:
                    with open(ev) as fh:
                        fh.seek(pos); sys.stdout.write(fh.read()); sys.stdout.flush(); pos = fh.tell()
                time.sleep(30)
        finally:
            if read_text(beat) == owner.strip():
                beat.unlink(missing_ok=True)
    elif mode == 'retire':
        sid = args[args.index('--session') + 1] if '--session' in args else None
        K.retire(sid)
    elif mode == 'status':
        print(f'keeper heartbeat {now() - mtime(K.k / "heartbeat"):.0f}s ago; coordinator live: {K.coordinator_live()}')
        for name in ('retired', 'coordinator.session'):
            if (K.k / name).exists():
                print(f'{name}: {(K.k / name).read_text().strip()}')
        for e in K.st['events'].values():
            print(f'  {e["kind"]:14} {e.get("name") or Path(e["lane"]).name}: {e["detail"]}')


if __name__ == '__main__':
    main()
