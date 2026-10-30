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
coordinator still running, or a registered coordinator transcript written in the last 20 minutes.

    lane-keeper.py tick   [--config F] [--dry-run]
    lane-keeper.py watch  [--config F]          # Monitor command for an interactive coordinator
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
Linux: systemd timer instead of launchd; CODEX_HOME is honoured; caffeinate is used only where it exists.
"""
import fcntl, json, os, re, shlex, shutil, signal, subprocess, sys, time
from pathlib import Path

DEFAULT_CONFIG = Path.home() / 'src/skard/.coordinator.json'
SCRIPTS = Path(__file__).resolve().parent
REG = Path.home() / '.agents/coordinators'
CLOSING = re.compile(r'^(DONE\.md|READY-[0-9][0-9a-z]*\.md|DRIVER-FINISHED\.json|RETIRED)$')
TRANSIENT = re.compile(r'at capacity|429 Too Many|retry limit')
FRESH_LIMIT, CONTINUE_LIMIT, QUIET = 25, 2, 3600
COORD_LIVE = 1200
_FNM = Path.home() / 'Library/Application Support/fnm/node-versions/v24.11.1/installation/bin'
PATH = ':'.join(dict.fromkeys([*([str(_FNM)] if _FNM.is_dir() else []), str(Path.home() / '.local/bin'), '/opt/homebrew/bin',
                               *os.environ.get('PATH', '').split(':'), '/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin']))
CODEX_SESSIONS = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')) / 'sessions'
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


def alive(pidfile):
    """A pid file is written after its process starts; a process that started later holds a recycled pid."""
    try:
        pid = int(Path(pidfile).read_text().split()[0])
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


class Keeper:
    def __init__(self, cfg_path, dry=False):
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
        subprocess.Popen([*self.cfg.get('detach', []), *argv], cwd=cwd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True, env={**os.environ, 'PATH': PATH, 'REPO': str(self.repo)})
        if self.cfg.get('detach'):
            time.sleep(1)   # systemd-run must leave the tick's cgroup before the tick ends

    # lanes
    def session_id(self, d):
        """Only an id with a session file: `codex exec resume <unknown id>` silently starts a new, briefless session."""
        try:
            m = re.search(r'session id: (\S+)', (d / 'codex.log').read_text(errors='ignore')[:6000])
        except OSError:
            return None
        if m and list(CODEX_SESSIONS.glob(f'*/*/*/*{m.group(1)}.jsonl')):
            return m.group(1)
        return None

    def acked(self, d):
        return max(mtime(d / 'HANDLED'), mtime(d / 'RETIRED'), mtime(d / 'MERGED.md'))

    def closing(self, d):
        files = [f for f in os.listdir(d) if CLOSING.match(f)]
        return max(files, key=lambda f: mtime(d / f)) if files else None

    def handoff(self, d):
        if (d / 'HANDOFF.md').exists():
            return True
        started = birth(d / 'codex.log')
        later = sorted((p for p in d.glob('HANDOFF.*.md') if mtime(p) >= started and started),
                       key=mtime, reverse=True)
        if later and not self.dry:
            (d / 'HANDOFF.md').write_text(later[0].read_text())
        return bool(later)

    def launch(self, C, d, why):
        n = int((d / 'fresh').read_text()) if (d / 'fresh').exists() else 0
        if n >= FRESH_LIMIT:
            return self.event(d, 'FRESH-LIMIT', f'{n} fresh sessions')
        lived = mtime(d / 'codex.log') - birth(d / 'codex.log')
        made_driver = mtime(d / 'driver.pid') >= birth(d / 'codex.log')
        fresh_at = float((d / 'fresh-at').read_text() or 0) if (d / 'fresh-at').exists() else 0
        if (d / 'codex.log').exists() and lived < 300 and not made_driver and fresh_at and birth(d / 'codex.log') >= fresh_at:
            return self.event(d, 'SHORT-SESSION', f'last fresh session lived {lived:.0f}s and started no driver')
        model = (d / 'model').read_text().split() if (d / 'model').exists() else []
        if not self.dry:
            (d / 'fresh').write_text(str(n + 1)); (d / 'fresh-at').write_text(str(int(now())))
        self.log(f'FRESH {d.name} ({why})')
        self.run(['bash', str(C / 'launch.sh'), d.name, *model], self.repo, C / 'autoresume.log')

    def resume(self, C, d, msg_file, why):
        if not self.session_id(d):
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
            for child in subprocess.run(['pgrep', '-P', pid], capture_output=True, text=True).stdout.split():
                os.kill(int(child), signal.SIGTERM)
            os.kill(int(pid), signal.SIGTERM)
            time.sleep(3)
            self.run(['bash', str(SCRIPTS / 'resume.sh'), str(C), name, str(C / 'msg-handoff.md')], self.repo, C / 'autoresume.log')
            time.sleep(1)
            (d / 'forced-handoff').write_text((d / 'pid').read_text().strip())

    def tend(self, C, d):
        L = self.lane(d)
        s_alive, d_alive = alive(d / 'pid'), alive(d / 'driver.pid')
        quiet = now() - newest_in(d)
        # A READY-<n> is a checkpoint for the integrator, not the end of the lane: report it even while work goes on.
        for f in os.listdir(d):
            if f.startswith('READY-') and CLOSING.match(f) and mtime(d / f) > self.acked(d):
                self.event(d, 'READY', f'{f} written {stamp(mtime(d / f))}', key=f'{d}|{f}|{int(mtime(d / f))}')
        if s_alive or d_alive:
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
        tail = (d / 'codex.log').read_text(errors='ignore')[-4000:] if (d / 'codex.log').exists() else ''
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
        if now() - mtime(self.k / 'watch.heartbeat') < 180:
            return 'watch heartbeat'
        if alive(self.k / 'coordinator.pid'):
            return 'keeper-started coordinator'
        for f in REG.glob('*.json') if REG.exists() else []:
            try:
                r = json.loads(f.read_text())
            except ValueError:
                continue
            if r.get('repo') == str(self.repo) and r.get('transcript') and now() - mtime(r['transcript']) < COORD_LIVE:
                return f'session {f.stem[:8]} active'
        return None

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
        if now() - S.get('last', 0) < c.get('min_gap_minutes', 10) * 60:
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
        prompt = f"""You are the integrator and coordinator for {self.repo}, started headless by lane-keeper at {stamp()} because \
events are pending and no coordinator was live. Read the newest handover first: {handover}. Then the repository rules.

Pending events (also in {self.k / 'EVENTS'}):
{events}

Handle each event: integrate READY checkpoints by the repository's integration rules, review DONE files (a DONE before \
the queue is exhausted is a failure: resume or relaunch the lane), fix or relaunch failed drivers. After handling a \
lane run `python3 {script} ack <lane-dir> "<one line: what you did>"`, or write RETIRED there with the reason.
Do not watch long jobs from this session and do not arm monitors. Start long jobs detached under \
`{SCRIPTS / 'drive-loop.sh'}` in a folder under one of the campaign dirs (use --once for a job that must not rerun on \
failure, such as a release cycle); the keeper reports how they end and starts a coordinator then.
Before you end, write {self.state_dir}/keeper-HANDOVER-<YYYYMMDD-HHMM>.md (under 400 words): what you did, what runs, \
what each remaining item waits on. The next coordinator starts from it."""
        self.log(f'COORDINATOR-START ({len(pending)} events; handover {handover and handover.name}) → {out.name}')
        if self.dry:
            return
        (self.k / 'coordinator-prompt.txt').write_text(prompt)
        p = subprocess.Popen([*CAFFEINATE, 'claude', '-p', prompt, '--permission-mode', 'bypassPermissions'],
                             cwd=self.repo, stdin=subprocess.DEVNULL, stdout=open(out, 'a'), stderr=subprocess.STDOUT,
                             start_new_session=True, env={**os.environ, 'PATH': PATH})
        (self.k / 'coordinator.pid').write_text(str(p.pid))

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
        while True:
            (K.k / 'watch.heartbeat').touch()
            if ev.stat().st_size > pos:
                with open(ev) as fh:
                    fh.seek(pos); sys.stdout.write(fh.read()); sys.stdout.flush(); pos = fh.tell()
            time.sleep(30)
    elif mode == 'status':
        print(f'keeper heartbeat {now() - mtime(K.k / "heartbeat"):.0f}s ago; coordinator live: {K.coordinator_live()}')
        for e in K.st['events'].values():
            print(f'  {e["kind"]:14} {e.get("name") or Path(e["lane"]).name}: {e["detail"]}')


if __name__ == '__main__':
    main()
