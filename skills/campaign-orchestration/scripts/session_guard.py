#!/usr/bin/env python3
"""Serialize launch/resume admission before either script changes lane files."""
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from driver_state import prepare as prepare_driver_state


def tail_text(path, limit=4000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell()-limit))
            return stream.read(limit).decode(errors='replace')
    except OSError: return ''


SID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
HEADER = re.compile(rb'session id: ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\r?', re.I)


def last_session_id(path):
    """Latest complete CLI header or thread.started event, never echoed prose.

    Read backwards with bounded memory. Do not fall back to an older identity
    when the latest real header names a missing session; the caller refuses it.
    """
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2); offset = stream.tell(); suffix = b''
            while offset:
                size = min(offset, 65536); offset -= size; stream.seek(offset)
                lines = (stream.read(size)+suffix).split(b'\n')
                if offset:
                    suffix = lines.pop(0)
                    if len(suffix) > 256: suffix = b'\x00'  # oversized prose cannot be an identity line
                for line in reversed(lines):
                    match = HEADER.fullmatch(line)
                    if match: return match[1].decode().lower()
                    if line.startswith(b'{'):
                        try: row = json.loads(line)
                        except (ValueError, UnicodeError): continue
                        if isinstance(row, dict) and row.get('type') == 'thread.started':
                            sid = row.get('thread_id')
                            if isinstance(sid, str) and SID.fullmatch(sid): return sid.lower()
    except OSError: pass
    return None


def local_codex_session(sid, sessions=None):
    """Resume only a persisted local session whose metadata confirms the ID."""
    if not isinstance(sid, str) or not SID.fullmatch(sid): return None
    sessions = Path(sessions) if sessions is not None else Path(os.environ.get('CODEX_HOME', Path.home()/'.codex'))/'sessions'
    for path in sessions.glob('*/*/*/*'+sid+'.jsonl'):
        try:
            with path.open('rb') as stream:
                row = json.loads(stream.readline(1024*1024))
            if row.get('type') == 'session_meta' and (row.get('payload') or {}).get('id') == sid:
                return path
        except (OSError, ValueError, AttributeError): pass
    return None


def require_lane_workspace(lane):
    """Existing ownership is authoritative; a stale parked checkout is never recreated here."""
    lane = Path(lane)
    prepare_driver_state(lane)
    if (lane / 'RETIRED').exists():
        raise ValueError('retired lane: ' + str(lane))
    registry = Path(os.environ.get('LANE_REGISTRY_DIR', Path.home() / '.agents/lanes')) / (lane.name + '.json')
    if not registry.exists():
        return  # generic campaigns need not use a repository lane registry
    row = json.loads(registry.read_text())
    path = Path(row['path'])
    if row.get('released') or not path.is_dir():
        raise ValueError('stale lane registry: ' + str(path) + ' is released or missing')
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    def git(*args):
        result = subprocess.run(['git', '-C', str(path), *args], env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('invalid registered worktree: ' + str(path))
        return result.stdout.strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != path.resolve():
        raise ValueError('registered path is not a worktree root: ' + str(path))
    if git('branch', '--show-current') != row['branch']:
        raise ValueError('registered worktree branch differs: ' + str(path))


def main():
    if sys.argv[1] in ('--session-id', '--codex-session-id'):
        sid=last_session_id(sys.argv[2])
        if sid and (sys.argv[1] != '--codex-session-id' or local_codex_session(sid)):
            print(sid); return 0
        print('refused: no valid persisted session identity for '+sys.argv[2], file=sys.stderr)
        return 2
    lane=Path(sys.argv[1]).resolve(); lane.mkdir(parents=True,exist_ok=True)
    command=sys.argv[3:]
    try:
        require_lane_workspace(lane)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('refused: ' + str(error), file=sys.stderr); return 2
    with (lane/'session.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'refused: lane {lane.name} has a launch in progress');return 3
        try:
            pid=int((lane/'pid').read_text().strip())
            if pid>0:
                try: os.kill(pid,0)
                except ProcessLookupError: pass
                else:
                    print(f'refused: lane {lane.name} already has a live session (pid {pid})');return 3
        except (FileNotFoundError,ValueError): pass
        try: recent=time.time()-float((lane/'session-at').read_text())<120
        except (FileNotFoundError,ValueError): recent=False
        if recent:
            print(f'refused: lane {lane.name} launched in the last 120 s');return 3
        (lane/'session-at').write_text(str(int(time.time()))+'\n')
        return subprocess.run(command,env={**os.environ,'LANE_SESSION_GUARDED':str(lane)}).returncode


if __name__=='__main__': raise SystemExit(main())
