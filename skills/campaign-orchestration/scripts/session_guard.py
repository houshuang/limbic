#!/usr/bin/env python3
"""Serialize launch/resume admission before either script changes lane files."""
import fcntl
import os
import re
from pathlib import Path
import subprocess
import sys
import time


def tail_text(path, limit=4000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell()-limit))
            return stream.read(limit).decode(errors='replace')
    except OSError: return ''


def last_session_id(path):
    """Search append-only logs backwards with bounded memory, including legacy logs."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0,2); offset=stream.tell(); overlap=b''
            while offset:
                size=min(offset,65536);offset-=size;stream.seek(offset)
                block=stream.read(size)+overlap
                ids=re.findall(rb'session id: (\S+)',block,re.I)
                if ids: return ids[-1].decode()
                overlap=block[:256]
    except OSError: pass
    return None


def main():
    if sys.argv[1]=='--session-id':
        sid=last_session_id(sys.argv[2])
        if sid: print(sid); return 0
        return 2
    lane=Path(sys.argv[1]).resolve(); lane.mkdir(parents=True,exist_ok=True)
    command=sys.argv[3:]
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
