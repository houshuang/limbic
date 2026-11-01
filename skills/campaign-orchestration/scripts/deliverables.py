#!/usr/bin/env python3
"""Optional EXPECTS JSON: files, nonempty parseable jsonl, and a branch commit since base.

Paths are relative to the lane state directory; commit.repo is relative to it too.
Used by drive-loop and detached launch/resume, without changing lanes that have no contract.
"""
import os
import time
import json
import signal
from pathlib import Path
import subprocess
import sys
from driver_state import driver_dirs, owns_process

LANE_REGISTRY = Path.home() / '.agents/lanes'


def commit_repository(repo, branch):
    """A removed checkout may only resolve through its completed integration.

    A missing worktree is otherwise a refusal, never an invitation to recreate
    it or to accept a same-named branch in an arbitrary repository.
    """
    if repo.is_dir():
        return repo
    for path in LANE_REGISTRY.glob('*.json'):
        row = json.loads(path.read_text())
        if Path(row.get('path', '')).resolve() != repo.resolve() or row.get('branch') != branch:
            continue
        if not (row.get('released') and row.get('removed_utc') and row.get('integrated_commit')):
            break
        shared = Path(row['shared_root'])
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        def ancestor(a, b):
            return subprocess.run(['git', '-C', str(shared), 'merge-base', '--is-ancestor', a, b],
                                  env=env, capture_output=True).returncode == 0
        if shared.is_dir() and ancestor(branch, row['integrated_commit']) and ancestor(row['integrated_commit'], 'HEAD'):
            return shared
        break
    return repo


def live_driver(lane):
    for folder in driver_dirs(lane):
        try:
            path = folder / 'driver.pid'
            pid = int(path.read_text())
            if pid <= 0 or not owns_process(folder, pid):
                continue
            os.kill(pid, 0)
            result = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True)
            started = time.mktime(time.strptime(' '.join(result.stdout.split()), '%a %b %d %H:%M:%S %Y'))
            if started <= path.stat().st_mtime + 5:
                return True
        except (OSError, ValueError):
            continue
    return False


def missing(lane):
    lane = Path(lane)
    contract = lane / 'EXPECTS'
    if not contract.exists():
        return []
    try:
        spec = json.loads(contract.read_text())
        if not isinstance(spec, dict) or set(spec) - {'files', 'jsonl', 'commit'} or not spec:
            raise ValueError('expected files, jsonl or commit')
        errors = []
        for kind in ('files', 'jsonl'):
            paths = spec.get(kind, [])
            if not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths):
                raise ValueError(kind + ' must be a list of paths')
            for name in paths:
                path = lane / name
                if not path.is_file():
                    errors.append('missing file: ' + name)
                elif kind == 'jsonl':
                    try:
                        count = 0
                        with path.open() as stream:
                            for line in stream:
                                if line.strip():
                                    json.loads(line); count += 1
                        if not count:
                            errors.append('empty JSONL: ' + name)
                    except (OSError, ValueError) as error:
                        errors.append('invalid JSONL: ' + name + ': ' + str(error))
        if 'commit' in spec:
            commit = spec['commit']
            repo, base, branch = lane / commit['repo'], commit['base'], commit['branch']
            repo = commit_repository(repo, branch)
            # An explicit base is required: mere branch existence is not a deliverable.
            def git(*args):
                return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True,
                                      env={k: v for k, v in os.environ.items() if not k.startswith('GIT_')})
            ancestor = git('merge-base', '--is-ancestor', base, branch)
            count = git('rev-list', '--count', base + '..' + branch)
            if ancestor.returncode or count.returncode or not int(count.stdout.strip()):
                errors.append('missing commit: ' + branch + ' since ' + base + ' in ' + str(repo))
        return errors
    except (OSError, ValueError, TypeError, KeyError) as error:
        return ['invalid EXPECTS: ' + str(error)]


def check(lane):
    errors = missing(lane)
    if errors:
        message = '# Deliverable contract failed\n\n' + '\n'.join('- ' + e for e in errors) + '\n'
        (Path(lane) / 'DRIVER-FAILED.md').write_text(message)
        print(message, file=sys.stderr)
        return 65
    return 0


def main():
    mode, lane = sys.argv[1:3]
    if mode == 'check':
        return check(lane)
    process = subprocess.Popen(sys.argv[4:])
    def forward(signum, frame):
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    code = process.wait()
    if code < 0:
        code = 128 - code
    if code == 0 and not live_driver(lane):
        code = check(lane)
    (Path(lane) / 'SESSION-EXIT').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
