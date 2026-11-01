#!/usr/bin/env python3
"""Run a frozen list of independent checkpoint commands, parking each failure.

The caller owns the integration lease and supplies the repository's heavy wrapper.
Each item is {id, cwd, argv}; use already-authorized, isolated checkpoint commands.
An interrupted item is parked for manual inspection, never replayed automatically.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def require_lease(path, owner):
    lease = json.loads(Path(path).read_text())
    if lease.get('owner') != owner or float(lease['until']) <= time.time():
        raise ValueError('integration lease is expired or held by another owner')


def guarded(lease, owner, command):
    # Invoked by heavy.sh *after* admission; refuse if its wait expired the lease.
    try:
        require_lease(lease, owner)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print('lease check failed after heavy admission: ' + str(error), file=sys.stderr)
        return 75
    os.execvp(command[0], command)


def run(manifest, state, heavy, lease_file, owner):
    raw = Path(manifest).read_bytes()
    entries = json.loads(raw)
    if not isinstance(entries, list) or not entries:
        raise ValueError('manifest must contain checkpoint items')
    ids, workspaces = set(), set()
    for item in entries:
        if not isinstance(item, dict) or set(item) != {'id', 'cwd', 'argv'}:
            raise ValueError('each checkpoint requires id, cwd and argv')
        ident, cwd, argv = item['id'], item['cwd'], item['argv']
        if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', ident) or ident in ids:
            raise ValueError('invalid or duplicate checkpoint id')
        if not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise ValueError('checkpoint cwd must be an existing absolute directory')
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
            raise ValueError('checkpoint argv must be a nonempty string array')
        workspace = Path(cwd).resolve()
        root = subprocess.run(['git', '-C', str(workspace), 'rev-parse', '--show-toplevel'],
                              capture_output=True, text=True)
        workspace = Path(root.stdout.strip()).resolve() if root.returncode == 0 else workspace
        if any(workspace == other or workspace in other.parents or other in workspace.parents for other in workspaces):
            raise ValueError('each checkpoint requires a distinct, nonnested workspace')
        ids.add(ident); workspaces.add(workspace)
    state = Path(state); state.mkdir(parents=True, exist_ok=True)
    with (state / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results_path = state / 'results.json'
        digest = hashlib.sha256(raw).hexdigest()
        results = json.loads(results_path.read_text()) if results_path.exists() else {'manifest_sha256': digest, 'items': {}}
        if results.get('manifest_sha256') != digest:
            raise ValueError('checkpoint manifest changed; preserve results and create a new queue')
        outcomes = results['items']
        for item in entries:
            ident = item['id']
            if ident in outcomes:
                if outcomes[ident]['status'] == 'running':
                    outcomes[ident] = {'status': 'parked', 'reason': 'interrupted; inspect checkpoint before any retry'}
                    write(results_path, results)
                continue
            try:
                require_lease(lease_file, owner)
            except (OSError, KeyError, TypeError, ValueError) as error:
                outcomes[ident] = {'status': 'parked', 'reason': 'lease check failed: ' + str(error)}
                write(results_path, results)
                break
            outcomes[ident] = {'status': 'running'}; write(results_path, results)
            log = state / (ident + '.log')
            try:
                with log.open('ab') as out:
                    result = subprocess.run([str(heavy), sys.executable, str(Path(__file__).resolve()), '--guarded', str(lease_file), owner, *item['argv']], cwd=item['cwd'], stdin=subprocess.DEVNULL,
                                            stdout=out, stderr=subprocess.STDOUT,
                                            env={**os.environ, 'HEAVY_EXCLUSIVE': '1'}, pass_fds=(lock.fileno(),))
                outcomes[ident] = {'status': 'completed' if result.returncode == 0 else 'parked',
                                   'exit': result.returncode, 'log': str(log)}
            except OSError as error:
                outcomes[ident] = {'status': 'parked', 'reason': str(error), 'log': str(log)}
            write(results_path, results)
        return 1 if any(row['status'] != 'completed' for row in outcomes.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest'); parser.add_argument('state'); parser.add_argument('--heavy', required=True)
    parser.add_argument('--lease', required=True); parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    return run(args.manifest, args.state, args.heavy, args.lease, args.owner)


if __name__ == '__main__':
    if len(sys.argv) > 4 and sys.argv[1] == '--guarded':
        raise SystemExit(guarded(sys.argv[2], sys.argv[3], sys.argv[4:]))
    raise SystemExit(main())
