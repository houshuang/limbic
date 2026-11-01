#!/usr/bin/env python3
"""Remove abandoned scratch directories that killed jobs leave in the temp folder.

A job killed by SIGKILL (the keeper's reaper, sleepsafe on wake, the OOM path) never runs
its own cleanup, so every aborted export check leaves its scratch tree behind. On 1 Nov 2026
7,164 of them held 67 GB under $TMPDIR/skard-*. Measuring only worktrees missed them.

    tmp_sweep.py [--execute] [--min-age-minutes 60] GLOB [GLOB ...]

A directory is removed only when nothing under it is open by any process (one `lsof -Fn`
snapshot) and nothing in it was modified within the minimum age. Dry run by default.
"""
import argparse
import glob
import os
import shutil
import stat
import subprocess
import sys
import time


def open_paths():
    try:
        out = subprocess.run(['lsof', '-Fn'], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {line[1:] for line in out.splitlines() if line.startswith('n/')}


def newest_mtime(path, limit=20000):
    newest = os.lstat(path).st_mtime
    seen = 0
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            try:
                newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
            seen += 1
            if seen >= limit:
                return newest
    return newest


def size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_blocks * 512
            except OSError:
                pass
    return total


def _writable(func, path, _exc):
    parent = os.path.dirname(path)
    for p in (parent, path):
        try:
            os.chmod(p, os.lstat(p).st_mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        except OSError:
            pass
    func(path)


def candidates(patterns, opened, min_age, now):
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.expandvars(os.path.expanduser(pattern)))):
            if not os.path.isdir(path) or os.path.islink(path):
                continue
            prefix = path.rstrip('/') + '/'
            if any(p == path or p.startswith(prefix) for p in opened):
                yield path, 'open'
                continue
            if now - newest_mtime(path) < min_age * 60:
                yield path, 'recent'
                continue
            yield path, 'stale'


def sweep(patterns, min_age=60, execute=False, now=None, opened=None):
    now = time.time() if now is None else now
    if opened is None:
        opened = open_paths()
        if opened is None:
            return {'error': 'lsof failed; nothing removed', 'removed': 0, 'bytes': 0}
    removed, freed, kept = 0, 0, {'open': 0, 'recent': 0}
    for path, why in candidates(patterns, opened, min_age, now):
        if why != 'stale':
            kept[why] += 1
            continue
        b = size_bytes(path)
        if execute:
            shutil.rmtree(path, onerror=_writable)
            if os.path.exists(path):
                continue
        removed += 1
        freed += b
    return {'removed': removed, 'bytes': freed, 'kept': kept, 'execute': execute}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('patterns', nargs='+')
    ap.add_argument('--min-age-minutes', type=float, default=60)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args(argv)
    r = sweep(a.patterns, a.min_age_minutes, a.execute)
    if 'error' in r:
        print(r['error'])
        return 1
    verb = 'removed' if a.execute else 'would remove'
    print(f"tmp_sweep: {verb} {r['removed']} dirs, {r['bytes'] / 2**30:.1f} GiB; "
          f"kept {r['kept']['open']} open, {r['kept']['recent']} recent")
    return 0


if __name__ == '__main__':
    sys.exit(main())
