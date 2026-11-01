"""Bind driver markers to their lexical state directory before trusting them."""
import json
import os
from pathlib import Path
import subprocess

MARKERS = ('driver.pid', 'driver.cmd', 'DRIVER-EXIT', 'DRIVER-FAILED.md')


def spec(folder):
    folder = Path(folder).absolute()
    for name in MARKERS:
        path = folder / name
        if path.is_symlink() or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError('aliased driver marker: ' + str(path) + '; preserve and remove the alias before launch')
    row = json.loads((folder / 'driver.cmd').read_text())
    args = row['args']
    args = args[1:] if args and args[0] == '--once' else args
    if len(args) < 5 or args[3] != '--' or Path(args[0]).resolve() != folder.resolve():
        raise ValueError('driver.cmd state owner differs from ' + str(folder))
    if not Path(row['cwd']).is_absolute() or not Path(row['cwd']).is_dir():
        raise ValueError('driver.cmd cwd is missing: ' + str(row.get('cwd')))
    return row


def valid(folder):
    try:
        spec(folder)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def driver_dirs(lane):
    """Root and immediate children only; no recursive walk through caches."""
    lane = Path(lane)
    return [lane] + sorted(p for p in lane.iterdir() if p.is_dir() and not p.is_symlink()
                           and not any((p / name).is_symlink() for name in MARKERS)
                           and (valid(p) if (p / 'driver.cmd').exists() else (p / 'driver.pid').exists()))


def process_cwd(pid):
    try:
        if Path('/proc/self/cwd').exists():
            return os.readlink('/proc/%s/cwd' % pid)
        out = subprocess.run(['lsof', '-a', '-d', 'cwd', '-Fn', '-p', str(pid)], capture_output=True, text=True).stdout
        return next((line[1:] for line in out.splitlines() if line.startswith('n')), '')
    except OSError:
        return ''


def owns_process(folder, pid, command=None, cwd=None):
    """Destructive operations require positive state, argv AND cwd evidence."""
    try:
        row = spec(folder)
        if command is None:
            command = subprocess.run(['ps', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True).stdout
        if cwd is None:
            cwd = process_cwd(pid)
        args = row['args'][1:] if row['args'][0] == '--once' else row['args']
        # ps doesn't preserve shell quoting; compare the exact state argument between
        # the script and driver name, never a substring elsewhere in the payload.
        prefix = 'drive-loop.sh ' + ('--once ' if row['args'][0] == '--once' else '')
        tail = command.partition(prefix)[2]
        owner, separator, _ = tail.partition(' ' + args[1] + ' ' + str(args[2]) + ' -- ')
        return bool(separator and Path(owner).is_absolute() and Path(owner).resolve() == Path(folder).resolve()
                    and cwd and Path(cwd).resolve() == Path(row['cwd']).resolve())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(folder):
    """Refuse legacy aliases before the shell writes even its PID or exit status."""
    folder = Path(folder)
    for name in MARKERS:
        path = folder / name
        if path.is_symlink() or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError('aliased driver marker: ' + str(path) + '; preserve and remove the alias before launch')
    if (folder / 'driver.cmd').exists():
        spec(folder)  # a copied foreign owner is as unsafe as a symlink
    return 0


if __name__ == '__main__':
    import sys
    try:
        prepare(sys.argv[1])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('drive-loop refused: ' + str(error), file=sys.stderr)
        raise SystemExit(65)
