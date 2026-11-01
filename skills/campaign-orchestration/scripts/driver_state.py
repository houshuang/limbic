"""Bind driver markers to their lexical state directory before trusting them."""
import json
import os
import re
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
        pid = int(str(pid).strip())
        row = spec(folder)
        if command is None:
            command = subprocess.run(['ps', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True).stdout
        if cwd is None:
            cwd = process_cwd(pid)
        # A driver command quoted inside a model prompt is not the executable.
        invocation = re.match(r'^(?:(?:\S*/)?(?:bash|sh|zsh)\s+)?(?:\S*/)?drive-loop\.sh\s+', command.strip())
        if not invocation:
            return False
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
        if len(sys.argv) > 2 and sys.argv[2] == '--verify-claim':
            import fcntl
            fd = int(os.environ['DRIVE_LOOP_CLAIM_FD'])
            path = Path(sys.argv[1]) / '.driver-owner.lock'
            a, b = os.fstat(fd), path.lstat()
            if path.is_symlink() or a.st_nlink != 1 or (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise ValueError('invalid driver lock')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(0)
        prepare(sys.argv[1])
        if len(sys.argv) > 2:
            import fcntl
            lock = os.open(str(Path(sys.argv[1]) / '.driver-owner.lock'), os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            if os.fstat(lock).st_nlink != 1:
                raise ValueError('aliased driver lock')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('another drive-loop owns this state directory')
            pidfile = Path(sys.argv[1]) / 'driver.pid'
            if pidfile.exists() and owns_process(sys.argv[1], pidfile.read_text()):
                raise ValueError('a live drive-loop owns this state directory')
            os.set_inheritable(lock, True)
            os.environ['DRIVE_LOOP_CLAIM_FD'] = str(lock)
            os.execv('/bin/bash', ['/bin/bash', *sys.argv[2:]])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('drive-loop refused: ' + str(error), file=sys.stderr)
        raise SystemExit(65)
