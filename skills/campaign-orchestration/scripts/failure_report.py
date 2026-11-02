"""Capture one execution and render actionable failures; inspect never writes state."""
import json
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

AUTH = re.compile(r'token_revoked|refresh_token_invalidated|invalid_api_key|authentication[_ ](?:error|failed)|HTTP(?:/\S+)?\s+401|401\s+Unauthorized', re.I)


def auth_failure(text):
    # A worker discussing an old incident is not a provider failure. Require
    # CLI/provider error context as well as an authentication status.
    return any(AUTH.search(line) and re.search(
        r'PacketFailure|CodexCLIError|\bERROR\b|^Error:|^error:|"type"\s*:\s*"error"', line)
        for line in text.splitlines())


def tail_lines(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            offset = stream.tell(); chunks = []; newlines = 0
            while offset and newlines <= 20:
                size = min(offset, 65536); offset -= size; stream.seek(offset)
                chunk = stream.read(size); chunks.append(chunk); newlines += chunk.count(b'\n')
            return b''.join(reversed(chunks)).decode(errors='replace').splitlines()[-20:]
    except OSError:
        return []


def tail(path):
    return '\n'.join(tail_lines(path))


def archive(lane, names=('DRIVER-FAILED.md',)):
    lane = Path(lane)
    for name in names:
        path = lane / name
        if path.exists():
            dest = lane / 'failure-history'
            dest.mkdir(exist_ok=True)
            path.rename(dest / (str(time.time_ns()) + '-' + uuid.uuid4().hex + '-' + name))


def render(lane, argv, code, log=None, reason='', since=None):
    lane = Path(lane)
    sections = []
    # Structured command results are stronger evidence than guessing from a
    # filename. Never follow a result's log outside this lane.
    result = lane / 'RESULTS.json'
    try:
        if since is None or result.stat().st_mtime >= since:
            row = json.loads(result.read_text())
            for check in row.get('checks', []):
                command = check.get('command')
                path = Path(check.get('log', ''))
                if not path.is_absolute():
                    path = lane / path
                if (check.get('returncode') and isinstance(command, list)
                        and all(isinstance(a, str) for a in command)
                        and lane.resolve() in path.resolve().parents):
                    sections.append(('Failing check', command, check['returncode'], path))
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    sections.insert(0, ('Driver', argv, code, Path(log) if log else lane / 'driver.log'))
    # Legacy drivers redirected stdout to launch.log or check-N.log. Include
    # labelled nonempty evidence instead of silently reporting an empty tail.
    represented = {str(p.resolve()) for _, _, _, p in sections}
    extra = sorted(lane.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in extra:
        if (str(path.resolve()) not in represented and path.name != 'codex.log'
                and (since is None or path.stat().st_mtime >= since) and tail(path)):
            sections.append(('Additional log (command not recorded)', None, None, path))
            if len(sections) >= 5:
                break
    auth = any(auth_failure(tail(path)) for _, _, _, path in sections)
    text = '# Driver failure\n\nExit code: ' + str(code) + '\n'
    try:
        text += '\nRecorded driver.pid: ' + (lane / 'driver.pid').read_text().strip() + '\n'
    except OSError:
        pass
    if auth or code == 77:
        text += '\nProvider status: AUTHENTICATION-FAILED; process liveness/exit zero does not prove extraction.\n'
    if reason:
        text += '\n' + reason + '\n'
    for label, command, status, path in sections:
        text += '\n## ' + label + '\n'
        if command is not None:
            text += '\nargv: ' + json.dumps(command, ensure_ascii=False) + '\n\nCommand: ' + shlex.join(command) + '\n\nExit code: ' + str(status) + '\n'
        text += '\nLog: ' + str(path) + '\n\nLast 20 lines:\n```text\n' + (tail(path) or '(no output captured; see exact argv and exit code above)') + '\n```\n'
    return text


def write(lane, argv, code, log=None, reason='', since=None):
    message = render(lane, argv, code, log, reason, since)
    archive(lane)
    (Path(lane) / 'DRIVER-FAILED.md').write_text(message)
    return message


def run(lane, argv, *, publish=True, inherited=False):
    """Tee to a unique attempt log even when the caller redirects elsewhere."""
    lane = Path(lane)
    started = time.time()
    logs = lane / 'driver-attempts'
    logs.mkdir(exist_ok=True)
    log = logs / (str(time.time_ns()) + '-' + uuid.uuid4().hex + '.log')
    auth = False
    pending = b''
    session_log = lane / 'codex.log'
    offset = session_log.stat().st_size if inherited and session_log.exists() else 0
    with log.open('wb') as stream:
        def record(chunk):
            nonlocal auth, pending
            stream.write(chunk); stream.flush()
            pending = pending + chunk
            auth = auth or auth_failure(pending.decode(errors='replace'))
            pending = pending[-65536:]

        try:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=None if inherited else subprocess.PIPE, stderr=subprocess.STDOUT)
            previous = {}
            def forward(signum, frame):
                try:
                    process.send_signal(signum)
                except ProcessLookupError:
                    pass
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous[signum] = signal.signal(signum, forward)
            if not inherited:
                while True:
                    chunk = process.stdout.read1(65536)
                    if not chunk:
                        break
                    record(chunk)
                    sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
                process.stdout.close()
            code = process.wait()
            if inherited and session_log.exists():
                # Keep the CLI's original stdout fd semantics: a detached child
                # inheriting it must not keep the session wrapper waiting on EOF.
                with session_log.open('rb') as source:
                    source.seek(offset)
                    while True:
                        chunk = source.read(65536)
                        if not chunk:
                            break
                        record(chunk)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            code = 128 - code if code < 0 else code
        except OSError as error:
            stream.write((str(error) + '\n').encode()); stream.flush()
            code = 127
    process_code = code
    if auth and code == 0:
        code = 77
    if code:
        reported = argv if publish else argv[argv.index('--')+1:]
        message = render(lane, reported, code, log, since=started,
                         reason='Process exit code: ' + str(process_code) +
                         ('\n\nProvider status: AUTHENTICATION-FAILED (observed in this attempt).' if auth else ''))
        log.with_suffix('.md').write_text(message)
        if publish:
            archive(lane)
            (lane / 'DRIVER-FAILED.md').write_text(message)
    return code


def main():
    mode, lane = sys.argv[1:3]
    if mode in ('run', 'attempt'):
        return run(lane, sys.argv[4:], publish=mode == 'run')
    if mode == 'promote':
        reports = list((Path(lane) / 'driver-attempts').glob('*.md'))
        archive(lane)
        (Path(lane) / 'DRIVER-FAILED.md').write_bytes(max(reports, key=lambda p: p.stat().st_mtime_ns).read_bytes())
        return 0
    if mode == 'inspect':
        row = json.loads((Path(lane) / 'driver.cmd').read_text())
        args = row['args']; argv = args[args.index('--')+1:]
        print(render(lane, argv, int(sys.argv[3])))
        return 0
    raise ValueError('unknown mode: ' + mode)


if __name__ == '__main__':
    raise SystemExit(main())
