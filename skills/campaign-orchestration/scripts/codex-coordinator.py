#!/usr/bin/env python3
"""Stream one subscription Codex coordinator; record its actual thread ID.

CLI JSONL: https://learn.chatgpt.com/docs/non-interactive-mode
Authentication restriction: https://learn.chatgpt.com/docs/auth
The keeper owns scheduling/limits; this adapter never launches another turn.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def read(path):
    try: return Path(path).read_text().strip()
    except FileNotFoundError: return ''


def retired(keeper):
    return {line.split()[-1] for line in read(keeper/'retired').splitlines() if len(line.split()) == 3}


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text)
    temp.replace(path)


def register(args, sid):
    if not re.fullmatch(r'[0-9A-Za-z-]+', sid): raise ValueError('invalid Codex thread id')
    if sid in retired(args.keeper) or args.reservation in retired(args.keeper):
        raise ValueError('coordinator was retired before thread capture')
    if read(args.keeper/'coordinator.session') != args.reservation:
        raise ValueError('coordinator reservation has been replaced')
    atomic(args.session_file, sid+'\n')
    atomic(args.keeper/'coordinator.session', sid+'\n')
    atomic(args.registry/(sid+'.json'), json.dumps({'repo': str(args.repo.resolve()),
           'transcript': str(args.out.resolve()), 'engine': 'codex'}, sort_keys=True)+'\n')
    with (args.keeper/'keeper.log').open('a') as log:
        log.write(f'{dt.datetime.now().isoformat()} COORDINATOR-SESSION {sid}: log {args.out}; '
                  f'cd {args.repo} && codex exec resume {sid}\n')


def codex_command(args):
    command = ['codex', 'exec', '--json', '--dangerously-bypass-approvals-and-sandbox',
               '-c', 'forced_login_method="chatgpt"', '-c', 'model_provider="openai"', '-C', str(args.repo)]
    if args.model: command += ['--model', args.model]
    if args.effort: command += ['-c', 'model_reasoning_effort='+json.dumps(args.effort)]
    return [*command, args.prompt.read_text()]


def run(args):
    if read(args.keeper/'coordinator.session') != args.reservation or args.reservation in retired(args.keeper):
        raise ValueError('inactive coordinator reservation')
    env = {key: value for key, value in os.environ.items()
           if key not in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'ANTHROPIC_API_KEY', 'CLAUDE_CODE_SESSION_ID', 'CLAUDE_SESSION_ID')}
    process = subprocess.Popen(codex_command(args), cwd=args.repo, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding='utf-8', errors='replace', start_new_session=True)
    previous = {}
    def stop(signum, _frame):
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        previous[sig] = signal.signal(sig, stop)
    sid, failed, completed = None, False, False
    try:
        for line in process.stdout:
            try: row = json.loads(line)
            except ValueError:
                print(line, end='', flush=True); continue
            if not isinstance(row, dict):
                print(line, end='', flush=True); continue
            if row.get('type') == 'thread.started':
                actual = row.get('thread_id')
                if not isinstance(actual, str) or sid is not None:
                    raise ValueError('missing or repeated Codex thread id')
                register(args, actual); sid = actual
            if row.get('type') in ('turn.failed', 'error'): failed = True
            if row.get('type') == 'turn.completed': completed = True
            row['timestamp'] = dt.datetime.now(dt.timezone.utc).isoformat()
            print(json.dumps(row), flush=True)
        code = process.wait()
        if not sid or failed or not completed:
            print('Codex coordinator incomplete: missing session/completed turn or failure event', file=sys.stderr)
            return code or 2
        return code
    finally:
        # A killed stream adapter must not leave an untracked coordinator or its
        # command grandchildren alive. The child owns its whole process group.
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
        process.stdout.close()
        for sig, handler in previous.items(): signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'keeper', 'prompt', 'out', 'registry', 'session-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reservation', required=True)
    parser.add_argument('--model')
    parser.add_argument('--effort')
    args = parser.parse_args()
    try: return run(args)
    except (OSError, ValueError) as error:
        print(f'Codex coordinator refused: {error}', file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
