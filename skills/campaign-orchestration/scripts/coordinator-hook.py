#!/usr/bin/env python3
"""Claude Code hooks for campaign coordinators; a no-op in every other session.

Only a session owning a live lane-keeper watch heartbeat, or the keeper-started headless
session, may register. Command text (including launch commands and handover heredocs)
is never evidence of coordinator ownership. Retired IDs may not re-register.

Install this version alongside the keeper after integration:
    cp skills/campaign-orchestration/scripts/coordinator-hook.py ~/.agents/bin/coordinator-hook.py

    register  (PostToolUse, Bash): register an actual coordinator; check context size
    start     (SessionStart): inject lane-audit output into a registered session (startup/resume/compact)
    stop      (Stop): block ending a turn while lanes run and no lane-audit --loop heartbeat is fresh
"""
import json, os, subprocess, sys, time
from pathlib import Path

REG = Path.home() / '.agents/coordinators'
AUDIT = Path.home() / 'src/limbic/skills/campaign-orchestration/scripts/lane-audit.sh'
KEEPER = Path.home() / 'src/limbic/skills/campaign-orchestration/scripts/lane-keeper.py'
WARN, BLOCK = 200_000, 300_000


def config_for(cwd):
    for d in [Path(cwd), *Path(cwd).parents]:
        f = d / '.coordinator.json'
        if f.exists():
            return json.loads(f.read_text())
    return None


def registered(sid):
    f = REG / f'{sid}.json'
    return json.loads(f.read_text()) if f.exists() else None


def context_tokens(transcript):
    try:
        with open(transcript, 'rb') as fh:
            fh.seek(0, 2); fh.seek(max(0, fh.tell() - 400_000))
            lines = fh.read().decode('utf-8', 'ignore').splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        if '"usage"' not in line:
            continue
        try:
            u = json.loads(line).get('message', {}).get('usage') or {}
        except ValueError:
            continue
        if u:
            return u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0)
    return 0


def live_lanes(cfg):
    out = []
    for c in cfg.get('campaigns', []):
        for d in Path(c).iterdir() if Path(c).is_dir() else []:
            for name in ('pid', 'driver.pid'):
                p = d / name
                try:
                    os.kill(int(p.read_text().strip()), 0); out.append(d.name); break
                except (OSError, ValueError):
                    pass
    return out


def keeper_dir(cfg):
    # Registrations made before the config named a state dir lack it, so read the live config.
    live = config_for(cfg.get('repo', '.')) or {}
    return Path(live.get('state') or cfg.get('state') or '/nonexistent') / 'keeper'


def retired(sid, cfg):
    try:
        return any(len(parts := line.split()) == 3 and parts[-1] == sid
                   for line in (keeper_dir(cfg) / 'retired').read_text().splitlines())
    except OSError:
        return False


def owns_coordinator(sid, cfg):
    kd = keeper_dir(cfg)
    try:
        beat = kd / 'watch.heartbeat'
        owner, pid = beat.read_text().split()
        if owner == sid and time.time() - beat.stat().st_mtime < 180:
            os.kill(int(pid), 0)
            return True
    except (OSError, ValueError):
        pass
    try:
        if (kd / 'coordinator.session').read_text().strip() == sid:
            os.kill(int((kd / 'coordinator.pid').read_text()), 0)
            return True
    except (OSError, ValueError):
        pass
    return False


def heartbeat_fresh(cfg):
    """A watching monitor, or the launchd lane-keeper, which starts a successor coordinator when lanes need one."""
    beats = [Path(c) / '.audit-heartbeat' for c in cfg.get('campaigns', [])]
    fresh = lambda b, age: b.exists() and time.time() - b.stat().st_mtime < age
    return any(fresh(b, 180) for b in beats) or fresh(keeper_dir(cfg) / 'heartbeat', 300)


def main():
    mode = sys.argv[1]
    h = json.load(sys.stdin)
    sid = h.get('session_id', '')
    if h.get('agent_id') or '/.claude/worktrees/' in h.get('cwd', ''):
        return  # subagents share the coordinator's session id; their context and lanes are their own
    reg = registered(sid)
    cfg = reg or config_for(h.get('cwd', '.'))
    if cfg and retired(sid, cfg):
        (REG / f'{sid}.json').unlink(missing_ok=True)
        return
    if mode == 'register':
        if not reg:
            if cfg and sid and owns_coordinator(sid, cfg):
                REG.mkdir(parents=True, exist_ok=True)
                (REG / f'{sid}.json').write_text(json.dumps({**cfg, 'since': time.strftime('%FT%T'), 'transcript': h.get('transcript_path', '')}))
            return
        if h.get('transcript_path') and reg.get('transcript') != h['transcript_path']:
            (REG / f'{sid}.json').write_text(json.dumps({**reg, 'transcript': h['transcript_path']}))
        tokens = context_tokens(h.get('transcript_path', ''))
        mark = REG / f'{sid}.warned'
        if tokens >= BLOCK:
            print(json.dumps({'decision': 'block', 'reason': f'Coordinator context is {tokens//1000}k tokens (limit {BLOCK//1000}k). Write the handover now (state, live lanes, and for each next step the lane that owns it or the file it waits on; normally spaced sentences, at most 2,800 characters) to {reg.get("state", "the state dir")}/<session>-HANDOVER-<YYYYMMDD-HHMM>.md and stop. lane-keeper starts the successor from the newest *HANDOVER*.md there as soon as lanes raise events and no coordinator is live; leave the keeper armed and do not stop lanes or drivers.'}))
        elif tokens >= WARN and not mark.exists():
            mark.touch()
            print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PostToolUse', 'additionalContext': f'Coordinator context is {tokens//1000}k tokens. Plan the handover: finish the current step, then write it before {BLOCK//1000}k.'}}))
    elif mode == 'start' and reg:
        try:
            out = subprocess.run(['zsh', str(AUDIT), reg['repo'], *reg.get('campaigns', [])], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:
            out = f'lane-audit failed: {e}'
        note = ''
        try:
            kd = keeper_dir(reg)
            os.kill(int((kd / 'coordinator.pid').read_text()), 0)
            note = f'\nA keeper-started coordinator is running (pid {(kd / "coordinator.pid").read_text().strip()}, logs in {kd}); do not integrate in parallel with it.'
        except (OSError, ValueError):
            pass
        print(json.dumps({'hookSpecificOutput': {'hookEventName': 'SessionStart', 'additionalContext': 'Campaign coordinator session. lane-audit now:\n' + (out or 'all lanes healthy') + note + f'\nTo be woken on lane events while idle, arm `python3 {KEEPER} watch` under a Monitor; otherwise lane-keeper starts a headless successor 20 minutes after this session goes quiet.'}}))
    elif mode == 'stop' and reg and not h.get('stop_hook_active'):
        lanes = live_lanes(reg)
        if lanes and not heartbeat_fresh(reg):
            print(json.dumps({'decision': 'block', 'reason': f'{len(lanes)} lanes are running ({", ".join(lanes[:6])}) and neither lane-keeper nor a watching monitor is alive. Arm the monitor (Monitor tool: python3 {KEEPER} watch) or load the keeper (launchctl load ~/Library/LaunchAgents/com.stian.lane-keeper.plist) before ending the turn.'}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        pass
