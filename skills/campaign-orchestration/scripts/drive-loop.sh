#!/bin/bash
# Usage: drive-loop.sh [--once] <state-dir> <name> <limit-seconds> -- <driver command…>
# Runs a cache-resuming driver under sleepsafe until it exits 0 or a stop file appears.
#  - exit 124/125 (sleepsafe awake-time limit / wake) restarts at once: a time limit is not an outcome
#  - any other non-zero exit counts; three in a row writes DRIVER-FAILED.md (log tail) and stops,
#    so lane-audit.sh and the watcher see a failure instead of a quiet lane
#  - --once: the first non-zero exit other than 124/125 ends the loop (a release cycle must not rerun itself)
# Writes its own pid to <state-dir>/driver.pid and its arguments to driver.cmd; on exit it writes DRIVER-EXIT
# (the exit code, or «term»). lane-keeper.py restarts a drive-loop that was killed (no DRIVER-EXIT, or «term»),
# so stop one deliberately with <state-dir>/STOP, never with kill alone.
# Budget: a repeating loop waits (does not stop) while ~/.agents/budget/HOLD-<engine> exists (written by lane-keeper
# from budget.py); <state-dir>/engine names the engine (default codex; «none» for drivers that call no model).
# --once jobs (release cycles, gates) never wait.
# sleepsafe.py is $SLEEPSAFE, else ~/.agents/bin/sleepsafe.py, else the one on PATH. Runs under bash 3.2 (macOS) and Linux.
scripts=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().parent)' "${BASH_SOURCE[0]}")
once=0; [[ $1 == --once ]] && { once=1; shift; }
S=$(cd "$1" && pwd -P) || exit 2; name=$2; limit=$3; shift 3; [[ $1 == -- ]] && shift
SLEEPSAFE=${SLEEPSAFE:-$HOME/.agents/bin/sleepsafe.py}
[ -x "$SLEEPSAFE" ] || SLEEPSAFE=$(command -v sleepsafe.py) || { echo "drive-loop: no sleepsafe.py (set SLEEPSAFE)"; exit 2; }
python3 "$scripts/driver_state.py" "$S" || exit $?
echo $$ > "$S/driver.pid"
python3 -c 'import json, os, sys; print(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}))' \
  $([[ $once == 1 ]] && echo --once) $S $name $limit -- "$@" > $S/driver.cmd
rm -f $S/DRIVER-FAILED.md $S/DRIVER-EXIT
trap 'echo term > $S/DRIVER-EXIT; exit 143' TERM INT HUP
finish() { echo $1 > $S/DRIVER-EXIT; exit $1; }
failures=0
while true; do
  [[ -e $S/DONE.md && ! -e $S/STOP ]] && { python3 "$scripts/deliverables.py" check "$S" || finish $?; }
  [[ -e $S/STOP || -e $S/DONE.md ]] && { echo "$(date -u +%FT%TZ) drive-loop: stop file"; finish 0; }
  if (( ! once )); then
    engine=$(cat $S/engine 2>/dev/null || echo codex); hold=${AGENTS_BUDGET_DIR:-$HOME/.agents/budget}/HOLD-$engine
    if [[ $engine != none && -e $hold ]]; then
      echo "$(date -u +%FT%TZ) drive-loop: budget hold ($(cat $hold 2>/dev/null)); waiting"
      while [[ -e $hold && ! -e $S/STOP && ! -e $S/DONE.md ]]; do sleep ${BUDGET_POLL:-300}; done
      continue
    fi
  fi
  "$SLEEPSAFE" --name $name --limit $limit -- "$@"
  code=$?
  echo "$(date -u +%FT%TZ) drive-loop: exit $code"
  if (( code == 0 )); then
    python3 "$scripts/deliverables.py" check "$S" || finish $?
  fi
  (( code == 0 )) && { echo "$(date -u +%FT%TZ) drive-loop: driver finished"; finish 0; }
  if (( code == 124 || code == 125 )); then failures=0; else failures=$((failures + 1)); fi
  if (( failures >= 3 || (once && failures >= 1) )); then
    { echo "# $name: $failures failed run(s) in a row ($(date -u +%FT%TZ))"; echo; echo '```'; tail -40 $S/driver.log 2>/dev/null; echo '```'; } > $S/DRIVER-FAILED.md
    finish $code
  fi
  sleep 5
done
