#!/bin/zsh
# Usage: drive-loop.sh [--once] <state-dir> <name> <limit-seconds> -- <driver command…>
# Runs a cache-resuming driver under sleepsafe until it exits 0 or a stop file appears.
#  - exit 124/125 (sleepsafe awake-time limit / wake) restarts at once: a time limit is not an outcome
#  - any other non-zero exit counts; three in a row writes DRIVER-FAILED.md (log tail) and stops,
#    so lane-audit.sh and the watcher see a failure instead of a quiet lane
#  - --once: the first non-zero exit other than 124/125 ends the loop (a release cycle must not rerun itself)
# Writes its own pid to <state-dir>/driver.pid and its arguments to driver.cmd; on exit it writes DRIVER-EXIT
# (the exit code, or «term»). lane-keeper.py restarts a drive-loop that was killed (no DRIVER-EXIT, or «term»),
# so stop one deliberately with <state-dir>/STOP, never with kill alone.
once=0; [[ $1 == --once ]] && { once=1; shift; }
S=${1:A}; name=$2; limit=$3; shift 3; [[ $1 == -- ]] && shift
echo $$ > $S/driver.pid
python3 -c 'import json, os, sys; print(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}))' \
  $([[ $once == 1 ]] && echo --once) $S $name $limit -- "$@" > $S/driver.cmd
rm -f $S/DRIVER-FAILED.md $S/DRIVER-EXIT
trap 'echo term > $S/DRIVER-EXIT; exit 143' TERM INT HUP
finish() { echo $1 > $S/DRIVER-EXIT; exit $1; }
failures=0
while true; do
  [[ -e $S/STOP || -e $S/DONE.md ]] && { echo "$(date -u +%FT%TZ) drive-loop: stop file"; finish 0; }
  ~/.agents/bin/sleepsafe.py --name $name --limit $limit -- "$@"
  code=$?
  echo "$(date -u +%FT%TZ) drive-loop: exit $code"
  (( code == 0 )) && { echo "$(date -u +%FT%TZ) drive-loop: driver finished"; finish 0; }
  if (( code == 124 || code == 125 )); then failures=0; else (( failures++ )); fi
  if (( failures >= 3 || (once && failures >= 1) )); then
    { echo "# $name: $failures failed run(s) in a row ($(date -u +%FT%TZ))"; echo; echo '```'; tail -40 $S/driver.log 2>/dev/null; echo '```'; } > $S/DRIVER-FAILED.md
    finish $code
  fi
  sleep 5
done
