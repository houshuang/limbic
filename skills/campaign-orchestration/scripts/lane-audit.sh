#!/bin/zsh
# Usage: lane-audit.sh <repo> <campaign-dir>… [--loop SECONDS]
# Prints one line per lane that needs the coordinator; prints nothing when all is well.
# With --loop it repeats and prints only changes, so it can run as a Monitor command.
#   DRIVER-DEAD    driver.pid not running, no DONE.md (includes DRIVER-FAILED.md when present)
#   DRIVER-QUIET   driver running but driver.log/PROGRESS.md untouched for 20 min
#   SESSION-ENDED  worker session gone, no driver, no DONE.md
#   ORPHAN-BRANCH  worktree branch has commits not on main, last commit > 6 h ago, no READY/DONE
#   UNWATCHED      live lane that no running watch.sh names, while no lane-keeper heartbeat is fresh
# A lane is closed (skipped) only by an exact DONE.md, READY-<n>.md, DRIVER-FINISHED.json or RETIRED (one line: why),
# and only while nothing in it runs. DONE.partial*.md and similar never close a lane.
setopt null_glob
repo=$1; shift; dirs=(); loop=0
while (( $# )); do [[ $1 == --loop ]] && { loop=$2; shift 2; continue; }; dirs+=$1; shift; done
# A pid file is written after its process starts. A process that started later holds a recycled pid.
alive() {
  [[ -f $1 ]] || return 1
  local p=$(cat $1); kill -0 $p 2>/dev/null || return 1
  local started=$(date -j -f '%a %b %e %T %Y' "$(ps -o lstart= -p $p | sed 's/  */ /g')" +%s 2>/dev/null)
  (( ${started:-0} <= $(stat -f %m $1) + 5 ))
}
age() { local n=$(stat -f %m "$@" 2>/dev/null | sort -n | tail -1); echo $(( $(date +%s) - ${n:-0} )); }
audit() {
  local watched=$(ps -axo command | grep '[w]atch.sh')
  local kept=0; for k in $repo/../${repo:t}-state/keeper/heartbeat(N); do (( $(age $k) < 300 )) && kept=1; done
  for C in $dirs; do
    for d in $C/*(/); do
      local l=${d:t}
      [[ -f $d/pid || -f $d/driver.pid ]] || continue
      local closed=0; [[ -f $d/DONE.md || -f $d/DRIVER-FINISHED.json || -f $d/RETIRED ]] && closed=1; for f in $d/READY-<->.md; do closed=1; done
      (( closed )) && ! alive $d/pid && ! alive $d/driver.pid && continue
      if [[ -f $d/driver.pid ]]; then
        if alive $d/driver.pid; then
          (( $(age $d/driver.log $d/PROGRESS.md) > 1200 )) && echo "DRIVER-QUIET $l $(age $d/driver.log $d/PROGRESS.md)s"
        elif ! alive $d/pid; then
          echo "DRIVER-DEAD $l$([[ -f $d/DRIVER-FAILED.md ]] && echo ' (DRIVER-FAILED.md)'):$(tail -1 $d/driver.log 2>/dev/null | cut -c1-120)"
          continue
        fi
      elif [[ -f $d/pid ]] && ! alive $d/pid; then
        if tail -20 $d/codex.log 2>/dev/null | grep -qE "429|at capacity|retry limit"; then echo "SESSION-ENDED-RESUMABLE $l (429/capacity in log tail: resume it)"; else echo "SESSION-ENDED $l"; fi; continue
      fi
      (( ! kept )) && { alive $d/pid || alive $d/driver.pid; } && [[ $watched != *" $l"* && $watched != *"$l "* ]] && echo "UNWATCHED $l"
    done
  done
  git -C $repo worktree list --porcelain | awk '/^worktree /{print $2}' | while read w; do
    [[ $w == $repo ]] && continue
    local b=$(git -C $w branch --show-current 2>/dev/null); [[ -z $b ]] && continue
    local ahead=$(git -C $repo rev-list --count main..$b 2>/dev/null || echo 0)
    (( ahead > 0 )) || continue
    local t=$(( $(date +%s) - $(git -C $repo log -1 --format=%ct $b) ))
    (( t > 21600 )) || continue
    local signed=0; [[ -f $w/state/DONE.md ]] && signed=1; for f in $w/state/READY-<->.md; do signed=1; done
    for C in $dirs; do [[ -f $C/${w:t}/DONE.md ]] && signed=1; for f in $C/${w:t}/READY-<->.md; do signed=1; done; done
    (( signed )) || echo "ORPHAN-BRANCH ${w:t} ahead=$ahead idle=$(( t/3600 ))h"
  done
}
if (( loop )); then
  prev=""
  while true; do for C in $dirs; do touch $C/.audit-heartbeat; done; cur=$(audit | sort); comm -13 <(echo $prev) <(echo $cur); prev=$cur; sleep $loop; done
else
  audit
fi
