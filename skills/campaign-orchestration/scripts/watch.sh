#!/bin/bash
# Event-driven lane watcher. Costs no model tokens while lanes or their drivers run.
#  - DONE.md written            → exit (coordinator reviews)
#  - session died on capacity/429, no handoff → resume after 10 min (max 6)
#  - session gone + HANDOFF.md, driver running → wait
#  - session gone + HANDOFF.md, driver finished/absent → fresh small session (max 12 per lane)
#  - session gone, no HANDOFF/DONE, not transient → exit (coordinator looks)
#  - log and driver log silent > 60 min while running → exit
C=/Users/stian/src/skard-state/campaign-20261023
lanes="$*"
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
while true; do
  touch "$C/.watcher-heartbeat"
  # Enforced cost cap: a lane session over 150k context/turn, 60 turns or 15M tokens/h is stopped and made to hand off.
  for l in $("$C/costguard.py" $lanes); do
    d="$C/$l"; alive "$d/pid" || continue
    [ -f "$d/forced-handoff" ] && [ "$(cat "$d/forced-handoff")" = "$(cat "$d/pid")" ] && continue
    p=$(cat "$d/pid"); pgrep -P "$p" | xargs kill 2>/dev/null; kill "$p" 2>/dev/null; sleep 3
    "$C/resume.sh" "$l" "$C/msg-handoff.md" >> "$C/autoresume.log" 2>&1; cat "$d/pid" > "$d/forced-handoff"
    echo "$(date +%H:%M) forced handoff $l" >> "$C/autoresume.log"
  done
  for l in $lanes; do
    d="$C/$l"
    [ -f "$d/pid" ] || continue   # not launched yet
    if alive "$d/driver.pid"; then   # any live driver must show progress within 60 min
      newest=$(stat -f %m "$d/driver.log" "$d/PROGRESS.md" 2>/dev/null | sort -n | tail -1)
      [ $(( $(date +%s) - ${newest:-$(date +%s)} )) -gt 3600 ] && { echo "DRIVER-STALE $l"; exit 0; }
    fi
    [ -f "$d/DONE.md" ] && { echo "DONE $l"; exit 0; }
    if alive "$d/pid"; then
      age=$(( $(date +%s) - $(stat -f %m "$d/codex.log") ))
      [ $age -gt 3600 ] && { echo "STALE $l ${age}s"; exit 0; }
      continue
    fi
    if [ ! -f "$d/HANDOFF.md" ]; then   # sessions sometimes write HANDOFF.<n>.md; accept one newer than the session start
      latest=$(ls -t "$d"/HANDOFF.*.md 2>/dev/null | head -1)
      [ -n "$latest" ] && [ "$(stat -f %m "$latest")" -ge "$(stat -f %B "$d/codex.log")" ] && cp "$latest" "$d/HANDOFF.md"
    fi
    if [ -f "$d/HANDOFF.md" ]; then
      if alive "$d/driver.pid"; then
        newest=$(stat -f %m "$d/driver.log" "$d/PROGRESS.md" 2>/dev/null | sort -n | tail -1); age=$(( $(date +%s) - ${newest:-$(date +%s)} ))
        [ $age -gt 3600 ] && { echo "DRIVER-STALE $l ${age}s"; exit 0; }
        continue
      fi
      # A session that exits within 5 min without leaving a running driver is a loop, not progress.
      lived=$(( $(stat -f %m "$d/codex.log") - $(stat -f %B "$d/codex.log") ))
      started_driver=0; [ -f "$d/driver.pid" ] && [ "$(stat -f %m "$d/driver.pid")" -ge "$(stat -f %B "$d/codex.log")" ] && started_driver=1
      [ "$lived" -lt 300 ] && [ "$started_driver" = 0 ] && [ -f "$d/fresh-at" ] && [ "$(stat -f %B "$d/codex.log")" -ge "$(cat "$d/fresh-at")" ] && { echo "SHORT-SESSION $l (${lived}s, no driver.pid alive)"; exit 0; }
      f=$(cat "$d/fresh" 2>/dev/null || echo 0)
      [ "$f" -ge 25 ] && { echo "FRESH-LIMIT $l"; exit 0; }
      echo $((f+1)) > "$d/fresh"; date +%s > "$d/fresh-at"; "$C/launch.sh" "$l" >> "$C/autoresume.log" 2>&1; continue
    fi
    r=$(cat "$d/autoresumes" 2>/dev/null || echo 0)
    if grep -qE "at capacity|429 Too Many" "$d/codex.log" && [ "$r" -lt 6 ]; then
      echo $((r+1)) > "$d/autoresumes"; sleep 600; "$C/resume.sh" "$l" "$C/msg-capacity.md" >> "$C/autoresume.log" 2>&1; continue
    fi
    echo "ENDED $l"; tail -5 "$d/codex.log"; exit 0
  done
  sleep 120
done
