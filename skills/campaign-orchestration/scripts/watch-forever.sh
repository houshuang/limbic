#!/bin/bash
# Usage: watch-forever.sh <campaign-dir> <lanes…>
# watch.sh exits on its first event, so a watcher started in the background goes blind after one event.
# This keeps watching: each event is appended to <campaign-dir>/EVENTS, the lane that raised it leaves
# the watch list (the coordinator handles it and re-adds it by appending its name to <campaign-dir>/WATCH-ADD),
# and watch.sh restarts at once on the remaining lanes. Tail EVENTS from a Monitor to be woken.
C=$1; shift; lanes=("$@")
echo $$ > "$C/watch-forever.pid"
while true; do
  if [ -s "$C/WATCH-ADD" ]; then
    while read -r l; do [ -n "$l" ] && [[ " ${lanes[*]} " != *" $l "* ]] && lanes+=("$l"); done < "$C/WATCH-ADD"
    : > "$C/WATCH-ADD"
  fi
  if [ ${#lanes[@]} -eq 0 ]; then touch "$C/.watcher-heartbeat"; sleep 60; continue; fi
  out=$(bash "$C/watch.sh" "${lanes[@]}" 2>&1 | tail -6)
  event=$(printf '%s\n' "$out" | grep -E '^(DONE|ENDED|STALE|DRIVER-STALE|SHORT-SESSION|FRESH-LIMIT|DISK)' | head -1)
  echo "$(date '+%F %T') ${event:-WATCH-EXIT $(printf '%s' "$out" | tail -1)}" >> "$C/EVENTS"
  lane=$(printf '%s' "$event" | awk '{print $2}')
  if [ -n "$lane" ]; then
    keep=(); for l in "${lanes[@]}"; do [ "$l" != "$lane" ] && keep+=("$l"); done; lanes=("${keep[@]}")
  else
    sleep 60
  fi
done
