#!/bin/bash
# Usage: launch.sh <lane> [model] [effort]
set -euo pipefail
hb="/Users/stian/src/skard-state/campaign-20261023/.watcher-heartbeat"
{ [ -f "$hb" ] && [ $(( $(date +%s) - $(stat -f %m "$hb") )) -lt 600 ]; } || [ "${ALLOW_UNWATCHED:-}" = 1 ] || { echo "refused: start watch.sh first (it enforces the cost cap)"; exit 2; }
lane=$1; model=${2:-gpt-6.1-sol}; effort=${3:-medium}
C=/Users/stian/src/skard-state/campaign-20261023
[ -f "$C/$lane/pid" ] && kill -0 "$(cat "$C/$lane/pid")" 2>/dev/null && { echo "refused: lane $lane already has a live session (pid $(cat "$C/$lane/pid"))"; exit 3; }
mkdir -p "$C/$lane"
brief="$C/briefs/$lane.md"
prompt="You are the Codex worker for lane $lane. Your brief follows; it is complete and authorised. Begin now.

$(cat "$brief")"
if [ -f "$C/$lane/HANDOFF.md" ]; then
  k=$( (ls "$C/$lane"/HANDOFF.*.md 2>/dev/null || true) | wc -l | tr -d " ")
  prompt="$prompt

## Continue from the previous session's handoff (read it first; do not redo finished work)

$(cat "$C/$lane/HANDOFF.md")"
  mv "$C/$lane/HANDOFF.md" "$C/$lane/HANDOFF.$k.md"
  [ -f "$C/$lane/codex.log" ] && mv "$C/$lane/codex.log" "$C/$lane/codex.$(date +%H%M%S).log"
fi
if [ -f "$C/$lane/ADDENDUM.md" ]; then prompt="$prompt

## Addendum (read and apply)

$(cat "$C/$lane/ADDENDUM.md")"; fi
if [ -f "$C/WRAP" ]; then prompt="$(cat "$C/wrap-common.md")

$prompt"; fi
printf '%s' "$prompt" > "$C/$lane/PROMPT.txt"
cd /Users/stian/src/skard
nohup caffeinate -i env -u OPENAI_API_KEY codex exec -m "$model" -c model_reasoning_effort="$effort" --dangerously-bypass-approvals-and-sandbox -C /Users/stian/src/skard "$(cat "$C/$lane/PROMPT.txt")" < /dev/null > "$C/$lane/codex.log" 2>&1 &
echo $! > "$C/$lane/pid"; echo "$model $effort" > "$C/$lane/model"
echo "launched $lane ($model/$effort)"
