#!/bin/bash
# Usage: launch.sh <lane> [model] [effort]      (copy into a campaign dir as <campaign>/launch.sh, or symlink it)
# The campaign dir is this script's folder (or $CAMPAIGN_DIR); the worker runs in $REPO (default: the keeper's repo).
# Reads briefs/<lane>.md, a pending HANDOFF.md, ADDENDUM.md and (with WRAP) wrap-common.md.
set -euo pipefail
C=${CAMPAIGN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
REPO=${REPO:-/Users/stian/src/skard}
hb="$C/.watcher-heartbeat"
mt() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }
{ [ -f "$hb" ] && [ $(( $(date +%s) - $(mt "$hb") )) -lt 600 ]; } || [ "${ALLOW_UNWATCHED:-}" = 1 ] || { echo "refused: no keeper heartbeat in $C (the keeper enforces the cost cap)"; exit 2; }
lane=$1; model=${2:-gpt-6.1-sol}; effort=${3:-medium}
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
cd "$REPO"
caf=(); command -v caffeinate >/dev/null && caf=(caffeinate -i)
date +%s > "$C/$lane/session-at"   # Linux has no file birth time: the keeper reads the session start here
nohup ${caf[@]+"${caf[@]}"} env -u OPENAI_API_KEY codex exec -m "$model" -c model_reasoning_effort="$effort" --dangerously-bypass-approvals-and-sandbox -C "$REPO" "$(cat "$C/$lane/PROMPT.txt")" < /dev/null > "$C/$lane/codex.log" 2>&1 &
echo $! > "$C/$lane/pid"; echo "$model $effort" > "$C/$lane/model"
echo "launched $lane ($model/$effort)"
