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
# The shared admission lock covers keeper launches and manual resumes alike.
if [[ ${LANE_SESSION_GUARDED:-} != "$(cd "$C" && pwd -P)/$lane" ]]; then
  guard=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().with_name("session_guard.py"))' "${BASH_SOURCE[0]}")
  exec python3 "$guard" "$C/$lane" -- /bin/bash "$0" "$@"
fi
mkdir -p "$C/$lane"
brief="$C/briefs/$lane.md"
claude_model() { [[ $1 =~ ^(opus|sonnet|haiku|fable|claude-) ]]; }
engine=Codex; claude_model "$model" && engine=Claude
prompt="You are the $engine worker for lane $lane. Your brief follows; it is complete and authorised. Begin now.

$(cat "$brief")"
if [ -f "$C/$lane/HANDOFF.md" ]; then
  k=$( (ls "$C/$lane"/HANDOFF.*.md 2>/dev/null || true) | wc -l | tr -d " ")
  prompt="$prompt

## Continue from the previous session's handoff (read it first; do not redo finished work)

$(cat "$C/$lane/HANDOFF.md")"
  mv "$C/$lane/HANDOFF.md" "$C/$lane/HANDOFF.$k.md"
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
if [[ $engine == Claude ]]; then
  # Subscription, not API billing: the child must not see ANTHROPIC_API_KEY. Effort has no Claude CLI flag and is ignored.
  sid=$(python3 -c 'import uuid; print(uuid.uuid4())')
  echo "session id: $sid" >> "$C/$lane/codex.log"
  nohup ${caf[@]+"${caf[@]}"} env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u LANE_SESSION_GUARDED claude -p "$(cat "$C/$lane/PROMPT.txt")" --model "$model" --permission-mode bypassPermissions --session-id "$sid" --output-format stream-json --verbose < /dev/null >> "$C/$lane/codex.log" 2>&1 &
else
nohup ${caf[@]+"${caf[@]}"} env -u OPENAI_API_KEY -u LANE_SESSION_GUARDED codex exec -m "$model" -c model_reasoning_effort="$effort" --dangerously-bypass-approvals-and-sandbox -C "$REPO" "$(cat "$C/$lane/PROMPT.txt")" < /dev/null >> "$C/$lane/codex.log" 2>&1 &
fi
echo $! > "$C/$lane/pid"; echo "$model $effort" > "$C/$lane/model"
echo "launched $lane ($model/$effort)"
