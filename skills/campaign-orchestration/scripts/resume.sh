#!/bin/bash
# Usage: REPO=<repo> resume.sh <campaign-dir> <lane> <message-file>   — resumes the lane's own Codex or Claude session with a follow-up
# A lane whose model file names a Claude model but whose last session is not a Claude session (an engine switch)
# gets a fresh Claude session from launch.sh instead: brief + HANDOFF, the follow-up message is not needed then.
set -euo pipefail
original_args=("$@")
C=$(cd "$1" && pwd -P); shift   # campaign dir; set REPO to the repository root
lane=$1; msg=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
if [[ ${LANE_SESSION_GUARDED:-} != "$C/$lane" ]]; then
  guard=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().with_name("session_guard.py"))' "${BASH_SOURCE[0]}")
  exec python3 "$guard" "$C/$lane" -- /bin/bash "$0" "${original_args[@]}"
fi
model=(); [ -f "$C/$lane/model" ] && read -r -a model < "$C/$lane/model"
claude_lane=; [[ ${model[0]:-} =~ ^(opus|sonnet|haiku|fable|claude-) ]] && claude_lane=1
sid=$(python3 "$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().with_name("session_guard.py"))' "${BASH_SOURCE[0]}")" --session-id "$C/$lane/codex.log")
[[ -n $sid ]] || { echo "refused: no session id in $C/$lane/codex.log"; exit 2; }
if [[ -n $claude_lane ]] && ! compgen -G "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/projects/*/$sid.jsonl" > /dev/null; then
  echo "engine switch: $lane is a Claude lane and $sid is not a Claude session; starting a fresh session"
  launcher=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().with_name("launch.sh"))' "${BASH_SOURCE[0]}")
  exec env CAMPAIGN_DIR="$C" REPO="${REPO:-$PWD}" /bin/bash "$launcher" "$lane" "${model[@]}"
fi
n=$( (ls "$C/$lane"/DONE*.md 2>/dev/null || true) | wc -l | tr -d ' ')
if [ -f "$C/$lane/DONE.md" ]; then mv "$C/$lane/DONE.md" "$C/$lane/DONE.partial-$n.md"; fi
cd "${REPO:-$PWD}"
caf=(); command -v caffeinate >/dev/null && caf=(caffeinate -i)
date +%s > "$C/$lane/session-at"   # Linux has no file birth time: the keeper reads the session start here
if [[ -n $claude_lane ]]; then
  nohup ${caf[@]+"${caf[@]}"} env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u LANE_SESSION_GUARDED claude -p --resume "$sid" "$(cat "$msg")" --permission-mode bypassPermissions --output-format stream-json --verbose < /dev/null >> "$C/$lane/codex.log" 2>&1 &
else
nohup ${caf[@]+"${caf[@]}"} env -u OPENAI_API_KEY -u LANE_SESSION_GUARDED codex exec resume --dangerously-bypass-approvals-and-sandbox "$sid" "$(cat "$msg")" < /dev/null >> "$C/$lane/codex.log" 2>&1 &
fi
echo $! > "$C/$lane/pid"; echo "resumed $lane session $sid"
