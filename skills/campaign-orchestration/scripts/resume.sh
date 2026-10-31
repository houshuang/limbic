#!/bin/bash
# Usage: REPO=<repo> resume.sh <campaign-dir> <lane> <message-file>   — resumes the lane's own Codex session with a follow-up
set -euo pipefail
original_args=("$@")
C=$(cd "$1" && pwd -P); shift   # campaign dir; set REPO to the repository root
lane=$1; msg=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
if [[ ${LANE_SESSION_GUARDED:-} != "$C/$lane" ]]; then
  guard=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().with_name("session_guard.py"))' "${BASH_SOURCE[0]}")
  exec python3 "$guard" "$C/$lane" -- /bin/bash "$0" "${original_args[@]}"
fi
sid=$(python3 "$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().with_name("session_guard.py"))' "${BASH_SOURCE[0]}")" --session-id "$C/$lane/codex.log")
[[ -n $sid ]] || { echo "refused: no session id in $C/$lane/codex.log"; exit 2; }
n=$( (ls "$C/$lane"/DONE*.md 2>/dev/null || true) | wc -l | tr -d ' ')
if [ -f "$C/$lane/DONE.md" ]; then mv "$C/$lane/DONE.md" "$C/$lane/DONE.partial-$n.md"; fi
cd "${REPO:-$PWD}"
caf=(); command -v caffeinate >/dev/null && caf=(caffeinate -i)
date +%s > "$C/$lane/session-at"   # Linux has no file birth time: the keeper reads the session start here
nohup ${caf[@]+"${caf[@]}"} env -u OPENAI_API_KEY -u LANE_SESSION_GUARDED codex exec resume --dangerously-bypass-approvals-and-sandbox "$sid" "$(cat "$msg")" < /dev/null >> "$C/$lane/codex.log" 2>&1 &
echo $! > "$C/$lane/pid"; echo "resumed $lane session $sid"
