#!/bin/bash
# Usage: REPO=<repo> resume.sh <campaign-dir> <lane> <message-file>   — resumes the lane's own Codex session with a follow-up
set -euo pipefail
C=$(cd "$1" && pwd); shift   # campaign dir; set REPO to the repository root
lane=$1; msg=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
sid=$(grep -m1 -i "session id" "$C/$lane/codex.log" | awk '{print $NF}')
n=$( (ls "$C/$lane"/DONE*.md 2>/dev/null || true) | wc -l | tr -d ' ')
if [ -f "$C/$lane/DONE.md" ]; then mv "$C/$lane/DONE.md" "$C/$lane/DONE.partial-$n.md"; fi
mv "$C/$lane/codex.log" "$C/$lane/codex.$(date +%H%M%S).log"
cd "${REPO:-$PWD}"
nohup caffeinate -i env -u OPENAI_API_KEY codex exec resume --dangerously-bypass-approvals-and-sandbox "$sid" "$(cat "$msg")" < /dev/null > "$C/$lane/codex.log" 2>&1 &
echo $! > "$C/$lane/pid"; echo "resumed $lane session $sid"
