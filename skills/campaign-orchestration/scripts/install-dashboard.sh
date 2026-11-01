#!/bin/bash
# Usage: install-dashboard.sh <.coordinator.json> [--uninstall]
# Installs a launchd agent that keeps dashboard.py serving (127.0.0.1 only) next to the keeper's agent.
# Run it from the checkout launchd should use (normally the limbic main checkout): the plist points at this
# script's own dashboard.py. Logs go to <state>/keeper/dashboard.out and dashboard.err.
set -euo pipefail
cfg=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
here=$(cd "$(dirname "$0")" && pwd -P)
label=com.$(id -un).keeper-dashboard
plist=$HOME/Library/LaunchAgents/$label.plist
[[ ${2:-} == --uninstall ]] && { launchctl bootout gui/$(id -u)/$label 2>/dev/null || true; rm -f "$plist"; echo "removed $label"; exit 0; }
[[ -f $here/dashboard.py && -f $cfg ]] || { echo "need $here/dashboard.py and $cfg" >&2; exit 2; }
state=$(/usr/bin/python3 -c 'import json,sys,pathlib; c=json.load(open(sys.argv[1])); r=pathlib.Path(c["repo"]); print(c.get("state", r.parent / (r.name + "-state")))' "$cfg")
port=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("dashboard", {}).get("port", 4170))' "$cfg")
mkdir -p "$state/keeper"
cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>$here/dashboard.py</string>
    <string>serve</string>
    <string>--config</string>
    <string>$cfg</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>$HOME</string>
    <key>PATH</key><string>$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>StandardOutPath</key><string>$state/keeper/dashboard.out</string>
  <key>StandardErrorPath</key><string>$state/keeper/dashboard.err</string>
</dict>
</plist>
EOF
launchctl bootout gui/$(id -u)/$label 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$plist"
for _ in $(seq 1 20); do curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 && { echo "$label serving http://127.0.0.1:$port/"; exit 0; }; sleep 1; done
echo "$label installed but not answering on :$port; see $state/keeper/dashboard.err" >&2; exit 1
