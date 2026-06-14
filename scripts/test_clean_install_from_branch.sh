#!/usr/bin/env bash
#
# True external-user test: cold `pip install` of the GUI straight from the
# pushed branch (git URL) into a fresh venv, then actually LAUNCH the GUI and
# confirm it serves the bundled frontend. Answers "is any fidgeting needed?".
#
#   PYTHON=python3 scripts/test_clean_install_from_branch.sh [BRANCH]
set -uo pipefail

PYTHON="${PYTHON:-python3}"
BRANCH="${1:-claude/gui-bugfixes}"
URL="git+https://github.com/ma-gilles/recovar.git@${BRANCH}"
PORT="${PORT:-8199}"
WORK="$(mktemp -d -t recovar-branchinstall-XXXXXX)"
FAIL=0
note() { echo ">>> $*"; }
bad() { echo "!!! FIDGET/FAIL: $*"; FAIL=1; }

note "branch: $BRANCH"
note "work:   $WORK"

note "creating fresh venv (isolated from ~/.local)"
"$PYTHON" -m venv "$WORK/venv" || { bad "venv create failed"; exit 1; }
VPY="$WORK/venv/bin/python"
VBIN="$WORK/venv/bin"
export PYTHONNOUSERSITE=1
"$VPY" -m pip install -q --upgrade pip >/dev/null 2>&1

note "cold install: pip install 'recovar[gui] @ $URL'  (no cache, build-isolated — exactly what a user runs)"
if ! "$VPY" -m pip install --no-cache-dir "recovar[gui] @ ${URL}" 2>&1 | tail -3; then
  bad "pip install from branch failed"; rm -rf "$WORK"; exit 1
fi

note "is the 'recovar' command on PATH?"
[ -x "$VBIN/recovar" ] && echo "    yes: $VBIN/recovar" || bad "recovar console-script missing"

note "recovar gui --check (readiness)"
"$VBIN/recovar" gui --check 2>&1 | sed 's/^/    /' || bad "gui --check errored"

note "launch: recovar gui --no-browser --port $PORT"
"$VBIN/recovar" gui --no-browser --port "$PORT" > "$WORK/gui.log" 2>&1 &
GUIPID=$!
ready=""
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/api/system/info" 2>/dev/null)
  if [ "$code" = "200" ]; then ready="yes (${i}s)"; break; fi
  sleep 1
done
# the server may have advanced the port if $PORT was taken — read the real one
realport="$PORT"
if [ -z "$ready" ]; then
  rp=$(grep -oE "Uvicorn running on http://127.0.0.1:[0-9]+" "$WORK/gui.log" | grep -oE "[0-9]+$" | tail -1)
  if [ -n "$rp" ]; then realport="$rp"; code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${rp}/api/system/info"); [ "$code" = "200" ] && ready="yes (port $rp)"; fi
fi
if [ -n "$ready" ]; then echo "    API up: $ready"; else bad "GUI server did not come up"; echo "    --- gui.log ---"; tail -15 "$WORK/gui.log"; fi

note "does it SERVE the bundled frontend? (the static must be packaged in the wheel)"
HTML=$(curl -s "http://127.0.0.1:${realport}/" 2>/dev/null)
if echo "$HTML" | grep -qiE "<title>|<div id=\"root\"|assets/index-"; then
  echo "    yes — frontend HTML served:"; echo "$HTML" | grep -oE "<title>[^<]*</title>|assets/index-[A-Za-z0-9]+\.(js|css)" | head -4 | sed 's/^/      /'
else
  bad "frontend not served (static not bundled in the wheel?)"; echo "    got: $(echo "$HTML" | head -c 200)"
fi

kill "$GUIPID" 2>/dev/null || true
wait "$GUIPID" 2>/dev/null || true
rm -rf "$WORK"

echo
if [ "$FAIL" -eq 0 ]; then
  echo "================  RESULT: NO FIDGETING — pip install + recovar gui just works  ================"
else
  echo "================  RESULT: needs attention (see FIDGET/FAIL lines above)  ================"
fi
exit "$FAIL"
