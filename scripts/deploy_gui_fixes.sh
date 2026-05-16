#!/usr/bin/env bash
# Deploy the GUI subset-export / density-explore changes to della and verify.
#
# Authenticates ONCE (opens an SSH master connection), then rsyncs the changed
# backend + rebuilt static, restarts the GUI server, re-registers the project,
# and runs the live round-trip verification.
#
# Usage:
#   bash scripts/deploy_gui_fixes.sh
#
# Override any of these via the environment if your della env/ports differ:
#   HOST, VENV, PORT, PROJECT
set -euo pipefail

HOST="${HOST:-della-mol}"
VENV="${VENV:-/scratch/gpfs/GILLES/recovar_clean_test}"
PORT="${PORT:-8083}"
PROJECT="${PROJECT:-/scratch/gpfs/GILLES/gui_10073_20260605_005014}"

SRC="/Users/gilles/research/recovar_guifix/recovar/gui_v2/backend"
SCRIPTS="/Users/gilles/research/recovar_guifix/scripts"
DEST="$VENV/lib/python3.11/site-packages/recovar/gui_v2/backend"
LOG="/scratch/gpfs/GILLES/recovar_gui.log"
CM="$HOME/.ssh/cm-deploy-%r@%h:%p"

echo "== Opening SSH master connection (you will authenticate once) =="
ssh -M -S "$CM" -o ControlPersist=15m -fN "$HOST"
cleanup() { ssh -S "$CM" -O exit "$HOST" 2>/dev/null || true; }
trap cleanup EXIT
run() { ssh -S "$CM" "$HOST" "$@"; }

echo "== Sanity-check the target env exists =="
if ! run "test -d '$DEST'"; then
  echo "ERROR: $DEST does not exist on $HOST." >&2
  echo "Point VENV at the env that runs your GUI, e.g.:" >&2
  echo "  VENV=/path/to/your/recovar/env bash scripts/deploy_gui_fixes.sh" >&2
  exit 1
fi

echo "== rsync changed backend files + rebuilt static =="
rsync -az -e "ssh -S $CM" "$SRC/api/subsets.py"    "$HOST:$DEST/api/subsets.py"
rsync -az -e "ssh -S $CM" "$SRC/api/embeddings.py" "$HOST:$DEST/api/embeddings.py"
rsync -az --delete -e "ssh -S $CM" "$SRC/static/"  "$HOST:$DEST/static/"
run "mkdir -p '$VENV/_gui_verify'"
rsync -az -e "ssh -S $CM" "$SCRIPTS/verify_gui_fixes.py" "$HOST:$VENV/_gui_verify/verify_gui_fixes.py"

echo "== Import-check the deployed backend (catches syntax/import errors) =="
run "$VENV/bin/python - <<'PY'
import importlib
for m in ('recovar.gui_v2.backend.api.subsets','recovar.gui_v2.backend.api.embeddings'):
    importlib.import_module(m)
    print('import OK:', m)
PY"

echo "== Restart GUI server on :$PORT =="
run "PID=\$(ss -ltnpH \"sport = :$PORT\" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2); \
     if [ -n \"\$PID\" ]; then echo killing old server pid \$PID; kill \$PID; sleep 2; fi; \
     nohup '$VENV/bin/recovar' gui --no-browser --port $PORT > '$LOG' 2>&1 </dev/null & \
     sleep 5; echo '--- server log tail ---'; tail -4 '$LOG'"

echo "== Run live verification (re-registers the project, exercises both features) =="
run "$VENV/bin/python '$VENV/_gui_verify/verify_gui_fixes.py' --port $PORT --project '$PROJECT'"

echo
echo "Deploy + verify complete. If you have a tunnel up (ssh -L 8080:localhost:$PORT $HOST),"
echo "refresh the browser to pick up the new frontend."
