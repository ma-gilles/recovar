#!/usr/bin/env bash
#
# Clean-room install smoke test for the recovar web GUI.
#
# Reproduces a *user* install — a fresh virtualenv with only the declared
# dependencies, isolated from ~/.local — and verifies the backend can import
# and open a project database. This is the guard that would have caught issue
# #142: the [gui] extra was missing the `aiosqlite` driver, so
# `pip install recovar[gui]` installed cleanly but crashed the instant a user
# created a project.
#
# Usage:
#   scripts/test_gui_clean_install.sh [WHEEL]
#
#   WHEEL   Optional path to a prebuilt recovar wheel. If omitted, a wheel is
#           built from the current source tree.
#
# Environment:
#   PYTHON  Interpreter used to build the wheel and create the venv
#           (default: python3).
#
# Note: this installs recovar's full runtime stack into a throwaway venv and
# therefore needs network access; it is meant for CI, not the login node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d -t recovar-gui-cleaninstall-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Clean-room GUI install test"
echo "    repo:  $REPO_ROOT"
echo "    work:  $WORK"

WHEEL="${1:-}"
if [ -z "$WHEEL" ]; then
    echo "==> Building wheel from source"
    "$PYTHON" -m pip wheel --no-build-isolation --no-deps -w "$WORK/wheel" "$REPO_ROOT" >/dev/null
    WHEEL="$(ls -t "$WORK"/wheel/recovar-*.whl | head -1)"
fi
echo "    wheel: $WHEEL"

echo "==> Creating fresh virtualenv (isolated from ~/.local)"
"$PYTHON" -m venv "$WORK/venv"
VPY="$WORK/venv/bin/python"
export PYTHONNOUSERSITE=1

echo "==> Installing recovar[gui] (no cache — exactly what a user gets)"
"$VPY" -m pip install --quiet --upgrade pip >/dev/null
"$VPY" -m pip install --no-cache-dir "${WHEEL}[gui]"

echo "==> Smoke: import the app and open a project DB (the issue #142 path)"
"$VPY" - <<'PYEOF'
import asyncio
import tempfile
from pathlib import Path

# 1) The whole app must import with only the declared [gui] deps present.
from recovar.gui_v2.backend.main import create_app

assert create_app() is not None

# 2) The exact failing frame from issue #142: init_db opens sqlite+aiosqlite,
#    which loads the aiosqlite driver. Raises ModuleNotFoundError if undeclared.
from recovar.gui_v2.backend.db import close_all, init_db


async def main():
    with tempfile.TemporaryDirectory() as d:
        factory = await init_db(Path(d) / "recovar_project.db")
        assert factory is not None
    await close_all()


asyncio.run(main())
print("OK: app imported and project DB opened using only the declared [gui] deps")
PYEOF

echo "==> PASS: clean-room GUI install works"
