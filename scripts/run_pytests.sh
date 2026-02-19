#!/usr/bin/env bash
set -euo pipefail

# Standardized pytest entrypoint for local dev and CI usage.
# Modes:
#   fast         - default unit/smoke coverage, excludes slow/integration/gpu
#   integration  - includes integration tests
#   gpu          - includes GPU-marked tests
#   full         - includes integration + gpu + slow

MODE="${1:-fast}"

case "$MODE" in
  fast)
    shift || true
    pytest -m "unit and not slow and not integration and not gpu" "$@"
    ;;
  integration)
    shift
    pytest --run-integration -m "unit or integration" "$@"
    ;;
  gpu)
    shift
    pytest --run-gpu -m "unit or gpu" "$@"
    ;;
  full)
    shift
    pytest --run-integration --run-gpu --run-slow "$@"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [fast|integration|gpu|full] [extra pytest args...]"
    exit 2
    ;;
esac
