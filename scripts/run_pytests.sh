#!/usr/bin/env bash
set -euo pipefail

# Standardized pytest entrypoint for local dev and CI usage.
# Modes:
#   fast         - unit/smoke only; no slow/integration/gpu
#   integration  - unit + integration tests
#   gpu          - unit + GPU-marked tests
#   full         - unit + integration + gpu + slow (no large data needed)
#   tiny-metrics - full + tiny end-to-end metrics/outliers tests (no large data needed)
#   full-long    - full suite + long metrics regressions (requires large volumes)
#   real-regression - full suite + strict real-dataset quality gates
#   long-metrics - opt-in very long run_test_all_metrics regression (1h+)
#
# Single-command coverage tiers:
#
#   No large data (generates its own):
#     ./scripts/run_pytests.sh tiny-metrics
#     # equiv: pytest --run-integration --run-gpu --run-slow --run-tiny-metrics
#
#   With large data (cryo-ET, outliers, long metrics):
#     LONG_METRICS_VOLUMES_DIR=... LONG_METRICS_BASELINE_JSON=... \
#     LONG_METRICS_OUTPUT_BASE=/scratch/... \
#     ./scripts/run_pytests.sh full-long
#     # equiv: pytest --run-integration --run-gpu --run-slow --run-tiny-metrics
#     #        + run_long_metrics_regression.sh

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
  tiny-metrics)
    shift || true
    pytest --run-integration --run-gpu --run-slow --run-tiny-metrics "$@"
    ;;
  full-long)
    shift || true
    pytest --run-integration --run-gpu --run-slow --run-tiny-metrics "$@"
    ./scripts/run_long_metrics_regression.sh
    ;;
  real-regression)
    shift || true
    ./scripts/run_full_real_dataset_regression.sh "$@"
    ;;
  long-metrics)
    shift || true
    ./scripts/run_long_metrics_regression.sh "$@"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [fast|integration|gpu|full|tiny-metrics|full-long|real-regression|long-metrics] [extra pytest args...]"
    exit 2
    ;;
esac
