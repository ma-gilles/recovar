#!/usr/bin/env bash
set -euo pipefail

# Standardized pytest entrypoint for local dev and CI usage.
# Modes:
#   fast         - unit/smoke only; no slow/integration/gpu
#   smoke        - pipeline smoke tests only (SPA + cryo-ET, ~2 min on CPU, no data needed)
#   integration  - unit + integration tests (no slow/gpu); includes smoke tests
#   gpu          - unit + GPU-marked tests
#   full         - unit + integration + gpu + slow
#   tiny-metrics - full + tiny end-to-end metrics/outliers quality tests (~30 min on GPU)
#   long-test    - full quality regression suite: cryo-EM SPA, cryo-ET, pipeline with
#                  outliers, pipeline with --ind/--particle-ind (6-12 h on GPU).
#                  All data generated synthetically — no external files needed.
#                  Baselines auto-saved in tests/baselines/ on first run.
#   full-long    - full suite + long metrics regressions (legacy mode)
#   real-regression - full suite + strict real-dataset quality gates (requires volumes dir)
#   long-metrics - opt-in very long run_test_all_metrics regression (1h+)
#
# Coverage tiers (no external data required for any tier):
#
#   Fastest — unit tests only (~30 s):
#     ./scripts/run_pytests.sh fast
#
#   Pipeline smoke — full SPA + cryo-ET end-to-end, tiny dataset (~2 min CPU):
#     ./scripts/run_pytests.sh smoke
#     # or directly: pytest --run-integration tests/integration/test_pipeline_smoke.py
#
#   Quality check — tiny metrics + outlier detection (~30 min GPU):
#     ./scripts/run_pytests.sh tiny-metrics
#
#   Full quality regression — all cryo-EM/ET, outliers, with/without indices (6–12 h GPU):
#     LONG_METRICS_OUTPUT_BASE=/scratch/recovar_tests \
#     ./scripts/run_pytests.sh long-test

MODE="${1:-fast}"

case "$MODE" in
  fast)
    shift || true
    pytest -m "unit and not slow and not integration and not gpu" "$@"
    ;;
  smoke)
    shift || true
    pytest --run-integration tests/integration/test_pipeline_smoke.py "$@"
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
  long-test)
    # Full quality regression suite: cryo-EM SPA, cryo-ET, pipeline with
    # outliers, pipeline with --ind / --particle-ind.
    # --long-test implies --run-integration, --run-gpu, --run-slow.
    # Tests skip gracefully when required env vars are absent.
    shift || true
    pytest --long-test "$@"
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
  parallel)
    shift || true
    bash "$(dirname "$0")/run_tests_parallel.sh" full "$@"
    ;;
  parallel-long)
    shift || true
    bash "$(dirname "$0")/run_tests_parallel.sh" long-test "$@"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [fast|smoke|integration|gpu|full|tiny-metrics|long-test|full-long|real-regression|long-metrics|parallel|parallel-long] [extra pytest args...]"
    exit 2
    ;;
esac
