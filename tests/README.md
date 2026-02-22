# Test Suite Guide

This repository uses `pytest` with explicit test tiers to keep validation fast and reliable as coverage grows.

## Layout

- `tests/unit/`: fast, isolated tests for functions and modules.
- `tests/smoke/`: minimal import/sanity tests.
- `tests/integration/` (planned): multi-module behavioral tests.

## Markers

- `unit`: baseline tests expected to run frequently.
- `integration`: tests crossing module boundaries.
- `gpu`: tests requiring GPU runtime.
- `slow`: long-running tests.
- `io`: file-heavy tests.

## Running tests

- Fast default:
  - `./scripts/run_pytests.sh`
- Integration:
  - `./scripts/run_pytests.sh integration`
- GPU:
  - `./scripts/run_pytests.sh gpu`
- Full:
  - `./scripts/run_pytests.sh full`
- Full + long regressions:
  - `./scripts/run_pytests.sh full-long`
  - Runs full suite, then both long metrics regressions (standard + cryo-ET subsampling)
- Long metrics regression (opt-in, typically 1h+):
  - `./scripts/run_pytests.sh long-metrics`
  - Requires env vars described in `scripts/run_long_metrics_regression.sh`
  - Includes both:
    - standard (non-tilt) long metrics regression
    - cryo-ET long metrics regression with tilt/image/ntilts subsampling consistency checks
  - For large runs, set `LONG_METRICS_OUTPUT_BASE=/scratch/...` (or `/tigress/...`) to avoid HOME quota pressure.
- Tiny end-to-end metrics baseline gate (opt-in):
  - `RUN_TINY_METRICS_INTEGRATION=1 pytest --run-integration --run-gpu --run-slow -q tests/integration/test_run_test_all_metrics_tiny_regression_baseline.py`
  - Optional tolerance override: `RUN_TINY_METRICS_TOL_FRAC=0.10`

You can always pass extra pytest args, e.g.:

- `./scripts/run_pytests.sh fast -k fourier_transform_utils -q`

## Extending the suite

- Add new tests file-by-file in `tests/unit/` first.
- Use `integration` marker only when test setup spans multiple modules.
- Prefer deterministic random seeds and avoid global mutable state.
- Keep tests small and focused; one behavior per test.
