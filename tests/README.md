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

You can always pass extra pytest args, e.g.:

- `./scripts/run_pytests.sh fast -k fourier_transform_utils -q`

## Extending the suite

- Add new tests file-by-file in `tests/unit/` first.
- Use `integration` marker only when test setup spans multiple modules.
- Prefer deterministic random seeds and avoid global mutable state.
- Keep tests small and focused; one behavior per test.

