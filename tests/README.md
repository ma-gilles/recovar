# Test Suite Guide

## Quick reference

All tiers are accessible directly via `pytest`. The `./scripts/run_pytests.sh`
wrapper just adds the right flags automatically.

| Direct `pytest` command | Equivalent script | What runs | Time | Requires |
|---|---|---|---|---|
| `pytest` | `run_pytests.sh fast` | Unit tests only | ~30 s | nothing |
| `pytest --run-integration tests/integration/test_pipeline_smoke.py` | `run_pytests.sh smoke` | Full pipeline SPA + cryo-ET (tiny dataset) | ~2 min | CPU only |
| `pytest --run-integration --run-gpu --run-slow` | `run_pytests.sh full` | Unit + all integration + slow tests | ~10 min | GPU |
| `pytest --run-integration --run-gpu --run-slow --run-tiny-metrics` | `run_pytests.sh tiny-metrics` | Quality metrics + outlier detection (small data) | ~30 min | GPU |
| `pytest --long-test` | `run_pytests.sh long-test` | Full quality regression suite | 6–12 h | GPU |

No external data files are required for any tier — all data is generated synthetically.

---

## Tiers in detail

### `fast` — unit tests only

```bash
pytest -m "unit and not slow and not integration and not gpu"
# or: ./scripts/run_pytests.sh fast
```

Runs all unit tests in `tests/unit/` (~1 000 tests). No GPU, no subprocess calls,
no file I/O. Safe to run anywhere, anytime.

---

### `smoke` — full pipeline, tiny dataset, CPU

```bash
pytest --run-integration tests/integration/test_pipeline_smoke.py
# or: ./scripts/run_pytests.sh smoke
```

Generates a synthetic 100-image dataset (grid_size=32) and runs
`pipeline_with_outliers` end-to-end on CPU for both SPA and cryo-ET.
Checks that expected output files exist and that inliers + outliers = total
images. No quality baseline comparison — purely "does the code crash?".

**Env vars (all optional):**

| Var | Default | Meaning |
|---|---|---|
| `SMOKE_N_IMAGES` | 100 | SPA: number of images |
| `SMOKE_N_IMAGES_ET` | 300 | cryo-ET: number of images (÷27 tilts ≈ 11 particles) |
| `SMOKE_GRID_SIZE` | 32 | Grid size |
| `SMOKE_ZDIM` | 4 | Latent dimensionality |
| `SMOKE_K_ROUNDS` | 1 | Outlier-removal rounds |
| `SMOKE_PCT_OUTLIERS` | 0.20 | Fraction of synthetic outliers |

---

### `full` — unit + integration + slow + GPU

```bash
pytest --run-integration --run-gpu --run-slow
# or: ./scripts/run_pytests.sh full
```

Includes everything in `fast` plus all integration and slow tests that do
not require large datasets. Includes the smoke tests and the tiny
self-contained outlier regression (`test_outliers_pipeline_tiny_regression`).

---

### `tiny-metrics` — quality check, no external data

```bash
pytest --run-integration --run-gpu --run-slow --run-tiny-metrics
# or: ./scripts/run_pytests.sh tiny-metrics
```

Generates small synthetic datasets (grid_size=32, ~800 images) and runs the
full metrics pipeline and outlier pipeline end-to-end. Checks real quality
metrics (FSC, embedding error, outlier precision/recall) against committed
baselines in `tests/baselines/`. Requires a GPU.

Tests activated by this mode:

- `test_run_test_all_metrics_tiny_integration` — SPA metrics pipeline
- `test_run_test_outliers_pipeline_tiny_integration_spa` — SPA outlier pipeline
- `test_run_test_outliers_pipeline_tiny_integration_cryo_et` — cryo-ET outlier pipeline

---

### `long-test` — full quality regression suite

```bash
# Minimum — outputs go to pytest tmp_path:
pytest --long-test

# Recommended — redirect large outputs off home quota:
LONG_METRICS_OUTPUT_BASE=/scratch/gpfs/$USER/recovar_tests \
pytest --long-test

# or via script (identical):
LONG_METRICS_OUTPUT_BASE=/scratch/gpfs/$USER/recovar_tests \
./scripts/run_pytests.sh long-test
```

`--long-test` implies `--run-integration --run-gpu --run-slow` automatically,
so no other flags are needed.

Runs the complete set of quality-regression tests. All data is generated
synthetically (grid_size=128, n_images=10 000–50 000). Baselines are
auto-saved to `tests/baselines/` on first run, then compared on every
subsequent run.

**Tests included:**

| Test | What it checks | Approx time |
|---|---|---|
| `test_run_test_all_metrics_regression_against_baseline` | SPA: FSC, embedding, contrast metrics vs baseline | ~1–2 h |
| `test_run_test_all_metrics_cryo_et_subsampling_regression_against_baseline` | cryo-ET: same metrics + tilt subsampling consistency | ~2–3 h |
| `test_outliers_pipeline_regression_against_baseline` | SPA outliers: precision/recall/F1 vs baseline | ~1–2 h |
| `test_outliers_pipeline_cryo_et_regression_against_baseline` | cryo-ET outliers: particle-level metrics vs baseline | ~2–3 h |
| `test_pipeline_spa_with_ind_regression` | SPA with `--ind` (80% subset): quality vs baseline | ~1–2 h |
| `test_pipeline_cryo_et_with_particle_ind_regression` | cryo-ET with `--particle-ind`: particle quality vs baseline | ~2–3 h |

**Env vars (all optional):**

| Var | Default | Meaning |
|---|---|---|
| `LONG_METRICS_OUTPUT_BASE` | pytest `tmp_path` | Where to write large scratch outputs (set to `/scratch/...`) |
| `LONG_METRICS_RUN_ARGS` | `--grid-size 128 --n-images 50000 ...` | Extra args for `run_test_all_metrics` |
| `LONG_METRICS_TOL_FRAC` | 0.10 | Allowed relative metric degradation |
| `LONG_METRICS_WRITE_BASELINE` | 0 | Set to `1` to regenerate baselines |
| `OUTLIERS_N_IMAGES` | 10000 | Images for outlier tests |
| `OUTLIERS_GRID_SIZE` | 128 | Grid size for outlier tests |
| `OUTLIERS_K_ROUNDS` | 2 | Outlier-removal rounds |
| `OUTLIERS_TOL_FRAC` | 0.15 | Outlier metric tolerance |
| `OUTLIERS_WRITE_BASELINE` | 0 | Set to `1` to regenerate outlier baselines |
| `PIPELINE_IND_N_IMAGES` | 10000 | Images for with-ind tests |
| `PIPELINE_IND_FRAC` | 0.80 | Fraction of images kept via `--ind` |
| `PIPELINE_IND_TOL_FRAC` | 0.20 | Allowed degradation for with-ind tests |
| `PIPELINE_IND_WRITE_BASELINE` | 0 | Set to `1` to regenerate with-ind baselines |

You can also override volumes with real data:

| Var | Meaning |
|---|---|
| `LONG_METRICS_VOLUMES_DIR` | Real volume prefix (e.g. `/scratch/.../vols/vol`) — uses synthetic if unset |
| `OUTLIERS_VOLUMES_DIR` | Same for outlier tests |

#### First-run / regenerating baselines

On first run the tests write baselines to `tests/baselines/` and skip
(skipped = pass). Commit the new baseline files, then subsequent runs compare
against them. To force a regeneration:

```bash
LONG_METRICS_WRITE_BASELINE=1 \
OUTLIERS_WRITE_BASELINE=1 \
PIPELINE_IND_WRITE_BASELINE=1 \
LONG_METRICS_OUTPUT_BASE=/scratch/gpfs/$USER/recovar_tests \
./scripts/run_pytests.sh long-test
```

---

## Baseline files

Baselines are stored under `tests/baselines/` and committed to the repo:

```
tests/baselines/
  run_test_all_metrics/
    long_generated/
      all_scores.json          # SPA long-test baseline
      all_scores_cryo_et.json  # cryo-ET long-test baseline
    clean_<timestamp>/         # point-in-time snapshots (captured manually)
      all_scores.json
      metadata.json
  run_test_outliers_pipeline/
    long_generated/
      all_scores.json          # SPA outliers long-test baseline
      all_scores_cryo_et.json  # cryo-ET outliers long-test baseline
    tiny_baseline.json         # tiny self-contained outlier baseline
  pipeline_with_indices/
    spa_baseline.json          # SPA with --ind baseline
    cryo_et_baseline.json      # cryo-ET with --particle-ind baseline
  tilt_index_consistency/
    v1.json
```

---

## Running individual tests

```bash
# Unit tests for a specific module (no flags needed):
pytest tests/unit/test_core_slicing.py -v

# Smoke tests (--run-integration is the only flag needed):
pytest --run-integration tests/integration/test_pipeline_smoke.py -v

# Tiny metrics tests:
pytest --run-integration --run-gpu --run-slow --run-tiny-metrics \
    tests/integration/test_run_test_all_metrics_tiny_integration.py -v

# All long quality tests:
pytest --long-test -v

# One specific long test:
pytest --long-test -k test_run_test_all_metrics_regression_against_baseline -v

# All long tests except one:
pytest --long-test -k "not cryo_et" -v
```

---

## Markers

| Marker | Activated by | Meaning |
|---|---|---|
| `unit` | always | Fast, isolated unit test |
| `integration` | `--run-integration` | Crosses module boundaries / spawns subprocesses |
| `slow` | `--run-slow` | Takes more than a few seconds |
| `gpu` | `--run-gpu` | Requires CUDA GPU |
| `io` | always | Heavy file I/O |
| `tiny_metrics` | `--run-tiny-metrics` | Tiny end-to-end metrics/outliers quality test |
| `long_test` | `--long-test` | Multi-hour quality regression (synthetic data, no files needed) |

`--long-test` implies `--run-integration`, `--run-slow`, and `--run-gpu`.

---

## Adding new tests

- Put fast, isolated logic in `tests/unit/`.
- Put full-pipeline tests in `tests/integration/`.
  - Smoke tests (no quality check) → `test_pipeline_smoke.py`, mark `integration`.
  - Quality tests needing GPU but no large data → mark `integration`, `slow`, `gpu`, `tiny_metrics`.
  - Long multi-hour quality regressions → mark `integration`, `slow`, `gpu`, `long_test`.
- Keep tests deterministic: the `_set_deterministic_seed` autouse fixture sets
  `np.random.seed(0)` before every test.
- One behavior per test; prefer small focused tests over large omnibus ones.
