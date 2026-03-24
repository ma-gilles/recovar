# Test Development Rules

## Hard Rules (non-negotiable)

### NEVER widen tolerance to make tests pass
Do not change `_TOL`, `tol_frac`, `HIGH_VARIANCE_TOKENS`, or add skip/ignore logic for specific metrics. If a test fails, **fix the code**, not the test. You may **suggest** a tolerance change and wait for explicit approval, but never implement it unilaterally.

### NEVER modify files in `tests/baselines/`
Baselines are ground truth generated from the OLD published recovar code (`~/recovar`) with PDB volumes and GT mask. They represent the correct behavior of the published algorithm. Modifying them silently accepts regressions. Only exception: the user explicitly says "regenerate the baseline for X".

### NEVER use `pytest -q` for long-running tests
The `-q` flag suppresses all output until completion. For multi-hour GPU tests, this gives zero progress visibility. Use no flag or `-v` instead.

### Comparison tables must be visible
Use `logging.info()` or `sys.stderr` for regression comparison tables — NOT `print()` (pytest captures stdout on pass). Every regression test must save scores to JSON AND print a comparison table showing current vs baseline with % change.

## Test Tiers & Markers

| Marker | Flag | Purpose |
|--------|------|---------|
| `unit` | (always runs) | Fast isolated tests, no GPU, no subprocess |
| `integration` | `--run-integration` | Multi-module, may spawn subprocesses |
| `gpu` | `--run-gpu` | Requires CUDA GPU |
| `slow` | `--run-slow` | Takes more than a few seconds |
| `tiny_metrics` | `--run-tiny-metrics` | Quick quality check (32^3, ~800 images) |
| `long_test` | `--long-test` | Full regression suite (128^3, 50k images, 6-12h) |

`--long-test` implies `--run-integration`, `--run-slow`, `--run-gpu`.

All data is generated synthetically — no external downloads needed.

## Baseline Management

```
tests/baselines/
  run_test_all_metrics/long_generated/     # SPA + ET quality baselines (from OLD code)
  run_test_outliers_pipeline/long_generated/ # Outlier baselines (from OLD code)
  compute_state_regression/                 # compute_state baselines
  pipeline_functions_isolated/              # Per-function baselines
  pipeline_with_indices/                    # Subset selection baselines
  */perf_baseline*.json                     # Performance baselines (auto-updated per GPU)
```

**Quality baselines** (`all_scores*.json`): Sacred. From OLD code. NEVER auto-update.
**Performance baselines** (`perf_baseline*.json`): From NEW code. Auto-updated if missing hardware entry. Warn on regression, don't fail.

### Baseline generation workflow
1. **Current code** generates synthetic datasets
2. **Old `~/recovar` code** (conda env) runs pipeline on that data
3. **Current code** computes metrics on the old output

## How to Add Tests

- **Unit tests** → `tests/unit/`. Mark `@pytest.mark.unit`. No GPU, no subprocesses.
- **Integration tests** → `tests/integration/`. Mark `integration`, add `slow`/`gpu` as needed.
- **Quality regression** → mark `long_test`. Use `log_comparison_table()` from `helpers/metrics_regression.py`.
- Keep tests deterministic: `conftest.py` auto-seeds numpy with `seed=0`.
- One behavior per test; prefer small focused tests over omnibus.

## GPU Test Patterns

```python
# Subprocess tests must use gpu_subprocess_env() for proper GPU isolation
from conftest import gpu_subprocess_env
subprocess.run(cmd, check=True, env=gpu_subprocess_env())
```

This sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, auto-selects least-loaded GPU, and isolates Python paths.

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LONG_METRICS_OUTPUT_BASE` | pytest tmp_path | Redirect large outputs off home quota |
| `LONG_METRICS_TOL_FRAC` | 0.01 | Allowed relative metric degradation |
| `LONG_METRICS_WRITE_BASELINE` | 0 | Set to 1 to regenerate baselines |
| `CUDA_VISIBLE_DEVICES` | auto-selected | GPU selection |
