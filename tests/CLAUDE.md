# Test Development Rules

## Hard Rules (non-negotiable)

### NEVER widen tolerance to make tests pass
Do not change `_TOL`, `tol_frac`, `HIGH_VARIANCE_TOKENS`, or add skip/ignore logic for specific metrics. If a test fails, **fix the code**, not the test. You may **suggest** a tolerance change and wait for explicit approval, but never implement it unilaterally.

### Float64 companion required when tolerance is loosened
If a new or modified test must use a tolerance wider than machine-epsilon (e.g. `atol=1e-5` for float32 code), **add a float64 companion test** that runs the same comparison with tighter tolerances (e.g. `atol=1e-8`). The companion helps test whether rounding explains the gap. Pattern:
```python
def test_foo():
    """Float32 test — tolerance limited by single precision."""
    d = make_data(float_dtype=np.float32)
    compare(d, atol=1e-5)

def test_foo_f64():
    """Float64 companion — checks the same comparison at higher precision."""
    jax.config.update("jax_enable_x64", True)
    d = make_data(float_dtype=np.float64)
    compare(d, atol=1e-8)  # ≥3 orders tighter
```
The f64 test must tighten **by at least 3 orders of magnitude**. If it cannot, investigate the discrepancy before changing the implementation or claiming that rounding explains it.

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
| `gpu_memory_matrix` | `--long-test` | 14-cell GPU memory matrix (7 budgets x 2 backends); also exposed via `scripts/run_gpu_memory_matrix.sh` |

`--long-test` implies `--run-integration`, `--run-slow`, `--run-gpu`.

The `gpu_memory_matrix` marker exists alongside `long_test` so the
GPU integration matrix can be selected explicitly (e.g.
`pytest -m gpu_memory_matrix --long-test`) or driven from the Slurm
submitter for parallel cells.

Many tests generate synthetic data locally. EM replay/trajectory tests also
require curated external particle and RELION fixtures. Check the selected test's
input inventory before submitting. Missing fixtures may skip an optional local
check, but a qualification job must fail if a required case did not execute.

### Fast EM Guardrail

For dense/local EM refactor work, run:
`pixi run test-em-fast-guard`

This CPU-default guardrail runs tiny deterministic dense big-JIT, local exact
EM, Fourier-window, dtype-policy, and helper-path tests. It is intended to
finish in under about 60 seconds without the 5k parity dataset. To run it on a
local GPU, check `nvidia-smi` first and then use
`EM_FAST_GUARD_BACKEND=gpu pixi run test-em-fast-guard`.

The same command checks the helper/controller import boundary, captured replay
state and explicit HEALPix schedules. These cases use small in-memory fixtures;
they require no external RELION capture or GPU allocation on the default CPU
path. Keep their case inventory intact when reorganizing tests.

## Baseline Management

```
tests/baselines/
  run_test_all_metrics/long_generated/     # SPA + ET quality baselines (from OLD code)
  run_test_outliers_pipeline/long_generated/ # Outlier baselines (from OLD code)
  compute_state_regression/                 # compute_state baselines
  pipeline_functions_isolated/              # Per-function baselines
  pipeline_with_indices/                    # Subset selection baselines
  */perf_baseline*.json                     # Hardware-specific performance controls
```

**Quality baselines** (`all_scores*.json`): Sacred. From OLD code. NEVER auto-update.
**Performance baselines** (`perf_baseline*.json`): From NEW code, specific to
hardware and workload. Keep the existing 10% regression warning policy.
The legacy performance helper can write missing hardware entries; isolate its
output before running it. This behavior does not authorize modifying committed
baselines during cleanup. Store newly measured controls separately and obtain an
explicit user instruction before replacing any established expected result.

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

This sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, pins `XLA_PYTHON_CLIENT_MEM_FRACTION=.90` for reproducible baselines, can auto-select a least-loaded GPU, and isolates Python paths. Set the allowed device visibility explicitly first; automatic selection alone does not enforce the reserved-local-GPU rule below.

## Backend and run isolation

Before pytest collection, unset `PYTHONPATH`, `PYTHONHOME`, `CONDA_PREFIX` and
`VIRTUAL_ENV`; set `PYTHONNOUSERSITE=1`. CPU checks use
`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`. GPU checks use the allocated device
visibility with `JAX_PLATFORMS=cuda,cpu` and verify a GPU is actually the default
backend. RECOVAR also needs a CPU device for host transfers.

Local Della GPU 0 is reserved for other users. Use only idle physical GPUs 1–3
for short development checks, selected by UUID before imports. Slurm allocation
visibility remains authoritative on compute nodes. See
[the cluster runbook](../docs/development/della.md).

Save scores and ledgers under a unique run root, separately from immutable
baselines. Require the declared metric/stage inventory, finite numeric values,
expected executed counts and explicit skip/failure status. Existing partial
function tests declare their metric subset explicitly. A comparison over only
whatever keys survived cannot establish that the whole workload passed.

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LONG_METRICS_OUTPUT_BASE` | pytest tmp_path | Redirect large outputs off home quota |
| `LONG_METRICS_TOL_FRAC` | 0.01 | Allowed relative metric degradation |
| `LONG_METRICS_WRITE_BASELINE` | 0 | Set to 1 to regenerate baselines |
| `CUDA_VISIBLE_DEVICES` | auto-selected | GPU selection |
