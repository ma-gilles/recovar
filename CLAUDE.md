# RECOVAR Development Guide

RECOVAR analyzes conformational heterogeneity in cryo-EM/cryo-ET datasets using regularized covariance estimation and kernel regression. GPU-accelerated via JAX + custom CUDA kernels.

## Engineering Priorities (in order)
1. **Correctness** — new code must match or improve on baseline quality metrics. Add or run tests to verify.
2. **Performance** — optimize for GPU (speed, memory). Batch sizing, half-spectrum layouts, CUDA FFI.
3. **Clarity** — simple, readable code. Small targeted diffs. No drive-by formatting.

## Architecture

```
recovar/
  core/            # Cryo-EM primitives: ForwardModelConfig (Equinox), CTF, slicing, geometry
  data_io/         # Dataset loading: CryoEMDataset, ImageSource, metadata parsing
  heterogeneity/   # Science: covariance estimation, PCA, embedding, kernel regression volumes
  reconstruction/  # Mean reconstruction, noise estimation, regularization
  output/          # Results serialization, metrics (FSC, locres, PCS), plotting
  commands/        # CLI entry points: pipeline.py is the main one
  cuda/            # CUDA backprojection kernel (XLA FFI, auto-compiles)
  simulation/      # Synthetic data generation for testing
  em/              # EM algorithm (E-step, M-step, heterogeneous state)
  gui/             # Flask web interface
  utils/           # Helpers, NVTX profiling, multi-GPU
```

**Data flow**: `CryoEMDataset` → batch iterator → mean reconstruction → covariance estimation → PCA → embedding → kernel regression volumes → output

**Key types**: `ForwardModelConfig` (static Equinox struct, triggers JIT recompilation if changed), `CryoEMDataset` (single entry point for all data), `ModelState` (mean, mask, basis, eigenvalues)

## Build & Setup

**Primary (pixi)** — hermetic env with pinned deps:
```bash
pixi install
pixi run install-recovar       # editable install, no deps
pixi run smoke-import-recovar   # verify import works
```

**Alternative (pip)** — for end users:
```bash
pip install git+https://github.com/scikit-fmm/scikit-fmm.git "jax[cuda12]"==0.9.0.1 recovar
```

CUDA kernels auto-compile on first use. The Makefile uses the running Python to locate JAX FFI headers — always build/test through pixi or the correct Python.

## Testing

See `tests/CLAUDE.md` for full testing rules (critical — read before modifying tests).

| Tier | Command | Time | GPU | Purpose |
|------|---------|------|-----|---------|
| Fast | `pixi run test-fast` | ~30s | No | Unit tests, safe on login node |
| Smoke | `pytest --run-integration tests/integration/test_pipeline_smoke.py` | ~2 min | No | Pipeline sanity |
| Full | `pixi run test-full` | ~2h | Yes | All tests — **submit via Slurm** |
| Long | `pytest --long-test` | 6-12h | Yes | Quality regression — **Slurm only** |
| Parallel | `./scripts/run_tests_parallel.sh long-test` | ~2-3h | Yes | Long tests in parallel Slurm jobs |

## Branching & Commits

- `dev` is the active development branch. `main` is the old public release — do not target it.
- Never commit directly to `dev`. Create feature branches like `claude/<short-task-name>` off `dev`, then push the branch for PR into `dev`.
- Small targeted diffs. Do not commit large artifacts (checkpoints, datasets, binaries).
- Never force-push unless explicitly asked.

## Before Pushing / Creating a PR — MANDATORY

1. Rebase on `dev`: `git fetch origin && git rebase origin/dev`
2. Run the **full long-test suite** via parallel Slurm submission:
   ```bash
   ./scripts/run_tests_parallel.sh long-test
   ```
   This includes unit, smoke, downstream, SPA metrics regression,
   ET metrics regression, outlier regression, PDB trajectory, indices,
   stress tests, and isolated-function tests. **All must pass.**
3. Wait for the summary job to complete. Check the summary log.
4. If any test fails → **do not push**. Fix and resubmit.
5. Only push and create PR after **all tests pass including long-test**.

## PR Description — MANDATORY Format

Every PR description **must** include quality and performance comparison
tables extracted from the long-test results. Run:
```bash
pixi run python scripts/extract_regression_tables.py
```
after long-test completes, and paste the output into the PR body.

### Quality Baselines Table

Compare current scores against `tests/baselines/` JSON files. Include
both SPA and cryo-ET metrics. Higher `svd_relative_variance` is better,
lower errors are better.

```
### Quality Comparison (current vs baseline)
| Metric | Baseline | Current | % Change | Status |
|--------|----------|---------|----------|--------|
| spa_pcs_relative_variance_4 | 0.9234 | 0.9240 | +0.06% ↑ | OK |
| spa_locres_90pct | 9.18 | 9.18 | 0.0% | OK |
| et_pcs_relative_variance_4 | 0.6326 | 0.6330 | +0.06% ↑ | OK |
| outlier_recall_round_1 | 0.992 | 0.992 | 0.0% | OK |
```

### Performance Baselines Table

Compare pipeline/compute_state wall time and GPU memory against
`tests/baselines/.../perf_baseline_*.json`. Lower time is better,
lower memory is better. Flag regressions > 10%.

```
### Performance Comparison (current vs baseline, H100)
| Stage | Baseline (s) | Current (s) | % Change | Status |
|-------|-------------|-------------|----------|--------|
| spa_pipeline | 555.8 | 540.2 | -2.8% ↓ | OK |
| et_pipeline | 1886.7 | 1850.0 | -1.9% ↓ | OK |
| spa_compute_state | 193.0 | 152.0 | -21.2% ↓ | IMPROVED |
| et_compute_state | 145.9 | 140.0 | -4.0% ↓ | OK |
| spa_peak_gpu_gb | 40.4 | 40.4 | 0.0% | OK |
```

Use ↑ for increases, ↓ for decreases. Mark regressions > 10% as **REGRESSED**.

## End of Task

Provide: summary of changes, files modified, test commands run (with Slurm job IDs if relevant), how to reproduce, `git status` and `git diff --stat`.
