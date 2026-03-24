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
- Before pushing: rebase on `dev`, run full test suite.
- Small targeted diffs. Do not commit large artifacts (checkpoints, datasets, binaries).
- Never force-push unless explicitly asked.

## End of Task

Provide: summary of changes, files modified, test commands run (with Slurm job IDs if relevant), how to reproduce, `git status` and `git diff --stat`.
