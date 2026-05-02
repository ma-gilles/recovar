# RECOVAR Development Guide

RECOVAR analyzes conformational heterogeneity in cryo-EM/cryo-ET datasets using regularized covariance estimation and kernel regression. GPU-accelerated via JAX + custom CUDA kernels.

## Engineering Priorities (in order)
1. **Correctness** — new code must match or improve on baseline quality metrics. Add or run tests to verify.
2. **Performance** — optimize for GPU (speed, memory). Batch sizing, half-spectrum layouts, CUDA FFI.
3. **Clarity** — simple, readable code. Small targeted diffs. No drive-by formatting.

## Code Readability
- Prefer the shortest correct implementation that is easy to audit.
- Do not trade correctness, accuracy, or GPU performance for brevity.
- Replace large inline blocks, long argument lists, and repeated loop setup with small named helpers or simple data containers.
- Remove flags and branches when one clear path should exist. If modes are genuinely different, split them into separate helpers instead of threading conditionals through the hot path.
- Do not add fallbacks or failsafes that hide bugs. Validate assumptions early and fail loudly.
- Keep TODOs until the issue is actually fixed; when resolving one, remove only that TODO and mention the replacement in the summary or commit.

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
  gui_v2/          # Web GUI: FastAPI backend + React/TypeScript frontend
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
pip install "recovar[gpu]"
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

## Worktree & Branch Hygiene — MANDATORY before running tests

This repo has many parallel worktrees (`/scratch/gpfs/GILLES/mg6942/recovar_wt_*`)
and parity work that lives on long-lived branches not yet merged to `dev`.
Worktree directory names lie: a directory named `recovar_wt_parity_branch_*`
may have been re-checked out to an unrelated branch by a previous session.
Mixing up branches has cost days of debugging — apply these rules every time:

**Before running ANY test or benchmark in a worktree, always print:**
```bash
git -C <worktree> rev-parse HEAD          # commit you're actually on
git -C <worktree> symbolic-ref --short HEAD || echo "<detached>"
git -C <worktree> status --porcelain      # uncommitted state
python -c "import recovar; print(recovar.__file__)"  # confirm import path
```
The provenance gate must show: `recovar.__file__` inside the worktree, JAX
inside `.pixi/envs/default`, and a recognized commit/branch.

**Cite commits, not branches, in memory and reports.** Branch tips move
constantly; commit hashes are immutable. `claude/relion-parity-flag-audit
at e120cdfc` is reproducible. `the parity branch` is not.

**For parity work specifically**, `scripts/run_multi_iter_parity.py` now
prints a provenance banner and asserts known load-bearing parity commits
are in HEAD's ancestry. If it exits with code 2, the worktree is missing a
required fix and any "broken parity" result is not a regression — it's the
wrong branch. Switch branches before debugging deeper.

**Pre-PR parity smoke** (3 minutes, machine-precision check):
```bash
pixi run python scripts/run_multi_iter_parity.py \
  --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
  --data_star  /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star \
  --iter 3 --max_iter 1 \
  --gt_volume /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/reference_gt.mrc \
  --output_dir /tmp/parity_check
```
Pass criterion: `Final half1/half2 vs RELION it004 corr ≥ 0.999`. Anything
below that means kernel parity has regressed. Run before any push that
touches `recovar/em/`.

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
