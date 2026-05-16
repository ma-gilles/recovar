# EM/VDAM Post-Merge Regression Guard

This checklist protects the InitialModel/VDAM parity work on
`codex/vdam-postmerge-20260507` while merging related EM, VDAM, and PPCA
refinement branches.

## Scope

The guard is intentionally EM-scoped. Do not substitute the project-wide
SPA/ET long-test unless the merge crosses shared non-EM behavior.

Protected improvements:

- RELION-equivalent native InitialModel bootstrap Fourier ordering.
- VDAM class and direction priors, including direction-prior model STAR output.
- Sigma-offset and sigma2-noise updates with RELION/VDAM momentum.
- K-class sparse pass-2 joint class/pose behavior.
- Native InitialModel K=1 quality parity and K=2 map/scalar parity.

## Mandatory Commands

Run from the merged worktree with pixi, no conda environment:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu

.pixi/envs/default/bin/python -m pytest -q \
  tests/unit/initial_model/test_dense_adapter.py \
  tests/unit/initial_model/test_iteration_loop.py \
  tests/unit/initial_model/test_init_and_estep.py \
  tests/unit/initial_model/test_native_driver.py \
  tests/unit/test_k_class_joint_semantics.py \
  tests/unit/initial_model/test_vdam_abinitio_merge_guard.py
```

Then submit the Slurm merge guard:

```bash
./scripts/run_em_merge_guard_slurm.sh --watch
```

For final pre-merge quality/performance signoff, submit the EM-long guard:

```bash
./scripts/run_em_parity_long_slurm.sh --watch
pixi run python scripts/extract_em_parity_tables.py --tier long
```

## Acceptance Floors

Fast/unit guard:

- Focused InitialModel unit slice passes.
- `test_vdam_abinitio_merge_guard.py` passes.
- `run_em_merge_guard_slurm.sh` summary job completes.

K=1 native InitialModel parity observed on this branch:

- Native aligned GT corr: `0.521194`.
- RELION aligned GT corr: `0.521777`.
- Native must stay within `0.05` corr-vs-GT of the RELION InitialModel reference
  in the EM-long guard.
- Direct native-vs-RELION iter-8 same-frame map corr must stay `>= 0.999` in
  the EM-long guard.

K=2 native InitialModel parity observed on this branch:

- Iteration 1 mean map corr vs RELION: `0.999860`.
- Iteration 2 mean map corr vs RELION: `0.999825`.
- Iteration 2 class distribution: native `[0.369262, 0.630738]`, RELION
  `[0.365342, 0.634658]`.
- Iteration 2 sigma offset: native `4.533 A`, RELION `4.617 A`.
- Iteration 2 `pmax_abs_mean`: `0.056253`. This is the remaining known gap;
  a merge must not make map/scalar parity worse while this is being closed.

Performance reference from the same K=2 run:

- Native K=2 two-iteration wall time: `8:10.72`, max RSS `8678072 KB`.
- RELION K=2 two-iteration reference wall time: `0:15.11`, max RSS `489536 KB`.
- Treat native wall-time or max-RSS regressions over `10%` as blocking unless
  accompanied by a deliberate quality improvement.

## Saved Reference Runs

- K=1 native:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/codex_k1_5k_bootstrapfix_20260508_014405_1904`
- K=1 evaluation:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/codex_k1_eval_current_20260508_033608_2016994`
- K=2 native:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/codex_k2_noisemomentum_20260508_055652_27008`
- K=2 RELION reference:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/codex_k2_relion_write1_20260508_044635_6755/relion`

## Failure Policy

If any guard fails, do not push the merged branch. Save the command, Slurm job
IDs, log paths, `git status`, and `git diff --stat`, then fix the regression
before rerunning the same guard.
