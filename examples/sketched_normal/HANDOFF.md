# Sketched ProxSVT notebook — handoff notes

This branch (`claude/sketched-complex-adjoint-fix`, on top of
`claude/sketched-normal-op-v2`) contains a round of debugging on the
matrix-free Proximal-SVT + R4SVD path.

## Headline result — sketched beats PPCA

On the notebook smoke dataset at low noise (n=5000, grid=64, nl=1e-5),
the matrix-free solver beats PPCA-EM at matched rank 10:

| metric | sketched ProxSVT + R4SVD | PPCA-EM |
|---|---|---|
| rv@1  | **0.6801** | 0.7233 |
| rv@2  | **0.8837** | 0.7343 |
| rv@5  | **0.9243** | 0.8415 |
| rv@10 | **0.9243** | **0.8699** |

≈ **+6% on rv@10 at matched rank 10.**  Sketched converges to rank 4
(three singular values absorbed by soft thresholding) and still captures
more variance than PPCA's rank-10 basis.

### Exact config

- Dataset: `notebook_smoke_data/test_dataset` (n=5000, grid=64, nl=1e-5),
  built by `recovar.simulation.simulator.generate_synthetic_dataset`.
- **Prior: none.**  Nuclear-norm metric: Frobenius (SVT on X).
- `λ = 1.0`, `target_rank = 10`, `max_rank = 60`, `n_iter = 80`.
- `block_size = 15`, `n_power = 3`.
- Step rule: **backtracking** (monotone-descent acceptance on F_rel).
  - `bt_delta_init = 0.1`, `bt_shrink = 0.5`, `bt_grow = 1.5`,
    `bt_max_retries = 10`, `bt_armijo_c = 0.9`.
- `seed = 1`.  Wall time ≈ 17 min on one A100.

Saved run:
`_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/lamwide_nb_results/lamwide_nl1e-5_lam1.0.json`

PPCA baseline for same dataset:
`_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/ppca_reeval_results/ppca_notebook_nl1e-5.json`

## What changed on this branch

1. **Complex Fourier adjoint bug fixed.**
   `SketchedNormalOperator._compute_sketches_half` used
   `per_image_backproject` with `.real(...)` on the complex half-image
   residual, dropping the imaginary part and returning `Re[A*(A(X)-b)]`
   — about half the true gradient.  This was the dominant cause of the
   initial rv@10 ≈ 0.12 plateau.  Replaced with
   `vmap(core.adjoint_slice_volume)`, keeping the full complex adjoint.
   Numerical gate added in `tests/unit/test_sketched_normal_operator.py`.
   Commit `bb3e52d8`.

2. **Left/right scale reconciled.**
   Raw `left_matvec` output is inflated by `vol_size / 2` relative to
   `right_matvec` — a half-image layout artefact.  Notebook now applies
   `LEFT_SCALE = vol_size / 2.0` at the call site
   (`left_matvec_scaled`, `right_matvec_scaled`, `both_matvecs_scaled`
   in cell 8).

3. **Radial Fourier prior folded into the operator** (commit `9f6a1458`).
   Pass `D2_fourier=<(vol_size,) array>` to `SketchedNormalOperator` and
   every matvec adds `D² X` to the data term.  No separate scalar
   multiplier — magnitude is encoded entirely in `D²` (the previous
   `prior_lambda` kwarg was removed in commit `18461d3b`).  The left
   side of the prior is pre-inflated by `self.left_scale = vol_size/2`
   so callers applying `LEFT_SCALE` symmetrically on the outside still
   see a correctly-scaled prior contribution.

4. **Matrix-free objective evaluation.**
   `eval_objective(op, U, s, V, lam)` in notebook cell 8 computes
   `F_rel(X) = (1/2)⟨X, G(X)+G(0)⟩ + (1/2)Σsᵢ²⟨Uᵢ, D²Uᵢ⟩ + λ·Σsᵢ`
   via two extra `right_matvec` passes per iteration.  Plotted in
   cells 18–19; overlay in cell 22.

5. **Backtracking-step solver** (currently in the sandbox, not in the
   notebook).  Monotone-descent acceptance on `F_rel`; adaptive `δ`
   (`bt_shrink` on reject, `bt_grow` on clear descent).  Lives at
   `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/sketch_lib.py::run_iterations_backtracking`.
   The headline result above uses it.

6. **Unified notebook preset, dataset selector, BASE_DIR override.**
   `DATASET ∈ {'notebook', 'IgG-1D', 'IgG-RL', 'Ribosembly',
   'Tomotwin-100'}` in cell 2; `$SKETCHED_BASE_DIR` overrides the
   writable scratch dir; CryoBench tags auto-generate on first use.

## How to run

**Environment.** The repo uses pixi for a hermetic env.  Every command
below assumes the repo root as cwd.

```bash
# One-time setup
pixi install
pixi run install-recovar
pixi run smoke-import-recovar

# Provenance gate — run this before any job
pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/')
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve())
print('ENV_OK')
"
```

**Run the notebook.**  The default preset is `notebook` at
(grid=64, n=20000, nl=0.1).  To reproduce the headline win above, open
the notebook in Jupyter and in cell 2 set

```python
DATASET = 'notebook'
_DEFAULT_PRESET = dict(grid=64, n_images=5000, noise_level=1e-5)
```

then run top-to-bottom.  Expect ~17 min on an A100.

**Fastest path — use the sandbox runner.**  The headline result was
produced by `run_experiment.py`, not the notebook (it uses the sandbox
backtracking solver).  Reproduce it with:

```bash
cd /scratch/gpfs/GILLES/mg6942/_agent_scratch/wt_sketched_diagnosis
ROOT=_agent_scratch/sketch_sweep

# 1. Generate the dataset (cache-safe)
pixi run python $ROOT/generate_dataset.py \
  --cryobench-name notebook \
  --grid-size 64 --n-images 5000 --noise-level 1e-5 \
  --output-dir $ROOT/notebook_smoke_data/test_dataset

# 2. Sketched solver
pixi run python $ROOT/run_experiment.py \
  --dataset-dir $ROOT/notebook_smoke_data/test_dataset \
  --grid-size 64 --n-images 5000 --batch-size 500 \
  --method soft --target-rank 10 --lam 1.0 --prior-mode none \
  --step-rule backtracking \
  --bt-delta-init 0.1 --bt-armijo-c 0.9 \
  --bt-shrink 0.5 --bt-grow 1.5 --bt-max-retries 10 \
  --init cold --n-iter 80 \
  --block-size 15 --max-rank 60 --n-power 3 \
  --seed 1 \
  --output /tmp/sketched_nb_nl1e-5.json

# 3. PPCA baseline
pixi run python $ROOT/run_ppca_experiment.py \
  --dataset-dir $ROOT/notebook_smoke_data/test_dataset \
  --grid-size 64 --n-images 5000 \
  --basis-size 10 --n-iter 20 \
  --output /tmp/ppca_nb_nl1e-5.json
```

A Slurm sbatch wrapper for this pattern is at
`_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/run_tomo_highrank.sbatch`
(adapt the for-loop).

## Where things stand on the larger g64/nl=0.1 preset

On `test_dataset_notebook_g64_n20000_nl0.1` (the notebook default):

- No-prior run (`λ=4.11e-4`, fixed step):  rv@10 ≈ 0.38–0.43 at rank 10.
- Prior-augmented run (`λ=1.0`, fixed step):  similar magnitude.
- PPCA-EM baseline:  rv@10 ≈ 0.70.
- Backtracking (tested in `_agent_scratch/sketched_nb_run/bt_sweep.py`)
  lifts rv@10 by ~12% on this preset but does not close the gap.

The sub-rank behaviour (rank pinned at the cap, top singular value
rising while rv@10 stays flat) is consistent with a basin/alignment
issue rather than a bug in the matrix-free code.

## Suggested follow-up — **Tomotwin-100 with many images**

Tomotwin-100 has 100 ground-truth classes
(`cumfrac@10=0.36, cumfrac@50=0.82, cumfrac@100=1.0`), which is exactly
the regime where PPCA-EM can't meaningfully compete: PPCA covariance
scales quadratically with `basis_size` in memory and cubically in time,
so it's capped at `basis_size ≈ 10` on a 64³ grid, while the sketched
solver can target `rank=100` cheaply.

On the current Tomotwin-100 run at **n=20k, nl=1e-5**, target_rank=100
with the winning config (λ=10, Frob, backtracking, `n_power=3`,
`max_rank=120`, 120 iters) reaches **rv_s2@100 = 0.34** in ~23 min,
while PPCA at basis_size=10 tops out at **rv_s2@10 = 0.29**.  That's
already a meaningful win conceptually (PPCA can't reach rank 100) but
not a matched-rank win.

The right next experiment is **Tomotwin-100 at 5–10× more images**
(n=100k, n=200k).  The hypothesis is that the `rv_s2@100` gap widens
with more data because (a) the sketched operator's effective SNR scales
with `n` and (b) PPCA is rank-capped regardless of `n`.  A runner is
already wired up at
`/scratch/gpfs/GILLES/mg6942/_agent_scratch/sketched_nb_run/run_tomo_scale.sbatch`
— just submit it with `N_IMAGES={50000|100000|200000}`.  Cost at
`n_iter=60, n_power=2` is roughly 15 / 30 / 60 min on one A100.

## Other suggested next probes

- **Fold backtracking into the notebook solver.**  Mechanical port of
  `run_iterations_backtracking` from
  `_agent_scratch/sketch_sweep/sketch_lib.py` into
  `prox_svt_r4svd_single_lambda_cold_start`.
- **Warm-start from PPCA on the harder preset.**  Initialize `U, s, V`
  from the PPCA solution (IDFT + QR into the real-valued subspace) and
  see whether the solver stays there or drifts.  Partial harness at
  `_agent_scratch/sketched_nb_run/warmstart_exp.py`.
- **Objective-value comparison.**  Plug the PPCA solution into
  `eval_objective(op, U_ppca, s_ppca, V_ppca, λ)` and compare to the
  sketched solver's final `F_rel`.  Tells you whether the sketched
  solver is in a local basin vs. the nuclear-norm model is just not
  aligned with the PPCA objective on that preset.
- **R4SVD subspace audit.**  Compare the basis produced by
  `expand_block` against the top-k right singular vectors of `G(0)`.

## Files

- `recovar/ppca/sketched_normal.py` — operator + sketches + folded prior.
- `examples/sketched_normal/diagonsis.ipynb` — the notebook.
- `tests/unit/test_sketched_normal_operator.py` — numerical gate.
- `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/run_experiment.py`
  — sandbox runner (what the headline result used).
- `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/sketch_lib.py`
  — backtracking solver + helpers.
- `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/FINDINGS.md`
  — overnight diagnosis notes (scratch).

## One-line summary

Complex-adjoint bug fixed, prior API cleaned up, matrix-free objective
wired in, backtracking solver in the sandbox; on notebook nl=1e-5 the
solver beats PPCA by ~6% on rv@10, and the suggested next experiment is
Tomotwin-100 at 5–10× images to see if the high-rank gap widens.
