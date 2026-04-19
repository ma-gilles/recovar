# Sketched ProxSVT notebook — handoff notes

This branch (`claude/sketched-complex-adjoint-fix`, on top of
`claude/sketched-normal-op-v2`) contains a round of debugging on the
matrix-free Proximal-SVT + R4SVD path.  This file records **what changed**,
**why**, and **where the solver currently is** so the next person can pick
up without re-deriving context.

## What was broken coming in

The sketched solver plateaued far below PPCA on the notebook benchmark
(`test_dataset_notebook_g64_n20000_nl0.1`): `rv@10 ≈ 0.12` while PPCA-EM
gave `rv@10 ≈ 0.70` on the same preset.

Two root causes were found:

### 1. Complex Fourier adjoint was being silenced

`SketchedNormalOperator._compute_sketches_half` used
`per_image_backproject` with `.real(...)` on the complex Hermitian
half-image residual, which dropped the imaginary part and effectively
returned `Re[A*(A(X) - b)]` — about half the true gradient.  This was the
dominant cause of the rv@10 ≈ 0.12 plateau.

Fix: commit `bb3e52d8` replaces the per-image path with
`vmap(core.adjoint_slice_volume)`, keeping the full complex adjoint.
Benchmarked against two alternatives at grid=128, n=20k, block ∈ {15..200}
on an A100; vmap+GEMM is correct and roughly block-independent.  A
numerical-gate test was added in
`tests/unit/test_sketched_normal_operator.py`.

### 2. Left/right scale mismatch

The raw `left_matvec` output is inflated by `vol_size / 2` relative to
`right_matvec` — an artefact of the half-image layout.  The notebook now
divides `left_matvec` (and the `both_matvecs` left output) by
`LEFT_SCALE = vol_size / 2.0` at the call site (see `left_matvec_scaled`,
`right_matvec_scaled`, `both_matvecs_scaled` in cell 8).  After this
division the left and right products agree to numerical tolerance.

## What the operator itself does now

Commit `9f6a1458` folds the radial Fourier prior
`R(X) = (1/2)||D X||_F^2` **into** `SketchedNormalOperator`: pass
`D2_fourier=<(vol_size,) array>` and every matvec adds the analytic prior
gradient `D^2 X` to the data term.

The prior now has **no** scalar multiplier — its magnitude is encoded
entirely in `D^2`.  The previous `prior_lambda` kwarg was removed (this is
in the uncommitted sketched_normal.py diff; commit it before merging).
Rationale: stacking a tunable λ_prior on top of an already-tuned radial
weighting was adding a second knob for one physical quantity and caused
debugging drift.

The left side of the prior is pre-inflated by `self.left_scale = vol_size/2`
inside the operator so that callers who apply `LEFT_SCALE` normalization
symmetrically on the outside still see a correctly-scaled prior
contribution.

## Notebook structure (diagonsis.ipynb)

Cells, top to bottom:

| # | What it does |
|---|---|
| 0  | Title + overview |
| 1-2 | Imports, config (`DATASET`, `GRID_SIZE`, `N_IMAGES`, `NOISE_LEVEL`) |
| 3-6 | Dataset load (supports notebook + CryoBench tags), GT states, GT eigenvectors |
| 7-8 | Solver + matrix-free `eval_objective` helpers |
| 9-10 | Build `SketchedNormalOperator` (no prior) + run PPCA-EM baseline |
| 11-13 | Cold-start single-λ ProxSVT+R4SVD run (no prior); evaluate vs GT |
| 14-16 | Build radial Fourier prior from `regularization`; rerun solver with `op_prior` |
| 17 | Iteration-history table for the no-prior run |
| 18-19 | Objective-trajectory plots (no-prior and prior-augmented) |
| 20 | No-prior cold-start summary |
| 21-22 | Comparison + overlay plots (PPCA vs sketched-no-prior vs sketched-prior) |

The **objective evaluation is matrix-free**: `eval_objective(op, U, s, V, lam)`
in cell 8 computes

```
F_rel(X) = (1/2) ⟨X, G(X) + G(0)⟩
         + (1/2) Σ sᵢ² ⟨Uᵢ, D² Uᵢ⟩        # 0 when D² not attached
         +  λ · Σ sᵢ
```

via two extra `right_matvec` passes per iteration (one on `U,s,V` with
`Q=V`, one on empty factors with `Q=V`).  This is F(X) − F(0), since F(0)
is a constant.

## Where the solver currently stands

On `test_dataset_notebook_g64_n20000_nl0.1`, cold-start single-λ with the
best fixed-step settings we found:

- no-prior run: `λ=4.11e-4`, cold start, fixed `STEP_SIZE`, R4SVD config in
  cell 2 → `rv@10 ≈ 0.38–0.43`, rank 10.  Activation window confirmed at
  this λ.
- prior-augmented run: `λ=1.0` (the operator scale changes when D² is
  added, so λ retunes).  Results similar to the no-prior run.
- PPCA-EM baseline on the same preset: `rv@10 ≈ 0.70`.

### Gap analysis (what we verified)

- Step-size invariance at cold start: rv@10 is determined by
  `λ/‖G(0)‖`, where `‖G(0)‖ ≈ 7.69e-3` on this preset.  Sweeping `step ∈
  {1, 100, 1000}` gave the same rv@10 once λ was held equal up to that
  ratio.
- Backtracking (monotone-descent acceptance on `F_rel`) was implemented as
  a separate script under `_agent_scratch/sketched_nb_run/bt_sweep.py` to
  test whether a fixed step was under-stepping; it lifted rv@10 by ~12%
  over the best fixed-step result but did not close the gap to PPCA on
  this preset.  Backtracking is **not** folded into the notebook —
  `prox_svt_r4svd_single_lambda_cold_start` still uses a fixed step.
- One dataset/SNR **does** flip the ordering: on `Tomotwin-100 g64 nl=1e-5`
  with `target_rank=100, max_rank=120, block_size=20, n_power=3, λ=10.0,
  backtracking, 120 iters`, the sketched solver reached `rv_s2@100 ≈ 0.34`
  vs PPCA-EM `rv_s2@10 ≈ 0.29` on the apples-to-apples metric.  That run
  is saved at
  `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/tomo_highrank_results/hirank_tomo_nl1e-5_lam10.0.json`.

### What's still open

On the notebook preset the rv@10 gap to PPCA is **not closed**.  The
sub-rank behaviour (rank pinned at the cap, top-singular-value rising
while rv@10 stays flat) is consistent with a basin/alignment problem
rather than a bug in the matrix-free implementation: the R4SVD-expanded
subspace is simply not the PPCA subspace.  Suggested next probes
(unexplored here):

- **Warm-start from PPCA**: initialize `U, s, V` from the PPCA solution
  (projected into the real-valued subspace via IDFT + QR) and see whether
  the solver stays there or drifts off.  A partial harness exists at
  `_agent_scratch/sketched_nb_run/warmstart_exp.py`; it was not finished.
- **Objective-value comparison**: plug the PPCA solution into
  `eval_objective(op_prior, U_ppca, s_ppca, V_ppca, λ)` and compare the
  value to the sketched solver's final `F_rel`.  If PPCA has a strictly
  lower F_rel, the sketched solver is stuck in a local basin; if they are
  comparable, the nuclear-norm model is simply not well-aligned with the
  PPCA objective on this preset.
- **R4SVD subspace audit**: compare the span of the basis produced by
  `expand_block` against the top-k right singular vectors of G(0).  If
  they disagree, the randomized expansion is under-powered.

## Files to look at

- `recovar/ppca/sketched_normal.py` — operator + sketches + folded prior.
- `examples/sketched_normal/diagonsis.ipynb` — the notebook.
- `tests/unit/test_sketched_normal_operator.py` — numerical gate on the
  complex-adjoint fix.
- `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/FINDINGS.md`
  — prior overnight diagnosis notes (scratch; safe to delete).

## Summary

- Complex-adjoint bug: fixed.
- Prior API: folded into operator, scale-consistent.
- Objective: matrix-free eval wired in and plotted.
- Notebook preset rv@10 gap to PPCA: narrowed from ≈0.12 → ≈0.38–0.43, but
  not closed.  The next meaningful step is probably an objective-value
  comparison against a PPCA warm-start, not more step-size tuning.
