# Sketched ProxSVT notebook — handoff notes

This branch (`claude/sketched-complex-adjoint-fix`, on top of
`claude/sketched-normal-op-v2`) contains a round of debugging on the
matrix-free Proximal-SVT + R4SVD path. This file records **what changed**,
**why**, and **where the solver currently is** so the next person can pick
up without re-deriving context.

## Headline result

On the notebook smoke dataset at low noise (n=5000, grid=64, nl=1e-5), the
matrix-free solver **beats PPCA-EM** on rv@10 at matched rank 10:

| metric | sketched ProxSVT + R4SVD | PPCA-EM |
|---|---|---|
| rv@1  | **0.6801** | 0.7233 |
| rv@2  | **0.8837** | 0.7343 |
| rv@5  | **0.9243** | 0.8415 |
| rv@10 | **0.9243** | **0.8699** |

~**+6% on rv@10 at matched rank 10**.  The solver converges to rank 4
(three singular values absorbed by soft thresholding) while still
capturing more variance than PPCA's rank-10 basis.

### Exact config that produced this

- Dataset: `notebook_smoke_data/test_dataset` (n=5000, grid=64, nl=1e-5),
  built by `recovar.simulation.simulator.generate_synthetic_dataset`
- **Prior: none** (no D² radial Fourier prior)
- Nuclear-norm metric: Frobenius (SVT on X)
- `λ = 1.0`, `target_rank = 10`, `max_rank = 60`, `n_iter = 80`
- `block_size = 15`, `n_power = 3`
- Step rule: **backtracking** (monotone-descent acceptance on F_rel)
  - `bt_delta_init = 0.1`, `bt_shrink = 0.5`, `bt_grow = 1.5`,
    `bt_max_retries = 10`, `bt_armijo_c = 0.9`
- Seed = 1. Wall time: ~17 min.

Saved run:
`_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/lamwide_nb_results/lamwide_nl1e-5_lam1.0.json`

PPCA baseline for same dataset:
`_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/ppca_reeval_results/ppca_notebook_nl1e-5.json`

The backtracking harness used for this run lives at
`_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/run_experiment.py`.
The notebook itself still uses the fixed-step solver
(`prox_svt_r4svd_single_lambda_cold_start`); folding backtracking into
the notebook is the first obvious follow-up.

## Bugs found & fixed coming in

### 1. Complex Fourier adjoint was being silenced

`SketchedNormalOperator._compute_sketches_half` used
`per_image_backproject` with `.real(...)` on the complex Hermitian
half-image residual, dropping the imaginary part and returning
`Re[A*(A(X) - b)]` — about half the true gradient.  This was the dominant
cause of the initial rv@10 ≈ 0.12 plateau.

Fix: commit `bb3e52d8` replaces the per-image path with
`vmap(core.adjoint_slice_volume)`, keeping the full complex adjoint.
Numerical gate added in `tests/unit/test_sketched_normal_operator.py`.

### 2. Left/right scale mismatch

The raw `left_matvec` output is inflated by `vol_size / 2` relative to
`right_matvec` — an artefact of the half-image layout.  The notebook now
divides `left_matvec` (and the `both_matvecs` left output) by
`LEFT_SCALE = vol_size / 2.0` at the call site (see `left_matvec_scaled`,
`right_matvec_scaled`, `both_matvecs_scaled` in cell 8).

Empirically the correct factor is ~8–10% below `vol_size/2` on individual
datasets (`ref_val = 131072`, measured ≈ 120k on notebook / 123k on
ribo), so a residual trace-identity error of 3% / 10% remains.  This is
small on notebook but worth revisiting on harder datasets.

## Operator-level API changes

Commit `9f6a1458` folds the radial Fourier prior
`R(X) = (1/2)||D X||_F^2` **into** `SketchedNormalOperator`: pass
`D2_fourier=<(vol_size,) array>` and every matvec adds the analytic prior
gradient `D^2 X` to the data term.

The prior has **no** scalar multiplier — its magnitude is encoded
entirely in `D^2`.  The previous `prior_lambda` kwarg was removed (commit
`18461d3b`).  Rationale: stacking a tunable λ_prior on top of an
already-tuned radial weighting was adding a second knob for one physical
quantity and caused debugging drift.

The left side of the prior is pre-inflated by
`self.left_scale = vol_size/2` inside the operator so callers who apply
`LEFT_SCALE` symmetrically on the outside still see a correctly-scaled
prior contribution.

## Notebook structure (examples/sketched_normal/diagonsis.ipynb)

| # | What it does |
|---|---|
| 0  | Title + overview |
| 1-2 | Imports, config (`DATASET`, `GRID_SIZE`, `N_IMAGES`, `NOISE_LEVEL`) |
| 3-6 | Dataset load (`'notebook'` + CryoBench tags), GT states, GT eigenvectors |
| 7-8 | Solver + matrix-free `eval_objective` helpers |
| 9-10 | Build `SketchedNormalOperator` (no prior) + PPCA-EM baseline |
| 11-13 | Cold-start single-λ ProxSVT + R4SVD (no prior); evaluate vs GT |
| 14-16 | Build radial Fourier prior; rerun solver with `op_prior` |
| 17 | Iteration-history table for the no-prior run |
| 18-19 | Objective-trajectory plots (no-prior / prior-augmented) |
| 20 | No-prior cold-start summary |
| 21-22 | Comparison + overlay plots |

The **objective evaluation is matrix-free**: `eval_objective(op, U, s, V, lam)`
in cell 8 computes

```
F_rel(X) = (1/2) ⟨X, G(X) + G(0)⟩
         + (1/2) Σ sᵢ² ⟨Uᵢ, D² Uᵢ⟩        (0 when D² not attached)
         +  λ · Σ sᵢ
```

via two extra `right_matvec` passes per iteration (one on `U,s,V` with
`Q=V`, one on empty factors with `Q=V`).  This is F(X) − F(0), since F(0)
is a constant.

## Where the solver stands on the larger g64/nl=0.1 preset

On `test_dataset_notebook_g64_n20000_nl0.1` (the unified preset used by
the notebook default path), cold-start single-λ with fixed step:

- no-prior run: `λ=4.11e-4`, rv@10 ≈ 0.38–0.43 at rank 10
- prior-augmented run: `λ=1.0`, similar magnitude
- PPCA-EM baseline: rv@10 ≈ 0.70

Backtracking (tested in scratch: `_agent_scratch/sketched_nb_run/bt_sweep.py`)
lifts rv@10 by ~12% over the best fixed step on this preset but does not
close the gap to PPCA.  The sub-rank behaviour (rank pinned at the cap,
top singular value rising while rv@10 stays flat) is consistent with a
basin/alignment problem rather than a bug in the matrix-free code.

## Suggested next steps

1. **Fold backtracking into the notebook solver.**  The headline win
   above used backtracking; the notebook still uses a fixed step.  This
   is a mechanical port of `solve_backtracking` from
   `_agent_scratch/sketched_nb_run/bt_sweep.py`.
2. **Warm-start from PPCA on the harder preset.**  Initialize `U, s, V`
   from the PPCA solution (IDFT + QR into the real-valued subspace) and
   see whether the solver stays there, drifts, or improves.  A partial
   harness exists at `_agent_scratch/sketched_nb_run/warmstart_exp.py`.
3. **Objective-value comparison.**  Plug the PPCA solution into
   `eval_objective(op, U_ppca, s_ppca, V_ppca, λ)` and compare to the
   sketched solver's final `F_rel`.  If PPCA has a strictly lower
   `F_rel`, the sketched solver is in a local basin; if comparable, the
   nuclear-norm model is simply not aligned with the PPCA objective on
   that preset.
4. **R4SVD subspace audit.**  Compare the span of the basis produced by
   `expand_block` against the top-k right singular vectors of `G(0)`.

## Files

- `recovar/ppca/sketched_normal.py` — operator + sketches + folded prior.
- `examples/sketched_normal/diagonsis.ipynb` — the notebook.
- `tests/unit/test_sketched_normal_operator.py` — numerical gate on the
  complex-adjoint fix.
- `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/FINDINGS.md`
  — overnight diagnosis notes (scratch).
- `_agent_scratch/wt_sketched_diagnosis/_agent_scratch/sketch_sweep/run_experiment.py`
  — the runner that produced the headline result.

## One-line summary

Complex-adjoint bug fixed, prior API cleaned up, matrix-free objective
wired in; on notebook nl=1e-5 the solver now beats PPCA by ~6% on rv@10;
the g64/nl=0.1 preset still has a ~0.27 gap to PPCA that looks like a
basin-alignment issue, not an implementation bug.
