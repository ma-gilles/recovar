# `ppca_refinement` — Reading Guide

> Goal: get a complete mental model of this package in <30 minutes.
>
> The package implements **one EM iteration** of probabilistic PCA over a
> cryo-EM particle stack, plus a multi-iteration loop on top. The math is
> in `engine.py`; everything else is plumbing.

---

## 1. The math, in one diagram

```
      ┌─────────────────┐
      │ sufficient stats│       ← rhs (P=q+1, n_voxels)
      │  (rhs, lhs_tri) │         lhs_tri (tri(P), n_voxels)
      └────────▲────────┘
               │
   ┌───────────┴────────────┐
   │ E-step: pose marginal  │   ← per image, posterior over (R,t)
   │ for every image accum  │     produces α_aug = E[z_aug],
   │ α_aug, G_aug into vols │              G_aug = E[z_aug z_aug^T]
   └───────────▲────────────┘
               │ images, μ, W
       ┌───────┴────────┐
       │ dataset +      │
       │ forward model  │
       └────────────────┘

then:

      ┌─────────────────┐
      │ augmented       │       ← per-voxel (P+1)×(P+1) linear solve for
      │ M-step solve    │         joint [μ, W] from rhs/lhs_tri
      │ → μ_new, W_new  │
      └────────▲────────┘
               │
      ┌────────┴────────┐
      │ postprocess     │       ← optional mask + grid correction
      │  (heuristic)    │
      └─────────────────┘
```

**Two flavors of E-step exist:**
- **Dense** — every image scored against a full HEALPix × translation grid
  (sparse-pass2 culls candidates between passes 1 and 2).
- **Exact-local** — every image scored against its own short candidate list
  (`LocalHypothesisLayout`).

Both feed the **same** M-step + postprocess.

A **halfset wrapper** runs the iteration twice (halfset 0 + halfset 1) and
FSC-combines for gold-standard scoring.

A **multi-iteration loop** (`refinement_loop.py`) stitches several iterations
together with schedule decisions (`current_size`, HEALPix order, when to
switch from single-set to halfset).

---

## 2. Reading order (~30 minutes)

| # | File | What you get | Time |
|---|---|---|---|
| 1 | `README.md` (this) | Math sketch + map | 5 min |
| 2 | `config.py` | Every tunable knob lives here. Read once. | 3 min |
| 3 | `state.py` | `PoseMarginalPPCAEMState` — what the loop carries between iterations. | 2 min |
| 4 | `engine.py` | The JIT-compiled E+M kernel. The hard math is in `fused_dense_pose_ppca_block` and (in `recovar/ppca/pose_marginal.py`) `compute_ppca_pose_scores_and_moments_no_contrast`. | 10 min |
| 5 | `dense_dataset.py` (or `local_dataset.py`) | Turns a `CryoEMDataset` into the blocks `engine.py` consumes, then loops over them and assembles the M-step input. | 5 min |
| 6 | `refinement_loop.py` | Multi-iteration driver: schedule, halfset gating, FSC checkpoints. | 5 min |

---

## 3. Files in this package

| File | What lives there |
|---|---|
| `__init__.py` | Public re-exports. |
| `config.py` | `GeometryConfig`, `ScheduleConfig`, `ScoringConfig`, `SparsePass2Config` plus re-exported `MeanRegularizationConfig` and `PostprocessConfig`. **Every tunable parameter lives here.** |
| `state.py` | `PoseMarginalPPCAEMState` — pytree carried between iterations (μ, W, priors, schedule state). |
| `schedule.py` | `PPCARefinementScheduleState` and the halfset-resolution gating decision. |
| `mean_regularization.py` | `MeanRegularizationConfig`, `resolve_mean_precision`, RELION/K-class tau filter helpers. The mean row of the augmented system gets RELION-style tau regularization; the W rows keep the variance-style prior. |
| `postprocess.py` | `PostprocessConfig` + `postprocess_ppca_half_volumes`. RELION-style soft mask + grid correction applied after the M-step. Marked clearly as a heuristic; the long-term target is masked-PCG. |
| `diagnostics.py` | `build_iteration_diagnostics`, `resolve_image_scale_range`. Single home for the diagnostic-dict contract. |
| `initialization.py` | `initialize_ppca_from_gt_volumes`, `loading_row_norm_variance_prior`, `volume_power_variance_prior` etc. |
| `engine.py` | JIT-compiled E+M kernel: `fused_dense_pose_ppca_block`, `dense_pose_ppca_score_stats_blocked`, `run_dense_ppca_fused_refinement_blocks`. Shared by both flavors. |
| `dense_dataset.py` | Dataset-facing dense-flavor iteration (`run_dense_ppca_fused_em_iteration`) + halfset wrapper + the dense block iterator. |
| `local_dataset.py` | Dataset-facing exact-local-flavor iteration (`run_local_ppca_fused_em_iteration`) + halfset wrapper + the per-image bucket iterator. |
| `refinement_loop.py` | `run_dense_ppca_refinement_loop` and `run_local_ppca_refinement_loop` — multi-iteration drivers. |
| `fixture_validation.py` | Test fixtures shared with `tests/unit/ppca_refinement/`. |

---

## 4. Public entry points

```python
from recovar.em.ppca_refinement import (
    # Single iteration
    run_dense_ppca_fused_em_iteration,         # dense flavor
    run_local_ppca_fused_em_iteration,         # exact-local flavor
    run_dense_ppca_halfset_fused_em_iteration, # gold-standard halfsets
    run_local_ppca_halfset_fused_em_iteration,
    # Multi-iteration loops
    run_dense_ppca_refinement_loop,
    run_local_ppca_refinement_loop,
    # State + schedule
    PoseMarginalPPCAEMState,
    PPCARefinementScheduleState,
    # Configs
    GeometryConfig, ScheduleConfig, ScoringConfig,
    SparsePass2Config, MeanRegularizationConfig, PostprocessConfig,
)
```

A typical caller passes 2–4 configs:

```python
result = run_dense_ppca_fused_em_iteration(
    dataset, mu, W,
    mean_prior=..., W_prior=..., noise_variance=...,
    rotations=..., translations=...,
    geometry=GeometryConfig(current_size=128, q=4),
    schedule=ScheduleConfig(image_batch_size=32, rotation_block_size=256),
    postprocess=PostprocessConfig(strategy="none", grid_correct=False),
    # other configs default
)
```

---

## 5. What this package deliberately does NOT own

- **Particle-stack I/O, CTF, dataset abstractions** → `recovar/data_io/`,
  `recovar/core/configs.py` (`ForwardModelConfig`).
- **The augmented M-step linear solve and PPCA score/moment math** →
  `recovar/ppca/`. `solve_augmented_ppca_mstep` and
  `compute_ppca_pose_scores_and_moments_no_contrast` are the two
  most-load-bearing imports.
- **Half-Fourier helpers, FFT conventions** → `recovar/core/fourier_transform_utils.py`.
- **Preprocessing, batch fetch, half-spectrum weights** →
  `recovar/em/dense_single_volume/helpers/`.
- **Noise model expansion** → `recovar/reconstruction/noise.py`.

The single rule: **if it's used by `dense_dataset.py` and `local_dataset.py`,
it's worth checking whether the canonical version lives in one of those
external dirs first.**

---

## 6. Hot tips for editing this package

- The PPCA augmented system is `[μ, W_1, ..., W_q]`; index 0 is always the
  mean row. `_enforce_augmented_x0` zeros the DC of the augmented stats
  before the M-step.
- `SparsePass2Config(enabled=False)` is the parity-test mode: pass 2
  backprojects every (image, pose) regardless of posterior weight. Use it
  when comparing against an older reference iteration.
- The two flavors converged by design — they share `engine.py` and the
  M-step / postprocess. If you're tempted to copy logic between
  `dense_dataset.py` and `local_dataset.py`, check whether it belongs in
  the shared `engine.py`, `diagnostics.py`, or `mean_regularization.py`
  first.
- **Don't widen test tolerance**, **don't modify `tests/baselines/`** —
  see `tests/CLAUDE.md`.
