# Pose-Marginalized PPCA Refinement — Implementation Notes (2026-05-01)

Companion to `docs/math/ppca_refine_plan_2026_05_01.md` (math, scope,
milestones) and `recovar/em/ppca_refinement/CLAUDE.md` (operating contract).

This file holds the implementation details: reuse pointers, code skeletons,
sufficient-stats dataclasses, line-level PCG audit, dense/sparse engine block
shapes, EM driver state, numerical contract, test list, CLI sketches.


## Naming map (legacy → canonical)

`recovar/ppca/ppca.py::_e_step_half_inner` returns sufficient statistics
under legacy names. New code uses canonical names.

| Legacy | Canonical | Shape | Dtype | Math |
|---|---|---|---|---|
| `y_norm_sq` | `y_norm` | `[..., ]` real | f32 | `<x,x>` |
| `t` | `t_mx` | `[..., ]` real | f32 | `<x,m>` |
| `nu` | `nu_mm` | `[..., ]` real | f32 | `<m,m>` |
| `g` | `g_zx` | `[..., q]` complex | c64 | `B* x` |
| `h` | `h_zm` | `[..., q]` complex | c64 | `B* m` |
| `H` | `Hzz` | `[..., q, q]` complex | c64 | `B* B` |

`B = Σ_i^{-1/2} C_i P_R W`, `m = Σ_i^{-1/2} C_i P_R μ`,
`x = T_{-t} Σ_i^{-1/2} y_i`. Augmented moments:
`alpha_aug [..., q+1]` is `E[c · [1; z]]`,
`G_aug_tri [..., (q+1)(q+2)/2]` is the upper triangle of
`E[c² · [1; z][1; z]*]`. Augmented parameter
`theta_aug = [μ, W₁..W_q]`, `r=0` is the mean component.


## Per-pose math (no contrast first)

```
M    = I_q + Hzz
b    = g_zx − h_zm
z̄    = M^{-1} b
S_z  = M^{-1}
ρ    = y_norm − 2 t_mx + nu_mm
ℓ    = − ½ [ ρ − b* M^{-1} b + log det M ]

alpha_aug  = [1; z̄]
G_aug      = [[1, z̄*], [z̄, S_z + z̄ z̄*]]
```

Required public function in `recovar/ppca/pose_marginal.py`:

```python
def compute_ppca_pose_scores_and_moments_no_contrast(
    y_norm, t_mx, nu_mm, g_zx, h_zm, Hzz,
    *, return_moments: bool,
) -> tuple[Array, Array | None, Array | None]:
    """Return (log_score, alpha_aug, G_aug_tri).

    Vectorized over arbitrary leading batch dimensions. JIT-friendly.
    Cholesky for M; log-det from L; symmetrize Hzz before factoring;
    cho_solve for S_z; jitter only behind explicit debug_jitter arg.
    No pinv. No explicit matrix inverse.
    """
```


## Augmented M-step — generalize `_pcg_hard_mstep` from q to q+1 in place

Generalize `recovar/ppca/ppca.py::_pcg_hard_mstep` with a runtime
`n_components` argument. The lines where `q` is hard-coded (verify against
the current revision before editing — function may have moved):

| Line ~ | Function | Change |
|---|---|---|
| 525 | `_mstep_AL_solve_fourier` | `D = L.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])` → use `q+1` and feed `reg_diag_aug = stack([mean_reg_diag, W_reg_diag], -1)` |
| 506 | `_mstep_batched_rfft` | `for j0 in range(0, q, _MSTEP_PC_BATCH)` → `range(0, q+1, ...)` |
| 672 | `_mstep_batched_irfft` | same loop change |
| 677 | `scatter` | output shape `(q, *vs) → (q+1, *vs)` |
| 686 | `gather` | reshape `q → q+1` |

Multimask: `pc_mask_assignment` becomes `[mean_mask_idx, *W_mask_assignment]`.
Mean component default mask is the standard solvent mask.

Backwards-compat regression test (Milestone 2 gate): legacy fixed-pose PPCA
on a tiny pinned fixture must produce identical outputs after the
generalization. Reuse the gridding kernel `K(x) = sinc²(x/D)` and the
half-Fourier convention unchanged.

Wrapper in `recovar/ppca/augmented_mstep.py`:

```python
def solve_augmented_ppca_mstep(
    stats: AugmentedPPCAStats,
    *,
    mean_prior, W_prior, masks, solver_opts,
    theta_init: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """PCG solve of augmented normal equations. Calls into the generalized
    _pcg_hard_mstep with n_components = q+1 and reg_diag stacked as
    [mean_reg_diag, W_reg_diag]."""
```


## Sufficient-statistics dataclass

```python
@dataclass
class AugmentedPPCAStats:
    rhs: Array           # [half_vol, q+1] complex64        — Σ γ α_r A* x
    lhs_tri: Array       # [half_vol, (q+1)(q+2)//2] real64 — Σ γ G_rs A* A
    residual_num: Array  # per-shell numerator
    residual_den: Array  # per-shell denominator
    log_likelihood: float
    n_images: int
    diagnostics: dict
```

Axis order matches `recovar/ppca/ppca.py::E_M_step_batch_half`. Do not
re-axis the codebase.


## Prior conventions (read together with §7 of CLAUDE.md)

  * `W_prior[half_vol, q]` is **variance**, float32. Regularizer is
    `1 / (W_prior + ε)`.
  * `τ_W(ξ, k) = max(τ_floor, α_prior · d_ppca(s(ξ)) / q_total)` with
    `q_total = opts.zdim`, `α_prior = 1.0`, `τ_floor = 1e-8`.
  * `d_ppca(s)` from
    `recovar/ppca/prior_estimation.py::estimate_hybrid_shell_prior_from_data`,
    returning `prior_info["W_prior"]` directly.
  * Latent prior identity in v1: `z_prior_precision_diag = jnp.ones(q_active)`,
    `contrast_lambdas = jnp.ones(q_active)`. Eigenvalues live in `W`, not
    in the latent prior.
  * Schedule: `prior_freeze_iters = 3`,
    `recompute_prior_once_after_iter = 5` (only when
    `opts.allow_prior_recompute=True`), `allow_every_iter_prior_update = False`.

Diagnostics dumped per iteration: `W_prior` radial curve, floor fraction,
raw shell total, repaired shell total, `|μ|²` fallback, reliable-shell
mask, `median_ratio`, `α_prior`, `q_total`, `q_active`, `latent_prior_mode`,
`pc_prior_mode`, mean / median `W_prior` inside mask, prior penalty
`Σ |W|² / W_prior`, data-vs-prior ratio.

Do not use as the loading prior: RELION `tau2`, `variance_prior`,
`prior_total_signal`, `prior_shell_subtracted`, the mean reconstruction
prior. They are different objects.


## Dense engine — reuse pointers (in this repo on this branch)

| File | Reuse |
|---|---|
| `recovar/em/dense_single_volume/em_engine.py::run_em` | Two-pass dense engine (logsumexp pass-1, accumulator pass-2); never materializes `[N,R,T]` |
| `recovar/em/dense_single_volume/dense_k_class_engine.py::run_dense_k_class_em_native` | K-class dense engine; closest analogue — `q+1` augmented components mirror K templates |
| `recovar/em/dense_single_volume/helpers/projection.py::project_half_spectrum`, `compute_projections_block` | Forward slice into half-spectrum |
| `recovar/em/dense_single_volume/helpers/backprojection.py::adjoint_slice_volume_half`, `batch_adjoint_slice_volume_half`, `accumulate_adjoint_pair` | Half-spectrum backprojection + fused (Ft_y, Ft_ctf) |
| `recovar/em/dense_single_volume/helpers/half_volume_mstep.py::half_volume_accumulator_shape`, `enforce_half_volume_x0`, `half_volume_accumulators_to_full` | Half-volume accumulator packing + x=0 Hermitian enforcement |
| `recovar/em/dense_single_volume/helpers/fourier_window.py::make_fourier_window_spec`, `quantize_current_size` | `current_size` scheduling |
| `recovar/em/dense_single_volume/helpers/orientation_priors.py`, `helpers/translation_prior.py` | Pose log-priors |
| `recovar/em/dense_single_volume/helpers/significance.py::_compute_significance_batched`, `helpers/sparse_pass2_bucketed.py`, `helpers/oversampling.py::compute_pass2_stats` | Sparse pass-2 + bucketed JIT |
| `recovar/em/dense_single_volume/helpers/types.py::MeanStats`, `RelionStats`, `NoiseStats` | Sufficient-stats dataclasses; extend, don't replace |

### Block shapes

```
B image batch · R rotation block · T translation block · F Fourier pixels at current_size
q PPCA dim   · p = q+1 augmented components

proj_mu  = project_half_spectrum(mu, rot_block, current_size)         # [R, F]
proj_W   = batched_project_half_spectrum(W, rot_block, current_size)  # [R, q, F]
proj_aug = concat([proj_mu[:, None, :], proj_W], axis=1)              # [R, p, F]
```

First-order (one half-spectrum GEMM, replaces K-class template bank):

```python
Y1 = build_shifted_whitened_images(image_batch, trans_block)          # [B*T, F]
D  = real_half_spectrum_gemm(Y1, conj(proj_aug)).reshape(B, T, R, p)
t_mx = D[..., 0]
g_zx = D[..., 1:]
```

Second-order (translation-independent):

```python
K_aug = einsum('bf, rpf, rqf -> brpq',
               ctf2_over_noise, conj(proj_aug), proj_aug).real         # [B,R,p,p]
nu_mm = K_aug[..., 0, 0]
h_zm  = K_aug[..., 1:, 0]
Hzz   = K_aug[..., 1:, 1:]
```

For large `q`, build only the upper triangle and chunk over R or F.

### Allowed / forbidden tensors

Allowed inside a block: `score [B,T,R]`, `alpha_aug [B,T,R,p]`,
`G_aug_tri [B,T,R,p(p+1)/2]`, `D [B,T,R,p]`, `K_aug_tri [B,R,p(p+1)/2]`.

**Forbidden:** `[N_images, N_rot, N_trans, *]`. CI enforces this via
`RECOVAR_DEBUG_ASSERT_NO_FULL_POSTERIORS=1`.

Memory split order: translation blocks → rotation blocks → image batches
→ contrast quadrature nodes (when contrast lands in M8).

### Two-pass loop skeleton

```python
def dense_pose_ppca_E_step_and_stats(state, dataset, image_indices, sampler, opts):
    # Pass 1: evidence + best poses + significance metadata
    for image_batch in batches(...):
        for rot_block in sampler.rotation_blocks():
            proj_aug = project_augmented(state.theta_score, rot_block, opts.current_size)
            K_aug    = compute_second_order_aug(proj_aug, ctf2_over_noise(image_batch))
            for trans_block in sampler.translation_blocks():
                Y1    = shifted_whitened_images(image_batch, trans_block)
                score = ppca_pose_score_no_contrast(Y1, proj_aug, K_aug)
                score = score + sampler.pose_log_prior(rot_block, trans_block, image_batch)
                update_logsumexp_best_significance(logZ, pmax, best_pose, sig_meta, score)

    # Pass 2: recompute and accumulate
    stats = AugmentedPPCAStats.zeros(...)
    for image_batch in batches(...):
        for rot_block in sampler.rotation_blocks():
            proj_aug = project_augmented(state.theta_score, rot_block, opts.current_size)
            K_aug    = compute_second_order_aug(proj_aug, ctf2_over_noise(image_batch))
            for trans_block in sampler.translation_blocks():
                Y1                       = shifted_whitened_images(image_batch, trans_block)
                score, alpha_aug, G_tri  = ppca_pose_score_and_moments_no_contrast(
                                              Y1, proj_aug, K_aug, return_moments=True)
                score                    = score + sampler.pose_log_prior(...)
                gamma                    = jnp.exp(score - logZ_for_image_batch)
                accumulate_augmented_ppca_stats(stats, gamma, alpha_aug, G_tri, ...)
    return stats, PosteriorDiagnostics(logZ, pmax, best_pose, sig_meta)
```


## Sparse / local engine — reuse pointers

| File | Reuse |
|---|---|
| `recovar/em/dense_single_volume/local_em_engine.py::run_local_em_exact` | Per-image neighborhood EM with adaptive bucketing |
| `recovar/em/dense_single_volume/local_k_class_engine.py::run_local_k_class_em_native` | K-class local engine — closest analogue to augmented PPCA |
| `recovar/em/dense_single_volume/local_layout.py::LocalHypothesisLayout`, `bucket_local_hypothesis_layout` | Flat per-image hypothesis storage; bucketed for static-shape JIT |
| `recovar/em/sampling.py::build_local_search_grid_metadata` | Local search metadata |
| `recovar/em/dense_single_volume/local_score_pass.py`, `helpers/local_search.py` | Local pass-2 score path |
| `recovar/em/dense_single_volume/local_backprojection.py::enforce_relion_half_volume_x0_hermitian` | x=0 Hermitian enforcement |

E-score and pruning: `s_ia = log π_ia + ℓ_ia`; `E_ia = s_ia − max_b s_ib`.
Same significance thresholds as the k-class code (`τ_sig`,
`max_significant`, parent-expansion oversampling, block-max E pruning).
Pruning may restrict support; it must not alter per-hypothesis scores.

Two modes: (A) coarse-to-fine — score coarse grid, retain significant
parents, expand to oversampled children, score with exact PPCA likelihood,
normalize over retained fine support, accumulate. (B) local refinement
around current poses — main M7 target; build local angular + shift
hypotheses with existing σ/cutoff logic, score, normalize, accumulate,
update hard poses from posterior maxima.

Sparse return:

```python
SparsePPCAPosterior(
    logZ, pmax, nr_significant,
    best_rot, best_shift,
    omitted_mass_estimate,
    per_image_counts_after_pruning,
)
```

M6 gate: sparse with all hypotheses retained equals dense bit-for-bit
(within float32 fused-multiply tolerance) on a tiny image batch and pose grid.


## EM driver state

`recovar/em/ppca_refinement/state.py`:

```python
@dataclass
class PoseMarginalPPCAEMState:
    mu_half:  tuple[Array, Array]
    W_half:   tuple[Array, Array]
    mu_score: Array                 # filtered avg of halfsets, used for E-step
    W_score:  Array

    W_prior:  Array                 # [half_vol, q] variance
    mean_prior: Any
    z_prior_precision_diag: Array   # ones(q) in v1

    noise_variance: Array
    contrast_params: ContrastParams # mode="none" until M8
    masks: MaskSpec

    pose_estimates: PoseTable
    pose_priors: Any
    refinement_schedule_state: Any
    hyperparams: Any
    diagnostics: dict
```

`mu_score`, `W_score` use the existing `combine_halfsets_for_scoring` path
(see `recovar/em/dense_single_volume/helpers/convergence.py::RefinementState`).

W initialization order: (1) run existing fixed-pose PPCA from current
poses + mean; (2) small random masked volumes RMS 1e-3..1e-2 × mean RMS;
(3) low-rank class-volume differences from a previous k-class run.


## Numerical contract

  * `complex64` images / projections / accumulators. `float64` available
    behind `--use-float64-scoring` (mirrors high-res EM).
  * Cholesky for `M` and log-det. No `pinv` in new code.
  * Symmetrize Hermitian matrices before factoring.
  * Jitter only behind an explicit flag; default zero; report failures.
  * Call `enforce_relion_half_volume_x0_hermitian` after every M-step
    accumulation. Don't skip on the augmented path.
  * FFT normalization: `1/N` forward IFFT, `1` backward FFT (RECOVAR).
    Image rfft via numpy: forward `1`, irfft `1/N`. The legacy
    `_pcg_hard_mstep` bakes the gridding kernel `K(x) = sinc²(x/D)` into
    the operator; reuse unchanged.
  * Half-spectrum weights: `1` at DC and Nyquist, `2` at interior columns
    for full-spectrum inner product. Scoring uses RELION unit weights via
    `make_scoring_half_image_weights(..., relion_half_sum=True)`. Don't
    "fix" the asymmetry.
  * Volume frame: RECOVAR `[z,y,x]`, RELION `[x,y,z]`,
    `vol_recovar = -np.transpose(vol_relion, (2,1,0))`. Use
    `recovar.utils.helpers.load_relion_volume(...)` for RELION MRCs.
  * Numeric parity escalation (from `recovar/em/CLAUDE.md`): RELION GPU
    parity `~1e-4` arithmetic; `1e-3` or pose flips or multi-iteration
    drift are escalations.


## Test plan

### Unit (in `tests/unit/ppca_refinement/`)

  1. `q=0` reduction → homogeneous pose-marginalized reconstruction.
  2. `W=0` reduction → homogeneous scores plus pose-independent constant.
  3. Fixed-pose fixed-mean parity vs `recovar/ppca/ppca.py` E/M stats.
  4. Fixed-pose free-mean toy: augmented PCG vs explicit dense normal eqs.
  5. Dense brute-force enumeration (tiny image, tiny grid) vs blockwise.
  6. Sparse equals dense when unpruned.
  7. Contrast posterior parity vs `contrast_posterior.py` (M8).
  8. Basis rotation invariance: `W ← W Q`, `Q` orthogonal.
  9. Prior convention: `W_prior == α_prior · d_ppca(shell) / q_total` after
     clipping; latent prior remains identity.
  10. Backwards-compat regression: legacy fixed-pose PPCA on a tiny pinned
      fixture produces identical outputs after the PCG generalization.

### Integration

  1. Fixed-pose synthetic homogeneous → mean reconstructs, `W` stays small.
  2. Fixed-pose synthetic linear hetero → `W` subspace recovers GT up to
     orthogonal rotation.
  3. Local-pose synthetic linear hetero → log-evidence improves over
     fixed-pose; subspace recovered.
  4. Dense vs unpruned sparse → match on a small ribosome subset.
  5. CryoBench ribosome refinement-first comparison: fixed-pose PPCA vs
     local-pose PPCA refinement vs k-class high-res EM vs existing RECOVAR
     PPCA without pose marginalization.

Track FSC, posterior `pmax`, `nr_significant`, pose changes per iter, log
evidence, noise spectrum, contrast distribution, `W` singular values,
halfset `W` subspace angles, runtime, GPU memory.

Phase success: at least as stable as fixed-pose PPCA, no consensus
degradation, pose likelihood improved or stabilized vs fixed poses.

Don't run the full long-test suite for interim parity work
(`feedback_no_longtest_for_parity.md`). Long-test required only before PR.


## Pitfalls (extended notes)

  * **DC pixel / x=0 Hermitian.** Backprojection doesn't enforce it
    automatically; call `enforce_relion_half_volume_x0_hermitian` on every
    half-volume accumulator before public exposure.
  * **Half-spectrum weights asymmetry is intentional.** Don't "fix" the
    RELION-style scoring weight imbalance.
  * **Translation handling.** Default GEMM with shifted-image copies (200×
    reuse, 45 ms); FFT-per-rotation 1.5G vs 327 GB but slower overall.
    Augmented engine follows GEMM path; don't introduce a new strategy.
  * **JIT recompilation.** Static-arg-jitted on `image_shape`,
    `proj_volume_shape`, `recon_volume_shape`, `disc_type`, `n_shells`,
    bucket sizes. Mid-iteration changes recompile. Pass these the same way
    `run_dense_k_class_em_native` does.
  * **Iter-1 RELION `--firstiter_cc`.** Hard winner-take-all CC; not
    Bayesian. Don't replicate.
  * **`tau2` vs `W_prior`.** Different priors. See prior section.
  * **PPCA E-step uses `pinv` today.** New code uses Cholesky. Don't carry
    `pinv` forward; consider migrating the legacy site in a separate
    parity-tested commit.
  * **Iterative eigenvalue refit during EM is harmful**
    (`project_ppca_eigenval_update_during_anneal_harmful.md`). Keep `s`
    frozen in the EM loop in v1; eigenvalue refit is post-EM only (M9 via
    `ppca_iterative_refitb.py`).
  * **`use_global_significant_support`** has a known regression on the
    relion-parity branch (per-image Python loop;
    `project_use_global_significant_support_path.md`). Sparse engine must
    not depend on it.
  * **Multimask gate.** Legacy `_pcg_hard_mstep` already supports
    multimask via `pc_mask_assignment`; the augmented version inherits
    this. M7 single-mask only; multimask gates on M9.
  * **`H` is overloaded in legacy code.** `H` in `_e_step_half_inner` is
    `Hzz`. `compute_H_B`, `compute_little_H_b` in
    `recovar/em/heterogeneity.py` are different objects.


## CLI sketches

```bash
# M3 — fixed pose
recovar ppca-refine particles.star \
  --out RefinePPCA/fixed_pose_test --init-mean consensus.mrc --init-poses poses.star \
  --zdim 6 --pose-mode fixed --contrast none --pc-prior hybrid_shell

# M7 — main local-pose refinement
recovar ppca-refine particles.star \
  --out RefinePPCA/local_pose --init-mean consensus.mrc --init-poses poses.star \
  --zdim 6 --pose-mode local --engine sparse --contrast none --pc-prior hybrid_shell \
  --reuse-kclass-pose-schedule

# M5 — dense low-res debug
recovar ppca-refine particles.star \
  --out RefinePPCA/dense_debug --init-mean consensus.mrc --init-poses poses.star \
  --zdim 6 --pose-mode dense --engine dense --max-resolution 20 --contrast none

# M8 — contrast restart
recovar ppca-refine particles.star \
  --out RefinePPCA/local_pose_contrast --init RefinePPCA/local_pose/state.pkl \
  --zdim 6 --pose-mode local --engine sparse --contrast marginalize \
  --allow-prior-recompute-once
```
