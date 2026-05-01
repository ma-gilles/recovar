# Pose-Marginalized PPCA in RECOVAR — Refinement-First Plan (2026-05-01)

> Original task spec for the `recovar ppca-refine` project. This document is
> the source-of-truth for math, scope, and milestones. The actionable
> distilled guide for agents lives at
> `recovar/em/ppca_refinement/CLAUDE.md` — if it contradicts this plan,
> the agent guide wins (it captures the corrected name table, the
> line-level PCG audit, and the branch-reality decisions).
>
> Branch: `claude/ppca-refine-pose-marginal`, based on
> `claude/relion-parity-local-search-fix` (which already contains both
> `recovar/em/dense_single_volume/` and `recovar/ppca/`).

This plan replaces the earlier ab-initio-first plan. The immediate goal is a
**non-InitialModel, high-resolution EM/refinement implementation** of PPCA
with pose marginalization. The ab-initio InitialModel/VDAM version is now a
later milestone, because the k-class InitialModel path still has unresolved
bugs and should not be the foundation for the first PPCA implementation.

The first production target is `recovar ppca-refine`. It starts from an
existing consensus/pose state — homogeneous refinement, k-class refinement,
or imported RELION-style poses — and jointly refines: mean volume μ, PPCA
loading volumes W₁..W_q, pose posterior over rotations and translations,
noise and contrast parameters, and PPCA loading priors.

The key implementation rule is: **reuse the high-resolution EM
pose-refinement machinery, not the InitialModel/VDAM machinery, for the
first working version.**


## In scope now

  1. Fixed-pose PPCA parity tests.
  2. Dense pose-marginalized PPCA E-step.
  3. Sparse/local pose-marginalized PPCA E-step with pruning from the
     likelihood/E-score.
  4. Joint augmented M-step for `[μ, W]`.
  5. No-contrast first.
  6. Profile contrast second.
  7. Marginalized contrast third.
  8. Multimask and existing PPCA hyperparameter logic after the no-contrast
     path works.
  9. CryoBench ribosome runs initialized from reliable non-ab-initio poses
     or maps.


## Out of scope for the first implementation

`recovar ppca-initial-model`, VDAM PPCA InitialModel, mean-only InitialModel
warmup, pseudo-halfset VDAM scheduling, InitialModel sparse support
scheduling, RELION BPref conversion for PPCA stats, native PPCA VDAM
momentum updates.


## Source branches and reuse targets

The new branch is based on `claude/relion-parity-local-search-fix`, which
contains both the high-resolution EM machinery and the fixed-pose PPCA
package. The deferred ab-initio prototype lives on `claude/ppca-abinitio-v0`
and is not authority — borrow names and logging only.

**PPCA reuse:** `recovar/ppca/ppca.py`, `contrast_posterior.py`,
`prior_estimation.py`, `ppca_iterative_projcov.py`,
`ppca_iterative_refitb.py`. Specific reuse: `_e_step_half_inner`-like
sufficient statistics, `E_M_step_batch_half` accumulation,
`_pcg_hard_mstep` PCG, contrast profile/marginalization, contrast
renormalization, automatic PPCA prior estimation, single-/multimask logic,
final basis orthonormalization.

**EM reuse:** `recovar/em/dense_single_volume/em_engine.py`,
`recovar/em/e_step.py`, `m_step.py`, `iterations.py`, `sampling.py`,
`states.py`, `heterogeneity.py`. Specific reuse: dense two-pass pose
normalization, local hypothesis layouts, rotation/translation schedules,
pose priors, hard-pose updates from posterior maxima, sparse significance
pruning, omitted-mass diagnostics, halfset splitting/scoring, noise update
conventions.

**InitialModel code is deferred.** Do not reuse `recovar/em/initial_model/*`,
RELION BackProjector / BPref VDAM paths, InitialModel subset schedules,
pseudo-halfsets, `tau2_fudge` schedules in v1.


## Model

For particle `i`, pose hypothesis `a=(R,t)`, latent `z_i`, optional contrast
`c_i`:

```
z_i ~ N(0, I_q),
x_ia = c_i A_ia (μ + W z_i) + ε_i,    ε_i ~ N(0, I)
A_ia v = Σ_i^{-1/2} C_i T_t P_R v.
```

Shifting the image instead of the projection:

```
x_ia  = T_{-t} Σ_i^{-1/2} y_i,
m_ia  = Σ_i^{-1/2} C_i P_R μ,
B_ia  = Σ_i^{-1/2} C_i P_R W,
x_ia  = c_i m_ia + c_i B_ia z_i + ε_i.
```

Augmented parameter `θ_aug = [μ, W₁..W_q]`, `r(z) = [1; z]`,
`μ + W z = θ_aug r(z)`.


## Naming convention (per-pose stats and moments)

```
y_norm   scalar       <x,x>
t_mx     scalar       <x,m>
nu_mm    scalar       <m,m>
g_zx     [q]          B* x
h_zm     [q]          B* m
Hzz      [q,q]        B* B

alpha_aug  [q+1]                         E[c · [1, z]]
G_aug_tri  [(q+1)(q+2)/2]                upper triangle of E[c² · [1,z][1,z]*]
```


## Per-pose posterior, no contrast

```
M     = I_q + Hzz
b     = g_zx − h_zm
z̄     = M^{-1} b
S_z   = M^{-1}
ρ     = y_norm − 2 t_mx + nu_mm
ℓ_ia  = − ½ [ ρ − b* M^{-1} b + log det M ]

alpha_aug = [1; z̄]
G_aug     = [[1, z̄*], [z̄, S_z + z̄ z̄*]]
```

Numerical: Cholesky of `M`, log-det from Cholesky, symmetrize `Hzz`, jitter
only behind a debug flag, no explicit inverse except for tiny `q`.


## Contrast (deferred to M8)

For fixed `c`:

```
M(c) = I_q + c² Hzz
b(c) = c · g_zx − c² · h_zm
ρ(c) = y_norm − 2 c · t_mx + c² · nu_mm
ℓ_ia(c) = − ½ [ ρ(c) − b(c)* M(c)^{-1} b(c) + log det M(c) ] + log p(c)
```

Marginalized contrast wraps `recovar/ppca/contrast_posterior.py`, which
returns `mean_c`, `second_c`, `mean_cz`, `mean_c2z`, `second_c2zz`,
`marginal_ll`. Build `alpha_aug`, `G_aug` from those moments.


## Pose posterior

```
γ_ia = (π_ia · exp(ℓ_ia)) / Σ_b (π_ib · exp(ℓ_ib))
```

Two-pass, k-class style. Pass 1 = scores + priors + logZ + pmax + best
rotation/shift + significance metadata. Pass 2 = recompute + γ + accumulate
augmented M-step stats + residual/noise diagnostics.


## Augmented M-step

```
∀ r ∈ {0,…,q}:  Σ_s (Σ_{i,a} γ_ia G_ia,rs A_ia* A_ia) θ_s + R_r θ_r
                            = Σ_{i,a} γ_ia α_ia,r A_ia* x_i

R = diag(R_μ, R_W, …, R_W)         R_r is block-diagonal over components

rhs_r += γ_ia · α_ia,r · A_ia* x_i
lhs_rs += γ_ia · G_ia,rs · A_ia* A_ia

residual = y_norm − 2 Σ_r α_ia,r D_ia,r + Σ_rs G_ia,rs K_ia,rs
D_ia,r   = <x_i, A_ia θ_r>
K_ia,rs  = <A_ia θ_r, A_ia θ_s>
```

PCG: generalize `_pcg_hard_mstep` from `q` to `q+1`. Component 0 uses mean
prior / homogeneous regularization. Components 1..q use `W_prior`. Single-
and multi-mask supported. Same gridding-kernel correction as the PPCA
branch. Same half-Fourier convention. No dense matrix inverse over voxels.


## PC prior parameters: exact convention

Two priors. Latent prior on `z`: identity in v1. Loading-volume prior on `W`:

```
p(W) ∝ exp(−½ Σ_k Σ_ξ |W_k(ξ)|² / (τ_W(ξ,k) + ε))

W_prior[half_vol, q]      # variance, larger means weaker regularization
W_reg_diag = 1 / (W_prior + ε)

reg_diag_aug[:, 0]  = mean_reg_diag
reg_diag_aug[:, 1:] = W_reg_diag
```

Default formula:

```
τ_W(ξ, k) = max(τ_floor, α_prior · d_ppca(s(ξ)) / q_total)
q_total = opts.zdim, α_prior = 1.0, τ_floor = 1e-8,
use_q_total_for_division = True, smooth_shell_prior = True,
latent_prior_mode = "identity", pc_prior_mode = "hybrid_shell".
```

Use `q_total`, not `q_active`. Estimate `d_ppca(s)` from current μ + poses
+ noise via `estimate_hybrid_shell_prior_from_data`.

Update schedule: `prior_freeze_iters = 3`,
`recompute_prior_once_after_iter = 5` (only if `allow_prior_recompute=True`),
`allow_every_iter_prior_update = False`. Diagnostics: `W_prior` radial
curve, floor fraction, raw shell total, repaired shell total, |μ|² fallback,
reliable shell mask, median ratio, α_prior, q_total/q_active, prior modes,
mean / median W_prior inside mask, prior penalty, data-vs-prior ratio.

Do **not** use RELION τ², `variance_prior`, `prior_total_signal`,
`prior_shell_subtracted`, or the mean reconstruction prior as the PPCA
loading prior.


## Dense engine

Block shapes `B / R / T / F / q / p=q+1`. Project augmented bank `[μ, W]` to
`proj_aug [R, p, F]`. One half-spectrum GEMM gives `D = Y1 · proj_aug* →
[B,T,R,p]`. Second-order `K_aug [B,R,p,p]` is translation-independent.
Two-pass loop: pass 1 → logZ/pmax/best/significance; pass 2 → recompute +
γ + accumulate. Allowed block tensors: `score [B,T,R]`,
`alpha_aug [B,T,R,p]`, `G_aug_tri [B,T,R,p(p+1)/2]`, `D [B,T,R,p]`,
`K_aug_tri [B,R,p(p+1)/2]`. Forbidden: `[N_images, N_rot, N_trans, *]`.


## Sparse engine

Reuse the high-resolution EM local-search machinery; replace only the
per-hypothesis score/moment function. E-score `s_ia = log π_ia + ℓ_ia`,
`E_ia = s_ia − max_b s_ib ≤ 0`. Same significance thresholds as the k-class
code. Two modes: dense/coarse → sparse/fine, and local refinement around
current poses (main target). Reuse `LocalHypothesisLayout`. Return
`SparsePPCAPosterior(logZ, pmax, nr_significant, best_rot, best_shift,
omitted_mass_estimate, per_image_counts_after_pruning)`.


## EM driver state

`PoseMarginalPPCAEMState` carries `mu_half`, `W_half`, `mu_score`,
`W_score`, `W_prior`, `mean_prior`, `z_prior_precision_diag = ones(q)`,
`noise_variance`, `contrast_params`, `masks`, `pose_estimates`,
`pose_priors`, `refinement_schedule_state`, `hyperparams`, `diagnostics`.
Initialization modes: fixed (mean+poses), local (mean+poses), dense (mean
optional rough poses). No random-map ab-initio. W initialization: existing
fixed-pose PPCA from current poses+mean (preferred), or small random masked
volumes RMS 1e-3..1e-2 × mean RMS, or low-rank class-volume differences.
Always `z ~ N(0, I)`; scale lives in `W`.


## Efficient accumulation

RHS: stack components and call batched backprojection. LHS: sum γ·G over
translations before CTF² backprojection (translation does not affect A*A).
Allowed block tensors as above. Memory split order: translations →
rotations → image batches → contrast quadrature nodes.


## Code organization

```
recovar/ppca/{pose_marginal,augmented_mstep,pose_accumulators,pc_prior_config}.py
recovar/em/ppca_refinement/{state,dense_engine,sparse_engine,iterations,cli}.py
```

Do not create an InitialModel PPCA package yet.


## CLI

Modes: `fixed` (parity sanity), `local` (main), `dense` (debug/exactness).
Engines: `dense`, `sparse`. Contrast: `none` → `profile` → `marginalize`.


## Testing

Unit: q=0, W=0, fixed-pose parity, free-mean toy, dense brute-force,
sparse=dense unpruned, contrast posterior parity, basis rotation
invariance, prior convention, backwards-compat regression for legacy PPCA.
Integration: synthetic homogeneous (mean reconstructs), synthetic linear
hetero (subspace recovers GT), local-pose synthetic (log evidence
improves), dense vs unpruned sparse on small ribosome, CryoBench ribosome
refinement-first: fixed / local / k-class / existing PPCA.

Track FSC, pmax, nr_significant, pose changes, log evidence, noise
spectrum, contrast distribution, W singular values, halfset W subspace
angles, runtime, GPU memory.

Phase success: at least as stable as fixed-pose PPCA, no consensus
degradation, pose likelihood improved or stabilized vs fixed poses. The
ab-initio criterion is deferred.


## Milestones

```
M0  audit + naming map
M1  per-pose math (compute_ppca_pose_scores_and_moments_no_contrast)
M2  augmented M-step (q+1 PCG generalization)
M3  fixed-pose driver — recovar ppca-refine --pose-mode fixed
M4  dense E-step, no contrast
M5  dense driver — --pose-mode dense --engine dense
M6  sparse E-step
M7  local driver (main) — --pose-mode local --engine sparse
M8  contrast: profile → marginalized → renormalize μ AND W
M9  multimask + final PPCA postprocessing
M10 CryoBench ribosome eval
```


## Deferred ab-initio / InitialModel

Later, add `recovar ppca-initial-model`. Plan: (1) k-class `K=1`
InitialModel warmup; (2) export stable mean, pose priors, noise, particle
state; (3) custom PPCA InitialModel E-step using the same score/moment
function as high-resolution EM; (4) custom PPCA VDAM M-step on augmented
`[μ, W]` normal equations; (5) damped target updates first; (6) VDAM
moment updates only after damped works; (7) never feed cross-coupled PPCA
augmented stats into a class-only VDAM M-step; (8) never apply RELION
BPref `−N²` / `N⁴` frame scaling to native PPCA augmented stats. Ab-initio
success: on CryoBench ribosome, PPCA InitialModel produces a consensus map
at least as good as k-class InitialModel and recovers meaningful low-D
heterogeneity. Deferred until k-class InitialModel is reliable.


## Non-negotiable rules

  1. Build high-res refinement first. Don't start with InitialModel/VDAM.
  2. Keep `z ~ N(0, I)` in v1.
  3. PC eigenvalues belong in `W`, not in the latent prior.
  4. `W_prior` is a loading-volume variance. Regularizer is `1 / W_prior`.
  5. `W_prior(ξ, k) = max(τ_floor, α_prior · d_ppca(shell(ξ)) / q_total)`.
  6. Do not use RELION τ² or mean reconstruction priors as the PPCA loading
     prior.
  7. No-contrast before profile or marginalized contrast.
  8. Dense and sparse engines call the same PPCA pose-score/moment function.
  9. Sparse pruning may restrict support; it must not alter per-hypothesis
     scores.
  10. The augmented M-step must include cross-component `G_aug[r,s]` terms.
  11. The fixed-pose PPCA in `recovar/ppca/` is the trusted starting
      point — reuse, don't fork.
  12. Ab-initio / InitialModel PPCA is deferred until the k-class
      InitialModel bugs are resolved.
