# Empirical-Bayes shell-prior eigenvalue calibration (design note, NOT implemented)

**Status:** future direction (Phase 6).
**Audience:** anyone considering implementing it.
**Why this exists:** the post-review evaluation (2026-04-25) flagged a fourth eigenvalue-calibration option that we recorded but did not implement in v0.

## Problem

In the v0 PPCA ab-initio model, eigenvalues `s = (s_1, …, s_q)` describe per-PC signal variance:

```
α_i ~ N(0, diag(s)),  y_i = CTF_i · S_t · A_R · (μ + U α_i) + ε_i
```

Three eigenvalue strategies have been considered in v0:

1. **Frozen `s = 1` flat (current default).** Validated as cheat-free. Likelihood Gram dominates the prior at cryo-EM SNR; EM trajectory is identical across 7 orders of magnitude of `s`. But absolute calibration is lost.
2. **Tipping-Bishop iterative `s` update during EM.** Empirically harmful — pose-discretization bias inflates eigenvalues 4× under annealing and ~25% even at f=1 post-anneal (Section 9.2 of `ppca_abinitio_status_20260416.md`). Default off.
3. **One-shot post-EM ProjCov refit (Phase 2, shipped).** Eigendecomposes the sample-averaged posterior covariance `Σ_α = (1/N) Σ_i E[α_i α_i^T | y_i]` and rotates `U`. Same posterior-bias concern as (2), but applied as a single one-shot pass at f=1, so it does not propagate biased s back into another E-step. Better than nothing, but still inherits some pose-discretization bias.

The fourth option is **empirical-Bayes shell hyperparameters**.

## The proposal

Replace the scalar prior precision `Λ⁻¹ = diag(1/s)` with a **shell-stratified, per-PC prior precision** that is itself estimated by maximum marginal likelihood on a **held-out validation split** of the data:

```
prior:  α_{i,k}(shell ℓ) ~ N(0, τ_{k,ℓ})    (per-PC, per-shell prior variance)
hyperprior on τ:  τ_{k,ℓ} ~ InvGamma(a, b)   (or weak conjugate)
```

The hyperparameters are `{τ_{k,ℓ}}` for k = 1..q, ℓ = 0..L-1, plus the InvGamma `(a, b)`.

### Estimation

1. **Train EM at flat-s on training split** — same v0 algorithm as today.
2. **Compute held-out marginal likelihood** `p(Y_val | μ, U, τ)` as a function of the τ hyperparameters, integrating over α analytically (PPCA closed form) and over poses on the discrete grid.
3. **Maximize wrt τ** — either gradient ascent on log-marginal, or an EB EM loop where:
   - E-step computes posterior moments of α at current τ
   - M-step closed-form updates `τ_{k,ℓ}` per shell from the posterior second moments restricted to shell ℓ
4. **Final τ** is the eigenvalue spectrum to report. Optional U rotation (Phase 2 style) on top.

### Why this should help

- **Pose discretization bias is amortized over a held-out set**, not the training data: the training-side bias that hurts Tipping-Bishop is replaced by a held-out objective that is closed-form and unbiased relative to the pose grid (each held-out image sees the same pose grid).
- **Shell-stratified prior** matches cryo-EM signal physics: low-frequency shells carry orders of magnitude more variance than high-frequency shells, and a single scalar `s_k` per PC cannot represent that.
- **Conjugate hyperprior** stabilizes the estimate in low-data shells.
- **Compatible with the existing v0 forward model** — no new linear algebra primitives, just a held-out marginal log-likelihood evaluator.

### Connection to W_prior regularization

The `W_prior` regularizer in the M-step (Phase 1) is the **regularization analog** of this hyperprior: it enforces shell-stratified per-PC prior magnitudes during the M-step solve, but uses the *current* `(U, s)` to estimate the shell magnitudes (a single-pass plug-in). The empirical-Bayes version would estimate those shell magnitudes from data via held-out marginal-likelihood maximization rather than reading them off the current iterate — replacing a heuristic with a principled estimator while keeping the same shape.

## Why we did not implement it in v0

Three reasons:
1. **Scope.** v0 ships an algorithm validation at vol=32; spectrum calibration is secondary to subspace recovery, and ProjCov is sufficient for v0's claims.
2. **Held-out marginal likelihood plumbing.** Phase 1 added the held-out lm trajectory as a model-selection signal but the Phase 1 sweep is not yet evaluated, so we don't know how well held-out lm tracks ground truth at the v0 forward-model fidelity. Empirical-Bayes only works if the held-out lm landscape is well-behaved.
3. **Conjugacy bookkeeping.** Per-shell, per-PC InvGamma updates require careful indexing into the half-volume rfft layout's radial shell structure. Doable but is its own engineering task.

## Pointers for the future implementer

- **Conjugate update derivation:** for InvGamma(a, b) prior on τ_{k,ℓ}, the EM M-step posterior is InvGamma(a + n_ℓ/2, b + 0.5 Σ_{i in shell ℓ} E[α_{i,k}^2]). See e.g. Bishop PRML §10.2 for the standard PPCA EB derivation; the shell stratification is a per-shell repeat.
- **Shell index:** existing radial labels live in `recovar/em/ppca_abinitio/factor_update.py::compute_W_prior_half`. Same labeling can drive the EB shell aggregation.
- **Held-out lm hook:** Phase 1's `--save-results` JSON dump already records the trajectory; the EB loop would read this as the optimization target for τ.
- **U rotation compatibility:** because the prior is now per-PC, U gauge changes during refit invalidate τ. Either freeze `U` (post-EM only) or carry a joint orthonormalization update on (U, τ).
- **Sister-branch precedent:** the `claude/ppca-refit-algorithms` branch has shell-stratified prior infrastructure in `recovar/ppca/prior_estimation.py::estimate_hybrid_shell_prior_from_data` — that is the closest existing implementation in the codebase, though it operates on the full pipeline forward model rather than v0.

## Acceptance criteria for a future EB implementation

- Calibration error `mean_k |log(s_eb / u_k^T Σ_gt u_k)|` on Ribosembly + IgG-1D + IgG-RL at SNR ≤ 1 reduces by ≥ 2× vs ProjCov.
- Held-out lm at the EB optimum is strictly higher than at flat-s.
- Subspace recovery (PE) does not regress.
- Cost: ≤ 2× wall time of one EM run (for the EB inner loop).
- Same `--cheat-free` contract: must not depend on `ds.s_true`.
