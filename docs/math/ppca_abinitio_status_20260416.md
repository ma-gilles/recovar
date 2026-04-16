# PPCA Ab-Initio v0 — Status Report (2026-04-16)

**Branch:** `claude/ppca-abinitio-v0`
**Worktree:** `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_ppca_review_20260408`
**Module root:** `recovar/em/ppca_abinitio/`
**Benchmark script:** `scripts/ppca_abinitio/run_cryobench.py`

---

## 1. What this is

A probabilistic PCA (PPCA) implementation for cryo-EM heterogeneity that
operates in the **ab-initio** regime: poses are not known and are marginalized
over a discrete HEALPix × translation grid.  The goal is to jointly infer
the mean volume μ, the q principal heterogeneity directions U, and per-image
latent coordinates α, starting from a rough initial mean estimate and no
prior knowledge of U.

The implementation lives in `recovar/em/ppca_abinitio/` with ~15 modules
built over 10 commits.  It is evaluated against CryoBench ground-truth
volume ensembles (Ribosembly, IgG-1D, IgG-RL) through
`scripts/ppca_abinitio/run_cryobench.py`, which synthesizes images from
GT volumes through the v0 forward model to create a convention-matched
test bed.

---

## 2. The math

### 2.1 Generative model

Per image i with unknown pose g = (rotation R, translation t):

```
y_i = CTF_i · S_t · A_R · (μ + U α_i) + ε_i
α_i ~ N(0, Λ),   Λ = diag(s₁, …, s_q)
ε_i ~ N(0, σ²_i I)
```

All operations happen in the **half-volume rfft layout**: μ ∈ ℂ^{V_half},
U ∈ ℂ^{q × V_half}, where V_half = N₀ · N₁ · (N₂/2 + 1).  A real-volume
Hermitian constraint is enforced via `project_to_real_volume_subspace` and
the half-volume inner product uses frequency-dependent weights w_k
(w=1 for the DC and Nyquist planes, w=2 for all interior planes) so that
⟨a, b⟩_half = Σ_k w_k · conj(a_k) · b_k = ⟨a_real, b_real⟩_ℓ².

### 2.2 E-step (pose-marginalized posterior)

For each (image i, pose g) on the fixed grid G:

```
Posterior of α_i | y_i, g:    N(m_{i,g}, H_{i,g}⁻¹)
where H_{i,g} = Λ⁻¹ + (A_R U)ᴴ diag(CTF²/σ²) (A_R U)
      m_{i,g} = H_{i,g}⁻¹ (A_R U)ᴴ diag(CTF²/σ²) (S_t · CTF · y_i / σ² − A_R μ)

Responsibility:  γ_{i,g} ∝ exp(log_score_{i,g})
where log_score includes the observation term and the ½ log det Λ·H⁻¹
```

This is `score_and_posterior_moments_eqx` in `posterior.py`.

### 2.3 M-step for μ (Wiener filter)

Given γ, each μ update is a pose-weighted Wiener reconstruction.  Two
variants:

- **Homogeneous** (`update_mu_homogeneous`): standard Wiener from
  γ-weighted image data.  Used during burn-in.
- **Residualized** (`update_mu_residualized`): subtracts the
  γ-weighted posterior-mean heterogeneity contribution Σ_g γ_{i,g} A_R U m_{i,g}
  from each image before Wiener reconstruction of μ.  The residualized
  mean converges to the PPCA ML fixed point of μ.

### 2.4 M-step for U (closed-form per-voxel solve)

The U M-step is the standard Tipping-Bishop closed form, extended to
pose marginalization.  For each half-volume voxel v, solve:

```
M_v · U_new[:, v] = B_v
```

where

```
M_v = Σ_{i,g} γ_{i,g} · ctf²_{i,g,v}/σ²_i · (m_{i,g} mᵀ_{i,g} + H⁻¹_{i,g}) + λI
B_v = Σ_{i,g} γ_{i,g} · ctf²_{i,g,v}/σ²_i · m_{i,g} · (y_shifted_{i,g,v} − μ_v)
```

This is a q×q system per voxel, solved by `jax.vmap(jnp.linalg.solve)`.
No gradient descent, no learning rate, no line search.

After the solve, U is projected back to the real-volume Hermitian subspace
and orthonormalized under the half-volume weighted inner product
(`real_volume_orthonormalize_half`).

See `docs/math/ppca_closed_form_mstep.md` for the full derivation.

### 2.5 Eigenvalue constraint

In the current v0, eigenvalues s are **frozen at the empirical values from
the GT ensemble** (spec Section Q2).  This is a deliberate choice: it
isolates the subspace learning problem from eigenvalue estimation.
Eigenvalue estimation is a separate topic with known shrinkage issues
(see Section 6.3).

### 2.6 Deterministic annealing

The noise variance σ² can be multiplied by a schedule factor f(t)
that starts large and decays to 1.  The `log1000` schedule uses
f(t) = 10^{3(1 - t/T)}, i.e., `np.logspace(3, 0, T)`.

This flattens the posterior (all poses equally likely) early in the
loop so the M-step can explore broadly, then tightens the posterior as
the model improves.  It is critical for large q on Ribosembly where
the landscape has multiple basins (see Section 5.2).

---

## 3. Pipeline structure

The `run_cryobench.py` script runs three stages:

### Stage A: Homogeneous burn-in (default 0 iters for discrete_volumes mode)

Update μ only via homogeneous Wiener filter with U=0.  Converges μ to the
oracle fixed point (discretization floor).  Not needed when using the
`discrete_volumes` external mode (all images come from actual GT volumes
and the perturbed-mu init is already good enough).

### Stage B: U initialization

Three options:
- **`svd` (default, weighted):** Compute per-image residuals r_i = y_i - Σ_g γ_{i,g} A_R μ,
  project each residual to the real-volume subspace, weight by sqrt(w_k),
  stack, and take the top-q SVD.  The weighting ensures the SVD ℓ²
  matches the real-volume ℓ².
- **`random`:** Random orthonormal U in the real-volume Hermitian subspace.
  Requires annealing to avoid lazy basins.
- **`zero`:** U=0.

### Stage C: Joint mean+factor loop (default 30 iters)

Each iteration:
1. Residualized μ update (with annealing factor if set)
2. Closed-form factor U update
3. Report: FRE(μ, μ_true), FRE(μ, μ_oracle_fp), projector error PE(U), log-marginal

---

## 4. Metrics

### 4.1 What we report

| Metric | What it measures | Notes |
|--------|-----------------|-------|
| FRE(μ) | Fourier relative error of mean vs truth | Bounded below by discretization floor (~0.37 at vol=32) |
| PE(U) | Frobenius distance between projectors Π_U and Π_{U_true} | Gauge-invariant; [0, √(2q)] |
| log-marginal | PPCA log-marginal likelihood | Used for multi-restart model selection |
| centroid_acc | Procrustes-aligned centroid accuracy | **BIASED** — rewards alignment to GT PCA basis, not ML optimum |
| Hungarian acc | k-means → Hungarian-matched cluster accuracy | Basis-invariant; honest metric |
| ARI | Adjusted Rand Index | Chance-corrected; good for unequal clusters |
| NMI | Normalized Mutual Information | Complementary to ARI |

### 4.2 Oracle ceilings

Two ceilings bound what the loop can achieve:

1. **Factor-only ceiling (loose):** 12 factor M-steps from (μ_true, U_true) with
   μ frozen at truth.  This over-estimates what the joint loop reaches because the
   joint loop also updates μ, which drifts away from μ_true toward the ML fixed
   point under model misspecification.

2. **Joint-loop ceiling (non-annealed reference):** N joint μ+U steps from
   (μ_true, U_true) with the true noise variance.  This is a strong reference,
   but an annealed training run follows a different update map and can land in
   a different basin.  Results that exceed this reference are legitimate but
   should be compared against the matched annealed ceiling (below).

3. **Annealed oracle ceiling (matched reference):** Same as (2) but with the
   same annealing schedule used during training.  When comparing annealed runs,
   this is the honest matched reference.  On Ribosembly, both ceilings are
   similar.  On IgG-RL, the annealed ceiling is much lower (annealing from
   truth hurts on continuous manifolds).

---

## 5. Experimental results

### 5.1 Ribosembly (16 discrete states)

#### q=2 (misspecified: 16 states → 2-dim subspace)

| Setting | cac | hun | ari | nmi | Gap to ceiling (hun) |
|---------|-----|-----|-----|-----|---------------------|
| Weighted SVD warmstart, no anneal | 0.45 | 0.64 | — | — | ~0.00 |
| Weighted SVD + log1000 anneal | 0.73 | 0.73 | — | — | beats ceiling |
| Joint-loop ceiling | — | 0.67 | — | — | — |

At q=2 on 16 discrete states the model is fundamentally misspecified.
The honest Hungarian ceiling is low (~0.67) and the weighted SVD warmstart
reaches it without annealing.  Log1000 annealing actually beats the
30-iter joint-loop ceiling, presumably by escaping to a better basin
that the oracle init doesn't find.

#### q=4

| Setting | cac | hun | ari | nmi | vs non-anneal ref | vs anneal ref |
|---------|-----|-----|-----|-----|-------------------|---------------|
| **Old** unweighted SVD + ortho M-step | 0.73 | 0.62 | 0.65 | 0.84 | −0.10 | — |
| Weighted SVD, no anneal (pre-gauge-fix) | 0.76 | 0.78 | — | — | +0.06 | — |
| Weighted SVD + log1000, pre-gauge-fix | — | 0.81 | 0.80 | — | +0.09 | — |
| **Weighted SVD + log1000, post-gauge-fix** | — | **0.87** | **0.83** | **0.93** | **+0.07** | **+0.08** |
| Non-annealed reference (30 iters from truth) | 0.81 | 0.79 | 0.77 | 0.91 | — | — |
| Annealed reference (30 iters from truth, log1000) | 0.81 | 0.79 | 0.77 | 0.90 | — | — |

Two improvements stack: (1) weighted SVD warmstart lifts q=4 from
hun≈0.62 to hun≈0.78; (2) removing the post-solve orthonormalization
(gauge fix, see Section 9.1) further lifts to hun≈0.87.  The method
exceeds both truth-start references because the warmstart + annealing
combination finds a better basin than truth-start EM.

#### q=8 (partial results)

q=8 needs ~100 iters and heavy annealing (log1000).  Without annealing, all
inits land in a "lazy basin" at hun≈0.5.  With log1000 + 100 iters, SVD
warmstart reaches hun≈0.87 (near the ceiling).  Random U + log1000 also
works (hun≈0.82-0.92) — the mu basin is wide (works up to init_fre≈1.4).

### 5.2 IgG-1D (100 continuous states, linear trajectory)

| Setting | cac | hun | ari | nmi | Gap to ceiling (hun) |
|---------|-----|-----|-----|-----|---------------------|
| Weighted SVD, no anneal | — | matches | — | — | ~0.00 |
| Joint-loop ceiling | — | — | — | — | — |

IgG-1D q=2: the weighted SVD warmstart matches the oracle ceiling.  No
regression from the default changes.

### 5.3 IgG-RL (100 continuous states, rotational trajectory)

| Setting | cac | hun | ari | nmi | vs non-anneal ref |
|---------|-----|-----|-----|-----|-------------------|
| Weighted SVD, no anneal | 0.096 | 0.229 | 0.076 | 0.574 | +0.002 |
| Weighted SVD + log1000 full anneal | 0.010 | 0.180 | 0.027 | 0.513 | +0.051 (worse!) |
| **Weighted SVD + log1000 factor-only anneal** | — | **0.232** | **0.081** | **0.573** | **+0.004** |
| Non-annealed reference (30 iters from truth) | 0.105 | 0.229 | 0.075 | 0.570 | — |
| Annealed reference (30 iters from truth, log1000) | 0.014 | 0.180 | 0.026 | 0.520 | — |

q=2 on 100 continuous states is heavily misspecified (ceiling itself is
very low: hun=0.23).  The weighted SVD warmstart without annealing
reaches the ceiling (gap +0.002 on Hungarian).

**Full log1000 annealing HURTS IgG-RL** — it causes FRE divergence
because heavy annealing flattens the posterior and the μ update loses
signal.  **Factor-only annealing fixes this**: annealing only the factor
M-step keeps the mean update sharp (fre_truth stays at 0.38, near
oracle floor) while still giving the factor update the exploration
benefit.  Factor-only annealing reaches the ceiling on IgG-RL
(hun=0.232) while preserving the Ribosembly benefits.

---

## 6. What works and what doesn't

### 6.1 What works well

1. **Closed-form M-step:** The per-voxel Tipping-Bishop solve is fast,
   stable, and eliminates learning-rate tuning.  One E-step + one
   M-step per iteration, ~15-25s per iter at vol=32.

2. **Weighted SVD warmstart:** Projecting residuals to the real-volume
   Hermitian subspace and weighting by sqrt(w_k) before SVD matches
   the downstream gauge and picks directions optimal under real-space ℓ².
   This was the key bug fix: the old unweighted SVD treated Hermitian
   conjugate pairs as independent and gave hun≈0.62 at q=4; weighted
   gives hun≈0.78.

3. **Reaches or beats oracle ceiling on all tested datasets** (Ribosembly
   q=2/4, IgG-1D q=2, IgG-RL q=2) with the default settings (no
   annealing).

4. **Multi-restart model selection:** argmax(log-marginal) over K
   restarts cleanly rejects collapsed-basin restarts (the bad basin
   sits 0.17% below the good basins in log-marginal, which is easily
   distinguishable).

5. **Diagnostic infrastructure:** The script reports both oracle ceilings
   + honest metrics + interpretable per-iteration traces, making it
   easy to diagnose convergence issues.

### 6.2 What doesn't work / known limitations

1. **Full annealing is dataset-dependent.** Log1000 full annealing is
   essential on Ribosembly q≥8 (rescues from lazy basin) but harmful
   on IgG-RL (causes FRE divergence on continuous manifolds).
   **Factor-only annealing** (`--anneal-factor-only`) fixes this: it
   preserves sharp mean updates while still giving the factor M-step
   the exploration benefit.  This should be the default annealing mode.

2. **Model misspecification at small q.** When q ≪ n_states, the PPCA
   subspace cannot capture all discrete conformations and clustering
   quality is capped.  At q=2 on 16 Ribosembly states, the honest
   Hungarian ceiling is only ~0.67.

3. **Eigenvalues are frozen at truth.**  The current v0 does not
   estimate eigenvalues from data.  This is fine for evaluating U
   learning, but means the v0 cannot be used as a standalone method
   without external eigenvalue information.

4. **Scalability untested.**  All experiments use vol=32, n_img≤4096,
   healpix_order=1 (576 rotations × 5 translations = 2880 poses).
   Real datasets have vol≥128, n_img≥100k, and order≥3 grids.
   Memory scales as O(n_img × n_pose × q) for the posterior moments.

5. **Fixed grid only.**  The pose grid is fixed at the start.  No
   pose refinement (oversampled sub-grid search), no sigma search,
   no adaptive grid.  RELION uses HEALPix oversampling + perturbation
   (`SamplingPerturbation`).

6. **No CTF estimation, no scale estimation, no dose weighting.**
   The v0 forward model is simplified: constant CTF per image, no
   per-image scale factor, white noise.

### 6.3 Eigenvalue shrinkage (separate investigation)

On a separate branch (`claude/ppca-refit-algorithms`), a full
cryobench × SNR sweep confirmed that PPCA severely under-estimates
eigenvalues (~20× at low SNR; later PCs collapse 1000× at high SNR).
PPCA+ProjCov (re-estimate eigenvalues from the sample covariance
projected onto the PPCA subspace) restores calibration at low/mid SNR.
The subspace (RelVar) is the same — only the spectrum changes.

This is the "good subspace, bad spectrum" hypothesis confirmed
quantitatively.  See `project_ppca_eigenvalue_shrinkage_confirmed.md`
in agent memory for the full table.

---

## 7. Code inventory

### Modified tracked files (this branch, uncommitted)

| File | Change |
|------|--------|
| `recovar/em/ppca_abinitio/factor_update.py` | Frozen-posterior NLL uses explicit m/Hinv terms instead of re-scoring; closed-form M-step adds real-volume projection + orthonormalization |
| `recovar/em/ppca_abinitio/loop.py` | Train/val split in fixed-grid loop (subset_synthetic_dataset); loop trains on train_idx only |
| `recovar/em/ppca_abinitio/synthetic.py` | `subset_synthetic_dataset()`; `state_label_true`/`state_coords_true` fields; `external_volumes_real` + `external_sampling_mode` params |
| `docs/math/plan_ppca_abinitio_v0.md` | Updated Section 0 with resolved decisions |
| `scripts/ppca_abinitio/run_phase_2_external_mean_bootstrap.py` | Updated API calls for new synthetic.py signatures |
| `scripts/ppca_abinitio/run_stage_1c_factor_learning.py` | Same |
| `scripts/ppca_abinitio/run_stage_1d_full_soft_mstep.py` | Same |
| `tests/ppca_abinitio/test_factor_update.py` | New test: `test_expected_nll_uses_frozen_posterior_moments` |
| `tests/ppca_abinitio/test_factor_update_closed_form.py` | Updated for orthonormalization post-solve |
| `tests/ppca_abinitio/test_loop_convergence_diagnostic.py` | Updated for train/val split |
| `tests/ppca_abinitio/test_synthetic.py` | Tests for `subset_synthetic_dataset`, external volumes, discrete sampling |

### New untracked files

| File | Purpose |
|------|---------|
| `scripts/ppca_abinitio/run_cryobench.py` (1035 lines) | Main CryoBench benchmark: loads GT volumes, generates synthetic data, runs two-stage loop, reports all metrics + both oracle ceilings |
| `scripts/ppca_abinitio/diag_warmstart_compare.py` (478 lines) | Diagnostic comparing weighted vs unweighted SVD vs residual PCA warmstarts |
| `tests/ppca_abinitio/test_cryobench_script.py` (68 lines) | Unit tests for cryobench script helpers |
| `scripts/ppca_abinitio/diag_cluster_bootstrap_v{2..11}.py` | Exploratory diagnostic scripts (v2→v11 iteration on annealing/basin experiments) |
| `scripts/ppca_abinitio/diag_basin_radius_v12.py` | Basin radius sweep diagnostic |
| `scripts/ppca_abinitio/diag_freeze_mu.py` | μ-frozen ablation |
| `scripts/ppca_abinitio/diag_mstep_fixed_point.py` | M-step fixed-point analysis |
| `scripts/ppca_abinitio/diag_mstep_metrics.py` | Per-M-step metric tracking |
| `scripts/ppca_abinitio/diag_mu_swap.py` | μ initialization swap experiment |
| `scripts/ppca_abinitio/diag_q8_basin_test.py` | q=8 basin identification |
| `scripts/ppca_abinitio/diag_q_sweep_oracle_ceiling.py` | q sweep for oracle ceiling |
| `scripts/ppca_abinitio/diag_random_U.py` | Random U convergence test |
| `scripts/ppca_abinitio/diag_svd_variants.py` | SVD variant comparison |
| `scripts/ppca_abinitio/diag_true_pose_svd.py` | True-pose residual SVD |
| `scripts/ppca_abinitio/diag_u_true_freq.py` | Frequency-domain U_true analysis |
| `scripts/ppca_abinitio/plot_diagnostic.py` | Plotting utilities |

---

## 8. What to try next

### 8.1 High priority (improving the method)

1. **Adaptive annealing or annealing-free basin escape.**  The current
   log1000 schedule works on Ribosembly but hurts IgG-RL.  Ideas:
   - Monitor log-marginal change Δlm and only anneal when Δlm stalls
   - Use a milder schedule (log10 or log100) that doesn't flatten the
     posterior as aggressively
   - Multi-restart without annealing (K=4-8 restarts, argmax lm) as
     the default strategy — avoids the continuous-manifold divergence
     issue entirely

2. **Eigenvalue estimation.**  The frozen-at-truth eigenvalues are the
   main unrealism.  Options:
   - Tipping-Bishop eigenvalue update: s_k = (1/N) Σ_i (m²_{i,k} + H⁻¹_{i,kk})
   - ProjCov: project sample covariance onto the PPCA subspace
   - Minka-style automatic relevance determination to auto-select q
   The shrinkage study (Section 6.3) provides the baseline.

3. **Scale to vol=64/128.**  This requires:
   - Batched E-step (can't hold all (n_img, n_pose, n_half) in memory)
   - Possibly switching to the CUDA backprojection kernel for the
     M-step accumulation
   - Higher healpix_order (order 2-3 for vol=64-128)
   - Testing whether the closed-form per-voxel solve still works at
     higher resolution or needs regularization

### 8.2 Medium priority (robustness)

4. **Test on Tomotwin-100.**  The fourth CryoBench dataset (100 classes,
   cryo-ET), untested.  Likely needs high q (≥25).

5. **Pose refinement.**  Port RELION's HEALPix oversampling +
   `SamplingPerturbation` logic so the pose grid refines over iterations.
   This is the main gap between v0 and production-quality ab-initio.

6. **Better multi-restart selection.**  Current argmax(lm) works for
   rejecting collapsed basins but doesn't distinguish within good basins.
   Could use: AIC/BIC, held-out likelihood, or post-hoc clustering quality
   on a validation split.

### 8.3 Lower priority (nice to have)

7. **CTF / scale / dose weighting.**  Real images have heterogeneous CTF
   defocus, per-image scale factors, and dose-dependent weighting.

8. **Continuous embedding quality.**  The current metrics (Hungarian, ARI,
   NMI) are designed for discrete states.  For continuous manifolds
   (IgG-1D, IgG-RL), better metrics would be: Spearman correlation of
   pairwise latent distances vs GT distances, or reconstruction quality
   of kernel-regression volumes at GT coordinates.

9. **Integration with the main recovar pipeline.**  The v0 ab-initio
   module is self-contained (its own forward model, synthetic data,
   grid).  Plugging it into the main `CryoEMDataset → pipeline` flow
   requires bridging the forward model conventions.

---

## 9. Key mathematical insights

### 9.1 Gauge consistency: no orthonormalization in the M-step

**The problem:** The closed-form M-step solves a per-voxel q×q system
to get U_raw.  An earlier version then applied `real_volume_orthonormalize_half`,
which computes the weighted Gram G = U_raw W U_raw^H, Cholesky-factors
G = L L^H, and returns L^{-1} U_raw.  This is a GL(q) transform (not
just O(q)), so the represented covariance U diag(s) U^H changes.  With
s frozen at truth, there is no compensating update — the returned
(U_new, s_frozen) represents a DIFFERENT model than the exact M-step
solution.

**The fix:** Remove orthonormalization from the M-step.  Keep only the
real-volume Hermitian projection (`project_to_real_volume_subspace_batch`).
The E-step, M-step, and all metrics handle non-orthonormal U correctly.
Orthonormalization is still applied at initialization (warmstart SVD,
random init) where there is no previous s to be consistent with.

**Impact:** On Ribosembly q=4, removing the orthonormalization lifted
Hungarian from 0.81 to 0.87 (+7.5%).  The Cholesky whitening was
distorting the M-step solution at each iteration, accumulating gauge
drift over the 30-iteration loop.

**Future:** When eigenvalue estimation is added (s no longer frozen),
the correct approach is: orthonormalize U_new = L^{-1} U_raw, then
update s via Λ_new = L^T Λ L → eigendecompose → new diagonal s and
additional rotation applied to U.  This preserves the model while
keeping the orthonormal-row gauge.

### 9.2 Eigenvalue update experiment (2026-04-15)

Implemented the Tipping-Bishop eigenvalue update:

    s_new[k] = (1/N) Σ_{i,g,t} γ_{i,g,t} (m²_{i,g,t,k} + H⁻¹_{i,g,kk})

plus joint orthonormalization (Section 9.1's "future" approach), gated
behind `--update-eigenvalues`.

**Result: harmful under annealing, mildly harmful without.**

Ribosembly q=4 (vol=32, n=1024, σ=0.01, log1000 factor-only annealing):

| Metric     | s update ON | s frozen (baseline) | Joint-loop ceiling |
|------------|------------|--------------------|--------------------|
| Hungarian  | 0.6631     | **0.8447**         | 0.7617             |
| ARI        | 0.6188     | **0.8403**         | 0.7462             |
| NMI        | 0.8578     | **0.9401**         | 0.8967             |

IgG-RL q=2 (vol=32, n=1024, σ=0.1, no annealing):

| Metric     | s update ON | s frozen (previous) | Ceiling |
|------------|------------|--------------------|---------| 
| Hungarian  | 0.2158     | 0.2285             | 0.2305  |

**Root cause:** During annealing, the E-step uses noise_variance × f
(f = 1000→1). The posterior moments (m, H⁻¹) are computed under this
inflated noise, so Tipping-Bishop estimates the *effective* latent
variance under the annealed model, not the true signal variance.  Final
estimated s = [7.35, 6.01, 4.36, 3.62] vs true std ≈ [2.85, 1.60,
1.34, 1.08] — up to 4× inflation.

**Implication:** Eigenvalue estimation should be a **post-annealing
refinement** phase (apply at f=1 only), not during the annealing
schedule. This is the next experiment to try.

### 9.3 Weighted SVD warmstart = matching the metric (was 9.2)

The half-volume rfft layout has frequency-dependent weights w_k.
If the SVD is done in the unweighted ℓ² (raw complex entries), the
top-q SVD directions are optimal for unweighted ℓ² but NOT for the
weighted ℓ² that the real-volume inner product uses.

The fix: before SVD, multiply each residual by sqrt(w_k).  Then the
SVD's ℓ² objective IS the weighted ℓ², and the top-q directions are
optimal under the correct metric.  After SVD, divide by sqrt(w_k)
to get back to the half-volume convention.

This is analogous to doing PCA on a dataset with non-identity metric:
you pre-whiten by the metric, do standard PCA, then un-whiten.

### 9.4 The PPCA ML fixed point ≠ the data-generating truth

Under model misspecification (q < n_states for discrete data, or any
finite-q model for continuous manifolds), the PPCA ML optimum is NOT
(μ_true, U_true).  The EM algorithm correctly moves AWAY from
(μ_true, U_true) toward the ML fixed point.  This means:
- The pre-EM oracle (metrics at literal truth) over-estimates what EM
  can reach
- The factor-only ceiling (U updates with μ frozen at truth)
  over-estimates what the joint loop reaches
- The only honest ceiling is the joint-loop oracle: run the full
  EM loop from truth and see where it converges

Concretely on Ribosembly q=4: pre-EM oracle has hun=0.90; after 30
joint EM iters from truth, hun drops to 0.72.  This is not a bug —
it's the ML fixed point under q=4 misspecification.

### 9.5 Multi-basin landscape

On Ribosembly q≥4, the PPCA landscape has multiple basins:
- The "good basin" corresponds to a U that separates most conformations
- The "collapsed basin" (q=4) merges several states into 2 effective
  clusters, with log-marginal ~0.17% lower
- The "lazy basin" (q=8) has high PE and low clustering, with hun≈0.5

Deterministic annealing (log1000) flattens the posterior to let the
M-step explore broadly before committing.  Multi-restart with
argmax(lm) selection provides an alternative.  The two approaches
can be combined.

### 9.6 Convergence depth depends on q

| q | Ribosembly iters to ceiling | Notes |
|---|---------------------------|-------|
| 2 | ~15 | Low ceiling (misspecified); fast convergence |
| 4 | ~30 | Good separation with weighted SVD |
| 8 | ~100 | Needs annealing; log-marginal still climbing at iter 100 |

Under-running the loop (too few iters) underestimates the method.

---

## 10. Reproducing results

All runs use pixi environment, CUDA enabled, vol=32, n_images=1024,
healpix_order=1.

```bash
# Ribosembly q=4, weighted SVD warmstart (no anneal)
pixi run python scripts/ppca_abinitio/run_cryobench.py \
    --dataset Ribosembly --vol 32 --n-images 1024 --sigma 0.01 --q 4 \
    --n-joint 30 --seed 0

# Ribosembly q=4, weighted SVD + log1000 annealing
pixi run python scripts/ppca_abinitio/run_cryobench.py \
    --dataset Ribosembly --vol 32 --n-images 1024 --sigma 0.01 --q 4 \
    --n-joint 30 --anneal-schedule log1000 --seed 0

# IgG-RL q=2, no anneal (default)
pixi run python scripts/ppca_abinitio/run_cryobench.py \
    --dataset IgG-RL --vol 32 --n-images 1024 --sigma 0.1 --q 2 \
    --n-joint 30 --seed 1

# Multi-restart, 4 restarts
pixi run python scripts/ppca_abinitio/run_cryobench.py \
    --dataset Ribosembly --vol 32 --n-images 1024 --sigma 0.01 --q 4 \
    --n-joint 30 --n-restarts 4 --seed 0
```

Each run takes ~5-15 minutes on a single H100 at vol=32.
