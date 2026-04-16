# PPCA Ab-Initio: Clean Algorithm Specification

## Model

Observation: $y_i = P_{\phi_i}(\mu + U \alpha_i) + \varepsilon_i$

where:
- $\mu \in \mathbb{C}^V$ is the consensus structure (half-volume Fourier)
- $U \in \mathbb{C}^{V \times q}$ is the factor loading matrix (columns span the heterogeneous subspace)
- $\alpha_i \sim \mathcal{N}(0, \text{diag}(s))$ are latent coordinates
- $P_\phi$ is CTF-modulated projection at pose $\phi = (R, t)$
- $\varepsilon_i \sim \mathcal{N}(0, \sigma^2 I)$ is white noise
- Poses $\phi_i$ are unknown and marginalized over a discrete grid

## Algorithm

```
Input:  images Y, CTFs, σ², rotation grid R, translation grid T,
        latent dimension q

Stage A — Mean initialization:
  if μ₀ provided (from prior homogeneous reconstruction):
    μ ← μ₀
  else:
    μ ← 0; run N_burnin homogeneous EM iterations

Stage B — Factor and eigenvalue initialization:
  Compute per-image residual backprojections at hard-argmax pose
  Apply sqrt(Hermitian weights), project to real-volume subspace
  SVD → top-q left singular vectors → U
  s ← 1  (flat; see §Eigenvalue Discussion)

Stage C — Joint EM refinement:
  for iter = 1, 2, ..., N:
    E-step:  (γ, m, H⁻¹) ← posterior(Y, μ, U, s, CTFs, σ²)
    M-step:  μ ← residualized Wiener backprojection
             U ← per-voxel closed-form solve, s frozen
    if Δ log_marginal < ε: break

Output: μ, U (s is a fixed hyperparameter, not estimated)
```

## Design Decisions

### 1. Eigenvalue initialization: s = 1 (flat)

**Why not ground truth?** Using s_true is cheating — unavailable in practice.

**Why not estimate from SVD?** The SVD singular values are in weighted
Fourier space; the mapping to model eigenvalues depends on CTFs, number
of images, and noise level. The calibration is dataset-specific and adds
a fragile step.

**Why flat works:** At typical cryo-EM SNR (σ ~ 0.01, s_true ~ 1–10),
the likelihood Gram in the E-step posterior precision H dominates the
prior precision diag(1/s) by 3–4 orders of magnitude. Setting s = 1 vs
s = s_true changes the prior contribution from ~0.1–0.8 to 1.0, while
the likelihood contributes ~600 per component. The posterior is
effectively MLE regardless.

**When it could fail:** At very low SNR (σ > 1) with few images, the
prior would matter. In that regime, a data-driven calibration (or
empirical Bayes) would be needed.

### 2. No annealing by default

Deterministic annealing (inflating σ² by factor f > 1) is a heuristic
for escaping local optima. Experiments show:

- Ribosembly q=4: no-anneal (hun=0.849) ≈ annealed (hun=0.845)
- The benefit of annealing depends on warmstart quality. With the
  weighted SVD warmstart, the algorithm starts in a good basin.

Annealing remains available as `--anneal-schedule` for hard cases
(random init, high q, low SNR) but is not part of the core algorithm.

### 3. No eigenvalue estimation during EM

The Tipping-Bishop update s_k = (1/N) Σ γ(m²_k + H⁻¹_kk) is
the MLE for standard PPCA (fully observed data). Under discrete-grid
pose marginalization, it overestimates eigenvalues because:

- Pose discretization injects additional variance into the posterior
  moments m, which the formula attributes to signal rather than
  discretization error.
- Experimentally: hun drops from 0.845 to 0.663 with s estimation.

Eigenvalue estimation is an open problem in the pose-marginalized
setting. For now, frozen s is the correct default.

### 4. Weighted SVD warmstart

The key innovation enabling a cheat-free pipeline. Weighting by
sqrt(Hermitian half-volume weights) before SVD ensures the Frobenius
norm matches the real-space ℓ² metric used in the model. This lifts
Hungarian accuracy from ~0.62 (unweighted) to ~0.78 (weighted) on
Ribosembly q=4.

### 5. Convergence criterion

Stop when Δ log_marginal < ε (e.g., ε = 0.01% of |log_marginal|).
Default: fixed 30 iterations (sufficient for q ≤ 4; increase for
larger q).

## What's NOT in the algorithm (and why)

| Removed | Reason |
|---------|--------|
| Multi-restart | Log-marginal is unreliable for restart selection (picks wrong basin) |
| Eigenvalue estimation | Tipping-Bishop formula biased under pose marginalization |
| Post-anneal s refinement | Same Tipping-Bishop bias; s diverges at f=1 too |
| Annealing (default) | Unnecessary with weighted SVD warmstart |

## Validation

Tested on 3 CryoBench datasets × 3 s-init modes (vol=32, σ=0.01).
All 9 runs used: weighted SVD warmstart, no annealing, 30 joint iters.

### Subspace recovery (projector error, lower is better)

| Dataset | q | s=truth | s=flat [1] | s=svd |
|---------|---|---------|------------|-------|
| Ribosembly (16 discrete) | 4 | **1.767** | **1.767** | **1.767** |
| IgG-1D (100 continuous) | 2 | **0.863** | **0.863** | **0.863** |
| IgG-RL (100 continuous) | 2 | **0.880** | **0.880** | **0.880** |

### Mean recovery (FRE vs oracle fixed point, lower is better)

| Dataset | s=truth | s=flat | s=svd |
|---------|---------|--------|-------|
| Ribosembly | **0.140** | **0.140** | **0.140** |
| IgG-1D | **0.015** | **0.015** | **0.015** |
| IgG-RL | **0.019** | **0.019** | **0.019** |

### Clustering (Ribosembly only — meaningful with 16 states)

| Metric | s=truth | s=flat | s=svd |
|--------|---------|--------|-------|
| Hungarian | **0.849** | **0.849** | **0.849** |
| ARI | **0.828** | **0.828** | **0.828** |
| NMI | **0.931** | **0.931** | **0.931** |

**All metrics identical across s-init modes.** The s values span 7 orders
of magnitude ([0.3] to [4×10⁶]) with zero effect on the EM trajectory.
This confirms the prior is negligible at cryo-EM SNR, and s=1 is the
correct cheat-free default.
