# Masked M-step solver decisions

Summary of the solver comparison study (2026-04-01) and rationale for
the chosen defaults.

## Decision

**Default solver: hard mask CG, float32, no preconditioner.**

- Naive per-voxel solve kept as a fast fallback option.
- Soft mask not used by default (extra parameter μ, worse CG stability).

## Study setup

- 128³, q=10, 50k images, 20 EM iterations
- B-factored synthetic dataset (Bfac=60, noise=1)
- GT mean, GT variance prior, GT moving mask
- Metric: RelVar (relative variance explained by top PCs vs GT)

## Key findings

### 1. Prior must act on KV, not V

The gridding kernel K appears in both data and prior terms of the
objective.  The correct operator is

    K · iFFT[(A + Λ) · FFT[K · V]]

An earlier version applied the prior to V directly (`iFFT[Λ·FFT[V]]`),
which broke the equivalence with naive and caused masked solvers to
underperform by ~10% RelVar.  See commit `ebf3855`.

### 2. Hard mask CG matches naive

With the corrected prior, hard mask CG (V=EZ reduced coordinates)
matches or slightly beats naive across precisions and tolerances:

| Method            | f64 tol=1e-6 | f64 tol=1e-4 | f32 tol=1e-4 |
|-------------------|-------------|-------------|-------------|
| naive             | 0.5176      | 0.5176      | 0.5176      |
| hard (no precond) | 0.5181      | 0.5171      | 0.5174      |
| hard+precond      | 0.5180      | 0.5180      | —           |

### 3. Preconditioner not worth it

The circulant preconditioner (scatter→FFT→(k²(A+Λ))⁻¹→iFFT→gather)
adds ~one FFT round trip per CG iteration.  Benefit is marginal:

- f64 tol=1e-4: hard=0.5171 vs hard+precond=0.5180 (+0.2%)
- f64 tol=1e-6: hard=0.5181 vs hard+precond=0.5180 (identical)

With tighter tolerance or more CG iterations, unpreconditioned CG
matches preconditioned.  Not worth the cost.

### 4. Float32 works fine for hard mask

f32 hard (0.5174) ≈ f64 hard (0.5181).  The reduced-coordinate
system is well-conditioned enough for f32.  CG converges in ~16–20
iterations at tol=1e-4.

### 5. Soft mask: slightly better quality but harder to use

Soft mask with μα² penalty can beat hard/naive at well-tuned μ:

| μ     | soft   | soft+precond |
|-------|--------|-------------|
| 0.001 | 0.5174 | 0.4993      |
| 0.01  | 0.5179 | 0.4988      |
| 0.1   | 0.5192 | 0.4999      |
| 1.0   | 0.5231 | 0.5009      |

But:
- Requires tuning μ (dataset-dependent)
- CG residuals oscillate wildly (rr spikes to 1e2 then recovers)
- f32 soft is significantly worse (0.5023) — needs f64
- Block preconditioner consistently hurts (~0.50 vs ~0.52)
- Full-grid variable is 24× larger than reduced hard-mask variable

### 6. Naive: fast but slightly worse

Naive is the fastest (~93s vs ~135s for hard) because it's a per-voxel
Fourier solve with no CG.  RelVar (0.5176) is 0.1% below hard (0.5181).
The mask is applied as post-processing, not as a constraint — so naive
doesn't enforce the support exactly.  Kept as a fallback for speed.

### 7. Grid correction handled correctly

When the CG solver has K in the operator, the EM must NOT apply
`griddingCorrect_square` again (double correction).  The solver returns
the deblurred V directly.  `use_gridding_correction=False` for CG paths.

## Slurm jobs

| Job     | Description                               |
|---------|-------------------------------------------|
| 6341162 | No-K ablation (K=1 vs K=G for all methods)|
| 6342932 | Double-K correction test                  |
| 6346360 | Prior fix verification                    |
| 6348424 | Tight tol + μ sweep                       |
| 6352042 | Float32 test                              |
