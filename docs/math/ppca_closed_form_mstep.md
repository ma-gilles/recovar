# PPCA in recovar — closed-form per-voxel M-step

## What this document covers

The standard PPCA M-step (Tipping & Bishop 1999) is closed-form: given
posterior moments from the E-step, the optimal `U` is the unique
solution of a quadratic problem and is obtained by solving one linear
system. This document explains how that closed-form is implemented
in recovar and how it adapts to the cryo-EM forward model.

It is the math reference for two pieces of code:

1. **`recovar/heterogeneity/ppca.py`** — the canonical, fixed-pose
   PPCA solver. Each image's pose is assumed known; the E-step
   produces one `(z_i, cov_z_i)` per image and the M-step is a single
   per-voxel `q×q` linear system over the dataset.

2. **`recovar/em/ppca_abinitio/factor_update.py::update_factor_closed_form`**
   — the pose-marginal extension used by the ab-initio loop. The
   E-step produces `(γ_{i,g}, m_{i,g}, H_{i,g}^{-1})` over an
   `(image, pose)` grid, and the M-step accumulates a soft sum
   weighted by `γ_{i,g}` into the same per-voxel `q×q` linear system.

The two implementations share the same math; only the outer
`Σ_i` becomes a `Σ_{i,g}` weighted by responsibilities.

## Why this matters: gradient descent is the wrong tool

PPCA is famously closed-form-solvable. Tipping & Bishop's original
1999 paper opens with the observation that EM for PPCA has a
closed-form M-step — that is the **point** of probabilistic PCA. A
PPCA implementation that uses gradient descent on `U` with a
hand-tuned learning rate is solving a quadratic problem with the
wrong tool. It will work from a near-optimum init, hit bistabilities
under bad init, and force the user to tune `lr`, line-search params,
inner-step counts, and convergence tolerances. None of that is
necessary.

The recovar `heterogeneity/ppca.py` path got this right from the
start. The `em/ppca_abinitio` v0 spec did not, and was reworked
after this document was written. See task #41 history if you need
the receipts.

## Model

Per image `i`, with pose `g_i = (rotation R_i, translation t_i)`
**known**:

```
y_i = CTF_i · S_{t_i} · A_{R_i} · (μ + U α_i) + ε_i
α_i ~ N(0, Λ),  Λ = diag(s)
ε_i ~ N(0, σ_i² I)
```

- `μ ∈ ℂ^V` is the volume mean (half-volume rfft layout for v0
  ab-initio; full-volume in `heterogeneity/ppca.py`).
- `U ∈ ℂ^{V×q}` is the factor matrix; columns are the `q` principal
  directions in volume space.
- `α_i ∈ ℝ^q` is the per-image latent coordinate.
- `A_R` is the slice operator at rotation `R` (linear interpolation
  or nearest gridpoint, depending on the discretization choice).
- `S_t` is a Fourier-domain phase shift.
- `CTF_i` and `σ_i²` are per-image and per-pixel.

For ab-initio, the pose `g_i` is **not** known and is marginalized
over a fixed grid `G` of `(R, t)` candidates with responsibilities
`γ_{i,g}`. The model is the same; the only change is the outer sum
in the M-step is over `(i, g)` with weight `γ_{i,g}`, not over `i`
alone.

## E-step (already done, this section is just notation)

Given `(μ, U, s)` from the previous iteration, the per-image
conditional posterior of `α_i` given `y_i` and pose `g` is Gaussian:

```
α_i | y_i, g ~ N(m_{i,g}, H_{i,g}^{-1})
```

with

```
H_{i,g} = Λ^{-1} + (CTF_i · S_t · A_R · U)^* (CTF_i · S_t · A_R · U) / σ_i²
        = diag(1/s) + Σ_pixel w_p · (CTF² / σ²)_p · u_proj_{i,g}^*[p] u_proj_{i,g}[p]

m_{i,g}  = H_{i,g}^{-1} · b_{i,g}
b_{i,g}  = Σ_pixel w_p · u_proj_{i,g}^*[p] · (CTF / σ²)_p · ŷ_{i,g}[p]
```

where `u_proj_{i,g}` is the slice of each `U` column through pose
`g`, `ŷ_{i,g}` is the pose-aligned image, and `w_p` is the
half-spectrum Hermitian weight for the rfft layout (1 at DC and
Nyquist, 2 elsewhere). For pose-marginal ab-initio, also compute
`γ_{i,g} ∝ p(y_i | g)` via softmax over the marginal log-likelihood.

`recovar/heterogeneity/ppca.py` calls
`embedding.get_coords_in_basis_and_contrast_3` for the E-step;
`recovar/em/ppca_abinitio/posterior.py::score_from_half_image_projections`
is the pose-marginal equivalent. Both produce
`(zs, cov_zs)` (or `(γ, m, H_inv)` for ab-initio) which are the
**only** inputs the M-step needs.

## M-step — derivation of the closed form

The expected complete-data log-likelihood, dropping `U`-independent
terms, is:

```
Q(U) = Σ_{i[,g]} γ_{i,g} · E_{α | y_i, g}[ -½ ‖y_i - CTF S A (μ + U α)‖² / σ² ]
     - ½ λ ‖U‖²                                              (ridge prior)
```

For fixed-pose PPCA the outer sum has no `g` and `γ_{i,g} ≡ 1`.

Expanding the squared norm and using the second-moment identity
`E[α α^T] = H^{-1} + m m^T = C` (real, symmetric, PSD):

```
Q(U) = const
     + Σ γ · Re tr( m_{i,g}^T · U^* · A_g^* (CTF / σ²) (y_i - CTF A_g μ) )
     - ½ Σ γ · tr( U^* · A_g^* (CTF² / σ²) A_g · U · C_{i,g} )
     - ½ λ ‖U‖²
```

This is **quadratic** in `U`. The Wirtinger gradient is

```
∂Q/∂U^* = Σ γ · A_g^* ((CTF / σ²) (y_i - CTF A_g μ)) · m_{i,g}^T
        - Σ γ · A_g^* (CTF² / σ²) A_g · U · C_{i,g}
        - λ U
```

Setting `∂Q/∂U^* = 0` gives the **U-equation**:

```
Σ γ · ( A_g^* (CTF² / σ²) A_g ) · U · C_{i,g}  +  λ U
   =  Σ γ · A_g^* ( (CTF / σ²) (y_i - CTF A_g μ) ) · m_{i,g}^T
```

This is a linear equation for `U` of the form `T(U) = R` where `T`
is a `(V·q) × (V·q)` operator (V the volume size, q the latent
dimension). It cannot be materialized as a dense matrix at any
realistic scale.

## The per-voxel direct solve (exact under nearest disc)

The operator `A_g^* (CTF² / σ²) A_g` is the back-projection of
`CTF² / σ²` through pose `g` followed by re-slicing through the
same pose. In voxel space, it is the slice-then-adjoint-slice
operator weighted per-pixel by `CTF² / σ²`.

**The v0 ab-initio path uses NEAREST discretization throughout**
(simulator + score kernel + mean update + M-step). Under nearest,
`A_g[pixel, voxel]` is binary: each pixel touches exactly one voxel.
As a consequence, the operator

```
A_g^* (CTF² / σ²) A_g  =  diag( Ψ_{i,g}[v] )       (EXACT under nearest)
Ψ_{i,g}[v] = Σ_pixel A_g[pixel, v] · (CTF_i² / σ_i²)[pixel]
           = adj_slice_g( CTF_i² / σ_i² )[v]
```

is **exactly** diagonal in the volume basis (no off-voxel coupling).
The matrix entry at voxel `v` is just the per-voxel back-projection
of CTF²/σ², a single number.

For comparison: under linear-interpolation slicing, each pixel touches
up to 8 voxels with fractional weights, so the operator has nearest-
neighbor coupling and the per-voxel diagonal version is only an
approximation (verified ~80% per-voxel relative error at vol 8 in
the v0 debugging session). The choice to keep v0 on nearest disc was
made specifically so the M-step stays a direct block solve. See the
top-of-module comment in `recovar/em/ppca_abinitio/posterior.py` for
the rationale and trade-offs.

The fixed-pose `recovar/heterogeneity/ppca.py::M_step` also uses
nearest disc — via `batch_get_nearest_gridpoint_indices` — so the
two implementations share the same exact per-voxel structure.
`HeterogeneousEMState` uses the same approach for its covariance
accumulators.

Because the operator is diagonal, the U-equation **decouples per
voxel**:

```
M[v] · U[v, :] = B[v, :]                              (for each v ∈ {1,...,V})
```

with

```
M[v]  = Σ_{i[,g]} γ_{i,g} · Ψ_{i,g}[v] · C_{i,g}    +  λ I_q       (q×q)
B[v]  = Σ_{i[,g]} γ_{i,g} · ψ_{i,g}^B[v] · m_{i,g}^T               (q-vector)

ψ_{i,g}^B[v] = ( A_g^* ( (CTF / σ²) · (y_i - CTF A_g μ) ) )[v]
```

A `q × q` linear solve at each voxel, vectorizable across all
voxels in one `vmap`. **No learning rate, no line search, no inner
loop** — one linear solve and you're done.

`recovar/heterogeneity/ppca.py` does exactly this:

```python
# M_step_batch (paraphrase)
second_moments = covariance_batch + outer(mean_batch, mean_batch)   # (n_img, q, q)
lhs[v] += second_moments * (CTF² / σ²)[pixel→v]                     # M[v]
rhs[v] += (CTF · y / σ²)[pixel→v] · mean_batch                      # B[v]

# M_step
W = batch_solve(lhs, rhs)                                            # per-voxel q×q solve
U, S, _ = svd(W); W = U @ diag(S)                                    # gauge fix
```

The pose-marginal v0 ab-initio version
(`update_factor_closed_form`) uses the same equations with the
soft sum over `(i, g)` and accumulates into the **half-volume rfft
layout** via `adjoint_slice_volume(half_volume=True)` instead of the
`grid_point_indices.at[].add()` scatter used in the fixed-pose path.

## Gauge fix

`U` is identified only up to a real `O(q)` rotation. The fixed-pose
path applies the standard **SVD orthogonalization**:

```python
U, S, _ = jnp.linalg.svd(W, full_matrices=False)
W = U @ jnp.diag(S)
```

(This actually keeps the singular values and just rotates so the
columns are orthogonal in the standard `ℓ²` inner product.)

The v0 ab-initio path applies a **real-volume gauge fix**:

```
U_band = radial_band_limit_half(U, volume_shape, k_max)
U_new  = real_volume_orthonormalize_half(U_band, weights, N_full)
```

where `real_volume_orthonormalize_half` is Cholesky-whitening of the
rfft-weighted Gram matrix and corresponds to orthonormality in the
**real-space** inner product on the decoded volumes (not the
rfft-coefficient inner product). This is what makes projector-error
metrics gauge-invariant — see
`recovar/em/ppca_abinitio/half_volume.py` for the implementation
and `tests/ppca_abinitio/test_half_volume.py` for the pinning
tests.

## Why does this avoid the bistability?

The gradient-descent factor update in the v0 ab-initio loop has two
known failure modes (documented in task #40 and the debugging
session of 2026-04-09):

1. **Bad μ poisons U.** From a perturbed-mean init the gradient
   step interprets the μ-error as heterogeneity and learns directions
   that absorb the error.
2. **Gauge fix poisons μ on the next outer iter** even when U barely
   moves.

Both failure modes come from the same underlying issue: **gradient
descent on a quadratic with a hand-tuned `lr` does not actually
minimize the quadratic in one step**. The closed-form M-step does:
each call yields the exact U-minimizer of the expected-NLL given
the current `(γ, m, H^{-1})`. There is no `lr` to tune, no inner
loop to converge, and the gauge fix is applied **once** per outer
iter (not after every gradient sub-step). The bistability simply
does not arise.

The empirical test of this claim is the random-init joint sweep at
vol 16/32/48/64 (see task #39 / #41); the closed-form path is
expected to converge from random init at all sizes, where the
gradient path diverged.

## Cross-references

- `recovar/heterogeneity/ppca.py` — fixed-pose closed-form
- `recovar/em/ppca_abinitio/factor_update.py::update_factor_closed_form`
  — pose-marginal closed-form (added in task #41)
- `recovar/em/ppca_abinitio/factor_update.py::update_factor_one_outer_step`
  — gradient version (Stage 1C, retained for parity testing only)
- `recovar/em/ppca_abinitio/posterior.py::score_from_half_image_projections`
  — E-step for pose-marginal
- `recovar/heterogeneity/embedding.py::get_coords_in_basis_and_contrast_3`
  — E-step for fixed-pose
- `recovar/em/states.py::HeterogeneousEMState` — closely related
  covariance-column path that also uses the per-voxel
  diagonal-approximation accumulators (`compute_H_B`, then
  `post_process_from_filter_v2`)

## References

- Tipping, M. & Bishop, C. (1999), "Probabilistic principal
  component analysis", *Journal of the Royal Statistical Society:
  Series B*, 61(3): 611–622. The closed-form M-step is in §3.
- Roweis, S. (1998), "EM algorithms for PCA and SPCA", *NIPS*. The
  EM derivation that recovar follows.
- Punjani et al. (2020), "3DFlex" (cryoSPARC heterogeneity), uses a
  similar per-voxel M-step structure.
