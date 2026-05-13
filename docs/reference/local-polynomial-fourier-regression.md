# Local Polynomial Fourier Regression

This note describes the experimental `local_poly` kernel-regression mode.  It is
intended to be compared against the existing `standard` kernel estimator and the
one-dimensional `deconvolved` kernel estimator without changing either path.

## Goal

For a target latent coordinate `z_star`, reconstruct the Fourier volume at that
coordinate while accounting for uncertainty in each observed one-dimensional
latent coordinate.  The image model is

```text
y_i = A_i V(x_i) + eps_i,
eps_i ~ complex Normal(0, N_i),
x_i | z_i ~ Normal(z_i, sigma_i^2),
sigma_i^2 = 1 / p_i.
```

`A_i` is RECOVAR's existing projection, CTF, pose, and translation operator, and
`p_i` is the `latent_precision_noreg` value from the zdim-1 embedding.  The
method integrates over `x_i | z_i`; it does not invert latent noise and does not
introduce signed deconvolution weights.

## Local Model

For one bandwidth `h`, use polynomial scale `s = h` and approximate each Fourier
voxel independently:

```text
V(k, x) ~= sum_{r=0}^d theta_r(k) phi_r((x - z_star) / h),
phi_r(t) = t^r / r!.
```

The desired target estimate is `theta_0(k)`.  Localization uses a positive
Gaussian window

```text
omega_h(x; z_star) = exp(-(x - z_star)^2 / (2 h^2)).
```

The fitted objective is the posterior expected weighted least-squares problem:

```text
min_theta sum_i E_{x | z_i} [
    omega_h(x; z_star)
    || y_i - A_i sum_r theta_r phi_r((x - z_star) / h) ||_{N_i^{-1}}^2
] + Fourier regularization.
```

This yields positive posterior-window moments

```text
m_ir  = E[omega_h(x; z_star) phi_r((x - z_star) / h)]
M_irs = E[omega_h(x; z_star) phi_r((x - z_star) / h)
                         phi_s((x - z_star) / h)].
```

RECOVAR then accumulates Fourier normal equations with its existing weighted
half-spectrum backprojection:

```text
rhs_r(k)  = sum_i m_ir  A_i^* N_i^{-1} y_i
lhs_rs(k) = sum_i M_irs A_i^* N_i^{-1} A_i.
```

For every Fourier voxel, solve the small system

```text
(lhs(k) + rho(k) I) theta(k) = rhs(k)
```

and pass `theta_0(k)` through the same post-division RELION-style iDFT, crop,
mask, and gridding correction used by standard kernel reconstruction.

## Closed-Form 1D Moments

Let

```text
diff_i = z_i - z_star,
var_i  = 1 / p_i.
```

For finite positive precision, the posterior-window normalizer is

```text
alpha_i = h / sqrt(h^2 + var_i)
          * exp(-0.5 * diff_i^2 / (h^2 + var_i)).
```

Under the product of the latent posterior and Gaussian window,

```text
x - z_star ~ Normal(mu_i, tau_i^2),
mu_i       = diff_i * h^2 / (h^2 + var_i),
tau_i^2    = var_i * h^2 / (h^2 + var_i).
```

For `T = (x - z_star) / h`,

```text
m_ir  = alpha_i E[T^r]     / r!
M_irs = alpha_i E[T^(r+s)] / (r! s!).
```

Degrees up to 4 only require Gaussian raw moments through order 8, computed by
the recurrence

```text
E[T^0] = 1
E[T^1] = mean(T)
E[T^n] = mean(T) E[T^(n-1)] + (n-1) var(T) E[T^(n-2)].
```

When latent noise goes to zero, `alpha_i` becomes the ordinary Gaussian window
weight at the observed coordinate and the moments reduce to deterministic local
polynomial features.  When `degree = 0`, `m_i0 = M_i00 = alpha_i`, so the
estimator is noise-aware Gaussian kernel regression.

## Bandwidth Grid

The default polynomial degree is 3.  The default multiplier grid is

```python
[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
```

For each target, compute `sigma_ref = median_i sqrt(1 / p_i)` and `r_min`, the
distance from the target to the `n_min_particles`-th closest observed latent
coordinate across both halfsets.  The minimum bandwidth is

```text
h_min = max(1.25 * sigma_ref, r_min)
```

and candidates are `h_l = h_min * multiplier_l`.  This keeps the smallest
candidate from being dominated by too few particles or by latent uncertainty.

## Comparison Protocol

Use the deconvolved-kernel regression protocol in
`docs/reference/deconvolved-kernel-regression-testing.md`, but add a third
candidate run:

```bash
pixi run python -m recovar.commands.compute_state \
  /path/to/pipeline_gtpc0_scaled_meansigma_seedYYYYMMDD \
  -o /path/to/output_local_poly_degree3 \
  --latent-points /path/to/target_latent_point_true_state32.txt \
  --zdim1 \
  --save-all-estimates \
  --kernel-regression-mode local_poly \
  --local-poly-degree 3 \
  --local-poly-bandwidth-multipliers 1,1.5,2,3,4,6,8,12
```

For the small development comparison, use grid size 128, noise level 10, and the
tight focus mask for local error.  The local ball used in the older 256-grid
experiments is too small for stable local-error estimates at 128.
