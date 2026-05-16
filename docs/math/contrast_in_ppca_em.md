# Contrast marginalization in the PPCA EM

## Model

Per-image forward model with contrast:

$$
y_n = c_n \, P_n(W z_n + \mu) + \varepsilon_n,
\qquad z_n \sim \mathcal{N}(0, I_K), \quad \varepsilon_n \sim \mathcal{N}(0, \sigma_n^2 I).
$$

After noise-whitening ($\tilde y = y/\sigma$, $\tilde P = P \cdot \mathrm{CTF}/\sigma$):

$$
\tilde y_n = c_n \, \tilde P_n(W z_n + \mu) + \tilde\varepsilon_n.
$$

$W$ is $(N_\mathrm{vol}, K)$ in Fourier and is kept orthogonalized as $W = U S$
after each M-step SVD, so $\Sigma = W W^* = U S^2 U^*$.

## E-step: sufficient statistics

Let $B_n = \tilde P_n W$ (noise-whitened projected basis, shape $(K, N_\mathrm{pix})$).
With Hermitian half-spectrum weights $w$, the six sufficient statistics per image are:

| Symbol | Code name | Formula | Shape |
|--------|-----------|---------|-------|
| $H_n$ | `AU_t_AU` | $\operatorname{Re}(B_n^* \operatorname{diag}(w)\, B_n)$ | $(K,K)$ |
| $g_n$ | `AU_t_images` | $\operatorname{Re}(B_n^* \operatorname{diag}(w)\, \tilde y_n)$ | $(K,)$ |
| $h_n$ | `AU_t_Amean` | $\operatorname{Re}(B_n^* \operatorname{diag}(w)\, \tilde P_n\mu)$ | $(K,)$ |
| $t_n$ | `image_T_A_mean` | $\operatorname{Re}(\tilde y_n^* \operatorname{diag}(w)\, \tilde P_n\mu)$ | scalar |
| $\nu_n$ | `A_mean_norm_sq` | $\|\tilde P_n\mu\|_w^2$ | scalar |
| $\|y\|^2_n$ | `image_norms_sq` | $\|\tilde y_n\|_w^2$ | scalar |

These are computed in `_e_step_half_inner` from the existing
noise-whitened half-spectrum quantities.

When `disc_type_mean="cubic"`, `EM_step_half` precomputes periodic cubic
B-spline coefficients once per EM step via `core.precompute_cubic_coefficients`
before `_e_step_half_inner` slices the mean. Passing raw Fourier samples
directly into cubic interpolation gives the wrong scale.

## E-step: latent posterior

Given the sufficient statistics, the per-image posterior over $(z, c)$ is
handled by `contrast_posterior.solve_latent_posterior()` with three modes:

- **none**: $c = 1$ deterministically.  Standard PPCA, no overhead.
- **profile**: MAP over $c$ via grid search (backward-compatible with RECOVAR).
- **marginalize**: exact quadrature over $c$ with the collapsed marginal score
  $J(c) = r(c) - q(c)^T A(c)^{-1} q(c) + \log\det A(c) - 2\log p(c)$,
  where $A(c) = I + c^2 H$, $q(c) = c(g - ch)$,
  $r(c) = \|y\|^2 - 2ct + c^2\nu$.

The solver returns the moments needed by the M-step:

| Moment | Used for |
|--------|----------|
| $E[z \mid y]$ | diagnostics |
| $E[cz \mid y]$ | RHS term 1 |
| $E[c^2 z \mid y]$ | RHS term 2 (mean correction) |
| $E[c^2 z z^T \mid y]$ | LHS |
| $E[c \mid y]$ | logging |

## Prior

The latent prior is $z \sim \mathcal{N}(0, I)$ always.  The signal scale
is absorbed into $W$: after SVD orthogonalization, $W = U \cdot \operatorname{diag}(S)$
and $\Sigma = W W^* = U S^2 U^*$.

The contrast solver's `lambdas` parameter is $\Lambda = I$ (the prior
covariance of $z$).  The Gram matrix $H = B^* B$ already absorbs $S$
through $W$.

**Warmup**: The first few EM iterations use random $W$, making $H \gg I$.
This causes the contrast posterior to collapse to the grid boundary.
We skip contrast estimation for the first 3 iterations
(`contrast_mode="none"`), then switch to the requested mode.

## M-step: contrast-weighted backprojection

The expected complete-data log-likelihood w.r.t. $(z_n, c_n)$ gives:

$$
\sum_n E_{z,c|y}\!\bigl[\|\tilde y_n - c_n \tilde P_n(W z_n + \mu)\|^2\bigr]
$$

Taking the gradient w.r.t. $W$ and setting to zero:

$$
\underbrace{\sum_n \tilde P_n^* \tilde P_n \otimes E[c_n^2 z_n z_n^T]}_{\text{LHS}}
\; W
=
\underbrace{\sum_n \tilde P_n^* \tilde y_n \, E[c_n z_n]^T}_{\text{RHS term 1}}
\;-\;
\underbrace{\sum_n \tilde P_n^* \tilde P_n \mu \, E[c_n^2 z_n]^T}_{\text{RHS term 2}}
$$

In the code (`E_M_step_batch_half`):

```
LHS(ξ) += backproject(CTF² ⊗ E[c²zz^T]_tri)    # CUDA fused kernel
RHS(ξ) += backproject(CTF · ỹ · E[cz]*)          # half-image adjoint
RHS(ξ) -= backproject(CTF · P̃μ · E[c²z]*)        # mean correction
```

When $c = 1$: $E[cz] = E[z]$, $E[c^2z] = E[z]$, $E[c^2zz^T] = E[zz^T]$,
and the two RHS terms combine to $\text{backproject}(\mathrm{CTF} \cdot (\tilde y - \tilde P\mu) \cdot E[z]^*)$
— identical to the original code.

## Code references

| What | Where |
|------|-------|
| Sufficient statistics | `recovar/ppca/ppca.py:_e_step_half_inner` |
| Contrast solver | `recovar/ppca/contrast_posterior.py:solve_latent_posterior` |
| Backprojection | `recovar/ppca/ppca.py:E_M_step_batch_half` |
| Cubic mean preparation | `recovar/ppca/ppca.py:_prepare_mean_estimate_for_slicing` |
| Eigenvalue / warmup | `recovar/ppca/ppca.py:EM` (loop body) |
| EM threading | `recovar/ppca/ppca.py:EM_step_half` |
| Spectral eigendecomp | `recovar/ppca/contrast_posterior.py:_spectral_decomposition` |
