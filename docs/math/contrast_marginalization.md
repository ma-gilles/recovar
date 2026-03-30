# Contrast Marginalization in RECOVAR's Latent Posterior Solver

*RECOVAR Development Notes --- March 2026*

This document derives the exact contrast-marginalized posterior over latent
coordinates used in RECOVAR's embedding stage. We describe three inference
modes---no contrast, profile-MAP, and marginalized contrast---and show how a
single eigendecomposition reduces the entire contrast quadrature to elementwise
arithmetic. We give the full set of posterior moments needed by PPCA/EM with
latent contrast.

---

## Table of Contents

1. [Statistical Model](#1-statistical-model)
2. [Conditional Posterior for Fixed Contrast](#2-conditional-posterior-for-fixed-contrast)
3. [Three Inference Modes](#3-three-inference-modes)
4. [Quadrature for Contrast Marginalization](#4-quadrature-for-contrast-marginalization)
5. [Spectral Reduction](#5-spectral-reduction)
6. [Posterior Moments](#6-posterior-moments)
7. [Numerical Stability](#7-numerical-stability)
8. [Scope Restriction](#8-scope-restriction)
9. [Implementation Map](#9-implementation-map)
10. [Pseudocode](#10-pseudocode)

---

## 1. Statistical Model

Work in whitened image coordinates for a single image (or a shared-contrast group). The generative model is:

$$
z \sim \mathcal{N}(0, \Lambda), \qquad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_k),
$$

$$
y \mid z, c \sim \mathcal{N}\bigl(c(m + Bz),\; I\bigr),
$$

where $y \in \mathbb{R}^p$ is the whitened image, $m \in \mathbb{R}^p$ is the whitened projected mean volume, $B \in \mathbb{R}^{p \times k}$ is the whitened projected basis, and $c > 0$ is the per-image contrast scalar.

### Sufficient statistics

All image-space information enters through six inner products computed once in RECOVAR's phase-1 forward pass:

$$
H = B^\top B, \quad g = B^\top y, \quad h = B^\top m, \quad t = y^\top m, \quad \nu = m^\top m, \quad \|y\|^2 = y^\top y.
$$

These have shapes $(k \times k)$, $(k)$, $(k)$, scalar, scalar, scalar per image (with a leading batch dimension in practice).

---

## 2. Conditional Posterior for Fixed Contrast

For fixed $c$, the negative log joint (up to constants independent of $z$ and $c$) is

$$
\Phi(z, c) = \tfrac{1}{2}\|y - c(m + Bz)\|^2 + \tfrac{1}{2} z^\top \Lambda^{-1} z - \log p(c).
$$

Expanding and collecting terms quadratic in $z$:

$$
\Phi(z, c) = \tfrac{1}{2} z^\top A(c)\, z - q(c)^\top z + \tfrac{1}{2} r(c) - \log p(c),
$$

where

$$
A(c) = \Lambda^{-1} + c^2 H,
$$

$$
q(c) = c\,(g - c\,h),
$$

$$
r(c) = \|y\|^2 - 2c\,t + c^2\,\nu.
$$

The conditional posterior is Gaussian:

$$
p(z \mid y, c) = \mathcal{N}(\mu_c, \Sigma_c), \qquad \mu_c = A(c)^{-1} q(c), \qquad \Sigma_c = A(c)^{-1}.
$$

---

## 3. Three Inference Modes

### 3.1 Mode 1: No Contrast ($c = 1$)

Set $c = 1$ deterministically:

$$
A = \Lambda^{-1} + H, \qquad q = g - h, \qquad \mu = A^{-1} q, \qquad \Sigma = A^{-1}.
$$

This is the fastest path and the correct default when contrast estimation is disabled.

### 3.2 Mode 2: Profile-MAP Contrast

RECOVAR's existing estimator jointly optimizes over $(z, c)$:

$$
\hat{c} = \arg\min_c \min_z \Phi(z, c).
$$

After analytically minimizing over $z$, the profile objective is

$$
\mathcal{J}_{\text{prof}}(c) = r(c) - q(c)^\top A(c)^{-1} q(c) - 2\log p(c).
$$

**There is no $\log\det A(c)$ term.** This is because optimizing $\min_z \Phi(z, c)$ for fixed $c$ yields $\Phi(\mu_c, c)$, which does not involve the determinant---the determinant only appears when *integrating* $z$ out.

### 3.3 Mode 3: Marginalized Contrast

For exact contrast marginalization we integrate out $z$:

$$
p(c \mid y) \propto p(c) \int p(y \mid z, c)\, p(z)\, dz.
$$

The Gaussian integral gives

$$
p(y \mid c) \propto |A(c)|^{-1/2} \exp\!\left(-\tfrac{1}{2}\bigl[r(c) - q(c)^\top A(c)^{-1} q(c)\bigr]\right),
$$

so the collapsed negative log marginal score is

$$
\boxed{\mathcal{J}_{\text{marg}}(c) = r(c) - q(c)^\top A(c)^{-1} q(c) + \log\det A(c) - 2\log p(c).}
$$

### Why $\log\det$ matters

Different contrast values $c$ yield different posterior covariance volumes $|\Sigma_c| = |A(c)|^{-1}$. When comparing marginal likelihoods across $c$, this volume factor is essential. It is:

- **irrelevant** when $c$ is fixed (Mode 1),
- **absent** in profile-MAP (Mode 2),
- **required** for marginalization (Mode 3).

The old RECOVAR code is *not* wrong---it targets a different estimator.

---

## 4. Quadrature for Contrast Marginalization

Let $\{(c_j, w_j)\}_{j=1}^C$ be a positive quadrature rule on $[c_{\min}, c_{\max}]$. The unnormalized posterior contrast weight is

$$
\tilde{\omega}_j = w_j \, p(c_j) \, |A(c_j)|^{-1/2} \exp\!\left(-\tfrac{1}{2}\bigl[r(c_j) - q(c_j)^\top A(c_j)^{-1} q(c_j)\bigr]\right),
$$

normalized as $\omega_j = \tilde{\omega}_j / \sum_\ell \tilde{\omega}_\ell$.

### Default rule

**16-point Gauss--Legendre on $[0, 3]$:**

- The integrand is smooth in $c$.
- 16 nodes achieve spectral convergence (errors $< 10^{-3}$ vs. 64-node reference, verified empirically).
- Far fewer nodes than a uniform 50-point search grid.

### Compatibility

If the user passes explicit nodes without weights, trapezoid weights are computed automatically.

### Contrast prior

If $\sigma_c^2 = \infty$, use a flat prior on $[c_{\min}, c_{\max}]$. If finite:

$$
\log p(c) = -\frac{(c - \mu_c)^2}{2\sigma_c^2} + \text{const}.
$$

---

## 5. Spectral Reduction

The key computational insight: the one-parameter family $A(c) = \Lambda^{-1} + c^2 H$ can be simultaneously diagonalized for *all* $c$ by a single eigendecomposition.

### Setup

Let $L = \Lambda^{1/2}$ and define the whitened Gram matrix

$$
G = L\, H\, L = Q\, \operatorname{diag}(d_1, \dots, d_k)\, Q^\top, \qquad d_\ell \geq 0,
$$

where $Q$ is orthogonal. Precompute:

$$
\alpha = Q^\top L\, g, \qquad \beta = Q^\top L\, h, \qquad T = L\, Q.
$$

### Per-node quantities

For each contrast node $c_j$, define:

$$
s_\ell(c) = \frac{1}{1 + c^2 d_\ell}, \qquad v_\ell(c) = c\,\alpha_\ell - c^2\,\beta_\ell, \qquad \rho_\ell(c) = s_\ell(c)\, v_\ell(c).
$$

Then:

$$
\mu_c = T\, \rho(c),
$$

$$
q(c)^\top A(c)^{-1} q(c) = \sum_{\ell=1}^k s_\ell(c)\, v_\ell(c)^2,
$$

$$
\log\det A(c) = \text{const} + \sum_{\ell=1}^k \log(1 + c^2 d_\ell).
$$

### Complexity

After the one-time $O(Bk^3)$ eigendecomposition, the entire contrast pass over $C$ nodes is $O(BCk)$ elementwise arithmetic and reductions over $[\text{batch}, C, k]$ arrays. **No $[B, C, k, k]$ tensors are materialized.**

### Why this is an optimization, not a requirement

Without the spectral trick, exact marginalization is still possible by computing a Cholesky factorization of $A(c_j)$ at each node. The eigendecomposition avoids $C$ repeated factorizations and gives 8--26x speedup on GPU:

| $B$ | $k$ | $C$ | Spectral (ms) | Cholesky (ms) | Speedup |
|-----|-----|-----|---------------|---------------|---------|
| 1,000 | 4 | 16 | 0.20 | 1.56 | 8x |
| 1,000 | 10 | 50 | 0.28 | 7.32 | 26x |
| 5,000 | 20 | 16 | 2.43 | 32.05 | 13x |
| 10,000 | 10 | 16 | 1.05 | 10.08 | 10x |
| 50,000 | 10 | 128 | 8.11 | --- | --- |

---

## 6. Posterior Moments

### 6.1 Latent moments

The marginalized posterior moments are computed as weighted sums over contrast nodes:

$$
\mathbb{E}[z \mid y] = \sum_j \omega_j\, \mu_j,
$$

$$
\mathbb{E}[zz^\top \mid y] = \sum_j \omega_j\, (\Sigma_j + \mu_j \mu_j^\top),
$$

$$
\operatorname{Cov}[z \mid y] = \mathbb{E}[zz^\top \mid y] - \mathbb{E}[z \mid y]\,\mathbb{E}[z \mid y]^\top.
$$

In the spectral basis, $\Sigma_j = T\,\operatorname{diag}(s(c_j))\,T^\top$ and $\mu_j = T\,\rho(c_j)$, so:

$$
\mathbb{E}[zz^\top \mid y] = T \left[\operatorname{diag}\!\left(\sum_j \omega_j\, s(c_j)\right) + \sum_j \omega_j\, \rho(c_j)\,\rho(c_j)^\top \right] T^\top.
$$

The matrix in brackets is $k \times k$ and built from $[B, C, k]$ reductions---no $[B, C, k, k]$ materialization.

### 6.2 Contrast moments

$$
\mathbb{E}[c \mid y] = \sum_j \omega_j\, c_j, \qquad \mathbb{E}[c^2 \mid y] = \sum_j \omega_j\, c_j^2.
$$

### 6.3 Cross moments for PPCA M-step

When the PPCA model includes contrast multiplicatively

$$
\tilde{y}_n(\xi) = C_n(\xi)\, c_n\, \bigl(\mu(\xi) + W(\xi,:)\, z_n\bigr) + \varepsilon_n(\xi),
$$

the M-step normal equations for row $w_\xi = W(\xi,:)$ are

$$
\left[\sum_n |C_n(\xi)|^2\, \mathbb{E}[c_n^2\, z_n z_n^* \mid y_n] + \operatorname{diag}(\text{Reg}(\xi))\right] w_\xi^* = \sum_n C_n(\xi)^*\, \tilde{y}_n(\xi)\, \mathbb{E}[c_n\, z_n^* \mid y_n] - \mu(\xi) \sum_n |C_n(\xi)|^2\, \mathbb{E}[c_n^2\, z_n^* \mid y_n].
$$

This requires the following contrast-weighted moments from the E-step:

$$
\mathbb{E}[c\, z \mid y] = \sum_j \omega_j\, c_j\, \mu_j,
$$

$$
\mathbb{E}[c^2\, z \mid y] = \sum_j \omega_j\, c_j^2\, \mu_j,
$$

$$
\mathbb{E}[c^2\, z z^\top \mid y] = \sum_j \omega_j\, c_j^2\, (\Sigma_j + \mu_j \mu_j^\top).
$$

### Summary of returned moments

| Field | Expression |
|-------|------------|
| `mean_z` | $\mathbb{E}[z \mid y]$ |
| `cov_z` | $\operatorname{Cov}[z \mid y]$ |
| `second_moment_z` | $\mathbb{E}[zz^\top \mid y]$ |
| `mean_c` | $\mathbb{E}[c \mid y]$ |
| `second_moment_c` | $\mathbb{E}[c^2 \mid y]$ |
| `mean_cz` | $\mathbb{E}[cz \mid y]$ |
| `mean_c2z` | $\mathbb{E}[c^2 z \mid y]$ |
| `second_moment_czz` | $\mathbb{E}[c^2 zz^\top \mid y]$ |
| `contrast_weights_posterior` | $\omega_j$ |

---

## 7. Numerical Stability

1. **Symmetrize $G$** before `eigh`: $G \leftarrow \tfrac{1}{2}(G + G^\top)$.
2. **Clip eigenvalues**: $d_\ell \leftarrow \max(d_\ell, 0)$ to handle small negative values from float32 rounding.
3. **Log-determinant via `log1p`**: $\log\det A(c) = \text{const} + \sum_\ell \log(1 + c^2 d_\ell)$ avoids catastrophic cancellation when $c^2 d_\ell \ll 1$.
4. **Logsumexp for weights**: $\omega_j = \mathrm{softmax}(\log\tilde{\omega}_j)$ rather than direct exponentiation.
5. **Symmetrize output**: $\mathbb{E}[zz^\top]$ and $\operatorname{Cov}[z]$ are explicitly symmetrized after the $T M T^\top$ transform to correct float32 asymmetry.

---

## 8. Scope Restriction

The current implementation supports exact marginalization only when there is **one contrast scalar per solve**. This covers:

- SPA per-image solves (one $c$ per image),
- Grouped shared-label solves with one shared $c$ across the group.

The case of one shared $z$ with multiple independent per-image contrasts inside a group is **not supported** and will raise `NotImplementedError`.

---

## 9. Implementation Map

| Module / Function | Role |
|-------------------|------|
| `contrast_posterior.py` | Standalone solver module |
| &nbsp;&nbsp; `make_contrast_quadrature` | Gauss--Legendre / trapezoid / custom |
| &nbsp;&nbsp; `solve_no_contrast` | $c=1$ fast path |
| &nbsp;&nbsp; `solve_profile_contrast` | Profile-MAP (no log-det) |
| &nbsp;&nbsp; `solve_marginalized_contrast` | Full marginalization |
| &nbsp;&nbsp; `solve_latent_posterior` | Dispatch wrapper |
| `embedding.py` | Integration layer |
| &nbsp;&nbsp; `_solve_batch_from_stats_v2` | Routes to solver by mode |
| &nbsp;&nbsp; `get_per_image_embedding_multi_zdim` | Accepts `contrast_mode` |

---

## 10. Pseudocode

**Algorithm: Marginalized contrast posterior (spectral form)**

**Input:** $H_{b,k,k'}$, $g_{b,k}$, $h_{b,k}$, $t_b$, $\nu_b$, $\|y\|^2_b$, $\lambda_k$, $c_j$, $w_j$

1. $L_k \gets \sqrt{\lambda_k}$ &emsp; *shape $[K]$*
2. $G_b \gets L \, H_b \, L$; symmetrize &emsp; *shape $[B, K, K]$*
3. $d_b, Q_b \gets \mathrm{eigh}(G_b)$; &ensp; $d \gets \max(d, 0)$ &emsp; *one batched eigh*
4. $\alpha_b \gets Q_b^\top (L \odot g_b)$, &ensp; $\beta_b \gets Q_b^\top (L \odot h_b)$, &ensp; $T_b \gets L \, Q_b$
5. **For each node $c_j$** (vectorized over $[B, C, K]$):
   - $s_{b,j,\ell} \gets 1/(1 + c_j^2\, d_{b,\ell})$
   - $v_{b,j,\ell} \gets c_j \alpha_{b,\ell} - c_j^2 \beta_{b,\ell}$
   - $\rho_{b,j,\ell} \gets s_{b,j,\ell}\, v_{b,j,\ell}$
6. $\text{quad}_{b,j} \gets \sum_\ell v \cdot \rho$
7. $\text{logdet}_{b,j} \gets \sum_\ell \log(1 + c_j^2 d_{b,\ell})$
8. $r_{b,j} \gets \|y\|^2_b - 2c_j t_b + c_j^2 \nu_b$
9. $\omega_{b,j} \gets \mathrm{softmax}_j\bigl(\log w_j + \log p(c_j) - \tfrac{1}{2}(r - \text{quad} + \text{logdet})\bigr)$
10. $\mathbb{E}[z]_b \gets T_b \sum_j \omega_{b,j}\, \rho_{b,j}$
11. $M_b \gets \operatorname{diag}(\sum_j \omega_{b,j}\, s_{b,j}) + \sum_j \omega_{b,j}\, \rho_{b,j}\,\rho_{b,j}^\top$ &emsp; *shape $[B, K, K]$*
12. $\mathbb{E}[zz^\top]_b \gets T_b\, M_b\, T_b^\top$; symmetrize
13. *(analogous for cross moments $\mathbb{E}[cz]$, $\mathbb{E}[c^2 zz^\top]$, etc.)*
