# Transferring the W-prior to an eigenvalue prior for projcov / refitB

## 1. Setup and notation

In PPCA we model each image as

$$y_i = C_i (μ + W z_i) + ε_i, \qquad z_i \sim \mathcal{N}(0, I_q)$$

where $W \in \mathbb{C}^{D \times q}$ is the loading matrix ($D$ = number of
Fourier voxels, $q$ = basis size) and $C_i$ encodes CTF + projection.

### 1.1 The current W-prior

The pipeline places a diagonal Gaussian prior on each column of $W$:

$$p(W) \propto \exp\!\Bigl(-\tfrac{1}{2}\sum_{k=1}^{q} w_k^* \,\text{diag}(\tau)^{-1} w_k\Bigr) = \exp\!\Bigl(-\tfrac{1}{2}\|D^{-1} W\|_F^2\Bigr)$$

where $\tau \in \mathbb{R}^D_{>0}$ is the per-voxel prior variance
(`W_prior[:, k]` in the code, same for every $k$) and we define
$D = \text{diag}(\sqrt{\tau})$, so $D^2 = \text{diag}(\tau)$.

In the M-step this appears as a Tikhonov regulariser on each column of $W$
(see `ppca.py:687`, `reg_half = 1 / (W_prior_half + ε)`).

The prior $\tau$ is estimated by `estimate_hybrid_shell_prior_from_data`:
half-set RELION-style variance → shell average → repair high-res tails with
$|\mu|^2$ fallback → broadcast back to per-voxel → divide by $q$ to get
per-column variance.

### 1.2 Orthogonalisation: W = U R

After EM converges (or at each projcov/refitB checkpoint), we factorise

$$W = U R, \qquad U^* U = I_q \text{ (Fourier-orthonormal)}, \quad R \in \mathbb{R}^{q \times q}$$

via SVD of $W$ in real space (`_orthonormalize_W_to_basis`). The eigenvalues
reported are $s_k = \sigma_k(W)^2 \cdot D_\text{vol}$ (where
$D_\text{vol} = \text{vol\_size}$), i.e.\ the diagonal of $R^T R$ after
rescaling.

After orthogonalisation we fix $U$ and re-estimate only the $q \times q$
matrix $B \approx R R^T$ via either:

- **ProjCov**: $B = (\text{lhs})^{-1} \text{rhs}$ from the projected
  empirical covariance $U^* \hat{\Sigma}_\text{emp} U$ with noise correction.
- **RefitB**: inner EM on $(G_i, h_i)$ posterior moments in span($U$),
  giving $B = \frac{1}{n} \sum_i T_i$.

Both currently operate **without any prior on $B$** (refitB has an optional
$\kappa$-ridge but it defaults to 0).

## 2. Transferring the W-prior to a prior on B

### 2.1 Derivation

Substituting $W = U R$ into the W-prior:

$$\|D^{-1} W\|_F^2 = \|D^{-1} U R\|_F^2 = \text{tr}(R^* U^* D^{-2} U R)$$

Define the **projected prior precision** (a $q \times q$ matrix):

$$G_\text{prior} = U^* D^{-2} U = U^* \,\text{diag}(\tau^{-1})\, U$$

where $U \in \mathbb{C}^{D \times q}$ are the current orthonormal basis
vectors. Then the W-prior penalty in $(U, R)$ coordinates is:

$$-\log p(W) \;\propto\; \text{tr}(R^* \, G_\text{prior} \, R)$$

### 2.2 Interpretation as a prior on B

Since $B = R R^T$ (symmetric PSD), we can write:

$$\text{tr}(R^* G_\text{prior} R) = \text{tr}(G_\text{prior} \, R R^*) = \text{tr}(G_\text{prior} \, B)$$

This is the log-density kernel of a **matrix-normal / inverse-Wishart-like
prior on $B$** with precision kernel $G_\text{prior}$.

More precisely, if we parameterise the prior as an exponential-family
penalty $-\frac{\kappa}{2} \text{tr}(G_\text{prior} \, B)$ added to the
log-likelihood, the MAP estimate of $B$ becomes:

$$\boxed{B_\text{MAP} = \frac{\sum_i T_i + \kappa \, G_\text{prior}^{-1}}{n + \kappa}}$$

for **refitB**, and

$$\boxed{B_\text{MAP} = (\text{lhs} + \kappa \, G_\text{prior})^{-1}\,(\text{rhs} + \kappa \, I)}$$

for **projcov** (regularised least-squares form of $A B = R$ with the
additional penalty).

### 2.3 Computing $G_\text{prior}$

$G_\text{prior}$ is cheap to compute given $U$ and $\tau$:

```python
# U_fourier: (D, q) complex, orthonormal columns
# tau: (D,) real, per-voxel prior variance = W_prior[:, 0]
inv_tau = 1.0 / np.maximum(tau, 1e-12)
G_prior = (np.conj(U_fourier) * inv_tau[:, None]).T @ U_fourier  # (q, q)
G_prior = G_prior.real  # should be real by Parseval since U is real-space orthonormal
```

This is a single matrix product — negligible cost compared to E-step or
projcov data passes.

### 2.4 Properties of $G_\text{prior}$

- **Diagonal dominance**: If $U$ columns live predominantly in the
  high-signal shells (where $\tau$ is large, so $\tau^{-1}$ is small),
  then $G_\text{prior}$ is small → weak regularisation. If $U$ reaches
  into low-signal / noise-dominated shells, those voxels contribute
  large $\tau^{-1}$ → stronger pull toward zero. This is exactly the
  desired behaviour: penalise eigenvalue inflation in poorly-constrained
  directions.

- **Not diagonal in general**: Off-diagonal terms in $G_\text{prior}$
  couple PCs whose basis vectors overlap in Fourier regions where $\tau$
  is small. This is richer than a simple per-eigenvalue shrinkage.

- **Trace constraint connection**: $\text{tr}(G_\text{prior} B) = \sum_k
  (G_\text{prior})_{kk} \, B_{kk} + \text{off-diag terms}$. When
  $G_\text{prior} \approx g \cdot I$ (isotropic), this reduces to a
  trace penalty $g \cdot \text{tr}(B)$, i.e. an L1-on-eigenvalues /
  nuclear-norm regulariser on $B$.

## 3. Implementation plan

### 3.1 Compute $G_\text{prior}$ at each projcov/refitB checkpoint

**Where**: In `ppca.py:EM()`, after `_orthonormalize_W_to_basis` produces
`U_real`, convert to `U_fourier` and compute $G_\text{prior}$.

**Input**: `W_prior` (already available in EM scope) and `U_fourier`.

```python
# After _orthonormalize_W_to_basis gives U_real, s_em, _:
U_fourier = ftu.get_dft3(U_real).reshape(q, vol_size).T  # (D, q)
tau = W_prior[:, 0]  # per-voxel prior variance (same for all PCs)
inv_tau = 1.0 / np.maximum(tau, 1e-12)
G_prior = (np.conj(U_fourier) * inv_tau[:, None]).T @ U_fourier  # (q, q)
G_prior = G_prior.real
```

### 3.2 Integrate into projcov

**Current code** (`ppca.py:1057`):
```python
refined_u, projcov_s = pca_by_projected_covariance(...)
```

`pca_by_projected_covariance` returns eigenvectors and eigenvalues of
the projected covariance solve `B = lhs⁻¹ rhs`. We need to modify
the solve to include the prior.

**Option A — post-hoc regularisation** (minimal code change):
After getting `projcov_s` (diagonal of $B$ in its eigenbasis), apply:

```python
# In the projcov eigenbasis, G_prior rotates too:
G_prior_eig = R.T @ G_prior @ R  # R = eigenvectors of B from projcov
B_diag_reg = (n * projcov_s + kappa * np.diag(np.linalg.inv(G_prior_eig))) / (n + kappa)
```

But this ignores off-diagonal coupling. Better:

**Option B — regularised solve inside projcov** (cleaner):
Modify `solve_covariance` in `heterogeneity.py:669` (or a wrapper) to
accept an optional $G_\text{prior}$ and strength $\kappa$:

```python
def solve_covariance_regularised(lhs, rhs, G_prior=None, kappa=0.0):
    # Current: B = lhs⁻¹ rhs  (solving lhs @ B = rhs)
    # New:     (lhs + κ G_prior) @ B = rhs + κ I
    if G_prior is not None and kappa > 0:
        lhs_reg = lhs + kappa * G_prior
        rhs_reg = rhs + kappa * np.eye(rhs.shape[0])
    else:
        lhs_reg, rhs_reg = lhs, rhs
    B = np.linalg.solve(lhs_reg, rhs_reg)
    return 0.5 * (B + B.T)
```

Then eigdecompose $B$ as before to get the regularised eigenvalues.

### 3.3 Integrate into refitB

**Current code** (`ppca.py:860`):
```python
def _refitb_em_steps(G_all, h_all, B_init, n_inner_iters=3, eps=1e-8,
                     B_prior=None, kappa=0.0):
    ...
    if kappa > 0:
        B = (sum_T + kappa * B_prior) / (n + kappa)
```

Currently `B_prior = B_init` (diagonal, from PPCA EM eigenvalues). Replace
with:

```python
B_prior = np.linalg.inv(G_prior)  # (q, q) — the prior covariance implied by the W-prior
B = (sum_T + kappa * B_prior) / (n + kappa)
```

This is exact MAP for the exponential-family prior
$p(B) \propto \exp(-\frac{\kappa}{2} \text{tr}(G_\text{prior} B))$.

### 3.4 Choosing $\kappa$

$\kappa$ controls how strongly the W-prior constrains the eigenvalues.

- $\kappa = 0$: current behaviour (no prior on B).
- $\kappa = 1$: one "pseudo-observation" worth of prior.
- $\kappa \sim \sqrt{n}$: moderate regularisation.
- $\kappa = n$: equal weight to data and prior.

Practical default: **$\kappa = 1$** (one pseudo-observation). This adds a
gentle pull without dominating. For the test sweep, try
$\kappa \in \{0, 1, 10, 100, n/10, n\}$.

### 3.5 New CLI flags

```
--ppca-eigenvalue-prior-kappa FLOAT   (default 0 = off)
```

When > 0, compute $G_\text{prior}$ at each projcov/refitB checkpoint and
apply the regularised solve with the given $\kappa$.

## 4. Experimental plan

### 4.0 Prerequisites

- Save `prior_info` into `params.pkl` so we don't need to recompute it
  (one-line fix in `pipeline.py`).
- Add a `--ppca-eigenvalue-prior-kappa` flag.

### 4.1 Diagnostic: inspect $G_\text{prior}$

Before running any sweep, compute $G_\text{prior}$ for existing
ppca-100 runs at SNR = {0.1, 1.0} and report:

- Eigenvalues of $G_\text{prior}$ (condition number, anisotropy).
- Diagonal elements vs PPCA eigenvalues $s_k$.
- $\text{tr}(G_\text{prior} \cdot \text{diag}(M_\text{gt}))$ vs
  $\text{tr}(G_\text{prior} \cdot \text{diag}(s_\text{ppca}))$ — does
  the prior "know" which eigenvalue profile is better?

### 4.2 Sweep: projcov + G_prior

Run on 4 datasets × {SNR 0.1, 1.0}:

| Method tag | Description |
|---|---|
| `ppca-projcov-end` | Baseline (no prior on B) |
| `ppca-projcov-Gp-k1` | projcov + $G_\text{prior}$, $\kappa=1$ |
| `ppca-projcov-Gp-k10` | projcov + $G_\text{prior}$, $\kappa=10$ |
| `ppca-projcov-Gp-k100` | projcov + $G_\text{prior}$, $\kappa=100$ |
| `ppca-projcov-Gp-kN10` | projcov + $G_\text{prior}$, $\kappa=n/10$ |

### 4.3 Sweep: refitB + G_prior

Same grid as above but for the refitB path:

| Method tag | Description |
|---|---|
| `ppca-refitb` | Baseline (no prior on B) |
| `ppca-refitb-Gp-k1` | refitB + $G_\text{prior}^{-1}$ as B_prior, $\kappa=1$ |
| `ppca-refitb-Gp-k10` | same, $\kappa=10$ |
| `ppca-refitb-Gp-k100` | same, $\kappa=100$ |
| `ppca-refitb-Gp-kN10` | same, $\kappa=n/10$ |

### 4.4 Evaluation

For each method, report:
1. **Eigenvalue calibration**: $B_\text{est,diag}$ vs $M_\text{gt,diag}$
   and trace ratio $\sum s_k / \sum M_{\text{gt},k}$.
2. **Downstream metrics**: `pc_metric`, `embed_metric`, `cluster_metric`.
3. **Comparison to oracle**: how close does the best $\kappa$ get to
   `ppca-oracle-eigs`?

### 4.5 Expected outcome

At low SNR (0.1), projcov and refitB currently overestimate eigenvalues
by 3–20×. The W-prior estimates total signal power to within 2–10× of
truth. With $\kappa$ tuned appropriately, $G_\text{prior}$ should pull
inflated eigenvalues back toward a physically reasonable scale, improving
embed_metric and cluster_metric without changing pc_metric (since $U$ is
fixed).

The key question is whether the W-prior's per-shell shape is informative
enough to improve individual eigenvalues (not just the trace), and whether
the off-diagonal structure in $G_\text{prior}$ helps or hurts.

## 5. Files to modify

| File | Change |
|---|---|
| `recovar/ppca/ppca.py` | Compute $G_\text{prior}$ at projcov/refitB checkpoints; pass to regularised solve |
| `recovar/ppca/ppca.py` (`_refitb_em_steps`) | Accept `G_prior` as alternative to `B_prior`; use $G_\text{prior}^{-1}$ |
| `recovar/heterogeneity/heterogeneity.py` (`solve_covariance`) | Add optional `G_prior, kappa` args for regularised solve |
| `recovar/commands/pipeline.py` | Add `--ppca-eigenvalue-prior-kappa` flag; wire through to EM call |
| `recovar/commands/pipeline.py` | Save `prior_info` to `ppca_info` in `params.pkl` |

## 6. Risks and open questions

1. **$G_\text{prior}$ conditioning**: If $\tau^{-1}$ varies by $10^6$
   across shells and $U$ columns span many shells, $G_\text{prior}$ may
   be ill-conditioned. May need to regularise $G_\text{prior}$ itself
   (e.g. $G_\text{prior} + \epsilon I$) or work with its eigendecomposition.

2. **Projcov RHS correction**: The clean MAP for the regularised projcov
   solve assumes the RHS correction is $\kappa I$ (identity). This is
   exact only if the prior on $R$ is row-independent with covariance
   $G_\text{prior}^{-1}$ per row. Verify this matches the W-prior
   factorisation.

3. **$G_\text{prior}$ is not diagonal**: The refitB inner EM currently
   assumes $B_\text{prior}$ is symmetric PSD. $G_\text{prior}^{-1}$
   should be PSD (since $G_\text{prior}$ is PSD) but verify numerically.

4. **Interaction with contrast estimation**: When contrast $c_i \neq 1$,
   the effective loading is $c_i W$, so the prior penalty becomes
   $\|D^{-1} c_i W\|^2 = c_i^2 \|D^{-1} W\|^2$. This couples $c$ and
   $B$ estimation. For now, assume $c_i$ is pre-estimated and fixed; the
   $G_\text{prior}$ formulation remains valid with $U$ absorbing the
   mean contrast.

5. **Diagonal approximation**: If $G_\text{prior}$ off-diagonals cause
   problems, fall back to $\text{diag}(G_\text{prior})$ — this gives a
   per-PC independent shrinkage that still respects the shell structure.
