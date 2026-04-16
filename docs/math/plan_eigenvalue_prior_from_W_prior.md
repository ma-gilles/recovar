# Transferring the W-prior to an eigenvalue prior for projcov / refitB

**Status**: v2 — revised after external review that found errors in the MAP
formulas. The $G_\text{prior}$ derivation is correct; the update equations
are now fixed.

## 0. The problem: eigenvalue overestimation degrades embeddings

At low SNR (σ²=0.1), the PPCA basis vectors $U$ are adequate (pc_metric
is comparable across methods), but the eigenvalue estimates from projcov
and refitB are inflated by 3–20× compared to ground truth. This matters
because eigenvalues enter the per-image MAP embedding solver as a Wiener
regulariser: $z_i^* = (U^T C_i^T C_i U + \text{diag}(s)^{-1})^{-1}
U^T C_i^T y_i$. Overestimated $s_k$ weaken the regulariser, letting
noise leak into the embedding.

**Evidence from oracle experiment**: Feeding ground-truth projected
eigenvalues $M_\text{gt,diag} = \text{diag}(U_\text{est}^* \Sigma_\text{gt}
U_\text{est})$ into the embedding solver improves embed_metric by +37%
on Ribosembly (0.411→0.562), and lifts IgG-1D from 0→0.178. pc_metric
is unchanged because $U$ is the same. This proves the embedding gap is
entirely an eigenvalue calibration problem.

**Eigenvalue trace ratios at σ²=0.1** (sum of estimated / sum of GT top-10):

| Method | Ribosembly | IgG-1D | IgG-RL | Tomotwin-100 |
|--------|-----------|--------|--------|-------------|
| PPCA EM (100 iters) | 1.01 | 1.99 | 3.35 | 1.09 |
| ProjCov-end | 4.30 | 9.94 | 20.11 | 3.00 |
| RefitB | 2.81 | 5.09 | 9.72 | 2.00 |
| Oracle | 1.01 | 1.99 | 3.35 | 1.09 |

PPCA EM is closest to truth; projcov and refitB systematically overshoot.
The goal is to regularise projcov/refitB eigenvalues back toward a
physically reasonable scale.

## 1. Setup and notation

In PPCA we model each image as

$$y_i = C_i (\mu + W z_i) + \varepsilon_i, \qquad z_i \sim \mathcal{N}(0, I_q)$$

where $W \in \mathbb{C}^{D \times q}$ is the loading matrix ($D$ = number
of Fourier voxels, $q$ = basis size) and $C_i$ encodes CTF + projection.

### 1.1 The current W-prior

The pipeline places a diagonal Gaussian prior on each column of $W$:

$$p(W) \propto \exp\!\Bigl(-\tfrac{1}{2}\sum_{k=1}^{q} w_k^* \,\text{diag}(\tau)^{-1} w_k\Bigr) = \exp\!\Bigl(-\tfrac{1}{2}\|D^{-1} W\|_F^2\Bigr)$$

where $\tau \in \mathbb{R}^D_{>0}$ is the per-voxel prior variance
(`W_prior[:, k]` in the code, same for every $k$) and $D = \text{diag}(\sqrt{\tau})$,
so $D^2 = \text{diag}(\tau)$.

In the M-step this appears as a Tikhonov regulariser on each column of $W$
(see `recovar/ppca/ppca.py:687`, `reg_half = 1 / (W_prior_half + ε)`).

The prior $\tau$ is estimated by `estimate_hybrid_shell_prior_from_data`
(`recovar/ppca/prior_estimation.py:266`): half-set RELION-style variance →
shell average → repair high-res tails with $|\mu|^2$ fallback → broadcast
to per-voxel → divide by $q$ for per-column variance.

### 1.2 Orthogonalisation: W = U R

After EM converges (or at each projcov/refitB checkpoint), we factorise

$$W = U R, \qquad U^* U = I_q, \quad R \in \mathbb{R}^{q \times q}$$

via real-space SVD of $W$ (`_orthonormalize_W_to_basis`, ppca.py:755).
The eigenvalues reported are $s_k = \sigma_k(W)^2 \cdot \text{vol\_size}$.

**FFT normalization caveat**: `get_dft3` uses `norm="backward"` (unnormalized
forward FFT). The orthonormality $U^* U = I$ holds in the Fourier convention
where columns have unit Fourier norm (= $1/\sqrt{\text{vol\_size}}$ real-space
norm). All formulas below assume $\tau$ is defined in the same Fourier
convention as $U$. Since `W_prior` is constructed in the half-Fourier domain
and $U$ is obtained from $W$ via the same FFT, they are consistent. But any
formula involving Parseval or cross-convention products must account for the
$\text{vol\_size}$ factor.

After orthogonalisation we fix $U$ and re-estimate only the $q \times q$
matrix $B \approx R R^T$ via either:

- **ProjCov**: solves a $q^2 \times q^2$ system $\mathcal{A}\,\text{vec}(B) = \text{vec}(R)$
  from the projected empirical covariance $U^* \hat{\Sigma}_\text{emp} U$
  with noise correction (`solve_covariance` in `em/heterogeneity.py:669`).
- **RefitB**: inner EM on per-image posterior moments $(G_i, h_i)$ in span($U$),
  giving $B = \frac{1}{n} \sum_i T_i$ (`_refitb_em_steps`, ppca.py:860).

Both currently operate **without any prior on $B$** (refitB has an optional
$\kappa$-ridge but it defaults to 0 and uses $B_\text{init}$ as the center).

## 2. Transferring the W-prior to a penalty on B

### 2.1 Derivation

Substituting $W = U R$ (with $R$ real) into the W-prior:

$$\|D^{-1} W\|_F^2 = \|D^{-1} U R\|_F^2 = \text{tr}(R^T U^* D^{-2} U R)$$

Define the **projected prior precision** ($q \times q$ Hermitian PD matrix):

$$G = U^* D^{-2} U = U^* \,\text{diag}(\tau^{-1})\, U$$

Since $R$ is real and $B = R R^T$:

$$\text{tr}(R^T G R) = \text{tr}(G R R^T) = \text{tr}(G B)$$

This identity requires $R$ real. If $R$ were complex, the natural object
would be $R R^*$, not $R R^T$.

So the W-prior penalty, transferred to the $(U, B)$ parameterisation, is:

$$\boxed{-\log p(W)\big|_{U\text{ fixed}} \;\propto\; \tfrac{1}{2}\,\text{tr}(G\,B)}$$

This is a **linear penalty in $B$** (linear in the eigenvalues), not a
quadratic ridge toward $G^{-1}$. It is analogous to a nuclear-norm / trace
regulariser weighted by the projected prior precision.

### 2.2 What this is NOT

The transferred penalty $\frac{\kappa}{2}\text{tr}(GB)$ is **not** the same
as any of these:

- **Quadratic ridge toward $G^{-1}$**: $\frac{\kappa}{2}\text{tr}[(B - G^{-1})^T G (B - G^{-1})]$
  → gives $(A + \kappa G)B = R + \kappa I$ in the projcov setting.
- **Frobenius ridge toward $G^{-1}$**: $\frac{\kappa}{2}\|B - G^{-1}\|_F^2$
  → gives $(A + \kappa I)B = R + \kappa G^{-1}$.
- **Conjugate covariance prior**: $p(B) \propto |B|^{-(\nu+q+1)/2} \exp(-\frac{1}{2}\text{tr}(\Psi B^{-1}))$
  → inverse-Wishart with scale $\Psi$.

The plan uses the transferred trace penalty. Alternative priors (quadratic
ridge, conjugate) are discussed in §2.5 as options for the experimental sweep.

### 2.3 Computing $G$

```python
# U_fourier: (D, q) complex — columns are the basis in Fourier convention
# tau: (D,) real, per-voxel prior variance = W_prior[:, 0]
inv_tau = 1.0 / np.maximum(tau, 1e-12)
G = (np.conj(U_fourier) * inv_tau[:, None]).T @ U_fourier   # (q, q)
# Hermitian symmetrize, then verify real
G = 0.5 * (G + G.conj().T)
if np.max(np.abs(G.imag)) > 1e-6 * np.max(np.abs(G.real)):
    raise ValueError("G has substantial imaginary part — FFT convention bug")
G = G.real
```

### 2.4 Properties of $G$

- **Hermitian PD**: since $\tau > 0$ and $U$ has linearly independent columns.
- **Condition number**: inherits the full dynamic range of $\tau^{-1}$ across
  shells. If $\tau$ spans $10^6$, $G$ can too. Must eigendecompose $G$ and
  floor small eigenvalues before any inversion.
- **Not diagonal**: off-diagonals couple PCs whose basis vectors overlap in
  low-$\tau$ (high-frequency) Fourier regions.
- **Rank $q$**: always full-rank $q \times q$ (assuming $U$ has $q$
  independent columns).

### 2.5 Three candidate priors to test

We consider three options, all built from $G$:

| Label | Penalty on $B$ | Interpretation |
|-------|---------------|----------------|
| **Trace** | $\frac{\lambda}{2}\text{tr}(GB)$ | Exact transfer of W-prior. Linear in eigenvalues. |
| **Ridge-G** | $\frac{\kappa}{2}\text{tr}[(B-G^{-1})^T G (B-G^{-1})]$ | Quadratic pull toward $G^{-1}$ in the $G$-metric. |
| **Ridge-I** | $\frac{\kappa}{2}\|B-G^{-1}\|_F^2$ | Frobenius pull toward $G^{-1}$. |

## 3. MAP updates

### 3.1 RefitB

The refitB objective with the data log-likelihood and penalty is:

$$Q(B) = -\frac{n}{2}\log|B| - \frac{1}{2}\text{tr}(B^{-1} S) - \frac{1}{2} P(B)$$

where $S = \sum_i T_i$ (sum of posterior second moments).

**Trace penalty** $P(B) = \lambda\,\text{tr}(GB)$:

Setting $\nabla_B Q = 0$ gives:

$$S = n B + \lambda\, B\,G\,B$$

This is a matrix Riccati equation. It has a closed-form solution via
$G$-whitening:

Let $G = Q \Lambda Q^T$ (eigendecompose). Define whitened variables:
$C = \Lambda^{1/2} Q^T B Q \Lambda^{1/2}$ and $T = \Lambda^{1/2} Q^T S Q \Lambda^{1/2}$.
Then $T = n C + \lambda C^2$, so $C$ and $T$ share eigenvectors. If
$T = V\,\text{diag}(t_j)\,V^T$, then:

$$\boxed{c_j = \frac{-n + \sqrt{n^2 + 4\lambda\,t_j}}{2\lambda}, \qquad B = G^{-1/2} V\,\text{diag}(c_j)\,V^T G^{-1/2}}$$

where $G^{-1/2} = Q \Lambda^{-1/2} Q^T$ (computed via the eigendecomposition,
with eigenvalue flooring for stability).

**Ridge-G penalty** $P(B) = \kappa\,\text{tr}[(B-G^{-1})^T G (B-G^{-1})]$:

This is equivalent to adding $\kappa$ pseudo-observations with sufficient
statistic $G^{-1}$:

$$\boxed{B_\text{MAP} = \frac{S + \kappa\,G^{-1}}{n + \kappa}}$$

This is the formula that was incorrectly attributed to the trace penalty
in v1. It is the correct MAP for the ridge-G prior.

**Ridge-I penalty** $P(B) = \kappa\,\|B - G^{-1}\|_F^2$:

$$\nabla_B Q = 0 \implies B^{-1} S B^{-1} = n B^{-1} + 2\kappa(B - G^{-1})$$

This has no clean closed form. In practice, iterate or use the ridge-G
variant instead.

### 3.2 ProjCov

The projcov solve is a $q^2 \times q^2$ linear system:

$$\mathcal{A}\,\text{vec}(B) = \text{vec}(R)$$

where $\mathcal{A}$ is accumulated from per-image Kronecker products of
projected CTF terms (see `em/heterogeneity.py:669`, `solve_covariance`).

**Trace penalty** $P(B) = \lambda\,\text{tr}(GB)$:

The projcov data term (simplified) is $\frac{1}{2}\text{vec}(B)^T \mathcal{A}\,\text{vec}(B) - \text{vec}(R)^T \text{vec}(B)$. Adding $\frac{\lambda}{2}\text{tr}(GB)$ gives:

$$\nabla = \mathcal{A}\,\text{vec}(B) - \text{vec}(R) + \frac{\lambda}{2}\text{vec}(G) = 0$$

$$\boxed{\mathcal{A}\,\text{vec}(B) = \text{vec}(R) - \frac{\lambda}{2}\,\text{vec}(G)}$$

The operator $\mathcal{A}$ is unchanged; only the RHS shifts by a constant.
This is a simple one-line change in the existing solve.

**Ridge-G penalty** $P(B) = \kappa\,\text{tr}[(B-G^{-1})^T G (B-G^{-1})]$:

The gradient adds $\kappa\, G \otimes I$ to the LHS operator:

$$\boxed{(\mathcal{A} + \kappa\,(I \otimes G))\,\text{vec}(B) = \text{vec}(R) + \kappa\,\text{vec}(I)}$$

Note: the Kronecker structure of the regulariser must match the vectorisation
convention used in `solve_covariance` (column-major via `.T.reshape(-1)`).
Verify whether $I \otimes G$ or $G \otimes I$ is correct for the specific
vec convention.

**Ridge-I penalty**: $(\mathcal{A} + \kappa I_{q^2})\,\text{vec}(B) = \text{vec}(R) + \kappa\,\text{vec}(G^{-1})$.

## 4. Implementation plan

### 4.0 Simpler baselines first

Before implementing the full $G$-based prior, test two zero-code-change
controls that address the same failure mode:

1. **Existing refitB $\kappa$-ridge with $B_\text{prior} = B_\text{EM}$**:
   The code already has this hook (`_refitb_em_steps`, kappa parameter).
   Since EM eigenvalues are within 1–3× of truth, using them as the ridge
   center with $\kappa > 0$ may already help. Test $\kappa \in \{1, 10, n/10\}$.

2. **Global trace rescaling**: For each method, compute
   $s_\text{corrected} = s_\text{method} \times (\sum s_\text{EM}) / (\sum s_\text{method})$.
   This preserves the spectral shape from projcov/refitB but forces the total
   trace to match EM. No code change needed — just a post-hoc rescale.

### 4.1 Compute $G$ at each projcov/refitB checkpoint

**Where**: In `ppca.py:EM()`, after `_orthonormalize_W_to_basis` produces
`U_real`, convert to `U_fourier` and compute $G$.

**Numerical safety**: Eigendecompose $G$, floor eigenvalues at
$\epsilon \cdot \lambda_\text{max}$ (e.g. $\epsilon = 10^{-6}$), compute
$G^{-1/2}$ from the floored eigendecomposition. Never invert $G$ directly.

```python
U_fourier = ftu.get_dft3(U_real).reshape(q, vol_size).T
tau = W_prior[:, 0]
inv_tau = 1.0 / np.maximum(tau, 1e-12)
G = ((np.conj(U_fourier) * inv_tau[:, None]).T @ U_fourier)
G = 0.5 * (G + G.conj().T)
assert np.max(np.abs(G.imag)) < 1e-6 * np.max(np.abs(G.real))
G = G.real

# Stable eigendecomposition
eigvals_G, eigvecs_G = np.linalg.eigh(G)
floor = eigvals_G[-1] * 1e-6
eigvals_G = np.maximum(eigvals_G, floor)
G_inv_sqrt = eigvecs_G @ np.diag(1.0 / np.sqrt(eigvals_G)) @ eigvecs_G.T
```

### 4.2 Integrate into refitB (trace penalty)

Modify `_refitb_em_steps` to accept `G_eigdecomp` and implement the
whitened Riccati solution:

```python
def _refitb_em_steps_trace_prior(G_all, h_all, B_init, n_inner_iters=3,
                                  lam=0.0, G_eigdecomp=None, eps=1e-8):
    q = B_init.shape[0]
    n = G_all.shape[0]
    B = B_init.copy()
    for _ in range(n_inner_iters):
        B_inv = np.linalg.inv(B + eps * np.eye(q))
        P_all = np.linalg.inv(B_inv[None] + G_all)
        m_all = np.einsum("nij,nj->ni", P_all, h_all)
        T_all = P_all + np.einsum("ni,nj->nij", m_all, m_all)
        S = np.sum(T_all, axis=0)

        if lam > 0 and G_eigdecomp is not None:
            eigvals_G, eigvecs_G = G_eigdecomp
            Lsqrt = np.diag(np.sqrt(eigvals_G))
            Linvsqrt = np.diag(1.0 / np.sqrt(eigvals_G))
            Q = eigvecs_G
            # Whiten: T_w = Λ^{1/2} Q^T S Q Λ^{1/2}
            T_w = Lsqrt @ Q.T @ S @ Q @ Lsqrt
            T_w = 0.5 * (T_w + T_w.T)
            eigvals_T, V = np.linalg.eigh(T_w)
            # Riccati: c_j = (-n + sqrt(n² + 4λt_j)) / (2λ)
            c = (-n + np.sqrt(n**2 + 4 * lam * np.maximum(eigvals_T, 0))) / (2 * lam)
            c = np.maximum(c, eps)
            # Unwhiten: B = G^{-1/2} V diag(c) V^T G^{-1/2}
            G_inv_sqrt = Q @ Linvsqrt @ Q.T
            B = G_inv_sqrt @ V @ np.diag(c) @ V.T @ G_inv_sqrt
        else:
            B = S / n
        B = 0.5 * (B + B.T) + eps * np.eye(q)
    return B
```

### 4.3 Integrate into projcov (trace penalty)

The trace penalty only shifts the RHS — one line change in `solve_covariance`:

```python
def solve_covariance(lhs, rhs, G_prior=None, lam=0.0):
    def vec(X):  return X.T.reshape(-1)
    def unvec(x): n = int(np.sqrt(x.size)); return x.reshape(n, n).T

    rhs_vec = vec(rhs)
    if G_prior is not None and lam > 0:
        rhs_vec = rhs_vec - (lam / 2) * vec(G_prior)
    covar = jax.scipy.linalg.solve(lhs, rhs_vec, assume_a="pos")
    return unvec(covar)
```

### 4.4 Integrate ridge-G variant into projcov

For the ridge-G variant, the LHS operator changes. Since `lhs` is
$(q^2, q^2)$, we add $\kappa (I \otimes G)$ (or $\kappa (G \otimes I)$
depending on vec convention — must verify):

```python
if G_prior is not None and kappa > 0:
    # Ridge-G: (A + κ(I⊗G)) vec(B) = vec(R) + κ vec(I)
    lhs_reg = lhs + kappa * np.kron(np.eye(q), G_prior)
    rhs_vec = vec(rhs) + kappa * vec(np.eye(q))
    covar = jax.scipy.linalg.solve(lhs_reg, rhs_vec, assume_a="pos")
```

### 4.5 CLI flags

```
--ppca-eigenvalue-prior-mode  {none, trace, ridge-G, ridge-I}  (default: none)
--ppca-eigenvalue-prior-strength FLOAT  (default: 0)
```

The strength parameter is $\lambda$ for trace mode and $\kappa$ for ridge
modes. For the sweep, parameterise as a dimensionless ratio
$\lambda_\text{rel} = \lambda / n$ (refitB) or $\lambda_\text{rel} = \lambda / \|\mathcal{A}\|$
(projcov) to make it rank- and dataset-invariant.

## 5. Experimental plan

### 5.1 Phase 0: zero-code baselines

| Tag | Description |
|-----|-------------|
| `refitb-existing-k10` | Existing refitB $\kappa=10$ with $B_\text{prior} = B_\text{EM}$ |
| `refitb-existing-kN10` | Same with $\kappa = n/10$ |
| `trace-rescale-projcov` | Post-hoc rescale projcov $s$ to match EM trace |
| `trace-rescale-refitb` | Post-hoc rescale refitB $s$ to match EM trace |

### 5.2 Phase 1: diagnostic — inspect $G$

Compute $G$ for existing ppca-100 results at SNR={0.1, 1.0}, 4 datasets.
Report:

- Eigenvalues of $G$ (condition number, dynamic range).
- $\text{tr}(G \cdot \text{diag}(M_\text{gt}))$ vs $\text{tr}(G \cdot \text{diag}(s_\text{projcov}))$:
  does the penalty correctly penalise the inflated spectrum more?
- $\text{diag}(G)$ vs $1/s_\text{ppca}$: is $G$ roughly proportional to
  the inverse eigenvalues (which would mean the trace penalty ≈ counting
  effective dimensions)?

### 5.3 Phase 2: trace penalty sweep

4 datasets × {SNR 0.1, 1.0} × {projcov, refitB} × $\lambda_\text{rel} \in \{0.01, 0.1, 1, 10\}$.

### 5.4 Phase 3: ridge-G sweep (if trace penalty shows promise)

Same grid but with ridge-G and $\kappa_\text{rel} \in \{0.01, 0.1, 1, 10\}$.

### 5.5 Evaluation

For each method:
1. **Eigenvalue calibration**: $B_\text{est,diag}$ vs $M_\text{gt,diag}$
   and trace ratio.
2. **Downstream metrics**: pc_metric, embed_metric, cluster_metric.
3. **Comparison to oracle**: how close does the best setting get to
   `ppca-oracle-eigs`?

### 5.6 Diagonal-only fallback

If the off-diagonal structure of $G$ causes problems (basis rotation inside
the span), test $\text{diag}(G)$ only — per-PC independent shrinkage that
still respects the shell structure. This is guaranteed basis-preserving.

## 6. Files to modify

| File | Change |
|------|--------|
| `recovar/ppca/ppca.py` | Compute $G$ at projcov/refitB checkpoints |
| `recovar/ppca/ppca.py` (`_refitb_em_steps`) | Add trace-penalty Riccati solver |
| `recovar/em/heterogeneity.py` (`solve_covariance`) | Add optional $G$, $\lambda$ args |
| `recovar/commands/pipeline.py` | Add CLI flags; wire through to EM |
| `recovar/commands/pipeline.py` | Save `prior_info` to `ppca_info` in params.pkl |

## 7. Risks and open questions

1. **$G$ conditioning**: $\tau^{-1}$ can span $10^6$ across shells. Always
   work via eigendecomposition of $G$ with floored eigenvalues. Never
   invert $G$ directly or discard imaginary parts without checking magnitude.

2. **vec convention**: `solve_covariance` uses `.T.reshape(-1)` (column-major
   vec). The Kronecker regulariser $I \otimes G$ vs $G \otimes I$ must match.
   Verify with a unit test before the sweep.

3. **$\lambda$ / $\kappa$ scale invariance**: The trace penalty strength
   depends on $q$ (since $G$ itself depends on how many PCs share the
   Fourier support). Normalise by $n$ for refitB and by $\|\mathcal{A}\|$
   for projcov to get rank-invariant behaviour.

4. **Off-diagonal $G$ rotates the basis**: The full $G$ prior can rotate
   eigenvectors within span($U$). Since the oracle experiment says the
   basis is already fine, this may be undesirable. The diagonal-only
   fallback (§5.6) avoids this.

5. **Interaction with contrast**: When contrast $c_i \neq 1$, the prior
   penalty becomes $c_i^2 \|D^{-1} W\|^2$, coupling $c$ and $B$. For now
   assume $c_i$ is pre-estimated and fixed.

## 8. References

- Tipping & Bishop (1999). Probabilistic principal component analysis. JRSS-B.
  — foundational PPCA EM formulation.
- Bishop (1999). Bayesian PCA. NIPS. — priors on W, automatic relevance
  determination, variational treatment.
- Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. — linear shrinkage baseline.
- Donoho, Gavish & Johnstone (2018). Optimal shrinkage of eigenvalues.
  — clean eigenvalue shrinkage theory for spiked covariance models.
