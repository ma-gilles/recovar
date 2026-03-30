# Masked volume reconstruction with soft boundary penalty

## 1. The reconstruction problem (independent of EM)

Given accumulated Fourier-space weights $d(\xi) \geq 0$ and backprojected
data $\hat{r}(\xi)$ at each 3D Fourier voxel $\xi$, find a real-space
volume $w(x)$ supported on a mask $\Omega$.

This problem arises in:
- **Homogeneous reconstruction** (mean estimation): $d(\xi)$ scalar, $q = 1$
- **PPCA M-step**: $D(\xi)$ is $q \times q$ SPD, solve for $q$ columns jointly

The structure is identical — only the per-voxel size changes.

### 1.1 Forward model and gridding

The cryo-EM forward operator for image $n$:

$$A_n\,w = C_n \cdot \mathcal{S}_{R_n}\,\mathcal{F}[G \cdot w]$$

- $G(x)$: real-space blurring from trilinear Fourier-slice interpolation
- $\mathcal{F}$: 3D DFT
- $\mathcal{S}_{R_n}$: 2D slice extraction at orientation $R_n$
- $C_n$: CTF multiplication

Trilinear interpolation in Fourier space convolves the discrete spectrum
with a triangle kernel $K$ (width 1 voxel per axis). This is equivalent
to multiplying in real space by:

$$G(x) = \operatorname{sinc}^2(x_1/D)\,\operatorname{sinc}^2(x_2/D)\,\operatorname{sinc}^2(x_3/D)$$

where $D$ is the grid size and $\operatorname{sinc}(t) = \sin(\pi t)/(\pi t)$.

### 1.2 Normal equations

The adjoint of $A_n$ is "backproject, then multiply by $G$". So:

$$A^H\!A\;w = G\,\mathcal{F}^{-1}\big[d \cdot \mathcal{F}[G\,w]\big]$$

where $d(\xi)$ is the accumulated weight (e.g.\ $\sum_n |C_n(\xi)|^2 / \sigma_n^2$
for the mean, or $\sum_n |C_n(\xi)|^2 \Sigma_n$ for PPCA).

Without a mask, the per-voxel solution is $\hat{w}(\xi) = \hat{r}(\xi) / d(\xi)$,
then divide by $G$ in real space (gridding correction).

### 1.3 The masked problem

We want $w(x) \approx 0$ outside a molecular support $\Omega$.
Instead of a hard constraint (which couples all Fourier voxels and requires
an iterative solver), we use a smooth penalty:

$$\min_w \;\underbrace{\sum_\xi |\widehat{Gw}(\xi)|^2\,d(\xi)
  - 2\operatorname{Re}\,\overline{\widehat{Gw}(\xi)}\,\hat{r}(\xi)}_{\text{data fidelity with gridding}}
  \;+\;\underbrace{\lambda \sum_{x \in \Omega_S} \alpha(x)\,|w(x)|^2}_{\text{soft boundary penalty}}$$

where $\Omega_S \supset \Omega$ is a generous outer support and
$\alpha(x)$ is the smooth penalty weight.

### 1.4 Penalty weight $\alpha(x)$

Built from the binary mask via signed distance transform $d(x)$
(negative inside, positive outside):

$$\alpha(x) = \begin{cases}
  0 & d(x) < -c \\[4pt]
  \tfrac{1}{2}\big(1 + \cos(\pi\,d(x)/c)\big) & -c \leq d(x) \leq 0 \\[4pt]
  1 & d(x) > 0
\end{cases}$$

Collar width $c$ should scale with grid size: $c \approx 0.04\,D$.

Hard outer support: $\Omega_S = \text{dilate}(\Omega,\, c + 3)$.
Variables live only on $\Omega_S$ (reduced coordinates).

### 1.5 Operator and CG

The normal equations on the support:

$$\big[P_S\,G\,\mathcal{F}^{-1}[d\,\mathcal{F}[G\,E_S\,\cdot\,]]\,
  + \lambda\,\operatorname{diag}(\alpha)\big]\,w = P_S\,G\,\mathcal{F}^{-1}[\hat{r}]$$

where $E_S$: scatter to grid, $P_S$: gather from grid.

Solved by unpreconditioned CG. Each matvec costs:
scatter → $G \cdot$ → rfft3 → $d \cdot$ → irfft3 → $G \cdot$ → gather → $+\lambda\alpha w$.

For the $q > 1$ case (PPCA), replace scalar $d(\xi)$ with $q \times q$ matrix $D(\xi)$
and solve for all $q$ columns jointly. The structure is identical.

## 2. Application to PPCA

Each image: $y_n = A_n(\mu + W z_n) + \varepsilon_n$, $z_n \sim \mathcal{N}(0, I_q)$.

E-step gives $\bar{z}_n$, $\Sigma_n$. M-step accumulates:
- $D(\xi) = \sum_n |C_n(\xi)|^2 \Sigma_n + \Lambda(\xi)$ — $q \times q$ LHS
- $\hat{r}(\xi) = \sum_n C_n^*(\xi)\,\hat{y}_n^c(\xi)\,\bar{z}_n^T$ — $q$-vector RHS

Then solve the masked reconstruction problem from Section 1 with these
$D$, $\hat{r}$.

## 3. Numerical scheme

See Section 1.5 for the operator. CG pseudocode:

```
r₀ = b − A w₀
p₀ = r₀
for k = 0, …, maxiter−1:
    αₖ = ⟨rₖ, rₖ⟩ / ⟨pₖ, Apₖ⟩
    wₖ₊₁ = wₖ + αₖ pₖ
    rₖ₊₁ = rₖ − αₖ Apₖ
    if (k+1) mod 10 = 0:  rₖ₊₁ = b − Awₖ₊₁     (float32 stability)
    βₖ = ⟨rₖ₊₁, rₖ₊₁⟩ / ⟨rₖ, rₖ⟩
    pₖ₊₁ = rₖ₊₁ + βₖ pₖ
```

Cost per iteration: $O(q \cdot D^3 \log D + q^2 \cdot D^3)$.

Warmstart: previous iteration's solution. Cold start: per-voxel Fourier
solve $D^{-1} \hat{r}$, gathered at support.

## 4. Results

### 128³ (q=10, 50k images, 20 EM iterations)

| Method | RelVar | Δ |
|--------|--------|---|
| soft-alpha λ=10 c=5 | 0.9580 | +0.8% |
| mask projection + gridding | 0.9496 | baseline |
| PCG circulant + gridding | 0.9347 | −1.5% |

### 256³ (q=10, 50k images, 20 EM iterations)

| Method | RelVar | Δ |
|--------|--------|---|
| soft-alpha λ=100 c=10 | 0.8278 | +4.6% |
| soft-alpha λ=10 c=10 | 0.8270 | +4.5% |
| mask projection + gridding | 0.7816 | baseline |

Collar ≈ 4% of grid. λ ∈ [10, 500] insensitive.
Improvement scales with resolution.

## 5. Implementation

`recovar/reconstruction/pcg_variants.py`:
- `solve()` — entry point (works for q=1 and q>1)
- `compute_gridding_kernel_real()` — G(x)
- `build_alpha_weight()` — α(x) and outer support
- `_matvec()` — CG operator
- `_cg()` — CG loop
