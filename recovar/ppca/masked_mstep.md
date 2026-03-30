# Masked PPCA M-step: Formulation and Solver

## Forward model and interpolation

The cryo-EM forward model for image $n$ at 2D Fourier pixel $\eta$:

$$\hat{y}_n(\eta) = C_n(\eta) \cdot \hat{v}_n(R_n^{-1}\eta) + \text{noise}$$

where $v_n(x) = \mu(x) + \sum_k z_{nk}\,w_k(x)$ is the heterogeneous volume,
$C_n$ is the CTF, and $R_n$ is the orientation.

The 3D volume is known on a discrete grid of size $D^3$. Evaluating
$\hat{v}(R_n^{-1}\eta)$ at off-grid Fourier coordinates requires interpolation.

### Trilinear interpolation and the gridding kernel

Trilinear interpolation in 3D Fourier space convolves the discrete spectrum
with a triangle kernel $K$ of width 1 voxel per axis. Convolution in
Fourier space is multiplication in real space by $\mathcal{F}(K)$:

$$G(x) = \mathcal{F}(K)(x) = \operatorname{sinc}^2\!\Big(\frac{x_1}{D}\Big)\,
  \operatorname{sinc}^2\!\Big(\frac{x_2}{D}\Big)\,
  \operatorname{sinc}^2\!\Big(\frac{x_3}{D}\Big)$$

So the effective forward model is not "slice $\hat{w}$" but
"slice $\mathcal{F}[G \cdot w]$":

$$\hat{y}_n(\eta) \approx C_n(\eta)\;\widehat{G\,v_n}(R_n^{-1}\eta)$$

The volume that gets projected is $G \cdot w$, not $w$.

### Consequence for reconstruction

The adjoint (backprojection) of this forward model is:

$$A^H y = G \cdot \mathcal{F}^{-1}\!\Big[\sum_n C_n^*\,\hat{y}_n\;\text{(inserted)}\Big]$$

The factor of $G$ appears because the adjoint of "multiply by $G$ then project"
is "backproject then multiply by $G$". The normal equations operator $A^H\!A$ is:

$$A^H\!A\;w = G\;\mathcal{F}^{-1}\!\big[D\;\mathcal{F}[G\,w]\big]$$

where $D(\xi) = \sum_n |C_n(\xi)|^2 \mathbb{E}[z_n z_n^T]$ is the per-voxel
$q \times q$ data matrix accumulated during the E-step (without interpolation
weights — those are handled by $G$).

**Standard approach (gridding as post-processing):** ignore $G$ in the solve,
get $\tilde{w} = (D + \Lambda)^{-1} r$ per Fourier voxel, then correct:
$w = \tilde{w} / G$. This is the RELION-style gridding correction.

**Correct approach (gridding in objective):** solve the true normal equations
directly, getting the deconvolved $w$ without post-processing.

## PPCA M-step with soft mask penalty

### PPCA model

Each image $y_n$ is generated from a latent variable $z_n \in \mathbb{R}^q$:

$$y_n = A_n(\mu + W z_n) + \varepsilon_n, \qquad z_n \sim \mathcal{N}(0, I_q)$$

where $A_n$ is the forward operator (project, CTF, interpolation) for image $n$,
$\mu$ is the mean volume, $W = [w_1 \mid \cdots \mid w_q]$ is the loading matrix
($q$ principal component volumes), and $\varepsilon_n$ is noise.

### M-step least squares

Given posterior statistics from the E-step:
- $\bar{z}_n = \mathbb{E}[z_n \mid y_n]$
- $\Sigma_n = \mathbb{E}[z_n z_n^T \mid y_n]$

The M-step minimizes over $W$:

$$\mathcal{L}(W) = \sum_n \mathbb{E}_{z_n|y_n}\!\big[\|y_n - A_n(\mu + Wz_n)\|^2\big]
  + \sum_\xi \hat{W}(\xi)^H \Lambda(\xi)\,\hat{W}(\xi)$$

where $\Lambda(\xi) = \operatorname{diag}(1/\tau(\xi))$ is the Tikhonov prior.

Expanding the quadratic and using $A_n = C_n \cdot \mathcal{S}_{R_n} \cdot \mathcal{F} \cdot G$
(CTF $\times$ slice extraction $\times$ DFT $\times$ gridding), the per-Fourier-voxel
normal equations are:

$$D(\xi)\,\hat{W}(\xi) = \hat{r}(\xi)$$

where:
- $D(\xi) = \sum_n |C_n(\xi)|^2\,\Sigma_n + \Lambda(\xi)$ — $q \times q$, SPD
- $\hat{r}(\xi) = \sum_n C_n^*(\xi)\,\hat{y}_n^{\text{centered}}(\xi)\,\bar{z}_n^T$ — $q$-vector

Without a mask, this decouples per Fourier voxel: $\hat{W}(\xi) = D(\xi)^{-1}\hat{r}(\xi)$.

### Adding the mask

We want $w(x) \approx 0$ outside the molecular support.
A hard constraint couples all Fourier voxels.
Instead, we add a soft penalty to the M-step objective:

$$\min_W \;\underbrace{\sum_\xi \widehat{GW}(\xi)^H D(\xi)\,\widehat{GW}(\xi)
  - 2\operatorname{Re}\,\widehat{GW}(\xi)^H\hat{r}(\xi)}_{\text{data fidelity + Tikhonov, with gridding}}
  \;+\;\underbrace{\frac{\lambda}{2}\sum_x \alpha(x)\,|W(x)|^2}_{\text{soft boundary penalty}}$$

The solution $W$ is the deconvolved, boundary-regularized loading matrix.

### Penalty weight $\alpha(x)$

Constructed from the binary mask via signed distance transform:

- **Core** ($d(x) < -c$): $\alpha = 0$ — no penalty
- **Collar** ($-c < d(x) \leq 0$): $\alpha = \frac{1}{2}(1 + \cos(\pi\,d/c))$ — smooth ramp
- **Outside** ($d(x) > 0$): $\alpha = 1$ — full penalty

where $d(x)$ is signed distance (negative inside mask, positive outside)
and $c$ is the collar width in voxels.

A generous hard outer support (mask dilated by $c + 3$ voxels) provides
exact zeros far from the molecule. Variables only live on this support
(reduced coordinates), saving memory and FLOPs.

### Choice of $\lambda$

The penalty strength $\lambda$ balances data fidelity against boundary
suppression. In practice:
- $\lambda = 10$–500 works across 128^3 and 256^3 at 50k images
- Insensitive within this range for collar = 4% of grid size
- Very large $\lambda$ with wide collar over-regularizes

### Choice of collar width

The collar should scale with grid size to maintain constant physical width:

$$c = \lfloor 0.04 \times D \rceil$$

- 128^3: $c = 5$
- 256^3: $c = 10$

## Numerical scheme

### Variables

Let $\Omega_S$ denote the outer support (mask dilated by $c + 3$),
$n_S = |\Omega_S|$ its cardinality.

The unknown is $w \in \mathbb{R}^{n_S \times q}$ — the $q$ loading
columns restricted to the support voxels.

### Operator

Define the linear operator $A : \mathbb{R}^{n_S \times q} \to \mathbb{R}^{n_S \times q}$:

$$A\,w = P_S\,G\,\mathcal{F}^{-1}\!\big[D\;\mathcal{F}[G\,E_S\,w]\big]
  + \lambda\,\text{diag}(\alpha_{|S})\,w$$

where:
- $E_S : \mathbb{R}^{n_S} \to \mathbb{R}^{D^3}$ scatters support values to the full grid (zero-fill)
- $P_S : \mathbb{R}^{D^3} \to \mathbb{R}^{n_S}$ gathers from the full grid at support voxels
- $G = \text{diag}(g)$ with $g(x) = \text{sinc}^2(x_1/D)\,\text{sinc}^2(x_2/D)\,\text{sinc}^2(x_3/D)$
- $\mathcal{F}$ and $\mathcal{F}^{-1}$ are the 3D DFT and inverse DFT
- $D(\xi) \in \mathbb{R}^{q \times q}$ is the per-voxel normal-equations matrix (SPD)
- $\alpha_{|S}$ is the penalty weight restricted to the support

The right-hand side is:

$$b = P_S\,G\,\mathcal{F}^{-1}[\hat{r}]$$

where $\hat{r}(\xi) \in \mathbb{C}^q$ is the accumulated RHS from the E-step.

$A$ is SPD on $\mathbb{R}^{n_S \times q}$ (proof: $D \succ 0$, $G \geq 0$,
$\alpha \geq 0$, and $G E_S P_S G$ is positive semidefinite).

### CG algorithm

Standard unpreconditioned CG on $Aw = b$:

```
r₀ = b - A w₀
p₀ = r₀
for k = 0, 1, ..., maxiter-1:
    αₖ = (rₖ, rₖ) / (pₖ, A pₖ)
    wₖ₊₁ = wₖ + αₖ pₖ
    rₖ₊₁ = rₖ - αₖ A pₖ
    if (k+1) mod 10 == 0:  rₖ₊₁ = b - A wₖ₊₁    # float32 stability
    βₖ = (rₖ₊₁, rₖ₊₁) / (rₖ, rₖ)
    pₖ₊₁ = rₖ₊₁ + βₖ pₖ
```

Inner product: standard Euclidean $(u, v) = \sum_{i,j} u_{ij} v_{ij}$.

### Matvec evaluation

Each application of $A w$ costs:

| Step | Operation | Complexity |
|------|-----------|------------|
| 1 | $E_S w$: scatter to $(q, D^3)$ | $O(q \cdot n_S)$ |
| 2 | $G \cdot$: pointwise multiply in real space | $O(q \cdot D^3)$ |
| 3 | $\mathcal{F}$: rfft3 (batched over $q$) | $O(q \cdot D^3 \log D)$ |
| 4 | $D(\xi) \cdot$: per-voxel $q \times q$ matvec | $O(q^2 \cdot D^3/2)$ |
| 5 | $\mathcal{F}^{-1}$: irfft3 | $O(q \cdot D^3 \log D)$ |
| 6 | $G \cdot$: pointwise multiply | $O(q \cdot D^3)$ |
| 7 | $P_S$: gather at support | $O(q \cdot n_S)$ |
| 8 | $+\lambda \alpha w$: penalty | $O(q \cdot n_S)$ |

Total: $O(q \cdot D^3 \log D + q^2 \cdot D^3)$ per CG iteration.

### Initialisation

- **EM iteration $k > 0$**: warmstart from $w_{k-1}$ (previous M-step output)
- **EM iteration 0**: cold start from per-voxel Fourier solve
  $\hat{w}_0(\xi) = D(\xi)^{-1} \hat{r}(\xi)$, then iFFT + gather at support

### Float32 considerations

CG residuals in float32 are unreliable due to rounding in the recurrence
$r \gets r - \alpha A p$. The true residual $||b - Aw||$ diverges from
the tracked residual after ~5–10 iterations.

Mitigation: recompute $r = b - Aw$ from scratch every 10 iterations.
This stabilizes residual tracking but does not eliminate float32 error.

Despite unreliable residuals, CG iterates are meaningful (RelVar improves
steadily). Tolerance $10^{-4}$ is never reached; solver runs to maxiter.

Budget: 50 CG iterations per M-step × 20 EM iterations = 1000 total.

## Implementation

Code: `recovar/reconstruction/pcg_variants.py`

- `solve()` — main entry point
- `compute_gridding_kernel_real()` — builds $G(x)$
- `build_alpha_weight()` — builds $\alpha(x)$ and outer support
- `_matvec()` — CG operator (scatter → G → FFT → D → iFFT → G → gather + penalty)
- `_cg()` — CG loop with periodic residual recomputation

Integration: `recovar/ppca/ppca.py`

- `EM_step_half()` dispatches to `mstep_solver_fn` if provided
- `EM()` accepts `mstep_solver_fn` and `use_gridding_correction` parameters
- When gridding is in the objective, set `use_gridding_correction=False`
- The EM applies `volume_mask` as a safety projection after each M-step;
  pass the outer support as `volume_mask` to preserve the collar

## Results (q=10, 50k images, 20 EM iterations)

All soft-alpha results below use gridding as post-processing.
Gridding-in-the-objective results pending.

### 128^3

| Method | RelVar | Δ vs baseline |
|--------|--------|---------------|
| soft-alpha λ=10 c=8 | 0.9585 | +0.9% |
| soft-alpha λ=100 c=5 | 0.9581 | +0.9% |
| mask projection + gridding | 0.9496 | baseline |
| unpreconditioned CG 50it + gridding | 0.9490 | -0.1% |
| PCG baseline + gridding | 0.9347 | -1.5% |

### 256^3

| Method | RelVar | Δ vs baseline |
|--------|--------|---------------|
| soft-alpha λ=10 c=10 (4% of grid) | 0.8270 | +4.5% |
| soft-alpha λ=10 c=8 | 0.8186 | +3.7% |
| soft-alpha λ=100 c=6 | 0.8123 | +3.1% |
| soft-alpha λ=10 c=6 | 0.8103 | +2.9% |
| soft-alpha λ=10 c=5 | 0.8059 | +2.4% |
| soft-alpha λ=10 c=3 | 0.7963 | +1.5% |
| noprecond CG 50it + gridding | 0.7835 | +0.2% |
| mask projection + gridding | 0.7816 | baseline |

Observations:
- Collar ≈ 4% of grid size is optimal (c=5 at 128, c=10 at 256)
- λ insensitive in range 10–500
- Improvement scales with resolution: +0.9% at 128, +4.5% at 256
- Circulant preconditioner hurts in float32; plain CG is better
