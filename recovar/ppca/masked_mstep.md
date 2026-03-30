# Masked PPCA M-step: Formulation and Solver

## Problem

At each EM iteration we solve the M-step: given accumulated data statistics
$D(\xi)$ (q x q, SPD) and right-hand side $r(\xi)$ (q-vector) at each Fourier
voxel $\xi$, find the loading matrix $W$ (q columns of a D^3 volume).

### Standard (unmasked) M-step

Per-voxel solve in Fourier:

$$\hat{W}(\xi) = D(\xi)^{-1} \hat{r}(\xi) \quad \forall \xi$$

where $D(\xi) = L(\xi) + \Lambda(\xi)$:
- $L(\xi) = \sum_n |C_n(\xi)|^2\, \mathbb{E}[z_n z_n^T \mid y_n]$ — data term
- $\Lambda(\xi) = \text{diag}(1/\tau(\xi))$ — Tikhonov regularization (prior)

This decouples across Fourier voxels. Cost: one q x q solve per voxel.

### Masked M-step

We want $W(x) \approx 0$ outside a molecular support mask $\Omega$.
This couples all Fourier voxels.

## Formulation: soft penalty + gridding in the objective

$$\min_W \;\underbrace{\sum_\xi \hat{W}(\xi)^H D(\xi)\, \hat{W}(\xi)
  - 2\,\text{Re}\,\hat{W}(\xi)^H \hat{r}(\xi)}_{\Phi(W)\text{: data fidelity + regularization}}
  + \underbrace{\frac{\lambda}{2} \sum_x \alpha(x)\, |W(x)|^2}_{\text{soft boundary penalty}}$$

### Gridding correction

The forward model uses trilinear interpolation for Fourier-slice extraction.
The interpolation kernel $G(x)$ acts in **real space** as a blurring:

$$G(x) = \text{sinc}^2(x_1/D)\;\text{sinc}^2(x_2/D)\;\text{sinc}^2(x_3/D)$$

where $\text{sinc}(t) = \sin(\pi t)/(\pi t)$.

Including $G$ in the forward model, the true normal equations are:

$$G\,\mathcal{F}^{-1}\!\big[D\;\mathcal{F}[G\,W]\big] = G\,\mathcal{F}^{-1}[\hat{r}]$$

The full objective with gridding becomes:

$$\min_W \;\sum_\xi \widehat{GW}(\xi)^H D(\xi)\,\widehat{GW}(\xi)
  - 2\,\text{Re}\,\widehat{GW}(\xi)^H \hat{r}(\xi)
  + \frac{\lambda}{2} \sum_x \alpha(x)\,|W(x)|^2$$

The solution $W$ is the **deconvolved** volume — no post-processing gridding
correction is needed.

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

## Solver: unpreconditioned CG in reduced coordinates

### Normal equations

$$\big[\underbrace{P_\Omega\,G\,\mathcal{F}^{-1}[D\,\mathcal{F}[G\,\cdot\,]]
  \,P_\Omega}_{\text{data operator}} + \underbrace{\lambda\,\text{diag}(\alpha)}_{\text{penalty}}\big]\,W
  = P_\Omega\,G\,\mathcal{F}^{-1}[\hat{r}]$$

where $P_\Omega$ is gather/scatter to the outer support.

### Matvec (one CG iteration)

For input $w$ on support voxels ($n_{\text{sup}} \times q$ reals):

1. **Scatter** to full grid: $(n_\text{sup}, q) \to (q, D, D, D)$
2. **Multiply by $G$** in real space: $w \gets G \cdot w$
3. **rfft3** to half-volume Fourier: $(q, D, D, D/2+1)$ complex
4. **$D \cdot$** per Fourier voxel: $q \times q$ matrix-vector, chunked
5. **irfft3** back to real space
6. **Multiply by $G$** again in real space
7. **Gather** at support voxels
8. **Add penalty**: $+ \lambda\,\alpha(x)\,w(x)$

Cost per CG iteration: 2 FFT pairs + 2 real-space pointwise + 1 chunked LHS multiply.

### Warmstart

At EM iteration $k > 0$, initialize CG from $W_{k-1}$ (previous M-step
output, already deconvolved when gridding is in the objective).

At EM iteration 0 (cold start): per-voxel Fourier solve $D^{-1} r$, then
iFFT and gather at support.

### Convergence

In float32, CG residuals are unreliable (oscillate due to rounding in the
recurrence $r \gets r - \alpha Ap$). Periodic recomputation from scratch
($r = b - Ax$ every 10 iterations) stabilizes tracking but does not fix
the underlying float32 limitation.

Despite unreliable residuals, the CG **iterates** are still meaningful:
RelVar improves steadily over EM iterations. The tolerance criterion
(default $10^{-4}$) is effectively never reached; the solver always runs
to maxiter.

Typical budget: 50 CG iterations per M-step, 20 EM iterations total.

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

## Results (128^3, q=10, 50k images, 20 EM iterations)

| Method | RelVar |
|--------|--------|
| soft-alpha λ=10 c=8 + post-gridding | 0.9585 |
| soft-alpha λ=100 c=5 + post-gridding | 0.9581 |
| mask projection + gridding | 0.9496 |
| unpreconditioned CG 50it + gridding | 0.9490 |
| PCG baseline + gridding | 0.9347 |
| PCG baseline (no gridding) | 0.9186 |

Note: these used gridding as post-processing. Results with gridding
in the objective (correct formulation) are pending.
