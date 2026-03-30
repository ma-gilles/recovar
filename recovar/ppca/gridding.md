# Gridding correction in cryo-EM reconstruction

## The problem

We represent the 3D Fourier transform of a volume on a discrete grid $\hat{v}[\xi]$, $\xi \in \mathbb{Z}^3$.
An image $n$ gives a 2D central slice of the continuous Fourier transform at orientation $R_n$.
The slice passes through non-integer 3D coordinates, so we need interpolation.

## Forward model with interpolation

The forward model (projection) for image pixel $\eta$:

$$\hat{y}_n(\eta) = C_n(\eta) \sum_{\xi} w(\eta, \xi)\;\hat{v}[\xi]$$

where $w(\eta, \xi)$ is the interpolation weight from 3D grid point $\xi$ to
2D slice point $\eta$. For trilinear interpolation, $w$ is a product of
triangle functions in each coordinate, nonzero only for the 8 nearest grid points.

## Adjoint (backprojection)

The adjoint scatters each image pixel back to the 3D grid with the same weights:

$$[\text{backproject}(y)]_\xi = \sum_n \sum_\eta w(\eta, \xi)\;C_n^*(\eta)\;y_n(\eta) / \sigma_n^2$$

This is what `Ft_y` accumulates. Similarly `Ft_ctf` accumulates:

$$d[\xi] = \sum_n \sum_\eta w(\eta, \xi)\;|C_n(\eta)|^2 / \sigma_n^2$$

Note: each $w(\eta,\xi)$ appears **once** — from the adjoint (backprojection).
The forward interpolation weights do not appear in $d$ or $\hat{r}$ because
the forward model is not applied during accumulation.

## Normal equations

If we used the exact forward–adjoint pair, the normal equations operator
$A^H A$ would involve $w$ twice (once from forward, once from adjoint):

$$[A^H A\;\hat{v}]_\xi = \sum_\eta w(\eta,\xi) \;|C(\eta)|^2 \sum_{\xi'} w(\eta,\xi')\;\hat{v}[\xi']$$

This couples different 3D voxels $\xi$ and $\xi'$ through shared image pixels.
It is not diagonal per voxel.

## The RELION approximation

RELION (and recovar) ignores the off-diagonal coupling and approximates:

$$[A^H A\;\hat{v}]_\xi \approx d[\xi]\;\hat{v}[\xi]$$

where $d[\xi]$ is the accumulated weight (one factor of $w$, from the adjoint only).
This makes the system diagonal per voxel, giving the Wiener filter:

$$\hat{v}[\xi] = \hat{r}[\xi] \;/\; d[\xi]$$

## What's missing

The true normal equations have $w^2$ (forward × adjoint) per voxel on the diagonal.
The RELION approximation uses $d$ which has only $w^1$ (adjoint only).
This means the Wiener solution is off by a factor of $w$ — the solution has too
much weight at voxels that are frequently visited by interpolation (grid centers)
and too little at voxels that are rarely visited (grid corners/edges).

## The gridding correction

When many images cover Fourier space uniformly, the interpolation weights
averaged over all orientations become a smooth function of position on the 3D grid.
For trilinear interpolation, this average weight is:

$$\bar{w}[\xi] \approx \text{sinc}^2(\xi_1/D)\;\text{sinc}^2(\xi_2/D)\;\text{sinc}^2(\xi_3/D) = G(\xi)$$

evaluated in **real space** after iDFT (because the interpolation kernel is
a triangle in Fourier, whose DFT is sinc²).

The gridding correction divides the real-space volume by $G$:

$$v_{\text{corrected}}(x) = v_{\text{Wiener}}(x) \;/\; G(x)$$

This compensates for the missing factor of $w$ in $d$.

## Relationship to the masked solve

For the unmasked diagonal solve, gridding correction as post-processing is
fine — it's just dividing by a known smooth function.

For the masked solve (CG on a coupled system), the situation is more subtle.
The CG solves:

$$\text{operator}(v) = \text{rhs}$$

The operator is $P \mathcal{F}^{-1}[d \cdot \mathcal{F}[P\,v]] + \lambda\alpha v$,
using the accumulated $d$ (with one factor of $w$). The gridding correction
is then applied to the CG output as post-processing, same as the unmasked case.

**Alternative**: incorporate the missing $w$ factor into the CG operator.
This requires multiplying $d$ by an additional factor of $\bar{w}$ in
the matvec, and the RHS by $\bar{w}$. The CG then directly produces the
corrected volume. However, this only makes sense if $d$ was accumulated
without the orientation-averaged $\bar{w}$ — which is NOT the case in the
current code (the trilinear scatter naturally produces $d$ with one $w$ factor).

## Summary

| Quantity | What it contains |
|----------|-----------------|
| $d[\xi]$ (Ft_ctf) | $\sum w(\eta,\xi) |C|^2/\sigma^2$ — one interpolation weight |
| $\hat{r}[\xi]$ (Ft_y) | $\sum w(\eta,\xi) C^* y/\sigma^2$ — one interpolation weight |
| Wiener solution | $\hat{r}/d$ — ratio cancels the one $w$, but missing the second $w$ from forward |
| Gridding correction | Divides real-space by $G(x) = \text{sinc}^2(x/D)$ per axis — compensates for the missing second $w$ |

The gridding correction is **not** about the forward model being wrong.
It's about the diagonal approximation dropping one of the two interpolation
weight factors from $A^H A$.
