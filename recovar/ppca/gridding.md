# Gridding correction: from post-processing to the objective

## Setup

The 3D volume is represented on a discrete grid. Extracting a 2D Fourier
slice at non-integer coordinates requires interpolation. Let $K$ denote
the interpolation operator (convolution with the interpolation kernel —
triangle for trilinear, sinc² in real space).

## Gridded vs deconvolved variable

Let $V$ be the true volume and $U = KV$ the kernel-smoothed (gridded) version.
The standard reconstruction works in the gridded variable:

$$\widehat{U} = \arg\min_U F(U), \qquad F(U) = \|AU - b\|^2$$

where $A$ is the measurement operator (slice extraction + CTF). One can show
that, in expectation, $\widehat{U} \approx KV_\star$ — the result is biased
by the interpolation kernel.

The standard post-processing step then deconvolves:

$$\widehat{V} = K^{-1} \widehat{U}$$

Since $K$ corresponds to multiplication by $G(x) = \text{sinc}^2(x/D)$ per
axis in real space, deconvolution is just dividing by $G$.

## Change of variables: $K$ in the objective

Rather than deconvolving after the fact, substitute $U = KV$ directly:

$$\widehat{V} = \arg\min_V F(KV) = \arg\min_V \|AKV - b\|^2$$

Normal equations:

$$K^* A^* A K\, \widehat{V} = K^* A^* b$$

Since $K$ is invertible:

$$\widehat{V} = K^{-1}(A^*A)^{-1} A^* b = K^{-1}\widehat{U}$$

**Without regularization, the two approaches are identical.** Solving in $V$
and post-processing deconvolution give the same answer.

## The regularized case

With Tikhonov regularization on the deconvolved volume:

$$\widehat{V}_\lambda = \arg\min_V \|AKV - b\|^2 + \lambda\|V\|^2$$

Normal equations:

$$(K^*A^*AK + \lambda I)\,\widehat{V}_\lambda = K^*A^*b$$

This gives:

$$\widehat{V}_\lambda = (K^*A^*AK + \lambda I)^{-1} K^*A^*b
= K^{-1}\bigl(A^*A + \lambda(KK^*)^{-1}\bigr)^{-1} A^*b$$

Compare with the standard approach (regularize in $U$, then deconvolve):

$$K^{-1}\widehat{U}_\lambda = K^{-1}(A^*A + \lambda I)^{-1} A^*b$$

**These are different.** The correct version has $\lambda(KK^*)^{-1}$ instead
of $\lambda I$. Since $K$ attenuates high frequencies, $(KK^*)^{-1}$ amplifies
them — the correct regularization penalizes high-frequency components of $V$
less than the naive approach (because $K$ already damps them in the data term).

## With the soft mask penalty

The full objective for the deconvolved, mask-regularized volume:

$$\widehat{V} = \arg\min_V \|AKV - b\|^2 + \lambda \sum_x \alpha(x)|V(x)|^2$$

Normal equations:

$$(K^*A^*AK + \lambda\,\text{diag}(\alpha))\,V = K^*A^*b$$

In the RELION diagonal approximation ($A^*A \approx \text{diag}(d)$):

$$K\,\text{diag}(d)\,K\,V + \lambda\,\alpha\,V = K\,\mathcal{F}^{-1}[\hat{r}]$$

where $K$ acts as multiplication by $G(x) = \text{sinc}^2(x/D)$ in real space,
and $d$, $\hat{r}$ are the accumulated Fourier-space weights and data.

The CG matvec is:

$$\text{operator}(V) = G \cdot \mathcal{F}^{-1}[d \cdot \mathcal{F}[G \cdot V]] + \lambda\,\alpha\,V$$

The RHS is:

$$b = G \cdot \mathcal{F}^{-1}[\hat{r}]$$

## What $d$ and $\hat{r}$ contain

The accumulated quantities from the backprojection pipeline:

$$d[\xi] = \sum_n \sum_\eta w_n(\eta,\xi)\;|C_n(\eta)|^2/\sigma_n^2$$

$$\hat{r}[\xi] = \sum_n \sum_\eta w_n(\eta,\xi)\;C_n^*(\eta)\,y_n(\eta)/\sigma_n^2$$

where $w_n(\eta,\xi)$ is the interpolation weight (trilinear scatter from
image pixel $\eta$ to grid point $\xi$).

In the RELION diagonal approximation, $d$ plays the role of $A^*A$ restricted
to the diagonal. The scatter weight $w$ in $d$ and $\hat{r}$ comes from the
adjoint operation only. The $K$ factors in the operator $K\,\text{diag}(d)\,K$
come from the change of variables $U = KV$.

## Summary

| Approach | Operator | Post-processing |
|----------|----------|-----------------|
| Standard (regularize $U$) | $\text{diag}(d) + \lambda I$ | deconvolve by $K^{-1}$ |
| Correct (regularize $V$) | $K\,\text{diag}(d)\,K + \lambda I$ | none |
| Standard + mask | $P\,\text{diag}(d)\,P + \lambda\alpha$ | deconvolve by $K^{-1}$ |
| Correct + mask | $K\,P\,\text{diag}(d)\,P\,K + \lambda\alpha$ | none |

The difference matters when regularization is nontrivial (large $\lambda$,
tight mask, or both).
