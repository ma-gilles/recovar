# Sketched Normal-Operator Products

## Setup

For image $i$, the whitened forward model is

$$
\mathcal{A}_i(x) = \frac{\mathrm{CTF}_i \cdot P_i(x)}{\sqrt{\sigma_i^2}},
\qquad
b_i = \frac{y_i - \mathrm{CTF}_i \cdot P_i(\mu)}{\sqrt{\sigma_i^2}},
$$

where $P_i$ is the Fourier slice operator (rotation $R_i$), $\mathrm{CTF}_i$
is the contrast transfer function, $\sigma_i^2$ is the noise variance, $\mu$
is the mean volume, and $y_i$ is the translated image.

Define the **whitened CTF**: $w_i = \mathrm{CTF}_i / \sqrt{\sigma_i^2}$.

Then $\mathcal{A}_i(x) = w_i \cdot P_i(x)$ and
$\mathcal{A}_i^*(r) = P_i^*(w_i \cdot r)$ (backprojection).

## Gradient

The residual for column $i$ of the iterate $X$ with column $x_i$:

$$
r_i = \mathcal{A}_i(x_i) - b_i.
$$

The gradient column: $g_i = \mathcal{A}_i^*(r_i) = P_i^*(w_i \cdot r_i)$.

The full gradient: $G(X) = [g_1, \ldots, g_n] \in \mathbb{R}^{p \times n}$.

## Factored input

$X = U \text{diag}(\sigma) V^T$ with $U \in \mathbb{R}^{p \times r}$,
$\sigma \in \mathbb{R}^r$, $V \in \mathbb{R}^{n \times r}$.

For image $i$: $x_i = U \text{diag}(\sigma) V_i^T$, so
$\mathcal{A}_i(x_i) = \sum_j \sigma_j V_{ij} \cdot w_i \cdot P_i(U_j)$.

This reuses `batch_over_vol_slice_volume_half` to project all $r$ columns
of $U$ at once.

## Right sketch: $G(X)\, Q$

$$
G(X) Q = \sum_{i=1}^n g_i\, q_i^T
= \sum_i P_i^*(w_i \cdot r_i)\, q_i^T.
$$

**Implementation:** for each image batch, `per_image_backproject` gives
$\mathrm{bp} \in \mathbb{R}^{p \times B}$ in one CUDA call (one column per
image).  Then $\mathrm{bp} \cdot Q_B$ is a matmul.  Accumulated over batches.

Cost is $O(B)$ backprojections per batch regardless of sketch rank $t$.

## Left sketch: $S\, G(X)$

$$
(S\, G(X))_{s,i} = \sum_k S_{s,k} \cdot g_{i,k}
= S_{s,:} \cdot P_i^*(w_i \cdot r_i).
$$

**Implementation:** same `per_image_backproject` gives bp, then
$S \cdot \mathrm{bp}$ is a matmul.  No separate forward projection of $S$
needed.

## Per-image backproject trick

Both sketches share the same backprojection.  The CUDA kernel
`per_image_backproject` scatters each image into its own volume column
(near-zero atomic contention), giving $(p, B)$ in one call.  Then:

- Right: $\mathrm{bp} \cdot Q_B \to (p, t)$
- Left: $S \cdot \mathrm{bp} \to (s, B)$

This is the same trick used in the PPCA M-step (`E_M_step_batch_half`)
for the LHS accumulation.

## Implementing functions

All in `recovar/ppca/sketched_normal.py`:

- `_sketched_normal_batch` — JIT-compiled per-batch kernel
- `_compute_sketches_half` — dataset loop (half-volume internals)
- `SketchedNormalOperator` — public API (real-space in/out)

## Notes

- All internal operations use half-volume rfft layout for ~2x memory savings.
- The public API accepts real-space $U$, $S$, $Q$ and converts internally.
- Mean is accepted in Fourier domain (complex) or real-space (detected automatically).
- `right_matvec_fourier` bypasses the real→Fourier conversion for $U$
  when you already have Fourier eigenvectors (e.g. from `gt.get_vol_svd()`).
