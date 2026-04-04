# Sketched Normal-Operator Products

This document derives the formulas for computing sketched products of the
normal-operator gradient $G(X) = \mathcal{A}^*(\mathcal{A}(X) - b)$ without
ever forming the dense $p \times n$ matrix $G(X)$.

## Whitening Convention

For image $i$, the whitened forward model is:

$$
\mathcal{A}_i(x) = \frac{\mathrm{CTF}_i \cdot P_i(x)}{\sqrt{\sigma_i^2}},
\qquad
b_i = \frac{y_i - \mathrm{CTF}_i \cdot P_i(\mu)}{\sqrt{\sigma_i^2}}
$$

where:
- $P_i$ is the slice operator (Fourier-space projection via rotation $R_i$),
- $\mathrm{CTF}_i$ is the contrast transfer function for image $i$,
- $\sigma_i^2$ is the noise variance,
- $\mu$ is the mean volume,
- $y_i$ is the observed (translated) image.

Define the **whitened CTF**: $\mathrm{CTF}_{w,i} = \mathrm{CTF}_i / \sqrt{\sigma_i^2}$.

Then $\mathcal{A}_i(x) = \mathrm{CTF}_{w,i} \cdot P_i(x)$ and
$\mathcal{A}_i^*(r) = P_i^*(\mathrm{CTF}_{w,i} \cdot r)$, where $P_i^*$
is the adjoint slice (backprojection).

This matches the convention in `recovar/heterogeneity/embedding.py`
(`_compute_batch_coords_p1`).

## Normal-Operator Gradient

The residual for image $i$ given iterate $X$ with column $x_i$:

$$
r_i = \mathcal{A}_i(x_i) - b_i
$$

The gradient column:

$$
g_i = \mathcal{A}_i^*(r_i) = P_i^*(\mathrm{CTF}_{w,i} \cdot r_i)
$$

The full gradient: $G(X) = [g_1, \ldots, g_n] \in \mathbb{R}^{p \times n}$.

## Factored Input

The iterate $X = U_X \, \mathrm{diag}(\sigma_X) \, V_X^\top$ with shapes
$U_X \in \mathbb{R}^{p \times r}$, $\sigma_X \in \mathbb{R}^r$,
$V_X \in \mathbb{R}^{n \times r}$.

For a batch $B$, the coefficient matrix
$C_B = V_X[B,:] \, \mathrm{diag}(\sigma_X) \in \mathbb{R}^{|B| \times r}$,
and the predicted whitened images are:

$$
\mathcal{A}_i(x_i) = \sum_{j=1}^r C_{i,j} \cdot \mathrm{CTF}_{w,i} \cdot P_i(U_{X,j})
$$

This reuses the existing `batch_vol_forward_from_map` for basis slicing.

## Right Sketch: $G(X) \, Q_R$

For $Q_R \in \mathbb{R}^{n \times t}$:

$$
G(X) Q_R = \sum_{i=1}^n g_i \, q_i^\top
= \sum_{i=1}^n P_i^*\!\bigl(\mathrm{CTF}_{w,i} \cdot r_i\bigr) \, q_i^\top
$$

For a batch $B$, the contribution is a **multi-channel weighted backprojection**:
for each sketch column $j$, backproject the weighted residuals
$\mathrm{CTF}_{w,i} \cdot r_i \cdot Q_{i,j}$ across all images $i \in B$.

This is implemented via `batch_adjoint_slice_volume` with shape
`(qrank, batch, image_size)` to produce `(qrank, volume_size)`.

**Implementing function:** `right_sketch_normal_residual_batch` in
`recovar/heterogeneity/ppca.py`.

## Left Sketch: $S_L \, G(X)$

For $S_L \in \mathbb{R}^{s \times p}$:

$$
(S_L G(X))_{s,i} = \sum_k S_{L,s,k} \cdot g_{i,k}
= \sum_k S_{L,s,k} \cdot P_i^*(\mathrm{CTF}_{w,i} \cdot r_i)_k
$$

Using the adjoint identity $\langle v, P_i^*(w) \rangle = \langle P_i(v), w \rangle$
and the relation $\sum_k a_k b_k = \langle \overline{a}, b \rangle$:

$$
(S_L G)_{s,i}
= \langle \overline{S_{L,s,:}}, P_i^*(\mathrm{CTF}_{w,i} \cdot r_i) \rangle
= \langle P_i(\overline{S_{L,s,:}}), \mathrm{CTF}_{w,i} \cdot r_i \rangle
$$

For nearest-neighbor slicing, $P_i(\overline{v}) = \overline{P_i(v)}$, so:

$$
(S_L G)_{s,i} = \sum_k P_i(S_{L,s,:})_k \cdot \mathrm{CTF}_{w,i,k} \cdot r_{i,k}
$$

**No backprojection is required.** We forward-project the rows of $S_L$
(via `batch_vol_forward_from_map` with `skip_ctf=True`), multiply by
$\mathrm{CTF}_w \cdot r$, and contract over pixels.

**Implementing function:** `left_sketch_normal_residual_batch` in
`recovar/heterogeneity/ppca.py`.

## Public API

```python
from recovar.heterogeneity.ppca import compute_normal_residual_sketches

result = compute_normal_residual_sketches(
    experiment_dataset,
    U_X, sigma_X, V_X,        # factored iterate X = U diag(s) V^T
    mean,                       # mean volume
    noise_variance,             # per-image or global
    batch_size,
    left_sketch=S_L,            # (s, volume_size) or None
    right_sketch=Q_R,           # (n_images, t) or None
)
# result["left"]  -> (s, n_images) = S_L @ G(X)
# result["right"] -> (volume_size, t) = G(X) @ Q_R
```

## Half-Image (rfft2) Convention

All batch-level operations use the half-image convention for ~2x memory and
compute savings.  Images are stored in rfft2-packed format with shape
`(H, W//2+1)` flattened to `(H*(W//2+1),)`.

**Right sketch** uses `batch_adjoint_slice_volume(..., half_image=True)`,
which is the true adjoint of half-image slicing ��� no additional weights needed.

**Left sketch** computes the full-spectrum sum
$\sum_k a[k] \cdot b[k]$ from half-spectrum data using Hermitian weights.
For data with Hermitian symmetry (real-valued in real space), the full sum
equals $\mathrm{Re}\bigl(\sum_{k \in \mathrm{half}} w[k] \cdot a[k] \cdot b[k]\bigr)$,
where $w[k] = \sqrt{w_k}^2$ and `rfft2_hermitian_weights` returns $\sqrt{w}$.
Both arrays are multiplied by $\sqrt{w}$ before the plain contraction, and the
real part is taken.

## Orientation Convention

`left_sketch` has shape `(s, p)` so the API literally computes $S_L G(X)$.
To use a basis-oriented form with $U_{\mathrm{basis}}$ of shape $(p, s)$,
call with `left_sketch = U_basis.T`.
