# RECOVAR EM for a single homogeneous volume on a dense pose grid

Reviewed against `ma-gilles/recovar`, branch `dev`, on 2026-03-31.

Primary files reviewed:
- `recovar/em/e_step.py`
- `recovar/em/m_step.py`
- `recovar/em/core.py`
- `recovar/em/iterations.py`
- `recovar/em/states.py`
- `recovar/em/sampling.py`
- `recovar/core/configs.py`
- `recovar/core/slicing.py`
- `recovar/core/geometry.py`
- `recovar/reconstruction/relion_functions.py`
- local note: `abinitio.pdf`

This document is meant to describe the stable algorithmic core, not the current transient API boundaries.

---

## 1. Scope

This document focuses on the dense-grid EM path for a **single homogeneous volume**.

It does **not** try to fully document:
- low-rank heterogeneity EM,
- covariance estimation,
- refined or hierarchical pose grids,
- SGD variants.

Those features are present in the package, but the dense homogeneous path is the cleanest core and should be the anchor for long-term documentation.

---

## 2. Forward model and notation

For image `i`, candidate rotation `r`, and candidate in-plane translation `t`, the forward model is
$$
y_i = S_t C_i P_r \mu + \varepsilon_i,
\qquad
\varepsilon_i \sim \mathcal N_{\mathbb C}(0, \Lambda_i).
$$
Where:
- `\mu` is the 3D volume in Fourier coordinates.
- `P_r` is the Fourier slice / projection operator at rotation `r`.
- `C_i` is the diagonal CTF operator for image `i`.
- `S_t` is the diagonal Fourier phase-shift operator for translation `t`.
- `\Lambda_i` is the diagonal noise covariance, represented in code by `noise_variance`.

The dense-grid assumption is simple:
- all candidate rotations are explicitly enumerated,
- all candidate translations are explicitly enumerated,
- EM sums over the full Cartesian product of these hidden states.

The current code also applies an image preprocessing function inside the kernels. Conceptually, that preprocessing is part of the observation operator used in the implementation.

---

## 3. E-step math

For the homogeneous model, the posterior weight for hidden state `(r, t)` is
$$
\gamma_{i,r,t}
\propto
\exp\left(
-\tfrac12 \|y_i - S_t C_i P_r \mu\|^2_{\Lambda_i^{-1}}
\right),
$$
followed by normalization over all rotations and translations for fixed `i`.

### 3.1 Expanded residual

Because `\Lambda_i` is diagonal and `S_t` is unitary, the weighted squared residual can be expanded as
$$
\|y_i - S_t C_i P_r \mu\|^2_{\Lambda_i^{-1}}
=
\|y_i\|^2_{\Lambda_i^{-1}}
- 2\,\Re\,\langle S_t(C_i y_i / \sigma_i^2),\; P_r \mu \rangle
+ \langle C_i^2 / \sigma_i^2,\; |P_r \mu|^2 \rangle.
$$
This is the key algebraic reduction in the code:
- the translation only appears in the cross term,
- the expensive hidden-state dependence can be written as matrix products.

### 3.2 Matrixized form

Define the precomputed projection matrix
$$
P[r, :] = P_r \mu.
$$
For a batch of images, define the shifted, CTF-weighted image matrix
$$
Y_{(i,t), :} = S_t\left(C_i y_i / \sigma_i^2\right).
$$
Then the cross term for all image / rotation / translation triples is computed by a single matrix multiplication:
$$
-2\,\Re\,(Y P^*)
\quad\text{or equivalently}\quad
-2\,\Re\,(\overline{Y} P^\top),
$$
depending on storage convention.

The projection-norm term is also a matrix multiplication. Let
$$
W[i, :] = C_i^2 / \sigma_i^2,
\qquad
Q[r, :] = |P_r \mu|^2.
$$
Then
$$
\|C_i P_r \mu\|^2_{\Lambda_i^{-1}} = W[i, :] \cdot Q[r, :].
$$
So the dense E-step is:
1. precompute all `P_r \mu`,
2. compute all translation-dependent cross terms by one large GEMM,
3. compute all norm terms by one second GEMM,
4. add the per-image constant `\|y_i\|^2_{\Lambda_i^{-1}}`,
5. apply a softmax-like normalization over `(r, t)`.

### 3.3 Numerically stable normalization

The current code divides the residual by `2`, subtracts the minimum value over hidden states for each image, exponentiates, and normalizes. That is exactly the right stabilization pattern for
$$
\gamma_{i,r,t} = \frac{\exp(-\tfrac12 d_{i,r,t})}{\sum_{r',t'} \exp(-\tfrac12 d_{i,r',t'})}.
$$
---

## 4. M-step math

Given posterior weights `\gamma_{i,r,t}`, the mean update is the weighted regularized least-squares problem
$$
\mu^{\text{new}}
=
\arg\min_\mu
\frac12\sum_{i,r,t}
\gamma_{i,r,t}
\|y_i - S_t C_i P_r \mu\|^2_{\Lambda_i^{-1}}
+
\frac12\|\mu\|^2_{\tau^{-1}}.
$$
The corresponding normal equation has the form
$$
\left(
\sum_{i,r,t} \gamma_{i,r,t}
P_r^* C_i \Lambda_i^{-1} C_i P_r
+ \tau^{-1}
\right)
\mu
=
\sum_{i,r,t}
\gamma_{i,r,t}
P_r^* C_i \Lambda_i^{-1} S_t^* y_i.
$$
### 4.1 Sufficient statistics

The dense homogeneous path never forms the full linear operator explicitly. Instead it accumulates two Fourier-domain sufficient statistics:
$$
F_t y
\approx
\sum_{i,r,t}
\gamma_{i,r,t}
P_r^* \big(C_i \Lambda_i^{-1} S_t^* y_i\big),
$$
and
$$
F_t \mathrm{ctf}
\approx
\sum_{i,r,t}
\gamma_{i,r,t}
P_r^* \big(C_i^2 \Lambda_i^{-1}\big) P_r.
$$
In code these are called `Ft_y` and `Ft_ctf`.

The structure is important:
- both objects are **additive over images**,
- both can be accumulated in batches,
- both are the only M-step quantities needed for the single-volume update.

### 4.2 RELION-style filtered solve

After accumulating `Ft_y` and `Ft_ctf`, the code calls a RELION-style post-processing routine:
1. regularize the filter with the prior variance `tau`,
2. divide `F_ty` by the regularized `Ft_ctf`,
3. inverse DFT to real space,
4. crop and apply the spherical mask,
5. apply grid correction,
6. DFT back if Fourier-space output is desired.

So the implemented M-step is best understood as a **filtered backprojection solve with regularization**, not as an explicitly assembled dense linear solve.

---

## 5. How the current code organizes compute

## 5.1 Static configuration

`ForwardModelConfig` packages static geometry, discretization, CTF evaluation, and preprocessing into a single compile-time object. This is a good design direction and should remain the canonical description of the forward model.

What belongs in this object:
- image shape,
- volume shape,
- voxel size,
- discretization type,
- CTF evaluator,
- preprocessing function,
- grid and upsampling metadata.

## 5.2 Projection precompute

At the start of each dense E-step, the code computes and stores all rotated projections
$$
P_r \mu, \qquad r = 1, \dots, n_{\text{rot}}.
$$
This is the core dense-grid precompute. It converts a 3D volume problem into a 2D batched comparison problem.

This precompute is done in rotation batches to fit memory.

## 5.3 Translation handling

The code does **not** translate each projection separately. Instead it translates the batch of images in Fourier space via phase shifts. This is mathematically equivalent and much cheaper because:
- translations are diagonal in Fourier space,
- the same shifted images can be compared against all precomputed projections.

This is one of the main reasons the dense-grid implementation is tractable.

## 5.4 Dense E-step kernel

For each image batch:
1. preprocess the images,
2. apply `CTF / noise_variance`,
3. generate all translated versions of the images,
4. flatten `(image, translation)` into one long batch,
5. multiply by `projections.T` to obtain all cross terms,
6. separately multiply `CTF^2 / noise_variance` by `|projections|^2.T` to obtain all norm terms,
7. normalize into posterior probabilities.

The computational heart is therefore two large GEMMs plus Fourier phase shifts.

## 5.5 Dense M-step kernel

For each image batch and rotation block:
1. preprocess and CTF-weight the batch,
2. generate all translated images,
3. flatten to `(image × translation)`,
4. reshape probabilities so that one matrix multiply performs
$$
   \sum_{i,t} \gamma_{i,r,t} \cdot \text{shifted\_image}_{i,t}
$$
   for every rotation `r`,
5. pack the result into half-spectrum layout,
6. backproject it into the 3D Fourier volume to accumulate `Ft_y`,
7. separately sum probabilities over translations and backproject the `CTF^2 / noise_variance` weights to accumulate `Ft_ctf`.

Again, the expensive work is a GEMM plus a batched adjoint slice operator.

## 5.6 Streaming over image batches

The high-level loop processes the dataset in image batches. For each batch it:
1. computes posteriors,
2. records hard assignments if desired,
3. accumulates `Ft_y` and `Ft_ctf`.

This is the right coarse-grained organization because the M-step sufficient statistics are additive over images.

---

## 6. Why this is efficient

The dense homogeneous path is efficient for the following structural reasons.

### 6.1 One projection precompute per EM iteration

Each candidate rotation is projected once per iteration, not once per image. This turns the cost profile into:
- one 3D-to-2D precompute over rotations,
- then many 2D batched comparisons.

### 6.2 Translations are pushed onto the images

Using Fourier phase shifts avoids repeatedly resampling the projections for every translation. This is a major reduction in work.

### 6.3 Hidden-state dependence is written as GEMMs

Both of the expensive E-step terms become matrix-matrix products:
- cross term,
- norm term.

Likewise, the main M-step accumulation becomes a matrix-matrix product before backprojection.

This is exactly the right thing to do on GPU.

### 6.4 The M-step is additive over image shards

`Ft_y` and `Ft_ctf` are sums over images. This makes the single-volume dense path naturally compatible with:
- image-batch streaming,
- distributed data parallelism,
- future multi-GPU all-reduce.

### 6.5 Half-spectrum packing is already used in the adjoint path

Before backprojection, the code packs 2D Fourier images into Hermitian half-image layout. That reduces redundant storage and aligns the backprojection with the Fourier symmetry structure.

### 6.6 GPU-aware slice and adjoint kernels already exist

The low-level slice / adjoint layer already dispatches to CUDA for the common interpolation orders on GPU. This is the correct substrate for keeping the dense EM path small and high-level.

---

## 7. Relationship to heterogeneity EM

The package also contains a low-rank heterogeneity extension, roughly of the form
$$
\mu + U z,
\qquad z \sim \mathcal N(0, \Gamma).
$$
The uploaded note derives the key E-step quantity after integrating out `z`. In that setting the posterior score becomes the homogeneous residual term plus a correction involving a small dense latent-space system:
$$
H_{i,r} = U^* P_r^* C_i \Lambda_i^{-1} C_i P_r U + \Gamma^{-1},
$$
and a linear term `b_{i,r,t}`. The implementation computes this correction through Cholesky solves in the principal-component dimension.

Conceptually, that extension sits on top of the same dense-grid skeleton:
- precompute mean projections,
- precompute basis projections,
- reuse shifted, CTF-weighted images,
- express the latent correction with matrix-matrix operations and small dense solves.

That is why the heterogeneous code should be treated as a later extension of the homogeneous core, not as the primary organizing principle for the module layout.

---

## 8. What must be preserved during cleanup

A refactor should preserve the following invariants.

1. **The dense homogeneous kernel is built around two E-step GEMMs and two additive M-step statistics.**
2. **Projection precompute is iteration-level state.**
3. **Translations are handled by Fourier phase shifts on images.**
4. **`Ft_y` and `Ft_ctf` are additive over image batches and therefore over devices.**
5. **The low-level slice and adjoint operators remain the only place that knows about CUDA / interpolation details.**
6. **Static forward-model metadata remains centralized in `ForwardModelConfig` or an equivalent typed object.**

If these six facts are preserved, the code can be made much smaller without changing the algorithm.

---

## 9. Observed caveats in the current implementation

These do not change the mathematical core, but they matter for cleanup.

### 9.1 The dense homogeneous path is mixed with other concerns

The current `recovar/em` package mixes:
- homogeneous dense-grid EM,
- low-rank heterogeneity EM,
- covariance work,
- refined-grid utilities,
- legacy and newer Equinox-based APIs.

That makes the simplest path look much more complicated than it really is.

### 9.2 The M-step API name is misleading

`M_with_precompute` does not actually perform the same style of forward projection precompute as `E_with_precompute`. It is better understood as “accumulate dense homogeneous mean statistics from probabilities”.

### 9.3 Accumulator lifecycle is not explicit enough

`Ft_y` and `Ft_ctf` are crucial iteration-local statistics. Their initialization, accumulation, and reset should be explicit in the public dense homogeneous API.

### 9.4 Discretization consistency should be made explicit

The dense homogeneous path should either:
- fully support a discretization choice end-to-end, or
- explicitly reject unsupported combinations.

The code should not expose a discretization option at the API level if the forward and adjoint sides do not both honor it.

### 9.5 Memory planning is heuristic and scattered

The batch-size logic is currently spread across multiple functions with ad hoc constants. The algorithm deserves a single planner object that owns those decisions.

---

## 10. Recommended long-term conceptual decomposition

For permanent documentation, the dense homogeneous path should be documented in exactly five conceptual layers:

1. **Model**
   - `y_i = S_t C_i P_r \mu + \varepsilon_i`
2. **Posterior evaluation**
   - precompute projections,
   - two GEMMs,
   - normalize over hidden states
3. **Mean-statistics accumulation**
   - accumulate `Ft_y`,
   - accumulate `Ft_ctf`
4. **Solve / post-process**
   - RELION-style regularized filtered reconstruction
5. **Execution policy**
   - batching,
   - GPU execution,
   - future multi-GPU reduction

If the code and docs both follow that decomposition, the implementation will become much easier to maintain.

---

## 11. One-sentence summary

The dense single-volume EM path in RECOVAR is fundamentally a very clean algorithm: **precompute all rotated projections once, evaluate posteriors by large batched Fourier-space matrix products, accumulate two additive Fourier sufficient statistics, and reconstruct the updated mean by a regularized filtered backprojection solve.**
