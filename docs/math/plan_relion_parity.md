# Plan: RELION-parity dense single-volume EM in JAX/CUDA

## 1. Goal

Bring `recovar/em/` to feature parity with RELION's `relion_refine --auto_refine`
for **single-class, single-volume** homogeneous refinement. Match RELION's output
(resolution, convergence trajectory, per-image assignments) while being at least
as fast on the same GPU.

**Out of scope**: K-class classification, heterogeneity/PPCA, helical symmetry,
tilt series, CTF refinement, Bayesian polishing.

---

## 2. Current state

We have a working dense-grid EM with:
- E-step: explicit phase-shifted copies + GEMM (all rotations × all translations)
- M-step: GEMM + CUDA adjoint backprojection
- RELION-style Wiener solve
- Split half-set support (`split_E_M_v2`)
- Full-spectrum operations (N = H×W = 16384 for 128px images)

**Performance** (5000 images, 128px, order 3 = 36,864 rotations, 7×7 translations, A100-80GB):
- Our code: 68s/iter (full spectrum), 19s/iter (half-spectrum benchmark)
- RELION 5.0.1 on same data: ~163s/iter (steady state), ~45s/iter (early, low-res)

**What RELION does that we don't**:
1. Half-spectrum (rfft) throughout — we use full spectrum for GEMMs
2. Fourier crop to current resolution — we always use full 128² pixels
3. Two-pass adaptive oversampling with significant weight pruning
4. FSC-driven resolution → controls Fourier crop size
5. Per-iteration noise and signal prior estimation from data

---

## 3. RELION reference infrastructure (Step 0 — do this FIRST)

Before any optimization, create a reproducible RELION reference run that we
compare against at every subsequent step.

### 3.1 Reference dataset

Use our existing synthetic dataset: 5000 images, 128px, noise_level=1.0,
`recovar/assets/vol` reference, at `/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/`.

### 3.2 RELION reference run

Run RELION with EXACTLY the parameters we want to match:

```bash
relion_refine \
  --i particles.star --ref reference_init.mrc \
  --o relion_ref/run \
  --auto_refine --split_random_halves \
  --particle_diameter 200 \
  --ini_high 30 \
  --healpix_order 3 \
  --offset_range 3 --offset_step 1 \
  --oversampling 1 \
  --pad 2 \
  --gpu 0 \
  --j 4
```

Use MPI (`mpirun -n 3 relion_refine_mpi ...`) to enable split_random_halves.

### 3.3 Extract RELION per-iteration reference data

Write a script `scripts/extract_relion_reference.py` that reads RELION's output
STAR files and saves per-iteration:

- `current_resolution` (Å) and `current_size` (pixels)
- Per-shell noise variance (`rlnSigma2Noise`)
- Per-shell signal prior (`rlnSignalPrior` / tau²)
- Per-image best rotation and translation (`rlnAngleRot/Tilt/Psi`, `rlnOriginXAngst/YAngst`)
- Reconstructed half-map volumes (MRC)
- FSC between half-maps
- Angular accuracy estimate
- Total number of significant orientations per image

Save as `.npz` or `.pkl` per iteration. These are the **ground truth** for all
subsequent steps.

### 3.4 Comparison framework

Write `tests/integration/test_relion_comparison.py` with helpers:

```python
def compare_hard_assignments(ours, relion, rotation_grid, tolerance_deg=15):
    """Compare best rotations allowing for angular grid discretization."""

def compare_volumes(ours, relion, volume_shape):
    """FSC between our reconstruction and RELION's."""

def compare_resolution_trajectory(ours, relion):
    """Compare current_size at each iteration."""
```

### Deliverable

A `relion_reference/` directory with per-iteration ground truth, and a
comparison test framework. All subsequent steps use this to validate.

---

## 4. Step 1: Half-spectrum GEMMs

### What changes

Replace the E-step and M-step GEMMs to operate on the rfft-packed half-spectrum:
N_half = H × (W//2 + 1) = 8320 instead of N = H × W = 16384.

### Mathematical justification

For real-valued images in real space, the Fourier transform is Hermitian:
f(-k) = conj(f(k)). The full inner product can be recovered from the half:

```
Re<a, b>_full = Re[sum_half w(k) · conj(a(k)) · b(k)]
```

where w(k) = 1 for DC and Nyquist columns, w(k) = 2 for all other columns.

### What to change (4 functions)

**1. E-step cross-term** (`core.py:82`, `compute_dot_products_eqx`):

Currently:
```python
shifted = batch_trans_translate_images(batch_ctf, translations, image_shape)  # (n_img, n_trans, N)
shifted_flat = shifted.reshape(n_img * n_trans, N)
cross = -2 * (conj(shifted_flat) @ projections.T).real  # GEMM on N=16384
```

Change to:
```python
shifted = batch_trans_translate_images(batch_ctf, translations, image_shape)
shifted_half = full_image_to_half_image(shifted.reshape(-1, N), image_shape)  # (-1, N_half)
proj_half = full_image_to_half_image(projections, image_shape)  # (n_rot, N_half)
cross = -2 * (conj(shifted_half) * half_weights @ proj_half.T).real  # GEMM on N_half=8320
```

Note: `half_weights` is absorbed into one side of the product (multiply once,
not per GEMM call). Precompute `proj_half_weighted = proj_half * half_weights`
at the start of the iteration.

**2. E-step norm-term** (`core.py:98`, `compute_CTFed_proj_norms_eqx`):

Currently:
```python
CTFs = config.compute_ctf(ctf_params) ** 2 / noise_variance
return CTFs @ projections.T  # (n_img, n_rot) via N=16384
```

Change to:
```python
CTFs_half = full_image_to_half_image(CTFs, image_shape)
proj_abs2_half = full_image_to_half_image(|projections|², image_shape)
return (CTFs_half * half_weights) @ proj_abs2_half.T  # N_half=8320
```

**3. M-step GEMM** (`m_step.py:146`, `sum_up_images_fixed_rots_eqx`):

Currently:
```python
P = probs.swapaxes(0,1).reshape(n_rot, n_img*n_trans)
summed = P @ shifted_flat                              # (n_rot, N=16384)
summed_half = full_image_to_half_image(summed, ...)    # convert AFTER GEMM
Ft_y = adjoint_slice_volume(summed_half, ..., half_image=True)
```

Change to:
```python
shifted_half = full_image_to_half_image(shifted_flat, image_shape)  # convert BEFORE GEMM
P = probs.swapaxes(0,1).reshape(n_rot, n_img*n_trans)
summed_half = P @ shifted_half                                       # (n_rot, N_half) — already half!
Ft_y = adjoint_slice_volume(summed_half, ..., half_image=True)       # no conversion needed
```

**4. M-step CTF term** (`m_step.py:153`):

Same pattern — convert CTF²/nv to half before the GEMM.

### What NOT to change

- `batch_trans_translate_images` still operates on full spectrum (phase shifts
  need all frequencies to be correct). Convert to half AFTER shifting.
- `adjoint_slice_volume` already accepts `half_image=True` — no change.
- `slice_volume` (projection precompute) — add `half_image=True` flag to
  output projections directly in half layout, avoiding the separate conversion.
  The CUDA kernel already supports this.

### Half-weights computation

```python
def make_half_image_weights(image_shape):
    H, W = image_shape
    w = 2.0 * jnp.ones((H, W // 2 + 1), dtype=jnp.float32)
    w = w.at[:, 0].set(1.0)    # first packed column (Nyquist freq along W)
    w = w.at[:, -1].set(1.0)   # last packed column (DC freq along W)
    return w.reshape(-1)        # (N_half,)
```

**CRITICAL**: verify this matches `get_real_fft_packed_last_axis_indices` column
ordering. Write a unit test that computes `sum_k |f(k)|²` via full and half
spectrum and verifies exact equality.

### Risk: phase shifts before half-image conversion

Phase shifts (`S_t`) multiply each Fourier pixel by `exp(2πi k·t)`. The full
spectrum has redundant conjugate pairs: `f(-k) = conj(f(k))` and the phase
shift preserves this symmetry: `S_t f(-k) = exp(-2πi k·t) conj(f(k)) = conj(S_t f(k))`.
So shifting then converting to half is safe. Converting to half then shifting
would also be safe if the phase shift function handles the half layout — but
our `batch_trans_translate_images` assumes full spectrum, so we shift first.

### Tests

1. `test_half_inner_product`: for random complex Hermitian arrays, verify
   `Re<a,b>_full == Re[sum(conj(a_half) * half_weights * b_half)]` to machine
   precision.

2. `test_e_step_half_matches_full`: run both full and half E-step on same
   inputs, verify probability outputs match within `atol=1e-5`.

3. `test_m_step_half_matches_full`: same for M-step, verify Ft_y and Ft_ctf
   match within `atol=1e-6`.

4. `test_full_iteration_half_matches`: one complete EM iteration, verify
   hard assignments match and mean volumes match within `atol=1e-4`.

### Expected outcome

- E-step: 72ms → 35ms (2.07×)
- M-step: 93ms → 62ms (1.51×)
- Total iteration (5K images, order 3, 7×7): 33s → 19s (1.71×)

---

## 5. Step 2: Fourier cropping to current resolution

### What changes

At each iteration, determine the current resolution from FSC, crop all
Fourier-space images and projections to a smaller grid of size
`current_size × current_size`, and run E-step + M-step on the small grid.

### Why this is the biggest win

At 30Å resolution (early iterations), `current_size ≈ 18`. The E-step GEMM
inner dimension drops from N_half=8320 to `18×10=180`: a **46× FLOP reduction**.

### Key design decisions

**Decision 1: How to crop images.**

Fourier cropping = select the low-frequency sub-grid from the centered spectrum.
For a (H, W) image cropped to (cs, cs):

```python
def crop_fourier_image(image_2d, current_size):
    """Crop centered Fourier image from (H, W) to (cs, cs)."""
    H, W = image_2d.shape
    c = H // 2  # center
    r = current_size // 2
    return image_2d[c-r:c+r, c-r:c+r]  # or c-r:c+r+1 for odd cs
```

For half-spectrum (H, W//2+1) → (cs, cs//2+1): similar crop along the first
axis, keep all columns of the half.

Existing code: `downsample_vol_by_fourier_truncation` in `utils/helpers.py:627`
does this for volumes. Adapt for 2D images.

**Decision 2: How to handle translations at cropped resolution.**

THIS IS THE HARD PART. Translations are phase shifts: `S_t(k) = exp(2πi k·t)`.
The frequency vector `k` depends on the grid size:
- At original (128, 128): `k_j = j/128` for `j = -64, ..., 63`
- At cropped (18, 18): the low-frequency subset is `k_j = j/128` for `j = -9, ..., 8`

The physical frequencies are the SAME — they're a subset of the original grid.
But `batch_trans_translate_images(image, translations, image_shape=(18,18))` would
compute `k_j = j/18`, which is WRONG.

**Solution**: Apply phase shifts at the ORIGINAL pixel spacing. Three options:

**(a) Shift before crop** (simple, slightly wasteful):
```python
shifted_full = batch_trans_translate_images(images, translations, original_image_shape)
shifted_cropped = crop_fourier(shifted_full)
```
Cost: N_full per translation for shifting, then crop. Shifting is O(n_img × n_trans × N)
which is cheap vs the GEMM.

**(b) Custom crop-aware shift** (optimal):
```python
def shift_cropped_images(images_cropped, translations, original_image_shape, current_size):
    """Phase shifts using original-resolution k vectors, applied to cropped images."""
    # Compute k vectors for the cropped sub-grid at original spacing
    k_y = jnp.fft.fftfreq(original_image_shape[0])[:current_size]  # low-freq subset
    k_x = jnp.fft.fftfreq(original_image_shape[1])[:current_size]
    phases = jnp.exp(2j * jnp.pi * (k_y[:, None] * t[0] + k_x[None, :] * t[1]))
    return images_cropped * phases
```
Cost: O(n_img × n_trans × cs²) — much cheaper at small cs.

**(c) Precompute phase shift matrices for the crop grid**:
Compute once per iteration, reuse across image batches.

**Recommendation**: Start with (a) for correctness, optimize to (b) later.

**Decision 3: Quantize current_size to avoid JIT recompilation.**

JAX recompiles every JIT'd function when input shapes change. With 20 iterations
at 20 different `current_size` values, this means 20 compilations × ~30s = 10
minutes of compilation, dominating compute savings.

**Solution**: Quantize to a fixed set of allowed sizes:
```python
ALLOWED_SIZES = [16, 24, 32, 48, 64, 96, 128]

def quantize_current_size(cs):
    return min(s for s in ALLOWED_SIZES if s >= cs)
```

This limits recompilations to ≤7. First compilation per size takes ~30s; all
subsequent iterations at that size are instant.

**Decision 4: Forward projection at cropped resolution.**

`slice_volume` can output at any `image_shape`. Pass `image_shape=(cs, cs)` to
project only the low-frequency pixels. The CUDA kernel samples the 3D volume at
the Fourier coordinates of the (cs, cs) grid — these are the correct low-frequency
voxels.

Verify: for a (128, 128, 128) volume, `slice_volume(vol, rots, (18, 18), (128, 128, 128))`
should return the same values as `crop(slice_volume(vol, rots, (128, 128), (128, 128, 128)))`.

**Decision 5: Adjoint at cropped resolution.**

`adjoint_slice_volume(images, rots, (18, 18), (128, 128, 128))` should insert
the (18, 18) images into the correct low-frequency voxels of the (128, 128, 128)
volume. Verify this with a forward-adjoint dot product test.

### Implementation outline

```python
def run_em_iteration_with_cropping(dataset, mean, ..., current_size):
    cs = quantize_current_size(current_size)
    image_shape_cropped = (cs, cs)

    # Projections at cropped resolution
    proj_half = slice_volume(mean, rotations, image_shape_cropped, volume_shape,
                             disc_type, half_image=True)

    for image_batch in dataset:
        # Shift at original resolution, then crop
        shifted = batch_trans_translate_images(batch_ctf, translations, original_image_shape)
        shifted_cropped_half = crop_to_half(shifted, cs)

        # E-step GEMM at cropped size: inner dim = cs*(cs//2+1)
        cross = -2 * (conj(shifted_cropped_half) * weights @ proj_half.T).real

        # ... softmax, M-step as before but at cropped size ...

        # M-step backproject into full volume
        Ft_y = adjoint_slice_volume(summed_half, rots, image_shape_cropped, volume_shape, ...)

    # Solve on full volume (Ft_y has zeros at high freq → regularization handles it)
    new_mean = post_process_from_filter(dataset, Ft_ctf, Ft_y, tau=prior, ...)
```

### Resolution estimation

After the solve, compute FSC between half-map volumes:
```python
fsc = get_fsc_gpu(mean_half1, mean_half2, volume_shape)
max_res_shell = find_fsc_threshold(fsc, threshold=0.143)
next_current_size = 2 * max_res_shell  # Nyquist: need 2 pixels per cycle
```

Existing code: `regularization.get_fsc_gpu` and `locres.find_fsc_resol`.

### "Oracle mode" for debugging

Before implementing our own FSC → current_size, add an option to READ RELION's
`current_size` from the reference data (Step 0). This isolates the compute
optimization from the statistical model:

```python
def run_em_iteration(..., current_size_override=None):
    if current_size_override is not None:
        cs = current_size_override  # use RELION's value
    else:
        cs = compute_current_size_from_fsc(...)
```

### Tests

1. `test_crop_fourier_roundtrip`: crop then pad with zeros → original up to
   the cropped frequencies.

2. `test_projection_at_cropped_resolution`: verify
   `slice_volume(vol, rots, (18,18), (128,128,128))` matches
   `crop(slice_volume(vol, rots, (128,128), (128,128,128)))`.

3. `test_adjoint_dot_product_cropped`: verify `<Ax, y> = <x, A*y>` at cropped
   resolution. This tests the forward-adjoint consistency of the CUDA kernels
   at non-standard image_shape.

4. `test_phase_shifts_at_cropped_resolution`: verify shift-then-crop matches
   custom cropped-shift function.

5. `test_iteration_with_oracle_current_size`: run one iteration at each of
   RELION's `current_size` values. Compare hard assignments and Ft_y against
   RELION reference.

6. `test_jit_recompilation_count`: verify that quantized sizes limit
   recompilations to ≤ len(ALLOWED_SIZES).

### Expected outcome

At `current_size=18` (early iterations):
- GEMM inner dim: 8320 → 180 (half of 18×10). **46× fewer FLOPs.**
- Iteration time: ~0.5s instead of 19s.
- At `current_size=64` (converged): ~5s. At `current_size=128`: back to 19s.

---

## 6. Step 3: Split half-sets and gold-standard FSC

### What changes

Run the E-step and M-step independently on two random half-sets of images.
Compute FSC between the two independent reconstructions to estimate resolution.

### Existing code

`split_E_M_v2` in `em/iterations.py:57` already does this. It:
1. Splits images into two halves
2. Runs `E_M_batches_2` on each half independently
3. Calls `finish_up_M_step` on each
4. Computes FSC via `get_fsc_gpu`
5. Computes RELION-style prior via `compute_relion_prior`
6. Estimates noise via `estimate_noise_level_no_masks`

### What needs to change

Wire the output FSC → `current_size` for Step 2. Currently `split_E_M_v2`
computes `current_pixel_res` but doesn't use it to control Fourier cropping.

Add: `current_size = 2 * current_pixel_res` after the FSC computation, pass it
to the next iteration's E-step and M-step.

### Tests

Compare `current_pixel_res` at each iteration against RELION's
`rlnCurrentResolution` from the reference run.

---

## 7. Step 4: Noise and signal prior estimation

### What changes

Ensure our per-iteration noise and signal prior updates match RELION's.

### Existing code

- `reconstruction/noise.py:716` — `estimate_noise_level_no_masks`
- `reconstruction/regularization.py:75` — `compute_relion_prior`
- `reconstruction/noise.py:1084` — `make_radial_noise`

### RELION's update formula

Signal prior (tau²) from FSC between half-maps:
```
tau²(k) = FSC(k) / (1 - FSC(k)) × sigma²(k)
```
where sigma²(k) is the noise power at shell k.

RELION applies this with smoothing and a floor to prevent tau² = 0 at high
frequencies.

### What needs verification

1. Does `compute_relion_prior` implement exactly this formula?
2. Does the noise estimation use the correct residuals (current assignments)?
3. Is the update schedule correct (every iteration, or every N iterations)?

### Tests

Compare per-shell noise and tau² against RELION's `rlnSigma2Noise` and
`rlnReferenceSigma2` at each iteration.

---

## 8. Step 5: Two-pass adaptive oversampling

### What changes

Split the E-step into two passes:
1. **Coarse pass**: evaluate ALL rotations at base sampling and current_size.
   Identify the significant orientations per image (top 99.9% of weight).
2. **Fine pass**: evaluate ONLY the significant orientations at oversampled
   angles (2× finer) and possibly higher Fourier resolution.

### Why this is architecturally hard

Pass 1 is a dense GEMM (our current approach — all images × all rotations).
Pass 2 is per-image sparse (each image has ~100-500 significant rotations).

The dense GEMM works because all images share the same rotation grid. In pass 2,
each image has a DIFFERENT set of candidate rotations (the children of its
significant coarse rotations).

### Approaches for pass 2

**(a) Padded batched evaluation** (recommended for first implementation):
```python
# max_sig = maximum significant orientations per image (cap at ~500)
# Each significant coarse rotation spawns 4 oversampled children → max_sig*4 fine rotations
sig_rot_indices = jnp.zeros((n_img, max_sig * 4), dtype=int)  # per-image rotation indices
# Gather projections: (n_img, max_sig*4, N_half) — one projection per (image, candidate)
# This breaks the GEMM structure but can be done as batched dot products
```

**(b) FFT cross-correlation** (natural for per-image):
At cropped resolution (cs=18), each FFT is tiny (18×18). For 200 images × 500
candidate rotations: 100K FFTs of 18×18 — about 0.5ms total. Very fast.

**(c) Group-by-rotation** (optimization):
Many images may share similar significant rotations. Group them and batch the
GEMM per group. Complex to implement but most efficient.

**Recommendation**: implement (b) first — it's simple, fast at cropped resolution,
and naturally handles per-image candidates. Optimize to (a) or (c) if needed.

### Oversampling grid

For each significant coarse rotation (healpix pixel), generate 4 child pixels
at the next healpix level. Existing code: `em/sampling.py` has
`get_healpix_children` and `get_oversampled_rotation_grid`.

For translations: if coarse step is 1 pixel, oversampled step is 0.5 pixel
within ±1 pixel of the coarse best. Existing: `get_oversampled_translation_grid`.

### Significance threshold

RELION uses `adaptive_fraction = 0.999`: keep orientations that contribute to
the top 99.9% of total posterior weight per image.

```python
def find_significant_mask(weights, adaptive_fraction=0.999):
    """Per-image mask of significant orientations."""
    sorted_w = jnp.sort(weights, axis=-1)[:, ::-1]
    cumsum = jnp.cumsum(sorted_w, axis=-1) / weights.sum(axis=-1, keepdims=True)
    threshold_idx = jnp.argmax(cumsum >= adaptive_fraction, axis=-1)
    threshold_val = sorted_w[jnp.arange(len(sorted_w)), threshold_idx]
    return weights >= threshold_val[:, None]
```

### Tests

1. `test_significance_mask`: verify mask keeps correct fraction of weight.
2. `test_oversampled_grid_matches_relion`: verify our child rotation generation
   matches RELION's `get_healpix_children`.
3. `test_pass2_matches_dense`: on a small problem, verify that pass 1 + pass 2
   gives the same posteriors as a single dense pass at the oversampled resolution.
4. `test_two_pass_matches_relion`: compare per-image significant counts and
   final assignments against RELION reference.

---

## 9. Step 6: Maximum significants cap

### What changes

After pass 1, cap the number of significant orientations per image to
`maximum_significants` (default: 500). This prevents pathological cases.

### Implementation

```python
def cap_significants(weights, max_sig=500):
    topk_indices = jnp.argsort(weights, axis=-1)[:, -max_sig:]
    mask = jnp.zeros_like(weights, dtype=bool)
    mask = mask.at[jnp.arange(len(mask))[:, None], topk_indices].set(True)
    return mask
```

### Tests

Verify capped mask still gives good convergence (same final resolution as uncapped).

---

## 10. Step 7: Full integration

### What changes

Wire everything into a single `refine_single_volume()` function that runs the
complete auto-refine loop matching RELION's behavior:

```python
def refine_single_volume(dataset, init_volume, init_resolution, ...):
    current_size = resolution_to_size(init_resolution)
    state_half1 = EMState(init_volume, ...)
    state_half2 = EMState(init_volume, ...)

    for iteration in range(max_iter):
        cs = quantize(current_size)

        # E+M on each half-set (fused, half-spectrum, Fourier-cropped)
        for state, half_dataset in [(state_half1, data1), (state_half2, data2)]:
            run_em_iteration(state, half_dataset, cs, ...)

        # FSC → resolution → next current_size
        fsc = get_fsc_gpu(state_half1.mean, state_half2.mean, ...)
        current_size = fsc_to_current_size(fsc)

        # Update noise and prior
        noise = estimate_noise(...)
        prior = compute_relion_prior(fsc, noise, ...)

        # Check convergence
        if angular_change < threshold and no_gain_count >= patience:
            break

    return mean, fsc, assignments
```

### Final comparison against RELION

Run both on EMPIAR-10028 subset (or our synthetic data) for 25 iterations.
Compare:
- Final resolution (FSC = 0.143): must match within 1 Fourier shell
- Final volume: FSC between RELION and our reconstruction > 0.99 at all shells below resolution
- Wall-clock time: must be ≤ RELION on same GPU
- Convergence trajectory: current_size sequence should match within ±1 step

---

## 11. Existing code inventory

| Need | Existing code | File | Status |
|------|--------------|------|--------|
| Half-spectrum conversion | `full_image_to_half_image` | `core/fourier_transform_utils.py:291` | Ready |
| Half-image forward slice | `slice_volume(..., half_image=True)` | `core/slicing.py:221` | Ready |
| Half-image adjoint slice | `adjoint_slice_volume(..., half_image=True)` | `core/slicing.py:331` | Ready |
| Fourier volume downsample | `downsample_vol_by_fourier_truncation` | `utils/helpers.py:627` | Ready |
| Image downsample | `downsample_images` | `data_io/downsample.py:24` | Ready |
| Frequency mask | `CryoEMDataset.get_valid_frequency_indices(rad)` | `data_io/cryoem_dataset.py:565` | Ready |
| FSC | `get_fsc_gpu(vol1, vol2, shape)` | `reconstruction/regularization.py:128` | Ready |
| FSC → resolution | `locres.find_fsc_resol(fsc, threshold)` | `heterogeneity/locres.py` | Ready |
| Signal prior from FSC | `compute_relion_prior(datasets, noise, v1, v2)` | `reconstruction/regularization.py:75` | Verify vs RELION |
| Noise estimation | `estimate_noise_level_no_masks(dataset, ...)` | `reconstruction/noise.py:716` | Verify vs RELION |
| Radial noise model | `make_radial_noise(PS, image_shape)` | `reconstruction/noise.py:1084` | Ready |
| RELION Wiener solve | `post_process_from_filter` | `reconstruction/relion_functions.py:427` | Ready |
| Split half-set EM | `split_E_M_v2` | `em/iterations.py:57` | Adapt for cropping |
| HEALPix children | `get_healpix_children(parents, level)` | `em/sampling.py` | Ready |
| Oversampled rot grid | `get_oversampled_rotation_grid(parents, level, order)` | `em/sampling.py` | Ready |
| Oversampled trans grid | `get_oversampled_translation_grid(parents, offset, order)` | `em/sampling.py` | Ready |
| Phase shifts | `batch_trans_translate_images` | `core/` | Ready |
| Dense E-step GEMM | `compute_dot_products_eqx` | `em/core.py:82` | Modify for half |
| Dense M-step GEMM | `sum_up_images_fixed_rots_eqx` | `em/m_step.py:117` | Modify for half |
| Fused engine | `engine_fused.py`, `engine_v2.py` | `em/dense_single_volume/` | Modify for crop |

New code needed: ~500 lines total.

---

## 12. Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Half-image weights wrong (wrong DC/Nyquist columns) | HIGH — silently wrong results | Unit test: full vs half inner product to machine precision |
| Phase shifts at cropped resolution use wrong k vectors | HIGH — wrong translations | Option (a): shift before crop. Test: shift-then-crop vs crop-aware-shift |
| JIT recompilation at every new current_size | MEDIUM — compilation dominates compute | Quantize to ALLOWED_SIZES = [16, 24, 32, 48, 64, 96, 128] |
| adjoint_slice_volume with mismatched image_shape/volume_shape | HIGH — garbage backprojection | Test: forward-adjoint dot product at (18,18) image, (128,128,128) volume |
| FSC/resolution differs from RELION → different crop schedule | MEDIUM — different convergence | Oracle mode: use RELION's current_size for debugging |
| Noise/prior estimation doesn't match RELION | MEDIUM — different regularization | Compare per-shell values against RELION's model.star |
| Pass 2 per-image sparse evaluation is slow | LOW — only at Step 5 | FFT approach at cropped resolution is fast (18×18 FFTs) |
| Memory: half-spectrum images + cropped projections + prob tensor | LOW | Cropping massively reduces memory pressure |

---

## 13. Implementation order and dependencies

```
Step 0: RELION reference data ──────────────────────────────┐
                                                            │
Step 1: Half-spectrum GEMMs ────────────────────────────────┤
  (no dependency on Step 0, but validate against it)        │
                                                            │
Step 2: Fourier cropping ──────────────────── needs Step 1 ─┤
  (half-spectrum + crop = maximum GEMM reduction)           │
                                                            │
Step 3: Split half-sets + FSC ─── needs Step 2 ─────────────┤
  (FSC drives current_size)                                 │
                                                            │
Step 4: Noise/signal prior ──── needs Step 3 ───────────────┤
  (prior from FSC, noise from residuals)                    │
                                                            │
Step 5: Two-pass adaptive ──── needs Steps 2+4 ─────────────┤
  (pass 1 at cropped res, pass 2 at oversampled)            │
                                                            │
Step 6: Max significants cap ── needs Step 5 ───────────────┤
                                                            │
Step 7: Full integration ───── needs all above ─────────────┘
```

Each step gets its own branch (`claude/em-relion-step-N`), its own tests, and
is validated against the RELION reference before merging and proceeding.
