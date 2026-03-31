# Plan: RELION-parity dense single-volume EM in JAX/CUDA

## Goal

Incrementally bring `recovar/em/dense_single_volume/` to feature parity with
RELION's single-class `relion_refine --auto_refine` on GPU, for a single
homogeneous volume (no K-classes, no heterogeneity). Each step adds exactly
one layer of complexity, with numerical comparison against RELION output at
every step.

## Scope

**In scope**: single volume, split half-sets, FSC-based resolution and
regularization, Fourier cropping, adaptive oversampling, significant weight
pruning, noise estimation, gold-standard FSC.

**Out of scope (for now)**: multiple classes, heterogeneity/PPCA, helical
symmetry, tilt series, CTF refinement, Bayesian polishing.

## Comparison methodology

At each step, run both RELION and our code on the **same synthetic dataset**
(5000 images, 128px, noise_level=1, `recovar/assets/vol` reference). Compare:
- Per-image hard assignments (best rotation, best translation)
- Posterior probability distribution (top-K weights)
- Reconstructed volume (FSC between RELION and recovar outputs)
- Per-iteration convergence trajectory (rotation error, translation error)
- Wall-clock time per iteration

RELION reference run: `relion_refine --healpix_order 3 --offset_range 3
--offset_step 1 --oversampling 0` on the same hardware.

---

## Step 0: Current baseline (DONE)

What we have:
- Dense E-step: explicit shift + GEMM on full spectrum
- Dense M-step: GEMM + adjoint backprojection on full spectrum
- RELION-style Wiener solve (`post_process_from_filter`)
- `split_E_M_v2` with half-set splitting, FSC, noise estimation

What's missing: everything below.

Measured: 68s/iter (full spectrum), 19s/iter (half spectrum benchmark).
RELION: 163s/iter on same data (but includes CPU M-step + overhead).

---

## Step 1: Half-spectrum throughout

**What**: Replace all E-step and M-step GEMMs to operate on the rfft-packed
half-spectrum (N_half = H × (W//2+1) instead of N = H × W).

**Why**: Demonstrated 1.7× speedup (72→35ms E-step, 93→62ms M-step).
Images are real-valued in real space → Hermitian-symmetric in Fourier space
→ half the pixels carry all the information.

**Implementation**:
- Convert images and projections to half-spectrum immediately after loading/slicing
- E-step GEMM: `conj(shifted_half) @ (proj_half * half_weights).T`
  where `half_weights = 2` for interior columns, `1` for DC/Nyquist
- M-step GEMM: `P @ shifted_half → summed_half` (already in half layout for adjoint)
- Forward slice: use `half_image=True` flag (already supported by CUDA kernel)
- Adjoint slice: already uses half-image

**Test**: Numerical equivalence to Step 0 (same hard assignments, Ft_y within 1e-6).

**Expected**: 19s/iter (from benchmark). ~3.6× vs original.

---

## Step 2: Fourier cropping to current resolution

**What**: At each iteration, crop all Fourier-space arrays to a grid matching
the current resolution estimate (from FSC). Work on an effective image size of
`current_size × current_size` instead of `128 × 128`.

**Why**: This is RELION's single biggest optimization. At early iterations
(30Å resolution, 4.25Å/pixel), `current_size ≈ 18`, and all GEMMs operate on
`18 × 10 = 180` half-pixels instead of `128 × 65 = 8320`. A **46× reduction**
in GEMM FLOPs.

**How RELION does it** (from `ml_optimiser.cpp`):
```
windowFourierTransform(Faux, Fimg, wsum_model.current_size);  // crop to current_size
```
Then all downstream compute uses `current_size²` pixels.

Resolution updates each iteration based on FSC:
```
mymodel.current_size = maxres * 2;  // maxres from FSC > 0.2 threshold
```

**Implementation**:
1. Add `crop_fourier_images(images_half, current_size)` — extracts the low-frequency
   sub-grid from the half-spectrum. Simple index selection.
2. Add `crop_fourier_volume(volume, current_size)` — extracts the low-frequency
   sub-volume. Forward slicing into the small grid.
3. Modify E-step: crop images and projections to `current_size` before GEMM.
4. Modify M-step: GEMM on cropped images, backproject into full volume at the
   correct frequency bins.
5. Modify solve: Wiener filter on full volume but only update shells ≤ current_size.
6. Add resolution estimator: FSC between half-maps → find shell where FSC > 0.143
   (or 0.5) → set `current_size = 2 × max_shell`.

**What gets cheaper**:
- E-step GEMM: `(n_img × n_trans, current_size²/2) @ (current_size²/2, n_rot)`
- M-step GEMM: same reduction
- Forward slice: fewer output pixels
- Adjoint slice: fewer input pixels, fewer voxels touched
- Phase shifts for translations: on smaller grid, cheaper
- Probability tensor: `n_img × n_rot × n_trans` unchanged (no pixel dependence)

**What stays the same**:
- Number of rotations, translations, images
- Probability normalization (softmax over rot × trans)
- Hard assignment logic

**Test**: Run 10 iterations of both RELION and recovar with `--ini_high 30`
(force 30Å initial resolution). Compare:
- `current_size` at each iteration (should match RELION's)
- Hard assignments at each iteration
- Final volume FSC

**Expected**: At `current_size=18`: GEMM on ~180 half-pixels vs 8320. About
**46× FLOP reduction** at early iterations, tapering as resolution improves.
Overall iteration time at early iterations: **~0.5s** instead of 19s.

---

## Step 3: Split half-sets and gold-standard FSC

**What**: Process two independent half-sets of images, each producing its own
reconstruction. Compute FSC between the two half-map volumes to estimate
resolution. This resolution drives `current_size` in Step 2.

**Why**: Gold-standard FSC is the standard method for resolution estimation
and overfitting prevention. RELION requires it. `split_E_M_v2` already
implements this, but the integration with Fourier cropping needs to be clean.

**Implementation**:
- Already have `split_E_M_v2` which runs `E_M_batches_2` on each half-set
- Add proper FSC computation → resolution → `current_size` feedback loop
- Each half-set gets its own reconstruction, own noise estimate
- Resolution from FSC drives the `current_size` for the NEXT iteration

**Test**: Compare half-map FSC curves and resolution estimates against RELION's
`run_itNNN_model.star` which records `rlnCurrentResolution`.

---

## Step 4: RELION-style noise and signal prior estimation

**What**: Estimate per-shell noise variance from data residuals, and signal
prior (tau²) from FSC between half-maps. Update both every iteration.

**Why**: RELION uses adaptive noise and signal estimates to weight the
likelihood properly. Without this, the prior/noise balance is wrong and
convergence suffers.

**How RELION does it**:
- Noise: from residuals of current assignments, radially averaged
- Signal prior (tau²): from FSC between half-maps via
  `tau² = FSC / (1 - FSC) × noise_power`  (RELION prior formula)

**Implementation**:
- Noise estimation: already in `reconstruction/noise.py`
  (`estimate_noise_level_no_masks`)
- Signal prior: already partially in `split_E_M_v2` via
  `regularization.compute_relion_prior`
- Need to wire these into the per-iteration loop properly, matching
  RELION's update schedule

**Test**: Compare per-shell noise and signal estimates against RELION's
`run_itNNN_model.star` fields `rlnSigma2Noise` and `rlnSignalPrior`.

---

## Step 5: Two-pass adaptive oversampling

**What**: Within each E-step, make TWO passes over orientations:
1. **Coarse pass**: evaluate all rotations at `current_size` (cheap). Identify
   which orientations have significant weight (top 99.9%).
2. **Fine pass**: evaluate only the significant orientations at full
   oversampled resolution. Skip all others.

**Why**: This is RELION's main algorithmic speedup. Instead of evaluating
all 36,864 rotations at full resolution, pass 1 identifies ~100-500
significant ones per image, and pass 2 only processes those.

**How RELION does it** (from `ml_optimiser.cpp`):
```cpp
for (int exp_ipass = 0; exp_ipass < nr_sampling_passes; exp_ipass++) {
    exp_current_oversampling = (exp_ipass == 0) ? 0 : adaptive_oversampling;
    getAllSquaredDifferences(...);
    convertAllSquaredDifferencesToWeights(..., exp_Mcoarse_significant, exp_significant_weight, ...);
}
```

Pass 1 uses `oversampling=0` (base grid). Pass 2 uses `adaptive_oversampling`
(e.g., 1 = 2× finer). `exp_Mcoarse_significant` is a boolean mask — only
`True` entries get the fine pass.

**Implementation**:
1. Pass 1: run full E-step on coarse grid (all rotations, base resolution).
   Compute weights. Find threshold such that top 99.9% of total weight is kept.
   Mark significant (rotation, translation) pairs per image.
2. Pass 2: for each image, only evaluate the significant orientations at
   oversampled resolution (2× finer angles, 2× finer translations).
   This is a per-image sparse evaluation — very different from the dense GEMM.
3. Combine weights from both passes for M-step accumulation.

**The per-image sparse pass 2 is where FFT cross-correlation shines**: each
image has only ~100-500 candidate rotations, too few for efficient GEMM batching.
The FFT approach (one iFFT per candidate rotation) is natural here.

**Test**: Compare significant weight masks and final posterior weights against
RELION at each iteration.

**Expected**: Pass 1 is the same cost as current dense grid. Pass 2 is
~100×–500× cheaper (only ~100–500 orientations/image instead of 36,864).
Total E-step: roughly 2× current cost for pass 1 + negligible pass 2.
But at oversampled resolution, so much better accuracy.

---

## Step 6: Maximum significants cap and early termination

**What**: Cap the number of significant orientations per image to
`maximum_significants` (default: 100–500 per class). Skip images that have
already converged (small angular change).

**Why**: Prevents pathological cases where many orientations have similar
weights (e.g., symmetric particles, low SNR early iterations).

**Implementation**:
- After pass 1, sort weights per image, keep top `maximum_significants`
- Track per-image angular change between iterations
- Optionally skip images with change < threshold (RELION doesn't do this,
  but cryoSPARC does)

**Test**: Compare `rlnMaxNumberOfPooledParticles` and per-image weight
distributions against RELION.

---

## Step 7: Integration and full RELION comparison

**What**: Wire all the above into a single `relion_refine_jax()` function
that runs the complete auto_refine loop:
1. Initialize from low-pass filtered reference
2. Split into half-sets
3. Per iteration:
   a. Compute current_size from FSC
   b. Crop images/projections to current_size
   c. E-step with two-pass adaptive oversampling
   d. M-step with half-spectrum, Fourier-cropped backprojection
   e. Wiener solve with FSC-based prior
   f. Estimate noise
   g. Update angular sampling if accuracy improves
4. Continue until convergence (angular change < threshold)

**Test**: Run on EMPIAR-10028 (or our synthetic data) and compare:
- Final resolution (FSC = 0.143 threshold)
- Final volume (FSC between RELION and recovar reconstructions)
- Total wall time
- Convergence trajectory

**Target**: Same resolution as RELION, ≤ RELION wall time on same GPU,
identical convergence behavior.

---

## Summary table

| Step | Feature | RELION equivalent | FLOP reduction | Effort |
|------|---------|-------------------|----------------|--------|
| 0 | Current baseline | — | 1× | DONE |
| 1 | Half-spectrum GEMMs | rfft throughout | 2× | Small |
| 2 | Fourier cropping | `windowFourierTransform` to `current_size` | 10–50× early | Medium |
| 3 | Split half-sets + FSC | Gold-standard FSC | — (enables Step 2) | Small |
| 4 | Noise/signal prior | `rlnSigma2Noise`, `rlnSignalPrior` | — (convergence) | Medium |
| 5 | Two-pass adaptive | `exp_ipass`, `exp_Mcoarse_significant` | 100× in pass 2 | Large |
| 6 | Max significants cap | `maximum_significants` | Safety bound | Small |
| 7 | Full integration | `relion_refine --auto_refine` | All combined | Medium |

Steps 1–2 are the highest-impact, lowest-effort optimizations.
Step 5 is the most complex but provides the final large speedup.
Steps 3–4 are prerequisites for Steps 2 and 5.

## Existing code to reuse

Most of the infrastructure already exists:

| Need | Existing code | Location |
|------|--------------|----------|
| Half-spectrum conversion | `full_image_to_half_image`, `half_image_to_full_image` | `core/fourier_transform_utils.py` |
| Half-image forward/adjoint slice | `slice_volume(..., half_image=True)`, `adjoint_slice_volume(..., half_image=True)` | `core/slicing.py` (CUDA) |
| Half-volume support | `half_volume_to_full_volume`, etc. | `core/fourier_transform_utils.py` |
| Fourier truncation / downsample | `downsample_vol_by_fourier_truncation` | `utils/helpers.py:627` |
| Image downsampling | `downsample_images` | `data_io/downsample.py:24` |
| Valid frequency mask | `CryoEMDataset.get_valid_frequency_indices(rad)` | `data_io/cryoem_dataset.py:565` |
| FSC computation | `get_fsc_gpu(vol1, vol2, volume_shape)` | `reconstruction/regularization.py:128` |
| Signal prior from FSC | `compute_relion_prior(datasets, noise, vol1, vol2)` | `reconstruction/regularization.py:75` |
| Noise estimation | `estimate_noise_level_no_masks(dataset, indices, mean, ...)` | `reconstruction/noise.py:716` |
| Radial noise model | `make_radial_noise(PS, image_shape)` | `reconstruction/noise.py:1084` |
| RELION Wiener solve | `post_process_from_filter`, `post_process_from_filter_v2` | `reconstruction/relion_functions.py:427,459` |
| Split half-set EM loop | `split_E_M_v2` | `em/iterations.py:57` |
| HEALPix rotation grid | `get_rotation_grid(nside_level)` | `em/sampling.py` |
| Translation grid | `get_translation_grid(max_pixel, pixel_offset)` | `em/sampling.py` |
| Phase shift for translations | `batch_trans_translate_images` | `core/` |

The main NEW code needed is:
- Half-spectrum GEMM wrappers (Step 1: ~50 lines)
- Fourier crop/uncrop for images at current_size (Step 2: ~100 lines)
- Two-pass adaptive E-step with significance mask (Step 5: ~200 lines)
- Orchestrator wiring it all together (Step 7: ~200 lines)

## Implementation order

**Phase A** (compute efficiency): Steps 1 → 2. Pure performance, no algorithmic change.

**Phase B** (RELION algorithm): Steps 3 → 4. Match RELION's statistical model.

**Phase C** (adaptive search): Steps 5 → 6 → 7. Match RELION's search strategy.

Each phase ends with a RELION comparison test on the synthetic dataset.
