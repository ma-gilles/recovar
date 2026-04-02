# RELION-Parity Implementation Plan v2

Supersedes: `plan_relion_parity.md` (v1).

Goal: bring `recovar/em/dense_single_volume/` to exact feature parity
with RELION 5's `relion_refine --auto_refine` for single-class,
single-volume homogeneous refinement, matching RELION's output
(resolution, convergence trajectory, per-image posteriors) while being
at least as fast on the same GPU hardware.

**Out of scope**: K-class classification, heterogeneity/PPCA, helical
symmetry, tilt series, CTF refinement, Bayesian polishing, multi-body
refinement.

**Single optics group assumption**: throughout this plan we assume a
single optics group.  RELION keeps per-optics-group image sizes and
noise spectra; we do not.  This limitation must be documented in the
code but is acceptable for the target use cases.

---

## 1. Current State Assessment

### Legend

- **DONE**: Implemented and verified against RELION output.
- **PARTIAL**: Implemented but differs from RELION in specific ways.
- **MISSING**: Not implemented at all.

### Feature Matrix

| # | RELION Feature | Status | Details |
|---|---|---|---|
| 1 | Half-spectrum GEMMs (efficiency, not a RELION feature per se) | **DONE** | `engine_v2.py` uses `N_half = H*(W//2+1)` for all E-step and M-step GEMMs. Half-image weights correct (DC=1, Nyquist=1, interior=2). |
| 2 | Fourier windowing (current_size) | **DONE** | `fourier_window.py` implements coordinate-preserving gather/scatter. `engine_v2.py:run_em_v2()` accepts `current_size` and restricts GEMMs to `n_windowed` frequencies. Allowed sizes: `[16, 24, 32, 48, 64, 96, 128]`. |
| 3 | FSC-driven current_size loop | **PARTIAL** | `refine.py:refine_single_volume()` computes FSC between half-maps and derives current_size via `fsc_to_current_size()`. However: uses FSC < 0.143 threshold, NOT RELION's `data_vs_prior < 1` criterion. Growth logic is simple quantization, not RELION's +25%/+10 shells rule. No `ave_Pmax` check. |
| 4 | Two-pass adaptive oversampling | **PARTIAL** | `adaptive.py` + `refine.py` implement pass 1 (coarse, dense) -> significance pruning -> pass 2 (oversampled children). However: pass 1 and pass 2 use the SAME current_size (no separate coarse/fine Fourier window). RELION uses a smaller `image_coarse_size` for pass 1 and full `image_current_size` for pass 2. The `image_coarse_size` formula from particle diameter is missing. |
| 5 | Significance pruning | **DONE** | `adaptive.py:find_significant_mask()` counts rot x trans SAMPLES (not just rotations), matching RELION's `_rlnNrOfSignificantSamples`. Uses `adaptive_fraction=0.999` and `max_significants` cap. |
| 6 | Tau2 (signal prior) from FSC | **PARTIAL** | `regularization.py:compute_fsc_prior_gpu()` computes `SNR = FSC / (1-FSC)` then `prior = SNR / bottom_avg`. But: (a) no smoothing of tau2 across shells, (b) the `bottom_of_fraction` denominator uses nearest-neighbor gridding counts (`batch_get_nearest_gridpoint_indices`) rather than RELION's `sigma2_reconstruction` from the actual backprojection weights, (c) no `tau2_fudge` factor, (d) no floor on tau2 for stability. The iterative prior (`prior_iteration_relion_style`) is more sophisticated but only used in the heterogeneity path. |
| 7 | Wiener reconstruction with tau2 | **DONE** | `relion_functions.py:adjust_regularization_relion_style()` adds `1/(padding_factor^3 * tau2)` to the Fourier weight denominator, matching RELION's `Fweight += invtau2`. Floor at `radial_avg/1000`. Gridding correction (sinc^2 for trilinear) is correct. |
| 8 | Noise estimation | **PARTIAL** | `noise.py:estimate_noise_level_no_masks()` uses HARD assignments (best rotation/translation only), operates on `experiment_datasets[0]` only (one half-set), uses `min(1000, n_units)` images, and subtracts projected mean. RELION uses POSTERIOR-WEIGHTED residuals from ALL images in BOTH half-sets using the MASKED image. |
| 9 | `data_vs_prior` resolution criterion | **MISSING** | RELION determines current resolution from the shell where `data_vs_prior_class[ires] < 1.0`, not from FSC < 0.143. `data_vs_prior = sum(CTF^2 * weight) / (1/tau2)` for each shell. We use FSC < 1/7 exclusively. |
| 10 | current_size growth logic | **MISSING** | RELION grows current_size by +25% of `ori_size/2` when `ave_Pmax > 0.1` and FSC is high at the limit, else by `+incr_size` (default 10 shells). We jump directly to the FSC-determined size with simple quantization. |
| 11 | Masked alignment vs unmasked reconstruction | **MISSING** | RELION applies a soft circular mask (particle_diameter/2, cosine edge 2 px) to images before FFT for the E-step (alignment), but uses UNMASKED images for the M-step (backprojection). We use the same image for both: `_preprocess_batch()` calls `config.process_fn(batch, apply_image_mask=False)` with no mask, then uses those images for both cross-term (E-step) and weighted sum (M-step). |
| 12 | Image normalization | **MISSING** | RELION normalizes each image: subtract mean, divide by standard deviation, both computed within the circular mask. We apply `process_images()` which does background subtraction but no per-image intensity normalization within a mask. |
| 13 | Per-group scale correction | **MISSING** | RELION estimates a per-group scale factor `scale_correction[group]` from the ratio `<X*A>/<A*A>` (least-squares fit of image to CTF-modulated reference) at shells where `data_vs_prior > 3`. The scale multiplies the reference projection before comparison: `Frefctf = CTF * scale * Fref`. We have no group scale concept. |
| 14 | `min_diff2` trick | **DONE (implicit)** | Our score computation is `scores = -0.5 * residuals` and we use streaming logsumexp (`_update_logsumexp`) which naturally subtracts the running maximum before exponentiation, achieving the same numerical stability as RELION's `exp(-(diff2 - min_diff2))`. |
| 15 | `exp_highres_Xi2` (high-res contribution beyond current_size) | **MISSING** | RELION computes `exp_highres_Xi2_img = sum_{k > current_size} |F_img(k)|^2 / sigma2_noise(k)` and adds `highres_Xi2 / 2` to every diff2. This ensures the total log-likelihood includes all frequencies, even those not used for alignment. We truncate at current_size and ignore the rest. |
| 16 | Convergence detection | **MISSING** | RELION tracks: (a) iterations without resolution gain, (b) angular accuracy `acc_rot` / `acc_trans`, (c) changes in hidden variables, (d) `has_fine_enough_angular_sampling`. Convergence triggers joining half-sets and a final Nyquist-resolution iteration. We run a fixed number of iterations. |
| 17 | Angular step refinement | **MISSING** | RELION increments HEALPix order when resolution stalls and assignments are stable, subject to `acc_rot` check (don't refine beyond 75% of estimated accuracy). We use a fixed HEALPix order throughout. |
| 18 | Local angular search | **MISSING** | RELION switches from global exhaustive to local Gaussian-prior search when HEALPix order >= 4 (~3.7 deg). Prior: `pdf_direction = exp(-angular_distance^2 / (2 * sigma_rot^2))` with `sigma_rot = 2 * 2 * step^2`. Sigma cutoff = 3 (only search within 3-sigma). We always do global exhaustive search. |
| 19 | Translation oversampling in pass 2 | **MISSING** | RELION oversamples translations in pass 2: each coarse translation spawns 4 sub-translations (2x2 within +/-0.5 step). We use the same translation grid for both passes. |
| 20 | Coarse image size from angular step + particle diameter | **MISSING** | RELION computes `image_coarse_size = 2 * ceil(pixel_size * ori_size / (angular_sampling * pi * particle_diameter / (360 * 1.2)))`. We use the same current_size for coarse and fine passes. |
| 21 | Weight formula: `pdf_orientation * pdf_offset * exp(-(diff2 - min_diff2))` | **PARTIAL** | We compute `weight = exp(score - log_Z)` with flat priors on orientation and translation. RELION multiplies by `pdf_orientation` (Gaussian in local mode, flat/estimated in global mode) and `pdf_offset` (Gaussian with estimated `sigma2_offset`). We have no orientation or translation priors. |
| 22 | `sigma2_offset` estimation and translation prior | **MISSING** | RELION estimates `sigma2_offset` from the data and uses it as a Gaussian prior on translations. |
| 23 | Solvent flattening / reference masking | **MISSING** | RELION applies a user-provided solvent mask to the reference volume after reconstruction (except at the final converged iteration). We do not apply any solvent mask to the reference between iterations. |
| 24 | Half-map gold-standard FSC-to-tau2 | **PARTIAL** | We compute FSC between half-maps and convert to tau2. But the conversion formula differs: we use `tau2 = SNR / bottom_avg` where `bottom_avg` is shell-averaged nearest-neighbor CTF counts. RELION uses `tau2 = (FSC/(1-FSC)) * tau2_fudge * sigma2_noise` where `sigma2_noise` is the reconstruction noise from the backprojection weight array. |
| 25 | Point-group symmetry enforcement | **MISSING** | RELION calls `symmetriseReconstructions()` after the E-step to enforce Hermitian + point-group symmetry on the backprojected Fourier volumes. We do not enforce symmetry. |

---

## 2. Implementation Phases

### Phase C1: Tau2 Estimation -- Match RELION's Formula

**Priority**: CRITICAL (affects every subsequent comparison).

#### What RELION does

In `ml_optimiser.cpp`, function `updateSSNRarrays()` (line 1041), when
`update_tau2_with_fsc` is true (gold-standard auto-refine):

```
for each shell ires:
    myfsc = clamp(FSC_halves[ires], 0.001, 0.999)
    SSNR  = myfsc / (1 - myfsc)
    SSNR *= tau2_fudge                          // default 1.0, user can set 2-4
    tau2[ires] = SSNR * sigma2[ires]            // sigma2 = reconstruction noise
```

Where `sigma2[ires]` (called `sigma2_reconstruction` or the denominator
of the Wiener filter) is derived from the actual Fourier weight array
of the backprojection.  Specifically, for each shell, RELION computes
the average of `1 / Fweight(k)` over all voxels in that shell (this is
the noise variance of the reconstruction at that shell given the CTF
coverage).

The `data_vs_prior` ratio is then:
```
data_vs_prior[ires] = sum_voxels_in_shell(Fweight(k)) / (N_voxels_in_shell / tau2[ires])
```
This ratio determines the resolution limit (where `data_vs_prior < 1`).

#### Current state in recovar

`regularization.py:compute_fsc_prior_gpu()` (line 238):
```python
SNR = fsc / (1 - fsc)
bottom_avg = average_over_shells(bottom_of_fraction.real, volume_shape)
prior_avg = SNR / bottom_avg
```

`bottom_of_fraction` is computed by `compute_prior_quantites()` (line 44)
which accumulates `CTF^2 / noise_variance` projected onto volume voxels
using nearest-neighbor gridding (`batch_get_nearest_gridpoint_indices`).
This is an approximation of the Fourier weight denominator, not the
actual backprojection weights from the reconstruction.

There is also a more sophisticated `compute_fsc_prior_gpu_v2()` (line 298)
that takes the actual `lhs` (Ft_ctf from backprojection) and iterates
with `prior_iteration_relion_style()`, but this is only used in the
heterogeneity path, not in `refine.py`.

#### Changes needed

1. In `refine.py:refine_single_volume()`, switch from `compute_relion_prior()`
   to using `compute_fsc_prior_gpu_v2()` or `prior_iteration_relion_style()`,
   passing the actual `Ft_ctf` from backprojection as the weight denominator.
   The code already has `Ft_ctf_0` and `Ft_ctf_1` available after the E+M
   step.

2. Add a `tau2_fudge` parameter (default 1.0) that multiplies the SNR
   before computing tau2.

3. The FSC clamping should match RELION: `clamp(fsc, 0.001, 0.999)`.
   Currently: `clamp(fsc, FSC_ZERO_THRESHOLD, 1 - FSC_ZERO_THRESHOLD)`.
   Check that `FSC_ZERO_THRESHOLD` matches 0.001.

4. Verify the FSC computation itself.  `get_fsc_gpu()` uses
   `average_over_shells` which indexes up to `volume_shape[0]//2 - 1`.
   RELION's FSC goes to `ori_size/2`.  Document any Nyquist-edge
   differences.

#### Verification

- Run both codes on the same synthetic dataset for one iteration.
- Compare per-shell `tau2` (RELION's `rlnReferenceSigma2` = tau2) at
  each shell.  Extract from RELION's `_model.star` file.
- Tolerance: < 5% relative error at all shells with significant signal
  (FSC > 0.1).

#### Files to modify

- `recovar/em/dense_single_volume/refine.py` (lines 558-562)
- `recovar/reconstruction/regularization.py` (add tau2_fudge parameter,
  potentially route to `compute_fsc_prior_gpu_v2`)
- New: comparison test `tests/integration/test_tau2_vs_relion.py`

---

### Phase C2: Wiener Reconstruction with Tau2 -- Audit Solver

**Priority**: HIGH (reconstruction quality depends on correct regularization).

#### What RELION does

In `backprojector.cpp:reconstruct()` (line 1465):

```
for each voxel (k, i, j) in 3D Fourier grid:
    ires = round(sqrt(k^2 + i^2 + j^2) / padding_factor)
    invtau2 = 1 / (oversampling_correction * tau2_fudge * tau2[ires])
    if ires >= minres_map:
        Fweight[k,i,j] += invtau2
// Then:
Fvol[k] = Fdata[k] / Fweight[k]
```

Where `oversampling_correction = padding_factor^3` for 3D.  The minimum
weight floor is `radial_avg(Fweight) / 1000`.

#### Current state in recovar

`relion_functions.py:adjust_regularization_relion_style()` (line 352):
```python
oversampling_factor = padding_factor**3
inv_tau = 1 / (oversampling_factor * safe_tau)
regularized_filter = filter_flat + inv_tau
avged_reg = average_over_shells(regularized_filter) / 1000
regularized_filter = max(regularized_filter, avged_reg_volume)
```

This matches RELION's formula.  The `tau` parameter passed to this
function comes from `compute_relion_prior()`, which is the quantity
audited in Phase C1.  The `tau2_fudge` factor is currently missing
from the inverse-tau computation (it should multiply `tau` before
taking the reciprocal).

#### Changes needed

1. Add `tau2_fudge` parameter to `adjust_regularization_relion_style()`
   and `post_process_from_filter_v2()`.  Apply it as:
   ```python
   inv_tau = 1 / (oversampling_factor * tau2_fudge * safe_tau)
   ```

2. Verify that `upscale_tau()` correctly maps radial tau2 to the 3D
   grid with `padding_factor`.  RELION uses
   `ires = round(sqrt(k^2+i^2+j^2) / padding_factor)` to index into
   the tau2 array.  Our code uses the same formula via
   `jnp.round(jnp.linalg.norm(pixels, axis=-1) / padding_factor)`.
   Verify edge behavior at `ires = 0` and `ires = max`.

3. The `minres_map` parameter: RELION only adds regularization for
   `ires >= minres_map` (default 0, so all shells).  We add it
   everywhere.  This should already match for default settings.

#### Verification

- Given identical `Ft_y`, `Ft_ctf`, and `tau2` arrays, compare the
  reconstructed volume voxel-by-voxel.  Should match to machine
  precision (< 1e-5 relative error).
- Test with `tau2_fudge = 1`, `2`, and `4`.

#### Files to modify

- `recovar/reconstruction/relion_functions.py`
  (lines 352-406: `adjust_regularization_relion_style`)
- `recovar/reconstruction/relion_functions.py`
  (lines 458-539: `post_process_from_filter_v2`)

---

### Phase C3: Noise Estimation -- Posterior-Weighted

**Priority**: HIGH (noise estimate feeds into tau2 and E-step weights).

#### What RELION does

In `storeWeightedSums()` (line 8626-8651):

```
for each significant (rotation, translation, oversampling):
    for each pixel n in Fourier image:
        diff_real = Frefctf[n].real - Fimg_shift[n].real
        diff_imag = Frefctf[n].imag - Fimg_shift[n].imag
        wdiff2 = weight * (diff_real^2 + diff_imag^2)
        wsum_sigma2_noise[ires] += wdiff2
```

Then in `maximizationOtherParameters()` (line 5246):
```
sigma2_noise[ires] = wsum_sigma2_noise[ires] / (2 * sumw_group * Npix_per_shell[ires])
```

Key properties:
- **Posterior-weighted**: every significant orientation contributes
  proportional to its weight.
- **Uses MASKED images** for the residual computation.
- **All images in both half-sets** contribute.
- **Per-optics-group** (we assume single group).
- The factor of 2 accounts for real+imaginary parts of complex Gaussian.

#### Current state in recovar

`noise.py:estimate_noise_level_no_masks()` (line 716):
```python
# For each image in image_subset (min(1000, n_units)):
#   translate image by stored best translation
#   subtract projected mean at stored best rotation
#   accumulate |residual|^2 averaged over shells
# Then divide by n_images (or CTF^2 for premultiplied)
```

Differences:
- **Hard assignment only**: uses only the best rotation/translation.
- **Subset of images**: `min(1000, n_units)` from half-set 0 only.
- **No masking**: processes unmasked images.
- **No posterior weighting**: each image contributes equally (weight=1
  for best pose, weight=0 for all others).

#### Changes needed

1. Create a new function `estimate_noise_posterior_weighted()` that:
   - Accepts `Ft_y`, `Ft_ctf`, and the current mean (or reference
     projections).
   - For each image batch, computes the posterior-weighted squared
     residual across all significant orientations.
   - Accumulates `sum_k weight * |CTF*ref_k - img_k|^2` per shell.
   - Divides by `2 * sum_weight * N_pixels_per_shell`.

2. This is computationally expensive (must re-evaluate residuals at
   all significant orientations).  Two implementation strategies:

   **Strategy A (accurate)**: During the M-step pass of `run_em_v2()`,
   simultaneously accumulate `wsum_sigma2_noise[ires]`.  This requires
   adding a noise accumulator to `_m_step_block()` /
   `_m_step_block_windowed()`.  The additional cost per block is one
   elementwise subtract + elementwise multiply + shell-reduction --
   negligible compared to the GEMM.

   **Strategy B (approximate)**: Keep the current hard-assignment noise
   estimation but use ALL images (not just 1000) and BOTH half-sets.
   This is simpler but still not posterior-weighted.

   **Recommendation**: Implement Strategy A.  It integrates into the
   existing M-step loop with minimal overhead.

3. Integrate the new noise estimator into `refine.py`.

#### Verification

- Compare per-shell `sigma2_noise` against RELION's
  `rlnSigma2Noise` from the `_model.star` file.
- Tolerance: < 10% relative error at shells with significant signal.
  The posterior-weighted vs hard-assignment difference should be small
  when the posterior is peaked.

#### Files to modify

- `recovar/em/dense_single_volume/engine_v2.py`
  (modify `_m_step_block`, `_m_step_block_windowed`, `run_em_v2`)
- `recovar/em/dense_single_volume/refine.py`
  (replace noise estimation call)
- `recovar/reconstruction/noise.py` (new function or route)
- New: `tests/integration/test_noise_vs_relion.py`

---

### Phase C4: `data_vs_prior` Resolution Criterion

**Priority**: HIGH (determines current_size trajectory, which controls
computational cost and convergence speed).

#### What RELION does

In `updateCurrentResolution()` (line 5579):
```
for ires = 1 to ori_size/2:
    if data_vs_prior_class[iclass][ires] < 1.0:
        break
maxres = ires - 1
```

`data_vs_prior` is the ratio of accumulated evidence (CTF-weighted
data) to prior strength (1/tau2).  When `data_vs_prior < 1`, the
prior dominates at that shell, meaning there is insufficient data to
determine the signal.

This is NOT the same as FSC < 0.143.  The `data_vs_prior` criterion
is more conservative at early iterations (restricts current_size more
aggressively) and less conservative at late iterations (allows higher
resolution once tau2 becomes well-estimated).

#### Current state in recovar

`refine.py:fsc_to_current_size()` (line 266):
```python
pixel_res = find_fsc_resol(fsc, threshold=1/7)
raw_size = 2 * pixel_res
```

Uses FSC < 0.143 exclusively.  No concept of `data_vs_prior`.

#### Changes needed

1. Compute `data_vs_prior` per shell from `Ft_ctf` (accumulated CTF^2
   weights) and `tau2`:
   ```python
   def compute_data_vs_prior(Ft_ctf, tau2, volume_shape, padding_factor=1):
       # Shell-average the Fourier weights
       avg_weight = average_over_shells(Ft_ctf, volume_shape)
       # data_vs_prior = avg_weight * tau2  (both are per-shell radial arrays)
       # (the actual formula: data_vs_prior = sum(Fweight) / (N_voxels / tau2)
       # = avg(Fweight) * tau2 * oversampling_correction)
       oversampling_correction = padding_factor ** 3
       data_vs_prior = avg_weight * tau2 * oversampling_correction
       return data_vs_prior
   ```

2. Find resolution from `data_vs_prior`:
   ```python
   def resolution_from_data_vs_prior(data_vs_prior):
       for ires in range(1, len(data_vs_prior)):
           if data_vs_prior[ires] < 1.0:
               return ires - 1
       return len(data_vs_prior) - 1
   ```

3. Replace `fsc_to_current_size()` in the refinement loop with the
   `data_vs_prior` criterion.  Keep FSC < 0.143 as a fallback / for
   reporting.

4. Additionally, RELION checks from the high-resolution side for cases
   where FSC dips and rises again (tight-mask artifact):
   ```
   for ires = ori_size/2 to 1 (reverse):
       if fsc[ires] > 0.5 and ires > maxres:
           maxres = ires
   ```
   Implement this check when a user mask is applied.

#### Verification

- Compare current_size trajectory vs RELION.  Should match within +/-1
  shell at each iteration.
- Log both FSC-based and `data_vs_prior`-based resolution at each
  iteration.

#### Files to modify

- `recovar/em/dense_single_volume/refine.py`
  (replace `fsc_to_current_size` logic)
- `recovar/reconstruction/regularization.py`
  (add `compute_data_vs_prior` function)
- New: `tests/unit/test_data_vs_prior.py`

---

### Phase C5: current_size Growth Logic

**Priority**: MEDIUM (affects convergence speed but not correctness).

#### What RELION does

In `updateImageSizeAndResolutionPointers()` (line 5684):
```
maxres = pixelFromResolution(current_resolution)
if ave_Pmax > 0.1 AND has_high_fsc_at_limit:
    maxres += round(0.25 * ori_size / 2)   // +25% of Nyquist
else:
    maxres += incr_size                      // default: +10 shells
current_size = min(2 * maxres, ori_size)
```

This means RELION ALWAYS adds extra shells beyond the current
resolution limit.  The +25% jump happens when the posterior is peaked
(`ave_Pmax > 0.1`) and there is significant signal at the resolution
limit (`has_high_fsc_at_limit`).  Otherwise, a conservative +10
shells is added.

`ave_Pmax` is the average over all images of the maximum posterior
probability (the weight of the best-matching orientation for each
image).  When images are well-aligned, this is high (approaching 1).

#### Current state in recovar

We compute `raw_size = 2 * pixel_res` from FSC and quantize to the
nearest allowed size.  No extra shells beyond the resolution limit.
No `ave_Pmax` tracking.

#### Changes needed

1. Track `ave_Pmax`:  In `engine_v2.py:run_em_v2()`, after computing
   `log_Z` and `best_score`, compute:
   ```python
   Pmax_per_image = jnp.exp(best_score - log_Z)
   ave_Pmax = jnp.mean(Pmax_per_image)
   ```
   Return `ave_Pmax` from `run_em_v2()`.

2. Implement the growth logic:
   ```python
   def grow_current_size(maxres_shell, ori_size, ave_Pmax, fsc_at_limit, incr_size=10):
       if ave_Pmax > 0.1 and fsc_at_limit > 0.5:
           maxres_shell += round(0.25 * ori_size / 2)
       else:
           maxres_shell += incr_size
       return min(2 * maxres_shell, ori_size)
   ```

3. Wire this into `refine.py`'s current_size determination.

#### Verification

- Compare current_size trajectory vs RELION.  The growth pattern
  should match: slow (+10) at early iterations, faster (+25%) once
  alignment stabilizes.

#### Files to modify

- `recovar/em/dense_single_volume/engine_v2.py`
  (return `ave_Pmax`)
- `recovar/em/dense_single_volume/refine.py`
  (implement growth logic)

---

### Phase C6: Masked Alignment vs Unmasked Reconstruction

**Priority**: HIGH (affects alignment quality and reconstruction artifacts).

#### What RELION does

In `getFourierTransformsAndCtfs()` (line 5840):
1. Read image, apply old offsets (shift in real space).
2. Apply soft circular mask (particle_diameter/2, cosine edge 2 px).
3. FFT of masked image -> `exp_Fimg` (used for E-step alignment).
4. FFT of unmasked (background-subtracted) image -> `exp_Fimg_nomask`
   (used for M-step backprojection).

Two separate Fourier images per particle.

#### Current state in recovar

`engine_v2.py:_preprocess_batch()` (line 79):
```python
processed = config.process_fn(batch, apply_image_mask=False)
ctf_weighted = processed * CTF / noise_variance
shifted = batch_trans_translate_images(ctf_weighted, ...)
```

There is only ONE processed image per particle, used for both E-step
and M-step.  `apply_image_mask=False` means no soft mask is applied.

#### Changes needed

1. Add a `particle_diameter` parameter to the refinement loop.

2. In `_preprocess_batch()`, produce TWO outputs:
   ```python
   # For alignment (E-step): apply soft circular mask before FFT
   processed_masked = config.process_fn(batch, apply_image_mask=True)
   # OR: apply mask in Fourier space as a convolution (expensive)
   # Better: apply real-space mask before FFT
   masked_real = apply_soft_circular_mask(batch_real, diameter, cosine_width=2)
   masked_ft = rfft2(masked_real)

   # For reconstruction (M-step): no mask (or just background subtraction)
   processed_nomask = config.process_fn(batch, apply_image_mask=False)
   ```

3. Use `shifted_half_masked` for E-step GEMMs and
   `shifted_half_nomask` for M-step GEMMs.

4. This approximately doubles the preprocessing cost but does not
   affect GEMM size (same images, same projections).

5. The soft circular mask function:
   ```python
   def soft_circular_mask(image_shape, radius_px, cosine_width=2):
       # radius_px = particle_diameter / (2 * pixel_size)
       # Cosine-edge mask transitioning from 1 to 0 over cosine_width pixels
       y, x = np.mgrid[...centered coordinates...]
       r = sqrt(x^2 + y^2)
       mask = np.where(r < radius_px - cosine_width,
                       1.0,
                       np.where(r < radius_px,
                                0.5 + 0.5 * cos(pi * (r - radius_px + cosine_width) / cosine_width),
                                0.0))
       return mask
   ```

#### Verification

- Run a reconstruction with and without masked alignment.
- Compare FSC to RELION's reconstruction.  The masked-alignment
  version should match more closely.
- Check for mask artifacts: the unmasked reconstruction should not
  show sharp edges at the mask boundary.

#### Files to modify

- `recovar/em/dense_single_volume/engine_v2.py`
  (`_preprocess_batch` and `run_em_v2`)
- `recovar/em/dense_single_volume/refine.py`
  (pass `particle_diameter`)
- `recovar/core/mask.py` or new utility (soft circular mask function)

---

### Phase C7: Image Normalization

**Priority**: MEDIUM (affects scale of likelihood, interacts with noise
estimation and scale correction).

#### What RELION does

Before computing the FFT, RELION normalizes each image:
1. Compute mean and variance within the circular mask.
2. Subtract mean.
3. Divide by standard deviation.

This ensures all images have approximately unit variance, which is
important for the Gaussian likelihood model.  The `rlnNormCorrection`
field stores the per-image normalization factor.

#### Current state in recovar

`experiment_dataset.process_images()` applies background subtraction
(subtract outer-ring mean) but does NOT normalize variance.  The
overall scale is handled implicitly by the noise variance estimate.

#### Changes needed

1. Add a per-image normalization step:
   ```python
   def normalize_image(image, mask):
       mean_in_mask = sum(image * mask) / sum(mask)
       image -= mean_in_mask
       var_in_mask = sum(image^2 * mask) / sum(mask)
       image /= sqrt(var_in_mask)
       return image, sqrt(var_in_mask)  # store norm factor
   ```

2. Store the normalization factor per image (needed for
   `rlnNormCorrection` output).

3. This is less critical when noise estimation adapts to the actual
   image scale.  Can be deferred if Phase C3 (posterior-weighted noise)
   is already matching RELION well.

#### Verification

- Compare image power spectra before/after normalization.
- Check that noise estimates match RELION more closely with
  normalization enabled.

#### Files to modify

- `recovar/em/dense_single_volume/engine_v2.py`
  (`_preprocess_batch`)
- Dataset preprocessing utilities

---

### Phase C8: Per-Group Scale Correction

**Priority**: LOW for single-group data, HIGH for multi-group data.

#### What RELION does

In `storeWeightedSums()` (line 8641-8650):
```
// Only for shells where data_vs_prior > 3:
sumXA += weight * (Frefctf.real * Fimg_shift.real + Frefctf.imag * Fimg_shift.imag)
sumA2 += weight * (Frefctf.real^2 + Frefctf.imag^2)
```

Then in `maximizationOtherParameters()` (line 5120-5170):
```
scale_correction[group] = wsum_signal_product[group] / wsum_reference_power[group]
```

Clipped to [median/5, 5*median], then renormalized so the
particle-weighted average is 1.0.

The scale correction multiplies the reference before comparison:
`Frefctf = CTF * scale * Fref`.

#### Current state in recovar

No per-group scale correction exists.  All images are treated as
having the same intensity scale.

#### Changes needed

1. During the M-step, accumulate per-group:
   ```python
   sumXA[group] += weight * Re(conj(Frefctf) * Fimg_shift)
   sumA2[group] += weight * |Frefctf|^2
   ```
   Only at shells where `data_vs_prior > 3`.

2. Compute `scale[group] = sumXA[group] / sumA2[group]`.

3. Clip and renormalize.

4. Apply in E-step: multiply reference projections by `scale[group]`
   before computing diff2.

5. For single-group data this reduces to a global scale factor of 1.0.
   Still useful as a sanity check.

#### Verification

- Compare per-group scale factors against RELION's
  `rlnGroupScaleCorrection` in the model STAR file.

#### Files to modify

- `recovar/em/dense_single_volume/engine_v2.py`
  (accumulate scale statistics)
- `recovar/em/dense_single_volume/refine.py`
  (apply scale correction)

---

### Phase C9: Weight Computation Formula Audit

**Priority**: MEDIUM (ensure we match RELION's exact formula).

#### What RELION does

In `convertAllSquaredDifferencesToWeights()` (line 7896):
```
weight = pdf_orientation * pdf_offset * exp(-(diff2 - min_diff2))
```

Where:
- `pdf_orientation`:
  - Global mode (NOPRIOR): `pdf_direction[iclass][idir]` (flat or
    estimated from data).
  - Local mode (PRIOR_ROTTILT_PSI): Gaussian prior centered on
    current best orientation.
- `pdf_offset`: `exp(-(offset - prior)^2 / (2*sigma2_offset)) / norm`
  - Normalized by the mean offset prior across all translations.
- `diff2 - min_diff2`: ensures the best orientation has weight
  `exp(0) = 1`.  Values with `diff2 - min_diff2 > 700` are zeroed.

#### Current state in recovar

`engine_v2.py:_e_step_block_scores()` (line 103):
```python
scores = -0.5 * residuals
```

Then in the logsumexp/normalize:
```python
probs = exp(scores - log_Z)
```

This is equivalent to `exp(-(diff2/2 - min_diff2/2)) / Z` with flat
priors on orientation and translation.  The `-0.5` factor matches
RELION's Gaussian model.  The logsumexp handles the `min_diff2` trick
automatically.

However:
1. No `pdf_orientation` prior (flat in both global and local modes).
2. No `pdf_offset` prior (flat over translation grid).
3. No `sigma2_offset` estimation.
4. No `exp_highres_Xi2` contribution beyond current_size.

#### Changes needed for global mode (Phase C9a)

For global exhaustive search (no local prior), RELION's
`pdf_orientation` is `pdf_direction[iclass][idir]`, initialized to
`1 / n_directions` (flat) and optionally updated from data.  In
practice for auto-refine this is flat, so our flat prior is correct.

The `pdf_offset` is Gaussian with estimated `sigma2_offset`.  During
global search this starts broad and narrows.  To match:

1. Add `sigma2_offset` as a refinement parameter.
2. Compute `pdf_offset` per translation:
   ```python
   pdf_offset = exp(-(tx^2 + ty^2) / (2 * sigma2_offset))
   ```
3. Multiply into the weight before normalization.
4. Update `sigma2_offset` from the posterior-weighted offset variance.

#### Changes needed for `exp_highres_Xi2` (Phase C9b)

1. Before the E-step, for each image compute:
   ```python
   highres_Xi2 = sum_{k > current_size//2} |F_img(k)|^2 / sigma2_noise(k)
   ```
   This is a per-image scalar.

2. Add `highres_Xi2 / 2` to every `diff2` (equivalently, subtract
   `highres_Xi2 / 2` from every score in `_e_step_block_scores`).

3. Since this is a per-image constant, it does not affect the
   argmax or relative weights.  It only affects the absolute likelihood
   and thus `ave_Pmax` and the normalization.  For matching RELION's
   `_rlnLogLikeliContribution`, this must be included.

#### Verification

- Compare per-image `_rlnLogLikeliContribution` against RELION's
  output.  Should match when all prior terms are included.
- Compare `_rlnNrOfSignificantSamples` distribution.

#### Files to modify

- `recovar/em/dense_single_volume/engine_v2.py`
  (`_e_step_block_scores`, `_preprocess_batch`)
- `recovar/em/dense_single_volume/refine.py`
  (sigma2_offset update)

---

### Phase S1: Expand Allowed current_size Set

**Priority**: LOW (convenience / convergence speed).

#### What RELION does

`current_size` can be any even integer up to `ori_size`.  There is no
restriction to specific sizes because RELION's `windowFourierTransform()`
works on any grid.

#### Current state in recovar

`fourier_window.py:ALLOWED_CURRENT_SIZES = [16, 24, 32, 48, 64, 96, 128]`

The comment says non-divisors of 128 are safe because we use
gather/scatter, not `image_shape` changes.  However, the restricted
set means we may overshoot (e.g., RELION wants `current_size=80` but
we jump to 96).

#### Changes needed

1. Add intermediate sizes: 40, 56, 72, 80, 88, 104, 108, 112, 120.
   Any even number <= 128 is valid.

2. Alternatively, allow ANY even number <= `ori_size`:
   ```python
   def quantize_current_size(cs, ori_size):
       cs = max(16, min(int(cs), ori_size))
       if cs % 2 != 0:
           cs += 1
       return cs
   ```

3. The concern about JIT recompilation is valid: each unique
   `n_windowed` value triggers a new JIT compilation.  In practice,
   current_size increases monotonically during refinement, so at most
   ~10 compilations total.  Each compilation takes ~10-15s.  Budget
   ~2 minutes total.

#### Verification

- Run a refinement with finer current_size granularity.
- Check that the current_size trajectory more closely matches RELION.

#### Files to modify

- `recovar/em/dense_single_volume/fourier_window.py`

---

### Phase S2: Differential Coarse/Fine Fourier Window

**Priority**: MEDIUM (significant speedup at early iterations).

#### What RELION does

- **Pass 1 (coarse)**: Uses `image_coarse_size` (smaller).
- **Pass 2 (fine)**: Uses `image_current_size` (larger).

The coarse size is computed from the angular step and particle diameter
(see Phase S3).  The fine size is the full current resolution.

Using a smaller size for pass 1 makes the dense GEMM much cheaper
(pass 1 evaluates ALL rotations), while pass 2 only evaluates the
significant subset at full resolution.

#### Current state in recovar

Both passes use the same `current_size`.  In
`refine.py:_compute_significance_batched()` (the pass 1 code), the
same `cs_for_engine` is used as in the main E+M call.

#### Changes needed

1. In `refine.py:refine_single_volume()`, compute two sizes:
   ```python
   cs_fine = quantize_current_size(resolution_shell)
   cs_coarse = min(cs_fine, coarse_size_from_angular_step(...))
   ```

2. Pass `cs_coarse` to `_compute_significance_batched()` for pass 1.

3. Pass `cs_fine` to `compute_pass2_stats()` for pass 2.

4. Also pass `cs_fine` to the M-step accumulation in pass 2 (the
   M-step uses the unmasked image at full current resolution).

#### Verification

- Verify that pass 1 assignments are not degraded by the smaller
  Fourier window (significant set should still capture the correct
  orientations).
- Wall-clock speedup measurement at early iterations.

#### Files to modify

- `recovar/em/dense_single_volume/refine.py`
  (compute and pass separate sizes)

---

### Phase S3: Coarse Image Size from Angular Step + Particle Diameter

**Priority**: MEDIUM (completes the coarse/fine split from Phase S2).

#### What RELION does

In `updateImageSizeAndResolutionPointers()` (line 5760):
```
rotated_distance = (angular_sampling / 360) * pi * particle_diameter
keepsafe_factor = 1.2   (3D)
coarse_resolution = rotated_distance / keepsafe_factor
image_coarse_size = 2 * ceil(pixel_size * ori_size / coarse_resolution)
```

This computes the minimum image resolution needed to distinguish two
orientations separated by `angular_sampling` degrees for a particle of
the given diameter.

#### Current state in recovar

No such computation exists.  The coarse pass uses the same resolution
as the fine pass.

#### Changes needed

1. Implement the formula:
   ```python
   def compute_coarse_size(angular_step_deg, particle_diameter_A,
                           pixel_size_A, ori_size):
       rotated_distance_A = (angular_step_deg / 360) * np.pi * particle_diameter_A
       coarse_resolution_A = rotated_distance_A / 1.2
       coarse_size = 2 * int(np.ceil(pixel_size_A * ori_size / coarse_resolution_A))
       return min(coarse_size, ori_size)
   ```

2. The angular step depends on the HEALPix order:
   ```python
   angular_step = 360 / (6 * 2**healpix_order)
   ```

3. Require `particle_diameter` and `pixel_size` as inputs to the
   refinement loop (they are already available from the dataset).

#### Verification

- Compare computed `image_coarse_size` against RELION's for a range of
  angular steps and particle diameters.

#### Files to modify

- `recovar/em/dense_single_volume/refine.py` (new function, wire in)

---

### Phase S4: Convergence Detection

**Priority**: HIGH (avoids wasting iterations / knowing when to stop).

#### What RELION does

Convergence when ALL of:
1. `has_fine_enough_angular_sampling` (can't get finer).
2. `nr_iter_wo_resol_gain >= 1` (1 iteration without resolution
   improvement).
3. `nr_iter_wo_large_hidden_variable_changes >= 1` (1 iteration
   without large orientation changes) OR `auto_ignore_angle_changes`.

"Large hidden variable changes" measured by tracking:
- Mean angular change between successive iterations' best orientations.
- `acc_rot`: angular accuracy estimated from the posterior width.
- `acc_trans`: translational accuracy from posterior width.

When converged:
- Join half-sets (`do_join_random_halves = true`).
- Use data to Nyquist (`do_use_all_data = true`).
- Run one final iteration at full resolution.

#### Current state in recovar

Fixed `max_iter` iterations.  No convergence tracking.

#### Changes needed

1. After each iteration, compute:
   ```python
   # Angular accuracy from posterior
   # (estimated from the spread of significant orientations)
   acc_rot = estimate_angular_accuracy(weights, rotations)

   # Mean change in best orientation from previous iteration
   mean_angular_change = mean_rotation_distance(
       best_rots_current, best_rots_previous
   )

   # Resolution gain
   resol_gain = current_pixel_res - previous_pixel_res
   ```

2. Track `nr_iter_wo_resol_gain` and
   `nr_iter_wo_large_hidden_variable_changes`.

3. Implement the convergence check:
   ```python
   has_converged = (
       has_fine_enough_sampling
       and nr_iter_wo_resol_gain >= 1
       and (nr_iter_wo_large_changes >= 1 or auto_ignore_angle_changes)
   )
   ```

4. When converged, run one final iteration:
   - Join half-sets (average volumes).
   - Set current_size = ori_size (use all frequencies).
   - Re-estimate noise and tau2 from the combined dataset.

#### Verification

- Check that convergence is detected at approximately the same
  iteration as RELION.
- The final resolution should match.

#### Files to modify

- `recovar/em/dense_single_volume/refine.py`
  (convergence tracking and final iteration)

---

### Phase S5: Angular Step Refinement

**Priority**: HIGH (enables resolution progress beyond the initial
angular sampling).

#### What RELION does

In `updateAngularSampling()` (line 9687), angular sampling is refined
when:
1. Resolution hasn't improved for 1 iteration OR
   `auto_resolution_based_angles` and resolution-implied step is finer.
2. Hidden variables haven't changed for 1 iteration OR
   `auto_ignore_angle_changes`.
3. Current step is NOT already finer than 75% of `acc_rot`.

When stepping to finer sampling:
- HEALPix order increments by 1 (doubles the number of directions).
- `new_step = 360 / (6 * round(2^(new_order + adaptive_oversampling)))`.
- Translation step: `min(1.5, 0.75 * acc_trans) * 2^adaptive_oversampling`.
- Translation range: `5 * current_changes_optimal_offsets` (capped at
  1.3x previous range).

#### Current state in recovar

Fixed HEALPix order and translation grid throughout refinement.

#### Changes needed

1. Before each iteration, evaluate whether to refine angular sampling:
   ```python
   def should_refine_angular_step(
       nr_iter_wo_resol_gain, nr_iter_wo_large_changes,
       current_step, acc_rot, auto_resolution_based_angles,
       resolution_based_step,
   ):
       condition1 = (nr_iter_wo_resol_gain >= 1 or
                     (auto_resolution_based_angles and
                      resolution_based_step < current_step))
       condition2 = (nr_iter_wo_large_changes >= 1 or
                     auto_ignore_angle_changes)
       condition3 = current_step > 0.75 * acc_rot
       return condition1 and condition2 and condition3
   ```

2. When refining:
   ```python
   new_order = current_order + 1
   new_rotations = get_healpix_grid(new_order)
   new_step = 360 / (6 * round(2**(new_order + adaptive_oversampling)))

   # Update translation grid
   new_trans_step = min(1.5, 0.75 * acc_trans) * 2**adaptive_oversampling
   new_trans_range = min(
       5 * current_changes_optimal_offsets,
       1.3 * current_trans_range,
   )
   new_translations = make_translation_grid(new_trans_step, new_trans_range)
   ```

3. The HEALPix grid generation already exists in `sampling.py`.

#### Verification

- Compare angular step and HEALPix order at each iteration against
  RELION.
- Compare final resolution: finer angular sampling should enable
  higher resolution.

#### Files to modify

- `recovar/em/dense_single_volume/refine.py`
  (angular refinement logic)
- `recovar/em/sampling.py`
  (translation grid generation with variable step/range)

---

### Phase S6: Local Angular Search

**Priority**: MEDIUM-HIGH (critical for final resolution, but only
kicks in at fine angular sampling).

#### What RELION does

When `healpix_order >= autosampling_hporder_local_searches` (default 4,
~3.7 deg before oversampling):

- Switch from global exhaustive to local search.
- Set `sigma2_rot = sigma2_psi = 2 * 2 * new_step^2`.
- For each image, only evaluate orientations within 3-sigma of the
  current best:
  ```
  angular_distance = acos(tr(R_current^T * R_test) - 1) / 2  [approximate]
  if angular_distance < 3 * sigma_rot:
      pdf = exp(-angular_distance^2 / (2 * sigma_rot^2))
  else:
      pdf = 0  (skip this orientation)
  ```

This reduces the search space from O(10^4-10^5) orientations to
O(100-500) per image.

#### Current state in recovar

Global exhaustive search always.  No per-image orientation priors.

#### Changes needed

1. Per-image prior orientations: store the best (rot, tilt, psi) from
   the previous iteration.

2. Before the E-step, for each image, compute which orientations are
   within 3*sigma of the prior:
   ```python
   def select_local_orientations(prior_rot, all_rotations, sigma_rot, sigma_cutoff=3):
       angular_distances = rotation_geodesic_distance(prior_rot, all_rotations)
       mask = angular_distances < sigma_cutoff * sigma_rot
       pdf = jnp.where(mask, jnp.exp(-angular_distances**2 / (2 * sigma_rot**2)), 0)
       return mask, pdf
   ```

3. This creates a PER-IMAGE orientation mask.  For the GEMM-based
   E-step, this is awkward because different images have different
   active orientations.  Two strategies:

   **Strategy A (union-based)**: Take the union of all active
   orientations across the image batch, evaluate densely, then mask.
   The union size depends on how spread out the priors are.

   **Strategy B (per-image sparse)**: Group images by similar prior
   orientations, batch each group with its own local grid.  More
   complex but more efficient for very peaked posteriors.

   **Recommendation**: Start with Strategy A.  If the union is too
   large (> 50% of all orientations), fall back to global search for
   that batch.

4. Multiply the orientation pdf into the weight:
   ```python
   scores += jnp.log(pdf_orientation + 1e-30)
   ```

#### Verification

- Compare per-image `_rlnAngleRot/Tilt/Psi` against RELION at late
  iterations when local search is active.
- Compare the number of orientations evaluated per image.

#### Files to modify

- `recovar/em/dense_single_volume/engine_v2.py`
  (per-image orientation masking)
- `recovar/em/dense_single_volume/refine.py`
  (switch to local search when appropriate)
- `recovar/em/sampling.py`
  (rotation distance computation)

---

### Phase S7: Translation Oversampling in Pass 2

**Priority**: LOW (small effect, only matters at sub-pixel accuracy).

#### What RELION does

In pass 2, each coarse translation spawns `2^oversampling`
sub-translations in each of (x, y):
- For `adaptive_oversampling = 1`: 4 sub-translations per coarse.
- Sub-translations: coarse +/- step/2 in each axis.

#### Current state in recovar

Same translation grid for both passes.  In `adaptive.py:compute_pass2_stats()`,
the `translations` argument is passed unchanged.

#### Changes needed

1. Generate oversampled translation grid for pass 2:
   ```python
   def oversample_translations(coarse_translations, step, oversampling_order=1):
       sub_step = step / 2**oversampling_order
       offsets = jnp.array([[-sub_step/2, -sub_step/2],
                            [-sub_step/2, +sub_step/2],
                            [+sub_step/2, -sub_step/2],
                            [+sub_step/2, +sub_step/2]])
       oversampled = coarse_translations[:, None, :] + offsets[None, :, :]
       return oversampled.reshape(-1, 2)
   ```

2. Use oversampled translations in pass 2 only.

3. The increased translation count (4x) is offset by the reduced
   rotation count (only significant rotations).

#### Verification

- Compare per-image best translations against RELION.  Should be
  closer with oversampling.

#### Files to modify

- `recovar/em/dense_single_volume/adaptive.py`
  (`compute_pass2_stats`)
- `recovar/em/sampling.py`
  (oversampled translation grid generator)

---

## 3. Phase Ordering and Dependencies

```
Phase C1 (tau2)
  |
  v
Phase C2 (Wiener solver)  <-- depends on C1 for correct tau2
  |
  v
Phase C4 (data_vs_prior)   <-- depends on C1 for tau2 and C2 for Ft_ctf
  |
  v
Phase C5 (growth logic)    <-- depends on C4 for correct resolution
  |
  +--- Phase C3 (noise)    <-- can run in parallel with C4/C5
  |
  v
Phase C6 (masked/unmasked) <-- independent, but affects alignment quality
  |
  +--- Phase C7 (normalization) <-- can run in parallel with C6
  |
  +--- Phase C8 (scale)        <-- can run in parallel, low priority
  |
  v
Phase C9 (weight formula)  <-- depends on C6 for masked images
  |
  ====== CORRECTNESS COMPLETE ======
  |
  v
Phase S1 (expand sizes)    <-- trivial, do anytime
  |
  v
Phase S3 (coarse size formula) <-- needed before S2
  |
  v
Phase S2 (coarse/fine split)  <-- depends on S3
  |
  v
Phase S4 (convergence)       <-- independent, do anytime after C phases
  |
  v
Phase S5 (angular refinement) <-- depends on S4 for convergence tracking
  |
  v
Phase S6 (local search)       <-- depends on S5 for angular step control
  |
  v
Phase S7 (translation oversampling) <-- low priority, do last
```

**Recommended implementation order**:
1. C1 -> C2 -> C4 -> C5 (tau2/resolution pipeline)
2. C3 (noise, can overlap with above)
3. C6 (masked alignment)
4. C9b (highres_Xi2, quick addition)
5. S1 (expand sizes, trivial)
6. S3 -> S2 (coarse/fine split)
7. S4 -> S5 (convergence + angular refinement)
8. S6 (local search)
9. C7, C8, C9a, S7 (polish)

---

## 4. Testing Strategy

### Synthetic Test Dataset

Use the existing dataset at
`/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/`
(5000 images, 128px, noise_level=1.0).

Additionally create a second dataset with:
- 1000 images, 128px, noise_level=0.5 (higher SNR, for quick tests)
- Known ground truth volume and orientations.

### RELION Reference Runs

Run RELION 5 on each dataset:
```bash
mpirun -n 3 relion_refine_mpi \
  --i particles.star --ref reference_init.mrc \
  --o relion_ref/run \
  --auto_refine --split_random_halves \
  --particle_diameter 200 \
  --ini_high 30 \
  --healpix_order 2 \
  --offset_range 5 --offset_step 1 \
  --oversampling 1 \
  --pad 2 \
  --gpu 0 \
  --j 4
```

Extract per-iteration:
- `rlnCurrentResolution` (Angstrom) and `rlnCurrentImageSize` (pixels)
- Per-shell `rlnSigma2Noise` (from `_model.star` -> `data_model_class_N`)
- Per-shell `rlnReferenceSigma2` (= tau2)
- Per-shell `rlnDataVsPrior`
- Per-image Euler angles and origins
- Per-image `_rlnNrOfSignificantSamples`
- Per-image `_rlnLogLikeliContribution`
- Reconstructed half-map MRC volumes
- FSC between half-maps

### Per-Phase Verification

| Phase | Metric | Extraction | Tolerance |
|---|---|---|---|
| C1 | Per-shell tau2 | `rlnReferenceSigma2` from `_model.star` | < 5% relative at shells with FSC > 0.1 |
| C2 | Reconstructed volume | Half-map MRC files | FSC > 0.999 at all resolved shells |
| C3 | Per-shell noise | `rlnSigma2Noise` from `_model.star` | < 10% relative at all shells |
| C4 | current_size trajectory | `rlnCurrentImageSize` per iteration | Exact match (+/-2 pixels) |
| C5 | current_size growth pattern | `rlnCurrentImageSize` per iteration | Match growth events |
| C6 | Final resolution | FSC = 0.143 shell | Within 1 shell |
| C9 | Per-image log-likelihood | `_rlnLogLikeliContribution` | < 1% relative (excluding constant terms) |
| S4 | Convergence iteration | Iteration where RELION converges | Within +/-2 iterations |
| S5 | HEALPix order trajectory | Infer from angular step in STAR | Exact match |
| S6 | Final Euler angles | `rlnAngleRot/Tilt/Psi` | < 1 angular step deviation |
| End-to-end | Final resolution | FSC = 0.143 | Within 1 shell |
| End-to-end | Final volume | FSC(ours, RELION) | > 0.99 at all resolved shells |
| End-to-end | Wall-clock time | Timing | <= RELION on same GPU |

### RELION STAR File Parsing

Use `starfile` Python package:
```python
import starfile
data = starfile.read("run_it025_data.star")
model = starfile.read("run_it025_model.star")
# Per-image: data['particles']['rlnAngleRot'], etc.
# Per-shell: model['model_class_1']['rlnSigma2Noise'], etc.
```

### Test Infrastructure

Each phase gets its own test file:
```
tests/integration/
    test_tau2_vs_relion.py          # Phase C1
    test_wiener_vs_relion.py        # Phase C2
    test_noise_vs_relion.py         # Phase C3
    test_data_vs_prior.py           # Phase C4
    test_current_size_trajectory.py # Phase C4+C5
    test_masked_alignment.py        # Phase C6
    test_weight_formula.py          # Phase C9
    test_convergence.py             # Phase S4
    test_angular_refinement.py      # Phase S5
    test_local_search.py            # Phase S6
    test_end_to_end_parity.py       # Final comparison
```

Run with:
```bash
./scripts/run_tests_parallel.sh full    # includes integration tests
```

---

## 5. Performance Targets

### Configuration 1: 128px / 5000 images

| Metric | Current | After Correctness Phases | After Speed Phases | RELION 5 |
|---|---|---|---|---|
| Iteration time (order 3, cs=128) | ~29s | ~30s (C phases add overhead) | ~29s | ~163s |
| Iteration time (order 3, cs=32) | ~3s | ~3s | ~2s (coarse/fine split) | ~5s (early iter) |
| Iteration time (order 4, cs=64, local) | N/A | N/A | ~5s | ~20s |
| Total convergence time | N/A (fixed iter) | ~300s (10 iter) | ~100s (auto-converge) | ~600s |

### Configuration 2: 256px / 50000 images (projected)

| Metric | Current (projected) | After All Phases | RELION 5 |
|---|---|---|---|
| Iteration time (order 3, cs=256) | ~240s | ~240s | ~1500s |
| Iteration time (order 3, cs=64) | ~15s | ~8s (coarse/fine) | ~30s |
| Iteration time (order 5, cs=128, local) | N/A | ~30s | ~120s |
| Total convergence time | N/A | ~30 min | ~4h |

### Key performance notes

- The E-step GEMM dominates at high current_size.  At low current_size,
  preprocessing (phase shifts, CTF weighting) can dominate -- this is
  why Phase S2 (differential coarse/fine) matters.
- The M-step is cheap compared to the E-step (one GEMM per rotation
  block vs the full E-step).
- Local search (Phase S6) is the biggest speedup for late iterations:
  O(100-500) orientations per image instead of O(10000+).
- Memory: the blockwise engine never materializes the full
  (n_images, n_rot, n_trans) tensor, so memory scales with
  `batch_size * block_size * n_trans`, not with the total grid.

### Memory budget (A100 80GB)

| Component | Current | After windowing (cs=32) |
|---|---|---|
| Volume (128^3 complex) | 16 MB | 16 MB |
| Projections (5000 * N_half) | 166 MB | ~11 MB |
| Shifted images (500 * 49 * N_half) | 815 MB | ~52 MB |
| Weights (not materialized) | 0 | 0 |
| Total peak | ~2 GB | ~0.5 GB |

---

## 6. Code Inventory -- What Exists vs What is Needed

### Existing code (ready to use)

| Component | Location | Status |
|---|---|---|
| Half-spectrum GEMM engine | `engine_v2.py` | DONE, working |
| Fourier windowing | `fourier_window.py` | DONE, working |
| FSC computation | `regularization.py:get_fsc_gpu()` | DONE |
| Wiener solver | `relion_functions.py:post_process_from_filter_v2()` | DONE |
| Gridding correction | `relion_functions.py:griddingCorrect_square()` | DONE |
| Significance pruning | `adaptive.py:find_significant_mask()` | DONE |
| HEALPix grids | `sampling.py` | DONE |
| Oversampled rotation grid | `sampling.py:get_oversampled_rotation_grid()` | DONE |
| Two-pass adaptive | `adaptive.py:compute_pass2_stats()` | DONE |
| Iterative prior | `regularization.py:prior_iteration_relion_style()` | DONE (heterogeneity path) |
| Half-volume backprojection | `relion_functions.py:_relion_kernel_batch_half()` | DONE |
| RELION volume convention | `utils/helpers.py:relion_volume_to_recovar()` | DONE |

### New code needed

| Component | Estimated lines | Phase |
|---|---|---|
| `compute_data_vs_prior()` | 20 | C4 |
| `grow_current_size()` | 15 | C5 |
| Posterior-weighted noise accumulator in M-step | 40 | C3 |
| Masked/unmasked image preprocessing | 30 | C6 |
| Soft circular mask function | 15 | C6 |
| `sigma2_offset` estimation | 20 | C9 |
| `highres_Xi2` computation | 15 | C9 |
| `ave_Pmax` tracking | 5 | C5 |
| Convergence tracking | 40 | S4 |
| Angular refinement logic | 30 | S5 |
| Local search orientation selector | 40 | S6 |
| Coarse image size formula | 10 | S3 |
| Translation oversampling | 15 | S7 |
| RELION reference extraction script | 200 | Testing |
| Integration test suite | 500 | Testing |
| **Total** | **~1000** | |

### Existing code to modify

| File | Changes | Phase |
|---|---|---|
| `refine.py` | tau2 source, growth logic, convergence, angular refinement, masked alignment | C1, C4, C5, C6, S4, S5 |
| `engine_v2.py` | noise accumulator, masked preprocessing, highres_Xi2, ave_Pmax, local search | C3, C6, C9, S6 |
| `regularization.py` | tau2_fudge, data_vs_prior | C1, C4 |
| `relion_functions.py` | tau2_fudge in solver | C2 |
| `fourier_window.py` | expand allowed sizes | S1 |
| `adaptive.py` | coarse/fine split, translation oversampling | S2, S7 |

---

## 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Tau2 formula mismatch cascades to all downstream metrics | CRITICAL | Implement C1 first, validate independently before proceeding |
| Masked alignment doubles preprocessing cost | HIGH | Profile to verify GEMMs still dominate; only apply mask at high current_size |
| Local search (S6) requires per-image orientation masks, incompatible with dense GEMM | HIGH | Use union-based approach; fall back to global if union > 50% |
| Posterior-weighted noise (C3) is expensive inside the M-step loop | MEDIUM | The additional computation per block is one elementwise op + reduction, negligible vs GEMM |
| JIT recompilation with many current_sizes | MEDIUM | current_size increases monotonically; at most ~10 compilations total |
| RELION uses iterative gridding correction (Pipe & Menon) while we use direct division | LOW | Our `skip_gridding`-equivalent path with floor matches RELION's when the weight is well-conditioned; verify on test data |
| Point-group symmetry not implemented | LOW | Only affects symmetric structures; document as limitation |

---

## 8. Agent Task Decomposition

Each phase maps to a branch (`claude/em-parity-CX` or
`claude/em-parity-SX`).  Each branch is validated before merge.

### Task 1: C1+C2 (tau2 + solver)
- Branch: `claude/em-parity-c1c2`
- Modify: `regularization.py`, `relion_functions.py`, `refine.py`
- Test: `test_tau2_vs_relion.py`
- Validation: per-shell tau2 matches RELION < 5%

### Task 2: C3 (posterior-weighted noise)
- Branch: `claude/em-parity-c3`
- Modify: `engine_v2.py`, `refine.py`
- Test: `test_noise_vs_relion.py`
- Validation: per-shell noise matches RELION < 10%

### Task 3: C4+C5 (data_vs_prior + growth)
- Branch: `claude/em-parity-c4c5`
- Modify: `regularization.py`, `refine.py`
- Test: `test_current_size_trajectory.py`
- Validation: current_size trajectory matches RELION +/-2

### Task 4: C6 (masked alignment)
- Branch: `claude/em-parity-c6`
- Modify: `engine_v2.py`, `refine.py`
- Test: `test_masked_alignment.py`
- Validation: FSC improvement, reconstruction without mask artifacts

### Task 5: C9b (highres_Xi2)
- Branch: `claude/em-parity-c9b`
- Modify: `engine_v2.py`
- Test: `test_weight_formula.py`
- Validation: log-likelihood matches RELION

### Task 6: S1+S3+S2 (sizes + coarse formula + coarse/fine split)
- Branch: `claude/em-parity-s123`
- Modify: `fourier_window.py`, `refine.py`, `adaptive.py`
- Test: timing benchmarks
- Validation: speedup at early iterations

### Task 7: S4+S5 (convergence + angular refinement)
- Branch: `claude/em-parity-s45`
- Modify: `refine.py`, `sampling.py`
- Test: `test_convergence.py`, `test_angular_refinement.py`
- Validation: convergence iteration matches RELION +/-2

### Task 8: S6 (local search)
- Branch: `claude/em-parity-s6`
- Modify: `engine_v2.py`, `refine.py`, `sampling.py`
- Test: `test_local_search.py`
- Validation: final resolution matches, orientations match

### Task 9: Polish (C7, C8, C9a, S7)
- Branch: `claude/em-parity-polish`
- Lower priority items
- Test: end-to-end comparison

### Task 10: End-to-end validation
- Branch: `claude/em-parity-e2e`
- Run full 25-iteration refinement on both codes.
- Test: `test_end_to_end_parity.py`
- Validation: all metrics in tolerance

---

## 9. Rules for Implementing Agents

1. **NEVER modify `heterogeneity.py`** (separate owners).
2. **NEVER widen test tolerances or modify baselines** without explicit
   approval.
3. **Always run `pixi run test-fast`** before pushing (2454 tests).
4. **Always run `./scripts/run_tests_parallel.sh long-test`** via Slurm
   before any PR.
5. **All GPU work via Slurm** for real jobs; login GPUs for quick
   benchmarks only.
6. **Small targeted diffs** -- no drive-by formatting.
7. **Each phase gets its own branch** off `dev`, validated against
   RELION reference before merging.
8. **Document all RELION-specific constants** (e.g., `tau2_fudge`,
   `adaptive_fraction`, `sigma_cutoff`, `incr_size`) with their
   default values and the RELION source location.
9. When in doubt about RELION's behavior, refer to
   `/scratch/gpfs/GILLES/mg6942/relion5_auto_refine_algorithm.md`
   and the RELION source at `/scratch/gpfs/GILLES/mg6942/relion/`.
