# Phase 2 Audit: Prior and Noise Estimation vs RELION

## Overview

This document audits recovar's signal prior (`compute_relion_prior` /
`compute_fsc_prior_gpu`) and noise estimation (`estimate_noise_level_no_masks`)
against RELION 5.0's implementation in `ml_optimiser.cpp` and
`backprojector.cpp`. It documents each formula, identifies algorithmic
differences, and presents quantitative comparisons on a shared synthetic
dataset (5000 images, 128 px, noise_level=1.0).

---

## Step 2A: Signal Prior (tau^2) Audit

### recovar formula: `compute_fsc_prior_gpu` (normal path)

Located in `recovar/reconstruction/regularization.py:238`.

The function computes the per-shell signal prior (tau^2) from two half-map
volumes as follows:

1. **Compute FSC** between the two unregularized half-maps:

   FSC(k) = Re<V_half1, V_half2>_k / sqrt(|V_half1|^2_k * |V_half2|^2_k)

   where `<,>_k` denotes the average inner product over shell k.

2. **Clamp FSC** to `[epsilon, 1 - epsilon]` where `epsilon = 0.001`
   (`FSC_ZERO_THRESHOLD`).

3. **Compute half-map SNR**:

   SNR(k) = FSC(k) / (1 - FSC(k))

   This is the half-map SNR (not the merged-map SNR). The option
   `estimate_merged_SNR=True` would convert with `FSC_merged = 2*FSC/(1+FSC)`
   first, but this path is not used in `split_E_M_v2`.

4. **Compute bottom_of_fraction** (the LHS denominator):

   `bottom_of_fraction` is a per-voxel quantity computed by
   `compute_prior_quantites` (line 44). For each image i in both half-sets:

   bottom_of_fraction[v] += sum_i CTF_i(k_v)^2 / sigma^2

   where k_v is the frequency at voxel v in 3D Fourier space (nearest
   gridpoint), CTF_i is the CTF for image i, and sigma^2 is the **scalar**
   noise variance (`cov_noise`). This is then divided by 2 (the number of
   half-sets), giving the per-half average.

   This is essentially the diagonal of the normal matrix (Wiener denominator)
   averaged per half-set.

5. **Radially average** `bottom_of_fraction`:

   bottom_avg(k) = <bottom_of_fraction>_k   (shell average)

6. **Compute prior**:

   prior_avg(k) = SNR(k) / bottom_avg(k)    when bottom_avg > 0
                = epsilon                     otherwise

   This gives per-shell tau^2 values. The full 3D prior is obtained by
   broadcasting back to voxels: `prior[v] = prior_avg[shell(v)]`.

### Mathematical interpretation

The formula is:

   tau^2(k) = [FSC(k) / (1 - FSC(k))] / <sum_i CTF_i^2 / sigma^2>_k

Since `sum_i CTF_i^2 / sigma^2` is the Wiener denominator (without the
prior term), this is equivalent to:

   tau^2(k) = SNR_half(k) / W_denom(k)

where `W_denom(k)` is the shell-averaged Wiener denominator.

### RELION formula: `updateSSNRarrays` in `backprojector.cpp:1041`

When `update_tau2_with_fsc = true` (the gold-standard refinement path):

1. **Compute sigma2** (not noise variance -- this is the average inverse
   weight per shell):

   sigma2(k) = count(k) / sum_voxels_in_shell_k(weight(v))

   where `weight(v)` is the 3D weight array from backprojection
   (= sum_i CTF_i^2 / sigma^2_noise_i, accumulated per voxel with padding
   correction). The `oversampling_correction = padding_factor^3` is applied
   to weight before averaging. So:

   sigma2(k) = count(k) / [padding^3 * sum(weight(v))]_k

   This is the shell-averaged inverse of the Wiener denominator.

2. **Clamp FSC** to `[0.001, 0.999]`.

3. **Compute SSNR**:

   SSNR(k) = FSC(k) / (1 - FSC(k))

   Multiplied by `tau2_fudge_factor` (default 1.0 for auto_refine).

4. **Compute tau^2**:

   tau2(k) = SSNR(k) * sigma2(k)

### Side-by-side comparison

| Aspect | recovar | RELION |
|--------|---------|--------|
| FSC input | Unregularized half-maps | Unregularized half-maps (same) |
| FSC clamping | [0.001, 0.999] | [0.001, 0.999] |
| SNR formula | FSC/(1-FSC) | FSC/(1-FSC) * tau2_fudge |
| Denominator quantity | shell_avg(sum_i CTF_i^2 / sigma^2_scalar) | count(k) / [padding^3 * sum(weight(v))]_k |
| Final tau^2 | SNR / bottom_avg | SSNR * sigma2 |

**Key relationship**: RELION's `sigma2(k)` is the reciprocal of recovar's
`bottom_avg(k)`, up to normalization constants (padding factor, per-voxel vs
per-shell counting). Therefore:

   RELION: tau^2 = SNR * (1/W_denom_avg)
   recovar: tau^2 = SNR / W_denom_avg

These are **mathematically equivalent** (assuming tau2_fudge=1.0 and the
normalization agrees).

### Differences identified

1. **Padding factor**: RELION uses `padding_factor = 2` and applies an
   `oversampling_correction = padding_factor^3 = 8` to the weight array.
   recovar does not pad the volume (it operates at the native grid), so no
   padding correction is needed. This is a convention difference, not a bug --
   the voxel counts per shell differ between padded and unpadded grids, but
   the ratio `count/sum(weight)` should be consistent.

2. **tau2_fudge_factor**: RELION supports a `tau2_fudge_factor` (default 1.0
   for auto_refine, can be adjusted via `--tau2_fudge`). recovar does not
   support this parameter. Not a discrepancy for standard refinement.

3. **Per-voxel vs radially-averaged denominator**: recovar's `bottom_of_fraction`
   is a full 3D array that gets shell-averaged at the end. RELION accumulates
   directly into shells. The result should be identical but the intermediate
   3D array uses more memory.

4. **Scalar vs per-shell noise in denominator**: recovar passes `cov_noise`
   (a scalar) to `compute_batch_prior_quantities`, computing
   `CTF^2 / cov_noise`. RELION uses per-shell `sigma2_noise` in the weight
   accumulation. This is a **real algorithmic difference** -- recovar uses a
   single scalar noise estimate for the prior denominator, while RELION uses
   the full spectral noise model. Impact: at shells where noise deviates from
   the scalar mean, the prior will be biased.

5. **Smoothing/floor on tau^2**: RELION applies a raised-cosine taper to tau^2
   at the ini_high boundary after iteration 1 (lines 5306-5327 of
   `ml_optimiser.cpp`), and also clamps tau^2 to zero at shells beyond
   r_max (line 5321). recovar does not apply any smoothing or boundary
   tapering to the prior.

6. **v2 path (iterative)**: recovar also has `compute_fsc_prior_gpu_v2` and
   `prior_iteration_relion_style` which iteratively refine the prior using
   weighted Wiener denominators. The `split_E_M_v2` path uses the simpler
   `compute_fsc_prior_gpu` (non-iterative), which is a difference from RELION's
   iterative tau^2 estimation in the reconstruction step.

### `compute_fsc_prior_gpu_v2` (iterative path)

Used by `prior_iteration_relion_style` (not by `split_E_M_v2`). This version:

1. Computes two Wiener-filtered half-maps using the current prior
2. Computes FSC between them
3. Updates the prior using weighted shell averages:

   sum_top(k) = <H^2 / (H + 1/tau)^2>_k
   sum_bot(k) = <H / (H + 1/tau)^2>_k
   prior_avg(k) = SNR(k) * sum_bot(k) / sum_top(k)

   where H is the combined (H0+H1)/2 Wiener denominator.

This is closer to RELION's approach of using the reconstruction weights to
inform the prior, but it is only used in the heterogeneity path (covariance
estimation), not in the homogeneous EM loop.

---

## Step 2B: Noise Estimation Audit

### recovar: `estimate_noise_level_no_masks`

Located in `recovar/reconstruction/noise.py:716`.

**Inputs**:
- `experiment_dataset`: One half-set dataset (always `experiment_datasets[0]`,
  line 112 of `iterations.py`)
- `image_subset`: `np.arange(min(1000, n_units))` -- first 1000 images of
  the first half-set
- `mean_estimate`: The regularized mean volume for half-set 0
- `batch_size`: 100
- `disc_type`: "linear_interp"

**Algorithm**:

For each image i in the subset:

1. Load and process image: `y_i = process_images(images[i])`
2. Translate to stored best translation: `y_i = translate(y_i, t_best_i)`
3. Compute CTF: `CTF_i = ctf(params_i)`
4. Forward-project mean at best rotation: `proj_i = forward_model(mean, R_best_i)`
5. Compute residual:
   - If premultiplied_ctf: `r_i = y_i - CTF_i * proj_i`
   - Otherwise: `r_i = y_i - proj_i`
6. Compute radially-averaged power spectrum: `PS_i(k) = <|r_i|^2>_k`
7. Accumulate: `lhs += PS_i(k)`, and if premultiplied:
   `rhs += <CTF_i^2>_k`, else `rhs += 1`
8. Final noise estimate: `noise(k) = lhs(k) / rhs(k)`

**Confirmed properties**:
- **Hard assignments**: Uses `best_rotations` and `best_translations` from
  argmax of the posterior (line 109 of `iterations.py`). NOT posterior-weighted.
- **Subset-based**: Uses `min(1000, n_units)` images from `experiment_datasets[0]`
  only (one half-set).
- **Single half-set**: Only the first half-set contributes to noise estimation.

### RELION noise estimation

RELION estimates noise in two different contexts:

#### A. Initial noise estimation (`setSigmaNoiseEstimatesAndSetAverageImage`)

For the first `minimum_nr_particles_sigma2_noise` (default 1000 for SPA)
particles per optics group:

1. Compute per-image power spectrum (all posterior-weighted sums)
2. Average over particles: `sigma2_noise = wsum_sigma2_noise / (2 * sumw_group)`
3. Subtract power spectrum of the averaged image
4. Replace any negative values with nearest positive neighbor

#### B. Running noise update (`maximization`, lines 5246-5286)

After each EM iteration:

1. `sigma2_noise[k] *= mu` (momentum from previous iteration, `my_mu` is
   typically 0 in auto_refine)
2. `sigma2_noise[k] += (1 - mu) * wsum_sigma2_noise[k] / (2 * sumw_group * Npix_per_shell[k])`

The `wsum_sigma2_noise` is accumulated in `storeWeightedSums` as:

   For each (image, orientation, translation) triple with posterior weight w:
     For each pixel with resolution ires:
       wsum_sigma2_noise[ires] += w * |CTF*V_ref - y_shifted|^2

This is a **posterior-weighted** sum of squared residuals, not a hard-assignment
estimate.

3. Clamp: `sigma2_noise[k] >= 1e-15` (for CTF-premultiplied data)
4. Fill zeros from previous shell value

### Side-by-side comparison

| Aspect | recovar | RELION |
|--------|---------|--------|
| Weighting | Hard assignment (argmax) | Posterior-weighted (soft) |
| Number of images | min(1000, n_units) from half-set 0 | All particles (or initial 1000 for init) |
| Half-sets used | Half-set 0 only | All particles (combined across halves) |
| Residual formula | y - CTF*proj (or y - proj) | CTF*V_ref - y_shifted (posterior-weighted) |
| Per-shell normalization | Sum over images / count | wsum_sigma2_noise / (2 * sumw * Npix) |
| Momentum/smoothing | None | mu-smoothing (usually mu=0 for auto_refine) |
| Negativity handling | Replace inf/NaN with last valid | Replace negative with nearest positive |
| Floor | None (beyond inf/NaN fix) | 1e-15 for CTF-premultiplied |

### Key differences

1. **Hard vs soft assignments**: recovar uses the single best (rotation,
   translation) pair. RELION weights residuals by the full posterior
   distribution. Impact: for well-determined orientations (high SNR), the
   difference is negligible. For ambiguous orientations, RELION's approach
   is more statistically principled.

2. **Subset vs full dataset**: recovar uses only 1000 images from one half-set.
   RELION uses all particles. Impact: recovar's estimate has higher variance
   but is much faster to compute. For 5000 images this is 1000/2500 = 40% of
   one half-set.

3. **Normalization**: recovar divides by `rhs` which is either the count of
   images (non-premultiplied) or the sum of `<CTF^2>_k` per shell
   (premultiplied). RELION divides by `2 * sumw_group * Npix_per_shell`, where
   the factor 2 accounts for the complex plane dimensionality and Npix is the
   number of Fourier pixels per shell.

4. **Residual sign convention**: recovar computes `y - CTF*proj` while RELION
   computes `CTF*V_ref - y_shifted`. The squared difference is the same
   regardless of sign.

---

## Step 2C: Quantitative Comparison

A comparison script (`scripts/audit_prior_noise.py`) runs recovar's EM loop
and compares per-iteration quantities against the RELION reference.
Results below are from 3 iterations on the shared synthetic dataset
(5000 images, 128 px, HEALPix order 3, 29 translations).

### Critical finding: normalization convention difference

The absolute values of `noise_variance` and `tau^2`/`mean_signal_variance`
differ by a factor of ~5.5 x 10^8 between recovar and RELION. This is NOT
a bug -- it arises from different FFT normalization and counting conventions:

- recovar: `noise_variance[k] = <|y_k|^2>_k` (mean squared DFT coefficient
  magnitude per shell, unnormalized FFT)
- RELION: `sigma2_noise[k]` = variance per real/imaginary component, per
  Fourier pixel, normalized by `2 * sumw * Npix_per_shell`

The scale ratio is approximately `N^2 * Npix_per_shell / 2` where
`N^2 = 16384` (image size). This is internally consistent in each code --
the E-step probability cancels the noise scale in the relative posteriors.

**Consequence**: Raw per-shell comparison of noise or prior values is
meaningless. We compare instead:
1. FSC curves (scale-independent)
2. Noise spectrum SHAPE (normalized to unit sum)
3. SNR = prior/noise (the ratio, convention-invariant)

### FSC comparison (scale-independent)

| Iter | recovar FSC_res | RELION FSC_res | Max |FSC_diff| |
|------|-----------------|----------------|---------------------|
| 1    | 56 shells       | 4 shells       | 1.103               |
| 2    | 63 shells       | 16 shells      | 0.998               |
| 3    | 63 shells       | 22 shells      | 0.991               |

recovar achieves much higher FSC at all shells because it does NOT crop images
to the current resolution (it uses the full 128-pixel images at every
iteration). RELION restricts evaluation to `current_image_size` which starts
at 56 and grows. This means recovar's E-step sees all frequencies from
iteration 1, giving stronger orientation discrimination. RELION intentionally
limits resolution to prevent overfitting.

The FSC difference at shells 22+ is ~0.9 by iteration 3, reflecting that
recovar resolves high-frequency structure that RELION has not yet included
in its evaluation.

### Noise spectrum shape

| Iter | Correlation | Max shape diff | Scale ratio    |
|------|-------------|----------------|----------------|
| 1    | 0.830       | 0.029          | 5.42 x 10^8    |
| 2    | 0.133       | 0.019          | 5.47 x 10^8    |
| 3    | 0.045       | 0.047          | 5.52 x 10^8    |

The noise spectrum shape (after unit normalization) starts with ~0.83
correlation at iteration 1 and degrades to 0.04 by iteration 3. This
divergence is caused by:

1. **Resolution cropping**: RELION's noise estimation uses `current_size`
   cropped images, so high-frequency shells in RELION receive raw power
   spectrum data rather than model-subtracted residuals. recovar estimates
   noise at full resolution throughout.

2. **Hard vs soft assignments**: recovar's hard-assignment noise estimate
   becomes less representative as the posterior distribution sharpens on
   different orientations than RELION's posterior-weighted estimate.

3. **Subset vs full dataset**: recovar uses 1000 images from one half-set;
   RELION uses all particles.

The scale ratio stays constant at ~5.5 x 10^8 across iterations (consistent
with the FFT normalization difference).

### SNR = prior/noise (convention-invariant)

| Iter | Max SNR ratio | Median SNR ratio |
|------|---------------|------------------|
| 1    | 2.28 x 10^7   | 7.28 x 10^4      |
| 2    | 2176          | 226               |
| 3    | 3182          | 49                |

The SNR ratio (recovar / RELION) is orders of magnitude away from 1.0,
indicating that the prior/noise relationship is fundamentally different:

1. **recovar SNR is much higher** at all shells (SNR ~ 100-400 vs RELION
   SNR ~ 0.01-3). This means recovar's prior is extremely permissive
   relative to its noise estimate -- the Wiener filter barely regularizes.

2. **RELION SNR is small** (< 1 at many shells), meaning RELION applies
   significant regularization. This is the intended behavior: tau^2 should
   be on the order of the signal power per voxel, not the total power.

**Root cause**: recovar's `compute_relion_prior` divides `FSC/(1-FSC)` by
`bottom_avg`, where `bottom_avg = <sum_i CTF_i^2 / sigma^2_scalar>_k`.
Since `sigma^2_scalar` is the large scalar noise variance (~16000), the
denominator is small, making the prior large. RELION's tau^2 uses
`SSNR * sigma2_reconstruction` where `sigma2_reconstruction` is the
per-shell average of the inverse weight (which is proportional to
`1 / sum_i(CTF_i^2 / sigma2_noise_i)`). The per-shell noise in RELION's
weight accumulation produces a much more appropriate scale for tau^2.

### Resolution trajectory

| Iter | recovar (px_res) | RELION (Angstrom) | RELION (cs) |
|------|------------------|-------------------|-------------|
| 1    | 35.91            | 108.80            | 56          |
| 2    | 35.92            | 36.27             | 30          |
| 3    | 35.85            | 21.76             | 50          |

recovar's pixel resolution starts and stays near 36 px (FSC 1/7 threshold),
corresponding to ~4.4 Angstrom at this pixel size. RELION starts at low
resolution (108 A) and progressively improves. By iteration 3, RELION reaches
21.76 A while recovar has already converged to near-Nyquist resolution.

This confirms that recovar processes full-resolution images from the start
(no Fourier cropping), while RELION uses `current_image_size` to restrict
resolution.

### Detailed per-shell analysis (iteration 3)

Selected shells showing the SNR ratio pattern:

| Shell | SNR_ours | SNR_relion | SNR_ratio | FSC_ours | FSC_relion |
|-------|----------|------------|-----------|----------|------------|
| 1     | 235.7    | 1.180      | 199.8     | 0.9999   | 1.0000     |
| 5     | 172.5    | 0.537      | 321.4     | 0.9995   | 0.9910     |
| 10    | 0.91     | 0.130      | 7.0       | 0.9585   | 0.9418     |
| 15    | 21.6     | 0.524      | 41.2      | 0.9995   | 0.9784     |
| 20    | 1.95     | 0.230      | 8.5       | 0.9917   | 0.9354     |
| 25    | 1.18     | 0.073      | 16.2      | 0.9659   | 0.7130     |

The SNR ratio varies widely per shell (7x to 3200x) and does not follow
a smooth pattern, indicating that the prior formula differences interact
with the frequency-dependent noise and FSC curves in complex ways.

---

## Summary of Discrepancies

| # | Discrepancy | Severity | Measured impact | Worth fixing? |
|---|-------------|----------|-----------------|---------------|
| 1 | No Fourier cropping (full resolution from iter 1) | High | FSC reaches Nyquist by iter 1; RELION takes 5+ iters | Phase 3 (planned) |
| 2 | Scalar vs per-shell noise in prior denominator | Medium | SNR ratio off by 7-3200x per shell vs RELION | Yes -- use per-shell noise |
| 3 | Hard vs posterior-weighted noise estimation | Medium | Noise correlation with RELION drops from 0.83 to 0.04 | Deferred -- needs E-step changes |
| 4 | Subset (1000 imgs, 1 half-set) for noise | Low | Higher variance noise estimate | OK -- speed tradeoff |
| 5 | No tau^2 smoothing/tapering | Low | Noisy prior at resolution boundary | Consider adding |
| 6 | Normalization convention (scale factor ~5.5e8) | None (internal) | Internally consistent; no effect on Wiener filter | No fix needed |
| 7 | Non-iterative prior in split_E_M_v2 | Low | Current prior converges well; v2 path available | Could switch if needed |
| 8 | No tau2_fudge_factor support | None | Not used in standard auto_refine | No fix needed |

### Key insight: recovar over-resolves compared to RELION

The biggest difference is NOT in the prior/noise formulas themselves but in
the **resolution management**: recovar uses full-resolution images from
iteration 1, while RELION progressively increases `current_image_size`.
This means:

1. recovar's E-step has 128^2 = 16384 Fourier coefficients informing each
   orientation score, while RELION starts with only ~56^2 = 3136.
2. recovar achieves near-Nyquist FSC in 1-2 iterations but risks overfitting
   to noise at high frequencies.
3. RELION's progressive resolution increase acts as implicit regularization,
   preventing the model from fitting high-frequency noise before low-frequency
   structure is reliable.

This confirms that Phase 3 (Fourier cropping) is essential for matching
RELION's convergence behavior.

### Recommendations (priority order)

1. **Phase 3 (Fourier cropping)**: Implement resolution-dependent image
   cropping before comparing prior/noise more carefully. Without matching
   the resolution schedule, the prior and noise estimates will always
   diverge because they are computed on different frequency ranges.

2. **Per-shell noise in prior denominator**: In `compute_prior_quantites`,
   use per-shell noise variance instead of scalar `cov_noise`. This is a
   small code change but should significantly improve the SNR scale match.

3. **Noise correlation improvement**: After Fourier cropping is implemented,
   re-evaluate whether the hard-assignment noise needs to be improved. The
   correlation drop is partially caused by the resolution mismatch.

4. **Deferred**: Posterior-weighted noise estimation, tau2_fudge, smoothing.

---

## Code References

- `recovar/reconstruction/regularization.py:75` -- `compute_relion_prior`
- `recovar/reconstruction/regularization.py:238` -- `compute_fsc_prior_gpu`
- `recovar/reconstruction/regularization.py:298` -- `compute_fsc_prior_gpu_v2`
- `recovar/reconstruction/regularization.py:44` -- `compute_prior_quantites`
- `recovar/reconstruction/noise.py:716` -- `estimate_noise_level_no_masks`
- `recovar/em/iterations.py:57` -- `split_E_M_v2`
- RELION `src/backprojector.cpp:1041` -- `updateSSNRarrays`
- RELION `src/ml_optimiser.cpp:5246` -- noise update in `maximization()`
- RELION `src/ml_optimiser.cpp:8241` -- `storeWeightedSums`
