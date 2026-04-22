# RELION Auto-Refine Algorithm -- Annotated Code Map

This document walks through the RELION 5.0 auto-refine algorithm step by step,
mapping each step to both RELION C++ source and the recovar Python implementation.
All line numbers refer to the state of the code on the `claude/relion-parity-flag-audit`
branch after cleanup. Code references are clickable links that open the file at the
correct line in VS Code.

## Overview

RELION's auto-refine performs iterative expectation-maximization (EM) to reconstruct
a single 3D volume from a set of 2D cryo-EM particle images. Each iteration:

1. **E-step**: score every (image, rotation, translation) triplet and normalize to
   posterior weights.
2. **M-step**: accumulate weighted backprojections into Fourier-space sufficient
   statistics and reconstruct the volume with Wiener regularization.
3. **Convergence check**: track assignment changes and resolution stalls to decide
   when to refine angular sampling or stop.

The algorithm uses gold-standard FSC: the dataset is split into two random halves
that are refined independently, joined only at low resolution (< 40 A).

```
                            +--------------------+
                            |  Initialization    |
                            | (grids, noise, vol)|
                            +--------+-----------+
                                     |
                            +--------v-----------+
                       +--->|  E-step: scoring   |
                       |    |  + normalization   |
                       |    +--------+-----------+
                       |             |
                       |    +--------v-----------+
                       |    |  M-step: accum.    |
                       |    |  + reconstruction  |
                       |    +--------+-----------+
                       |             |
                       |    +--------v-----------+
                       |    |  Noise update      |
                       |    +--------+-----------+
                       |             |
                       |    +--------v-----------+
                       |    |  Convergence check |
                       |    |  + angular refine  |
                       |    +--------+-----------+
                       |             |
                       |    converged? --no------+
                       |             |
                       |            yes
                       |             |
                       |    +--------v-----------+
                       +----| Final join & output|
                            +--------------------+
```

---

## 1. Initialization

Entry point for the full refinement loop. Sets up the rotation/translation grids,
bootstraps the initial current_size from the FSC of the init volume, and enters
the iteration loop.

**RELION source**: `ml_optimiser.cpp::iterate()` (top-level loop),
`ml_optimiser.cpp::initialiseGeneral()` (grid setup).

**recovar code**:
- [`refine.py:448`](../../recovar/em/dense_single_volume/refine.py#L448) -- `refine_single_volume()`: public API, dispatches to `_refine_relion_mode`.
- [`refine.py:931`](../../recovar/em/dense_single_volume/refine.py#L931) -- `_refine_relion_mode()`: the iteration loop. Sets up initial
  `RefinementState`, computes init FSC, bootstraps `current_size` via
  [`relion_init.py:110`](../../recovar/em/dense_single_volume/refine_dev_helpers/relion_init.py#L110) -- `_bootstrap_current_size_relion()`.

**Key state**:
- `mean` (complex, flat): two half-set volumes in centered Fourier space.
- `noise_variance`: per-shell noise power spectrum, one per half-set.
- `RefinementState`: tracks healpix_order, translation range/step, resolution,
  convergence counters, best assignments (see Section 8).

**Deliberate deviations from RELION**:
- RELION initializes from a low-pass-filtered reference and estimates noise from
  the data in iter 0. recovar takes pre-computed noise estimates and init volume
  as inputs.

---

## 2. Rotation and Translation Grids

The orientation search uses HEALPix for the two Euler angles (rot, tilt) that
define the projection direction, combined with a uniform grid for the third
angle (psi, in-plane rotation). Translations are searched on a Cartesian grid.
Both grids support adaptive oversampling: a coarse pass followed by refinement
around significant orientations.

**RELION source**: `healpix_sampling.cpp::initialise()`,
`healpix_sampling.cpp::selectOrientationsWithPerturbation()`.

**recovar code**:
- [`sampling.py:713`](../../recovar/em/sampling.py#L713) -- `get_relion_rotation_grid(order)`: generates the coarse
  HEALPix grid at a given order. Returns rotation matrices.
- [`sampling.py:523`](../../recovar/em/sampling.py#L523) -- `get_oversampled_rotation_grid_from_samples(parent_pixels,
  ...)`: subdivides parent HEALPix pixels into child orientations at finer
  oversampling. Used in pass 2 of adaptive oversampling.
- [`sampling.py:264`](../../recovar/em/sampling.py#L264) -- `get_translation_grid(max_pixel, pixel_offset)`: Cartesian
  translation grid in pixel units.
- [`sampling.py:639`](../../recovar/em/sampling.py#L639) -- `get_oversampled_translation_grid(parent_translations, ...)`:
  subdivides parent translations for pass-2 oversampling.

**Perturbation** (RELION `_rlnSamplingPerturbFactor`):
- [`sampling.py:297`](../../recovar/em/sampling.py#L297) -- `advance_relion_perturbation()`: generates per-iteration
  random perturbation parameters, matching RELION's `selectOrientationsWithPerturbation`.
- [`sampling.py:326`](../../recovar/em/sampling.py#L326) -- `apply_relion_rotation_perturbation()`: applies random
  rotation perturbation to the HEALPix grid.
- [`sampling.py:357`](../../recovar/em/sampling.py#L357) -- `apply_relion_translation_perturbation()`: applies random
  translation perturbation.

**Grid sizes** (order -> number of rotations):
| Order | Directions | Psi steps | Total rotations | Angular step |
|-------|-----------|-----------|-----------------|--------------|
| 1     | 48        | 10        | 480             | 22.5 deg     |
| 2     | 192       | 18        | 3,456           | 11.25 deg    |
| 3     | 768       | 36        | 27,648          | 5.625 deg    |
| 4     | 3,072     | 72        | 221,184         | 2.8125 deg   |

With `adaptive_oversampling=1`, each coarse orientation has 8 (rot/tilt) x 4 (psi)
= 32 children, but only significant orientations are subdivided (Section 10).

---

## 3. E-step: Scoring

The E-step computes log-posterior scores for every (image, rotation, translation)
triplet. The forward model is:

```
y_i = S_t C_i P_r mu + eps_i
```

where `S_t` is the translation phase shift, `C_i` is the CTF for image `i`,
`P_r` is the projection operator for rotation `r`, `mu` is the 3D volume,
and `eps_i ~ N(0, sigma^2_i)` is per-image Gaussian noise.

The log-likelihood for a single (i, r, t) decomposes into:

```
log p(y_i | r, t) = -1/(2*sigma^2) * ||y_i - S_t C_i P_r mu||^2
                  = cross(i,r,t) + norm(r) + data(i)
```

where:
- **cross term**: `-2 Re <y_i / sigma^2, S_t C_i P_r mu>` -- the expensive GEMM
- **norm term**: `||C_i P_r mu||^2 / sigma^2` -- depends on (r) only via projections
- **data term**: `||y_i||^2 / sigma^2` -- constant w.r.t. (r, t), cancels in softmax

The cross term is computed as a matrix multiply: images (shifted by all translations)
form the LHS matrix of shape `(n_img * n_trans, N_pix)`, projections form the RHS
matrix of shape `(N_pix, n_rot)`. This GEMM approach reads each image once for all
rotations, giving 200x better data reuse than FFT-based cross-correlation.

**RELION source**: `ml_optimiser.cpp::getAllSquaredDifferences()` (cross + norm terms),
`projector.cpp::get2DFourierTransform()` (projection via trilinear interpolation).

**recovar code**:
- [`engine_v2.py:503`](../../recovar/em/dense_single_volume/engine_v2.py#L503) -- `run_em_v2()`: orchestrates the two-pass EM iteration.
  Pass 1 computes scores and collects normalization statistics (logsumexp).
  Pass 2 recomputes scores and applies normalized weights to the M-step.
- [`engine_v2.py:174`](../../recovar/em/dense_single_volume/engine_v2.py#L174) -- `_e_step_block_scores()`: computes cross + norm terms for a
  block of rotations against all images. Full-resolution path.
- [`engine_v2.py:226`](../../recovar/em/dense_single_volume/engine_v2.py#L226) -- `_e_step_block_scores_windowed()`: same computation but
  restricted to the Fourier window defined by `current_size` (Section 3a).
- [`engine_v2.py:493`](../../recovar/em/dense_single_volume/engine_v2.py#L493) -- `_compute_projections_block()`: forward-projects the volume
  at a block of rotations via `slice_volume(half_image=True)`.
- [`engine_v2.py:105`](../../recovar/em/dense_single_volume/engine_v2.py#L105) -- `_preprocess_batch()`: loads and preprocesses an image batch
  (CTF weighting, noise normalization, translation phase shifts).

**Half-spectrum optimization**: all GEMMs operate on the rfft half-image layout
(`N_half = H * (W//2 + 1)` instead of `N = H * W`), giving ~2x speedup.
Hermitian conjugate weights are absorbed into projections (precomputed once per
rotation block).

### 3a. Fourier Windowing (current_size)

At early iterations, the volume is resolved only to low frequency. RELION crops
images and projections to `current_size` pixels before the E-step GEMM, reducing
the pixel count from `N_half` to `N_window << N_half`. This is the single biggest
speedup at early iterations (50x+ fewer pixels at order 2).

**RELION source**: `ml_optimiser.cpp::getFourierTransformsAndCtfs()` (window images),
`backprojector.cpp::get2DFourierTransform()` (window projections).

**recovar code**:
- [`fourier_window.py:53`](../../recovar/em/dense_single_volume/refine_dev_helpers/fourier_window.py#L53) -- `make_fourier_window_indices(image_shape, current_size)`:
  computes the index set of half-spectrum pixels within the frequency window.
- [`fourier_window.py:145`](../../recovar/em/dense_single_volume/refine_dev_helpers/fourier_window.py#L145) -- `quantize_current_size()`: snaps current_size to an
  allowed grid for JIT cache efficiency.
- [`relion_init.py:129`](../../recovar/em/dense_single_volume/refine_dev_helpers/relion_init.py#L129) -- `fsc_to_current_size()`: converts an FSC curve to the
  current_size threshold (using FSC > 1/7).

---

## 4. E-step: Posterior Normalization (Two-Pass Logsumexp)

The posterior weight for triplet (i, r, t) is:

```
gamma(i, r, t) = exp(score(i,r,t) - logsumexp_i) * prior(r) * prior(t)
```

Computing `logsumexp_i = log sum_{r,t} exp(score(i,r,t))` requires seeing all
scores for image `i` before normalizing. Since the full `(n_img, n_rot, n_trans)`
tensor is too large to materialize (~100 GB at order 3), `run_em_v2` uses a
streaming two-pass approach:

**Pass 1** (normalization): iterate over all rotation blocks, computing scores and
updating a running `(max, sum_exp)` pair per image via the logsumexp identity:
```
logsumexp(a, b) = max(a,b) + log(exp(a - max(a,b)) + exp(b - max(a,b)))
```
This gives the exact logsumexp without storing all scores.

**Pass 2** (accumulation): iterate over all rotation blocks again, recompute scores,
normalize using the logsumexp from pass 1, and accumulate into the M-step.

**RELION source**: `ml_optimiser.cpp::expectationOneParticle()` (single-pass with
explicit weight storage), `ml_optimiser.cpp::convertAllSquaredDifferencesToWeights()`.

**recovar code**:
- [`engine_v2.py:344`](../../recovar/em/dense_single_volume/engine_v2.py#L344) -- `_update_logsumexp(max_s, sum_exp, scores_block)`: streaming
  logsumexp update. Called once per rotation block in pass 1.
- [`engine_v2.py:503`](../../recovar/em/dense_single_volume/engine_v2.py#L503) -- `run_em_v2()`: lines ~680-750 (pass 1 loop), lines ~760-850
  (pass 2 loop).

**Key difference from RELION**: RELION stores the full weight tensor in memory
(feasible on CPU with sparse storage). recovar never materializes it, recomputing
scores in pass 2. This doubles the projection cost but eliminates the memory
bottleneck, enabling GPU execution.

---

## 5. M-step: Accumulation

The M-step accumulates two sufficient statistics in Fourier space:

```
Ft_y[k]   += sum_{i,r,t} gamma(i,r,t) * P_r^*(S_t^* C_i y_i)[k] / sigma^2_i[k]
Ft_ctf[k] += sum_{i,r,t} gamma(i,r,t) * |C_i[k]|^2 / sigma^2_i[k]
```

where `P_r^*` is the adjoint projection (backprojection via `adjoint_slice_volume`),
`S_t^*` is the conjugate translation shift, and the sums run over all images,
rotations, and translations weighted by their posterior probabilities.

The accumulation is fused with the E-step in pass 2: after normalizing a block of
scores, the weighted image sum is formed by a GEMM (weights matrix times shifted
images), then each resulting rotation slice is backprojected into the 3D Fourier grid.

**RELION source**: `ml_optimiser.cpp::storeWeightedSums()` (accumulate Ft_y, Ft_ctf),
`backprojector.cpp::set2DFourierTransform()` (backprojection via trilinear insertion).

**recovar code**:
- [`engine_v2.py:364`](../../recovar/em/dense_single_volume/engine_v2.py#L364) -- `_m_step_block()`: accumulates Ft_y and Ft_ctf for a block
  of rotations. Full-resolution path. Computes `weights @ shifted_images` GEMM,
  then `adjoint_slice_volume` per rotation.
- [`engine_v2.py:269`](../../recovar/em/dense_single_volume/engine_v2.py#L269) -- `_m_step_block_windowed()`: same but restricted to the
  Fourier window indices. Uses windowed adjoint for the backprojection.
- [`engine_v2.py:312`](../../recovar/em/dense_single_volume/engine_v2.py#L312) -- `_adjoint_slice_volume_windowed()`: backprojects a windowed
  1D slice into the full 3D Fourier grid, inserting only at the window indices.

**Data flow**: `run_em_v2()` returns `(Ft_y, Ft_ctf)` as flat complex arrays of
shape `(volume_size,)` in centered Fourier space. These are the inputs to the
reconstruction step (Section 6).

---

## 6. M-step: Reconstruction

Given the accumulated sufficient statistics `(Ft_y, Ft_ctf)`, the volume is
reconstructed via Wiener filtering with a spectral prior (tau2):

```
mu[k] = Ft_y[k] / (Ft_ctf[k] + 1/tau2[k])
```

where `tau2[k]` is the signal power spectrum (regularization prior), estimated from
the FSC between the two half-set reconstructions. The reconstruction proceeds in
three sub-steps:

1. **Unregularized reconstruction**: `mu_unreg = Ft_y / Ft_ctf` (or with a small
   epsilon for numerical stability).
2. **FSC computation**: compare the two half-set unregularized reconstructions to
   get the gold-standard FSC curve.
3. **Prior estimation**: convert FSC to tau2 via `tau2 = FSC / (1 - FSC) * sigma^2 / Ft_ctf`.
4. **Regularized reconstruction**: apply the Wiener filter with the estimated tau2.
5. **Resolution update**: compute `data_vs_prior` ratio per shell and find the
   resolution where signal dominates noise.

**RELION source**: `ml_optimiser.cpp::maximization()`,
`backprojector.cpp::reconstruct()` (Wiener inversion),
`ml_optimiser.cpp::updateCurrentResolution()`.

**recovar code**:
- [`relion_functions.py:716`](../../recovar/reconstruction/relion_functions.py#L716) -- `relion_reconstruct()`: full RELION-parity Wiener
  reconstruction. Takes `(Ft_y, Ft_ctf)`, computes gridding correction, applies
  Wiener filter with tau2 regularization, returns the reconstructed volume.
- [`regularization.py:75`](../../recovar/reconstruction/regularization.py#L75) -- `compute_relion_prior()`: computes tau2 from the FSC
  between two half-set unregularized volumes. Matches RELION's
  `BackProjector::reconstruct()` prior computation.
- [`regularization.py:671`](../../recovar/reconstruction/regularization.py#L671) -- `compute_data_vs_prior()`: computes the per-shell
  `Ft_ctf / tau2` ratio. Used to determine current resolution.
- [`regularization.py:709`](../../recovar/reconstruction/regularization.py#L709) -- `resolution_from_data_vs_prior()`: finds the shell
  where `data_vs_prior > 1` (signal dominates noise), returns current resolution.
- [`relion_functions.py:503`](../../recovar/reconstruction/relion_functions.py#L503) -- `adjust_regularization_relion_style()`: applies
  RELION's tau2 fudge factor and regularization adjustments.
- [`regularization.py:895`](../../recovar/reconstruction/regularization.py#L895) -- `join_halves_at_low_resolution()`: below a threshold
  (40 A default), replaces each half-map with the average of both halves. Prevents
  half-set divergence at low frequencies.

**Key variables**:
- `tau2`: 1D array of shape `(n_shells,)`, signal power spectrum.
- `data_vs_prior`: 1D array, ratio of data support to prior strength per shell.
- `current_size`: integer, diameter in pixels corresponding to current resolution.

---

## 7. Noise Update

After reconstruction, the noise model is updated using the posterior-weighted
residuals. For each image, the expected squared residual under the posterior is:

```
sigma^2_new[k] = (1/N) sum_i sum_{r,t} gamma(i,r,t) * |y_i[k] - S_t C_i P_r mu[k]|^2
```

In practice, this is accumulated during the M-step as an additional sufficient
statistic (`wsum_sigma2_noise`) that tracks the weighted sum of squared residuals.

**RELION source**: `ml_optimiser.cpp::storeWeightedSums()` (accumulates `wsum_sigma2_noise`),
`ml_optimiser.cpp::updateOtherParams()` (normalizes to per-shell noise variance).

**recovar code**:
- [`engine_v2.py:423`](../../recovar/em/dense_single_volume/engine_v2.py#L423) -- `_compute_noise_block()`: accumulates the noise sufficient
  statistic for a block of rotations during pass 2 of `run_em_v2`. Enabled when
  `accumulate_noise=True`.
- [`noise.py:898`](../../recovar/reconstruction/noise.py#L898) -- `normalize_wsum_to_sigma2_noise()`: normalizes the accumulated
  `wsum_sigma2_noise` by the total posterior weight to get the per-shell noise
  variance estimate.

**Data flow**: `run_em_v2()` returns `wsum_sigma2_noise` and `wsum_img_power` alongside
`(Ft_y, Ft_ctf)`. The caller (`_refine_relion_mode`) normalizes via
`normalize_wsum_to_sigma2_noise` and updates the noise model for the next iteration.

---

## 8. Convergence Detection

RELION auto-refine tracks three signals to decide when to stop:

1. **Resolution stall**: no improvement in current resolution for N consecutive
   iterations (`nr_iter_wo_resol_gain >= 1`).
2. **Assignment stability**: fraction of images whose best orientation changed by
   less than one HEALPix step (`nr_iter_wo_assignment_changes >= 1`).
3. **Already at finest sampling**: healpix_order has reached `max_healpix_order`
   and cannot be refined further.

Convergence is declared when all three hold AND the angular sampling has already
been refined to the finest level (i.e., there's nothing left to try).

**RELION source**: `ml_optimiser.cpp::checkConvergence()` (all three checks),
`ml_optimiser.cpp::updateAngularSampling()` (angular refinement trigger).

**recovar code**:
- [`convergence.py:78`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L78) -- `RefinementState`: dataclass tracking all convergence
  state. Fields include `nr_iter_wo_resol_gain`, `nr_iter_wo_assignment_changes`,
  `has_converged`, `do_local_search`, `fraction_changed`, etc.
- [`convergence.py:570`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L570) -- `check_convergence(state)`: returns True when convergence
  criteria are met. Matches RELION's three-condition check.
- [`convergence.py:803`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L803) -- `update_refinement_state()`: called after each iteration
  to update the state with new assignments, resolution, and angular/translational
  changes. This is the main bookkeeping function.
- [`convergence.py:277`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L277) -- `compute_assignment_changes()`: computes the fraction of
  images whose best rotation changed by more than one HEALPix angular step.
- [`convergence.py:333`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L333) -- `compute_translation_changes()`: computes RMS change in
  best translations between iterations.
- [`convergence.py:546`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L546) -- `compute_ave_Pmax()`: mean of per-image maximum posterior
  probability. Used by RELION to modulate `current_size` growth.
- [`convergence.py:500`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L500) -- `calculate_expected_angular_errors()`: estimates angular
  and translational accuracy from the posterior distribution. Used for local
  search sigma estimation.

---

## 9. Angular Refinement

When the algorithm detects that it has converged at the current angular sampling
(resolution stalled + assignments stable), it increases the HEALPix order by 1,
doubles the number of directions, halves the angular step, and adjusts the
translation grid correspondingly.

When `healpix_order >= 4` (angular step <= 2.8 deg), the algorithm switches from
global search to **local search**: each image searches only in a neighborhood
around its current best orientation, using a Gaussian prior with sigma estimated
from the posterior width.

**RELION source**: `ml_optimiser.cpp::updateAngularSampling()` (order increment +
local search transition), `healpix_sampling.cpp::getLocalSearchGrid()`.

**recovar code**:
- [`convergence.py:620`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L620) -- `should_refine_angular_sampling(state)`: returns True
  when resolution has stalled and assignments are stable, matching RELION's trigger
  conditions.
- [`convergence.py:691`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py#L691) -- `refine_angular_sampling(state)`: increments healpix_order,
  updates angular_step, adjusts translation range/step, and enables local search
  when order >= 4.
- [`refine.py:90`](../../recovar/em/dense_single_volume/refine.py#L90) -- `_run_grouped_local_search_em()`: the local-search EM iteration.
  Groups images by their best prior orientation into GPU-friendly batches, builds
  per-group HEALPix neighborhoods, and runs `run_em_v2` with image-specific rotation
  priors.
- [`sampling.py:772`](../../recovar/em/sampling.py#L772) -- `get_local_rotation_grid_fast()`: generates the HEALPix
  neighborhood for a single direction at a given order. Uses exact pixel neighbor
  lookup to find all orientations within the search radius.
- [`sampling.py:102`](../../recovar/em/sampling.py#L102) -- `build_local_search_grid_metadata()`: precomputes the local
  search grid (neighbor lists + Gaussian prior weights) for a set of seed directions.
- [`local_search.py:92`](../../recovar/em/dense_single_volume/refine_dev_helpers/local_search.py#L92) -- `_partition_local_search_groups()`: groups images by their
  seed direction so that images with similar neighborhoods share the same rotation
  grid on the GPU.
- [`relion_priors.py:16`](../../recovar/em/dense_single_volume/refine_dev_helpers/relion_priors.py#L16) -- `make_relion_translation_log_prior()`: computes Gaussian
  translation log-prior centered on each image's previous best translation.
- [`relion_priors.py:118`](../../recovar/em/dense_single_volume/refine_dev_helpers/relion_priors.py#L118) -- `make_relion_direction_log_prior()`: computes per-direction
  prior weights from the accumulated posterior over directions. Used for direction
  prior in local search.

**Key insight**: in local search, each image sees a different rotation grid (its
own neighborhood). To exploit GPU parallelism, images are grouped by seed direction
so that images in the same group share the same rotation grid and can be processed
in a single batched GEMM.

---

## 10. Adaptive Two-Pass Oversampling

Within each EM iteration, the orientation search uses two passes:

**Coarse pass (pass 1)**: evaluate all rotations at the base HEALPix resolution.
Identify the "significant" orientations whose posterior weight exceeds a threshold
(`adaptive_fraction`, default 0.999 of cumulative weight).

**Fine pass (pass 2)**: for each significant coarse orientation, generate child
orientations at the oversampled resolution (32 children per parent at
`adaptive_oversampling=1`). Re-evaluate only these children in the E-step and
use their weights for the M-step.

This reduces the effective number of fine orientations from `n_rot * 32` (all
children) to `n_significant * 32` (typically 100-500 significant orientations
per image), a 50-200x reduction.

**RELION source**: `ml_optimiser.cpp::expectationOneParticle()` (two-pass logic,
`exp_ipass` loop), `healpix_sampling.cpp::getOrientations()` (child generation).

**recovar code**:
- [`adaptive.py:59`](../../recovar/em/dense_single_volume/refine_dev_helpers/adaptive.py#L59) -- `find_significant_mask()`: given a flat weight array, returns
  a boolean mask of entries whose cumulative weight covers `adaptive_fraction` of
  the total. Used to select significant (rotation, translation) pairs.
- [`adaptive.py:114`](../../recovar/em/dense_single_volume/refine_dev_helpers/adaptive.py#L114) -- `find_significant_rotations()`: maps the significant mask
  from the flat `(n_rot * n_trans)` space back to unique rotation indices.
- [`adaptive.py:162`](../../recovar/em/dense_single_volume/refine_dev_helpers/adaptive.py#L162) -- `compute_pass2_stats()`: computes statistics for pass 2
  (number of significant orientations, child grid metadata).
- [`engine_v2.py:503`](../../recovar/em/dense_single_volume/engine_v2.py#L503) -- `run_em_v2()`: the pass-1 / pass-2 logic is handled by the
  caller (`_refine_relion_mode` or `_run_grouped_local_search_em`), which calls
  `run_em_v2` twice: once for coarse scoring, once for fine scoring with the
  oversampled grid restricted to significant orientations.
- [`significance.py:12`](../../recovar/em/dense_single_volume/refine_dev_helpers/significance.py#L12) -- `_compute_significance_batched()`: batched wrapper that calls
  `find_significant_rotations` per image and collects the significant rotation
  indices for pass 2.

---

## 11. Final Join and Output

After convergence, the two half-set volumes are:
1. Individually post-processed (gridding correction, Wiener filter).
2. Joined at low resolution (< 40 A) to produce the final merged volume.
3. The final FSC is computed between the two half-sets.

**RELION source**: `ml_optimiser.cpp::do_join_random_halves()` at the final
iteration, followed by `ml_optimiser.cpp::writeOutput()`.

**recovar code**:
- [`regularization.py:895`](../../recovar/reconstruction/regularization.py#L895) -- `join_halves_at_low_resolution()`: replaces each
  half-map with the average of both below the join threshold.
- [`refine.py:931`](../../recovar/em/dense_single_volume/refine.py#L931) -- `_refine_relion_mode()`: the final iteration block
  (after convergence) performs reconstruction, computes the final FSC, and
  returns the merged volume along with all refinement metadata.
- [`relion_functions.py:716`](../../recovar/reconstruction/relion_functions.py#L716) -- `relion_reconstruct()`: final Wiener
  reconstruction of the merged volume.

**Outputs**:
- `mean`: final 3D volume in centered Fourier space (flat complex array).
- `fsc`: gold-standard FSC curve between half-sets.
- `noise_variance`: final per-shell noise estimate.
- `RefinementState`: final convergence state with best assignments per image.

---

## File Index

| Module | Path | Purpose |
|--------|------|---------|
| refine | [`recovar/em/dense_single_volume/refine.py`](../../recovar/em/dense_single_volume/refine.py) | Top-level refinement loop and local search |
| engine_v2 | [`recovar/em/dense_single_volume/engine_v2.py`](../../recovar/em/dense_single_volume/engine_v2.py) | Two-pass EM engine (E-step + M-step) |
| convergence | [`recovar/em/dense_single_volume/refine_dev_helpers/convergence.py`](../../recovar/em/dense_single_volume/refine_dev_helpers/convergence.py) | Convergence detection and angular refinement |
| adaptive | [`recovar/em/dense_single_volume/refine_dev_helpers/adaptive.py`](../../recovar/em/dense_single_volume/refine_dev_helpers/adaptive.py) | Significant weight selection for oversampling |
| sampling | [`recovar/em/sampling.py`](../../recovar/em/sampling.py) | HEALPix/translation grid generation |
| regularization | [`recovar/reconstruction/regularization.py`](../../recovar/reconstruction/regularization.py) | Tau2 prior estimation and Wiener filter |
| relion_functions | [`recovar/reconstruction/relion_functions.py`](../../recovar/reconstruction/relion_functions.py) | RELION-parity reconstruction helpers |
| noise | [`recovar/reconstruction/noise.py`](../../recovar/reconstruction/noise.py) | Noise variance estimation |
| fourier_window | [`recovar/em/dense_single_volume/refine_dev_helpers/fourier_window.py`](../../recovar/em/dense_single_volume/refine_dev_helpers/fourier_window.py) | Fourier windowing (current_size) |

## Known Parity Gaps

1. **Hermitian weights**: RELION uses `w=1` for all half-spectrum pixels. The correct
   approach uses `w=2` for interior frequencies (which have a conjugate partner) and
   `w=1` for DC/Nyquist. We match RELION for parity; the fix (`make_half_image_weights`
   in [`engine_v2.py:57`](../../recovar/em/dense_single_volume/engine_v2.py#L57)) is ready but not yet enabled.

2. **Iter-1 hard CC**: RELION binarizes posterior weights at iter 1 when
   `--firstiter_cc` is set (winner-take-all). recovar uses soft Bayesian posterior
   at all iterations. This causes a transient Pmax gap at iter 1-6 that converges.

3. **Gridding correction**: RELION applies a trilinear gridding correction during
   backprojection. recovar uses the same trilinear kernel but applies the correction
   at reconstruction time rather than insertion time. Both produce identical results
   when the volume is resolved (tested to rel_err < 1e-12).
