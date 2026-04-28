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
- [`iteration_loop.py:448`](../../recovar/em/dense_single_volume/iteration_loop.py#L448) -- `refine_single_volume()`: public entry point.
  Accepts init volume, noise, dataset; dispatches to `_run_relion_iteration_loop` when
  `relion_mode=True`, otherwise runs legacy dense-grid EM.
- [`iteration_loop.py:931`](../../recovar/em/dense_single_volume/iteration_loop.py#L931) -- `_run_relion_iteration_loop()`: the main iteration loop.
  Each iteration: build grids with perturbation, call `run_em` for E+M step,
  reconstruct via `relion_reconstruct`, update noise/convergence, check stopping.
  Bootstraps `current_size` from init-FSC via
  [`resolution.py:110`](../../recovar/em/dense_single_volume/helpers/resolution.py#L110) -- `_bootstrap_current_size_relion()`.

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
- [`sampling.py:713`](../../recovar/em/sampling.py#L713) -- `get_relion_rotation_grid(order)`: calls the C++ binding
  `get_coarse_orientations` to produce RELION's exact HEALPix grid, then reindexes
  from RELION's (direction-slow, psi-fast) to recovar's (psi-slow, direction-fast)
  so that `index % n_pixels` gives the HEALPix pixel. Returns `(N, 3, 3)` rotation
  matrices in recovar's frame.
- [`sampling.py:523`](../../recovar/em/sampling.py#L523) -- `get_oversampled_rotation_grid_from_samples(...)`: given
  coarse-grid rotation indices, generates oversampled children. Each parent HEALPix
  pixel splits into 4 child pixels (via NEST indexing: `child = 4*parent + {0,1,2,3}`),
  and each parent psi bin splits into 2 midpoint sub-bins (matching RELION's
  `pushbackOversampledPsiAngles`), yielding 8 children per parent. Returns child
  matrices + a `parent_map` array linking each child back to its parent index.
- [`sampling.py:264`](../../recovar/em/sampling.py#L264) -- `get_translation_grid(max_pixel, pixel_offset)`: builds a
  Cartesian 2D grid of (dx, dy) shifts within a circular mask of radius `max_pixel`,
  spaced at `pixel_offset` pixels.
- [`sampling.py:639`](../../recovar/em/sampling.py#L639) -- `get_oversampled_translation_grid(...)`: subdivides each
  parent translation cell into `4^oversampling_order` children (2x2 sub-grid per
  level), halving the step size. Returns child translations + parent_map.

**Perturbation** (RELION `_rlnSamplingPerturbFactor`):

RELION randomly shifts the entire HEALPix grid each iteration to break
discretization bias. The perturbation is a scalar `p` (in units of angular step)
that right-multiplies every rotation matrix by `R(p, p, p)` and adds `p * step`
to every translation. The scalar accumulates across iterations with random increments,
wrapped to `[-pf, +pf]`.

- [`sampling.py:297`](../../recovar/em/sampling.py#L297) -- `advance_relion_perturbation()`: `p += uniform(0.5*pf, pf)`,
  then wrap to `[-pf, +pf]`. Ports `HealpixSampling::resetRandomlyPerturbedSampling`.
- [`sampling.py:326`](../../recovar/em/sampling.py#L326) -- `apply_relion_rotation_perturbation(R, p, step)`: computes
  `R_perturbed = R @ R_from_relion(p*step, p*step, p*step)` for every rotation.
  Ports `healpix_sampling.cpp:1909-1934`.
- [`sampling.py:357`](../../recovar/em/sampling.py#L357) -- `apply_relion_translation_perturbation(T, p, step)`: adds
  `p * step` to both x and y of every translation. Ports `healpix_sampling.cpp:1810-1820`.

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
log p(y_i | r, t) = -1/2 * sum_k |y_i[k] - S_t[k] C_i[k] (P_r mu)[k]|^2 / sigma^2_i[k]
                  = cross(i,r,t) + norm(i,r) + data(i)
```

where `sigma^2_i[k]` is the per-group, per-shell noise power spectrum (one spectrum
per optics group in RELION; one per half-set in recovar), and:
- **cross term**: `sum_k -2 Re( y_i[k]^* S_t[k] C_i[k] (P_r mu)[k] ) / sigma^2_i[k]` -- the expensive GEMM
- **norm term**: `sum_k |C_i[k] (P_r mu)[k]|^2 / sigma^2_i[k]` -- depends on (i, r) since sigma^2 is per-image
- **data term**: `sum_k |y_i[k]|^2 / sigma^2_i[k]` -- constant w.r.t. (r, t), cancels in softmax

The cross term is computed as a matrix multiply: images (shifted by all translations)
form the LHS matrix of shape `(n_img * n_trans, N_pix)`, projections form the RHS
matrix of shape `(N_pix, n_rot)`. This GEMM approach reads each image once for all
rotations, giving 200x better data reuse than FFT-based cross-correlation.

**RELION source**: `ml_optimiser.cpp::getAllSquaredDifferences()` (cross + norm terms),
`projector.cpp::get2DFourierTransform()` (projection via trilinear interpolation).

**recovar code**:
- [`em_engine.py:503`](../../recovar/em/dense_single_volume/em_engine.py#L503) -- `run_em()`: orchestrates the two-pass EM iteration.
  Pass 1 computes scores and collects normalization statistics (logsumexp).
  Pass 2 recomputes scores and applies normalized weights to the M-step.
- [`em_engine.py:174`](../../recovar/em/dense_single_volume/em_engine.py#L174) -- `_e_step_block_scores()`: computes cross + norm terms for a
  block of rotations against all images. Full-resolution path.
- [`em_engine.py:226`](../../recovar/em/dense_single_volume/em_engine.py#L226) -- `_e_step_block_scores_windowed()`: same computation but
  restricted to the Fourier window defined by `current_size` (Section 3a).
- [`em_engine.py:493`](../../recovar/em/dense_single_volume/em_engine.py#L493) -- `_compute_projections_block()`: forward-projects the volume
  at a block of rotations via `slice_volume(half_image=True)`.
- [`em_engine.py:105`](../../recovar/em/dense_single_volume/em_engine.py#L105) -- `_preprocess_batch()`: loads and preprocesses an image batch
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
- [`fourier_window.py:53`](../../recovar/em/dense_single_volume/helpers/fourier_window.py#L53) -- `make_fourier_window_indices(image_shape, current_size)`:
  computes the index set of half-spectrum pixels within the frequency window.
- [`fourier_window.py:145`](../../recovar/em/dense_single_volume/helpers/fourier_window.py#L145) -- `quantize_current_size()`: snaps current_size to an
  allowed grid for JIT cache efficiency.
- [`resolution.py:129`](../../recovar/em/dense_single_volume/helpers/resolution.py#L129) -- `fsc_to_current_size()`: converts an FSC curve to the
  current_size threshold (using FSC > 1/7).

---

## 4. E-step: Posterior Normalization (Two-Pass Logsumexp)

The posterior weight for triplet (i, r, t) is:

```
gamma(i, r, t) = exp(score(i,r,t) - logsumexp_i) * prior(r) * prior(t)
```

Computing `logsumexp_i = log sum_{r,t} exp(score(i,r,t))` requires seeing all
scores for image `i` before normalizing. Since the full `(n_img, n_rot, n_trans)`
tensor is too large to materialize (~100 GB at order 3), `run_em` uses a
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
- [`em_engine.py:344`](../../recovar/em/dense_single_volume/em_engine.py#L344) -- `_update_logsumexp(max_s, sum_exp, scores_block)`: streaming
  logsumexp update. Called once per rotation block in pass 1.
- [`em_engine.py:503`](../../recovar/em/dense_single_volume/em_engine.py#L503) -- `run_em()`: lines ~680-750 (pass 1 loop), lines ~760-850
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
- [`em_engine.py:364`](../../recovar/em/dense_single_volume/em_engine.py#L364) -- `_m_step_block()`: accumulates Ft_y and Ft_ctf for a block
  of rotations. Full-resolution path. Computes `weights @ shifted_images` GEMM,
  then `adjoint_slice_volume` per rotation.
- [`em_engine.py:269`](../../recovar/em/dense_single_volume/em_engine.py#L269) -- `_m_step_block_windowed()`: same but restricted to the
  Fourier window indices. Uses windowed adjoint for the backprojection.
- [`em_engine.py:312`](../../recovar/em/dense_single_volume/em_engine.py#L312) -- `_adjoint_slice_volume_windowed()`: backprojects a windowed
  1D slice into the full 3D Fourier grid, inserting only at the window indices.

**Data flow**: `run_em()` returns `(Ft_y, Ft_ctf)` as flat complex arrays of
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
- [`em_engine.py:423`](../../recovar/em/dense_single_volume/em_engine.py#L423) -- `_compute_noise_block()`: accumulates the noise sufficient
  statistic for a block of rotations during pass 2 of `run_em`. Enabled when
  `accumulate_noise=True`.
- [`noise.py:898`](../../recovar/reconstruction/noise.py#L898) -- `normalize_wsum_to_sigma2_noise()`: normalizes the accumulated
  `wsum_sigma2_noise` by the total posterior weight to get the per-shell noise
  variance estimate.

**Data flow**: `run_em()` returns `wsum_sigma2_noise` and `wsum_img_power` alongside
`(Ft_y, Ft_ctf)`. The caller (`_run_relion_iteration_loop`) normalizes via
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
- [`convergence.py:78`](../../recovar/em/dense_single_volume/helpers/convergence.py#L78) -- `RefinementState`: dataclass tracking all convergence
  state. Fields include `nr_iter_wo_resol_gain`, `nr_iter_wo_assignment_changes`,
  `has_converged`, `do_local_search`, `fraction_changed`, etc.
- [`convergence.py:570`](../../recovar/em/dense_single_volume/helpers/convergence.py#L570) -- `check_convergence(state)`: returns True when convergence
  criteria are met. Matches RELION's three-condition check.
- [`convergence.py:803`](../../recovar/em/dense_single_volume/helpers/convergence.py#L803) -- `update_refinement_state()`: called after each iteration
  to update the state with new assignments, resolution, and angular/translational
  changes. This is the main bookkeeping function.
- [`convergence.py:277`](../../recovar/em/dense_single_volume/helpers/convergence.py#L277) -- `compute_assignment_changes()`: computes the fraction of
  images whose best rotation changed by more than one HEALPix angular step.
- [`convergence.py:333`](../../recovar/em/dense_single_volume/helpers/convergence.py#L333) -- `compute_translation_changes()`: computes RMS change in
  best translations between iterations.
- [`convergence.py:546`](../../recovar/em/dense_single_volume/helpers/convergence.py#L546) -- `compute_ave_Pmax()`: mean of per-image maximum posterior
  probability. Used by RELION to modulate `current_size` growth.
- [`convergence.py:500`](../../recovar/em/dense_single_volume/helpers/convergence.py#L500) -- `calculate_expected_angular_errors()`: estimates angular
  and translational accuracy from the posterior distribution. Used for local
  search sigma estimation.

---

## 9. Angular Refinement and Local Search

### 9a. When does angular refinement trigger?

After each iteration, `update_refinement_state` updates two stall counters:

- `nr_iter_wo_resol_gain`: increments when `current_resolution` did not improve over
  `previous_resolution`. Resets to 0 on any resolution gain.
- `nr_iter_wo_large_hidden_variable_changes`: increments when the per-particle
  angular change (median of `|R_new - R_old|` in degrees) AND per-particle
  translation change (RMS in Angstroms) AND class assignment change are all below
  RELION's hardcoded thresholds (`smallest_changes_optimal_orientations` etc.).
  Resets to 0 if any of the three exceeds its current sticky minimum.

`should_refine_angular_sampling` returns True when BOTH counters have stalled
for at least 1 iteration AND the angular step has not yet exceeded 75% of
the estimated angular accuracy (`acc_rot`, from the posterior width).

**RELION source**: `ml_optimiser.cpp:9772-9790` (`updateAngularSampling`),
`ml_optimiser.cpp:10135-10204` (`checkConvergence`).

### 9b. What happens when refinement triggers?

[`convergence.py:691`](../../recovar/em/dense_single_volume/helpers/convergence.py#L691) -- `refine_angular_sampling(state)`:
1. **HEALPix order += 1**: doubles the number of directions (e.g. 768 -> 3072),
   halves the angular step (e.g. 5.625 -> 2.8125 deg).
2. **Translation step update**: `new_step = min(1.5, 0.75 * acc_trans) * 2^oversampling`.
   This uses the accuracy estimated from the posterior width, not a fixed halving.
3. **Translation range update**: `new_range = 5 * changes_optimal_offsets`, capped at
   1.3x the previous range. `changes_optimal_offsets` is the RMS offset change from
   the last iteration, so the search window shrinks as translations converge.
4. **Stall counters reset**: both `nr_iter_wo_resol_gain` and
   `nr_iter_wo_large_hidden_variable_changes` go back to 0, along with all sticky
   per-particle change baselines. This gives the new finer grid time to converge
   before the next refinement can trigger.
5. **Local search activation**: when `new_order >= 4` (angular step <= 2.8 deg),
   `do_local_search = True`. The initial local-search sigma is
   `sigma = sqrt(8) * angular_step / 2^oversampling` (in radians).

### 9c. How does local search work?

Local search is still full soft-weighted EM — the same `run_em` engine runs the
two-pass E-step (scoring + logsumexp normalization) and M-step (weighted
backprojection). The only change is **which rotations each image evaluates**: instead
of ALL rotations in the grid (infeasible at order >= 4 with 221k+ rotations), each
image scores only a ~100-1500 rotation neighborhood around its previous best
orientation. The Gaussian rotation prior drives out-of-neighborhood rotations to
zero posterior weight, so the soft posterior over the neighborhood is the effective
posterior. The M-step accumulates `Ft_y` and `Ft_ctf` using these soft weights
exactly as in global search.

**Neighborhood construction** --
[`sampling.py:772`](../../recovar/em/sampling.py#L772) -- `get_local_rotation_grid_fast()`:

For each image with prior orientation `(rot_i, tilt_i, psi_i)`:
1. **Direction selection**: compute the angular distance between the image's prior
   viewing direction (z-column of its rotation matrix) and every HEALPix direction
   on the grid. Keep all directions within `3 * max(sigma_rot, sigma_psi)` degrees.
   Typically 20-100 directions out of 3072+ at order 4.
2. **Psi selection**: keep all in-plane angles within `3 * sigma_psi` of the image's
   prior psi. Typically 5-15 psi angles out of 72.
3. **Cartesian product**: the selected rotations are `{selected_directions} x {selected_psi}`,
   typically 100-1500 rotations per image (vs 221k for the full grid).
4. **Gaussian log-prior**: each selected rotation gets a factored prior
   `log_prior[d,p] = log_gauss(diffang_d, sigma_rot) + log_gauss(diffpsi_p, sigma_psi)`,
   normalized so the prior sums to 1. Unselected rotations get `-1e30` (zero weight).

**Bucketization for GPU efficiency** --
[`local_layout.py`](../../recovar/em/dense_single_volume/local_layout.py) --
`build_local_hypothesis_layout()` and `bucket_local_hypothesis_layout()`:

Different images have different prior orientations, so their neighborhoods differ.
The active local path keeps that per-image support explicit:
1. Build a flat `LocalHypothesisLayout` containing each image's selected rotations,
   rotation log-priors, translation priors, and row offsets.
2. Group images by padded local-rotation count so nearby shapes reuse the same
   compiled XLA programs.
3. Build `LocalBucketSpec` batches with per-image rotation masks and `-inf` prior
   on padded rows, preserving the exact per-image posterior support.

**Per-bucket EM iteration** --
[`local_em_engine.py`](../../recovar/em/dense_single_volume/local_em_engine.py) --
`run_local_em_exact()`:

For each local bucket:
1. Preprocess the selected images and CTF rows.
2. Score each image only on its own local rotation neighborhood and translation grid.
3. Normalize posteriors, accumulate `Ft_y`/`Ft_ctf`, noise stats, hard assignments,
   and per-direction posterior sums across buckets.

The per-direction posterior sums feed back into `make_relion_direction_log_prior`
for the next iteration's direction prior (RELION's `pdf_direction`).

**RELION source**: `ml_optimiser.cpp::updateAngularSampling()`,
`healpix_sampling.cpp::selectOrientationsWithNonZeroPriorProbability()`,
`ml_optimiser.cpp::expectationOneParticle()`.

**recovar code summary**:
- [`convergence.py:620`](../../recovar/em/dense_single_volume/helpers/convergence.py#L620) -- `should_refine_angular_sampling()`: trigger check
- [`convergence.py:691`](../../recovar/em/dense_single_volume/helpers/convergence.py#L691) -- `refine_angular_sampling()`: order bump + parameter update
- [`sampling.py:772`](../../recovar/em/sampling.py#L772) -- `get_local_rotation_grid_fast()`: per-image neighborhood with Gaussian prior
- [`sampling.py:102`](../../recovar/em/sampling.py#L102) -- `build_local_search_grid_metadata()`: precomputes direction vectors + psi grid for fast neighbor lookup
- [`local_layout.py`](../../recovar/em/dense_single_volume/local_layout.py) -- `build_local_hypothesis_layout()`: per-image local hypothesis layout
- [`local_layout.py`](../../recovar/em/dense_single_volume/local_layout.py) -- `bucket_local_hypothesis_layout()`: padded shape buckets for exact local EM
- [`local_em_engine.py`](../../recovar/em/dense_single_volume/local_em_engine.py) -- `run_local_em_exact()`: per-bucket EM with per-image priors
- [`orientation_priors.py:16`](../../recovar/em/dense_single_volume/helpers/orientation_priors.py#L16) -- `make_relion_translation_log_prior()`: Gaussian translation prior from `sigma_offset_angstrom` and previous best offset
- [`orientation_priors.py:118`](../../recovar/em/dense_single_volume/helpers/orientation_priors.py#L118) -- `make_relion_direction_log_prior()`: accumulates per-direction posterior across iterations for the direction prior

### 9d. Greedy commitment and the hard-assignment seed

Although local search uses soft EM weights for the M-step, the **neighborhood
center** for each image at iteration `t+1` is determined by the hard MAP
assignment from iteration `t` (the single rotation with the highest posterior
weight). This creates a greedy commitment:

1. At the first local-search iteration, each image's neighborhood is centered on
   its best rotation from the last global-search iteration.
2. After one local EM iteration, the new MAP rotation becomes the next center.
3. If the true orientation lies outside the initial 3-sigma cone, it is never
   evaluated and the image is permanently committed to a local basin.

The soft weights within the neighborhood do NOT prevent this — they only affect
the M-step contribution, not the neighborhood placement. Two rotations that tie
at iteration `t` are broken by first-encountered ordering
([`em_engine.py:1291`](../../recovar/em/dense_single_volume/em_engine.py#L1291):
`improved = block_best > best_score`, strict inequality), and the loser's
neighborhood is never explored.

This is identical to RELION's behavior. Both codes commit to a single local
basin per image once local search begins. The Gaussian prior softens the
within-basin posterior but does not maintain alternative basins.

**Mitigation factors**: (1) the transition from global to local search happens
at order >= 4 (angular step <= 2.8 deg), where most images have already
converged to a narrow posterior peak; (2) the Gaussian sigma starts wide
(`sqrt(8) * angular_step`) and the neighborhood covers 3 sigma, so the initial
cone is typically 15-25 degrees; (3) RELION's 20+ years of production use
suggest this greedy commitment works well in practice for single-class
refinement.

**Design alternative — hierarchical refinement**: instead of committing to one
neighborhood center, maintain the top-K candidate orientations per image across
angular refinements. At each order increment, expand only the surviving
candidates' children and re-prune. This is a natural generalization of the
two-pass oversampling (Section 10) — the fine pass is already one level of
hierarchical refinement; recursing it would prune dead branches early instead
of expanding all children at once. The cost would be ~K evaluations per level
instead of 8^(oversampling_order) per significant parent. Not implemented in
either RELION or recovar; flagged as a post-parity design consideration.

---

## 10. Adaptive Two-Pass Oversampling

Within each EM iteration, the orientation search uses two passes:

**Coarse pass (pass 1)**: evaluate all rotations at the base HEALPix resolution.
Identify the "significant" orientations whose posterior weight exceeds a threshold
(`adaptive_fraction`, default 0.999 of cumulative weight).

**Fine pass (pass 2)**: for each significant coarse orientation, generate child
orientations at the oversampled resolution. At `adaptive_oversampling=1` (RELION
default), each parent produces 4 child HEALPix pixels × 2 child psi midpoints =
8 children. At `adaptive_oversampling=K`, subdivision recurses K times: 4^K child
pixels × 2^K child psi = 8^K children per parent (64 at K=2, 512 at K=3). All K
levels are expanded in one shot — there is no intermediate pruning between levels.

This reduces the effective number of fine orientations from `n_rot * 8^K` (all
children) to `n_significant * 8^K` (typically 100-500 significant orientations
per image), a 50-200x reduction. The lack of intermediate pruning means the cost
grows exponentially with K, which is why RELION uses K=1 in practice.

**RELION source**: `ml_optimiser.cpp::expectationOneParticle()` (two-pass logic,
`exp_ipass` loop), `healpix_sampling.cpp::getOrientations()` (child generation).

**recovar code**:
- [`oversampling.py:59`](../../recovar/em/dense_single_volume/helpers/oversampling.py#L59) -- `find_significant_mask()`: given a flat weight array, returns
  a boolean mask of entries whose cumulative weight covers `adaptive_fraction` of
  the total. Used to select significant (rotation, translation) pairs.
- [`oversampling.py:114`](../../recovar/em/dense_single_volume/helpers/oversampling.py#L114) -- `find_significant_rotations()`: maps the significant mask
  from the flat `(n_rot * n_trans)` space back to unique rotation indices.
- [`oversampling.py:162`](../../recovar/em/dense_single_volume/helpers/oversampling.py#L162) -- `compute_pass2_stats()`: computes statistics for pass 2
  (number of significant orientations, child grid metadata).
- [`em_engine.py:503`](../../recovar/em/dense_single_volume/em_engine.py#L503) -- `run_em()`: the pass-1 / pass-2 logic is handled by the
  caller (`_run_relion_iteration_loop` or `_run_local_search_iteration`), which calls
  `run_em` twice: once for coarse scoring, once for fine scoring with the
  oversampled grid restricted to significant orientations.
- [`significance.py:12`](../../recovar/em/dense_single_volume/helpers/significance.py#L12) -- `_compute_significance_batched()`: batched wrapper that calls
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
- [`iteration_loop.py:931`](../../recovar/em/dense_single_volume/iteration_loop.py#L931) -- `_run_relion_iteration_loop()`: the final iteration block
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
| iteration_loop | [`recovar/em/dense_single_volume/iteration_loop.py`](../../recovar/em/dense_single_volume/iteration_loop.py) | Top-level refinement loop and local search |
| em_engine | [`recovar/em/dense_single_volume/em_engine.py`](../../recovar/em/dense_single_volume/em_engine.py) | Two-pass EM engine (E-step + M-step) |
| convergence | [`recovar/em/dense_single_volume/helpers/convergence.py`](../../recovar/em/dense_single_volume/helpers/convergence.py) | Convergence detection and angular refinement |
| oversampling | [`recovar/em/dense_single_volume/helpers/oversampling.py`](../../recovar/em/dense_single_volume/helpers/oversampling.py) | Two-pass adaptive oversampling (significance pruning) |
| sampling | [`recovar/em/sampling.py`](../../recovar/em/sampling.py) | HEALPix/translation grid generation |
| regularization | [`recovar/reconstruction/regularization.py`](../../recovar/reconstruction/regularization.py) | Tau2 prior estimation and Wiener filter |
| relion_functions | [`recovar/reconstruction/relion_functions.py`](../../recovar/reconstruction/relion_functions.py) | RELION-parity reconstruction helpers |
| noise | [`recovar/reconstruction/noise.py`](../../recovar/reconstruction/noise.py) | Noise variance estimation |
| fourier_window | [`recovar/em/dense_single_volume/helpers/fourier_window.py`](../../recovar/em/dense_single_volume/helpers/fourier_window.py) | Fourier windowing (current_size) |
| local_search | [`recovar/em/dense_single_volume/helpers/local_search.py`](../../recovar/em/dense_single_volume/helpers/local_search.py) | Per-image neighborhood grouping for GPU batching |
| orientation_priors | [`recovar/em/dense_single_volume/helpers/orientation_priors.py`](../../recovar/em/dense_single_volume/helpers/orientation_priors.py) | Direction and translation prior construction |
| resolution | [`recovar/em/dense_single_volume/helpers/resolution.py`](../../recovar/em/dense_single_volume/helpers/resolution.py) | FSC-to-resolution, coarse image size computation |
| significance | [`recovar/em/dense_single_volume/helpers/significance.py`](../../recovar/em/dense_single_volume/helpers/significance.py) | Batched significance computation for pass 2 |

## Known Parity Gaps

1. **Hermitian weights**: RELION uses `w=1` for all half-spectrum pixels. The correct
   approach uses `w=2` for interior frequencies (which have a conjugate partner) and
   `w=1` for DC/Nyquist. We match RELION for parity; the fix (`make_half_image_weights`
   in [`em_engine.py:57`](../../recovar/em/dense_single_volume/em_engine.py#L57)) is ready but not yet enabled.

2. **Iter-1 hard CC**: RELION binarizes posterior weights at iter 1 when
   `--firstiter_cc` is set (winner-take-all). recovar uses soft Bayesian posterior
   at all iterations. This causes a transient Pmax gap at iter 1-6 that converges.

3. **Gridding correction**: RELION applies a trilinear gridding correction during
   backprojection. recovar uses the same trilinear kernel but applies the correction
   at reconstruction time rather than insertion time. Both produce identical results
   when the volume is resolved (tested to rel_err < 1e-12).

## Post-Parity Design Considerations

1. **Hierarchical refinement**: replace the flat coarse→fine expansion (Section 10)
   and greedy local-search commitment (Section 9d) with recursive prune-and-expand.
   At each angular refinement level, expand only the top-K surviving candidates per
   image, score them, prune, and recurse. This is strictly cheaper than increasing
   `adaptive_oversampling` (which expands 8^K children without intermediate pruning)
   and avoids the greedy commitment problem where a single hard MAP assignment locks
   each image into one local basin. The two-pass oversampling is already one level
   of this hierarchy — the generalization is to recurse it. Trade-off: requires
   maintaining per-image candidate lists across iterations (memory) and complicates
   the GPU batching strategy (different images have different candidate counts).

2. **Correct Hermitian weights**: see Known Parity Gap #1. Enabling
   `make_half_image_weights` would sharpen posteriors without changing MAP rankings.

3. **Soft local-search seeding**: use the top-K posterior peaks (not just the MAP)
   to seed multiple overlapping neighborhoods per image. Combined with hierarchical
   refinement, this would let local search explore alternative basins that the
   current greedy commitment discards.
