# RELION-Parity Dense Single-Volume EM: Corrected Development Plan

## Goal

Bring `recovar/em/` to feature parity with RELION's `relion_refine --auto_refine` for single-class, single-volume homogeneous refinement. Match RELION's output (resolution, convergence trajectory, per-image assignments) while being at least as fast on the same GPU.

**Out of scope**: K-class classification, heterogeneity/PPCA, helical symmetry, tilt series, CTF refinement, Bayesian polishing.

**Primary execution target**: `engine_v2.py` (the optimized blockwise path), not the legacy Equinox path in `core.py`/`m_step.py`.

**Scale targets**: The benchmark dataset is 5000 images at 128px, but the implementation must be designed to handle production-scale problems: **up to 512×512 images and 1M+ particles**, potentially across multiple GPUs. Every design decision (memory layout, batching strategy, data structures) should be evaluated against this target, not just the benchmark. Hardcoded sizes, in-memory probability tensors that scale with n_images × n_rot, and single-GPU assumptions are not acceptable in the final code. The benchmark is for rapid iteration; the architecture is for production. RELION handles this scale routinely (hours to days on multi-GPU clusters). We should match that capability.

---

## Critical Corrections from Critique

This plan incorporates the following corrections to the original spec. Agents should treat these as ground truth.

### C1: Fourier cropping is NOT "use smaller image_shape"

RECOVAR's `slice_volume` / `adjoint_slice_volume` use `upsampling = volume_shape[0] // image_shape[0]`, which assumes an integer ratio. Passing `image_shape=(18,18)` with `volume_shape=(128,128,128)` gives `128//18 = 7`, destroying frequency spacing. RELION instead windows the original Fourier grid to a smaller radius while preserving physical frequency spacing. The correct implementation is a **coordinate-preserving Fourier window/mask on the original grid**, not a change to `image_shape`.

### C2: The original plan's prior formula is wrong

The plan described `tau^2(k) = FSC/(1-FSC) * sigma^2(k)`. The actual RECOVAR code in `compute_relion_prior` / `compute_fsc_prior_gpu` forms `SNR = FSC / (1 - FSC)` and sets `prior_avg = SNR / bottom_avg` in the normal path, or `prior_avg = SNR * noise_level` only in the `from_noise_level=True` path (which the code labels "outdated"). The plan must be built around the actual code path.

### C3: The packed half-spectrum endpoint labels are reversed

In `fourier_transform_utils.py`, for even W: packed column 0 is the shifted **DC** column and packed column -1 is the shifted **Nyquist** column, not the other way around as the original Step 1 comments say. The numeric weights happen to be symmetric (both get weight 1), so computation is unaffected, but comments and any endpoint-specific crop/mask logic must use the correct labels.

### C4: Noise estimation is hard-assignment and subset-based

`split_E_M_v2` converts hard assignments into best rotations/translations, then estimates noise on `experiment_datasets[0]` only, over `min(1000, n_units)` images. This is not a posterior-weighted noise estimate. This is a known algorithmic difference from RELION, not a minor verification item.

### C5: RELION's "significant samples" includes translations, not just orientations

`_rlnNrOfSignificantSamples` counts significant assignments from the first pass of adaptive oversampling (orientations AND translations together). The original plan's Step 6 capped only orientations.

### C6: Single optics group assumption must be stated

RELION keeps `image_coarse_size`, `image_current_size`, and `image_full_size` as vectors (one per optics group). Our plan assumes a single optics group. This must be documented explicitly.

---

## Architecture: Where to Make Changes

All optimization work targets the fast path:

```
recovar/em/dense_single_volume/engine_v2.py   -- primary target
recovar/em/dense_single_volume/types.py        -- DensePoseGrid, DenseEMPlan
recovar/em/dense_single_volume/plan.py         -- memory planner
recovar/em/iterations.py                       -- split_E_M_v2 orchestrator
recovar/core/fourier_transform_utils.py        -- half-spectrum layout
recovar/core/slicing.py                        -- CUDA forward/adjoint slice
```

Do NOT modify `heterogeneity.py` (separate owners). Avoid modifying the legacy `core.py` / `m_step.py` path unless specifically needed for shared utilities.

---

## Implementation Order

```
Phase 0: Convention Lock + RELION Reference
    Step 0A: Internal RECOVAR convention lock
    Step 0B: RELION reference harness

Phase 1: Half-Spectrum GEMMs (in engine_v2)

Phase 2: Prior/Noise Alignment with RELION
    (moved AHEAD of Fourier cropping)

Phase 3: Coordinate-Preserving Fourier Windowing
    Step 3A: Window operator design + restricted size set
    Step 3B: Integration with engine_v2
    Step 3C: Preprocessing benchmark

Phase 4: FSC-Driven Resolution Loop

Phase 5: Two-Pass Adaptive Oversampling
    (crop schedule is part of this, not separate)

Phase 6: Full Integration + Convergence Testing
```

---

## Phase 0: Convention Lock and RELION Reference

### Step 0A: Internal RECOVAR Convention Lock

**Rationale**: Before comparing against RELION, prove that RECOVAR's own internal representations are self-consistent. Otherwise you will burn time chasing RELION mismatches that are actually RECOVAR representation bugs.

**Deliverable**: A test module `tests/unit/test_convention_lock.py` that freezes:

1. **Packed-half column ordering**: Verify that packed column 0 is DC, packed column -1 is Nyquist (for even W). Write a test that creates a known signal, packs it, and checks specific bin positions.

2. **Default `max_r` semantics**: Document that `_default_max_r` excludes the Nyquist edge by default. `average_over_shells` indexes up to `volume_shape[0] // 2 - 1`. Lock this with a test.

3. **FFT normalization**: Document `DEFAULT_FFT_NORM` and verify forward-inverse roundtrip.

4. **Nyquist inclusion**: RELION clips at `N//2`. RECOVAR excludes it. Document this difference explicitly.

5. **Full-vs-half path agreement**: For random Hermitian 2D data, verify:
   - `Re<a, b>_full == Re[sum(conj(a_half) * half_weights * b_half)]` to machine precision
   - `full_image_to_half_image` then `half_image_to_full_image` roundtrips correctly
   - Forward projection at full resolution converted to half == forward projection with `half_image=True`

**Agent instructions**: Run on Della via Slurm. Do not modify any production code in this step. Only add test files.

### Step 0B: RELION Reference Harness

**Dataset**: Existing synthetic dataset (5000 images, 128px, noise_level=1.0) at `/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/`.

**RELION run**:
```bash
mpirun -n 3 relion_refine_mpi \
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

**Extraction script** (`scripts/extract_relion_reference.py`): Parse RELION STAR files per iteration and save:

- `rlnCurrentResolution` (Angstrom) and `rlnCurrentImageSize` (pixels) -- use these names exactly
- Per-shell `rlnSigma2Noise`
- Per-shell `rlnSignalPrior` / `rlnReferenceSigma2`
- Per-image Euler angles (`rlnAngleRot/Tilt/Psi`) and origins (`rlnOriginXAngst/YAngst`)
- Reconstructed half-map MRC volumes
- FSC between half-maps
- `_rlnNrOfSignificantSamples` per image (this is significant SAMPLES, not just orientations)

**Assumption documented**: Single optics group. If RELION's STAR files contain multiple optics groups, assert there is exactly one and fail loudly otherwise.

**Comparison framework** (`tests/integration/test_relion_comparison.py`):

```python
def compare_hard_assignments(ours, relion, rotation_grid, tolerance_deg=7.5):
    """Compare best rotations. Tolerance = healpix_order 3 angular step (7.5 deg),
    NOT 15 deg -- that would hide entire neighboring-cell mismatches."""

def compare_volumes(ours, relion, volume_shape):
    """FSC between our reconstruction and RELION's."""

def compare_resolution_trajectory(ours, relion):
    """Compare rlnCurrentImageSize at each iteration."""

def compare_noise_spectra(ours, relion):
    """Per-shell noise variance comparison."""

def compare_prior_spectra(ours, relion):
    """Per-shell signal prior comparison."""
```

**First checkpoint is deliberately small**: Compare only `rlnCurrentImageSize`, `rlnCurrentResolution`, FSC, and `_rlnNrOfSignificantSamples`. Delay per-image Euler-angle parity until Phase 3+.

**Intermediate map comparisons**: Compare unregularized half-maps, regularized half-maps, AND merged/post-processed maps separately. Comparing only final maps confounds E-step/M-step errors with reconstruction/post-processing differences.

---

## Phase 1: Half-Spectrum GEMMs

### What Changes

Modify `engine_v2.py` to perform E-step and M-step GEMMs on the rfft-packed half-spectrum: `N_half = H * (W//2 + 1) = 8320` instead of `N = H * W = 16384`.

### Mathematical Justification

For real-valued images, `f(-k) = conj(f(k))`. The full inner product is recovered from the half:

```
Re<a, b>_full = Re[sum_half w(k) * conj(a(k)) * b(k)]
```

where `w(k) = 1` for DC and Nyquist columns, `w(k) = 2` for all others.

### Half-Weights (CORRECTED)

```python
def make_half_image_weights(image_shape):
    H, W = image_shape
    w = 2.0 * jnp.ones((H, W // 2 + 1), dtype=jnp.float32)
    w = w.at[:, 0].set(1.0)    # packed column 0 = DC (CORRECTED from original)
    w = w.at[:, -1].set(1.0)   # packed column -1 = Nyquist (CORRECTED from original)
    return w.reshape(-1)        # (N_half,)
```

**Note**: The numeric values are the same as before (both endpoints get weight 1), but the *comments* are now correct per C3. This matters for any future endpoint-specific masking.

### Changes to engine_v2.py

The key insight: `engine_v2.py` already has the blockwise E-step/M-step structure with rotation padding for fixed shapes. Modify the GEMM kernels inside `_e_step_block_scores` and `_m_step_block` to operate on half-spectrum data.

**Pattern for all GEMMs**: Convert to half AFTER phase shifting (since `batch_trans_translate_images` assumes full spectrum), but BEFORE the GEMM. Precompute `proj_half_weighted = proj_half * half_weights` once at iteration start.

**What NOT to change**:
- `batch_trans_translate_images` still operates on full spectrum (phase shifts need all frequencies)
- `adjoint_slice_volume` already accepts `half_image=True`
- `slice_volume` already supports `half_image=True` output

### Tests

1. **test_half_inner_product**: Random Hermitian arrays, verify `Re<a,b>_full == Re[sum(conj(a_half) * w * b_half)]` to machine precision.

2. **test_e_step_half_matches_full**: Both paths on same inputs, probability outputs match within `atol=1e-5`.

3. **test_m_step_half_matches_full**: `Ft_y` and `Ft_ctf` match within `atol=1e-6`.

4. **test_full_iteration_half_matches**: One complete EM iteration, hard assignments match, mean volumes match within `atol=1e-4`.

### Expected Outcome

E-step: 72ms -> ~35ms (2.07x). M-step: 93ms -> ~62ms (1.51x). Total iteration: ~19s (from 33s).

---

## Phase 2: Prior/Noise Alignment with RELION

**Rationale**: Moved AHEAD of Fourier cropping. If prior and noise estimation differ from RELION, then every subsequent "current_size trajectory" comparison is comparing against a moving regularization model. Fix the statistics before optimizing the geometry.

### Step 2A: Audit compute_relion_prior

Read `compute_relion_prior` in `reconstruction/regularization.py:75` and `compute_fsc_prior_gpu`. Document exactly what formula is implemented. Compare against RELION's `src/ml_optimiser.cpp` (search for `wsum_signal_product_spectra` and `tau2_class`).

Key questions:
- Does the normal path use `SNR / bottom_avg` or `SNR * noise_level`?
- Does RELION apply smoothing or a floor to tau^2?
- What is `bottom_of_fraction` computing, and does it match RELION?

**Deliverable**: A side-by-side comparison document and a test that feeds identical FSC + noise curves and verifies matching tau^2 output.

### Step 2B: Audit noise estimation

Read `estimate_noise_level_no_masks` in `reconstruction/noise.py:716`. Document:
- It uses hard assignments (not posterior-weighted)
- It operates on `experiment_datasets[0]` only (one half-set)
- It uses `min(1000, n_units)` images
- It translates images by stored translations and subtracts a projected mean

Compare against RELION's noise estimation in `ml_optimiser.cpp` (search for `sigma2_noise`).

**Deliverable**: Test comparing per-shell noise from both codes given identical inputs.

### Step 2C: Wire corrected prior/noise into split_E_M_v2

If discrepancies are found, fix them. If we choose to keep RECOVAR's simpler approximation, document it as a known algorithmic difference and quantify the impact on FSC trajectories.

### Tests

Compare per-shell noise and tau^2 against RELION's `rlnSigma2Noise` and `rlnReferenceSigma2` at each iteration, using the reference harness from Step 0B.

---

## Phase 3: Coordinate-Preserving Fourier Windowing

**This replaces the original "Step 2: Fourier cropping to current resolution."**

### Critical Design Change

Do NOT pass a smaller `image_shape` to `slice_volume`/`adjoint_slice_volume`. Instead, implement a **Fourier window operator** that selects a low-frequency sub-grid from the original-resolution packed half-spectrum while preserving physical frequency spacing.

### Step 3A: Window Operator Design

The missing abstraction: "window the original Fourier grid to the current resolution while keeping the original physical frequency spacing."

**For 2D images in packed-half layout** `(H, W//2+1)`:

```python
def make_fourier_window_mask(image_shape, current_size, volume_shape):
    """Create a boolean mask on the (H, W//2+1) half-spectrum grid
    that selects frequencies within the current resolution shell.

    current_size: diameter in pixels (RELION's rlnCurrentImageSize)
    Returns: 1D boolean mask of shape (N_half,) on the packed layout.
    """
    H, W = image_shape
    r_max = current_size // 2
    # Frequency indices on the ORIGINAL grid (not a resized grid)
    freq_indices = get_frequency_indices(image_shape, volume_shape)
    radii = compute_radial_distances(freq_indices)
    mask = radii <= r_max
    return full_image_to_half_image(mask.reshape(1, -1), image_shape).reshape(-1)
```

**For forward projection**: Project at full resolution with `half_image=True`, then apply the mask. This is slightly more work than a native cropped projection but is correct.

**For adjoint (backprojection)**: Zero out masked frequencies before backprojecting at full resolution. The volume still gets updates only at the included shells.

**For GEMMs**: The mask creates a reduced index set. Extract only the unmasked frequencies from both sides of the GEMM. This gives the same FLOP reduction as actual cropping.

**Alternative (more efficient, second pass)**: Implement a compact representation that physically stores only the unmasked frequencies, with a gather/scatter to convert to/from full layout. This avoids wasted memory bandwidth on zeros.

### Restricted Size Set

Use ONLY sizes that correspond to valid shell radii: `[32, 64, 128]` initially.

Do NOT use the original plan's `[16, 24, 32, 48, 64, 96, 128]`. The sizes 16, 24, 48, 96 are non-divisors of 128 and break RECOVAR's CUDA scaling rule. Expand this set only after validating the basic windowing.

### Step 3B: Integration with engine_v2

Modify `_e_step_block_scores` and `_m_step_block` to accept a frequency mask / index set and perform GEMMs on the reduced dimension.

**JIT recompilation management**: Each unique mask shape is a new static argument. With 3 allowed sizes, this means at most 3 compilations per JIT'd function. This is manageable. The original plan's 7 sizes would produce 7 compilations * multiple kernels per size = significant compile overhead.

**Translation handling** (CORRECTED):

Option (a) "shift before crop" is NOT acceptable as a permanent solution. At small `current_size`, the full-resolution shift materialization becomes first-order cost, erasing most of the GEMM speedup.

Implement option (b) from the start:
```python
def shift_and_window(images_half, translations, image_shape, freq_mask):
    """Phase shifts using original-resolution k vectors,
    then apply frequency window mask."""
    # Phase shifts at original spacing
    shifted = batch_trans_translate_images(images_half, translations, image_shape)
    # Window: extract only unmasked frequencies
    return shifted[..., freq_mask]
```

### Step 3C: Preprocessing Benchmark

**Mandatory**: Add a dedicated benchmark that isolates `process_fn + CTF weighting + translation materialization + window extraction` separately from the GEMM. Given `engine_v2`'s layout, preprocessing can become dominant after windowing reduces the GEMM dimension.

Benchmark at each allowed current_size and report the preprocessing-to-GEMM ratio.

### Tests

1. **test_window_roundtrip**: Window then pad with zeros returns original up to windowed frequencies.

2. **test_projection_windowed_matches_full**: `window(slice_volume(vol, rots, full_shape))` matches `slice_volume(vol, rots, full_shape)[..., mask]`.

3. **test_adjoint_dot_product_windowed**: `<Ax, y> = <x, A*y>` for the windowed operator.

4. **test_phase_shifts_at_windowed_resolution**: Verify `shift_and_window` matches `shift_full_then_window`.

5. **test_iteration_with_oracle_current_size**: Run one iteration at each of RELION's `rlnCurrentImageSize` values. Compare hard assignments and `Ft_y` against RELION reference.

### Expected Outcome

At `current_size=32` (early iterations): GEMM inner dim drops from 8320 to ~528. ~15x fewer FLOPs. At `current_size=64`: ~4x fewer. At `current_size=128`: no change.

---

## Phase 4: FSC-Driven Resolution Loop

### What Changes

Wire FSC -> `current_size` -> frequency window into the iteration loop.

### Implementation

After the M-step solve, compute FSC between half-map volumes:
```python
fsc = get_fsc_gpu(mean_half1, mean_half2, volume_shape)
max_res_shell = find_fsc_threshold(fsc, threshold=0.143)
next_current_size = 2 * max_res_shell  # Nyquist
next_current_size = quantize(next_current_size, ALLOWED_SIZES)
```

Existing code: `regularization.get_fsc_gpu` and `locres.find_fsc_resol`.

### Oracle Mode

Before trusting our own FSC -> current_size mapping, support reading RELION's `rlnCurrentImageSize` from the reference data:

```python
def run_em_iteration(..., current_size_override=None):
    if current_size_override is not None:
        cs = current_size_override
    else:
        cs = compute_current_size_from_fsc(...)
```

This isolates compute optimization from the statistical model.

### Tests

Compare `current_size` trajectory against RELION's. Must match within +/-1 quantized step.

---

## Phase 5: Two-Pass Adaptive Oversampling

**Critical note from critique**: RELION couples image cropping with the two-pass structure. The coarse pass uses smaller images (`image_coarse_size`), the fine pass uses larger images (`image_current_size`). These are not independent optimizations. Implement them together.

### Pass 1: Coarse Evaluation

Evaluate ALL rotations at base angular sampling using the **coarse** frequency window (smaller `current_size`). This is a dense GEMM, exactly our current approach but at reduced resolution.

### Significance Pruning

After pass 1, identify significant orientations per image. RELION uses `adaptive_fraction = 0.999`: keep assignments contributing to the top 99.9% of total posterior weight.

```python
def find_significant_mask(weights, adaptive_fraction=0.999):
    sorted_w = jnp.sort(weights, axis=-1)[:, ::-1]
    cumsum = jnp.cumsum(sorted_w, axis=-1) / weights.sum(axis=-1, keepdims=True)
    threshold_idx = jnp.argmax(cumsum >= adaptive_fraction, axis=-1)
    threshold_val = sorted_w[jnp.arange(len(sorted_w)), threshold_idx]
    return weights >= threshold_val[:, None]
```

**Cap**: Limit to `maximum_significants` (default 500) per image. This caps SAMPLES (orientations x translations), matching RELION's `--maxsig` semantics (not just orientations as the original plan stated).

### Pass 2: Fine Evaluation

For each image, evaluate only its significant coarse orientations at oversampled angles (children from `get_healpix_children`) using the **fine** frequency window (larger `current_size`).

**Implementation**: Use FFT cross-correlation for pass 2. At small cropped resolution, each FFT is tiny. For 200 images * 500 candidate rotations at 32x32: ~100K FFTs of 32x32, about ~1ms total. `engine_v2.py` already contains an FFT-based path. Reuse it rather than inventing a second FFT engine.

**Translation oversampling**: If coarse step is 1 pixel, oversampled step is 0.5 pixel within +/-1 pixel of coarse best. Use `get_oversampled_translation_grid`.

### Tests

1. **test_significance_mask**: Verify mask keeps correct fraction of weight.
2. **test_oversampled_grid**: Verify child rotation generation.
3. **test_two_pass_vs_dense**: On small problem, verify pass 1 + pass 2 gives same posteriors as single dense pass at oversampled resolution.
4. **test_significant_counts_vs_relion**: Compare `_rlnNrOfSignificantSamples` (not just orientation counts) against RELION reference.

---

## Phase 6: Full Integration

### New Top-Level API

Only create this AFTER all representation issues are settled:

```python
def refine_single_volume(dataset, init_volume, init_resolution, ...):
    current_size = resolution_to_size(init_resolution)
    state_half1 = EMState(init_volume, ...)
    state_half2 = EMState(init_volume, ...)

    for iteration in range(max_iter):
        cs_coarse, cs_fine = quantize_pair(current_size)

        for state, half_dataset in [(state_half1, data1), (state_half2, data2)]:
            # Pass 1: coarse angular + coarse Fourier
            weights = coarse_e_step(state, half_dataset, cs_coarse)
            sig_mask = find_significant_mask(weights)

            # Pass 2: fine angular + fine Fourier (only significant)
            run_fine_pass(state, half_dataset, sig_mask, cs_fine)

        # FSC -> resolution -> next current_size
        fsc = get_fsc_gpu(state_half1.mean, state_half2.mean)
        current_size = fsc_to_current_size(fsc)

        # Update noise and prior
        noise = estimate_noise(...)
        prior = compute_relion_prior(fsc, noise, ...)

        if converged(angular_change, no_gain_count):
            break

    return mean, fsc, assignments
```

### Final Comparison

Run both on synthetic data for 25 iterations. Compare:
- Final resolution (FSC = 0.143): match within 1 Fourier shell
- Final volume: FSC > 0.99 at all shells below resolution
- Wall-clock time: <= RELION on same GPU
- `current_size` sequence: match within +/-1 step
- `_rlnNrOfSignificantSamples`: match distribution

---

## Risk Register (Corrected)

| Risk | Severity | Original | Mitigation |
|------|----------|----------|------------|
| Fourier cropping geometry mismatch (C1) | **CRITICAL** | Medium | Frequency-window approach, NOT smaller `image_shape`. Restricted size set `[32, 64, 128]` only. |
| Nyquist/shell-index conventions (C3) | **HIGH** | Low | Convention lock tests in Phase 0A before any optimization work. |
| Prior/noise formula mismatch (C2, C4) | **HIGH** | Medium | Phase 2 moved ahead of cropping. Audit actual code paths, not simplified formulas. |
| JIT recompilation | **HIGH** | Medium | 3 allowed sizes (not 7). Each size produces multiple compiled kernels (E-step, M-step, block variants). Budget ~90s total compile. |
| Preprocessing dominates after windowing | **HIGH** | Not assessed | Mandatory benchmark in Phase 3C. Implement crop-aware shifts from the start. |
| Sparse pass 2 (translations + orientations) | **HIGH** | Low | RELION couples finer angular with finer translational sampling in pass 2. FFT approach handles this naturally. |
| RELION significant-count comparison | **MEDIUM-HIGH** | Low | Need exact `_rlnNrOfSignificantSamples` from STAR files, not inferred values. Separate "count parity" from "candidate-set parity". |
| Half-image weights wrong | HIGH | HIGH | Convention lock + inner product test (no change, still critical). |
| Phase shifts at windowed resolution | HIGH | HIGH | Implement crop-aware shifts directly, not shift-before-crop. |
| Memory: reduced data + prob tensor | LOW | LOW | Windowing massively reduces memory pressure (no change). |

---

## Code Inventory (Corrected)

| Need | Existing code | Status |
|------|--------------|--------|
| Half-spectrum conversion | `full_image_to_half_image` in `fourier_transform_utils.py` | Ready |
| Half-image forward slice | `slice_volume(..., half_image=True)` in `slicing.py` | Ready |
| Half-image adjoint slice | `adjoint_slice_volume(..., half_image=True)` in `slicing.py` | Ready |
| Fourier volume downsample | `downsample_vol_by_fourier_truncation` in `helpers.py` | Reference only (do NOT use for image cropping) |
| FSC | `get_fsc_gpu` in `regularization.py` | Ready |
| FSC -> resolution | `find_fsc_resol` in `locres.py` | Ready |
| Signal prior from FSC | `compute_relion_prior` in `regularization.py` | **AUDIT REQUIRED** (Phase 2) |
| Noise estimation | `estimate_noise_level_no_masks` in `noise.py` | **AUDIT REQUIRED** (Phase 2) |
| RELION Wiener solve | `post_process_from_filter` in `relion_functions.py` | Ready |
| Split half-set EM | `split_E_M_v2` in `iterations.py` | Adapt for windowing |
| HEALPix children | `get_healpix_children` in `sampling.py` | Ready |
| Oversampled grids | `sampling.py` | Ready |
| Phase shifts | `batch_trans_translate_images` | Ready (full spectrum) |
| FFT cross-correlation | `crosscorr_from_ft` in `core.py` | Ready (reuse for pass 2) |
| Blockwise E-step | `_e_step_block_scores` in `engine_v2.py` | **MODIFY** for half + window |
| Blockwise M-step | `_m_step_block` in `engine_v2.py` | **MODIFY** for half + window |

**New code needed**:
- Frequency window mask generator (~50 lines)
- Crop-aware phase shift function (~30 lines)
- Significance pruning (~40 lines)
- RELION reference extraction script (~200 lines)
- Convention lock tests (~150 lines)
- Integration tests (~300 lines)
- Total: ~800 lines

---

## Agent Task Decomposition

Each phase maps to a Claude Code agent task. Agents should follow these rules:

1. **Always run `pixi run test-fast` before pushing** (2454 tests).
2. **Run `./scripts/run_tests_parallel.sh long-test` via Slurm before any PR.**
3. **NEVER widen test tolerances or modify baselines without explicit approval.**
4. **NEVER modify `heterogeneity.py`.**
5. **All GPU work via Slurm** for real jobs; login GPUs for quick benchmarks only.
6. **Each phase gets its own branch** (`claude/em-relion-phase-N`), validated against RELION reference before merging.

### Agent Task 0A: Convention Lock
- **Input**: Current codebase on `claude/dense-em-refactor`
- **Output**: `tests/unit/test_convention_lock.py` with 5+ tests
- **Validation**: All tests pass, no production code modified

### Agent Task 0B: RELION Reference
- **Input**: Synthetic dataset path, RELION binary
- **Output**: `scripts/extract_relion_reference.py`, `relion_reference/` directory, `tests/integration/test_relion_comparison.py`
- **Validation**: Reference data extracted, comparison helpers work on dummy data

### Agent Task 1: Half-Spectrum in engine_v2
- **Input**: Phase 0 complete
- **Output**: Modified `engine_v2.py`, new tests
- **Validation**: All existing tests pass, new equivalence tests pass, 1.7x measured speedup

### Agent Task 2: Prior/Noise Audit
- **Input**: Phase 0 complete (needs RELION reference)
- **Output**: Audit document, corrected code if needed, comparison tests
- **Validation**: Per-shell noise and prior match RELION within documented tolerance

### Agent Task 3: Fourier Windowing
- **Input**: Phases 1 and 2 complete
- **Output**: Window operator, modified engine_v2, preprocessing benchmark
- **Validation**: Adjoint dot-product test passes, oracle-current_size iterations match RELION, preprocessing benchmark shows GEMM is still dominant at each allowed size

### Agent Task 4: FSC Loop
- **Input**: Phase 3 complete
- **Output**: `current_size` wired into iteration loop
- **Validation**: Trajectory matches RELION within +/-1 step

### Agent Task 5: Two-Pass Adaptive
- **Input**: Phases 3+4 complete
- **Output**: Coarse/fine pass structure, significance pruning
- **Validation**: Significant sample counts match RELION, convergence matches

### Agent Task 6: Integration
- **Input**: All previous phases complete
- **Output**: `refine_single_volume()` API, full 25-iteration run
- **Validation**: Resolution, volume FSC, wall-clock time all meet targets

---

## Performance Targets

| Configuration | Current | After Phase 1 | After Phase 3 (cs=32) | After Phase 5 |
|---|---|---|---|---|
| E-step GEMM dim | 16384 | 8320 | ~528 | ~528 (pass 1) + sparse (pass 2) |
| Iteration time (5K img, order 3) | 33s | ~19s | ~2-3s | ~1-2s |
| RELION equivalent | 163s | 163s | ~45s (early) | ~45s (early) |

**Wall-clock target**: Match or beat RELION at every point in the convergence trajectory.
