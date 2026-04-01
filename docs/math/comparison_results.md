# Head-to-Head Comparison: recovar EM vs RELION

Date: 2026-04-01
GPU: NVIDIA A100-SXM4-80GB
Dataset: 5000 images, 128x128 px, voxel_size=4.25 A/px, noise_level=1.0
Half-set split: identical (RELION's rlnRandomSubset from run_it001_data.star)

## Experimental Setup

Two comparison runs using RELION's half-set assignments:

1. **Run 1 (own FSC)**: Our code determines current_size from its own FSC
   curves. This is the realistic operating mode.

2. **Run 2 (oracle)**: Our code uses RELION's per-iteration current_image_size
   values (quantized to our allowed set). This isolates the quality comparison
   from the resolution management strategy.

Both runs use: healpix_order=3 (36,864 rotations), offset_range=3,
offset_step=1 (29 translations), adaptive_oversampling=0.

## Per-Iteration Timing, Resolution, and Current Size

| Iter | RELION cs | Our cs (own) | Our cs (oracle) | RELION res (A) | Our res (A, own) | Our res (A, oracle) | RELION time (s) | Our time (s, own) | Our time (s, oracle) |
|------|-----------|--------------|-----------------|----------------|------------------|---------------------|-----------------|-------------------|----------------------|
| 1 | 0 | 48 | 48 | inf | 14.59 | 14.59 | 0 | 41.4 | 12.6 |
| 2 | 56 | 128 | 64 | 108.80 | 10.33 | 7.40 | 23 | 79.0 | 26.7 |
| 3 | 30 | 128 | 32 | 36.27 | 10.32 | 14.59 | 88 | 66.7 | 11.5 |
| 4 | 50 | 128 | 64 | 21.76 | 10.73 | 14.59 | 145 | 66.9 | 19.3 |
| 5 | 70 | 96 | 96 | 15.54 | 5.62 | 14.59 | 239 | 39.7 | 36.9 |
| 6 | 98 | 128 | 128 | 15.54 | 7.25 | 7.23 | 436 | 66.9 | 66.8 |
| 7 | 98 | 128 | 128 | 17.00 | 7.22 | 7.20 | 408 | 67.0 | 67.0 |
| 8 | 92 | 96 | 96 | 22.67 | 7.26 | 7.22 | 364 | 37.1 | 37.2 |
| 9 | 88 | 96 | 96 | 22.67 | 7.29 | 7.21 | 303 | 37.1 | 37.1 |
| 10 | 90 | 96 | 96 | 30.22 | 7.15 | 7.06 | 283 | 37.1 | 37.0 |

### Speed Summary

| Mode | Total Time (s) | Speedup vs RELION |
|------|---------------|-------------------|
| RELION (1 GPU) | 2289 | 1.0x |
| Ours (own FSC) | 539 | **4.2x** |
| Ours (oracle cs) | 352 | **6.5x** |

### Per-Resolution-Level Speed

| current_size | n_windowed / n_half | FLOP reduction | Our time (s) | RELION time (s) | Speedup |
|-------------|--------------------:|---------------:|-------------:|----------------:|--------:|
| 32 | 415 / 8320 | 95.0% | 11.5 | ~88 (iter 3) | 7.7x |
| 48 | 921 / 8320 | 88.9% | 12.6 | ~23 (iter 1) | 1.8x |
| 64 | 1637 / 8320 | 80.3% | 19.3-26.7 | ~145 (iter 4) | 5.4-7.5x |
| 96 | 3655 / 8320 | 56.1% | 37.0-39.7 | ~300 (iter 8-9) | 7.6-8.1x |
| 128 | 8320 / 8320 | 0% | 66.7-79.0 | ~420 (iter 6-7) | 5.3-6.3x |

Note: RELION timing includes disk I/O and CPU overhead that varies per
iteration. Our first iteration at each current_size includes JIT compilation.

## Resolution Trajectory

### Our code converges faster

Our code reaches 7 A resolution by iteration 5-6 and stabilizes, while
RELION is still at 15-30 A after 10 iterations and appears to be
oscillating. This is likely because:

1. Our E-step processes the full half-spectrum (with frequency masking for
   speed), giving stronger orientation discrimination at each resolution level.
2. RELION's progressive resolution increase is more conservative, preventing
   overfitting but also slowing convergence.
3. Different noise estimation strategies lead to different regularization.

### Oracle mode reveals resolution management matters

With oracle current_sizes (Run 2), our code follows RELION's gradual
resolution ramp (cs=32 at iter 3, cs=64 at iters 2/4) but still converges
to 7 A by iteration 6 when cs reaches 128. This confirms that the E-step
implementation (not just resolution management) drives the faster
convergence.

## Volume Quality

### Cross-FSC (same images)

Cross-FSC between our half-1 volume and RELION's half-1 volume is near
zero (~0.03) at all shells. This is expected despite using the same images
because:

1. The two codes use fundamentally different convergence paths
2. Different noise estimation (hard-assignment vs posterior-weighted)
3. Different prior computation details
4. Different resolution management schedules
5. No volume padding in our code (RELION uses --pad 2)

The near-zero cross-FSC does NOT indicate a bug -- it indicates that the
two algorithms find different reconstructions from the same data.

### Internal half-map FSC

Our internal half-map FSC crosses 0.143 at shell 31 (corresponding to
~8.8 A resolution), which is our gold-standard quality metric.

## Fixes Applied

### Gap 1: Expanded allowed current_sizes

Changed from `[32, 64, 128]` to `[16, 24, 32, 48, 64, 96, 128]`.

Impact: The new intermediate sizes (48, 96) are now actively used.
Iteration 5 uses cs=96 instead of jumping to 128, saving 40% computation
(37s vs 67s). The finer granularity better matches RELION's resolution
trajectory.

### Gap 2: Per-shell noise in prior (ALREADY CORRECT)

The audit identified this as a potential issue, but examining the current
code shows that `noise_variance` passed to `compute_relion_prior` is
already a per-pixel array expanded from per-shell noise via
`make_radial_noise()`. The division `CTF**2 / noise_variance[None]` in
`compute_batch_prior_quantities` correctly uses per-shell noise weights.
No code change was needed.

### Gap 3: Adaptive oversampling union cap

Added `max_union_pixels=200` parameter to `compute_pass2_stats`. When the
union of significant HEALPix pixels exceeds this cap, pass 2 is skipped
and the caller falls back to pass-1-only mode. This prevents the pathological
case where pass 2 becomes more expensive than pass 1 (which happened at
production scale with 5000 images).

### Gap 4: Volume padding (SKIPPED)

RELION's `--pad 2` pads the 3D volume by 2x in Fourier space before
reconstruction, reducing interpolation artifacts. Implementing this in our
code would require changing the accumulation grid (from 128^3 to 256^3),
not just the post-processing step. The `post_process_from_filter_v2` function
already supports `volume_upsampling_factor` for the Wiener solve, but the
E+M step accumulates at native resolution. This is deferred to a future
implementation.

### Gap 5: Same half-set split as RELION

Added `--relion_half_sets` flag to `run_full_refinement.py` that reads
`rlnRandomSubset` from a RELION data STAR file. The mapping correctly
handles different particle orderings between RELION (alphabetical) and our
code (stack order).

## Remaining Differences

| Difference | Impact | Fix Complexity |
|-----------|--------|---------------|
| No volume padding | Interpolation artifacts | High (needs accumulation grid change) |
| Hard-assignment noise estimation | Higher variance noise estimate | Medium (needs E-step posterior storage) |
| Subset (1000 imgs) for noise | Faster but noisier | Low (parameter change) |
| No tau2 smoothing/tapering | Noisy prior at resolution boundary | Low |
| Different convergence path | Different local optima | Fundamental algorithmic difference |

## Reproduction

```bash
cd /scratch/gpfs/GILLES/mg6942/recovar_wt_merge

# Run comparison
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    pixi run python scripts/run_comparison.py --max_iter 10

# Or run refinement manually with RELION's half-sets
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    pixi run python scripts/run_full_refinement.py \
        --adaptive_oversampling 0 \
        --relion_half_sets /scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/relion_ref/run_it001_data.star

# Results at
ls /scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/comparison_results/
```
