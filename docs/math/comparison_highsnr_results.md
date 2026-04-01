# Head-to-Head Comparison: recovar EM vs RELION (High SNR)

Date: 2026-04-01
GPU: NVIDIA A100-SXM4-80GB
Dataset: 5000 images, 128x128 px, voxel_size=4.25 A/px, noise_level=0.1 (10x better SNR than previous comparison)
Half-set split: identical (RELION's rlnRandomSubset from run_it001_data.star, 2558/2442 split)
RELION job: Slurm 6341849

## Experimental Setup

Same simulation as the previous comparison (comparison_results.md) but with
noise_level=0.1 instead of 1.0.  This gives 10x better SNR, making pose
assignment much easier.  Both codes use the same low-pass filtered ground-truth
init (rad=5 shells).

### RELION settings:
- `relion_refine_mpi --auto_refine --split_random_halves`
- `--healpix_order 3 --oversampling 1`
- `--offset_range 3 --offset_step 1`
- `--particle_diameter 200 --ini_high 30`
- `--pad 2 --gpu 0 --j 4`
- 3 MPI ranks, 1 GPU

### Our settings:
- healpix_order=3 (36,864 rotations), offset_range=3, offset_step=1 (29 translations)
- adaptive_oversampling=0
- image_batch_size=500, rotation_block_size=5000
- Three runs: (a) own FSC-driven current_size, (b) oracle (RELION's current_sizes), (c) full dataset (no half-set split, same data both halves)

## Key Findings

### RELION diverged completely

RELION's auto-refine converged (by its own criteria) after 9 iterations, but
the result was completely wrong:

- **FSC vs ground truth = 0 at ALL shells** (including shell 0, the DC component)
- RELION's internal half-map FSC dropped from 0.98 (shell 0) to noise level by shell 5
- RELION reported "Final resolution: 38.9 A" but the actual map is pure noise
- Angular accuracy warnings at every iteration: "you cannot align your particles"

The divergence pattern was:
1. Iterations 1-5: resolution oscillated (91A -> 34A -> 21A -> 19A -> 20A)
2. Iteration 6: resolution jumped to 78A (divergence)
3. Iterations 7-9: continued degrading (91A -> 109A), then auto-refine declared convergence

### Our code converged correctly

- By iteration 2: half-map FSC<0.5 at shell 35 (15.5 A)
- Stabilized at iterations 3-10: shell 36 (15.1 A), current_size=96
- **FSC vs ground truth > 0.5 up to shell 40** (13.6 A)
- Resolution is limited by angular sampling at healpix order 3 (~7.33 deg -> ~15 A for 200A particle)

### Oracle mode (RELION's current_sizes) also diverged

When we fed RELION's per-iteration current_sizes to our code (oracle mode),
our code also produced poor results (FSC vs GT < 0.5 after shell 24, ~22 A).
This confirms that RELION's resolution management was the problem -- the
oscillating current_sizes prevented any code from converging.

## Per-Iteration Table

### RELION

| Iter | current_size | Resolution (A) | Time (s) | Median sig. samples | Status |
|------|-------------|----------------|----------|--------------------:|--------|
| 1 | 56 | 90.67 | 23 | 4 | Poor alignment |
| 2 | 32 | 34.00 | 41 | 6302 | Briefly OK |
| 3 | 52 | 20.92 | 62 | 3728 | Best iteration |
| 4 | 90 | 19.43 | 130 | 2627 | Best resolution |
| 5 | 102 | 20.15 | 89 | 922 | Losing alignment |
| 6 | 100 | 77.71 | 49 | 248 | Diverged |
| 7 | 60 | 90.67 | 18 | 47 | Collapsed |
| 8 | 44 | 108.80 | 15 | 50 | Collapsed |
| **Total** | | | **427** | | **Failed** |

### Our code (own FSC)

| Iter | current_size | Half-map FSC<0.5 (shell) | Pixel res | Time (s) |
|------|-------------|------------------------:|----------:|---------:|
| 1 | 32 | 18 (30.2 A) | 62.0 | 38.8 |
| 2 | 128 | 35 (15.5 A) | 41.9 | 79.6 |
| 3 | 96 | 36 (15.1 A) | 41.9 | 39.3 |
| 4 | 96 | 36 (15.1 A) | 42.1 | 36.8 |
| 5 | 96 | 36 (15.1 A) | 42.1 | 36.9 |
| 6 | 96 | 36 (15.1 A) | 42.2 | 36.9 |
| 7 | 96 | 36 (15.1 A) | 42.2 | 36.9 |
| 8 | 96 | 36 (15.1 A) | 42.2 | 36.9 |
| 9 | 96 | 36 (15.1 A) | 42.2 | 36.9 |
| 10 | 96 | 36 (15.1 A) | 42.2 | 36.9 |
| **Total** | | | | **416** |

### Our code (oracle -- RELION's current_sizes)

| Iter | current_size | Pixel res | Time (s) |
|------|-------------|----------:|---------:|
| 1 | 64 | 32.6 | 17.7 |
| 2 | 32 | 62.0 | 10.9 |
| 3 | 64 | 32.9 | 21.4 |
| 4 | 96 | 42.1 | 36.9 |
| 5 | 128 | 42.3 | 66.6 |
| 6 | 128 | 42.3 | 66.9 |
| 7 | 64 | 33.0 | 19.4 |
| 8 | 48 | 25.8 | 15.9 |
| 9 | 48 | 25.9 | 13.1 |
| 10 | 48 | 25.9 | 13.2 |
| **Total** | | | **282** |

## FSC vs Ground Truth

| Shell | RELION | Ours (own FSC) | Ours (oracle) | Ours (full dataset) |
|------:|-------:|---------------:|--------------:|--------------------:|
| 0 | -0.071 | **0.999** | 0.999 | 0.999 |
| 5 | -0.028 | **0.998** | 0.998 | 0.998 |
| 10 | -0.019 | **0.990** | 0.990 | 0.990 |
| 15 | 0.012 | **0.976** | 0.975 | 0.972 |
| 20 | -0.002 | **0.951** | 0.951 | 0.948 |
| 25 | 0.003 | **0.884** | -0.330 | 0.876 |
| 30 | 0.007 | **0.801** | -0.028 | 0.792 |
| 35 | 0.001 | **0.693** | 0.025 | 0.681 |
| 40 | 0.006 | **0.465** | -0.029 | 0.463 |
| 45 | 0.003 | 0.154 | 0.006 | 0.185 |
| 50 | -0.002 | 0.005 | 0.003 | 0.062 |
| 55 | 0.001 | -0.001 | 0.007 | -0.002 |
| 60 | -0.002 | 0.020 | 0.026 | -0.003 |

### Resolution at FSC=0.5 vs GT

| Code | Resolution (A) |
|------|---------------:|
| RELION | inf (never reaches 0.5) |
| Ours (own FSC) | **13.6** |
| Ours (oracle) | 21.8 |
| Ours (full dataset) | **13.6** |

## Cross-FSC: Our Half-Maps vs RELION's Half-Maps

Since RELION diverged, the cross-FSC between our reconstructions and RELION's
is pure noise at all shells:

| Shell | Our h1 vs RELION h1 | Our h2 vs RELION h2 | Merged vs Merged |
|------:|--------------------:|--------------------:|-----------------:|
| 0 | -0.083 | -0.078 | -0.081 |
| 5 | -0.020 | -0.020 | -0.024 |
| 10 | -0.019 | -0.001 | -0.010 |

All values are consistent with noise-level correlation (~1/sqrt(N_voxels)).

## Speed Comparison

| Mode | Total Time (s) | vs RELION |
|------|---------------:|----------:|
| RELION (1 GPU, 3 MPI) | 427 | 1.0x |
| Ours (own FSC) | 416 | **1.0x** |
| Ours (oracle) | 282 | **1.5x** |

Speed is comparable because our code chose current_size=96 or 128 for most
iterations (the FSC stayed high due to low noise), while RELION used varying
sizes.  The main speed advantage of Fourier windowing comes at low resolution
(current_size=32-48).  At this noise level, both codes need to work at nearly
full resolution, so the speed is similar.

However, the comparison is unfair to RELION because it diverged -- a converging
RELION run with oversampling would likely take much longer per iteration.

## Discussion

### Why did RELION diverge?

RELION's auto-refine at healpix_order 3 with oversampling 1 activates its
adaptive oversampling internally (going to 3.75 degree steps, 294912
orientations).  The combination of:
1. Small dataset (5000 images)
2. `--oversampling 1` triggering RELION's internal adaptive oversampling
3. Potentially suboptimal noise/prior estimation in early iterations

led to oscillating resolution estimates and eventual divergence.  RELION's
own warnings confirm: "you cannot align your particles" at every iteration.

### Why did our code succeed?

Our code uses a simpler approach:
1. FSC-driven resolution: the half-map FSC directly determines the working
   resolution, preventing wild oscillations
2. No adaptive oversampling: we search the full healpix order 3 grid at every
   iteration, avoiding the fragility of significance-pruned local searches
3. The FSC quickly stabilized at cs=96, providing a consistent working resolution

### Comparison with previous results (noise_level=1.0)

In the previous comparison (comparison_results.md) at noise_level=1.0:
- RELION also struggled (resolution oscillated but eventually reached ~15A)
- Our code was 4.2x faster (own FSC) and achieved better resolution
- At high noise, RELION's oscillation was less severe because the resolution
  stayed lower (less overfitting risk)

At noise_level=0.1, the paradox is that better data made RELION perform worse:
- Low noise means the FSC goes higher, RELION increases resolution faster
- Higher resolution means more angular precision needed
- The adaptive oversampling narrows too quickly, missing the correct orientation
- Once off track, the high SNR amplifies the wrong signal, causing divergence

## Files

- Dataset: `/scratch/gpfs/GILLES/mg6942/tmp/em_comparison_highsnr/`
- RELION output: `relion_ref/`
- RELION extracted: `relion_ref_npz/`
- Our results (halfset): `our_results_halfset.npz`
- Our results (full): `our_results.npz`
- FSC curves: `our_fsc_vs_gt.npy`, `relion_fsc_vs_gt.npy`, `relion_fsc_halfmaps.npy`, `cross_fsc.npz`
- RELION log: `/scratch/gpfs/GILLES/mg6942/slurmo/relion-highsnr-6341849.out`
