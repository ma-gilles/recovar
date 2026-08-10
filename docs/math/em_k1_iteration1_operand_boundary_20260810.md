# K=1 iteration-1 first-operand boundary (2026-08-10)

## Frozen acceptance status

- [x] Fixed matrix remains 34 evaluated cases.
- [x] Strict signed shellwise FSC/FSC-AUC: 28/34 (82.4%).
- [x] Exact topology: 32/34.
- [ ] No fixed-scorecard case is promoted by this diagnostic.
- [x] `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` remained unset.
- [x] `RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER` remained unset.
- [x] Correlation was not used for acceptance or localization.

The purpose of this panel is to find the first unequal numerical operand, not
to infer improvement from a complete trajectory.  It uses three fixed case-22
particles (stack indices 252, 591, and 2124) whose iteration-1 class, pose,
translation, support, and posterior are identical between RELION and RECOVAR.
Each has one accepted hypothesis with posterior exactly 1.

## First unequal boundary

The first unequal boundary is the per-particle BPref operand, before
inter-particle accumulation.  The control relative-L2 ranges are:

| Operand | stack 252 | stack 591 | stack 2124 |
|---|---:|---:|---:|
| BPref numerator | 3.35825e-7 | 2.57363e-7 | 2.97589e-7 |
| BPref denominator | 3.81888e-7 | 1.86090e-7 | 2.40510e-7 |

The authoritative primitive report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/analysis/CASE22_IT1_BPREF_PRIMITIVE_BOUNDARY_V9.json`.
Its complete per-pixel arrays are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/analysis/it1_bpref_primitive_dumps_v9/`.

## Demonstrated causes

### CTF

- [x] RECOVAR's JAX CTF differs from RELION at 842--932 of 1,227 pixels
  (relative L2 8.64e-8--1.20e-7).
- [x] RELION's C++ CTF evaluator, fed the original STAR decimal values,
  reproduces all captured CTF pixels bit-for-bit for all three particles.
- [x] Feeding that evaluator RECOVAR's float32 CTF metadata leaves 4--6
  mismatched pixels, demonstrating a second, smaller premature-metadata-cast
  error.

### Initial inverse noise

- [x] RECOVAR rounds the fresh live radial noise spectrum to float32 before
  reconstructing its reciprocal; 305 of 1,227 active pixels then differ.
- [x] Computing `float32(1 / sigma2_float64)` reproduces every captured RELION
  inverse-noise pixel bit-for-bit.
- [x] Once the spectrum has been rounded to float32, algebraic rearrangement
  cannot recover the missing reciprocal values.

The exact float64 spectrum is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_first_boundary_20260810T0945ET/analysis/live_initial_sigma2_recomputed_f64.npy`.

### Fourier image and translation

- [x] The control's host NumPy FFT differs from RELION at 1,180--1,196 of
  1,227 pixels (relative L2 1.22e-7--1.39e-7).
- [x] JAX/cuFFT on the identical float32 real image reproduces every captured
  RELION Fourier pixel bit-for-bit for all three particles.
- [x] RECOVAR's separate JAX phase-table path differs from RELION's fused CUDA
  `sincosf` BPref translation path (relative L2 1.02e-7--1.44e-7).
- [x] Replaying RELION's fused translation arithmetic reduces the translated
  image residual to 2.63e-8--6.39e-8.  Rebuilding the replay as CUDA-12.6
  compute-80 PTX, matching the deployed RELION build mode, gives the same
  result; the residual is not a CUDA-version or target-code explanation.

### Joint substitution

- [x] Source-precision CTF plus float64-before-reciprocal noise reproduces the
  RELION weighted-CTF operand bit-for-bit.
- [x] Adding cuFFT and the fused `sincosf` translation reduces numerator
  relative L2 from 2.57e-7--3.36e-7 to 3.33e-8--9.24e-8.
- [x] The remaining denominator residual is 4.81e-8--5.17e-8 and is reproduced
  by RELION's own scalar multiplication/cast ordering.

## Live one-iteration A/B

Slurm job `12210268` ran the same iteration-1 boundary with only
`--image-fourier-backend relion_cuda` changed.  Wall time was 83 seconds on an
H100.  The raw Fourier operand becomes exact, but the combined numerator
relative-L2 changes by -5.36%, +2.13%, and +1.92% for stacks 252, 591, and
2124; denominator error is unchanged.  The isolated backend switch is
therefore rejected as a production fix because the remaining operand errors
partially cancel it.

The A/B report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_cuda_fft_ab_20260810T1029ET/analysis/CASE22_IT1_CUDA_FFT_BPREF_AB.json`.

## Next acceptance gate

The next experiment must be another one-iteration case-22 A/B, not a full
trajectory.  One guarded arm must change the four coupled operands together:

1. preserve source CTF precision and evaluate with RELION-compatible scalar
   arithmetic;
2. preserve the float64 radial sigma2 and materialize its float32 reciprocal
   before any float32 variance cast;
3. use cuFFT for the unmasked reconstruction image;
4. fuse translation and weighted-CTF multiplication using the RELION BPref
   CUDA arithmetic.

The arm passes the local gate only if it improves all three fixed particles at
the numerator and denominator boundaries without changing their tuple set,
posterior, support, winner, or topology.  Only then is a short multi-iteration
trajectory justified.  K=4 remains out of scope until the K=1 fixed scorecard
closes.
