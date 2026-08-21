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

## Exact primitive and live-boundary update

The four coupled operands have now passed that local gate.

### Literal native-topology probe

Slurm jobs `12213234`, `12213499`, and `12213578` progressively matched the
deployed RELION kernel's 128-thread block/pass topology, CUDA 12.6 target, and
compiler flags.  The final report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_bpref_literal_20260810/LITERAL_BPREF_PROBE.json`.
The captured native and replay kernels both use 32 registers, 32 bytes of
stack, 8 bytes of constant bank 2, and 464 bytes of constant bank 0.

For all three fixed particles:

- [x] weighted CTF is bitwise identical;
- [x] the BPref denominator is bitwise identical;
- [x] the first remaining unequal value is the translated complex image;
- [x] none of 16 phase/FMA source variants removes that residual;
- [x] the ordinary compiler source variant has the lowest mismatch count.

The translated-image relative-L2 residual is `2.60e-8--2.97e-8`.  The SASS
captures are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_bpref_literal_20260810/native_capture.sass`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_bpref_literal_20260810/our_capture.sass`.
They show the same `FMUL; FFMA` phase structure and `sincosf` polynomial but
different register scheduling and memory layout.  This is currently treated
as a compiler/code-generation residual, not evidence for another scientific
operand mismatch.

### Live float64-noise boundary

Slurm job `12213753` preserved live `sigma2_noise` in float64 until RELION's
float32 reciprocal boundary.  Its report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_exact_bpref_noise64_20260810T1150ET/analysis/CASE22_IT1_EXACT_BPREF_AB.json`.

| Particle | numerator relative-L2 | baseline removed | denominator |
|---|---:|---:|---:|
| stack 252 | 3.33104e-8 | 90.08% | bitwise exact |
| stack 591 | 3.51305e-8 | 86.35% | bitwise exact |
| stack 2124 | 9.23922e-8 | 68.95% | bitwise exact |

Candidates, posterior, support, and topology did not change.  Stacks 252 and
591 are at the literal kernel residual.  Stack 2124 remains larger because
its selected x translation angle is one float32 ULP above RELION: RECOVAR
`3178343904`, RELION `3178343903`.  The preceding source translation was
prematurely cast to float32; promoting it afterward cannot recover RELION's
angle.

### Current focused intervention

The accepted candidate preserves a separate host-float64 fine-translation grid
through RELION's float32 angle conversion while retaining float32 scoring,
priors, pose IDs, and output translations.  K=4 is explicitly kept on the
existing float32 handoff.  Slurm job `12214131` completed in 94 seconds and
passed the focused three-particle, one-iteration gate:

| Particle | numerator relative-L2 before | after host-float64 grid | denominator after |
|---|---:|---:|---:|
| stack 252 | 3.33104e-8 | 3.33104e-8 | bitwise exact |
| stack 591 | 3.51305e-8 | 3.51305e-8 | bitwise exact |
| stack 2124 | 9.23922e-8 | 3.16324e-8 | bitwise exact |

The accepted report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_exact_translation64_20260810T120255ET/analysis/CASE22_IT1_EXACT_BPREF_AB.json`
with SHA-256
`06a36c0b38a29ad7527ffb26e873c395fac73a11336e6b440df6adfd34d576bc`.
Pose/rotation mapping remained exact.  Stack 2124 now lies in the same literal
compiler residual range as stacks 252 and 591, so the source-translation ULP
error is closed at this boundary.

## Iteration-2 serialized-boundary discriminator

The first attempt to compare iteration-2 fine operands exposed two restart
confounds rather than a scorer result:

- stock continuation job `12215227` changed RELION's one-time physical order,
  producing expected accuracy `19.110 deg / 16.218 A` instead of the
  uninterrupted `16.730 deg / 14.888 A`;
- order-restored job `12215500` recovered `16.750 deg / 14.956 A`, but the
  restart chose sampling perturbation `-0.08248` instead of the uninterrupted
  `+0.409489929676`.

Job `12215737` restored both quantities and completed in 5 minutes 49 seconds.
The deployed diagnostic binary reported the exact requested perturbation and
the serialized `run_it002_sampling.star` recorded `0.409490`.  The factor
capture passed validation at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_restart_same_serialized_20260810T1230ET/retry3_order_perturb_restored/analysis/capture_validation.json`.

With order and perturbation fixed, the fine geometry still differs before raw
fine scoring:

| Particle | RELION coarse-parent count | RECOVAR coarse-parent count | Shared fine rotations |
|---|---:|---:|---:|
| stack 252 | 15 | 15 | 120 / 120 exact |
| stack 591 | 9 | 10 | all 72 RELION rows exact; RECOVAR has 8 extra rows |
| stack 2124 | 19 | 16 | 128 exact shared rows; RELION has 24 extra rows |

All shared rotation matrices are bitwise identical.  The first demonstrated
iteration-2 mismatch is therefore the coarse significant-support / fine-parent
boundary, not fine candidate construction or raw fine scoring.

Slurm job `12216042` completed a three-particle RECOVAR coarse dump in 80
seconds.  It saved all `1,069,056` coarse tuples per particle, including raw
pre-prior score, score after priors, normalized weight, support mask, cutoff
rank, Pmax, images, CTF/noise operands, rotations, and translations under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse_recovar_20260810T1255ET/coarse/`.
The RECOVAR support counts are 29, 18, and 39 tuples for stacks 252, 591, and
2124, respectively, at the fixed `0.999` cumulative-mass boundary.

The next causal gate is an aligned native RELION coarse capture on the same
serialized maps, particle order, perturbation, GPU type, and three particles.
Comparison proceeds in the fixed order: tuple geometry, raw `diff2`, direction
and translation priors, combined log weights, normalization, sorted cutoff,
then support indices.  No trajectory or scorecard run is justified until the
first unequal item in that chain is identified.

## All-particle iteration-2 localization

The three-particle support mismatch did not account for the material aggregate
BPref discrepancy.  Slurm job `12223009` compared all 1,490 half-1 particles
at the iteration-2 boundary.  The maximum per-particle reconstruction-mass
relative-L2 was `7.14e-5`, too small to explain the aggregate denominator
error.  Its report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rotation_mass_compare_20260810T1715ET/analysis/K1_CASE22_IT2_HALF1_ROTATION_MASS_ALL.json`
with SHA-256
`9cfe90b1a8b3eaaa700fc19d873a452de247378c827a001eda405a51616e2a82`.

The next discriminator held every captured particle operand fixed and changed
only the order in which particles entered the float32 BPref accumulator:

| execution arm | half-1 denominator relative-L2, shells 16--30 |
|---|---:|
| ordinary RECOVAR order, normal production batching | `1.89620e-3` |
| ordinary RECOVAR order, one-particle calls | `1.85833e-3` |
| RELION physical order, one-particle calls | `1.72623e-4` |
| independent native-repeat floor | `1.71914e-4` |

The ordinary-order one-particle control rules out microbatch shape as the
explanation.  The RELION-order arm reaches the independent repeat floor, so
particle execution order is causal at the first material K=1 aggregate
boundary.

The H100 reports are:

- ordinary order:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_default_single_order_20260810T1655ET/analysis/K1_CASE22_IT2_HALF1_DEFAULT_SINGLE_ORDER.json`,
  SHA-256
  `dbd590d5b6256005432e55fdb09d02d6ae043afb4be29059d8b280bf1e58e98d`;
- RELION order:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_native_exec_order_20260810T1745ET/analysis/K1_CASE22_IT2_HALF1_NATIVE_EXEC_ORDER.json`,
  SHA-256
  `0e5a69f795e2e32d5bb697bc184fdd27aed291f77dc96d730cb8f9f29b851aa2`.

An independent local A100 order-only A/B reproduced the effect with denominator
relative-L2 movement `2.04707e-3`.  Its report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_execution_order_local_a100_20260810T1640ET/analysis/K1_CASE22_IT2_HALF1_EXECUTION_ORDER_A100_AB.json`
with SHA-256
`3a9c4f2dab08887cca639dd7e1dc4336b7e1d1425709d37afc5815f7f9dcfa26`.

## Guarded fresh-K=1 production route

The candidate now constructs one authoritative fresh AutoRefine order before
half-dataset creation.  It reproduces RELION's paired RNG semantics: one
`srand(seed + iter)`, half 1 shuffle, continued-stream half 2 shuffle, then a
stable numeric optics-group sort within each half.  Expected accuracy consumes
the first 100 rows of the already physical half-1 order directly.  The BPref
path then executes particles in that physical order.  The guard is deliberately
narrow: fresh K=1 only, with continuation, frozen-boundary replay, perturbation
replay, and K>1 left unchanged.

Slurm job `12223784` validated the complete production route through physical
iteration 1 in 2 minutes 43 seconds.  It used all 3,000 case-22 particles and
compared the raw half-1 accumulator against native RELION:

| quantity | relative-L2 |
|---|---:|
| denominator | `6.49466e-7` |
| numerator | `2.76034e-6` |
| average | `2.58089e-6` |

Support Jaccard was `1.0` with zero support mismatches.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fresh_order_production_it1_v2_20260810T1735ET/analysis/K1_CASE22_IT1_FRESH_ORDER_PRODUCTION.json`
with SHA-256
`79d51a527a08d3d3ee4d6793b43b2dab2b5fad42177119e2d6fe1a3ec2ecae69`.

The complete two-iteration launch independently saved the candidate's first
reconstructed maps.  Comparing them to RELION iteration 1 with the repository's
canonical volume transforms (`load_mrc` for RECOVAR and `load_relion_volume`
for RELION) gives:

| half | cross-engine FSC-AUC | map relative-L2 |
|---|---:|---:|
| 1 | `0.999999999947` | `1.04941e-6` |
| 2 | `0.999999999944` | `7.13611e-7` |

This closes the full first reconstruction boundary, not only the selected raw
accumulator shells.  The signed comparison uses the canonical RELION-to-RECOVAR
frame conversion; no acceptance metric was made absolute-valued.

Slurm job `12223827` completed the focused two-iteration production
integration gate in 31 minutes 23 seconds.  It captured and compared the
physical iteration-2 raw accumulator rather than waiting for a complete
refinement trajectory:

| quantity, shells 16--30 | relative-L2 |
|---|---:|
| denominator | `1.72190e-4` |
| numerator | `2.16588e-5` |
| average | `5.32594e-4` |

Support Jaccard was `1.0` with zero support mismatches.  The denominator result
is at the independent native-repeat floor (`1.71914e-4`) and the isolated
RELION-order arm (`1.72623e-4`), versus `1.89620e-3` before the intervention.
Thus the complete guarded production route removes about 91% of the material
denominator discrepancy and closes particle execution order as the cause at
the first divergent EM update.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fresh_order_production_it2_v3_20260810T1740ET/analysis/K1_CASE22_IT2_FRESH_ORDER_PRODUCTION.json`
with SHA-256
`e24ace08f69c35811706925eac66c5dcda691857ad03d9921dd0886257b278ea`.
The run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fresh_order_production_it2_v3_20260810T1740ET/`,
and both the run and runtime roots contain `SAFE_TO_DELETE` markers.

This reference route deliberately uses one-particle calls.  It establishes
causality and correctness but is not the final performance design: physical
iteration 2 spent 1,718 seconds in the variable-support update.  The next
focused gate is a contiguous physical-order chunk-size ablation against this
same saved boundary, followed by the shortest trajectory required to verify
that the controller remains on the RELION branch.  A fixed-suite scorecard
promotion is not permitted until a frozen case is rerun.

## Iteration-2 map closure and execution ablations

The regularized maps produced by job `12223827` close the complete second
reconstruction boundary.  Canonical signed cross-engine comparisons are:

| physical iteration | half | FSC-AUC | map relative-L2 |
|---:|---:|---:|---:|
| 1 | 1 | `0.999999999947` | `1.04941e-6` |
| 1 | 2 | `0.999999999944` | `7.13611e-7` |
| 2 | 1 | `0.999999999744` | `2.20262e-5` |
| 2 | 2 | `0.999999999720` | `2.21180e-5` |

The iteration-2 physical-order reference was then rerun with only execution
chunking changed.  Every successful physical-order chunk arm preserved exact
support and stayed at the native-repeat denominator floor, but none improved
runtime:

| particles per call | job | wall time | denominator relative-L2 | result |
|---:|---:|---:|---:|---|
| 1 | `12223144` | 15m31s | `1.72623e-4` | correctness/performance reference |
| 2 | `12224952` | 16m24s | `1.72660e-4` | parity pass, slower |
| 4 | `12224473` | 16m31s | `1.72710e-4` | parity pass, slower |
| 8 | `12224511` | 19m31s | `1.72771e-4` | parity pass, slower |
| 16 | `12224474` | 1m28s | not produced | OOM requesting 14.75 GiB |
| 64 | `12224475` | 1m25s | not produced | OOM requesting 59.01 GiB |

The successful reports and SHA-256 values are:

- chunk 2:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_order_chunk2_20260810T1828ET/analysis.json`,
  `57c9df1fbd3a986efd476f420739f6bfda3a9d421d8a1c077badf8ff9d4ceb09`;
- chunk 4:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_order_chunk4_20260810T1815ET/analysis.json`,
  `e7060b6fc67948846b3eb8733960791aa78f4f22a2668100333d8653e1b79659`;
- chunk 8:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_order_chunk8_20260810T1818ET/analysis.json`,
  `45698f5858ec3894ac3f546cacdc355eb4475c4f077338b586a63cec960fe27a`.

Two plausible shortcuts were separately falsified:

| arm | job | wall time | denominator relative-L2 | numerator relative-L2 |
|---|---:|---:|---:|---:|
| ordinary size grouping, float64 accumulators | `12224518` | 15m48s | `8.82173e-3` | `1.98070e-2` |
| RELION order stable only within size buckets | `12225119` | 15m36s | `1.84894e-3` | `8.42975e-5` |

Their reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_standard_double_20260810T1822ET/analysis.json`
(SHA-256
`605864b4dedaecf1cb8bf6af7c941a3ee215ffbb1a74021071285de636ef0c99`)
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_native_grouped_20260810T1835ET/analysis.json`
(SHA-256
`6bfe4367310aa181d6abfb8669df6227e71e5d411ef7c1c4b47fc2a5c96e6e0d`).

Therefore the required invariant is the global RELION float32 particle
accumulation sequence.  Stable order only inside sparse size buckets is not
sufficient, and float64 accumulation does not provide a parity-equivalent
order-independent substitute.  Among the tested direct routes, one particle
per call is both the most accurate and the fastest.

## Consecutive-support scoring batches

The production scheduler now separates the two relevant boundaries.  It may
score a contiguous run of physical particles together when every row has the
same padded support size, but it backprojects the resulting BPref operands one
particle at a time in the original physical order.  It never groups
non-adjacent equal-size rows.  Concatenating the scheduled batches therefore
reproduces the authoritative physical permutation exactly.

Focused H100 job `12226818` compared this scheduler directly with the sealed
one-score-call-per-particle iteration-1 route.  For half 1, 1,490 scoring calls
became two calls and sparse E+M time fell from `49.88` to `15.54` seconds.  For
half 2, 1,510 calls became two calls and sparse E+M time fell from `45.43` to
`8.36` seconds.  Total scientific wall time was `84` seconds, versus `156`
seconds for the singleton job.

Batched and singleton accumulators differ only at relative-L2
`1.12e-9`--`1.41e-9`, consistent with rowwise JAX batching arithmetic.  The
decisive native RELION boundary is unchanged:

| iteration-1 half-1 quantity | singleton relative-L2 | consecutive-batch relative-L2 |
|---|---:|---:|
| denominator | `6.49466497e-7` | `6.49463588e-7` |
| numerator | `2.76033522e-6` | `2.76033358e-6` |
| average | `2.58089213e-6` | `2.58089348e-6` |

Both routes retain support Jaccard `1.0` with zero mismatches.  The batching
route is therefore parity-equivalent at the measured physical boundary, not
bitwise identical to the singleton implementation.

The singleton comparison and native comparison reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_consecutive_batch_it1_20260810T1945ET/analysis/K1_CASE22_IT1_BATCH_VS_SINGLETON.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_consecutive_batch_it1_20260810T1945ET/analysis/K1_CASE22_IT1_BATCH_VS_NATIVE.json`,
with SHA-256 values
`0dd458a5ac1b75c34d5d2a994e1a59acc5c3d67399aae3e9bf03762242faf553`
and
`78580b73677bef63deabc061f1ef0a7de52f7da72d8379189ed008a2e14a15d3`.

This clears the practical gate for the fixed 100,000-particle case 7.  Focused
job `12227271` is the bounded two-iteration discriminator combining the
previously positive exact coarse-Gaussian tree with live initial noise, exact
BPref operands, fresh physical dispatch, and the global per-particle BPref
sequence.  It saves numbered maps, particle state, and the complete
iteration-2 accumulator; it does not run a full trajectory or final pass.

## Case-26 two-iteration generalization gate

Focused H100 job `12228267` applied the same production order and exact-BPref
route to the independent 1,000-particle case 26.  The EM computation completed
two physical iterations in 438.5 seconds and wrote all requested maps,
particle state, and iteration-2 accumulator artifacts.  The Slurm wrapper then
returned status 2 only because the complete-trajectory auditor correctly
reported that a deliberately truncated two-iteration run does not have the
11 numbered iterations of the RELION reference.  The focused scientific
artifacts themselves are complete.

The numbered map boundary is essentially closed:

| physical iteration | half 1 FSC-AUC | half 2 FSC-AUC | merged FSC-AUC |
|---:|---:|---:|---:|
| 1 | `0.999999999962` | `0.999999999957` | `0.999999999973` |
| 2 | `0.999999999855` | `0.999999999824` | `0.999999999887` |

At physical iteration 2, the signed merged GT FSC-AUC differs from RELION by
only `-1.28434e-7`.  Particle identity, current size (`56`), HEALPix order
(`3`), and all hard rotations and translations are preserved.  Pmax
relative-L2 is `1.22543e-5`; 65 of 1,000 significant-support counts differ by
exactly one.  This remaining support boundary is numerically small and does
not prevent closure of the reconstructed-map boundary, but it remains the
first discrete intermediary to investigate before claiming exact posterior
parity.

The reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it2_20260810T1925ET/analysis/case26_ordered_exact_fsc_it2.json`,
  SHA-256 `55b47ab9934bb31b55260014d3a8d58a652ddd665b9075860a811e892a8fd5a7`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it2_20260810T1925ET/analysis/case26_ordered_exact_state_it2.json`,
  SHA-256 `ef502beb52cbf82cb9d165d80e991dde22bd64e5590905248c15bf49cc7227bd`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it2_20260810T1925ET/analysis/case26_ordered_exact_particle_state_it2.json`,
  SHA-256 `c6ce62ac145ff5f97479043e9d3d71beac49e6751735aeb281c23e05e9ab3694`.

This independent case confirms that the order correction is not a case-22
overfit.  It also narrows the next posterior investigation to the cumulative
significance threshold for the 65 one-count disagreements: dump the ordered
weights and cumulative mass immediately around each boundary, then compare
raw fine score, additive priors, normalization constant, and threshold
strictness in that order.

## Case-26 iteration-3 localization

Focused H100 job `12230608` extended the same autonomous candidate by exactly
one numbered iteration.  It did not run convergence or final reconstruction.
The iteration-3 merged map remains close to RELION, but the posterior boundary
has begun to amplify:

| quantity | physical iteration 2 | physical iteration 3 |
|---|---:|---:|
| merged map FSC-AUC | `0.999999999882` | `0.999999772413` |
| Pmax relative-L2 | `1.34011e-5` | `1.06287e-3` |
| Pmax RMSE | `2.71936e-7` | `4.95422e-4` |
| significant-count mismatches | `65 / 1000` | `53 / 1000` |

Current size (`66`) and HEALPix order (`3`) remain exactly on the RELION
topology.  Compared with the prior live-noise arm, the candidate reduces the
iteration-3 map FSC-AUC deficit by about 73%, Pmax RMSE by about 74%, and
support mismatches from 167 to 53.  This is material improvement, but not
posterior closure.

The reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it3_20260810T1935ET/analysis/case26_ordered_exact_it3_fsc.json`,
  SHA-256 `f4302f03934c2004603e219645e159098bdabf80d1e928c48da82b01a8641b6b`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it3_20260810T1935ET/analysis/case26_ordered_exact_it3_state.json`,
  SHA-256 `43eb76dc6f92a0b9e80be206106480292cde4a751d8bf61b3866fe92209a82ec`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_ordered_exact_it3_20260810T1935ET/analysis/case26_ordered_exact_it3_particle_state.json`,
  SHA-256 `6e87b5fbb84b4d7e5334b8425b95cb67ba65d756e2c2b3095496817ca0417afe`.

Job `12234888` then held the physical iteration-3 scorer fixed while replacing
its complete incoming boundary with exact RELION iteration-2 state.  This is
the causal discriminator requested by the first-divergence strategy.  The
Pmax relative-L2 collapsed from `1.06287e-3` to `4.19309e-5` (about 96%), Pmax
RMSE collapsed to `1.95448e-5`, and support mismatches fell from 53 to 5.  All
five previously selected tail particles had exact significant counts.  The
largest remaining Pmax error is `2.17471e-4` at source particle 819.

The five raw pass-2 captures are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_exact_it3_dump_retry1_20260810T1955ET/pass2/`.
This experiment largely exonerates the physical iteration-3 scorer,
normalizer, significance selection, and BPref routing.  The dominant residual
is inherited from the small iteration-2 state discrepancy.  The next gate is
therefore exact RELION iteration 1 into RECOVAR iteration 2, with raw captures
for the largest iteration-2 Pmax tails.  That gate is Slurm job `12237191`.

The exact-input wrapper completed its scientific output but exposed an audit
instrumentation omission: `refinement_results.npz` did not contain the two
source-order half-index arrays needed by the generic particle auditor.  The
runner now saves `half1_indices` and `half2_indices` before writing the NPZ;
the focused regression panel passes 61 tests.  This changes only diagnostic
capture, not EM arithmetic.

The apparent iteration-1 direction-prior relative-L2 values (`7.10e-5` and
`6.26e-5`) are a STAR-serialization floor, not evidence of a live-state
formula mismatch.  RECOVAR's positive values are exact normalized count
multiples (`1/517` and `1/483`) and sum to one.  RELION's saved model STAR
rounds each probability to six decimal places; its half sums are
`0.9999410242` and `0.9999579987`.  The two engines have the same nonzero-bin
counts (287 and 250), and iteration-1 hard poses/support are identical up to
STAR Euler/translation precision.  Consequently, a STAR-seeded exact-input
replay is a conservative discriminator with a known serialization residual;
it cannot be called an exact copy of RELION's private in-memory direction
prior.  The newly captured pre-collapse rotation-posterior trajectory avoids
conflating posterior aggregation with this output rounding on future runs.

## Case-26 iteration-2 serialization floor

Job `12237191` completed the exact-RELION-iteration-1 to RECOVAR-iteration-2
discriminator.  Contrary to the iteration-3 replay, loading the serialized
RELION boundary made the already small iteration-2 posterior discrepancy
worse:

| iteration-2 quantity | autonomous candidate | serialized RELION boundary |
|---|---:|---:|
| Pmax relative-L2 | `1.34011e-5` | `2.86879e-5` |
| Pmax RMSE | `2.71936e-7` | `5.82137e-7` |
| significant-count mismatches | `65 / 1000` | `418 / 1000` |

All 418 support-count differences remain exactly one.  This is not evidence
that the live autonomous state is farther from RELION.  It demonstrates that
the autonomous state is already closer than the six-decimal model/data STAR
serialization floor.  The serialized boundary is therefore not a valid
oracle for localizing the residual iteration-2 error.

Five source-particle pass-2 panels were captured under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_exact_it2_dump_20260810T2005ET/pass2/`.
For every panel,
`scores_with_prior == scores_pre_prior + rotation_log_prior + translation_log_prior`
is bitwise exact, and an independent float64 softmax reproduces the stored
probabilities to approximately `1e-15`.  This exonerates RECOVAR's internal
prior addition and normalization at the captured boundary.  It does not
compare the pre-prior scores with native in-memory RELION scores because the
incoming RELION maps, noise, and priors have already been rounded during STAR
serialization.

The particle-state report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_exact_it2_dump_20260810T2005ET/analysis/case26_exact_it2_particle_state.json`,
with SHA-256
`12e562005193b7216c71664e470df0c262c681a6637d85d553e17b24a7ab41ac`.
The Pmax comparison has SHA-256
`d89d4f8700500191fc7aed7494cbbc1b4dde18eac395ed5b4f98314fb78cffd9`.
Both the run root and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_exact_it2_dump_20260810T2005ET/`
contain `SAFE_TO_DELETE` markers.

The next decisive iteration-2 comparison must therefore capture native
RELION values before six-decimal metadata serialization, or compare two
RECOVAR arms whose only difference is a single unrounded incoming operand.
Another broad trajectory from the rounded STAR boundary would not resolve
this question.

## Native iteration-2 fine-score boundary

Job `12240570` completed both requested RELION iterations and captured all five
part-specific ACC panels before its wrapper rejected the absent legacy CPU
dump files.  The scientific ACC capture is complete; no rerun was needed.
The generic verbose arrays identify RELION `part_id=634`, source particle 66,
and contain the complete compact pass-2 boundary.

The source-66 native comparison is decisive at the candidate and score
boundary:

| causal boundary | result |
|---|---:|
| candidate keys | `22656 / 22656` exact |
| rotation matrices | median Frobenius `0`, max `4.21e-8` after transpose |
| translations | max absolute difference `5.96e-8` |
| top pose key | exact: `(2841, 57)` |
| translation log prior | bit-identical |
| rotation log prior | max absolute difference `2.38e-7` |
| raw pre-prior centered residual | p95 `4.52e-5`, max `1.08e-4` |
| posterior L1 after common renormalization | `8.44e-6` |

Thus the first meaningful native iteration-2 mismatch is the raw fine score,
not candidate generation, priors, normalization, or support routing.

The streamed operand comparison then placed both engines in the same packed
current-size Fourier topology.  On score-relevant pixels the relative-L2
operand differences are `8.84e-7` for the projected reference, `1.76e-7` for
the shifted corrected image, and `6.34e-8` for the pixel weight.  A fixed
three-factor intervention attributes only `0.92%` of centered residual-energy
removal to replacing the pixel weight, while replacing reference and shifted
image together removes `67.16%`.  The remaining replay floor is approximately
`1e-5`, principally because the passive decomposition does not capture the
private high-resolution Xi2 addend before float32 addition.  The two
intervention directions are correspondingly asymmetric, so the factorial does
not yet assign the residual to one operand.  It does establish that all three
operands are extremely close and that their float32 interaction, rather than a
candidate or prior mismatch, produces the remaining score-scale discrepancy.

The serialized-RELION-boundary arm makes this more specific.  Relative to the
autonomous arm it improves projected-reference relative-L2 from `8.84e-7` to
`2.41e-7`, but worsens shifted-image relative-L2 from `1.76e-7` to `8.14e-7`
and pixel-weight relative-L2 from `6.34e-8` to `1.94e-7`.  Its native raw-score
p95 residual consequently worsens from `5.79e-5` to `2.72e-4`.  This explains
why wholesale STAR-boundary replay regressed: it fixes the incoming reference
while injecting larger serialized image/noise operands.  The next causal test
must swap one unrounded live operand at a time.

Reports:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_estep_20260810T2034ET/analysis/native_part634_source000066_generic_matrix.json`,
  SHA-256 `c900947402d5b0104d6de918038c18164611e10c18a29a3ae243b60fd3536983`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_estep_20260810T2034ET/analysis/native_part634_source000066_operand_factorial.json`,
  SHA-256 `60270cd4d5de7f02a3561d84d83866ac8fbd7a137a18493197d471cf91b668a6`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_estep_20260810T2034ET/analysis/native_part634_source000066_exact_star_matrix.json`,
  SHA-256 `b1d1787374f6b25627fa2046c5e5f11ecb8ed6d35e2b2f0df13c0989b6dd39a7`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_estep_20260810T2034ET/analysis/native_part634_source000066_exact_star_operands.json`,
  SHA-256 `f134067cbf40a1df392c2fa704a8fe3a1f46c0d7ab952e273bec9bf294187730`.

## Case-7 two-iteration dispatch discriminator

Job `12227271` also completed its two scientific iterations.  Its wrapper exit
2 is solely the expected complete-trajectory audit mismatch (two RECOVAR
iterations versus the 15-iteration RELION reference).  At iteration 2 the
cross-engine signed merged FSC-AUC is `0.9999997822479294` (half 1
`0.9999997802837153`, half 2 `0.9999997822816254`).  This is essentially
unchanged from the prior case-7 iteration-2 result, so the combined physical
dispatch and global particle accumulation correction does not close case 7's
early residual.  Particle order remains demonstrated for case 22, but is now
falsified as a sufficient case-7 root cause at this boundary.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case07_combined_it2_20260810T1920ET/analysis/case07_combined_fsc_it2.json`,
SHA-256 `f351ce6d4b72235789bc8371db5977c5f5abf0a632d209916de46849bd793990`.

## Case-26 iteration-2 boundary after first-iteration atomic closure

The two-iteration candidate prefix, Slurm job `12249681`, demonstrates that
closing the first-iteration fused-atomic boundary is necessary locally but is
not sufficient to close the next posterior.  Relative to native RELION,
iteration-2 Pmax relative-L2 changes from `1.225434e-5` in the previous
control to `1.293417e-5` with the fused first-iteration reconstruction; the
dominant direction-prior and noise residuals are essentially unchanged.
Iteration-1 raw accumulators move to the native repeat floor as intended, but
the iteration-2 hard rotations, translations, coarse assignments, fine
assignments, and noise remain identical to the old RECOVAR arm.  The fixed
scorecard therefore remains `28/34` strict, `32/34` topology, and `34/34`
evaluated.

Native RELION iteration-2 raw-accumulator job `12249766` then compared the
same physical boundary with the fused-first-iteration RECOVAR prefix.  The
complete raw numerator/denominator relative-L2 values are
`5.804861e-6` / `6.590637e-6` for half 1 and
`7.708592e-6` / `4.711303e-6` for half 2.  Aggregate Fourier support is
identical, with Jaccard `1.0` and no support-coordinate mismatch.  This
localizes a real iteration-2 difference before reconstruction, but aggregate
accumulators alone do not distinguish changed posterior operands from
soft-posterior reduction order.

The existing source-66 native fine-score capture provides the earlier
boundary.  Candidate keys, the selected pose, and translation prior agree;
the first meaningful mismatch is the raw pre-prior fine score.  Inspection of
the exact deployed RELION source identifies one previously unmatched float32
operation order in its pixel weight:

```text
RELION: Minvsigma2 * (CTF * CTF)
RECOVAR: (Minvsigma2 * CTF) * CTF
```

These are not bit-equivalent.  For the observed sensitive operands
`Minvsigma2=22520.71875` and `CTF=-0.7145116329193115`, RELION's order yields
float32 bits `0x4633a5bb`, while RECOVAR's order yields `0x4633a5ba`.
The candidate changes only the fine-score `corr_img` construction.  It keeps
the reconstruction/BPref multiplication order unchanged because those
pre-scatter operands are already bit-exact against native RELION.  A focused
two-iteration A/B must reduce the native raw-score residual before this is
accepted as causal; no full trajectory is justified before that gate.

The native iteration-2 accumulator reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_raw_accum_20260811T0120ET/analysis/CASE26_IT2_RAW_ACCUM_HALF1.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_raw_accum_20260811T0120ET/analysis/CASE26_IT2_RAW_ACCUM_HALF2.json`,
with SHA-256 values
`38078ea16526513521576ed69e245715b559a5246d2a29f2b5190fd5762970b5`
and
`30590eebf18d758363bf36d59ef4f4c2521a3c0e44586ce02401fe34984ef1e6`.
Both run and runtime roots contain `SAFE_TO_DELETE` markers.

## Case-26 `corr_img` multiplication-order falsification

Slurm job `12250645` completed the same-H100, two-iteration candidate with
five source-ID-aligned raw pass-2 panels. It changes only the Gaussian
fine-score pixel-weight construction from `(Minvsigma2 * CTF) * CTF` to
`Minvsigma2 * (CTF * CTF)`; first-iteration BPref atomics and later BPref
operands remain unchanged.

The source-66 focused native join falsifies this operation-order change as the
remaining root cause:

| boundary | prior fused-firstiter arm | `corr_img` candidate |
|---|---:|---:|
| pixel-weight relative-L2 | `6.33847e-8` | `6.60746e-8` |
| centered raw-score RMS | `2.05813e-5` | `2.05441e-5` |
| common-renormalized posterior L1 | `8.44159e-6` | `7.76724e-6` |
| iteration-2 Pmax relative-L2 | `1.29342e-5` | `1.30326e-5` |
| merged cross-engine FSC-AUC | `0.999999999885` | `0.999999999880` |

The raw accumulator moves by less than one percent: half-1 numerator and
denominator relative-L2 change from `5.80486e-6` / `6.59064e-6` to
`5.78506e-6` / `6.58397e-6`; half 2 changes from `7.70859e-6` /
`4.71130e-6` to `7.63898e-6` / `4.70787e-6`. Support remains exact.

A direct common-pixel bit audit is more specific. The historical
reconstruction-order expression matches `641/1226` native pixels exactly;
the source-spelling candidate matches `618/1226`. Native values are within
two ULPs of both, but neither expression is globally authoritative: the old
order is closer on 225 pixels, the candidate on 199, and they tie on 802.
This means an earlier factor, cast, or deployed CUDA compiler instruction is
still unmatched. The candidate scoring change is removed rather than
promoted, and no three-iteration or complete-case run is justified.

Reports are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_corrimg_order_it2_20260811T0124ET/analysis/`.
The run completed as Slurm `12250645` in `00:17:45`; both run and runtime
roots contain `SAFE_TO_DELETE` markers. The fixed K=1 scorecard remains
`28/34` strict, `32/34` topology, and `34/34` evaluated.

## Native RFLOAT/XFLOAT stage resolution

The rejected arm reproduced the source parentheses after first narrowing the
CTF to float32.  A staged native capture now identifies that earlier cast as
the missing operation.  The deployed build has `DoublePrec_CPU=ON` and
`DoublePrec_ACC=OFF`, hence `RFLOAT=double` and `XFLOAT=float`.  RELION keeps
`local_Fctf` in RFLOAT while `Minvsigma2` and `corr_img` are XFLOAT.

Native job `12250962` captured the unrounded RFLOAT CTF, its RFLOAT square,
the XFLOAT Minvsigma2, both candidate arithmetic paths, and the actual
`corr_img` for iteration 2 particle 634.  The exact source-semantics replay

```text
float32(float32(Minvsigma2) * (float64(CTF) * float64(CTF)))
```

matches `1624/1624` pixels bit-for-bit.  The narrowed float32-CTF-square path
matches only `991/1624`, has relative-L2 `6.607464652e-8`, and differs by up
to two ULP.  The source STAR plus pinned RELION binding independently
reconstruct the native RFLOAT CTF magnitudes bit-for-bit on all `1275/1275`
RECOVAR score-window pixels; the sign is intentionally opposite and cancels
under the square.

The native instrumentation is inert: job `12250962` completed in `00:01:51`,
preserved selected topology at both iterations, and had a minimum merged
signed FSC-AUC of `0.999999999987` against the control.  The stage report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_corr_stages_rfloat_20260811T0159ET/analysis/CORR_STAGE_BOUNDARY.json`.
Both the run and runtime roots contain `SAFE_TO_DELETE` markers.

This demonstrates the first operand-level cause, but not yet a trajectory
rescue.  The guarded score-only RFLOAT-square candidate is being tested by
the two-iteration, five-particle-panel job `12251045`; reconstruction/BPref
operand order remains unchanged.  No complete-case run is justified until
that candidate improves raw fine score, posterior, and iteration-2 state.

The adjacent corrected-image path has now been staged as well.  Native job
`12251220` completed in `00:01:52` and was inert at the same map/topology
gates.  RELION first forms an XFLOAT pixel correction, compound-divides it by
the RFLOAT CTF, multiplies the RFLOAT Fourier image by that XFLOAT correction,
and assigns the result to XFLOAT `Fimg_`.  The source replay matches both real
and imaginary `Fimg_` arrays at `1624/1624` pixels.  Replacing the RFLOAT CTF
division with a narrowed float32 division matches only `1267/1624`, with
relative-L2 `6.410369846e-8` and a maximum one-ULP difference.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_it2_fimg_stages_20260811T0211ET/analysis/CORR_AND_FIMG_STAGE_BOUNDARY.json`.

Thus the CTF precision defect has two coupled score operands: `corr_img` and
the corrected/shifted image.  Job `12251045` deliberately changes only the
first so its result remains a clean attribution experiment; it cannot be
expected to make the complete raw score bit-exact while the second boundary
is unchanged.

Job `12251045` completed successfully in `00:17:42` and validates that
attribution.  On source 66, the score pixel-weight relative-L2 closes from
`6.607464e-8` to exactly `0.0`, with all `22656` candidate keys and the top
pose unchanged.  Centered raw-score RMS improves from `2.054413e-5` to
`2.045918e-5`.  Global iteration-2 Pmax relative-L2 improves from the fused
control's `1.293417e-5` to `1.218472e-5` (about `5.8%`).  Raw numerator
relative-L2 moves from `5.804861e-6` / `7.708592e-6` to
`5.743178e-6` / `7.627677e-6` for halves 1/2; support remains exact.

The standalone arm is not sufficient: source-66 common posterior L1 is
`9.991521e-6`, and the complete raw-score residual remains dominated by the
shifted image and projected reference.  Therefore no longer trajectory was
scheduled.  The coupled RFLOAT-square plus RFLOAT-pixel-correction prefix was
the next same-size gate, Slurm `12251311`.

Job `12251311` completed successfully in `00:17:42`.  The coupled expression
improves the shifted-corrected-image relative-L2 from `1.758960e-7` to
`1.700775e-7` and the centered raw-score RMS from `2.045918e-5` to
`2.044150e-5`.  Candidate keys, top pose, translation prior, pixel weights,
and aggregate support remain exact.  The downstream result is mixed:
source-66 common posterior L1 improves from the RFLOAT-square-only arm's
`9.991521e-6` to `9.637696e-6`, but iteration-2 Pmax relative-L2 regresses
from `1.218472e-5` to `1.290916e-5`, nearly the fused control's
`1.293417e-5`.  Half-1 raw numerator relative-L2 changes from
`5.743178e-6` to `5.822836e-6`, while half 2 improves from `7.627677e-6` to
`7.418843e-6`; denominators are effectively unchanged.  This proves both CTF
precision corrections locally but does not justify a longer trajectory.

With weight, candidates, priors, and support closed, the first live operands
are the shifted image before/after translation and the projected reference.
Relative translation-phase ratios agree at approximately `9e-8`, so the next
causal gate is a same-boundary map intervention rather than translation-grid
tuning.  Initial state-swap attempts `12252000` (`all_relion`) and `12252001`
(`recovar_maps`) failed closed before iteration 1 because the launcher still
enabled the cold-start-only live-noise flag while requesting trajectory
replay.  They produced no scientific output.  Replacement jobs `12252029`
(`all_relion`) and `12252030` (`recovar_maps`) use the same complete RELION
iteration-2 state and vary only whether the scoring maps come from RELION or
RECOVAR.  They stop after two numbered iterations and write the same
five-particle fine-score and BPref panels.  No full-case or 12-hour run is
scheduled.

A direct map-array check strengthens this localization without replacing the
state-swap experiment.  Before the iteration-2 E-step, iteration-1 RECOVAR to
RELION map relative-L2 is `7.608370e-7` for half 1 and `1.523678e-6` for half
2; after an optimal scalar it is `5.539472e-7` and `1.444176e-6`.  The
source-66 projected-reference relative-L2 is `6.392237e-7`, on the same scale
as its incoming half-1 map discrepancy.  After iteration 2 the map
relative-L2 has grown to `8.463864e-6` / `8.428652e-6`.  This is strong
evidence for inherited reference drift, but only the matched `all_relion`
versus `recovar_maps` boundary can distinguish it causally from projector
arithmetic.

## Iteration-2 map-only state swap result

Replacement jobs `12252029` (`all_relion`) and `12252030` (`recovar_maps`)
completed both scientific iterations.  Their Slurm wrappers exited only in
post-run assertions that incorrectly required fresh-order log lines during a
replay; the refinement results, five score panels, maps, and accumulator dumps
are complete and passed the manual state auditor.  The launcher now scopes
those assertions to non-replay runs.

With the complete serialized iteration-2 RELION state held fixed, substituting
RELION maps reduces source-66 projected-reference relative-L2 from
`3.457410e-6` to `2.410177e-7`, a `14.3x` collapse.  Candidate Jaccard remains
`1.0`, all `22656` candidate keys remain common, and the selected pose remains
exact.  The two arms differ only in projected reference, score, posterior, and
two support-mask entries.  This demonstrates that most of the live projected-
reference discrepancy is inherited from the incoming map rather than created
by the projector.

The serialized replay boundary is not a native-private exact operand boundary:
both arms retain pixel-weight relative-L2 `1.854959e-7` and shifted-image
relative-L2 `8.129298e-7`.  Therefore this experiment is used only for map
causality, not as a trajectory-improvement score.

## First post-accumulator boundary: reconstruction

The production fused iteration-1 accumulator was compared with the passive
native post-low-resolution-join BPref capture.  Complete raw numerator and
denominator relative-L2 values remain at the native repeat floor:
`1.472631e-8` / `1.403778e-8` for half 1 and
`1.326482e-8` / `1.417275e-8` for half 2.  Support Jaccard is `1.0` with no
mismatched coordinates.  The first iteration-1 map discrepancy is therefore
introduced after the half join, inside reconstruction/regularization.

An offline same-input reconstruction routes the exact captured native BPref
arrays through both `BackProjector::reconstruct` and RECOVAR.  FSC and the
untapered per-half tau2 are recomputed from those same arrays.  With the
historical float32 tau2 reconstruction operand, native-versus-RECOVAR premask
relative-L2 is `1.127772e-6` for half 1 and `3.547779e-6` for half 2.  Removing
the final sinc-squared gridding correction leaves essentially the same gap, so
that correction is not causal.

A passive native Fourier dump immediately after Wiener division localizes the
first unequal arithmetic to the inverse-tau term.  Raw weights are exactly
equal after frame conversion.  RECOVAR explicitly returns the tau2 volume as
float32, so `1 / (padding_factor**3 * tau2)` is rounded in float32; the deployed
RELION build uses double `RFLOAT` for this division.  Promoting the unchanged
tau2 values to float64 only at the reconstruction call reduces the premask
gap to `3.010432e-7` for half 1 and `4.833783e-7` for half 2, improvements of
`3.7x` and `7.3x`.  The stored/controller tau2 state remains unchanged.

The native S4/S5 captures, recomputed tau2 arrays, and frame-aligned premask
volumes are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_reconstruct_boundary_20260811/`;
the disposable run and runtime roots contain `SAFE_TO_DELETE` markers.  A
single-iteration production validation of the narrow promotion, Slurm job
`12252525`, completed `0:0` in `00:01:24`.  Half-1 map relative-L2 improves
from `7.609880e-7` to `3.200862e-7`; half 2 improves from `1.523602e-6` to
`3.216119e-7`.  After an optimal scalar the residuals are
`1.658642e-7` / `1.638572e-7`, and merged signed FSC-AUC remains
`0.999999999976`.  The two previously asymmetric half errors have collapsed
to the same residual floor.

The focused two-iteration gate, Slurm `12252541`, completed `0:0` in
`00:08:12` (maximum RSS `9,969,680K`).  The improved iteration-1 maps do not
materially close the iteration-2 posterior discrepancy: Pmax relative-L2 is
`1.291868e-5`, versus `1.290916e-5` in the otherwise matching coupled RFLOAT
CTF arm.  The iteration-2 shellwise FSC relative-L2 does improve from
`5.120341e-7` to `4.719380e-7`, and the raw pre-join BPref errors improve in
both halves: numerator/denominator relative-L2 changes from
`3.289099e-6` / `4.644272e-6` to `2.708196e-6` / `4.534888e-6` for half 1,
and from `5.311427e-6` / `2.889127e-6` to
`4.004386e-6` / `2.747994e-6` for half 2.  Support remains exact
(`Jaccard=1.0`, zero mismatched coordinates).

Therefore the inverse-tau precision mismatch is a demonstrated K=1
reconstruction bug and reduces the next M-step map error, but it is not the
dominant cause of the remaining iteration-2 posterior gap.

## Iteration-2 fine-score operand panel

Focused H100 job `12253082` captured the complete source-66 iteration-2,
half-2 fine panel after the reconstruction-precision fix.  The valid capture
contains all `22656` candidate keys; candidate Jaccard, the top candidate, and
the pixel weights are exact.  Projected-reference and shifted-corrected-image
relative-L2 are `2.7987092e-7` and `1.7007746e-7`.  Translation priors are
exact; rotation-prior maximum absolute error is `2.384e-7` with p95 equal to
zero.  Centered raw-score maximum error is `5.65e-5`, posterior L1 after a
common renormalization is `9.919424e-6`, and support remains exact.

Replacing only the projected reference removes `54.5%` of the centered
raw-score residual energy: RMS changes from `1.745746e-5` to `1.177507e-5`.
Replacing the shifted image does not rescue the residual.  This independently
localizes most of the first iteration-2 fine-score mismatch to the incoming
reference reconstructed at iteration 1, not candidate generation, priors,
normalization, support, or translation.  Reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_tau64_pass2_panel_it2_20260811T0348ET/analysis/native_part634_source000066_tau64_matrix.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_tau64_pass2_panel_it2_20260811T0348ET/analysis/native_part634_source000066_tau64_operands.json`.

## Exact pre-IFFT denominator-floor boundary

An environment-gated capture now records the raw numerator, raw denominator,
regularized denominator, support mask, tau operand, and divided Fourier volume
after Wiener division and before Fourier windowing.  The first attempt, job
`12253342`, failed only when the diagnostic tried to convert a JAX tracer to a
NumPy array; it produced no boundary files.  The writer was moved to an
ordered JAX host callback and its jitted unit test passes.  Replacement job
`12253382` completed `0:0` in `00:01:24` with maximum RSS `4,172,772K` and
wrote both half captures under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_wiener_boundary_callback_it1_20260811T0445ET/wiener_boundary/`.

Against the passive native S4 dump, the captured divided-volume relative-L2
is `6.494010e-7` for half 1 and `1.101362e-6` for half 2.  Replacing only the
numerator leaves `6.484593e-7` / `1.100861e-6`; replacing only the regularized
denominator collapses the residual to `3.331524e-8` / `3.163885e-8`.  The ten
largest half-1 residual voxels explain `96.4%` of squared residual energy; the
ten largest half-2 voxels explain `98.96%`.  Every one is on padded radius 56,
where RELION applies the shell-27 denominator floor.

The exact mismatch is the population used for that floor.  RELION computes
the `1/1000` shell average over its stored FFTW x-half.  RECOVAR expanded the
Hermitian accumulator to a full cube and averaged that cube, counting x>0 and
x<0 partners twice while counting the x=0 plane once.  At shell 27 this gives
`11189.4212137` instead of RELION's `11193.3290538` in half 1 and
`10451.3293231` instead of `10455.0850665` in half 2.  Selecting the exact
nonredundant x bins reproduces the native constants to below `5e-9` absolute.

The candidate fix therefore performs only the shell reduction over the
nonredundant x-half and broadcasts the resulting floor back to the existing
full layout.  A same-input CPU replay of the immutable job-`12253382` operands,
with RELION's `minres_map=5`, predicts divided-volume relative-L2
`4.785819e-8` / `4.626395e-8`.  Four focused unit tests pass.  H100 production
validation job `12253435` is pinned to diff
`7bc4666d4e6d406ded17136a8ba5fb0553773f0caeff0d74ec6f5da9ce3d1b17`;
Slurm scheduled it for `2026-08-11T18:00:00` because of H100 maintenance.  It
has not been cancelled, modified, or treated as completed evidence.

This closes the first unequal iteration-1 reconstruction operation, but it is
not yet a scorecard promotion.  The fixed K=1 score remains `28/34` strict,
`32/34` topology, and `34/34` evaluated.  After the short H100 confirmation,
the next bounded gate is a two-iteration replay to measure how much the exact
floor correction reduces the already-captured iteration-2 reference,
raw-score, posterior, and BPref residuals; no full trajectory is justified
before that result.

## Exact S4-to-S5 post-Wiener boundary

The same immutable native S4 capture was then propagated one operation at a
time through Fourier windowing, inverse FFT, and `softMaskOutsideMap`.  The
RELION binding and RECOVAR Fourier-window implementations are bit-exact for
both halves: relative-L2 and maximum absolute difference are both zero.  With
the corrected nonredundant shell floor, RECOVAR's complete S4-to-S5 result
still differs from native S5 by `1.211760e-7` / `1.230728e-7`.  Supplying the
exact native S4 directly leaves essentially the same `1.198380e-7` /
`1.216213e-7`, so this residual is created downstream rather than inherited
from Wiener division.

The mismatch is spatially exhaustive to a single operation.  After matching
the native float32 S5 dump, every voxel strictly inside radius 64 and every
voxel outside radius 67 is exact; all measurable error lies in the
`64 <= r <= 67` raised-cosine transition.  RECOVAR constructed the coordinates
with the shared float32 Fourier-grid helper and therefore evaluated the
radius and cosine in float32 even though the reconstruction volume is float64.
The deployed RELION build evaluates these quantities in double `RFLOAT`.

Keeping mask geometry at the real dtype of the input volume reduces the exact
native-S4-to-S5 relative-L2 to `1.368256e-16` for half 1 and `4.205085e-15`
for half 2.  In each half, `2,097,151` of `2,097,152` float32 output voxels are
bit-exact; the sole remaining value differs only at roundoff
(`3.55e-15` / `1.14e-13` before the native float32 cast).  Float32 callers
retain float32 mask arithmetic.  This identifies and repairs the next unequal
iteration-1 reconstruction operation without running a trajectory.

Focused verification is
`tests/unit/test_core_mask.py::TestSoftMaskOutsideMap::test_float64_transition_matches_relion_double_arithmetic`;
the complete core-mask unit file passes `55/55`, and the focused EM guardrail
passes.  This is again same-input causal evidence, not an autonomous scorecard
promotion: the fixed K=1 score remains `28/34` strict, `32/34` topology, and
`34/34` evaluated.

## Two-iteration discriminator after the reconstruction fixes

A local A100 prefix propagated the combined tau-operand, nonredundant
shell-floor, and float64 solvent-mask fixes through two physical iterations.
The run completed successfully under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_softmask_local_a100_it2_20260811T0512ET/`;
the run and runtime roots contain `SAFE_TO_DELETE`.  It is diagnostic rather
than scoring because the qualified case-26 reference was produced on H100.

Both numbered FSC gates remain extremely close: signed merged FSC-AUC is
`0.9999999999764076` at iteration 1 and `0.9999999999586052` at iteration 2.
The iteration-2 Pmax relative-L2 is `1.3101096e-5`, compared with the prior
H100 tau64 prefix's `1.291868e-5`.  The iteration-2 pre-join native comparisons
are:

| half | numerator relative-L2 | denominator relative-L2 | support |
|---:|---:|---:|---:|
| 1 | `2.771475e-6` | `4.536796e-6` | Jaccard `1.0`, zero mismatches |
| 2 | `3.784376e-6` | `2.750142e-6` | Jaccard `1.0`, zero mismatches |

These values are essentially unchanged from the preceding H100 tau64 prefix,
apart from a modest half-2 numerator improvement.  The reconstruction fixes
are exact and necessary at their isolated same-input boundaries, but they are
falsified as the dominant cause of the iteration-2 posterior discrepancy.
The first material remaining boundary is therefore before reconstruction, in
the score/prior/posterior operands that feed the iteration-2 accumulator.

The run also exposed an audit-harness defect.  The launcher constrained pass-2
panel dumps by current size 56 but not by iteration; case 26 uses that size in
both iterations, so the write-once files captured iteration 1.  They are
explicitly marked invalid in the run root and were not used above.  The
launcher now exports `RECOVAR_PASS2_DUMP_ITERATION=${MAX_ITER}` so the next
five-particle capture cannot silently bind to the wrong physical iteration.
The fixed score remains `28/34` strict, `32/34` topology, and `34/34`
evaluated.
