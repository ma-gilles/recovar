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
