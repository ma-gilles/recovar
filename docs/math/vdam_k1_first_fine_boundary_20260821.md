# VDAM K=1 first fine-score boundary (2026-08-21)

This note records a causal K=1 InitialModel correction. It does not update the
frozen 12-case scorecard or relax any acceptance threshold.

## Boundary

For `vdam-01`, iteration 1, stack image 1060 (RECOVAR original index 1059),
RELION and RECOVAR had the same 608 fine candidates, centered priors, winner,
input image, pixel weights, and bitwise-identical serialized iteration-0 map.
The projected-reference operand contained the full residual. Of 52,448 unique
rotation/pixel values, the material differences were confined to the rounded
current-image rim, especially `(kx, ky) = (19, +/-1)` and `(1, 19)`.

RELION's fine CUDA projector evaluates its model-radius predicate after the
rotation and converts the float squared radius to `int`. The ordinary supplied-
map EM path also applies an independent exact current-image disk. That disk is
correct for the existing EM contract but is not part of InitialModel's rounded
fine-score support, so InitialModel now disables it through an explicit engine
option while the EM default remains enabled.

The first unmasked replay exposed a second, coupled error: the VDAM driver
scaled a separately rounded `rnd_unif(0, 1)` draw. RELION evaluates
`rnd_unif(low, high)` inside the source float function. At seed 0 / iteration 1
the correct perturbation is `-0.07990610599517822`, rather than
`-0.07990613579750061`. The one-ulp perturbation difference changed the
integer-truncated radius decision. VDAM now reuses the exact perturbation
sequence already implemented by the EM sampler.

## Evidence

One-iteration Slurm job `12705559` on one GPU produced:

- exact candidate support (`608/608`, Jaccard `1.0`);
- the same top candidate `(rotation 77, translation 52)`;
- centered fine-score maximum error `1.220703125e-4`, down from `7.72343e-2`;
- posterior L1 error `2.18086e-5`, down from `1.12452e-2`;
- top probability `0.584262447` versus RELION `0.584272844`.

The remaining score residual is at the established CUDA float-reduction scale.
The full paired trajectory job `12705606` confirmed iteration-1 FSC-AUC
`0.9999999947` and iteration-2 FSC-AUC `0.9999984885`. It still fails the
frozen map gate at iteration 4 (`0.9985366004`) and iteration 8
(`0.9909807355`). The corrected first hard particle-state divergence is at
iteration 2 (stack images 108 and 1171), so that is the next localization
boundary.

Disposable evidence root:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_full_exact_perturb_rim_df055428_20260821T064500Z`

## Rejected coarse-pass extension

The fine-pass disk correction must not be copied into the coarse significance
pass.  Same-GPU Slurm job `12706043` disabled the explicit current-image disk
there as a controlled counterfactual.  It changed the iteration-2 target from
four to five significant samples, but retained only three fine rotation
parents and selected a different fifth sample than RELION.  Centered coarse
score RMS error increased from `0.23792` to `0.89085`; iteration-1 FSC-AUC
worsened from `0.9999999947` to `0.9999956200`, and iteration-8 FSC-AUC worsened
from `0.9909807355` to `0.9901001496`.  The counterfactual was reverted.

## StoreWavg boundary

The subsequent production-path StoreWavg audit separates posterior error from
M-step arithmetic.  The initial replay appeared to show a `1.90e-3` translated
image-operand error, but it was confined to the Fourier origin.  RELION's
pre-StoreWavg `Minvsigma2` capture has a zero at the origin; immediately before
accumulation, `ml_optimiser.cpp` restores that element to
`1 / (sigma2_fudge * sigma2_noise[0])`.  The analyzer now performs the same
restoration from the captured `sigma2_fudge.bin` and `sigma2_noise.bin` rather
than treating the pre-StoreWavg dump as the accumulator input.

With the corrected replay (Slurm job `12712005`), the same-posterior controls
are all at float-rounding scale:

- translated image/CTF/inverse-noise operand relative L2: `2.43e-7`;
- CTF-squared/inverse-noise operand relative L2: `4.99e-8`;
- gradient numerator relative L2: `2.12e-7`;
- gradient denominator relative L2: `5.46e-8`;
- scattered data and weight relative L2: `1.96e-7` and `3.64e-8`.

Using RECOVAR's actual posterior instead leaves `1.87e-5` and `1.63e-5`
relative errors in the scattered data and weight, respectively, consistent
with the measured `1.97e-5` posterior relative L2 error.  This proves that the
remaining iteration-1 boundary is fine scoring/posterior formation, not the
reconstruction accumulator.

Two production counterfactuals were rejected:

- applying RELION CUDA `sincosf` translation to reconstruction operands changed
  the operand by only about `1.5e-7` and slightly worsened the posterior;
- replacing the big-JIT unmasked FFT with a centered host FFT changed the
  reconstruction operand by only about `1.9e-7` and did not change the apparent
  origin residual.

A third counterfactual zeroed the RECOVAR reconstruction origin to match the
pre-StoreWavg dump.  Although that made the uncorrected boundary replay appear
exact, full-trajectory Slurm job `12711875` worsened iteration-1 FSC-AUC from
`0.999999609002` to `0.999946224077`.  RELION's source-level DC restoration
explains the failure; the change was reverted.

Corrected replay evidence:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_storewavg_restored_dc_p69_df055428_20260822T004500Z`

The next localization target is therefore the remaining fine-score operand and
float-reduction residual that produces the iteration-1 posterior mismatch and,
after propagation, the two first hard particle-state divergences at iteration 2.

## Fused preprocessing and live fine-score boundary

The InitialModel driver selected the shared `relion_cuda` image backend, but
the fused local big-JIT previously bypassed that backend and used the generic
JAX mask/FFT path.  The guarded K=1 exact-operand route now calls the same
source-order CUDA normalization, integer pre-shift, and soft-mask primitive as
the EM route, followed by a per-image cuFFT.  The unmasked reconstruction image
uses the corresponding normalized, unmasked real-space output so scoring and
StoreWavg do not silently use different normalizations.

Paired full-trajectory job `12712263` changed iteration-1/2 FSC-AUC only at the
last few digits (`0.999999994669` and `0.999998488644`) and retained the two
known iteration-4/8 failures.  On the focused particle, however, posterior
relative L2 improved from `1.9687e-5` to `1.7252e-5` and the actual-posterior
StoreWavg data/weight errors improved to `1.5892e-5` and `1.3864e-5`.  The
change is therefore locally closer and reuses the established EM primitive,
but is not by itself a trajectory closure.

The live production operand audit (Slurm jobs `12713277`, `12713441`, and
`12713508`) corrected an earlier comparison against an obsolete adapter dump:

- projected reference: bitwise exact for all `362368/362368` candidate-pixels;
- score weight: bitwise exact for all `596/596` active pixels;
- CUDA translation of RELION's unweighted corrected image: bitwise exact for
  all `362368/362368` candidate-pixels;
- live preweighted shifted image: relative L2 `1.0967e-7`;
- centered live raw-score residual: RMS `3.5358e-5`.

Preweighting the captured native image before translation, instead of
translating the corrected image and applying `corr_img` inside diff2, accounts
for a `5.5543e-8` relative operand residual by itself.  The strict preprocessing
replay matches `519/596` Fourier values exactly, with only `1.4263e-8` relative
L2 over the remainder.  Both available deterministic CUDA background-reduction
orders produce the same result on this particle.

Most importantly, replaying the source operation order with the shared direct
diff2 CUDA kernel matches `563/608` native raw fine costs bit-for-bit and cuts
the centered raw-score RMS residual to `1.7677e-5`.  Its computed high-resolution
tail is `0.0226478204`, versus the `0.0226440430` addend inferred from the native
capture; the small common difference is mostly absorbed by float32 addition.
This makes corrected-image formation followed by unweighted translation and
direct diff2 the next production counterfactual.  The prior attempt to apply
direct diff2 to the preweighted/algebraic operands was not equivalent: it
worsened the posterior to `2.2284e-5` and was reverted.

Live-boundary evidence:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_native_translation_boundary_df055428_20260822T030000Z/report_v8.json`

## Direct fine diff2 and first aggregate boundary

The source-order direct fine scorer is now available behind
`RECOVAR_INITIAL_MODEL_EXACT_FINE_DIFF2=1`.  It translates the corrected,
unweighted source image with the RELION CUDA primitive, forms the exact
`corr_img`/half-spectrum weight, runs the direct float32 fine-diff2 tree, adds
the high-resolution tail, and applies RELION's common-min posterior ordering.
It fails closed unless the exact BPref operands, big-JIT path, and current-size
window are all active.

For the captured iteration-1 particle, the production posterior relative L2
falls from `1.7252e-5` to `2.2671e-6`, with identical 41-sample support.  The
StoreWavg data/weight scatter residuals fall to `1.1451e-6` and `7.7936e-7`.
The full eight-iteration run costs `163.6 s`, only about 1.2% above the
`161.6 s` strict RECOVAR control, but retains the iteration-4/8 trajectory
failures (`0.9985366025` and `0.9909807346` FSC-AUC).  The direct scorer is
therefore a validated local improvement, not yet a complete K=1 closure.

A matched native/RECOVAR iteration-1 M-step dump locates the next unequal
boundary before reconstruction.  The incoming `Igrad1` halves and `Igrad2`
are bitwise exact (`1,064,960/1,064,960` values each), and the real-space input
reference agrees to `9.43e-16` relative L2.  The first mismatch is the raw
BPref accumulator: data relative L2 is `5.7286e-4` / `7.7170e-4` and weight
relative L2 is `1.4284e-4` / `2.1709e-4` for the two pseudo-halves.  The final
reconstructed reference differs by `8.8245e-5` relative L2.  This rules out
`reconstructGrad` as the first cause and moves the active boundary to the
aggregate E-step contribution stream.

Two aggregate-order counterfactuals were rejected.  Sequential float32
translation reduction leaves the raw accumulator and trajectory unchanged at
reported precision while increasing runtime to `166.2 s`.  One fused launch
per particle likewise changes the accumulator only around `3e-7` relative and
does not materially improve the native residual.  Both routes were removed.

The fine-lane mapping matrix also rejects radial compact lane assignment.
Native current-grid and parent physical-grid mappings give the same direct
score result (`563/608` exact; inferred native high-resolution mode
`0.0226440430` for 563 hypotheses), while radial compact lanes fall to
`377/608`.  Native atomic soft-mask reduction improves the focused score to
`571/608`, but its full aggregate and trajectory changes are mixed and
negligible: iteration-8 FSC-AUC is `0.9909807505`.  Because it introduces
schedule-dependent preprocessing without a parity gain, the deterministic
production preprocessor remains selected.

Aggregate-boundary evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_mstep_boundary_exact_diff2_df055428_20260822T060000Z/mstep_boundary_report.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_fine_lane_mapping_matrix_df055428_20260822T110000Z/report.json`

## Multi-particle posterior panel and native powerClass reduction

A paired iteration-1 panel confirms that the focused-particle result is not an
outlier.  RELION's debug hook is not thread-safe, so only 10 of the 200
eight-thread captures were internally consistent and complete.  Across those
10 particles, the direct scorer has zero support and argmax mismatches.  Its
posterior relative L2 has median `1.4687e-5`, maximum `3.8170e-5`, and pooled
value `1.8327e-5`.  Replaying RECOVAR's posterior through the native StoreWavg
operands gives pooled data/weight relative L2 of `1.4424e-5` and `1.3855e-5`.
This independently confirms that most of the aggregate mismatch is inherited
from the posterior rather than introduced by the scatter.

Running RELION with one thread produces clean captures for all 200 particles,
but it is not a valid replacement baseline: the denovo bootstrap itself is
thread-count dependent and changes candidate sets.  The paired analyzer
therefore rejects incomplete and internally inconsistent threaded captures
instead of silently combining them.

RELION's CUDA `powerClass` computes the high-resolution image tail with
128-lane block reductions and atomic accumulation.  The exact fine route now
reuses a matching CUDA FFI primitive.  Its H100 GPU contract test passes, and
the focused posterior relative L2 improves from `2.2671e-6` to `2.0822e-6`;
the same-posterior StoreWavg data/weight errors are `1.1302e-6` and
`9.5984e-7`.  The full run remains `159 s` and has the same trajectory within
reported precision: iteration-4/8 FSC-AUC are `0.9985365971` and
`0.9909807328`.  This source-faithful reduction is retained as a local parity
improvement, but the next boundary remains fine-kernel accumulation and
high-resolution-tail addition order.

Panel and atomic-reduction evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_all200_score_panel_df055428_20260822T123000Z/storewavg_panel_report.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_atomic_powerclass_exact_diff2_df055428_20260822T150000Z/storewavg_panel_report.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_atomic_powerclass_exact_diff2_df055428_20260822T150000Z/vdam-01/trajectory_audit.json`

## Coarse-grid and aggregate counterfactuals

The adaptive coarse pass now obtains its Euler matrices from the same
device-side float32 RELION sampler used by the fine pass.  The captured coarse
matrices are bitwise exact, but paired trajectory job `12719051` is unchanged.
This rules out host Euler conversion as the cause of the iteration-2
four-versus-five support split.  A support-boundary analyzer now reports
matched and unmatched posterior mass, cumulative threshold state, and winners
for that first hard divergence.  On original image 107 (stack image 108), the
native fine support has 32 rotations versus RECOVAR's 24.  The eight
native-only rotations carry `0.9937483` posterior mass, and the dominant
missing rotation alone carries `0.9742115`; it contains the native winner at
probability `0.6623028`.  The RECOVAR winner is consequently a different
rotation and translation.  The late trajectory divergence is therefore a
coarse-threshold support error with a known missing parent, not a small
same-support posterior perturbation.

Changing the adaptive fraction is not a viable way to mask the split.  A
controlled `0.9992` run (job `12720469`) worsened every recorded checkpoint:
iteration-1/2/4/8 FSC-AUC became `0.99976629`, `0.99981799`, `0.99822291`, and
`0.98991555`.  RELION's default float32 `0.999` contract remains fixed.

A native-shaped one-particle SGD BPref launch was also rejected.  It worsened
the iteration-1 gate to `0.9999998512` and left the later trajectory effectively
unchanged (`0.9985361155` and `0.9909848864` at iterations 4 and 8; job
`12720193`).  More importantly, the matched capture made the raw accumulator
much less native: data relative errors rose to `2.208e-2` / `2.258e-2` and
weight errors to `8.26e-3` / `8.80e-3`, compared with the accepted sub-`8e-4`
boundary.  A single coupled particle launch does not reproduce RELION's
aggregate BPref schedule, so the implementation and diagnostic flag were
removed.

Rejected-aggregate evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_native_sgd_mstep_capture_20260822T221500Z/mstep_boundary_report.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_it2_hard108_capture_6833cb22_20260822T174500Z/posterior_support_report.json`
- Slurm jobs `12719051`, `12720193`, and `12720469`

## Native fused coarse projector boundary

The iteration-2 hard split was caused by separating projection from RELION's
coarse diff2 kernel.  Replaying the captured in-memory `PPref` through the
existing preprojected path leaves a centered raw-diff2 RMS error of
`0.2371851` (maximum `1.00757`).  A fused CUDA diagnostic now follows the
native source topology: it stages the compact `PPref` in texture memory,
projects 16 Euler matrices per 128-thread block, applies the translation with
the native `sincosf` arithmetic, and accumulates diff2 in the same lane/atomic
topology.  It also clips the projector to `min(mdlMaxR, imgX - 1)`, which is
essential for RELION's even current-image crop.

The fused replay reduces the centered native score residual to `2.3677e-5`
RMS and `9.1553e-5` maximum; `8,912/16,704` hypotheses are bitwise exact.  On
the formerly failing original image 107, both engines now select hypothesis
14891 and retain the same five coarse hypotheses at the `0.999` cumulative
cutoff.  The exact K=1 route therefore uses one all-rotation fused launch,
preserving RELION's 128-orientation main segment and one-orientation tail.

Paired trajectory job `12723377` confirms that this closes the first hard
boundary.  Iterations 1 and 2 have zero pose/translation mismatches across all
3,000 particles, and their FSC-AUC values are `0.99999999995` and
`0.99999999994`.  A separate divergence begins at iteration 3 (94 particles),
so iteration 4 and 8 still fail at `0.9985507` and `0.9910791`.  RECOVAR wall
time remains `163.29 s` versus RELION's `22.45 s`; the fused change is a
correctness closure, not yet the required performance closure.

Fused-coarse evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_native_ppref_coarse_replay_20260823T021500Z/report_v5.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_fused_coarse_20260823T110000Z/vdam-01/trajectory_audit.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_fused_coarse_20260823T110000Z/vdam-01/particle_state_audit.json`

## K=1 resolution schedule and eight-iteration closure

After the fused coarse scorer removed all iteration-2 winner mismatches, the
next divergence was not another fine-score defect.  RELION entered iteration
3 with a 34-pixel fine grid, while RECOVAR used 30 pixels.  Their model STARs
localized the cause to the previous M-step spectrum: RELION selected shell 7
(`77.714286 A`) after iteration 2, while RECOVAR selected shell 5
(`108.8 A`).

Two independent native-boundary errors caused the spectral gap:

1. `getSpectrum` and `Projector::computeFourierTransformMap` were called with
   RECOVAR-frame real-space volumes.  RELION's half-spectrum traversal is not
   invariant to RECOVAR's X/Z frame swap.  On the captured iteration-1
   reference, the old call exactly reproduces RECOVAR's discrepant tau2,
   whereas applying `recovar_volume_to_relion` first reproduces RELION's tau2
   to displayed precision for every inspected shell.
2. `updateSSNRarrays` was given `0.5 * (accum_h0.weight +
   accum_h1.weight)` for K=1.  RELION actually calls it on
   `BPref[iclass].weight`; the pseudo-halfset backprojector participates in
   the gradient moments but not this SsnrMap calculation.  The captured
   primary RECOVAR and native weights agree shell-by-shell to about `2e-5`
   relative or better over the decisive low shells.

Using only the primary weight improves iteration-8 FSC-AUC from `0.9910791`
to `0.9942334` but does not close the run (job `12725109`).  Applying both
native-boundary corrections makes the resolution and image-size schedules
identical through iteration 8 and eliminates every particle winner mismatch
at iterations 1, 2, 3, 4, and 8.  The paired trajectory then passes all map
checkpoints:

| Iteration | Cross-engine FSC-AUC |
| ---: | ---: |
| 1 | `0.999999999947` |
| 2 | `0.999999999937` |
| 4 | `0.999999999898` |
| 8 | `0.999999999437` |

The remaining Pmax differences do not change winners: mean absolute error is
`3.38e-5` at iteration 3 and `1.33e-4` at iteration 8.  Runtime is not yet
closed: this cold-cache run took `191.57 s` for RECOVAR versus `22.29 s` for
RELION.  K=1 correctness can now expand to the rest of the fixed parameter
matrix while performance remains a separate required workstream.

Closure evidence:

- Slurm job `12725298`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_relion_frame_ssnr_20260823T150000Z/vdam-01/trajectory_audit.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_relion_frame_ssnr_20260823T150000Z/vdam-01/particle_state_audit.json`

## Frozen K=1 matrix and large-grid memory closure

The corrected scorer, M-step spectrum, and resolution schedule pass every
scientifically evaluated case in the frozen 12-case GUI-default K=1 matrix.
The first matrix screen produced 11 passing reports; `vdam-09` terminated
before evaluation because a 256-pixel A100 run inherited the user-facing
500-image batch and attempted a transient allocation of about 17 GB after the
projector state was resident.  This was an execution-policy failure, not a
parity failure.

InitialModel now treats the requested batch as an upper bound on 256-pixel and
larger inputs.  Its internal cap scales approximately with image area and
available accelerator memory, from 32 images at 256 pixels on a 40 GB device.
The requested and effective values are written to every iteration metadata
file.  A separate packed-M-step safety cap scales to 15 percent of the
smallest visible device, with the existing environment variable remaining an
explicit override.

The automatic default-policy rerun of `vdam-09` completed all eight
iterations on the A100 with an effective batch of 33 and passed every map
checkpoint.  Its minimum cross-engine FSC-AUC was `0.999999997955`; RECOVAR
took `228.21 s` versus RELION's `50.26 s`.  A deliberately conservative
25-image counterfactual also passed, with minimum FSC-AUC `0.999911635235`.
Thus the frozen scientific screen is 12/12 passing, while formal post-commit
matrix qualification and runtime convergence remain open.

Matrix and memory evidence:

- Slurm array `12725777` (11 evaluated passes and the original operational
  `vdam-09` OOM)
- Slurm job `12726759` (automatic default-policy `vdam-09` pass)
- Slurm job `12726698` (explicit 25-image safety counterfactual pass)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam09_auto_batch_a100_20260823T162000Z/vdam-09/trajectory_audit.json`

## InitialModel accumulator-finalization profile

The K=1 profile showed another avoidable JAX cost after every local M-step.
InitialModel's changing reconstruction grid caused device-side x=0 Hermitian
enforcement and half-to-full layout expansion to compile new volume-shaped
programs at most resolution steps.  On `vdam-01`, this finalization took about
3.1--3.8 seconds for the first pseudo-halfset and only about 0.017 seconds for
the second, demonstrating compilation rather than arithmetic as the dominant
cost.

InitialModel consumes both arrays on the host immediately, so its sparse local
route now explicitly uses the existing source-equivalent host implementations
for x=0 enforcement and public-layout expansion.  This is scoped to the
InitialModel adapter; the shared EM engine retains its existing size-based
default.  The explicit host counterfactuals preserve the closed trajectory:

| Case | RECOVAR before | RECOVAR host finalize | RELION | Minimum FSC-AUC |
| --- | ---: | ---: | ---: | ---: |
| `vdam-01` (3k/128) | `175.81 s` profiled | `156.00 s` profiled | `22.52 s` | `0.999999999440` |
| `vdam-09` (3k/256) | `228.21 s` | `176.00 s` | `29.68 s` | `0.999999998020` |

This removes roughly 11 percent and 23 percent respectively without changing
the scientific result.  It does not by itself meet the runtime objective: the
remaining profile is dominated by changing-shape coarse and local score/M-step
compilations.

Performance evidence:

- Slurm job `12726796` (device-finalization profile)
- Slurm job `12727257` (`vdam-01` host-finalization counterfactual)
- Slurm job `12727430` (`vdam-09` host-finalization counterfactual)

## Formal fine-score capture and native repeatability floor

The patched RELION fine-score capture now records the production candidate
table and one selected operand without relying on the older verbose dump's
inferred common addend. Same-H100 Slurm job `12743706` captured iteration 1,
internal part 69 / stack image 1060 / class 1 for `vdam-01`. The formal table
contains all 608 active candidates and records the actual high-resolution
`sum_init` as `0.022647816687822342`.

The selected RELION projector row is bitwise identical to RECOVAR in all 596
complex score pixels. Passive and SASS replays reproduce the selected native
raw cost exactly. Across two independent RELION executions, however, only
467/608 raw costs are bitwise equal; the centered residual has RMS
`3.09e-5` and is confined to one or two float32 ULPs. RECOVAR's posterior is
closer to the formal run than the older RELION run is:

| Comparison | posterior L1 | posterior relative L2 | max absolute |
| --- | ---: | ---: | ---: |
| older RELION vs formal RELION | `1.55e-5` | `1.37e-5` | `6.67e-6` |
| RECOVAR vs formal RELION | `7.02e-6` | `6.02e-6` | `3.00e-6` |

All three select the same candidate. This establishes a native GPU
repeatability floor and rejects the remaining fine-score ULP residual as the
causal explanation for the long-trajectory failure. The opt-in exact fused
fine scorer remains useful for boundary validation, but it is not promoted as
a trajectory repair.

Formal evidence root:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_vdam01_formal_fine_capture_20260823T191500Z`

## K=1 long-trajectory boundary

The frozen 25-iteration K=1 suite passes through the short-trajectory region
but fails all three long cases. Slurm array `12739560` produced minimum
cross-engine FSC-AUC `0.9869212`, `0.9855999`, and `0.9613005` for
`vdam-l01`, `vdam-l02`, and `vdam-l03`. Independent RELION repetitions stay
above approximately `0.9999999996` through iteration 25, so the late split is
not native run-to-run nondeterminism.

For `vdam-l01`, RECOVAR-vs-RELION map relative L2 grows gradually from about
`1.4e-5` at iteration 1 to `3.96e-5` at iteration 19, then jumps to
`6.41e-3` at iteration 20. Pose and translation winners remain essentially
identical through iteration 15. Iteration 20 is the scheduled transition from
Healpix order 1 to 2; both engines use the same grid schedule. The fine grid
therefore amplifies an accumulated map/posterior drift rather than introducing
a scheduler mismatch.

Two bounded counterfactuals were rejected:

- preserving physical particle order for K=1 BPref accumulation (job
  `12744453`) slightly worsened the RECOVAR-vs-RELION map relative L2 at every
  checked iteration and was reverted;
- replacing the expected-accuracy denominator's scheduled `tau2_fudge` with a
  literal 1 (array `12744685`) was a production no-op because the schedule has
  already deflated the factor to 1 by the accuracy-estimation iteration; it
  reproduced the three prior failures and was reverted.

Aggregate repeatability job `12745125` completed both full `vdam-01` pairs on
one physical H100; only its final analyzer invocation failed, because the
original shell loop nested arm B under arm A and invoked the analyzer with a
non-package script path. The captures were complete and the repaired analyzer
produced `mstep_repeatability.json` without rerunning science.

RELION's raw BPref numerator varies by `9.62e-6` / `9.41e-6` relative L2
between identical runs, while RECOVAR varies by only `2.61e-7` / `2.79e-7`.
The two cross-engine numerator comparisons are `1.44e-5--1.65e-5`, or only
about 1.5--1.7 times the native repeat floor. The reconstructed RELION map
varies by `2.65e-6`; the cross-engine maps differ by `6.60e-6` and `8.21e-6`.
Thus bitwise accumulator closure is impossible on the native GPU path, but a
small systematic reduction-order residual remains above that floor.

RELION does not scatter every orientation/translation hypothesis separately.
`cuda_kernel_backproject3D` loops translations inside each orientation/pixel
thread, then scatters one reduced numerator/weight row. The current RECOVAR
path instead reduces validated BPref operands in XLA before its CUDA scatter.
An opt-in CUDA primitive was tested that owned only this translation reduction
in native storage order. Same-H100 job `12745587` passed the eight-iteration
trajectory, but its iteration-1 boundary was unchanged: paired raw numerator
errors were `1.60e-5` / `1.40e-5` and reconstructed-map error was `7.72e-6`.
Against the two earlier RELION captures, changes from baseline were mixed at
about `1e-7` relative and stayed inside native variability. The primitive and
its routing were therefore reverted without a 25-iteration run. CUDA
instruction ownership of the already validated translation reduction is not
the long-trajectory cause.

Repeatability evidence:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_mstep_repeatability_20260821T234848Z/mstep_repeatability.json`

Rejected CUDA-reduction evidence:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_cuda_reduce_it1_20260824T003500Z/mstep_boundary.json`

## Iteration-19 incoming-state boundary

Same-H100 paired job `12746179` captured every M-step stage at iteration 19
of `vdam-l01`. The expected frozen trajectory failure occurs after all 25
iterations, but the late capture shows that reconstruction is not a new first
boundary. Its incoming gradient halves already differ by `3.04e-2` and
`2.65e-2` relative L2, and the incoming second moment differs by `2.01`.
The RELION-frame reference differs by `1.84e-3` before reconstruction and
`4.81e-3` after it. These internal pre-solvent values are more sensitive than
the saved masked-map FSC, but their ordering is decisive: the expectation and
running-gradient state have diverged before the iteration-19 M-step begins.

The first focused RECOVAR pass-2 selector in that job was invalid. RELION
internal particle 69 maps through `Experiment::read` ordering to RECOVAR input
row 159 (`160@particles.128.mrcs`); stack image number 1060 is not a dataset
row. Jobs `12746548` and `12746857` corrected and broadened the row selector,
respectively, but exposed a second diagnostic constraint: InitialModel's local
engine does not execute the adaptive sparse pass-2 writer used by the supplied-
map EM path. Disabling local big-JIT in job `12747611` did not change that
ownership, so the job was cancelled after it passed the target iteration
without a dump. Job `12747788` instead uses the local engine's dedicated fused-
posterior hook, which records the actual production fused path without changing
its arithmetic. The M-step tensors and production trajectories from the earlier
jobs remain valid.

Late-boundary evidence:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it19_boundary_20260824T004500Z/mstep_boundary_it19.json`

The production fused-posterior capture from job `12747788` closes the selected
iteration-19 particle to the native numerical floor: the argmax and both
reconstruction hypotheses are identical, posterior relative L2 is `1.54e-5`,
L1 is `2.18e-5`, and Pmax is `0.9978859` versus native `0.9978750`. This is the
same scale as the `1.37e-5` posterior relative-L2 difference between two native
RELION captures. It rules out a new late-iteration posterior topology defect for
this particle, but does not explain the aggregate half-map drift.

Whole-map repeatability does rule out stochastic native drift as the explanation.
Two independent paired runs have native RELION cross-repeat FSC-AUC
`0.99999999995`, `0.99999999993`, `0.99999999994`, `0.99999999991`,
`0.99999999978`, and `0.99999999963` at iterations 1, 8, 12, 16, 20, and 25.
RECOVAR repeats are similarly stable (iteration-25 FSC-AUC
`0.99999999871`). The cross-engine decay is therefore systematic even though a
single late posterior is native-floor equivalent. The next boundary is the
first resolution-expansion interval after iteration 12. Direct map relative L2
stays at `1.54e-5` through iteration 13, then jumps to `2.20e-4` at iteration
14 and grows monotonically thereafter. At iteration 14, shell-relative error is
largest at shells 25--28 (`0.0062`, `0.0091`, `0.0123`, `0.0156`), exactly at
the newly admitted edge of the size-56 Fourier window. Job `12748071` captures
both iteration-14 M-steps, and job `12748177` captures the production fused
posterior for the same selected particle at that first expansion boundary.

Fused-posterior evidence:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it19_fused_posterior_20260824T021000Z/fused_posterior_boundary.json`

The iteration-14 M-step makes the expansion failure asymmetric. Its incoming
reference is still only `1.55e-5` relative L2 from RELION, but raw accumulator
data differs by `4.35e-4` in half 0 and `6.81e-2` in half 1; reconstruction
turns that into a `6.42e-4` reference difference. Comparing all 631 visited
particles identifies a single topology change: source row 41
(`42@particles.128.mrcs`, RELION internal part 357) changes Pmax from
`0.994889` to `0.440294` and is the only different winning pose. The other
selected particle used as a control retains identical argmax/support, although
its posterior relative L2 has already grown to `1.08e-3` at the new shell edge.
Jobs `12748468` and `12748469` retarget the production posterior and raw operand
captures to the failing particle.

The particle history moves the first topology boundary back one iteration. Its
pose agrees through iteration 12 (Pmax `0.994789` versus `0.994158`), but at
iteration 13 the engines choose different fine poses even though Pmax remains
close (`0.366682` versus `0.366445`). Iteration 14 then searches around those
different local centers and produces the large Pmax split. Jobs `12748544` and
`12748545` capture the failing particle at iteration 13, the first pose-choice
boundary, rather than only its amplified iteration-14 consequence.

Iteration-14 M-step evidence:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it14_boundary_20260824T023000Z/mstep_boundary_it14.json`

## Iteration-13 score operand and high-shell noise boundary

The failing iteration-13 particle is a genuine near tie. RELION assigns
posterior `0.366620` to mapped key `[6, 57]` and `0.366151` to `[14, 58]`;
RECOVAR reverses them at `0.366301` and `0.366359`. The projector differs by
only `5.07e-6` relative L2, while the inverse-noise score weights differ by
`3.02e-4` and the weighted shifted image by `2.81e-4`. The resulting top-pair
raw-score odds are biased by about `-1.19e-3`; the pose-prior bias is only
about `6.7e-6`. This localizes the systematic operand to the evolving noise
spectrum rather than geometry or the pose prior.

The opt-in exact fine path avoids that particular tie and remains pose-exact
through iteration 15, but moves the first tie to iteration 16. There RELION's
top probabilities are `0.498213` and `0.498031`, while RECOVAR returns
`0.497909` and `0.498334`. Projector relative L2 is `1.17e-5`; score-weight
relative L2 is `1.61e-4`. Exact fine scoring therefore changes which near tie
fails but does not remove the upstream operand drift.

Native per-particle noise captures exposed a separate local-engine defect.
Within the currently scored shells, RECOVAR's raw noise numerator matches
RELION at roughly `2e-6--5e-6` relative L2. Above the current window, however,
RECOVAR multiplied `power_img` by the retained significance/Pmax mass even
though RELION adds it once per particle. At iteration 13 the common high-shell
ratio was exactly the retained-mass ratio, `568.847/569 = 0.999732`. This tail
feeds future iterations when the Fourier window expands.

The exact-local engine now uses the same mass topology already implemented by
the shared sparse EM path: current shells remain posterior weighted, and valid
high shells are unweighted and owned by exactly one class. Focused CPU tests
pass `18/18`; the related InitialModel/EM noise subset passes `90/90`. In
25-iteration jobs `12749438` and `12749439`, both default and exact-fine routes
have zero pose mismatches through iteration 19 and cross-engine FSC-AUC above
`0.9999999996` through iteration 16. The earlier iteration-13 and iteration-16
single-particle failures are removed.

This correction does not close the full trajectory. Both routes transition
from zero pose mismatches at iteration 19 to 557 mismatches at iteration 20,
when the sampling schedule advances from HEALPix order 1 to 2. Iteration-20
FSC-AUC is `0.9989211`; iteration-25 values are `0.9871984` (default) and
`0.9872198` (exact fine). Since the two scorers are indistinguishable at this
boundary, the remaining long-run problem is now isolated to the order-change
sampling/significance path rather than fine diff2 or high-shell noise power.

Evidence roots:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it13_part357_fused_20260824T031000Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it13_part357_score_20260824T031000Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_exact_it16_part691_fused_20260824T033000Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_exact_it16_part691_score_20260824T033000Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_noise_terms_20260824T034500Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_highshell_fix_default_20260824T041500Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_highshell_fix_exact_20260824T041500Z`

## Iteration-20 order-promotion coarse-size boundary

Verbose native and RECOVAR significance captures for source particle 0 show
that the order-1-to-2 failure occurred before fine scoring. RELION's coarse
pass selected rotation 3800 with four significant translations, while RECOVAR
selected rotation 3824 with two. The captured coarse device Euler tables are
bit-exact after the expected transpose, and replaying RECOVAR's coarse scorer
with RELION's in-memory PPref changes centered diff2 by only `0.00285` RMS.
The incoming map was therefore not the cause.

The actual operand topology differed: RELION scored a 26-by-26 coarse Fourier
window (364 half-spectrum pixels), while RECOVAR scored 50-by-50 (1300
pixels). RELION computes `image_coarse_size` before `updateAngularSampling`, so
an expectation that promotes HEALPix order must size pass 1 from the incoming
order while using the promoted order for the orientation grid and fine-child
expansion. InitialModel now carries that pre-update order explicitly into its
sparse pass-2 adapter. A focused regression pins the observed order-1-to-2
boundary to `coarse_size=26`.

Paired H100 job `12751069` closes the frozen 25-iteration trajectory gate.
Cross-engine FSC-AUC is `0.9999999857` at iteration 20 and `0.9999987771` at
iteration 25, compared with `0.9989211` and about `0.9872` before the fix. All
1,000 poses/translations agree at iteration 20; two near-tie rows differ at
iteration 25. The full InitialModel adapter/driver unit suites pass `71/71`.
External wall time remains non-comparable on this small case: RECOVAR takes
`326.5 s` versus RELION's `23.0 s`, so performance is still an explicit K=1
work item after correctness qualification.

Evidence roots:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it20_native_verbose_part0_forced_20260824T053000Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it20_recovar_coarse_part0_20260824T052000Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_it20_order2_part0_boundary_20260824T043000Z`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_preupdate_coarse_size_default_20260824T073000Z`

## Exact fine arithmetic promoted to the K=1 default

With the order-promotion boundary closed, the exact RELION CUDA fine-diff2
path removes the two remaining iteration-25 translation-grid flips. Paired
H100 job `12751275` has no pose or translation mismatch above `1e-4` across
all 1,000 particles at iteration 25. Its minimum trajectory cross-engine
FSC-AUC is `0.999999999223`, versus `0.999998777126` with the former default,
and its worst RECOVAR-minus-RELION GT FSC-AUC delta is `-3.54e-7`.

The exact path is therefore the default for the guarded K=1 RELION-projector
route. `RECOVAR_INITIAL_MODEL_EXACT_FINE_DIFF2=0` remains an explicit
diagnostic/performance opt-out. This default does not affect CPU/dense
fallbacks or routes without the exact RELION projector operands.

Evidence root:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_l01_preupdate_coarse_size_exact_20260824T083000Z`

## Rejected compact shared-pass-2 execution topology

InitialModel's qualified exact-local route pads each image's retained fine
rotations to a bounded rectangular bucket.  As a performance experiment, the
adapter can instead call the shared supplied-map EM compact sparse pass 2 with
`RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2=1`.  The experiment reuses the
same coarse significance lists and RELION child grids.  The shared sparse
engine gained one explicit VDAM operation that it previously lacked: before
backprojection it can subtract
`posterior_mass * projected_reference * CTF^2 / sigma^2` from the translated
image sufficient statistic, matching the proven exact-local gradient formula.

Same-H100 frozen case job `12764156` completed and passed the unchanged audit,
so the residual extension is scientifically coherent.  It is not a candidate
default.  Cross-engine FSC-AUC fell to `0.9994907915` at iteration 8, while
RECOVAR wall time rose to `334.44 s` versus RELION's `7.70 s` (`43.4x`).  The
shared compact implementation generated thousands of small JAX cache entries
and compiled changing support shapes; steady-state iterations also remained
slower.  This is about twice the wall time of the qualified local route on the
same scale of frozen fixture.

The switch therefore remains opt-in and fail-closed for K>1 and
zero-oversampling.  Production keeps unified exact-local buckets.  Future
performance work should reuse compact pair indexing inside the qualified
single-shape local scorer, rather than switching wholesale to the current
shared sparse execution driver.

Evidence root:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_compact_pass2_76039662_20260820T120000Z`

## Unified InitialModel CLI and GUI contract

RECOVAR now exposes the native path as `recovar initial_model`. The lightweight
`GuiInitialModelDefaults` dataclass is the sole default source for that command,
`NativeInitialModelOptions`, the legacy parity runner, the GUI defaults API,
and the GUI command builder. The GUI retrieves those values from
`GET /api/jobs/initial-model/defaults`, so a backend default change cannot be
silently shadowed by a TypeScript literal.

The default form matches the RELION GUI contract (`nr_iter=200`, `K=1`,
`tau2_fudge=4`, C1 refinement, particle diameter 200 Angstrom, Healpix order 1,
oversampling 1, offset range/step 6/2 pixels) and defaults to GPU 0 with the
custom-CUDA runtime gate enabled. It exposes the important parity and scaling
controls, including masks/CTF, symmetry, K, sampling, offsets, batching,
padding, image Fourier backend, deterministic diagnostics, and iteration
artifact cadence. Job clones preserve the complete submitted parameter map.

This interface work does not change the qualified K=1 numerical path. The
remaining scientific frontier is K>1 class-matched parity; the remaining K=1
engineering frontier is the real-data runtime ratio documented above.

Focused CPU coverage passes `90/90`, including the public command, legacy
runner, native driver, compact-pass diagnostic, and project registry. H100
Slurm smoke job `12766060` exercised the public command with the resolved GUI
K=1 controls; the CUDA gate passed and the expected iteration/final artifacts
were written in 38 seconds. A focused Chromium check also verified the live
defaults endpoint and rendered form against the built static bundle.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_unified_cli_smoke_20260822T061500Z`
- `/tmp/gui_qa/screenshots/initial_model_focused.png`

The repository-wide `gui_qa.sh` now waits for slow server imports and includes
an InitialModel form assertion. Its legacy full-journey portion cannot run on
this machine because both hard-coded `old_regression_scores_v2` fixture roots
are absent; the focused live-browser check covers the changed InitialModel
surface independently.

## Default-GUI 200-iteration expansion

The earlier 25-iteration qualification does not establish parity across the
full default InitialModel schedule.  The frozen
`vdam-k1-gui-default-full-v1` suite now covers 22 small- and midscale fixtures
and requires every artifact from iteration 0 through 200.  It spans uniform,
Kent, and anisotropic poses; white and radial noise over low through very-high
noise levels; no-CTF, contrast/noise scaling, translations, resolution and
scale changes; junk particles; and outlier fractions through 70 percent.

The first terminal tranche contains no passes.  These are strict
point-reference failures; the fixed cross-engine FSC-AUC gate remains 0.999
and the fixed RECOVAR-minus-RELION GT FSC-AUC floor remains -0.002.

| Case | First failing iteration | Minimum cross-engine FSC-AUC | Minimum GT delta |
|---|---:|---:|---:|
| gf01 | 73 | 0.821270076 | -0.002129885 |
| gf05 | 73 | 0.724679574 | -0.004098185 |
| gf06 | 72 | 0.532028366 | -0.020190610 |
| gf07 | 37 | 0.537170303 | -0.000228475 |
| gf08 | 93 | 0.938161585 | -0.001214697 |
| gf10 | 13 | 0.508864782 | -0.004549611 |
| gf11 | 94 | 0.606768758 | -0.003905791 |
| gf12 | 74 | 0.511178696 | -0.018881116 |
| gf13 | 59 | 0.721548645 | -0.000959252 |
| gf14 | 82 | 0.673242209 | -0.002398550 |
| gf15 | 77 | 0.658667288 | -0.007948350 |
| gf16 | 76 | 0.957551841 | -0.000651299 |
| gf17 | 69 | 0.236270982 | -0.000836103 |
| gf18 | 31 | 0.675630118 | -0.003830814 |
| gf19 | 65 | 0.681088003 | -0.001006762 |
| gf21 | 84 | 0.719066226 | -0.001117729 |
| gf22 | 75 | 0.868005545 | -0.004747162 |

The no-CTF gf16 row is an important discriminator: its GT-quality delta stays
inside the frozen nondegradation gate while its cross-engine trajectory still
fails.  Exact RELION dynamics and output quality are therefore separate open
requirements.  The gf10 two-native/two-candidate same-GPU panel also fails its
repeat envelope from iteration 13 through 200, so that late divergence is not
accepted as stock RELION multimodality.

The final long-running gf17 contrast/noise-scale cell is also complete.  Its
particle state first differs at iteration 25, its strict FSC-AUC gate first
fails at iteration 69, and its iteration-200 cross-engine FSC-AUC is
`0.236270982`.  RECOVAR takes `8035.53 s` versus RELION's `1169.54 s`, a
`6.87x` runtime ratio.  The full 22-case suite has therefore finished; every
completed cell is a tracked parity failure, not an infrastructure failure.

For gf01, all 3,000 poses and translations agree through iteration 32.  At
iteration 33 exactly one particle, `1003@particles.128.mrcs`, chooses a
different fine pose.  Capturing that boundary required keeping three identity
spaces separate: RECOVAR original row 1002, RELION shuffled internal particle
6, and stack index 1003.  RELION continuation also regenerates the sampling
perturbation unless it is forced; the sealed full run used 0.322510.  The
capture harness now pins that perturbation, requires all replayed poses and
translations to match the sealed iteration-33 state, and gates the target
particle's Pmax independently of unrelated native Pmax variability.

A preliminary corrected A100 comparison has exact 298-sample reconstruction
support but reverses a two-candidate posterior near tie.  Native favors mapped
key `[53, 58]` at 0.110033609 over `[52, 54]` at 0.109983251; RECOVAR favors
`[52, 54]` at 0.110071361 over `[53, 58]` at 0.110054567.  Posterior relative
L2 is 9.93e-4 and the maximum probability residual is 1.11e-4.  This localizes
the first discrete split to fine-score/posterior arithmetic rather than
significance support.  Same-GPU production and raw-score operand captures
`12862068` and `12862069` are the acceptance evidence; no source fix is
authorized from the preliminary cross-job comparison alone.

The generic RECOVAR full/long suite is intentionally not part of this
campaign.  Validation is the focused VDAM unit and merge guards plus frozen
trajectory, repeat-envelope, parameter, scale, K>1, and real-data evidence.

The exact-from-zero same-H100 iteration-1 soft-mask panel `12894797` and GPU
audits `12894863` and `12895553` calibrate the first image boundary without
changing that strict status.  All eight normalized/shifted inputs are
bit-identical, while stock RELION's background spans 15 float32 ULP.  The
current deterministic block-first result lies inside that range and one ULP
from its nearest sampled value.  Across all 480 score candidates, RECOVAR's
nearest centered native RMS is `1.5012e-5`, below the native/native maximum
`2.7816e-5`; 296 candidates are inside the coordinatewise native envelope and
184 remain outside.  Native atomics add schedule dependence without
guaranteeing a closer trajectory, so no production topology change is
accepted.  The next discriminator is the aggregate iteration-2 noise update
under deterministic-lane and native-atomic modes.

That discriminator is complete.  Isolated diagnostic commit `f35844a9a`
passes 4/4 focused routing/fail-closed tests.  Same-input iteration-2 jobs
`12896342`--`12896345` all complete.  Shell 15 is `+1.5602497e-5` for the
block-first control, `+1.5520540e-5` for deterministic native-lane, and
`+1.5639750e-5` / `+1.5539167e-5` for two native-atomic runs.  Lane order
improves only `8.20e-8` (0.53 percent), and native atomics straddle the same
error.  All modes remain outside the native repeat floor, so they are rejected
without a 200-iteration run.  Failed preflight submissions
`12896228`--`12896231` produced no science.

The controlled same-GPU gf01 repeat-4 panel `12880351` also completed its
evidence before the expected strict-audit exit.  Candidate repeatability first
falls below the native envelope at iteration 34.  Its worst
candidate-repeat-minus-native-repeat FSC-AUC margin is `-0.0452569`; at
iteration 200 the candidate repeat floor is `0.8348111` versus native
`0.8753012`.  Candidate/native matching and GT quality later fail as well.
The long-run instability is therefore not only a choice among stock RELION
modes; RECOVAR's own repeat spread is materially larger.

## Per-particle cutoff-shell noise boundary

Tracking commit `f343c5bb3` adds a fail-closed analyzer and targeted runner for
the already-qualified production big-JIT debug triplet.  It maps native
RELION `part_id` through stable STAR image identities, requires complete
particle coverage and one common cutoff shell, converts by the exact `N^4`
frame factor, and reports direct residual, `AA`, `XA`, inferred image power,
and retained support mass independently.  Its two focused unit tests, Ruff,
and shell syntax checks pass; no generic RECOVAR suite ran.

Slurm `12897360` captured all 200 frozen gf01 iteration-1 particles at shell
19.  RECOVAR/native aggregate direct residual is
`0.1293679055`/`0.1293675770`, a `+3.28e-7` difference.  `AA` is lower for
all 200 particles (sum error `-5.77e-8`); `XA` and inferred image-power sum
errors are `+3.05e-7` and `+9.96e-7`, so the coupled terms substantially
cancel.  Support-mass relative L2 is `2.90e-7`.  The job's production capture
is complete and valid, although its post-run analyzer initially exited on an
absolute-script import error; the preserved artifacts were analyzed by module
invocation, and `d8314fb27` repairs that setup path.

This rejects iteration 1 cutoff aggregation as the already-full-size source
of the iteration-2 shell-15 `+1.55e-5` error.  The identical iteration-2
panel `12897664` completes cleanly at tracking head `be6ee1f13` and reproduces
the defect: its direct-residual sum error is `+1.6006e-5`.  The `AA` sum error
is `-3.8716e-6` and is negative for 197/200 particles.  The `XA` sum error is
`-9.9872e-6`; through `AA - 2*XA`, it contributes about `+1.9974e-5` and is
the dominant term.  Inferred image-power sum error is only `-8.64e-8`.
Per-particle direct error correlates `-0.9979` with `XA` error and `0.1355`
with image-power error.  A single support-mass outlier at native part `59` has
error `+2.11e-5` but contributes only `-1.10e-7` direct error; removing it
slightly increases the aggregate residual mismatch.

The first material aggregate boundary is therefore posterior-weighted
image/reference cross correlation, with a smaller systematic reference-power
deficit.  Image-power formation, soft-mask topology, and total support mass
are rejected as dominant causes.  The next bounded experiment will replay
native and candidate posteriors on the same captured operands for the largest
`XA` contributors (beginning with parts `27`, `65`, `175`, `96`, and `123`)
to separate posterior error from cross-term operand/reduction error before any
production change.

That bounded replay is complete.  Slurm `12899028` serially captured parts
`5`, `27`, `28`, `65`, `96`, `123`, `138`, and `175` in independent j8 RELION
runs using the true 200-iteration schedule, stopping only after iteration 2.
All eight runs reproduce the frozen iteration-1/2 hard pose and translation
states exactly.  Serial capture is required: an earlier multi-target attempt
is quarantined because RELION's shared diagnostic prefix is not thread-safe.

The fail-closed reference decomposition reuses the EM Wavg component replay
and closes both cutoff terms.  Across the eight largest contributors, the
candidate-posterior change on native operands sums to only `+4.65e-11` for
`XA` and `+2.58e-12` for `AA`.  Replacing only the native reference projection
with the production RECOVAR projection accounts for `-2.2125802e-6` of `XA`
and `-5.6053283e-7` of `AA`; the unexplained residuals are only
`-3.63e-13` and `-4.00e-15`.  Per-particle reference-projection relative L2 is
`4.21e-7`--`1.52e-6`.  Thus the candidate projection plus native masked image,
CTF, translations, posterior, and Wavg reduction reproduces the production
candidate `XA/AA` to about `4.1e-13`/`2.3e-14` absolute.

The iteration-2 cutoff failure is therefore propagated reference-state error,
not a posterior, image-power, masked-image, CTF, translation, or Wavg-reduction
defect.  The causal search moves back to the iteration-1 BPref/M-step
accumulator mismatch already observed before reconstruction.  Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_it02_native_xa_top8_serial_40076ddd_20260824/analysis/state_audit.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_it02_native_xa_top8_serial_40076ddd_20260824/analysis/posterior_panel.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_it02_native_xa_top8_serial_40076ddd_20260824/analysis/reference_decomposition.json`

## Hopper PTX/JIT M-step resource discriminator

The separate-accumulator fused arm also tested whether the remaining native
repeat-scale difference was caused by CUDA resource topology.  Static
inspection first appeared to show stock RELION's SGD kernel at 40 registers
and 48 bytes of shared storage.  The sm80-only candidate build in job
`12913534` matched those cubin values exactly, but focused job `12913597`
correctly rejected it on H100 with `no kernel image is available`: RELION's
binary also embeds compute-80 PTX, so Hopper is not executing that inspected
sm80 cubin directly.  Two source/routing guards passed before the two CUDA
executions stopped; no boundary science ran.  The build's final inspection
command also used a node-local path that was absent, producing exit 127 after
the binary had built.  Both failures are retained as provenance failures.

The corrected isolated commit `8a471b29e` is PTX-faithful.  RELION's native
SGD PTX declares nine float32 Euler entries (36 bytes), so the candidate pads
its six-entry compact rotation to the same PTX shared declaration and removes
the artificial register cap.  PTX-only build job `12913683` completed with
CUDA digest
`d7566386a3c2fb6223f796f626af812b5f6036e9f73c8e4736ab70881464aa61`;
the dumped candidate and native PTX both declare 36 shared bytes.  H100 job
`12913762` then passed all four focused source, fused-interior, SGD-y-boundary,
and fail-closed routing tests.  Initial submissions `12913773` and `12913774`
stopped before work because their new disposable roots lacked the mandatory
`SAFE_TO_DELETE` marker.  Resubmitted jobs `12913835` and `12913836` completed
on distinct H100 UUIDs in 52 seconds each using the true 200-iteration
schedule and stopping after the iteration-1 M-step capture.

The result is null.  Native repeat relative L2 is
`1.0812071e-5`/`1.0094684e-5` for the two data halves,
`2.4511435e-6`/`1.9858739e-6` for weight, and `2.4726651e-6`
after reconstruction.  Candidate repeat is much tighter, but cross-engine
data/weight residuals remain `0.977`--`1.254x` the native repeat floor.  The
reconstructed reference is already inside that floor at `0.725`/`0.820x`.
Matching RELION's PTX target and static shared declaration therefore does not
reproduce its atomic distribution and is rejected without a long trajectory.
The isolated experimental commit remains unpushed.

A new fail-closed paired M-step analyzer now loads both arms' qualified cross
reports, compares native/native and RECOVAR/RECOVAR using their respective
on-disk schemas, and emits per-stage native-floor ratios.  Its two focused
unit tests pass; Ruff and diff checks pass.  No generic RECOVAR suite ran.
Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_ptx_jit_repeat_panel_8a471b29_20260824.json`
- `/scratch/gpfs/GILLES/mg6942/slurmo/relion-patched-ptx-20260824.txt`
- `/scratch/gpfs/GILLES/mg6942/slurmo/vdam-ptx80-jit-ptx-12913683.txt`

## Accepted-posterior inline-projector discriminator

Isolated commit `9cc94a796` reapplies the previously tested inline RELION
texture projector on top of the accepted float32 posterior, fused residual
formation, and separate real/imaginary/weight accumulator storage.  It remains
an intermediate control with separate 36-byte projection and backprojection
Euler tensors, not the final single-Euler native kernel body.  PTX build job
`12914645` produced CUDA digest
`4aadd0eea82aa70da29f573dce8059d5bae117aa73f7d8367903907f7399b026`;
the job exited only at the final hash because the fresh worktree lacked the
unchanged RELION Python binding artifact.  After linking the already-qualified
binding, H100 job `12914671` passed all five focused projector, fused-interior,
SGD-y-boundary, separate-storage, and routing gates.

Paired true-200-schedule boundary jobs `12914695`/`12914696` completed on
distinct H100 UUIDs in 51 seconds each.  Native repeat relative L2 is
`8.97751e-6`/`9.38897e-6` for data, `1.68587e-6`/`2.27024e-6`
for weight, and `1.39658e-6` after reconstruction.  Cross/native ratios are
`0.951`--`1.287` for data, `0.917`--`1.624` for weight, and
`1.479`/`1.590` for the reconstructed reference.  Inline projection worsens
the repeat-aware reconstructed boundary and is rejected without a long
trajectory.  The experimental commit remains unpushed.  The remaining exact
kernel discriminator must use one native Euler tensor for both projection and
scatter and mirror the native SGD control flow, rather than combining two
independent approximations.  Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_inline_sep_repeat_panel_9cc94a79_20260824.json`
- `/scratch/gpfs/GILLES/mg6942/slurmo/vdam-inline-separate-ptx-12914645.txt`

## Single-Euler inline-projector discriminator

Isolated commit `defa4e075` compile-time-specializes the fused kernel and uses
one nine-float Euler tensor for both inline texture projection and native-order
scatter.  Its compute-80 PTX has exactly one 36-byte shared allocation in the
inline specialization.  PTX-only build job `12914984` completed with CUDA
digest
`aee4ec5ca832b0b3d844fde2bb238332774630c9a9f7b2844867bd5430bfa8d2`.
Focused H100 job `12915212` passed all five source, fused-interior,
SGD-y-boundary, zero-projector, and fail-closed routing tests.

Paired true-200-schedule boundary jobs `12915367`/`12915368` completed in
59/60 seconds on distinct H100 UUIDs.  Native-repeat relative L2 is
`9.34808e-6`/`1.07796e-5` for data, `2.24983e-6`/`2.48560e-6`
for weight, and `1.93312e-6` after reconstruction.  The two cross/native
ratios are `0.920`--`1.044` for data, `0.922`--`1.087` for weight, and
`1.041`/`1.245` for the reconstructed reference.  The candidate is therefore
still at the native nondeterminism floor rather than materially closer, and is
rejected without a long trajectory.  Its experimental commit remains
unpushed.

This closes the bounded resource/layout variants: matching PTX target, shared
Euler footprint, accumulator storage, inline projection, and one-Euler reuse
does not reproduce RELION's atomic distribution.  The next discriminator must
mirror RELION's native SGD kernel body and control flow, including its image,
translation, significant-weight, and atomic-triplet paths, before another
trajectory is justified.  Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_single_euler_repeat_panel_defa4e07_20260824.json`
- `/scratch/gpfs/GILLES/mg6942/slurmo/vdam-single-euler-ptx-12914984.txt`
- `/scratch/gpfs/GILLES/mg6942/slurmo/vdam-single-euler-focused-12915212.out`

## Native-SGD production boundary closure

The final isolated arm now mirrors RELION's native VDAM SGD kernel body rather
than composing individually plausible projector and scatter variants.  The
accepted production chain culminates in `fb162a1e1` and retains the earlier
layout, physical-particle packing, one-Euler, and inline-projector changes only
as prerequisites of that source-faithful kernel.  Its normalized compute-80
PTX matches native RELION line for line: all 523 of 523 lines, including ABI,
register, and 36-byte shared-Euler declarations.  Focused H100 job `12916461`
passed all five CUDA source, routing, projection, boundary, and accumulator
tests in 14.48 seconds.  The subsequently integrated PR head passed the full
15-test CUDA translation file in focused H100 job `12920426` in 5.56 seconds.

Aggregate paired iteration-1 jobs `12916504` and `12916505` still land at the
native/native atomic repeatability floor.  That boundary cannot distinguish a
source-faithful implementation from a different atomic distribution, so it is
no longer used as the acceptance oracle.  Source-level contribution capture
first established exact alignment for all 80 of 80 selected rotations in job
`12917549`; the accepted posterior was already bitwise identical.

The authoritative discriminator reruns RELION's production
`runBackProjectKernel` for one selected particle into a zeroed temporary GPU
BPref and compares that result with RECOVAR's inline production contribution.
Native capture job `12919940` produced the complete iteration-1
`(41, 41, 21)` data and weight buffers for the exact 3,000-particle gf01 input
under the true 200-iteration schedule.  The wrapper's Slurm state is failed
only because it still required the older `Fimg_unweighted_nomask` diagnostic;
the new GPU BPref artifacts themselves are complete.  The runner now makes
that obsolete image diagnostic optional in particle-BPref mode so subsequent
capture status reflects the scientific artifact gate.

Fail-closed analyzer job `12920218` completed successfully.  Native RELION and
RECOVAR production contributions agree to float32 precision:

- data relative L2 `3.107632513845835e-7`, cosine
  `0.9999999999999516`;
- weight relative L2 `2.0207509932595694e-7`, cosine
  `0.9999999999999835`;
- posterior bitwise exact, with zero support mismatch.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_native_particle_bpref_it1_20260825/analysis/storewavg_img0_native_gpu_bpref.json`
(SHA-256
`2f7ba86f54c249e0540cf936a48180329a0464a9f51707dee6c4febb5dbdd663`).
The native real, imaginary, and weight capture SHA-256 values are respectively
`6656db9a7cc8f7d5c5eb663dec82ad87471d6b471c570b397ee483ea78f136b1`,
`d3fd7740c0ebcfa63e5ef8eda127e91ae2ba7ee254f4c59126a033f9fc4c7e2a`,
and `fc1f49f7d6495700d5e63b9d488f6133391077452987d0e410fc012e54e04493`.

This formally closes the K=1 iteration-1 production posterior/scatter
boundary.  The earlier CPU binding replay's 6.16% data and 3.65% weight
differences are rejected as a false diagnostic caused by using a CPU scatter
path that is not RELION's production mechanism.  The next boundary is the
iteration-2 cutoff/noise state propagated from the now-qualified iteration-1
M-step, followed by one representative true 200-iteration K=1 trajectory
before expanding the trajectory matrix.  No generic RECOVAR full or long test
suite ran.

## Iteration-2 qualified-M-step propagation gate

H100 Slurm `12920824` reran the complete 200-particle iteration-2 cutoff/noise
panel at accepted head `25d7b6db0`, against the unchanged frozen native RELION
data and component table.  It completed `0:0` in 61 seconds with 3.2 GB peak
RSS.  Preflight `12920781` failed closed in two seconds because the submitted
full Git SHA was mistyped; no science ran in that attempt.

The source-faithful native-SGD body does not by itself close the propagated
state.  At shell 15, direct-residual signed error is
`+1.5975690968e-5`, versus the prior `+1.60060e-5`; `AA` and `XA` errors are
`-3.8716139640e-6` and `-9.9874876848e-6`.  Inferred image-power error is only
`-1.1731079750e-7`.  Thus the iteration-2 failure remains the reference-driven
`XA/AA` boundary identified earlier, despite the now-qualified per-particle
posterior and production GPU BPref arithmetic.

The remaining implementation difference is aggregate launch topology.  Native
RELION launches `orientation_num` blocks for each particle.  RECOVAR's current
FFI launches the bucket-padded `rotation_count` for every particle and masks
inactive rows by zero posterior; for the qualified particle-0 capture this is
80 active rotations in a 1,024-row padded axis.  That produces the correct
isolated contribution but changes block scheduling during full-particle atomic
accumulation.  The next bounded discriminator passes exact per-particle
rotation counts to the otherwise unchanged 523-line-matched kernel and repeats
the iteration-1 aggregate plus iteration-2 cutoff gates.  It will receive no
200-iteration trajectory unless those gates materially improve.  No generic
RECOVAR suite ran.

## Exact per-particle rotation-count discriminator

Isolated commit `90fa08f16` changes only the source-faithful FFI launch grid:
each particle launches its number of packed active rotations instead of the
bucket-padded rotation axis.  The 523-line-matched SGD kernel body, posterior,
projector, translations, accumulator layout, and reconstruction are unchanged.
H100 build/test job `12921237` completed in 52 seconds with all 15 CUDA
translation tests passing.  Focused job `12921329` additionally proved that
rows beyond the supplied particle count are excluded from both scatter and
denominator.  Build attempt `12921218` used a relative Python path from
`make -C` and test-selection attempt `12921308` matched no tests; both failed
before scientific execution.

The scientific result is negative.  Matched iteration-1 job `12921404`
completed in 50 seconds.  Raw accumulator relative L2 is
`8.72817e-6`/`1.10321e-5` for data and
`2.08442e-6`/`2.24206e-6` for weight; reconstructed-reference relative L2
worsens to `2.67134e-6`.  Frozen-native iteration-2 job `12921405` completed
in 63 seconds.  Shell-15 direct-residual error changes from
`+1.5975691e-5` to `+1.5874443e-5`, only a 0.63% improvement; `AA` and `XA`
remain `-3.8716367e-6` and `-9.9874480e-6`.

Exact active grid size is therefore rejected as the remaining propagated
reference cause.  The experimental commit remains unpushed and receives no
200-iteration trajectory.  Reports:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_exact_counts_mstep_90fa08f1_20260825/analysis/mstep_boundary.json`
  (SHA-256
  `1e59a7202c8d3eacb72371c884bb0840cb5faadfc30875d5b7b8e02f459c5766`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_exact_counts_it02_90fa08f1_20260825/analysis/cutoff_particle_panel.json`
  (SHA-256
  `6a89701254e9d357097dfab08524e2edaee3a2dad95698fa4deb0956397fbe13`).

The next bounded gate must preserve physical particle order and one shared
accumulator across RECOVAR's bucket/FFI boundaries, separating outer launch
grouping from the now-rejected per-particle grid size.  No generic RECOVAR
suite ran.

## RELION pool-3 stream-topology discriminator

Source inspection closes RELION's outer CUDA schedule.  With the GUI-default
`--pool 3 --j 8`, `expectationSomeParticles` gives the three particles in one
pool to one-particle OpenMP tasks.  Each worker owns a CUDA class stream and
launches `runBackProjectKernel` into the `MlDeviceBundle`'s shared
BackProjector.  The worker synchronizes its class stream after the particle;
the OpenMP barrier then closes the three-particle pool.  RELION therefore has
three potentially concurrent particle kernels, shared float32 atomics, and a
barrier after each group of three.

Isolated, unpushed commit `431083949` reproduces that schedule inside each
candidate FFI call with three ordinary CUDA streams.  It also lets the
fail-closed Slurm gates reuse a separately qualified Pixi environment, without
changing VDAM defaults or science.  H100 build/test job `12922230` completed
in 55 seconds with 1.74 GB peak RSS and all 15 focused CUDA translation/VDAM
tests passing.  Preflight jobs `12922443` and `12922444` failed closed in two
seconds because their expanded Git SHA was mistyped; they produced no science.

The correctly pinned iteration-1 M-step job `12922464` completed in 51 seconds
with 2.71 GB peak RSS.  Pool-3 streams improve raw accumulator relative L2 to
`7.24985e-6`/`8.52167e-6` for data and
`1.74336e-6`/`2.15094e-6` for weight.  The reconstructed-reference relative L2
improves to `1.41860e-6`.  Independent native RELION M-step captures differ by
`8.64718e-6`/`9.52974e-6` for data,
`2.18868e-6`/`2.38532e-6` for weight, and `1.66747e-6` after reconstruction.
The candidate aggregate and reconstructed map are therefore already inside
the measured native aggregate repeat envelope; bitwise aggregate equality is
not a valid stock-RELION target.

The propagated-state gate is nevertheless negative.  Iteration-2 job
`12922463` completed in 61 seconds with 3.22 GB peak RSS.  Its shell-15 direct
residual error is `+1.5958762627e-5`, essentially unchanged from the accepted
`+1.5975690968e-5`; `AA` and `XA` remain
`-3.8717364616e-6` and `-9.9876405570e-6`.  By contrast, two native RELION
iteration-2 captures differ by only `2.18058e-7` relative L2 in direct
residual and retain identical hard poses/translations.  The structured
candidate reference-projection error is therefore not explained by native
aggregate nondeterminism.

Pool-3 streams alone are rejected and receive no 200-iteration trajectory.
The remaining source-topology mismatch is now more specific: RECOVAR executes
pseudo-halfsets as two separate E-steps, while RELION processes the globally
shuffled particle stream once and routes each particle to one of two shared
BackProjectors by `part_id % 2`.  The next bounded gate must keep global
particle/pool order while atomically routing into two halfset accumulators;
only then can pool boundaries also remain continuous across bucket/FFI
boundaries.  Reports:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_pool3_mstep_43108394_20260825/analysis/mstep_boundary.json`
  (SHA-256
  `6370a7041aa70daa8c1214ac6c2611331d43d5258663b15d683543ecc27cd7bc`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_pool3_mstep_43108394_20260825/analysis/native_repeat_cross_candidate.json`
  (SHA-256
  `cc631c1e22614aba7da7279188bfc5298c7dbd0fe783552d2edceacb2bff1728`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_pool3_it02_43108394_20260825/analysis/cutoff_particle_panel.json`
  (SHA-256
  `0da4e9f372cb44df3dc754a22ef1748c2f84dc7408019ec0e79de9d0f5b81be1`).

No generic RECOVAR full or long test suite ran.

## Joint pseudo-halfset particle-stream discriminator

Isolated, unpushed commit `650c8d54b` replaces RECOVAR's two serial
pseudo-halfset E-steps with one globally ordered K=1 exact E-step.  Particle
half ids travel through the existing local scorer and are consumed only by the
source-faithful CUDA BPref boundary, which routes each particle's atomic
updates into one of two grouped accumulators.  Pool-of-three chunk boundaries
are kept continuous by forcing one bucket shape and rounding every non-tail
FFI particle chunk down to a multiple of three.  Other engines and K>1 remain
fail-closed on the old path.

Focused H100 build/test job `12924054` completed in 2 minutes 52 seconds with
1.98 GB peak RSS.  All 19 selected source, CUDA, group-routing, adapter, and
pool-boundary tests passed in 6.16 seconds.  The grouped CUDA output agrees
with separate one-half launches within the native float32 atomic tolerance;
the qualified CUDA SHA-256 is
`6e29e0cf2d6f880928493893fede6445bcd4520e9ecc68eae7e297865ef613de`.
Build preflight `12924013` failed before compilation because `nvcc` was absent
from the compute-node PATH; the runner now pins the qualified CUDA 13.1
compiler.  Science preflight `12924228` likewise failed in two seconds before
GPU work because the new disposable output root had not yet received its
mandatory `SAFE_TO_DELETE` marker.

The correctly pinned iteration-2 science job `12924274` completed in 59
seconds with 3.13 GB peak RSS.  Both iteration-1 and iteration-2 metadata
confirm `joint_halfset_particle_stream=true`, but the propagated boundary is
unchanged:

- direct-residual signed sum error `+1.5988481367e-5`;
- `AA` signed sum error `-3.8716156432e-6`;
- `XA` signed sum error `-9.9874459921e-6`;
- inferred image-power signed sum error `-1.0443533354e-7`;
- support-mass signed sum error `+2.1182779812e-5`.

The prior pool-only direct-residual error was `+1.5958762627e-5`, so globally
interleaving pseudo-halfsets does not explain the structured iteration-2
reference error.  This topology is rejected and receives no 200-iteration
trajectory.  Its report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_joint_halfset_it02_650c8d54_20260825/analysis/cutoff_particle_panel.json`
(SHA-256
`96a15d3b63549bf2ff839241808a10d4a633c5a8455b0af78bd35198178956e5`).

The next bounded gate is shell-resolved incoming-reference localization.  The
candidate iteration-1 map is inside the global native atomic repeat envelope,
yet native iteration-2 cutoff components are much more repeatable than the
candidate error.  Before another production change, compare candidate/native
and native/native reconstructed references and texture-projector samples by
Fourier shell, with special attention to the iteration-2 cutoff shell and the
eight already-qualified largest `XA` contributors.  This will distinguish a
low-shell reconstruction/layout defect from harmless high-shell atomic
variation.  No generic RECOVAR full or long test suite ran.

## Rounded-shell Wavg reconstruction-support acceptance

The shell-resolved audit found a deterministic boundary mismatch hidden by
global map norms.  At iteration-2 `current_size=30`, RELION's Wavg/noise loop
issues the full FFTW rectangle and accepts pixels whose *rounded* shell is at
most 15.  RECOVAR projected and reconstructed only the exact Euclidean disk.
On the 32-pixel particle grid, exact support contains 355 pixels while rounded
support contains 372; 17 of RELION's 39 shell-15 pixels were therefore issued
with image power but zero candidate `XA/AA` reference terms.

The corrected top-eight captured-particle decomposition ran as H100 Slurm
`12925327` in 15 seconds.  The omitted rim explains the prior reference error
to numerical closure: summed `XA` error is `-2.2125492e-6`, with a
`-2.2125974e-6` rim term; summed `AA` error is `-5.6053194e-7`, with a
`-5.6055199e-7` rim term.  Posterior and inside-exact-disk reference terms are
only order `1e-11`.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_joint_halfset_it02_650c8d54_20260825/analysis/rounded_rim_decomposition_top8_v2.json`
(SHA-256
`9615da6a7f98e0bb09f91aa77ea585f701bd3ec4040eb3e5ba8c8780dab7d4f3`).

Experimental commit `2a4ccbeeb` propagates InitialModel's already-declared
`recon_exact_radius=False` contract into the exact-local Fourier window.  The
rounded pixels now participate in projected-reference residual and Wavg/noise
accounting; the RELION-style CUDA BackProjector independently retains its
exact rotated 3-D insertion cutoff, so the outer rim is not inserted into the
volume.  A direct part-5 projector audit at experimental head `9b0735f1a`
matches RELION's complete 372-pixel projection at `2.20831e-9` relative L2.
Restoring the 17 pixels cancels the particle's `XA` defect from
`-1.1806893e-6` to `-1.48614e-11`, and its `AA` defect to `+3.36908e-12`.
Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_joint_halfset_it02_650c8d54_20260825/analysis/rounded_projection_part005.json`
(SHA-256
`4f8d2c0710c85d64612978e277ab5f990f7ca48806da90a291bb80d667cd55b0`).

Focused H100 build/test job `12925660` completed in 2 minutes 51 seconds with
2.04 GB peak RSS and all 22 selected Fourier-window, Wavg, adapter,
joint-halfset, pool-boundary, and CUDA routing tests passing.  The full frozen
200-particle iteration-2 gate `12925793` completed in 58 seconds with 3.14 GB
peak RSS and 55 seconds of RECOVAR wall time.  It removes the propagated
reference defect:

- direct-residual signed error: `+7.7336232e-9`, down from
  `+1.5988481e-5` (about 2,067-fold);
- `XA` signed error: `+1.6671798e-11`, down from `-9.9874460e-6`;
- `AA` signed error: `+2.9276316e-10`, down from `-3.8716156e-6`;
- inferred image-power signed error: `+1.7833844e-8`;
- support-mass signed error: `+1.9669235e-5`.

The production report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_rounded_rim_it02_9b0735f1_20260825/analysis/cutoff_particle_panel.json`
(SHA-256
`69f9ca3e7ae543ab4f5e0d285f3182b1807eebe63bf5d03e10736394157a358f`).
This causal gate is accepted.  It justifies the next iteration-1 accumulator
boundary check and a true 200-iteration K=1 trajectory.  The remaining small
particle-wise residual/image-power and support-mass differences stay tracked;
they are not being declared exact parity yet.  No generic RECOVAR full or long
test suite ran.

## First rounded-rim 200-iteration trajectory

The first true 200-iteration GUI-default baseline ran at experimental head
`78cdc97ed` as H100 Slurm array task `12926196_1`.  Both engines completed,
but the strict trajectory audit failed after 1 hour 7 minutes; peak batch RSS
was 11.00 GB.  Native RELION required about 14 minutes, while RECOVAR required
about 48 minutes, so runtime parity is not yet achieved.  Cases 2--22 remain
held: no outlier, alternate pose/noise distribution, or scale trajectory is
being released until this baseline transition is causally closed.

The rounded-rim correction keeps maps numerically equivalent through a long
prefix.  Cross-engine FSC AUC is `0.999999999939` at iteration 2, remains near
one through iteration 69, and is `0.999999758` at iteration 85.  The first
particle-state difference occurs at iteration 24 for exactly one particle
(`1172@particles.128.mrcs`, RECOVAR row 193): only its psi child differs by
`3.75000187` degrees, with a `2e-6` pmax difference; it reconverges at
iteration 25.

The consequential divergence localizes to the adaptive sampling transition:

- iterations 82--87 have no divergent particles;
- iterations 88 and 89 have two divergent particles each;
- iteration 90 has 440 divergent particles, pose match `0.989`, and
  translation match `0.8533`;
- RECOVAR iteration 90 changes from 116 to 52 translations and records
  `sampling_updated=true` at `current_size=64`, Healpix order 3;
- the first cross-engine FSC gate failure is iteration 92, at
  `0.9988480497` versus the `0.999` threshold;
- minimum cross-engine FSC AUC is `0.9170677807` at iteration 141, while the
  minimum RECOVAR-minus-RELION ground-truth FSC delta is `-0.0029190173`;
- terminal iteration 200 cross-engine FSC AUC is `0.9845245403`, with
  ground-truth FSC delta `+0.000196409`.

Thus scientific quality versus ground truth often stays close, but exact
trajectory parity fails and the current RECOVAR path is about 3.4 times slower
than native for the 200-iteration engine phase.  The next bounded gate compares
only RELION and RECOVAR sampling metadata at iterations 86--92 (offset range,
offset step, angular order/step, perturbation, and translation count), then
tests the transition without rerunning all 200 iterations.  Repeated JIT shapes
and slower late-size steady-state kernels are separately tracked for runtime.

Artifacts:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_full_rounded_78cdc97e_20260825/vdam-gf01/trajectory_audit.json`
  (SHA-256
  `e828f09b6fa136b5effdd0f0d3af69e662e103e999aba944361af1c8f95f7f45`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_full_rounded_78cdc97e_20260825/vdam-gf01/particle_state_trajectory_audit.json`
  (SHA-256
  `a7a388630288ef71eb548c5ad598526750543e46c3a7cb8cae417c9a6c68765e`).

No generic RECOVAR full or long test suite ran.

## Adaptive-sampling expected-accuracy discriminator

The iteration-90 transition failure is a crossed-parameter defect, not random
trajectory drift.  RELION and RECOVAR agree through iteration 89 on Healpix
order 3, psi step 7.5 degrees, translation range `9.002823` Angstrom, coarse
translation step `3.0` Angstrom, and every sampling perturbation.  At iteration
90 RELION changes to range `6.482290` Angstrom and step `2.575500` Angstrom;
RECOVAR changed its range to `6.487914` Angstrom but incorrectly retained the
`3.0` Angstrom step.

The expected-accuracy inputs identify why.  RELION's
`calculateExpectedAngularErrors` divides by the separate, fixed
`sigma2_fudge=1`.  RECOVAR instead passed the dynamic reference-regularization
`tau2_fudge_factor`.  This is nearly invisible while tau2 fudge is one, then
amplifies with the VDAM schedule: native tau2 fudge is `1.854242` at iteration
70, `3.821947` at iteration 80, and `3.995253` at iteration 90.  Accordingly,
RECOVAR's iteration-90 accuracy estimate was `(3.730, 3.281 Angstrom)` versus
RELION's `(1.823, 1.717 Angstrom)`.

The saved-artifact source-boundary replay with the correct independent noise
fudge exactly reproduces RELION at iteration 70: rotation accuracy `4.630` and
translation accuracy `3.48075` Angstrom.  Replaying RECOVAR's saved iteration-89
state gives rotation accuracy `1.828` (native `1.823`) and translation accuracy
`1.717` Angstrom exactly.  The latter yields RELION's coarse translation step
`min(1.5, 0.75 * 1.717) * 2 = 2.5755` Angstrom.  Both the native range and the
slightly drifted saved-candidate range select the same 21 coarse / 84 fine
translations at that corrected step.

Experimental commit `82370f0f5` separates the two parameters, adds a direct
saved-artifact expected-accuracy auditor, and guards against reusing dynamic
tau2 in the accuracy binding.  All 53 focused native InitialModel driver tests
pass; no generic RECOVAR full or long test suite ran.  The iteration-90 report
is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_full_rounded_78cdc97e_20260825/vdam-gf01/analysis/expected_accuracy_sigma2_boundary_it090.json`
(SHA-256
`937f5f3844ad2ae44463db65a28d5c59c433e6fd0b5acd387d5383ced2caa0df`).

Targeted H100 Slurm `12929393`, at experimental runner head `c7581d0fd`, is the
production gate.  It runs only RECOVAR through iteration 92, reuses the
existing native trajectory and compilation cache, and audits maps plus
particle states across iterations 69--92.  Preflights `12929306` and `12929363`
failed in two seconds before computation because of, respectively, an
incorrectly expanded commit SHA and an unpadded iteration filename in the new
runner; both are infrastructure-only failures with no science output.  The 21
remaining 200-iteration robustness cases stay held until this transition gate
passes.

## Live RFLOAT accuracy boundary

H100 Slurm `12929393` completed the 92-iteration RECOVAR trajectory in 11
minutes 25 seconds with 7.86 GB peak step RSS; the batch was marked failed only
after computation because the reused permutation auditor unnecessarily
rejected `K=1`.  The corrected auditor was run against those fixed artifacts,
with a focused unit regression for its 1-by-1 assignment/FSC path.

The independent-noise-fudge correction takes the right iteration-90 topology:
range `6.48229004175` Angstrom, 21 coarse / 84 fine translations, and
`sampling_updated=true`.  The signed shellwise FSC gate passes every requested
checkpoint from iteration 69 through 92, with minimum FSC-AUC
`0.999125104061` at iteration 92 and class-assignment accuracy one.  Map report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_sampling_sigma2_c7581d0f_20260825/analysis/sampling_transition_maps.json`
(SHA-256
`59e9f226eee5bcd6f591b775047a2d0068e08fd80f78b567fd2aeb56d83026ef`).

The particle gate does not pass.  Live RECOVAR estimated translation accuracy
`1.734` Angstrom and therefore used step `2.601` Angstrom, versus RELION's
`1.717` and `2.5755`.  All 440 visited iteration-90 particles consequently
move onto the neighbouring translation grid; translation-match fraction is
`0.853333`, falling to `0.848` at iteration 92.  Particle report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_sampling_sigma2_c7581d0f_20260825/analysis/sampling_transition_particles.json`
(SHA-256
`346e87b2c06c3e2483e527a1dcceb2e44ede4754a8b9f32cdeb4b41fb9e6568e`).

The same iteration-89 state replayed from its serialized float32 MRC gives
RELION's `(1.823, 1.717 Angstrom)` exactly, while the live in-memory float64
reference gives `(1.824, 1.734 Angstrom)`.  The enriched discriminator records
both rather than mistaking serialized replay agreement for live agreement:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_sampling_sigma2_c7581d0f_20260825/analysis/expected_accuracy_it090.json`
(SHA-256
`278a36653d26a645d94621a4bacfac5dcca8e08c84d47f43319c9067d379ca9c`).

Experimental commit `acc4929f0` now crosses RELION's RFLOAT/float32 reference
boundary before the double-argument expected-accuracy replay binding.  The 58
focused native-driver and trajectory-auditor tests pass.  Causal H100 rerun
`12930126` is in progress through iteration 92 and reuses the same native
trajectory.  It must reproduce accuracy `(1.823, 1.717 Angstrom)`, step
`2.5755`, map FSC-AUC at least `0.999`, and the native particle translation
grid before any 200-iteration rerun or robustness-case release.  No generic
RECOVAR full or long test suite ran.

The RFLOAT-reference hypothesis is rejected.  Slurm `12930126` completed in 6
minutes 41 seconds of RECOVAR wall time (6 minutes 50 seconds including audits,
7.49 GB peak step RSS) and reproduced the prior live accuracy
`(1.824, 1.734 Angstrom)`, step `2.601`, and iteration-90 translation-match
fraction `0.853333`.  Its minimum map FSC-AUC remains passing at
`0.999124612323`, so this is a particle-grid failure rather than a gross map
failure.  The float32 cast is removed at experimental head `c811548ef`.

That head adds an opt-in exact live-operand capture for the expected-accuracy
binding and extends the serialized replay report with operand-by-operand exact
and maximum-absolute comparisons.  H100 Slurm `12930466` is the next bounded
gate.  It reruns only through iteration 92, dumps the iteration-90 in-memory
reference, Euler angles, particle IDs, classes, noise spectrum, optics vectors,
and scalar binding inputs, then replays them from disk.  The first differing
operand will determine the next production correction; no 200-iteration or
robustness trajectory is released on this diagnostic alone.

The capture identified the omitted live argument.  RELION seeds every
expected-accuracy trial with Experiment's internal `part_id`; for this sorted
default stream those are `0..99`.  RECOVAR passed no explicit seed ids, so the
binding fell back to original dataset rows `0, 9, 99, 999, ...`.  Supplying
internal part ids makes the exact captured live operands replay to the same
accuracy as the serialized path.  Experimental commit `0bd239f19` propagates
the parallel stored RELION part-id stream, including shuffled nonzero-seed
subsets, and records it in metadata and diagnostic captures.  The focused test
now checks the actual `sigma2_fudge` argument (index 18), rather than
accidentally checking the fixed interpolator argument at index 17, and checks
the independent internal seed-id vector.  All 58 focused tests pass.

Slurm `12930466` also exposed a separate repeatability failure before the
sampling transition.  Although it ran the same scientific source on the same
physical H100 as `12929393`, one borderline particle first selected a different
pose/translation at iteration 19.  Five particles differed at iteration 20,
and feedback grew the map relative-L2 difference from `2.53e-6` at iteration
19 to `2.77e-3` at iteration 20 and `6.97e-2` at iteration 69.  Cross-engine
FSC-AUC was therefore already `0.986749` at iteration 69 and the map gate
failed, independently of the iteration-90 seed fix.  The exact input capture
is preserved at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_sampling_live_inputs_c811548e_20260825/accuracy_inputs/iter090_expected_accuracy_inputs.npz`
(SHA-256
`e5d7a9f5e73f8917e9f9440717159a08663366fa69e5d491c06d6dfc0caba72c`).

Three post-fix 92-iteration repeats are running as H100 Slurm `12930904`,
`12930905`, and `12930906`.  They gate both the corrected live sampling seed
contract and run-to-run stability.  A single passing repeat is insufficient:
the transition fix must reproduce across the panel, and the pre-transition
iteration-19 branch sensitivity must be bounded or removed before the full
200-iteration baseline is released.  No generic RECOVAR full or long test
suite ran.

All three post-fix repeats pass.  They completed in 6 minutes 30--33 seconds of
RECOVAR wall time and 6 minutes 44--47 seconds including audits, with 7.13--7.15
GB peak step RSS.  Every run reproduces native expected accuracy exactly at
all exposed long boundaries: iteration 70 is `(4.630, 3.48075 Angstrom)`,
iteration 80 is `(1.721, 1.60225 Angstrom)`, and iteration 90 is
`(1.823, 1.717 Angstrom)`.  Iteration 90 also matches range
`6.48229004175` Angstrom, step `2.5755` Angstrom, and 84 fine translations.

The three map gates pass through iteration 92 with minimum FSC-AUC
`0.999170286449`--`0.999170740520` and assignment accuracy one.  The old
systematic grid split is removed: iteration-90 divergent-particle count falls
from 440 to 15, translation match rises from `0.853333` to `0.996`, and
iteration-92 translation match is `0.985333`--`0.985667`.  The residual 49--50
iteration-92 particle differences are on the common native grid and remain a
repeatability/near-tie boundary for the full trajectory, not a scheduler
defect.  Per-repeat map-report SHA-256 values are, in repeat order,
`563512ade05a490515e8e3c26857bbde1c9f709ae8e5bbac4830b0dfff90ff87`,
`2a977c266c722fe583fde108d7a5d5cb138f0c5801efd406722c4b9d265367e3`,
and `9f368b2cccc492c2921b3a19743cdd088e9dee00eb3ed5f6d08c94eb51002f7b`.

This accepts the sampling transition and justifies one RECOVAR-only
200-iteration completion gate against the existing native trajectory.  Slurm
`12931162`, at experimental head `c4aabc839`, is running with opt-in per-stage
profiles and all 201 map and particle checkpoints.  It preserves science and
profile evidence even if a strict audit fails.  The profiler reports schedule,
subset, projector refresh, expectation/pass-1/pass-2, M-step, state update,
and artifact-boundary wall time by early, transition, and adaptive/late phase.
The 22-case robustness matrix remains held pending the complete baseline and
runtime result.  No generic RECOVAR full or long test suite ran.

## Profiled 200-iteration completion gate

The corrected iteration-90 seed contract is necessary but not sufficient for
the complete trajectory.  Focused H100 Slurm `12931162` completed all 200
RECOVAR science iterations at experimental head `c4aabc839` in 32 minutes
42.87 seconds, with 9.62 GiB peak step RSS reported by Slurm (`10083596K`).  The
batch then failed only its strict parity audits, as intended.  This is faster
than the earlier approximately 48-minute candidate run but remains about 2.3x
the approximately 14-minute native RELION baseline, so runtime parity is not
accepted.

The all-checkpoint map gate is effectively exact through iteration 89:
iteration-89 FSC-AUC is `0.999999995229`.  Divergence begins on the corrected
fine grid at iteration 90 (`0.999806085574`), first crosses the `0.999` gate at
iteration 93 (`0.998852359624`), reaches its minimum at iteration 134
(`0.938214029885`), and ends at only `0.987099107010` at iteration 200.  The
report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_full_seed_profile_c4aabc83_20260825/analysis/full_trajectory_maps.json`
(SHA-256
`50dfc50c6602ee143a41724e795c3c8499a107e6ab7d30f1ff5a584e2c0c942a`);
its shell archive SHA-256 is
`b95d88a2f0c5b30845127eedcc76a1024b6578f5758c9a9083cad86917c20a08`.

The post-hoc particle audit starts at iteration 1 because RELION's input-state
iteration 0 has no posterior-Pmax column.  It finds one low-Pmax pose split at
iteration 24, no divergent particles at iteration 89, then 15 at iteration
90, 49 at iteration 92, and 520 at iteration 100 after the first differing
translation grid.  At iteration 200 all 1,000 visited particles differ;
pose-match fraction is `0.950667` and translation-match fraction is
`0.666667`.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_full_seed_profile_c4aabc83_20260825/analysis/full_trajectory_particles.json`
(SHA-256
`45316c9b53d536a32626e728c13172ca2c3e8d3b30e9e660d2a4366be15da918`).

A new full adaptive-sampling auditor makes the downstream control failure
explicit.  HEALPix order and random perturbation remain exact for all 200
iterations.  Offset range and step first differ at iteration 100, translation
topology first differs at iteration 110, and the update/no-update decision
first differs at iteration 140.  At iteration 170 RECOVAR uses range
`3.202826` Angstrom, step `1.549125` Angstrom, and 52 fine translations,
where RELION retains range `4.109667` Angstrom, step `1.447125` Angstrom, and
the larger topology.  The first geometry error is already explained by the
iteration-99 hidden-variable change: RECOVAR records `1.182642293` Angstrom
versus RELION's `1.183376` Angstrom.  The schedule is therefore a downstream
state symptom and must not be forced from oracle values.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_full_seed_profile_c4aabc83_20260825/analysis/full_sampling_trajectory.json`
(SHA-256
`01405d051bc072fd7c767ee525eba86d64687ab37b03ca36c27ad5a46cd233fe`).

The profile assigns 1,883.88 of 1,925.00 measured pre-artifact seconds to the
expectation step.  Sparse coarse pass 1 costs 1,212.42 seconds, sparse fine
pass 2 costs 555.82 seconds, and the M-step only 29.45 seconds.  In iterations
90--200, mean expectation time is 14.04 seconds: pass 1 averages 9.40 seconds
and pass 2 averages 3.98 seconds.  Within pass 2, the combined big-JIT bucket
time is 302.49 seconds; backprojection, packing, and raw-cache construction
cost 56.64, 42.51, and 41.01 seconds respectively.  Runtime work must therefore
target pass-1 scoring first and pass-2 big-JIT buckets second, not the M-step.
The profile is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_full_seed_profile_c4aabc83_20260825/analysis/iteration_profile_summary.json`
(SHA-256
`53c742dd268543ffe04bf01b6dc18e2419dea753ddac71b7cd631dfd6240f562`).

Experimental commit `815d4a0e4` adds the full schedule auditor, two focused
unit regressions, excludes the structurally unauditable iteration-0 particle
state from the full runner, and preserves schedule status independently of
map, particle, and profile status.  Commit `581e498b2` makes single-particle
native continuation captures target-scoped and pins them to the same H100
class as the accepted trajectory.  All 62 focused native-driver and audit
tests pass.  H100 Slurm `12932544` is capturing the iteration-90 fused
posterior and exact score operands for the first strongly divergent particle,
`1010@particles.128.mrcs` (original row 1009, RELION internal part-id 14).
Infrastructure-only attempts `12932098`, `12932360`, and `12932421` stopped
before RECOVAR science because of, respectively, script import context, a
mistyped pinned commit, and an invalid whole-subset assertion after the native
capture deliberately stopped at the target particle.  No generic RECOVAR
full or long test suite ran, and all 21 additional 200-iteration robustness
cases remain held.

## Iteration-90 local-prior transition boundary

The target-scoped continuation capture completed its science on H100 Slurm
`12932544`.  Its final batch status is infrastructure-only: a test-only commit
moved the experimental checkout after the job pinned its source head, so the
post-run immutable-head assertion failed even though the relevant production
source did not change.  The captured particle is
`1010@particles.128.mrcs` (original row 1009, RELION internal part-id 14).
Native target replay Pmax is `0.952955`, within `2.41e-4` of the sealed native
trajectory value; RECOVAR remains on its accepted approximately `0.4969`
branch.

The fused fine-posterior audit proves that this is not only a shared-fine-score
rounding defect.  Native retains 64 fine candidates, whereas RECOVAR routes
32 parents across 84 translations.  The native best mapped key is `(25, 41)`
at probability `0.952955`; RECOVAR selects `(18, 41)` at `0.496859`, and only
`0.48747` of RECOVAR probability lies on native support.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_it090_first_particle_581e498b_20260825/analysis/fused_posterior.json`
(SHA-256
`0ec6fbc3961290394c6915d700fd92b240df99ad5e0bfaa4c42b4854a92b2e05`).

The translated-image, score-weight, noise, and shared fine-score operands are
already close enough that they cannot explain the support split.  Image and
score-weight relative-L2 are approximately `4.9e-7` and `3.33e-7`; substituting
RECOVAR's live projected reference reproduces the remaining shared-support raw
score residual (RMS `0.057119`, maximum `0.123916`).  The iteration-89 MRCs
differ by only `4.68026e-5` relative-L2, but projection amplifies that to
`5.346e-4`.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_it090_first_particle_581e498b_20260825/analysis/native_translation.json`
(SHA-256
`c4a71e9f63dbae37d9ffe2e29e20b52e41859a5b4b184902dc61751b123e3da9`).

A verbose coarse capture then isolated the discrete cause.  Of three same-head
H100 repeats, only `12933447` followed the accepted pre-transition branch;
`12933142` and `12933448` already had 417 iteration-89 particle differences
and map minima near `0.7405`, so their dumps are quarantined.  The accepted
coarse dump is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_it090_coarse_02a8583d_r2_20260825/significance/significance_orig001009_it090_cs064.npz`
(SHA-256
`5dc56cd498db0d647fd35e58c8cada3568d87b3a476e474bb5e86aa7982a222c`).

On that fixed state, native RELION has exactly two significant coarse
hypotheses, `(31098, 10)` and `(31385, 10)`, and selects `(31098, 10)`.
RECOVAR has five, adding `(31336, 10)`, `(31337, 10)`, and `(31385, 11)`, and
selects `(31385, 10)`.  The centered raw-score residual is only `0.059668` RMS,
and is `0.00684`/`0.01843` at the two native hypotheses; it cannot cause the
approximately `6.25` log-weight winner reversal.  The rotation prior does:
native's live GPU orientation log prior is uniform at `-10.5149908066`, while
RECOVAR applies learned values `-6.15377665` at rotation 31098 and
`+0.08624665` at 31385.  Translation priors already match.

The corresponding RELION state transition is explicit in the frozen output:
`rlnOrientationalPriorMode` is `0` through iteration 89 and becomes `1` at
iteration 90 while HEALPix remains 3.  Its stored angular-prior widths are
zero, so the live operands are uniform direction and psi priors
(`1/768 * 1/48`).  RECOVAR incorrectly continued to score with the nonuniform
`pdf_direction`.  Replaying only the accepted RECOVAR coarse table with a
uniform rotation prior restores RELION's exact winner and two-hypothesis
support; learned-prior Pmax is `0.953176`, while the uniform-prior replay Pmax
is `0.959020`.

Experimental commit `41a773cf1` encodes the gradient InitialModel local-prior
transition as persistent sampling state and switches the E-step rotation prior
to uniform at that boundary.  Six focused transition/prior/E-step plumbing
tests pass, along with Ruff, py_compile, and diff checks.  The commit remains
unpushed.

Three independent H100 iteration-92 gates completed as Slurm `12934827`,
`12934828`, and `12934829`.  The two runs that enter iteration 90 on the
accepted state close the systematic transition failure:

- `12934827` has zero divergent particles at iterations 89, 90, and 92;
  minimum transition FSC-AUC is `0.999999908564`;
- `12934829` has zero at iterations 89 and 90 and one near-tie particle at
  iteration 92; minimum transition FSC-AUC is `0.999999987632`.

Their respective map-report SHA-256 values are
`0a54e65c5b14fc30872c0a8de14edb5901d65c2326c2eee7c0dfaf74f3ddf888`
and
`a3e94fe0e7820431aadc692f1508ec8acd9b79a6b4754fd956a135d05e43f252`;
particle-report SHA-256 values are
`9c764920b5f1b13368389b2789c58848dbf5ab207e4996a130d28ff0feb75dc6`
and
`4d9dddbfecacb759a3d0a07f953fb5ff4228da33d45792818d4774edb8f4261a`.
This accepts the local-prior correction for its causal boundary.

Repeat `12934828` independently reproduces the unresolved earlier branch:
one particle first differs at iteration 19, five differ at iteration 20, and
417 already differ at iteration 89.  Its minimum transition FSC-AUC is
`0.740519727593`.  The first particle is
`1037@particles.128.mrcs` (RECOVAR sorted row 43): at iteration 19 it differs
by `7.527689` degrees and `1.5` Angstrom while Pmax differs by only `4e-6`.
The post-hoc all-iteration particle report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_it092_prior_41a773cf_r2_20260825/analysis/pretransition_particles.json`
(SHA-256
`de6f4a2e96a2cf334ad576ec3c0ec233a9ed4afe59bf37ca0a074b92057df5e4`).
The next bounded gate captures this particle's iteration-19 coarse/fine score
boundary across repeated H100 runs, accepting only repeats whose first 18
iterations remain exact.  The full 200-iteration rerun and all 21 additional
outlier/noise/pose/scale trajectories remain held, and no generic RECOVAR full
or long suite ran.

Experimental commit `011e8e639` adds that bounded gate without changing the
science path.  `AUDIT_MODE=pretransition` preserves the 200-iteration schedule
but stops after iteration 19, audits all map checkpoints 0--19 and particle
states 1--19, and skips iteration-90-only expected-accuracy capture.  The
adaptive-sampling auditor now also checks the persistent
`rlnOrientationalPriorMode` state that closed the accepted iteration-90 cause.
Three focused sampling-auditor tests and the new runner-contract test pass,
along with Ruff, shell syntax, py_compile, and diff checks.  The experimental
commit remains unpushed.

Six H100 repeats are submitted as Slurm `12935875`--`12935880`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_it019_repeat_011e8e639_rN_20260825`.
Each records the production fused local posterior and coarse significance
operands for original particle row 1036 at iteration 19/current size 30.  The
panel is designed to capture both the accepted and divergent sides of the
approximately `4e-6` Pmax boundary so their incoming iteration-18 continuous
states and fixed-particle scores can be compared directly.  The 200-iteration
gate and 22-case matrix remain held pending this classification; no generic
RECOVAR full or long test suite is being run.

All six bounded captures completed successfully in 63--74 seconds.  Every map
prefix passes with minimum FSC-AUC `0.9999999999168`--`0.9999999999190`.
Repeat 1 independently reproduces the one-particle iteration-19 split, while
repeats 2--6 have no discrete particle differences through iteration 19.  The
panel therefore contains both sides of the boundary under identical code,
input, H100 class, and diagnostics.

The captured cause is an exact float32 fine-posterior tie, not different
candidate support.  All six runs retain the same 48 local rotations, 116
translations, and 185 reconstruction samples.  In repeat 1, global rotations
7585 and 8642 both serialize at probability `0.0630029514432`, so the earlier
local flat index 635 wins.  In the five matching repeats, rotation 8642 wins
by only `1.92e-6`--`5.77e-6` probability.  Bad-versus-good coarse raw-score
RMS is `7.70e-5`, but an independent good-versus-good pair is slightly larger
at `8.64e-5`.  Likewise, all iteration-18 candidate maps differ from one
matching repeat by `1.68e-6`--`2.04e-6` relative L2; the divergent repeat is
not an outlier at `1.88e-6`.  This rejects a new deterministic scoring or
support formula defect at iteration 19 and localizes the branch to the
already-measured CUDA atomic repeat envelope.

Experimental commit `2558d79f9` adds an observer-free mode to the existing
stock-RELION continuation repeat panel; its focused contract test and shell
syntax check pass, and it remains unpushed.  Twelve same-GPU RELION
continuations from the exact sealed iteration-18 optimiser are running as
H100 Slurm `12936211`, with all capture observers disabled.  This directly
tests whether stock RELION itself preserves or flips the same iteration-19
near-tie from a fixed incoming state before choosing the acceptance rule for
long trajectories.  No generic RECOVAR suite ran.

The 12 observer-free continuations complete in 30 seconds total and are
repeat-stable against each other: zero hard pose/translation differences and
maximum Pmax jitter `1.8e-5`.  They do **not**, however, replay the sealed
autonomous iteration.  RELION continuation reconstructs the iteration-local
1,000-particle subset/RNG stream rather than serializing and restoring its
original autonomous history; all 1,000 reprocessed particles differ from the
sealed iteration-19 state, and the target changes from fine Pmax `0.063007` to
coarse Pmax about `0.295`.  This panel is therefore retained as a continuation
repeat control but rejected as evidence about the autonomous near-tie.

Experimental commit `11b140358` adds a fail-closed autonomous native-repeat
gate.  It copies the exact sealed GUI-default RELION command, requires
`--iter 200`, pins executable SHA-256
`2d070d6456ae439c3890fcbd0f6e9c8e4e56bcc2ef79f55bfc58574e50f7a11b`,
and audits all 201 maps and all 200 posterior particle states against the
original native run.  Its focused contract test and shell syntax check pass;
the commit remains unpushed.  Four independent full native H100 repeats are
running as Slurm `12936720`--`12936723`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_native_full_repeat_11b140358_rN_20260825`.
These runs define the scientifically valid autonomous native envelope before
the RECOVAR 200-iteration and 22-case gates are released.  No generic RECOVAR
full or long suite ran.

The fresh autonomous controls already reproduce both sides of the target
boundary before completion: native repeats 1--3 retain the sealed
`(-122.42695, 30.332644, 13.941839)`-degree pose, while native repeat 4 selects
`(-117.5254, 36.313398, 6.136272)` with the same 1.5-Angstrom translation
change as RECOVAR's divergent repeat.  Native Pmax spans only
`0.063003`--`0.063007`.  The iteration-19 hard split is therefore a measured
stock-RELION autonomous mode, and a single-reference exact-particle gate is
invalid for this chaotic long trajectory.

This releases a repeat-aware full default gate, not the broad matrix.  A
four-repeat paired GUI-default 200-iteration panel is submitted as H100 Slurm
`12936969`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_default_repeat_panel_11b140358_20260825`.
Each repeat runs stock RELION and RECOVAR on the same physical GPU, audits all
201 checkpoints, retains the frozen FSC/GT thresholds, and additionally
requires every candidate run to match a native mode and every native run to
have a candidate mode.  The remaining 21 outlier/noise/pose/scale cases stay
held until this default envelope passes.  No generic RECOVAR suite ran.

Paired-panel attempt `12936969` stopped before science in three seconds.  Its
GPU/CUDA/provenance preflight passed, but the clean-environment launcher ran a
repository script by file path after unsetting `PYTHONPATH`, so the child
could not import `recovar`.  Experimental commit `ef0d5bce8` switches that
boundary to `python -m scripts.run_vdam_relion_parity_case`; a focused import
contract test and shell syntax check pass.  The corrected fresh-root panel is
running as Slurm `12937169` at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_default_repeat_panel_ef0d5bce_retry_20260825`;
it passed the GPU/CUDA/provenance preflight.  No science output from the failed
attempt is reused, and no generic RECOVAR suite ran.

All four fresh native science trajectories complete 200 iterations in
290--327 seconds.  Their original Slurm jobs exit 1 only because both post-run
auditors were still invoked by file path under the clean environment; the
201-map and 200-particle artifacts are complete and preserved.  Experimental
commit `c69c636ca` applies module execution to those auditors, its focused
contract test and shell syntax check pass, and the audits are running post hoc
without repeating GPU science.

Three completed all-checkpoint native audits already prove substantial stock
RELION long-trajectory chaos relative to the sealed original:

- repeat 2 first crosses below FSC-AUC `0.999` at iteration 86, reaches
  `0.919961346983`, and ends at `0.982637886189`;
- repeat 3 first crosses at iteration 118, reaches `0.953235762642`, and ends
  at `0.992013386123`;
- repeat 4, which takes the iteration-19 alternate pose, first crosses at
  iteration 31, reaches `0.668868711092`, and ends at `0.850538616596`.

Native iteration-200 hard-state divergence is respectively 1,000, 276, and
1,000 of the 1,000 visited particles for those repeats.  Thus a single-native
late exact-map/particle gate cannot have zero failures even when evaluating
RELION against itself.  The acceptance contract remains exact before the
measured native repeat floor; after that boundary it must compare the paired
candidate/native ensembles and require nondegraded GT quality, native-scale
repeat variability, schedule coverage, and runtime rather than oracle
identity.  The paired panel remains active, and no generic RECOVAR suite ran.

The fourth native audit completes the same classification.  Repeat 1 first
crosses below FSC-AUC `0.999` at iteration 86, reaches `0.922091532234`, and
ends at `0.985347414552`; all 1,000 visited iteration-200 particles differ.
Map/particle report SHA-256 values for native repeats 1--4 are respectively:

- `f8f24c39a9592a86362092f73cf743cd5f1dd693027e3314ef780776d8eb81da` /
  `268ce7a521e8181ab59cfdd7679fb6c7b6e4afb367815e5ee87d4c9c12ed4101`;
- `77f2dc765df449b4d7820010daa6ab67a65a5c661d13838a78eea45552cf87bc` /
  `1047aa1ece4b44a448088149b42b158c557f1f077b901caebbd6fcdd7ffc65f7`;
- `a7654846d4ce85390d6849d67db8235710eabbc702c800bae657b7f0253a47e8` /
  `ec2ca4ea99d6c4fd1ad4b56361b7f01d6ec701928b9e502952c05a8e877f45f2`;
- `44b8a794496c25e9027e6181f576dfb930cdebdc2eeceb909da5a076e59b2233` /
  `9406b82186ff0f0c72e12f9d526e7276148f74f9f416afc6ee778c1f4902e3df`.

The native schedule is itself mode-dependent while preserving HEALPix order
3 and the perturbation stream.  At iteration 90 the sealed/native-repeat
range-step pairs include `6.482290/2.5755`, `6.567306/2.569125`, and the
iteration-19 alternate mode `5.961148/2.607375`; the latter matches the
previous RECOVAR alternate step exactly.  At iteration 200 native ranges span
`3.220735`--`4.109667` Angstrom and steps span `1.44075`--`1.581` Angstrom.
Schedule equality must therefore be assessed as native-mode coverage after
the repeat floor, not equality to one oracle schedule.  No generic RECOVAR
suite ran.

Because Slurm denied a non-disruptive extension of paired job `12937169`'s
four-hour limit, experimental commit `bd1327fa6` adds a restart-safe CPU-only
ensemble summary.  It validates all four trajectory/provenance/GPU reports,
reuses a completed summary if present, or runs the same full 4-by-4
all-checkpoint audit if the parent science job reaches its time limit during
the expensive final pairwise FSC stage.  Its focused contract test and shell
syntax check pass.  Dependent CPU Slurm `12938147` is queued `afterany` on the
paired panel, so completed GPU science cannot be lost or repeated merely to
finish analysis.  No generic RECOVAR suite ran.

Paired job `12937169` completed repeat-1 RELION and RECOVAR science but was
correctly rejected before its trajectory audit because the repository head
changed from launch head `ef0d5bce8` to `bd1327fa6` while the job was active.
The intervening commits changed runners/auditors rather than production
science, but the immutable-source evidence contract does not permit that
distinction after launch.  The preserved run took `276.6543 s` in stock
RELION and `2305.7316 s` in RECOVAR (`8.33x`); a focused CPU-only post-hoc
0--200 map, particle, and schedule audit is Slurm `12939499`.  It remains
standalone evidence and is not mixed into a formal repeat ensemble.  The
dependent summary `12938147` failed closed in three seconds because repeats
2--4 were absent.

A new immutable snapshot worktree at exact source head `ef0d5bce8` now owns
the formal repeat gate.  Its compiled CUDA and RELION-binding artifacts are
independent copies with SHA-256
`e0aab7e1be93159e67c42c62f622062232a66646c91353e82c5917b0c0b24d39`
and `fcbb2a8356c2f7ee88e947fa92c9f5bfc41535ed0a2c6a9124a2fad781a63b83`,
respectively, so later rebuilds cannot mutate the running checkout.  The
fresh four-repeat same-allocation H100 panel is Slurm `12939263`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gui_default_repeat_panel_ef0d5bce_frozen_retry3_20260825`;
restart-safe CPU summary `12939264` is queued `afterany`.  Two setup attempts
(`12939154` and the corresponding summary `12939155`) stopped before science
because the new worktree initially lacked the ignored compiled RELION
binding.  No output from those preflight failures is reused.

Runtime investigation is isolated from correctness at local unpushed head
`1bc960705`.  The accepted 200-iteration profile spends `1212.42 s` in sparse
pass 1, versus `555.82 s` in sparse pass 2.  The one active hypothesis is
that the fixed 200-million-float coarse-score budget is too conservative for
an 80 GB H100: late iterations use only 5--13 images per batch and make about
40--154 pass-1 calls.  An opt-in, fail-fast budget override and eight focused
unit tests are complete (`8 passed`; Ruff, pycompile, and diff checks pass).
The bounded same-GPU default-versus-800-million-float iteration-20 probe is
Slurm `12939443`.  Earlier attempts `12939000`, `12939262` failed before
science on, respectively, the missing ignored RELION binding and the CUDA
binary SHA guard after the shared symlink target was rebuilt.  The corrected
probe uses independent compiled-artifact copies and rejects on any particle,
map, support, memory, or less-than-20-percent runtime failure.  No full VDAM
trajectory and no generic RECOVAR test suite is spent on this performance
probe.

The standalone repeat-1 CPU audit finishes as Slurm `12939499` without
repeating GPU science.  RECOVAR is exact to the paired native run through
iteration 18; the first hard-state difference is the already classified
iteration-19 native near-tie (one particle, `1037@particles.128.mrcs`), and
five particles differ at iteration 20.  The first map checkpoint below
FSC-AUC `0.999` is iteration 31 (`0.997566498504`), the minimum is
`0.668409764290` at iteration 120, and the final is `0.856422986141`.
Hard-state divergence grows to 440 visited particles at iteration 90 and
1,000 at iteration 200.  The schedule audit first distinguishes translation
topology/range at iteration 20, offset step at iteration 90, and the
mode-dependent `sampling_updated` decision at iteration 140.  This is valid
standalone evidence, but the launch-time source mutation means it remains
excluded from the formal ensemble.  Map, particle, and sampling report
SHA-256 values are respectively
`8468df1ab80071ab2647827c84386299ec23e58315a741894ad662646dea4c94`,
`bcaea5d3857279baa48b48faef09fd08403ad4daa0cc358ac6aa36648c380a69`,
and `ad8b9126b96b0771d9a75dc693234d51e394c3662f56f9b99f4209284a6bec85`.

The first bounded runtime hypothesis is rejected by same-H100 Slurm
`12939602`.  Raising the score-tensor budget from 200 million to 800 million
floats changes iteration-20 pass-1 time from `1.8761578 s` to `3.4010768 s`
(`+81.3%`) while pass 2 remains `1.5857396 s` versus `1.5677228 s`.
All hard particle states are identical through iteration 20 and the minimum
between-arm map FSC-AUC is `0.9999999999101582`; map/particle comparison
report SHA-256 values are
`b78bad0ffdee7445a64c421da83ee0edcd1300f6d6b10fda036c1e07bae4701b`
and `733abcb5105f7ba9899822c9f0c27f520c739a6be02f51cbbc28c3e9fd78e3fd`.
Slurm `12939819` observes `1.9202456 s` versus `6.1568847 s` pass-1 wall for
the default and 103-million-float arms, but its original "three-image batch"
interpretation was wrong and the result is not a sealed steady-state
rejection.  The iteration-20 pass-1 grid is the coarse `36,864` rotations by
`29` translations, not the fine `294,912` by `116` grid in the summary.  The
two caps are therefore 187 and 96 images.  For 200 selected particles they
score two padded calls/374 rows and three padded calls/288 rows.  The 96-image
arm compiled this previously unseen JAX shape during its timed iteration, so
its `6.1568847 s` is a cold-shape observation.

The intervening CUDA resource-setup hypothesis is rejected by timing-only
Nsight Systems H100 Slurm `12940146`.  Across all 21 coarse-projector calls
through iteration 20, total call wall is `2733.548 ms` and the actual scoring
kernels account for `2712.538 ms`; allocation, texture fill/copy,
synchronization slack, destruction, and frees total only `21.010 ms` (`0.77%`
of call time and `0.17%` of the `12.429 s` aggregate pass-1 wall).  The CUDA
API/kernel/memory report SHA-256 values are
`f0546d36eee1446987b1fdfb27afeb4c8b651fb0a9b83600b5a057d4a12a1213`,
`0faf9d899d43b32d356d8c45946eb4bb33c9b245254c8511eb4ede2b37bf7823`,
and `31ee66d8a7d367725c7d17ddec74b7187888d49ed3d875a1b3fd6a267c950f07`.
No resource-lifetime change is justified.

The active isolated runtime discriminator is same-H100 Slurm `12940613`,
which repeats default versus 103 million floats after the 96-image executable
has populated the immutable compilation cache.  It accepts only at least 20%
pass-1 speedup with zero hard-state differences, native-envelope maps, and no
OOM.  Setup attempt `12940105` stopped before science in two seconds on an
incorrect typed source-SHA guard; no output is reused.  Runtime changes remain
local and unpushed.  The immutable four-repeat default panel `12939263` and
restart-safe summary `12939264` continue independently; the snapshot checkout
and its copied compiled artifacts have not changed.  No generic RECOVAR suite
ran, and the remaining 21 frozen trajectory cases remain held until the
repeat-aware default gate is classified.

The corrected warm-shape result closes score-budget tuning.  Slurm `12940613`
still measures `1.9373736 s` default versus `5.9438605 s` at the 96-image cap
for iteration-20 pass 1 (`+206.8%`).  Direct all-checkpoint comparison first
selects the measured native iteration-19 alternate for one particle and differs
for five at iteration 20; minimum between-arm FSC-AUC is
`0.999989537494`.  Map/particle/shell report SHA-256 values are
`57929ebd8c485824ef469d01258b17e8f8cbd571907e4253948079418b795c85`,
`36ae899cac5b0e51908f8545a6ce45a423b12c5f6a9dfaca7e086f3918485955`,
and `02206f5a8cffe3ea19a4eaded5a9ca2bac91855f46372bb4eda27e26f384d166`.

An isolated opt-in kernel candidate now mirrors RELION's shared-reference
staging without changing RECOVAR's already-qualified compact rotation frame,
pixel/lane accumulation order, or atomic output topology.  Representative
H100 Slurm `12941717` reduces a 187-image by 36,864-rotation by 29-translation
call from median `0.3741495 s` to `0.0656711 s` (`82.45%`).  Candidate
same-kernel atomic jitter (relative L2 `3.85e-8`, one synthetic hard near-tie)
classifies the legacy-vs-candidate delta (`4.38e-8`, three synthetic hard
near-ties); report SHA-256 is
`e43a2352942d890137040c915497557f5abdb0fc5351c35db2a86a37f06ab0e6`.

The real iteration-0--20 same-H100 gate, Slurm `12941792`, has zero hard
particle differences from RELION in both arms and all maps pass.  Candidate
minimum FSC-AUC is `0.9999999999195193`.  Iteration-20 pass 1 falls from
`1.9457793 s` to `1.3525169 s` (`30.49%`), with pass 2 stable at
`1.6432538 s` versus `1.6584010 s`.  Candidate map/particle/profile report
SHA-256 values are
`e328d9ccf0c0cb37e5de6923ca10f11f32b4d99432f7107d34f397b186d6e3a2`,
`52e7a5358ed5c608d913cfcdcfabb6e5f3a8f5ea3b3398aef5794593e52a011f`,
and `a34eb2629701044b3c0cbc12741b6e1398a30374277023345984c072a5b86737`.
Focused source/flag tests, Ruff, pycompile, shell syntax, CUDA compilation, and
diff checks pass; no generic suite ran.

Full-qualification setup attempt `12941960` stopped before science in two
seconds because the direct runner requires its output root to carry a
pre-existing `SAFE_TO_DELETE` marker.  Retry `12942022` also stopped before
science in five seconds: Slurm inherited the tracking checkout as its current
directory, so Python's empty-path import precedence correctly tripped the
runtime-checkout provenance guard.  The corrected opt-in 200-iteration
default qualification is H100 Slurm `12942089`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_shared_full_c2583f890_retry2_20260825`;
Slurm's working directory is explicitly the immutable runtime checkout and
the import/GPU/CUDA preflight has passed.
It is pinned to immutable local unpushed source
`c2583f890` and experimental H100 CUDA SHA-256
`b3a329a5f5559ca3bb3c436ab5a051bfe1666c317a2c2834daea1ae3c2c6939c`.
It profiles every iteration and retains all 201 maps, 200 particle states, and
schedule checkpoints.  Acceptance requires native-ensemble/GT/schedule
coverage after the measured iteration-19 floor and at least 20% total RECOVAR
wall reduction.  The formal immutable four-repeat production panel
`12939263` continues independently, with `12939264` queued for restart-safe
summary.  The performance worktree will remain unchanged while `12942089`
runs, and the frozen 21-case expansion remains held.

The shared-reference candidate clears the complete default runtime gate.
Slurm `12942089` produces all 201 maps, all 200 particle-state artifacts, the
full schedule audit, and the iteration-90 expected-accuracy replay from the
pinned source/binary above.  The RECOVAR trajectory itself finishes normally
in `948.44 s`, versus `2294.3134 s` for the accepted legacy repeat: a
`58.66%` reduction.  The paired legacy stock RELION wall is `285.1376 s`, so
the candidate is still `3.326x` slower and comparable runtime remains open.
Slurm's final `1:0` is the expected fail-closed single-oracle audit result,
not a run or artifact failure.

Against that one sealed native trajectory, the first hard particle difference
is iteration 42 (one particle), the first map below `0.999` is iteration 119,
the minimum cross-engine FSC-AUC is `0.9890884153` at iteration 139, and the
iteration-200 value is `0.9916669728`.  This does not pass the frozen point
gate and is not waived; final science classification waits for the immutable
four-repeat native-mode/GT/schedule envelope `12939263`/`12939264`.  The map,
particle, sampling, profile, and expected-accuracy report SHA-256 values are
`4066a150e3347267c6194b742c68c21640b1764c59b74cb6b5f9722e7fc54160`,
`fdfa4da66e7ecb3d3f88a174fe83f7ad33cdee23b731bd1ffa8d48908a17ec72`,
`46d5847ea78730d3e358a837277a1b044a7962700924da389d6c558380f23dca`,
`b2749dce5ef221bc881be659f476a4d77d9e965bc80cdd71690294673088b2ee`,
and `f04eed3db6b09be3833159131a21ca8c03b90b9c7fcf769ddbe2914e7425d61`.

The profile also freezes the next bounded performance target.  Across all 200
iterations, expectation takes `869.0291 s`, pass 1 `335.5154 s`, and pass 2
`415.9826 s`; over iterations 90--200, pass 1 is `266.8355 s` and pass 2 is
`300.5365 s`.  The next experiment will profile a representative late pass-2
bucket before any new implementation change.  The existing full GUI-default
matrix already covers 22 complete 0--200 trajectories; its remaining 21 cases
stay held until the default ensemble gate is classified.  No generic RECOVAR
suite ran.

Local unpushed correctness-tooling commit `77c9bde5a` adds a fail-closed
candidate-versus-native-envelope trajectory auditor.  It requires at least two
complete native repeats with one immutable source/executable/physical-GPU
provenance set, verifies candidate source/CUDA/GPU provenance and every frozen
artifact/checkpoint, and retains the unchanged scientific gates: the candidate
must match at least one native mode at FSC-AUC `0.999`, while its GT FSC-AUC
must remain within `-0.002` of the **best** native repeat.  Particle and
schedule envelopes remain explicit separate gates rather than being implied by
the map result.  Focused auditor tests pass `6/6`; Ruff and byte-compilation
also pass.  This commit is not pushed while the formal repeat panel is still
running.

Local unpushed runtime commits `b56280859` and `7bd6a8953` add measurement-only
CUDA-profiler start/stop around one explicitly selected VDAM expectation
iteration and make the existing Nsight wrapper select its stop iteration,
audit mode, and trace label.  Default execution does not load the profiler.
Focused validation passes `3/3` for the iteration/profiler contract and `1/1`
for the wrapper, plus Ruff, byte-compilation, and shell syntax.  Setup attempt
`12944169` stopped before science in two seconds because the abbreviated Git
SHA did not satisfy the exact 40-character provenance guard.  Corrected H100
Slurm `12944211` runs through iteration 150 from immutable source
`7bd6a8953761b070a8c86e872e9f77aaa418ff42`, captures CUDA/NVTX/OSRT only for
expectation iteration 150, and pins the already-qualified shared CUDA binary
SHA-256 `b3a329a5f5559ca3bb3c436ab5a051bfe1666c317a2c2834daea1ae3c2c6939c`.
No setup-failure output is reused.

Bounded H100 Slurm `12944211` completes successfully and identifies host
compilation, not fine projection arithmetic, as the late pass-2 bottleneck.
At iteration 150, RECOVAR records `10.0448 s` for pass 2 and `6.4461 s` in two
big-JIT buckets.  Nsight attributes only about `27.44 ms` to the two
fine-score CUDA launches and `0.451 ms` to texture projection, while two
`jit_run_local_bucket_big_jit` compilations consume `5.1137 s` and 20 XLA
autotuner compilations consume another `2.5484 s`.  The trace, SQLite,
CUDA-API, and kernel-summary SHA-256 values are
`460d56e31292e69c962f38b99fe22255bbfe383672707765330f09fe6844ed6c`,
`d996d1c3accdfbb3c43bf75f8e3259c517965af25934c1dd17ea3d80b3312161`,
`bd69659433ed81b7b8b447dff2098eba0c12419f55382ba3bf0d7af89b027cdc`,
and `df0e91a77c7e9efaa417bd359b4d62398144643feff754e5f772fb4056909726`.
This evidence rejects projection-cache work as the next experiment.

Inspection then exposes a signature-drift defect: the big-JIT decorator says
it donates loop-carried `Ft_y`/`Ft_ctf`, but positional donation `(4, 5)` now
names `corr_img_rfloat_square_half` and `mean_for_proj`.  The profile log emits
unused-donation warnings for the changing score operands throughout the late
trajectory.  Local unpushed runtime commit `250cf27c7` donates the two
accumulators by name and adds a source regression guard.  Ruff,
byte-compilation, and the guard pass.  A larger numerical unit reaches an
unsupported login-node CUDA binary (`no kernel image`) and is classified as an
environment failure; focused H100 replay `12945961` is the authoritative
numeric/runtime A/B against `12944211`.  Setup attempt `12945899` is excluded
before science after another abbreviated exact-SHA guard failure.  No runtime
commit is pushed.

The immutable default panel has produced all 200 RELION iterations in all four
repeats.  Repeat 3 is fully audited: its first hard particle difference is
iteration 72, minimum cross-engine FSC-AUC is `0.9197498003`, and minimum
RECOVAR-minus-RELION GT FSC-AUC is `-0.0003295897`, so its GT trajectory passes
the unchanged `-0.002` gate while its single-oracle map trajectory does not.
Repeat 4's audit and the restart-safe ensemble summary remain in progress
under `12939263`/`12939264`; the candidate-envelope auditor will run only after
those immutable artifacts close.

The accumulator-donation experiment is rejected and restored locally.  H100
Slurm `12945961` reduces iteration-150 pass 2 from `10.0448 s` to `7.6935 s`
(`23.41%`), but leaves big-JIT time effectively unchanged (`6.4461 s` to
`6.4240 s`) and changes scientific state: candidate/control active-particle
states are exact through iteration 90, then all 840 selected particles differ
at iteration 140 and all 920 differ at iteration 150.  Candidate/control map
FSC-AUC is `0.9999999250` at iteration 90, `0.9934749859` at iteration 140,
and `0.9936399177` at iteration 150.  Local commit `23b499342` restores the
production decorator and records the rejection; neither the experimental nor
restoration commits are pushed.  The next profiler target must be a warm late
iteration, because iteration 150 is a cold schedule/shape transition.

The correctness pool now has explicit map, particle-state, and sampling-mode
ensemble gates.  Local commits `08bd550f1`, `cb2091ce1`, and `8cd1798d0`
require every active candidate particle to match at least one native-repeat
pose/translation state, require the complete candidate sampling state to match
one whole native mode (not a cross-repeat field mixture), ignore expected
accuracy fields only while they are explicitly unused, and fail closed on
cross-GPU provenance.  The reusable CPU Slurm wrapper runs both gates even if
one fails and records their independent statuses.  Focused validation is
`19/19` auditor tests plus the wrapper test, Ruff, byte-compilation, and shell
syntax; no generic suite ran.  These tooling commits remain local and unpushed.

All four formal native/candidate pairs completed 200 iterations on the same
physical H100.  Repeat 4's first hard particle difference is iteration 42;
its minimum cross-engine FSC-AUC is `0.9107028723`, and its minimum
RECOVAR-minus-RELION GT FSC-AUC is `-0.0014313519`, still inside the frozen
`-0.002` GT gate.  Parent `12939263` exits `1:0` only because individual
single-oracle audits are fail-closed; restart-safe ensemble summary `12939264`
is running over all four complete repeats.

The prior shared candidate was produced on a different physical H100, so the
new provenance guard correctly excludes it from the formal panel comparison.
Candidate-only same-GPU rerun `12947769` is pinned to `della-h19g2`, source
`23b4993428d015d46fd574281489c2f8e1b0a5a8`, and CUDA SHA-256
`b3a329a5f5559ca3bb3c436ab5a051bfe1666c317a2c2834daea1ae3c2c6939c`.
Dependent CPU job `12947773` will run the map and state/schedule envelopes only
after both that candidate and summary `12939264` finish.  Setup attempt
`12947583` stopped before science because the output root lacked its required
`SAFE_TO_DELETE` marker; its dependent `12947714` was cancelled before start
and no artifact is reused.

Same-H100 candidate `12947769` has now completed all 200 scientific iterations
with status `0:0` for the timed RECOVAR step and `14:34.87` wall.  Provenance
records physical GPU UUID `GPU-235ec3bc-ca9f-1c0e-88eb-c8b37c5e0480`, exactly
the UUID used by the four-repeat panel.  Iteration-90 expected rotation and
translation accuracies replay within `4.45e-16` and `1.56e-15` Angstrom.  The
outer job exits `1:0` only because the retained single-oracle audits fail:
first map failure is iteration 132, minimum cross-engine FSC-AUC is
`0.9544534622` at iteration 173, final FSC-AUC is `0.9931815500`, and first
hard particle divergence is iteration 84.  These point failures are not
waived; candidate acceptance still belongs exclusively to dependent ensemble
job `12947773` after repeat summary `12939264` finishes.

The same run also identifies a sharper runtime trace target.  Iteration 159 is
the maximum pass-2 event at `9.9204 s`: `current_size` changes 72 to 74, the
profiled half-set changes from two sparse big-JIT buckets to four, padded rows
jump from 31,488 to 63,488, and big-JIT wall reaches `7.0056 s`.
Measurement-only commits `4adfafb9d`/`c2f9d66c0` record that one hypothesis
without changing the restored production source behavior.  Same-node bounded
Nsight job `12948880` is active on `della-h19g2`, pinned to full head
`c2f9d66c0e4b64e49bdaea2199493e54dee224bd` and the qualified CUDA SHA above,
and captures only iteration 159.  No optimization is accepted from host
timing alone.

The formal four-repeat summary `12939264` completes its full computation and
exits `1:0` because the scientific result is `fail`, not because of the CPU
cache warnings.  The first failed checkpoint is iteration 84 and 117/201
checkpoints fail.  All four RECOVAR runs satisfy the frozen GT nondegradation
gate at every checkpoint (minimum delta `-0.0014313519`), but cross-engine mode
coverage is incomplete: the minimum candidate-best-native FSC-AUC is
`0.9630186047` at iteration 173, and minimum native-best-candidate is
`0.9120287616` at iteration 133.  RECOVAR repeat variability stays within the
native envelope at 200/201 checkpoints; the one real spread failure is
iteration 173, where the RECOVAR repeat floor is `0.9528717628` versus native
`0.9760519203`.  This is an active K=1 default failure, not a tolerance
candidate.  Shared-candidate ensemble job `12947773` is now released from its
dependency and pending for CPU priority.

Bounded Nsight `12948880` completes `0:0` but samples a warm stochastic state,
not the intended transition: iteration 159 is `current_size=72` with two local
buckets and pass 2 takes `2.3208 s`.  The two big-JIT modules execute in
`0.1002`/`0.1031 s`, the fine-score kernels total `24.87 ms`, and there is no
XLA compile range for them.  Five coarse launches instead total `12.0966 s`.
The trace and SQLite SHA-256 values are
`cf2e568d0e18762babb969ab056165ce2d7e7f1ad52272d826a4ddedbc27ae70`
and `5574e66a5b62c9071a1cce0935086b8eab9b1501694dce3e2a61ad73135795f9`.

Measurement-only local commit `b32ebc8e4` replaces unstable physical-iteration
selection with a fail-closed exact-local shape trigger.  It captures the first
non-score call matching an explicit `current_size:bucket_count`, is inert by
default, records the target in run provenance, and rejects simultaneous
iteration/shape selectors.  Focused validation passes 4/4 tests plus Ruff,
byte-compilation, shell syntax, and diff checks; no generic suite ran.  Same-
node setup `12950773` stops before science in two seconds because an incorrect
expanded SHA fails the exact-head guard; its output is excluded.  Corrected
shape capture `12950828` is active for `74:4`, source
`b32ebc8e4c2f5dfb01a9b417ebbbdd698ac59ce9`, and the qualified CUDA SHA.  The
measurement commit remains local and unpushed.

The one formal repeat-spread failure is now localized to an exact adaptive-
resolution boundary, not a generic random map excursion.  At iteration 173,
RECOVAR repeats 2 and 4 still use `current_size=74`, while both matching
RELION repeats already use `72`; repeats 1 and 3 agree with RELION at `72`.
The lag is one iteration for repeat 2 and two iterations for repeat 4, and
the RECOVAR repeat floor recovers from `0.9528717628` at iteration 173 to
`0.9859` at iteration 174.  Shell 27 is the decisive threshold: in repeat 2
at iteration 172 RELION records `rlnSsnrMap=0.989837` while RECOVAR records
`1.004969`; in repeat 4 at iteration 171 the values are `0.995230` and
`1.021395`.  The difference is dominated by the higher RECOVAR reference
tau2 rather than reference sigma2, so the next correctness boundary is the
late reference/tau2 formation that feeds `updateSSNRarrays`.

Local unpushed audit commit `3be85425e` closes the test blind spot exposed by
that failure.  The full sampling-trajectory auditor now checks RELION's
serialized current resolution and current image size at every selected
checkpoint, while retaining the distinction between the resolution chosen
after the M-step and the size used by that iteration.  Focused validation is
`14/14` sampling/envelope tests plus Ruff and byte-compilation.  A replay of
formal repeat 2 detects the real boundary exactly: first resolution mismatch
at iteration 172, first image-size mismatch at iteration 173, and agreement
again at iteration 174.  No generic RECOVAR suite ran.

Exact-shape Nsight job `12950828` also closes without a usable trace for a
scientifically explained reason.  Its inner 200-iteration run completes
`0:0`, but the outer fail-closed profiler gate exits `1:0` because this
stochastic repeat never produces the requested `74:4` exact-local shape.  At
its slow size-74 phase it has three non-score buckets, including iteration
170, where expectation takes `25.9544 s`; the complete run's maximum pass-2
time is `12.1812 s`.  Dependent same-H100 job `12952330` is therefore pinned
to the observed `74:3` shape, the same source
`b32ebc8e4c2f5dfb01a9b417ebbbdd698ac59ce9`, and the same qualified CUDA
SHA-256.  Candidate ensemble job `12947773` remains active independently.

Same-H100 candidate envelope `12947773` is now complete and fails the frozen
scientific contract.  Its map report SHA-256 is
`28e0e343ea6c111cd26e0a6f08c0b09e939e3fc6e56cafb8b95c2d4fdb23e603`.
Ground-truth nondegradation passes at all 201 checkpoints, but the candidate
first leaves every native map mode at iteration 139 and misses at 62/201
checkpoints through iteration 200.  The minimum candidate-best-native FSC-AUC
is `0.9654143125` at iteration 173; the neighboring values are `0.99486076`
at 172 and `0.99207845` at 174.  Its minimum RECOVAR-minus-RELION GT delta is
`-0.0018244161` at iteration 160, still inside the unchanged `-0.002` gate.

The first state-envelope report also fails, but its same-iteration
`rlnNumberOfIterWithoutResolutionGain` comparison is invalid: RECOVAR records
the counter before the E/M-step, while RELION serializes it after
`updateCurrentResolution`.  Local unpushed commit `619352eba` therefore keeps
that counter diagnostic-only while retaining complete sampling geometry,
accuracy, update, current-resolution, and current-size checks as active
acceptance fields.  It also incorporates the exact resolution/size auditor and
locks the requested 22-case full-trajectory stress matrix (uniform, Kent, and
anisotropic poses; white, radial, high, and low noise; CTF/translation/scale;
junk and 20--70% outliers).  Focused validation is 15 state/sampling tests,
the explicit stress-contract test, Ruff, byte-compilation, and shell syntax;
no generic RECOVAR suite ran.  Corrected state-only rerun `12953825` is pending
against the already completed candidate and native panel; it does not repeat
the 35-minute map audit.

Exact-shape profiler `12952330` captures the first cold `74:3` pass at
iteration 161 before the outer trajectory stalls.  That pass takes
`24.4791 s`, versus `1.8201 s` once warm.  Nsight attributes the cold boundary
primarily to two `run_local_bucket_big_jit` compilations (`5.1710 s` total),
two fused M-step compilations (`1.4992 s`), and XLA autotuning (`2.2344 s`);
all CUDA kernels together take about `0.610 s`.  The leading kernels are the
coarse VDAM SGD kernel (`0.2081 s`), XLA accumulation (`0.1609 s`), fine diff2
(`0.0454 s`), and M-step sums (`0.0262 s`).  The trace SHA-256 is
`43169b6e3d6e3d27825f125fc050dd2216e354b9ec68f9e5e3f2ffa9a8252534`.
The scheduler denies a wall-limit extension and then fails to terminate at
40 minutes, so the run is cancelled at `41:04` after all trace reports are
safely materialized.  This is an orchestration cancellation, not a trajectory
parity result.  The runtime target is now cold shape/compile reuse without
changing the qualified numerical path.

The correctness attack moves to the earlier causal boundary rather than
rerunning the frozen matrix.  Across the formal repeats, candidate/native
shell-27 tau2 first exceeds a mean ratio of `1.005` at iteration 98 and rises
to about `1.017` by iteration 166, while sigma2 remains close.  Full-schedule
M-step capture `12955511` is running on one H100 from local unpushed head
`7f59c67176163990351318e07a67741fd4518b7b`, qualified CUDA SHA-256
`b3a329a5f5559ca3bb3c436ab5a051bfe1666c317a2c2834daea1ae3c2c6939c`,
and RELION binding SHA-256
`fcbb2a8356c2f7ee88e947fa92c9f5bfc41535ed0a2c6a9124a2fad781a63b83`.
At iteration 98 it compares raw BPref accumulators, post-apply data and noise
momenta, and reference input/output formation.  Its focused harness validation
is 4/4 tests plus Bash syntax, Ruff, and byte-compilation.  The remaining 21
frozen stress trajectories stay held until the GUI-default K=1 gate closes.

Full-schedule capture `12955511` completes both scientific arms through
iteration 98 in `14:04`.  The Slurm wrapper reports `1:0` only because the
post-run analyzer originally treated RELION's `(1, 1, 65)` MultidimArray shell
header and RECOVAR's equivalent `(65,)` NumPy vector as incompatible.  Local
unpushed runtime commit `d1b98e7b0` canonicalizes singleton-only shape
differences and adds a regression test; the complete focused analyzer suite is
`4/4`, with Ruff and byte-compilation also passing.  The already-complete
trajectory artifacts are reused, so no GPU rerun is required.

The iteration-98 result rejects the late M-step schedule as the causal source.
RECOVAR computes `tau2_fudge=3.9997504916` and
`grad_current_stepsize=0.5334314472`, which serialize exactly as RELION's
`3.999751` and `0.533431`; effective step size is also `0.533430` in both
captures.  The reconstruct input FSC spectrum is bitwise exact (`65/65`).
However, the M-step is already receiving a divergent trajectory: incoming
first-moment relative L2 is `0.9838199`/`1.0085798`, incoming second moment is
`1.0275489`, and the reference before reconstruction is `0.2501801`.  Raw
halfset accumulator relative L2 is `1.1917784`/`1.2561409`, post-apply data is
`1.2497818`, moment-noise power is `0.0747206`, and the reconstructed reference
is `0.2884436`.  Thus iteration 98 is a consequence boundary, not the first
cause; the earliest proved causal boundary remains iteration 1, where incoming
gradient state is bitwise exact and the first difference appears in raw BPref
accumulation.

As a secondary single-run diagnostic, shell 27 becomes populated at iteration
76 and immediately differs: RECOVAR/native tau2 is `1.0969720` and DVP is
`1.1018750`; the maximum through iteration 98 occurs at iteration 90
(`1.2245999` tau2, `1.2185535` DVP), while sigma2 remains close.  At iteration
98 the shell-27 ratios are `1.1539018` tau2, `1.1507121` DVP, and `1.0027706`
sigma2.  This single captured run is not substituted for the formal
four-repeat envelope; it reinforces that high-shell tau2/DVP amplifies an
earlier expectation/accumulator difference.  No generic RECOVAR suite ran,
and the 21 held stress trajectories remain gated.

The corrected state-only envelope closes the counter-temporality audit.  The
first allocation, `12953825`, waited for CPU priority and then stopped before
analysis because the submitted expanded SHA did not equal the actual
`619352eba61d1439cd68f7b2b7e5dba2d92a9d27`; no report was produced by that
attempt.  Corrected 2-CPU/16-GB job `12956653` completes the report in 52
seconds and exits `1:0` for the retained scientific failure.  Particle state
first leaves all four native modes at iteration 84 and fails 97/200
checkpoints.  Whole-mode schedule state first fails at iteration 6 and fails
108/200 checkpoints: optimal-offset change misses at 100 iterations, offset
range at 101, offset step at 10, rotation accuracy at 10, and translation
accuracy at 5.  The newly active resolution fields isolate exactly one current
resolution miss at iteration 172 and one current-size miss at iteration 173.
The pre/post-M-step resolution-gain counter is diagnostic-only and differs at
181 iterations, as expected.  Report SHA-256 is
`e180a0d5c1055333a38eb8bfa796efd3e7b71c1e2c95bc3e70d97fe6df8b28c7`.

Source inspection then corrects the iteration-1 launch-topology hypothesis.
The matched RELION job uses `--j 8`; RELION computes
`nr_pool = --pool * --j` and gives one particle task at a time to eight OpenMP
workers, each with a blocking class CUDA stream.  The prior RECOVAR experiment
incorrectly interpreted `--pool 3` as three worker streams with a barrier after
each group of three.  Local unpushed experiment `e479c2a0f` instead keeps one
in-flight particle per eight worker streams.  Setup job `12956321` stops before
build on another incorrect expanded SHA; corrected H100 job `12956370`
completes in 52 seconds, passes all 16 focused CUDA translation/routing tests,
and produces binary SHA-256
`79f05a8676b5f6bfaf1669a4cda70be8b93ac1763212d4573ff70c03e9c6df93`.

The worker-8 discriminator materially closes propagated iteration-2 state.
Although its iteration-1 raw-accumulator relative L2 remains
`8.0012e-6`/`9.3739e-6` and its reconstructed reference is `2.0513e-6`, the
frozen iteration-2 component comparison improves by about four orders of
magnitude: signed direct-residual error changes from about `+1.60e-5` to
`-1.25e-9`, `AA` from `-3.87e-6` to `+1.82e-10`, and `XA` from `-9.99e-6`
to `-2.99e-10`.  Two independent repeats `12956491`/`12956492` confirm the
closure: direct-residual relative L2 stays in
`[6.73e-7, 6.81e-7]`, `AA` in `[2.29e-6, 2.47e-6]`, and `XA` in
`[3.70e-6, 3.87e-6]`.  Each repeat takes 60--62 seconds, comparable to the
57--61 second controls.  Report SHA-256 values are
`c7fd756b27270f648ec6ab8975a56f6048d7d2468676b51a7db9c6ec771b2b8d`,
`7badaf7c00ecbda91f342969ff705530c6ab2c45a5003047896eb27315e77e88`,
and `ef382f7984b613da0c2fa9a7ad43cc001db990ece404fba3e56c1e5f6970ba1b`.

Local unpushed head `0f3642045` adds only the already-qualified singleton
shell-vector analyzer fix, leaving the worker-8 CUDA binary unchanged.  Its
first promoted 0--20 trajectory (`12956564`) passes all 21 map checkpoints
with minimum cross-engine FSC-AUC `0.9999999999189698` and has zero active
pose/translation divergences through iteration 20.  The wrapper exits `1:0`
only because the standalone schedule audit still treats unused early accuracy
sentinels and small serialized accuracy/optimal-offset differences as active;
map and particle audit statuses are both zero.  Same-H100 0--200 promotion
`12956768` is pinned to `della-h19g2`, source `0f3642045`, and the worker-8
binary above.  It will be compared against the formal four-repeat native
envelope before any stress-matrix release.  No generic RECOVAR suite ran; the
remaining 21 frozen cases remain held.

The worker-8 0--200 formal-GPU trajectory is now complete.  Job `12956847`
ran all 200 scientific iterations on physical GPU
`GPU-235ec3bc-ca9f-1c0e-88eb-c8b37c5e0480` in `28:54.49`; its outer Slurm
status is `1:0` only because the retained single-oracle audits fail.  Against
the frozen repeat-1 RELION trajectory, all map checkpoints pass the unchanged
`0.999` gate through iteration 128 and the first failure is iteration 129.
FSC-AUC is `0.9999998501` at iteration 104, `0.9998592106` at 120, and
`0.9936677537` at 200; the minimum across the trajectory is `0.9901038851`.
The map, particle, sampling, and profile report SHA-256 values are
`e40a49a6ccbcb90cabd07a829b07adc45c7f2d6fcfe837d803e317f082466745`,
`f126f2921b14eab698d1a18b2411ccfeaf0341bb4b1088b6281baa3d4de6cfbd`,
`714cbe007d975ba4d091323e3f72d79600c8552394a1251936e39e16002d85ea`,
and `1a91de404c25064e25236f806b2e96f513c0deb8a547af15913cea97a3b2b763`.

The single-oracle particle path is initially sparse and non-monotone rather
than immediately unstable.  Exactly one hard winner differs at iterations 84,
92, and 97, with exact hard state restored between those checkpoints.  The
candidate again has zero hard differences at iteration 103, one translation
winner at 104 and 105, and zero at 106.  Divergence then grows from one at 107
to 36/3000 at iteration 120.  This is a large improvement over the prior
candidate, but the late single-oracle trajectory is not closed.  Independent
cross-GPU diagnostic `12956768` finishes its scientific step in `29:35.45`
and enters a different late RELION mode; it is retained as robustness evidence
and is not used for formal acceptance.  CPU job `12957075` is the preregistered
map/state comparison against all four frozen native modes.

The first iteration-104 native continuation tests, but does not expand, the
accepted one-particle boundary.  Exact-formal-GPU job `12958359` restarts native RELION
from the frozen repeat-1 iteration-103 optimiser, preserves perturbation
`0.158883`, and captures internal particle 344 / source row 1307 / stack image
1308.  The continued process selects the alternate translation seen in the
worker-8 candidate, but replay Pmax is `0.471658` versus `0.470291` in the
frozen uninterrupted run.  The fail-closed harness therefore correctly aborts
before RECOVAR because the `5e-4` native-replay gate is exceeded.  Because
`--continue` reloads the serialized iteration-103 reference while the sealed
0--200 process carries its reference in memory, this near-tie flip may be a
continuation/serialization artifact and is not an accepted live native mode.
The threshold is not weakened and the resulting verbose native score/posterior
capture remains diagnostic-only.
Sibling allocation `12958358` stops in four seconds on the wrong physical GPU.
Candidate-only production fused-posterior capture `12958453` completes on the
exact formal GPU in `13:56`; sibling `12958452` likewise stops before science
on the wrong GPU.  The iteration-104 full-trajectory winner therefore remains
an unresolved parity boundary unless an uninterrupted native repeat or an
in-memory-exact boundary replay demonstrates the alternate mode.

The earlier iteration-84 boundary is now reproduced without continuation.
Exact-formal-GPU job `12958796` runs native RELION uninterrupted from iteration
0 through the selected iteration-84 particle, then the wrapper deliberately
stops the live process after writing the verbose capture.  It identifies
internal particle 323 / source row 128 / stack image 129 and completes in 56
seconds; sibling `12958795` stops before science on the wrong GPU.  Against
this fresh live capture, the worker-8 trajectory still differs at exactly that
one particle with a `3.697531` degree pose error, while translation agrees to
about `2e-6` Angstrom.  Candidate-only exact-GPU job `12958889` independently
replays the production trajectory through iteration 84 in `9:35` and writes
the target fused posterior; sibling `12958888` stops at the UUID guard.

The schema-correct fused-posterior comparison localizes the hard-state flip.
RELION and RECOVAR have exactly the same 64-candidate support, the same
20-sample reconstruction mask, and matching rotation matrices to at most
`2.98e-8` Frobenius distance.  RELION's two leading hypotheses are mapped keys
`[1,86]` and `[5,86]`, with probabilities `0.29173377` and `0.29130675`;
RECOVAR assigns `0.29122007` and `0.29178941`, reversing their order.  Posterior
L1 is `0.00154873`, relative L2 is `0.00164336`, and maximum absolute residual
is `0.000513704`.  Thus iteration 84 is an upstream score-ordering boundary,
not a support-pruning or adaptive-threshold boundary.  Focused job `12959527`
passes all 16 float32 posterior tests before correctly failing to apply the
older contribution-dump analyzer to this newer fused-posterior schema; the
dedicated fused analyzer produces the completed report instead.  Exact score
operand replay `12959751` is running to split the remaining difference among
image preprocessing, projected reference, diff2, and prior terms.

Source inspection exposes one remaining mismatch in the worker-8 experiment:
RELION's eight host workers dynamically claim one particle at a time, while
the first RECOVAR worker-8 candidate assigned particles to streams in a fixed
single-host loop.  Local unpushed experiment `38e795895` replaces that loop
with eight host threads and an atomic next-particle distributor; each thread
owns one blocking CUDA stream and synchronizes its particle before claiming
the next, matching RELION's `ThreadTaskDistributor` topology.  H100 gate
`12959706` builds binary SHA-256
`b436c7eefa6c022639e7a6a0995797cefe3e2bba4d11368b0e6a04b243b9e207`
and passes all 16 focused translation/fused-M-step GPU tests.  Earlier gate
allocations `12959482`, `12959499`, and `12959682` are setup-only failures from
two incorrect expanded commit pins and one wrong provenance source filename;
`12959499` nevertheless compiled and passed the same 16 tests before its final
hash step, and none ran trajectory science.  An exact-GPU iteration-1 raw
M-step discriminator is queued behind the score capture.  No generic RECOVAR
tests ran, and the remaining 21 stress cases remain held until this causal
boundary is resolved.

The first full-operand score replay, job `12959751`, is excluded from causal
use.  Although it completed on the exact formal H100, its autonomous candidate
trajectory had already selected a different hard state at iteration 19 and the
selected iteration-84 bucket was `16x100` instead of the sealed production
`8x116` bucket.  Its target Pmax was `0.739860`, versus `0.291789` in the
sealed fused-only candidate.  Those operands therefore describe a different
trajectory mode and are not substituted for the iteration-84 boundary.

Local unpushed commits `d3d928e0a` and `090c2ba1f` add a narrower diagnostic:
the selected fused path returns only its already-computed raw/total score and
rotation/translation priors, without materializing the large image/projector
operands.  Focused H100 job `12960361` passes all 3 score-capture/analyzer
tests.  Exact-formal-GPU job `12960380` then completes the true 200-iteration
schedule through iteration 84 in `9:33` and writes the sealed target score
table.  Across iterations 1--84, this independent run has only one transient
hard-state difference from the earlier fused-only candidate (one particle at
iteration 24); the selected iteration-84 particle retains the same pose and
translation, and its Pmax changes by only `4.1e-5`.  Earlier Pmax differences
occur before the iteration-selective diagnostic is active and reflect the
known atomic repeat floor rather than observer mutation.

The score decomposition is decisive.  On all 64 common candidates, centered
combined rotation/translation prior has maximum error `2.6822e-6`, while the
centered pre-prior likelihood score has maximum error `0.00403327` and RMS
`0.00161887`.  For the two competing winners, native RELION favors mapped key
`[1,86]` over `[5,86]` by log-weight `+0.00146484`; RECOVAR reverses that
pair with margin `-0.00170898`.  Candidate support, 20-sample reconstruction
mask, and mapped rotations remain identical.  The first iteration-84 hard
flip is therefore raw likelihood spacing, not priors, support, pruning, or
posterior normalization.  Report SHA-256 is
`607c095f7d302ca78c3026c93aa8107afe5fa7e8e7de5b14f33ebad5fb5b3c20`.

The first global dynamic-worker experiment does not close the earlier causal
aggregate boundary.  Job `12959830` completes the iteration-1 M-step in 37
seconds, but raw data residuals worsen to `1.00283e-5`/`1.00209e-5`, weight
residuals are `2.15370e-6`/`1.87133e-6`, and reconstructed-reference residual
is `2.12247e-6`.  Iteration-2 job `12960400` completes in 48 seconds; its
incoming gradient and second-moment states are already nonexact, while its
final reference residual is `1.03153e-6`.  The corresponding 0--20 run
`12960527` completes the scientific step in 53 seconds.  Every one of its 21
map checkpoints passes, with iteration-20 FSC-AUC `0.999999999944`, and every
particle pose/translation agrees through iteration 20.  Its wrapper failure is
the separately retained single-oracle sampling-accuracy/optimal-offset audit,
not a map or hard-state failure.  Map and particle report SHA-256 values are
`eed3121f8bdb1cef176667b922d976d063b94468c0aee3273e3aeda4b5e8f9e7`
and `9d3a3a92018f7d46df0c50d3e58d58c51595b8c42ba9fc24f79778c73f563437`.

RELION source inspection explains why the global distributor was the wrong
counterfactual.  `--pool 3 --j 8` creates sequential 24-particle input pools;
inside each pool, eight workers dynamically claim one particle at a time, and
all workers barrier before the next pool is loaded.  Local unpushed commit
`f0dee51c8` adds exactly those pool boundaries while leaving the accepted
523-line native SGD kernel body and padded rotation launch unchanged.  H100
gate `12961504` builds CUDA SHA-256
`4162c378682f49837243b06627f78192af9e8ab382e1eb59c83d08b960cdadaf`
and passes all 16 focused CUDA translation/routing tests in 7.18 seconds.
Iteration-1 and iteration-2 bounded M-step jobs `12961519`/`12961520` are
running; no 0--200 promotion is authorized unless those gates improve the
raw-accumulator and propagated-state boundaries.  No generic RECOVAR suite
ran.

The pool-barrier-only result is negative.  Jobs `12961519`/`12961520`
complete in 41/48 seconds.  Iteration-1 raw data residuals are
`8.98969e-6`/`9.27484e-6`, weight residuals are
`2.36457e-6`/`2.15104e-6`, and reconstructed-reference residual worsens to
`2.30904e-6`.  Iteration 2 already receives nonexact gradient state; its final
reference residual is `9.94930e-7`.  Pool barriers without particle-dependent
worker timing are therefore rejected as causal closure.

The coupled discriminator adds the exact active rotation-block count for each
particle, because that count determines when a worker becomes available to
claim its next task.  Local unpushed commits `7c09b83ec`/`bfd5a9a5f` combine
that existing exact-count experiment with the 24-particle/eight-worker pool.
H100 job `12961709` passes all 16 CUDA routing tests; setup job `12961621`
stops before build on a mistyped full SHA, and `12961628` builds successfully
but exposes and stops on one grouped-test call that had not yet supplied the
new count operand.  Neither failed job runs boundary science.

The exact-count/pool result is materially closer at the first M-step.  Job
`12961793` completes in 42 seconds with bitwise-exact incoming gradients and
second moment.  Raw data residuals are `8.12595e-6`/`8.91127e-6`, weight
residuals are `1.82100e-6`/`2.29253e-6`, and reconstructed-reference residual
falls to `1.50647e-6`, versus `2.0513e-6` for static worker-8 and
`2.12247e-6` for the global dynamic distributor.  Iteration-2 job `12961794`
completes in 51 seconds; its incoming state is nonexact and its final reference
residual is `1.20159e-6`.  Report SHA-256 values are
`ef9f1c867c05264c49e369563da2b58dfce0cc9d80114cc21bcef341af8ae29e`
and `f3700ed0d5c4137f12c5746358137e3f82a86383617716145d7a7d64c00ecb41`.

A stricter outer-bucket experiment is also rejected at the causal boundary.
Guarded local commit `dd8f6d814` keeps each heterogeneous 24-particle physical
pool in one exact-count FFI call instead of splitting whenever padded rotation
size changes; its three focused layout tests and H100 job `12961927` all pass.
Iteration-1 job `12962010` worsens raw data to
`9.77431e-6`/`1.10544e-5` and reconstructed reference to `2.93899e-6`.
Although iteration-2 job `12962011` happens to end at `8.95758e-7`, it starts
from a nonexact trajectory and cannot override the earlier causal regression.
No long trajectory is authorized for this bucket variant.

The exact-count/pool candidate passes its bounded scientific promotion.
Job `12962194` completes the 0--20 science step in `64.40 s`; all 21 map
checkpoints pass with minimum FSC-AUC `0.9999999999192933`, and all 3,000
poses/translations agree at every iteration through 20.  Iteration-20 FSC-AUC
is `0.9999999999606846`.  The outer `1:0` is exclusively the retained
single-oracle sampling-accuracy/optimal-offset audit; map and particle statuses
are both zero.  Map and particle report SHA-256 values are
`ac2c770094fbca98a91b20712e2f8fe0dd96d7662d234053d4f64c7f452fc275`
and `27fadbc238260c4de808406b31029cc45f526b846fcce5077cb8ba95f8e2cc1c`.

Local unpushed orchestration commit `dbb55742f` adds an optional physical-GPU
UUID guard to the trajectory runner; its focused contract test and Bash/Ruff
checks pass.  Exact-formal-H100 0--200 siblings `12962260`/`12962263` are
pending.  Only an allocation with UUID
`GPU-235ec3bc-ca9f-1c0e-88eb-c8b37c5e0480` may enter science; any sibling on
another GPU fails before launching RECOVAR.  No generic RECOVAR suite ran and
the remaining 21 stress cells remain held until this full trajectory is
classified.

The exact-formal result rejects exact-count/pool as the production candidate.
The UUID guard works as intended: sibling `12962260` lands on the wrong
physical GPU and fails closed in three seconds before science, while sibling
`12962263` acquires the sealed RELION GPU and completes iterations 0--200 in
`31:49` science / `33:12` including audits.  Its first hard particle mismatch
is iteration 42 (one translation choice for
`1136@particles.128.mrcs`), earlier than static worker-8's iteration 84.  The
first strict map failure is iteration 131 at FSC-AUC
`0.9989390314608421`; minimum trajectory FSC-AUC is
`0.9533737779301175`, and iteration 200 is `0.9926056852032877`.  Static
worker-8 instead first fails the strict map gate at iteration 129, reaches a
materially better minimum `0.9901038850705235`, and ends at
`0.9936677536689421`.  Exact-count/pool report SHA-256 values are
`c53d86aa869e53b557f5ff2a2c072abeec19bdc845dcce0b3e599748b66f6af4`
for maps, `727967b49833ba8b426d208dfd2cd2ac2c0fd1379689754724b2e624c5230022`
for particle state, and
`652d63a01de81ca5de4a0d3870a83430ed304a432a9a9971368f4c781d78154a`
for sampling.  The remaining 21 stress cells stay held.

The completed four-repeat static-worker envelope also fails, rather than
reclassifying the iteration-84 boundary as ordinary native repeatability.
Job `12957075` completes both auditors in `26:25`; its nonzero exit is the
expected gate result, not an execution failure.  The first particle outside
all four native runs is iteration 84, again
`129@particles.128.mrcs`.  The first map outside the frozen native-mode gate is
iteration 131; minimum best-native FSC-AUC is `0.9831481130943629`, while the
candidate-minus-best-native ground-truth floor remains nondegraded at
`-0.0009750295208153792` versus the allowed `-0.002`.  The map, state, status,
and shell report SHA-256 values are respectively
`535f6da63eaed983f862348ad914600bf2fcff36417a13f3a087770563f42895`,
`bb5cb89c5d6413f76f4f3461e48ff21df30d1c1b6a3a7f84a6b96b6d64ad1ba9`,
`e5b8bba0df321c8095e668ec3f7960ea4f90f95c8253ce7193bc9ec43c394533`,
and `9f987819f5f1c1c0189b293d2967724e896e00d1dee8047bd78dd8e5fc3ca5e6`.

These two long gates close the scheduler-guess branch.  RELION source confirms
that the actual topology is a 24-particle outer pool plus a one-particle
OpenMP task distributor feeding eight blocking worker streams.  Rotation-count
timing alone does not recover the authoritative worker-to-particle handoff.
The next bounded discriminator is therefore a passive per-pool OpenMP owner
trace in RELION and the matching RECOVAR worker trace, reusing the EM program's
fail-closed capture/provenance pattern.  No further 0--200 or stress-matrix run
is authorized until that observed schedule is replayed at the iteration-1 raw
accumulator boundary.  No generic RECOVAR suite ran.

The passive native trace is now qualified, and it changes the interpretation
of the long boundary.  Isolated RELION commits `70dbf37e`/`33c6acfe` add an
environment-gated recorder at the actual CUDA worker claim site; with the
environment unset the code path is inert.  Build job `12965076` completes in
`1:43`, producing binary SHA-256
`590afb03d769bccb2bcd10cb12b30a38b0f9a77a25440d39dddf67f459a56ceb`
and full instrumentation-patch SHA-256
`e813f4eb2be7045121b33914a224ff3992e0a9b928f2bb4ba784f7ba43088415`.
The first capture exposed and fail-closed on a recorder placement in the CPU
rather than CUDA worker loop; no scientific iteration completed in that run.
Focused validator/contract checks after the correction pass `7/7`, together
with Bash syntax and Ruff checks.

Native trace jobs `12965214` and `12965225` then complete the full 0--200
trajectory on physical H100 UUIDs `GPU-e2c3190a-...` and
`GPU-ddb1592d-...` in `4:57` and `4:50`.  Both sealed 200-row iteration-1
schedules pass the exact 24-particle-pool/eight-worker topology check.  Every
worker owns exactly 25 particles, but the two native assignments differ at
169 of 200 positions.  Despite that owner variation, each iteration-1 map
matches the canonical native run at FSC-AUC about `0.999999999954`, and all
3,000 pose/translation choices agree.  Worker labels are therefore genuinely
runtime-nondeterministic and are not an invariant production schedule.

The newly completed full native-vs-native audit is more important than an
owner replay.  Against the canonical native trajectory made on physical UUID
`GPU-235ec3bc-...`, both new native runs first fail the frozen `0.999` map gate
at iteration 31 and reach their minima at iteration 119: `0.6676413803940733`
for the e2 run and `0.6674598091049447` for the ddb run.  Their iteration-200
FSC-AUC values are `0.8998051966651099` and `0.9095889691013362`, and their
first hard particle differences versus canonical are already iteration 19.
The two new native trajectories are substantially closer to one another, but
still first fail at iteration 130, reach minimum FSC-AUC
`0.9648137755552504` at iteration 176, end at `0.9933418978829993`, and first
differ in hard particle state at iteration 72.  Map-report SHA-256 values are
`22ee685c68f7d8de6e8b8c40a396e44a34d1230a73ee3bde279b84a99b1cef29`,
`599940632b64d9b1afc15e42e0f5c779a57bc10862d5b5a4a7b538167e124ed8`,
and `211057b72394105a4dbbcfe4e0bd294f40fb75290c387235593feef0305aa8b9`.

Consequently, literal independent-run 0--200 identity is not a valid RELION
parity contract: native RELION itself violates it strongly across physical
H100s.  The frozen suite's same-physical-GPU requirement remains mandatory,
and native run-to-run variability on one UUID must be measured before changing
the candidate gate or adding replay logic.  Same-UUID repeat job `12965740` is
running on the e2 H100 for that discriminator.  The other 21 robustness cells
remain held until this native envelope is sealed.  No generic RECOVAR tests
ran.

The same-UUID discriminator confirms that the late trajectory is not a
point-valued native oracle.  The second e2 run (`12965740`) completes on the
same physical H100 as `12965214`; its 200 particle-to-worker assignments differ
at 175 positions even though every worker still claims exactly 25 particles.
Full CPU-Slurm audit `12965960` then compares the two independent native runs
at every checkpoint from 0 through 200.  The old fixed `0.999` gate first
fails at iteration 31, the minimum native-vs-native FSC-AUC is
`0.6792833012912778` at iteration 118, and iteration 200 is
`0.8892913459263536`.  Hard particle state first differs at iteration 19; by
iteration 200, pose and translation match fractions are `0.7486667` and
`0.6666667`.  The nonzero Slurm result is the expected old-gate
classification, not an execution failure.

This makes the earlier fixed point-oracle contract internally inconsistent:
native RELION itself can be far below `0.999` on the same input, binary,
physical GPU, and command.  The replacement contract remains fail-closed but
uses native repeatability as the scientific floor:

- at every map checkpoint, the candidate must meet the lower of the frozen
  `0.999` gate and the observed same-GPU native repeat floor;
- at every map checkpoint, candidate-to-GT FSC-AUC may trail the best native
  repeat by no more than the frozen `0.002` allowance;
- at every positive checkpoint, the candidate's nearest complete native hard
  particle state may differ for no more particles than the worst native pair;
- exact discrete search topology is mandatory through the first native hard
  particle-state branch; later adaptive topology and all continuous schedule
  quantities remain reported diagnostics rather than a false point oracle;
- input identity, default parameters, artifact topology, RELION executable,
  candidate CUDA binary, source heads, and same physical GPU remain mandatory.

The static worker-8 baseline now has zero failures under that contract.  The
sealed 201-checkpoint map report is reclassified without recomputing its maps
or shell curves: the parent map and shell hashes remain
`535f6da63eaed983f862348ad914600bf2fcff36417a13f3a087770563f42895`
and `9f987819f5f1c1c0189b293d2967724e896e00d1dee8047bd78dd8e5fc3ca5e6`.
The original strict point audit has 70 failures beginning at iteration 131;
the repeatability-calibrated audit has zero failures.  Its minimum
candidate-best-native FSC-AUC is `0.9831481130943629`, while the minimum GT
delta remains `-0.0009750295208153792`, inside the frozen `-0.002` allowance.
The derived report SHA-256 is
`b12c06e31c5c0e0f3d0d73284a01b51e8b1799e8470af0e81e8975ee4e297199`.

Focused CPU-Slurm state audit `12967101` completes in `1:24`.  All 200 positive
particle checkpoints pass, the first four-repeat native hard-state branch is
iteration 24, and the candidate-minus-native mismatch ceiling never exceeds
zero.  All gated pre-branch search-topology checkpoints pass.  The separately
retained strict continuous/adaptive diagnostic first differs at iteration 6
and differs at 110 checkpoints, so the evidence is not hidden or rewritten.
State report SHA-256 is
`2ea5d94d8e1f0975db4ce0f406cf0f825c61c4738b3308f3c3622fa71dc819ce`.

Two provenance guards also behaved as intended.  State-only job `12966993`
stopped in 16 seconds when a mistakenly selected archived candidate named a
different H100 UUID; pending job `12966333` was cancelled before allocation
after the same path error was found.  The valid candidate is
`vdam_worker8_formalgpu_attempt1_0f3642045_20260825`, not the later cross-GPU
diagnostic root.  Correctly pinned combined recomputation `12967102` was then
cancelled at zero elapsed time because the sealed parent map/shell evidence
could be reclassified losslessly.  No science outputs were deleted and no GPU
was allocated for these corrections.

Focused contract evidence is `16 passed` for the state-boundary and 200-step
suite checks, plus `27 passed` for the map, state, and sealed-reclassification
auditors; Ruff, Bash syntax, and `git diff --check` pass.  No generic RECOVAR
or generic long-test suite ran.

The K=1 baseline gate is therefore sealed.  The next execution step is the
remaining 21 full 0--200 GUI-default cells already frozen in
`vdam_k1_gui_default_full_suite_v1.json`: outlier and junk fractions,
uniform/anisotropic/Kent pose distributions, white/radial and low-to-very-high
noise, no-CTF, contrast/noise scaling, translations, resolution, small-N, and
midscale data.  Each cell must carry its own same-GPU native repeat panel and
the same map, GT, hard-state, topology, runtime, and provenance gates; failures
will be localized and fixed before moving to parameter, scale, real-data, or
K>1 qualification.

The first robustness cell now passes that complete contract.  `vdam-gf02`
(`k1-16`: anisotropic poses, outliers, and high noise) runs all checkpoints
0--200 plus four independent native RELION trajectories in one H100 allocation
as Slurm job `12967505_2`.  The panel completes in `55:35` on physical UUID
`GPU-202f2d43-...`, with science source `db89de857` and manifest SHA-256
`3e5be259d1961552c4a34c3cfcb1771670f2d9688c1b57dcdea687afd43df70c`.
The four native runtimes are `254.7305`, `253.1328`, `252.7026`, and
`250.5184` seconds.  RECOVAR takes `1691.4024` seconds, or `6.6876x` the
median native runtime, so runtime parity remains explicitly unqualified even
though the correctness gate passes.

Focused CPU audit `12970038_2` completes in `25:31` with exit zero and peak RSS
`768116K`.  The repeatability-calibrated map gate passes all 201 checkpoints:
minimum candidate-best-native FSC-AUC is `0.9025892080924967`, and minimum
candidate-minus-best-native GT FSC-AUC is `-0.0009234422988315949`, inside the
frozen `-0.002` allowance.  Hard particle state passes all 200 positive
checkpoints, the first native hard-state split is iteration 54, and all exact
pre-split topology gates pass.  The separately retained strict adaptive and
continuous diagnostic fails first at iteration 10.  Map, shell, state, and
status SHA-256 values are respectively
`e998e17e8f0efe166d1b6c33c7c4aa03836eb12d7a9cd371545a73aa951fe4f1`,
`b5a24cfe7e90dbac7468dbbf6275d0efdb26b5143b2d60a7bf0e0e9d30970d7a`,
`308e752718a16ccb247568496f77f9626ab081bd7ba156c2afa051431c8db68c`,
and `58bfdbfeb51c1d6901e7419677e1ac8d8df3226d0474b8934d657118acb4a78e`.

The first audit attempt exposed an orchestration-schema mismatch rather than a
science failure: new panels seal `run_provenance.json` and
`paired_gpu_uuid.json`, while the auditor accepted only the older
`provenance/*.txt` layout.  The current auditor now validates both formats,
including launch/completion source identity, clean tracked state, CUDA-library
digest, and exact paired GPU identity.  Focused provenance/map/state/wrapper
coverage passes `28/28`; Ruff, Bash syntax, and diff checks pass.  The fix is
local implementation commit `7d78beb89` and remains unpushed.

With the pilot qualified, full GUI-default robustness array `12970815` is now
running `vdam-gf03` through `vdam-gf22` at four concurrent H100s.  This covers
the remaining Kent/anisotropic/uniform pose, junk/outlier, noise, no-CTF,
translation, contrast, resolution, small-N, and midscale cells.  Each task
runs one RECOVAR trajectory and four same-allocation native repeats before it
can write `SCIENCE_COMPLETED`; the current CPU envelope audit is applied only
after that marker.  No generic RECOVAR test, full suite, or long-test ran.
