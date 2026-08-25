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
