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
