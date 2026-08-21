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
