# K=1 case-22 fresh-order numbered-prefix gate

The fresh K=1 physical-order candidate closes the previously observed
iteration-2 map and posterior divergence to approximately the independent
RELION repeat floor. The same candidate remains above the fixed FSC gate and
on the exact RELION controller topology through iteration 3. This is a
three-iteration causal prefix, not a completed
case-22 trajectory, so it does not change the fixed `28/34` strict or `32/34`
topology scorecard.

The machine-readable record is
`docs/math/em_k1_case22_fresh_order_prefix_scorecard_v1.json`.

## Fixed progress metric

| Boundary | Historical RECOVAR | Fresh-order candidate | Improvement |
|---|---:|---:|---:|
| Iteration-2 merged signed FSC-AUC | 0.9999994611 | 0.9999999998805353 | 4,510.96x smaller deficit from 1 |
| Iteration-2 Pmax relative L2 | 0.00429 | 0.0000070238815 | 610.77x smaller |
| Iteration-2 hard pose/shift mismatches | not used as the primary metric | 1 / 3,000 | one native-repeat-scale boundary |
| Iteration-2 support-count mismatch | not used as the primary metric | 461 / 3,000, each by one sample | bounded latent residual |
| Iteration-3 merged signed FSC-AUC | not captured in the first prefix | 0.9999985920055022 | passes the fixed 0.995 gate |
| Iteration-3 Pmax relative L2 | not captured in the first prefix | 0.0038268203361925997 | first material posterior recurrence |

Iteration 1 has merged cross-engine FSC-AUC `0.9999999999771184`, exact Pmax
for all 3,000 particles, exact support counts, and no hard pose/shift mismatch.
Iteration 2 has merged cross-engine FSC-AUC `0.9999999998805353`; its merged
GT FSC-AUC differs from RELION by only `+3.2268e-8`.

## First remaining unequal boundary

The only iteration-2 hard mismatch is stack image 1204. A passive native
fine-score capture contains all 2,349,792 active hypotheses. RELION's selected
candidate has raw `diff2=130.35562133789062`; the runner-up has
`diff2=130.3556365966797`. The margin is one float32 ULP. The two candidates
share the same native orientation and translation priors.

RECOVAR's production direction-prior value for native coarse direction 751 is
bit-identical to the native captured log prior (`-5.109306812286377`). A
one-particle replay had a slightly different prior and rounded the two final
scores together, so that replay cannot support a tie-break code change. The
current classification is residual arithmetic sensitivity at the native-repeat
scale, not a demonstrated candidate-order defect.

A live in-memory RELION capture at the iteration-2 scoring boundary rules out
startup noise weighting as that residual. All 65 float32 inverse-noise shell
values consumed by the scorer are bit-identical between fresh RECOVAR and
RELION in both halves. The earlier shell-1-through-4 discrepancy came from
RELION model-STAR decimal serialization: the serialized replay is respectively
696, 157, 1,894, and 4,633 ULP from the live inverse-noise values. It is not a
valid production fix target.

The serialized one-particle replay also had 2,349,760 valid fine tuples versus
2,349,792 active native RELION tuples, a difference of 32. This is not yet a
fresh-run finding because pass-1 parent selection can inherit serialized-state
differences. Exact rotation-matrix alignment shows 64 RELION-only tuples in
two complete `8 rotations x 4 translations` child blocks and 32 RECOVAR-only
tuples in one complete `8 rotations x 4 translations` child block. That
structure points to coarse-parent routing rather than fine-grid construction
or padding. The compact diagnostic now records global candidate count and
score/posterior extrema as scalars for the next stop-after-target capture, so
this boundary can be tested without dumping the full 17.5-million-entry mask.

Native capture job `12276441` wrote the complete fine-score sidecar before its
optional BPref geometry capture exceeded the configured 2 GB safety cap. The
fine sidecar is therefore usable for this score boundary; the failed job is not
an inert full-trajectory claim.

## Current bounded gate

Slurm job `12276962` completed the same guarded candidate through numbered
iteration 3 on one H100. Its run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fresh_order_it3_20260812T0040ET/`.
The merged cross-engine FSC-AUC values are `0.9999999999980128`,
`0.9999999999667332`, and `0.9999985920055022` for iterations 1--3;
current-size and HEALPix-order topology remain exact (`56/3`, `60/3`, `80/3`).

The earlier compact pass-2 route was abandoned after its ordinary scorer
materialized a 17.05-GiB bucket before the bounded dump. It produced no
scientific artifact and is not used below.

## Exact native coarse boundary

Native capture job `12280812` and focused RECOVAR dump job `12279997` now
locate the first discrete production mismatch before pass 2. For stack image
1204 at physical iteration 2, native RELION retains 73,431 coarse parents and
RECOVAR retains 73,432. After exact rotation/translation alignment, the native
parent set is a strict subset and the only RECOVAR-only key is rotation 4188,
translation 15. That one parent creates exactly 32 extra fine children.

The complete native boundary shows:

| Stage | Result |
|---|---:|
| Candidate topology | exact, 1,069,056 candidates |
| Valid prior support | exact, 631,968 candidates |
| Orientation-prior support and finite values | bit-exact |
| Translation-prior maximum absolute residual | 1.9073486328125e-6 |
| Target translation-prior residual | exact |
| Common best candidate | exact, `(13039, 20)` in RECOVAR coordinates |
| Target raw score relative-to-best delta | +1.5735626220703125e-5 RECOVAR minus RELION |
| Target combined log-weight relative delta | +1.52587890625e-5 |
| Native/RECOVAR support mismatches | exactly 1 |

This makes the first material mismatch the coarse Gaussian score. Candidate
generation, order, prior support, target priors, posterior normalization, CUB
sort/scan, and threshold comparison are downstream or closed at this boundary.
The native capture is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_coarse_boundary_h100_retry2_20260812T0255ET/capture/part229_stack1204.coarse-v1.bin`
(SHA-256 `724370b46c7a04f415764aa8d73c83f54184e022572f2dc1778056a19754cd9b`).

Focused causal job `12281174` enabled the existing RELION-exact coarse
operand/scorer route, ran for 122 seconds, wrote only the target boundary, and
stopped before pass 2/M-step. It produces exactly 73,431 parents, with zero
native-only and zero RECOVAR-only parent keys. Across all valid candidates its
centered raw-score residual versus native improves from median/p95/max
`1.335e-5 / 3.934e-5 / 1.073e-4` to
`0 / 1.526e-5 / 4.578e-5`. The path therefore causally closes this discrete
defect but does not yet make every raw score bit-identical.

Two additional stop-after-dump factorial arms reduce the causal change further.
Job `12281516` used the RELION CUDA Gaussian reduction, native `sincosf`
translation, and exact CUB support but retained the ordinary RECOVAR image/CTF
operands. Job `12281588` also removed native `sincosf`, leaving the RELION CUDA
Gaussian reduction as the only score-path change. Both arms retain exactly
73,431 parents with zero support mismatches, and both make the target raw score
relative to the common best exact. Their full-table centered raw residuals are
median/p95/max `0 / 1.526e-5 / 4.578e-5`.

Therefore the minimal demonstrated cause of the discrete parent defect is the
ordinary RECOVAR/JAX coarse reduction arithmetic/order. Candidate generation,
priors, translation phases, special per-image FFT/CTF operands, normalization,
and the CUB cutoff are not required to explain or repair this boundary.

The exact dump and report are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fresh_exact_coarse_stack1204_it2_20260812T0315ET/`.
Bounded two-iteration prefix job `12281298` tests the diagnostic all-exact arm
on the complete 3,000-particle posterior/map state. Job `12281662` repeats that
same bounded gate using only the minimal CUDA Gaussian score reduction. Native
component job `12281434` captured reference-norm, cross-term, and live operands
for the target and common best; its individual files validate, while its
directory-level provenance header incorrectly declared three expected follower
ranks although the explicit particle was owned by one. The launcher is
corrected to one expected follower for future reproduction.

## Complete two-iteration coarse intervention

The minimal CUDA-Gaussian prefix job `12281662` finishes with iteration-2
merged cross-engine FSC-AUC `0.9999999999667588`, Pmax relative L2
`5.9364088535e-6`, 121 support-count mismatches, and one hard pose/shift
mismatch. The matching control has 461 support-count mismatches, while the
all-exact coarse arm has 122. Thus the coarse intervention causally removes
340 discrete support errors, but it does not move the worst soft-posterior
error or the numbered map FSC materially. It is a real local repair, not the
remaining dominant K=1 fix.

The worst remaining iteration-2 Pmax particle is stack image 1574. Its native
and minimal-coarse candidate topology, support, best coarse tuple, direction
priors, and translation priors all agree. The complete coarse posterior total
variation is only `1.0390690454e-6`, so the material Pmax residual begins in
fine scoring.

## First fine-score mismatch for the worst Pmax particle

A fresh in-process two-row capture avoids the STAR-serialization artifacts of
the earlier one-particle replay. For stack image 1574 and translation 56:

| Fine tuple | Native raw `diff2` | RECOVAR raw `diff2` | Difference |
|---|---:|---:|---:|
| rotation 41 / RECOVAR row 17 | 144.39877319335938 | 144.39877319335938 | exact |
| rotation 48 / RECOVAR row 48 (best) | 143.056884765625 | 143.05685424804688 | -2 float32 ULP |

The resulting best-versus-competitor raw margin differs by exactly
`-3.0517578125e-5`. Its posterior-max change is `+7.1026093250e-6`, which
reproduces the full 3,000-particle maximum Pmax error. Direction and
translation priors are bit-exact. This reduces the dominant iteration-2 soft
posterior gap to one best fine tuple before normalization.

The exact native `PPref` projector reproduces both captured native references
bitwise. On the best tuple, RECOVAR's incoming projected reference differs by
relative L2 `5.6706771179e-8`, and substituting it alone lowers native raw
`diff2` by one ULP. RECOVAR's translated image differs by relative L2
`7.3102294674e-8`, and substituting it alone also lowers native raw `diff2` by
one ULP. Correction weights and the high-resolution contribution are exact.
The ordinary JAX score is two ULP low; the source-faithful fused CUDA reduction
on RECOVAR operands is one ULP low. Therefore a reduction-only fine-score fix
is insufficient: the incoming iteration-1 reference state and the image path
both contribute before posterior normalization.

Focused job `12282767` captured RECOVAR's unshifted fine-score input at this
same live boundary and stopped immediately after writing it. The corrected,
authoritative one-tuple GPU replay is job `12283141`; job `12282844` used an
incorrect report-only conversion for the native translation angle and is
superseded.

The translation angles in job `12283141` are bit-exact. Replaying RECOVAR's
unshifted input through the exact CUDA translation reproduces the live RECOVAR
translated operand bitwise. The first unequal boundary is therefore the
unshifted score input, before translation: relative L2
`8.364632316960114e-8`, maximum absolute difference
`1.1920928955078125e-7`, and 847 of 1,461 complex values differ. The optimal
positive scalar is `1.0000000136036735`, but the residual relative L2 after
that scalar remains `8.253271911854404e-8`; this is not a single normalization
factor error. Native input replay through the same CUDA translation differs
from native production by only `2.3182520824272172e-8`, the remaining
compiler/code-generation floor for that replay.

This closes translation as the source of the observed image mismatch and moves
the first unresolved boundary to normalized-real image construction, soft-mask
background subtraction, or the real-to-Fourier transform. Native passive
preprocessing capture job `12283196` and its dependent stage-by-stage analysis
job `12283200` tested only those stages for stack image 1574; they did not run a
new RECOVAR trajectory. The passive capture remained map-stable against its
control at both numbered iterations, with minimum merged/half-map FSC-AUC
`0.9999999999870683` and exact selected topology.

Normalization plus integer shifting is bit-exact for all 16,384 real pixels.
Starting from that exact image, RECOVAR's deterministic default soft-mask
reduction produces a background value two float32 ULP from native RELION. This
changes 13,940 masked pixels at very small magnitude and 530 of 1,860 complex
Fourier values, with Fourier relative L2 `2.5018971530180866e-8`. Feeding the
native masked real image through RECOVAR's per-image FFT reproduces all 1,860
native Fourier values bitwise, so the FFT is closed.

RELION's deployed mask reduction atomically accumulates 128 blocks into one
slot per lane and is schedule-dependent. RECOVAR's diagnostic native-atomic
replay happened to reproduce the captured native background, every masked
pixel, and every Fourier value bitwise on its first invocation. Across eight
additional invocations it produced seven background bit patterns and did not
hit the captured value again. Thus this is a demonstrated preprocessing cause,
but it cannot be made a deterministic exact-parity fix by simply enabling the
native atomic schedule. The default mask discrepancy is also smaller than the
complete live unshifted-input discrepancy. Focused job `12283431` therefore
dumps the live Fourier image and every scalar/pixelwise correction immediately
before the score input to localize the remaining component.

## Same-run fine-image input and native-repeat envelope

Focused H100 job `12283990` completed in 8 minutes 21 seconds and passively
captured, for stack image 1574 at iteration 2, every native windowed Fourier
value, signed `local_Fctf`, `pixel_correction`, and corrected fine-score image.
The capture root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1574_fine_image_input_capture_20260812T0645ET/`.

The same-run boundary closes completely:

| Boundary | Result |
|---|---:|
| Native preprocessing Fourier to native host scoring Fourier | 3,720 / 3,720 float32 components bit-exact |
| Native host `Fourier * pixel_correction` product | 3,720 / 3,720 components bit-exact |
| Native `pixel_correction` versus negative RECOVAR correction | 1,461 / 1,461 pixels bit-exact |

The correction sign is the established RELION-versus-RECOVAR CTF convention,
not a numerical defect. An independent host binary64 evaluation from the
source STAR followed by RELION's float32 cast also matches RECOVAR's saved
correction for all 1,461 pixels. CTF evaluation, reciprocal construction,
multiplication, and the host-to-device input boundary are therefore closed.

The remaining stack-1574 difference is smaller than native repeatability. Two
independent RELION iteration-2 runs have Pmax relative L2
`1.7975910502174958e-5`, while fresh-order RECOVAR versus the selected native
run is `6.475004069871885e-6`. The two native runs differ in the captured
normalization-plus-mask Fourier input by relative L2
`2.5554650078131437e-6`; this is larger than RECOVAR's
`8.364632316960114e-8` input difference from the selected native run. The
stack-1574 two-ULP fine-score residual is consequently classified as inside
the native repeat envelope, not as an actionable RECOVAR bug.

Iteration 3 also requires a repeat-aware interpretation. The selected
`currenthead_ref` RELION trajectory differs from two mutually agreeing RELION
repeats in Pmax by relative L2 about `0.003532`; those two repeats agree with
each other to `4.2091974026696356e-5`. Fresh-order RECOVAR differs from the
selected outlier by `0.0038268203361925997`, but from each agreeing repeat by
about `0.00148`. Map FSC is more decisive: the two agreeing native repeats
have merged FSC-AUC `0.9999999999848632`; RECOVAR versus them is
`0.9999997938157471` and `0.9999997942982827`, whereas the selected native
outlier versus them is only about `0.9999988447`. Thus the fresh-order
iteration-3 map is closer to the native consensus than the selected reference
run itself.

The next first-divergence gate moves to iteration 4. It must compare the same
fresh-order RECOVAR prefix against all three existing native trajectories and
use signed FSC/FSC-AUC plus controller topology as the primary decision; Pmax
against one schedule-sensitive native run is no longer sufficient evidence of
a code defect.

## Repeat-aware iteration-4 gate

Pinned H100 job `12284182` completed the four-iteration prefix in 2,398
seconds. The output root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fresh_order_it4_pinned_20260812T0625ET/`.
All four numbered iterations pass the strict signed FSC/FSC-AUC and GT-delta
gates against both agreeing native trajectories. At iteration 4, RECOVAR and
RELION exactly agree on `current_size=70` and HEALPix order 4. The merged
cross-engine FSC-AUC is `0.9999867024772974` against `robust_ref` and
`0.9999867017948157` against `fulltraj_ref`; the corresponding merged GT
FSC-AUC deltas are positive `1.3077524260995954e-5` and
`1.3056468526173592e-5`.

The first repeat-robust latent mismatch is now sharply bounded. RECOVAR has no
hard pose or translation differences at iterations 1 or 2. At iteration 3,
exactly four of 3,000 particles differ from the mutually agreeing robust and
full-trajectory RELION runs: stack images 941, 992, 1514, and 1770. Three
remain different at iteration 4: 992, 1514, and 1770. The iteration-3 Pmax
relative L2 is about `0.00148`, versus a native repeat floor of
`4.2091974026696356e-5`; it grows to about `0.007488` at iteration 4.

Stack image 992 is the next bounded target because its different rotation and
translation persist and become one of the two dominant iteration-4 posterior
errors. RECOVAR job `12284649` dumps its complete iteration-3 candidate set,
raw score, priors, posterior, support, and winner, then stops. Native RELION
job `12284675` captures the corresponding complete fine-score boundary. The
comparison order is tuple identity, raw `diff2`, priors, centered log weights,
normalization, support cutoff, and winner.

## Exact stack-992 iteration-3 winner attribution

Job `12284649` completed the requested stop-after-target boundary in 4,496
seconds. Its only scientific payload is the 10.1-MB stack-992 pass-2 dump at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack992_it3_full_boundary_20260812T0720ET/pass2/pass2_orig000991_cs080.npz`
(SHA-256 `e001e7d693c8a50b29445114e481610496649ba0241f98c6cb56fe13ec52b360`).
The native fine-score and BPref-geometry capture in job `12284675` validates
all 1,760 active candidate records and native score/prior algebra exactly.

The tuple and support boundaries close: all 1,760 native active tuples exist
in RECOVAR, and both engines retain exactly the same 498 significant tuples.
The unequal quantities are:

| Quantity | Relative L2 | Maximum absolute difference |
|---|---:|---:|
| raw `diff2` | 0.00045128099714698 | 1.108642578125 |
| centered raw score | 0.00005955839123586985 | 0.00390625 |
| orientation log prior | 0.0012879705658397763 | 0.025127410888671875 |
| translation log prior | 4.016911749389204e-8 | 2.384185791015625e-7 |
| centered combined log weight | 0.000432386875939445 | 0.01806640625 |
| normalized active posterior | 0.0028178595838177596 | 0.00042097882359871247 |

The native winner is local rotation/translation `(98, 87)`, mapped to RECOVAR
row `(194, 87)`. Its native margin over the RECOVAR winner `(38, 59)`, mapped
to `(54, 59)`, is only `0.0039522647857666016` log units. Substituting only
RECOVAR raw scores leaves the native winner ahead by `0.0027315616607666016`.
Substituting only RECOVAR priors also leaves it ahead, by
`0.0005182027816772461`. Applying both reverses the margin to
`-0.0007025003433227539`. The exact additive attribution is:

| Contribution to native-winner minus RECOVAR-winner margin | Log units |
|---|---:|
| Native complete margin | +0.0039522647857666016 |
| Raw-`diff2` substitution | -0.001220703125 |
| Orientation-prior substitution | -0.0034341812133789062 |
| Translation-prior substitution | +1.1920928955078125e-7 |
| RECOVAR complete margin | -0.0007025003433227539 |

Thus there is no missing candidate, support-cutoff, tie-break, or translation-
prior defect at this first repeat-robust hard split. It is a coupled
sub-margin flip dominated by the learned orientation-prior state, with a
smaller centered raw-score contribution.

The complete learned direction-prior vector first differs materially after
iteration 2. Against the two agreeing native trajectories, RECOVAR relative
L2 is about `0.000724908`/`0.000735287` for halves 1/2, while native-repeat
relative L2 is only `1.0844e-8`/`8.1879e-9`; zero support is identical. The
apparent iteration-1 vector residual (`7.24e-5`/`7.31e-5`) is bounded by the
six-decimal model-STAR serialization error: iteration-1 Pmax, support, and hard
poses are exact, and those priors are integer winner counts. Therefore the
first actionable learned-state mismatch remains the iteration-2 soft-posterior
aggregation, not iteration 1.

The machine-readable boundary report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack992_it3_full_boundary_20260812T0720ET/analysis/K1_FINE_SCORE_BOUNDARY_STACK992_IT3_WITH_WINNER_ATTRIBUTION.json`
(SHA-256 to be recorded with the committed scorecard). The next focused gate
is an iteration-2 posterior-aggregation panel: compare native and RECOVAR
per-particle weights for a small set that spans the largest direction-prior
residuals, then compare their direction-bin operands before global reduction.

## Earlier iteration-1 normalization boundary

An exact native update capture for stack image 1462 exposed an earlier scalar
boundary than the iteration-2 learned-prior drift. RELION's iteration-1
normalization update is `25091866.04904771` in RECOVAR's unnormalised Fourier
units. The historical RECOVAR path produced `25091868.5`, yielding a one-ULP
low next-iteration image factor (`0x3f7ad9e1` instead of native
`0x3f7ad9e2`). Candidate weights are bit-exact for all 928 captured fine
candidates, so this is an accumulation boundary rather than a posterior-set
error.

The discrepancy splits into two source-level reductions:

1. RELION's `powerClass` emits an atomically accumulated shell spectrum for
   the high-shell normalization tail. RECOVAR incorrectly reused the separate
   block-tree `highres_Xi2` scalar used by fine scoring. The exact high-shell
   native value is `12060356.349819839`; the old RECOVAR value was
   `12060360.0`.
2. RELION's Wavg kernel forms image power after applying each float32
   translation and retains one float32 `wdiff2` accumulator per Fourier pixel
   until a host RFLOAT sum. RECOVAR used untranslated image power and collapsed
   `A2`/`XA` to GPU scalars first. Native low-shell image power is
   `18968761.142488837`; the historical RECOVAR value is
   `18968759.994586825`. Replaying RECOVAR's captured translated images with
   the bit-exact candidate weights gives `18968761.057280898`, closing about
   `1.06` of the `1.15`-unit image-power gap.

The power-spectrum-only counterfactual in job `12288900` makes the target
normalization factor bit-exact and reduces its full update-input error by about
52%. The combined captured-operand replay at commit `74dae7fb` gives
`25091865.932280898`, only `0.11676681041717529` from native, a 95.2% closure
relative to the historical 2.451-unit error.

The first combined integration attempt, job `12289683`, failed before state
serialization because it incorrectly supplied the noise-weighted score image
to the Wavg power boundary. Commit `58394280` instead routes the raw masked
image through the exact CUDA Wavg translation. Corrected one-iteration job
`12289913` completed `0:0` in `00:01:50`. For stack image 1462, the resulting
RECOVAR factor is bit-exact with the captured native in-memory value:
`0x3f7ad9e2` in both engines. Across all 3,000 particles, the combined path
changes 1,213 factors relative to the historical control, with a maximum
distance of two float32 ULP and a 95th percentile of one ULP in each half.
All 3,000 Pmax values and all 3,000 significant-support counts remain exact;
hard assignments are unchanged. The iteration-1 merged signed non-DC
FSC-AUC is `0.9999999999978554`.

The exact tagged-particle report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_translated_wavg_norm_it1_retry2_20260812T1110ET/analysis/K1_STACK1462_NORM_STATE_BOUNDARY.json`.
The population A/B report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_translated_wavg_norm_it1_retry2_20260812T1110ET/analysis/K1_NORM_POPULATION_AB.json`.
Both changes remain opt-in diagnostics. Two-iteration high-shell-only job
`12289068` is running, and combined two-iteration causal gate `12290158` is
pending. Neither changes the fixed `28/34` strict, `32/34` topology,
`34/34` evaluated scorecard before its iteration-2 comparison passes.

## Normalization propagation and raw-score closure

The clean same-executable control `12290572`, high-shell-only arm `12289068`,
and combined arm `12290158` complete the bounded two-iteration factorial. The
high-shell correction reduces iteration-2 Pmax relative L2 from
`6.2472020820489016e-6` to `5.161458798740429e-6` (17.4%) and support-count
mismatches from 79 to 76. Adding translated Wavg gives
`5.008111676411559e-6` and leaves support mismatches at 76. Merged signed
FSC-AUC remains saturated at `0.9999999999664306`,
`0.9999999999667298`, and `0.9999999999665399`, respectively. Normalization
is causal for soft posterior state but is not the dominant trajectory cause.

The stop-at-first-target job `12292842` nevertheless closes a concrete raw
score. For stack image 1462 the corrected factor is the native float32 word
`0x3f7ad9e2`. It reduces native-host-versus-RECOVAR preprocessed-image
relative L2 from `1.265141074023961e-7` to `4.540843268410586e-8`, and the
corrected score-input relative L2 from `2.3217287954294786e-7` to
`6.530776649444363e-8`. Mismatching float components fall from 2,249 to
1,230 of 2,922. The previously first unequal native production raw score is
then bit-exact at `120.35405731201172`. The remaining image-input floor begins
at RELION's schedule-dependent soft-mask background reduction; a native-style
atomic replay is nondeterministic and is not promoted as an exact fix.

## Significant-pruned direction-mass bug

The next first wrong intermediate is a source bug in the large
membership-chunked sparse pass-2 route. RELION's `collect2jobs` accumulates
`pdf_direction` from the significant-pruned weights used by
`storeWeightedSums`. RECOVAR's unchunked route did the same, but its chunked
route accumulated `rotation_posterior_sums` from the unpruned posterior.
Case 22 exercises that chunked route.

Commit `61157af7` routes the chunked statistic through `mstep_probs` when
RELION fine M-step pruning is active. Commit `2fb00d91` additionally proves
that pruning decreases total direction mass, so chunked/unchunked equality
cannot pass vacuously. The focused chunking test passes all four parameter
cases; the normalization panel passes 12/12.

Same-H100 job `12292691` closes the immediate learned-prior boundary. Half-1
relative L2 falls from `7.202765521650175e-4` to
`5.480119744181026e-5` (92.4% closure), and half 2 from
`7.332348967195873e-4` to `5.164298054672068e-5` (93.0%). The half-1 retained
mass is `1488.5108335142213`, only `2.9862e-5` from the native
significant-pruned replay. Iteration-2 map FSC-AUC remains
`0.999999999966509`, as expected because this prior is first consumed by
iteration 3. Three-iteration job `12294318` is the only downstream causal
gate; it tests whether the correction removes the known stack-992
orientation-prior/winner split before any complete trajectory is attempted.

The frozen scorecard remains `28/34` strict, `32/34` topology, and `34/34`
evaluated until a full fixed case passes. K=4 remains parked.

## Iteration-3 causal propagation

Pinned same-H100 job `12294318` completed the three-iteration treatment in
2,319 seconds. The direction-mass correction removes all four repeat-robust
iteration-3 hard pose/translation mismatches (stack images 941, 992, 1514,
and 1770). Identity-aligned Pmax relative L2 falls from
`0.0014794409843944163` to `0.0003689264413191558`, a 75.1% reduction, and
significant-support count mismatches fall from 65 to 5, a 92.3% reduction.
The five remaining support-count residuals are stack images 232, 262, 1241,
2233, and 2828; each differs by exactly one candidate and none changes the
hard pose or translation.

The sealed stack-992 counterfactual predicted this result before the live
iteration completed. Replacing only the old learned prior with the corrected
iteration-2 prior changes its native-winner-minus-old-winner margin from
`-0.000732421875` to `+0.0026998519897460938`. The two decisive corrected
orientation log priors are each one float32 ULP from the captured RELION
values. The live result therefore demonstrates both the operand-level cause
and its next-iteration consequence.

All numbered map and topology gates remain passing. Iteration-3 merged signed
FSC-AUC is `0.9999999931331159` against `robust_ref` and
`0.9999999930589739` against `fulltraj_ref`; merged GT FSC-AUC deltas are
positive `3.726083386335066e-6` and `3.820017639766249e-6`. The complete
artifact root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_direction_prune_it3_20260812T1240ET/`.

The next boundary is not a complete trajectory. Capture stack 232, the
largest remaining support-boundary residual with a one-candidate support
difference, through tuple identity, raw `diff2`, priors, normalized posterior,
threshold margin, and significant support. Stack 2739 is the separate largest
Pmax residual (`0.010401692754745506`) and remains a secondary target if stack
232 does not explain the remaining soft-posterior tail. The frozen full-case
scorecard remains `28/34` strict, `32/34` topology, and `34/34` evaluated
until a full fixed case is rerun. K=4 remains parked.

## Focused residual localization

The next stopped-boundary experiments refine the chronology. Stack 232 is
exact through iteration 2: Pmax differs by only `1.0521820784439861e-10` and
its retained coarse support is exactly `83095` in both engines. Its `53` versus
`52` support split is therefore an iteration-3 amplifier, not the original
iteration-2 seed.

The production RECOVAR iteration-3 coarse dump and the locally stable target
particle from native capture have identical candidate topology, prior support,
and winning coarse candidate. They differ by exactly one retained candidate,
at RECOVAR coordinates `(rotation=18202, translation=15)`. That candidate has
native/RECOVAR posterior `8.26739197e-5`/`8.28443808e-5` and a centered
preexponent residual of `+0.00205040`. The full centered raw-score residual has
median absolute `0.00181198`, 95th percentile `0.00682068`, and maximum
`0.0237503`; the orientation-prior residual is only `9.54e-7` at the 95th
percentile and the translation-prior residual is `6.10e-6`. Removing a
per-rotation median explains `84.4%` of raw residual sum of squares, while
removing a per-translation median explains only `0.3%`. This localizes the
observed stack-232 support split to rotation-dependent projection/reference
scoring before posterior normalization, not translation phase or a support
tie rule.

This is a diagnostic localization, not yet a production-causality result. The
native capture globally fails its all-particle inertness gate, although stack
232 itself retains the same hard state and support as the robust native run.
The decisive A/B is therefore a second stopped RECOVAR arm that changes only
the coarse scorer to the exact RELION Gaussian operand/reduction path.

A separate two-minute iteration-2 RECOVAR panel captured stacks 1462, 1569,
2276, and 1574. Stack 1462 has matching coarse support `12`, the same winning
candidate, exact orientation priors, zero support mismatch, and coarse
posterior total variation only `8.61e-8`; its reported Pmax residual therefore
belongs to fine pass 2, not coarse pass 1. Stacks 1569 and 2276 demonstrate a
different cutoff-sensitive regime: their selected-minimum and
excluded-maximum posterior weights are respectively
`7.99773900e-8`/`7.99761679e-8` and
`1.30199538e-7`/`1.30130033e-7`. Small upstream posterior changes can thus
produce several reported support-count changes without changing a hard pose.

Artifacts:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack232_coarse_it3_20260812T1342ET/analysis/K1_STACK232_IT3_COARSE_BOUNDARY_V2.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse_panel_20260812T1404ET/coarse/significance_orig001461_it002_cs060.npz`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse_panel_b_20260812T1407ET/coarse/`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1462_native_coarse_it2_20260812T1410ET/analysis/K1_STACK1462_IT2_COARSE_BOUNDARY.json`

## Fine negative control and repeat-stable soft-posterior target

The complete iteration-2 fine table for stack image 1462 provides a negative
control. All 384 native active tuples are present, orientation priors are
bit-exact, and both engines retain the same 195 significant tuples. The first
exact inequality is raw fine `diff2`: 107/384 values differ, with maximum
absolute residual `1.52587890625e-5` (one or two binary32 ULP at this scale).
The normalized posterior relative L2 is `1.725562371054519e-6`, while the
direction-binned posterior relative L2 is `6.274222675024169e-7`. Replaying
RECOVAR operands with the source-faithful native reduction lowers raw-score
relative L2 only from `3.800734571580958e-8` to
`3.6341973571518895e-8`. Because this native capture trajectory is itself
outside the robust repeat envelope at iteration 2, the residual is not a
production-fix target. The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1462_full_fine_it2_20260812T1420ET/analysis/K1_STACK1462_IT2_FINE_BOUNDARY.json`.

Stack image 2739 is the repeat-stable soft-posterior control. After the
direction-mass fix, RECOVAR has Pmax `0.2612946927547455`; two independent
RELION controls have `0.250893` and `0.250892`, with identical pose, shift,
and support count 40. Native coarse-capture job `12298305` retains those same
discrete values and has Pmax `0.250913`, so the target particle remains locally
valid even though passive capture may perturb unrelated particles. Its native
coarse artifact is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2739_native_coarse_it3_20260812T1435ET/capture/part1933_stack2739.coarse-v1.bin`.
The stopped stack-232 exact-coarse arm `12297876` removes the single support
mismatch and retains RELION's 52 candidates, but does not improve its soft
posterior. Coarse Pmax changes from production `0.4345438778400421` to
`0.43452826142311096`, versus native `0.434539794921875`; posterior total
variation versus the captured native table changes from `0.00015537` to
`0.00017368`. This classifies exact coarse arithmetic as a cutoff-amplifier
repair, not the remaining trajectory seed, and the broad diagnostic path is
not promoted on this evidence.

The completed production fine dump makes the structural consequence exact.
Native stack 232 has 248 unique fine rotations and 1,664 active tuples;
RECOVAR has the same 248 rotations and tuples plus eight child rotations and
32 active tuples from its one extra coarse parent. There are zero native-only
fine tuples. Thus the coarse cutoff mismatch is neither a reporting-only
support difference nor a fine-grid construction bug: it directly changes the
evaluated fine candidate set.

Stopped RECOVAR stack-2739 coarse job `12298296` is the next staged gate. The
matching native fine-side job `12298637` completed in 517 seconds and wrote
coarse, complete fine-score, and BPref-geometry captures under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2739_native_fine_it3_20260812T1442ET/capture/`.
Its target particle again has the same pose, shift, and support count 40, with
Pmax `0.250902`. If the RECOVAR coarse posterior is close, the first material
boundary is fine pass 2; otherwise the coarse raw-score/prior decomposition is
continued. No full trajectory is authorized by these captures.

Job `12298296` completed the coarse gate in 1,888 seconds. Candidate topology,
the best coarse tuple, and the support count `40` are exact. Coarse posterior
total variation is `7.017404100055894e-5`; the maximum posterior residual is
`6.243586540222168e-5`, versus the final Pmax residual
`0.010401692754745506`. The only discrete difference is a two-parent swap at
the cutoff: RECOVAR includes `(15173,20)` instead of native `(29447,7)`, with
posterior values around `5.21e-5`. The first material amplification is thus
the coarse-support-to-fine-candidate boundary, or fine scoring on the common
children, rather than coarse normalization itself. Fine-table job `12299407`
failed its pre-run diff-hash provenance gate in four seconds and produced no
science output. Retry `12299447` captures the production table, while exact-
coarse arm `12299506` tests whether restoring the parent set closes the fine
boundary; both stop before any M-step.
