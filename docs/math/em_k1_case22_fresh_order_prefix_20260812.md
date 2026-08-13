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

Both stopped fine arms completed. Native has 232 fine rotations and 1,280
active tuples. Production RECOVAR has 224 rotations and 1,280 active tuples:
1,248 tuples are common, 32 native tuples are missing, and 32 different
RECOVAR tuples are present. The missing eight rotations are exactly the eight
oversampled children of native's cutoff parent `(29447,7)`. Their 32 tuples
carry `0.03972759475048382` of native posterior mass. Removing only that mass
raises native Pmax from `0.2509015676827406` to the common-domain value
`0.2612816732573048`, matching RECOVAR's `0.2612628035416517`; common-domain
posterior total variation is only `9.322834516230378e-5`.

The exact-coarse arm is a causal null for this boundary. It retains the same
224/232 rotation topology and the same 32 missing native tuples; Pmax changes
by only `-2.3092870528e-7`, to `0.2612625726129465`, and common-domain TV is
`9.371408716865815e-5`. Thus the full observed stack-2739 Pmax error is
explained by the single coarse-parent swap, not by common-tuple fine scoring,
posterior normalization, or the exact-coarse arithmetic intervention. Stable
JSON reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2739_full_fine_it3_retry_20260812T1505ET/analysis/K1_STACK2739_IT3_PARTIAL_FINE_TOPOLOGY.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2739_exact_coarse_full_fine_it3_20260812T1508ET/analysis/K1_STACK2739_IT3_PARTIAL_FINE_TOPOLOGY.json`.

For the two swapped parents, native's combined margin favors `(29447,7)` by
`+0.000732421875`; RECOVAR favors `(15173,20)` by `-0.0008287429809570312`.
The raw-score substitution contributes `-0.0016651153564453125`, while the
orientation-prior contribution is only `+9.5367431640625e-7` and the
translation-prior contribution is zero. The next stopped gate therefore
substitutes native iteration-3 reference projections for only these two
coarse orientations. No full trajectory is authorized before that margin
counterfactual is known.

## Pixel-weight factorization at the flipped parent

The stopped native-reference counterfactual rules out the reference as the
source of the stack-2739 parent swap. Replacing RECOVAR's reference with the
native reference changes the raw parent margin from `-2.1453857421875` to
`-2.1455078125`, slightly farther from native `-2.1435546875`. Replacing both
the image and reference is also insufficient. The remaining structured input
is the per-pixel correction, `scale_correction**2 / sigma2_noise`.

Native iteration-2 noise dumps and RECOVAR's stopped sufficient statistics
show an exact current-size boundary at shell 31. Shell counts match exactly,
and residual terms are zero above shell 30 in both engines. RELION adds the
high-shell image power once per particle but divides by retained significant
mass; RECOVAR support-weighted that numerator, cancelling the retained-mass
denominator. The native/RECOVAR high-shell raw ratio is
`1.0010009351925655` for half 1, matching the predicted
`1490 / 1488.5108337193947 = 1.0010004403373298`. The source-faithful fix is
covered by targeted weighted-power tests.

That exact bug is secondary for the flipped parent. Correcting only the 1,112
high-shell pixels moves the native-reference margin by one float32 score word,
`-2.1455078125 -> -2.1453857421875`. The low-shell pixel-weight ratio instead
factorizes across every active shell into a global scale-squared term and a
shellwise noise term. The inferred RECOVAR/native scale ratio is
`1.0000531839184625`, predicting target scale `0.38682357169921305` versus
native model-STAR `0.386803`.

A nine-second stopped intervention multiplying only by the inverse scale
factor closes `0.00048828125` of the `0.001953125` raw parent-margin gap.
Correcting the shellwise noise update and the scale factor together reproduces
the exact-native-weight margin `-2.143798828125`. This closes
`0.001708984375` of the gap; the residual `0.000244140625` is one float32 score
quantum and is unchanged by additionally substituting the native translated
image. The first actionable source for this parent swap is therefore the
iteration-2 pixel-weight state: active-shell noise is dominant, global scale
normalization is secondary, and the high-shell image-power bug is small.

The active-shell numerator is decomposed at the same stopped iteration-2
boundary. In half 1, RECOVAR minus RELION is `0.0018037725022833628`; the
image-power contribution is `0.001403336802610844` (`77.8001%`) and the
`AA - 2*XA` contribution is `0.0004006865262677919` (`22.2138%`). In half 2,
the corresponding values are `0.0015070404835628715`,
`0.001132095846530734` (`75.1205%`), and `0.00037423391943178447`
(`24.8324%`). The active-shell RECOVAR/RELION image-power median ratios are
`1.0000257599571096` and `1.000022860207772`. Iteration-1 normalization
amplitude ratios predict power ratios within about `1.6e-6` of one, so they
cannot explain this `2.3e-5`--`2.6e-5` effect. This rules out the preceding
normalization scalar, but decomposition alone does not identify how RELION's
captured `image_power` is formed.

Stopped two-iteration treatment `12304787` tested the translated-Wavg
formation directly and failed the preregistered gate. The half-1 median active
image-power ratio changes from `1.0000257599571096` to
`1.0000258518141674`, while half 2 remains exactly
`1.000022860207772`. Active signed image-power delta is also slightly worse:
`0.0014064362441391642` and `0.001134241613742648`, versus control
`0.001403336802610844` and `0.001132095846530734`. Thus translated-Wavg
formation is rejected as the cause at this accumulator boundary. The
unvalidated source treatment is removed and no iteration-3 parent run is
authorized.

The same job confirms the independent default high-shell fix: all-shell noise
relative L2 improves to `2.7736e-5` and `2.6419e-5`, and high-shell median
native/RECOVAR ratios are approximately `1.000000256` and `1.000000215`.
That demonstrated correction is retained; it does not change the active-shell
image-power residual.

The stopped scale-statistics replay also closes the secondary factorization.
For stack 2739, both engines clip the local raw scale to exactly `0.2`, so its
own XA/AA mismatch contributes zero. The final scale difference comes entirely
from the all-particle normalization average: RECOVAR/RELION is
`0.9999469120210713` in half 1, whose inverse reproduces the target scale
inflation. After the exact `128**4` unit conversion, the population median
RECOVAR/RELION ratios are `1.000010726856586` for XA and
`1.000072096628941` for AA in half 1; half 2 gives
`1.0000091042658723` and `1.0000641335285645`. Thus the next scale boundary is
reference-power AA before the global normalization reduction. The focused
capture will stop on one strong unclipped group and compare projection values,
squared projection power, CTF posterior factors, support weights, scale masks,
per-pixel AA contributions, and their shell sums. If local operands agree but
group totals do not, the cause is reduction/order; otherwise the first unequal
operand is the cause.

Artifacts are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_noise_terms_native_retry_it2_20260812T1820ET/`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_noise_terms_recovar_it2_20260812T1750ET/`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2739_high_shell_noise_cf_it3_retry_20260812T1750ET/`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2739_scale_cf_it3_20260812T2025ET/`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2739_all_noise_cf_it3_20260812T2040ET/`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_noise_terms_recovar_it2_20260812T1750ET/analysis/noise_update_terms_components_half1.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_noise_terms_recovar_it2_20260812T1750ET/analysis/noise_update_terms_components_half2.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_scale_terms_recovar_it2_20260812T2015ET/analysis/scale_update_terms_native_units_half1.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_scale_terms_recovar_it2_20260812T2015ET/analysis/scale_update_terms_native_units_half2.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_translated_wavg_noise_it2_20260812T2110ET/analysis/noise_update_terms_components_half1.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_translated_wavg_noise_it2_20260812T2110ET/analysis/noise_update_terms_components_half2.json`

The fixed scorecard remains `28/34` strict, `32/34` topology, and `34/34`
evaluated. No full trajectory is justified before the stopped AA
operand-versus-reduction boundary identifies a causal source.

## Scale-AA first-unequal boundary

The initial jobs `12305992` and `12306274` were instrumentation-invalid or
out of memory and produced no scientific result. The corrected stopped
capture is job `12306304`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_scale_aa_orig1096_it2_aaonly_20260812T1912ET/`.
It proves that RECOVAR's local products and reductions replay exactly, but the
stack-1097 shell-AA vector already differs from RELION with relative L2
`2.4432335e-4` and median RECOVAR/RELION ratio `1.000243619`.

Native per-pixel job `12306654` moves the boundary earlier. All 333 active
pixels join exactly by Fourier coordinate and shell, and their AA values
differ before shell reduction: relative L2 `2.4830625e-4`, median ratio
`1.0002006355`, and 5th--95th percentile ratio
`[1.0000696998, 1.0003576807]`. A fixed-order replay reproduces the shell
residual. Global group reduction and shell reduction are therefore ruled out.

Candidate jobs `12306994`, `12306995`, and `12307389` then separate posterior
from projection operands. The complete native active rotation set (160,679
rotations) and all 116 translations match; there is no unmatched rotation
mass. Posterior total variation is only `1.977928e-6`. Replacing RECOVAR's
rotation weights with native RELION weights changes the shell-AA residual
from `2.4432005e-4` to `2.4443916e-4`, a closure fraction of
`-4.8748e-4`. CTF-squared values agree to relative L2 `9.7865e-8` with median
ratio `0.9999999888`. Candidate topology, support pruning, posterior weights,
translation indexing, and CTF-squared are not the dominant cause.

Job `12307739` compares the incoming `Projector::data` arrays. Their complex
relative L2 is `7.48958e-8`, with median magnitude ratio exactly 1. A GPU
replay in job `12307952` replaces RECOVAR's projector input with the native
RELION array while leaving RECOVAR projection arithmetic unchanged. The shell
AA residual changes from `2.4432005e-4` to `2.4434960e-4`, for closure
`-1.20919e-4`. The incoming reference is therefore a causal null.

The first remaining boundary is now inside one operation: texture
interpolation/projection, formation of `ref_real**2 + ref_imag**2`, or Wavg's
float32 accumulation. Native compact projection-panel job `12308183` captures
every active rotation at the same 333 pixels at iteration 2. It is designed
to distinguish projected-reference power from Wavg accumulation without
running a later iteration or complete trajectory. The fixed scorecard remains
`28/34` strict, `32/34` topology, and `34/34` evaluated; K=4 remains parked.

Job `12308183` completed in 498 seconds and identifies that operation. Native
and RECOVAR per-rotation/per-shell projected power differ by only
`1.4297574e-7` relative L2, with median ratio `0.9999999350`. Nevertheless,
combining native projected power with native posterior weights using a
high-accuracy/tree-style rotation reduction still differs from RELION's
actual per-pixel Wavg AA by `2.4847813e-4`, the same residual as RECOVAR
(`2.4829921e-4`).

Replaying RELION's two reduction levels separates the cause. The sequential
116-translation inner loop changes rotation masses by only `9.06217e-8`
relative L2. Accumulating the 164,464 rotation rows in forward float32 order,
however, reduces the per-pixel residual to `1.3704161e-6`; reverse order gives
`5.87178e-5`. The source-order float32 replay therefore closes `99.45%` of
the observed discrepancy. The first material inequality is RELION Wavg's
float32 rotation accumulation (CUDA atomics), not any incoming mathematical
operand.

The immutable reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_project_panel_it2_20260812T2030ET/analysis/projected_power_boundary.json`
  (SHA-256 `75c87f832422edb47171ecb96334ea275c960c84e7068dd5a2e7a3dd214123fa`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_project_panel_it2_20260812T2030ET/analysis/wavg_accumulation_boundary_v2.json`
  (SHA-256 `1cb7c077c8e88192093104e76e5f4fe323109961c063266c03091ae3d6eac76c`).

Focused H100 job `12308654` now tests a compact RECOVAR CUDA primitive with
the same one-block-per-rotation, per-pixel float32 atomic topology on the
saved 333-pixel panel. This is an artifact-only acceptance gate; it does not
run another EM iteration. Only if it reproduces native Wavg will the reducer
be wired behind a diagnostic scale-AA path for a stopped iteration-2 A/B.

Jobs `12308853`, `12308897`, `12309162`, and `12309318` resolve the remaining
ordering variable. A chunked CUDA replay in native orientation order gives
pixel relative L2 `1.47938e-6`, but the first stopped EM integration in
RECOVAR order gives `1.12334e-4`, only a 55% closure. Replaying identical
native terms in RECOVAR's live order reproduces that value (`1.12422e-4`).
Only 16 of 160,679 active rotations occupy the same rank before the fix.

The permutation is structural: all 20,558 eight-child fine-parent groups are
bitwise identical internally, but RECOVAR orders parents psi-slow and
direction-fast whereas RELION orders them direction-slow and psi-fast.
Stopped job `12309318` applies only that stable parent-key transpose plus the
atomic AA reducer. The live 160,679-active-rotation sequence and the full
164,464 real-rotation prefix then become bitwise identical to native RELION.
Pixel AA relative L2 falls from `2.48295e-4` to `1.50412e-6` and fixed-order
shell relative L2 falls from `2.44742e-4` to `5.27891e-7`. This closes 99.39%
and 99.78% respectively at the stopped boundary. The fixed complete-case
scorecard remains `28/34`; the next gate is the corresponding XA/BPref
accumulation boundary before a short iteration-3 trajectory test.

## Fused XA/AA Wavg boundary

Stopped H100 job `12310062` reproduces RELION's complete per-pixel atomic
issue order (`XA`, then `AA`, then `diff2`) after applying the exact fine-parent
execution order.  It stops at the same iteration-2, half-1, stack-1097
boundary; no later EM state contributes to the comparison.

Both scale sufficient statistics close to the precision of the native text
capture.  Ordinary RECOVAR XA has pixel relative L2 `7.71488e-5` and
fixed-order shell relative L2 `4.72518e-5`; the fused native-order result gives
`1.83539e-6` and `6.70922e-7`.  Ordinary AA remains `2.48300e-4` pixelwise and
`2.44743e-4` shellwise, while the fused result gives `1.47309e-6` and
`5.50279e-7`.  This rules out a separate XA operand defect at the captured
boundary and confirms the coupled fine-parent order/Wavg float32 atomic
schedule as the complete material scale-statistic cause there.

The accepted capture is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_triplet_order_stopped_it2_20260812T2120ET_retry/pass2/scale_aa_chunked_orig001096_half1_cs060.npz`
(SHA-256 `bfd01259ca085c5fd1d5919b9cf4a2173b3ac56802c6eddd99c15d7663761ec0`).
The analysis is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_triplet_order_stopped_it2_20260812T2120ET_retry/analysis/scale_xa_aa_triplet_pixels.json`
(SHA-256 `611b44efc5c355044a3d01e123a9fd2fcb658b8d523ead692ee00524f855cbb2`).
Failed job `12309946` exited before iteration 1 because its diagnostic CUDA
artifact contained only `sm_80`; it has no scientific result.  The accepted
job used the pinned `sm_90` artifact with SHA-256
`871bc03e399bdd55fdcf07d0d42c430976d82e3850d7ae5e8bcc7b04c3aead9a`.

The fixed complete-case scorecard remains `28/34`.  Short job `12310265` is
the next and only authorized downstream gate: three numbered iterations with
the native fine-parent order and fused Wavg scale accumulators, followed by
the same identity-aligned Pmax/support/hard-pose and signed FSC audits used by
the previous iteration-3 discriminator.

## Fused Wavg iteration-3 propagation

Pinned H100 job `12310265` completed `0:0` in 42 minutes 25 seconds. The
iteration-2 group-scale relative L2 against the contemporaneous instrumented
native control is `3.57417e-7` in half 1 and `4.70333e-7` in half 2, near the
six-decimal model-STAR output floor. Comparisons to the older robust and
full-trajectory references are approximately `9.1e-6`; those two older
references agree with each other at about `2e-7` but differ from the same-day
instrumented native Wavg run. The exact same-day comparison is therefore the
appropriate operand-boundary acceptance result, while both older runs remain
the trajectory references.

The downstream discriminator is positive. At iteration 3, identity-aligned
Pmax relative L2 improves from `3.6892644e-4` to `1.2986044e-4`, a `64.8%`
reduction, and maximum absolute Pmax error improves from `0.0104017` to
`0.00231342`. All 3,000 hard poses and translations remain within the strict
`0.01` degree/Angstrom identity gate. Support-count mismatches change from
five to six rather than improving monotonically; the remaining one-count
residuals are stacks `79`, `232`, `262`, `2110`, `2544`, and `2659`.

Merged iteration-3 signed FSC-AUC improves from `0.9999999931331159` to
`0.9999999990985996`, closing `86.9%` of the previous FSC deficit. The exact
size/order schedule remains unchanged. This establishes the coupled
fine-parent/Wavg schedule as a real K=1 trajectory cause, not just a stopped
arithmetic match. It is not yet a complete-case score increase: the frozen
scorecard remains `28/34` strict, `32/34` topology, and `34/34` evaluated.

The accepted run is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_order_wavg_triplet_it3_20260812T2125ET/`.
The particle audit SHA-256 is
`a4ddbcca26befbfaf178adbf02c59b62331015ac4373c7e8ffd2fcf917309b03`;
the signed FSC prefix audit SHA-256 is
`9051699cd3a9351b7bd9934ec52afb1b5e0df1cefc12d1328a5b286cb2189feb`.
The next stopped target is the earliest of the six iteration-3 support
boundaries, with numerator, denominator, normalized weight, cumulative mass,
and exact threshold margin captured before another complete trajectory.

## Direct Wavg residual and branch-consistency gate

The fused Wavg capture was extended with the third native atomic lane,
`diff2`. Corrected stopped job `12313872` shows that the direct residual uses
the raw translated preprocessed image rather than RECOVAR's CTF/noise-weighted
BPref numerator. On the exact-radius reconstruction pixels, atomic AA and XA
match native at approximately `1.4e-6` and `1.8e-6` pixel relative L2, while
their shell reductions match at approximately `4.3e-7` and `5.9e-7`.

Three-iteration diagnostic job `12314354` replaced complete shells 0--29 with
that direct fused residual, but only in rotation-chunked buckets. It improved
the iteration-2 low-shell raw-noise relative L2 to `6.71e-6` and `5.51e-6`
for halves 1 and 2 and reduced the iteration-3 support-count mismatch from six
particles to one. The downstream result was mixed: iteration-3 Pmax relative
L2 worsened from the accepted fused-XA/AA control's `1.29860e-4` to
`3.19634e-4`, merged signed FSC-AUC worsened from `0.999999999099` to
`0.999999996315`, and one hard pose changed by `3.692` degrees. The hard-pose
outlier is identity row 940; the sole support-count mismatch is identity row
1240, so they are distinct effects.

The implementation audit found the confound: naturally unchunked sparse-pass
buckets still used the algebraic XA/AA and noise residual. The direct run was
therefore an internally mixed arithmetic treatment, not a clean Wavg A/B.
It also left RECOVAR's per-particle normalization correction on the algebraic
`A2 - 2*XA + image power` path, whereas RELION adds the same direct Wavg
`wdiff2` pixels to both `thr_wsum_sigma2_noise` and
`exp_wsum_norm_correction`. This is a second coupled inconsistency capable of
moving the next iteration's normalized images even when the shellwise noise
operand improves.
Retry job `12316478` applies the same fused triplet to both bucket paths and is
capped at two iterations with an exact noise-update dump. Failed predecessor
`12316256` stopped before science because the centralized guard incorrectly
rejected iteration 1's intentional absence of scale groups; the retry scopes
the diagnostic to iterations where scale groups exist.

The exact deployed Wavg source and native `scale_aa_pixels.tsv` also establish
the complete pixel topology. At current size 60, the CUDA launch walks the
full `60 x 31 = 1860` FFTW rectangle. The host later consumes 1,462 rounded
support pixels: 1,411 exact-radius reference pixels plus 51 cutoff-rim pixels
whose native XA and AA are exactly zero but whose translated-image residual is
nonzero. A reconstructed FFTW gather matches all 1,860 native `j` positions
and every native shell label bitwise. After the unified bucket gate, the next
stopped treatment must reproduce this full launch topology; compactly adding
only the 51 rim pixels would not reproduce the native CUDA issue schedule. It
must use the resulting direct pixels for the shellwise noise numerator and the
per-particle normalization correction, then add the independently matched
high-resolution `powerClass` tail.

The fixed complete-case scorecard remains `28/34` strict, `32/34` topology,
and `34/34` evaluated. No threshold has changed, K=4 remains parked, and no
complete case-22 trajectory is justified before the unified iteration-2
operand result.

Unified chunked/unchunked job `12316478` completed successfully in 34 minutes
45 seconds. It removes the mixed-bucket confound, but remains restricted to
the 1,411 exact-radius pixels and therefore leaves cutoff shell 30 on the
algebraic path. The iteration-2 active-shell raw-noise relative L2 is
`5.76849e-6` in half 1 and `5.31592e-6` in half 2. This modestly improves the
partial half-1 result but does not close the boundary; its largest residual is
at the omitted cutoff-rim treatment. The accepted reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_direct_unified_it2_retry_20260813T0022ET/analysis/noise_update_terms_components_half1.json`
(SHA-256 `4c5b2594f0a8ff4050e4c50da34e165142efc2efab0b84bfd17c16ded6bb699c`)
and the corresponding half-2 report (SHA-256
`0d5e014041dff544ff0bec35e9ccfc2d74381a74e3dfff69a80dd061f86fd01d`).

The next stopped boundary is job `12317473`. It emits the complete 1,860-pixel
FFTW-ordered Wavg stream, consumes all 1,462 valid rounded-support pixels
through shell 30, retains the 1,411 exact BPref pixels for XA/AA, and uses the
same direct `diff2` pixels for per-particle normalization before adding the
independently matched high-shell `powerClass` tail. It also writes an
identity-aligned split of current-size and high-shell norm operands. Focused
unit tests cover the exact `1860/1462/1411/51/398` topology and all eight
noise/scale tests pass. No iteration-3 or complete trajectory is authorized
until this joint shell-noise/norm gate is analyzed.

## Full Wavg rectangle and iteration-2 first-divergence gate

Job `12317473` completed `0:0` in 39 minutes 22 seconds. The complete
rectangle closes the active current-size direct residual to relative L2
`1.48730e-6` and `2.31445e-6` in halves 1 and 2. The identity-aligned total
normalization operand has relative L2 `1.32125e-5`; the prior algebraic direct
normalization operand was `3.16660e-5`. The independently matched high-shell
tail remains at `3.27909e-7` relative L2.

The downstream iteration-2 state is close but not closed: Pmax RMSE is
`8.47805e-7`, 581 particles differ in significant-support count, and stack
image 1204 has the only hard pose difference. The merged signed FSC-AUC is
`0.9999999999255613`. This is a bounded two-iteration result, not a complete
case-22 acceptance result, so the fixed scorecard remains `28/34` strict,
`32/34` topology, and `34/34` evaluated.

The exact reports are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_rect_norm_it2_20260813T0042ET/analysis/`.
The full Wavg comparison is `wavg_norm_comparison.json`; the identity-aligned
particle report is `K1_PARTICLE_STATE_IT1_IT2_ROBUST.json`; and the signed FSC
report is `K1_FSC_IT1_IT2_ROBUST_PREFIX.json`.

The iteration-1 correction state feeding stack 1204's iteration-2 score was
then captured independently in job `12319723`. The EM computation and all
requested state artifacts completed in 81.8 seconds; the Slurm wrapper exited
1 only because its generic post-run check requested an iteration-2 noise dump
from a one-iteration run, while RELION intentionally skips the iteration-1
noise update. At the operational boundary, RECOVAR and serialized RELION both
produce float32 image/normalization factor `1.274451732635498` with zero ULP
difference. This rules out the incoming normalization scalar as the stack-1204
winner-flip cause. The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_rect_it1_state_retry2_20260813T0218ET/analysis/K1_CASE22_STACK1204_ITER1_NORM_STATE.json`
(SHA-256 `92a0b4c03acd8c6db79599a82ca23f3ca01e30599ffd6a142d9d27d64c82d390`).

Job `12318935` attempted the corresponding stop-after-target iteration-2
fine-table capture. It reached stack 1204 after traversing the full physical
half order, then failed before writing the target artifact because the dump
path requested one 17.05 GiB raw-score allocation. This is a capture-memory
failure rather than a scientific result. The replacement experiment replays
only stack 1204 from the lossless iteration-1 internal Fourier references
saved by job `12319723`, avoiding the full-half traversal and oversized
capture allocation.
Job `12319987` is a separate two-iteration composition gate that adds only the
already demonstrated CUDA coarse-Gaussian reduction to the complete Wavg
treatment; native `sincosf` remains disabled. Neither job authorizes a full
trajectory before its first-divergence report is complete.

## Coarse-boundary composition and lossless-noise replay

Job `12319987` completed `0:0` in 36 minutes 29 seconds. Its composition of
the complete Wavg treatment, CUDA Gaussian coarse scoring, and RELION-style
float32 coarse support is a strong positive iteration-2 result. Relative to
the complete-Wavg control, Pmax RMSE falls from `8.47805e-7` to
`2.37753e-7`, the Pmax maximum absolute error falls from `1.13457e-5` to
`4.08752e-6`, and significant-support-count mismatches fall from 581 to 118.
All remaining support discrepancies are one count. The only hard pose and
translation mismatch is removed, and merged signed cross-engine FSC-AUC
improves from `0.9999999999255613` to `0.9999999999675001`. The bounded
controller topology remains matched. This identifies the coarse score/support
boundary as the dominant mediator of the remaining iteration-2 posterior
drift; it does not yet establish a complete case-22 trajectory.

The accepted reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_rect_coarse_gaussian_it2_20260813T0210ET/analysis/K1_PARTICLE_STATE_IT1_IT2_ROBUST.json`
(SHA-256 `b74521fb093515f131ebc8b2a553e8024e2782017a119227bd551ca82d83b1bd`)
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_rect_coarse_gaussian_it2_20260813T0210ET/analysis/K1_FSC_IT1_IT2_ROBUST_PREFIX.json`
(SHA-256 `15831312c2ce93a1df4a04ac547995b47d597afe24fba378c95e294e7be946b4`).

The one-particle coarse-Gaussian/float32-support replay in job `12320501`
produced an artifact byte-identical to the map-and-normalization replay
control. That replay begins after its coarse parent/support set has already
been inherited, so it is a null intervention for the missing 64 fine tuples
and is not evidence against the positive fresh two-iteration composition.

Job `12320925` then replayed stack 1204 from the exact iteration-1 internal
Fourier references, exact float32 normalization factor
`1.274451732635498`, and lossless full-pixel iteration-1 noise arrays instead
of fixed-decimal STAR shells. It completed `0:0` in 2 minutes 28 seconds.
The centered pre-prior score relative L2 falls from `7.37616e-6` to
`1.27489e-6`, and its maximum absolute error falls from `1.54972e-4` to
`3.52859e-5`. Most importantly, RECOVAR now selects RELION's exact winner,
local fine rotation 145623 and translation 82. This causally confirms that
initial-noise serialization was sufficient to trigger the previous
stack-1204 winner flip when combined with the small reference residual.

The lossless-noise report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_live_noise_ref_it2_20260813T0247ET/analysis/K1_CASE22_STACK1204_FINE_SCORE_STAGES_LIVE_NOISE_REF.json`
(SHA-256 `6730b691e6c9880bb7c119381395b602aeecd3219303706af288f9cf5d0e3fff`).
Its pass-2 artifact SHA-256 is
`d32beaa106df1c4b07c79c54eb501cf7542a55f44eb51b52478d51c7cfef48b7`.
The first remaining discrete mismatch is still candidate-tuple presence: 64
native tuples from two coarse parent/translation groups are absent. Across
the common 2,349,728 tuples, the next largest staged discrepancy is the
orientation log prior (`2.22372e-5` relative L2), followed by centered
combined score (`1.42819e-5`). The next compact discriminator therefore
compares coarse tuple generation and direction-prior operands at iteration 2;
another complete trajectory is not the next experiment.

The fixed complete-case scorecard remains `28/34` strict, `32/34` topology,
and `34/34` evaluated until a complete case-22 run crosses its frozen gates.

## Lossless direction state and exact stack-1204 topology

The model STAR is also a lossy boundary for the learned direction prior. The
lossless iteration-1 half priors have 768 float32 entries and exactly unit
mass. Against those arrays, the serialized half-1 and half-2 STAR vectors have
relative L2 `7.23599e-5` and `7.31304e-5`. For stack 1204, the maximum staged
orientation-log-prior error, `0.00021028518676757812`, is exactly the maximum
error predicted by taking the logarithm of the serialized half-1 STAR vector.
The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_rect_it1_state_retry2_20260813T0218ET/analysis/K1_CASE22_DIRECTION_PRIOR_SERIALIZATION.json`
(SHA-256 `cbc0ddccb31d817898082b0efaac7bfa553ae247cdb4317d1d0793c425418723`).

Focused job `12321216` restored the lossless half priors together with the
lossless references, noise, and normalization scalar. It completed `0:0` in
2 minutes 28 seconds. All 64 formerly missing native fine tuples return, the
orientation prior becomes bit-exact, and the RELION winner remains exact.
RECOVAR still carries 32 additional tuples: eight oversampled rotations and
four translations from one extra coarse parent. The centered pre-prior score
relative L2 is `1.33802e-6`, and the first non-topological boundary is the raw
fine score. The version-2 report, which checks candidate-set equality in both
directions, is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_live_state_ref_it2_20260813T0259ET/analysis/K1_CASE22_STACK1204_FINE_SCORE_STAGES_LIVE_STATE_REF_V2.json`
(SHA-256 `ece91f665baf0247914066bc032b70403a099726a71b579de561014b29111b33`).

Exact-operand analysis job `12321365` then compared both top candidates. Their
native and RECOVAR raw `diff2`, orientation prior, translation prior, combined
pre-exponent, and pairwise winner margin are bit-exact. The projected
reference, translated image, correction, and high-resolution tail retain tiny
pixel-level differences, but they quantize to the exact production raw score
for both candidates. The first unequal top-candidate scalar is therefore the
normalized posterior, not the scorer. The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_live_state_exact_operand_20260813T0305ET/analysis/K1_CASE22_STACK1204_EXACT_OPERAND_PAIR.json`.

Job `12321624` added only RELION-style float32 coarse support composition to
that lossless-state replay. It completed `0:0` in 2 minutes 24 seconds and
closed every discrete boundary for stack 1204:

| Boundary | Native | RECOVAR | Result |
|---|---:|---:|---|
| Active fine tuples | 2,349,792 | 2,349,792 | exact |
| Significant fine tuples | 1,931,199 | 1,931,199 | exact set |
| Hard winner | `(145623, 82)` | `(145623, 82)` | exact |
| Missing or extra tuples | 0 | 0 | exact |

Its report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_live_state_f32support_ref_it2_20260813T0317ET/analysis/K1_CASE22_STACK1204_FINE_SCORE_STAGES_LIVE_STATE_F32SUPPORT_REF.json`
(SHA-256 `0b179e6bcc70c77a144445be8156d88ab5be63c75557e9c44c5f999d0a5956f7`).
The pass-2 artifact SHA-256 is
`68c79bb980cef5d53b927d5a63e29648b103c575f115f4244aac7066dba35ec8`.

The remaining posterior relative L2 is `6.75397e-6`, with exact topology and
support. A normalization counterfactual rules out softmax arithmetic as its
cause: native production probabilities versus a float64 mathematical softmax
of the native score field differ by only `3.85615e-8` relative, RECOVAR's
production probabilities versus the same mathematical operation differ by
`8.67414e-16`, and applying mathematical softmax to the two respective score
fields reproduces `6.75252e-6` relative error. The remaining continuous error
is therefore inherited from the aggregate raw fine-score field. The next
compact target is the shared tuple with the largest absolute posterior error,
native local rotation 69630 and translation 82; no complete trajectory is
needed to localize that operand.
