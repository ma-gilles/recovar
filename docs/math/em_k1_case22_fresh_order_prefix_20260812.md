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
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_live_state_f32support_ref_it2_20260813T0317ET/analysis/K1_CASE22_STACK1204_FINE_SCORE_STAGES_LIVE_STATE_F32SUPPORT_REF_V2.json`
(SHA-256 `73a0ee789c88217bb9c79dac58b0343880a5abac18f30a908d48cf2783f8bcf6`).
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

Native operand capture job `12321720` and GPU analysis job `12321911` close
that target. RECOVAR's raw `diff2` is exactly two float32 ULP high
(`130.71279907226562` versus `130.7127685546875`). Replacing only RECOVAR's
projected reference with the exact native `PPref` removes one ULP. Replacing
only its correction weights with the native weights also removes one ULP.
Replacing both makes the production raw score bit-exact. The correction
effect comes from shells 5 and above; replacing only shells 1--4 does not
change the RECOVAR production value. The translated image and high-resolution
tail do not change the quantized raw score.

The combined report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_impact_operand_combined_20260813T0342ET/analysis/K1_CASE22_STACK1204_EXACT_OPERAND_PAIR.json`
(SHA-256 `c5235164494c1fb02904b293c2cf1d0dd088dc54a3c6be3b39dedb7b35ab9463`).
Both unequal operands are iteration-start state derived from the preceding
map/noise update. This rejects a new fine-score formula, exponentiation, or
normalization defect at this boundary and returns the first-cause search to
the remaining Wavg/reference and noise-update residuals.

## Process-start RFLOAT noise boundary

The correction residual is now localized to a premature dtype conversion,
not to the CTF formula or noise-state values. The radial-state comparison in
`K1_CASE22_STACK1204_SCORING_NOISE_BOUNDARY.json` showed all 65
RECOVAR-derived native-unit inverse-noise shells equal to the live RELION
float32 words. That comparison was necessary but incomplete: it reconstructed
the reciprocal directly from the saved binary64 radial spectrum and therefore
bypassed RECOVAR's MPI process-start helper.

The direct pre-fix report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_live_operands_20260813T0410ET/analysis/K1_CASE22_STACK1204_LIVE_PASS2_NOISE_PREFIX.json`
(SHA-256 `b7a4d71ff2729d10c92d9281237b5e4ea33076e27189b4ef46cc1660814563bb`).

Focused telemetry job `12322645` recorded the operands actually delivered to
the fine-score correction operation. The source-precision binary64 CTF is
bit-exact, and replaying the recorded operands reproduces RECOVAR's stored
correction exactly. The live inverse-noise operand, however, differs by one
float32 ULP at 304 score pixels on shells 3, 7, 8, 14, 18, 24, and 27. The
process-start helper was unconditionally converting the full binary64 noise
image to float32 before the scorer took its reciprocal. RELION retains
`sigma2_noise` as RFLOAT and casts only the reciprocal to XFLOAT.

Preserving the incoming noise dtype fixes the demonstrated boundary. Focused
job `12322770` records zero correction-weight mismatches across all 1,461
supported pixels for stack 1204. Its raw `diff2` residual falls from two
float32 ULP to one: `130.71278381347656` versus native
`130.7127685546875`. With exact `PPref`, RECOVAR's remaining operands now
quantize to the native raw score, so the sole demonstrated scalar residual for
this tuple is the already localized incoming projected-reference state.

The fixed direct-noise report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_rfloat_noise_fix_20260813T0411ET/analysis/K1_CASE22_STACK1204_LIVE_PASS2_NOISE_FIXED.json`
(SHA-256 `23ffa43d8dacace928069bb8922e4d069a6b96f5ec03f72875dad1371779ded7`);
all 1,461 live score-window inverse-noise words are exact.

The fixed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack1204_rfloat_noise_fix_20260813T0411ET/analysis/K1_CASE22_STACK1204_RFLOAT_NOISE_FIX.json`
(SHA-256 `9caf772dd4ebeedd468ade63da9386cd304118d0f5e09de31ebc862644da30bc`).
The pre-fix and fixed pass-2 artifact SHA-256 values are respectively
`e1b17489b29428d58d71d2f8e036687a646045bea1ddf8809398c6da0cb3ec08`
and
`92a3cb48fbc8039595e491fca3bfd197436a5bff9e83f206fe437108cd28f8d3`.

Two-iteration composition job `12322851` is the causal full-dataset gate for
this change. Until that job is analyzed against the fixed native envelope,
the frozen complete-case scorecard remains `28/34` strict, `32/34` topology,
and `34/34` evaluated.

## Iteration-1 BPref repeat envelope

The remaining one-ULP projected-reference contribution is not currently a
cross-engine defect. Native first-iteration raw-accumulator repeat job
`12323251` completed the RELION iteration and wrote both BPref halves and both
numbered maps, but its wrapper exited after science because this capture
binary did not emit the optional downsampled text files required by the
launcher. The raw artifacts were therefore validated separately by their
headers, exact expected `(58, 115, 115)` half-grid shape, finite values, and
completed numbered output before comparison.

RELION's own original-versus-repeat downsampled-average relative L2 is
`1.97976e-7` in half 1 and `2.01006e-7` in half 2. By contrast, the current
RECOVAR iteration-1 accumulator is only `5.13104e-8` and `5.21303e-8` from
the new native repeat. Its numerator/denominator relative L2 values against
that repeat are `1.29312e-8`/`6.93128e-9` and
`1.37403e-8`/`6.58710e-9` for halves 1 and 2. RECOVAR already uses one
particle-owned BPref launch per particle in physical RELION order at this
boundary. The observed `4.28503e-8` projected-reference residual is therefore
inside native RELION's CUDA atomic repeat envelope; promoting the accumulator
or changing reconstruction arithmetic is not justified by this evidence.

The structured repeat report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_it1_repeat_20260813T0430ET/analysis/NATIVE_IT1_RAW_REPEAT.json`
(SHA-256 `d9adfef735ab076ddef968f6d7769c896312663b95309a1f3decbf382bd7972f`).
This closes the incoming-reference branch at native-repeat precision and
leaves the RFLOAT scoring-noise fix as the first demonstrated actionable
iteration-2 operand difference.

The pre-fix three-iteration propagation job `12322049` also completed before
that scorer correction. Its iteration-2 result landed on the less favorable
atomic branch seen in the earlier complete-Wavg control: Pmax RMSE
`8.48175e-7`, 579 support-count differences, and one hard pose/translation
difference. By iteration 3, one sensitive particle has Pmax error `0.12442`,
a `3.69209` degree pose difference, and a 12-count support difference; merged
signed FSC-AUC is still `0.999998722835`. This run is a pre-fix stability
baseline, not evidence against the RFLOAT correction. Its particle and FSC
reports are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_wavg_rect_coarse_gaussian_it3_20260813T0342ET/analysis/`.

## RFLOAT-noise two-iteration causal gate

Fixed full-dataset job `12322851` completed `0:0` in 35 minutes 51 seconds on
`della-h19g1`. It used the same complete-Wavg, CUDA coarse-Gaussian, and
float32 coarse-support composition as the preceding two-iteration control,
with only the process-start RFLOAT-noise correction added. Against the same
paired native reference, iteration-2 Pmax RMSE is `2.37473e-7`, its maximum
absolute error is `4.08752e-6`, and 128 particles differ in significant
support by exactly one candidate. All hard poses and translations remain
matched to output precision, current size and HEALPix order are exact, and
merged signed FSC-AUC is `0.999999999966664`.

The corresponding pre-fix composition values were Pmax RMSE `2.37753e-7`,
118 one-candidate support differences, and merged signed FSC-AUC
`0.999999999967500`. The full-distribution change is therefore neutral at
this boundary: the focused RFLOAT fix is operand-correct, but it does not
materially move the aggregate iteration-2 trajectory by itself.

This paired result is already inside the wider native CUDA-atomic envelope.
The paired native reference differs from two independently archived RELION
runs by Pmax RMSE `8.47412e-7` and `8.57384e-7`, with 596 and 604 support-count
differences and maximum support difference 5. RECOVAR versus the paired
native reference is smaller on every reported distribution bound. That does
not prove later-iteration parity: it means the remaining iteration-2
one-candidate support differences are not an actionable cross-engine defect
without a tighter same-operand discriminator.

The primary particle-state and signed-FSC reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_it2_20260813T0417ET/analysis/K1_PARTICLE_STATE_IT1_IT2_NATIVE_CONTROL.json`
(SHA-256 `68048ac42c439121922395d48ad327aef1e772869e4e89ea0c8bf7b5f828eba5`)
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_it2_20260813T0417ET/analysis/K1_FSC_IT1_IT2_NATIVE_CONTROL_PREFIX.json`
(SHA-256 `c6f502be63f134ab0217ab8df9eaa92b97320e7d37e0a19928bd117418f018b1`).
The native-envelope reports are in the same analysis directory. The frozen
complete-case scorecard remains `28/34` strict, `32/34` topology, and `34/34`
evaluated; a two-iteration prefix is not a case-22 closure.

## Full-rectangle/direct-residual composition is a negative K=1 A/B

Two fixed three-iteration jobs isolate the later full-rectangle treatment
from the earlier fused-XA/AA trajectory improvement. Job `12323552` used the
full power-class spectrum norm, translated-Wavg norm, and direct residual
without fine-parent order. At iteration 3 it had Pmax relative L2
`0.0035337733`, maximum absolute Pmax error `0.12444288`, four support-count
mismatches, one hard pose/translation mismatch, and merged signed FSC-AUC
`0.999998836850`.

Job `12323640` added exact fine-parent execution order to that same treatment.
It did not rescue the composition: iteration-3 Pmax relative L2 was
`0.0036552940`, maximum absolute error was `0.12439862`, seven support-count
mismatches remained, one hard pose/translation mismatch remained, and merged
signed FSC-AUC was `0.999998725866`.

The positive job `12310265` used a different and smaller composition: exact
fine-parent execution order, the fused RELION-order XA/AA Wavg atomic schedule,
and coarse Gaussian scoring. It explicitly did not enable the power-class
spectrum norm, translated-Wavg norm, or direct residual. Its iteration-3 Pmax
relative L2 was `0.0001298604`, its maximum absolute error was `0.002313416`,
all hard poses/translations matched, and merged signed FSC-AUC was
`0.999999999099`.

Therefore the full-rectangle/direct-residual treatment is rejected as a
candidate composition even though it improves isolated operand comparisons.
It remains diagnostic-only. The production candidate and external validation
prompt explicitly unset its three environment variables. This negative result
does not change the frozen complete-case scorecard.

## Native stack-232 fine BPref scan is exact in both RECOVAR scan paths

Native iteration-3 capture job `12323968` recorded all 1,664 active fine
candidates for stack 232 together with RELION's production significance
scalars. Its EM science completed; the outer wrapper alone exited nonzero
because its capture-inertness comparison used a stricter map-repeat threshold
than this diagnostic requires.

Replaying the exact captured float32 weights through both JAX/XLA and
RECOVAR's RELION-compatible CUB primitive reproduces RELION exactly:

- weight-sum bits: `1678890944` (`0x6411d7c0`);
- threshold bits: `1541859727` (`0x5be6e98f`);
- accepted hypotheses: `229` in native RELION, the production fine helper,
  XLA, and CUB.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack0232_native_it3_20260813T0523ET/native/analysis/K1_STACK0232_NATIVE_FINE_SIGNIFICANCE.json`
with SHA-256
`34485c75464910c24c9a458868cb06bce3771a0a2b8f1187f6a3c5b89b503f70`.
This count is not `_rlnNrOfSignificantSamples`. RELION writes that STAR field
from pass-1 coarse support (`52` here), while `229` is the number of pass-2
fine hypotheses admitted to BPref. Therefore this exact replay falsifies only
fine BPref sort/scan arithmetic when given identical fine weights. It does not
falsify the `53` versus `52` coarse-parent residual, which was already
localized to one rotation-dependent coarse raw-score boundary. Any new fine
capture addresses the separate Pmax and common-fine-tuple residual, not the
STAR significant-sample count.

## Positive composition with RFLOAT noise through iteration 3

Full three-iteration job `12324725` combines the previously positive narrow
composition with the demonstrated process-start RFLOAT scoring-noise fix. It
uses exact fine-parent execution order, the fused RELION-order XA/AA Wavg
atomic schedule, CUDA coarse-Gaussian scoring, and float32 coarse support. It
does not enable the rejected power-class spectrum norm, translated-Wavg norm,
or direct-residual paths.

The result improves the previous best iteration-3 trajectory. Pmax relative
L2 decreases from `0.0001298604424` to `0.0001119701484`, and maximum absolute
Pmax error decreases from `0.0023134160` to `0.0009019981`. The same six
particles retain one-count coarse-support residuals, while every hard pose and
translation remains matched and current size/HEALPix topology is exact. The
iteration-3 merged signed FSC-AUC increases from `0.9999999990985996` to
`0.9999999993275085`.

The primary reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_order_wavg_it3_20260813T0554ET/analysis/PARTICLE_STATE.json`
  (SHA-256 `77ada261dfabaee3a9c97fa9a2dbafc4a027ce9f8a15419f93679a2e8b32dc07`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_order_wavg_it3_20260813T0554ET/analysis/INTERMEDIATE_TRAJECTORY.json`
  (SHA-256 `20aa153e7031138a42154db9abf5b24d0b26fe5c915fd8ad7a54b2b5662af177`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_order_wavg_it3_20260813T0554ET/analysis/FSC_TRAJECTORY.json`
  (SHA-256 `ce424a445cd3b9d2ad29ab096ce355a0da1cc833886af44a766d423cbd577649`).

This confirms that the narrow candidate continues to close the trajectory
without a hard-state regression. It does not close case 22: the complete-case
scorecard remains `28/34` strict, `32/34` topology, and `34/34` evaluated. The
next discriminator is the exact production candidate table for stack 232,
captured without disabling projection caching or forcing its bucket
unchunked. That experiment tests whether the one extra coarse parent is the
sole source of the remaining fine-table and Pmax difference.

## Production-preserving stack-232 fine boundary

Job `12325540` emitted the requested stack-232 shard during the ordinary
cache-on, chunked iteration-3 production path. The comparison is complete
even though the rest of that full diagnostic run is still allowed to finish.
The production and native tables have exactly the same 248 rotations, 1,664
active fine tuples, and 229 fine BPref-significant tuples. There are no
native-only or RECOVAR-only tuples. This closes the formerly observed extra
fine-parent topology as an artifact of the earlier pre-RFLOAT/control path,
not a residual in the current positive composition.

The first unequal current boundary is the centered fine score before priors:
maximum absolute residual `0.0024957656860351562` and relative L2
`2.732477875007952e-5`. Orientation and translation priors differ only at
float32-rounding scale (`9.5367431640625e-7` maximum). Fine significant
support is exact. The selected native capture's Pmax is
`0.4817918629430238`, versus RECOVAR `0.48166962572426897`, but that native
capture is not a repeat-stable soft-posterior reference: the pinned robust
native and production RECOVAR values are both near `0.48168`. Therefore this
raw-score residual is recorded as a repeat-envelope diagnostic and does not
authorize a scorer change.

After removing the score convention's global offset
(`2353.2583045959473`), a per-rotation median removes `74.4436%` of the
pre-prior residual energy, while a translation-only median removes `10.2098%`.
The residual after rotation and translation medians has RMS
`0.0002211176118` and maximum absolute value `0.00128173828125`. This
localizes the repeat-envelope residual mainly to rotation-dependent scoring,
but does not make stack 232 a repeat-stable treatment target.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack0232_production_capture_it3_20260813T063215ET/analysis/K1_STACK232_PRODUCTION_FINE_BOUNDARY.json`
(SHA-256
`65e2fdebe5e8e0207798b79830b7f0631ada829d097e8baa6f31bcd7754e0d59`).

## Iteration-3 continuation is not an exact live-state oracle

Job `12325969` showed that an ordinary continuation from the serialized
iteration-2 optimiser is invalid: process-local order and sampling RNG state
are reset, changing all 3,000 hard poses. Job `12326034` restored the exact
fresh order (`1723`) and uninterrupted perturbation
`-0.293744921684`. This recovers stack 262's hard pose and shift and improves
half-map FSC-AUC to `0.9997891214`/`0.9995387243`, but its Pmax/support remain
`0.115130`/`32`, versus uninterrupted robust RELION `0.122029`/`18`.
Serialized continuation therefore remains a controlled serialization arm,
not the native operand authority.

The uninterrupted fresh capture instead gives stack-262 Pmax/support
`0.121111`/`19`, nearly matching RECOVAR `0.1211270019`/`19`. Its apparent
`0.0009019981` error against the pinned robust run is mainly native repeat
drift, so stack 262 is removed from the actionable target set. Stacks 79,
469, 2498, 2544, and 2659 retain the same one-count RECOVAR support excess
against both independent native runs. Multi-particle native job `12326152`
received only stack 79 because commas in the `sbatch --export` value were
interpreted as environment-variable separators. Its stack-79 state is locally
admissible: the hard pose, shift, and native support count 47 match both
reference runs, while Pmax is `0.358746` versus `0.358734` and `0.358731`.
Corrected native job `12326366` passes the five targets with colon separators;
it completed `0:0` in 8 minutes 39 seconds and wrote all ten expected fine-score
and geometry sidecars. All five target hard poses, shifts, and native coarse
support counts match both reference runs. The target Pmax values remain inside
the native repeat envelope. Its outer audit exits 1 only because the selected
control directory has no iteration-3 STAR, so target-local admissibility is
used explicitly rather than claiming global capture inertness.

The original production job `12326241` was likewise verified from its live
environment to contain only stack 79. Corrected production job `12326426`
uses colon-separated stack IDs and exposes original indices
`78,468,2497,2543,2658` inside the running process. Both original jobs are
allowed to finish naturally. `scripts/analyze_k1_partial_fine_panel.py` now
turns the corrected artifacts into fixed-denominator exactness counts for
rotation topology, active tuples, pre-prior score, both priors, posterior, and
fine support, and records each particle's first unequal boundary.

## Repeat-stable stack-79 fine boundary

The original single-target production job `12326241` completed normally and
wrote stack 79 from the exact cache-on, chunked positive composition. Native
RELION has 248 fine rotations and 1,504 active tuples. RECOVAR has those exact
rotations and tuples plus the eight children and 32 tuples of its one extra
coarse parent. None of the 32 extra tuples survives fine significance: native
and RECOVAR both retain the same 251 fine/BPref tuples with zero identity
mismatches. For stack 79, the one-count coarse STAR residual therefore changes
evaluated work but not the M-step operand set.

On the 1,504 common tuples the first numerical boundary is the centered fine
pre-prior score, with maximum absolute residual `0.0026378631591796875` and
common-domain posterior total variation `7.009615608681395e-5`. The native
Pmax is `0.3587464629` and the production RECOVAR Pmax is `0.3586772378`.
Rotation medians remove `66.4028%` of the global-offset-removed residual
energy; translation medians remove only `5.6087%`. The production winner is
RECOVAR local rotation 239, global fine rotation 262871, parent 32858, and
translation 62, matching native local rotation 231. This is now the bounded
pixel/operand target.

The full 1,069,056-candidate native coarse score table was also replayed
through RECOVAR's float32 support selector. It reproduced all 47 native tuple
identities and the exact threshold bits. The CPU replay total differed by one
float32 ULP, without moving the cutoff; the production GPU route uses the same
CUB sort/scan primitive as RELION. This rules out the cutoff rule itself for
stack 79 and places any coarse support difference upstream in score/prior
operands.

The fine-panel report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_repeatstable5_production_capture_it3_20260813T071338ET/analysis/K1_STACK79_PRODUCTION_FINE_PANEL.json`
(SHA-256
`a3ead6be90c74ceee7d890f7c6f924e9f32943c66bd2f21480d44634feb82faa`).
Native tuple job `12327217` and matching stopped RECOVAR operand job
`12327237` capture exactly the winning tuple in iteration 3. Their staged
comparison proceeds through projected reference, shifted image, correction
weight, high-resolution sum, pixel contributions, 256 lane partials, and raw
`diff2`; no complete trajectory is needed for this discriminator.

## Fixed five-particle production fine panel

The corrected production capture emitted all five repeat-stable targets from
the same ordinary cache-on, chunked positive composition. The fixed panel
shows fine-rotation topology exact for `2/5`, active fine-tuple topology exact
for `0/5`, and final fine/BPref support exact for `2/5`:

| stack | native/RECOVAR fine rotations | native/RECOVAR active tuples | native/RECOVAR fine support | support identity |
|---:|---:|---:|---:|---|
| 79 | 248 / 256 | 1,504 / 1,536 | 251 / 251 | exact |
| 469 | 584 / 592 | 3,392 / 3,424 | 169 / 171 | RECOVAR +2 |
| 2498 | 176 / 184 | 864 / 896 | 102 / 102 | exact |
| 2544 | 6,096 / 6,096 | 40,160 / 40,192 | 8,045 / 8,050 | RECOVAR +5 |
| 2659 | 168 / 168 | 928 / 960 | 259 / 263 | RECOVAR +4 |

There are no native-only rotations, active tuples, or fine-support tuples in
the panel. The coarse support excess therefore produces a RECOVAR superset.
For stacks 79 and 2498 its 32 extra children die before BPref. For stacks 469,
2544, and 2659, two to five extra tuples cross the fine reconstruction cutoff
and change the actual M-step operand set. Stacks 2544 and 2659 demonstrate the
important case where the extra coarse tuple reuses a rotation whose other
translation was already selected: rotation topology stays exact while active
tuple topology changes.

The probability-aware support audit separates direct and indirect
propagation. In stack 469, one of the two extra BPref tuples is an extra-parent
child with posterior `4.23666e-5`; the other is a common active tuple at the
shifted cutoff. In stack 2659, two of four extras are extra-parent children
(`2.55498e-5` and `1.81264e-5`) and two are common active tuples. In stack
2544, all five extras are common active tuples: none of the 32 extra-parent
children survives directly, but their probability mass changes the cumulative
fine cutoff and admits five common tuples. This establishes the discrete
causal chain `+1 coarse tuple -> +32 active fine tuples -> changed fine mass
and cutoff -> +2/+5/+4 BPref tuples` for the three affected particles.

The common-domain fine-score residual is small but systematic across all five:
maximum centered pre-prior residual ranges from `0.00255585` to `0.00270081`,
and common-domain posterior TV ranges from `3.26608e-5` to `1.14760e-4`.
Rotation medians remove between `54.20%` and `80.76%` of residual energy. This
panel proves that the residual is not only a serialized coarse-count field,
but it does not yet identify a source-faithful formula change. The bounded
stack-79 pixel/operand comparison and the full coarse raw/prior capture are
the next gates.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_repeatstable5_production_retry_it3_20260813T0731ET/analysis/K1_REPEAT_STABLE5_PRODUCTION_FINE_PANEL.json`
(SHA-256
`c1341d318ca727c9910b34371e11f7db35821bc286aad68386f4afeef9f840f6`).
This fixed particle-panel denominator is separate from, and does not change,
the complete-case scorecard of `28/34` strict, `32/34` topology, and `34/34`
evaluated.

## Stack-79 coarse first-divergence boundary

The complete 1,069,056-candidate native/RECOVAR coarse join is now available.
Candidate topology, finite-prior support, and the hard best tuple are exact.
Native retains 47 coarse tuples while RECOVAR retains 48; the sole mismatch is
RECOVAR rotation 33997 / translation 15, corresponding to native rotation
9884 / translation 15. Its native descending rank is 47 (zero-based), exactly
one below the 47-tuple retained set.

The mismatch is already present before posterior normalization. Relative to
the shared best, the tuple's raw score is `0.000244140625` higher in RECOVAR;
its orientation prior differs by `9.53674e-7` and translation prior by
`2.38419e-7`. Across all 631,968 common valid tuples, centered raw-score
median/p95/max absolute residuals are `0.000732422`, `0.00219727`, and
`0.00756836`; removing a per-rotation median removes `52.9831%` of residual
energy, while translation medians remove only `2.1981%`. The native cutoff
tuple itself has exactly matched relative raw and combined scores, so the
extra count arises when the distributed rotation-dependent raw-score residual
changes the lower-tail mass boundary, not from a different comparison or tie
rule.

The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_repeatstable5_coarse_production_it3_20260813T0741ET/analysis/K1_STACK79_NATIVE_COARSE_BOUNDARY.json`
(SHA-256
`4b262d980353fc5bc7bd3149cedea1ea9979a7f5da080e2dd9f3adc36177bd7a`).
This establishes raw coarse scoring as the first unequal computed variable for
stack 79. The next bounded capture requests only the mismatching rotation and
the shared-best rotation, then compares reference norm, cross term, projected
reference, shifted image, correction weight, and native-reduction replay.

## Exact fine-posterior arithmetic: local closure, trajectory falsification

Commit `503092461d132a913af381a198ab77fca554b3ca` adds an opt-in
`RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1` route that reproduces the native
CUDA fine-posterior boundary on the fixed stack-117 table. Candidate count
`730,976`, raw `diff2`, raw weights, CUB scan-sum bits `1701728133`, cutoff
bits `1474915410`, significant count `493,009`, hard winner, and float32
division all close exactly. The route remains diagnostic and is not the
default.

The three-iteration case-22 experiment `12359936` does not convert that local
closure into a better trajectory. Iteration-3 merged signed FSC-AUC changes
from `0.999999997337489` in the prior arm to `0.999999992973701` with exact
posterior arithmetic. Iteration-2 Pmax RMSE improves slightly from
`8.5775e-7` to `8.4822e-7`, while the aggregate map result is neutral to
slightly worse. The apparent stack-941 hard-pose regression is a native
near-tie: the sealed native run chooses one pose, while two independent
instrumented native runs choose the other. On the matching instrumented
capture, the exact route reduces stack-941 posterior relative L2 from
`4.882857e-4` to `1.220748e-4` and selects the same native winner.

The primary trajectory artifacts are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_exact_fine_posterior_it3_50309246_20260813T2321ET/analysis/`.

## Matching-repeat BPref correction and physical-iteration-2 gate

The earlier physical-iteration-2 accumulator report mixed two native repeats.
Against the matching native repeat, the old RECOVAR half-1 shell-16--30
relative L2 values are `3.1737151e-6` for the numerator, `4.3954929e-6` for
the denominator, and `9.3618347e-6` for the reconstructed average. The
corrected three-edge report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_half1_native_exec_order_20260810T1745ET/analysis/K1_CASE22_IT2_HALF1_NATIVE_EXEC_ORDER_CORRECTED_EDGES.json`
(SHA-256
`6d7287dadecd7431f9dcbdc2a0817c23b47d084a07179159b8ac0ce2681ced94`).
The accumulator analyzer now records RECOVAR against both native captures so
this provenance error cannot recur silently.

Job `12362186` repeated that same-input physical-iteration-2 boundary with
exact posterior arithmetic, native particle execution order, and one
particle-owned accumulator launch per particle. It is neutral to slightly
worse: numerator `3.2196618e-6`, denominator `4.3990191e-6`, and average
`9.3779787e-6`. More decisively, particle support-count mismatches increase
from `47/1490` to `276/1490`, all by one count, while Pmax relative L2
increases from `5.561264e-6` to `6.472475e-6`. The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_sameinput_exact_posterior_native_order_20260814T0200ET/analysis/K1_CASE22_IT2_SAMEINPUT_PARTICLE_STATE.json`
(SHA-256
`3445ef46fea2e894bdc2ed887af6c7dfd2d1ab051e96236a81677886c7d9b5a2`).

This falsifies posterior arithmetic as the remaining aggregate iteration-2
root. Exact arithmetic closes when the local candidate table is fixed, but
the normal production path supplies a different candidate sequence or set for
many particles. The next gate is therefore the normal, non-oracle production
candidate table itself. Submission `12362830` failed before science work
because the submit-side root variable name was wrong. Focused H100 retry
`12362864` captures stack 2690 at physical iteration 2 and stops at that
boundary; its run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_normal_candidates_it2_20260814T0345ET_retry1/`.

The complete-case scorecard remains `28/34` strict, `32/34` topology, and
`34/34` evaluated.

## Guarded fresh-K=1 default checkpoint (2026-08-14)

The direct RFLOAT scoring-noise update is now active by default only in the
guarded fresh-K=1 physical-order route.  Case-26 job `12371765` supplied the
causal full-run result: final signed cross-engine FSC-AUC improved from
`0.954914` to `0.997747`, with exact numbered topology.  Passing-case-25 job
`12372874` preserved its result within `2.69e-7` FSC-AUC.  Job `12373695`
then verified direct activation with both launcher diagnostic flags set to
zero.  These results promote case 26 and move the fixed complete-case
scorecard to `29/34` strict and `32/34` topology.

The next bounded case-22 capture tests the earliest remaining ordinary
Gaussian coarse boundary, not another full trajectory.  Job `12374036`
captured the frozen 14-particle mismatch/control panel at physical iteration
2 with the native CUDA Gaussian reduction and float32 support calculation.
All `14/14` parent sets, winners, and prior-support sets agree with RELION;
all posterior total-variation values pass the fixed `1e-4` gate.  Individual
candidate masks are exact for `12/14`.  Stack indices 2322 and 2994 retain
one fewer translation candidate each, with zero parent-set mismatch.  The
historical 13 parent-side discrepancies have therefore closed; the analyzer's
`expected_side_reproduced=0/13` wording records that none of those old
RELION-only/RECOVAR-only sides still exists.  It is not evidence that the
current parent sets disagree.  The accepted report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse14_gaussian_db76b99e_20260814T0745ET_retry1/analysis/K1_IT2_COARSE14_GAUSSIAN_BOUNDARY.json`
(SHA-256
`f2a84d4f5ccd5a80851365930958365826b75226db37ded9ace1d7e782087fdf`).

A deliberately zero-Gaussian case-26 prefix, job `12373948`, also completes
with exact topology through iteration 3 and merged cross-engine FSC-AUC
`0.9999999999288698` at iteration 3.  This shows that Gaussian coarse is not
needed to rescue the already-closed case-26 prefix, while the case-22 raw
capture shows that it is the component that removes the first known
candidate-parent discrepancy.  The implementation consequently defaults the
Gaussian FFI and float32 coarse-support calculation only when the existing
fresh-K=1 physical-order guard is active.  Explicit environment values still
override that default; continuation, frozen-boundary replay, perturbation
replay, and K>1 remain unchanged.

## Exact iteration-3 fine-score operand split and direct-noise gate

Focused positive-control job `12368430` completed naturally with exit code
zero and captured one immutable stack-79 tuple at physical iteration 3. Native
production and the native SASS-tree replay are bit-exact at
`2186.461181640625`; RECOVAR production and its replay are bit-exact at
`2186.413330078125`. The complete four-operand factorial localizes the gap:

| Substitution into RECOVAR operands | Raw `diff2` |
| --- | ---: |
| none | `2186.413330078125` |
| native correction weight | `2186.453369140625` |
| native shifted image | `2186.421630859375` |
| native projected reference | `2186.412841796875` |
| all native operands | `2186.461181640625` |

The correction substitution closes about `83.7%` of the raw gap. The
score-active correction, shifted-image, and reference relative-L2 values are
`1.6828434e-5`, `1.9396349e-6`, and `2.0788233e-5`. The shifted-image error is
almost a pure preprocessing scalar: fitting real scale
`1.000001939344767` reduces its residual to `3.3763e-8`. Translation
arithmetic is therefore not the material image boundary. The accepted report
is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack0079_positive_raw_operand_it3_20260814T040817ET/analysis/K1_STACK79_FINE_OPERAND_POSITIVE.json`
(SHA-256
`058953dfbbfaa7e6b5961088e0525b381e3d187d05aab1f34cf2b41003579d88`).

Direct-noise-only production job `12368908` also completed naturally with exit
code zero, but the pre-recorded immutable-tuple gate rejects a row-wise operand
comparison. The intervention changes the iteration-3 candidate table: compact
row 239 is global rotation `262871` in the positive control and `269047` in
the direct-noise arm, while candidate rotation count changes from `256` to
`248`. The two row-239 raw scores are consequently different tuples and must
not be compared. This is captured in the fail-closed report
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack0079_direct_noise_raw_operand_it3_20260814T0426ET/analysis/K1_STACK79_DIRECT_NOISE_IDENTITY_GATE.json`
(SHA-256
`b1c4c13e75dfd3b9623d83f8ee7508423b3c636fc39182aba1041faf40c1bf9c`).
The treatment does leave the stored float32 preprocessing normalization factor
unchanged at `1.2500802278518677`, so it isolates shell noise rather than
silently applying the rejected coupled normalization treatment.

The case-level causal gate is the same-physical-H100 case-26 control/treatment
pair. Its first wrapper submission `12369782` failed before either arm started
because it pinned an incorrect empty-diff hash. That audit also found that the
tracked launcher exported and then unset the Gaussian-coarse override. Commit
`059beba9` fixes the ordering and adds a static regression. Corrected job
`12370338` runs control and direct-noise-only sequentially inside one H100
allocation. No fixed score is changed until that case-level gate completes;
the scorecard remains `28/34` strict, `32/34` topology, and `34/34` evaluated.

Job `12370338` completed naturally with exit code zero in 33 minutes 31
seconds. Both arms used H100 UUID
`GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a` and preserve the exact
iteration-1--3 size/HEALPix topology (`56, 56, 66` and `3, 3, 3`). The
direct-noise-only arm passes every predeclared causal gate:

| Metric | Control | Direct noise only | Change |
| --- | ---: | ---: | ---: |
| iteration-2 half-1 shell 0--28 noise rel-L2 | `1.3556491e-5` | `1.2426626e-7` | `109.1x` closer |
| iteration-2 half-2 shell 0--28 noise rel-L2 | `9.1033654e-6` | `2.4863711e-7` | `36.6x` closer |
| iteration-3 Pmax rel-L2 | `4.1614107e-5` | `3.8456602e-5` | `7.59%` lower |
| iteration-3 support-count mismatches | `10/1000` | `4/1000` | `60%` fewer |
| iteration-3 merged signed FSC-AUC | `0.999999999971256` | `0.9999999999758163` | `+4.56e-12` |

The iteration-3 maximum hard-pose and translation errors remain unchanged at
`1.3511150e-5` degrees and `4.4172941e-6` Angstrom. Thus source-faithful
direct low-shell residual accumulation is causal for the early case-26 latent
state rather than merely changing a downstream final map. The accepted report
is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_direct_noise_pair_it3_ddfe8dbd_20260814T0503ET/analysis/K1_CASE26_DIRECT_NOISE_PAIR_GATE.json`
(SHA-256
`58ae699e1f3e8f0ab81e3d9688ce6210ed7a7a30dd1fc94e0b6eb8fd6112e071`).

This result authorizes a complete case-26 treatment run, but does not yet
promote the diagnostic to default behavior or change a frozen score. The
complete-case scorecard therefore remains `28/34` strict, `32/34` topology,
and `34/34` evaluated.

## Complete case-26 direct-noise gate

Pinned full-trajectory job `12371765` completed naturally with exit code zero
in 31 minutes 29 seconds on the same H100 UUID used by the accepted prefix
factorial.  It ran from commit `36dac0171859b09bb57358ff0a86ed8a03615975`
with an empty tracked diff, `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset, and
the forced-after-maximum final override unset.  All eleven numbered
iterations preserve RELION's exact current-size and HEALPix schedule:
`56, 56, 66, 60, 60, 60, 60, 60, 60, 60, 60` and
`3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4`.

The authentic final all-data pass clears both fixed scientific gates:

| Metric | Frozen case 26 | Direct-noise candidate | Fixed gate |
| --- | ---: | ---: | ---: |
| final merged cross-engine signed FSC-AUC | `0.954914` | `0.997747377` | `>= 0.995` |
| final merged RECOVAR-minus-RELION GT FSC-AUC | not promoted | `+0.008697039` | `>= -0.002` |

The final half-map cross-engine FSC-AUC values are `0.999991562` and
`0.999995803`.  Every numbered merged FSC-AUC rounds to `1.0`; iteration-11
Pmax relative L2 is `3.328973e-4`.  An independent immutable-particle audit
also passes the exact schedule gate.  Its final pose residuals are limited to
STAR precision: angular p95 `9.0934264e-6` degrees and translation p95
`1.3020949e-6` Angstrom.  Final Pmax absolute p95 is `2.7954081e-4`.

The accepted artifacts are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_direct_noise_full_36dac017_20260814T0610ET/analysis/case26_ordered_exact_full_fsc.json`
  (SHA-256
  `587913505160261847b7fcc7db01fc03f50ddafc9ccc3de23d4fbf3f5448416c`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_direct_noise_full_36dac017_20260814T0610ET/analysis/case26_ordered_exact_full_state.json`
  (SHA-256
  `846b1d6fc0870f4640308bb25f662d891b6849685b6408113398fc757629cf25`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_direct_noise_full_36dac017_20260814T0610ET/analysis/case26_ordered_exact_full_particle_state.json`
  (SHA-256
  `88e3bad8d36a4909e9dc509ae60a5163160906379e530755149f686485a5f02e`).

This closes case 26 scientifically for the direct-noise candidate and would
move the fixed denominator to `29/34` strict if a passing-case preservation
run confirms that the opt-in intervention can be promoted safely.  Until
that preservation gate completes, the repository default and published fixed
scorecard remain deliberately unchanged at `28/34` strict, `32/34` topology,
and `34/34` evaluated.

## Passing-case preservation and guarded promotion

Case-25 preservation job `12372874` completed all scientific work on the same
H100 UUID as the case-26 gate.  It preserved the exact eight-iteration
current-size schedule `56, 56, 52, 50, 50, 50, 52, 52` and HEALPix schedule
`3, 3, 3, 3, 3, 4, 4, 4`, then ran the authentic final all-data pass with
both final overrides unset.  The final result is repeat-level stable:

| Metric | Frozen case 25 | Direct-noise candidate | Change |
| --- | ---: | ---: | ---: |
| final merged cross-engine signed FSC-AUC | `0.998192576` | `0.998192307` | `-2.69e-7` |
| final half-1 cross-engine signed FSC-AUC | `0.999992125` | `0.999992129` | `+3.93e-9` |
| final half-2 cross-engine signed FSC-AUC | `0.999997340` | `0.999997341` | `+1.25e-9` |
| final merged RECOVAR-minus-RELION GT FSC-AUC | not gated historically | `+0.009182130` | passes `>= -0.002` |

Every numbered merged FSC-AUC is at least `0.999999999914046`.  The accepted
FSC and state reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_direct_noise_preservation_7f0e2348_20260814T0645ET/analysis/case25_direct_noise_preservation_full_fsc.json`
(SHA-256
`eae4cd9ab938c6c0761013402ec29bd635014311f10bbeb966e46dc50dba658a`)
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_direct_noise_preservation_7f0e2348_20260814T0645ET/analysis/case25_direct_noise_preservation_full_state.json`
(SHA-256
`7fb31a48ab98e17056857479018ae35ea00efb6634a2098b2508da25f093af96`).

Slurm records job `12372874` as exit `1:0` only because the generalized
launcher still asserted case 26's literal effective order seed `1727` after
case 25 correctly used `1726`.  The refinement, final pass, and artifacts had
already completed; the two manual auditors both return pass.  The launcher
now derives this assertion as `run_seed + 1`.

The direct-noise-only path is therefore promoted as the default only when
both existing guards are active: fresh K=1 RELION physical particle order and
exact RELION BPref operands.  It remains dormant for iteration 1 before scale
groups exist.  Explicit environment value `0` disables it, and continuation,
frozen-boundary replay, perturbation replay, and K>1 remain unchanged because
they do not set the fresh K=1 order guard.  This evidence-backed promotion
moves the fixed K=1 scorecard to `29/34` strict, `32/34` topology, and `34/34`
evaluated.  Remaining strict failures are cases 4, 5, 7, 10, and 22; cases 7
and 22 remain the two topology failures.

## Uninterrupted iteration-3 BPref boundary

The first iteration-3 native accumulator attempt was rejected before causal
classification.  Jobs `12366689` and `12366711` continued from a serialized
`run_it002_optimiser.star`, so their process-local particle order and sampling
RNG state were reset.  The existing continuation audit already demonstrates
that this changes all 3,000 hard poses.  Those dumps are valid continuation
controls, but they are not an oracle for RELION's uninterrupted physical
iteration 3.  The same limitation applies to reconstruction-stage job
`12366854`.

Fresh uninterrupted native jobs `12367256` and `12367339` instead ran physical
iterations 1--3 from the original inputs and captured all three raw BPref
calls.  Physical iteration 3 is explicitly `call0002`.  Independent
reconstruction-stage job `12367257` also ran fresh and captured only call 2.
Its instrumentation is inert against `12367256`: all three numbered
iterations retain exact selected topology, every half-map FSC-AUC is at least
`0.9999999999030741`, and no pose or shift differs.  The inertness report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_fresh_it3_reconstruct_stage_20260814T0332ET/analysis/RELION_FRESH_CAPTURE_INERTNESS.json`
(SHA-256
`f38c99494dcd2a09895e5da800ee3bd97f9b1502bc0b6d9e19b5145f5e2f4595`).

The positive narrow candidate from job `12324725` had already saved its
post-low-resolution-join iteration-3 accumulators losslessly.  They were
wrapped in the stage-identified archive
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_order_wavg_it3_20260813T0554ET/analysis/recovar_bpref_accum_it003_from_saved_intermediates.npz`
(SHA-256
`6705dab4ef34c51f2bcd6a23fcf481821d6c799d7220e26e74f810fbadbaaa85`)
and compared to both uninterrupted native call-2 captures:

| Half | Quantity | RECOVAR/native A rel-L2 | RECOVAR/native B rel-L2 | Native A/B rel-L2 |
| ---: | --- | ---: | ---: | ---: |
| 1 | numerator | `9.7378619e-3` | `9.7376086e-3` | `1.7747644e-5` |
| 1 | denominator | `1.0991199e-3` | `1.0991362e-3` | `3.1480538e-6` |
| 1 | downsampled average | `7.2921677e-3` | `7.2917311e-3` | `3.1927836e-5` |
| 2 | numerator | `3.9458379e-4` | `3.9525465e-4` | `2.3472742e-5` |
| 2 | denominator | `3.0620719e-5` | `3.1605459e-5` | `2.9545365e-6` |
| 2 | downsampled average | `6.5895098e-4` | `6.6124531e-4` | `5.9721500e-5` |

Current-head boundary job `12366688` completed independently with the same
fresh physical order and explicit post-join archive.  Relative to native A,
its half-1 numerator/denominator/average errors are
`9.7439534e-3`/`1.1063982e-3`/`7.3087027e-3`; half 2 is
`4.0332047e-4`/`5.0522523e-5`/`6.6643738e-4`.  The positive narrow candidate
therefore improves every reported edge, most visibly the half-2 denominator,
but leaves the dominant half-1 boundary essentially unchanged.  The baseline
reports are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it3_postjoin_boundary_20260814T0310ET/analysis/K1_CASE22_BASELINE_FRESH_IT3_POSTJOIN_RAW_ACCUM_HALF1_V5.json`
(SHA-256
`a75e3560eab55c631c00eb63e6661168ee2e89790dbf02f2a31654d8c121a3e2`)
and the corresponding half-2 report (SHA-256
`226f5af7728f4fd77ae49f307a3870f389f25b8f5d3c18c3dfcd3e4814fa5102`).

The accepted three-edge reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_order_wavg_it3_20260813T0554ET/analysis/K1_CASE22_POSITIVE_FRESH_IT3_POSTJOIN_RAW_ACCUM_HALF1_V5.json`
  (SHA-256
  `ee5bd29467a10d874d8d8b8f0bfbb7f83a10c991d73e1cb6bb49ce84ed7c7a6b`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_rfloat_noise_order_wavg_it3_20260813T0554ET/analysis/K1_CASE22_POSITIVE_FRESH_IT3_POSTJOIN_RAW_ACCUM_HALF2_V5.json`
  (SHA-256
  `c8e951eb071168c2c27d7c2ddd1be8d9c9fdd8ec5441a847f267bbc912eca2da`).

This is the first repeat-stable material iteration-3 boundary.  It is already
present in post-join BPref numerators and denominators, is strongly dominated
by half 1, and is hundreds of native-repeat units in the half-1 numerator.
Tau2 regularization, Wiener division, inverse FFT, and gridding are downstream
and are not the next fix target.

The next bounded discriminator stays at physical iteration 3 and half 1.  It
must join the ordinary production path to fresh native RELION by immutable
particle and class-pose tuple identity, then compare normalized posterior
mass, significant BPref tuple membership, per-tuple numerator and denominator
operands, destination indices, and particle-owned partial sums.  Start with
the largest identity-aligned Pmax residual and the previously repeat-stable
five-particle panel, but retain full-half hashes and mass totals so a sparse
panel cannot hide a distributed scale error.

The fixed complete-case scorecard remains deliberately unchanged at `28/34`
strict, `32/34` topology, and `34/34` evaluated.  This focused boundary
localizes the remaining trajectory input error; it does not yet add a passing
complete case.

## Physical-iteration-2 candidate order and the first missing parent

Focused H100 retry `12362864` completed in 3 minutes 16 seconds and captured
the ordinary, non-oracle stack-2690 fine table at physical iteration 2.  The
native and RECOVAR tables have the same `115,224` fine rotations.  Native has
`913,792` active tuples and RECOVAR has `913,760`: the only set difference is
one native coarse parent and its 32 fine children.  On the `913,760` common
tuples, the ordinary RECOVAR traversal order is not RELION's order; only 128
positions agree after removing the missing block.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_normal_candidates_it2_20260814T0345ET_retry1/analysis/K1_STACK2690_NORMAL_CANDIDATE_TOPOLOGY.json`
(SHA-256
`86af51227b1529e6a98466162352fb6e7f6a2405dd6e14de6f19261eac33ea64`).

Job `12363117` repeated the same table with the existing RELION parent-order
diagnostic.  After restricting both tables to their common tuples, all
`913,760` positions are exactly ordered and both byte sequences have SHA-256
`43bb8d91e14dcb044d139e4252fd90869bb67d53defe84d73f39e833d737e12d`.
The full sequences first differ only where the native-only 32-tuple block is
inserted.  This proves that `RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER=1`
reproduces the required fine-candidate traversal rather than merely applying
a different arbitrary permutation.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_relion_candidate_order_it2_20260814T0110ET/analysis/K1_STACK2690_RELION_CANDIDATE_ORDER_V2.json`
(SHA-256
`0afdc2330ea06066e548841f4c8c81094da0c820941aacdcef2027fbebc3ac56`).

The missing block comes from native coarse rotation 25632 / translation 13,
which maps to RECOVAR rotation 534 / translation 13.  The complete coarse
join has exact candidate topology, finite-prior support, orientation prior,
and shared hard winner.  Native retains 28,556 tuples while RECOVAR retains
28,555; this target is the only support mismatch.  Its raw score relative to
the shared best is lower in RECOVAR by `2.384185791015625e-5`, before
posterior normalization or the cumulative cutoff.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_native_coarse_it2_20260814T0100ET/analysis/K1_STACK2690_NATIVE_COARSE_BOUNDARY.json`
(SHA-256
`a300cdc4b9680c2a56aff4163b893e583845ebc186ad360e49ce2f18950ad170`).

The RELION-float32 coarse scorer in job `12363128` reduces the broad centered
raw-score RMS from `2.95865e-5` to `1.57258e-5` and posterior total variation
from `1.10784e-5` to `5.30878e-6`, but it still retains 28,555 tuples and
drops the same target.  Float32 cutoff arithmetic is therefore not the root;
the next unequal boundary is inside the target/best raw score operands or
their reduction.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_relion_f32_coarse_it2_20260814T0115ET/analysis/K1_STACK2690_RELION_F32_COARSE_BOUNDARY.json`
(SHA-256
`1053e4d815fd2be78b80e6a6aff40d0b466acf72f5d2ab17dda724d4eff7b715`).

The fixed complete-case scorecard is deliberately unchanged at `28/34`
strict, `32/34` topology, and `34/34` evaluated.  These focused experiments
localize two implementation boundaries; they do not yet claim an autonomous
case-level pass.

## Same-input RELION fine-order A/B

Job `12363443` completed the physical-iteration-2 half-1 boundary in 832
seconds with exact fine-posterior arithmetic, exact BPref operands, native
particle order, one particle-owned accumulator launch per particle, and the
verified RELION fine-parent traversal.  Fine traversal order has no effect on
the posterior boundary: all 1,490 Pmax values and all 1,490 significant-count
values are bitwise identical to the exact-posterior arm without the traversal
change.  Both arms retain 276 support-count mismatches and Pmax relative L2
`6.472474761705944e-6`, versus 47 mismatches and `5.5612638434996375e-6` in
the non-exact-posterior baseline.  This falsifies fine traversal order as the
posterior/support root.  The fixed-denominator report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_sameinput_exact_posterior_relion_rotation_order_20260814T0130ET/analysis/K1_CASE22_IT2_SAMEINPUT_PARTICLE_STATE_ORDER_AB.json`
(SHA-256
`10b5f6df30bee95dc0442494a09e9b31ef7fa784261fcfa7dd23058da9837cce`).

The traversal does have the expected small BPref reduction-order effect.  In
shells 16--30, RECOVAR-to-matching-native numerator, denominator, and
reconstructed-average relative L2 improve from
`3.2196618e-6`/`4.3990191e-6`/`9.3779787e-6` to
`3.1672351e-6`/`4.2640364e-6`/`9.2098410e-6`.  This is a mediator-level
improvement, not closure; the posterior and raw coarse-score mismatch remains
upstream.  The accumulator report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_sameinput_exact_posterior_relion_rotation_order_20260814T0130ET/analysis/K1_CASE22_IT2_HALF1_RELION_ROTATION_ORDER_MATCHED_NATIVE.json`
(SHA-256
`37ea3689bb18fbdaa5121e6b89859a65854d213a6fa49b5652a1972021a4198a`).

## Bounded coarse operand panel

Native job `12363432` captured stack 2690's complete coarse component table
and two projected-reference operand rows.  Matching RECOVAR job `12363923`
completed in 120 seconds and captured those same mapped rows, 534 and 19000.
The target inputs are already close: projected-reference, weighted-shifted-
image, and correction relative L2 are `6.9157e-8`, `3.61485e-7`, and
`4.33019e-7`.  Substituting the captured native operands does not reduce the
centered two-rotation score residual; each individual and joint substitution
increases its energy by approximately `5.86x`.  This disfavors a large
projected-reference, shifted-image, or correction operand error and raises
the native CUDA lane/reduction order as the next boundary.

This result is not yet classification-ready.  The exact native operand replay
passes all fixed gates, including production-diff2 centered p95 and maximum
`1.52587890625e-5`.  The broader component replay has centered p95
`6.103515625e-5`, just above its fixed `5e-5` gate, so the report remains
explicitly rejected for causal classification.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_native_components_it2_20260814T0120ET/analysis/K1_STACK2690_COARSE_OPERAND_BOUNDARY_V3_NATIVE_PAIR.json`
(SHA-256
`b285a898626df551b1a3bf45d59dea4a1b77561dfca1e977a19adb94f40da55c`).
The next decisive capture must pair the missing target with the true shared
best rotation under the same validated operand schema, then compare exact
per-lane partials and their reduction sequence rather than weakening the
component gate.

## Native coarse lanes and the earlier iteration-1 M-step boundary

Native RELION lane-capture job `12364438` completed the scientific step and
wrote all three requested artifacts for half-1 particle 1879 / one-based
stack index 2690.  The Slurm batch ended nonzero only in the post-run Python
validator because it invoked a repository module by file path.  Manual module
validation passes the fixed schema and completeness gates.  The run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_native_coarse_lanes_it2_20260814T1435ET/`, with a `SAFE_TO_DELETE` marker and
Slurm logs under its `logs/` directory.  The earlier capture attempt's empty
output was an infrastructure error: `sbatch --export=NONE` also set
`SLURM_EXPORT_ENV=NONE`, preventing the nested `srun` from receiving the
capture variables.  The corrected launcher exports `SLURM_EXPORT_ENV=ALL`
after sanitizing the environment.

The native production output is attainable from its captured four lane
partials for all `58/58` target/best translation values.  Enumerating all 24
legal lane orders proves that lane ordering cannot explain the missing
target: the target rotation 25632 / translation 13 relative to anchor rotation
19619 / translation 14 is invariant across every order.  The best single
fixed order matches `50/58` production values, but no fixed order is required
for the target/anchor conclusion.  Passive operand replay still differs from
the active kernel (`192/232` lane values exact, maximum absolute difference
`7.62939453125e-6`), so the first unresolved native coarse boundary is now
inside the production lane operands or their in-kernel arithmetic, not the
atomic lane reduction order.  The exact analyzer output is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2690_native_coarse_lanes_it2_20260814T1435ET/analysis/K1_STACK2690_NATIVE_LANE_ORDER_V2.json`.

The corrected RECOVAR operand-pair job `12364551` used rotations 534 and
27288, the actual mappings of native rotations 25632 and 19619.  It happened
to reproduce the sealed native target/anchor relative raw score exactly,
`-9.772491455078125`, and retained the target with `28,556` tuples.  Earlier
otherwise identical H100 runs produced `-9.772506713867188` and `28,555`
tuples.  This one-ULP boundary is real, but a deterministic coarse-reduction
A/B falsifies coarse atomic scheduling as the trajectory fix.

Deterministic coarse replicas `12365071` and `12365072` completed on separate
H100s.  Both reproduce the native target/anchor delta, both retain `28,555`
tuples, and their significant masks and indices are bitwise identical.  Their
iteration-2 score tables nevertheless differ in 458 pre-prior entries and 260
post-prior/weight entries.  More importantly, their incoming iteration-1
hard-assignment, FSC, noise, rotation, translation, and tau2 artifacts are
bitwise identical, while the iteration-1 M-step accumulators already differ:

- half-1 `Ft_y`: 3,352 unequal values, relative L2 `2.15166e-9`, maximum
  absolute difference `3.72529e-9`;
- half-2 `Ft_y`: 3,632 unequal values, relative L2 `2.90811e-9`, maximum
  absolute difference `3.72529e-9`;
- half-1 `Ft_ctf`: 630 unequal values, relative L2 `2.21240e-9`, maximum
  absolute difference `1.45519e-11`;
- half-2 `Ft_ctf`: 740 unequal values, relative L2 `2.64693e-9`, maximum
  absolute difference `2.91038e-11`.

The regularized half maps then differ and feed the later coarse-score split.
The opt-in deterministic coarse reducer is therefore not retained as a code
fix.  The next bounded discriminator moves earlier, to physical iteration 1:
join identical particles and retained candidates, then compare normalized
weights, per-particle BPref numerator/denominator operands, destination
indices, microbatch partials, and the first unequal global accumulator.  This
tests the demonstrated earliest varying boundary directly and does not wait
for another full trajectory.

The fixed complete-case scorecard remains deliberately unchanged at `28/34`
strict, `32/34` topology, and `34/34` evaluated.  These experiments eliminate
a coarse reduction-order hypothesis; they do not yet add a passing case.

## Physical-iteration-1 BPref closure and diagnostic stage correction

Fresh native BPref replicas `12365573` and `12365686` and fresh RECOVAR
replicas `12365651` and `12365687` initially appeared to expose a large
iteration-1 accumulator difference.  Comparing the RECOVAR pre-join buffer to
RELION's `BackProjector::getDownsampledAverage` raw buffer gave half-1/half-2
numerator relative L2 `0.1119287/0.1254246`.  Shellwise inspection showed that
the difference was confined to shells 1--15 and was nearly identical between
halves; shells 16--28 already agreed at approximately `1e-8`.

This was a diagnostic stage mismatch.  RELION calls
`BackProjector::getDownsampledAverage` during reconstruction, after
`MlOptimiserMpi::joinTwoHalvesAtLowResolution` has replaced both halves' low
shells.  The RECOVAR archive was explicitly
`recovar-bpref-prejoin-v2`.  Comparing the native raw buffer to RECOVAR's
`recovar-bpref-accum-v2` post-join archive closes the aggregate boundary:

| Half | Numerator rel-L2 | Denominator rel-L2 | Native repeat numerator | Native repeat denominator |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `1.2428771e-8` | `7.3257757e-9` | `1.1391068e-8` | `6.1481220e-9` |
| 2 | `1.4007627e-8` | `6.3395368e-9` | `1.1348108e-8` | `5.8959598e-9` |

The RECOVAR repeat residual is smaller still: half-1/half-2 numerator
`1.3573186e-9/1.4721324e-9` and denominator
`8.3289025e-10/1.0289959e-9`.  The accepted reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_recovar_it1_prejoin_20260814T0240ET/analysis/K1_CASE22_IT1_POSTJOIN_RAW_ACCUM_HALF1_V4.json`
  (SHA-256
  `ea67275969a5a483d554d75093343819cd1dec3dcff08ebe897c608831dd0101`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_recovar_it1_prejoin_20260814T0240ET/analysis/K1_CASE22_IT1_POSTJOIN_RAW_ACCUM_HALF2_V4.json`
  (SHA-256
  `20048a82da712b5277f0cbe17e8f0dbf15794c4914517851f853b7aefada19e7`).

Independent particle-state audit at this same boundary has exact Pmax and
significant-support values for all `3000/3000` particles.  Hard-pose residuals
are limited to STAR metadata precision (angular p95 `9.24145e-6` degrees and
translation p95 `2.68214e-6` Angstrom).  Focused scatter replay `12366092`
also closes rotation matrices, Fourier coordinates, Hermitian folding, all
eight neighbor indices, and interpolation coefficients exactly for all three
fixed particles.  Its classification is
`fixed_panel_scatter_geometry_closes` in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_scatter_geometry_replay_20260814T0250ET/analysis/CASE22_IT2_SCATTER_GEOMETRY.json`.

Physical iteration 1 therefore closes through particle state, captured
pre-scatter operands, scatter geometry, post-join BPref accumulation, and the
regularized map at the measured repeat envelope.  The deterministic coarse
replicas' tiny iteration-1 differences are ordinary engine repeat variation,
not a demonstrated parity root; native raw BPref itself varies by about
`1e-8` between repeats.

The analyzer now treats BPref stage as a mandatory identity field.  Raw
`getDownsampledAverage` captures default to `post_lowres_join`; explicit MPI
state captures declare `pre_lowres_join` or `post_lowres_join`.  Cross-stage
comparisons fail closed.  The fresh case-22 launcher now compares post-join
to post-join, while the case-26 MPI state launcher explicitly declares its
pre-join capture.

The first material K=1 target remains physical iteration 2.  Existing
same-input evidence localizes it before reconstruction: the fixed 64-particle
cohort has exact candidate sets for `54/64`, exact positive rotation sets for
`30/64`, and exact significant-sample counts for only `5/64`, even though all
`64/64` normalized reconstruction-mass comparisons pass the fixed `1e-3`
gate.  The next focused discriminator is therefore the normal production
candidate grid and cumulative-support boundary, joined by immutable particle
and candidate identity; reconstruction reduction is downstream until those
inputs agree.

The later guarded direct-noise checkpoint promotes case 26.  The current
complete-case scorecard is `29/34` strict, `32/34` topology, and `34/34`
evaluated.  Job `12374036` is the first focused follow-up at this boundary:
its Gaussian/F32 coarse capture closes all 14 parent sets and winners, leaving
only two one-translation candidate-mask differences.

## Gaussian coarse cutoff and production-lane localization

The two residual job-`12374036` support differences were stacks 2322 and
2994. Replaying native RELION raw `diff2` with the current RECOVAR priors
through RECOVAR's exact CUDA CUB normalization and cutoff closes both
particles exactly. The replay reproduces native `sum_weight`, significant
threshold, significant count, and the complete significant mask for both
particles. This rules out the current priors, normalization domain, CUB sum,
threshold comparison, and tie handling at this boundary. The accepted report
is `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_native_raw_gpu_replay_178966c2_20260814T0815ET/NATIVE_RAW_RECOVAR_PRIOR_CUB_REPLAY.json`
(SHA-256
`37545ca414c64f984c40f9c5d41d96e70deec27d02999cf75ca75207ddc9461e`).

The bounded translation-arithmetic A/B does not supply a universal fix. The
CUDA `sincosf` arm closes stack 2322 but leaves stack 2994 unchanged; the
exact-operand arm moves stack 2322's mismatch to a different candidate and
also leaves stack 2994 unchanged. Neither opt-in is promoted. A later default
operand-panel replay happened to close the complete support mask for both
particles while preserving centered raw-score residuals of at most
`6.103515625e-5` and `4.57763671875e-5`. This demonstrates that the
one-candidate support result can move within the scorer's float32 execution
envelope and is not, by itself, a stable scientific regression metric.

Native production-lane jobs `12375426` and `12375427` then captured the
actual four coarse CUDA lane partials for three rotations and all 29
translations per particle. All `87/87` production scores for each particle
are exactly reachable by a legal ordering of the four native partials. The
passive operand replay is not exact: `285/348` and `286/348` active lane
values are bitwise equal, with maximum absolute discrepancies
`3.814697265625e-6` and `7.62939453125e-6`. Each particle has three
production candidates that cannot be reached from the passive operand replay
even though every candidate is reachable from the captured active lanes. The
first unresolved raw-scorer boundary is therefore the active kernel's
per-pixel operands or arithmetic, followed by atomic arrival order; it is not
the posterior cutoff implementation. The accepted lane reports are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2322_native_coarse_lanes_it2_20260814T0820ET_retry1/analysis/RELION_LANE_VALIDATION.json`
  (SHA-256
  `321468c252a8525a21320c1267f778716bb7decc0e9f98a1a1b9617da713fe43`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack2994_native_coarse_lanes_it2_20260814T0820ET_retry1/analysis/RELION_LANE_VALIDATION.json`
  (SHA-256
  `07458556b8051648596a6cbfe9768352232f3af67f24b47c9343a62e5c593075`).

The active-lane atomic envelopes explain `71/87` and `63/87` of RECOVAR's
captured centered panel scores exactly. The remaining distances are only one
float32 ULP at p95 and at most two ULPs. There is no single fixed order of the
four lane atomics that reproduces all native candidates, so replacing the
scorer with one arbitrary deterministic lane order would overfit the panel.
Operand comparisons remain small: projected-reference relative L2 is
`5.06e-8`/`5.96e-8`, weighted shifted-image relative L2 is
`4.02e-7`/`3.10e-7`, and correction relative L2 is
`5.26e-7`/`2.29e-7` for stacks 2322/2994. Because passive RELION operands do
not reproduce RELION's own active lanes, these substitutions are diagnostic
only; the next exact scorer experiment must capture per-pixel values from
inside the active production kernel or compare its generated instruction
sequence. Scientific acceptance remains the fixed trajectory and FSC-AUC,
not bitwise overfitting of one cutoff candidate.

## Full guarded-default case-22 closure

The autonomous guarded-default trajectory resolves the scientific question
without overfitting the remaining one-ULP scorer envelope.  Science job
`12377247` and audit job `12377829` completed with exit code zero from clean
commit `e791e87502b51fee8b8c2ba9c7de80ee922421ec`.  RECOVAR converged at
physical iteration 11 and exactly matched RELION's full current-size schedule
`56, 60, 80, 70, 76, 70, 72, 70, 70, 70, 70` and HEALPix schedule.

Every numbered merged signed FSC-AUC passes; the minimum is
`0.9999968583440365` at iteration 10.  The authentic final all-data result is:

| Metric | Result | Frozen gate |
| --- | ---: | ---: |
| final merged cross-engine signed FSC-AUC | `0.9977673455104741` | `>= 0.995` |
| final half-1 signed FSC-AUC | `0.9989153794653891` | diagnostic |
| final half-2 signed FSC-AUC | `0.999237320197774` | diagnostic |
| final RECOVAR-minus-RELION merged GT FSC-AUC | `+0.009642596478529442` | `>= -0.002` |

The strict report set `completion_claim=true` and has no failures:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_default_full_e791e875_20260814T0922ET/analysis/case22_default_full_strict_fsc.json`
(SHA-256
`f9e7ae42c4ef1dfd3778392bbd0ba6f808fe6bb7930f0b2e26c06605a4f71a60`).
The immutable superseding ledger is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_default_full_e791e875_20260814T0922ET/analysis/em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v11.json`
(SHA-256
`ea466f54e32a7d29043bda06e56ebff7a781b193e76dedbf6c36dc93813260e7`).

This promotes fixed case `k1-22` and moves the frozen-denominator metric to
`30/34` strict, `33/34` exact topology, and `34/34` evaluated.  The remaining
K=1 strict failures are cases `4`, `5`, `7`, and `10`; only case `7` still
fails topology.

Separate stopped-boundary controls also rule out CUDA 12.6-versus-12.8 and
compute-80 PTX JIT as causes: current-source CUDA 12.6 PTX, CUDA 12.6 sm90,
and CUDA 12.8 sm90 reproduce the same stack-1204 winner/support result.  The
observed library difference is therefore source-version arithmetic, not the
compiler or PTX route.  It is retained as a preservation diagnostic and is
not required for the accepted default case-22 closure.
