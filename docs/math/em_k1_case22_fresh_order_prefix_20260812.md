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
