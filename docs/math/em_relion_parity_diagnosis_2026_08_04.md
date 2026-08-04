# RECOVAR–RELION EM parity diagnosis, 2026-08-04

This decision record converts the external causal review into the active parity
workflow. It distinguishes structural correctness from causal effectiveness and
does not change any fixed acceptance threshold.

## Fixed progress metrics

- K=1 strict signed-FSC/FSC-AUC cases: **28 / 34**.
- K=1 topology cases: **32 / 34**.
- K=1 evaluated cases: **34 / 34**.
- K=4 direct class-iteration comparisons: **41 / 60**.
- K=4 all-class iterations: **9 / 15**.
- K=4 RECOVAR all-class boundary captures: **4 / 4**; this is one-sided
  capture completeness, not cross-engine parity.
- K=4 RECOVAR all-class boundary repeatability gates: **9 / 9**; this makes
  the fixed iteration-2 RECOVAR boundary stable, not cross-engine-equal.
- K=4 native auxiliary-stream repeatability gates: **12 / 13** (historical
  predecessor panel).
- K=4 native soft-mask-partial repeatability gates: **13 / 14**.
- K=4 native high-resolution-Xi2 observer repeatability gates: **15 / 15**.
- K=4 stable native operand-capture admission: **1 / 1**; this is an observer
  admission result, not cross-engine posterior/BPref/map parity.
- K=4 native highres-treatment/all-class admission: **19 / 21**; rejected on
  the fixed global Pmax and support-count repeatability envelopes.
- K=4 native classes-2--4 target-artifact admission: **32 / 32**; accepted for
  target-local use only.
- K=4 target-local classes with exact tuples and support but first unequal raw
  `diff2`: **4 / 4**.
- Fresh K=1 dispatch alignment: **2 / 2** cases verified.
- Fresh K=1 dispatch standalone rescue: **0 / 2** cases.

The K=1 strict failures remain cases 04, 05, 07, 10, 22, and 26. Cases 07
and 22 are the topology failures. The K=4 failures are the late iterations
10--15.

## Current causal decisions

### Evidence-status ledger

The external review's evidence labels are retained, but later fixed-panel
experiments supersede its ranking where they provide direct causal evidence.

| Conclusion | Status | Current evidence |
| --- | --- | --- |
| RELION uses one continued C RNG stream for the two half shuffles and then stable integer-optics sorts | Demonstrated structurally | Exact deployed-revision source and paired-helper tests |
| The guarded RECOVAR candidate constructs the intended permutation and keeps the first 100 expected-accuracy identities, order, and runtime CTF rows aligned | Demonstrated for cases 22 and 26 | Complete candidate-order, inverse-map, identity, and CTF gates passed in both A/Bs |
| The guarded physical dispatch is a sufficient fix for case 22 or case 26 | Falsified at the tested boundary | Alignment passed 2/2; standalone rescues were 0/2; case 22 changed by `+0.000013408214354915238` FSC-AUC and case 26 by `-0.00005334034513027053` |
| Particle order can remain a numerical mediator through stable support buckets and reductions | Mechanistically supported, not causal here | Stable within-bucket execution preserves input order, but no tested scientific rescue |
| The iteration-12 controller split is the original case-7 defect | Strongly disfavored | Material Pmax/map drift is reported from iteration 2 through 11 |
| The iteration-12 split amplifies earlier drift | Strongly inferred | It changes current size and angular search after earlier continuous-state divergence |
| Final fine scoring or final reconstruction is the common dominant K=1 defect | Strongly disfavored | Joint pose/reference oracles and exact-boundary/float64 interventions rescue or nearly preserve the final path |
| K=4 fixed-label failures are class swaps | Falsified for the fixed exact-A100 trajectory | Identity map and particle class assignments are optimal at all 15 iterations |
| K=4 current-source RECOVAR iteration-2 all-class boundary is stable | Demonstrated | Job `11994138` passed all 9/9 exact-byte gates across two independent executions on one A100 |
| K=4 target-local tuples and significant support agree in all four admitted classes | Demonstrated | Exact `247232/247232` tuples and `66986/66986` support rows across four independent target-local joins |
| K=4 raw `diff2` is the first unequal measured boundary in all four admitted classes | Demonstrated in the narrow target-local scopes | 55,658 float32 mismatches, at most three ULPs; broad native all-class admission remains rejected |
| Native lane-first soft-mask reduction closes the pinned class-1 shifted-image residual | Falsified at the tested tuple | Native-lane relative L2 was `1.0312363428376726e-07`, slightly worse than default RELION-CUDA at `9.863882911656713e-08` |
| K=4 reduction order alone is the primary cause | Disfavored, not eliminated | Determinism improves repeatability without improving the fixed 41/60 cross-engine score; identical all-class operands/destinations have not yet been shown |
| Full deployed RELION in-memory order beyond the reconstructed source semantics has been directly hashed | Unknown | The candidate is source-faithful and fully internally checked, but no native full-order runtime hash is available |

### Active hypothesis queue

For K=1, the next discriminator is deliberately broader than a guessed fix:

1. inherited incoming reference/state drift at the iteration-2 boundary;
2. raw coarse/fine score, CTF, phase, interpolation, or normalization operands;
3. additive direction/translation priors or particle-aligned metadata;
4. joint normalization, ordered significant-support threshold, or tie semantics;
5. per-particle BPref operands or destination indexing;
6. reduction order, padding, or scatter after identical support and operands;
7. particle dispatch as a possible mediator, but not a sufficient production
   fix on the current 0/2 causal result.

For K=4, the current order is:

1. raw fine-score input/operand or arithmetic-order mismatch, provisionally
   observed for class 1;
2. combined class--rotation and translation-prior construction;
3. flattened joint class-pose normalization and global support semantics;
4. evaluated all-class tuple sets and `--firstiter_cc` global-winner routing;
5. class-specific BPref operands and accumulator destinations;
6. reduction arithmetic alone.

The first unequal boundary in an admitted native capture scope, rather than
this provisional ordering, chooses the implementation target. Broad native
all-class attribution is currently prohibited by its failed 19/21 admission;
the RECOVAR side is stable at 9/9 exact repeatability gates.

### K=1 particle order

The exact RELION fresh-run order is structurally established:

1. source insertion order;
2. stable random-subset partition;
3. one `srand(random_seed + iter)` call;
4. half-1 `std::random_shuffle`;
5. half-2 `std::random_shuffle` using the continued RNG stream;
6. separate stable integer-optics-group sorts within the two halves.

Case 22 and case 26 now verify the proposed complete physical order, ordered
first-100 expected-accuracy identities, runtime CTF rows, inverse mapping, and
alignment invariants. Neither case is rescued by the physical-order treatment.
Particle order is therefore retained as a required structural invariant and a
possible numerical mediator, but the tested implementation is rejected as a
sufficient standalone K=1 parity fix. It must not be promoted to production on
the basis of the current experiments.

These claims apply to the sealed **local** candidate lineage, not the older
public PR head. The initial guarded implementation is `71477afd`, explicit
identity physical order for expected accuracy is `c9eec5ab`, and the scientific
A/B source is sealed at `e2893cb3`. The static intervention audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_dispatch_static_audit_e2893cb3_20260802T1523ET/provenance/STATIC_INTERVENTION_AUDIT.json`
(SHA-256
`144b67dfcba4fed28d70b19f7a24a240147f759ea03d7b8a8f879ce161d08bf5`).
It verifies all 13 STAR columns after immutable-ID inverse join, the complete
treatment order, the ordered first-100 identities, and byte-exact runtime
float64 CTF rows in both cases. The candidate remains unpushed because its
scientific A/B result is negative.

The physical-reorder versus internal-execution-permutation experiment remains
conditional on a positive clean dispatch A/B. Running it after two unsupported
A/B results would not be the next decisive experiment.

If physical ordering is revisited, one authoritative order plan must own both
gather and scatter maps. It must be applied before all particle-aligned caches,
poses, priors, CTF overrides, scale rows, and replay arrays are constructed.
Once the dataset is physically ordered, expected accuracy should consume the
identity local order and assert the ordered first 100 immutable source IDs and
CTF rows rather than reconstructing identity through another shuffle/inverse.

### K=1 trajectory and final-only failures

The first reported material case-7 posterior mismatch is iteration 2. The
iteration-12 current-size/HEALPix split is a downstream discrete controller
bifurcation and likely amplifier; it is not treated as the original defect.
Controller code is not changed unless identical FSC, data-versus-prior,
tau-squared, accuracy, size/order, and convergence-history inputs produce a
different RELION decision.

The final-only failures are treated as upstream last-numbered pose/reference
state errors. Joint pose-and-reference oracles and exact-boundary runs strongly
disfavor the final fine scorer and final reconstruction as the common dominant
cause. A residual final-boundary mismatch remains possible and will only be
revisited after the numbered upstream boundary is repaired.

The next K=1 discriminator is a case-7 iteration-2 same-input comparison in
this order:

1. immutable source IDs, images, CTFs, incoming maps, and controller state;
2. candidate tuple keys;
3. raw coarse score components and additive priors;
4. coarse stabilization, normalizer, weights, and winners;
5. fine parent/child tuples;
6. raw fine score components and additive priors;
7. fine stabilization, normalizer, weights, support threshold/margin, and
   ordered support;
8. Pmax, hard assignments, and per-particle BPref operands;
9. partial reductions, final accumulators, and half maps.

The fixed-panel BPref analyzer must require exact significant-support identity
before it may classify the particle pre-scatter boundary as closed. Operand
agreement evaluated only on RELION's retained tuples is insufficient when
RECOVAR retains a different tuple set. The active Slurm launch is hash-pinned
and is not mutated; its original report will be preserved, then a corrected
post-terminal classification will place `support_exact` ahead of operand and
translation-reduction closure.

The first unequal level determines the next code change. A RELION-state
substitution can distinguish inherited drift from a local RECOVAR mismatch,
but native cross-engine raw equality requires an immutable-ID-aligned RELION
capture or a source-faithful reference calculation.

### K=4

Reduction order alone is disfavored because deterministic accumulation improved
repeatability without materially improving direct class FSC. It is not ruled
out until identical per-particle operands and destination sequences are shown.

K=4 localization proceeds through one staged boundary, joined by particle ID,
half, class, rotation ID, and translation ID:

1. evaluated class-pose tuple sets;
2. raw `diff2` and its image/reference/cross components;
3. direction, translation, and class log priors;
4. flattened joint class-pose stabilization and normalization;
5. global coarse winning class/pose and `--firstiter_cc` fine-parent routing;
6. global class-pose significant-support ordering, threshold, and margin;
7. class-specific weighted-image and CTF-squared BPref operands;
8. accumulator destinations and partial/final reductions;
9. reconstructed class maps.

At the deployed no-orientation-prior boundary, RELION's captured
`orientation_log_prior` is not a direction-only term: it is constructed from
`mymodel.pdf_direction[exp_iclass]`, whose class row carries the class mass.
The skip-align branch uses `mymodel.pdf_class[exp_iclass]` directly. RECOVAR's
captured `rotation_log_prior` likewise contains the combined class--rotation
term. The staged analyzer must therefore label this a **combined
class--rotation prior** comparison. If that is the first unequal boundary, the
next capture must also retain `pdf_class` and the conditional direction prior
separately; it must not attribute the mismatch to direction or class priors
from the combined value alone.

The existing fixed-target operand replay closes exactly while the complete
active raw table does not. Current prior mismatches are inert for the observed
support and are routed downstream rather than treated as the causal boundary.

A provisional class-1 join between the admitted native observer and the
same-UUID RECOVAR capture now gives exact tuple and support identities:
`109184/109184` active tuples and `38982/38982` significant tuples. The first
unequal measured boundary is raw `diff2`: `25687/109184` float32 values differ,
with maximum absolute delta `0.0001220703125`, relative L2
`4.9158666958484043e-08`, median mismatch distance one ULP, p95 two ULPs, and
maximum three ULPs. The first unequal tuple is native/RECOVAR rotation-local
zero and translation 66. The combined class--rotation prior differs in 15,264
rows by at most one ULP; the translation prior differs in 40,800 rows by at
most two ULPs. The resulting unnormalized score differs in 101,388 rows and
the class-1 posterior differs bitwise in all 109,184 rows, while global
significant support remains exact. This is evidence for a small scorer/operand
or arithmetic-order discrepancy before normalization, not evidence that such
a discrepancy causes the late map failures. It is accepted only in the narrow
class-1 target scope; the rejected broad native admission prevents an
all-class generalization.

The accepted class-1 support identity is not an immediate class-local tie. In
the native raw float32 weights, the minimum retained value equals the recorded
threshold and is 1,917 float32 ULPs above the maximum excluded active value;
after native float32 normalization, the retained/excluded gap is
`8.618883384770015e-12`, or 2,426 ULPs. RECOVAR's original float64
probabilities have a corresponding class-local gap of
`8.614616903405886e-12`. These margins weaken tie behavior as the owner for
this target/class. They do not establish the global four-class margin or rule
out later particles near a support boundary.

The provisional mismatch is distributed rather than confined to one tuple:
25,687/109,184 active candidates differ (23.5%), with 18,766 positive and
6,921 negative RECOVAR-minus-RELION deltas, signed sum `0.628997802734375`,
and L1 sum `1.267913818359375`. The ten largest rotation strata account for
only 1.06% of mismatch L1, whereas the ten largest translation strata account
for 23.9%. The predeclared representative-selection rule chooses native
rotation-local 1790, mapped RECOVAR rotation row 2878, translation 28: native
raw `diff2` `512.4172973632812` versus RECOVAR `512.4174194335938`, a
two-ULP delta. This representative supersedes a simple first-mismatch target
for a bounded operand capture only in a native scope that passes its own fixed
admission. The broad all-class native panel did not pass, so the representative
cannot authorize a broad join.

The fixed RECOVAR iteration-2 boundary for stack identity 53723 is now complete
for all four classes. Its four artifacts preserve 247,232 active class-pose
tuples, 66,986 significant tuples, and a joint probability mass of
`0.9999999999999997`; an independent fixed-order host replay agrees with the
stored joint probabilities to maximum absolute error
`4.336808689942018e-19`. This accepts only the RECOVAR side of the join. Native
RELION capture must come from the same accepted observer lineage before tuple,
raw-score, prior, normalization, support, or BPref comparisons are causal.

That RECOVAR boundary is now independently repeatable. Slurm job `11994138`
completed two fresh executions sequentially on node `della-l07g2`, physical
A100 UUID `GPU-f3e94635-d095-bea9-dbe3-26e91dd3ea27`, and passed all **9/9**
predeclared exact-byte gates: both-arm validity, immutable identity, geometry
and candidate tuples, raw `diff2`, priors, unnormalized scores, joint
posterior, and global significant support. The four artifacts in each arm have
identical SHA-256 digests, preserving 247,232 active and 66,986 significant
class-pose tuples. The result allows stable attribution on the RECOVAR side;
it does not override the rejected broad native all-class admission and cannot
change the frozen K=4 FSC/FSC-AUC score. The immutable report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/analysis/RECOVAR_ALLCLASS_REPEATABILITY_11994138.json`
(SHA-256
`3e2341222a1a2e00a014995245709f8c5383eed9d611adcf23128bbc34f7f4cd`).
The checked fixed-denominator repo record is
`docs/math/em_k4_allclass_recovar_repeatability_scorecard_v1.json`.

A prior independent RECOVAR same-allocation raw-operand pair, job `11831421`
at source `db1ab501`, is an explicit warning that one-sided captures need a
repeatability control. Both arms self-replayed all 109,184 active raw costs
bitwise, but 11,918 cross-arm raw costs differed by at most
`0.0001220703125`. Of the captured operand families, only the shifted image
differed (16,983/69,136 complex entries); substituting that one family closed
all 11,918 raw-cost mismatches. Projection, score weights, half weights, the
high-shell scalar, identity, and topology were byte-exact. The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_sameallocation_effective_operands_retry3_db1ab501_20260731T0810ET/analysis/RAW_OPERAND_REPEATABILITY.json`
(SHA-256
`8feb772f79278a09f2b9363f4c92c0c6d7189d3fabde513edb275e9506f9666f`).
This is not stable cross-engine attribution because it used an older source and
a different physical A100; it motivated the current-source exact-GPU
repeatability gate described above.

The relevant source change after `db1ab501` is known: `e0c51765` introduced a
deterministic soft-mask background reduction and `8aa573f5` parallelized it
without restoring schedule-dependent atomics. The previous 128-lane atomic
background accumulation was replaced by fixed per-block reductions followed by
a deterministic second-stage sum. Therefore the old shifted-image instability
is not presumed to survive in current source `223e7e81`; job `11994138` is the
direct independent-process regression test for that claim.

Source comparison now identifies a narrower, falsifiable preprocessing
hypothesis even if both implementations are repeatable. The accepted native
observer and current RECOVAR use the same 128 blocks, 128 lanes, pixel-owner
formula, radius tests, and per-lane pixel traversal, but they use different
float32 addition trees:

1. The native observer writes all `128 * 128` block--lane partials, adds a
   fixed lane across blocks in increasing block order, and finally applies a
   CUB reduction to the 128 lane totals.
2. RECOVAR applies a CUB block reduction across the 128 lanes first, writes
   128 block totals, and finally applies a CUB reduction to those block totals.

The two trees are deterministic but not arithmetically identical. This makes
soft-mask background grouping a concrete candidate for stable shifted-image
and raw-`diff2` ULP differences; it does not yet establish causality for the
late class-map failures. No production change is admissible from source
inspection alone. Current-source RECOVAR repeatability subsequently passed,
while broad native treatment/capture admission failed. A bounded same-input
diagnostic is therefore permitted only in a narrower native target scope that
passes its own fixed admission. The hypothesis is falsified if matching the
native tree does not close those first unequal operands.

A local diagnostic implementation now exposes the native lane-first tree under
a separate FFI symbol while preserving the accepted block-first symbol and ABI
as the default. On one local A100 and a fixed four-image 256-by-256 smoke input,
both modes repeated byte-exactly four times; normalization and integer
translation were exact between modes. The default masked output was also
byte-exact to the sealed pre-change library (SHA-256
`811e8a647e0feba27fb5c4955f6092fb81f2756da50e47c9dc546f799c8f4caf`).
The native-lane tree changed 56,683/262,144 masked float32 elements, with
maximum absolute delta `5.960464477539063e-08`. This proves that the proposed
intervention changes the intended arithmetic boundary while leaving the
default path unchanged; it is still only a local diagnostic, not cross-engine
causal evidence or a scorecard improvement. The immutable smoke report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_softmask_native_lane_bfaaec4d_20260804T0750ET/provenance/LOCAL_GPU_SMOKE.json`.

The bounded fine-operand comparator now reports the native-lane path as a
separate `relion_cuda_native_lane` preprocessing counterfactual alongside the
captured production path, dataset-native JAX FFT, and default RELION-CUDA
preprocessing. It uses identical normalization, CTF, translation, projection,
candidate, and score-reduction inputs; the only additional intervention is the
soft-mask background addition tree. This is a diagnostic report field only
and does not select the native-lane path in production. The focused
CPU/GPU/symbol/routing/comparator panel passes 128/128 tests. New high-precision
pass-2 captures also record the selected reduction tree explicitly, and all
replay/comparison paths default legacy captures to the established block-first
tree rather than inferring the mode from an output difference.

The predeclared one-particle, one-candidate counterfactual then falsified the
native lane-first tree as a local improvement at the pinned class-1 tuple. The
sign-aligned shifted-image residual was relative L2
`1.0312363428376726e-07` with maximum absolute delta
`0.0004920811613951751`, compared with default RELION-CUDA relative L2
`9.863882911656713e-08` and maximum absolute delta
`0.00034526698300124393`. Dataset-native JAX FFT was worse at relative L2
`2.88579426490984e-07`. All three centered score residuals are zero because the
capture contains only one candidate, so they are non-informative and cannot
support a posterior conclusion. The final immutable report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_lane_operand_counterfactual_02681e7e_20260804T0850ET/analysis/NATIVE_LANE_OPERAND_COUNTERFACTUAL_V2.json`
(SHA-256
`df8b29cb834782abc478f556542d71847866977e6d86086d15a0145574af50af`).
The old-source contribution's independent-process shifted-image warning still
applies. This result therefore removes the soft-mask reduction tree from the
leading fix queue without changing the fixed K=4 scorecards or the need for a
current-source, multi-candidate native capture admitted in the same narrow
scope.

A separate source audit found that current joint K-class support pruning uses
the normalized posterior, whereas the deployed RELION GPU path thresholds its
flattened float32 raw weights after the `+50`/`expf` transform and a float32
cumulative scan. The existing source-faithful RECOVAR float32 helper was
therefore applied as a bounded counterfactual to the complete four-class
iteration-2 target capture. It retained exactly the same `66,986` tuples as
the current joint-probability path: class counts `38,982`, `14,076`, `11,804`,
and `2,124`, with zero mask differences in four byte-exact repetitions. This
boundary is not an immediate tie: the minimum retained and maximum excluded
float32 log scores are separated by `6.103515625e-05`, or 32 float32 ULPs.
This falsifies that arithmetic distinction as a support owner for the pinned target;
it does not establish native RELION support equality or rule out threshold
differences at later, near-boundary particles. The immutable report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_joint_f32_support_probe_055ea5ca_20260804T0937ET/analysis/JOINT_F32_SUPPORT_PROBE_V2.json`
(SHA-256
`93428560797d1b7f24e66f6519068794e827d3df5270c7d68ee316dac8649277`).

As a separate provisional clue, that RECOVAR capture's high-shell scalar and
the accepted native observer's per-candidate `sum_init` are bitwise identical:
float32 value `0.07816561311483383`, bits `1033901387`. This weakens a pure
high-shell-scalar explanation but does not close it at the current same-GPU
boundary. The bounded follow-up should first compare shifted-image operands,
then projections and score weights, before reduction arithmetic.

The next all-class capture is intended to expose the first unequal global
class-pose normalization/support boundary. Class permutation is already ruled
out for the current exact-A100 fixed trajectory. Its signed 4-by-4
cross-engine FSC-AUC matrices select the identity map assignment at every one
of 15 numbered iterations, and the independently joined particle assignments
also select the identity class permutation at every iteration. At iteration
15 the matrix is

```text
[[0.993726871, 0.168255188, 0.325498694, 0.037487868],
 [0.170138501, 0.991635133, 0.320108931, 0.027329405],
 [0.324291781, 0.319346012, 0.990841324, 0.026526720],
 [0.035612276, 0.027311323, 0.028779669, 0.994521524]]
```

The source report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_full15_phaseffi_exactgpu_retry1_31c4a0ca_20260727T040500ET/analysis/relion_cuda/k4_fsc_trajectory.json`
(SHA-256
`ff35e7d548c4c0157aac349e83aadf1c1814a791f9904a8c6c93c813eae9615e`).
This excludes a label swap for the fixed late-trajectory failures; it does not
waive fixed-label acceptance or replace the operand-level localization.

Native RELION observer repeatability is an admission gate for this comparison.
Stream-replay changes are observer controls, not RECOVAR parity fixes. The
corrected same-A100 auxiliary-stream pair rejected exact capture repeatability
at 12/13 fixed gates. Its first unequal captured level was preprocessing rather
than tuple generation or priors: fixed-target preprocessed samples and
`sum_init` differed while reference pixels and tuple identities were exact,
propagating to 16,735/109,184 fine `diff2` rows and BPref `weight_norm`.

The next default-off soft-mask block-partial pair completed on job `11990914`
and rejected at 13/14 fixed gates. It made the complete fine-score artifact and
geometry-only BPref artifact byte-identical. The fine-operand artifact differed
in only three bytes, solely in its two copies of `sum_init`; all 1,520
per-pixel fields, lane partials, production/replay raw `diff2`, target state,
dispatch, topology, capture identities, and validators were exact. The two
`sum_init` values were `0.07816566526889801` and `0.07816564291715622`, a
three-ULP float32 delta (`2.2351741790771484e-08`). All four same-label signed
normalized non-DC FSC-AUC preservation values passed: `0.9999999942598741`,
`0.9999999806617705`, `0.9999999895280383`, and `0.9999999866926460`.

Source localization identified `sum_init = highres_Xi2_img[img_id] / 2` as the
only remaining field. The deployed power-class kernel uses the original
128-lane tree within each block, then schedule-dependent inter-block atomics for
the scalar `highres_Xi2`. The default-off observer control preserves all
per-pixel operations and the within-block tree, writes one scalar partial per
block, and finalizes those partials in increasing block order. Its build passed
as Slurm job `11992806`. Initial pair job `11992900` is infrastructure-invalid:
the deterministic wrapper bypassed the predecessor function that emitted the
required powerClass-stream marker, so the hash-pinned launcher stopped after a
successful arm A and arm B never ran. That job is not a scientific result and
its outputs remain preserved. Marker-only commit `17a97690` built successfully
as job `11993050`.

Corrected fixed pair job `11993105` completed on the same A100 and passed all
15/15 predeclared observer gates. Fine score, fine operand, and BPref artifacts
were byte-identical. The four signed normalized non-DC same-label map FSC-AUC
values were `0.9999999961725112`, `0.9999999807100690`,
`0.9999999921474709`, and `0.9999999852511000`, each above the fixed
`0.999999` threshold. Dispatch, topology, capture identities, target state,
runtime, and all observer-marker families also passed. Across the full 100,000
particle state, hard pose/class/shift stayed exact, while 13 Pmax rows and 15
support-count rows differed, with a maximum support-count delta of one. This
admits stable native operand localization; it does not establish joint
posterior/BPref/map parity or change the 41/60 cross-engine score.

The next K=4 causal gate separates observer treatment from capture effects. A
fixed same-A100 panel must compare predecessor/default-off high-resolution Xi2
against deterministic high-resolution Xi2 with capture disabled, then compare
that treatment against per-class capture arms. It must preserve exact hard
pose/class/shift and topology, require signed map FSC-AUC at least `0.999999`,
measure Pmax/support against the admitted native envelope, and capture all four
classes for the immutable-ID-aligned join to RECOVAR. Cross-engine attribution
remains prohibited until treatment preservation and capture inertness pass.

That seven-arm panel completed as Slurm job `11993773` on the pinned A100 and
was rejected at **19/21** fixed gates. The following gates passed: all hard
pose/class/shift fields, target-particle state, dispatch bytes and row counts,
topology, all signed map FSC-AUC checks, all four fine-score and geometry-only
BPref validators, global stabilization bits, observer markers, manifests, and
runtime checks. The minimum same-label signed normalized non-DC map FSC-AUC was
`0.9999999849558822`, above the fixed `0.999999` threshold. The two failed
gates were the prospectively fixed global Pmax and significant-support native
envelopes:

| Comparison | Pmax mismatches (max abs) | Support-count mismatches (max abs) |
| --- | ---: | ---: |
| source control vs default-off | 36 (`2.0e-5`) | 22 (`1`) |
| default-off vs highres treatment | 26 (`1.5e-5`) | 17 (`1`) |
| treatment vs class-1 capture | 10 (`2.0e-5`) | 22 (`1`) |
| treatment vs class-2 capture | 9 (`2.0e-5`) | 19 (`1`) |
| treatment vs class-3 capture | 12 (`2.7e-5`) | 17 (`1`) |
| treatment vs class-4 capture | 9 (`5.0e-6`) | 18 (`1`) |

The admitted envelope was at most 13 Pmax rows and 15 support rows, so it is
not weakened after seeing this result. The report classifies the panel as
`rejected_highres_treatment_or_allclass_capture`, sets
`allclass_operand_localization_allowed=false`, and prohibits the all-class
cross-engine join. The immutable report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_highres_treatment_allclass_capture_17a9769_20260804T0626ET/analysis/TREATMENT_ALLCLASS_RESULT_11993773.json`
(SHA-256
`c4b68323df6e31c12f5c6b32668187a082f2ab4e44a79601d32e18664f789ad7`).

The narrower class-1 target artifacts remain stable: the new fine-score SHA
`de5816046f21266c2f675c74cdbed799046bd3654e88d4e40210860ec2ede24b`
and BPref SHA
`5d1c9f08eac3e46d9ecb6aa6b1040ec28ffed929cc128637f757309ca82d7f57`
are byte-identical to both arms of accepted job `11993105`. This does not
override the failed all-class admission. It supports a narrower prospective
per-target artifact-repeatability experiment for classes 2--4 while the
independent RECOVAR side is now fixed by the accepted 9/9 repeatability gate.
Only a passing target-artifact admission can authorize a target-local join.

## Active decisive captures

The prospective classes 2--4 target-artifact repeat, Slurm job `11996846`,
completed on the pinned A100 in 25m32s and passed all **32/32** fixed gates.
All three classes passed byte-exact fine-score and BPref artifact comparison,
artifact validation, dispatch, hard pose/class/shift, target state, topology,
runtime replay, and signed normalized non-DC map FSC-AUC preservation. This
authorizes target-local use only; the rejected broad all-class join remains
prohibited. The immutable admission report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_target_artifact_repeat_17a9769_20260804T0915ET/analysis/NATIVE_TARGET_ARTIFACT_REPEAT_RESULT_11996846.json`
(SHA-256
`da59157b92956fca4095b87d2dce850cc53d1e21e4e3321474d12bd651f3c4b8`).

The fail-closed join to the RECOVAR 9/9 repeatability boundary localizes the
first unequal target-local boundary to raw `diff2` independently for classes
2, 3, and 4, matching the already admitted class-1 result. Candidate tuples
and significant support are exact in every class:

| Class | Active tuples | Raw `diff2` mismatches | Max float32 ULP | Significant support |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 109,184 | 25,687 | 3 | 38,982 / 38,982 exact |
| 2 | 65,952 | 14,503 | 3 | 14,076 / 14,076 exact |
| 3 | 64,704 | 13,806 | 3 | 11,804 / 11,804 exact |
| 4 | 7,392 | 1,662 | 3 | 2,124 / 2,124 exact |

Across the four target-local joins, all 247,232 candidate tuples and 66,986
retained tuples agree, while 55,658 raw scores differ by at most three float32
ULPs. Priors and normalized posteriors also differ, but raw `diff2` is earlier
in the causal ordering. The result prioritizes score inputs and arithmetic; it
does not prove that these small score differences cause the late class-map
failures. The immutable target-local report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_target_local_join_11996846_11994138_20260804T1104ET/analysis/K4_TARGET_LOCAL_BOUNDARY.json`
(SHA-256
`66695693709c34ec036632aaec9de91f024af9a596d4882f4312ccea3e0f6a27`).

The class-1 raw-score mismatch is independently reproducible over the admitted
narrow tuple join: 25,687 of 109,184 raw `diff2` values differ by at most three
float32 ULPs while tuple, prior, and significant-support identities remain
exact. A one-candidate operand dump cannot localize this residual because
centering one value removes all candidate-relative information. The new panel
therefore freezes native rotation-local `1790`, which maps to RECOVAR class-local
row `2878` and global rotation ID `4446`. It has 96 active translations. In the
already admitted fine-score/pass-2 artifacts, 29 of those 96 raw values differ;
the centered RECOVAR-minus-RELION residual has L2
`0.0002845808477116318` and maximum absolute value
`0.00011507670084635417`.

Two fail-closed jobs implement the staged comparison:

- Slurm job `12001297` captures two independent RECOVAR class-1 contribution
  bundles in one allocation. It requires exact bytes for pass-2 state plus raw
  operands, the full high-precision contribution archive, and its device
  signature: a prospective 3/3 gate. Its root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_class1_contribution_repeat_223e7e81_20260804T1034ET`.
- Slurm job `12001296` captures the 96 native RELION fine operands. It requires
  the accepted native observer, exact source/binary and replay streams, exact
  predecessor class-1 fine-score and BPref hashes, the fine-operand validator,
  and the fixed candidate panel: a prospective 7/7 gate. Its root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_class1_fine_operands_rot1790_17a9769_20260804T1034ET`.

Native job `12001296` completed in 8m38s and passed all **7/7** fixed gates,
including the 96-candidate scope, artifact validators, exact predecessor
fine-score and BPref hashes, replay/dispatch markers, and exact treatment
source and binary. Its immutable admission report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_native_class1_fine_operands_rot1790_17a9769_20260804T1034ET/analysis/NATIVE_FINE_OPERAND_ADMISSION_12001296.json`
(SHA-256
`37d58d00dbc250a0d12dd834920c07b1c8b193b47cd1645c561a7c5f0089c60c`).

RECOVAR job `12001297` is infrastructure-invalid: its emptiness guard counted
the intentionally pre-created empty capture leaf directories and exited in two
seconds before GPU validation, imports, or either science arm. Corrected retry
job `12003112` passed that guard but received the other two A100s on the node;
the prospective physical-GPU UUID gate stopped it before imports or science.
Both failed roots are preserved with empty science logs and no capture
artifacts. Neither contributes a scientific result, weakens the 3/3 gate, or
permits the cross-engine operand join.

Corrected retry `12004190` is queued with the same prospective 3/3 science
gate, a new immutable root, and a scheduler begin time after the pinned GPU's
current allocation. It retains the exact physical-GPU UUID gate and must fail
before imports or science if that device is absent.

If and only if the RECOVAR admission passes, the multi-candidate component
substitutions compare shifted-image, projection/reference, score-weight, and
score-arithmetic boundaries after removing only the common
candidate-independent offset. The arithmetic arm evaluates RECOVAR's actual
JAX/XLA direct-Gaussian tree on the captured native RELION reference, shifted
image, correction weights, and `sum_init`, gathered through RECOVAR's
production full-grid-to-compact lookup with its explicit zero-gap lane
topology; this prevents XLA pointwise FMA or lookup rounding from being
misattributed to an input operand. Classification uses
only native candidates whose passive replay is bitwise equal to production;
the complete 96-candidate panel remains telemetry. This is target/class-local
causal evidence. It cannot change the fixed 41/60 K=4 scorecard or establish
all-class parity. The repository wrapper
`scripts/analyze_em_k4_admitted_fine_operands.py` enforces both admissions,
artifact hashes, same-physical-GPU identity, the 96-candidate scope, and
production-exact centered-score classification before producing that report.

The contribution-repeatability metrics remain deliberately distinct. The
historical fixed panel is immutable at 0/3, and the deterministic candidate
panel is already 3/3 but non-scoring pending its separate quality A/B. If job
`12004190` passes, it will be recorded as a third, current-source
production-path 3/3 panel. It will not rewrite either historical result or
change an FSC/FSC-AUC acceptance count.

## Mandatory telemetry conventions

Every staged capture must include immutable particle identity, base/physical
row coordinates, optics/noise groups, bucket and microbatch positions, exact
class/rotation/translation tuple keys, original device dtypes and byte hashes,
log-prior components, normalization constants, ordered support and threshold
margins, BPref operands and accumulator destinations, partial-reduction hashes,
and full signed shellwise FSC/controller arrays.

Each dump must state gather versus scatter direction, zero- versus one-based
indices, radians/degrees, pixels/Angstroms, translation sign, rotation
convention, x-half/Hermitian layout, conjugation, padding factor, and whether a
value is before or after float32 casting. Host float64 conversion is
supplemental and never replaces the original float32 bit pattern.

## Falsification and acceptance

- Dispatch order is not accepted unless a clean fixed-GPU A/B both verifies
  all aligned arrays and improves the first divergence without degrading any
  passing case. The current result is alignment 2/2 and rescue 0/2.
- Iteration 12 is not a root cause unless identical controller inputs produce
  different controller outputs.
- K=4 tuple differences localize candidate generation or first-iteration
  routing; raw-score differences localize scoring inputs/formulae; prior-only
  differences localize metadata/prior construction; normalization-only
  differences localize the joint log-sum-exp domain; support-only differences
  localize significance/tie semantics; operand-only differences localize
  BPref formula/destinations; reduction-only differences localize ordering,
  padding, scatter, or arithmetic.
- A code fix is accepted only after the first unequal boundary is demonstrated,
  the intervention closes or materially improves that boundary, all currently
  passing fixed cases are preserved, and signed normalized non-DC FSC-AUC and
  topology improve without relaxed thresholds.

`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` remains unset/default-off. Final all-data
is never forced after K-class nonconvergence. RELION `--firstiter_cc` keeps the
global best-coarse/winner-subset semantics, and current-size BPref joins retain
the explicit RELION padding factor.
