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
- K=4 native auxiliary-stream repeatability gates: **12 / 13** (historical
  predecessor panel).
- K=4 native soft-mask-partial repeatability gates: **13 / 14**.
- K=4 native high-resolution-Xi2 observer repeatability gates: **15 / 15**.
- K=4 stable native operand-capture admission: **1 / 1**; this is an observer
  admission result, not cross-engine posterior/BPref/map parity.
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
| K=4 class-1 tuples and significant support agree at the accepted observer boundary | Demonstrated provisionally | Exact `109184/109184` tuples and `38982/38982` support rows in the current one-class join |
| K=4 raw `diff2` is the first unequal measured class-1 boundary | Demonstrated provisionally | 25,687 float32 mismatches, one-ULP median and three-ULP maximum; all-class admission and RECOVAR repeatability are still pending |
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

The first unequal boundary after the pending native all-class admission and
RECOVAR repeatability controls, rather than this provisional ordering, chooses
the implementation target.

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
a discrepancy causes the late map failures. It remains provisional until the
four native class captures and their treatment/capture-inertness admission
pass.

The provisional mismatch is distributed rather than confined to one tuple:
25,687/109,184 active candidates differ (23.5%), with 18,766 positive and
6,921 negative RECOVAR-minus-RELION deltas, signed sum `0.628997802734375`,
and L1 sum `1.267913818359375`. The ten largest rotation strata account for
only 1.06% of mismatch L1, whereas the ten largest translation strata account
for 23.9%. The predeclared representative-selection rule chooses native
rotation-local 1790, mapped RECOVAR rotation row 2878, translation 28: native
raw `diff2` `512.4172973632812` versus RECOVAR `512.4174194335938`, a
two-ULP delta. This representative supersedes a simple first-mismatch target
for a bounded operand capture if the pending admission and repeatability gates
accept the all-class attribution.

The fixed RECOVAR iteration-2 boundary for stack identity 53723 is now complete
for all four classes. Its four artifacts preserve 247,232 active class-pose
tuples, 66,986 significant tuples, and a joint probability mass of
`0.9999999999999997`; an independent fixed-order host replay agrees with the
stored joint probabilities to maximum absolute error
`4.336808689942018e-19`. This accepts only the RECOVAR side of the join. Native
RELION capture must come from the same accepted observer lineage before tuple,
raw-score, prior, normalization, support, or BPref comparisons are causal.

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
a different physical A100; it motivates the pending current-source exact-GPU
repeatability gate.

The relevant source change after `db1ab501` is known: `e0c51765` introduced a
deterministic soft-mask background reduction and `8aa573f5` parallelized it
without restoring schedule-dependent atomics. The previous 128-lane atomic
background accumulation was replaced by fixed per-block reductions followed by
a deterministic second-stage sum. Therefore the old shifted-image instability
is not presumed to survive in current source `223e7e81`; job `11994138` is the
direct independent-process regression test for that claim.

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
