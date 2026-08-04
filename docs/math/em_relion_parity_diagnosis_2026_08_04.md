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
- K=4 native auxiliary-stream repeatability gates: **12 / 13**.
- K=4 stable native operand-capture admission: **0 / 1**.
- Fresh K=1 dispatch alignment: **2 / 2** cases verified.
- Fresh K=1 dispatch standalone rescue: **0 / 2** cases.

The K=1 strict failures remain cases 04, 05, 07, 10, 22, and 26. Cases 07
and 22 are the topology failures. The K=4 failures are the late iterations
10--15.

## Current causal decisions

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

The existing fixed-target operand replay closes exactly while the complete
active raw table does not. Current prior mismatches are inert for the observed
support and are routed downstream rather than treated as the causal boundary.

The fixed RECOVAR iteration-2 boundary for stack identity 53723 is now complete
for all four classes. Its four artifacts preserve 247,232 active class-pose
tuples, 66,986 significant tuples, and a joint probability mass of
`0.9999999999999997`; an independent fixed-order host replay agrees with the
stored joint probabilities to maximum absolute error
`4.336808689942018e-19`. This accepts only the RECOVAR side of the join. Native
RELION capture must come from the same accepted observer lineage before tuple,
raw-score, prior, normalization, support, or BPref comparisons are causal.
The next all-class capture is intended to expose the first unequal global
class-pose normalization/support boundary. Before direct class FSC is given a
causal interpretation, a signed 4-by-4 class FSC-AUC matrix should diagnose
class permutation separately from map-content error; a permutation rescue is
diagnostic and does not satisfy a fixed-label scorecard.

Native RELION observer repeatability is an admission gate for this comparison.
Stream-replay changes are observer controls, not RECOVAR parity fixes. The
corrected same-A100 auxiliary-stream pair now rejects exact capture
repeatability at 12/13 fixed gates. Dispatch, tuple identities, priors, target
state, topology, capture validators, marker counts, and all four class-map
signed FSC-AUC gates pass, but fine-score, fine-operand, and geometry-only
BPref bytes differ.

The first unequal captured level is preprocessing rather than tuple generation
or priors. For the fixed class-1 target, preprocessed image samples and
`sum_init` differ while reference pixels and tuple identities are exact. This
propagates to 16,735/109,184 raw fine `diff2` rows, posterior weights, and BPref
`weight_norm`. The default-off deterministic soft-mask block-partial experiment
is therefore activated. It preserves the deployed 128-block by 128-lane work
decomposition and replaces only schedule-dependent inter-block lane atomics
with increasing-block float32 finalization. It remains an observer control,
not a RECOVAR parity fix, and must itself pass exact native repeatability and
capture-inertness before cross-engine attribution.

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
