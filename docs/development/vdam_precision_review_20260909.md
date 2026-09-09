# VDAM precision investigation — September 9, 2026

This dated evidence record preserves the reviewed private `907b02ce` /
`fe8472947` investigation. It does not qualify newer source or authorize
precision adoption. The [current EM status](em_status.md) owns the next check
and current integration decisions. Relative scratch paths below use
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/`; source-review paths use
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/`.

VDAM's separate two-path full-float32-product candidate `907b02ce` is now frozen
and reviewed: exactly two `HIGHEST` contraction keywords differ in production;
all 32 recorded H100 cases pass (13638400), after two expected-red cases
(13638324). Native-cutoff candidate replay 13638431 is diagnostic only. See
`handoffs/em_clean_vdam_full_float32_review_20260909.json`. Prepared-state jobs
13638710/13638814 on that private source now preserve complete incoming/candidate
inputs and particle decisions/Pmax. Noise/BPref arrays vary even within a policy;
the first panel's one-ULP support-sum policy association did not reproduce in the
second. Only the rectangle-power helper executes in this replay; the atomic
fallback has focused-test coverage only. These are not full-state equivalence,
trajectory or runtime acceptance.

VDAM's private composition `fe8472947` carries the same `907b02ce` patch atop
compact-CTF merge `5a39eab29`; only two documentation files separate that parent
from the assigned `cbff0b092` base. The candidate applies cleanly to the current
primary, but is **not adopted**. H100 13639506 completed four natural 200-iteration
old/new/new/old trajectories (0:0, 1,819 s). Whole-process old times are
449.269/454.861 s and new times 456.089/457.051 s: **1.009965×** by mean,
a small-fixture diagnostic only. The completed 804-file/201-checkpoint metadata
ledger is schema-valid but fails numeric comparison at 199 iterations (2–200).
Old-policy repeats first differ in pose at 32 and resolution shell at 45;
new-policy repeats first differ in pose at 52 and keep the same resolution shell.
Their winning margins and convergence implications remain unresolved.

The completed producer FSC review reports old-repeat minimum cross-FSC-AUC
0.969668 at iteration 155, versus 0.998219 for new repeats at 86. The two paired
old/new minima are 0.969722 and 0.998219. These are cross-map diagnostics.
The registered-GT follow-up below now covers these checkpoints; historical
native source-to-binary closure remains missing.
The candidate cross-native minima span 0.998196–0.998767 at iteration 86,
versus a native-repeat minimum of 0.9997155. The second old-policy arm is closer
to both native endpoints than either candidate arm: no uniform candidate win.
The producer reports exact independent agreement for all 15 terminal AUCs and
shell curves; em_clean's review here is the pinned report, not a fresh curve
recomputation. See `handoffs/vdam_full201_fsc_terminal_20260909.json` and
`vdam_f32_full200_20260909/fsc_trajectory_figure/fsc_trajectories.png` in scratch.
em_clean verified the review's seven artifact hashes, without independently
recomputing its arrays. No full-state, trajectory or runtime acceptance follows.
See `vdam_f32_full200_20260909/analysis_integrated_v2_review/result_summary.json`
under the scratch artifact root; review receipt is
`dense_firstiter_arguments_20260909/full200_peer_report_review.json` under the
source-review root below.

Prepared E/M job 13639984 failed in diagnostic callable serialization before
science E/M completion; the failed evidence remains. VDAM's artifact-only v2
repair completed as H100 13640121 (0:0, 145 s) on frozen fe847. Report review
verified five artifact hashes and four completion receipts, without independently
recomputing the numeric array comparisons. It reports exact incoming/compact-CTF
operands and local decisions, but coarse scores vary up to 1.220703125e-4,
including same-policy repeats. Accumulators and six post-state arrays vary;
crossed maxima are not uniformly bounded by the two same-policy comparisons.
Single-boundary cross-map FSC near one does not establish trajectory quality.

Follow-up raw FFI job **13640613** completed on frozen fe847. Independent CPU
recomputation of all 16 saved `(200,576,29)` float32 outputs verifies variation
in all 15 comparisons with repeat 0: maximum 1.220703125e-4, p95 3.0517578125e-5.
All 200 pre-prior raw winners remain exact across 16 runs; the minimum represented
score margin is 0.0013885498046875. Thirteen manifest pins, 17 native source
inputs in both checkouts, the library hash and terminal receipt were checked.
The frozen harness compiles the raw scorer once and uses distinct retained
output buffers; it reports exact negation/max controls on one fixed output.

The captured route is the shared-pretranslated direct float32 FFI, whose CUDA
source merges lane sums with atomics. Recompilation and the later Wavg product
intervention are not necessary for this observed variation. Actual atomic order
was not recorded, and these are raw pre-prior margins, not posterior margins or
evidence for later trajectory flips. No arithmetic change or quality acceptance
follows. The [durable audit archive](evidence/vdam-coarse-repeat-20260909/README.md) preserves
metrics, hashes, the exact CPU audit script and its reproduction command. Original
outputs are under `vdam_coarse_atomic_repeat_20260909` in scratch.

Paired-history H100 job **13641091** completed two fresh old-policy prefixes on
frozen fe847, through M31 and ordinary E32 plus three prepared repeats, stopping
before M32. Both choose rotation111738 for particle1367; signed competing margins
(best163798 minus best111738) are −0.0009765625 and −0.000244140625. Full target
candidate geometry/order agrees (3,360 cells, 256 finite). Target debug payloads
are exact within each history, and all 200 published decisions/Pmax agree across
ordinary/clean/debug calls. Whole E outputs still vary in 17 accumulator/noise
leaves. Incoming model, momentum, noise and priors differ before target preparation;
the cross-history score gap0.00115966796875 is not a same-input arithmetic bound.
Neither fresh history reproduces the original F200 pose flip, whose saved inputs
are unavailable. Reconstructed raw scores are not an independent arithmetic trace.

The bounded producer RNG audit found no scientific consumer of the differing
Python/legacy NumPy globals under these options; target preparation/E32 preserves
them within each history. Native RNG state and independent generators are outside
those snapshots. No global-seeding patch or automatic tie classification follows.
em_clean verified 11 report/manifest hashes, two terminal child receipts and
summary margin arithmetic; it did not independently recompute snapshot arrays or
repeat the full RNG reachability audit. See board handoff
`vdam_paired_history_terminal_20260909.json` and review/reproduction script under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/paired_history_review_20260909/`.
That prefix audit is now complete: all 64 metadata files for checkpoints 0–31
have matching schemas. Initial stored map arrays are exact; small accumulator,
noise and offset-sum differences appear at iteration 1, followed by support mass
at 2. Significant counts first differ at 18 for particle 1290 (36 versus 35);
no saved pose change occurs through 31. Full incoming E1 operands were not saved,
so initial-map equality does not establish identical E1 inputs or identify the
first divergent operation. The support-count tie remains unclassified.

The producer's momentum audit reports matching update formulas and no near-zero
square-root denominator (minimum 0.98634); the ten largest Fourier-cell differences
account for 98.68% of squared second-moment difference. Neither observation proves
the cause or its effect on poses. em_clean checked both pinned reports and the
prefix CPU completion receipt, without recomputing arrays or independently auditing
the full momentum formula. See `handoffs/vdam_prefix_first_divergence_20260909.json`
and `pass2_raw_schema_20260909/prefix_review.json` under the source-review root.
The registered-GT v2 diagnostic is complete: four planted controls plus native/null
optimizer checks pass. Fitting on native_1 final only, translation raises its
GT-AUC from 0.126986 to 0.388388 at fixed rotation. The same frozen transform was
then applied to all six histories and 201 checkpoints: 1,206 existing maps, no
refits or new scientific/GPU runs. The producer reports 443.88 CPU seconds, stable
source/input pins and all six terminal curves exact against the final-map pilot.
Registration controls validate this artifact, not a globally optimal or unique
frame; a final-map transform can miss earlier frame motion.

em_clean independently recomputed all 2,412 all-shell/held-out AUC values from the
saved curves (maximum discrepancy 1.12e-16) and verified all eight RECOVAR/native
trajectory summaries plus seven artifact hashes. This does not repeat fitting,
FFT computation or full source/native/MRC provenance closure. The unchanged
all-shell GT deficit condition is −0.002: old1 first fails at 155, reaching
−0.004248842 versus native1; it fails at 42 checkpoints versus native1 and 30
versus native2. New1, new2 and old2 remain within that condition at all 201
checkpoints against both native repeats. Their minima versus native1 are
−0.000508038, −0.001345653 and −0.000507449 respectively. Terminal deficits are
−0.000294190, −0.001214108 and −0.000312159; old1 ends at −0.002285102.
There is no uniform policy win across these two repeats.

Held-out shells 9–62 are descriptive: the immutable analyzer incorrectly attached
the all-shell threshold to them and native-repeat summaries. The preserved raw
report is superseded by `report_scoped.json`, which removes only those labels.
No scientific threshold changed. Review/reproduction script and receipt:
`pr179_current_api_20260909/gt_full201_review.{py,json}` under the source-review
root; board handoffs `vdam_registered_GT_terminal_20260909.json` and
`vdam_registered_GT_full201_20260909.json`. Strict accumulator/support/pose,
cross-engine FSC, convergence, robust K1/K4 and current-source runtime gates
remain open. Precision907 stays private; no production/native source change is
assigned.

The reviewed runs use **float32 scoring with an existing double-precision
M-step**. Effective precision is stage-specific:

| Stage in the reviewed composition | Effective execution | Evidence/scope |
| --- | --- | --- |
| Captured coarse scoring/projector operands | float32/complex64 | Captured-path evidence, not a claim about every intermediate or route |
| Candidate Wavg products | Two `HIGHEST` contraction keywords on float32 products | Same output dtypes; no M-step precision change |
| JAX VDAM M-step | float64/complex128 numerical computation in both arms | Explicit device casts and host call in [relion_vdam_mstep.py](../../recovar/em/dense_single_volume/helpers/relion_vdam_mstep.py); actual M executed in 13640121 |

The M helper is byte-identical in shared source and frozen fe847 (SHA-256
`0ba75b68202380372e0ffbf777e23bf5b1a82145437723a519429b7507524f2c`).
This is more than metadata precision and is not complete float32 M qualification.
No existing arithmetic is changed or newly accepted by this reporting correction;
the intended production-float32 goal remains. See board handoff
`vdam_effective_precision_boundary_20260909.json` and review receipt
`effective_precision_reporting_20260909/review.json` under the source-review root.
Warm prepared E is 268.953 → 284.035 ms,
**1.056075× (+5.61%)** in this small preloaded-data panel; no full-runtime
acceptance. Only rectangle power executes; full local score surfaces are absent.
VDAM's completed registered-GT diagnostic is reviewed above; both Wavg
helpers remain fixed. Preserve fe847 and its original evidence; no shared
precision adoption, new arithmetic change or duplicate GPU job. See
`handoffs/em_clean_prepared_em_composition_review_20260909.json` and the producer
`vdam_wavg_composition_v2_20260909/{RESULTS.md,result_summary.json}` under the
scratch artifact root. These runs precede newer structural cleanup and do not
qualify the current primary.

The separate907 correctness review now explicitly incorporates the completed
registered-GT trajectory measurement through board addendum
`handoffs/em_clean_precision907_registered_gt_review_20260909.json`.
This closes the missing registered-GT measurement for the reviewed fixture:
both candidates and old2 satisfy the unchanged all-shell delta condition at every
checkpoint against both native repeats; old1 first fails at155 and reaches
−0.004248842. It does not close strict cross-FSC, state/support/pose, convergence,
source/native provenance, broader K1/K4 quality or representative runtime gates.
No uniform policy win or adoption follows. The independent saved-curve audit
and its exact scope remain recorded above; no new fit, FFT or GPU run was made.
