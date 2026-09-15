# Reusable accuracy and performance benchmarks

A cleanup comparison measures the candidate against an unchanged RECOVAR
control. Report standing against RELION and synthetic ground truth separately.
Historical results from another commit remain historical evidence until their
workload is reproduced at the intended source.

## Keep definitions, runs and reports separate

Maintain three independently identified artifacts:

1. **Fixture definition:** immutable particle/metadata/map/mask inputs, exact
   generation or acquisition recipe, seeds and complete content hashes.
2. **Run record:** immutable source/build/environment identity, exact command,
   output root, job/process status, all measurements and artifact hashes.
3. **Comparison report:** derived from admitted run records and unchanged
   acceptance policy. Regenerate it without rewriting fixtures or past results.

Admission checks require every declared case/stage/half/class and unique input
identity. Reject missing or nonfinite required measurements, duplicate records,
wrong units, stale ledgers and incomplete outputs. Preserve the original failure
and its log. Distinguish execution failure, invalid evidence, quality failure,
performance regression and not measured; report generation success is not a
scientific pass. Expected counts come from the workload definition, not the
results that happened to be produced.

## Workload coverage

| Workload | Preserve and measure |
| --- | --- |
| Synthetic K1 | Fixed 34-case inventory, including known failures; small cases first, then robustness and completion trajectories |
| Synthetic K4 | Exactly four classes, Hungarian matching, every class and iteration, worst class, occupancy/collapse and assignment behavior; required 100k/256 completion pairs |
| Synthetic K2/K8/K16 | Existing three-seed stress cells, including preferred orientations and unequal class balance; retain per-class results |
| Real K1 | Fixed EMPIAR-10076 cohort, halfsets, masks, metadata and initialization; autonomous trajectory plus separately labeled fixed-state diagnostics |
| Real K>1 | Dense Class3D and K4 VDAM are separate workflows, each with matched per-class reference results and repeat/seed controls |
| Shared covariance/PPCA pipeline | SPA, cryo-ET, outliers and downstream tests, plus canonical real paper datasets from the [Della runbook](della.md) |

K4 is required; another K does not substitute for it. PPCA's latent dimension
is not class count. Dense Class3D's duplicated half-map outputs are not
independent-half FSC evidence. Real reference agreement measures consistency,
not absolute ground-truth accuracy.

Record whether a run is fixed-state replay, dispatch/perturbation replay,
fresh autonomous refinement, serialized continuation, Class3D or VDAM. State
which particle/noise/prior/controller values came from the oracle at each
iteration. A replay can diagnose arithmetic without qualifying autonomous
convergence. Capture-enabled and ordinary execution may use different paths;
qualify instrumentation effects before interpreting interventions or timings.

## Numerical contract

Production EM qualification uses float32. Double-precision EM runs are
diagnostic references for identifying arithmetic or implementation errors;
they cannot replace the final K1/K4 quality and performance evidence. Record
effective scoring, projection, accumulator and numerical M-step precision in
every run. Distinguish configured precision from captured operand/output dtypes
and explicit computation casts. Float32 scoring with an existing float64/complex128
M-step is a mixed-precision trajectory, not an all-float32 qualification. A
float32-only patch does not establish the precision of untouched stages. Match
the input state and candidates before comparing precisions, and do not infer
numerical noise merely because a discrepancy shrinks in double. Preserve
intentional higher-precision host/metadata operations; this is not a blanket
array-narrowing policy. See the mandatory
[EM precision rule](../../recovar/em/AGENTS.md).

Retain the existing metric directions, tolerances and quantitative EM gates.
Map quality uses FSC curves, FSC-AUC and established FSC summaries against GT
and RELION. Map correlation is diagnostic and cannot override those gates.
For K4, use Hungarian matching and show all class results; never accept a poor
class because the mean passes. Keep initial/per-iteration/final products and
convergence/finalization identities explicit. See [EM numeric rules](../../recovar/em/AGENTS.md)
and [the program gates](../math/em_parity_program.md).

Preserve casts, FFT frames, half-spectrum support, normalization, reduction
order and candidate domains when comparing implementations. Match full state,
including noise, scales, priors, posterior normalizers, accumulators and control
state, before attributing a trajectory difference to a kernel. Diagnose the
first divergent boundary and keep independent analytic/float64 references.

## Performance contract

Run repeated control/candidate pairs sequentially on the same physical GPU,
with identical inputs, initialization, seeds, masks, batches, memory budgets,
thread/MPI layout and analysis settings. Retain all samples and report measured
variability. Record queue wait separately from compute time.

Separate cold compilation, warmed device compute, data preparation and complete
process wall time. Synchronize pending device work at timing boundaries when
needed. Per-stage durations can overlap; do not sum overlapping stages and call
the result end-to-end time. Reuse must retain original generation/compute time.

Measure GPU memory for the relevant process tree and assigned devices, plus
host RSS, with sampling interval and method recorded. Whole-device memory can
include other users; endpoint RSS is not a continuously sampled stage peak.
Zero/missing/unavailable memory readings do not establish zero memory use.
Use UUID-aware device selection; an empty visibility mask means no visible GPU.

Keep the established 10% regression warning policy. Do not waive an accuracy
failure because performance improved, or claim equivalent speed from unmatched
hardware, dirty mixed revisions, cached-only runs or one historical sample.

## Minimum run record

- Full RECOVAR commit, branch, diff SHA256 and untracked source manifest; frozen
  lock hash, imported RECOVAR/JAX paths and versions.
- Actual CUDA library path/hash, source/header/compiler identity before/after;
  RELION source, patches, binary, command and metadata where used.
- Resolved and hashed inputs, particle IDs/order, half/class assignments, seed,
  grid/voxel size, CTF/noise/contrast, masks and all effective overrides.
- Exact commands, run mode, requested/executed cells, Slurm IDs, node, physical
  GPU UUID/model, process exits, full logs, output paths and artifact hashes.
- Quality values and baseline deltas per required class/half/iteration;
  timing/memory samples with compilation, warmup and sampling methodology.
- Explicit acceptance policy identity and separate execution, evidence,
  quality and performance outcomes. No missing cell is silently dropped.

Bulky disposable outputs belong in scratch with a `SAFE_TO_DELETE` marker.
Keep enough small evidence to reproduce and audit a result after scratch cleanup.

## Recorded source comparisons

The [K1 case25 archive from 7 September 2026](evidence/k1-case25-20260907/README.md)
preserves a measured control/candidate pair, commands, fixture identities and
all shellwise FSC curves. It is a scoped historical run record, not an expected
baseline or qualification of the current checkout.

The [shared SPA/ET checkpoint](evidence/shared-spa-et-20260907/README.md) preserves
historical quality/performance comparisons, generated fixture identities and
six shellwise FSC curves recovered from saved outputs. Its threshold-frequency
summaries saturate; use the archived curves when reviewing map-quality changes.

The [VDAM coarse-repeat audit from 9 September 2026](evidence/vdam-coarse-repeat-20260909/README.md)
preserves independent CPU comparisons of 16 saved float32 score outputs, their
hashes and the exact audit script. Same-input raw winners are stable despite
score variation; this scoped diagnostic does not qualify trajectory decisions,
M-step precision or performance.

The [frozen VDAM full201 cell at4f9](evidence/vdam-full200-4f9-20260909/README.md)
preserves all1,005 shellwise curves/AUCs, commands, source/input/build pins, timing
boundaries and a self-contained read-only audit. Both map conditions pass at all201
checkpoints on one3k/128 K1 natural200 H100 pair; strict state parity still fails.
The1.4883× wall ratio is one measured pair, not representative speed qualification.
This is historical evidence, not a baseline replacement or later-tip acceptance.

The [frozen canonical-pixel real-prefix cell at5ca9](evidence/vdam-canonical-pixel-prefix20-20260910/README.md)
now has integrator-verified source/input pins,21 exact FSC integrals, a worst-cell
raw-map recheck and initial-map byte equality. All0–20 cross-map conditions pass
on real10076 10k/256 K1; count/Pmax differences remain. This closes the missing
integrated-repair prefix measurement, not later-tip or completion qualification.

## K4 audit integrity and reviewed saved comparisons — September 10

The trajectory audit now requires identical, unique particle-ID sets and finite
integer class labels in0..K. Missing rows, duplicate IDs and out-of-range labels
fail with `AuditError`; zero remains the established unassigned sentinel. Valid
rows may appear in different orders. The common-assigned accuracy policy and
all numerical thresholds are unchanged.26 focused audit/runner tests pass;
80 original synthetic/real STAR files retain exactly their prior audit results.

The integrator reintegrated all3,200 saved shell curves from10 original/repeat
comparisons, verified their4×4 score matrices and optimal class assignments, and
recomputed the existing per-checkpoint pass/fail decisions. These are frozen5ca9
InitialModel20-iteration diagnostics, synthetic5k/128 and real10k/256, not the
required100k/256 completion workloads. Full run/source/build admission and raw-map
recomputation are not claimed by this saved-curve audit.

| Saved comparison | Minimum matched FSC-AUC | Minimum assignment accuracy | Fixed-gate failure iterations |
| --- | ---: | ---: | --- |
| Synthetic candidate / original native |0.9976801201|0.9986|19–20|
| Synthetic candidate / native repeat A |0.9999999943|1.0|None|
| Real candidate / original native |0.6489987975|0.9638|14–20|
| Real candidate / native repeat B |0.9983333846|0.9958|20|

Some native/native pairs also fail the same gates. Reproducing a branch is
useful diagnostic evidence, but does not change the fixed thresholds or prove
all competing-score/state differences are numerical. Preserve both original
failures and repeat comparisons; the peer's “closed at repeat-band level”
wording is not an integrator acceptance decision. Large reported K-class runtime
ratios also remain an open performance concern, not a qualified comparison here.

[All comparisons and pinned saved artifacts](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_audit_identity_guard_20260910/saved_review.json);
[reproducible CPU review](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_audit_identity_guard_20260910/review_saved.py);
[red/green test commands, source hashes and limits](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_audit_identity_guard_20260910/result.json).

## Robustness GT-curve review — September 10

The10-case synthetic K1 InitialModel matrix reports frozen5ca9, 128²,
natural200, seed29. Its original GT reports contain independently aligned
FSC curves at100 and200, not one prespecified shared rigid transform across
all201 checkpoints. The integrator recomputed all40 original GT integrals and
24 repeat integrals from these saved curves. Original40 values match exactly;
registration, raw GT curves and full source/build closure were not requalified.
Treat these as screening measurements, not full registered-GT admission.

| Original-control cell | Candidate minus native GT FSC-AUC |
| --- | ---: |
| Case13, iteration100 |−0.0022288483325973926|
| Case22, iteration200 |−0.01213269735834411|
| Case32, iteration200 |−0.008132623310265132|

These three cells fall below the existing−0.002 screening threshold. Do not
replace that threshold with the peer summary's±0.003 “equal quality” interval,
or use correlation to declare other cases closed. Other sampled cells are
above−0.002; unsampled GT checkpoints remain unmeasured. All10 reported
cross-engine AUC series include values below0.999; repeat variability alone
is not a waiver of their original controls.

For case22 final maps, the six saved native GT AUCs span
0.06039478833224346–0.07195972826887516. The two candidate AUCs are
0.05973366118166213 and0.05973931223337803. Eight of12 cross-engine pairings
fall below−0.002, four do not. Correlation values0.280 versus0.286–0.304
must not be substituted for these FSC-based comparisons.

The claim of bit-identical candidate repeats is contradicted by the two final
MRC arrays:97,376 of128³ voxels differ, maximum absolute difference
0.027517318725585938. This is a byte/value comparison, not a map-quality gate
or an attribution to numerical noise. The reported case22 fixed-state35→36
replay also retains coarse support10 versus7, so “all discrete outputs exact”
is not established. Neither these summaries nor double contraction justifies
a deterministic-accumulation rewrite during structural cleanup.

[All full-precision values and input hashes](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/robustness_fsc_review_20260910/result.json);
[reproducible CPU audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/robustness_fsc_review_20260910/review.py);
[execution/source receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/robustness_fsc_review_20260910/verification/receipt.json).
Next scientific review requires the actual first-divergence score/support
records and registration/source admission; no kernel change or rerun is implied.
