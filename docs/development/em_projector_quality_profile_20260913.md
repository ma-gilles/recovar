# Projector precision: paired regressions and profiling boundary

Frozen candidate `5e841111c116a07806ed153c922507cfceea9fce` restores the
production C64 projector consumer boundary on parent
`37faa499886c3a08742f4d1a6d4e5de0484e99bc`. Native host preparation and explicit
double diagnostics are unchanged. The separate half-staging optimization is
not included in these pairs. The EM-clean lead owns shared integration.

## K1 coverage limitation

H100 pair13829869 and CPU report13831400 completed successfully, one unchanged
three-iteration 5k/128 test per arm, no skips. Canonical halfmap AUC changes
+5.096e-9, merged cross-RELION AUC changes−1.706e-9, GT AUC changes+8.802e-8.
Support arrays are exact between arms; native mismatches1121/658/9 remain.
Cold wall1552.50 versus1547.76 seconds establishes no substantial speedup.

All six half-iterations in each arm have `supplied_ppref=False`: the global
zero-oversampling test does **not** exercise the native-projector repair.
This is regression coverage, not affected-path trajectory qualification.
[Canonical report](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_consumer_k1_report_20260913/result.json).

## Exactly-K4 affected-path result

H100 pair13832403 completed in7m03, each arm one unchanged 5k/128 strict-os1
three-iteration regression/no skips. Native PPref is supplied in all three
iterations. Captured native dispatch/perturbation replay is not autonomous
convergence. Cold process walls205.0/203.7 seconds are essentially unchanged.

| Class | Control / RELION FSC-AUC | Candidate / RELION FSC-AUC | Candidate minus control GT AUC |
| --- | ---: | ---: | ---: |
| 1 | 0.9905022642 | 0.9926882310 | −0.0000513780 |
| 2 | 0.9950925279 | 0.9951875016 | −0.0000084763 |
| 3 | 0.9966282233 | 0.9968341946 | −0.0001029207 |
| 4 | 0.9992095992 | 0.9992185012 | +0.0000076020 |

All cross-engine AUCs improve, but class1 remains below the0.995 gate. Every
candidate GT AUC exceeds the corresponding native value. Iteration2 support
mismatches improve222→0 and maximum Pmax error0.70146→0.000081924.
Iteration3 remains unresolved:424 support mismatches,72/5000 class mismatches
and maximum Pmax error0.66824. Neither full quality nor speed is admitted.

**Reporter correction:** the original report13832440 read RECOVAR MRCs raw,
omitting the transpose performed by `helpers.load_mrc`, while transforming
RELION correctly. Its cross-native/native-GT metrics are invalid and preserved
as failed evidence. The corrected report uses both canonical loaders; an
asymmetric MRC round-trip test explicitly catches the old error. Seven CPU
component checks and full regeneration passed (sessions53077/43100).
[Corrected curves, state and hashes](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_k4_report_20260913_v2/result_13832403.json),
[reporter and reproduction entry point](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_projector_k4_report_20260913_v2/report.py).
Report SHA256: `8f7112acf60361854ed562ac4d4b0b20d45fbbc09d6fdf33e8259ad425a3afa4`.
No scientific rerun or baseline edit was required. The K1 reporter was unaffected.

## Short-profile outcome and next measurement

The prepared real10073 one-particle Nsight run hit its175-second deadline before
completing any engine call (session39225, exit124, elapsed175.836s). All source
and library pins were unchanged, cleanup left no survivors, and no profile
timing is admitted. Its captured controller inputs exactly match the earlier
successful replay, including `mean_variance`. Native host projector preparation
took48.35s versus16.00s in the earlier run; this is not an isolated attribution
to Nsight, GPU arithmetic or a source regression.
[Preserved terminal receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_sparse_warm_profile_20260913/supervisor/result.json).

Next separate preparation from measurement: checkpoint actual dataset/backend
objects and engine operands once, then profile short warm calls without the
full controller startup. Private CPU round-trip checks preserve nested subset
identity, actual masked/unmasked full/half preprocessing, NumPy/JAX types and
dtypes; changed hashes and memmap payloads are rejected. These five checks pass
(92415); the real object graph and GPU placement still need validation.
[CPU checkpoint checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_input_checkpoint_20260913/cpu_result.json).
The checkpoint is private, trusted and source-pinned, not a public persistence
format. Actual preparation80700 subsequently completed in53.822s with zero E/M
calls. The1.43GB real checkpoint preserves all operand/raw/mask/processing/type
identities in a same-process round-trip. All source and library checks pass.

## Completed checkpoint-based warm trace

Profile31150 completed0 in135.491s on immediately idle A1001; no survivors or
changed pins. Three unchanged calls took60.303s cold,5.650s warm and3.618s
traced warm. Only the third was traced. All inputs, including mean_variance,
match the earlier successful real replay. GPU activity covers42.846% of the
traced interval (any recorded work, not occupancy).

The strongest substep target is projection staging:135 sparse projection
helper calls consume1.519s of nested host wall. Texture fill takes0.214s of
summed GPU time, versus0.033s for actual texture sampling. Nsight records308GB
of logical device-to-device copy bytes and36.7GB device-to-array; host-to-device
is0.54GB.142 stream-synchronize calls account for1.026s API wall. These times
overlap and must not be summed.

Source5e still expands the half projector to a cubic embedding at each static
projection block. The already-qualifiedb9/8dd half-staging patch addresses
this route; the next discriminator is its actual-input replay, not a new
implementation or full refinement. The lead's6118f0207 port is separate from
the frozen5e source measured here.

Warm/traced best scores, Pmax, poses and significant counts are byte-exact;35/41 fields match.
Six accumulation/noise fields differ. Against the original real replay,33/41
fields match; eight accumulation/noise fields differ. All arrays and differences
are retained; no full quality or general speed admission follows.
[Trace analysis](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_checkpoint_profile_20260913/analysis.json),
[output comparisons](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_checkpoint_profile_20260913/output_audit.json),
[terminal/source/library receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_checkpoint_profile_20260913/supervisor/result.json).

## Actual-input half-staging comparison

Fresh private candidate `32a636925fbd064cbd166377adfdf09a1f75397f` adds only
the existing b9/8dd staging package to frozen5e. The helper and staging tests
are byte-identical to8dd. No native source, binary, precision policy or scoring
formula changed.51 focused CPU tests,90 fast-guard cases and checkpoint/source/
command checks passed. Session38327 completed0 in110.215s, no survivors or
changed pins, on the same immediately idle physical A1001 UUID as control31150.
All three calls use the identical trusted actual-controller checkpoint; only
the third is traced. The frozen control and protected13785908 are unchanged.

| Measurement | Control5e | Half-staging32a | Direction |
| --- | ---: | ---: | --- |
| Traced call wall | 3.6182s | 2.7825s | Better,23.1% lower |
| Sparse projection helper,135 calls | 1.5194s | 0.8602s | Better |
| Logical device-to-device bytes | 308.02GB | 51.36GB | Better |
| Texture fill device time,140 calls | 0.2139s | 0.0484s | Better |
| Texture sampling device time,140 calls | 0.0330s | 0.0339s | Slightly worse |
| Rectangular fine-score device time,135 calls | 0.28943s | 0.28946s | Essentially same |
| Device-to-array bytes | 36.74GB | 36.74GB | Same |
| Stream synchronization API wall,142 calls | 1.0263s | 0.5864s | Better |

Cold/warm/traced calls were64.770/2.895/2.782s, versus60.303/5.650/3.618s
in the control. These separately launched, three-call arms are not a replicated
wall-time benchmark or a full-refinement speed ratio. GPU activity union falls
from1.550 to0.829s (42.8% to29.8% of wall), because unnecessary GPU work was
removed; this is not an occupancy measurement. Nested host/API/device times
overlap. Logical copy bytes are not measured DRAM traffic. Repeated texture
allocation, staging, synchronization and cleanup remain in the existing FFI.

The saved best scores, Pmax, poses and significant counts are byte-exact between arms.
However,26/41 output fields match, not all41: posterior totals and accumulation/
noise fields differ. Maximum rotation-posterior-sum delta is7.45e-9 and M-step
weight-total delta6.25e-8; scale-correction AA has maximum absolute delta1.928
and relative L2 delta1.04e-7. Within the candidate, warm/traced35/41 fields match,
with the six previously observed accumulation/noise differences. The extra
between-arm differences are retained and need first-operation attribution;
no bitwise, full-quality or general performance admission follows. Full support
masks and full score surfaces were not saved here. A measured
total-weight ratio does not explain all changed aggregates, so no correction
or fitted score offset is applied.

CPU analysis78419, output audit82374 and posterior audit99371 completed.
[Paired trace and output comparison](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_half_staging_profile_20260913/comparison.json),
[all saved-array differences](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_half_staging_profile_20260913/output_audit.json),
[posterior magnitudes](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_half_staging_profile_20260913/posterior_audit.json),
[terminal provenance](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_half_staging_profile_20260913/supervisor/result.json).
Comparison SHA256:`10b4dddd94de19211c6bdf843d971abf5b18c1f465244380b7c19b351050b342`.
Reproduce the read-only comparison with the frozen pixi Python and
`compare_profile.py` in that artifact root (exclusive-create report: use a fresh
output copy). `preparation.json` pins the exact bounded GPU command. Do not rerun
the consumed output root. Next isolate the new posterior deltas on the same
saved input before broader staging qualification or another speed change.

## Projection population and posterior-repeat attribution

Two bounded diagnostics narrow the extra posterior differences without changing
production code, masks, scoring, native libraries or frozen sources.

First, session63415 completed0 in15.861s (8.928s projection loop). It compares
the exact5e helper body extracted from Git against32a on the saved real C64
projector, rmax100,pf2, all36,864 coarse rotations at80 and294,912 fine rotations
at202. All6,197,280,768 complex outputs are **bitwise identical** across324
blocks/648 helper calls, including signed-zero checks. CPU preparation85643
tests actual helper wiring and bitwise/nonfinite detection; terminal coverage
audit60088 verifies every block and unchanged pins. This excludes changed
projector values at that geometry, not downstream scaling/fusion or scheduling.
[Population audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_projection_population_20260913/audit.json).

Second, session93955 completed0 in24.049s. One actual E-step prefix stops at
the first fine-posterior boundary, before downstream M-step work. The original
helper receives the exact same1,376,256 float32 scores three times; probabilities,
full fine masks, counts, denominator and threshold are bitwise identical.
Each returns8045 significant fine candidates and retained mass0.999000481368.
These fine reconstruction counts are not the controller's coarse-support count.
CPU observer check90096 covers actual helper calls, early stop, repeat bounds,
input mutation rejection and caller wiring; terminal audit21183 passes.
[Posterior-repeat audit and saved scores](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_posterior_repeat_20260913/audit.json).

Both use immediately idle A1001, sealed unchanged dependencies, bounded cleanup
and no surviving processes. The posterior result is a null finding over three
repeats, not proof of determinism under all scheduling conditions. It does not
retroactively identify the prior between-source delta. Source inspection shows
the measured rectangular fine scorer uses an explicit fixed reduction tree,
not unordered score atomics; the posterior uses the existing native float32
sort/scan. Next compare actual scorer operands and score surfaces between the
control and staging prefixes, including processed images and weights, to find
the first divergence. No fitted offsets, tolerances or production flags changed.

## Matched scorer-boundary terminal result

The follow-up prefixes on frozen control5e841111c and staging32a636925 both
completed: local sessions75493/89877, exit0, 40.270/43.266s. CPU wiring checks
11219/37767 and terminal posterior/pair audits completed. Same checkpoint,
raw inputs, sealed libraries, GPU UUID and effective environment were verified;
no changed pins or surviving owned processes. No new source or native changes.

All135 observed calls (45 raw,45 prior-adjusted,45 scored) have identical
operand fingerprints and byte-exact score outputs. The full1,376,256-score
posterior input and all five outputs—probabilities, support mask, count,
denominator and threshold—are byte-exact between arms. Each arm also repeats
the posterior three times exactly. Fine significant count is8045.
[Canonical pair comparison](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_real_score_boundary_candidate_20260913/pair_comparison.json)
has SHA256 `06d5b0f1784b6590dfefe96ad5b7f672f01499ebc0794acdcfc05f7917e055b6`.

This establishes value equivalence through this synchronized posterior boundary,
not the cause of the earlier tiny aggregate differences. The observer hashes
and copies operands, changing synchronization; only the first projection slab
is saved as an array, with later slabs fingerprinted. Neither original scheduling,
M-step equivalence, full refinement quality nor general speed is qualified here.
Do not use these capture wall times as a performance ratio or infer universal
scan determinism. Preserve earlier non-exact full-call outputs. Reproduction
uses the pinned `compare_pair.py` and `audit.py` in the candidate artifact root
with frozen pixi CPU Python; exclusive-create reports require a fresh output
copy. Consumed GPU roots must not be rerun.

## Pruning-score reuse: fewer calls, mixed timing evidence

Private82399613b on32a retains the existing budget-admitted prior-adjusted
scores for pruning instead of projecting/scoring again. No numerical kernel,
GEMM, preprocessing, projection slab cache or native library change. Existing
actual normalization/pruning arrays match byte-for-byte across45 chunks;
retention is5.25MiB. Expanded CPU tests pass9 cases and fast guard90, after the
parent fails the new redundant-call assertion. Unchanged parent RuffI001 remains.

Short A100 session16287 completed0 in110s with sealed source/input/library checks
and no survivors. Projection calls135→90, fine score kernel time.28946→.19324s,
device-to-array logical bytes36.74→24.93GB. Traced call2.7825→2.3268s is16.4%
lower, but untraced warm2.895→4.850s is higher: robust wall-time improvement is
not established. Separate processes, three calls each, third profiled. Next use
short repeated warm measurements without Nsight, not another full trajectory.

Saved best scores/Pmax/poses/significant counts match; only26/41 fields are
byte-exact between sources.15 aggregate/accumulation fields differ, including
posterior total6.25e-8 and rotation sums7.45e-9. Within candidate warm/traced,
35/41 fields match. Full support/margins and FSC are absent; preserve differences
and do not attribute them to harmless roundoff without evidence.
[Paired profile and full difference inventory](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_pruning_score_reuse_profile_20260913/comparison.json),
[reproduction, source, tests and remaining gates](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/em_pruning_score_reuse_terminal_20260913.md).
