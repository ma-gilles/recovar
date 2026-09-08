# Current EM development scope

The current milestone is behavior-preserving cleanup and development setup
across RECOVAR, with EM prioritized. GUI/frontend work and the new HIA engine
are outside this milestone. EM public APIs may change when they simplify the
code; migrate callers, tests and documentation together. Preserve non-EM APIs,
saved formats, numerical behavior and scientific defaults. Present numerical
repairs separately for a user decision.

The user clarified on 2026-09-08 that production EM remains float32. Double
precision is a diagnostic reference for separating numerical roundoff from
implementation bugs; it is not a proposed production solution. The mandatory
[precision policy](../../recovar/em/AGENTS.md)
requires final quality and performance qualification in float32, including K1
and exactly K4. A discrepancy shrinking in double alone does not establish
that it is numerical noise. PR180's double-precision evidence is interpreted
under this rule.

On 2026-09-08 the user requested integration of
[PR180](https://github.com/ma-gilles/recovar/pull/180), including its cleanup.
The full PR is integrated locally at
`42a3d6184c6d05a9f4f97bd00120e62d0081f1d3` on `codex/integrate-pr180`, with its
numerical changes tracked separately from the structural series. It has not
been pushed. The preserved cleanup
branch ends at `681c2e6ed03c9b63b94f07bc47a4a8367b1b4de3`; the pinned incoming
head is `1e2f229b3e0e8edaec029d2b604937f692148578`. Both descend from the PR158
control below. Earlier unrelated runtime/validation repair proposals remain
separate. Frozen benchmark checkouts and their results are unchanged.

PR180 introduces grouped refinement options, iteration history, precision
propagation through shared data/CTF and EM paths, CUDA arithmetic changes, and
replay-cutoff ownership changes. The integrated source therefore needs fresh
scientific qualification; historical structural-equivalence evidence does not
qualify these numerical changes. Integration details and failed/retried checks
are recorded under `pr180_integration_20260908/` in the review root below.

The integration has passed 180 focused helper/controller cases, 208 shared
precision/API/K-class cases, and the 38-case CPU fast guard. A separate
controller selection initially passed 44 of 45 cases; removing a stale unused
local-scoring keyword fixes the remaining finalization case, which passes
alongside the three explicit K1/K2/K4 dispatch cases. Counts overlap. Eleven
helper-body comparisons and the normalized main-loop comparison preserve
PR180's computations across the retained cleanup interfaces. The separate CPU
reference binding supplies PR180's added orientation interface; the historical
benchmark binding is unchanged. These checks do not establish GPU or
end-to-end quality/performance qualification.

The additional precision-interface check passes 116 cases and skips 17
GPU-only cases. All 6,818 non-GUI unit cases collect; collection is not
execution. The previously failing
`test_fused_sparse_k_class_pass2_matches_existing_two_pass_path` passes on the
integration candidate: PR180 supplies the missing spectrum-normalization
setting on that path. The earlier frozen CPU result below remains unchanged.

The merged source now passes all 16 selected GPU kernel cases without skips
on an H100 80 GB (Slurm13623670, `della-h20g2`). These cover native scoring,
translation, rotation and backprojection, including double-precision scoring
and translation. The run compiled an exclusive CUDA 12.8 library against the
frozen checkout's JAX headers, then loaded a read-only copy and verified its
path and SHA-256 in the test process before and after execution. The package
inventory matched the frozen environment before execution; source,
reference-library and CUDA-library identities remained unchanged afterward.
This is kernel evidence; it does not establish K1/K4 trajectory quality or
performance. Exact commands and results are under
`pr180_integration_20260908/gpu_kernels/` in the review root.

Broad CPU job Slurm13623235 completed on frozen `42a3d6184` in 53 minutes
(`della-h14n4`): 6,452 passed, 340 skipped and 25 failed. Source identities
remained unchanged. The 6,817-case selection retains the previous GUI and
shared-`/tmp` exclusions. The raw-ID comparison matches the reviewed inventory
(123 added, 19 removed); no failure is waived. Relative to the prior CPU
checkpoint, 14 failures remain, the K-class undefined-variable case now passes,
and 11 cases change from passing to failing. Eight are local-search test stubs
rejecting PR180's `dtype` keyword; one is the stale controller metadata guard
fixed below. The other two were a sampling-grid fixture/ownership mismatch and
a norm-reduction dtype expectation. Focused follow-ups for all 25 failures are
recorded below; the original CPU result is preserved.

Exact results and the raw comparison are in
`pr180_integration_20260908/cpu_checkpoint/` under the output root, with sealed
commands and source manifest at the corresponding path in the review root.
The sampling helper now uses its direct sampling dependency, with the
controller import removed. The two grid-precision cases and eight local-search
cases pass after migrating dtype stubs and removing a redundant patch that
overwrote the replay fixture with the real reader. Scientific assertions are
unchanged. Direct grid comparisons preserve both precisions at orders 0–3;
calling the helper leaves the controller unloaded. The explicit diagnostic
norm-reduction test now matches the contract
established by PR180 `a91bce65a`: float64 output retains low bits that a final
float32 cast loses. Both input precisions are covered, and a separate default
case verifies float32 output for complex64 inputs. All nine norm/capture cases
and two existing scoring/M-step default-precision guards pass. Runtime code and
tolerances are unchanged. Evidence for both migrations is under
`pr180_sampling_contracts_20260908/` in the review root.

All 11 newly failing CPU cases have now been addressed in focused checks.
The 14 inherited failures were checked separately on the same frozen source.
The eight tiny GPU-dependent cases pass on frozen `42a3d6184` in
Slurm13625819 (H100 80 GB, `della-h20g2`, 15 seconds), with no skips and source
and native-library identities unchanged. Their original CPU failures remain
recorded. The five dry-run tool checks and the float32 comparison also pass on the
same H100 in Slurm13625952, with the pinned real RELION executable supplied
explicitly. The binary is only resolved/hashed by the dry-run checks; no RELION
benchmark is launched. Source and native-library identities remain unchanged.

The [versioned CPU-failure reconciliation](evidence/pr180-unit-checkpoint-20260908/README.md)
matches all 25 original failures to these focused checks: 24 retain their case
IDs, and the diagnostic norm case has two explicitly recorded precision
replacements. The original CPU run remains failed. The float32 CPU-versus-NumPy
assertion is still a CPU failure; its GPU pass does not classify the K1 Pmax
residual. These follow-ups span recorded revisions, so neither a fresh full-suite
pass nor current-source trajectory/performance qualification is claimed.

The float32 K1 captured-state replay at frozen `42a3d6184` completed in
Slurm13624326 (5,000 particles, 128 pixels, iteration 3 to 4, H100 80 GB).
Its audit fails both particle gates: Pmax absolute-gap p95 is `0.00164212`,
maximum `0.0325985`, and 292 particles exceed `1e-3`. Merged cross FSC-AUC
is `0.995594079`; merged GT FSC-AUC delta is `+0.000254144`. Both map gates
pass, but these do not override the particle failures. The maximum angular
difference is 7.500004 degrees and competing candidate margins were not saved.
Source, fixtures and native libraries remained unchanged. The oracle artifacts
are hashed, but their generating RELION commit/build is unknown. This is not
quality, convergence or performance qualification.

The paired diagnostic in Slurm13624629 uses the existing
`--continuous-relion-noise-state` option on the same source and physical H100.
Particle/half alignment is verified; only the initial noise policy changes.
Half-1 Pmax stays exactly unchanged. Half-2 particles exceeding `1e-3` fall
from 289 to 3, and its 7.5-degree angular difference disappears. Six particles
still exceed the Pmax limit overall: p95 `0.000172648`, maximum `0.001695766`.
Both map gates pass again; both particle gates still fail. The remaining
zero-based particle rows are 901, 1257, 1300, 1414, 3694 and 4568. Their cause
is unresolved; neither double execution nor a wider tolerance was used.

The [versioned noise-state comparison](evidence/pr180-k1-noise-state-20260908/README.md)
contains both failures, full FSC curves, identities, exact commands and the
half-1 negative control. The next quality diagnostic should capture matched
state/candidate scores for the remaining rows before attributing the residual
to arithmetic. Follower replay-completion validation now resides with its
owner, preserving all three return paths, logs and failures. The duplicate
missing-statistics branch was unreachable after the strict preceding guard
and is removed. All 65 focused cases, 38 CPU fast-guard cases and 56 exact
original/new comparisons pass; all four guard truth-table cases agree.
Evidence is under `pr180_follower_completion_20260908/` in the review root.
Frozen `42a3d6184` remains unchanged. Next, make the active status concise while
preserving the detailed dated evidence; then continue normalization ownership
and the matched-candidate diagnostic for the six unresolved K1 rows.

The legacy fast-tier tests use map correlation and
write ledgers under `tests/baselines`; running them unchanged cannot establish
the FSC-based quality contract or authorize writes to established baselines.

The [cleanup plan](cleanup_plan.md) tracks the remaining work and its smallest
useful checks.

Follower dispatch and input remapping now belong to `relion_worker_scale.py`.
The controller supplies its selected precision, numbered iteration and logger;
scheduling stays in the controller. All 95 affected CPU tests pass. Independent
comparison to the pre-change functions matches 360 scenarios (138 successful
updates and 222 errors), including both precisions, unequal/empty halves,
numbered replay and final dispatch. Mutations and telemetry match; normalized
function bodies and the retained controller/worker/test bodies are unchanged.
The scale owner imports without loading the controller. Evidence is under
`pr180_follower_owner_20260908/` in the review root. This structural change does
not modify the frozen PR180 source used by CPU job Slurm13623235.

Follower scale/image-correction updates now share the scale owner as well.
The controller supplies the existing setup, norm/scale statistics, precision
and logger, and retains scheduling and history recording. All 122 focused
cases and the 38-case CPU fast guard pass. Direct comparison to the original
block matches 432 cases (240 updates and 192 errors), including both precisions,
1/2/4 followers, empty halves, first-iteration CC, zero normalization and
malformed inputs. Dtypes, mutations, errors, list identities and logs match;
normalized computation and retained-controller syntax trees match. Evidence
is under `pr180_follower_corrections_20260908/` in the review root. This does
not establish GPU memory or end-to-end quality/performance equivalence.

The first focused run passed 121 cases and exposed a stale dependency guard
that also fails on frozen `42a3d6184`: model-metadata loading moved from the
controller to replay in PR180. The guard now checks availability on the replay
consumer. No numerical tolerance, baseline or runtime behavior was changed.

## Source and review

The control is PR158 commit
`44d770de3f9336ab2f3f6a34203394bae8d1aeed`. The implementation branch is
`codex/recovar-structural-cleanup`, created directly from that control and
preserved at the pre-PR180 checkpoint above. Active work continues on
`codex/integrate-pr180`.
An earlier preparation branch mixes structural changes, runtime repairs and
GUI work; its commits and benchmark results do not qualify this selected series.

[Draft PR179](https://github.com/ma-gilles/recovar/pull/179) is stacked on
[PR158](https://github.com/ma-gilles/recovar/pull/158) so its diff contains only
the cleanup. It was opened at the user's request as a review checkpoint before
qualification is complete. Both changes remain unready for merge until the
applicable checks establish that they preserve the required behavior.

The full source review covered 1,654 text files and 89 other assets. Its finding
inventory, selection evidence, exact test commands and live job records are
stored on Della under:

```text
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/
```

Use `WORK_STATE.json`, `REVIEW_FOR_SCOPE_DECISION.md`,
`PREPARED_CHANGE_INVENTORY.md` and `structural_cleanup/` there. Preserve their
source identities when incorporating evidence into the draft or later results.

## Selected cleanup evidence

| Change | Evidence |
| --- | --- |
| Four unused private diagnostic/test helpers and one constant | `e8aaf5fad`; retained AST unchanged; 28 tests passed, Slurm13561657 |
| Unused public external-normalizer EM M-step wrapper | `5f6c35a40`; retained executable AST unchanged; 24 tests passed, Slurm13561804 |
| Fourteen private plotting, projection, PPCA and test helpers | `3ca69ab3f`; retained AST unchanged; 48 tests passed, Slurm13561946 |
| Dormant single-class significance scorer | `a525e87ef`; existing tests migrated to the active K1-via-Kclass path; 5 passed, Slurm13562507 |
| Duplicate EM environment parsers | `20a4b8235`; parser bodies unchanged; 5 caller checks passed, Slurm13562574 |
| API documentation and isolated docs environment | `1d1d6371f`; 14 tests and strict docs build passed, Slurm13562695; all 58 reviewed API functions rendered; all 496 original packages unchanged |
| Private embedding/state/linalg helpers | `48776ce7b`; retained AST unchanged apart from unused standard-library import; 143 passed, 7 GPU cases deselected, Slurm13562799 |
| Unreachable cubic-mask branch | `924187933`; existing linear policy preserved; 4 tests and 32 exact original/candidate comparisons passed, Slurm13562921 |
| Ten identical JSON conversion helpers | `358db66f3`; shared body unchanged; 16 tests and 10 CLI import/help checks passed, Slurm13563062 |
| Twenty-five identical file-hash helpers | `0ab668be8`; 139 CPU and 5 GPU cases passed, Slurm13565059/13565290; all 25 module CLIs pass; 12 direct-entry import failures match PR158, Slurm13565356 |
| Sixteen historical scorecard CLIs | `df09caa3b`; validators and renderers unchanged; 88 tests, 32 help checks and 32 pinned-report checks passed without site packages, Slurm13565574 |
| Exhaustive orientation-test memory | `7f8200e09`; blocked comparison matches dense float32/float64 results exactly; all 32 cases pass with the original angular gate, 592640 KiB peak RSS, Slurm13565654 |
| Direct local-search dependencies | 79 cases passed, Slurm13566158; planner/kernel bindings and numerical AST unchanged; tests migrated to the modules that call each dependency |
| Contributor workflow map and development navigation | `3a7c26863`; guide links, strict docs and four rendered development pages passed, Slurm13566512 |
| Unused controller constant copies | `bebd22faf`; 27 cache, batch and fine-grid cases passed, Slurm13566781; active owner values unchanged |
| Shared diagnostic original-image mapping | `352681db9`; 10 dump-targeting tests and 36 exact mapping/precedence/error comparisons passed, Slurm13567159 |
| Remaining hash-helper importers | `159059131`; three missed script imports caused collection failure in Slurm13567024; after migration, three tests, three CLI helps and collection of 6,708 non-GUI tests passed, Slurm13567239 |
| Canonical EM helper and PPCA bridge imports | `6ebaf5fad`; 32 focused tests and non-GUI collection passed, Slurm13569139; controller computations unchanged |
| Versioned K1 run archive | `02e5d79ec`; 84 fixture identities and 189 losslessly converted FSC curves; archive integrity and strict docs build passed |
| Shared dense/local translation-prior center | `be1f2913e`; 7 existing caller tests, 48 exact value/dtype/error comparisons and non-GUI collection passed, Slurm13569721 |
| Replay/state-swap diagnostic ownership | `db9576bd6`; 28 caller tests and 122 exact variant/snapshot/mutation/error comparisons passed, Slurm13572837; variant choices work without importing the controller |
| Frozen scoring-state integrity ownership | `05976cdf4`; two function ASTs unchanged; 74 focused tests and 44 exact payload/hash/error checks passed, Slurm13572933; final import/style check includes 100 passing cases, Slurm13572973 |
| Pipeline downsample/cache preparation | `2531c66b3`; 22 caller tests and 50 exact callback/mutation/error comparisons passed, Slurm13573180; public API and numerical stages unchanged |
| Shared pre-/post-join accumulator writer | `e05ca337f`; 48 caller/format tests and 144 exact payload/gate/error comparisons passed, Slurm13576398; numerical controller AST outside dump bodies unchanged |
| Unused EM helper inputs | `9f1fde1a7`; three unused keyword parameters and nine callers migrated; 30 cases passed, Slurm13576908; retained function bodies and assertions unchanged |
| Redundant module aliases and stale comments | `facf535c3`; 17 cases and 228 exact noise/metadata/precision-flag comparisons passed, Slurm13577241; calculations and defaults unchanged |
| Numeric convergence environment overrides | `e7533f762`; 6 existing caller tests and 240 exact value/type/warning comparisons passed; parser bodies and surrounding computations unchanged |
| Halfset output ownership | `89d901f99`; 4 existing output/offload tests passed; documentation and two local parameter names clarified, with executable AST otherwise unchanged |
| Adaptive batch-plan ownership | `58e2069cd`; 42 tests and 160 exact K1/K2/K4/K8/K16 plan/callback/error comparisons passed, Slurm13581512; 9 moved declarations unchanged, K-class replay CLI loads, all 1,780 source hashes stable during validation |
| Restart/replay policy ownership | `30163cdcd`; four moved function ASTs unchanged; 67 tests passed, Slurm13585434; controller and sampling bindings verified |
| Replay sampling dependencies | `cbfbcdc8c`; 12 tests and 90 exact K1/K2/K4/K8/K16 grid/value/dtype/error comparisons passed, Slurm13585641; shared grid policy now belongs to sampling |
| Metadata sampling dependencies and dead Euler-grid helper | `1bb616533`; 17 tests passed, Slurm13586745; retained numerical AST unchanged; 53 implementation helper modules have no direct controller imports |
| Dense/local output interfaces | `03aa2dde2`; 133 tests passed, Slurm13587342, plus 120 exact K1/K2/K4/K8/K16 two-half mutation/return/error comparisons; list identities and normalized controller AST unchanged |
| Unused controller imports and ineffective test stubs | `dbada1eb5`; 23 tests passed, Slurm13588076; four unused aliases removed; tests now patch active call-site dependencies; retained executable controller AST unchanged |
| Scoring result module | `4ccd65f78`; 142 tests passed, Slurm13589209, plus 120 exact K1/K2/K4/K8/K16 two-half comparisons; six moved declarations and retained controller AST unchanged; all source hashes stable |
| Materialized dense posterior test reference | `f35712af9`; 42 adaptive-oversampling tests passed, Slurm13589912; 12 exact weight/assignment comparisons; moved function and retained engine executable AST unchanged |
| Local-engine cache/timing imports | `8b51711f4`; 13 tests passed, Slurm13590097; 13 unused aliases removed and three test constant imports migrated; active owner bindings and executable engine/test AST unchanged |
| Dead local-cache memory wrapper | `f8ceeea2a`; whole-repository caller search found only the definition; retained module AST unchanged; five existing cache-limit tests and strict docs passed |
| Named dense-engine result | `c4713c5a0`; 352 selected tests passed, Slurm13590978; all eight optional-flag combinations matched the old engine numerically on a tiny CPU fixture; script imports, CLI and strict docs passed; historical fixture cases were collected only |
| Shared controller test setup | `eabc7ee66`; 12 tests and 36 exact original/extracted fixture comparisons passed, Slurm13591659; remaining test setup and assertion AST unchanged; 105 net duplicate lines removed |
| Named local-search wrapper result | `3d38f0db9`; 150 tests, 160 exact optional-field identity cases and 32 matching errors passed, Slurm13592280; kernel dispatch unchanged; controller no longer repacks/decodes this result |
| Nine unused private numerical/provenance helpers | `4280e0cdc`; 166 CPU tests passed, Slurm13592709, with 11 GPU-marked cases explicitly deselected; 213 lines removed; retained executable AST identical in all eight modules |
| Unused historical script helpers | `540fdf005`; 44 report/table tests and four CLI checks passed, Slurm13592996; 84 lines removed; retained executable AST and report acceptance/formatting unchanged |
| Remaining scoring-output helpers | `3bdda606b`; 73 tests passed, Slurm13593442; 70 exact value/identity/mutation cases and 14 matching errors; four moved declarations and retained controller/output-module AST unchanged |
| Paired historical scorecard validation | `db4aab19a`; 12 tests passed, Slurm13614908; 86 exact values, 2,286 matching errors and 40 standalone CLI checks; all four pinned Markdown reports unchanged |
| Captured sampling grid ownership | `6fb793698`; 104 local CPU tests passed; 108 exact outputs and 22 matching errors; three moved function ASTs and retained controller/replay AST unchanged; strict docs passed; Slurm13616833 was cancelled before execution |
| Explicit sampling schedule ownership | `1331659e7`; 84 local CPU tests passed; 36 exact schedules, 87 matching errors and 36 state-transition comparisons; moved functions and retained numerical AST unchanged |
| RELION projector preparation ownership | `9cf18fb34`; 24 focused CPU tests passed; 24 reference/cache/dump comparisons, six captured-array identity cases and 18 matching errors; retained numerical AST unchanged; strict docs passed |
| Captured replay test organization | `3535f1a4d`; 10 moved cases passed; the 352-case original module inventory equals 342 remaining plus 10 moved cases, with parameter IDs and assertions preserved. The initial independent-import check exposed the package dependency fixed below. |
| Helper/controller import boundary | `433ed8761`; 12 focused tests and the then-16-case CPU guard passed; seven fresh helper imports leave the controller unloaded; all 20 retained exports preserve object identities. All 6,714 non-GUI unit cases collected; collection alone is not execution. |
| Fast guard replay and sampling contracts | `517daf1c7`; 28 CPU cases passed in 47.70 seconds; all 16 previous cases retained, plus 12 existing replay/schedule cases. Fresh-process import boundary, ancestry, mirrored guides and strict docs passed. |
| Controller box-size names | `13318f807`; 17 focused FSC/current-size CPU tests passed; bijective local-name normalization preserves the entire controller AST, casts, keyword arguments and stored fields |
| Standalone diagnostic hashing | `f0612975f`; 387 CPU tests passed; 518 exact hashes and 222 matching errors across 74 scripts. All 74 module CLIs and 50 direct CLIs pass; 24 direct-entry failures reproduce on the unchanged parent. Strict docs pass; 14 lint diagnostics and 66 formatting requests remain inherited. |
| Resolution scheduling ownership | `ddb26a78b`; six unchanged helpers moved out of the controller; 16 CPU tests, 509 exact values, 36 matching errors, 545 warning comparisons and 806 input-mutation checks passed. |
| Independent resolution scheduling tests | `f85cdbdf6`; ten existing cases moved into `test_resolution_scheduling.py`; all pass without importing the controller. The 342-case original inventory equals 332 retained plus ten moved cases, including fixtures and markers. |
| Broader CPU fast guard | `5cc4724a7`; all 38 cases pass, retaining the previous 28 cases and adding the ten resolution contracts. Helper imports remain independent of the controller. |
| Unread private controller parameters | `bbf1d6901`; three unused inputs and forwarding keywords removed. Private loop body and public signature are unchanged; 24 CPU tests, 27 exact dispatch comparisons and five matching errors passed. Public legacy no-op inputs are documented accurately. |
| Approximate-accuracy convergence ownership | `7e7d92695`; policy and six constants moved to convergence helpers; two boolean wrappers removed. All 86 CPU cases, 532 exact policy values, two matching errors, 534 warning comparisons and 1,068 state-mutation checks pass; the original caller logger is preserved. |
| Angular-grid policy ownership | `966bd0c64`; four unchanged helpers and the exhaustive-grid cap moved to convergence. All 119 CPU cases, 530 exact values, 288 matching errors and 1,636 input-mutation checks pass, including global/local and K-class finalization callers. |
| Independent convergence policy tests | `3dd2cb107`; eight pure cases moved into `test_convergence.py`; all 90 convergence cases pass without importing the controller. All 414 case identities, fixtures and markers across the two affected modules are preserved. |
| First-iteration adaptive dispatch ownership | `df6fc04bc`; dispatcher moved to `firstiter_cc.py` with the caller's logger. All 58 CPU cases, 2,400 exact dispatch comparisons and 24 matching errors pass. The initial final whitespace check failed; a recorded AST-identical whitespace correction passes. |
| Precision selection ownership | `9d25ee5d7`; two selectors moved to dtype policy helpers with the original static settings supplied explicitly. All 12 CPU cases, 491 exact values, 805 matching errors and 1,296 configuration-read-order comparisons pass. |
| Explicit K1/K2/K4 local dispatch coverage | `681c2e6ed`; the former K4-labeled fixture actually constructed two classes. Its K1/K2 cases are retained and a genuine K4 case now checks four mean/prior entries. All three pass; all 324 previous case contracts are preserved, with one added case. This is dispatch coverage, not K4 quality qualification. |

These are focused checks, not full quality or performance qualification.
Counts describe each validation job and may overlap. No scientific tolerance
or established baseline was changed.

The retired public EM wrapper is
`local_score_pass.fused_score_normalize_mstep_abs2_with_log_z_on_demand`.
It had no repository callers. Its external-normalizer support calculation
remains available through
`fused_score_normalize_support_probs_abs2_with_log_z_on_demand`; active callers
perform M-step reductions after packing significant rows. The common fused
`fused_score_normalize_mstep_abs2_on_demand` API remains available.

Local-search kernels are imported from `local_em_engine.run_local_em_exact`
and `k_class.run_local_k_class_em`. Their unused aliases in `iteration_loop`
have been removed. `local_search_iteration` imports its layout builder, batch
planner and kernels directly; its tests patch those call-site bindings.
`iteration_loop.build_local_hypothesis_layout` remains an active dependency of
the controller's adaptive parent-layout construction. The batch planner imports
the shared memory-query utilities directly, avoiding a reverse dependency on
the controller. No scoring, batching formula or output layout changed.
Replay and rotation metadata also use sampling dependencies directly. Dense and
local scoring receive the existing `PerHalfOutputs` container, preserving its
list identities and the order of per-half updates. Their separate output-list
arguments have been removed; common payload storage and K-class summaries
retain their existing owners.

Report scripts now share `recovar.utils.json_utils.to_jsonable`. Conversion
preserves the old NumPy/path/nested-container behavior. It does not establish
that metrics are finite or that a report satisfies an acceptance policy.

## Current qualification gaps

The benchmark contract covers synthetic and real data, including K1, K4 and
K2/K8/K16; see [benchmarks](benchmarks.md). Preserve the existing failed cells.
The earlier mixed candidate passes the scoped K2 comparison, but its real K1
trajectory converges after 16 iterations versus the control's 17, and its final
merged-map direct FSC-AUC is 0.949891246, below the unchanged 0.995 gate. Repeated
synthetic K1 measurements also retain an unresolved host-RSS warning over 10%.
Those results cannot establish a behavior-preserving or performance-qualified
cleanup. Follow the external ledger for the complete reports and pending jobs.

PR158's K-class path has a separately identified undefined-variable bug. Its
prepared one-line policy repair is held outside this structural series pending
its separate correctness decision. Label any baseline containing that repair
as a repaired control, never as the unchanged PR158 source.

The local-search output review also identifies an inherited K-class wrapper
bug when `return_best_pose_details=False`: the wrapper packs pose fields that
its positional decoder then reads as statistics. The tuple construction and
decoder are unchanged from PR158. A controlled-kernel wrapper diagnostic covers
16 combinations of K2/K4, pose, noise and class-detail flags: the eight with
pose details enabled route correctly; the eight with them disabled do not.
An unapplied patch routes all 16 correctly. The controller's K-class scoring
call explicitly enables pose details, so this diagnostic does not explain the
trajectory failures. The patch and reproduction script are in
`structural_cleanup/local_kernel_contract_review/` under the review root.
Keep this repair separate from a behavior-preserving output-interface migration.

## Checkpoint results and next checks

The initial structural series through `430f46325` passed three paired cold
synthetic K1 comparisons in Slurm13564282/audit13564283. All eight compared
controller fields agree; all 81 direct map curves pass, with minimum FSC-AUC
0.999999989582. Pairwise wall-time changes are -0.415%, +0.253% and -0.835%.
Peak host-RSS changes are +0.055%, +3.505% and +11.772%; the last pair retains
the existing memory-regression warning. All six memory traces are admissible,
but these capture-enabled runs do not qualify ordinary execution performance.
Subsequent structural commits have their own focused checks; these results
do not establish end-to-end qualification for later source revisions.

Source `159059131` passed a further cold K1 case25 pair in
Slurm13567539/audit13567540. All eight compared controller fields match and all
27 direct map curves pass; minimum FSC-AUC is 0.999999998203. Wall time changed
by +0.081%, GPU peak by 0%, and host RSS by +1.561%. The
[versioned archive](evidence/k1-case25-20260907/README.md) preserves the commands,
fixture identities, measurements and shellwise curves. An additional audit of
37 explicitly image-ordered fields found exact recorded pose/support decisions;
maximum Pmax difference was 0.000155002 and the largest per-field p95 difference
was 0.0000503063. This still does not compare all candidate scores or accumulators.

A new broad CPU checkpoint at frozen `966bd0c64` finished in
Slurm13620809 with **6,361 passed, 337 skipped and 15 failed** in 57 minutes
45 seconds. It uses the same declared non-GUI CPU selection as the previous
checkpoint, including its one shared-`/tmp` exclusion. Its separate pixi
environment has the same lockfile and all 317 Python package versions. The
automatic audit accounts for twenty previously moved replay/resolution cases:
all 6,713 common case statuses are unchanged, with no added or removed cases.
Source hashes stayed unchanged. The same fifteen failures remain failures;
this result does not qualify later source changes or the PR180 integration.

A later broad CPU checkpoint at frozen `f8ceeea2a` finished in Slurm13590326
with **6,361 passed, 337 skipped and 15 failed**. All 6,713 collected cases were
accounted for, and source/native-library hashes stayed unchanged. The 15 failure
IDs match the earlier run below; no common case changed status. Eight added
accumulator-dump cases passed. Two removed tests covered retired helpers; the
translation-prior case has the same inputs, expected arrays and tolerances in
the retained shared-helper test. This remains a failing suite, and it predates
the named dense/local result changes. The inventory and failure comparison are
recorded under `structural_cpu_checkpoint/` in the Della review directory.

The broad CPU inventory at `159059131` finished in Slurm13567313 with 6,355
passed, 337 skipped and 15 failed. It excludes three deferred GUI files and one
test whose existing production call can delete a shared `/tmp` directory.
Ten failures reproduce on unchanged PR158 in comparisons13569182/13569637;
the other five are known GPU-only diagnostic cases. All eight GPU-dependent
failures pass on frozen source `be1f2913e` in Slurm13569842 with a verified CUDA
library. The remaining failures concern unavailable cluster tools, one float32
comparison and the separately proposed K-class undefined-variable repair.
No tolerance or test marker was changed. This is not full-suite or hosted-CI
qualification; the original import-failed attempt13567024 is also preserved.

Both unchanged-PR158 real-data repeats in Slurm13562724 emitted 16 numbered
iterations on the same physical H100. Audit13562837 fails the unchanged 0.995
direct-map gate starting at iteration 8; final merged FSC-AUC is 0.964376533.
All eight compared controller fields agree. A separate audit of 69 image-ordered
fields finds the first support/Pmax differences at iteration 2, and two images
change translations by 0.5 pixel at iteration 3. An earlier boundary is visible
in [15 saved first-iteration arrays](evidence/real10076-repeatability-20260907/README.md):
the four accumulator arrays differ, while the saved noise, tau2, FSC, grid and
assignment arrays match exactly. Audit13575086 verifies all 30 original output
hashes and leaves the files unchanged. These arrays are saved after
reconstruction; complete live E-step input identity is not established, and the
cause remains unresolved.
Three shortened first-iteration controls in Slurm13576748 locate different
accumulators before the low-resolution half join. Within each captured run,
post-join buffers exactly equal the late saved arrays. Audit13576847 verifies
49 consumed files; the other 11 first-iteration fields also match the original
autonomous run by file hash. This narrows the next diagnostic to upstream
scoring/accumulation without identifying the responsible kernel.

Unchanged-source capture13579420 and operand comparison13579421 completed.
Both runs contain all 5,000 half-1 particles and 40,000 valid rotation rows in
23 packets. All recorded scatter operands and identities match exactly.
The pre-join data and weight arrays still differ: maximum absolute differences
are 1.53598e-8 and 2.32831e-10, respectively. The audit verifies the sealed
source/input/output identities. [Fixed-input replay13585254](evidence/real10076-scatter-repeatability-20260907/README.md)
then repeats the first captured stream three times through the unchanged
production accumulation function from fresh zeros on the same physical H100.
Native data differ in 2,137–2,177 elements, with maximum gap 1.66600e-8;
native weights differ in 274–295 elements, with maximum gap 1.16415e-10.
The two warm trials also differ. All 1,774 recorded identities remain unchanged.
This demonstrates non-bitwise repeatability of the production accumulation
function, including its operand preparation. It does not identify a particular
low-level instruction or prove the cause of later trajectory divergence.
Replay synchronization and transfer timings do not qualify ordinary performance.

This negative control means a single real-data candidate mismatch cannot be
attributed to cleanup alone. The [structural real-data pair at `be1f2913e`](evidence/real10076-structural-pair-20260907/README.md)
completed in Slurm13569949. Audit13569953 verifies source/artifact integrity
but fails 27 of 51 map comparisons, starting with half 1 in iteration 8;
final merged FSC-AUC is 0.972350364. Audit13577514 fails 60 of 69 recorded
particle fields, starting with support counts in iteration 2 and a 0.5-pixel
translation change in iteration 3. Both runs execute 16 numbered iterations
and all eight controller fields agree. The result remains unqualified;
baseline repeatability and the cause of the differences are unresolved.

The no-intermediate-capture K1 study at `be1f2913e` completed seven trajectories
in Slurm13570225; audit13570230 completed its comparison but exits 2 because
one of three control/candidate pairs fails particle-level gates. All final-map
and reference gates pass, and all eight controller fields agree. The failed
pair changes one support count at iteration 7 and exceeds the existing Pmax
limit at iterations 7 and 8. Comparing the existing repeats requires no further
GPU execution: all three unchanged-source repeat pairs pass the recorded-field
contract, while the first candidate differs from both later candidate runs at
image 685's iteration-7 support count. This is unresolved; it does not establish
that the structural changes caused the variation or qualify the candidate.

The capture/no-capture calibration passes the recorded quality contract.
All seven memory traces have no sampling errors or unrelated GPU processes.
Candidate wall-time changes are +4.789%, +6.711% and +6.101%; sampled host-RSS
changes are -10.676%, +4.723% and -0.996%. The unchanged-source calibration itself
has +10.709% sampled host RSS despite -0.421% GNU-time process high-water RSS.
These are distinct memory measurements; neither cancels the other. A further
audit of 15 recorded state families finds first-iteration shell-statistic
differences in both control and candidate repeats; maximum shell-sum differences
are approximately 5e-10 to 1e-9. These precede the particle-level failure but do
not establish its cause. The failed
particle gate prevents a performance-qualified result. Stage timing remains
enabled, and this single case does not qualify broader workloads.

The older mixed K16 comparison13561074 fails its historical RELION gate. Its
legacy direct-comparison report passes aggregate thresholds, but one particle
changes class and pose at iteration 5. Competing score margins were not
captured, so strict equivalence is not established. The older K4 pair remains
running in Slurm13560202 on an A100 80GB since September 8; its dependent audit
is Slurm13560356. Neither old result qualifies the selected structural revision.

Shared SPA and cryo-ET 50k/128 regression tests pass in jobs13569618/13569619
at frozen source `6ebaf5fad`. The first external inventory audit fails because
the committed baselines retain ten retired aliases for metrics now emitted
under canonical names. Historical commits establish those aliases, and every
alias equals its canonical baseline value. Audit-only job13570324 verifies all
16 canonical required metric keys pass in each workload, plus four finite
local-resolution measurements without historical baselines. Source, library
and baseline contents remain unchanged; no GPU workload was rerun for this audit.

Historical performance comparisons retain warnings over 10%, including dataset
generation, metrics, cryo-ET state computation and recorded stage GPU memory.
These stage memory values are the existing endpoint/cumulative measurements,
not independently sampled process peaks. The runs do not establish paired
performance qualification. The [versioned shared archive](evidence/shared-spa-et-20260907/README.md) also
preserves generated fixture identities and six complete FSC curves; their
threshold-frequency summaries all saturate. Outlier/downstream and remaining
shared checks are still outstanding.

Complete these comparisons and classify failures before larger executor
extraction. The K-class correctness, workflow and benchmark-validation proposals
remain outside this series pending their separate decisions. New-engine work
remains a later milestone.

The [historical program](../math/em_parity_program.md),
[parity notes](../math/relion_parity_agent_notes.md) and
[completion records](../math/em_parity_best_metrics.md) retain quantitative
gates and dated evidence. Their old next actions are historical context.
