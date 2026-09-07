# Current EM development scope

The current milestone is behavior-preserving cleanup and development setup
across RECOVAR, with EM prioritized. GUI/frontend work and the new HIA engine
are outside this milestone. EM public APIs may change when they simplify the
code; migrate callers, tests and documentation together. Preserve non-EM APIs,
saved formats, numerical behavior and scientific defaults. Present numerical
repairs separately for a user decision.

The [cleanup plan](cleanup_plan.md) tracks the remaining work and its smallest
useful checks.

## Source and review

The control is PR158 commit
`44d770de3f9336ab2f3f6a34203394bae8d1aeed`. The implementation branch is
`codex/recovar-structural-cleanup`, created directly from that control.
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
queued; neither old result qualifies the selected structural revision.

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
