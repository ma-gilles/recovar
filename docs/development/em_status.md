# Current EM development scope

The current milestone is behavior-preserving cleanup and development setup
across RECOVAR, with EM prioritized. GUI/frontend work and the new HIA engine
are outside this milestone. EM public APIs may change when they simplify the
code; migrate callers, tests and documentation together. Preserve non-EM APIs,
saved formats, numerical behavior and scientific defaults. Present numerical
repairs separately for a user decision.

## Source and review

The control is PR158 commit
`44d770de3f9336ab2f3f6a34203394bae8d1aeed`. The implementation branch is
`codex/recovar-structural-cleanup`, created directly from that control.
An earlier preparation branch mixes structural changes, runtime repairs and
GUI work; its commits and benchmark results do not qualify this selected series.

The full source review covered 1,654 text files and 89 other assets. Its finding
inventory, selection evidence, exact test commands and live job records are
stored on Della under:

```text
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/
```

Use `WORK_STATE.json`, `REVIEW_FOR_SCOPE_DECISION.md`,
`PREPARED_CHANGE_INVENTORY.md` and `structural_cleanup/` there. Preserve their
source identities when incorporating evidence into a future PR.

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

## Next check

The initial structural series through `430f46325` passed three paired cold
synthetic K1 comparisons in Slurm13564282/audit13564283. All eight compared
controller fields agree; all 81 direct map curves pass, with minimum FSC-AUC
0.999999989582. Pairwise wall-time changes are -0.415%, +0.253% and -0.835%.
Peak host-RSS changes are +0.055%, +3.505% and +11.772%; the last pair retains
the existing memory-regression warning. All six memory traces are admissible,
but these capture-enabled runs do not qualify ordinary execution performance.
Subsequent structural commits have their own focused checks; these results
do not establish end-to-end qualification for later source revisions.

The corrected source `159059131` is frozen for broad CPU inventory job13567313.
The inventory excludes three deferred GUI files and one test whose existing
production call can delete a shared `/tmp` directory. The five known GPU-only
cases without markers remain in the unchanged CPU selection; their failures
must be recorded explicitly. The earlier import-failed attempt13567024 is
preserved. This selection is not full-suite or GitHub-hosted CI qualification.
The separate K-class correctness decision is pending. Two
unchanged-PR158 real-data repeats are running in Slurm13562724, with comparison
job13562837, to test repeatability before attributing the earlier candidate's
trajectory difference to a source change. The two repeats share an allocated
H100; its UUID differs from the original control, which the report must retain.

Finish the baseline/evidence review, then freeze a candidate for the applicable
shared and EM regression workloads. Larger executor extraction and the new
engine remain later work. Keep numerical fixes separate until approved.

The [historical program](../math/em_parity_program.md),
[parity notes](../math/relion_parity_agent_notes.md) and
[completion records](../math/em_parity_best_metrics.md) retain quantitative
gates and dated evidence. Their old next actions are historical context.
