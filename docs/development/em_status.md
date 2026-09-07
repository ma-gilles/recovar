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

These are focused checks, not full quality or performance qualification.
No scientific tolerance or established baseline was changed.

## Current qualification gaps

The benchmark contract covers synthetic and real data, including K1, K4 and
K2/K8/K16; see [benchmarks](benchmarks.md). Preserve the existing failed cells.
The earlier mixed candidate passes the scoped K2 comparison, but its real K1
trajectory has a different iteration count from the original control. Repeated
synthetic K1 measurements also retain an unresolved host-RSS warning over 10%.
Those results cannot establish a behavior-preserving or performance-qualified
cleanup. Follow the external ledger for the complete reports and pending jobs.

PR158's K-class path has a separately identified undefined-variable bug. Its
prepared one-line policy repair is held outside this structural series pending
its separate correctness decision. Label any baseline containing that repair
as a repaired control, never as the unchanged PR158 source.

## Next check

Validate the mirrored instructions and linked runbooks, then continue reviewed
EM dead-code removal and duplicate-helper consolidation. Freeze the resulting
candidate for the applicable shared and EM regression workloads. Keep new
scientific work separate until cleanup is qualified.

The [historical program](../math/em_parity_program.md),
[parity notes](../math/relion_parity_agent_notes.md) and
[completion records](../math/em_parity_best_metrics.md) retain quantitative
gates and dated evidence. Their old next actions are historical context.
