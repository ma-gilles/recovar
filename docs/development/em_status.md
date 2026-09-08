# Current EM development scope

The milestone is behavior-preserving cleanup and development setup across
RECOVAR, with EM first. GUI/frontend and the new HIA engine are excluded.
Remove demonstrated dead/duplicate code, clarify ownership and APIs, maintain
consistent contributor instructions, and establish reproducible synthetic,
real-data and K-class accuracy/performance evidence for the selected source.
The [cleanup plan](cleanup_plan.md) tracks the remaining engineering work.

EM public APIs may change with callers, tests and documentation migrated
together. Preserve non-EM APIs, saved formats, scientific defaults and numerical
behavior during cleanup. Keep separately proposed numerical/runtime repairs
outside the structural series until authorized. Full local integration of
PR180 is authorized; its numerical changes need fresh qualification.

**Production EM remains float32.** Double precision is diagnostic only; a gap
shrinking in double does not prove numerical noise. Final K1 and exactly-K4
quality/performance evidence must use float32. Preserve deliberate
higher-precision metadata and host operations. Only an explicit user decision
can change the [mandatory precision policy](../../recovar/em/AGENTS.md).

## Source and review

| Source | Identity and role |
| --- | --- |
| PR158 control | `44d770de3f9336ab2f3f6a34203394bae8d1aeed`; preserve unchanged |
| Preserved structural branch | `codex/recovar-structural-cleanup` at `681c2e6ed03c9b63b94f07bc47a4a8367b1b4de3` |
| Local PR180 integration | `42a3d6184c6d05a9f4f97bd00120e62d0081f1d3`, incorporating PR180 head `1e2f229b3e0e8edaec029d2b604937f692148578` |
| Active implementation branch | `codex/integrate-pr180`; includes follower-completion and normalization-owner cleanup |

The active checkout is
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907/`.
Read its actual `git rev-parse HEAD`, dirty diff hash and untracked manifest
before validation. Frozen benchmark checkouts remain immutable.

[Draft PR179](https://github.com/ma-gilles/recovar/pull/179) is stacked on
[PR158](https://github.com/ma-gilles/recovar/pull/158) for review of the structural
series and the integrated [PR180](https://github.com/ma-gilles/recovar/pull/180)
checkpoint. Local PR180 integration
does not mean the GitHub PR is merged or that later commits passed old runs.
On September 8 the user explicitly authorized pushing the current state to the
review branch with incomplete validation. This publication exception does not
waive scientific gates or authorize merging; PR179 remains a draft. Check its
actual remote HEAD to identify the latest published snapshot.

The original review covered 1,654 text files and 89 other assets. Its inventory,
decision records, exact commands and job records live under the **review root**:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/`.
Use `WORK_STATE.json`, `REVIEW_FOR_SCOPE_DECISION.md`,
`PREPARED_CHANGE_INVENTORY.md` and the named experiment directories there.

## Selected cleanup evidence

The [dated cleanup history](em_cleanup_history_20260908.md) preserves the full
previous status text, including source identities, failures, shared-code work
and old next actions. Historical checks qualify only their recorded source.
Current owners are mapped in the [codebase guide](codebase.md).

Recent work after PR180 integration:

| Change | Focused evidence |
| --- | --- |
| Follower dispatch and runtime input remapping owned by `relion_worker_scale.py` | 95 CPU cases; 360 exact old/new scenarios, including 222 errors |
| Follower scale/image-correction updates at the same owner | 122 focused cases, 38 CPU fast-guard cases; 432 exact scenarios, including 192 errors |
| Sampling helper uses its direct dependency without importing the controller | 10 affected cases and overlapping 19-case sampling selection; 8 exact grids at orders 0–3 in both precisions |
| Explicit diagnostic norm dtype and production-default tests | 9 norm/capture cases plus 2 existing precision-default guards; runtime and tolerances unchanged |
| Replay-completion validation at its owner; unreachable duplicate statistics guard removed | 65 focused cases, 38 CPU fast-guard cases; 56 exact result/error/log comparisons and all 4 guard combinations |
| Norm/scale arithmetic and seven formula tests moved to `relion_normalization.py` | 72 focused cases, 38 CPU fast-guard cases; 576 exact output and 17 exact error comparisons; all 380 collected cases retained through seven explicit ID migrations |
| Duplicate final half-map reconstruction calls consolidated; overwritten initialization removed | 31 affected cases, 38 CPU fast-guard cases and 3 K-class checks; 56 exact call/output/error scenarios across K1/K2/K4; the existing K1 convergence test now checks all five saved reconstruction products |
| Separately authorized K-class optional-field routing correction | New wrapper regression: 8 failed/8 passed before, all 16 passed after; 29 affected cases and 38 CPU fast-guard cases pass. The mismatch is inherited from PR158 and PR180; the tracked production controller's pose-enabled K-class call is unchanged |
| Local engine returns `LocalEMResult`; positional packer and both decoders removed | 188 focused cases and 38 CPU fast-guard cases pass; all 511 existing case IDs retained plus 4 new capture-routing cases; 16 engine flag combinations and both consumer mappings preserve object identity; 1,218 existing assertions/numerical checks preserved through explicit field mapping |

The replay-completion change preserves all three controller return paths, output keys,
validation exceptions and log messages. Detailed evidence is in
`pr180_follower_completion_20260908/` under the review root. These checks do
not establish GPU memory or end-to-end quality/performance equivalence.

## Current qualification gaps

**The current source is not yet scientifically or performance qualified.**
No baseline or tolerance has been widened, and no failed run is waived.

- The frozen PR180 CPU checkpoint (`13623235`, clean `42a3d6184`) has
  **6,452 passed, 340 skipped and 25 failed**. All 25 have explicit passing
  focused follow-ups: 11 after reviewed ownership/test-contract changes, and
  14 on the unchanged source with GPU/tools (`13625819`: 8 passed;
  `13625952`: 6 passed). Twenty-four case IDs are retained; one diagnostic norm
  case has two explicit precision replacements. These checks span revisions
  and environments; they are not a fresh full-suite pass. The float32
  CPU-versus-NumPy assertion still fails on CPU. See the
  [failure reconciliation](evidence/pr180-unit-checkpoint-20260908/README.md).
- Frozen PR180 passes 16 selected native GPU kernel cases on H100 80 GB in
  `13623670`, without skips and with verified source/library identities.
  This covers the selected kernels, not every double path or a trajectory.
- Two float32 K1 iteration-3-to-4 replays on the same physical H100 isolate a
  half-2 noise-state mismatch. Using continuous per-half noise reduces Pmax
  gaps above `1e-3` from 292 to 6 and removes a 7.5-degree pose difference;
  half-1 Pmax is exactly unchanged. Both particle gates still fail:
  p95 `0.000172648`, maximum `0.001695766`. Both map gates pass and cannot
  override the particle failures. Remaining zero-based rows are **901, 1257,
  1300, 1414, 3694 and 4568**. Candidate margins are absent, and the producing
  RELION build is unknown. The [paired evidence](evidence/pr180-k1-noise-state-20260908/README.md)
  preserves both failed audits (`13624326`, `13624629`).
  The [six-particle follow-up](evidence/pr180-k1-targeted-capture-20260908/README.md)
  in `13627156` exactly reproduces all six production Pmax values and saved
  poses. The existing score dump disables big-JIT and changes Pmax by at most
  `3.05e-5`. For row 901, float64 normalization of its captured float32 scores
  changes Pmax by only `5.01e-10`; the gap to RELION remains `0.00168`.
  Final normalization does not close this captured gap; its score operands,
  priors and correspondence to RELION still need comparison.
- Historical synthetic K1 and real10076 repeatability failures remain open.
  Fixed-input production scatter replays differ even with identical captured
  operands; this does not establish the cause of later support, pose or
  convergence changes. Historical K2 passes and K8/K16 failures do not qualify
  the current source. The exact-K4 repaired-control/older-candidate pair is
  **running** in `13560202`; audit `13560356` is pending dependency (last checked
  September 8). Poll Slurm before treating this status as current.
- Historical shared SPA/ET runs pass their 16 canonical metric keys, but
  preserve performance warnings and do not qualify current source. Outlier,
  downstream and other applicable shared checks remain outstanding. Current
  synthetic, real, exact-K4 completion and paired speed/memory evidence are
  still required by the [benchmark contract](benchmarks.md).

PR180 fixes the K-class undefined-variable path on the merged source; its
focused test passes. The historical K-class pose-detail decoder repair,
benchmark-validation changes and unrelated runtime proposals remain separate.
Keep historical repaired controls labeled as such. Preserve the reviewed
final-grid-correction default during cleanup; resolving its mismatch with the
strict-parity target needs separate scientific qualification.

## Checkpoint results and next checks

1. The completed local-result migration is frozen at
   `6ee49dc5c8255a35d16930dd948341a9fad515eb` in
   `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_api_checkpoint_20260908/`.
   CPU job `13630086` and GPU job `13630096` cover the same 566 cases in ten
   affected EM test modules. CPU expects five explicitly GPU-marked skips;
   GPU must execute every case. Results are pending, not qualification.
   The driver, source/library hashes and case inventory are under
   `em_api_checkpoint_20260908/` in the review root; logs use the same directory
   name under `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/`.
   Review these results before further engine changes. Engine numerical statements and
   K-class computation are unchanged; the named result removes both positional
   decoders. It also fixes the related capture-routing ambiguity where an
   implicit profile could be returned as significant counts when hidden by the
   wrapper. Four new cases preserve explicit profile visibility/copy behavior
   and require actual counts. Evidence is in `pr180_local_named_result_20260908/`
   under the review root. The earlier separately authorized K-class routing fix
   remains recorded under `pr180_local_result_contract_20260908/`. Neither
   routing correction explains or waives existing benchmark failures.
2. For K1, use the reproduced six-particle case to compare matched score
   operands, priors and candidate geometry, starting with row 901. Preserve
   the measured dump-versus-production difference and the unknown generating
   oracle build. Captured-score normalization is not the explanation; a
   matched RELION surface is needed before attributing the residual to noise
   or proposing a production repair.
3. Poll the existing K4 pair and audit; preserve its source and outputs.
   Complete the source/fixture requirements before new scientific checkpoint
   runs. Qualify K1, real particles and exact K4 on the selected source.
4. Group related cleanup into immutable checkpoints for broader CPU and
   applicable GPU checks. Run completion and performance workloads at those
   checkpoints, not after every helper edit. Finish the shared validation and
   publication requirements before pushing.

Legacy fast-tier map-correlation gates and baseline-writing ledgers cannot
substitute for the current FSC-based quality contract. Never run them unchanged
as qualification or authorize baseline writes implicitly.

Quantitative gates and detailed historical findings remain in the
[program archive](../math/em_parity_program.md),
[parity notes](../math/relion_parity_agent_notes.md) and
[completion records](../math/em_parity_best_metrics.md). Their old next actions
are historical context. The new engine remains a later milestone.
