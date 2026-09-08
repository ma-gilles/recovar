# Current EM development scope

## Shared PR179 integration workflow — September 8

The user authorized one integration workflow for `em_clean`, `em`, and `vdam`.
`em_clean` is the sole publisher of `origin/codex/recovar-structural-cleanup`
(PR179). Private branches hand off commits for review; do not force-push or
edit another session's worktree. The live coordination board is
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md`.
Read its `status/em_clean.json`, `status/em.json`, `status/vdam.json` and exact
path ownership before edits. Each session owns its status record; updates use
an atomic rename. The board records the immutable published integration HEAD.

The completed cleanup anchor is `0b52c995ab1a01e04ac0642149f0b929d43ee1d2`.
It separates BPref diagnostic state and capture writers from sparse scoring,
removes duplicate hashing, and preserves named results and numerical defaults.
The shared checkpoint also retains VDAM's published BigJIT cache fix and notes
through `e727d9c9814f9bce823d47be9cd6d1c16e2f0e12`. No new tests or GPU jobs
were launched for this coordination/publication request; existing evidence
qualifies only its recorded source, and the combined checkpoint is unqualified.

After publication, VDAM owns reconciliation of its 24 explicitly requested
paths in its private integration checkout. `em_clean` holds further overlapping
cleanup, including the proposed K-class package-import migration, and retains
shared status documentation and publication ownership. EM remains paused and
its evidence worktree and running jobs stay untouched. VDAM must preserve the
separate global pass-one window correction `a0a86f19e9` during reconciliation;
its full K1 FSC acceptance remains pending. The hashed patch and handoff are
recorded on the board. The unresolved VDAM catch-up merge is not part of this
published cleanup checkpoint.

## Shared VDAM performance transfer — September 8

The user authorized transferring applicable VDAM performance fixes onto PR179
while preserving its API and ownership cleanup. The first bounded hypothesis
is that equivalent dataset wrappers should reuse the shared local BigJIT
executable: its preprocessing does not call the dataset-bound `process_fn`,
which currently adds an irrelevant identity to the static compilation key.
The transfer keeps the full configuration on consumers that use preprocessing.
The port follows PR179 through `6ee49dc5c` and uses its named `LocalEMResult`
fields. Validation covers full local reconstruction/noise and score-only calls;
the prior static-key behavior is replayed at the same numeric boundary.
All scientific result fields agree exactly, including dtype. Both cache tests
failed before the fix (two compilations rather than one).

| Transfer validation | Result |
| --- | --- |
| Cache reuse, exact old-key replay, existing BigJIT/split comparisons | 4 passed in 27.77 s |
| CPU EM fast guard, including helper import boundary | 38 passed in 47.93 s |
| Named-result and K-class caller contracts | 22 passed |
| VDAM sparse adapter class/pseudo-halfset routing | 2 passed in 3.38 s with the matching RELION binding |

The first caller attempt had two missing-binding failures. Reusing the
source-matched PR180 binding (SHA256
`2c56e67c08df885fad762e0f70707f8dc6b89f6ee04033b3d4822f22db8c6a21`)
resolved them; no test was skipped or weakened. These CPU checks qualify the
bounded port, not GPU trajectories, map quality, or end-to-end ordinary EM speed.

### Active integration and next quality check

The transfer is published as `0b24aeb3c`. Active VDAM integration work now uses
`codex/pr179-shared-bigjit-20260908`, tracking
`origin/codex/recovar-structural-cleanup`, at
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr179_shared_bigjit_20260908`.
Fetch and reconcile the current PR head before publication. Preserve the
separate running VDAM source and its results. Rechecking dense/global EM found
that its BigJIT already takes explicit arrays and scalar options without the
dataset configuration, so the same cache-key edit is not applicable there.

The completed historical VDAM noise audit (`13628018`, metadata `13628114`,
maps `13628020`; source `8ab1a44be1`) localizes the first discrete disagreement
to iteration 32, selected row 68 / particle ID 1367. Candidate repeat 2 chooses
rotation ID 163798; both controls and candidate repeat 1 choose 111738.
Selected particle identities agree. Competing-score margins are unavailable;
this is not yet an adjudicated numerical tie. Continuous-state gates already
fail at iteration 2, including control repeats.

Candidate/control cross-FSC gates first fail at iterations 80 and 71, with
120 and 130 failing checkpoints respectively. Control-repeat cross-FSC also
fails at 120 checkpoints from iteration 80. RELION-repeat map checks pass all
201 checkpoints. No GT-AUC delta gate fails, but that cannot override failed
state/cross-FSC checks. These results do not qualify PR179's source.

The next bounded quality hypothesis is that particle 1367's pose disagreement
depends on incoming trajectory state rather than the same-state noise toggle.
Capture the complete iteration-31 state during uninterrupted replay and hold
iteration-32 candidates fixed across off/on and same-path repeats. Compare
scores and posterior margins before interpreting the pose decision; move
earlier if fixed-state arithmetic agrees. Existing map/model/data STAR files
alone have not been proven to preserve the complete momentum/adaptive state.
The read-only locator and input hashes are in `quality_triage.py` and
`quality_triage.json` under the transfer evidence root below. No new GPU replay
has been launched while local GPUs 1–3 remain occupied by the 100k panel.

The VDAM experiment source remains separately frozen at `8ab1a44be1` while
100k/256 timing pairs and a compilation profile run on local A100 GPUs 1–3.
The user now accepts approximately 1.5× RELION runtime as a threshold for
prioritizing VDAM quality parity. This changes work priority, not scientific
tolerances, the definition of equal speed, or completion requirements. The
completed 3k/128 native-noise experiment measured 1.452–1.460× RELION but failed
metadata equivalence; it cannot qualify representative speed or quality.

The reusable design lessons are to exclude unused Python identities from JIT
keys, keep logical counts separate from physical capacities, and preserve
device residency across substantial stages. Existing translation and
intermediate-shape variants still need measured treatment. Padding, different
reduction orders, and new CUDA posterior/noise routes remain separate
experiments until both numerical equivalence and runtime benefit are shown.
Do not import the VDAM development branch wholesale over the structural work.

PR179 carries a marked status block for these transfers. Refresh it from the
current remote body and preserve concurrent cleanup updates. Local transfer
evidence is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em2_recovery_20260905T2044Z/pr179_sync_20260908/`.

## Cleanup scope

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
| BPref capture context, validation and writers have a direct diagnostic owner | 50 baseline cases pass; 55 affected/hash cases and 38 CPU guard cases pass after migration. Exact source comparison preserves moved definitions and executable caller bodies through explicit owner mappings. One source-inspection assertion was migrated to the qualified call name |

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
   affected EM test modules. GPU passes all 566 cases with no skips; CPU has
   561 passed and the five expected GPU-marked skips. The CPU job itself failed
   because its outer reporter converted a test class name into a file path.
   `audit_saved_results.py` verifies all 566 saved JUnit identities and exactly
   the expected skip set on both backends without rerunning tests or rewriting
   that failed outcome. Source and native-library identities stayed unchanged.
   The driver, source/library hashes and case inventory are under
   `em_api_checkpoint_20260908/` in the review root; logs use the same directory
   name under `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/`.
   This qualifies the declared API/ownership test inventory, not an autonomous
   trajectory, performance or later source changes. Engine numerical statements and
   K-class computation are unchanged; the named result removes both positional
   decoders. It also fixes the related capture-routing ambiguity where an
   implicit profile could be returned as significant counts when hidden by the
   wrapper. Four new cases preserve explicit profile visibility/copy behavior
   and require actual counts. Evidence is in `pr180_local_named_result_20260908/`
   under the review root. The earlier separately authorized K-class routing fix
   remains recorded under `pr180_local_result_contract_20260908/`. Neither
   routing correction explains or waives existing benchmark failures.
   The next implementation checkpoint separates BPref diagnostics from sparse
   scoring. Its exact source comparisons, before/after cases and import-boundary
   checks are under `bpref_diagnostic_owner_20260908/` in the review root.
   Continue from its final focused results and caller inventory; preserve the
   diagnostic state lifetime, saved schema and production execution decisions.
   Its standalone import probe found that `dense_single_volume.__init__` still
   eagerly loads K-class execution and sparse scoring. Review and migrate the
   remaining K-class package re-exports next; the controller-import guard passes,
   but it does not prove that helper imports avoid every execution module.
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
