# Current EM development scope

## Frozen combined-source qualification — September 9

Checkpoint `d21f52d726c70acd27d99cbbfff65a79d341d497` is frozen in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr179_validation_d21f52d72_20260909`.
The current CUDA source builds for H100 (`sm_90`), and the current RELION
bindings build against archived source `f2c1a384400aec37dc6805856a5ba645650a44f1`.
Both libraries are pinned by hash outside shared builds. All 194 selected shared
CPU cases pass in 25.08 seconds, including the new InitialModel bindings,
configuration, build selection, metadata export and project/CLI contracts.
Three GPU cases are explicitly deselected from this CPU panel.

H100 job **13634222** on `della-h20g4` passes 222 native GPU cases in 62.51
seconds, but fails strict admission because the rectangular CUDA/JAX case is
explicitly skipped for a known pixel-ordering mismatch. That skip also exists
at pinned PR158 `44d770de3f`; it remains an unresolved coverage gap. No test
assertion failed, but the full 223-case native panel is not accepted. The
sequential launcher stopped before API execution. Independent H100 job
**13634313** completed the unchanged 604-case API panel: **592 passed, 12
failed**, no skips, in 686.27 seconds. Exact case identities and unchanged
source, libraries and baselines are verified. The panel is not accepted.
Two newly imported operand-order cases fail while patching the removed
`local_engine._sparse_pass2_diagnostics` alias. Test-only repair `21bb76086`
imports the canonical `helpers.sparse_pass2_bucketed` owner directly. Reversing
that import/patch-target mapping produces the identical complete test-module
AST; all assertions and production source remain unchanged. Both cases pass
on H100 in 3.53 seconds in targeted job **13634507**. The original failed
result remains preserved. Ten other failures remain for focused review:
three unbound projector-size closure cases, three normalization dtype/bitwise
checks, one result-comparison case and three source-contract guards. These
are failure descriptions, not conclusions about correctness or roundoff.

The launcher checks exact collected and executed case identities, rejects
unexpected skips, and verifies source,
native libraries and baseline files before and after each panel. Production
float32 defaults remain selected; existing float64 companion tests are diagnostic.

The first CPU attempt passed 193 cases and failed the spawned-worker case
because the test launcher ran pytest at import time. The corrected launcher
uses a `__main__` guard; all 194 unchanged tests then pass. Both attempts are
preserved. No production source, test assertions or baselines changed.

Reproduction configuration and scripts (`shared.sh`, `gpu.sbatch`, `inputs.json`):
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pr179_validation_20260909_v2/`.
Run `bash shared.sh` for CPU or `sbatch --parsable gpu.sbatch` for H100 after
copying the configuration to a new output root; drivers reject existing results.
Logs, XML and outcomes are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr179_validation_d21f52d72_20260909/v2/`.
The parent output directory retains the original launcher failure and build logs.
The coordination handoff `em_clean_frozen_native_validation_20260909.json`
records commands, inventories and hashes. These selected contracts do not
qualify full shared workflows, real K1, exactly K4 or performance parity.
The caller-migration handoff `em_clean_operand_caller_migration_20260909.json`
records its exact two-case command and dirty status-document fingerprint.
Its logs and XML are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr179_operand_caller_20260909/`;
the corresponding `pr179_operand_caller_20260909` directory under the review
root above contains the pinned inputs and `api_only.sbatch` reproduction script.

VDAM has resumed its private quality audit and speed monitoring at `5056e760d`,
based on d21. It owns `scripts/run_vdam_prepared_estep_replay.py` and
`tests/unit/initial_model/test_prepared_estep_replay.py`; production source is
read-only pending a bounded ownership request. Its replay and existing 100k
sources stay frozen. em_clean retains shared status/docs, publication and
GPU/API qualification. The coordination handoff
`em_clean_vdam_resume_ack_20260909.json` records the current scope and jobs.

## Structural cleanup after VDAM integration — September 9

Fourteen duplicated local binary-flag readers were replaced with direct calls
to `helpers.env_flags.parse_env_binary_flag` in `7b687dde8`. The engine shrinks
by 98 lines and production Python by 90 net lines. All 336 frozen comparisons
match return values, error types/messages and environment lookups. The engine
AST is unchanged after the explicit call mapping; existing tests retain their
assertions and parameters under that mapping. Permissive flag readers and
scientific defaults are unchanged.

The first affected run passed 322 cases and exposed one inherited frozen
packing-checksum failure. The pre-edit and current packing blocks were identical;
the only difference from pinned `6d11f325` was its former float32 rotation cast.
A separate test repair retains the original checksum and explicitly normalizes
only that reviewed expression. Four new live packing cases check float32 and
float64 rotation dtype/values; injecting the old cast into the test's extracted
AST makes both float64 cases fail. Production code is untouched by this repair.
The final affected CPU panel passes 327 cases in 21.38 seconds, retaining all
323 prior cases and adding four; 48 GPU cases are explicitly deselected. The
38-case numerical fast guard also passes. No GPU or Slurm jobs were launched.

Exact commands, source hashes and the original failure are recorded under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/` in
`binary_flags_affected_20260909`, `binary_flags_affected_final_20260909`,
`binary_flags_packing_migration_20260909`,
`binary_flags_packing_precision_red_20260909` and
`binary_flags_fast_guard_20260909`. The coordination handoffs
`em_clean_binary_flags_20260909.json` and
`em_clean_packing_checksum_migration_20260909.json` link reproducible comparison
and fault-injection scripts. The existing unrelated import-order lint findings
are unchanged; the shared parser and repaired test pass Ruff.

The materialized fine-grid significance mask now belongs to
`tests/helpers/fine_grid_significance_reference.py`. Only two test modules use
it, to check production lazy masks and explicit/complement support. The moved
70-line function is AST-identical; all other K-class statements are unchanged.
`k_class.py` shrinks by 72 lines including spacing. Both modules retain the same
228 collected cases and identical test bodies; all 51 affected K-class and
support tests pass in 4.73 seconds. No tolerance or baseline changed.

Evidence is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/`:
`fine_grid_reference_collect_before_20260909`,
`fine_grid_reference_collect_after_20260909`, and
`fine_grid_reference_affected_20260909`. Their `outcome.json` files record the
commands and source/native fingerprints. The coordination handoff
`handoffs/em_clean_fine_grid_reference_20260909.json` records reference identity,
caller review, case inventory and validation. The affected test command uses
the wrapper below with the full `test_k_class_joint_semantics.py` module and
the four explicit/complement support cases recorded in that outcome.

Following published integration `2b4596e99`, remove three unused package-level
K-class exports. Repository callers already import `KClassEMResult`,
`run_dense_k_class_em` and `run_local_k_class_em` from `k_class.py`. Definitions,
numerical implementations and saved module identities remain unchanged. No
lazy forwarding layer replaces the retired aliases.

The strengthened CPU guard reproduces the prior problem: a helper import loads
K-class orchestration, both engines, local BigJIT, significance and sparse
pass-two execution. After removing the exports, all six remain unloaded and
all 38 numerical fast-guard cases pass (47.02 s). The existing CLI subprocess
provenance test also passes, checking both accepted and rejected checkout
identities. Source and reference native-library hashes are unchanged during
each validation run. Ruff and shell syntax checks pass.

Exact commands use the existing evidence wrapper:

```bash
CHECKS=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pr180_integration_20260908/run_checks.sh
bash "$CHECKS" kclass_import_boundary_green_20260909 --fast-guard
bash "$CHECKS" kclass_import_boundary_cli_20260909 --run-integration -v tests/integration/test_em_runner_import_provenance_subprocess.py --tb=short
```

Recorded output roots are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/`:
`kclass_import_boundary_red_20260909`,
`kclass_import_boundary_green_20260909`, and
`kclass_import_boundary_cli_20260909`. Each contains logs and `outcome.json`
with source fingerprints and exact commands; green runs include JUnit XML.
The wrapper refuses existing output directories: choose a new label to rerun.
The coordination handoff records the zero-caller AST/text audit and current
ownership. This structural change adds no scientific acceptance claim; the
quality and shared workflow gaps below remain open.

## Combined integration checkpoint — September 9

Merge `7dba6abce728161b1cc491685962ec441c591a51` incorporates VDAM's
reconciliation `c38fc62f0` and the separate global-window correction `0219caa35`.
The production `recovar/` tree is identical to `0219caa35`. Named local results,
the sampling and BPref owners, and dtype-preserving diagnostic capture survive
the integration. The original global-window correction is `a0a86f19e9` from
EM's separate object store; its full K1 FSC acceptance remains pending.

VDAM's frozen selected CPU panel passes all 576 cases at clean `0219caa35`.
The seven global-window cases also pass after the adapted fixture demonstrated
three production failures before the fix. These checks include shared image
loading, Fourier utilities and STAR array serialization; they do not qualify
GPU execution or complete shared workflows. The combined merge independently
passes all ten selected packaging/build-contract cases in 10.56 seconds.
The manifest retains VDAM's complete explicit CUDA input list, including the
required header. Packaging tests build a wheel from an isolated source archive
and inspect both archives, so Git discovery cannot conceal omitted inputs.

VDAM's saved A100-target CUDA build matches the integrated CUDA sources. It
compiles only `sm_80`; neither GPU execution nor H100 qualification is implied.
Existing 100k runs remain frozen at `8ab1a44be1`, which is a different source.
Current-source real K1, exactly K4, full shared SPA/ET/downstream quality, speed
and memory acceptance remain open. Production EM remains float32; double is
diagnostic only. This is a draft integration checkpoint, not merge acceptance.

Evidence and exact reproduction scripts:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr179_vdam_reconcile_20260908/`
  (`run_frozen_cpu.sh`, `frozen_cpu.xml`, global-window red/green records and
  `cuda_build_result.json`).
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/vdam_combined_packaging_20260909/`
  (`tests.log`, `tests.xml`, `outcome.json`).
- The coordination board below records the published HEAD, evidence admission
  and current ownership. VDAM has frozen its reconciliation checkout for
  integration; do not edit frozen benchmarks or infer a new source assignment.

## Shared PR179 integration workflow — September 8

The user authorized one integration workflow for `em_clean`, `em`, and `vdam`.
`em_clean` is the sole publisher of `origin/codex/recovar-structural-cleanup`
(PR179). Private branches hand off commits for review; do not force-push or
edit another session's worktree. The live
[coordination board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md)
is on the shared filesystem.
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
recorded on the board. That catch-up merge was absent from the September 8
cleanup publication; the September 9 integration above now incorporates it.

### Integration review: callers outside Git conflicts

A static review of the incoming VDAM source `8ab1a44be1` and its unresolved
merge index found a named-result migration gap outside the 24 conflicted files.
`test_local_host_result_publication.py` still supplies a tuple-returning mock,
indexes the result positionally, and compares only tuple structures recursively.
Migrate these to `LocalEMResult` fields while retaining exact byte/dtype checks.

The incoming dense adapter needs separate owners for BPref context and its
live sparse-backend selector. A new accumulator-analysis test also imports a
BPref helper from its former sparse owner. Port the incoming semicolon selector
and optional capture fields to `helpers/bpref_diagnostics.py`; preserve the
current first-iteration topology parameter and decisions, image-index helper,
and capture precision. The incoming writer's explicit float32 translation cast
must not silently replace the current dtype-preserving capture.

Exact paths, source/index hashes, the three differing moved definitions and
review actions are in `handoffs/em_clean_integration_review_actions_20260908.json`
under the coordination root. The board's current assignment includes the
companion global-window test and these additional migrations; read its exact
path list before editing rather than relying on a historical path count.
This is static review evidence, not a combined-source test pass. The index was
unchanged during review; no source in VDAM's checkout or frozen runs was edited.

### Shared behavior and packaging review

The VDAM catch-up also changes shared image loading/prefetch, STAR export,
GPU detection, project registration and the process-wide compilation-cache
threshold. The translation/halfset export fix is a correctness repair, and
the cache threshold and buffer changes affect performance. Describe them
explicitly; an EM guard pass cannot qualify their shared callers.
`handoffs/em_clean_shared_validation_scope_20260909.json` on the coordination
board records 13 review areas, immutable source blobs and applicable checks.
Run affected checks first and broader shared workflows at the resolved frozen
checkpoint, preserving the existing source and hardware requirements.

Packaging fix `1014ca3b2` includes CUDA headers and source fragments in source
and wheel distributions. On isolated incoming source `8ab1a44be1`, the stronger
archive test fails before the fix because `noise_residual.cuh` is absent, then
passes after it. Standard isolated PEP517 wheel builds with the declared build
dependencies also omit the header before the fix and include it afterward when
started from a source archive. These version-1.0.0b1 wheels and the earlier
no-build-isolation version-0.0.0 archives are test artifacts. Live Git discovery
is a separate case; it must not conceal a broken archive manifest. No CUDA
kernel is compiled or scientifically qualified by these packaging checks.
Exact commands, archives, logs and member hashes are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_packaging_review_20260909/`.
The original failed test remains recorded. The integrator owns the packaging
change independently of VDAM's source reconciliation.

The permanent packaging fixture now copies source inputs into a private directory
without Git, stale egg-info or compiled objects, builds the sdist there, then
builds the wheel from that sdist. All nine packaging tests pass (7.26 s) with
source/native identities unchanged. The revised fixture still fails for the
original missing header and passes with the manifest fix on incoming source.
The whole-module log and source fingerprint are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/packaging_archive_fixture_20260909/`.

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
