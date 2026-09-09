# Current EM development scope

This is the current workboard, not a run history. Read the relevant linked
evidence for the active change. The [September 9 archive](em_cleanup_history_20260909.md)
preserves the previous status verbatim, including failures and superseded next
actions. Earlier history remains in the
[September 8 archive](em_cleanup_history_20260908.md).

## Milestone and invariants

Complete cleanup and professional development setup across RECOVAR, **EM first;
GUI and the new HIA engine are excluded**. Remove demonstrated dead/duplicate
code, clarify ownership/APIs, maintain consistent agent/contributor guidance, and
establish reproducible synthetic, real-data and K-class accuracy/performance
evidence for the selected source. See the [cleanup plan](cleanup_plan.md),
[codebase map](codebase.md) and [benchmark contract](benchmarks.md).

EM APIs may change with callers/tests/docs migrated together. Preserve non-EM
APIs, saved formats, scientific defaults, numerical behavior, reduction order,
JIT boundaries and memory lifetime during structural cleanup. Keep proposed
numerical/runtime repairs separate until authorized.

**Production EM remains float32. Double is diagnostic only.** A smaller double
gap does not prove roundoff. Preserve deliberate higher-precision metadata,
host calculations and necessary non-EM mathematics. Do not widen tolerances,
change baselines, dismiss discrete/convergence mismatches without evidence, or
use map correlation in place of FSC/FSC-AUC. Follow the
[EM operating contract](../../recovar/em/AGENTS.md).

## Source and ownership

| Item | Identity or rule |
| --- | --- |
| Implementation | `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907/`, branch `codex/integrate-pr180` |
| Latest production cleanup | `b3b51f0c8`: diagnostic reference gating/map loading joins the replay validators in `relion_replay.py`; controller 7,814 lines |
| Authorized performance integration | `5a39eab29` merges compact-CTF `b1d57608d`; host gather before stacking, full-grid default preserved |
| Latest test repair | `44d414a6b`: stale InitialModel callers and native source guards repaired; production unchanged |
| Pinned PR158 control | `44d770de3f9336ab2f3f6a34203394bae8d1aeed`; preserve unchanged |
| Incorporated history | PR180 `1e2f229b3`; cleanup anchor `0b52c995a`; VDAM merge `c38fc62f0`; separate global-window correction `0219caa35` |
| Publication | [Draft PR179](https://github.com/ma-gilles/recovar/pull/179), stacked on [PR158](https://github.com/ma-gilles/recovar/pull/158); em_clean is sole integrator/publisher |
| Live coordination | [Shared board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md); exact scopes/handoffs supersede historical grants |

Read actual HEAD, branch, dirty fingerprint and untracked manifest before tests
or claims. This table is a checkpoint record, not a moving HEAD pointer. WIP
draft publication with incomplete checks is user-authorized; merging and
scientific gates are not waived. The combined branch includes behavioral
integration changes as well as structural cleanup.

em_clean owns shared docs/status and publication. EM remains paused with frozen
source/jobs untouched. User-authorized compact-CTF `b1d57608d` is incorporated by
merge `5a39eab29`, preserving the separate performance commit. All six transferred
files match the frozen candidate byte-for-byte. Combined-source CPU checks pass
241 cases (23 GPU deselected) plus the 38-case guard. Reviewed H100 13638270
compares four synthetic configurations, six pairs each, with all five operands
and coarse scores byte-exact. This is **not trajectory/runtime qualification**.
See `handoffs/em_clean_compact_ctf_integration_20260909.json`; the private
candidate and its binaries remain frozen.

VDAM's separate two-path full-float32-product candidate `907b02ce` is now frozen
and reviewed: exactly two `HIGHEST` contraction keywords differ in production;
all 32 recorded H100 cases pass (13638400), after two expected-red cases
(13638324). Native-cutoff candidate replay 13638431 is diagnostic only. See
`handoffs/em_clean_vdam_full_float32_review_20260909.json`; full prepared-state,
trajectory and speed-cost checks remain open, with no shared production adoption.
Raw-prefetch source is unassigned. The RELION header lock is
released: five diagnostic insertions remain, and shared benchmark binaries plus
private native captures/builds stay frozen. No shared native writer/build is
assigned. Local GPU 0 is always reserved; check GPUs 1–3 immediately before use
and restrict by idle-device UUID, or use Slurm visibility. Avoid duplicate jobs.

## Engineering work and recent evidence

The latest audit repaired two stale test callers. Native source guards now
inspect the complete denominator argument list and each FFI handler's own macro,
preserving static/runtime buffer counts 17/18. Production code is unchanged;
all other test statements and numeric assertions survive explicit owner mapping.

| Checkpoint | Executed evidence | Limit |
| --- | --- | --- |
| Compact-CTF integration `5a39eab29` | 241 CPU passed/23 GPU deselected (31.13 s), 38-case guard; all peer cases retained and six transferred files exact; peer H100 13638270 byte-exact operands/scores | Full-image powerClass and source precision preserved; allocation microbenchmark only, no end-to-end or trajectory claim |
| Reference-replay ownership `b3b51f0c8` | Same 39-case control/candidate inventory, including tiny K4 map loading and state-swap order; 38-case CPU guard. Exact function/retained-controller AST; no new Ruff findings | Controller −95 lines by relocation; logger namespace follows replay owner; no GPU or trajectory claim |
| InitialModel test repair `44d414a6b` | Fourier-window control 49 passed/3 failed, then 52 passed (8.73 s); coarse-audit control 8 passed/4 failed, then 12 passed (2.75 s). Identical inventories; four in-memory native-contract mutations rejected; no new Ruff findings | CPU/test-source contracts, not compiled CUDA or trajectory qualification |
| Replay-order ownership `4da8954ae` | Same 24-case control/candidate inventory, 493 deselections; 38-case CPU guard. Separate parser-caller repair `1895bc50e`: 28 passed | AST-exact relocation; controller −73 lines, not net deletion |
| Diagnostic selectors `acbf6545e` | Same 93-case control/final inventory; 38-case guard; AST-exact relocation | Controller −126 lines; diagnostic logger namespace follows owner |
| Unused helpers `b52a78705` | Eight functions removed, 202 production lines; same 309 passed/1 opt-in CUDA skip/10 GPU deselections before/after | Skip unqualified; combined branch remains larger than PR158/PR180 |

Static review covered 1,404 tracked non-GUI Python files, 3,229 explicit EM
import names and 2,213 accesses through imported EM modules. Both stale callers
were confirmed by failing tests before migration. This does not prove dynamic
imports/registration or every runtime path. A tracked-reference scan found no
further unreferenced, undecorated top-level private EM function candidates;
do not remove code merely to reduce line counts.

Compact-CTF integration commands, fingerprints and case inventories are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/compact_ctf_integration_20260909/validation.json`.
Its CPU logs use `compact_ctf_combined{,_guard}_20260909` under the same CPU run
root below. No duplicate GPU job was launched.

Reference-replay commands, fingerprints and AST/case inventories are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/reference_replay_owner_20260909/validation.json`.
Control/after/guard logs use `reference_replay_{control,after,guard}_20260909`
under the CPU run root below. The previous test repair’s commands, fingerprints,
original failures and mutation/AST audits are in
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/em_import_audit_20260909/validation.json`.
CPU logs/XML are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/{stable_window_import,coarse_audit_import}_{control,after}_20260909/`.
Use fresh output roots when reproducing recorded commands. Earlier evidence is
in the [dated archive](em_cleanup_history_20260909.md); the original full-code
review and decisions remain under `hia_source_review_20260906/`.

Next, continue controller/state/kernel ownership and duplicate-code review with
bounded changes and proportional tests. Parsers with different blank/unknown-
token semantics are not interchangeable. Keep production source frozen during
its tests/jobs. Group changes into checkpoints for broader qualification; do
not run completion workloads after each helper edit.

## Unresolved validation gates

**The selected source is not scientifically or performance qualified.** Historical
checks qualify only their recorded source. This consolidation waives no failure,
skip, baseline or tolerance.

| Evidence or gap | Current interpretation |
| --- | --- |
| Frozen combined `d21f52d72` shared CPU | 194 passed, 3 GPU deselected; selected contracts, not complete shared workflows |
| H100 native 13634222 | 222 passed, 1 skipped; inherited rectangular CUDA/JAX ordering case remains unqualified |
| H100 API 13634313 | 592 passed, 12 failed; nine original failures have targeted follow-ups, not a fresh full-panel pass |
| Normalization 13636581 | Dtype migration CPU 8/8; GPU 4 passed/4 failed bytewise spectrum cases. Strict failures remain |
| Normalization replay 13636814 | Exact first-bucket inputs; ordinary results vary in 11/12 same-setting and 14/16 crossed pairs, max norm gap 0.00018310546875. Captured deferred equality does not erase uninstrumented failures; diagnostic precision lane |
| PR180 CPU 13623235 | 6,452 passed/340 skipped/25 failed; targeted repairs/environment reruns do not create a fresh suite pass. [Reconciliation](evidence/pr180-unit-checkpoint-20260908/README.md) preserves CPU-only failure and inventory mapping |
| K1 matched-noise replay | Six Pmax rows still fail; candidate margins and generating oracle identity are missing. [Paired failures](evidence/pr180-k1-noise-state-20260908/README.md), [targeted captures](evidence/pr180-k1-targeted-capture-20260908/README.md). Maps cannot override particle failures |
| Synthetic/real repeatability and global window | Current float32 K1 support/Pmax/pose/convergence and real10076 repeatability remain unresolved; full FSC acceptance of `0219caa35` is pending |
| Exact K4 and shared workflows | No selected-source ≥100k/256 K1 + exactly-K4 same-GPU completion pair; K2/K15 proxies do not count. Shared SPA/ET, outlier/downstream and speed/memory coverage remain outstanding |

Completion requires FSC curves/FSC-AUC against GT and RELION, tie-aware discrete
comparisons, exact convergence/finalization, Hungarian matching and per-class K4
results, identical fixtures/seeds/masks/initial maps and matched GPU models.
Close synthetic K1 trajectory parity, then a characterized real-particle check,
then exact K4. Preserve the reviewed final-grid-correction default during cleanup;
its difference from strict parity requires separate scientific work. Quantitative
gates remain in the [program archive](../math/em_parity_program.md);
[parity notes](../math/relion_parity_agent_notes.md) and
[completion records](../math/em_parity_best_metrics.md) retain detailed evidence.

## Frozen jobs and representative performance

Slurm was checked on September 9 UTC: K4 pair **13560202** was running on
`della-l08g4` (12:37:54 elapsed), audit **13560356** pending. K1 real-data window
job **13610518** was running on `della-h20g1` (12:17:12), with **13610539** and
**13629099** pending. Poll these exact jobs before using this status as current.
Preserve their frozen sources/outputs; they do not automatically qualify later
commits. No new jobs were launched for this audit.

Frozen `8ab1a44be` completed RECOVAR2 in **14,399.809 s** versus native1
**7,966.705 s**: **1.8075× unpaired**, on different physical A100s, launch order
and shared-host conditions. Native2 was running at that report timestamp; its
current state is not inferred here. The separate instrumented profile took
14,764.576 s. Backend compilation was **195.662 s / 1.33%** of profiled exclusive
time (excluding tracing/lowering). Prefetch queue waiting was 5,777.491 s; CTF
host `np.stack` was 3,709.627 s exclusive versus 69.060 s in the native CTF
binding. The helper's 4,557.538 s cumulative overlaps its children. These costs
are not guaranteed removable time; queue waiting is not proven disk I/O.
No paired speed or current-source quality acceptance follows.

The frozen 21-checkpoint cross-engine diagnostic reports final FSC-AUC
**0.9924569** and minimum **0.9900306 at iteration 70**. Iteration 40 is the first
saved checkpoint below 0.999, not the first divergence or an executed acceptance
gate. Only iterations 0,10,…200 are available. Initial cross-FSC 1 does not imply
bit-identical initial maps. Raw GT metrics are invalid for quality because
registration is unverified; the later shared rotation/mirror check omitted
translation search and did not establish adequate registration. Exact curves,
commands and producer provenance are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_speed_quality_inventory_20260909/refresh_030533/`;
see board acknowledgment `em_clean_vdam_cross_fsc_ack_20260909.json` for limits.
The 8.68 → 1.74 ms compact-CTF CPU allocation measurement is a microbenchmark,
not an end-to-end runtime prediction.

Native build **13636510** completed; bounded capture **13636553** obtained exact
inputs/residual rows, then deliberately stopped. Wavg replay must use masked
`Fimgs` planes 0/1, not unmasked planes 2/3. Later private arithmetic diagnostics,
including the full-float32 cutoff-product proposal, remain outside trajectory
acceptance and structural cleanup.

Report and hashes:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_native_residual_audit_20260909T024139Z/report.md`,
plus `handoffs/em_clean_vdam_terminal_100k_ack_20260909.json` on the board.
The new engine remains a later milestone.
