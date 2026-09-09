# Current EM development scope

Current decisions belong here; immutable details stay behind links. The previous
status page is preserved [byte-for-byte at structural checkpoint f0a8804e2](em_cleanup_history_20260909_f0a8804e2.md),
including earlier archives. Historical next actions and job observations are not
current assignments. Read live source/ownership before acting.

## Milestone and invariants

Complete RECOVAR cleanup/professionalization **EM first, GUI excluded**, before
new-engine development. Remove proven dead/duplicate code, clarify APIs/owners,
maintain contributor/agent guidance, and establish reproducible synthetic, real
and exactly-K4 accuracy/performance evidence. See [cleanup plan](cleanup_plan.md),
[codebase map](codebase.md) and [benchmark contract](benchmarks.md).

Preserve scientific defaults, casts/reductions/JIT boundaries, buffer lifetime,
non-EM APIs and saved formats during structural work. Keep numerical/runtime
repairs separate. Preserve canonical sampler Euler metadata; derive matrices.
Double is diagnostic, not a production parity remedy or proof of numerical noise.
Never widen tolerances/baselines or waive discrete differences without competing
scores/margins. Follow the [EM contract](../../recovar/em/AGENTS.md).

User priority: short-iteration parity, then final FSC/FSC-AUC; up to2× native
runtime provisionally. This changes neither accuracy gates nor the long-term
speed goal. Full-production-F32, broad quality and completion remain unproved.

## Source and ownership

| Item | Current identity or rule |
| --- | --- |
| Primary checkout | `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907`, branch `codex/integrate-pr180` |
| Latest production source | `f0a8804e2945ca36862491376db3820ada137df5`; the current archive/status change is docs-only. Actual HEAD/diff/untracked manifest takes precedence |
| Publication | [Draft PR179](https://github.com/ma-gilles/recovar/pull/179), stacked on [PR158](https://github.com/ma-gilles/recovar/pull/158), pinned base `44d770de3f9336ab2f3f6a34203394bae8d1aeed`; em_clean is sole integrator/publisher |
| Coordination | [Compact handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/CURRENT_TASK.md); [board and live scopes](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md). EM paused; VDAM owns private evidence, em_clean shared source/docs/publication |
| Frozen scientific source | `4f9a194923b084c649c7d9ce929eec7ae9f78902`, private `recovar_vdam_quality_prefix_integrated_20260909`; later cleanups are outside its run scope |

No overlapping source/build writers. Preserve frozen inputs/checkouts/binaries.
Shared RELION source/build changes require coordination; no lock is granted here.
Leave physical local GPU0 free; use only immediately verified idle1–3 by UUID.
Inside Slurm retain scheduler visibility. Do not duplicate peer jobs or analyses.

## Agent efficiency package — September 9

Compact handoffs and batched validation/publication are active. Opt-in Astra/high
and Terra/medium reader selection were verified, but the reader inherited write
permissions and no child-close tool was exposed. Delegation remains disabled;
direct Astra is the fallback. Worker execution, fresh read-only recovery and
long-term savings remain unverified. No automatic model polling/wakeup promise.
See [agent workflow](agent_workflow.md) and the board's delegated setup record.

## Engineering work and recent evidence

Latest structural batch `dd65b3563`/`f0a8804e2` simplifies candidate caches/class
assembly, removes two unreachable error arms and two compact-pair adapters, and
puts pair materialization beside host bucket builders. **78 fewer production
lines**, no numerical/default/scheduling change. Final30 focused CPU and38 guard
cases pass;13/17 archived-control cases and exact host/alias/error comparisons
pass. [Receipt and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/compact_pair_adapters_20260909/result.json).

Separate integrations remain distinguished in the
[archived integration ledger](em_cleanup_history_20260909_f0a8804e2.md#engineering-work-and-recent-evidence):
compact CTF, rigid reporting, opt-in F32 M/DC/CLI route, and projector test-contract
repair. Canonical Euler repairs479e888f9/db52d1bca and publication1624dd396 are
integrated. Private precision907 remains unmerged. The inherited F64/C128 M default
is preserved; opting into F32 M does not close higher-precision construction,
noise/prior and other stage boundaries. See [precision review](vdam_precision_review_20260909.md).

Frozen4f9 fresh prefix20 job13652879 passes both map conditions at all21 checkpoints;
one coarse count58/57 at14/image109 remains unclassified. Its separate natural200
pair13653485 now passes **both map conditions at all201 checkpoints**:

| Frozen4f9 map quantity | Result |
| --- | ---: |
| Minimum cross-engine FSC-AUC | 0.9997253944709251 at193, gate >=0.999 |
| Minimum registered-GT AUC delta | -0.0002512155449481135 at173, gate >=-0.002 |
| Final cross-engine FSC-AUC / GT delta | 0.9997294862648998 / -0.00009942153684500132 |

[The reusable archive](evidence/vdam-full200-4f9-20260909/README.md) preserves all1,005
curves/integrals, source/build/input pins, commands and a scratch-independent CPU
audit. Admission verified exact integrals, complete finite inventory, evidence
hashes and rejection of missing/corrupt archives. VDAM independently recomputed
worst and final curves from raw MRCs. This qualifies the map conditions on one
3k/128 K1 cell only; it is not strict-state or later-source acceptance.

## Unresolved validation gates

**The current selected source is not scientifically or performance qualified.**

- Frozen4f9 strict state:3,726 coarse-count mismatches/112,400 updated rows;
  first32/image313. First selected Pmax gap>=1e-3 at61/image2605; maximum0.699197
  at154/image1332, with large late pose/origin differences. Classes match, but
  fine support/competing margins are missing; no noise waiver.
- Counter187 differs0/5. [Source-bound peer audit](evidence/vdam-full200-4f9-20260909/late_counter_scope.md)
  proves it is monitor-only in this fixed200 gradient InitialModel configuration.
  It remains a strict-state discrepancy; **no ordinary auto-refine/K4 waiver**.
  Four native adaptive fields remain uncaptured. Map agreement cannot supply them.
- Current-source robustness, characterized real-particle confirmation, production
  >=100k/256 K1 and exactly-K4 same-GPU completion pairs remain missing. K4 audit
  13560356 failed2:0; both arms first missed the FSC gate at10/class2. No automatic
  retry. K2/K15 are not substitutes. Shared SPA/ET/outlier/downstream gates remain.
- Historical failures remain failed: API13641893 has6 failures; older13634313 has12;
  normalization13636581 has4 GPU bytewise failures; PR180 CPU has25 failures;
  K1 matched-noise replay has6 Pmax failures with missing margins/oracle identity.
  Partial targeted repairs do not qualify entire panels. Additional source-CTF,
  block-map, global-window and repeatability limits remain in the
  [complete failure ledger](em_cleanup_history_20260909_f0a8804e2.md#unresolved-validation-gates).

Keep [quantitative gates](../math/em_parity_program.md) unchanged. Completion needs
per-class Hungarian matching, convergence/finalization evidence and full precision
inventory, not only successful execution or final maps. Preserve the reviewed
final-grid-correction default during cleanup; its strict-target discrepancy needs
separate scientific qualification.

## Frozen jobs and representative performance

Real10076 10k/256 prefix20 **13654154 completed0:0** (sacct checked September9),
on the same frozen4f9 source. Cross-FSC/state analysis remains peer-owned and
pending review. Real GT is unavailable; no absolute-accuracy claim or duplicate job.

| Timing evidence | Result and limit |
| --- | --- |
| Frozen4f9,3k/128 H100 natural200 | 452.295/303.905s=1.48827652×. One pair/order; includes asymmetric candidate harness overhead, cold work/I/O. No GPU-memory measurement; storage contention unmeasured |
| Older8ab1a44be,100k A100 pairs | Forward1.647590×, reverse2.036693×; paired geometric1.831839×, predates compact CTF, quality unqualified |
| Privatebae959dab,3k/128 H100 | 1.436880×; not current100k or an isolated F32-M speedup |

See [archived timing boundaries](evidence/vdam-full200-4f9-20260909/README.md#state-precision-and-performance-limits)
and the [earlier performance ledger](em_cleanup_history_20260909_f0a8804e2.md#frozen-jobs-and-representative-performance).
Real10073 frozen-source evidence remains in the [real-data review](k1_real_window_review_20260909.md).
Next: review the peer real10076 handoff when ready; continue bounded structural
cleanup with proportional CPU checks and batched publication. No new GPU/build
work or change to scientific defaults is implied by this status page.
