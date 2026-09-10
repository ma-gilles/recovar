# Current EM development scope

Current decisions belong here; update this page when a decision changes, not for
every test or publication. Detailed receipts belong behind links. The
[previous page is preserved byte-for-byte](em_cleanup_history_20260910_8d157f765.md)
at `8d157f765`; its historical next actions are superseded here.

## Milestone and invariants

Complete RECOVAR cleanup **EM first, GUI excluded**, before new-engine work:
remove proven dead/duplicate code, clarify owners/APIs and establish reproducible
synthetic, real and exactly-K4 accuracy/performance benchmarks. The goal remains
incomplete. See [cleanup plan](cleanup_plan.md), [codebase map](codebase.md),
[benchmark contract](benchmarks.md) and [EM operating rules](../../recovar/em/AGENTS.md).

Structural changes preserve defaults, casts/reductions/JIT order, buffer lifetime,
non-EM APIs, saved formats and independent references. Canonical source Euler
angles and host pixel geometry stay metadata; derive computation arrays from them.
Double is diagnostic, not a production remedy or proof of noise. No tolerance or
baseline changes. User priority is short-prefix parity then final FSC, with up to
2× native runtime provisionally; this does not waive accuracy or completion gates.

## Source and ownership

- Primary: `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907`,
  branch `codex/integrate-pr180`. Sampling/initialization cleanup follows reviewed `8d157f7651ca44a7d625a4c50e1b763bf79298b8`.
  Recheck HEAD, diff and untracked files; earlier trajectories do not qualify it.
- **em_clean is sole integrator/publisher**, [draft PR179](https://github.com/ma-gilles/recovar/pull/179)
  on pinned PR158 base `44d770de3f9336ab2f3f6a34203394bae8d1aeed`.
  [Compact handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/CURRENT_TASK.md)
  and [ownership board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md)
  govern assignments. EM is paused; VDAM owns private scientific evidence.
- Preserve frozen checkouts, jobs, inputs and binaries. No shared RELION writer
  lock is granted here. No active em_clean job; peer job state must be checked
  before acting, not inferred from historical records. Never duplicate peer work.
  Leave local GPU0 free; only immediately idle GPUs1–3 by UUID; respect Slurm visibility.

## Engineering work and recent evidence

The current sampling/initialization batch consolidates coarse sizing, initial
resolution seeding and explicit schedule validation in their existing owners.
Duplicate checks/calculations are removed; defaults and state order are preserved.
All105 combined CPU cases pass; exact old/new scalar, validation and state traces
are recorded in the [batch receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/initial_resolution_owner_20260910/result.json).
This remains structural qualification only.

An earlier runtime batch gives follower state one owner, moves reconstruction
captures to their owner and removes three test-only diagnostics from runtime.
Controller −126 lines, net production +59; **95 combined CPU cases pass**, with
12 exact paired follower updates and 64 exact capture comparisons. This is
structural evidence, not trajectory qualification.
[Receipt and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/follower_state_single_owner_20260910/result.json).
The subsequent benchmark admission fix rejects invalid particle IDs/class labels:
26 CPU cases pass, 80 valid STAR results unchanged. [K4 audit](benchmarks.md#k4-audit-integrity-and-reviewed-saved-comparisons--september-10).

Canonical Euler/pixel repairs, compact CTF, rigid reporting and opt-in F32 M/DC/CLI
capability are integrated. **Private precision907 remains unmerged**; inherited
F64/C128 M defaults and other higher-precision stages remain disclosed.
[Precision review](vdam_precision_review_20260909.md).
All earlier cleanup receipts and line counts remain in the archive above.

## Unresolved validation gates

**Moving HEAD is not scientifically or performance qualified.** Map passes below
apply only to their frozen source and fixture; strict state is a separate gate.

| Evidence | Established result and remaining limit |
| --- | --- |
| [Frozen4f9 synthetic K1, 3k/128, full200](evidence/vdam-full200-4f9-20260909/README.md), job13653485 | All201 map gates pass: minimum cross-AUC .9997253945, worst GT delta −.000251216. Strict state: 3,726 coarse-count mismatches (first32), selected Pmax gaps from61, maximum .699197, late pose/origin differences. Missing fine support/margins; no noise waiver |
| [Frozen5ca9 real10076 K1, 10k/256, prefix20](evidence/vdam-canonical-pixel-prefix20-20260910/README.md), job13664081 | All21 cross-map gates pass, minimum .9999968978; map0 byte-exact. 57 count differences from3; 294 selected-row Pmax gaps >=.001 from13. No GT, timing ratio or full200 acceptance |
| [Frozen5ca9 real10076 full200](evidence/vdam-real-full200-5ca9-20260910/README.md), job13664965 | All four final cross-engine AUCs .968354–.974833 fail .999; native/native .975107 also fails. Twelve raw-map AUCs at100/200 independently exact; no GT/repeat-band waiver. Descriptive process ratios1.277–1.447×; not current-source acceptance |
| [K4 saved comparisons](benchmarks.md#k4-audit-integrity-and-reviewed-saved-comparisons--september-10), reported5ca9, 20 iterations | 3,200 saved curves/classes rechecked. Original synthetic minimum .99768012 and real .64899880 fail. Synthetic repeat passes; closest real repeat .99833338 still fails at20. Source/build and raw-map admission incomplete; native variation is not a waiver |
| [Robustness screening](benchmarks.md#robustness-gt-curve-review--september-10) | 64 GT integrals rechecked; original13/100,22/200,32/200 fail −.002 screening. Case22 fails8/12 repeat comparisons. Independent per-map alignments are not the prespecified shared-transform GT gate; source/build admission remains open |

Rejected explanations: canonical pixel narrowing was a demonstrated metadata bug
([scalar causal gate](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10076_pixel_prefix4_20260909/RESULTS.md)),
not a reason to promote arithmetic to double. Case22 candidate repeats are **not
byte-identical**: 97,376 final voxels differ. Correlation/repeat-band summaries
cannot close FSC or discrete gates. No deterministic-accumulation rewrite follows
without actual score/support evidence. Counter187 is
[monitor-only in fixed200 InitialModel](evidence/vdam-full200-4f9-20260909/late_counter_scope.md),
not an ordinary auto-refine/K4 waiver; four native adaptive fields are uncaptured.

Historical failures stay open: API13641893 has6 failures (older13634313:12),
normalization13636581 has4 GPU bytewise failures, PR180 CPU has25 failures, and
K1 matched-noise replay has6 Pmax failures with incomplete margins/oracle identity.
K4 job13560356 failed2:0 at10/class2. Partial repairs do not qualify those panels.
[Complete failure ledger](em_cleanup_history_20260909_f0a8804e2.md#unresolved-validation-gates).
The optional M-step rotation override cleared by class-prior layout remains a
[separate correctness question](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k_class_input_owner_20260909/result.json).
Buffer lifetime changes likewise require measured evidence, not cleanup assumptions.

Completion still needs current-source robustness, real confirmation, production-F32
inventory, >=100k/256 K1 and exactly-K4 matched-GPU pairs, Hungarian per-class
results, convergence/finalization and shared downstream checks. Keep
[quantitative gates](../math/em_parity_program.md) unchanged. Preserve the reviewed
final-grid-correction default; its strict-target discrepancy needs separate qualification.

## Frozen jobs and representative performance

Frozen4f9 full200 H100: 452.295/303.905s = **1.4883×**, one3k/128 pair with asymmetric
harness/I/O and unmeasured contention/memory. Older8ab100k A100 paired ratio **1.831839×**
predates compact CTF and is quality-unqualified. Reported K4 slowdowns (~20× synthetic,
~8.9× real) need source-closed timing review. None establishes moving-tip performance.
For newer pending evidence consult [VDAM status](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/status/vdam.json);
this page does not schedule or authorize duplicate runs.

## Next action and efficient execution

Resume one bounded structural package from the cleanup plan: inventory remaining
controller/helper duplication, select a behavior-preserving simplification with
clear callers and focused checks, then implement/review one cohesive batch.
Keep first-divergence numerical diagnosis and remaining real-full200/source/build
admission separate; peer summary labels are not acceptance.

Use [the workflow](agent_workflow.md), existing `scripts/em_work_package.py` receipts,
focused CPU tests and one combined validation/publication per package. Reuse the
[verified native CPU binding](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/native_binding_cpu_restore_20260909/result.json)
only after checking its source/dependency/loaded-file pins; do not rebuild it implicitly.

## Agent efficiency package — September 9

Direct Astra medium remains the default for this workstream. Built-in delegation
is disabled; isolated Terra smokes passed but the reviewed small-task pilot used
1.625× input and 1.667× elapsed time versus direct Astra. No token savings proved.
Compact handoffs, scripted receipts and batched publication continue. No automatic
wakeup/model-polling promise. [Setup, measured limits and recovery](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/delegated/README.md).
