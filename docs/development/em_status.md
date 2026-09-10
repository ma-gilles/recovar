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
  govern assignments. Consult peer status for current numerical work and jobs;
  historical paused states do not establish current ownership or availability.
- Preserve frozen checkouts, jobs, inputs and binaries. No shared RELION writer
  lock is granted here. No active em_clean job; peer job state must be checked
  before acting, not inferred from historical records. Never duplicate peer work.
  Leave local GPU0 free; only immediately idle GPUs1–3 by UUID; respect Slurm visibility.

## Engineering work and recent evidence

Frozen5ca9 K4 cache diagnostic13687592 has reviewed profiled wall times
799.98→204.62s (74.4% reduction), with40 saved iteration timings and45 evidence
files checked. This is candidate-only, not direct compile-time attribution or
quality acceptance. Cache entries include primitive specializations; shape
unification remains separately qualified performance work. Full200 K4 job13687409
was running when checked; no duplicate run.
[Review](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_cache_timing_review_20260910/RESULTS.md).

Replay numbering and cutoff helpers now live in `relion_replay.py`, with direct
controller/test callers. Moved helper ASTs and remaining controller AST match
the parent after ownership normalization; 147 CPU guard/affected cases pass.
[Receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/replay_boundary_owner_20260910/result.json).

Final-pass admission and gridding selectors now belong to `finalization_policy.py`;
the controller passes its logger explicitly. All168 old/new decision and warning
comparisons match, and the remaining controller AST is unchanged after owner
migration. The combined CPU guard/affected panel passes 113 cases. Defaults and
K-class convergence requirements are preserved; no new trajectory qualification.
[Receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/finalization_policy_owner_20260910/result.json).

PPCA cleanup removes59 net lines of unused private helpers across refinement
postprocessing and shared PPCA; remaining live arithmetic and public exports
are unchanged. Two native-dependent PPCA tests now declare GPU requirements.
CPU companion:11 pass,3 GPU skipped. On clean9b09516ce, all3 GPU cases execute
and pass on local A100 GPU1 with an explicitly built, sealed sm80 library;
source/binary/header hashes remain unchanged. The earlier CPU auto-build attempts
were stopped and preserved; no dispatch repair or full-pipeline acceptance.
[GPU qualification and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/ppca_gpu_9b09516ce_20260910/result.json).

Particle-state reporting now rejects null/blank image IDs before alignment and
shares its identity validator. Non-finite or negative comparison tolerances also
fail explicitly; zero and established finite defaults are preserved. Sixteen
malformed cases exposed missing validation;80 reporting CPU tests pass and30
valid reports are unchanged. Empty trajectory requests now fail explicitly;
reverse-ordered requests report the earliest checked divergence while preserving
row order. Two reproduced failures are fixed, with51 affected CPU cases passing;
this does not establish behavior at unsampled iterations. No engine or gate changes.
[Iteration receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/trajectory_iteration_admission_20260910/result.json).
[Identity receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/particle_identity_admission_20260910/result.json),
[threshold receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/particle_tolerance_admission_20260910/result.json).

Batch-plan adjustment logging now belongs to the immutable batch-plan type;
the controller retains estimator inputs and scheduling.100 affected CPU cases
and the CPU guard pass;48 log and108 controller comparisons are exact.
Controller −20 lines, net production +14; no numerical or performance claim.
The single-class sparse scorer also sheds12 net lines of unused compact-pair
mode flags, retaining early environment validation and the live K-class flags.
Four focused CPU cases and16 exact validation comparisons pass; the remaining
module AST is unchanged.
[Dead-state receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_dead_mode_flags_20260910/result.json).
[Batch reporting receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/batch_report_owner_20260910/result.json).

Final sampling-file admission/selection now belongs to the replay owner,
including searched-path provenance for the missing-file diagnostic. All111 final
CPU guard/controller cases pass;162 old/new file-state comparisons are exact.
The initial private extraction missed that diagnostic consumer; a real-branch
regression now covers it. [Receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/final_sampling_owner_20260910/result.json).

The current convergence batch gives pose-stack preparation one owner, reuses
`state.fraction_changed`, and removes unused rotation-count/HEALPix parameters
from the assignment API and its controller/PPCA callers. The metric is documented
as an index comparison, not an angular-distance threshold. All209 combined CPU
cases pass; the numerical bodies are AST-identical after argument migration.
The earlier200 pose-stack and48 fraction comparisons remain recorded. Local
checkpoint, not trajectory/performance qualification.
[Latest receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/assignment_api_20260910/result.json).

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
| [Frozen5ca9 real10076 K1 InitialModel, 100k/256, full200](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real100k_admission_20260910/review.json), job13683192 | Saved201 summary independently recomputed; worst115 and final200 raw-map AUCs exact. Minimum 0.920691882,140/201 below .999, final .993747847: fail. Paired process ratio 2.148386× exceeds provisional2×. Raw state at21 already has a count difference; at32 Pmax gap .02103 and2 count differences. No exact-through31, chaos/noise, GT or moving-source acceptance; full native/input closure remains open |
| [K4 saved comparisons](benchmarks.md#k4-audit-integrity-and-reviewed-saved-comparisons--september-10), reported5ca9, 20 iterations | 3,200 saved curves/classes rechecked. Original synthetic minimum .99768012 and real .64899880 fail. Synthetic repeat passes; closest real repeat .99833338 still fails at20. Source/build and raw-map admission incomplete; native variation is not a waiver |
| [Robustness screening](benchmarks.md#robustness-gt-curve-review--september-10) | 64 GT integrals rechecked; original13/100,22/200,32/200 fail −.002 screening. Case22 fails8/12 repeat comparisons. Independent per-map alignments are not the prespecified shared-transform GT gate; source/build admission remains open |

[E6 row942 coarse-cap review](evidence/vdam-coarse-cap-tie-20260910/README.md)
confirms measured0–3 ULP competing scores and exact threshold/support agreement
in six instrumented candidate histories. This local near-tie evidence does not
establish the incoming-state cause, native matched-input parity or a trajectory
waiver; six-digit scalar equality in the producer report is corrected.

[Saved-state repeat audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/same_state_admission_20260910/result.json)
independently checks nine candidate metadata comparisons from reported5ca9/job13665172.
All200 support counts match in each comparison; t12 row2765 differs in Pmax by
1.3709068e-6, and saved noise/power/BPref summaries differ. This is not exact
full-state agreement or proof of an M-step-only cause; native inputs/maps and
complete build provenance are outside this audit.

Rejected explanations: canonical pixel narrowing was a demonstrated metadata bug
([scalar causal gate](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10076_pixel_prefix4_20260909/RESULTS.md)),
not a reason to promote arithmetic to double. Case22 candidate repeats are **not
byte-identical**: 97,376 final voxels differ. Correlation/repeat-band summaries
cannot close FSC or discrete gates. No deterministic-accumulation rewrite follows
without actual score/support evidence. Counter187 is
[monitor-only in fixed200 InitialModel](evidence/vdam-full200-4f9-20260909/late_counter_scope.md),
not an ordinary auto-refine/K4 waiver; four native adaptive fields are uncaptured.

The dense/local fast guard now rejects undefined names before JAX startup.
A broader static scan found existing `picked_frequencies` use-before-assignment
in `recovar/em/heterogeneity.py:971`; outside this guard's scope, unfixed.
A [caller audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/legacy_pca_callers_20260910/result.json)
finds no repository caller of that78-line legacy function and no package export.
The main PCA pipeline uses a distinct implementation. An unapplied removal
candidate is preserved; external/dynamic uses remain unverified.
[Guard check and exact finding](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/undefined_name_guard_20260910/result.json).

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
