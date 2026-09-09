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
| Source checkpoint | Structural series on canonical host-pixel checkpoint `5ca9c8fff`; actual HEAD/diff/untracked manifest takes precedence |
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

EM parity result isolation: fast/long and InitialModel cases now write ledgers
beside their temporary outputs. Slurm/merge-guard reporting reads the explicit
run root; the extractor rejects duplicate or malformed current ledgers without
falling back to historical results. Baselines, metrics and assertions are unchanged.
54 CPU cases pass; disposable summary checks reject missing/duplicate/corrupt
results, and four generated shell scripts pass syntax checks. No jobs were submitted.
Use a **fresh** pytest `--basetemp` directory (pytest clears that directory), then
`python scripts/extract_em_parity_tables.py --ledger-root <run-root> --tier all`.
Legacy correlation gates and incomplete metric coverage still need separate review;
this change establishes output isolation, not scientific acceptance.
[Checks, commands and limits](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/parity_ledger_isolation_20260909/result.json).

The earlier power-class CPU negative-control failure is a demonstrated fixture
defect:32x32 scales by an exact power of two, so normalization order need not
produce different values. The randomized independent-reference assertion remains;
a single high-shell pixel at30x30 now discriminates divide-before-square exactly
(one float32 ULP), with32x32 retained as an equality control. A square-first
mutation is rejected at30 and accepted at32 as expected. Combined sparse/BPref
panel107/107 passes with no skips; production and tolerances are untouched.
Historical failures remain in their original receipts. [Proof, mutation check
and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/powerclass_order_fixture_20260909/result.json).

K1/fused-K-class BPref preprocessing capture assembly now shares one diagnostic
builder; callers retain the original device operands and capture gates. The
expanded execution AST and original tests are unchanged. Baseline65/candidate75
CPU cases pass, including10 added metadata cases;38 guard cases pass. Runtime
shrinks6 lines after readable call formatting. This is capture-schema cleanup,
not a numerical precision change. [Receipt and checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/bpref_preprocess_owner_20260909/result.json).

Six sparse helpers with no production/CLI/notebook consumers now live under
`tests/helpers/sparse_pass2_test_support.py`, retaining all tests and independent
NumPy references. The scorer loses162 runtime lines; this is relocation, not
162 lines deleted from the entire repository. All callable ASTs/JIT decorators
and test bodies remain exact. Baseline/candidate each pass29/fail1: the unchanged
power-class test expects generic and reference float32 reductions to differ,
but they coincide on this CPU; its independent expected-value assertion passes.
No waiver or test repair was made.38 CPU guard and9 final import/caller cases
pass; no new lint, new helper format clean. [Evidence and exact commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_test_support_20260909/result.json).

Exact-local batching policies now live in `local_batch_planning.py`:13 unchanged
functions and their constants leave the engine. The x-half reporting summarizer
imports no execution modules after this and the preceding audit-owner extraction.
It still initializes a JAX CPU backend through dependencies; this is not a
JAX-free reporting claim. Baseline/candidate each pass52 CPU cases;38 guard cases
pass. Engine shrinks374 lines; net production grows34 lines for the new owner.
Memory-query semantics and logger messages remain; warnings use the planning
logger namespace. The existing all-device memory probe needs a separately
qualified resource-policy review. [Receipt and checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_batch_owner_20260909/result.json).

Coarse selector validation, exact support hashing and result-profile attachment
now share the existing `helpers.coarse_score_diagnostics` owner. Five function
bodies and the wrapper registry are unchanged; controllers, analyzers and tests
use direct imports. The late-pair analyzer no longer loads `significance`; the
x-half summarizer still has a separate local-engine budget dependency.
Baseline and candidate each pass300 CPU cases;38 guard cases pass, with no skips
or new lint findings. Significance/scheduling shrink367 lines; net production
grows13 lines. This structural checkpoint adds no numerical qualification.
[Receipt and exact checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/coarse_audit_owner_20260909/result.json).

The K-class result-owner series on430c0763d separates the shared assembler, result
type, publication/noise helpers and subset-stat expansion into `k_class_results.py`.
Dense and sparse firstiter paths now share their identical six-statement pose
scatter. Accumulator offloading, list retention, dtype casts and operation order
remain explicit and unchanged. **485 lines leave scheduling; net production grows53
lines for the owner, documentation and compatibility.** This is an ownership gain,
not a net deletion claim. Exact definition/inlined-body comparisons and unchanged
numerical assertions cover the refactor. Historical result pickle GLOBAL/alias is
preserved; canonical result imports load no scoring engines.
Combined243 CPU cases pass,3 native-binding cases skip, and38 guard cases pass.
Mirrored instructions/codebase links pass; prior fixture and inspection corrections
remain recorded. [Combined receipt and exact checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k_class_subset_publication_20260909/result.json).
Frozen5ca9 qualification stays separate from this structural successor.

The next structural checkpoint on `5ca9c8fff` completes compact-pair host planning
ownership in `helpers.sparse_bucket_arrays`: four live helpers move with direct
callers; one test-only count scanner and its vacuous spy are removed. Three unused
tail-coalescer parameters disappear; actual hypothesis caps remain in the bucket
builders. **144 lines leave the scorer; net production shrinks26 lines.** All231
remaining function/class bodies match after only the unused-argument normalization;
all test assertions remain exact. Original22 caller cases, final23 focused cases
and38 CPU guard pass without skips. [Receipt, commands and source comparisons](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/compact_bucket_owner_20260909/result.json).
The following internal API migration removes unused mean_variance from the two
bucketed scorers and unused class priors from two firstiter helpers. Dense reference
reconstruction still receives mean_variance; caller prior validation and existing
raw-score firstiter semantics remain. All120 caller cases pass on both source
versions;38 guard cases pass on the combined source. Three whole-module ASTs match
after only the four argument/caller migrations; all assertions remain exact.
Combined structural production shrinks34 lines, scorer146 lines. [Combined source
receipt and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_scoring_contracts_20260909/result.json).
VDAM's canonical-pixel qualification stays pinned to5ca9c8fff, not a moving tip.

The canonical host-pixel repair resolves serialized STAR/pickle/CS geometry before
computational casts and stores one Python-float `dataset.voxel_size`. Source STAR
Angstrom origins use that same geometry; subsets preserve it. Legacy `StarFile.apix`
and default public pickle-array behavior remain unchanged. Computational CTF/pose/
image dtypes remain F32/C64 or explicitly selected F64/C128; no driver overrides.
Source-aware pickle loading checks each selected row's physical grid instead of
broadcasting row0. Missing metadata retains existing fallbacks, not invented precision.

Validation: **188 shared loader tests +38 CPU guard pass**; broader EM callers
**126 pass/3 fail**, all three missing-native-binding failures reproduced on exact
unchanged base. New tests fail30/pass4 on original source with a valid fixture;
36 new cases now pass within the shared panel. All10,000 real fixture STAR rows
resolve1.6375 exactly before F32 computation casts. Four production lint findings
are unchanged from base. [Exact commands, provenance and limitations](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/canonical_host_pixels_20260909/result.json).
This correctness checkpoint needs VDAM's integrated-source short real-prefix gate;
no GPU job/build was launched by em_clean.

Structural batch `a95050c95` on `5fd41da6f`: remove two test-only candidate helpers and
collapse two bucket-reporting adapters, **59 fewer production lines**. The padding
test now checks the live joint-log-Z normalizer; image IDs use an explicit fixture.
All expected values/tolerances and independent numerical references are preserved.
Eight focused CPU cases and38 guard pass;36 bucket-report comparisons are exact,
and the remaining computation AST is unchanged. [Receipt and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_test_only_helpers_20260909/result.json).

The following extraction moves four bucket-planning/coalescing functions and
three defaults into `helpers.sparse_bucket_arrays`, beside host array assembly.
Execution-policy selection stays in the scorer. **323 lines leave the scorer**;
net production grows6 lines for ownership imports/documentation, not a deletion
claim. All planning bodies/defaults and caller assertions are unchanged;26 CPU
cases pass before and after the move. The owner loads without scoring engines.
Final combined34 focused CPU cases,38 guard cases and parity CLI help pass.
[Combined-source validation and publication receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_bucket_planning_owner_20260909/result.json).

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

Real10076 10k/256 prefix20 **13654154 completed0:0** on frozen4f9. The
[reviewed peer report](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10076_prefix20_4f9a19492_20260909/RESULTS.md)
passes all21 cross-FSC conditions, minimum0.9990900154943916 at20; GT unavailable.
Six handoff pins match; peer independently recomputed all21 AUCs and the worst
raw-map curve. Strict state remains open: count mismatch at3, two Pmax gaps>=.001
at4, worst0.210868 with equal39 coarse counts and a one-pixel Y shift. Pixel-size
narrowing1.6375→1.6375000476837158 explains the initial-map discrepancy in the
[matched CPU bootstrap](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10076_prefix20_4f9a19492_20260909/pixel_bootstrap/README.md):
with the original1,000 images/seed29, changing only pixel size and derived ini_high
reduces native relative-L2 residual8.65e-8→4.65e-19, leaving one voxel at3.47e-18.
Rounded repeats are byte-identical; five evidence pins match. The separate
[canonical scalar prefix4 causal gate](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10076_pixel_prefix4_20260909/RESULTS.md)
(job13656403, original200 schedule/seed29, same H100) now makes map0 bitwise exact,
reduces primary E4 Pmax gap0.210854→6.35e-6 and removes the one-pixel Y flip.
All800 updated Pmax gaps are below1e-4; E4 cross-FSC-AUC is0.9999999777.
Six handoff pins checked. One coarse-count difference at3 remains unclassified;
secondary image89904 varies native-versus-native and is not credited to the fix.
These scalar-intervention results do not qualify the integrated loader repair or
provide a runtime ratio, final-real/absolute-accuracy or strict-state acceptance.

| Timing evidence | Result and limit |
| --- | --- |
| Frozen4f9,3k/128 H100 natural200 | 452.295/303.905s=1.48827652×. One pair/order; includes asymmetric candidate harness overhead, cold work/I/O. No GPU-memory measurement; storage contention unmeasured |
| Older8ab1a44be,100k A100 pairs | Forward1.647590×, reverse2.036693×; paired geometric1.831839×, predates compact CTF, quality unqualified |
| Privatebae959dab,3k/128 H100 | 1.436880×; not current100k or an isolated F32-M speedup |

See [archived timing boundaries](evidence/vdam-full200-4f9-20260909/README.md#state-precision-and-performance-limits)
and the [earlier performance ledger](em_cleanup_history_20260909_f0a8804e2.md#frozen-jobs-and-representative-performance).
Real10073 frozen-source evidence remains in the [real-data review](k1_real_window_review_20260909.md).
Next: VDAM qualifies a frozen copy of the integrated canonical host-pixel repair
on the bounded real prefix. Keep prior controls and native builds frozen; do not
duplicate the peer job. Further structural work remains separate from this
metadata correction and its scientific acceptance.
