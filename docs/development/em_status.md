# Current EM development scope

This page holds the active work, ownership and unresolved gates. Detailed receipts
through source `121144413` are preserved verbatim in the
[checkpoint archive](em_cleanup_history_20260909_121144413.md), which links earlier
history. Update current entries here; keep dated run histories in the linked
archives or dedicated reviews. Historical checks qualify only their pinned source.

## Milestone and invariants

Complete cleanup and professional development setup across RECOVAR, **EM first;
GUI and the new HIA engine are excluded**. Remove demonstrated dead/duplicate
code, clarify ownership/APIs, maintain consistent agent/contributor guidance, and
establish reproducible synthetic, real-data and exactly-K4 accuracy/performance
evidence. See the [cleanup plan](cleanup_plan.md), [codebase map](codebase.md)
and [benchmark contract](benchmarks.md).

EM APIs may change with callers/tests/docs migrated together. Preserve non-EM
APIs, saved formats, scientific defaults, numerical behavior, reduction order,
JIT boundaries and memory lifetime during structural cleanup. Keep proposed
numerical/runtime repairs separate until authorized.

**Do not switch production EM to double to close parity gaps.** Double replay
is diagnostic; a smaller gap does not prove roundoff. The inherited VDAM M-step
still defaults to F64/C128; an explicit F32 M/state route is now integrated,
with higher-precision stages disclosed below. Neither establishes full-production
F32 closure. Preserve deliberate higher-precision metadata, host calculations
and necessary non-EM mathematics. Do not widen tolerances, change baselines,
dismiss discrete/convergence mismatches without evidence, or use map correlation
in place of FSC/FSC-AUC. Follow the [EM operating contract](../../recovar/em/AGENTS.md).

Current VDAM priority is short-iteration parity, then final FSC/FSC-AUC quality;
up to2× RELION runtime is provisionally acceptable. The
[program board](../math/em_parity_program.md#current-vdam-quality-priority--september-9)
records the user decision and its limits. Monitor speed without an optimization
chase while quality remains open.

## Source and ownership

| Item | Identity or rule |
| --- | --- |
| Implementation | `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907/`, branch `codex/integrate-pr180` |
| Latest structural source checkpoint | Host candidate preparation following `c6a5eccd2`; local engine 7,834, sparse scorer 16,531, half scorer 1,363, controller 6,051 |
| Pinned PR158 control | `44d770de3f9336ab2f3f6a34203394bae8d1aeed`; preserve unchanged |
| Incorporated history | PR180 `1e2f229b3`; cleanup anchor `0b52c995a`; VDAM merge `c38fc62f0`; separate global-window correction `0219caa35` |
| Publication | [Draft PR179](https://github.com/ma-gilles/recovar/pull/179), stacked on [PR158](https://github.com/ma-gilles/recovar/pull/158); em_clean is sole integrator/publisher |
| Live coordination | [Shared board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md); exact scopes/handoffs supersede historical grants |

Read actual HEAD, branch, dirty fingerprint and untracked manifest before tests
or claims. This table is a checkpoint record, not a moving HEAD pointer. WIP
draft publication with incomplete checks is user-authorized; merging and
scientific gates are not waived. The branch includes behavioral integrations
as well as structural cleanup.

em_clean owns shared docs/status and publication. EM remains paused; frozen
sources/jobs stay untouched. Source-Euler repairs `df9975ee` and `f91eed7d6` are
locally integrated as `479e888f9` and `db52d1bca`, with the unified local result
caller migration. The combined source passes326 focused CPU cases,38 guard cases
and the captured trial16 Euler-only replay. See the
[combined receipt](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/em_clean_source_euler_combined_20260909.json).
Canonical sampler angles remain the source of truth; matrices are computation
inputs, not a route for reconstructing known metadata. The engine agent is
read-only auditing remaining conversions on the combined source; shared source
changes await a bounded handoff.
These source-Euler commits are included in this draft checkpoint. No additional private F32-M ancestry or
precision907 policy is included. Combined GPU/trajectory/K4/current100k gates
remain open. VDAM owns its private Pmax24 capture. Native job13649706 failed its schema
validator, but its parent receipt verifies172 reusable native payload files; no
native rerun is needed. Candidate continuation13649895 is now observed FAILED1:0
on h19g1; its cause is not reviewed here. The native hook dumps a stale coarse
threshold index0; VDAM reports exact reconstruction579862/rank41 from sorted
weights, mask and STAR. Preserve that diagnostic erratum; do not use the stale
index as a scientific comparison. No duplicate job is assigned.
Raw-prefetch source is unassigned. The RELION header lock is released; shared
benchmark binaries/private captures remain frozen.

VDAM's frozen integrated-prefix20 pair13652879 at4f9a19492 meets both map
conditions at all21 checkpoints: minimum cross-FSC-AUC0.9999999995633239 and
worst GT delta−8.33139925244e-8. Classes and available discrete schedules agree.
One coarse count differs at14/image109 (58 versus57); its Pmax gap is1.79358e-6,
while the overall selected Pmax maximum gap is1.1403436e-4. Fine support and
competing margins are missing, so no tie/noise waiver follows. VDAM reports its
independent105-AUC and worst raw-map curve checks; this update reads that report,
not a new em_clean recomputation. Supervised native stopping prevents a timing
ratio claim. Qualification covers only the tested3k/128 K1 prefix at4f9; later
cleanups, full200/robustness/K4/current100k remain open. Its full200 follow-up
13653485 is observed RUNNING on the same frozen source; no duplicate job.
[Prefix report](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integrated_prefix20_4f9a19492_20260909/RESULTS.md).

Local GPU 0 remains reserved. Check GPUs 1–3 immediately before use and restrict
by idle-device UUID; within Slurm preserve scheduler visibility. Avoid duplicate
jobs and preserve source snapshots while their jobs run.

Fixed-native-input coarse posterior replay13650770 on frozen df9975ee reproduces
all41 mask bits, retained count, final sum and cutoff exactly in four calls.
Intermediate prefix scans differ; this is not full intermediate bitwise equality.
Candidate normalization replay13650337 also preserves the saved candidate Pmax
and mask. The extra42nd parent therefore requires an incoming score/state audit
for this capture, not support pruning or a normalization change. See
[VDAM's report](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_source_euler_metadata_20260909/PMAX24_RESULTS.md).

Native-input follow-ups13651334/13651600 on frozen df9975ee reconstruct all840
saved Fourier values and validate21×2 coarse phases plus84 fine controls. Both
shared-pretranslated and fused scorers compare774,144 scores in each of six calls,
with maximum absolute error0.00018310546875; all calls preserve the native minimum
and41 support bits. Original Xi2 was not captured: three independent measured
initializations were used, with no fitted offset. This one-particle/B1 result
supports auditing incoming reference/image state; it does not establish universal
roundoff bounds, trajectory quality, or a kernel/cutoff repair. Terminal and
independent-review receipts are linked from the report; no duplicate job is assigned.

Reference-only replay13651920 on frozen df9975ee produces coarse counts
41/42/42/41 and exactly the extra candidate242860 while other scorer operands
stay fixed. VDAM's independent review reports a96.273% centered-RMS reduction;
the remaining maximum centered residual0.00128174 is not dismissed. This
localizes the observed support change to the prepared reference, without
justifying a scoring-kernel or cutoff patch. The linked report now also records
native CPU/FFTW construction from exact candidate Iref: its C64 upload matches
candidate PPref bitwise. Original native internal Iref is unavailable; rounded
native MRC and CPU/GPU construction remain explicit limitations. These are
VDAM-reported checks, not a new em_clean array audit or trajectory acceptance.
Incoming map history remains VDAM's diagnostic scope; no duplicate job is assigned.

Frozen df9975ee short-prefix map conditions pass through5/10/20/24 on 3k/128
K1 seed29: worst cross-FSC-AUC0.9999999987093361 and GT deficit1.09505193524e-7.
Fresh diagnostic native diverges at19/particle1660 while candidate matches both
completed native controls; competing raw margins are absent, so no numerical-noise
classification follows. Current integrated-source20-prefix planning belongs to
VDAM; no source/build/job change is implied. See the
[short-prefix handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/vdam_short_prefix_quality_and_reference_boundary_20260909.json).

Publication follow-up `974aa11cf` is integrated separately as `1624dd396`.
It retains canonical source Euler metadata in `score_outputs` and `local_debug`,
with the matrix-only fallback preserved. Fresh combined-source checks pass all
17 focused and 38 CPU guard cases (zero skips), plus Ruff and import/provenance
checks. The four pending cleanup/documentation files survived integration
unchanged. Debug output intentionally gains an origin field and retains F64
source angles; computation matrices and scientific precision policy are unchanged.
See the [integration receipt](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/em_clean_euler_publication_review_20260909.json).
This commit is included in the following draft batch. Global-grid Euler handling,
GPU/trajectory/K4 qualification and private907 remain outside this integration.

## Agent efficiency package — September 9

The opt-in Astra/high lead and Terra/medium reader were runtime-verified. Reader
isolation failed (workspace-write inherited), and the smoke exposed no child-close
tool. Delegation is disabled pending bounded requalification; direct Astra is the
fallback. Worker execution and fresh read-only recovery remain untested. Compact
handoffs and batched validation are in use; long-term savings are unmeasured.
See [the workflow](agent_workflow.md) and its opt-in setup record. Global defaults
and scientific gates are unchanged. The prior routing/import cleanup is published
at `732e2cf3`, with11 focused and38 guard cases.

Frozen df9975ee full200 job13648609 completed0:0: all201 registered-GT conditions
pass both native references, but strict cross-FSC remains open (min0.9698486 at155).
Different physical H100, same class/driver: timing descriptive only. K4 audit
13560356 failed2:0; both control/candidate first miss RELION FSC-AUC0.995 at
iteration10,class2. No current100k speed or broad quality acceptance is claimed.

## Engineering work and recent evidence

Per-image fine candidate preparation now lives beside the host bucket builders in
`helpers.sparse_bucket_arrays`; coarse support encodings have a small independent
owner, `helpers.significant_samples`. This removes their dependency on coarse
scoring. The class, three encoding functions and preparation function retain exact
ASTs, including source-Euler permutations, priors, shared buffers and dtype/order.
All affected callers and independent-reference imports migrate without changing
reference expressions. The complement class retains its legacy pickle identity
and annotation types. Final31 CPU cases and38 guard cases pass; archived bodies
also pass31 cases, and254 old/new support/pickle payload hashes are exact. One new
import-order finding was fixed; final checks rerun, pre-existing scorer I001 retained.
No GPU/build/job changes. Receipt with source inventories and exact commands:
[Candidate preparation](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/per_image_candidate_owner_20260909/result.json).

VDAM full200 job13653485 on frozen4f9 is terminal COMPLETED0:0 (Slurm checked
September9); scientific analysis remains peer-owned and pending review. This run
cannot qualify later structural checkpoints merely by successful execution.

Rectangular and compact sparse bucket arrays now have one host assembly owner,
`helpers.sparse_bucket_arrays`. Five exact function bodies move with direct
engine/test/parity-script callers; scheduling and candidate selection stay in the
engine. Existing12 CPU cases pass on current and archived builders, including
padding, M-step rotation aliases, explicit F64 inputs and dense-reference score
comparisons. The38-case guard passes and checks the new helper import boundary;
parity CLI help passes. No assertions or numerical tolerances changed. Sparse
scorer−241 lines, new owner246; net production+5. Part of the host-preparation
publication batch; no GPU/trajectory/runtime qualification.
[Receipt and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_bucket_arrays_owner_20260909/result.json).

Host candidate masks, compact pair indices and flat fine-job plans now live in
`helpers.compact_candidates`, independently of sparse numerical execution.
The class and eight function bodies are preserved; local-engine, test and
benchmark callers import the owner directly. The redundant private nonzero
forwarder is removed. The legacy class import identity remains for stored data:
eight protocol4/5 old/new pickle payloads are byte-exact and old payloads load.
Current/archived37-case CPU panels pass (one GPU case deselected); final combined
source passes37 candidate cases,14 prior-expansion cases and38 CPU guard cases.
The guard now checks this owner without execution imports; benchmark CLI help
passes. Existing style findings retained; new owner/tests are Ruff-clean. Sparse
scorer−321 lines, new owner328; net production+8. This and the prior expansion
form one batch; neither qualifies GPU trajectories or current100k/exactK4.
[Receipt and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/compact_candidate_owner_20260909/result.json).

Sparse K1 and fused K-class prior expansion now share
`helpers.translation_prior.expand_fine_translation_prior`. Caller-owned casts,
coarse buffers, NumPy gather/broadcast storage and error behavior are preserved;
the two translation-grid constructions remain separate because their precision
and validation differ. Fourteen CPU cases pass with current and archived callers,
128 exact host-array/storage/error comparisons pass, and the38-case CPU guard
passes. The fused caller test now covers no/shared/per-image priors with its
original assertions unchanged. Sparse scorer−13 lines; total production+14 for
one explicit owner. Local checkpoint for the next batch, no GPU/quality claim.
[Receipt and commands](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/fine_translation_prior_owner_20260909/result.json).

The shared adaptive pass-2 grid builder now belongs to
`helpers.oversampling.build_adaptive_pass2_grids`, used directly by ordinary
K1/K-class scoring and first-CC dispatch. The calculation body and six/seven-array
return contract are unchanged; sampling imports remain lazy in the helper owner.
All caller/test bodies match under the symbol migration. Twenty-four archived/new
cases match bitwise, including alias relationships, and malformed-ID errors match.
Affected CPU25/25 and guard38/38 pass with a hash-verified existing RELION binding;
the initial23-pass/two-missing-binding failure is preserved. Production Ruff passes;
pre-existing test import-order and formatting findings remain. This ownership
change adds6 net production lines; the three-commit batch with replay/fused-tail
cleanup removes22 net production lines. No GPU/trajectory/runtime claim. See the
[package receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/adaptive_grid_owner_20260909/result.json).

The local fused-score branches now join at one posterior-dump/timing tail,
removing three identical copies. All kernel calls and profile synchronization
remain in their original branches; distributing the common tail reproduces the
whole original module AST. Local engine7,895→7,833 lines (−62). Twenty affected
CPU cases pass; the score-only test now explicitly covers big-JIT on/off, both
passing on current and archived original code with unchanged numerical assertions.
Runtime line tracing reaches all four fused branches and the split route. The
initial forced-mode backend-count failure is preserved in the
[receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/fused_publication_tail_20260909/result.json).
This is structural evidence, not timing or trajectory qualification.

The validated controller batch moves post-update optimiser convergence replay to
`relion_replay.apply_optimiser_convergence_replay`. The controller still applies
accuracy before the native state update and replay counters afterward. Missing
fields, partial mutation on malformed metadata, and unnumbered final-pass detection
retain their original behavior; restart defaults intentionally remain separate.
Controller6,123→6,051 lines; the explicit interface/documentation grows total
production by34 lines. All61 affected CPU and38 final guard cases pass, zero skips;12 new boundary
cases also pass against the archived original controller block. Inlining proves
all22 controller function ASTs exact. An initial six-case filename-renaming failure
was fixed and preserved in the [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/optimiser_replay_owner_20260909/result.json).
Existing E402/I001 style findings reproduce on control. This is a local
validated checkpoint for the next publication; no GPU or trajectory claim follows.

The diagnostic batch separates reconstruction-window normalization and group-scale
captures into `helpers.norm_scale_diagnostics`, with direct sparse-scorer/test
callers and no forwarding wrappers. `pass2_diagnostics` retains score/raw-operand
captures and shared target selection; its K1/K-class writers share the exact gate
validation order. The score owner shrinks 1,614→943 lines; the new owner is638.
Combined production is33 lines smaller, including formatting of the moved owner
(51 lines); the extraction itself adds22 and gate consolidation removes4.
All eight diagnostic function ASTs match before/after extraction, and243 sparse
functions match after the two owner substitutions. Combined affected38/38 and
CPU guard38/38 pass with zero skips. The new owner imports without execution
modules. Ruff passes except the unchanged pre-existing sparse-scorer I001;
legacy whole-file formatting findings are retained. New-owner formatting is AST
identical to the tested source. See the
[batch receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/norm_scale_capture_owner_20260909/result.json).

The latest structural batch moves four unchanged host score-reporting
functions from `helpers.scoring` to `helpers.coarse_score_diagnostics` and migrates
`significance` and tests directly. Remaining kernel and moved-function ASTs are
exact;58 affected CPU cases, import boundaries and Ruff pass. Scoring loses286
lines; the new owner has293, with net production growth6 after an unused import
is removed. This improves ownership, not net code size. See the
[batch receipt](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/em_clean_coarse_diagnostics_owner_20260909.json).

The current package moves split bucket preparation into `local_preprocessing`,
shares the duplicated exact BPref translation operation and gives profiled and
unprofiled execution one result constructor. All numerical operations match the
old syntax tree after inlining the shared operation; profile ordering and every
returned field are preserved. Caller imports and operand-capture patches migrate
with the owner. Local engine size falls8,199→7,868; the new owner has339 lines,
so combined production source grows by8 lines. This is clearer ownership and
removal of duplicate operations, not a net line-deletion claim.

The control affected CPU panel has55 passes and two GPU-only score-translation
failures. Candidate has56 passes (including an additional single-image dense/local
oracle case) and the exact same two failures. Final CPU guard passes38. Package evidence is in
`hia_source_review_20260906/local_preprocessing_publication_20260909/` under the
implementation checkout's parent. No GPU or trajectory qualification follows.

The preceding batch removes a tautological projector eligibility condition, an
unread debug counter and 13 stale imports; eight adjacent imports from the
projection owner are grouped together. The window alias is immutable and all
128 routing combinations are unchanged. The import-only follow-up preserves
the executable AST. Existing routing/debug/cache coverage passes 11 CPU cases
(seven routing cases also pass on the control); the final guard passes 38.
No tests or tolerances changed. Engine size falls 8,229 → 8,199 lines; the main
routine still has 5,662 lines. No GPU or runtime qualification follows.
Commands, original/new source fingerprints and receipts:
`hia_source_review_20260906/local_engine_import_cleanup_20260909/` under the
implementation checkout's parent directory. The prior projection-assembly
checkpoint and its 83-case panel remain recorded in the linked archive.

Earlier checkpoints gave projection caching, progress reporting and VDAM block-map
serialization explicit owners; the [codebase map](codebase.md) identifies them.
The [archive](em_cleanup_history_20260909_121144413.md#engineering-work-and-recent-evidence)
preserves exact diffs, comparison counts, errors and test commands. Moving code
to an owner is distinguished there from deleting duplicate production code.

| Separate integration | Accepted scope and remaining limits |
| --- | --- |
| Compact CTF `5a39eab29` from `b1d57608d` | Host gather before stacking; full-grid default preserved. Combined CPU 241 passed/23 GPU deselected plus guard 38; H100 13638270 has bitwise operands/coarse scores. No current trajectory/runtime qualification |
| Rigid reporting `49efca34e` from `383772fa0` | Explicit fit-once/apply-many reporting, legacy schemas/defaults preserved; CPU 78 plus guard 38. See [GT reporting](gt_reporting.md) |
| M capability/DC `29d7e38a5` from `1b4f2adee` + `b17913a91` | Explicit M precision and real DC before inverse FFT; inherited F64 default preserved. CPU 173 plus guard 38; fixed-M F32 error 6.93e-5 → 2.36e-7 is not trajectory acceptance |
| CLI M route `cb897cda5` from `bae959dab` | Explicit `--mstep-backend jax --mstep-compute-dtype float32` changes M, six persistent state fields and post-M mask arithmetic. CPU 243 plus guard 38; two-update H100 13644423 preserves pose/support with continuous-state differences |
| Native projector test contract `5066d57a5` | Remove expected-array C64 narrowing and check preserved native dtype: 3 failures/10 passes → 13 passes, no production change; historical failed panels remain failed |

Bootstrap, corrected E projector/tau2 preparation, normalization/noise, BPref
export and M shell geometry retain higher-precision numerical work. Full-stage
precision and integration evidence live in the
[precision review](vdam_precision_review_20260909.md). The [full200 review](vdam_f32_m_full200_review_20260909.md)
records all four registered-GT conditions passing across 201 checkpoints on
private `bae959dab`, while all four strict cross-engine FSC histories fail
0.999, first at 71. It does not qualify the newer primary composition or 100k runs.

The private Euler repair `df9975ee2` preserves the native source triple instead
of deriving it from a float32 matrix. Peer H100 prefix20 job13647974 reports
100 exact angular/translation crossings and Euler triples, 21 passing map gates
(minimum cross-FSC-AUC 0.99999999957), 657 available discrete comparisons exact
and 36 unavailable. Pmax maximum is 8e-5 versus native_off1 and 1.89e-4 across
three controls; saved Euler/origin maxima are 5e-6. These are the historical private prefix results. The initial integrator review
verified15 source hashes and reproduced source-Euler loss with class-specific
priors; that repair is now integrated at479e888f9/db52d1bca, with canonical-angle
publication at1624dd396. See the integration evidence above and the historical
`em_clean_source_euler_readonly_review_20260909.json` handoff. Broader K4 and
strict trajectory coverage remain open; precision907 remains separate.

Next work:

1. Continue separating local execution stages and setup data flow; preserve
   backend selection, JIT boundaries and allocation lifetime.
2. Review VDAM's completed frozen4f9 full200 analysis when the peer handoff is
   ready. Keep numerical acceptance separate from structural cleanup and907.
3. Reconcile current-source validation failures and the failed exact-K4 audit;
   do not repeat long jobs after each helper edit.
4. Continue shared non-EM cleanup and plan the required shared/real/K4 milestone
   checks after cohesive checkpoints. GUI and the new engine remain deferred.

## Unresolved validation gates

**The selected source is not scientifically or performance qualified.** Historical
checks qualify only their recorded source. This consolidation waives no failure,
skip, baseline or tolerance.

| Evidence or gap | Current interpretation |
| --- | --- |
| Frozen combined `d21f52d72` shared CPU | 194 passed, 3 GPU deselected; selected contracts, not complete shared workflows |
| H100 native 13634222 | 222 passed, 1 skipped; inherited rectangular CUDA/JAX ordering case remains unqualified |
| H100 API 13634313 | 592 passed, 12 failed; nine original failures have targeted follow-ups, not a fresh full-panel pass |
| Frozen H100 API 13641893 |727 passed/6 failed/0 skipped at `1b046647f`; one stale guard repaired separately, five strict byte-equality failures remain |
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

Additional retained failures: the source-CTF owner panel has 179 passes and two
unchanged CUDA-only failures on CPU (23 GPU deselected); the VDAM block-map panel
has 34 passes and the existing native trace-clock ordering failure. Neither was
suppressed or repaired by structural cleanup. Detailed errors and source pins
remain in the archive. Private F32-M map results do not close strict state/tie
or full-production-F32 gates.

## Frozen jobs and representative performance

Frozen older-source K4 producer **13560202 completed 0:0** on `della-l08g4`
(16:19:39). Its audit **13560356** was RUNNING on `della-r3c1n7` when checked
September 9 during this consolidation. Poll the exact job before relying on that
observation. Audit outputs are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/hia_k4_cleanup_pair_20260907/quality/`.
Process completion alone does not establish scientific acceptance.

Real-data K1 **13610518**, **13610539** and **13629099** completed 0:0.
The [real-data review](k1_real_window_review_20260909.md) verifies 92 hashes and
388 numeric fields, natural convergence 20/final 21, on frozen `a0a86` with
138,899 real 10073 particles at 380. Masked AUC improves +0.001742/+0.002335
against modern RECOVAR repeats but remains −0.000768/−0.000605 below RELION.
The high-shell deficit persists; this is not GT or current-source acceptance.

| Frozen timing evidence | RECOVAR / native wall time | Limits |
| --- | ---: | --- |
| `8ab1a44be`, 100k, forward same-physical-A100 pair | 13,125.865 / 7,966.705 s = 1.647590× | Whole-process, changing shared-host contention |
| Same source, reverse same-physical-A100 pair | 14,399.809 / 7,070.191 s = 2.036693× | Paired geometric mean 1.831839×; predates compact CTF; quality unqualified |
| `bae959dab`, 3k/128 H100, full200 13644924 | Paired geometric mean 1.436880× | F32 M route with other higher-precision stages; not current 100k or an isolated M speedup |
| Compact-CTF CPU allocation | 8.68 → 1.74 ms | Microbenchmark only, not end-to-end runtime |

The separate 14,764.576 s profile attributed 195.662 s (1.33%) to backend
compilation, 5,777.491 s to prefetch queue waiting and 3,709.627 s exclusive to
CTF host stacking versus 69.060 s in its native binding. Cumulative helper time
overlaps its children; these are not guaranteed removable costs or proof of I/O.
The old 100k saved cross-FSC has final AUC 0.9924569/minimum 0.9900306 at 70;
its GT registration remains unqualified. All raw receipts, incomplete controller
summaries, native-capture limits and earlier timings remain in the
[full checkpoint archive](em_cleanup_history_20260909_121144413.md#frozen-jobs-and-representative-performance).
No new GPU jobs or builds were launched for this cleanup batch.
