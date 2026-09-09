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

## Source and ownership

| Item | Identity or rule |
| --- | --- |
| Implementation | `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907/`, branch `codex/integrate-pr180` |
| Latest structural source checkpoint | Local routing/import cleanup following `121144413`; local engine 8,199 lines, sparse scorer 17,383, half scorer 1,358, controller 6,123 |
| Pinned PR158 control | `44d770de3f9336ab2f3f6a34203394bae8d1aeed`; preserve unchanged |
| Incorporated history | PR180 `1e2f229b3`; cleanup anchor `0b52c995a`; VDAM merge `c38fc62f0`; separate global-window correction `0219caa35` |
| Publication | [Draft PR179](https://github.com/ma-gilles/recovar/pull/179), stacked on [PR158](https://github.com/ma-gilles/recovar/pull/158); em_clean is sole integrator/publisher |
| Live coordination | [Shared board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md); exact scopes/handoffs supersede historical grants |

Read actual HEAD, branch, dirty fingerprint and untracked manifest before tests
or claims. This table is a checkpoint record, not a moving HEAD pointer. WIP
draft publication with incomplete checks is user-authorized; merging and
scientific gates are not waived. The branch includes behavioral integrations
as well as structural cleanup.

em_clean owns shared docs/status and publication. EM remains paused; its frozen
sources/jobs stay untouched. VDAM's private source-Euler repair is based on
`bae959dab`, on `codex/vdam-source-euler-metadata-20260909`; 12 EM Python owners and three tests are private scope only. Host metadata stays
outside JIT/FFI; no shared source adoption or native build is assigned by that
handoff. Precision proposal `907b02ce` remains private/unmerged. Raw-prefetch
source is unassigned. The RELION header lock is released: five diagnostic
insertions remain, and shared benchmark binaries/private captures stay frozen.

Local GPU 0 remains reserved. Check GPUs 1–3 immediately before use and restrict
by idle-device UUID; within Slurm preserve scheduler visibility. Avoid duplicate
jobs and preserve source snapshots while their jobs run.

## Agent efficiency package — September 9

The user authorized Terra/medium for routine cleanup, Astra for difficult review,
compact file handoffs and one publication per cohesive package. See
[the workflow](agent_workflow.md). The coordination README/em_clean status now
link full archives rather than repeating history. Scientific gates are unchanged.
The prior routing/import cleanup is published at `732e2cf3`, not an unpublished
seven-case patch; final focused coverage was11 plus guard38.

Peer reports private class-prior repair `f91eed7d6` atop frozen `df9975ee`,
red2fail/2pass then42 focused/38guard; review/adoption remains separate. Replacement
full200 K1 job13648609 was observed RUNNING;13648479 failed pre-science UUID
preflight. Different physical H100, same class/driver: timing descriptive only.
K4 audit13560356 also remains live. No duplicate job or shared Euler adoption.

## Engineering work and recent evidence

The latest batch removes a tautological projector eligibility condition, an
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
three controls; saved Euler/origin maxima are 5e-6. Full200 and broader gates
remain open. Integrator review verifies 15 source hashes and no textual merge
conflicts, but reproduces metadata loss in `_local_layout_for_class` with
class-specific priors. Forwarding plus a regression is required before shared
K-class coverage acceptance. Review: board handoff
`em_clean_source_euler_readonly_review_20260909.json`. No source adoption,
scoring-precision change, cutoff change or duplicate GPU run is assigned.

Next work:

1. Continue separating local execution stages and setup data flow; preserve
   backend selection, JIT boundaries and allocation lifetime.
2. Review the private Euler repair at a clean handoff, with a failing regression
   and fixed-state evidence before integration. Keep numerical review separate
   from structural cleanup and precision907.
3. Reconcile current-source validation failures and inspect the existing exact-K4
   audit when terminal; do not repeat long jobs after each helper edit.
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
