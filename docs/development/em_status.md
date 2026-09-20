# Current EM/VDAM development scope

Updated September 16, 2026. The cleanup and scientific qualification are unfinished.
The [task queue](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/CURRENT_TASK.md)
records the full work list, current evidence and next actions. This page describes
scope and ownership; experiment histories belong in the private archive.

## Scope and invariants

Refactor EM/VDAM for readable, succinct code and clean APIs; integrate qualified
peer work; remove demonstrated dead or duplicate code and obsolete experiments.
CUDA implementation, FFI and their unique numerical coverage are included.
Avoid generic executors or mode switches that make the code harder to follow.

**Validation for this task is EM/VDAM-only.** Do not run RECOVAR-wide SPA/ET,
downstream, outlier, indices, stress or heterogeneity suites, and do not regenerate
their baselines. Preserve main heterogeneity APIs and serialized formats through
source review and scoped checks. Historical plans calling for broader suites
are superseded by the user's scope correction.

Structural changes preserve scientific defaults, casts, reduction order, JIT
boundaries, layouts and buffer lifetime. Keep canonical sampler Euler angles
and host geometry metadata. EM/VDAM APIs and CLI compatibility may change when
callers migrate together; main-pipeline APIs remain protected. Keep independent
numerical references independent. Do not widen tolerances or modify baselines.

## Source and ownership

The lead owns integration and publication from `codex/integrate-pr180` in
`recovar_structural_cleanup_20260907`. [PR179](https://github.com/ma-gilles/recovar/pull/179)
is a draft stacked on PR158's pinned `44d770de3` control; do not merge, retarget,
force-push or alter frozen validation snapshots. Exact published source and local
changes are in [lead status](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/status/em_clean.json).

The existing EM and VDAM peers retain exclusive ownership of their implementations.
Use the [coordination board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md)
to review candidates and synchronize writable branches at natural integration
checkpoints. Preserve measured sources, user-selected models and unsent human
messages. Verify live job handles before restarting or declaring work stopped.
Continue selectively reviewing Roey's `dense_em_refactor` branch, retaining the
cleaner implementation when approaches overlap.

## Architecture and current integration

`recovar/em/` is the implementation root; `dense_single_volume/` is retired.
Controllers and schedules remain distinct where their semantics differ. Share
sampling, scoring, candidate layouts and accumulation where semantics match.
Replay/RELION adapters and diagnostics have explicit owners; runtime RELION
bindings are used by supported workflows and are not obsolete glue.
See the [codebase map](codebase.md) and [implementation guide](em_implementation.md).

The Fourier-window padding optimization is integrated into the draft with focused
and frozen-source evidence. The EM launch-ladder and initial-scale candidates,
and the VDAM unified class-segmented engine, remain under review. Passing a helper
check does not qualify a trajectory. Retire replaced engines only after callers
have migrated and their numerical, dtype, metadata and performance contracts pass.
Do not count the temporary coexistence of old and new engines as code reduction.

## Unresolved validation gates

**The current source is not fully quality-accepted or performance-qualified.**
Historical K1/K4 failures remain evidence for their recorded sources; archival
does not resolve them. The supported K4 comparison still has a class below its
established FSC-AUC gate. Saved launch-ladder contributors agree in a bounded
sample, while GPU accumulators are not bitwise repeatable even on the control.
Neither observation warrants a tolerance change or a rounding-noise dismissal.

Follow the unchanged [quantitative gates](../math/em_parity_program.md) and
[EM validation ladder](em_parity_runbook.md#validation-ladder): matched-state
scores/support/posteriors/poses/accumulators, synthetic then real K1, then exactly
K4 with Hungarian matching and per-class FSC/FSC-AUC. Correlation is diagnostic.
Completion requires production float32, matched inputs/seeds/maps/masks,
convergence and finalization, and at least 100,000 particles at 256x256 or larger,
with completed RECOVAR/RELION pairs on the same GPU class. Diagnostic double,
partial iterations and missing measurements do not satisfy these gates.
Preserve the current final-grid-correction default; its strict-target discrepancy
needs separate qualification. Scientific acceptance precedes speed qualification;
the provisional 2x runtime target does not waive quality gates.

Use frozen pixi environments, Slurm for integration/long GPU work, and sealed,
identified native libraries. Follow the local GPU0 reservation and scoped
[EM contract](../../recovar/em/AGENTS.md), [benchmark contract](benchmarks.md)
and [agent workflow](agent_workflow.md). Repeat checks only for changed behavior,
failures, unresolved concerns or required qualification.

## K=1 auto-refine speed program (September 18-20, 2026)

Active conclusion: on the 10k EMPIAR-10097 fixture at 256 px on one H100, the
device-resident K=1 candidate runs the cold auto-refine in 1087-1249 s against
RELION's 617-678 s band, **1.68x at matched iteration count and 1.81x at the
median**, from 3.26x at the program's start; the unchanged compact engine is
2.9x. Quality is neutral: regime-matched FSC-AUC against RELION's per-iteration
maps agrees with the control to 0.0011 at orders 2 and 3, and the order-4
difference is explained entirely by how many order-3 iterations preceded, not by
the engine. Speed work therefore continues from a quality-accepted checkpoint
**for this fixture only**; the 100k/256 K=1 and exactly-K=4 completion gates
above are untouched by this program and remain open.

Per-iteration cost is now shape-dependent rather than uniformly behind. RELION's
current-size and order schedule is identical to ours iteration by iteration, and
against it an iteration that repeats an already-compiled shape costs 0.81-0.99x
RELION while an iteration that introduces a new shape costs 1.34-1.98x. Four of
seventeen iterations are true repeats, so nearly the whole remaining factor is
paid at shape transitions and in the final all-data iteration.

The three open levers, largest first:

* **The final all-data iteration**, 178-195 s against RELION's rough 60-75. The
  local pass-2 route pins `use_translate_sum_kernel=False` and takes its weighted
  sums from XLA, so at current size 256 the CUDA translate-and-sum and Wavg
  kernels that serve the global route do not run at all. Node-granularity traces
  put 64.6 s of the 71.2 s of per-half GPU work in XLA fusions and only 6.6 s in
  our kernels.
* **First sight of a new shape.** A warm persistent compilation cache removes the
  whole XLA-compile part of it; whether production runs warm is a user decision,
  and the gate here runs cold. Hiding compile behind pass one was measured and
  rejected: it regressed the new-size regime by 60 s.
* **Convergence trajectory.** The candidate takes 18-23 iterations where the
  control takes 17-18 and RELION 17, all of the excess at order 3, where the
  hidden-variable stall test misses its 3 percent window in our favour. This is a
  parity question, not a speed one.

Next check: the four-arm qualification at the merged head (controls bracketing
the native-projector and device-projector candidates) decides whether the
projector default flips, and the full-box M-step needs the kernel path before its
130 s can be claimed. Evidence, job identities and the ticket board are in the
[coordination archive](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/em_speed_phase0_budget_20260918.md).

A durable correctness finding came out of this program: the BPref accumulator can
become non-finite on the compact engine, seen three times at iterations 2, 9 and
15 on three different GPUs, once as a CUDA illegal address under the platform
allocator. The once-per-iteration accumulator guard is now on by default in raise
mode, at a measured cost inside the wall band, so the failure stops located
instead of producing a silently corrupt map. The mechanism is an open defect.

## Historical evidence

The [previous status](https://github.com/ma-gilles/recovar-experiments/blob/5dc0795a8fb2f96ec994e29937951f7488df5c4d/docs/development/em_status_20260916_e4bca3705.md) preserves the detailed
scientific findings, failures, repairs and source-specific measurements. The
[superseded cleanup plan](https://github.com/ma-gilles/recovar-experiments/blob/5dc0795a8fb2f96ec994e29937951f7488df5c4d/docs/development/cleanup_plan_20260916_e4bca3705.md) is historical,
not current authorization. Both originals were archived byte-for-byte.
Keep completed experiment scripts, reports and results in
[recovar-experiments](https://github.com/ma-gilles/recovar-experiments), with source
identity and reproduction records. Main should retain implementation guides,
maintained workflow tools and unique numerical tests, rather than status histories.

The [original PR158 VDAM reuse plan](https://github.com/ma-gilles/recovar-experiments/blob/342c5c2164bd3595fe7748b66a6453439afc9af5/docs/math/vdam_relion_parity_reuse_plan.md)
preserves the historical ownership map and suite contract. Its retired package paths
and proposed work are historical; use the current codebase map and task queue above.

The fixed July case-11 six-arm report sealer is preserved in the
[experiment archive](https://github.com/ma-gilles/recovar-experiments/blob/7e8c1543c057a8590f119448fd658f92daf10c64/scripts/seal_six_arm_global_membership_repeat_join.py).
Reusable global-winner analysis and its numerical coverage remain in this repository.

The rejected August case-4/5/10 treatment-prefix report, inputs, generator and
exclusive report tests are in the [experiment archive](https://github.com/ma-gilles/recovar-experiments/tree/a9cd2fb51fc1b1637a830cc7dcf1e60e4632e681/experiments/k1_selected_treatment_prefix_20260822).
Its original **0/3** outcome is preserved; current qualification gates are unchanged.

Standalone timing, fine-score, prehalf, texture-tie and early parity investigation programs
now live with their exclusive tests and retired launchers in the
[experiment archive](https://github.com/ma-gilles/recovar-experiments/tree/545591c0935542c882ef92b60224300a43c66de7/scripts).
Independent formulations are preserved unchanged; production kernels, numerical
tests, maintained guards and reusable profiling tools remain here.
