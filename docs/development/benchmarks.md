# Reusable accuracy and performance benchmarks

A cleanup comparison measures the candidate against an unchanged RECOVAR
control. Report standing against RELION and synthetic ground truth separately.
Historical results from another commit remain historical evidence until their
workload is reproduced at the intended source.

## Keep definitions, runs and reports separate

Maintain three independently identified artifacts:

1. **Fixture definition:** immutable particle/metadata/map/mask inputs, exact
   generation or acquisition recipe, seeds and complete content hashes.
2. **Run record:** immutable source/build/environment identity, exact command,
   output root, job/process status, all measurements and artifact hashes.
3. **Comparison report:** derived from admitted run records and unchanged
   acceptance policy. Regenerate it without rewriting fixtures or past results.

Admission checks require every declared case/stage/half/class and unique input
identity. Reject missing or nonfinite required measurements, duplicate records,
wrong units, stale ledgers and incomplete outputs. Preserve the original failure
and its log. Distinguish execution failure, invalid evidence, quality failure,
performance regression and not measured; report generation success is not a
scientific pass. Expected counts come from the workload definition, not the
results that happened to be produced.

## Workload coverage

| Workload | Preserve and measure |
| --- | --- |
| Synthetic K1 | Fixed 34-case inventory, including known failures; small cases first, then robustness and completion trajectories |
| Synthetic K4 | Exactly four classes, Hungarian matching, every class and iteration, worst class, occupancy/collapse and assignment behavior; required 100k/256 completion pairs |
| Synthetic K2/K8/K16 | Existing three-seed stress cells, including preferred orientations and unequal class balance; retain per-class results |
| Real K1 | Fixed EMPIAR-10076 cohort, halfsets, masks, metadata and initialization; autonomous trajectory plus separately labeled fixed-state diagnostics |
| Real K>1 | Dense Class3D and K4 VDAM are separate workflows, each with matched per-class reference results and repeat/seed controls |
| Shared covariance/PPCA pipeline | SPA, cryo-ET, outliers and downstream tests, plus canonical real paper datasets from the [Della runbook](della.md) |

K4 is required; another K does not substitute for it. PPCA's latent dimension
is not class count. Dense Class3D's duplicated half-map outputs are not
independent-half FSC evidence. Real reference agreement measures consistency,
not absolute ground-truth accuracy.

Record whether a run is fixed-state replay, dispatch/perturbation replay,
fresh autonomous refinement, serialized continuation, Class3D or VDAM. State
which particle/noise/prior/controller values came from the oracle at each
iteration. A replay can diagnose arithmetic without qualifying autonomous
convergence. Capture-enabled and ordinary execution may use different paths;
qualify instrumentation effects before interpreting interventions or timings.

## Numerical contract

Retain the existing metric directions, tolerances and quantitative EM gates.
Map quality uses FSC curves, FSC-AUC and established FSC summaries against GT
and RELION. Map correlation is diagnostic and cannot override those gates.
For K4, use Hungarian matching and show all class results; never accept a poor
class because the mean passes. Keep initial/per-iteration/final products and
convergence/finalization identities explicit. See [EM numeric rules](../../recovar/em/AGENTS.md)
and [the program gates](../math/em_parity_program.md).

Preserve casts, FFT frames, half-spectrum support, normalization, reduction
order and candidate domains when comparing implementations. Match full state,
including noise, scales, priors, posterior normalizers, accumulators and control
state, before attributing a trajectory difference to a kernel. Diagnose the
first divergent boundary and keep independent analytic/float64 references.

## Performance contract

Run repeated control/candidate pairs sequentially on the same physical GPU,
with identical inputs, initialization, seeds, masks, batches, memory budgets,
thread/MPI layout and analysis settings. Retain all samples and report measured
variability. Record queue wait separately from compute time.

Separate cold compilation, warmed device compute, data preparation and complete
process wall time. Synchronize pending device work at timing boundaries when
needed. Per-stage durations can overlap; do not sum overlapping stages and call
the result end-to-end time. Reuse must retain original generation/compute time.

Measure GPU memory for the relevant process tree and assigned devices, plus
host RSS, with sampling interval and method recorded. Whole-device memory can
include other users; endpoint RSS is not a continuously sampled stage peak.
Zero/missing/unavailable memory readings do not establish zero memory use.
Use UUID-aware device selection; an empty visibility mask means no visible GPU.

Keep the established 10% regression warning policy. Do not waive an accuracy
failure because performance improved, or claim equivalent speed from unmatched
hardware, dirty mixed revisions, cached-only runs or one historical sample.

## Minimum run record

- Full RECOVAR commit, branch, diff SHA256 and untracked source manifest; frozen
  lock hash, imported RECOVAR/JAX paths and versions.
- Actual CUDA library path/hash, source/header/compiler identity before/after;
  RELION source, patches, binary, command and metadata where used.
- Resolved and hashed inputs, particle IDs/order, half/class assignments, seed,
  grid/voxel size, CTF/noise/contrast, masks and all effective overrides.
- Exact commands, run mode, requested/executed cells, Slurm IDs, node, physical
  GPU UUID/model, process exits, full logs, output paths and artifact hashes.
- Quality values and baseline deltas per required class/half/iteration;
  timing/memory samples with compilation, warmup and sampling methodology.
- Explicit acceptance policy identity and separate execution, evidence,
  quality and performance outcomes. No missing cell is silently dropped.

Bulky disposable outputs belong in scratch with a `SAFE_TO_DELETE` marker.
Keep enough small evidence to reproduce and audit a result after scratch cleanup.
