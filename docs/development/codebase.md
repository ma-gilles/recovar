# Codebase map for contributors

Start at the entry point for the workflow you are changing, then follow the
state and array layouts into its kernels. RECOVAR has several EM workflows;
sharing a numerical primitive does not make their controllers interchangeable.
The [development contract](../../AGENTS.md) defines change scope and validation.

## Workflow entry points

| Workflow | Entry point | Main implementation |
| --- | --- | --- |
| Covariance pipeline | [`standard_recovar_pipeline`](../../recovar/commands/pipeline.py), the `recovar pipeline` command | [`principal_components`](../../recovar/heterogeneity/principal_components.py), covariance estimation, then embedding |
| Pipeline PPCA | The same pipeline with `--use-ppca`; `_run_ppca_refinement` selects the PPCA path | [`recovar.ppca.ppca.EM`](../../recovar/ppca/ppca.py), using the supplied dataset poses |
| RELION-style K1/K-class refinement | [`scripts/run_full_refinement.py`](../../scripts/run_full_refinement.py) resolves inputs and options | [`iteration_loop.refine_single_volume`](../../recovar/em/dense_single_volume/iteration_loop.py); despite the name, this controller also handles K-class refinement |
| Pose-marginal PPCA refinement | [`refinement_loop`](../../recovar/em/ppca_refinement/refinement_loop.py) exposes dense and local refinement loops | [`dense_dataset`](../../recovar/em/ppca_refinement/dense_dataset.py), [`local_dataset`](../../recovar/em/ppca_refinement/local_dataset.py), and their fused kernels |
| InitialModel/VDAM | [`initial_model.iteration_loop.run_vdam_iterations`](../../recovar/em/initial_model/iteration_loop.py) | Initial-model schedules, subset selection, state and reconstruction |
| Earlier EM API | [`recovar.em`](../../recovar/em/__init__.py) exports `EMState`, `SGDState`, `HeterogeneousEMState` and batch routines | [`states`](../../recovar/em/states.py), [`iterations`](../../recovar/em/iterations.py), E-step/M-step and heterogeneity modules; the tracked [`em_test` notebook](../../recovar/em/em_test.ipynb) still uses this API |

Pipeline PPCA and pose-marginal PPCA have different entry points and state
contracts. Choose the implementation reached by the actual command. The
pipeline PPCA path currently rejects tilt-series input. Consult the
[PPCA refinement guide](../../recovar/em/ppca_refinement/AGENTS.md) when working
on pose refinement, and the [paper-data runbook](della.md) for pinned inputs.

## Shared data and numerical boundaries

| Boundary | Owner | Contract to inspect |
| --- | --- | --- |
| Particle loading and batch identity | [`CryoEMDataset`](../../recovar/data_io/cryoem_dataset.py), image loaders and half-set utilities | Original image/particle IDs, subset-local positions, half-set membership, image backend and CTF metadata |
| Forward-model configuration and state | [`core.configs`](../../recovar/core/configs.py) | `ForwardModelConfig` static fields versus dynamic `ModelState` arrays; changing a static value may change JIT specialization |
| Fourier transforms and volume I/O | [`fourier_transform_utils`](../../recovar/core/fourier_transform_utils.py), [`utils.helpers`](../../recovar/utils/helpers.py) | Centered Fourier conventions, flattened arrays, full versus half spectrum, and the RELION axis/sign conversion |
| Mean, noise and regularization | [`homogeneous`](../../recovar/reconstruction/homogeneous.py), [`noise`](../../recovar/reconstruction/noise.py), [`regularization`](../../recovar/reconstruction/regularization.py) | Half-set ownership, shell support, normalization, prior construction and reconstruction units |
| Saved results | [`output`](../../recovar/output/output.py), [`ResultPaths`](../../recovar/output/output_paths.py) | Serialized field names, shapes, original IDs, and downstream `PipelineOutput` consumers |
| CUDA and RELION references | [`cuda_backproject`](../../recovar/cuda_backproject.py), [`relion_bind`](../../recovar/relion_bind/__init__.py) | Loaded binary identity, device placement, native layouts and independent reference behavior |

The [source conventions](../../recovar/CLAUDE.md) give the exact FFT and
RELION-frame rules. Follow those helpers when loading volumes for a comparison;
raw MRC arrays and uncentered FFT calls are not interchangeable with them.

## Dense and local EM ownership

The [refinement controller](../../recovar/em/dense_single_volume/iteration_loop.py)
owns iteration history, half-set dispatch, sampling updates, convergence and
finalization. Its exact local-search stage is implemented in
[`local_search_iteration`](../../recovar/em/dense_single_volume/local_search_iteration.py).
That module builds local pose neighborhoods, asks
[`batch_planning`](../../recovar/em/dense_single_volume/batch_planning.py) for
batch sizes, calls the selected kernel and returns `_LocalSearchIterationResult`
with named accumulators, pose fields, statistics and optional class summaries.
The controller reads those fields directly.

[`relion_replay`](../../recovar/em/dense_single_volume/relion_replay.py) validates
whether overrides contain numbered trajectory state and whether physical BPref
particle ordering is legal at a fresh, imported or sealed boundary. The controller
calls these guards before refinement and when choosing final replay behavior;
the replay module owns the rules and errors. Initial-only overrides remain
separate from numbered trajectory replay.
The same owner selects diagnostic reference replay and loads the requested
half/class maps, including shared-class fallback files. The controller keeps
the replacement boundary; source casts, Fourier/frame conversion and errors
remain unchanged. Replay messages use that module’s logger.

The local kernel returns `LocalEMResult` from
[`helpers.types`](../../recovar/em/dense_single_volume/helpers/types.py):
`Ft_y`, `Ft_ctf`, `hard_assignments`, `stats`, optional best-pose fields,
`noise_stats`, `profile` and `significant_counts`. All sixteen return-flag
combinations have the same field layout; disabled fields are `None`. The
local-search wrapper and K-class orchestration read these fields directly;
the positional packer and both decoders are removed. The result stores array
references without copying or synchronizing them.

Requesting reconstruction probabilities or sample IDs enables the engine's
profile that carries those captures. The wrapper still exposes a profile only
when requested, and copies its dictionary before adding wrapper timings.
Significant counts retain their own field even when that internal profile is
hidden. `tests/unit/test_local_search_result_contract.py` covers this routing,
plus K2/exact-K4 pose, noise and class-summary settings. Returned arrays retain
their layouts, dtypes and identities; saved refinement field names are unchanged.

Per-half dispatch belongs to
[`half_scoring`](../../recovar/em/dense_single_volume/half_scoring.py).
Its dense and local adapters prepare engine arguments, retain adaptive/first-CC
routing, and write class/pose fields into the caller-owned `PerHalfOutputs`.
The controller calls its two BPref-scoped entry points and retains iteration
scheduling, state transitions, reconstruction and device-buffer lifetime.
[`scoring_policy`](../../recovar/em/dense_single_volume/scoring_policy.py)
owns shared padding/window constants, the import-time static kwargs object,
and the existing call-time environment selectors. Their override precedence,
invalid-value handling and float32 defaults are preserved. The controller and
scorers share the same kwargs object; policy imports do not load engine modules.
These owners have no imports back into the controller. Log messages are unchanged,
with namespaces following the owner of each moved function.

The dense single-class kernel is
[`em_engine.run_em`](../../recovar/em/dense_single_volume/em_engine.py).
It returns `DenseEMResult` from
[`helpers.types`](../../recovar/em/dense_single_volume/helpers/types.py), with
named `mean`, `hard_assignments`, `Ft_y`, `Ft_ctf`, `stats`, `noise_stats` and
`profile` fields. Optional outputs are `None` when their existing flags are
disabled; changing flags no longer changes tuple positions. The container does
not copy arrays. Controller and K-class callers read these fields directly.
The local single-class kernel is
[`local_em_engine.run_local_em_exact`](../../recovar/em/dense_single_volume/local_em_engine.py).
[`k_class`](../../recovar/em/dense_single_volume/k_class.py) supplies dense,
adaptive and local K-class orchestration. Class evidence and posterior mass
must be handled at the K-class level, not inferred from independently normalized
single-class probabilities.

Fixed-capacity call selection and validation belong to
[`fixed_capacity_local.py`](../../recovar/em/dense_single_volume/fixed_capacity_local.py),
with the sealed plan/operand/hypothesis binding types. The engine delegates those
checks at the same pre-JIT boundaries. Callers use the general call-index API;
the test-only call-0 wrappers are removed. Canonical byte/dtype checks, poisoned-tail
rejection and authoritative dataset fetch order remain mandatory. Bucket geometry
and the dtype-preserving adjoint rotation accessor belong to
[`local_layout.py`](../../recovar/em/dense_single_volume/local_layout.py).
Both owners import independently of execution modules.

Scale-group ID validation and full-axis sizing have one host owner,
[`helpers/scale_groups.py`](../../recovar/em/dense_single_volume/helpers/scale_groups.py).
Local EM, both sparse scorers and the K-class subset router use it. Explicit
counts retain groups absent from a class subset; missing IDs disable engine
scale-statistics allocation, while routing still retains an explicit count.
Empty ID arrays retain the existing one-group convention. Engine callers check
the flattened image axis; the router has no image-count constraint. The helper
preserves existing casts and errors and imports independently of execution.

External normalization inputs are prepared by
[`helpers/normalization_inputs.py`](../../recovar/em/dense_single_volume/helpers/normalization_inputs.py).
`prepare_local_normalization_inputs` returns named log-Z, log-evidence, Pmax and
reconstruction-threshold arrays. It owns the local modes' validation order and
exclusivity; sparse and K-class callers reuse only optional F64 image-vector
conversion and retain their different semantic checks. This owner prepares inputs;
posterior arithmetic and normalization kernels remain at their execution sites.
It preserves input strides and avoids copying already suitable F64 arrays.

Sealed VDAM worker and block-chronology replay lives in
[`helpers/vdam_replay.py`](../../recovar/em/dense_single_volume/helpers/vdam_replay.py).
It owns NPZ schema validation, four cached loaders, stack-ID joins, worker/launch
ordering, iteration selectors and physical-row gathers. The local engine calls
this owner directly while retaining kernel execution and candidate block-map
publication. Replay defaults, caches, stable ordering and error behavior are
preserved; importing the helper does not initialize the execution engine.

The exact coarse Gaussian path in
[`helpers/significance.py`](../../recovar/em/dense_single_volume/helpers/significance.py)
passes its existing host pixel indices to the shared source-precision CTF owner
[`helpers/relion_ctf.py`](../../recovar/em/dense_single_volume/helpers/relion_ctf.py).
Coarse, local and sparse scoring call that owner directly; its single process
cache and native binding remain independent of the execution engines.
That loader gathers each cached CTF row before stacking and device placement;
index order and duplicates are preserved. Omitting pixel indices retains the
full-grid contract and cache. Only scoring operands are compacted: full-image
powerClass inputs, source precision, scale correction and padding semantics
remain intact. The [current evidence](em_status.md) separates operand equivalence
and the allocation microbenchmark from pending full-runtime/trajectory checks.

The production fine-grid significance mask is lazy: `_ClassFineGridSignificanceMask`
and `_PerClassFineGridSignificanceMask` generate only the requested image/rotation
block. The materialized NumPy comparison lives in
[`tests/helpers/fine_grid_significance_reference.py`](../../tests/helpers/fine_grid_significance_reference.py).
It has no production callers and retains a separate mask-building algorithm
for checking lazy blocks and explicit/complement coarse support.

[`score_outputs`](../../recovar/em/dense_single_volume/score_outputs.py) owns
the scoring containers and class/coarse-grid result adapters. It also owns
optional half-accumulator combination, shape/axis resolution and profile-row
recording. The controller retains scheduling and device-buffer offloading.
`HalfScoreResult` carries one halfset's common scoring output.
`PerHalfOutputs` owns separate two-slot lists for a scoring phase; slot 0/1
always selects the halfset, including for class-related fields. Image arrays
retain each halfset's local order and size. The K-class adapters populate class
assignments, posterior summaries and per-class noise statistics separately from
`update_from`. That method preserves existing optional pose fields when the
new result omits them, while always replacing accumulator-layout metadata.

Local-search dependencies are imported from their owners. Tests that replace a
kernel for a local dispatch check patch its binding in `local_search_iteration`.
`half_scoring` has its own active `build_local_hypothesis_layout` binding for
adaptive parent-layout construction. Tests of whole-controller dispatch patch
engine bindings in `half_scoring`; tests of local chunk execution patch
`local_search_iteration`. A shared sizing function mocked across both the
controller and scorer must be patched at both consumers. Patch the call site
exercised by the test; do not add reverse imports to preserve an old monkeypatch
location.

[`debug_dumps`](../../recovar/em/dense_single_volume/debug_dumps.py) owns the
half-selection policy for terminating significance/noise captures and numbered
BPref device captures, together with iteration dump writers. The controller
calls those selectors at the same dispatch boundaries and retains diagnostic
completion/stop control. Selector errors and log messages are unchanged; their
logger namespace follows `debug_dumps`. Capture state and counters belong to the
separate diagnostic owner below.

[`helpers.pass2_diagnostics`](../../recovar/em/dense_single_volume/helpers/pass2_diagnostics.py)
owns K1/K-class score dumps, norm-residual and chunked scale-AA writers, and
target-row selection, including staging effective K-class raw operands after
scoring. It reads the shared numbered-half context from `bpref_diagnostics`;
it does not import sparse scoring. The scorer retains scheduling and numerical
operand preparation, calling the capture helpers at their original boundaries. Capture schemas, casts and reduction order are unchanged.

[`helpers.bpref_diagnostics`](../../recovar/em/dense_single_volume/helpers/bpref_diagnostics.py)
owns the numbered-half capture context, contribution and membership counters,
membership selectors/rotation-mass writers, device-panel state, capture validation
and artifact writers shared by sparse and exact-local EM. Fused K-class capture-row
materialization lives here beside the signature/shadow consumer; compact-pair
expansion, selected-row order and reconstruction-posterior fields are preserved. The controller and replay scripts set and clear that context through this
owner. Sparse scoring retains candidate planning, numerical kernels and live
accumulation; it asks the diagnostic owner for scoped capture decisions. The
diagnostic module has no direct import of sparse scoring or the iteration
controller. The package initializer exposes options and sampling/statistics
helpers; import K-class results and execution directly from `k_class.py`.
Standalone helper imports do not load dense/local engines or sparse scoring.
Tests replace capture functions and state at this owner, including optional
native signature panels. Dump schemas, precision, counter order and error
behavior remain unchanged. The boolean parser is shared through
`helpers.env_flags.parse_env_flag`; file identities use `utils.file_hash`.

Strict local capacity/packing selectors call `helpers.env_flags.parse_env_binary_flag`
directly. It accepts only `0` and `1` after stripping whitespace, defaults to
false when unset, and rejects blank or textual boolean values. Its behavior
differs from the permissive diagnostic parser above; do not interchange them
during structural cleanup. The engine retains the order of reads and mode checks.

Unused constant copies in `iteration_loop` have also been retired. Batch and
raw-image-cache limits, first-iteration reconstruction caps, dense K-class
hypothesis budgets and adaptive pass plans belong to `batch_planning`.
`firstiter_cc` constructs the first-iteration coarse/fine grids; the fine-grid
precomputation limit belongs to `local_search_iteration`. Their values and
environment overrides are unchanged.

The PPCA schedule bridge and its dense/local wrappers are imported from
[`ppca_bridge`](../../recovar/em/dense_single_volume/ppca_bridge.py).
Their unused controller re-exports have been retired. Helper-only callers also
import sign alignment and combined noise statistics from `mean_helpers`, rotation
metadata from `relion_metadata`, and replay iteration mapping from `relion_replay`.

[`relion_normalization`](../../recovar/em/dense_single_volume/relion_normalization.py)
owns per-image norm and per-group scale formulas and their result type. It
depends on NumPy/JAX, not the controller, mean reconstruction or follower
dispatch. The controller retains state installation and temporary lifetimes;
`relion_worker_scale` handles follower-specific corrections. The seven formula
tests live in `tests/unit/test_relion_normalization.py` and import this owner
directly. The old `mean_helpers` normalization exports are removed.

Dense and local scoring share `orientation_priors.relion_translation_prior_center`;
the duplicate `relion_local_translation_prior_center` entry point has been removed.
Both use `(prior - rounded_old_offset) / pixel_size`, as before. The separate
`relion_sigma_offset_prior_center` serves sufficient statistics and keeps its
pixel-space formula without that division.

Raw and processed-image cache limits belong to
[`local_caches.py`](../../recovar/em/dense_single_volume/local_caches.py).
Bounded RELION projection caches belong to
[`local_projection_cache.py`](../../recovar/em/dense_single_volume/local_projection_cache.py):
budget parsing, stable bucket sorting/grouping, rotation-ID mapping and chunked
projection construction share that owner. The local engine retains cache
enablement, construction timing, group advancement and release. Only mapped
valid rows may be consumed; unused capacity keeps its existing uninitialized
padding. Cache consumers and tests import the owner directly, without engine
re-exports. Importing it does not initialize execution controllers.
Profile field definitions belong to
[`local_timing.py`](../../recovar/em/dense_single_volume/local_timing.py).
Their unused local-engine re-exports have been removed. Tests import cache
limit names directly from their owner.

The former runtime `compute_e_step_weights` API had only test consumers.
Its materialized dense posterior implementation is preserved in
[`tests/helpers/dense_posterior_reference.py`](../../tests/helpers/dense_posterior_reference.py).
The adaptive-oversampling tests still compare its complete posterior with active
significance paths. This reference keeps separate orchestration but shares
production preprocessing/scoring kernels; it does not independently validate
those kernels. Production significance belongs to `helpers/significance.py`.

Pure convergence-policy cases live in `tests/unit/test_convergence.py`; collecting
this module does not import the iteration controller. Full iteration smoke tests
remain in `test_refine_relion_mode.py`.

The first-iteration winner-take-all dispatcher lives with its grid builder in
`firstiter_cc.py`. It calls the batch planner and K-class engine directly for
both K=1 and K-class scoring. The controller supplies its logger and chooses
whether the batch clamp also updates the caller’s argument dictionary.

Precision selectors belong to `helpers/dtype_policy.py`. The controller passes
its existing static argument mapping; diagnostic iteration selection still
reads the environment at call time. Moving the selectors does not evaluate a
second set of import-time defaults or change any selected dtype.

Angular-grid order policies belong to `helpers/convergence.py`: exhaustive-grid
capping, final parent/fine orders, perturbation order and direction-prior order.
The controller supplies the active state and captured final-sampling metadata;
the helpers preserve their distinct order choices.

For an extraction, identify the actual boundary first: array layout, casts,
reduction order, JIT scope, device placement, buffer ownership and returned
statistics. Preserve those contracts during structural cleanup. The
[EM development guide](../../recovar/em/AGENTS.md) and
[mathematical algorithm map](../math/relion_refinement_algorithm.md) describe the
validation ladder and scientific state transitions.

## Diagnostics and reusable evidence

Diagnostic scripts compare specific captures, layouts and policies. Similar
names or similar-looking reductions are insufficient evidence of duplication.
Keep an independent numerical reference separate from the implementation it tests.

Common transport and command mechanics have narrow owners:

- [`file_hash.sha256_file`](../../recovar/utils/file_hash.py) hashes files in
  8 MiB blocks for RECOVAR-dependent diagnostics. A hash alone does not make a
  mutable file immutable or validate a manifest.
- [`json_utils.to_jsonable`](../../recovar/utils/json_utils.py) converts NumPy
  values, paths and nested containers. Finite-value and report acceptance rules
  remain the caller's responsibility.
- [`scripts.file_hash.sha256_file`](../../scripts/file_hash.py) supplies the same
  8 MiB hashing contract to standalone diagnostics without importing RECOVAR or
  JAX. Keep the runtime helper in the installed package and this script helper
  usable through both direct entry points and package imports.
- [`scorecard_cli`](../../scripts/scorecard_cli.py) supplies print/write/check
  handling for compatible historical scorecards. Each renderer retains its own
  fixed case inventory, validator and Markdown format. These CLIs remain usable
  without importing the scientific environment.
- [`scorecard_validation`](../../scripts/scorecard_validation.py) validates the
  paired baseline/treatment cases shared by four historical EM scorecards.
  Their fixed inventories, Markdown rendering and refusal to overwrite an
  existing report remain in the individual scripts.

Use `python -m scripts.<name>` from the checkout for diagnostics that import
other script modules. Some older direct-file entry points still fail their
imports; the current review records that debt rather than treating failed help
commands as successful checks.

The [benchmark contract](benchmarks.md) defines source, fixture, library,
quality and performance evidence. [Current EM status](em_status.md) separates
the selected source from historical results and records open qualification gaps.
Historical scorecards describe their pinned runs; they do not qualify a new
checkout merely because the same report can still be rendered.

### Replay state diagnostics

`helpers/state_swap_probe.py` owns the supported component variants and CLI
validation. It can enumerate variants without importing the refinement
controller. `helpers/state_swap_runtime.py` owns snapshot copying, map-amplitude
scaling and restoration of the selected components. The controller still owns
when the snapshot is taken and applies it after the RELION replay override.

Snapshots preserve the existing ownership contract: array inputs are copied,
while `state_fields` is a shallow copy of `state.__dict__`. Changing that
ownership, the ordered return tuple or the restoration sequence requires its
own behavior review. The in-memory scoring-state inventory and overwrite guard are owned by
`dense_single_volume/frozen_boundary.py`, alongside the sealed-boundary loader.
The controller takes and checks those snapshots at the existing boundaries.

Captured sampling grids belong to `dense_single_volume/relion_replay.py`, which
also applies replay state overrides. Its helpers construct Euler/translation
grids, canonical coarse rotation IDs and direction log priors directly from
sealed sampling metadata. They preserve the recorded direction/psi order and
convert translations from Angstroms to pixels using the supplied voxel size.
The controller selects when to use these grids.

Refinement now receives one `RefinementOptions` container. Its groups own
scheduling, adaptive search, parity behavior, local search, class setup, replay,
diagnostics and batching. `helpers/iteration_history.py` owns the per-iteration
history lists and their established result-dictionary keys. The controller
still chooses when each snapshot is recorded.

Precision is explicit at extracted boundaries: replay grids, resolution
curves and scoring-output adapters receive the caller's dtype. These helpers
do not import the controller to discover runtime settings. PR180's numerical
changes and their qualification state are tracked on the
[EM status page](em_status.md).

`helpers/convergence.py` owns angular-refinement state transitions, including
validation and application of explicit HEALPix schedules used for oracle runs.
The controller selects the iteration's requested order; the convergence helper
advances through the existing angular and translation updates without coarsening
an active state.
It also owns the approximate-accuracy convergence gate and its environment
overrides. The controller supplies its logger so malformed-override warnings
keep their existing routing.

`helpers/resolution.py` owns current-size growth inputs and first-iteration
resolution rules: the inclusive FSC/data-vs-prior boundary, raw versus corrected
K1 scheduling, the initial high-resolution cutoff, and the tau2 reporting taper.
The controller retains their timing within the refinement loop. The pure
scheduling cases live in `tests/unit/test_resolution_scheduling.py`; the
reconstruction/taper ordering check remains with the controller tests.

`dense_single_volume/projector_preparation.py` prepares RELION reference slabs
for the controller's scoring calls. It owns native reference conversion, cache
keys and files, optional dumps, and validation of captured projector geometry.
`relion_replay.py` retains the captured-state type and parser; the controller
selects the native or captured path and passes the resulting slabs to scoring.

Captured sampling and projector-state tests live in
`tests/unit/test_relion_replay_state.py`. They exercise the replay and projector
owners directly, including immutable copied arrays and suppression of external
metadata reads. End-to-end controller behavior remains in
`test_refine_relion_mode.py`; capture-file parsing remains in
`test_relion_projector_capture.py`.

Import execution entry points explicitly from their owners:

```python
from recovar.em.dense_single_volume.iteration_loop import refine_single_volume
from recovar.em.dense_single_volume.k_class import (
    KClassEMResult,
    run_dense_k_class_em,
    run_local_k_class_em,
)
```

The package initializer does not re-export these names. The CPU fast guard
checks that importing replay, normalization, projector, result and diagnostic
helpers leaves the controller, K-class orchestration, dense/local engines and
sparse scoring unloaded. Existing callers already import from these owners;
the definitions and their serialized module identities are unchanged.

## Ground-truth reporting

[`initial_model/gt_registration.py`](../../recovar/em/initial_model/gt_registration.py)
owns the optional CPU rigid fitter and immutable fit-once transform. The existing
[`gt_metrics.py`](../../recovar/em/initial_model/gt_metrics.py) keeps its legacy
rotation-only alignment API and result type. The reporting CLI
[`evaluate_ab_initio_gt.py`](../../scripts/evaluate_ab_initio_gt.py) opts into the
new fitter or applies a saved transform without fitting. [The reporting guide](gt_reporting.md)
explains geometry, common-frame comparisons and limitations. These are diagnostic
reporting tools; E/M execution, precision and quality gates are independent.
