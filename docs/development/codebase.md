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
batch sizes, calls the selected kernel and packs statistics for the controller.

The dense single-class kernel is
[`em_engine.run_em`](../../recovar/em/dense_single_volume/em_engine.py).
The local single-class kernel is
[`local_em_engine.run_local_em_exact`](../../recovar/em/dense_single_volume/local_em_engine.py).
[`k_class`](../../recovar/em/dense_single_volume/k_class.py) supplies dense,
adaptive and local K-class orchestration. Class evidence and posterior mass
must be handled at the K-class level, not inferred from independently normalized
single-class probabilities.

Local-search dependencies are imported from their owners. Tests that replace a
kernel for a local dispatch check patch its binding in `local_search_iteration`.
The controller still has its own active `build_local_hypothesis_layout` binding
for adaptive parent-layout construction. Patch the call site exercised by the
test; do not add reverse imports merely to preserve an old monkeypatch location.

Unused constant copies in `iteration_loop` have also been retired. Batch and
raw-image-cache limits belong to `batch_planning`; first-iteration reconstruction
and dense K-class hypothesis budgets belong to `firstiter_cc`; the fine-grid
precomputation limit belongs to `local_search_iteration`. Their values and
environment overrides are unchanged.

The PPCA schedule bridge and its dense/local wrappers are imported from
[`ppca_bridge`](../../recovar/em/dense_single_volume/ppca_bridge.py).
Their unused controller re-exports have been retired. Helper-only callers also
import sign alignment and combined noise statistics from `mean_helpers`, rotation
metadata from `relion_metadata`, and replay iteration mapping from `relion_replay`.

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
- [`scorecard_cli`](../../scripts/scorecard_cli.py) supplies print/write/check
  handling for compatible historical scorecards. Each renderer retains its own
  fixed case inventory, validator and Markdown format. These CLIs remain usable
  without importing the scientific environment.

Use `python -m scripts.<name>` from the checkout for diagnostics that import
other script modules. Some older direct-file entry points still fail their
imports; the current review records that debt rather than treating failed help
commands as successful checks.

The [benchmark contract](benchmarks.md) defines source, fixture, library,
quality and performance evidence. [Current EM status](em_status.md) separates
the selected source from historical results and records open qualification gaps.
Historical scorecards describe their pinned runs; they do not qualify a new
checkout merely because the same report can still be rendered.
