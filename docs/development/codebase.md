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
| RELION-style K1/K-class refinement | [`scripts/run_full_refinement.py`](../../scripts/run_full_refinement.py) resolves inputs and options | [`iteration_loop.refine_single_volume`](../../recovar/em/refinement/iteration_loop.py); despite the name, this controller also handles K-class refinement |
| Pose-marginal PPCA refinement | [`refinement_loop`](../../recovar/em/ppca_refinement/refinement_loop.py) exposes dense and local refinement loops | [`dense_dataset`](../../recovar/em/ppca_refinement/dense_dataset.py), [`local_dataset`](../../recovar/em/ppca_refinement/local_dataset.py), and their fused kernels |
| InitialModel/VDAM | [`initial_model.iteration_loop.run_vdam_iterations`](../../recovar/em/vdam/iteration_loop.py) | Initial-model schedules, subset selection, state and reconstruction |
| Earlier EM API | [`recovar.em`](../../recovar/em/__init__.py) exports `EMState`, `SGDState`, `HeterogeneousEMState` and batch routines | [`states`](../../recovar/em/states.py), [`iterations`](../../recovar/em/iterations.py), E-step/M-step and heterogeneity modules; the tracked [`em_test` notebook](../../recovar/em/em_test.ipynb) still uses this API |

Pipeline PPCA and pose-marginal PPCA have different entry points and state
contracts. Choose the implementation reached by the actual command. The
pipeline PPCA path currently rejects tilt-series input. Consult the
[PPCA refinement guide](../../recovar/em/ppca_refinement/AGENTS.md) when working
on pose refinement, and the [paper-data runbook](della.md) for pinned inputs.

## EM package layout

`recovar/em/` is the common implementation package. Standard refinement and
VDAM have separate controllers and schedules, and share numerical owners where
their semantics already match:

| Directory | Responsibility |
| --- | --- |
| `refinement/` | Standard EM iteration, convergence/finalization, options and map updates |
| `vdam/` | InitialModel driver, subset schedule, learning rates and VDAM state transitions |
| `classification/` | K-class routing, inputs and joint result assembly |
| `dense/`, `local/` | Dense and exact-local execution |
| `scoring/`, `sparse_pass2/` | Coarse scores/support and sparse second-pass execution |
| `helpers/` | Shared array layouts, operators, batching, precision and statistics |
| `relion/` | Runtime RELION metadata, normalization, CTF and native adapters |
| `diagnostics/` | Optional capture writers, replay and intervention tools |
| `ppca_refinement/` | Pose-marginal PPCA workflow and its K-class bridge |

There is no second EM stack for VDAM. Its adapters supply the existing shared
kernels with VDAM-specific inputs. Scheduling and state transitions remain with
their workflow. The retired `dense_single_volume/` and `initial_model/` source
directories are gone. Import current owners directly: EM/VDAM is work in
progress, with no backward compatibility requirement for its Python APIs, CLIs
or old Python object names. Main heterogeneity pipeline compatibility remains
required. Historical logger names remain stable.

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

Use the [EM implementation reference](em_implementation.md#dense-and-local-em-ownership)
for detailed module contracts. Start with the boundary being changed:

| Boundary | Main owner |
| --- | --- |
| Iteration scheduling and state mutation | [`iteration_loop.py`](../../recovar/em/refinement/iteration_loop.py) |
| Dense E/M execution | [`em_engine.py`](../../recovar/em/dense/em_engine.py) |
| Local search orchestration and kernels | [`local_search_iteration.py`](../../recovar/em/local/local_search_iteration.py), [`local_em_engine.py`](../../recovar/em/local/local_em_engine.py) |
| Class routing and joint result assembly | [`k_class.py`](../../recovar/em/classification/k_class.py), [`k_class_results.py`](../../recovar/em/classification/k_class_results.py) |
| Replay selection and final-pass admission | [`relion_replay.py`](../../recovar/em/diagnostics/relion_replay.py), [`finalization_policy.py`](../../recovar/em/refinement/finalization_policy.py) |
| Coarse/sparse scoring | [`helpers/significance.py`](../../recovar/em/scoring/significance.py), [`helpers/sparse_pass2_bucketed.py`](../../recovar/em/sparse_pass2/sparse_pass2_bucketed.py) |

Import execution APIs directly from their defining modules; helpers must not
initialize controllers or scoring engines. During structural cleanup preserve
casts, reduction/JIT order, array lifetime, scientific defaults and saved formats.
Canonical source Euler angles and host pixel geometry remain metadata; derive
computation arrays from them. Required validation comes from the scoped guides,
not from the size of this overview. Current evidence belongs in [EM status](em_status.md).

## Diagnostics and reusable evidence

The optional iteration, reconstruction, pass-2 operand and normalization capture
writers live in [`recovar/em/diagnostics`](../../recovar/em/diagnostics/__init__.py).
Production engines call them at the existing capture boundaries. The package
initializer imports nothing; the individual writers still use shared numerical
utilities and the BPref capture context. This is an ownership boundary, not a
claim that all diagnostics have already been removed from normal import paths.

[`recovar/relion_bind`](../../recovar/relion_bind/__init__.py) currently mixes
native runtime dependencies with independent validation interfaces. RELION-style
EM uses its sampling, particle ordering, CTF and reconstruction routines; those
are not removable merely because the package also supports parity tests.
Oracle-only bindings, replay tools and historical experiment scripts need a
consumer/reproduction audit before relocation or deletion. Keep the independent
references under tests separate from the production functions they validate.

See [diagnostic owners](em_implementation.md#diagnostics-and-reusable-evidence)
for score, posterior, BPref, noise, normalization and output capture boundaries.
Use existing provenance/test wrappers and immutable evidence roots. Historical
runs qualify only their recorded source and inputs; missing cells are not passes.

### Replay state diagnostics

See [replay/state ownership](em_implementation.md#replay-state-diagnostics)
for frozen snapshots, intervention ordering, sampling, convergence, resolution
and projector preparation. Read these contracts before changing those boundaries.

## Ground-truth reporting

See [GT reporting owners](em_implementation.md#ground-truth-reporting) and
[the reporting guide](gt_reporting.md). Rigid fit-once/apply-many reporting is
opt-in; it does not change E/M execution or scientific acceptance gates.
