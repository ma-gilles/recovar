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
| InitialModel/VDAM | [`vdam.iteration_loop.run_vdam_iterations`](../../recovar/em/vdam/iteration_loop.py) | Initial-model schedules, subset selection, state and reconstruction |
| Earlier and independent EM references | Import directly from the owning module; `recovar.em` performs no workflow imports | [`states`](../../recovar/em/reference/states.py), [`iterations`](../../recovar/em/reference/iterations.py), the E-step/M-step and heterogeneity modules, and the independent [normalized-CC](../../recovar/em/reference/normalized_cc_replay.py) and [Gaussian-reduction](../../recovar/em/reference/gaussian_reduction_replay.py) replays |

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
| `reference/` | Independent earlier EM/covariance formulations and deterministic numerical replays |
| `ppca_refinement/` | Pose-marginal PPCA workflow and its K-class bridge |

RELION diagnostic checkpoint restoration lives in [`relion/vdam_checkpoint.py`](../../recovar/em/relion/vdam_checkpoint.py), separate from the VDAM execution driver. Native moment/reference and BPref overrides, including post-M-step reference-map replay, live in [`diagnostics/vdam_mstep_replay.py`](../../recovar/em/diagnostics/vdam_mstep_replay.py); [`vdam/mstep_single_class.py`](../../recovar/em/vdam/mstep_single_class.py) retains the reconstruction transaction and its numerical boundary calls.

Particle bootstrap is owned by [`vdam/bootstrap_iref.py`](../../recovar/em/vdam/bootstrap_iref.py): it loads the bootstrap images and constructs the initial reference/state. [`vdam/init.py`](../../recovar/em/vdam/init.py) contains the state-only initialization formulas, while [`relion/initial_noise.py`](../../recovar/em/relion/initial_noise.py) owns the image iterator, initial noise estimate, single-optics noise input and MPI process-start half-set noise policy. The driver coordinates these stages; sampling geometry stays in [`vdam/native_sampling.py`](../../recovar/em/vdam/native_sampling.py).

VDAM coarse-call naming and result diagnostic packaging live with the shared [`coarse_gaussian_diagnostics.py`](../../recovar/em/diagnostics/coarse_gaussian_diagnostics.py) and [`coarse_score_diagnostics.py`](../../recovar/em/diagnostics/coarse_score_diagnostics.py) owners. The sparse E-step invokes them but does not implement report bookkeeping. These extracted helpers remain counted in the VDAM size budget.

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
| Fixed-capacity hypothesis packing and execution binding | [`fixed_capacity_local.py`](../../recovar/em/local/fixed_capacity_local.py) |
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

The completion reporter and final-BPref replay share NumPy FSC calculations in
[`scripts/fsc_metrics.py`](../../scripts/fsc_metrics.py). This module remains
independent of production scoring and does not select a JAX backend. Include it
in source manifests when freezing or copying either reporter; the reporter file
alone no longer contains the full metric implementation.

Shared RELION projector construction lives in
[`relion_projector_setup.py`](../../recovar/em/relion/relion_projector_setup.py):
`reference_to_relion_projector_half_maps_and_power` selects native/JAX setup and
performs the established frame and dtype conversion; the maps-only wrapper
releases the unused power spectrum. EM projector caching and VDAM both use this
owner directly. VDAM's `dense_adapter` retains state-specific preparation and
accumulator conversion, so EM no longer imports the VDAM execution adapter to
construct projectors.

## VDAM code budgets

The original `fbdf23f9` InitialModel snapshot contains 5,014 Python lines.
At `afa3d6d46`, the same accounting scope contains 8,626, including code moved
into shared owners. The user approved replacing the inherited 6,100-line cap
with audited responsibility budgets on September 13, 2026. This revises a
structural guard; numerical tolerances, baselines and scientific gates are unchanged.

| Responsibility | Audited lines | Budget | Retained scope |
| --- | ---: | ---: | --- |
| Controller and schedules | 2,053 | 1,655 | Driver, iteration/subset schedules, options and launcher defaults |
| Initialization | 467 | 500 | Bootstrap, initial state and shared initial-reference filter |
| Sampling and layout | 818 | 950 | Native sampling updates, canonical pose metadata and frame conversions |
| E-step | 2,370 | 2,525 | E-step configuration, batching, dense/local/compact routing, statistics, support and projector setup |
| Reconstruction and state | 684 | 790 | Single-class M-step transaction, precision checks, state and class dispatch |
| Input/output | 1,218 | 1,270 | STAR metadata, startup artifacts, RELION checkpoint import and initial noise |
| Diagnostics | 1,016 | 1,160 | GT registration, native moment/reference replay and coarse report bookkeeping |
| **Total** | **8,626** | **8,850** | **224 lines of total headroom (2.6%)** |

Noise failure reports and optional noise-boundary captures now live in
[`diagnostics/vdam_noise.py`](../../recovar/em/diagnostics/vdam_noise.py);
`vdam/estep_meta_updates.py` owns the numerical update. This move transfers
110 budget lines from E-step to diagnostics without increasing the 8,850 total.
Solvent masking now lives with reconstruction in `vdam/m_step.py`; state precision
preparation lives beside its dtype definitions in `vdam/mstep_single_class.py`.
Their move transfers 90 budget lines from controllers to reconstruction/state,
again preserving the 8,850 total.
VDAM translation and class-orientation prior construction now lives beside the
sampling state/plan in `vdam/native_sampling.py`; this transfers 100 budget lines
from controller to sampling. The earlier `deeb4b5ed` noise-adapter extraction
added 55 shared lines to the counted input/output owner, reaching 1,269; 20 more
lines of controller headroom now cover that responsibility (1,270 allowance).
The combined allowance remains 8,850.
Projector refresh/consume lifecycle now lives beside its builders in
`vdam/dense_adapter.py`; its unchanged stale-state checks and single-use handoff
transfer 40 budget lines from controllers to E-step, preserving the total.
Image-mask setup and normalized-spectrum conversion also live in the E-step
adapter; this transfers another 35 budget lines from controllers to E-step.
The audited counts and headroom above describe the original snapshot, not the current tip.

The largest retained routine grew from 228 to 733 lines before the recent
11-line dead-prior cleanup: sparse pass-2 orchestration now covers additional
compact/local, zero-oversampling, exact-operand and execution-policy cases.
Other identifiable additions include the 430-line native checkpoint adapter,
96-line continuation subset-order replay, 409-line rigid-registration owner,
and native M-step/reference replay diagnostics. These have distinct scientific
or diagnostic consumers; their size alone does not justify deletion. The audit
also identifies controller, particle-input and reconstruction growth for further
simplification. The budgets are limits, not a declaration that all code is necessary.

The [budget guard](../../tests/unit/initial_model/test_refactor_invariants.py)
assigns every VDAM Python module to exactly one responsibility, requires all
listed files to exist, and counts shared extractions with the previous spacing,
import and alias allowances. A move must migrate its accounting; a new module
must receive an explicit owner. Each responsibility must fit independently, so
spare diagnostic budget cannot conceal growth in the E-step. Review justified
new functionality before revising any budget. Preserve separate numerical paths
when merging them would complicate control flow or change arithmetic.

The [source inventory and growth audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_budget_history_audit_20260913/REVIEW.md)
records the original comparison at `4f83abed4` (8,633 lines). The subsequent
prior cleanup removed 11 lines and the startup metadata boundary added four;
the table above accounts for both. Historical file/name counts distinguish
relocation from new names but are not a semantic proof of dead-code completeness.
