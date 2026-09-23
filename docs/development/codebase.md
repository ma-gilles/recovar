# Codebase map for contributors

Start at the entry point for the workflow you are changing, then follow the
state and array layouts into its kernels. RECOVAR and relax have several EM workflows;
sharing a numerical primitive does not make their controllers interchangeable.
The [development contract](../../AGENTS.md) defines change scope and validation.

## Workflow entry points

| Workflow | Entry point | Main implementation |
| --- | --- | --- |
| Covariance pipeline | [`standard_recovar_pipeline`](../../recovar/commands/pipeline.py), the `recovar pipeline` command | [`principal_components`](../../recovar/heterogeneity/principal_components.py), covariance estimation, then embedding |
| Pipeline PPCA | The same pipeline with `--use-ppca`; `_run_ppca_refinement` selects the PPCA path | [`recovar.ppca.ppca.EM`](../../recovar/ppca/ppca.py), using the supplied dataset poses |
| RELION-style K1/K-class refinement | [`scripts/run_full_refinement.py`](https://github.com/ma-gilles/relax/blob/main/scripts/run_full_refinement.py) resolves inputs and options | [`iteration_loop.refine_single_volume`](https://github.com/ma-gilles/relax/blob/main/relax/refinement/iteration_loop.py); despite the name, this controller also handles K-class refinement |
| Pose-marginal PPCA refinement | [`refinement_loop`](https://github.com/ma-gilles/relax/blob/main/relax/ppca_refinement/refinement_loop.py) exposes dense and local refinement loops | [`dense_dataset`](https://github.com/ma-gilles/relax/blob/main/relax/ppca_refinement/dense_dataset.py), [`local_dataset`](https://github.com/ma-gilles/relax/blob/main/relax/ppca_refinement/local_dataset.py), and their fused kernels |
| InitialModel/VDAM | [`relax.vdam.iteration_loop.run_vdam_iterations`](https://github.com/ma-gilles/relax/blob/main/relax/vdam/iteration_loop.py) | Initial-model schedules, subset selection, state and reconstruction |
| Earlier and independent EM references | Import directly from the owning relax module | [`states`](https://github.com/ma-gilles/relax/blob/main/relax/reference/states.py), [`iterations`](https://github.com/ma-gilles/relax/blob/main/relax/reference/iterations.py), the E-step/M-step and heterogeneity modules, and the independent [normalized-CC](https://github.com/ma-gilles/relax/blob/main/relax/reference/normalized_cc_replay.py) and [Gaussian-reduction](https://github.com/ma-gilles/relax/blob/main/relax/reference/gaussian_reduction_replay.py) replays |

Pipeline PPCA and pose-marginal PPCA have different entry points and state
contracts. Choose the implementation reached by the actual command. The
pipeline PPCA path currently rejects tilt-series input. Consult the
[PPCA refinement guide](https://github.com/ma-gilles/relax/blob/main/relax/ppca_refinement/AGENTS.md) when working
on pose refinement, and the [paper-data runbook](della.md) for pinned inputs.

## EM code

The RELION-style EM refinement, InitialModel/VDAM, pose-marginal PPCA refinement,
their CUDA library, RELION bindings, diagnostics and GT reporting now live in the
[relax repository](https://github.com/ma-gilles/relax). Its [codebase map](https://github.com/ma-gilles/relax/blob/main/docs/development/codebase.md)
describes that package layout, ownership boundaries and code budgets.

## Shared data and numerical boundaries

| Boundary | Owner | Contract to inspect |
| --- | --- | --- |
| Particle loading and batch identity | [`CryoEMDataset`](../../recovar/data_io/cryoem_dataset.py), image loaders and half-set utilities | Original image/particle IDs, subset-local positions, half-set membership, image backend and CTF metadata |
| Forward-model configuration and state | [`core.configs`](../../recovar/core/configs.py) | `ForwardModelConfig` static fields versus dynamic `ModelState` arrays; changing a static value may change JIT specialization |
| Fourier transforms and volume I/O | [`fourier_transform_utils`](../../recovar/core/fourier_transform_utils.py), [`utils.helpers`](../../recovar/utils/helpers.py) | Centered Fourier conventions, flattened arrays, full versus half spectrum, and the RELION axis/sign conversion |
| Mean, noise and regularization | [`homogeneous`](../../recovar/reconstruction/homogeneous.py), [`noise`](../../recovar/reconstruction/noise.py), [`regularization`](../../recovar/reconstruction/regularization.py); EM-only RELION variants in [`relax.reconstruction`](https://github.com/ma-gilles/relax/blob/main/relax/reconstruction/regularization_relion.py) | Half-set ownership, shell support, normalization, prior construction and reconstruction units |
| Saved results | [`output`](../../recovar/output/output.py), [`ResultPaths`](../../recovar/output/output_paths.py) | Serialized field names, shapes, original IDs, and downstream `PipelineOutput` consumers |
| CUDA and RELION references | [`cuda_backproject`](../../recovar/cuda_backproject.py) (pipeline library `libcuda_backproject.so` and its loader), [`cuda_build`](../../recovar/cuda_build.py) (`NativeLibrary`, public CUDA headers), [`relax.cuda.kernels`](https://github.com/ma-gilles/relax/blob/main/relax/cuda/kernels.py) (EM library), [`relax.relion_bind`](https://github.com/ma-gilles/relax/blob/main/relax/relion_bind/__init__.py) | Loaded binary identity, device placement, native layouts and independent reference behavior |

The [source conventions](../../recovar/CLAUDE.md) give the exact FFT and
RELION-frame rules. Follow those helpers when loading volumes for a comparison;
raw MRC arrays and uncentered FFT calls are not interchangeable with them.

