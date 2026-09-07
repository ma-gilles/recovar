# CUDA kernel development

Read the [source conventions](../CLAUDE.md),
[environment and validation contract](../../CONTRIBUTING.md), and
[Della resource policy](../../docs/development/della.md) before GPU work.
Use the checkout's pixi Python so compilation uses its JAX FFI headers.

## Build and library ownership

For a development build in the default cache:

```bash
pixi run build-custom-cuda
```

For a benchmark, first create an exclusive `RUN_ROOT`. Build into that run's
library directory instead of rebuilding a library another process may use:

```bash
# RUN_ROOT must identify a new run directory.
PIXI_PY="$(pixi run which python)"
PYTHON="$PIXI_PY" make -C recovar/cuda LIB="$RUN_ROOT/library/libcuda_backproject.so" all
sha256sum "$RUN_ROOT/library/libcuda_backproject.so"
```

Record source, lockfile, compiler, JAX headers and binary identity. Follow the
qualification procedure in CONTRIBUTING.md: load a fresh, immutable copy;
verify its actual loaded path and hash before loading, immediately afterward,
and after each run. Preserve the scheduler's GPU assignment and private caches.

## Architecture selection

The [Makefile](Makefile) selects targets from the detected nvcc version:

- `sm_80`, `sm_86`, `sm_89` and `sm_90` are in the default target list.
- `sm_70` and `sm_75` are included only when nvcc is older than 13.
- nvcc 12.8 and newer add `sm_100`, `sm_120` and `compute_120` PTX.
  Earlier versions use `compute_90` PTX as the fallback.

These are build-selection rules, not proof that every toolkit/driver/device
combination works. `CUDA_ARCH` can override the list when the selected compiler
supports the requested targets. Record overrides with benchmark results.

## Runtime and interface

`recovar/cuda_backproject.py` registers kernels through JAX XLA FFI.
`core/slicing.py` dispatches projection/backprojection operations to custom
CUDA by default on GPU. The loader can build a missing or stale library;
staleness includes source/Makefile modification times and missing FFI symbols.
An explicit `RECOVAR_CUDA_LIB` path alone does not establish binary identity.

`RECOVAR_DISABLE_CUDA=1` selects the JAX-native path. Treat this as a different
execution backend and record it explicitly; its speed and memory use require
measurement on the workload being compared.

## Coordinates and validation

CUDA kernels use k0=row, k1=column. JAX `meshgrid(indexing="xy")` produces
coord[0]=column, coord[1]=row. Preserve the explicit conversion, Fourier frame,
packed layout, dtypes and accumulation order when changing a kernel boundary.

Use [build-configuration tests](../../tests/unit/test_cuda_build_config.py)
for build changes and the affected
[CUDA/JAX equivalence tests](../../tests/unit/test_cuda_jax_equivalence.py)
for numerical changes. Follow the scoped EM ladder for RELION-specific paths.
Run GPU checks under the assigned visibility with the validated library;
successful import alone is not kernel or scientific qualification.
