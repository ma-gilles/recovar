# CUDA Kernel Development

## Build
```bash
PYTHON="$(pixi run which python)" make -C recovar/cuda clean all
```
The Makefile uses `jax.ffi.include_dir()` to locate JAX XLA FFI headers. Always use the pixi Python.

For the default cache-backed build path:
```bash
pixi run build-custom-cuda
```

## Architecture Targets
SM 80 (A100), 86 (RTX 3090), 89 (RTX 4090), 90 (H100). All are compiled into `libcuda_backproject.so`.

## Runtime behavior
`recovar/cuda_backproject.py` does not auto-compile the kernel on import or first use. Standard GPU execution uses JAX by default. To use RECOVAR's custom CUDA kernels, build the shared library explicitly and set `RECOVAR_ENABLE_CUSTOM_CUDA=1`.

## Interface
The kernel is exposed via JAX XLA FFI (custom call). `adjoint_slice_volume()` in `core/slicing.py` only dispatches to these kernels when `RECOVAR_ENABLE_CUSTOM_CUDA=1`; otherwise it uses the default JAX implementation.

## Coordinate Convention
CUDA kernels use k0=row, k1=col. JAX `meshgrid(indexing="xy")` has coord[0]=col, coord[1]=row. This mismatch caused a historical bug — any changes to coordinate handling must be validated against the `test_cuda_jax_equivalence.py` tests.
