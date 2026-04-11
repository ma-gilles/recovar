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
`recovar/cuda_backproject.py` auto-builds the kernel into the cache-backed shared library path on first GPU use when needed. Standard GPU execution prefers RECOVAR's custom CUDA kernels by default because they are faster than the JAX fallback. If the custom build/load fails, RECOVAR raises a clear error. `RECOVAR_DISABLE_CUDA=1` forces the slower JAX GPU path.

## Interface
The kernel is exposed via JAX XLA FFI (custom call). `adjoint_slice_volume()` in `core/slicing.py` dispatches to these kernels by default on GPU unless `RECOVAR_DISABLE_CUDA=1` is set.

## Coordinate Convention
CUDA kernels use k0=row, k1=col. JAX `meshgrid(indexing="xy")` has coord[0]=col, coord[1]=row. This mismatch caused a historical bug — any changes to coordinate handling must be validated against the `test_cuda_jax_equivalence.py` tests.
