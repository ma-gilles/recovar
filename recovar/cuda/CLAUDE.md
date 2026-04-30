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
Minimum compute capability: **7.0** (Volta). The default `CUDA_ARCH` compiles SASS for sm_70 (V100), sm_75 (T4, RTX 20-series), sm_80 (A100), sm_86 (RTX 30-series, A40), sm_89 (RTX 40-series, L40), sm_90 (H100, H200), plus a compute_75 PTX fallback for future architectures. Pascal (sm_60/61) is opt-in via `CUDA_ARCH` override.

To rebuild for a specific GPU:
```bash
cd recovar/cuda
make clean
make CUDA_ARCH="-gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60"
```

As a temporary alternative, `RECOVAR_DISABLE_CUDA=1` forces the slower JAX-native path (≈2x slower; matches recovar 0.4.5 behavior).

## Runtime behavior
`recovar/cuda_backproject.py` auto-builds the kernel into the cache-backed shared library path on first GPU use when needed. Standard GPU execution prefers RECOVAR's custom CUDA kernels by default because they are faster than the JAX fallback. If the custom build/load fails, RECOVAR raises a clear error. `RECOVAR_DISABLE_CUDA=1` forces the slower JAX GPU path.

## Interface
The kernel is exposed via JAX XLA FFI (custom call). `adjoint_slice_volume()` in `core/slicing.py` dispatches to these kernels by default on GPU unless `RECOVAR_DISABLE_CUDA=1` is set.

## Coordinate Convention
CUDA kernels use k0=row, k1=col. JAX `meshgrid(indexing="xy")` has coord[0]=col, coord[1]=row. This mismatch caused a historical bug — any changes to coordinate handling must be validated against the `test_cuda_jax_equivalence.py` tests.
