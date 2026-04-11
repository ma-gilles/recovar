"""
CUDA backprojector / projector — JAX JIT-compatible via XLA FFI.

Provides ``backproject`` and ``project`` that drop into ``@jax.jit``
compiled functions.  Also exposes a low-level ctypes path for
standalone benchmarks.

Quick start::

    from recovar.cuda_backproject import backproject, project

    @jax.jit
    def step(vol, images, rots):
        vol = backproject(vol, images, rots, image_shape, volume_shape, order=1)
        imgs = project(vol, rots, image_shape, volume_shape, order=1)
        return vol, imgs
"""

from __future__ import annotations

import ctypes
import functools
import logging
import pathlib
import subprocess
import threading
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Build / load shared library
# ──────────────────────────────────────────────────────────────────────

_LIB_DIR = pathlib.Path(__file__).resolve().parent / "cuda"
_LIB_PATH = _LIB_DIR / "libcuda_backproject.so"
_lib_handle = None  # ctypes CDLL


def _build_lib():
    if _LIB_PATH.exists():
        return
    import sys

    logger.info("Building %s …", _LIB_PATH)
    # Pass the current Python so the Makefile finds JAX headers correctly.
    subprocess.check_call(["make", "-C", str(_LIB_DIR), f"PYTHON={sys.executable}"])
    if not _LIB_PATH.exists():
        raise RuntimeError(f"Build failed — {_LIB_PATH} not found")


def _cuda_fallback_warning(exc: BaseException) -> str:
    return (
        f"Could not build or load RECOVAR's CUDA extension ({exc}). "
        "Falling back to the JAX implementation. JAX GPU support alone is not enough to build "
        "this extension: ensure a local CUDA toolkit/compiler is available via NVCC or CUDACXX, "
        "make sure nvcc is on PATH, or set LOCAL_CUDA_PATH/CUDA_HOME/CUDA_PATH."
    )


def _get_lib():
    global _lib_handle
    if _lib_handle is None:
        _build_lib()
        _lib_handle = ctypes.CDLL(str(_LIB_PATH))
    return _lib_handle


# ──────────────────────────────────────────────────────────────────────
# Register XLA FFI targets  (done once, thread-safe)
# ──────────────────────────────────────────────────────────────────────

_ffi_registered = False
_ffi_lock = threading.Lock()

# FFI target name constants
_TARGET_BACKPROJECT = "cuda_backproject"
_TARGET_PROJECT = "cuda_project"
_TARGET_BATCH_BACKPROJECT = "cuda_batch_backproject"
_TARGET_BATCH_PROJECT = "cuda_batch_project"


def _ensure_ffi():
    global _ffi_registered
    if _ffi_registered:
        return
    with _ffi_lock:
        if _ffi_registered:
            return
        lib = _get_lib()
        jax.ffi.register_ffi_target(_TARGET_BACKPROJECT, jax.ffi.pycapsule(lib.Backproject), platform="CUDA")
        jax.ffi.register_ffi_target(_TARGET_PROJECT, jax.ffi.pycapsule(lib.Project), platform="CUDA")
        jax.ffi.register_ffi_target(_TARGET_BATCH_BACKPROJECT, jax.ffi.pycapsule(lib.BatchBackproject), platform="CUDA")
        jax.ffi.register_ffi_target(_TARGET_BATCH_PROJECT, jax.ffi.pycapsule(lib.BatchProject), platform="CUDA")
        _ffi_registered = True
        logger.debug("Registered CUDA FFI targets")


_cuda_ok = None  # cached result: None = not checked, True/False = result


def cuda_available() -> bool:
    """Return True if CUDA backproject/project kernels can be used (cached).

    Set env var ``RECOVAR_DISABLE_CUDA=1`` to force-disable.
    """
    global _cuda_ok
    if _cuda_ok is not None:
        return _cuda_ok
    import os

    if os.environ.get("RECOVAR_DISABLE_CUDA", "0") == "1":
        _cuda_ok = False
        logger.info("CUDA kernels disabled via RECOVAR_DISABLE_CUDA")
        return _cuda_ok
    try:
        if not any(d.platform == "gpu" for d in jax.devices()):
            _cuda_ok = False
        else:
            _ensure_ffi()
            _cuda_ok = True
            logger.info("CUDA backproject/project kernels enabled")
    except (ImportError, OSError, RuntimeError, AttributeError, subprocess.SubprocessError) as e:
        _cuda_ok = False
        logger.warning(_cuda_fallback_warning(e))
        logger.debug("CUDA backproject not available", exc_info=True)
    return _cuda_ok


# ──────────────────────────────────────────────────────────────────────
# Public JAX API  (JIT-compatible)
# ──────────────────────────────────────────────────────────────────────


def _rot_to_compact(rotation_matrices: jax.Array, real_dtype=None) -> jax.Array:
    """Extract first two rows of each 3×3 rotation matrix → (n, 6).

    Rows are swapped so that the CUDA kernel's row-major pixel loop
    (k0=row, k1=col) matches the JAX coordinate convention established by
    ``get_k_coordinate_of_each_pixel(..., indexing="xy")`` where
    coord[0]=col_freq and coord[1]=row_freq.

    Without the swap, CUDA computes  rk = k0*R[0,:] + k1*R[1,:]
    = row_freq*R[0,:] + col_freq*R[1,:], but JAX expects
    col_freq*R[0,:] + row_freq*R[1,:].  Swapping the two rows fixes this.

    Parameters
    ----------
    real_dtype : optional dtype to cast the result to.  The CUDA kernel
        reads these as ``T*`` where ``T`` matches the volume's real component
        (float32 for C64, float64 for C128).  If the rotation matrices have a
        different dtype (e.g. float64 rotations with a complex64 volume), the
        kernel will reinterpret the bytes incorrectly.
    """
    n = rotation_matrices.shape[0]
    compact = rotation_matrices[:, [1, 0], :].reshape(n, 6)
    if real_dtype is not None:
        compact = compact.astype(real_dtype)
    return compact


def _volume_real_dtype(volume: jax.Array):
    """Return the real component dtype of a volume (float32 for complex64, etc.)."""
    return jnp.finfo(volume.dtype).dtype if jnp.issubdtype(volume.dtype, jnp.complexfloating) else volume.dtype


def _validate_inputs(volume_shape, image_shape, order, half_volume, half_image):
    """Validate parameters at trace time (not inside JIT)."""
    ih, iw = image_shape
    N0, N1, N2 = volume_shape
    if ih <= 0 or iw <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    if N0 <= 0 or N1 <= 0 or N2 <= 0:
        raise ValueError(f"volume_shape must be positive, got {volume_shape}")
    if order not in (0, 1, 3):
        raise ValueError(f"order must be 0, 1, or 3, got {order}")
    if N0 % ih != 0:
        raise ValueError(f"volume_shape[0] ({N0}) must be divisible by image_shape[0] ({ih})")


def _encode_max_r(max_r):
    """Encode max_r as int64 max_r2_x4 for FFI (quarter-pixel² precision).

    -1 means disabled (no sphere clipping).
    """
    if max_r is None:
        return np.int64(-1)
    return np.int64(int(round(float(max_r) * float(max_r) * 4)))


def _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r=None):
    """Compute the shared FFI scalar keyword arguments (used by all 4 targets)."""
    ih, iw_full = image_shape
    N0, N1, N2 = volume_shape
    ups = N0 // ih
    iw_eff = iw_full // 2 + 1 if half_image else iw_full
    return (
        dict(
            image_h=np.int64(ih),
            image_w=np.int64(iw_eff),
            N0=np.int64(N0),
            N1=np.int64(N1),
            N2=np.int64(N2),
            upsampling=np.int64(ups),
            order=np.int64(order),
            half_volume=np.int64(int(half_volume)),
            half_image=np.int64(int(half_image)),
            full_image_w=np.int64(iw_full),
            max_r2_x4=_encode_max_r(max_r),
        ),
        ih,
        iw_eff,
    )


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def backproject(
    volume: jax.Array,
    images: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
) -> jax.Array:
    """Back-project *images* into *volume* (accumulate in-place via aliasing).

    Parameters
    ----------
    volume : complex64 | complex128 | float32 | float64, shape ``(prod(vol_shape),)``
        Existing volume to accumulate into.  Pass zeros for a fresh start.
        For real-valued Fourier quantities (CTF^2, noise variance), use
        float32/float64 for 2x memory and scatter efficiency.
    images : same dtype as volume, shape ``(n_images, n_pixels)``
    rotation_matrices : float32 | float64, shape ``(n_images, 3, 3)``
    image_shape : (H, W) — real-space image dimensions.
        When half_image=True, images have shape ``(n, H*(W//2+1))``.
    volume_shape : (N0, N1, N2)   — full dimensions even when half_volume=True.
    order : 0 (nearest) or 1 (trilinear).
    half_volume : if True, volume is rfft-packed ``(N0*N1*(N2//2+1),)``.
    half_image : if True, images are rfft-packed ``(n, H*(W//2+1))``.
        Hermitian conjugates are scattered automatically.
    max_r : if not None, skip pixels whose rotated frequency radius
        exceeds this value (RELION-style sphere clipping).

    Returns
    -------
    Updated volume (same shape and dtype as *volume*).
    """
    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image)
    kw, ih, iw_eff = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volume))
    out_type = jax.ShapeDtypeStruct(volume.shape, volume.dtype)

    return jax.ffi.ffi_call(
        _TARGET_BACKPROJECT,
        out_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(images, rot6, volume, **kw)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def project(
    volume: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
) -> jax.Array:
    """Project *volume* to 2D images.

    Parameters
    ----------
    half_image : if True, output images are rfft-packed ``(n, H*(W//2+1))``.
    max_r : if not None, zero pixels whose rotated frequency radius
        exceeds this value (RELION-style sphere clipping).

    Returns
    -------
    complex array, shape ``(n_images, n_pixels)``  (n_pixels = H*W or H*(W//2+1)).
    """
    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image)
    kw, ih, iw_eff = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    n_images = rotation_matrices.shape[0]
    n_pixels = ih * iw_eff
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volume))
    out_type = jax.ShapeDtypeStruct((n_images, n_pixels), volume.dtype)

    return jax.ffi.ffi_call(
        _TARGET_PROJECT,
        out_type,
        vmap_method="sequential",
    )(volume, rot6, **kw)


# ──────────────────────────────────────────────────────────────────────
# Batched API  (multiple volumes, shared rotations)
# ──────────────────────────────────────────────────────────────────────


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def batch_backproject(
    volumes: jax.Array,
    images: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
) -> jax.Array:
    """Back-project images into a batch of volumes in a single kernel launch.

    Parameters
    ----------
    volumes : complex | real, shape ``(batch, vol_flat_size)``
        Existing volumes to accumulate into.  Supports float32/float64
        for real-valued Fourier quantities (2x efficiency).
    images : same dtype as volumes, shape ``(batch, n_images, n_pixels)``
        Per-volume images (e.g. differently weighted).
    rotation_matrices : real, shape ``(n_images, 3, 3)``
        Shared across all volumes in the batch.
    image_shape, volume_shape, order, half_volume, half_image :
        Same semantics as ``backproject()``.
    max_r : if not None, skip pixels whose rotated frequency radius
        exceeds this value (RELION-style sphere clipping).

    Returns
    -------
    Updated volumes, shape ``(batch, vol_flat_size)``.
    """
    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image)
    kw, ih, iw_eff = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volumes))
    out_type = jax.ShapeDtypeStruct(volumes.shape, volumes.dtype)

    return jax.ffi.ffi_call(
        _TARGET_BATCH_BACKPROJECT,
        out_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(images, rot6, volumes, **kw)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def batch_project(
    volumes: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
) -> jax.Array:
    """Project a batch of volumes to 2D images via vmap over single-volume project.

    Parameters
    ----------
    volumes : complex, shape ``(batch, vol_flat_size)``
    rotation_matrices : real, shape ``(n_images, 3, 3)``
        Shared across all volumes in the batch.
    max_r : if not None, zero pixels whose rotated frequency radius
        exceeds this value (RELION-style sphere clipping).

    Returns
    -------
    complex array, shape ``(batch, n_images, n_pixels)``.
    """
    # TODO: The underlying CUDA batch_project_kernel loops over volumes
    # sequentially per-thread (`for b in 0..batch_size`), causing L2 cache
    # thrashing when batch is large (e.g. 161 basis vectors × 256³ volumes).
    # Each thread jumps 128 MB between volumes, far exceeding L2 capacity
    # (40 MB on A100), so every read is a cache miss.
    #
    # Fix options for the CUDA kernel:
    #   1. Parallelize over volumes in the grid dimension (one thread-block
    #      per volume×image×pixel-tile) so all threads in a block read from
    #      the same volume → cache-friendly.
    #   2. Tile the volume batch loop with shared memory staging.
    #
    # For now, vmap over single-volume `project` is ~30-40x faster for large
    # batches because each kernel launch processes one volume that stays
    # cache-hot for all images.
    return jax.vmap(
        lambda v: project(
            v,
            rotation_matrices,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_volume,
            half_image=half_image,
            max_r=max_r,
        )
    )(volumes)


# ──────────────────────────────────────────────────────────────────────
# ctypes helpers  (for standalone benchmarks without JAX JIT overhead)
# ──────────────────────────────────────────────────────────────────────

_cudart = None


def _get_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    import glob as _glob

    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            _cudart = ctypes.CDLL(name)
            return _cudart
        except OSError:
            continue
    for p in sorted(_glob.glob("/usr/local/cuda*/lib64/libcudart.so"), reverse=True):
        try:
            _cudart = ctypes.CDLL(p)
            return _cudart
        except OSError:
            continue
    raise RuntimeError("Cannot find libcudart.so")


class GpuArray:
    """Minimal GPU allocation managed via cudart."""

    def __init__(self, data: np.ndarray):
        self.shape, self.dtype, self.nbytes = data.shape, data.dtype, data.nbytes
        data = np.ascontiguousarray(data)
        rt = _get_cudart()
        self._ptr = ctypes.c_void_p()
        assert rt.cudaMalloc(ctypes.byref(self._ptr), ctypes.c_size_t(self.nbytes)) == 0
        assert (
            rt.cudaMemcpy(
                self._ptr, data.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(self.nbytes), ctypes.c_int(1)
            )
            == 0
        )

    def as_float_ptr(self):
        return ctypes.cast(self._ptr, ctypes.POINTER(ctypes.c_float))

    def to_numpy(self):
        out = np.empty(self.shape, dtype=self.dtype)
        _get_cudart().cudaMemcpy(
            out.ctypes.data_as(ctypes.c_void_p), self._ptr, ctypes.c_size_t(self.nbytes), ctypes.c_int(2)
        )
        return out

    def free(self):
        if self._ptr:
            try:
                _get_cudart().cudaFree(self._ptr)
            except Exception:
                logger.debug("cudaFree failed", exc_info=True)
            self._ptr = ctypes.c_void_p()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass  # destructors must not raise


def _random_rotations_6(n, rng=None):
    """(n, 6) float32: first two rows of random rotation matrices."""
    if rng is None:
        rng = np.random.default_rng()
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return q[:, :2, :].reshape(n, 6).astype(np.float32)


class CudaBenchmarker:
    """Benchmark helper using ctypes (no JAX overhead)."""

    def __init__(self, image_shape, volume_shape, order=1, half_volume=False, half_image=False):
        self.ih, self.iw_full = image_shape
        self.N0, self.N1, self.N2 = volume_shape
        self.order = order
        self.half_volume = int(half_volume)
        self.half_image = int(half_image)
        self.ups = self.N0 // self.ih
        self.center = float(self.N0 // 2)
        self.N2_eff = self.N2 // 2 + 1 if half_volume else self.N2

        if half_image:
            self.iw = self.iw_full // 2 + 1
        else:
            self.iw = self.iw_full
        self.n_pixels = self.ih * self.iw

        self._lib = _get_lib()

    def benchmark(self, n_images, n_iters=100, kind="backproject"):
        rng = np.random.default_rng(42)
        vol_size = self.N0 * self.N1 * self.N2_eff
        vol_f32 = rng.standard_normal(vol_size * 2).astype(np.float32)
        img_f32 = rng.standard_normal(n_images * self.n_pixels * 2).astype(np.float32)
        rots = _random_rotations_6(n_images, rng)

        vol_d = GpuArray(vol_f32)
        img_d = GpuArray(img_f32)
        rot_d = GpuArray(rots)

        fn = self._lib.benchmark_backproject_c if kind == "backproject" else self._lib.benchmark_project_c
        fn.restype = ctypes.c_float
        ms = fn(
            vol_d.as_float_ptr(),
            img_d.as_float_ptr(),
            rot_d.as_float_ptr(),
            n_images,
            self.n_pixels,
            self.ih,
            self.iw,
            self.N0,
            self.N1,
            self.N2,
            self.ups,
            ctypes.c_float(self.center),
            self.order,
            self.half_volume,
            self.half_image,
            self.iw_full,
            n_iters,
        )

        vol_d.free()
        img_d.free()
        rot_d.free()
        return {
            "ms_total": float(ms),
            "ms_per_iter": float(ms) / n_iters,
            "throughput_img_per_s": n_images * n_iters / (float(ms) / 1000.0),
        }
