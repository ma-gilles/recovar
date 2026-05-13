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
import os
import pathlib
import subprocess
import threading
from contextlib import contextmanager
from types import ModuleType
from typing import Tuple

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows does not support custom CUDA builds
    fcntl: ModuleType | None = None
else:
    fcntl = _fcntl

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Build / load shared library
# ──────────────────────────────────────────────────────────────────────

_LIB_DIR = pathlib.Path(__file__).resolve().parent / "cuda"
_PACKAGE_LIB_PATH = _LIB_DIR / "libcuda_backproject.so"
_lib_handle = None  # ctypes CDLL
_loaded_lib_path = None

_DISABLE_CUSTOM_CUDA_ENV = "RECOVAR_DISABLE_CUDA"
_CUDA_LIB_ENV = "RECOVAR_CUDA_LIB"
_CUDA_CACHE_DIR_ENV = "RECOVAR_CUDA_CACHE_DIR"
_BUILD_LOCKFILE = ".build.lock"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def custom_cuda_requested() -> bool:
    """Return True unless the user explicitly disables custom CUDA."""
    return not _env_flag(_DISABLE_CUSTOM_CUDA_ENV)


def _cache_root() -> pathlib.Path:
    override = os.environ.get(_CUDA_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return pathlib.Path(xdg_cache_home).expanduser() / "recovar" / "cuda"

    return pathlib.Path.home().expanduser() / ".cache" / "recovar" / "cuda"


def _cached_lib_path() -> pathlib.Path:
    return _cache_root() / "libcuda_backproject.so"


def _configured_lib_path() -> pathlib.Path | None:
    override = os.environ.get(_CUDA_LIB_ENV)
    if not override:
        return None
    return pathlib.Path(override).expanduser()


def _candidate_lib_paths() -> list[pathlib.Path]:
    candidates = []
    for path in (_configured_lib_path(), _cached_lib_path(), _PACKAGE_LIB_PATH):
        if path is None:
            continue
        resolved = path.expanduser()
        if resolved not in candidates:
            candidates.append(resolved)
    return candidates


def _lib_is_stale(lib_path: pathlib.Path) -> bool:
    """Return True if the lib's mtime is older than the source files.

    Catches the case where a user installed before a kernel/Makefile fix
    landed (e.g. issue #131's Blackwell widening): without this check,
    `_existing_lib_path()` would happily return the stale cached `.so`
    forever, and the user would never pick up the new arch coverage.
    """
    try:
        lib_mtime = lib_path.stat().st_mtime
    except OSError:
        return False
    for src_name in ("cuda_backproject.cu", "Makefile"):
        src = _LIB_DIR / src_name
        try:
            if src.stat().st_mtime > lib_mtime:
                return True
        except OSError:
            continue
    return False


def _existing_lib_path() -> pathlib.Path | None:
    for candidate in _candidate_lib_paths():
        if candidate.exists():
            if _lib_is_stale(candidate):
                logger.info(
                    "RECOVAR CUDA library %s is older than its source — "
                    "will rebuild (this happens once after a kernel/Makefile update).",
                    candidate,
                )
                continue
            return candidate.resolve()
    return None


def _default_build_lib_path() -> pathlib.Path:
    configured = _configured_lib_path()
    if configured is not None:
        return configured.expanduser().resolve()
    return _cached_lib_path().resolve()


@contextmanager
def _build_file_lock(lock_path: pathlib.Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fd:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)


def _build_lock_path(lib_path: pathlib.Path) -> pathlib.Path:
    return lib_path.parent / _BUILD_LOCKFILE


def build_custom_cuda(output_path: str | os.PathLike[str] | None = None, force: bool = False) -> pathlib.Path:
    """Build RECOVAR's preferred custom CUDA extension and return its path."""
    import sys

    global _auto_build_attempted, _auto_build_error, _cuda_ok, _ffi_registered, _lib_handle, _loaded_lib_path

    lib_path = pathlib.Path(output_path).expanduser().resolve() if output_path else _default_build_lib_path()
    stale = lib_path.exists() and _lib_is_stale(lib_path)
    if lib_path.exists() and not force and not stale:
        logger.info("Using existing RECOVAR CUDA extension at %s", lib_path)
        _auto_build_attempted = True
        _auto_build_error = None
        return lib_path
    if stale:
        logger.info(
            "RECOVAR CUDA extension at %s is older than its source — rebuilding.",
            lib_path,
        )

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building %s", lib_path)
    make_cmd = ["make"]
    if force or stale:
        make_cmd.append("-B")
    make_cmd.extend(["-C", str(_LIB_DIR), f"PYTHON={sys.executable}", f"LIB={lib_path}"])
    subprocess.check_call(make_cmd)
    if not lib_path.exists():
        raise RuntimeError(f"Build failed — {lib_path} not found")
    _auto_build_attempted = True
    _auto_build_error = None
    _cuda_ok = None
    _ffi_registered = False
    _lib_handle = None
    _loaded_lib_path = None
    return lib_path


def _cuda_unavailable_message(exc: BaseException | None = None) -> str:
    searched = ", ".join(str(path) for path in _candidate_lib_paths())
    detail = ""
    if exc is not None:
        detail = f" Last error: {exc!s}."
    return (
        "RECOVAR's preferred custom CUDA backproject/project extension is unavailable."
        f"{detail} RECOVAR tries to use these kernels by default on GPU because they are substantially faster. "
        "Fix your CUDA compiler setup (`NVCC`, `CUDACXX`, `PATH`, `LOCAL_CUDA_PATH`, `CUDA_HOME`, or `CUDA_PATH`) "
        "or run `recovar build_custom_cuda` to build the shared library manually. "
        "If you need to bypass this temporarily, set `RECOVAR_DISABLE_CUDA=1` to force the slower JAX GPU path; "
        f"that workaround is supported but not preferred. Searched: {searched}"
    )


def cuda_unavailable_error() -> RuntimeError:
    return RuntimeError(_cuda_unavailable_message(_auto_build_error))


def _missing_lib_error(exc: BaseException | None = None) -> RuntimeError:
    return RuntimeError(_cuda_unavailable_message(exc))


_auto_build_attempted = False
_auto_build_error = None
_auto_build_lock = threading.Lock()


def _ensure_lib_path() -> pathlib.Path | None:
    global _auto_build_attempted, _auto_build_error

    existing = _existing_lib_path()
    if existing is not None:
        return existing.resolve()

    target = _default_build_lib_path()
    with _auto_build_lock:
        existing = _existing_lib_path()
        if existing is not None:
            return existing.resolve()
        if _auto_build_attempted:
            return None

        _auto_build_attempted = True
        with _build_file_lock(_build_lock_path(target)):
            existing = _existing_lib_path()
            if existing is not None:
                _auto_build_error = None
                return existing.resolve()
            try:
                built = build_custom_cuda(output_path=target)
            except Exception as exc:  # pragma: no cover - exercised in GPU envs
                _auto_build_error = exc
                logger.debug("Automatic CUDA build failed", exc_info=True)
                return None

        _auto_build_error = None
        return built.resolve()


def _get_lib():
    global _lib_handle, _loaded_lib_path
    lib_path = _existing_lib_path()
    if lib_path is None:
        lib_path = _ensure_lib_path()
    if lib_path is None:
        raise _missing_lib_error(_auto_build_error)
    lib_path = pathlib.Path(lib_path).resolve()

    if _lib_handle is None or _loaded_lib_path != lib_path:
        _lib_handle = ctypes.CDLL(str(lib_path))
        _loaded_lib_path = lib_path
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
_TARGET_PER_IMAGE_BACKPROJECT = "cuda_per_image_backproject"
_TARGET_BATCH_PROJECT = "cuda_batch_project"


_preflight_ok: bool | None = None  # None = not checked yet


def _detect_gpu_compute_cap() -> tuple[str, str] | None:
    """Return (gpu_name, compute_cap) like ("NVIDIA A100", "80"), or None."""
    # Try JAX first
    try:
        dev = jax.devices("gpu")[0]
        cap = getattr(dev, "compute_capability", None)
        if cap:
            name = getattr(dev, "device_kind", "GPU")
            # cap may be "8.0" or "80" or (8, 0)
            if isinstance(cap, tuple):
                cap_str = f"{cap[0]}{cap[1]}"
            elif isinstance(cap, str) and "." in cap:
                cap_str = cap.replace(".", "")
            else:
                cap_str = str(cap)
            return name, cap_str
    except Exception:
        pass
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            if len(parts) == 2:
                name = parts[0].strip()
                cap_str = parts[1].strip().replace(".", "")
                return name, cap_str
    except Exception:
        pass
    return None


def _detect_so_arches(so_path: pathlib.Path) -> tuple[set[str], set[str]]:
    """Return (sass_arches, ptx_arches) from cuobjdump --list-elf."""
    sass: set[str] = set()
    ptx: set[str] = set()
    try:
        result = subprocess.run(
            ["cuobjdump", "--list-elf", str(so_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import re

            for line in result.stdout.splitlines():
                m = re.search(r"sm_(\d+)", line)
                if m:
                    sass.add(m.group(1))
        # Also check for PTX
        result2 = subprocess.run(
            ["cuobjdump", "--list-ptx", str(so_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result2.returncode == 0:
            import re

            for line in result2.stdout.splitlines():
                m = re.search(r"sm_(\d+)", line)
                if m:
                    ptx.add(m.group(1))
    except Exception:
        pass
    return sass, ptx


def _detect_nvcc_version() -> str | None:
    """Return nvcc version string like '12.8.93' or None."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import re

            m = re.search(r"release (\d+\.\d+)", result.stdout)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _preflight_check(so_path: pathlib.Path) -> None:
    """One-time check that the loaded .so supports the running GPU.

    Raises RuntimeError with a detailed, actionable message if the GPU's
    compute capability is not covered by the .so's compiled targets.
    Silently succeeds (logs a warning) if the probe tools are unavailable.
    """
    global _preflight_ok
    if _preflight_ok is not None:
        return
    _preflight_ok = True  # assume OK; set False only on confirmed mismatch

    gpu_info = _detect_gpu_compute_cap()
    if gpu_info is None:
        logger.warning(
            "recovar could not preflight the CUDA kernel against your GPU; "
            "if the next call fails with 'no kernel image', see "
            "https://github.com/ma-gilles/recovar/issues/131"
        )
        return

    gpu_name, gpu_cap = gpu_info
    sass_arches, ptx_arches = _detect_so_arches(so_path)
    if not sass_arches and not ptx_arches:
        # cuobjdump not available — can't check
        logger.warning(
            "recovar could not inspect the CUDA kernel targets (cuobjdump not found); "
            "if the next call fails with 'no kernel image', see "
            "https://github.com/ma-gilles/recovar/issues/131"
        )
        return

    # Check if GPU is covered by SASS or compatible PTX
    gpu_cap_int = int(gpu_cap)
    sass_covered = gpu_cap in sass_arches
    ptx_covered = any(int(p) <= gpu_cap_int for p in ptx_arches)

    if sass_covered or ptx_covered:
        _preflight_ok = True
        return

    # Not covered — build the detailed error message
    _preflight_ok = False
    nvcc_ver = _detect_nvcc_version()
    makefile_dir = str(_LIB_DIR)
    makefile_path = str(_LIB_DIR / "Makefile")
    so_arches_str = ", ".join(f"sm_{a}" for a in sorted(sass_arches))
    ptx_desc = f"targets {', '.join(f'sm_{p}' for p in sorted(ptx_arches))} or higher only" if ptx_arches else "none"

    msg = f"""\
recovar's custom CUDA kernel cannot run on your GPU.

What's going wrong
------------------
Your GPU:           {gpu_name} (compute capability sm_{gpu_cap})
Loaded kernel:      {so_path}
Compiled targets:   {so_arches_str}
PTX fallback:       {ptx_desc}

The kernel was built only for compute capabilities that don't include
yours, so the CUDA driver has no executable code to dispatch to. This
is what produced the underlying "no kernel image is available for
execution on the device" error.

The right fix (recommended)
---------------------------
Rebuild the kernel for your GPU. This restores full performance.

    cd {makefile_dir}
    make clean
    make CUDA_ARCH="-gencode arch=compute_{gpu_cap},code=sm_{gpu_cap} \\
                    -gencode arch=compute_{gpu_cap},code=compute_{gpu_cap}"

Then re-run your recovar command. The new .so will live at {so_path}."""

    # CUDA 13 edge case
    if nvcc_ver and nvcc_ver.startswith("13.") and gpu_cap_int < 75:
        msg += f"""

NOTE: nvcc {nvcc_ver} (CUDA toolkit 13) does not support compute
capabilities below 7.5. Your GPU (sm_{gpu_cap}) requires a CUDA 12
toolkit. Install one alongside, e.g.:
    conda install -c nvidia cuda-toolkit=12.4
or download from https://developer.nvidia.com/cuda-12-4-0-download-archive
Then re-run the make command above with that nvcc on PATH."""

    msg += f"""

The temporary bypass (slower)
-----------------------------
If you can't rebuild right now, set this environment variable to fall
back to the JAX-native projection/backprojection path:

    export RECOVAR_DISABLE_CUDA=1

This is the same code path recovar 0.4.5 used. WARNING: it is roughly
2x slower for typical pipelines and uses noticeably more GPU memory.
For a one-off run on a small dataset it's fine; for production, rebuild.

Why this happened
-----------------
recovar's CUDA kernel ships precompiled targets for sm_70..sm_90 plus a
compute_75 PTX fallback (covers V100, T4, RTX 20/30/40-series, A100,
A40, H100). Your GPU is outside that range. Override CUDA_ARCH at make
time to add it. See {makefile_path} for the full default and
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-compilation
for what the gencode flags mean."""

    raise RuntimeError(msg)


def _ensure_ffi():
    global _ffi_registered
    if _ffi_registered:
        return
    with _ffi_lock:
        if _ffi_registered:
            return
        lib = _get_lib()
        # Preflight: check that the .so covers this GPU before FFI registration
        if _loaded_lib_path:
            _preflight_check(pathlib.Path(_loaded_lib_path))
        jax.ffi.register_ffi_target(_TARGET_BACKPROJECT, jax.ffi.pycapsule(lib.Backproject), platform="CUDA")
        jax.ffi.register_ffi_target(_TARGET_PROJECT, jax.ffi.pycapsule(lib.Project), platform="CUDA")
        jax.ffi.register_ffi_target(_TARGET_BATCH_BACKPROJECT, jax.ffi.pycapsule(lib.BatchBackproject), platform="CUDA")
        jax.ffi.register_ffi_target(
            _TARGET_PER_IMAGE_BACKPROJECT,
            jax.ffi.pycapsule(lib.PerImageBackproject),
            platform="CUDA",
        )
        jax.ffi.register_ffi_target(_TARGET_BATCH_PROJECT, jax.ffi.pycapsule(lib.BatchProject), platform="CUDA")
        _ffi_registered = True
        logger.debug("Registered CUDA FFI targets")


_cuda_ok = None  # cached result: None = not checked, True/False = result


def cuda_available() -> bool:
    """Return True if CUDA backproject/project kernels can be used (cached).

    RECOVAR prefers these kernels by default on GPU and will try to build the
    shared library automatically into the cache directory when needed. Set
    ``RECOVAR_DISABLE_CUDA=1`` to force the slower JAX GPU path instead.
    """
    global _auto_build_error, _cuda_ok
    if _cuda_ok is not None:
        return _cuda_ok

    if not custom_cuda_requested():
        _cuda_ok = False
        logger.info("CUDA kernels disabled via %s", _DISABLE_CUSTOM_CUDA_ENV)
        return _cuda_ok
    try:
        if not any(d.platform == "gpu" for d in jax.devices()):
            _cuda_ok = False
        else:
            _ensure_ffi()
            _cuda_ok = True
            _auto_build_error = None
            logger.info("CUDA backproject/project kernels enabled")
    except (ImportError, OSError, RuntimeError, AttributeError, subprocess.SubprocessError) as e:
        _cuda_ok = False
        if _auto_build_error is None:
            _auto_build_error = e
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


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def per_image_backproject(
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
    """Back-project each image into its own output volume.

    ``volumes`` has shape ``(n_images, vol_flat_size)``. This is useful when
    many weighted estimators share the same images and rotations: backproject
    once per image, then reduce with a dense weight matrix.
    """
    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image)
    kw, ih, iw_eff = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    out_dtype = jnp.result_type(images, volumes)
    images = images.astype(out_dtype)
    volumes = volumes.astype(out_dtype)
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volumes))
    out_type = jax.ShapeDtypeStruct(volumes.shape, volumes.dtype)

    return jax.ffi.ffi_call(
        _TARGET_PER_IMAGE_BACKPROJECT,
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
