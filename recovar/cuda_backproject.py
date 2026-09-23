"""
CUDA backprojector / projector — JAX JIT-compatible via XLA FFI.

Provides ``backproject`` and ``project`` that drop into ``@jax.jit``
compiled functions.

Quick start::

    from recovar.cuda_backproject import backproject, project

    @jax.jit
    def step(vol, images, rots):
        vol = backproject(vol, images, rots, image_shape, volume_shape, order=1)
        imgs = project(vol, rots, image_shape, volume_shape, order=1)
        return vol, imgs
"""

from __future__ import annotations

import contextvars
import ctypes
import functools
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
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
_CUDA_BUILD_SOURCE_NAMES = (
    "device_scratch.cuh",
    "../em/cuda/noise_residual.cuh",
    "../em/cuda/vdam_trace.cuh",
    "../em/cuda/relion_preprocess.cuh",
    "../em/cuda/relion_vdam_mstep.cuh",
    "../em/cuda/relion_scoring.cuh",
    "../em/cuda/relion_posterior.cuh",
    "../em/cuda/sparse_pass2_posterior.cuh",
    "../em/cuda/relion_translate_sum.cuh",
    "cuda_backproject.cu",
    "../em/cuda/relion_coarse_diff2_projector_body.inc",
    "Makefile",
)
_lib_handle = None  # ctypes CDLL
_loaded_lib_path = None

_DISABLE_CUSTOM_CUDA_ENV = "RECOVAR_DISABLE_CUDA"
_CUDA_LIB_ENV = "RECOVAR_CUDA_LIB"
_CUDA_CACHE_DIR_ENV = "RECOVAR_CUDA_CACHE_DIR"
_BUILD_LOCKFILE = ".build.lock"
_RELION_X_HALF_BP_BLOCK_TOPOLOGY_ENV = "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY"
_BPREF_DEVICE_SIGNATURE_DUMP_DIR_ENV = "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR"
_bpref_device_signature_scope = contextvars.ContextVar(
    "recovar_bpref_device_signature_scope",
    default=None,
)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def relion_x_half_bp_block_topology_enabled() -> bool:
    """Return whether the diagnostic RELION GPU pixel-pass topology is enabled.

    The value is consumed while JAX traces the caller. Change it only between
    fresh processes; changing the environment after compilation does not
    invalidate an already cached executable.
    """

    if os.environ.get(_BPREF_DEVICE_SIGNATURE_DUMP_DIR_ENV, "").strip():
        if _bpref_device_signature_scope.get() is not True:
            return False
    return relion_x_half_bp_block_topology_requested()


def relion_x_half_bp_block_topology_requested() -> bool:
    """Return the raw diagnostic request without changing live-path scope."""

    return _env_flag(_RELION_X_HALF_BP_BLOCK_TOPOLOGY_ENV)


def backproject_skip_zero_requested() -> bool:
    """Return whether indexed backprojection skips exactly-zero pixels.

    Opt-in (``RECOVAR_BACKPROJECT_SKIP_ZERO=1``). Sparse pass-2 M-step rows are
    padded to bucket size and pruned rows are entirely zero; scattering them
    only adds ``+0.0`` to every touched voxel. Skipping removes those atomics.
    """

    return _env_flag(_BACKPROJECT_SKIP_ZERO_ENV)


def custom_cuda_requested() -> bool:
    """Return True unless the user explicitly disables custom CUDA.

    Routes through ``recovar.utils.cuda_env`` so the codebase reads the
    canonical ``RECOVAR_DISABLE_CUDA`` env var in one place.
    """
    from recovar.utils.cuda_env import custom_cuda_disabled_from_env

    disabled, _ = custom_cuda_disabled_from_env()
    return not disabled


_CACHE_ROOT_FALLBACK_WARNED = False


def _path_is_writable(path: pathlib.Path) -> bool:
    """Return True iff *path* exists or can be created and a tiny file
    written and unlinked there.

    Used to gate fallbacks in :func:`_cache_root` when the user's
    ``XDG_CACHE_HOME`` (or ``$HOME``) points at a directory that exists
    but cannot be written to — e.g. issue #136 where ``XDG_CACHE_HOME``
    resolves to ``/home/levans`` but the real home is ``/mnt/home/levans``.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError):
        return False
    probe = path / ".recovar_write_test"
    try:
        probe.write_text("ok")
    except (PermissionError, OSError):
        return False
    try:
        probe.unlink()
    except (PermissionError, OSError):
        return False
    return True


def _warn_cache_root_fallback(rejected: list[tuple[str, pathlib.Path]], chosen: pathlib.Path) -> None:
    parts = ", ".join(f"{label}={path}" for label, path in rejected)
    logger.warning(
        "RECOVAR cuda cache: %s is not writable, falling back to %s. Set %s to silence this warning.",
        parts,
        chosen,
        _CUDA_CACHE_DIR_ENV,
    )


def _cache_root() -> pathlib.Path:
    """Pick a writable cache directory for the CUDA shared library.

    Resolution order, each step gated on writability (except the
    explicit override):

    1. ``RECOVAR_CUDA_CACHE_DIR`` — explicit override, returned as-is.
    2. ``$XDG_CACHE_HOME/recovar/cuda``
    3. ``~/.cache/recovar/cuda``
    4. ``tempfile.gettempdir()/recovar_cuda_cache/cuda`` — last-resort
       fallback when neither of the above is writable. Logs a WARNING.

    Steps 2 and 3 each attempt ``mkdir(parents=True, exist_ok=True)``
    plus a tiny probe-file write and delete. ``PermissionError`` and
    ``OSError`` are caught and trigger a fall-through to the next
    candidate.
    """
    global _CACHE_ROOT_FALLBACK_WARNED

    override = os.environ.get(_CUDA_CACHE_DIR_ENV)
    if override:
        return pathlib.Path(override).expanduser()

    candidates: list[tuple[str, pathlib.Path]] = []
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(("XDG_CACHE_HOME", pathlib.Path(xdg_cache_home).expanduser() / "recovar" / "cuda"))
    candidates.append(("HOME", pathlib.Path.home().expanduser() / ".cache" / "recovar" / "cuda"))

    rejected: list[tuple[str, pathlib.Path]] = []
    for label, path in candidates:
        if _path_is_writable(path):
            if rejected and not _CACHE_ROOT_FALLBACK_WARNED:
                _warn_cache_root_fallback(rejected, path)
                _CACHE_ROOT_FALLBACK_WARNED = True
            return path
        rejected.append((label, path))

    fallback = pathlib.Path(tempfile.gettempdir()) / "recovar_cuda_cache" / "cuda"
    if not _CACHE_ROOT_FALLBACK_WARNED:
        _warn_cache_root_fallback(rejected, fallback)
        _CACHE_ROOT_FALLBACK_WARNED = True
    return fallback


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


def _lib_missing_required_symbols(lib_path: pathlib.Path) -> str | None:
    """Return the name of the first missing required symbol, or None if all present.

    Catches binary/source skew that mtime alone misses: e.g. ``~/.cache`` holds an
    ``.so`` built last week from an older ``.cu`` that didn't export
    ``BackprojectIndexed``, and today's ``cuda_backproject.py`` registers FFI
    targets that need it. The mtime check sees ``cached .so newer than source``
    and reuses it; ``_ensure_ffi`` then crashes deep inside ``ctypes.__getattr__``,
    ``cuda_available()`` silently returns False, and the user runs on the slow
    JAX fallback without warning. Dlopen + symbol lookup catches this cheaply.
    """
    try:
        lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_LOCAL)
    except OSError:
        # Can't even dlopen — treat as binary-incompatible.
        return "<dlopen failed>"
    for _target, symbol in _FFI_REGISTRATIONS:
        if not hasattr(lib, symbol):
            return symbol
    return None


def _lib_is_stale(lib_path: pathlib.Path) -> bool:
    """Return True if the lib's mtime is older than the source files OR if
    it's missing a symbol that the current ``cuda_backproject.py`` expects.

    Catches:
      - mtime skew: user installed before a kernel/Makefile fix landed (e.g.
        issue #131's Blackwell widening).
      - binary skew: cached ``.so`` from an older branch doesn't export a
        symbol the current source requires (e.g. ``BackprojectIndexed``).
        Without this check ``_ensure_ffi`` would crash inside ``ctypes`` and
        ``cuda_available()`` would silently fall back to JAX.
    """
    try:
        lib_mtime = lib_path.stat().st_mtime
    except OSError:
        return False
    for src_name in _CUDA_BUILD_SOURCE_NAMES:
        src = _LIB_DIR / src_name
        try:
            if src.stat().st_mtime > lib_mtime:
                return True
        except OSError:
            continue
    missing = _lib_missing_required_symbols(lib_path)
    if missing is not None:
        logger.info(
            "RECOVAR CUDA library %s is missing symbol '%s' required by the current source — will rebuild.",
            lib_path,
            missing,
        )
        return True
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


def _discover_system_nvcc() -> str | None:
    """Find nvcc in common system install locations across Linux distros.

    Last-resort cluster-agnostic fallback for users who didn't ``module load``
    or set ``CUDA_HOME``. Returns the highest-version nvcc found, or None.
    Covers:
      - ``/usr/local/cuda*`` (RHEL/CentOS/Della-style symlinks + versioned dirs)
      - ``/opt/cuda*`` (Arch, some HPC)
      - ``/opt/nvidia/cuda*`` (some HPC)
      - ``/usr/lib/nvidia-cuda-toolkit/bin/nvcc`` (Debian/Ubuntu apt)
    """
    candidates: list[pathlib.Path] = []
    for pattern in (
        "/usr/local/cuda*/bin/nvcc",
        "/opt/cuda*/bin/nvcc",
        "/opt/nvidia/cuda*/bin/nvcc",
    ):
        candidates.extend(pathlib.Path("/").glob(pattern.lstrip("/")))
    candidates.append(pathlib.Path("/usr/lib/nvidia-cuda-toolkit/bin/nvcc"))
    candidates = [p for p in candidates if p.is_file() and os.access(p, os.X_OK)]
    if not candidates:
        return None

    def _version_key(p: pathlib.Path) -> tuple[int, int, str]:
        # Sort by versioned dir name (e.g. cuda-13.2, cuda-12.8), highest first.
        # Falls back to lexicographic for unversioned (cuda symlink).
        parent = p.parent.parent.name  # ".../cuda-13.2/bin/nvcc" → "cuda-13.2"
        suffix = parent.split("-", 1)[-1] if "-" in parent else ""
        parts = suffix.split(".") if suffix else []
        try:
            major = int(parts[0]) if len(parts) >= 1 else -1
            minor = int(parts[1]) if len(parts) >= 2 else -1
        except ValueError:
            major, minor = -1, -1
        return (major, minor, str(p))

    candidates.sort(key=_version_key, reverse=True)
    return str(candidates[0])


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
    make_env = os.environ.copy()
    # If the user hasn't surfaced nvcc through any of the Makefile's normal
    # discovery channels (NVCC/CUDACXX/PATH/LOCAL_CUDA_PATH/CUDA_HOME/CUDA_PATH),
    # do a last-resort sweep of common system install paths. This makes
    # ``import recovar; cuda_available()`` Just Work on most clusters without
    # requiring ``module load cudatoolkit`` first.
    nvcc_already_visible = (
        make_env.get("NVCC")
        or make_env.get("CUDACXX")
        or shutil.which("nvcc")
        or any(
            os.access(os.path.join(make_env.get(var, ""), "bin", "nvcc"), os.X_OK)
            for var in ("LOCAL_CUDA_PATH", "CUDA_HOME", "CUDA_PATH")
            if make_env.get(var)
        )
    )
    if not nvcc_already_visible:
        discovered = _discover_system_nvcc()
        if discovered is not None:
            logger.info("Discovered nvcc at %s (system fallback)", discovered)
            make_env["NVCC"] = discovered

    make_cmd = ["make"]
    if force or stale:
        make_cmd.append("-B")
    make_cmd.extend(["-C", str(_LIB_DIR), f"PYTHON={sys.executable}", f"LIB={lib_path}"])
    subprocess.check_call(make_cmd, env=make_env)
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
    if _ffi_registered and _lib_handle is not None and _loaded_lib_path is not None:
        # XLA FFI registrations last for the process, and every loaded copy of
        # the library keeps its own CUDA state (the persistent-texture registry
        # among it). Stay on the library whose symbols XLA holds: a second copy
        # would hand out handles the registered kernels cannot see.
        configured = _configured_lib_path()
        if configured is not None and configured.resolve() != _loaded_lib_path:
            raise RuntimeError(
                f"RECOVAR_CUDA_LIB={configured} asks for a different CUDA library than the one "
                f"this process's XLA FFI handlers are bound to ({_loaded_lib_path}); "
                "restart the process to switch libraries"
            )
        return _lib_handle
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
_TARGET_BACKPROJECT_INDEXED = "cuda_backproject_indexed"
_TARGET_BACKPROJECT_INDEXED_SKIP_ZERO = "cuda_backproject_indexed_skip_zero"
_BACKPROJECT_SKIP_ZERO_ENV = "RECOVAR_BACKPROJECT_SKIP_ZERO"
_TARGET_BACKPROJECT_INDEXED_SIGNATURE = "cuda_backproject_indexed_signature"
_TARGET_PROJECT = "cuda_project"
_TARGET_PROJECT_RELION_HALF_RUNTIME = "cuda_project_relion_half_runtime"
_TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS = "cuda_project_relion_half_image_radius"
_TARGET_PROJECT_INDEXED = "cuda_project_indexed"
_TARGET_BATCH_BACKPROJECT = "cuda_batch_backproject"
_TARGET_BATCH_BACKPROJECT_INDEXED = "cuda_batch_backproject_indexed"
_TARGET_PER_IMAGE_BP = "cuda_per_image_bp"
_TARGET_RELION_PROJECTOR_HALF_TEXTURE_F32 = "cuda_relion_projector_half_texture_f32"
_TARGET_RELION_PROJECTOR_PERSISTENT_HALF_TEXTURE_F32 = "cuda_relion_projector_persistent_half_texture_f32"
_TARGET_RELION_FIRSTITER_BPREF_FUSED_X_HALF = "cuda_relion_firstiter_bpref_fused_x_half"
_TARGET_RELION_FUSED_X_HALF_BP = "cuda_relion_fused_x_half_bp"
_TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID = (
    "cuda_relion_fused_x_half_bp_particle_grid"
)
_TARGET_RELION_FUSED_X_HALF_BP_SIGNATURE = "cuda_relion_fused_x_half_bp_signature"
_TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_COMPLEX_RANGE = (
    "cuda_relion_point_group_symmetrise_bpref_complex_range"
)
_TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_SPLIT_RANGE = (
    "cuda_relion_point_group_symmetrise_bpref_split_range"
)
_TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF = (
    "cuda_relion_point_group_symmetrise_bpref"
)
_TARGET_RELION_PREPROCESS_REAL_F32 = "cuda_relion_preprocess_real_f32"
_TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_LANE = (
    "cuda_relion_preprocess_real_f32_native_lane"
)
_TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_ATOMIC = (
    "cuda_relion_preprocess_real_f32_native_atomic"
)
_TARGET_RELION_MAKE_SCORING_ROTATIONS_F32 = "cuda_relion_make_scoring_rotations_f32"
_TARGET_RELION_MAKE_SCORING_ROTATIONS_F64 = "cuda_relion_make_scoring_rotations_f64"
_TARGET_RELION_TRANSLATE_SCORE_F32 = "cuda_relion_translate_score_f32"
_TARGET_RELION_TRANSLATE_SCORE_F64 = "cuda_relion_translate_score_f64"
_TARGET_RELION_TRANSLATE_BPREF_F32 = "cuda_relion_translate_bpref_f32"
_TARGET_RELION_TRANSLATE_BPREF_F64 = "cuda_relion_translate_bpref_f64"
_TARGET_BPREF_PARTICLE_PACK = "cuda_bpref_particle_pack"
_TARGET_DEFERRED_VDAM_HOST_PACK = "cuda_deferred_vdam_host_pack"
_TARGET_NOISE_PIXEL_PACK = "cuda_noise_pixel_pack"
_TARGET_NOISE_RESIDUAL_STATISTICS = "recovar_noise_residual_statistics"
_TARGET_SPARSE_PASS2_LOG_Z_F64 = "recovar_sparse_pass2_log_z_f64"
_TARGET_SPARSE_PASS2_POSTERIOR_F32 = "recovar_sparse_pass2_posterior_f32"
_TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64 = (
    "cuda_sparse_pass2_segmented_log_z_f64"
)
_TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32 = (
    "cuda_sparse_pass2_segmented_posterior_f32"
)
_TARGET_RELION_VDAM_MSTEP_SUMS_F32 = "cuda_relion_vdam_mstep_sums_f32"
_TARGET_RELION_VDAM_MSTEP_DENOMINATOR_F32 = (
    "cuda_relion_vdam_mstep_denominator_f32"
)
_TARGET_RELION_VDAM_MSTEP_FUSED_X_HALF = "cuda_relion_vdam_mstep_fused_x_half"
_TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_X_HALF = (
    "cuda_relion_vdam_mstep_fused_projector_x_half"
)
_TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_RUNTIME_X_HALF = (
    "cuda_relion_vdam_mstep_fused_projector_runtime_x_half"
)
_TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F32 = (
    "cuda_relion_coarse_diff2_rectangular_f32"
)
_TARGET_RELION_COARSE_DIFF2_RECTANGULAR_RUNTIME_F32 = (
    "cuda_relion_coarse_diff2_rectangular_runtime_f32"
)
_TARGET_RELION_COARSE_DIFF2_ROTATION_BLOCKS_F32 = (
    "cuda_relion_coarse_diff2_rotation_blocks_f32"
)
_TARGET_RELION_COARSE_DIFF2_ROTATION_BLOCKS_RUNTIME_F32 = (
    "cuda_relion_coarse_diff2_rotation_blocks_runtime_f32"
)
_TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32 = (
    "cuda_relion_coarse_diff2_projector_f32"
)
_TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32 = (
    "cuda_relion_coarse_diff2_projector_multistream_f32"
)
_TARGET_RELION_COARSE_DIFF2_PROJECTOR_LANES_F32 = (
    "cuda_relion_coarse_diff2_projector_lanes_f32"
)
_TARGET_RELION_COARSE_DIFF2_NATIVE_TEXTURE_RECTANGULAR_F32 = (
    "cuda_relion_coarse_diff2_native_texture_rectangular_f32"
)
_TARGET_RELION_COARSE_NORMALIZED_CC_PAIRS_F32 = (
    "cuda_relion_coarse_normalized_cc_pairs_f32"
)
_TARGET_RELION_COARSE_NORMALIZED_CC_NATIVE_TEXTURE_PAIRS_F32 = (
    "cuda_relion_coarse_normalized_cc_native_texture_pairs_f32"
)
_TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F64 = (
    "cuda_relion_coarse_diff2_rectangular_f64"
)
_TARGET_RELION_FINE_DIFF2_RECTANGULAR_F32 = (
    "cuda_relion_fine_diff2_rectangular_f32"
)
_TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32 = (
    "cuda_relion_fine_diff2_rectangular_masked_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RECTANGULAR_F32 = (
    "cuda_relion_fine_diff2_fused_translate_rectangular_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_FLAT_ROWS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_flat_rows_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_FLAT_ROWS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_flat_rows_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_RECTANGULAR_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_rectangular_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_PAIRS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_pairs_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_PAIRS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_pairs_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_JOBS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_jobs_f32"
)
_TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_JOBS_F32 = (
    "cuda_relion_fine_diff2_fused_translate_runtime_jobs_f32"
)
_TARGET_RELION_FINE_DIFF2_PAIRS_F32 = "cuda_relion_fine_diff2_pairs_f32"
_TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_F32 = (
    "cuda_relion_powerclass_spectrum_highres_f32"
)
_TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_RUNTIME_F32 = (
    "cuda_relion_powerclass_spectrum_highres_runtime_f32"
)
_TARGET_RELION_EXPONENTIATE_F32 = "cuda_relion_exponentiate_f32"
_TARGET_RELION_EXPONENTIATE_BATCHED_F32 = "cuda_relion_exponentiate_batched_f32"
_TARGET_RELION_DIVIDE_F32 = "cuda_relion_divide_f32"
_TARGET_RELION_FINE_DIFF2_RECTANGULAR_F64 = (
    "cuda_relion_fine_diff2_rectangular_f64"
)
_TARGET_RELION_FINE_DIFF2_PAIRS_F64 = "cuda_relion_fine_diff2_pairs_f64"
_TARGET_RELION_DIVIDE_BATCHED_F32 = "cuda_relion_divide_batched_f32"
_TARGET_RELION_CUB_SORT_SCAN_F32 = "cuda_relion_cub_sort_scan_f32"
_TARGET_RELION_CUB_SORT_SCAN_BATCHED_F32 = "cuda_relion_cub_sort_scan_batched_f32"
_TARGET_RELION_CUB_POSITIVE_SORT_SCAN_F32 = (
    "cuda_relion_cub_positive_sort_scan_f32"
)
_TARGET_RELION_WAVG_ROTATION_ATOMIC_TRIPLET_ADD_F32 = (
    "cuda_relion_wavg_rotation_atomic_triplet_add_f32"
)
_TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32 = (
    "cuda_relion_wavg_rotation_atomic_runtime_triplet_add_f32"
)
_TARGET_RELION_WAVG_SEQUENTIAL_TRIPLET_F32 = (
    "cuda_relion_wavg_sequential_triplet_f32"
)
_TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_TRIPLET_F32 = (
    "cuda_relion_wavg_sequential_runtime_triplet_f32"
)
_TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32 = (
    "cuda_relion_wavg_sequential_runtime_flat_rows_triplet_f32"
)
_TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32 = (
    "cuda_relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32"
)
_TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32 = (
    "cuda_relion_translate_sum_flat_rows_f32"
)

_TARGET_DUAL_WEIGHTED_SUMS_F32 = "cuda_dual_weighted_sums_f32"
_TARGET_DUAL_WEIGHTED_SUMS_PAIRS_F32 = "cuda_dual_weighted_sums_pairs_f32"
_TARGET_DUAL_WEIGHTED_SUMS_PAIRS_ROWS_F32 = "cuda_dual_weighted_sums_pairs_rows_f32"

# Single source of truth: (FFI target name, C symbol exported by libcuda_backproject.so).
# Used by ``_ensure_ffi`` to register kernels AND by ``_lib_missing_required_symbols``
# to verify a cached .so is binary-compatible with the current source. When the kernel
# adds/removes a symbol, update this tuple — that automatically invalidates stale caches.
_FFI_REGISTRATIONS: tuple[tuple[str, str], ...] = (
    (_TARGET_BACKPROJECT, "Backproject"),
    (_TARGET_BACKPROJECT_INDEXED, "BackprojectIndexed"),
    (_TARGET_BACKPROJECT_INDEXED_SIGNATURE, "BackprojectIndexedSignature"),
    (_TARGET_PROJECT, "Project"),
    (_TARGET_PROJECT_INDEXED, "ProjectIndexed"),
    (_TARGET_BATCH_BACKPROJECT, "BatchBackproject"),
    (_TARGET_BATCH_BACKPROJECT_INDEXED, "BatchBackprojectIndexed"),
    (_TARGET_PER_IMAGE_BP, "PerImageBackproject"),
    (_TARGET_RELION_FUSED_X_HALF_BP, "RelionFusedXHalfBackproject"),
    (_TARGET_RELION_FIRSTITER_BPREF_FUSED_X_HALF, "RelionFirstiterBprefFusedXHalf"),
    (_TARGET_RELION_PROJECTOR_HALF_TEXTURE_F32, "RelionProjectorHalfTextureF32"),
    (_TARGET_RELION_PROJECTOR_PERSISTENT_HALF_TEXTURE_F32, "RelionProjectorPersistentHalfTextureF32"),
    (
        _TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID,
        "RelionFusedXHalfBackprojectParticleGrid",
    ),
    (
        _TARGET_RELION_FUSED_X_HALF_BP_SIGNATURE,
        "RelionFusedXHalfBackprojectSignature",
    ),
    (
        _TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF,
        "RelionPointGroupSymmetriseBpref",
    ),
    (
        _TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_SPLIT_RANGE,
        "RelionPointGroupSymmetriseBprefSplitRange",
    ),
    (
        _TARGET_RELION_POINT_GROUP_SYMMETRISE_BPREF_COMPLEX_RANGE,
        "RelionPointGroupSymmetriseBprefComplexRange",
    ),
    (_TARGET_RELION_PREPROCESS_REAL_F32, "RelionPreprocessRealF32"),
    (
        _TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_LANE,
        "RelionPreprocessRealF32NativeLane",
    ),
    (
        _TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_ATOMIC,
        "RelionPreprocessRealF32NativeAtomic",
    ),
    (
        _TARGET_RELION_MAKE_SCORING_ROTATIONS_F32,
        "RelionMakeScoringRotationsF32",
    ),
    (
        _TARGET_RELION_MAKE_SCORING_ROTATIONS_F64,
        "RelionMakeScoringRotationsF64",
    ),
    (_TARGET_RELION_TRANSLATE_SCORE_F32, "RelionTranslateScoreF32"),
    (_TARGET_RELION_TRANSLATE_SCORE_F64, "RelionTranslateScoreF64"),
    (_TARGET_RELION_TRANSLATE_BPREF_F32, "RelionTranslateBprefF32"),
    (_TARGET_RELION_TRANSLATE_BPREF_F64, "RelionTranslateBprefF64"),
    (_TARGET_RELION_VDAM_MSTEP_SUMS_F32, "RelionVdamMstepSumsF32"),
    (
        _TARGET_RELION_VDAM_MSTEP_DENOMINATOR_F32,
        "RelionVdamMstepDenominatorF32",
    ),
    (_TARGET_RELION_VDAM_MSTEP_FUSED_X_HALF, "RelionVdamMstepFusedXHalf"),
    (
        _TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_X_HALF,
        "RelionVdamMstepFusedProjectorXHalf",
    ),
    (
        _TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_RUNTIME_X_HALF,
        "RelionVdamMstepFusedProjectorRuntimeXHalf",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F64,
        "RelionCoarseDiff2RectangularF64",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F32,
        "RelionCoarseDiff2RectangularF32",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_RECTANGULAR_RUNTIME_F32,
        "RelionCoarseDiff2RectangularRuntimeF32",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_ROTATION_BLOCKS_F32,
        "RelionCoarseDiff2RotationBlocksF32",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_ROTATION_BLOCKS_RUNTIME_F32,
        "RelionCoarseDiff2RotationBlocksRuntimeF32",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32,
        "RelionCoarseDiff2ProjectorF32",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32,
        "RelionCoarseDiff2ProjectorMultistreamF32",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_PROJECTOR_LANES_F32,
        "RelionCoarseDiff2ProjectorLanesF32",
    ),
    (
        _TARGET_RELION_COARSE_DIFF2_NATIVE_TEXTURE_RECTANGULAR_F32,
        "RelionCoarseDiff2NativeTextureRectangularF32",
    ),
    (
        _TARGET_RELION_COARSE_NORMALIZED_CC_PAIRS_F32,
        "RelionCoarseNormalizedCcPairsF32",
    ),
    (
        _TARGET_RELION_COARSE_NORMALIZED_CC_NATIVE_TEXTURE_PAIRS_F32,
        "RelionCoarseNormalizedCcNativeTexturePairsF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_RECTANGULAR_F32,
        "RelionFineDiff2RectangularF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RECTANGULAR_F32,
        "RelionFineDiff2FusedTranslateRectangularF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_FLAT_ROWS_F32,
        "RelionFineDiff2FusedTranslateFlatRowsF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_FLAT_ROWS_F32,
        "RelionFineDiff2FusedTranslateRuntimeFlatRowsF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_RECTANGULAR_F32,
        "RelionFineDiff2FusedTranslateRuntimeRectangularF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_PAIRS_F32,
        "RelionFineDiff2FusedTranslatePairsF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_PAIRS_F32,
        "RelionFineDiff2FusedTranslateRuntimePairsF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_JOBS_F32,
        "RelionFineDiff2FusedTranslateJobsF32",
    ),
    (
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_JOBS_F32,
        "RelionFineDiff2FusedTranslateRuntimeJobsF32",
    ),
    (_TARGET_RELION_FINE_DIFF2_PAIRS_F32, "RelionFineDiff2PairsF32"),
    (
        _TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_F32,
        "RelionPowerClassSpectrumHighresF32",
    ),
    (
        _TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_RUNTIME_F32,
        "RelionPowerClassSpectrumHighresRuntimeF32",
    ),
    (_TARGET_RELION_EXPONENTIATE_F32, "RelionExponentiateF32"),
    (_TARGET_RELION_EXPONENTIATE_BATCHED_F32, "RelionExponentiateBatchedF32"),
    (_TARGET_RELION_DIVIDE_F32, "RelionDivideF32"),
    (
        _TARGET_RELION_FINE_DIFF2_RECTANGULAR_F64,
        "RelionFineDiff2RectangularF64",
    ),
    (_TARGET_RELION_FINE_DIFF2_PAIRS_F64, "RelionFineDiff2PairsF64"),
    (_TARGET_RELION_DIVIDE_BATCHED_F32, "RelionDivideBatchedF32"),
    (_TARGET_RELION_CUB_SORT_SCAN_F32, "RelionCubSortScanF32"),
    (_TARGET_RELION_CUB_SORT_SCAN_BATCHED_F32, "RelionCubSortScanBatchedF32"),
    (
        _TARGET_RELION_CUB_POSITIVE_SORT_SCAN_F32,
        "RelionCubPositiveSortScanF32",
    ),
    (
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_TRIPLET_ADD_F32,
        "RelionWavgRotationAtomicTripletAddF32",
    ),
    (
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32,
        "RelionWavgRotationAtomicRuntimeTripletAddF32",
    ),
    (
        _TARGET_RELION_WAVG_SEQUENTIAL_TRIPLET_F32,
        "RelionWavgSequentialTripletF32",
    ),
    (
        _TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_TRIPLET_F32,
        "RelionWavgSequentialRuntimeTripletF32",
    ),
    (_TARGET_DUAL_WEIGHTED_SUMS_F32, "DualWeightedSumsF32"),
    (_TARGET_DUAL_WEIGHTED_SUMS_PAIRS_F32, "DualWeightedSumsPairsF32"),
    (_TARGET_DUAL_WEIGHTED_SUMS_PAIRS_ROWS_F32, "DualWeightedSumsPairsRowsF32"),
)


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
        for target, symbol in _FFI_REGISTRATIONS:
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, symbol)), platform="CUDA")
        _ffi_registered = True
        logger.debug("Registered CUDA FFI targets")


_TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF = (
    "cuda_relion_vdam_mstep_fused_projector_capacity_x_half"
)


_optional_ffi_registered: set[str] = set()
_OPTIONAL_FFI_REGISTRATIONS = {
    _TARGET_NOISE_RESIDUAL_STATISTICS: (
        "NoiseResidualStatistics",
        "Explicit CUDA build with NoiseResidualStatistics required",
    ),
    _TARGET_PROJECT_RELION_HALF_RUNTIME: (
        "ProjectRelionHalfRuntime",
        "Projector capacity was requested but the loaded CUDA library lacks ProjectRelionHalfRuntime; explicitly rebuild the custom CUDA library",
    ),
    _TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS: (
        "ProjectRelionHalfImageRadius",
        "Image-radius projection requires ProjectRelionHalfImageRadius; explicitly rebuild the custom CUDA library",
    ),
    _TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF: (
        "RelionVdamMstepFusedProjectorCapacityXHalf",
        "BPref projector capacity requires RelionVdamMstepFusedProjectorCapacityXHalf; explicitly rebuild the custom CUDA library",
    ),
    _TARGET_NOISE_PIXEL_PACK: (
        "NoisePixelPack",
        "CUDA noise pixel packing requires an explicit build with NoisePixelPack",
    ),
    _TARGET_DEFERRED_VDAM_HOST_PACK: (
        "DeferredVdamHostPack",
        "CUDA host-plan packing requires an explicit build with DeferredVdamHostPack",
    ),
    _TARGET_BPREF_PARTICLE_PACK: (
        "BprefParticlePack",
        "CUDA BPref packing requires an explicit build with BprefParticlePack",
    ),
    _TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32: (
        "RelionFineDiff2RectangularMaskedF32",
        "Masked rectangular fine diff2 requires an explicit CUDA build with RelionFineDiff2RectangularMaskedF32",
    ),
    _TARGET_SPARSE_PASS2_LOG_Z_F64: (
        "SparsePass2LogZF64",
        "The fused sparse pass-2 log-Z requires an explicit CUDA build with SparsePass2LogZF64",
    ),
    _TARGET_SPARSE_PASS2_POSTERIOR_F32: (
        "SparsePass2PosteriorF32",
        "The fused sparse pass-2 posterior requires an explicit CUDA build with SparsePass2PosteriorF32",
    ),
    _TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64: (
        "SparsePass2SegmentedLogZF64",
        "The segmented sparse pass-2 log-Z requires an explicit CUDA build with SparsePass2SegmentedLogZF64",
    ),
    _TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32: (
        "SparsePass2SegmentedPosteriorF32",
        "The segmented sparse pass-2 posterior requires an explicit CUDA build with SparsePass2SegmentedPosteriorF32",
    ),
    _TARGET_BACKPROJECT_INDEXED_SKIP_ZERO: (
        "BackprojectIndexedSkipZero",
        "RECOVAR_BACKPROJECT_SKIP_ZERO requires an explicit CUDA build with BackprojectIndexedSkipZero",
    ),
    _TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32: (
        "RelionWavgSequentialRuntimeFlatRowsTripletF32",
        "Flat-row RELION Wavg requires an explicit CUDA build with RelionWavgSequentialRuntimeFlatRowsTripletF32",
    ),
    _TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32: (
        "RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32",
        "Flat-row RELION Wavg atomics require an explicit CUDA build with RelionWavgRotationAtomicRuntimeFlatRowsTripletAddF32",
    ),
    _TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32: (
        "RelionTranslateSumFlatRowsF32",
        "Flat-row translate-and-sum requires an explicit CUDA build with RelionTranslateSumFlatRowsF32",
    ),
}


def _ensure_optional_ffi(target):
    """Register a requested optional ABI without invalidating older libraries."""
    _ensure_ffi()
    if target in _optional_ffi_registered:
        return
    with _ffi_lock:
        if target in _optional_ffi_registered:
            return
        symbol_name, error = _OPTIONAL_FFI_REGISTRATIONS[target]
        symbol = getattr(_get_lib(), symbol_name, None)
        if symbol is None:
            raise RuntimeError(error)
        jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(symbol), platform="CUDA")
        _optional_ffi_registered.add(target)


_cuda_ok = None  # cached result: None = not checked, True/False = result
_texture_debug_keys = set()


def cuda_available() -> bool:
    """Return True if CUDA backproject/project kernels can be used (cached).

    RECOVAR prefers these kernels by default on GPU and will try to build the
    shared library automatically into the cache directory when needed. Set
    ``RECOVAR_DISABLE_CUDA=1`` to force the slower JAX GPU path instead.
    """
    global _auto_build_error, _cuda_ok
    if not custom_cuda_requested():
        logger.info("CUDA kernels disabled via %s", _DISABLE_CUSTOM_CUDA_ENV)
        return False
    if _cuda_ok is not None:
        return _cuda_ok
    try:
        if not any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in jax.devices()):
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


def _relion_x_half_backproject_rotation_to_kernel(
    rotation_matrices: jax.Array,
    target_dtype=None,
) -> jax.Array:
    """Map RELION/RECOVAR scorer rotations to the CUDA ``(z, y, xhalf)`` scatter frame.

    RELION's BackProjector stores the Fourier x-axis as the packed half axis but
    RECOVAR's generic CUDA half-volume kernel packs its last coordinate. RELION
    computes a numerical ``A.inv()`` rather than using a transpose; boundary
    pixels at Nyquist can differ by one ulp, so mirror that inverse before
    reversing to the kernel's ``(z, y, x)`` scatter coordinates.
    """

    if target_dtype is not None and jnp.dtype(target_dtype) == jnp.dtype(jnp.float32):
        # RELION builds an orthonormal Euler matrix in CPU RFLOAT, inverts it
        # there, and only then stores the result as accelerated XFLOAT.  For a
        # single-precision ACC build this is the float-cast transpose, not the
        # inverse of an already rounded float32 matrix.
        return rotation_matrices.astype(jnp.float32)[..., [2, 1, 0]]
    inverse = jnp.linalg.inv(rotation_matrices.astype(jnp.float64))
    return jnp.swapaxes(inverse, -1, -2)[..., [2, 1, 0]]


def _prepare_relion_x_half_block_topology_operands(images, pixel_indices, image_shape, max_r):
    """Expand compact x-half rows into RELION's native current-size square.

    RELION launches one block per orientation over a cropped FFTW array with
    ``2*max_r`` rows and ``max_r+1`` packed-x columns. RECOVAR normally keeps
    only the nonzero circular support. The diagnostic restores the omitted
    square positions as explicit zeros so the CUDA kernel can enumerate native
    FFTW pixel positions in the same 128-thread serial-pass topology. This is
    deliberately a pixel-pass diagnostic, not full launch-order equivalence:
    RELION also uses per-particle launches and couples translation reduction
    with data/weight scattering, while RECOVAR performs those stages separately.
    Values
    absent from RECOVAR's compact support cannot be recovered: in particular,
    redundant negative-y ``x=0`` lanes remain zero. RELION's normal 2-D input
    also gives those lanes zero inverse-noise weight, but unusual callers with
    nonzero values there are not exactly emulated by this diagnostic.
    """

    full_height, full_width = (int(image_shape[0]), int(image_shape[1]))
    full_half_width = full_width // 2 + 1
    if max_r is None:
        current_height = full_height
    else:
        current_height = 2 * int(round(float(max_r)))
    current_half_width = current_height // 2 + 1
    if current_height <= 0 or current_height > full_height:
        raise ValueError(f"invalid RELION block-topology current height {current_height} for image shape {image_shape}")

    pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32).reshape(-1)
    full_rows = pixel_indices // full_half_width
    columns = pixel_indices % full_half_width
    signed_rows = jnp.where(full_rows <= full_height // 2, full_rows, full_rows - full_height)
    current_rows = jnp.mod(signed_rows, current_height)
    current_indices = current_rows * current_half_width + columns
    dense_pixels = current_height * current_half_width
    dense_images = jnp.zeros((*images.shape[:-1], dense_pixels), dtype=images.dtype)
    dense_images = dense_images.at[..., current_indices].set(images)
    dense_indices = jnp.arange(dense_pixels, dtype=jnp.int32)
    return dense_images, dense_indices, current_height, current_half_width


def _volume_real_dtype(volume: jax.Array):
    """Return the real component dtype of a volume (float32 for complex64, etc.)."""
    return jnp.finfo(volume.dtype).dtype if jnp.issubdtype(volume.dtype, jnp.complexfloating) else volume.dtype


def _infer_backproject_upsampling(image_shape, volume_shape, max_r=None):
    """Infer Fourier oversampling for standard and RELION BackProjector grids."""

    ih, _ = image_shape
    N0, N1, N2 = volume_shape
    if N0 % ih == 0:
        ups = N0 // ih
        if ups <= 0:
            raise ValueError(f"volume_shape[0] ({N0}) must be at least image_shape[0] ({ih})")
        return int(ups)
    if N0 != N1 or N0 != N2:
        raise ValueError(
            "non-cubic volume_shape requires standard integer upsampling along axis 0, "
            f"got image_shape={image_shape}, volume_shape={volume_shape}"
        )
    if max_r is not None and float(max_r) > 0:
        ups = int((float(N0) - 3.0) / (2.0 * float(max_r)) + 0.5)
        expected = 2 * (int(float(ups) * float(max_r) + 0.5) + 1) + 1
        if ups > 0 and expected == int(N0):
            return ups
    full_support = float(ih) / 2.0
    ups = int((float(N0) - 3.0) / (2.0 * full_support) + 0.5)
    expected = 2 * (int(float(ups) * full_support + 0.5) + 1) + 1
    if ups > 0 and expected == int(N0):
        return ups
    raise ValueError(
        f"volume_shape[0] ({N0}) must be divisible by image_shape[0] ({ih}) "
        "or match RELION pad_size=2*(int(padding_factor*support_radius+0.5)+1)+1"
    )


def _validate_inputs(volume_shape, image_shape, order, half_volume, half_image, max_r=None):
    """Validate parameters at trace time (not inside JIT)."""
    ih, iw = image_shape
    N0, N1, N2 = volume_shape
    if ih <= 0 or iw <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    if N0 <= 0 or N1 <= 0 or N2 <= 0:
        raise ValueError(f"volume_shape must be positive, got {volume_shape}")
    if order not in (0, 1, 3):
        raise ValueError(f"order must be 0, 1, or 3, got {order}")
    _infer_backproject_upsampling(image_shape, volume_shape, max_r=max_r)


def _encode_max_r(max_r):
    """Encode max_r as int64 max_r2_x4 for FFI (quarter-pixel² precision).

    -1 means disabled (no sphere clipping).
    """
    if max_r is None:
        return np.int64(-1)
    return np.int64(int(round(float(max_r) * float(max_r) * 4)))


def _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r=None):
    """Compute shared FFI scalar keyword arguments.

    ``max_r`` is in image Fourier-pixel coordinates, matching
    ``recovar.core.slicing`` and ``relion_interp``. The CUDA kernels compare
    against coordinates already multiplied by ``upsampling``, so encode the
    radius in padded-volume coordinates here and nowhere else.
    """
    ih, iw_full = image_shape
    N0, N1, N2 = volume_shape
    ups = _infer_backproject_upsampling(image_shape, volume_shape, max_r=max_r)
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
            max_r2_x4=_encode_max_r(None if max_r is None else float(max_r) * float(ups)),
        ),
        ih,
        iw_eff,
    )


def _project_ffi_kwargs(
    image_shape,
    volume_shape,
    order,
    half_volume,
    half_image,
    max_r=None,
    relion_texture_interp: bool = False,
):
    """FFI kwargs for forward projection.

    ``relion_texture_interp=True`` enables CUDA texture interpolation for
    full-volume order-1 projections, including RELION's positive even-box
    Nyquist convention. Generic projections retain RECOVAR's centered-grid
    convention.
    """
    kw, ih, iw_eff = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    kw["relion_texture_interp"] = np.int64(int(relion_texture_interp))
    return kw, ih, iw_eff


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
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image, max_r=max_r)
    kw, ih, iw_eff = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volume))
    out_type = jax.ShapeDtypeStruct(volume.shape, volume.dtype)

    return jax.ffi.ffi_call(
        _TARGET_BACKPROJECT,
        out_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(images, rot6, volume, **kw)


def sparse_pass2_segmented_supported() -> bool:
    """Return whether the loaded library exports both segmented pass-2 targets.

    The segmented handlers are an optional ABI, so a library built before them
    stays loadable and callers fall back to the rectangular handlers.
    """

    try:
        _ensure_ffi()
        lib = _get_lib()
        return all(
            getattr(lib, _OPTIONAL_FFI_REGISTRATIONS[target][0], None) is not None
            for target in (
                _TARGET_SPARSE_PASS2_SEGMENTED_LOG_Z_F64,
                _TARGET_SPARSE_PASS2_SEGMENTED_POSTERIOR_F32,
            )
        )
    except Exception:
        return False


def _backproject_indexed_target(use_relion_block_topology: bool) -> str:
    """Pick the indexed-backprojection FFI target for the current gate.

    The zero-skipping variant is opt-in and is unavailable under the RELION
    block topology, which re-expands operands onto the dense FFTW rectangle.

    ``backproject_indexed`` is jitted, so this runs at trace time and the
    chosen target is baked into the cached executable. Changing
    ``RECOVAR_BACKPROJECT_SKIP_ZERO`` part-way through a process therefore has
    no effect on already-traced shapes; set it before the first call.
    """

    if backproject_skip_zero_requested() and not use_relion_block_topology:
        _ensure_optional_ffi(_TARGET_BACKPROJECT_INDEXED_SKIP_ZERO)
        return _TARGET_BACKPROJECT_INDEXED_SKIP_ZERO
    return _TARGET_BACKPROJECT_INDEXED


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def backproject_indexed(
    volume: jax.Array,
    images: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
    relion_x_half: bool = False,
) -> jax.Array:
    """Back-project images whose pixels are stored in a compact indexed layout.

    ``pixel_indices`` contains the flattened pixel positions in the original
    image grid (or packed half-image grid when ``half_image=True``). The kernel
    interprets ``images[:, j]`` as the value at ``pixel_indices[j]``.
    """
    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image, max_r=max_r)
    if relion_x_half and not (half_volume and half_image):
        raise ValueError("relion_x_half requires half_volume=True and half_image=True")
    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    kw["relion_fold_x"] = np.int64(int(relion_x_half))
    use_relion_block_topology = bool(relion_x_half and relion_x_half_bp_block_topology_enabled())
    kw["relion_block_topology"] = np.int64(int(use_relion_block_topology))
    pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32).reshape(-1)
    if use_relion_block_topology:
        logger.info("RELION x-half diagnostic: 128-thread one-block backprojection topology enabled")
        images, pixel_indices, current_height, current_half_width = _prepare_relion_x_half_block_topology_operands(
            images, pixel_indices, image_shape, max_r
        )
        kw["image_h"] = np.int64(current_height)
        kw["image_w"] = np.int64(current_half_width)
        kw["full_image_w"] = np.int64(current_height)
    if relion_x_half:
        rotation_matrices = _relion_x_half_backproject_rotation_to_kernel(
            rotation_matrices,
            _volume_real_dtype(volume),
        )
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volume))
    out_type = jax.ShapeDtypeStruct(volume.shape, volume.dtype)

    target = _backproject_indexed_target(use_relion_block_topology)
    return jax.ffi.ffi_call(
        target,
        out_type,
        input_output_aliases={3: 0},
        vmap_method="sequential",
    )(images, pixel_indices, rot6, volume, **kw)


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def _backproject_indexed_signature_impl(
    volume: jax.Array,
    images: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    canonical_rotation_keys: jax.Array,
    signature_row_indices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float,
) -> tuple[jax.Array, ...]:
    """Run the ordinary indexed production kernel plus its inert signature companion."""

    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(f"ordinary indexed signature requires an odd BPref grid, got {volume_shape}")
    if volume.dtype != jnp.complex64 or images.dtype != jnp.complex64:
        raise TypeError("ordinary indexed signature volume/images must be complex64")
    if pixel_indices.dtype != jnp.int32:
        raise TypeError("ordinary indexed signature pixel indices must be int32")
    if canonical_rotation_keys.dtype != jnp.int32 or signature_row_indices.dtype != jnp.int32:
        raise TypeError("ordinary indexed signature keys/selected rows must be int32")
    if images.ndim != 2 or pixel_indices.shape != (images.shape[1],):
        raise ValueError("ordinary indexed signature requires rank-2 rows and matching pixel indices")
    if rotation_matrices.shape != (images.shape[0], 3, 3):
        raise ValueError("ordinary indexed signature rotations must have shape (n_rows,3,3)")
    if canonical_rotation_keys.shape != (images.shape[0],):
        raise ValueError("ordinary indexed signature rotation key length mismatch")
    expected_volume_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
    if volume.shape != (expected_volume_size,):
        raise ValueError("ordinary indexed signature accumulator shape mismatch")

    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    kw["relion_fold_x"] = np.int64(1)
    kw["relion_block_topology"] = np.int64(0)
    kernel_rotations = _relion_x_half_backproject_rotation_to_kernel(rotation_matrices, jnp.float32)
    rot6 = _rot_to_compact(kernel_rotations, jnp.float32)
    signature_shape = (int(signature_row_indices.shape[0]), int(images.shape[1]))
    out_types = (
        jax.ShapeDtypeStruct(volume.shape, volume.dtype),
        jax.ShapeDtypeStruct(signature_shape, jnp.int32),
        jax.ShapeDtypeStruct(signature_shape, jnp.int32),
        jax.ShapeDtypeStruct(signature_shape, jnp.int32),
        jax.ShapeDtypeStruct((*signature_shape, 5), jnp.float32),
        jax.ShapeDtypeStruct((*signature_shape, 8), jnp.int32),
        jax.ShapeDtypeStruct((*signature_shape, 8), jnp.float32),
        jax.ShapeDtypeStruct((*signature_shape, 8), jnp.int32),
        jax.ShapeDtypeStruct(volume.shape, volume.dtype),
        jax.ShapeDtypeStruct(images.shape, images.dtype),
        jax.ShapeDtypeStruct(pixel_indices.shape, pixel_indices.dtype),
        jax.ShapeDtypeStruct(rot6.shape, rot6.dtype),
        jax.ShapeDtypeStruct(canonical_rotation_keys.shape, canonical_rotation_keys.dtype),
        jax.ShapeDtypeStruct(signature_row_indices.shape, signature_row_indices.dtype),
    )
    return jax.ffi.ffi_call(
        _TARGET_BACKPROJECT_INDEXED_SIGNATURE,
        out_types,
        input_output_aliases={5: 0},
        vmap_method="sequential",
    )(
        images,
        pixel_indices,
        rot6,
        canonical_rotation_keys,
        signature_row_indices,
        volume,
        **kw,
    )


def backproject_indexed_signature(
    volume: jax.Array,
    images: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    canonical_rotation_keys: jax.Array,
    signature_row_indices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float,
) -> tuple[jax.Array, ...]:
    """Capture ordinary indexed CUDA geometry without changing its atomic launch."""

    row_indices = np.asarray(signature_row_indices)
    n_rows = int(images.shape[0])
    if (
        row_indices.ndim != 1
        or row_indices.dtype != np.dtype(np.int32)
        or row_indices.size == 0
        or np.any(row_indices < 0)
        or np.any(row_indices >= n_rows)
        or (row_indices.size > 1 and np.any(np.diff(row_indices) <= 0))
    ):
        raise ValueError("ordinary indexed signature rows must be nonempty, unique, strictly increasing, and in range")
    selected = jnp.asarray(row_indices, dtype=jnp.int32)
    pixels = jnp.asarray(pixel_indices, dtype=jnp.int32).reshape(-1)
    keys = jnp.asarray(canonical_rotation_keys, dtype=jnp.int32).reshape(-1)
    outputs = _backproject_indexed_signature_impl(
        volume,
        images,
        pixels,
        rotation_matrices,
        keys,
        selected,
        image_shape,
        volume_shape,
        max_r,
    )
    kernel_rotations = _relion_x_half_backproject_rotation_to_kernel(rotation_matrices, jnp.float32)
    rot6 = _rot_to_compact(kernel_rotations, jnp.float32)
    expected = (images, pixels, rot6, keys, selected)
    observed = outputs[9:14]
    mismatches = [
        name
        for name, lhs, rhs in zip(
            ("images", "pixel_indices", "rot6", "rotation_keys", "signature_rows"),
            expected,
            observed,
            strict=True,
        )
        if not _bitwise_array_equal(lhs, rhs)
    ]
    if not _bitwise_array_equal(outputs[0], outputs[8]):
        mismatches.insert(0, "accumulator")
    if mismatches:
        raise RuntimeError(
            "ordinary indexed signature deterministic inertness gate failed for " + ", ".join(mismatches)
        )
    return outputs[:8]


def _bitwise_array_equal(left, right) -> bool:
    """Compare diagnostic device snapshots without numeric tolerance."""

    left_np = np.asarray(left)
    right_np = np.asarray(right)
    return (
        left_np.shape == right_np.shape
        and left_np.dtype == right_np.dtype
        and left_np.tobytes(order="C") == right_np.tobytes(order="C")
    )


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
def batch_backproject_indexed(
    volumes: jax.Array,
    images: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
    relion_x_half: bool = False,
) -> jax.Array:
    """Back-project compact indexed images into a batch of volumes."""
    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image, max_r=max_r)
    if relion_x_half and not (half_volume and half_image):
        raise ValueError("relion_x_half requires half_volume=True and half_image=True")
    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    kw["relion_fold_x"] = np.int64(int(relion_x_half))
    use_relion_block_topology = bool(relion_x_half and relion_x_half_bp_block_topology_enabled())
    kw["relion_block_topology"] = np.int64(int(use_relion_block_topology))
    pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32).reshape(-1)
    if use_relion_block_topology:
        logger.info("RELION x-half diagnostic: batched 128-thread one-block backprojection topology enabled")
        images, pixel_indices, current_height, current_half_width = _prepare_relion_x_half_block_topology_operands(
            images, pixel_indices, image_shape, max_r
        )
        kw["image_h"] = np.int64(current_height)
        kw["image_w"] = np.int64(current_half_width)
        kw["full_image_w"] = np.int64(current_height)
    if relion_x_half:
        rotation_matrices = _relion_x_half_backproject_rotation_to_kernel(
            rotation_matrices,
            _volume_real_dtype(volumes),
        )
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volumes))
    out_type = jax.ShapeDtypeStruct(volumes.shape, volumes.dtype)

    return jax.ffi.ffi_call(
        _TARGET_BATCH_BACKPROJECT_INDEXED,
        out_type,
        input_output_aliases={3: 0},
        vmap_method="sequential",
    )(images, pixel_indices, rot6, volumes, **kw)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def _project_impl(
    volume: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
    relion_texture_interp: bool = False,
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
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image, max_r=max_r)
    kw, ih, iw_eff = _project_ffi_kwargs(
        image_shape,
        volume_shape,
        order,
        half_volume,
        half_image,
        max_r,
        relion_texture_interp=relion_texture_interp,
    )
    n_images = rotation_matrices.shape[0]
    n_pixels = ih * iw_eff
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volume))
    out_type = jax.ShapeDtypeStruct((n_images, n_pixels), volume.dtype)

    return jax.ffi.ffi_call(
        _TARGET_PROJECT,
        out_type,
        vmap_method="sequential",
    )(volume, rot6, **kw)


def project(
    volume: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
    relion_texture_interp: bool = False,
) -> jax.Array:
    """Project *volume* to 2D images.

    ``relion_texture_interp=True`` uses RELION-style CUDA texture
    interpolation, including RELION's positive even-box Nyquist convention,
    where the FFI backend supports it. The flag is forwarded as a static
    argument so manual and texture traces cannot alias in JAX caches.
    """
    global _texture_debug_keys
    debug_key = (
        relion_texture_interp,
        getattr(volume, "dtype", None),
        order,
        half_volume,
        half_image,
        image_shape,
        volume_shape,
        max_r,
    )
    if os.environ.get("RECOVAR_DEBUG_TEXTURE_PROJECT", "0") == "1" and debug_key not in _texture_debug_keys:
        print(
            "[RECOVAR_TEXTURE_PROJECT]",
            f"enabled={relion_texture_interp}",
            f"dtype={getattr(volume, 'dtype', None)}",
            f"order={order}",
            f"half_volume={half_volume}",
            f"half_image={half_image}",
            f"image_shape={image_shape}",
            f"volume_shape={volume_shape}",
            f"max_r={max_r}",
            flush=True,
        )
        _texture_debug_keys.add(debug_key)
    return _project_impl(
        volume,
        rotation_matrices,
        image_shape,
        volume_shape,
        order,
        half_volume,
        half_image,
        max_r,
        relion_texture_interp,
    )


def _optional_target_supported(target: str) -> bool:
    """Return whether the loaded library exports an optional target's symbol."""

    try:
        _ensure_ffi()
        symbol_name = _OPTIONAL_FFI_REGISTRATIONS[target][0]
        return getattr(_get_lib(), symbol_name, None) is not None
    except Exception:
        return False


def relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg target."""

    return _optional_target_supported(
        _TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_FLAT_ROWS_TRIPLET_F32
    )


def relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported() -> bool:
    """Return whether the loaded library exports the flat-row Wavg atomics."""

    return _optional_target_supported(
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_FLAT_ROWS_TRIPLET_ADD_F32
    )


def relion_translate_sum_flat_rows_f32_supported() -> bool:
    """Return whether the loaded library exports the translate-and-sum target."""

    return _optional_target_supported(_TARGET_RELION_TRANSLATE_SUM_FLAT_ROWS_F32)


@jax.jit
def dual_weighted_sums_f32(
    probabilities: jax.Array,
    first_values: jax.Array,
    second_values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute two real-probability/complex-value contractions in one FFI call.

    This is a guarded VDAM runtime primitive.  It deliberately keeps the two
    output reductions independent so their float32 accumulation does not
    change when the two pixel axes have different sizes.
    """

    _ensure_ffi()
    probabilities = jnp.asarray(probabilities)
    first_values = jnp.asarray(first_values)
    second_values = jnp.asarray(second_values)
    if probabilities.dtype != jnp.float32 or probabilities.ndim != 3:
        raise ValueError(
            "dual_weighted_sums_f32 expects float32 probabilities with shape "
            "[batch, rotation, translation]"
        )
    if first_values.dtype not in (jnp.complex64, jnp.complex128):
        raise ValueError(
            "dual_weighted_sums_f32 expects complex64 or complex128 value arrays"
        )
    if second_values.dtype != first_values.dtype:
        raise ValueError(
            "dual_weighted_sums_f32 expects both value arrays to have the same dtype"
        )
    for name, values in (("first_values", first_values), ("second_values", second_values)):
        if values.ndim != 3:
            raise ValueError(
                f"dual_weighted_sums_f32 expects {name} to be complex "
                "[batch, translation, pixel]"
            )
        if values.shape[:2] != (probabilities.shape[0], probabilities.shape[2]):
            raise ValueError(
                f"dual_weighted_sums_f32 {name} batch/translation axes do not match "
                f"probabilities: {values.shape} vs {probabilities.shape}"
            )
    output_types = (
        jax.ShapeDtypeStruct(
            (probabilities.shape[0], probabilities.shape[1], first_values.shape[2]),
            first_values.dtype,
        ),
        jax.ShapeDtypeStruct(
            (probabilities.shape[0], probabilities.shape[1], second_values.shape[2]),
            second_values.dtype,
        ),
    )
    return jax.ffi.ffi_call(
        _TARGET_DUAL_WEIGHTED_SUMS_F32,
        output_types,
        vmap_method="sequential",
    )(probabilities, first_values, second_values)


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
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image, max_r=max_r)
    kw, ih, iw_eff = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volumes))
    out_type = jax.ShapeDtypeStruct(volumes.shape, volumes.dtype)

    return jax.ffi.ffi_call(
        _TARGET_BATCH_BACKPROJECT,
        out_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(images, rot6, volumes, **kw)


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def per_image_backproject(
    volumes: jax.Array,
    base_images: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    max_r: float | None = None,
) -> jax.Array:
    """Per-image backproject: output ``(half_vol, n_images)``.

    Each image scatters to its own column — near-zero atomicAdd contention.
    Follow with GEMM to reduce: ``result @ smz_tri → (half_vol, n_ch)``.
    """
    _ensure_ffi()
    N0, N1, N2 = volume_shape
    H, W = image_shape
    ups = N0 // H
    max_r2_x4 = -1 if max_r is None else int(4 * max_r * max_r)
    base_images = base_images.astype(volumes.dtype)
    rot6 = _rot_to_compact(rotation_matrices, volumes.dtype)
    out_type = jax.ShapeDtypeStruct(volumes.shape, volumes.dtype)

    return jax.ffi.ffi_call(
        _TARGET_PER_IMAGE_BP,
        out_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(
        base_images,
        rot6,
        volumes,
        image_h=H,
        image_w=W,
        vol_n0=N0,
        vol_n1=N1,
        vol_n2=N2,
        upsampling=ups,
        max_r2_x4=max_r2_x4,
    )


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8))
def batch_project(
    volumes: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
    relion_texture_interp: bool = False,
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
            relion_texture_interp=relion_texture_interp,
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


