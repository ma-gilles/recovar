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

import contextvars
import ctypes
import functools
import logging
import math
import operator
import os
import pathlib
import shutil
import subprocess
import sys
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
    "noise_residual.cuh",
    "relion_vdam_mstep.cuh",
    "relion_scoring.cuh",
    "cuda_backproject.cu",
    "relion_coarse_diff2_projector_body.inc",
    "Makefile",
)
_lib_handle = None  # ctypes CDLL
_loaded_lib_path = None

_DISABLE_CUSTOM_CUDA_ENV = "RECOVAR_DISABLE_CUDA"
_CUDA_LIB_ENV = "RECOVAR_CUDA_LIB"
_CUDA_CACHE_DIR_ENV = "RECOVAR_CUDA_CACHE_DIR"
_BUILD_LOCKFILE = ".build.lock"
_RELION_X_HALF_BP_BLOCK_TOPOLOGY_ENV = "RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY"
_RELION_BATCHED_POSTERIOR_PRIMITIVES_ENV = (
    "RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES"
)
_BPREF_DEVICE_SIGNATURE_DUMP_DIR_ENV = "RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR"
_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY_ENV = (
    "RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY"
)
_VDAM_EXTERNAL_HOST_REPLAY_REPORT_DIR_ENV = (
    "RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_REPORT_DIR"
)
_VDAM_EXTERNAL_HOST_REPLAY_CAPTURE_DIR_ENV = (
    "RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_CAPTURE_DIR"
)
_vdam_external_host_replay_lock = threading.Lock()
_vdam_external_host_replay_call = 0
_bpref_device_signature_scope = contextvars.ContextVar(
    "recovar_bpref_device_signature_scope",
    default=None,
)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _next_vdam_external_host_replay_call() -> int:
    global _vdam_external_host_replay_call
    with _vdam_external_host_replay_lock:
        call = _vdam_external_host_replay_call
        _vdam_external_host_replay_call += 1
    return call


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


def relion_batched_posterior_primitives_requested() -> bool:
    """Return whether exact row-wise posterior CUDA calls are batched.

    The value is consumed while JAX traces the posterior caller. Change it only
    between fresh processes so an already cached executable cannot silently
    change its execution contract.
    """

    return _env_flag(_RELION_BATCHED_POSTERIOR_PRIMITIVES_ENV)


@contextmanager
def bpref_device_signature_scope(active: bool):
    """Scope trace-time CUDA diagnostic flags to one explicit score boundary."""

    token = _bpref_device_signature_scope.set(bool(active))
    try:
        yield
    finally:
        _bpref_device_signature_scope.reset(token)


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
_TARGET_BACKPROJECT_INDEXED_SIGNATURE = "cuda_backproject_indexed_signature"
_TARGET_PROJECT = "cuda_project"
_TARGET_PROJECT_RELION_HALF_RUNTIME = "cuda_project_relion_half_runtime"
_TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS = "cuda_project_relion_half_image_radius"
_TARGET_PROJECT_INDEXED = "cuda_project_indexed"
_TARGET_BATCH_BACKPROJECT = "cuda_batch_backproject"
_TARGET_BATCH_BACKPROJECT_INDEXED = "cuda_batch_backproject_indexed"
_TARGET_BATCH_BP_INTERLEAVED = "cuda_batch_bp_interleaved"
_TARGET_FUSED_BP = "cuda_fused_bp"
_TARGET_PER_IMAGE_BP = "cuda_per_image_bp"
_TARGET_RELION_FUSED_X_HALF_BP = "cuda_relion_fused_x_half_bp"
_TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID = (
    "cuda_relion_fused_x_half_bp_particle_grid"
)
_TARGET_RELION_FUSED_X_HALF_BP_SIGNATURE = "cuda_relion_fused_x_half_bp_signature"
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
_TARGET_RELION_BPREF_OPERANDS_F32 = "cuda_relion_bpref_operands_f32"
_TARGET_BPREF_PARTICLE_PACK = "cuda_bpref_particle_pack"
_TARGET_DEFERRED_VDAM_HOST_PACK = "cuda_deferred_vdam_host_pack"
_TARGET_NOISE_PIXEL_PACK = "cuda_noise_pixel_pack"
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
_TARGET_RELION_WAVG_ROTATION_ATOMIC_F32 = "cuda_relion_wavg_rotation_atomic_f32"
_TARGET_RELION_WAVG_ROTATION_ATOMIC_ADD_F32 = "cuda_relion_wavg_rotation_atomic_add_f32"
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
_TARGET_RELION_WAVG_NATIVE_PREFIX_F32 = "cuda_relion_wavg_native_prefix_f32"
_TARGET_RELION_WAVG_NATIVE_PREFIX_DEBUG_F32 = "cuda_relion_wavg_native_prefix_debug_f32"
_wavg_native_prefix_ffi_registered = False

_TARGET_DUAL_WEIGHTED_SUMS_F32 = "cuda_dual_weighted_sums_f32"

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
    (_TARGET_BATCH_BP_INTERLEAVED, "BatchBackprojectInterleaved"),
    (_TARGET_FUSED_BP, "FusedBackproject"),
    (_TARGET_PER_IMAGE_BP, "PerImageBackproject"),
    (_TARGET_RELION_FUSED_X_HALF_BP, "RelionFusedXHalfBackproject"),
    (
        _TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID,
        "RelionFusedXHalfBackprojectParticleGrid",
    ),
    (
        _TARGET_RELION_FUSED_X_HALF_BP_SIGNATURE,
        "RelionFusedXHalfBackprojectSignature",
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
    (_TARGET_RELION_BPREF_OPERANDS_F32, "RelionBprefOperandsF32"),
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
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_F32,
        "RelionWavgRotationAtomicF32",
    ),
    (
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_ADD_F32,
        "RelionWavgRotationAtomicAddF32",
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


_projector_capacity_ffi_registered = False
_projector_image_radius_ffi_registered = False
_bpref_projector_capacity_ffi_registered = False
_TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF = (
    "cuda_relion_vdam_mstep_fused_projector_capacity_x_half"
)


def _ensure_projector_capacity_ffi():
    """Register the opt-in ABI without invalidating qualified older libraries."""
    global _projector_capacity_ffi_registered
    _ensure_ffi()
    if _projector_capacity_ffi_registered:
        return
    with _ffi_lock:
        if _projector_capacity_ffi_registered:
            return
        lib = _get_lib()
        symbol = getattr(lib, "ProjectRelionHalfRuntime", None)
        if symbol is None:
            raise RuntimeError(
                "Projector capacity was requested but the loaded CUDA library lacks "
                "ProjectRelionHalfRuntime; explicitly rebuild the custom CUDA library"
            )
        jax.ffi.register_ffi_target(
            _TARGET_PROJECT_RELION_HALF_RUNTIME,
            jax.ffi.pycapsule(symbol),
            platform="CUDA",
        )
        _projector_capacity_ffi_registered = True


def _ensure_projector_image_radius_ffi():
    """Load the image-radius ABI only when the corrected projector is used."""
    global _projector_image_radius_ffi_registered
    _ensure_ffi()
    if _projector_image_radius_ffi_registered:
        return
    with _ffi_lock:
        if _projector_image_radius_ffi_registered:
            return
        symbol = getattr(_get_lib(), "ProjectRelionHalfImageRadius", None)
        if symbol is None:
            raise RuntimeError(
                "Image-radius projection requires ProjectRelionHalfImageRadius; "
                "explicitly rebuild the custom CUDA library"
            )
        jax.ffi.register_ffi_target(
            _TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS,
            jax.ffi.pycapsule(symbol),
            platform="CUDA",
        )
        _projector_image_radius_ffi_registered = True


def _ensure_bpref_projector_capacity_ffi():
    """Register only on explicit use, preserving qualified legacy libraries."""
    global _bpref_projector_capacity_ffi_registered
    _ensure_ffi()
    if _bpref_projector_capacity_ffi_registered:
        return
    with _ffi_lock:
        if _bpref_projector_capacity_ffi_registered:
            return
        symbol = getattr(_get_lib(), "RelionVdamMstepFusedProjectorCapacityXHalf", None)
        if symbol is None:
            raise RuntimeError(
                "BPref projector capacity requires RelionVdamMstepFusedProjectorCapacityXHalf; "
                "explicitly rebuild the custom CUDA library"
            )
        jax.ffi.register_ffi_target(
            _TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF,
            jax.ffi.pycapsule(symbol), platform="CUDA",
        )
        _bpref_projector_capacity_ffi_registered = True


def _ensure_wavg_native_prefix_ffi():
    """Optional targets must never invalidate a qualified legacy library."""
    global _wavg_native_prefix_ffi_registered
    _ensure_ffi()
    if _wavg_native_prefix_ffi_registered:
        return
    with _ffi_lock:
        if _wavg_native_prefix_ffi_registered:
            return
        registrations = (
            (_TARGET_RELION_WAVG_NATIVE_PREFIX_F32, "RelionWavgNativePrefixF32"),
            (_TARGET_RELION_WAVG_NATIVE_PREFIX_DEBUG_F32, "RelionWavgNativePrefixDebugF32"),
        )
        symbols = [(target, getattr(_get_lib(), symbol, None)) for target, symbol in registrations]
        if any(symbol is None for _, symbol in symbols):
            raise RuntimeError("Native Wavg prefix requires an explicitly rebuilt CUDA library")
        for target, symbol in symbols:
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(symbol), platform="CUDA")
        _wavg_native_prefix_ffi_registered = True


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


@functools.partial(jax.jit, static_argnums=(2,))
def relion_make_scoring_rotations_f32(
    eulers_deg: jax.Array,
    right_matrix: jax.Array,
    do_right: bool = True,
) -> jax.Array:
    """Build scorer rotations with RELION's accelerated float32 arithmetic.

    This reproduces ``cuda_kernel_make_eulers_3D<true,false,do_right>``
    through its Euler construction and optional ``B = A @ R`` product.  The
    returned matrices use RECOVAR's scorer convention ``B``.  RELION stores
    the inverse-projector convention ``B.T``, so a frozen RELION device dump
    must be transposed before a bitwise comparison with this result.

    This is a strict CUDA primitive: it has no CPU or JAX arithmetic fallback.
    """

    if eulers_deg.dtype != jnp.float32:
        raise TypeError(f"eulers_deg must be float32, got {eulers_deg.dtype}")
    if right_matrix.dtype != jnp.float32:
        raise TypeError(f"right_matrix must be float32, got {right_matrix.dtype}")
    if eulers_deg.ndim != 2 or eulers_deg.shape[1:] != (3,):
        raise ValueError(f"eulers_deg must have shape (N, 3), got {eulers_deg.shape}")
    if right_matrix.shape != (3, 3):
        raise ValueError(f"right_matrix must have shape (3, 3), got {right_matrix.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION scorer-rotation construction requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION scorer-rotation construction was explicitly requested but custom CUDA is disabled")
    _ensure_ffi()

    out_type = jax.ShapeDtypeStruct((eulers_deg.shape[0], 3, 3), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_MAKE_SCORING_ROTATIONS_F32,
        out_type,
        vmap_method="sequential",
    )(
        eulers_deg,
        right_matrix,
        do_right=np.int64(int(do_right)),
    )


@functools.partial(jax.jit, static_argnums=(2,))
def relion_make_scoring_rotations_f64(
    eulers_deg: jax.Array,
    right_matrix: jax.Array,
    do_right: bool = True,
) -> jax.Array:
    """Build scorer rotations with RELION's accelerated float64 arithmetic.

    This is the ``ACC_DOUBLE_PRECISION`` specialization of RELION's CUDA
    ``make_eulers_3D`` operation. It deliberately executes trigonometry and
    the optional right-matrix product on the device, matching coarse scoring.
    """

    if eulers_deg.dtype != jnp.float64:
        raise TypeError(f"eulers_deg must be float64, got {eulers_deg.dtype}")
    if right_matrix.dtype != jnp.float64:
        raise TypeError(f"right_matrix must be float64, got {right_matrix.dtype}")
    if eulers_deg.ndim != 2 or eulers_deg.shape[1:] != (3,):
        raise ValueError(f"eulers_deg must have shape (N, 3), got {eulers_deg.shape}")
    if right_matrix.shape != (3, 3):
        raise ValueError(f"right_matrix must have shape (3, 3), got {right_matrix.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION scorer-rotation construction requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION scorer-rotation construction was explicitly requested but custom CUDA is disabled")
    _ensure_ffi()

    out_type = jax.ShapeDtypeStruct((eulers_deg.shape[0], 3, 3), jnp.float64)
    return jax.ffi.ffi_call(
        _TARGET_RELION_MAKE_SCORING_ROTATIONS_F64,
        out_type,
        vmap_method="sequential",
    )(
        eulers_deg,
        right_matrix,
        do_right=np.int64(int(do_right)),
    )


@functools.partial(jax.jit, static_argnums=(3,))
def relion_translate_score_f32(
    images: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate score images with RELION's accelerated float32 arithmetic.

    ``translation_angles`` contains RELION's per-translation ``(tx, ty)``
    radians. ``pixel_indices`` use RECOVAR's centered half-spectrum layout.
    The CUDA primitive evaluates ``sincosf(x*tx + y*ty)`` and the explicit
    real/imaginary products used by RELION's fine Gaussian scorer. The output
    is flattened in image-major, translation-major order to match
    :func:`apply_half_translation_phases`.

    This is a strict CUDA primitive. Callers that support a non-CUDA fallback
    must select it before invoking this function.
    """

    if images.dtype != jnp.complex64:
        raise TypeError(f"images must be complex64, got {images.dtype}")
    if translation_angles.dtype != jnp.float32:
        raise TypeError(
            f"translation_angles must be float32, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION score translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION score translation was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_TRANSLATE_SCORE_F32,
        out_type,
        vmap_method="sequential",
    )(
        images,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(3,))
def relion_translate_score_f64(
    images: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate score/M-step images with RELION's CUDA double arithmetic."""

    if images.dtype != jnp.complex128:
        raise TypeError(f"images must be complex128, got {images.dtype}")
    if translation_angles.dtype != jnp.float64:
        raise TypeError(
            f"translation_angles must be float64, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION score translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION score translation was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex128,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_TRANSLATE_SCORE_F64,
        out_type,
        vmap_method="sequential",
    )(
        images,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@jax.jit
def relion_exponentiate_f32(values: jax.Array, add: jax.Array) -> jax.Array:
    """Apply RELION's fine-weight ``expf(value + add)`` CUDA kernel."""

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if add.dtype != jnp.float32 or add.ndim != 0:
        raise TypeError(f"add must be a float32 scalar, got {add.dtype} {add.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION float32 exponentiation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION float32 exponentiation was requested but custom CUDA is disabled")
    _ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_EXPONENTIATE_F32,
        output_type,
        vmap_method="sequential",
    )(values, add)


@jax.jit
def relion_exponentiate_batched_f32(
    values: jax.Array,
    add: jax.Array,
) -> jax.Array:
    """Apply RELION's float32 posterior exponentiation to a row batch."""

    if values.dtype != jnp.float32 or values.ndim != 2:
        raise TypeError(f"values must be a float32 matrix, got {values.dtype} {values.shape}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be nonempty, got {values.shape}")
    if add.dtype != jnp.float32 or add.shape != (values.shape[0],):
        raise TypeError(
            "add must be one float32 scalar per row, got "
            f"{add.dtype} {add.shape} for values {values.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION batched float32 exponentiation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION batched float32 exponentiation was requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    return jax.ffi.ffi_call(
        _TARGET_RELION_EXPONENTIATE_BATCHED_F32,
        jax.ShapeDtypeStruct(values.shape, jnp.float32),
    )(values, add)


@jax.jit
def relion_divide_f32(values: jax.Array, divisor: jax.Array) -> jax.Array:
    """Apply RELION's CUDA ``float / float`` posterior normalization."""

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if divisor.dtype != jnp.float32 or divisor.ndim != 0:
        raise TypeError(
            f"divisor must be a float32 scalar, got {divisor.dtype} {divisor.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION float32 division requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION float32 division was requested but custom CUDA is disabled")
    _ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_DIVIDE_F32,
        output_type,
        vmap_method="sequential",
    )(values, divisor)


@jax.jit
def relion_divide_batched_f32(
    values: jax.Array,
    divisor: jax.Array,
) -> jax.Array:
    """Apply RELION float32 division to all posterior rows in one FFI call."""

    if values.dtype != jnp.float32 or values.ndim != 2:
        raise TypeError(f"values must be a float32 matrix, got {values.dtype} {values.shape}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be nonempty, got {values.shape}")
    if divisor.dtype != jnp.float32 or divisor.shape != (values.shape[0],):
        raise TypeError(
            "divisor must be one float32 scalar per row, got "
            f"{divisor.dtype} {divisor.shape} for values {values.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION batched float32 division requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION batched float32 division was requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    return jax.ffi.ffi_call(
        _TARGET_RELION_DIVIDE_BATCHED_F32,
        jax.ShapeDtypeStruct(values.shape, jnp.float32),
    )(values, divisor)


@jax.jit
def relion_cub_sort_scan_f32(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sort and inclusively scan one float32 vector with RELION's CUB calls.

    This is a strict diagnostic primitive for the coarse-significance boundary.
    RELION invokes ``cub::DeviceRadixSort::SortKeys`` followed by
    ``cub::DeviceScan::InclusiveSum`` on each particle's positive weights.
    Keeping both intermediate arrays observable lets parity tests identify a
    sort discrepancy separately from a scan discrepancy.
    """

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION CUB sort/scan requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION CUB sort/scan was requested but custom CUDA is disabled")
    _ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_CUB_SORT_SCAN_F32,
        (output_type, output_type),
        vmap_method="sequential",
    )(values)


@jax.jit
def relion_cub_sort_scan_batched_f32(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run the exact RELION CUB sort/scan row order in one FFI call.

    Rows remain serialized on the caller's XLA stream. The implementation
    reuses one stream-ordered scratch allocation, so this changes dispatch and
    allocation topology without changing any row's CUB arithmetic.
    """

    if values.dtype != jnp.float32 or values.ndim != 2:
        raise TypeError(f"values must be a float32 matrix, got {values.dtype} {values.shape}")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"values must be nonempty, got {values.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION batched CUB sort/scan requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION batched CUB sort/scan was requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_CUB_SORT_SCAN_BATCHED_F32,
        (output_type, output_type),
    )(values)


@jax.jit
def relion_cub_positive_sort_scan_f32(
    values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Select positive weights before RELION's CUB sort and scan.

    RELION's coarse posterior passes only strictly positive weights to its
    radix sort and inclusive scan. This explicit experimental primitive
    mirrors that sequence while retaining fixed JAX output shapes: selected
    values and their cumulative sums are right-aligned behind a zero prefix.
    The existing :func:`relion_cub_sort_scan_f32` behavior is unchanged.
    """

    if values.dtype != jnp.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError(f"values must be a nonempty 1-D array, got {values.shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION positive CUB sort/scan requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION positive CUB sort/scan was requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    output_type = jax.ShapeDtypeStruct(values.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_CUB_POSITIVE_SORT_SCAN_F32,
        (output_type, output_type),
        vmap_method="sequential",
    )(values)


@functools.partial(jax.jit, static_argnums=(4,))
def relion_translate_bpref_f32(
    images: jax.Array,
    weighted_ctf: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate raw images, then apply BPref weights in RELION float32 order.

    RELION evaluates ``sincosf`` and the explicit complex translation before
    multiplying each translated real/imaginary component by ``weighted_ctf``.
    Keeping this as one CUDA primitive avoids the extra rounding boundary in
    RECOVAR's algebraically equivalent weight-then-translate path.
    """

    if images.dtype != jnp.complex64:
        raise TypeError(f"images must be complex64, got {images.dtype}")
    if weighted_ctf.dtype != jnp.float32:
        raise TypeError(f"weighted_ctf must be float32, got {weighted_ctf.dtype}")
    if translation_angles.dtype != jnp.float32:
        raise TypeError(
            f"translation_angles must be float32, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if weighted_ctf.shape != images.shape:
        raise ValueError(
            f"weighted_ctf must have shape {images.shape}, got {weighted_ctf.shape}"
        )
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION BPref translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION BPref translation was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex64,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_TRANSLATE_BPREF_F32,
        out_type,
        vmap_method="sequential",
    )(
        images,
        weighted_ctf,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(4,))
def relion_translate_bpref_f64(
    images: jax.Array,
    weighted_ctf: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
) -> jax.Array:
    """Translate raw images, then apply BPref weights in RELION double order."""

    if images.dtype != jnp.complex128:
        raise TypeError(f"images must be complex128, got {images.dtype}")
    if weighted_ctf.dtype != jnp.float64:
        raise TypeError(f"weighted_ctf must be float64, got {weighted_ctf.dtype}")
    if translation_angles.dtype != jnp.float64:
        raise TypeError(
            f"translation_angles must be float64, got {translation_angles.dtype}"
        )
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if weighted_ctf.shape != images.shape:
        raise ValueError(
            f"weighted_ctf must have shape {images.shape}, got {weighted_ctf.shape}"
        )
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError(
            "translation_angles must have shape (translations, 2), got "
            f"{translation_angles.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION BPref translation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION BPref translation was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    out_type = jax.ShapeDtypeStruct(
        (
            images.shape[0] * translation_angles.shape[0],
            images.shape[1],
        ),
        jnp.complex128,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_TRANSLATE_BPREF_F64,
        out_type,
        vmap_method="sequential",
    )(
        images,
        weighted_ctf,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
    )


@functools.partial(jax.jit, static_argnums=(6, 7))
def relion_bpref_operands_f32(
    images: jax.Array,
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior_over_weight_norm: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    image_shape: Tuple[int, int],
    arithmetic_variant: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate RELION BP.cuh's native-unit numerator and denominator terms."""

    if images.dtype != jnp.complex64:
        raise TypeError(f"images must be complex64, got {images.dtype}")
    for name, value in (
        ("ctf", ctf),
        ("minvsigma2", minvsigma2),
        ("posterior_over_weight_norm", posterior_over_weight_norm),
        ("translation_angles", translation_angles),
    ):
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if ctf.shape != images.shape or minvsigma2.shape != images.shape:
        raise ValueError("ctf and minvsigma2 must have the same shape as images")
    if translation_angles.ndim != 2 or translation_angles.shape[1:] != (2,):
        raise ValueError("translation_angles must have shape (translations, 2)")
    expected_posterior_shape = (images.shape[0], translation_angles.shape[0])
    if posterior_over_weight_norm.shape != expected_posterior_shape:
        raise ValueError(
            "posterior_over_weight_norm must have shape "
            f"{expected_posterior_shape}, got {posterior_over_weight_norm.shape}"
        )
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(
            f"pixel_indices must have shape ({images.shape[1]},), got "
            f"{pixel_indices.shape}"
        )
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if int(arithmetic_variant) not in range(24):
        raise ValueError("arithmetic_variant must be in [0, 23]")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION BPref operands require a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION BPref operands were explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    half_width = image_w // 2 + 1
    output_shape = (
        images.shape[0] * translation_angles.shape[0],
        images.shape[1],
    )
    output_types = (
        jax.ShapeDtypeStruct(output_shape, jnp.complex64),
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
        jax.ShapeDtypeStruct(output_shape, jnp.complex64),
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_BPREF_OPERANDS_F32,
        output_types,
        vmap_method="sequential",
    )(
        images,
        ctf,
        minvsigma2,
        posterior_over_weight_norm,
        translation_angles,
        pixel_indices,
        image_h=np.int64(image_h),
        image_half_width=np.int64(half_width),
        arithmetic_variant=np.int64(arithmetic_variant),
    )


@functools.partial(jax.jit, static_argnums=(7,))
def relion_vdam_mstep_sums_f32(
    images: jax.Array,
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior_over_weight_norm: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    reference: jax.Array,
    image_shape: Tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    """Reduce RELION VDAM BPref residual operands in native statement order.

    The returned arrays have shape ``(batch, rotations, pixels)``.  Translation
    weights are consumed sequentially inside the CUDA kernel, matching
    ``cuda_kernel_backproject3D_SGD`` before its particle-grid scatter.
    """

    if images.dtype != jnp.complex64 or reference.dtype != jnp.complex64:
        raise TypeError("images and reference must be complex64")
    for name, value in (
        ("ctf", ctf),
        ("minvsigma2", minvsigma2),
        ("posterior_over_weight_norm", posterior_over_weight_norm),
        ("translation_angles", translation_angles),
    ):
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
    if pixel_indices.dtype != jnp.int32:
        raise TypeError(f"pixel_indices must be int32, got {pixel_indices.dtype}")
    if images.ndim != 2:
        raise ValueError(f"images must have shape (batch, pixels), got {images.shape}")
    if ctf.shape != images.shape or minvsigma2.shape != images.shape:
        raise ValueError("ctf and minvsigma2 must have the same shape as images")
    if posterior_over_weight_norm.ndim != 3:
        raise ValueError("posterior_over_weight_norm must have shape (batch, rotations, translations)")
    if posterior_over_weight_norm.shape[0] != images.shape[0]:
        raise ValueError("posterior batch dimension must match images")
    if reference.shape != (
        images.shape[0],
        posterior_over_weight_norm.shape[1],
        images.shape[1],
    ):
        raise ValueError(
            "reference must have shape (batch, rotations, pixels), got "
            f"{reference.shape}"
        )
    if translation_angles.shape != (posterior_over_weight_norm.shape[2], 2):
        raise ValueError("translation_angles must have shape (translations, 2)")
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError(f"pixel_indices must have shape ({images.shape[1]},)")
    if len(image_shape) != 2 or any(int(size) <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive sizes, got {image_shape}")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION VDAM M-step sums require a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION VDAM M-step sums were explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    image_h, image_w = (int(size) for size in image_shape)
    output_shape = reference.shape
    output_types = (
        jax.ShapeDtypeStruct(output_shape, jnp.complex64),
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_VDAM_MSTEP_SUMS_F32,
        output_types,
        vmap_method="sequential",
    )(
        images,
        ctf,
        minvsigma2,
        posterior_over_weight_norm,
        translation_angles,
        pixel_indices,
        reference,
        image_h=np.int64(image_h),
        image_half_width=np.int64(image_w // 2 + 1),
    )


_bpref_particle_pack_ffi_registered = False
_deferred_vdam_host_pack_ffi_registered = False
_noise_pixel_pack_ffi_registered = False


def _ensure_noise_pixel_pack_ffi():
    global _noise_pixel_pack_ffi_registered
    _ensure_ffi()
    if _noise_pixel_pack_ffi_registered:
        return
    with _ffi_lock:
        if _noise_pixel_pack_ffi_registered:
            return
        symbol = getattr(_get_lib(), "NoisePixelPack", None)
        if symbol is None:
            raise RuntimeError("CUDA noise pixel packing requires an explicit build with NoisePixelPack")
        jax.ffi.register_ffi_target(_TARGET_NOISE_PIXEL_PACK, jax.ffi.pycapsule(symbol), platform="CUDA")
        _noise_pixel_pack_ffi_registered = True


def _noise_pixel_pack_shapes(probs, projection, ctf_probs, indices, spare_index, *, target_batch):
    if type(target_batch) is not int or target_batch <= 0:
        raise ValueError("Noise pixel target batch must be a positive integer")
    values = (probs, projection, ctf_probs, indices, spare_index)
    types = ((jnp.float32, jnp.float64), (jnp.complex64, jnp.complex128),
             (jnp.float32, jnp.float64), (jnp.int32, jnp.int64), (jnp.int32, jnp.int64))
    for value, allowed, rank in zip(values, types, (3, 3, 3, 1, 0), strict=True):
        if value.dtype not in allowed:
            raise TypeError("Noise pixel input dtype differs")
        if value.ndim != rank or any(n <= 0 for n in value.shape):
            raise ValueError("Noise pixel input rank or dimension differs")
    batch, rotations, _ = probs.shape
    if (projection.shape[:2] != (batch, rotations) or ctf_probs.shape != projection.shape
            or indices.shape != (batch,) or spare_index.dtype != indices.dtype or target_batch < batch):
        raise ValueError("Noise pixel input geometry or index dtype differs")
    return tuple(jax.ShapeDtypeStruct((target_batch, *v.shape[1:]), v.dtype) for v in values[:4])


@functools.partial(jax.jit, static_argnames=("target_batch",))
def pad_noise_pixels_cuda(probs, projection, ctf_probs, indices, spare_index, *, target_batch):
    """Preserve all prefix bytes and fill zero/data or spare-index tails on device."""
    args = (probs, projection, ctf_probs, indices, spare_index)
    outputs = _noise_pixel_pack_shapes(*args, target_batch=target_batch)
    if jax.default_backend() != "gpu" or not custom_cuda_requested():
        raise RuntimeError("CUDA noise pixel packing requires an enabled JAX GPU backend")
    _ensure_noise_pixel_pack_ffi()
    return tuple(jax.ffi.ffi_call(_TARGET_NOISE_PIXEL_PACK, outputs, vmap_method="sequential")(
        *args, target_batch=target_batch,
    ))


def _ensure_deferred_vdam_host_pack_ffi():
    global _deferred_vdam_host_pack_ffi_registered
    _ensure_ffi()
    if _deferred_vdam_host_pack_ffi_registered:
        return
    with _ffi_lock:
        if _deferred_vdam_host_pack_ffi_registered:
            return
        symbol = getattr(_get_lib(), "DeferredVdamHostPack", None)
        if symbol is None:
            raise RuntimeError("CUDA host-plan packing requires an explicit build with DeferredVdamHostPack")
        jax.ffi.register_ffi_target(
            _TARGET_DEFERRED_VDAM_HOST_PACK, jax.ffi.pycapsule(symbol), platform="CUDA",
        )
        _deferred_vdam_host_pack_ffi_registered = True


def _deferred_vdam_host_pack_shapes(
    posterior, sum_t, images, ctf, minvsigma2, flat_projection, take, mask, flat_take,
):
    for value, dtype, rank in zip(
        (posterior, sum_t, images, ctf, minvsigma2, flat_projection, take, mask, flat_take),
        (jnp.float32, jnp.float32, jnp.complex64, jnp.float32, jnp.float32,
         jnp.complex64, jnp.int32, jnp.bool_, jnp.int32),
        (3, 2, 2, 2, 2, 2, 2, 2, 2), strict=True,
    ):
        if value.dtype != dtype:
            raise TypeError("CUDA host-plan packing input dtype differs")
        if value.ndim != rank or any(n <= 0 for n in value.shape):
            raise ValueError("CUDA host-plan packing requires positive dimensions and matching ranks")
    batch, rotations = take.shape
    if posterior.shape[0] < batch or images.shape[0] < batch or sum_t.shape != posterior.shape[:2]:
        raise ValueError("CUDA host-plan packing posterior/image batch geometry differs")
    if ctf.shape != images.shape or minvsigma2.shape != images.shape or flat_projection.shape[1] != images.shape[1]:
        raise ValueError("CUDA host-plan packing pixel geometry differs")
    if mask.shape != take.shape or flat_take.shape != take.shape:
        raise ValueError("CUDA host-plan packing descriptor shapes differ")
    pixels, translations = images.shape[1], posterior.shape[2]
    return tuple(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in (
        ((batch, rotations, translations), jnp.float32),
        ((batch, rotations), jnp.float32),
        ((batch, pixels), jnp.complex64),
        ((batch, pixels), jnp.float32),
        ((batch, pixels), jnp.float32),
        ((batch, rotations, pixels), jnp.complex64),
        ((batch, rotations, pixels), jnp.float32),
    ))


@jax.jit
def pack_deferred_vdam_host_plan_cuda(
    posterior, sum_t, images, ctf, minvsigma2, flat_projection, take, mask, flat_take,
):
    """Gather/mask on device and reuse the original sequential CUDA denominator."""
    args = (posterior, sum_t, images, ctf, minvsigma2, flat_projection, take, mask, flat_take)
    outputs = _deferred_vdam_host_pack_shapes(*args)
    if jax.default_backend() != "gpu" or not custom_cuda_requested():
        raise RuntimeError("CUDA host-plan packing requires an enabled JAX GPU backend")
    _ensure_deferred_vdam_host_pack_ffi()
    return tuple(jax.ffi.ffi_call(
        _TARGET_DEFERRED_VDAM_HOST_PACK, outputs, vmap_method="sequential",
    )(*args))


def _ensure_bpref_particle_pack_ffi():
    """Register only when requested, preserving compatibility with older builds."""
    global _bpref_particle_pack_ffi_registered
    _ensure_ffi()
    if _bpref_particle_pack_ffi_registered:
        return
    with _ffi_lock:
        if _bpref_particle_pack_ffi_registered:
            return
        symbol = getattr(_get_lib(), "BprefParticlePack", None)
        if symbol is None:
            raise RuntimeError("CUDA BPref packing requires an explicit build with BprefParticlePack")
        jax.ffi.register_ffi_target(
            _TARGET_BPREF_PARTICLE_PACK, jax.ffi.pycapsule(symbol), platform="CUDA",
        )
        _bpref_particle_pack_ffi_registered = True


def _bpref_particle_pack_shapes(columns, capacity):
    """Validate the six-column BPref copy ABI without inspecting array values."""
    if type(capacity) is not int or not 1 <= capacity <= 256:
        raise ValueError("BPref packing capacity must be an integer from 1 to 256")
    if len(columns) != 6 or not 1 <= len(columns[0]) <= 256:
        raise ValueError("BPref packing requires six nonempty columns")
    count = len(columns[0])
    if any(len(column) != count for column in columns):
        raise ValueError("BPref packing bucket counts differ")
    dtypes = (jnp.complex64, jnp.float32, jnp.float32, jnp.float32, jnp.float32, jnp.int32)
    ranks = (2, 2, 2, 3, 4, 1)
    for column, dtype, rank in zip(columns, dtypes, ranks, strict=True):
        for value in column:
            if value.dtype != dtype:
                raise TypeError("BPref packing field dtype differs")
            if value.ndim != rank or any(n <= 0 for n in value.shape):
                raise ValueError("BPref packing field shape differs")
            if value.shape[1:] != column[0].shape[1:]:
                raise ValueError("BPref packing non-particle shapes differ")
    counts = tuple(value.shape[0] for value in columns[0])
    if sum(counts) > capacity:
        raise ValueError("BPref packing capacity cannot truncate active particles")
    for column in columns:
        if tuple(value.shape[0] for value in column) != counts:
            raise ValueError("BPref packing particle axes differ")
    if columns[1][0].shape != columns[0][0].shape or columns[2][0].shape != columns[0][0].shape:
        raise ValueError("BPref packing image, CTF and inverse-noise shapes differ")
    if columns[4][0].shape[1:] != (columns[3][0].shape[1], 3, 3):
        raise ValueError("BPref packing rotation matrices must match posterior rotations")
    return (
        *(jax.ShapeDtypeStruct((capacity, *column[0].shape[1:]), dtype)
          for column, dtype in zip(columns, dtypes, strict=True)),
        jax.ShapeDtypeStruct((capacity,), jnp.int32),
        jax.ShapeDtypeStruct((capacity,), jnp.int32),
    )


@functools.partial(jax.jit, static_argnums=(1,))
def pack_bpref_particle_fields(columns, capacity):
    """Copy bounded BPref inputs, preserving original per-bucket worker IDs.

    No floating-point arithmetic, support selection, or host device-array reads.
    Group tails are -1; other tails are positive zero. Inputs remain untouched.
    """
    outputs = _bpref_particle_pack_shapes(columns, capacity)
    if jax.default_backend() != "gpu" or not custom_cuda_requested():
        raise RuntimeError("CUDA BPref packing requires an enabled JAX GPU backend")
    _ensure_bpref_particle_pack_ffi()
    return tuple(jax.ffi.ffi_call(
        _TARGET_BPREF_PARTICLE_PACK, outputs, vmap_method="sequential",
    )(*(value for column in columns for value in column)))


@jax.jit
def relion_vdam_mstep_denominator_f32(
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior_over_weight_norm: jax.Array,
) -> jax.Array:
    """Reduce the RELION VDAM denominator without materializing its numerator.

    This exposes the denominator pass already shared by the fused VDAM scatter.
    Translation weights are consumed sequentially in RELION statement order.
    """

    for name, value in (
        ("ctf", ctf),
        ("minvsigma2", minvsigma2),
        ("posterior_over_weight_norm", posterior_over_weight_norm),
    ):
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
    if ctf.ndim != 2:
        raise ValueError(f"ctf must have shape (batch, pixels), got {ctf.shape}")
    if minvsigma2.shape != ctf.shape:
        raise ValueError("minvsigma2 must have the same shape as ctf")
    if posterior_over_weight_norm.ndim != 3:
        raise ValueError(
            "posterior_over_weight_norm must have shape "
            "(batch, rotations, translations)"
        )
    if posterior_over_weight_norm.shape[0] != ctf.shape[0]:
        raise ValueError("posterior batch dimension must match ctf")
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION VDAM M-step denominator requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION VDAM M-step denominator was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    output_shape = (
        ctf.shape[0],
        posterior_over_weight_norm.shape[1],
        ctf.shape[1],
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_VDAM_MSTEP_DENOMINATOR_F32,
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
        vmap_method="sequential",
    )(
        ctf,
        minvsigma2,
        posterior_over_weight_norm,
    )


@functools.partial(jax.jit, static_argnums=(10, 11, 12))
def relion_vdam_mstep_fused_x_half(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    images: jax.Array,
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior_over_weight_norm: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    reference: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Form and scatter VDAM residual rows in RELION's particle launch topology."""

    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(f"RELION fused VDAM M-step requires an odd BPref grid, got {volume_shape}")
    if images.dtype != jnp.complex64 or reference.dtype != jnp.complex64:
        raise TypeError("images and reference must be complex64")
    if data_volume.dtype != jnp.complex64 or weight_volume.dtype != jnp.float32:
        raise TypeError("VDAM accumulators must be complex64/float32")
    for name, value in (
        ("ctf", ctf),
        ("minvsigma2", minvsigma2),
        ("posterior_over_weight_norm", posterior_over_weight_norm),
        ("translation_angles", translation_angles),
        ("rotation_matrices", rotation_matrices),
    ):
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
    if pixel_indices.dtype != jnp.int32:
        raise TypeError("pixel_indices must be int32")
    if images.ndim != 2 or ctf.shape != images.shape or minvsigma2.shape != images.shape:
        raise ValueError("images, ctf, and minvsigma2 must have matching (particle,pixel) shapes")
    if posterior_over_weight_norm.ndim != 3 or posterior_over_weight_norm.shape[0] != images.shape[0]:
        raise ValueError("posterior must have shape (particle,rotation,translation)")
    n_particles, n_rotations, n_translations = map(int, posterior_over_weight_norm.shape)
    if reference.shape != (n_particles, n_rotations, images.shape[1]):
        raise ValueError("reference must have shape (particle,rotation,pixel)")
    if rotation_matrices.shape != (n_particles, n_rotations, 3, 3):
        raise ValueError("rotation_matrices must have shape (particle,rotation,3,3)")
    if translation_angles.shape != (n_translations, 2):
        raise ValueError("translation_angles must have shape (translation,2)")
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError("pixel_indices must match the compact pixel dimension")
    _ensure_ffi()

    dense_images, dense_indices, current_h, current_w = _prepare_relion_x_half_block_topology_operands(
        images, pixel_indices, image_shape, max_r
    )
    dense_ctf, ctf_indices, ctf_h, ctf_w = _prepare_relion_x_half_block_topology_operands(
        ctf, pixel_indices, image_shape, max_r
    )
    dense_minvsigma2, noise_indices, noise_h, noise_w = _prepare_relion_x_half_block_topology_operands(
        minvsigma2, pixel_indices, image_shape, max_r
    )
    flat_reference = reference.reshape(n_particles * n_rotations, images.shape[1])
    dense_reference, reference_indices, reference_h, reference_w = (
        _prepare_relion_x_half_block_topology_operands(
            flat_reference, pixel_indices, image_shape, max_r
        )
    )
    dense_reference = dense_reference.reshape(n_particles, n_rotations, -1)
    if (ctf_h, ctf_w) != (current_h, current_w) or (noise_h, noise_w) != (current_h, current_w):
        raise ValueError("VDAM fused operand topology metadata mismatch")
    if (reference_h, reference_w) != (current_h, current_w):
        raise ValueError("VDAM fused reference topology metadata mismatch")
    if not (ctf_indices.shape == noise_indices.shape == reference_indices.shape == dense_indices.shape):
        raise ValueError("VDAM fused dense pixel topology mismatch")

    kernel_rotations = _relion_x_half_backproject_rotation_to_kernel(
        rotation_matrices.reshape(n_particles * n_rotations, 3, 3), jnp.float32
    )
    rot6 = _rot_to_compact(kernel_rotations, jnp.float32).reshape(n_particles, n_rotations, 6)
    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    data_real_volume = jnp.asarray(data_volume.real, dtype=jnp.float32)
    data_imag_volume = jnp.asarray(data_volume.imag, dtype=jnp.float32)
    output_types = (
        jax.ShapeDtypeStruct(data_real_volume.shape, jnp.float32),
        jax.ShapeDtypeStruct(data_imag_volume.shape, jnp.float32),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
        jax.ShapeDtypeStruct(dense_reference.shape, jnp.float32),
    )
    fused_real, fused_imag, fused_weight, dense_denominator = jax.ffi.ffi_call(
        _TARGET_RELION_VDAM_MSTEP_FUSED_X_HALF,
        output_types,
        input_output_aliases={7: 0, 8: 1, 9: 2},
        vmap_method="sequential",
    )(
        dense_images,
        dense_ctf,
        dense_minvsigma2,
        posterior_over_weight_norm,
        translation_angles,
        dense_reference,
        rot6,
        data_real_volume,
        data_imag_volume,
        weight_volume,
        image_h=np.int64(current_h),
        image_w=np.int64(current_w),
        N0=kw["N0"],
        N1=kw["N1"],
        N2=kw["N2"],
        upsampling=kw["upsampling"],
        max_r2_x4=kw["max_r2_x4"],
    )
    fused_data = jax.lax.complex(fused_real, fused_imag)
    full_h, full_w = map(int, image_shape)
    full_half_w = full_w // 2 + 1
    full_rows = pixel_indices // full_half_w
    columns = pixel_indices % full_half_w
    signed_rows = jnp.where(full_rows <= full_h // 2, full_rows, full_rows - full_h)
    current_indices = jnp.mod(signed_rows, current_h) * current_w + columns
    compact_denominator = jnp.take(dense_denominator, current_indices, axis=-1)
    return fused_data, fused_weight, compact_denominator


def _run_vdam_external_host_replay_callback(
    projector_full,
    dense_images,
    dense_ctf,
    dense_minvsigma2,
    posterior_over_weight_norm,
    translation_angles,
    projector_eulers,
    compact_rotations,
    reconstruction_group_ids,
    worker_lane_ids,
    particle_trace_ids,
    rotation_replay_order,
    rotation_replay_counts,
    particle_start_offsets_ns,
    data_real_volume,
    data_imag_volume,
    weight_volume,
    *,
    image_h: int,
    image_w: int,
    volume_shape: tuple[int, int, int],
    upsampling: int,
    max_r2_x4: int,
    physical_image_size: int,
    projector_max_r: int,
    projection_padding_factor: int,
    reconstruction_group_count: int,
    parallel_worker_replay: bool,
):
    """Materialize one VDAM accumulation through a fresh CUDA process."""

    library_text = os.environ.get(_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY_ENV, "").strip()
    if not library_text:
        raise RuntimeError("external VDAM host replay library is not configured")
    library = pathlib.Path(library_text).expanduser().resolve()
    if not library.is_file():
        raise FileNotFoundError(library)
    exact_ptx = os.environ.get("RECOVAR_VDAM_EXACT_NATIVE_PTX", "").strip()
    if not exact_ptx:
        raise RuntimeError("external VDAM host replay requires exact native PTX")

    helper = pathlib.Path(__file__).resolve().parents[1] / "scripts" / (
        "run_vdam_exact_native_host_replay.py"
    )
    if not helper.is_file():
        raise FileNotFoundError(helper)
    call = _next_vdam_external_host_replay_call()
    temp_parent = os.environ.get("TMPDIR", "").strip() or None
    with tempfile.TemporaryDirectory(
        prefix=f"recovar-vdam-host-replay-{call:04d}-",
        dir=temp_parent,
    ) as temporary_directory:
        root = pathlib.Path(temporary_directory)
        input_path = root / "input.npz"
        output_path = root / "output.npz"
        report_path = root / "report.json"
        n_particles, rotation_count, translation_count = map(
            int, np.asarray(posterior_over_weight_norm).shape
        )
        pixel_count = int(np.asarray(dense_images).shape[1])
        projector_size = int(np.asarray(projector_full).shape[0])
        np.savez(
            input_path,
            projector_full=np.asarray(projector_full, dtype=np.complex64),
            images=np.asarray(dense_images, dtype=np.complex64),
            ctf=np.asarray(dense_ctf, dtype=np.float32),
            minvsigma2=np.asarray(dense_minvsigma2, dtype=np.float32),
            posterior_over_weight_norm=np.asarray(
                posterior_over_weight_norm, dtype=np.float32
            ),
            translation_angles=np.asarray(translation_angles, dtype=np.float32),
            projector_eulers=np.asarray(projector_eulers, dtype=np.float32),
            compact_rotations=np.asarray(compact_rotations, dtype=np.float32),
            reconstruction_group_ids=np.asarray(
                reconstruction_group_ids, dtype=np.int32
            ),
            worker_lane_ids=np.asarray(worker_lane_ids, dtype=np.int32),
            particle_trace_ids=np.asarray(particle_trace_ids, dtype=np.int32),
            rotation_replay_order=np.asarray(rotation_replay_order, dtype=np.int32),
            rotation_replay_counts=np.asarray(
                rotation_replay_counts, dtype=np.int32
            ),
            particle_start_offsets_ns=np.asarray(
                particle_start_offsets_ns, dtype=np.int32
            ),
            data_real_volume=np.asarray(data_real_volume, dtype=np.float32).reshape(
                reconstruction_group_count, -1
            ),
            data_imag_volume=np.asarray(data_imag_volume, dtype=np.float32).reshape(
                reconstruction_group_count, -1
            ),
            weight_volume=np.asarray(weight_volume, dtype=np.float32).reshape(
                reconstruction_group_count, -1
            ),
            projector_size=np.int64(projector_size),
            n_particles=np.int64(n_particles),
            rotation_count=np.int64(rotation_count),
            translation_count=np.int64(translation_count),
            pixel_count=np.int64(pixel_count),
            image_h=np.int64(image_h),
            image_w=np.int64(image_w),
            volume_n0=np.int64(volume_shape[0]),
            volume_n1=np.int64(volume_shape[1]),
            volume_n2=np.int64(volume_shape[2]),
            upsampling=np.int64(upsampling),
            max_r2_x4=np.int64(max_r2_x4),
            physical_image_size=np.int32(physical_image_size),
            projector_max_r=np.int32(projector_max_r),
            projection_padding_factor=np.int32(projection_padding_factor),
            reconstruction_group_count=np.int32(reconstruction_group_count),
            parallel_worker_replay=np.int32(parallel_worker_replay),
        )
        capture_dir_text = os.environ.get(
            _VDAM_EXTERNAL_HOST_REPLAY_CAPTURE_DIR_ENV, ""
        ).strip()
        if capture_dir_text:
            capture_dir = pathlib.Path(capture_dir_text).expanduser().resolve()
            capture_dir.mkdir(parents=True, exist_ok=True)
            capture_path = capture_dir / (
                f"pid-{os.getpid()}-call-{call:04d}-input.npz"
            )
            if capture_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite VDAM host-replay capture {capture_path}"
                )
            shutil.copy2(input_path, capture_path)
        command = [
            sys.executable,
            str(helper),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--library",
            str(library),
            "--report",
            str(report_path),
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "external VDAM host replay failed with exit "
                f"{result.returncode}: stdout={result.stdout!r} stderr={result.stderr!r}"
            )
        report_dir_text = os.environ.get(
            _VDAM_EXTERNAL_HOST_REPLAY_REPORT_DIR_ENV, ""
        ).strip()
        if report_dir_text:
            report_dir = pathlib.Path(report_dir_text).expanduser().resolve()
            report_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                report_path,
                report_dir / f"pid-{os.getpid()}-call-{call:04d}.json",
            )
        with np.load(output_path, allow_pickle=False) as output:
            return (
                np.asarray(output["data_real_volume"], dtype=np.float32).reshape(
                    np.asarray(data_real_volume).shape
                ),
                np.asarray(output["data_imag_volume"], dtype=np.float32).reshape(
                    np.asarray(data_imag_volume).shape
                ),
                np.asarray(output["weight_volume"], dtype=np.float32).reshape(
                    np.asarray(weight_volume).shape
                ),
                np.asarray(output["denominator_sum"], dtype=np.float32),
            )


@functools.partial(
    jax.jit,
    static_argnums=(10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28, 32, 33),
)
def relion_vdam_mstep_fused_projector_x_half(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    images: jax.Array,
    ctf: jax.Array,
    minvsigma2: jax.Array,
    posterior_over_weight_norm: jax.Array,
    translation_angles: jax.Array,
    pixel_indices: jax.Array,
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float,
    projector_max_r: int,
    projection_padding_factor: int,
    reconstruction_group_ids: jax.Array | None = None,
    worker_lane_ids: jax.Array | None = None,
    particle_trace_ids: jax.Array | None = None,
    rotation_replay_order: jax.Array | None = None,
    rotation_replay_counts: jax.Array | None = None,
    particle_start_offsets_ns: jax.Array | None = None,
    serial_rotation_replay: bool = False,
    float64_accumulator_replay: bool = False,
    reverse_rotation_replay: bool = False,
    rotation_replay_stride: int = 0,
    native_trace_shape_replay: bool = False,
    parallel_worker_replay: bool | None = None,
    candidate_trace_active: bool = False,
    persistent_serial_rotation_replay: bool = False,
    stable_dense_positions: jax.Array | None = None,
    logical_current_size: jax.Array | int | None = None,
    runtime_projector_radius: jax.Array | None = None,
    return_denominator: bool = True,
    particle_tail_mask: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Project, form residuals, and scatter VDAM rows in one native launch.

    Explicit ``runtime_projector_radius`` opts into fixed projector storage:
    ``projector_full`` then holds a C64 [z,y,x>=0] capacity HALF slab with shape
    [pf*Q+3,pf*Q+3,pf*Q//2+2], centered on y/z, including the logical ghost
    planes (e.g. from ``prepare_relion_projector_capacity``). The static
    ``projector_max_r`` must be zero; the independent logical projector radius
    is an S32 scalar with 1 <= radius <= Q//2, validated by CUDA. Requires the
    existing stable image geometry operands and pf=1 or 2. Legacy replay/trace
    diagnostics remain on the original route.

    CUDA stages the original logical texture extent and origin, not a larger
    padded texture. It reads the radius with the existing particle metadata
    D2H synchronization, moved earlier only for this opt-in route. This changes
    preparation timing, not the subsequent particle/rotation/atomic program.

    ``return_denominator=False`` returns ``None`` as the third result and
    skips the separate terminal denominator kernel and its dense/compact
    storage. The scatter arithmetic and final stream synchronization remain
    unchanged. This requires a CUDA library supporting the empty-output ABI.

    ``particle_tail_mask=True`` permits a contiguous suffix of group IDs equal
    to -1. CUDA excludes these rows from the particle scatter loop after its
    existing metadata synchronization. The active prefix must be nonempty;
    interior negative IDs and other invalid active metadata are rejected.
    This opt-in requires grouped accumulators, no denominator, explicit serial
    particle scheduling, and no replay/trace or projector-capacity diagnostics.
    """

    if not isinstance(return_denominator, bool):
        raise TypeError("return_denominator must be a Python bool")
    if not isinstance(particle_tail_mask, bool):
        raise TypeError("particle_tail_mask must be a Python bool")
    if particle_tail_mask and (
        return_denominator or reconstruction_group_ids is None
        or parallel_worker_replay is not False
        or runtime_projector_radius is not None
        or rotation_replay_order is not None or rotation_replay_counts is not None
        or particle_start_offsets_ns is not None
        or serial_rotation_replay or persistent_serial_rotation_replay
        or float64_accumulator_replay or reverse_rotation_replay
        or rotation_replay_stride != 0 or native_trace_shape_replay
        or candidate_trace_active
    ):
        raise ValueError(
            "particle_tail_mask requires grouped accumulator-only work, "
            "parallel_worker_replay=False and no replay/trace or projector capacity"
        )
    if not return_denominator and os.environ.get(
        _VDAM_EXTERNAL_HOST_REPLAY_LIBRARY_ENV, ""
    ).strip():
        raise ValueError("external host replay requires the denominator output")
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(
            f"RELION fused VDAM projector requires an odd BPref grid, got {volume_shape}"
        )
    projector_capacity = runtime_projector_radius is not None
    if projector_capacity:
        # Validate before this body's local conversion. The public jitted
        # entry has already applied JAX argument canonicalization; use the
        # qualified x64 environment and explicit C64/S32 device operands.
        if getattr(projector_full, "dtype", None) != np.dtype("complex64"):
            raise ValueError("BPref projector capacity requires complex64 storage")
        if (
            getattr(runtime_projector_radius, "dtype", None) != np.dtype("int32")
            or getattr(runtime_projector_radius, "shape", None) != ()
        ):
            raise ValueError("runtime_projector_radius must be an S32 scalar")
    projector_full = jnp.asarray(projector_full)
    if projector_capacity:
        shape = projector_full.shape
        if projection_padding_factor not in (1, 2):
            raise ValueError("BPref projector capacity requires padding factor 1 or 2")
        if (
            projector_full.dtype != jnp.complex64 or len(shape) != 3
            or not 5 <= shape[0] <= 1025 or shape[0] != shape[1]
            or shape[0] % 2 != 1 or shape[2] != shape[0] // 2 + 1
            or (shape[0] - 3) % (2 * projection_padding_factor) != 0
        ):
            raise ValueError("BPref projector capacity requires a C64 centered half slab for even Q")
        if projector_max_r != 0:
            raise ValueError("BPref projector capacity requires static projector_max_r=0")
        runtime_projector_radius = jnp.asarray(runtime_projector_radius)
        if runtime_projector_radius.dtype != jnp.int32 or runtime_projector_radius.shape != ():
            raise ValueError("runtime_projector_radius must be an S32 scalar")
        if stable_dense_positions is None or logical_current_size is None:
            raise ValueError("BPref projector capacity requires stable image geometry")
        if (
            rotation_replay_order is not None or rotation_replay_counts is not None
            or particle_start_offsets_ns is not None or serial_rotation_replay
            or persistent_serial_rotation_replay or float64_accumulator_replay
            or reverse_rotation_replay or rotation_replay_stride
            or native_trace_shape_replay or candidate_trace_active
            or os.environ.get(_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY_ENV, "").strip()
        ):
            raise ValueError("BPref projector capacity does not support replay/trace diagnostics")
    elif (
        projector_full.dtype != jnp.complex64
        or projector_full.ndim != 3
        or projector_full.shape[0] <= 0
        or projector_full.shape[1:] != (projector_full.shape[0], projector_full.shape[0])
    ):
        raise TypeError(
            "projector_full must be a nonempty complex64 cube, got "
            f"{projector_full.shape} {projector_full.dtype}"
        )
    if (not projector_capacity and int(projector_max_r) <= 0) or int(projection_padding_factor) <= 0:
        raise ValueError("projector radius and projection padding factor must be positive")
    if images.dtype != jnp.complex64:
        raise TypeError("images must be complex64")
    if data_volume.dtype != jnp.complex64 or weight_volume.dtype != jnp.float32:
        raise TypeError("VDAM accumulators must be complex64/float32")
    for name, value in (
        ("ctf", ctf),
        ("minvsigma2", minvsigma2),
        ("posterior_over_weight_norm", posterior_over_weight_norm),
        ("translation_angles", translation_angles),
        ("rotation_matrices", rotation_matrices),
    ):
        if value.dtype != jnp.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
    if pixel_indices.dtype != jnp.int32:
        raise TypeError("pixel_indices must be int32")
    if images.ndim != 2 or ctf.shape != images.shape or minvsigma2.shape != images.shape:
        raise ValueError("images, ctf, and minvsigma2 must have matching shapes")
    if posterior_over_weight_norm.ndim != 3 or posterior_over_weight_norm.shape[0] != images.shape[0]:
        raise ValueError("posterior must have shape (particle,rotation,translation)")
    n_particles, n_rotations, n_translations = map(int, posterior_over_weight_norm.shape)
    if rotation_matrices.shape != (n_particles, n_rotations, 3, 3):
        raise ValueError("rotation_matrices must match particle/rotation axes")
    if translation_angles.shape != (n_translations, 2):
        raise ValueError("translation_angles must have shape (translation,2)")
    if pixel_indices.shape != (images.shape[1],):
        raise ValueError("pixel_indices must match the compact pixel dimension")
    stable_capacity = logical_current_size is not None or stable_dense_positions is not None
    if stable_capacity and (logical_current_size is None or stable_dense_positions is None):
        raise ValueError(
            "stable VDAM BPref storage requires both logical_current_size and "
            "stable_dense_positions"
        )
    if stable_capacity:
        physical_current_size = 2 * int(round(float(max_r)))
        logical_current_size = jnp.asarray(logical_current_size, dtype=jnp.int32)
        if logical_current_size.shape != ():
            raise ValueError(
                "stable VDAM BPref logical_current_size must be an S32 scalar, "
                f"got {logical_current_size.shape}"
            )
        stable_dense_positions = jnp.asarray(stable_dense_positions, dtype=jnp.int32)
        if stable_dense_positions.shape != pixel_indices.shape:
            raise ValueError(
                "stable_dense_positions must match the compact pixel dimension"
            )
    if reconstruction_group_ids is None:
        if data_volume.ndim != 1 or weight_volume.ndim != 1:
            raise ValueError("ungrouped VDAM accumulators must be rank 1")
        reconstruction_group_ids = jnp.zeros((n_particles,), dtype=jnp.int32)
        reconstruction_group_count = 1
    else:
        reconstruction_group_ids = jnp.asarray(reconstruction_group_ids)
        if reconstruction_group_ids.dtype != jnp.int32:
            raise TypeError("reconstruction_group_ids must be int32")
        if reconstruction_group_ids.shape != (n_particles,):
            raise ValueError("reconstruction_group_ids must match the particle axis")
        if data_volume.ndim != 2 or weight_volume.ndim != 2:
            raise ValueError("grouped VDAM accumulators must be rank 2")
        if data_volume.shape != weight_volume.shape or data_volume.shape[0] <= 0:
            raise ValueError("grouped VDAM accumulators must have matching nonempty shapes")
        reconstruction_group_count = int(data_volume.shape[0])
    if particle_tail_mask and reconstruction_group_count <= 1:
        raise ValueError("particle_tail_mask requires at least two accumulator groups")
    captured_worker_lanes = worker_lane_ids is not None
    if parallel_worker_replay is None:
        parallel_worker_replay = captured_worker_lanes
    else:
        parallel_worker_replay = bool(parallel_worker_replay)
    if worker_lane_ids is None:
        worker_lane_ids = jnp.arange(n_particles, dtype=jnp.int32) % 8
    else:
        worker_lane_ids = jnp.asarray(worker_lane_ids)
        if worker_lane_ids.dtype != jnp.int32:
            raise TypeError("worker_lane_ids must be int32")
        if worker_lane_ids.shape != (n_particles,):
            raise ValueError("worker_lane_ids must match the particle axis")
    if particle_trace_ids is None:
        particle_trace_ids = jnp.arange(n_particles, dtype=jnp.int32)
    else:
        particle_trace_ids = jnp.asarray(particle_trace_ids)
        if particle_trace_ids.dtype != jnp.int32:
            raise TypeError("particle_trace_ids must be int32")
        if particle_trace_ids.shape != (n_particles,):
            raise ValueError("particle_trace_ids must match the particle axis")
    captured_rotation_replay = rotation_replay_order is not None
    if rotation_replay_order is None:
        rotation_replay_order = jnp.broadcast_to(
            jnp.arange(n_rotations, dtype=jnp.int32),
            (n_particles, n_rotations),
        )
    else:
        rotation_replay_order = jnp.asarray(rotation_replay_order)
        if rotation_replay_order.dtype != jnp.int32:
            raise TypeError("rotation_replay_order must be int32")
        if rotation_replay_order.shape != (n_particles, n_rotations):
            raise ValueError("rotation_replay_order must match particle/rotation axes")
    if rotation_replay_counts is None:
        rotation_replay_counts = jnp.full(
            (n_particles,),
            n_rotations,
            dtype=jnp.int32,
        )
    else:
        rotation_replay_counts = jnp.asarray(rotation_replay_counts)
        if rotation_replay_counts.dtype != jnp.int32:
            raise TypeError("rotation_replay_counts must be int32")
        if rotation_replay_counts.shape != (n_particles,):
            raise ValueError("rotation_replay_counts must match the particle axis")
    captured_particle_timing_replay = particle_start_offsets_ns is not None
    if particle_start_offsets_ns is None:
        particle_start_offsets_ns = jnp.zeros((n_particles,), dtype=jnp.int32)
    else:
        particle_start_offsets_ns = jnp.asarray(particle_start_offsets_ns)
        if particle_start_offsets_ns.dtype != jnp.int32:
            raise TypeError("particle_start_offsets_ns must be int32")
        if particle_start_offsets_ns.shape != (n_particles,):
            raise ValueError("particle_start_offsets_ns must match the particle axis")
    if persistent_serial_rotation_replay and not serial_rotation_replay:
        raise ValueError(
            "persistent serial VDAM rotations require serial_rotation_replay"
        )
    if projector_capacity:
        _ensure_bpref_projector_capacity_ffi()
    else:
        _ensure_ffi()

    if stable_capacity:
        physical_current_size = 2 * int(round(float(max_r)))
        pixel_capacity = physical_current_size * (physical_current_size // 2 + 1)

        def _pack_stable_dense(values):
            dense = jnp.zeros((*values.shape[:-1], pixel_capacity), dtype=values.dtype)
            return dense.at[..., stable_dense_positions].set(values)

        dense_images = _pack_stable_dense(images)
        dense_ctf = _pack_stable_dense(ctf)
        dense_minvsigma2 = _pack_stable_dense(minvsigma2)
        dense_indices = stable_dense_positions
        ctf_indices = stable_dense_positions
        noise_indices = stable_dense_positions
        current_h = physical_current_size
        current_w = current_h // 2 + 1
        ctf_h = noise_h = current_h
        ctf_w = noise_w = current_w
    else:
        dense_images, dense_indices, current_h, current_w = _prepare_relion_x_half_block_topology_operands(
            images, pixel_indices, image_shape, max_r
        )
        dense_ctf, ctf_indices, ctf_h, ctf_w = _prepare_relion_x_half_block_topology_operands(
            ctf, pixel_indices, image_shape, max_r
        )
        dense_minvsigma2, noise_indices, noise_h, noise_w = _prepare_relion_x_half_block_topology_operands(
            minvsigma2, pixel_indices, image_shape, max_r
        )
        pixel_capacity = current_h * current_w
    if (ctf_h, ctf_w) != (current_h, current_w) or (noise_h, noise_w) != (current_h, current_w):
        raise ValueError("VDAM fused-projector operand topology metadata mismatch")
    if not (ctf_indices.shape == noise_indices.shape == dense_indices.shape):
        raise ValueError("VDAM fused-projector dense pixel topology mismatch")

    flat_rotations = rotation_matrices.reshape(n_particles * n_rotations, 3, 3)
    eulers = jnp.swapaxes(flat_rotations, -1, -2).reshape(
        n_particles, n_rotations, 9
    )
    kernel_rotations = _relion_x_half_backproject_rotation_to_kernel(
        flat_rotations, jnp.float32
    )
    rot6 = _rot_to_compact(kernel_rotations, jnp.float32).reshape(
        n_particles, n_rotations, 6
    )
    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    denominator_type = jax.ShapeDtypeStruct(
        (n_particles, n_rotations, pixel_capacity) if return_denominator else (0,),
        jnp.float32,
    )
    data_real_volume = jnp.asarray(data_volume.real, dtype=jnp.float32)
    data_imag_volume = jnp.asarray(data_volume.imag, dtype=jnp.float32)
    output_types = (
        jax.ShapeDtypeStruct(data_real_volume.shape, jnp.float32),
        jax.ShapeDtypeStruct(data_imag_volume.shape, jnp.float32),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
        denominator_type,
    )
    external_host_replay = bool(
        os.environ.get(_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY_ENV, "").strip()
    )
    if external_host_replay:
        if stable_capacity:
            raise ValueError(
                "stable VDAM BPref storage is unsupported by external host replay"
            )
        if (
            captured_rotation_replay
            or serial_rotation_replay
            or persistent_serial_rotation_replay
            or float64_accumulator_replay
            or reverse_rotation_replay
            or rotation_replay_stride
            or native_trace_shape_replay
            or captured_particle_timing_replay
            or candidate_trace_active
        ):
            raise ValueError(
                "external exact-native host replay cannot mix with rotation, "
                "timing, trace, or float64 diagnostics"
            )
        callback = functools.partial(
            _run_vdam_external_host_replay_callback,
            image_h=current_h,
            image_w=current_w,
            volume_shape=tuple(map(int, volume_shape)),
            upsampling=int(kw["upsampling"]),
            max_r2_x4=int(kw["max_r2_x4"]),
            physical_image_size=int(image_shape[0]),
            projector_max_r=int(projector_max_r),
            projection_padding_factor=int(projection_padding_factor),
            reconstruction_group_count=int(reconstruction_group_count),
            parallel_worker_replay=bool(parallel_worker_replay),
        )
        fused_real, fused_imag, fused_weight, dense_denominator = (
            jax.experimental.io_callback(
                callback,
                output_types,
                projector_full,
                dense_images,
                dense_ctf,
                dense_minvsigma2,
                posterior_over_weight_norm,
                translation_angles,
                eulers,
                rot6,
                reconstruction_group_ids,
                worker_lane_ids,
                particle_trace_ids,
                rotation_replay_order,
                rotation_replay_counts,
                particle_start_offsets_ns,
                data_real_volume,
                data_imag_volume,
                weight_volume,
                ordered=True,
            )
        )
    else:
        target = (
            _TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_CAPACITY_X_HALF
            if projector_capacity
            else _TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_RUNTIME_X_HALF
            if stable_capacity
            else _TARGET_RELION_VDAM_MSTEP_FUSED_PROJECTOR_X_HALF
        )
        operands = (
            projector_full,
            dense_images,
            dense_ctf,
            dense_minvsigma2,
            posterior_over_weight_norm,
            translation_angles,
            eulers,
            rot6,
            reconstruction_group_ids,
            worker_lane_ids,
            particle_trace_ids,
            rotation_replay_order,
            rotation_replay_counts,
            particle_start_offsets_ns,
            data_real_volume,
            data_imag_volume,
            weight_volume,
        )
        if stable_capacity:
            operands += (logical_current_size,)
        if projector_capacity:
            operands += (runtime_projector_radius,)
        fused_real, fused_imag, fused_weight, dense_denominator = jax.ffi.ffi_call(
            target,
            output_types,
            input_output_aliases={14: 0, 15: 1, 16: 2},
            vmap_method="sequential",
        )(
            *operands,
            image_h=np.int64(current_h),
            image_w=np.int64(current_w),
            pixel_capacity=np.int64(pixel_capacity),
            N0=kw["N0"],
            N1=kw["N1"],
            N2=kw["N2"],
            upsampling=kw["upsampling"],
            max_r2_x4=kw["max_r2_x4"],
            physical_image_size=np.int64(image_shape[0]),
            projector_max_r=np.int64(projector_max_r),
            projection_padding_factor=np.int64(projection_padding_factor),
            reconstruction_group_count=np.int64(reconstruction_group_count),
            parallel_worker_replay=np.int64(parallel_worker_replay),
            captured_rotation_replay=np.int64(captured_rotation_replay),
            serial_rotation_replay=np.int64(
                2 if persistent_serial_rotation_replay else serial_rotation_replay
            ),
            float64_accumulator_replay=np.int64(float64_accumulator_replay),
            reverse_rotation_replay=np.int64(reverse_rotation_replay),
            rotation_replay_stride=np.int64(rotation_replay_stride),
            native_trace_shape_replay=np.int64(native_trace_shape_replay),
            captured_particle_timing_replay=np.int64(captured_particle_timing_replay),
            candidate_trace_active=np.int64(candidate_trace_active),
            particle_tail_mask=np.int64(particle_tail_mask),
        )
    fused_data = jax.lax.complex(fused_real, fused_imag)
    if not return_denominator:
        return fused_data, fused_weight, None
    if stable_capacity:
        compact_denominator = jnp.take(
            dense_denominator,
            stable_dense_positions,
            axis=-1,
        )
    else:
        full_h, full_w = map(int, image_shape)
        full_half_w = full_w // 2 + 1
        full_rows = pixel_indices // full_half_w
        columns = pixel_indices % full_half_w
        signed_rows = jnp.where(full_rows <= full_h // 2, full_rows, full_rows - full_h)
        current_indices = jnp.mod(signed_rows, current_h) * current_w + columns
        compact_denominator = jnp.take(dense_denominator, current_indices, axis=-1)
    return fused_data, fused_weight, compact_denominator


def _validate_relion_fine_diff2_inputs(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    *,
    real_dtype=jnp.float32,
) -> None:
    real_dtype = jnp.dtype(real_dtype)
    complex_dtype = jnp.dtype(jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64)
    if reference.dtype != complex_dtype:
        raise TypeError(f"reference must be {complex_dtype.name}, got {reference.dtype}")
    if shifted_image.dtype != complex_dtype:
        raise TypeError(f"shifted_image must be {complex_dtype.name}, got {shifted_image.dtype}")
    if weight.dtype != real_dtype:
        raise TypeError(f"weight must be {real_dtype.name}, got {weight.dtype}")
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            f"full_to_compact must be int32, got {full_to_compact.dtype}"
        )
    if full_to_compact.ndim != 1 or full_to_compact.shape[0] <= 0:
        raise ValueError(
            "full_to_compact must be a nonempty rank-1 array, got "
            f"{full_to_compact.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION fine diff2 requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION fine diff2 was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()


@jax.jit
def relion_coarse_diff2_rectangular_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate RELION's coarse Gaussian CUDA reduction topology.

    Shapes are ``reference=(R,N)``, ``shifted_image=(B,T,N)``,
    ``weight=(B,N)``, ``initial_diff2=(B,)``, and
    ``full_to_compact=(F,)``. The kernel uses RELION's 128-thread,
    16-orientation blocks and its native per-translation lane assignment. The
    lane partials are combined with CUDA atomics on top of the supplied
    high-resolution image term, as in RELION's coarse scorer.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    if initial_diff2.dtype != jnp.float32:
        raise TypeError(
            f"initial_diff2 must be float32, got {initial_diff2.dtype}"
        )
    if reference.ndim != 2 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "rectangular coarse diff2 expects reference rank 2, shifted rank "
            f"3, and weight rank 2, got {reference.shape}, "
            f"{shifted_image.shape}, {weight.shape}"
        )
    if (
        shifted_image.shape[0] != weight.shape[0]
        or reference.shape[1] != shifted_image.shape[2]
        or reference.shape[1] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[1] > 128
        or initial_diff2.shape != (shifted_image.shape[0],)
    ):
        raise ValueError(
            "rectangular coarse diff2 operands have inconsistent shapes or "
            "more than 128 translations: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    out_type = jax.ShapeDtypeStruct(
        (shifted_image.shape[0], reference.shape[0], shifted_image.shape[1]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact)


def _pack_runtime_logical_prefix_rows(
    values: jax.Array,
    logical_count: jax.Array,
) -> jax.Array:
    """Pack fixed-capacity rows at their dynamic logical stride.

    The coarse CUDA kernel's native atomic admission order depends on operand
    memory timing.  Leaving capacity padding between logical rows can move a
    legal atomic sum by a few float32 ULPs.  Pack each logical row contiguously
    at the front of the same fixed-size buffer so the runtime kernel observes
    the exact compact strides of the unpadded scorer without changing its XLA
    shape.
    """

    physical_count = values.shape[-1]
    flat = values.reshape(-1)
    flat_index = jnp.arange(flat.size, dtype=jnp.int32)
    safe_count = jnp.maximum(logical_count, jnp.int32(1))
    logical_total = jnp.int32(flat.size // physical_count) * logical_count
    logical_row = flat_index // safe_count
    logical_column = flat_index - logical_row * safe_count
    source = logical_row * jnp.int32(physical_count) + logical_column
    source = jnp.minimum(source, jnp.int32(flat.size - 1))
    packed = jnp.where(
        flat_index < logical_total,
        flat[source],
        jnp.zeros((), dtype=values.dtype),
    )
    return packed.reshape(values.shape)


def _validate_coarse_rectangular_runtime_inputs(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    logical_full_pixel_count: jax.Array,
) -> jax.Array:
    """Shared operand contract for packed and physical-stride coarse scoring."""

    logical_full_pixel_count = jnp.asarray(
        logical_full_pixel_count,
        dtype=jnp.int32,
    )
    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    if initial_diff2.dtype != jnp.float32:
        raise TypeError(
            f"initial_diff2 must be float32, got {initial_diff2.dtype}"
        )
    if logical_full_pixel_count.shape != ():
        raise ValueError("logical_full_pixel_count must be an int32 scalar")
    if reference.ndim != 2 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "runtime rectangular coarse diff2 expects reference rank 2, "
            f"shifted rank 3, and weight rank 2, got {reference.shape}, "
            f"{shifted_image.shape}, {weight.shape}"
        )
    if (
        shifted_image.shape[0] != weight.shape[0]
        or reference.shape[1] != shifted_image.shape[2]
        or reference.shape[1] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[1] > 128
        or initial_diff2.shape != (shifted_image.shape[0],)
    ):
        raise ValueError(
            "runtime rectangular coarse diff2 operands have inconsistent "
            f"shapes or more than 128 translations: {reference.shape}, "
            f"{shifted_image.shape}, {weight.shape}"
        )
    return logical_full_pixel_count


@jax.jit
def relion_coarse_diff2_rectangular_runtime_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    logical_full_pixel_count: jax.Array,
) -> jax.Array:
    """Evaluate a stable-capacity coarse table over its logical pixel prefix.

    The array shapes and lookup capacity remain static while the scalar logical
    count limits the CUDA traversal. This preserves the RELION lane assignment
    and avoids executing zero-weight capacity rows.
    """
    logical_full_pixel_count = _validate_coarse_rectangular_runtime_inputs(
        reference, shifted_image, weight, initial_diff2,
        full_to_compact, logical_full_pixel_count,
    )
    out_type = jax.ShapeDtypeStruct(
        (shifted_image.shape[0], reference.shape[0], shifted_image.shape[1]),
        jnp.float32,
    )
    reference = _pack_runtime_logical_prefix_rows(
        reference,
        logical_full_pixel_count,
    )
    shifted_image = _pack_runtime_logical_prefix_rows(
        shifted_image,
        logical_full_pixel_count,
    )
    weight = _pack_runtime_logical_prefix_rows(
        weight,
        logical_full_pixel_count,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_RECTANGULAR_RUNTIME_F32,
        out_type,
        vmap_method="sequential",
    )(
        reference,
        shifted_image,
        weight,
        initial_diff2,
        full_to_compact,
        logical_full_pixel_count,
    )


_TARGET_RELION_COARSE_SHARED_PRETRANSLATED_RUNTIME_F32 = (
    "cuda_relion_coarse_diff2_shared_pretranslated_runtime_f32"
)

_TARGET_RELION_COARSE_POSTERIOR_TRANSACTION_F32 = "cuda_relion_coarse_posterior_transaction_f32"
_coarse_posterior_transaction_registered = False


def _validate_coarse_posterior_transaction(scores, raw_max, actual_count, adaptive_fraction, max_significants):
    if scores.dtype != jnp.float32 or scores.ndim != 2:
        raise TypeError("posterior scores must be a float32 matrix")
    if min(scores.shape) < 1 or math.prod(scores.shape) > np.iinfo(np.int32).max:
        raise ValueError("posterior score capacity must be nonempty and fit int32 support positions")
    if raw_max.dtype != jnp.float32 or raw_max.shape != scores.shape[:1]:
        raise TypeError("raw maxima must be float32 with one value per score row")
    if actual_count.dtype != jnp.int32 or actual_count.ndim != 0:
        raise TypeError("actual row count must be an int32 device scalar")
    fraction = float(adaptive_fraction)
    if not np.isfinite(fraction) or not 0 < fraction <= 1 or np.float32(fraction) <= 0:
        raise ValueError("adaptive_fraction must be finite in (0, 1]")
    maximum = 0 if max_significants is None else operator.index(max_significants)
    if maximum < 0 or maximum > np.iinfo(np.int32).max:
        raise ValueError("max_significants must be None or a nonnegative int32")
    return np.float32(fraction), np.int64(maximum)


@functools.partial(jax.jit, static_argnames=("adaptive_fraction", "max_significants"))
def relion_coarse_posterior_transaction_f32(
    scores, raw_max, actual_count, *, adaptive_fraction=0.999, max_significants=500,
):
    """Unused exact-threshold posterior/support transaction for dense or compact rows.

    Inputs already include priors. Outputs are float32 [B,4] statistics
    (best score, Pmax, sum weight, threshold), int32 [B,4] indices/counts
    (best local pose, winning local pose, significant count, cutoff count),
    flat row-major support positions with capacity B*N, and an int32 count.
    Only the first count support entries are populated; remaining entries are
    -1. No max-significant truncation is applied to threshold ties. Source-block
    mapping and unchanged float64 logZ are the caller's responsibility.
    """
    fraction, maximum = _validate_coarse_posterior_transaction(
        scores, raw_max, actual_count, adaptive_fraction, max_significants,
    )
    if jax.default_backend() != "gpu" or not custom_cuda_requested():
        raise RuntimeError("Coarse posterior transaction requires explicit custom CUDA on GPU")
    global _coarse_posterior_transaction_registered
    _ensure_ffi()
    with _ffi_lock:
        if not _coarse_posterior_transaction_registered:
            symbol = getattr(_get_lib(), "RelionCoarsePosteriorTransactionF32")
            jax.ffi.register_ffi_target(
                _TARGET_RELION_COARSE_POSTERIOR_TRANSACTION_F32,
                jax.ffi.pycapsule(symbol), platform="CUDA",
            )
            _coarse_posterior_transaction_registered = True
    batch = scores.shape[0]
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_POSTERIOR_TRANSACTION_F32,
        (jax.ShapeDtypeStruct((batch, 4), jnp.float32),
         jax.ShapeDtypeStruct((batch, 4), jnp.int32),
         jax.ShapeDtypeStruct((math.prod(scores.shape),), jnp.int32),
         jax.ShapeDtypeStruct((), jnp.int32)),
    )(scores, raw_max, actual_count, fraction=fraction, maxsig=maximum)


_coarse_shared_pretranslated_registered = False


def _ensure_coarse_shared_pretranslated_ffi():
    global _coarse_shared_pretranslated_registered
    _ensure_ffi()
    with _ffi_lock:
        if not _coarse_shared_pretranslated_registered:
            symbol = getattr(
                _get_lib(), "RelionCoarseDiff2SharedPretranslatedRuntimeF32",
            )
            jax.ffi.register_ffi_target(
                _TARGET_RELION_COARSE_SHARED_PRETRANSLATED_RUNTIME_F32,
                jax.ffi.pycapsule(symbol), platform="CUDA",
            )
            _coarse_shared_pretranslated_registered = True


@jax.jit
def relion_coarse_diff2_shared_pretranslated_runtime_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    logical_full_pixel_count: jax.Array,
) -> jax.Array:
    """Reuse projection tiles while scoring existing translated image batches.

    Arrays retain their physical row strides; the device scalar limits logical
    traversal without prefix-packing copies. CUDA preserves the source-16,
    128-thread lane order and atomic accumulation. This optional entry point
    does not change production dispatch.
    """
    logical_full_pixel_count = _validate_coarse_rectangular_runtime_inputs(
        reference, shifted_image, weight, initial_diff2,
        full_to_compact, logical_full_pixel_count,
    )
    _ensure_coarse_shared_pretranslated_ffi()
    out_type = jax.ShapeDtypeStruct(
        (shifted_image.shape[0], reference.shape[0], shifted_image.shape[1]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_SHARED_PRETRANSLATED_RUNTIME_F32,
        out_type, vmap_method="sequential",
    )(
        reference, shifted_image, weight, initial_diff2,
        full_to_compact, logical_full_pixel_count,
    )


@jax.jit
def relion_coarse_diff2_rotation_blocks_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    rotation_block_ids: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Rescore selected native RELION coarse-rotation blocks.

    Shapes are ``reference=(R,N)``, ``shifted_image=(B,T,N)``,
    ``weight=(B,N)``, ``initial_diff2=(B,)``,
    ``rotation_block_ids=(B,Q)``, and ``full_to_compact=(F,)``. Each block ID
    addresses the same aligned group of 16 rotations used by the rectangular
    coarse scorer. The output has shape ``(B,Q,16,T)``. A negative or
    out-of-range block ID other than the reserved ``-1`` padding value emits
    ``NaN`` so callers fail closed; ``-1`` and rotations beyond a valid final
    partial block are padded with ``+inf`` and cannot become selectable. A
    partial final block intentionally follows the existing rectangular
    16-register path; production promotion still requires a separately gated
    one-Euler RELION-tail implementation.
    """

    if initial_diff2.dtype != jnp.float32:
        raise TypeError(
            f"initial_diff2 must be float32, got {initial_diff2.dtype}"
        )
    if rotation_block_ids.dtype != jnp.int32:
        raise TypeError(
            "rotation_block_ids must be int32, got "
            f"{rotation_block_ids.dtype}"
        )
    if (
        reference.ndim != 2
        or shifted_image.ndim != 3
        or weight.ndim != 2
        or rotation_block_ids.ndim != 2
    ):
        raise ValueError(
            "rotation-block coarse diff2 expects reference/weight/block IDs "
            "at ranks 2/2/2 and shifted image at rank 3, got "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}, "
            f"{rotation_block_ids.shape}"
        )
    if (
        shifted_image.shape[0] != weight.shape[0]
        or rotation_block_ids.shape[0] != shifted_image.shape[0]
        or reference.shape[1] != shifted_image.shape[2]
        or reference.shape[1] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[1] > 128
        or rotation_block_ids.shape[1] <= 0
        or initial_diff2.shape != (shifted_image.shape[0],)
    ):
        raise ValueError(
            "rotation-block coarse diff2 operands have inconsistent shapes "
            "or more than 128 translations: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}, "
            f"{initial_diff2.shape}, {rotation_block_ids.shape}"
        )
    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    out_type = jax.ShapeDtypeStruct(
        (
            shifted_image.shape[0],
            rotation_block_ids.shape[1],
            16,
            shifted_image.shape[1],
        ),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_ROTATION_BLOCKS_F32,
        out_type,
        vmap_method="sequential",
    )(
        reference,
        shifted_image,
        weight,
        initial_diff2,
        rotation_block_ids,
        full_to_compact,
    )


@jax.jit
def relion_coarse_diff2_rotation_blocks_runtime_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    rotation_block_ids: jax.Array,
    full_to_compact: jax.Array,
    logical_full_pixel_count: jax.Array,
) -> jax.Array:
    """Rescore source-16 blocks over a dynamic logical pixel prefix."""

    logical_full_pixel_count = jnp.asarray(
        logical_full_pixel_count,
        dtype=jnp.int32,
    )
    if initial_diff2.dtype != jnp.float32:
        raise TypeError(
            f"initial_diff2 must be float32, got {initial_diff2.dtype}"
        )
    if rotation_block_ids.dtype != jnp.int32:
        raise TypeError(
            "rotation_block_ids must be int32, got "
            f"{rotation_block_ids.dtype}"
        )
    if logical_full_pixel_count.shape != ():
        raise ValueError("logical_full_pixel_count must be an int32 scalar")
    if (
        reference.ndim != 2
        or shifted_image.ndim != 3
        or weight.ndim != 2
        or rotation_block_ids.ndim != 2
    ):
        raise ValueError(
            "runtime rotation-block coarse diff2 operands have invalid ranks"
        )
    if (
        shifted_image.shape[0] != weight.shape[0]
        or rotation_block_ids.shape[0] != shifted_image.shape[0]
        or reference.shape[1] != shifted_image.shape[2]
        or reference.shape[1] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[1] > 128
        or rotation_block_ids.shape[1] <= 0
        or initial_diff2.shape != (shifted_image.shape[0],)
    ):
        raise ValueError(
            "runtime rotation-block coarse diff2 operands have inconsistent "
            f"shapes or counts: {reference.shape}, {shifted_image.shape}, "
            f"{weight.shape}, {initial_diff2.shape}, {rotation_block_ids.shape}"
        )
    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    out_type = jax.ShapeDtypeStruct(
        (
            shifted_image.shape[0],
            rotation_block_ids.shape[1],
            16,
            shifted_image.shape[1],
        ),
        jnp.float32,
    )
    reference = _pack_runtime_logical_prefix_rows(
        reference,
        logical_full_pixel_count,
    )
    shifted_image = _pack_runtime_logical_prefix_rows(
        shifted_image,
        logical_full_pixel_count,
    )
    weight = _pack_runtime_logical_prefix_rows(
        weight,
        logical_full_pixel_count,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_ROTATION_BLOCKS_RUNTIME_F32,
        out_type,
        vmap_method="sequential",
    )(
        reference,
        shifted_image,
        weight,
        initial_diff2,
        rotation_block_ids,
        full_to_compact,
        logical_full_pixel_count,
    )


@jax.jit
def relion_coarse_normalized_cc_pairs_f32(
    shifted_image: jax.Array,
    score_weight: jax.Array,
    reference: jax.Array,
    half_weights: jax.Array,
    packed_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate bounded coarse normalized-CC pairs with RELION CUDA arithmetic.

    The first three operands have shape ``(B,C,N)`` and the output has shape
    ``(B,C)``. One 128-thread CUDA block evaluates each candidate, preserving
    RELION's operand contraction, lane tree, ``sqrtf``, and 128 identical
    atomic additions. ``packed_to_compact`` restores current-size FFTW order.
    """

    for name, value, dtype in (
        ("shifted_image", shifted_image, jnp.complex64),
        ("score_weight", score_weight, jnp.float32),
        ("reference", reference, jnp.complex64),
        ("half_weights", half_weights, jnp.float32),
        ("packed_to_compact", packed_to_compact, jnp.int32),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
    if (
        shifted_image.ndim != 3
        or score_weight.shape != shifted_image.shape
        or reference.shape != shifted_image.shape
        or half_weights.shape != (shifted_image.shape[-1],)
        or packed_to_compact.ndim != 1
        or packed_to_compact.shape[0] <= 0
        or shifted_image.shape[0] <= 0
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[2] <= 0
    ):
        raise ValueError(
            "RELION coarse normalized-CC pairs have inconsistent shapes: "
            f"{shifted_image.shape}, {score_weight.shape}, {reference.shape}, "
            f"{half_weights.shape}, {packed_to_compact.shape}",
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION coarse normalized-CC pairs require a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION coarse normalized-CC pairs require the custom CUDA extension",
        )
    _ensure_ffi()
    out_type = jax.ShapeDtypeStruct(shifted_image.shape[:-1], jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_NORMALIZED_CC_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(
        shifted_image,
        score_weight,
        reference,
        half_weights,
        packed_to_compact,
    )


@functools.partial(jax.jit, static_argnums=(6, 7, 8, 9))
def relion_coarse_normalized_cc_native_texture_pairs_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    shifted_image: jax.Array,
    score_weight: jax.Array,
    half_weights: jax.Array,
    packed_to_compact: jax.Array,
    current_size: int,
    padding_factor: int,
    projector_max_r: int,
    return_components: bool = False,
    translation_angles: jax.Array | None = None,
    numerator_weight: jax.Array | None = None,
) -> jax.Array:
    """Evaluate bounded normalized-CC pairs from RELION's CUDA texture.

    Each row is one candidate. Its projection is sampled from
    ``projector_full`` inside the same 128-thread CUDA block that contracts the
    normalized-CC numerator and reference norm. When ``translation_angles`` is
    supplied, the image is translated in that same block with RELION's
    coarse-kernel ``sincosf`` arithmetic. This avoids materializing a RECOVAR
    projection or translated image at a different arithmetic boundary.
    """

    projector_full = jnp.asarray(projector_full)
    rotation_matrices = jnp.asarray(rotation_matrices)
    shifted_image = jnp.asarray(shifted_image)
    score_weight = jnp.asarray(score_weight)
    half_weights = jnp.asarray(half_weights)
    packed_to_compact = jnp.asarray(packed_to_compact)
    if translation_angles is None:
        translation_angles = jnp.zeros(
            (shifted_image.shape[0], 2),
            dtype=jnp.float32,
        )
    else:
        translation_angles = jnp.asarray(translation_angles)
    if numerator_weight is None:
        numerator_weight = jnp.ones_like(score_weight, dtype=jnp.float32)
    else:
        numerator_weight = jnp.asarray(numerator_weight)
    for name, value, dtype in (
        ("projector_full", projector_full, jnp.complex64),
        ("rotation_matrices", rotation_matrices, jnp.float32),
        ("shifted_image", shifted_image, jnp.complex64),
        ("score_weight", score_weight, jnp.float32),
        ("half_weights", half_weights, jnp.float32),
        ("packed_to_compact", packed_to_compact, jnp.int32),
        ("translation_angles", translation_angles, jnp.float32),
        ("numerator_weight", numerator_weight, jnp.float32),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
    if (
        projector_full.ndim != 3
        or projector_full.shape[0] <= 0
        or projector_full.shape[1:]
        != (projector_full.shape[0], projector_full.shape[0])
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[1:] != (3, 3)
        or rotation_matrices.shape[0] <= 0
        or shifted_image.ndim != 2
        or shifted_image.shape[0] != rotation_matrices.shape[0]
        or shifted_image.shape[1] <= 0
        or score_weight.shape != shifted_image.shape
        or translation_angles.shape != (shifted_image.shape[0], 2)
        or numerator_weight.shape != shifted_image.shape
        or half_weights.shape != (shifted_image.shape[1],)
        or packed_to_compact.shape
        != (int(current_size) * (int(current_size) // 2 + 1),)
        or int(current_size) <= 0
        or int(padding_factor) <= 0
        or int(projector_max_r) <= 0
    ):
        raise ValueError(
            "native texture normalized-CC pair operands have inconsistent shapes: "
            f"projector={projector_full.shape}, rotations={rotation_matrices.shape}, "
            f"image={shifted_image.shape}, weight={score_weight.shape}, "
            f"half_weights={half_weights.shape}, lookup={packed_to_compact.shape}",
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "RELION native texture normalized-CC pairs require a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION native texture normalized-CC pairs require the custom CUDA extension"
        )
    _ensure_ffi()
    eulers = jnp.swapaxes(rotation_matrices, -1, -2).reshape(
        rotation_matrices.shape[0], 9
    )
    out_type = jax.ShapeDtypeStruct((shifted_image.shape[0], 3), jnp.float32)
    components = jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_NORMALIZED_CC_NATIVE_TEXTURE_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        eulers,
        shifted_image,
        translation_angles,
        score_weight,
        numerator_weight,
        half_weights,
        packed_to_compact,
        current_size=np.int64(current_size),
        padding_factor=np.int64(padding_factor),
        projector_max_r=np.int64(projector_max_r),
    )
    return components if return_components else components[:, 0]


def _prepare_relion_coarse_diff2_projector_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
) -> tuple[jax.Array, jax.ShapeDtypeStruct]:
    """Validate and prepare the one shared fused coarse-projector ABI."""
    if projector_full.dtype != jnp.complex64 or projector_full.ndim != 3:
        raise TypeError(
            "projector_full must be a rank-3 complex64 array, got "
            f"{projector_full.shape} {projector_full.dtype}"
        )
    if images.dtype != jnp.complex64 or images.ndim != 2:
        raise TypeError(
            f"images must be rank-2 complex64, got {images.shape} {images.dtype}"
        )
    if rotation_matrices.dtype != jnp.float32 or rotation_matrices.ndim != 3:
        raise TypeError(
            "rotation_matrices must be rank-3 float32, got "
            f"{rotation_matrices.shape} {rotation_matrices.dtype}"
        )
    if rotation_matrices.shape[1:] != (3, 3):
        raise ValueError(
            f"rotation_matrices must end in (3, 3), got {rotation_matrices.shape}"
        )
    if translation_angles.dtype != jnp.float32 or translation_angles.ndim != 2:
        raise TypeError(
            "translation_angles must be rank-2 float32, got "
            f"{translation_angles.shape} {translation_angles.dtype}"
        )
    if translation_angles.shape[1] != 2 or translation_angles.shape[0] > 128:
        raise ValueError(
            "translation_angles must have shape (T, 2) with T <= 128, got "
            f"{translation_angles.shape}"
        )
    if weight.dtype != jnp.float32 or weight.shape != images.shape:
        raise TypeError(
            f"weight must be float32 with shape {images.shape}, got {weight.shape} {weight.dtype}"
        )
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (images.shape[0],):
        raise TypeError(
            "initial_diff2 must be float32 with one value per image, got "
            f"{initial_diff2.shape} {initial_diff2.dtype}"
        )
    if full_to_compact.dtype != jnp.int32 or full_to_compact.shape != (
        int(current_size) * (int(current_size) // 2 + 1),
    ):
        raise TypeError(
            "full_to_compact must be the current-size packed int32 lookup, got "
            f"{full_to_compact.shape} {full_to_compact.dtype}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION fused coarse projector requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION fused coarse projector was requested but custom CUDA is disabled"
        )
    _ensure_ffi()
    compact_rotations = _rot_to_compact(rotation_matrices, jnp.float32)
    out_type = jax.ShapeDtypeStruct(
        (images.shape[0], rotation_matrices.shape[0], translation_angles.shape[0]),
        jnp.float32,
    )
    return compact_rotations, out_type


def _validate_relion_coarse_single_lane_canonical(
    translation_count: int,
    *,
    canonical_reduction: bool,
    single_lane_canonical: bool,
) -> None:
    """Fail closed unless one CUDA thread owns each coarse translation."""

    if not single_lane_canonical:
        return
    if not canonical_reduction:
        raise ValueError(
            "single_lane_canonical=True requires canonical_reduction=True",
        )
    if not 65 <= int(translation_count) <= 128:
        raise ValueError(
            "single_lane_canonical=True requires 65--128 translations, got "
            f"{translation_count}",
        )


def _validate_relion_coarse_prehalf_weight(
    *,
    canonical_reduction: bool,
    single_lane_canonical: bool,
    prehalf_weight: bool,
) -> None:
    """Restrict the prehalved-weight experiment to native atomics."""

    if prehalf_weight and (canonical_reduction or single_lane_canonical):
        raise ValueError(
            "prehalf_weight=True requires canonical_reduction=False and single_lane_canonical=False",
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "current_size",
        "physical_image_size",
        "model_max_r",
        "canonical_reduction",
        "single_lane_canonical",
        "prehalf_weight",
    ),
)
def relion_coarse_diff2_projector_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    canonical_reduction: bool = False,
    single_lane_canonical: bool = False,
    prehalf_weight: bool = False,
) -> jax.Array:
    """Run the parity-locked shared RELION fused coarse projector.

    ``canonical_reduction`` replaces the final atomic lane admission with a
    fixed lane-index sum.  It retains the same projector, interpolation,
    translation, pixel traversal, and per-lane arithmetic and is intended for
    source-order parity qualification at marginal adaptive cutoffs.

    ``single_lane_canonical`` is a default-off compile-time specialization for
    65--128 translations, where one CUDA thread is the sole contributor to
    each translation.  It preserves the canonical initial-plus-lane addition
    while omitting the generic lane staging buffer.

    ``prehalf_weight`` is a default-off native-atomic experiment that matches
    RELION's coarse source order by staging ``weight * 0.5`` once per pixel
    instead of multiplying every orientation's square sum by ``0.5``.  It is
    not supported by either canonical-reduction specialization.
    """

    _validate_relion_coarse_single_lane_canonical(
        translation_angles.shape[0],
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
    )
    _validate_relion_coarse_prehalf_weight(
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
        prehalf_weight=prehalf_weight,
    )

    compact_rotations, out_type = _prepare_relion_coarse_diff2_projector_f32(
        projector_full,
        rotation_matrices,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=model_max_r,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_PROJECTOR_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        compact_rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=np.int64(current_size),
        physical_image_size=np.int64(physical_image_size),
        model_max_r=np.int64(model_max_r),
        canonical_reduction=np.int64(bool(canonical_reduction)),
        single_lane_canonical=np.int64(bool(single_lane_canonical)),
        prehalf_weight=np.int64(bool(prehalf_weight)),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "current_size",
        "physical_image_size",
        "model_max_r",
        "canonical_reduction",
        "single_lane_canonical",
        "prehalf_weight",
    ),
)
def relion_coarse_diff2_projector_multistream_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    actual_batch_size: jax.Array,
    canonical_reduction: bool = True,
    single_lane_canonical: bool = False,
    prehalf_weight: bool = False,
) -> jax.Array:
    """Score physical particle rows over RELION's eight worker streams.

    This dispatcher calls the same production coarse projector kernel as
    :func:`relion_coarse_diff2_projector_f32`, including either its fixed-order
    canonical reduction or its native RELION atomic lane admission.  Only
    particle launch scheduling differs: one shared projector texture feeds
    eight independent blocking streams, and rows at or beyond
    ``actual_batch_size`` are initialized but never scored.  The actual row
    count is a runtime scalar operand so different final-batch occupancies at
    one physical shape reuse the same compilation.
    """

    _validate_relion_coarse_single_lane_canonical(
        translation_angles.shape[0],
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
    )
    _validate_relion_coarse_prehalf_weight(
        canonical_reduction=canonical_reduction,
        single_lane_canonical=single_lane_canonical,
        prehalf_weight=prehalf_weight,
    )
    compact_rotations, out_type = _prepare_relion_coarse_diff2_projector_f32(
        projector_full,
        rotation_matrices,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=model_max_r,
    )
    actual_batch_size = jnp.asarray(actual_batch_size)
    if actual_batch_size.shape != () or actual_batch_size.dtype != jnp.int32:
        raise ValueError(
            "actual_batch_size must be a scalar int32 runtime operand, got "
            f"{actual_batch_size.shape} {actual_batch_size.dtype}",
        )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_PROJECTOR_MULTISTREAM_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        compact_rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        actual_batch_size,
        current_size=np.int64(current_size),
        physical_image_size=np.int64(physical_image_size),
        model_max_r=np.int64(model_max_r),
        canonical_reduction=np.int64(bool(canonical_reduction)),
        single_lane_canonical=np.int64(bool(single_lane_canonical)),
        prehalf_weight=np.int64(bool(prehalf_weight)),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "current_size",
        "physical_image_size",
        "model_max_r",
        "prehalf_weight",
    ),
)
def relion_coarse_diff2_projector_lanes_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    images: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    *,
    current_size: int,
    physical_image_size: int,
    model_max_r: int,
    prehalf_weight: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Expose pre-atomic lanes from the shared fused coarse projector.

    This is a bounded diagnostic for exact RELION cutoff investigations.  It
    uses a compile-time capture specialization of the production kernel and
    returns ``(diff2, lanes)`` with lanes shaped ``(B, R, 128)``.  For a
    translation ``t``, its active partials are lanes ``t + q*T`` for
    ``q < 128 // T``; the remaining lanes are positive zero.

    ``prehalf_weight=True`` captures the explicit native-atomic prehalved
    specialization; the default captures the unchanged production arithmetic.
    """

    compact_rotations, out_type = _prepare_relion_coarse_diff2_projector_f32(
        projector_full,
        rotation_matrices,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
        physical_image_size=physical_image_size,
        model_max_r=model_max_r,
    )
    lane_type = jax.ShapeDtypeStruct(
        (images.shape[0], rotation_matrices.shape[0], 128),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_PROJECTOR_LANES_F32,
        (out_type, lane_type),
        vmap_method="sequential",
    )(
        projector_full,
        compact_rotations,
        images,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=np.int64(current_size),
        physical_image_size=np.int64(physical_image_size),
        model_max_r=np.int64(model_max_r),
        prehalf_weight=np.int64(bool(prehalf_weight)),
    )


@functools.partial(jax.jit, static_argnums=(7, 8, 9))
def relion_coarse_diff2_native_texture_rectangular_f32(
    projector_full: jax.Array,
    rotation_matrices: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
    current_size: int,
    padding_factor: int,
    projector_max_r: int,
) -> jax.Array:
    """Run RELION's fused texture-projection coarse Gaussian topology.

    This diagnostic mirrors ``cuda_kernel_diff2_coarse<true, false, 128,
    16, 4>``: it loads 16 Euler matrices into shared memory, projects the
    reference from a CUDA texture, translates each image, and accumulates the
    coarse score in one kernel. ``projector_full`` is the centered full-cube
    embedding of RELION ``Projector::data`` and must already carry the dense
    scorer scale.
    """

    projector_full = jnp.asarray(projector_full)
    rotation_matrices = jnp.asarray(rotation_matrices)
    image = jnp.asarray(image)
    translation_angles = jnp.asarray(translation_angles)
    weight = jnp.asarray(weight)
    initial_diff2 = jnp.asarray(initial_diff2)
    full_to_compact = jnp.asarray(full_to_compact)
    if projector_full.dtype != jnp.complex64:
        raise TypeError(
            f"projector_full must be complex64, got {projector_full.dtype}"
        )
    if rotation_matrices.dtype != jnp.float32:
        raise TypeError(
            "rotation_matrices must be float32, got "
            f"{rotation_matrices.dtype}"
        )
    if image.dtype != jnp.complex64:
        raise TypeError(f"image must be complex64, got {image.dtype}")
    if translation_angles.dtype != jnp.float32:
        raise TypeError(
            f"translation_angles must be float32, got {translation_angles.dtype}"
        )
    if weight.dtype != jnp.float32 or initial_diff2.dtype != jnp.float32:
        raise TypeError("weight and initial_diff2 must be float32")
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            f"full_to_compact must be int32, got {full_to_compact.dtype}"
        )
    if (
        projector_full.ndim != 3
        or projector_full.shape[0] <= 0
        or projector_full.shape[1:]
        != (projector_full.shape[0], projector_full.shape[0])
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[1:] != (3, 3)
        or rotation_matrices.shape[0] <= 0
        or image.ndim != 2
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or translation_angles.ndim != 2
        or translation_angles.shape[0] <= 0
        or translation_angles.shape[0] > 128
        or translation_angles.shape[1] != 2
        or weight.shape != image.shape
        or initial_diff2.shape != (image.shape[0],)
        or full_to_compact.shape
        != (int(current_size) * (int(current_size) // 2 + 1),)
        or int(current_size) <= 0
        or int(padding_factor) <= 0
        or int(projector_max_r) <= 0
    ):
        raise ValueError(
            "native texture coarse diff2 operands have inconsistent shapes: "
            f"projector={projector_full.shape}, rotations={rotation_matrices.shape}, "
            f"image={image.shape}, translations={translation_angles.shape}, "
            f"weight={weight.shape}, initial={initial_diff2.shape}, "
            f"lookup={full_to_compact.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION native texture coarse diff2 requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION native texture coarse diff2 was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    # RELION stores its scorer Euler matrices as a row-major transpose of the
    # RECOVAR projection matrices at this handoff.
    eulers = jnp.swapaxes(rotation_matrices, -1, -2).reshape(
        rotation_matrices.shape[0], 9
    )
    out_type = jax.ShapeDtypeStruct(
        (image.shape[0], rotation_matrices.shape[0], translation_angles.shape[0]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_NATIVE_TEXTURE_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(
        projector_full,
        eulers,
        image,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=np.int64(current_size),
        padding_factor=np.int64(padding_factor),
        projector_max_r=np.int64(projector_max_r),
    )
@jax.jit
def relion_coarse_diff2_rectangular_f64(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    initial_diff2: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate RELION's coarse Gaussian CUDA topology in double precision."""

    _validate_relion_fine_diff2_inputs(
        reference, shifted_image, weight, full_to_compact, real_dtype=jnp.float64
    )
    if initial_diff2.dtype != jnp.float64:
        raise TypeError(f"initial_diff2 must be float64, got {initial_diff2.dtype}")
    if reference.ndim != 2 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError("rectangular coarse diff2 expects reference/image/weight ranks 2/3/2")
    if (
        shifted_image.shape[0] != weight.shape[0]
        or reference.shape[1] != shifted_image.shape[2]
        or reference.shape[1] != weight.shape[1]
        or shifted_image.shape[1] <= 0
        or shifted_image.shape[1] > 128
        or initial_diff2.shape != (shifted_image.shape[0],)
    ):
        raise ValueError("rectangular coarse diff2 operands have inconsistent shapes")
    out_type = jax.ShapeDtypeStruct(
        (shifted_image.shape[0], reference.shape[0], shifted_image.shape[1]),
        jnp.float64,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_COARSE_DIFF2_RECTANGULAR_F64,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact)


@jax.jit
def relion_fine_diff2_rectangular_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Evaluate RELION's SM80 fine Gaussian tree on a rectangular grid.

    Shapes are ``reference=(B,R,N)``, ``shifted_image=(B,T,N)``,
    ``weight=(B,N)``, ``full_to_compact=(F,)``, and ``initial_diff2=(B,)``.
    One CUDA block evaluates each output hypothesis ``(B,R,T)`` with the
    production 256-lane topology, explicit binary32 FMA rounding boundaries,
    and RELION's final in-kernel ``sum + sum_init`` operation.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "rectangular fine diff2 expects reference/shifted rank 3 and "
            f"weight rank 2, got {reference.shape}, {shifted_image.shape}, "
            f"{weight.shape}"
        )
    if (
        reference.shape[0] != shifted_image.shape[0]
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != shifted_image.shape[2]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "rectangular fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((reference.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        reference.shape[0],
    ):
        raise ValueError(
            "rectangular fine diff2 initial_diff2 must be float32 with shape "
            f"({reference.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, initial_diff2, full_to_compact)


@functools.partial(
    jax.jit,
    static_argnames=("xdim", "ydim", "resolution_limit"),
)
def relion_powerclass_spectrum_highres_f32(
    relion_image: jax.Array,
    *,
    xdim: int,
    ydim: int,
    resolution_limit: int,
) -> jax.Array:
    """Run RELION's 2-D CUDA ``powerClass`` spectrum/high-tail atomics.

    ``relion_image`` is the native uncentred RFFT layout ``(B, ydim*xdim)``.
    The result contains ``xdim`` shell bins followed by the high-resolution
    Xi2 scalar, preserving the native spectrum atomics that influence block
    arrival order at the final scalar atomic.
    """

    relion_image = jnp.asarray(relion_image)
    if relion_image.dtype != jnp.complex64:
        raise TypeError(f"relion_image must be complex64, got {relion_image.dtype}")
    if (
        relion_image.ndim != 2
        or relion_image.shape[0] <= 0
        or xdim <= 0
        or ydim <= 0
        or relion_image.shape[1] != int(xdim) * int(ydim)
        or resolution_limit < 0
        or resolution_limit > xdim
    ):
        raise ValueError(
            "RELION powerClass operands have inconsistent dimensions: "
            f"image={relion_image.shape}, xdim={xdim}, ydim={ydim}, "
            f"resolution_limit={resolution_limit}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION powerClass requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION powerClass was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()
    out_type = jax.ShapeDtypeStruct(
        (relion_image.shape[0], int(xdim) + 1),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_F32,
        out_type,
        vmap_method="sequential",
    )(
        relion_image,
        xdim=int(xdim),
        ydim=int(ydim),
        resolution_limit=int(resolution_limit),
    )


@functools.partial(jax.jit, static_argnames=("xdim", "ydim"))
def relion_powerclass_spectrum_highres_runtime_f32(
    relion_image: jax.Array,
    resolution_limit: jax.Array,
    *,
    xdim: int,
    ydim: int,
) -> jax.Array:
    """Run powerClass with a device-side high-shell cutoff scalar."""

    relion_image = jnp.asarray(relion_image)
    resolution_limit = jnp.asarray(resolution_limit, dtype=jnp.int32)
    if relion_image.dtype != jnp.complex64:
        raise TypeError(f"relion_image must be complex64, got {relion_image.dtype}")
    if (
        relion_image.ndim != 2
        or relion_image.shape[0] <= 0
        or xdim <= 0
        or ydim <= 0
        or relion_image.shape[1] != int(xdim) * int(ydim)
    ):
        raise ValueError(
            "runtime RELION powerClass operands have inconsistent dimensions: "
            f"image={relion_image.shape}, xdim={xdim}, ydim={ydim}"
        )
    if resolution_limit.shape != ():
        raise ValueError("resolution_limit must be an int32 scalar")
    if jax.default_backend() != "gpu":
        raise RuntimeError("runtime RELION powerClass requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("runtime RELION powerClass requires custom CUDA")
    _ensure_ffi()
    out_type = jax.ShapeDtypeStruct(
        (relion_image.shape[0], int(xdim) + 1),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_POWERCLASS_SPECTRUM_HIGHRES_RUNTIME_F32,
        out_type,
        vmap_method="sequential",
    )(
        relion_image,
        resolution_limit,
        xdim=int(xdim),
        ydim=int(ydim),
    )


def _prepare_relion_fine_diff2_fused_translate_rectangular_operands(
    reference,
    image,
    translation_angles,
    weight,
    full_to_compact,
    initial_diff2,
):
    """Validate operands shared by static- and runtime-cutoff fine scoring."""

    reference = jnp.asarray(reference)
    image = jnp.asarray(image)
    translation_angles = jnp.asarray(translation_angles)
    weight = jnp.asarray(weight)
    full_to_compact = jnp.asarray(full_to_compact)
    if reference.dtype != jnp.complex64 or image.dtype != jnp.complex64:
        raise TypeError(
            "fused RELION fine diff2 reference/image must be complex64, got "
            f"{reference.dtype} and {image.dtype}"
        )
    if translation_angles.dtype != jnp.float32 or weight.dtype != jnp.float32:
        raise TypeError(
            "fused RELION fine diff2 angles/weight must be float32, got "
            f"{translation_angles.dtype} and {weight.dtype}"
        )
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            "fused RELION fine diff2 lookup must be int32, got "
            f"{full_to_compact.dtype}"
        )
    if (
        reference.ndim != 3
        or image.ndim != 2
        or translation_angles.ndim != 2
        or translation_angles.shape[1] != 2
        or weight.ndim != 2
        or full_to_compact.ndim != 1
        or reference.shape[0] != image.shape[0]
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != image.shape[1]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or reference.shape[2] <= 0
        or translation_angles.shape[0] <= 0
    ):
        raise ValueError(
            "fused RELION fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {image.shape}, {translation_angles.shape}, "
            f"{weight.shape}, {full_to_compact.shape}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((reference.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        reference.shape[0],
    ):
        raise ValueError(
            "fused RELION fine diff2 initial_diff2 must be float32 with shape "
            f"({reference.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    return reference, image, translation_angles, weight, full_to_compact, initial_diff2


@functools.partial(jax.jit, static_argnames=("current_size",))
def relion_fine_diff2_fused_translate_rectangular_f32(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
    *,
    current_size: int,
) -> jax.Array:
    """Evaluate RELION fine diff2 with translation inside the score kernel.

    Shapes are ``reference=(B,R,N)``, ``image=(B,N)``,
    ``translation_angles=(T,2)``, ``weight=(B,N)``, and
    ``full_to_compact=(F,)``, and ``initial_diff2=(B,)``. The CUDA kernel
    follows RELION's 256-lane REF3D topology: seven shared-memory translation
    slots, with the deployed job builder filling at most four. The per-image
    high-resolution addend is applied inside the kernel after the reduction,
    matching RELION's final ``sum + sum_init`` operation. It returns
    ``(B,R,T)``.

    This entry point is intentionally separate from the production scorer
    while the fused translation boundary is being qualified against native
    RELION operand captures.
    """

    (
        reference,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    ) = _prepare_relion_fine_diff2_fused_translate_rectangular_operands(
        reference,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )
    current_size = int(current_size)
    expected_full_pixels = current_size * (current_size // 2 + 1)
    if current_size <= 0 or full_to_compact.shape != (expected_full_pixels,):
        raise ValueError(
            "fused RELION fine diff2 lookup does not match current_size: "
            f"current_size={current_size}, lookup={full_to_compact.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("fused RELION fine diff2 requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "fused RELION fine diff2 was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], translation_angles.shape[0]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(
        reference,
        image,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
    )


def _prepare_relion_fine_diff2_fused_translate_flat_rows_operands(
    reference: jax.Array,
    row_image_ids: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    reference = jnp.asarray(reference)
    row_image_ids = jnp.asarray(row_image_ids)
    image = jnp.asarray(image)
    translation_angles = jnp.asarray(translation_angles)
    weight = jnp.asarray(weight)
    full_to_compact = jnp.asarray(full_to_compact)
    if reference.dtype != jnp.complex64 or image.dtype != jnp.complex64:
        raise TypeError(
            "flat-row RELION fine diff2 reference/image must be complex64, got "
            f"{reference.dtype} and {image.dtype}"
        )
    if row_image_ids.dtype != jnp.int32:
        raise TypeError(
            "flat-row RELION fine diff2 row_image_ids must be int32, got "
            f"{row_image_ids.dtype}"
        )
    if translation_angles.dtype != jnp.float32 or weight.dtype != jnp.float32:
        raise TypeError(
            "flat-row RELION fine diff2 angles/weight must be float32, got "
            f"{translation_angles.dtype} and {weight.dtype}"
        )
    if full_to_compact.dtype != jnp.int32:
        raise TypeError(
            "flat-row RELION fine diff2 lookup must be int32, got "
            f"{full_to_compact.dtype}"
        )
    if (
        reference.ndim != 2
        or row_image_ids.shape != (reference.shape[0],)
        or image.ndim != 2
        or translation_angles.ndim != 2
        or translation_angles.shape[1] != 2
        or weight.shape != image.shape
        or reference.shape[1] != image.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or image.shape[0] <= 0
        or translation_angles.shape[0] <= 0
    ):
        raise ValueError(
            "flat-row RELION fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {row_image_ids.shape}, {image.shape}, "
            f"{translation_angles.shape}, {weight.shape}, {full_to_compact.shape}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((image.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        image.shape[0],
    ):
        raise ValueError(
            "flat-row RELION fine diff2 initial_diff2 must be float32 with shape "
            f"({image.shape[0]},), got {initial_diff2.shape} {initial_diff2.dtype}"
        )
    return (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )


@functools.partial(jax.jit, static_argnames=("current_size",))
def relion_fine_diff2_fused_translate_flat_rows_f32(
    reference: jax.Array,
    row_image_ids: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
    *,
    current_size: int,
) -> jax.Array:
    """Evaluate source-ordered flat rotation rows with RELION fine arithmetic.

    ``reference`` is ``(Q,N)`` and ``row_image_ids`` maps every packed row to
    one of the ``B`` image/weight rows. The CUDA implementation shares the
    rectangular scorer's translation, pixel traversal, binary32 update, and
    256-lane reduction body; only the row-to-image address calculation differs.
    The output shape is ``(Q,T)``.
    """

    (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    ) = _prepare_relion_fine_diff2_fused_translate_flat_rows_operands(
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )
    current_size = int(current_size)
    expected_full_pixels = current_size * (current_size // 2 + 1)
    if current_size <= 0 or full_to_compact.shape != (expected_full_pixels,):
        raise ValueError(
            "flat-row RELION fine diff2 lookup does not match current_size: "
            f"current_size={current_size}, lookup={full_to_compact.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "flat-row fused RELION fine diff2 requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "flat-row fused RELION fine diff2 was explicitly requested but "
            "custom CUDA is disabled"
        )
    _ensure_ffi()

    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], translation_angles.shape[0]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_FLAT_ROWS_F32,
        out_type,
        vmap_method="sequential",
    )(
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        current_size=current_size,
    )


@jax.jit
def relion_fine_diff2_fused_translate_runtime_flat_rows_f32(
    reference: jax.Array,
    row_image_ids: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    logical_current_size: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Evaluate packed rotation rows inside a fixed physical pixel capacity.

    The scalar logical size controls the exact RELION rectangle traversal;
    physical-only score rows remain inert. This is the packed-row counterpart
    of :func:`relion_fine_diff2_fused_translate_runtime_rectangular_f32` and
    reaches the same templated CUDA scoring kernel.
    """

    (
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    ) = _prepare_relion_fine_diff2_fused_translate_flat_rows_operands(
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )
    logical_current_size = jnp.asarray(logical_current_size, dtype=jnp.int32)
    if logical_current_size.shape != ():
        raise ValueError(
            "runtime flat-row RELION fine diff2 logical_current_size must be a scalar, "
            f"got {logical_current_size.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "runtime flat-row fused RELION fine diff2 requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "runtime flat-row fused RELION fine diff2 was explicitly requested "
            "but custom CUDA is disabled"
        )
    _ensure_ffi()

    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], translation_angles.shape[0]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_FLAT_ROWS_F32,
        out_type,
        vmap_method="sequential",
    )(
        reference,
        row_image_ids,
        image,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        logical_current_size,
    )


def _prepare_relion_fine_diff2_fused_translate_pairs_operands(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    pair_reference_rows: jax.Array,
    pair_translation_ids: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Validate operands for compact selected-pair fused fine scoring."""

    reference = jnp.asarray(reference)
    image = jnp.asarray(image)
    translation_angles = jnp.asarray(translation_angles)
    weight = jnp.asarray(weight)
    pair_reference_rows = jnp.asarray(pair_reference_rows)
    pair_translation_ids = jnp.asarray(pair_translation_ids)
    full_to_compact = jnp.asarray(full_to_compact)
    if reference.dtype != jnp.complex64 or image.dtype != jnp.complex64:
        raise TypeError(
            "pair-indexed RELION fine diff2 reference/image must be complex64, got "
            f"{reference.dtype} and {image.dtype}"
        )
    if (
        pair_reference_rows.dtype != jnp.int32
        or pair_translation_ids.dtype != jnp.int32
        or full_to_compact.dtype != jnp.int32
    ):
        raise TypeError(
            "pair-indexed RELION fine diff2 pair ids/lookup must be int32, got "
            f"{pair_reference_rows.dtype}, {pair_translation_ids.dtype}, and "
            f"{full_to_compact.dtype}"
        )
    if translation_angles.dtype != jnp.float32 or weight.dtype != jnp.float32:
        raise TypeError(
            "pair-indexed RELION fine diff2 angles/weight must be float32, got "
            f"{translation_angles.dtype} and {weight.dtype}"
        )
    if (
        reference.ndim != 2
        or image.ndim != 2
        or translation_angles.ndim != 2
        or translation_angles.shape[1] != 2
        or weight.shape != image.shape
        or pair_reference_rows.ndim != 2
        or pair_translation_ids.shape != pair_reference_rows.shape
        or pair_reference_rows.shape[0] != image.shape[0]
        or full_to_compact.ndim != 1
        or reference.shape[1] != image.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or image.shape[0] <= 0
        or translation_angles.shape[0] <= 0
        or pair_reference_rows.shape[1] <= 0
    ):
        raise ValueError(
            "pair-indexed RELION fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {image.shape}, {translation_angles.shape}, "
            f"{weight.shape}, {pair_reference_rows.shape}, "
            f"{pair_translation_ids.shape}, {full_to_compact.shape}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((image.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        image.shape[0],
    ):
        raise ValueError(
            "pair-indexed RELION fine diff2 initial_diff2 must be float32 with "
            f"shape ({image.shape[0]},), got {initial_diff2.shape} "
            f"{initial_diff2.dtype}"
        )
    return (
        reference,
        image,
        translation_angles,
        weight,
        initial_diff2,
        pair_reference_rows,
        pair_translation_ids,
        full_to_compact,
    )


@functools.partial(jax.jit, static_argnames=("current_size",))
def relion_fine_diff2_fused_translate_pairs_f32(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    pair_reference_rows: jax.Array,
    pair_translation_ids: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
    *,
    current_size: int,
) -> jax.Array:
    """Evaluate selected rotation/translation pairs without dense expansion.

    ``reference`` is ``(Q,N)``. The two ``(B,P)`` index arrays select one
    reference row and one translation for each output pair; ``-1`` is an
    inert padding sentinel. The output is ``(B,P)`` and uses the same RELION
    translation, pixel traversal, binary32 update, and 256-lane reduction as
    the accepted rectangular and flat-row kernels.
    """

    operands = _prepare_relion_fine_diff2_fused_translate_pairs_operands(
        reference,
        image,
        translation_angles,
        weight,
        pair_reference_rows,
        pair_translation_ids,
        full_to_compact,
        initial_diff2,
    )
    current_size = int(current_size)
    expected_full_pixels = current_size * (current_size // 2 + 1)
    if current_size <= 0 or operands[-1].shape != (expected_full_pixels,):
        raise ValueError(
            "pair-indexed RELION fine diff2 lookup does not match current_size: "
            f"current_size={current_size}, lookup={operands[-1].shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "pair-indexed fused RELION fine diff2 requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "pair-indexed fused RELION fine diff2 was explicitly requested but "
            "custom CUDA is disabled"
        )
    _ensure_ffi()
    out_type = jax.ShapeDtypeStruct(operands[5].shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(*operands, current_size=current_size)


@jax.jit
def relion_fine_diff2_fused_translate_runtime_pairs_f32(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    pair_reference_rows: jax.Array,
    pair_translation_ids: jax.Array,
    full_to_compact: jax.Array,
    logical_current_size: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Evaluate selected fine pairs inside fixed physical pixel capacity."""

    operands = _prepare_relion_fine_diff2_fused_translate_pairs_operands(
        reference,
        image,
        translation_angles,
        weight,
        pair_reference_rows,
        pair_translation_ids,
        full_to_compact,
        initial_diff2,
    )
    logical_current_size = jnp.asarray(logical_current_size, dtype=jnp.int32)
    if logical_current_size.shape != ():
        raise ValueError(
            "runtime pair-indexed RELION fine diff2 logical_current_size must "
            f"be a scalar, got {logical_current_size.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "runtime pair-indexed fused RELION fine diff2 requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "runtime pair-indexed fused RELION fine diff2 was explicitly "
            "requested but custom CUDA is disabled"
        )
    _ensure_ffi()
    out_type = jax.ShapeDtypeStruct(operands[5].shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(*operands, logical_current_size)


def _prepare_relion_fine_diff2_fused_translate_jobs_operands(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    job_plan: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Validate the global compact fine-job ABI."""

    reference = jnp.asarray(reference)
    image = jnp.asarray(image)
    translation_angles = jnp.asarray(translation_angles)
    weight = jnp.asarray(weight)
    job_plan = jnp.asarray(job_plan)
    full_to_compact = jnp.asarray(full_to_compact)
    if reference.dtype != jnp.complex64 or image.dtype != jnp.complex64:
        raise TypeError(
            "job-indexed RELION fine diff2 reference/image must be complex64, got "
            f"{reference.dtype} and {image.dtype}"
        )
    if job_plan.dtype != jnp.int32 or full_to_compact.dtype != jnp.int32:
        raise TypeError(
            "job-indexed RELION fine diff2 plan/lookup must be int32, got "
            f"{job_plan.dtype} and {full_to_compact.dtype}"
        )
    if translation_angles.dtype != jnp.float32 or weight.dtype != jnp.float32:
        raise TypeError(
            "job-indexed RELION fine diff2 angles/weight must be float32, got "
            f"{translation_angles.dtype} and {weight.dtype}"
        )
    if (
        reference.ndim != 2
        or image.ndim != 2
        or translation_angles.ndim != 2
        or translation_angles.shape[1] != 2
        or weight.shape != image.shape
        or job_plan.ndim != 2
        or job_plan.shape[0] <= 0
        or job_plan.shape[1] != 4
        or full_to_compact.ndim != 1
        or reference.shape[1] != image.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or image.shape[0] <= 0
        or translation_angles.shape[0] <= 0
    ):
        raise ValueError(
            "job-indexed RELION fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {image.shape}, {translation_angles.shape}, "
            f"{weight.shape}, {job_plan.shape}, {full_to_compact.shape}"
        )
    if initial_diff2 is None:
        initial_diff2 = jnp.zeros((image.shape[0],), dtype=jnp.float32)
    else:
        initial_diff2 = jnp.asarray(initial_diff2)
    if initial_diff2.dtype != jnp.float32 or initial_diff2.shape != (
        image.shape[0],
    ):
        raise ValueError(
            "job-indexed RELION fine diff2 initial_diff2 must be float32 with "
            f"shape ({image.shape[0]},), got {initial_diff2.shape} "
            f"{initial_diff2.dtype}"
        )
    return (
        reference,
        image,
        translation_angles,
        weight,
        initial_diff2,
        job_plan,
        full_to_compact,
    )


@functools.partial(jax.jit, static_argnames=("current_size",))
def relion_fine_diff2_fused_translate_jobs_f32(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    job_plan: jax.Array,
    full_to_compact: jax.Array,
    initial_diff2: jax.Array | None = None,
    *,
    current_size: int,
) -> jax.Array:
    """Score one global source-ordered compact fine-job plan.

    ``job_plan`` has shape ``(J, 4)`` and stores image row, projected-reference
    row, dense rotation row, and translation id.  ``-1`` rows are inert static
    padding.  The dense rotation field is returned to the caller unchanged for
    scattering; CUDA consumes the other three fields and returns ``(J,)``.
    """

    operands = _prepare_relion_fine_diff2_fused_translate_jobs_operands(
        reference,
        image,
        translation_angles,
        weight,
        job_plan,
        full_to_compact,
        initial_diff2,
    )
    current_size = int(current_size)
    expected_full_pixels = current_size * (current_size // 2 + 1)
    if current_size <= 0 or operands[-1].shape != (expected_full_pixels,):
        raise ValueError(
            "job-indexed RELION fine diff2 lookup does not match current_size: "
            f"current_size={current_size}, lookup={operands[-1].shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("job-indexed fused RELION fine diff2 requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "job-indexed fused RELION fine diff2 was explicitly requested but "
            "custom CUDA is disabled"
        )
    _ensure_ffi()
    out_type = jax.ShapeDtypeStruct((operands[5].shape[0],), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_JOBS_F32,
        out_type,
        vmap_method="sequential",
    )(*operands, current_size=current_size)


@jax.jit
def relion_fine_diff2_fused_translate_runtime_jobs_f32(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    job_plan: jax.Array,
    full_to_compact: jax.Array,
    logical_current_size: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Score compact fine jobs inside a fixed physical pixel capacity."""

    operands = _prepare_relion_fine_diff2_fused_translate_jobs_operands(
        reference,
        image,
        translation_angles,
        weight,
        job_plan,
        full_to_compact,
        initial_diff2,
    )
    logical_current_size = jnp.asarray(logical_current_size, dtype=jnp.int32)
    if logical_current_size.shape != ():
        raise ValueError(
            "runtime job-indexed RELION fine diff2 logical_current_size must "
            f"be a scalar, got {logical_current_size.shape}"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "runtime job-indexed fused RELION fine diff2 requires a JAX GPU backend"
        )
    if not custom_cuda_requested():
        raise RuntimeError(
            "runtime job-indexed fused RELION fine diff2 was explicitly "
            "requested but custom CUDA is disabled"
        )
    _ensure_ffi()
    out_type = jax.ShapeDtypeStruct((operands[5].shape[0],), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_JOBS_F32,
        out_type,
        vmap_method="sequential",
    )(*operands, logical_current_size)


@jax.jit
def relion_fine_diff2_fused_translate_runtime_rectangular_f32(
    reference: jax.Array,
    image: jax.Array,
    translation_angles: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
    logical_current_size: jax.Array,
    initial_diff2: jax.Array | None = None,
) -> jax.Array:
    """Evaluate fine diff2 with a runtime cutoff inside fixed-capacity buffers.

    ``reference``, ``image``, ``weight``, and ``full_to_compact`` may have a
    shared physical capacity larger than the active logical window.  The
    scalar ``logical_current_size`` controls both the rectangular issue count
    and the native coordinates inside CUDA.  Pixels after that logical prefix
    are never issued, so the 256-lane accumulation tree is unchanged.
    """

    (
        reference,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    ) = _prepare_relion_fine_diff2_fused_translate_rectangular_operands(
        reference,
        image,
        translation_angles,
        weight,
        full_to_compact,
        initial_diff2,
    )
    logical_current_size = jnp.asarray(logical_current_size, dtype=jnp.int32)
    if logical_current_size.shape != ():
        raise ValueError(
            "runtime fused RELION fine diff2 logical_current_size must be a scalar, "
            f"got {logical_current_size.shape}",
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("runtime fused RELION fine diff2 requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "runtime fused RELION fine diff2 was explicitly requested but custom CUDA is disabled"
        )
    _ensure_ffi()

    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], translation_angles.shape[0]),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_FUSED_TRANSLATE_RUNTIME_RECTANGULAR_F32,
        out_type,
        vmap_method="sequential",
    )(
        reference,
        image,
        translation_angles,
        weight,
        initial_diff2,
        full_to_compact,
        logical_current_size,
    )


@jax.jit
def relion_fine_diff2_pairs_f32(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """Evaluate RELION's SM80 fine Gaussian tree for compact candidate pairs.

    Shapes are ``reference=(B,P,N)``, ``shifted_image=(B,P,N)``,
    ``weight=(B,N)``, and ``full_to_compact=(F,)``; output is ``(B,P)``.
    """

    _validate_relion_fine_diff2_inputs(
        reference,
        shifted_image,
        weight,
        full_to_compact,
    )
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "pair fine diff2 expects reference/shifted rank 3 and weight rank "
            f"2, got {reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    if (
        reference.shape != shifted_image.shape
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "pair fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    out_type = jax.ShapeDtypeStruct(reference.shape[:2], jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_PAIRS_F32,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_fine_diff2_rectangular_f64(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """ACC_DOUBLE_PRECISION specialization of the rectangular fine tree."""

    _validate_relion_fine_diff2_inputs(
        reference, shifted_image, weight, full_to_compact,
        real_dtype=jnp.float64,
    )
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "rectangular fine diff2 expects reference/shifted rank 3 and "
            f"weight rank 2, got {reference.shape}, {shifted_image.shape}, "
            f"{weight.shape}"
        )
    if (
        reference.shape[0] != shifted_image.shape[0]
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != shifted_image.shape[2]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or shifted_image.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "rectangular fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    out_type = jax.ShapeDtypeStruct(
        (reference.shape[0], reference.shape[1], shifted_image.shape[1]),
        jnp.float64,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_RECTANGULAR_F64,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@jax.jit
def relion_fine_diff2_pairs_f64(
    reference: jax.Array,
    shifted_image: jax.Array,
    weight: jax.Array,
    full_to_compact: jax.Array,
) -> jax.Array:
    """ACC_DOUBLE_PRECISION specialization of the compact-pair fine tree."""

    _validate_relion_fine_diff2_inputs(
        reference, shifted_image, weight, full_to_compact,
        real_dtype=jnp.float64,
    )
    if reference.ndim != 3 or shifted_image.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "pair fine diff2 expects reference/shifted rank 3 and weight rank "
            f"2, got {reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    if (
        reference.shape != shifted_image.shape
        or reference.shape[0] != weight.shape[0]
        or reference.shape[2] != weight.shape[1]
        or reference.shape[0] <= 0
        or reference.shape[1] <= 0
        or reference.shape[2] <= 0
    ):
        raise ValueError(
            "pair fine diff2 operands have inconsistent shapes: "
            f"{reference.shape}, {shifted_image.shape}, {weight.shape}"
        )
    out_type = jax.ShapeDtypeStruct(reference.shape[:2], jnp.float64)
    return jax.ffi.ffi_call(
        _TARGET_RELION_FINE_DIFF2_PAIRS_F64,
        out_type,
        vmap_method="sequential",
    )(reference, shifted_image, weight, full_to_compact)


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def relion_preprocess_real_f32(
    images: jax.Array,
    normalization_factors: jax.Array,
    integer_shifts: jax.Array,
    radius: float,
    cosine_width: float,
    apply_mask: bool = True,
    native_lane_reduction: bool = False,
    native_atomic_reduction: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Apply RELION's accelerated float32 real-space preprocessing.

    Returns ``(normalized_shifted, masked)`` so captured operand tests can
    gate both stored RELION boundaries.  The implementation preserves
    RELION's separate float32 normalization and zero-filled translation and
    CUDA ``sqrtf``/``cospif`` mask arithmetic.  The default uses RECOVAR's
    accepted deterministic block-first addition tree.  The diagnostic-only
    ``native_lane_reduction`` mode instead deterministically reproduces the
    native observer's lane-across-blocks tree before its final CUB sum. The
    diagnostic ``native_atomic_reduction`` mode reproduces RELION's actual
    schedule-dependent atomic lane accumulation.
    """

    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION CUDA preprocessing requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("RELION CUDA preprocessing was explicitly requested but custom CUDA is disabled")
    _ensure_ffi()
    if images.dtype != jnp.float32:
        raise TypeError(f"images must be float32, got {images.dtype}")
    if normalization_factors.dtype != jnp.float32:
        raise TypeError(f"normalization_factors must be float32, got {normalization_factors.dtype}")
    if integer_shifts.dtype != jnp.int32:
        raise TypeError(f"integer_shifts must be int32, got {integer_shifts.dtype}")
    if images.ndim != 3 or images.shape[-2] != images.shape[-1]:
        raise ValueError(f"images must have shape (batch, D, D), got {images.shape}")
    batch_size = images.shape[0]
    if normalization_factors.shape != (batch_size,):
        raise ValueError(f"normalization_factors must have shape ({batch_size},), got {normalization_factors.shape}")
    if integer_shifts.shape != (batch_size, 2):
        raise ValueError(f"integer_shifts must have shape ({batch_size}, 2), got {integer_shifts.shape}")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius must be finite and positive, got {radius}")
    if not np.isfinite(cosine_width) or cosine_width <= 0.0:
        raise ValueError(f"cosine_width must be finite and positive, got {cosine_width}")
    if native_lane_reduction and native_atomic_reduction:
        raise ValueError("native lane and native atomic reductions are mutually exclusive")

    out_type = jax.ShapeDtypeStruct(images.shape, jnp.float32)
    target = (
        _TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_ATOMIC
        if native_atomic_reduction
        else (
            _TARGET_RELION_PREPROCESS_REAL_F32_NATIVE_LANE
            if native_lane_reduction
            else _TARGET_RELION_PREPROCESS_REAL_F32
        )
    )
    return jax.ffi.ffi_call(
        target,
        (out_type, out_type),
        vmap_method="sequential",
    )(
        images,
        normalization_factors,
        integer_shifts,
        radius=np.float32(radius),
        cosine_width=np.float32(cosine_width),
        apply_mask=np.int64(int(apply_mask)),
    )


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

    return jax.ffi.ffi_call(
        _TARGET_BACKPROJECT_INDEXED,
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


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def relion_fused_x_half_backproject_indexed(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    data_rows: jax.Array,
    weight_rows: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Fused RELION x-half data/weight backprojection diagnostic.

    This target is intentionally narrower than :func:`backproject_indexed`:
    it accepts matching complex64/float32 or complex128/float64 rows, expands
    both to RELION's native current-size FFTW square, and updates the matching
    data and weight accumulators in one 128-thread CUDA grid.  The two
    output buffers alias their corresponding input accumulators.
    """

    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if max_r is None:
        raise ValueError("RELION fused x-half backprojection requires an explicit support radius")
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(f"RELION fused x-half backprojection requires an odd BPref grid, got {volume_shape}")

    if data_volume.dtype not in {jnp.dtype(jnp.complex64), jnp.dtype(jnp.complex128)}:
        raise TypeError(f"RELION fused x-half data volume must be complex64 or complex128, got {data_volume.dtype}")
    real_dtype = jnp.dtype(jnp.float64 if data_volume.dtype == jnp.dtype(jnp.complex128) else jnp.float32)
    if weight_volume.dtype != real_dtype:
        raise TypeError(f"RELION fused x-half weight volume must be {real_dtype}, got {weight_volume.dtype}")
    if data_rows.dtype != data_volume.dtype:
        raise TypeError(f"RELION fused x-half data rows must match the volume dtype, got {data_rows.dtype}")
    if weight_rows.dtype != real_dtype:
        raise TypeError(f"RELION fused x-half weight rows must be {real_dtype}, got {weight_rows.dtype}")
    if pixel_indices.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"RELION fused x-half pixel indices must be int32, got {pixel_indices.dtype}")
    if not jnp.issubdtype(rotation_matrices.dtype, jnp.floating):
        raise TypeError(f"RELION fused x-half rotations must be floating point, got {rotation_matrices.dtype}")

    if data_rows.ndim != 2 or weight_rows.ndim != 2:
        raise ValueError(
            "RELION fused x-half rows must both have shape (n_rotations, n_pixels), "
            f"got {data_rows.shape} and {weight_rows.shape}"
        )
    if data_rows.shape != weight_rows.shape:
        raise ValueError(
            f"RELION fused x-half data/weight row shape mismatch: {data_rows.shape} vs {weight_rows.shape}"
        )
    if data_rows.shape[0] <= 0 or data_rows.shape[1] <= 0:
        raise ValueError(f"RELION fused x-half rows must be nonempty, got {data_rows.shape}")
    if pixel_indices.ndim != 1 or pixel_indices.shape[0] != data_rows.shape[1]:
        raise ValueError(
            "RELION fused x-half pixel indices must match the row pixel axis, "
            f"got {pixel_indices.shape} for rows {data_rows.shape}"
        )
    if rotation_matrices.shape != (data_rows.shape[0], 3, 3):
        raise ValueError(
            f"RELION fused x-half rotations must have shape ({data_rows.shape[0]}, 3, 3), got {rotation_matrices.shape}"
        )
    if data_volume.ndim != 1 or weight_volume.ndim != 1:
        raise ValueError(
            f"RELION fused x-half accumulators must be flat, got {data_volume.shape} and {weight_volume.shape}"
        )
    expected_volume_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
    if data_volume.shape != (expected_volume_size,) or weight_volume.shape != (expected_volume_size,):
        raise ValueError(
            "RELION fused x-half accumulator shape mismatch: expected "
            f"{(expected_volume_size,)}, got {data_volume.shape} and {weight_volume.shape}"
        )

    dense_data_rows, dense_indices, current_height, current_half_width = _prepare_relion_x_half_block_topology_operands(
        data_rows, pixel_indices, image_shape, max_r
    )
    dense_weight_rows, weight_dense_indices, weight_height, weight_half_width = (
        _prepare_relion_x_half_block_topology_operands(weight_rows, pixel_indices, image_shape, max_r)
    )
    if (weight_height, weight_half_width) != (current_height, current_half_width):
        raise ValueError("RELION fused x-half data/weight topology metadata mismatch")
    # Both preparations use the same original indices, shape, and radius, so
    # their dense index vectors are identical. Keep a shape assertion here and
    # pass only one vector to the FFI target.
    if weight_dense_indices.shape != dense_indices.shape:
        raise ValueError("RELION fused x-half data/weight dense index shape mismatch")

    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    kw["image_h"] = np.int64(current_height)
    kw["image_w"] = np.int64(current_half_width)
    kw["full_image_w"] = np.int64(current_height)
    # Sparse pass-2 supplies the dedicated M-step matrices generated through
    # RELION's host ``generateEulerMatrices(..., inverse=true)`` emulation.
    # They are already in RECOVAR's transposed convention; applying the
    # generic scorer-to-backprojector numerical inverse here would invert them
    # a second time.  The packed half-axis permutation is still required.
    rotation_matrices = rotation_matrices.astype(real_dtype)[..., [2, 1, 0]]
    rot6 = _rot_to_compact(rotation_matrices, real_dtype)

    out_types = (
        jax.ShapeDtypeStruct(data_volume.shape, data_volume.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FUSED_X_HALF_BP,
        out_types,
        input_output_aliases={4: 0, 5: 1},
        vmap_method="sequential",
    )(
        dense_data_rows,
        dense_weight_rows,
        dense_indices,
        rot6,
        data_volume,
        weight_volume,
        **kw,
    )


@functools.partial(jax.jit, static_argnums=(6, 7, 8))
def relion_fused_x_half_backproject_particle_grid_indexed(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    data_rows: jax.Array,
    weight_rows: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Launch one ordered native fused grid for every particle in one FFI call.

    Inputs retain explicit ``(particle, rotation, pixel)`` ownership.  The
    native handler loops over particles on the caller's CUDA stream and
    launches the same RELION-shaped grid used by the single-particle target,
    eliminating framework dispatch between particle-owned launches without
    combining their atomic streams.
    """

    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if max_r is None:
        raise ValueError("RELION fused x-half particle grid requires an explicit support radius")
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(f"RELION fused x-half particle grid requires an odd BPref grid, got {volume_shape}")
    if data_volume.dtype != jnp.dtype(jnp.complex64) or data_rows.dtype != jnp.dtype(jnp.complex64):
        raise TypeError("RELION fused x-half particle-grid data must be complex64")
    if weight_volume.dtype != jnp.dtype(jnp.float32) or weight_rows.dtype != jnp.dtype(jnp.float32):
        raise TypeError("RELION fused x-half particle-grid weights must be float32")
    if pixel_indices.dtype != jnp.dtype(jnp.int32):
        raise TypeError("RELION fused x-half particle-grid pixel indices must be int32")
    if data_rows.ndim != 3 or weight_rows.shape != data_rows.shape:
        raise ValueError(
            "RELION fused x-half particle-grid rows must have matching "
            f"(particle, rotation, pixel) shapes, got {data_rows.shape} and {weight_rows.shape}"
        )
    n_particles, rows_per_particle, n_pixels = map(int, data_rows.shape)
    if n_particles <= 0 or rows_per_particle <= 0 or n_pixels <= 0:
        raise ValueError(f"RELION fused x-half particle-grid rows must be nonempty, got {data_rows.shape}")
    if pixel_indices.shape != (n_pixels,):
        raise ValueError(
            f"RELION fused x-half particle-grid pixel indices must have shape {(n_pixels,)}, "
            f"got {pixel_indices.shape}"
        )
    if rotation_matrices.shape != (n_particles, rows_per_particle, 3, 3):
        raise ValueError(
            "RELION fused x-half particle-grid rotations must have shape "
            f"{(n_particles, rows_per_particle, 3, 3)}, got {rotation_matrices.shape}"
        )

    flat_data = data_rows.reshape(n_particles * rows_per_particle, n_pixels)
    flat_weight = weight_rows.reshape(n_particles * rows_per_particle, n_pixels)
    dense_data, dense_indices, current_height, current_half_width = (
        _prepare_relion_x_half_block_topology_operands(
            flat_data, pixel_indices, image_shape, max_r
        )
    )
    dense_weight, weight_dense_indices, weight_height, weight_half_width = (
        _prepare_relion_x_half_block_topology_operands(
            flat_weight, pixel_indices, image_shape, max_r
        )
    )
    if (weight_height, weight_half_width) != (current_height, current_half_width):
        raise ValueError("RELION fused x-half particle-grid topology metadata mismatch")
    if weight_dense_indices.shape != dense_indices.shape:
        raise ValueError("RELION fused x-half particle-grid dense index shape mismatch")

    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    kw.update(
        image_h=np.int64(current_height),
        image_w=np.int64(current_half_width),
        full_image_w=np.int64(current_height),
        n_particles=np.int64(n_particles),
        rows_per_particle=np.int64(rows_per_particle),
    )
    kernel_rotations = _relion_x_half_backproject_rotation_to_kernel(
        rotation_matrices.reshape(n_particles * rows_per_particle, 3, 3),
        jnp.float32,
    )
    rot6 = _rot_to_compact(kernel_rotations, jnp.float32)
    out_types = (
        jax.ShapeDtypeStruct(data_volume.shape, data_volume.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FUSED_X_HALF_BP_PARTICLE_GRID,
        out_types,
        input_output_aliases={4: 0, 5: 1},
        vmap_method="sequential",
    )(
        dense_data,
        dense_weight,
        dense_indices,
        rot6,
        data_volume,
        weight_volume,
        **kw,
    )


@functools.partial(jax.jit, static_argnums=(8, 9, 10))
def _relion_fused_x_half_backproject_signature_indexed_impl(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    data_rows: jax.Array,
    weight_rows: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    canonical_rotation_keys: jax.Array,
    signature_row_indices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float | None,
) -> tuple[jax.Array, ...]:
    """Run the native fused x-half scatter and capture exact device geometry.

    The handler first launches the ordinary production accumulator kernel.  A
    stream-ordered signature-only kernel then executes the same templated
    float32 coordinate/fold/neighbor code without atomics and writes unique
    ``(row, pixel[, neighbor])`` slots.  The ordinary accumulator launch still
    receives every source row; ``signature_row_indices`` limits only the
    subsequent signature-only launch and its output allocation.  Thus neither
    signature stores nor contributor filtering can perturb the accumulator's
    atomic schedule.  ``canonical_rotation_keys`` is passed explicitly from
    the candidate grid; no Euler nearest-neighbor matching is performed.
    """

    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, 1, True, True, max_r=max_r)
    if max_r is None:
        raise ValueError("RELION fused x-half signature requires an explicit support radius")
    if int(volume_shape[2]) % 2 == 0:
        raise ValueError(f"RELION fused x-half signature requires an odd BPref grid, got {volume_shape}")
    if data_volume.dtype != jnp.complex64 or data_rows.dtype != jnp.complex64:
        raise TypeError("RELION fused x-half signature data rows/volume must be complex64")
    if weight_volume.dtype != jnp.float32 or weight_rows.dtype != jnp.float32:
        raise TypeError("RELION fused x-half signature weight rows/volume must be float32")
    if (
        pixel_indices.dtype != jnp.int32
        or canonical_rotation_keys.dtype != jnp.int32
        or signature_row_indices.dtype != jnp.int32
    ):
        raise TypeError("RELION fused x-half signature pixel, canonical rotation keys, and row indices must be int32")
    if data_rows.ndim != 2 or data_rows.shape != weight_rows.shape:
        raise ValueError("RELION fused x-half signature rows must have matching rank-2 shapes")
    if pixel_indices.shape != (data_rows.shape[1],):
        raise ValueError("RELION fused x-half signature pixel index length mismatch")
    if rotation_matrices.shape != (data_rows.shape[0], 3, 3):
        raise ValueError("RELION fused x-half signature rotations must have shape (n_rows, 3, 3)")
    if canonical_rotation_keys.shape != (data_rows.shape[0],):
        raise ValueError("RELION fused x-half signature canonical key length mismatch")
    if signature_row_indices.ndim != 1 or signature_row_indices.shape[0] <= 0:
        raise ValueError("RELION fused x-half signature row indices must be a nonempty rank-1 array")
    expected_volume_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
    if data_volume.shape != (expected_volume_size,) or weight_volume.shape != (expected_volume_size,):
        raise ValueError("RELION fused x-half signature accumulator shape mismatch")

    dense_data_rows, dense_indices, current_height, current_half_width = _prepare_relion_x_half_block_topology_operands(
        data_rows, pixel_indices, image_shape, max_r
    )
    dense_weight_rows, weight_dense_indices, weight_height, weight_half_width = (
        _prepare_relion_x_half_block_topology_operands(weight_rows, pixel_indices, image_shape, max_r)
    )
    if (weight_height, weight_half_width) != (current_height, current_half_width):
        raise ValueError("RELION fused x-half signature topology metadata mismatch")
    if weight_dense_indices.shape != dense_indices.shape:
        raise ValueError("RELION fused x-half signature dense index shape mismatch")

    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, 1, True, True, max_r)
    kw["image_h"] = np.int64(current_height)
    kw["image_w"] = np.int64(current_half_width)
    kw["full_image_w"] = np.int64(current_height)
    rotation_matrices = _relion_x_half_backproject_rotation_to_kernel(rotation_matrices, jnp.float32)
    rot6 = _rot_to_compact(rotation_matrices, jnp.float32)
    signature_shape = (int(signature_row_indices.shape[0]), int(dense_data_rows.shape[1]))
    out_types = (
        jax.ShapeDtypeStruct(data_volume.shape, data_volume.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
        jax.ShapeDtypeStruct(signature_shape, jnp.int32),
        jax.ShapeDtypeStruct(signature_shape, jnp.int32),
        jax.ShapeDtypeStruct(signature_shape, jnp.int32),
        jax.ShapeDtypeStruct((*signature_shape, 6), jnp.float32),
        jax.ShapeDtypeStruct((*signature_shape, 8), jnp.int32),
        jax.ShapeDtypeStruct((*signature_shape, 8), jnp.float32),
        jax.ShapeDtypeStruct((*signature_shape, 8), jnp.int32),
        jax.ShapeDtypeStruct(data_volume.shape, data_volume.dtype),
        jax.ShapeDtypeStruct(weight_volume.shape, weight_volume.dtype),
        jax.ShapeDtypeStruct(dense_data_rows.shape, dense_data_rows.dtype),
        jax.ShapeDtypeStruct(dense_weight_rows.shape, dense_weight_rows.dtype),
        jax.ShapeDtypeStruct(dense_indices.shape, dense_indices.dtype),
        jax.ShapeDtypeStruct(rot6.shape, rot6.dtype),
        jax.ShapeDtypeStruct(canonical_rotation_keys.shape, canonical_rotation_keys.dtype),
        jax.ShapeDtypeStruct(signature_row_indices.shape, signature_row_indices.dtype),
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_FUSED_X_HALF_BP_SIGNATURE,
        out_types,
        input_output_aliases={6: 0, 7: 1},
        vmap_method="sequential",
    )(
        dense_data_rows,
        dense_weight_rows,
        dense_indices,
        rot6,
        canonical_rotation_keys,
        signature_row_indices,
        data_volume,
        weight_volume,
        **kw,
    )


def _bitwise_array_equal(left, right) -> bool:
    """Compare diagnostic device snapshots without numeric tolerance."""

    left_np = np.asarray(left)
    right_np = np.asarray(right)
    return (
        left_np.shape == right_np.shape
        and left_np.dtype == right_np.dtype
        and left_np.tobytes(order="C") == right_np.tobytes(order="C")
    )


def _require_signature_inertness_outputs(outputs, expected_operands) -> None:
    """Fail closed unless signature capture is bitwise inert and input-exact."""

    accumulator_pairs = (
        ("data_accumulator", outputs[0], outputs[9]),
        ("weight_accumulator", outputs[1], outputs[10]),
    )
    operand_names = (
        "data_rows",
        "weight_rows",
        "pixel_indices",
        "rot6",
        "canonical_rotation_keys",
        "signature_row_indices",
    )
    operand_pairs = tuple(
        (name, expected, shadow)
        for name, expected, shadow in zip(operand_names, expected_operands, outputs[11:17], strict=True)
    )
    mismatches = [
        name
        for name, expected, observed in (*accumulator_pairs, *operand_pairs)
        if not _bitwise_array_equal(expected, observed)
    ]
    if mismatches:
        raise RuntimeError(
            "RELION fused x-half signature deterministic inertness gate failed for " + ", ".join(mismatches)
        )


def relion_fused_x_half_backproject_signature_indexed(
    data_volume: jax.Array,
    weight_volume: jax.Array,
    data_rows: jax.Array,
    weight_rows: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    canonical_rotation_keys: jax.Array,
    signature_row_indices: jax.Array,
    image_shape: Tuple[int, int],
    volume_shape: Tuple[int, int, int],
    max_r: float | None,
) -> tuple[jax.Array, ...]:
    """Validate contributor row selection before invoking the jitted FFI."""

    row_indices_np = np.asarray(signature_row_indices)
    n_rows = int(data_rows.shape[0])
    if (
        row_indices_np.ndim != 1
        or row_indices_np.dtype != np.dtype(np.int32)
        or row_indices_np.size == 0
        or np.any(row_indices_np < 0)
        or np.any(row_indices_np >= n_rows)
        or (row_indices_np.size > 1 and np.any(np.diff(row_indices_np) <= 0))
    ):
        raise ValueError(
            "RELION fused x-half signature row indices must be nonempty, unique, "
            "strictly increasing, and within the source-row range"
        )
    signature_row_indices_jax = jnp.asarray(row_indices_np, dtype=jnp.int32)
    outputs = _relion_fused_x_half_backproject_signature_indexed_impl(
        data_volume,
        weight_volume,
        data_rows,
        weight_rows,
        pixel_indices,
        rotation_matrices,
        canonical_rotation_keys,
        signature_row_indices_jax,
        image_shape,
        volume_shape,
        max_r,
    )
    dense_data_rows, dense_indices, _, _ = _prepare_relion_x_half_block_topology_operands(
        data_rows, pixel_indices, image_shape, max_r
    )
    dense_weight_rows, weight_dense_indices, _, _ = _prepare_relion_x_half_block_topology_operands(
        weight_rows, pixel_indices, image_shape, max_r
    )
    if not _bitwise_array_equal(dense_indices, weight_dense_indices):
        raise RuntimeError("RELION fused x-half signature prepared pixel indices disagree by operand")
    kernel_rotations = _relion_x_half_backproject_rotation_to_kernel(rotation_matrices, jnp.float32)
    rot6 = _rot_to_compact(kernel_rotations, jnp.float32)
    _require_signature_inertness_outputs(
        outputs,
        (
            dense_data_rows,
            dense_weight_rows,
            dense_indices,
            rot6,
            canonical_rotation_keys,
            signature_row_indices_jax,
        ),
    )
    # Shadow outputs are diagnostic gates only.  The public result retains the
    # established accumulator + seven signature-array contract.
    return outputs[:9]


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


@functools.partial(jax.jit, static_argnames=("image_shape", "padding_factor"))
def project_relion_half_capacity(
    projector_half: jax.Array,
    rotation_matrices: jax.Array,
    logical_r_max: jax.Array,
    *,
    image_shape: Tuple[int, int],
    padding_factor: int = 1,
    image_r_max: jax.Array | None = None,
) -> jax.Array:
    """Opt-in RELION texture projection with physical storage and runtime radius.

    ``projector_half`` is C64 [z,y,x>=0], centered on y/z, with physical
    shape (pf*Q+3, pf*Q+3, pf*Q//2+2). Center-pad the *logical* projector,
    including its ghost planes; rebuilding at the physical radius is not
    equivalent. Rotations use the same F32 matrix convention as ``project``.
    Radius is an S32 scalar in unpadded units, with 0 <= r <= Q//2. Invalid
    runtime values return NaNs without a host readback; producers must enforce
    this range. Physical geometry, rotations and output shape control caching.

    Returns C64 [rotation, H*(W//2+1)] with the existing texture projector's
    centered rows/positive Nyquist convention. Compact pixel gathering and
    current-image masks remain the existing caller's responsibility unless
    ``image_r_max`` is supplied. That S32 scalar applies RELION's rotated
    float32 radius test at min(logical_r_max, image_r_max), preserving model
    texture staging. Do not additionally apply an exact source-pixel disk:
    it rejects valid boundary contributions. Image radius must be between
    zero and H//2; invalid runtime values return NaNs. This
    transaction includes texture allocation, staging, synchronization and
    cleanup; there is no persistent resource cache or automatic fallback.
    """
    projector_half = jnp.asarray(projector_half)
    rotation_matrices = jnp.asarray(rotation_matrices)
    logical_r_max = jnp.asarray(logical_r_max)
    shape = projector_half.shape
    if padding_factor not in (1, 2):
        raise ValueError("padding_factor must be 1 or 2")
    if (
        projector_half.dtype != jnp.complex64
        or len(shape) != 3
        or shape[0] < 5
        or shape[0] > 1025
        or shape[0] != shape[1]
        or shape[0] % 2 != 1
        or shape[2] != shape[0] // 2 + 1
        or (shape[0] - 3) % (2 * padding_factor) != 0
    ):
        raise ValueError("projector_half must be C64 [pf*Q+3,pf*Q+3,pf*Q//2+2] for even Q")
    if (
        rotation_matrices.dtype != jnp.float32
        or rotation_matrices.ndim != 3
        or rotation_matrices.shape[1:] != (3, 3)
        or not 0 < rotation_matrices.shape[0] <= 65535
    ):
        raise ValueError("rotation_matrices must be nonempty F32 [rotation,3,3]")
    if logical_r_max.dtype != jnp.int32 or logical_r_max.shape != ():
        raise ValueError("logical_r_max must be an S32 scalar")
    if image_r_max is not None:
        image_r_max = jnp.asarray(image_r_max)
        if image_r_max.dtype != jnp.int32 or image_r_max.shape != ():
            raise ValueError("image_r_max must be an S32 scalar")
    if (
        len(image_shape) != 2
        or image_shape[0] != image_shape[1]
        or not 0 < image_shape[0] <= 4096
        or image_shape[0] % 2 != 0
    ):
        raise ValueError("image_shape must be a positive even square <=4096")
    n_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    if rotation_matrices.shape[0] * n_pixels > np.iinfo(np.int32).max:
        raise ValueError("projection output exceeds int32 kernel indexing")
    if image_r_max is None:
        _ensure_projector_capacity_ffi()
        target = _TARGET_PROJECT_RELION_HALF_RUNTIME
    else:
        _ensure_projector_image_radius_ffi()
        target = _TARGET_PROJECT_RELION_HALF_IMAGE_RADIUS
    rot6 = _rot_to_compact(rotation_matrices, jnp.float32)
    output = jax.ShapeDtypeStruct((rotation_matrices.shape[0], n_pixels), jnp.complex64)
    operands = (projector_half, rot6, logical_r_max)
    if image_r_max is not None:
        operands += (image_r_max,)
    return jax.ffi.ffi_call(
        target, output, vmap_method="sequential"
    )(
        *operands,
        image_h=np.int64(image_shape[0]),
        image_w=np.int64(image_shape[1]),
        padding_factor=np.int64(padding_factor),
    )


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


@jax.jit
def relion_wavg_rotation_atomic_f32(terms: jax.Array) -> jax.Array:
    """Reduce ``[batch, rotation, pixel]`` terms with RELION Wavg atomics."""

    _ensure_ffi()
    terms = jnp.asarray(terms)
    if terms.dtype != jnp.float32 or terms.ndim != 3:
        raise ValueError(
            "relion_wavg_rotation_atomic_f32 expects a float32 [batch, rotation, pixel] array"
        )
    output_type = jax.ShapeDtypeStruct((terms.shape[0], terms.shape[2]), jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_F32,
        output_type,
        vmap_method="sequential",
    )(terms)


@jax.jit
def relion_wavg_rotation_atomic_add_f32(
    terms: jax.Array,
    accumulator: jax.Array,
) -> jax.Array:
    """Atomically add ``[batch, rotation, pixel]`` terms into ``[batch, pixel]``."""

    _ensure_ffi()
    terms = jnp.asarray(terms)
    accumulator = jnp.asarray(accumulator)
    if terms.dtype != jnp.float32 or terms.ndim != 3:
        raise ValueError(
            "relion_wavg_rotation_atomic_add_f32 expects float32 [batch, rotation, pixel] terms"
        )
    if accumulator.dtype != jnp.float32 or accumulator.shape != (
        terms.shape[0],
        terms.shape[2],
    ):
        raise ValueError(
            "relion_wavg_rotation_atomic_add_f32 expects a matching float32 [batch, pixel] accumulator"
        )
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_ADD_F32,
        output_type,
        input_output_aliases={1: 0},
        vmap_method="sequential",
    )(terms, accumulator)


@jax.jit
def relion_wavg_rotation_atomic_triplet_add_f32(
    terms: jax.Array,
    accumulator: jax.Array,
) -> jax.Array:
    """Atomically add Wavg ``[XA, AA, diff2]`` terms in RELION source order."""

    _ensure_ffi()
    terms = jnp.asarray(terms)
    accumulator = jnp.asarray(accumulator)
    if terms.dtype != jnp.float32 or terms.ndim != 4 or terms.shape[-1] != 3:
        raise ValueError(
            "relion_wavg_rotation_atomic_triplet_add_f32 expects float32 "
            "[batch, rotation, pixel, 3] terms"
        )
    if accumulator.dtype != jnp.float32 or accumulator.shape != (
        terms.shape[0],
        terms.shape[2],
        3,
    ):
        raise ValueError(
            "relion_wavg_rotation_atomic_triplet_add_f32 expects a matching "
            "float32 [batch, pixel, 3] accumulator"
        )
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={1: 0},
        vmap_method="sequential",
    )(terms, accumulator)


def relion_wavg_native_prefix_f32(
    raw_rectangle,
    rectangle_image_power,
    projections,
    full_ctf,
    scale,
    posterior,
    exact_positions,
    recon_window_indices,
    logical_exact_count,
    logical_rectangle_count,
    *,
    debug=False,
):
    """Native exact-triplet/rectangle/atomic prefix, without host value reads.

    Inputs preserve packed B/R/T axes. Maps must be unique and in range, with
    capacity-only exact positions in the inert rectangle tail. Invalid runtime
    maps/counts produce NaNs, matching the runtime-kernel fail-closed convention.
    Debug additionally returns the actual preatomic rectangle buffer; production
    allocates that buffer only as stream-ordered private CUDA scratch.
    """
    if type(debug) is not bool:
        raise TypeError("debug must be a static bool")
    operands = (
        raw_rectangle,
        rectangle_image_power,
        projections,
        full_ctf,
        scale,
        posterior,
        exact_positions,
        recon_window_indices,
        logical_exact_count,
        logical_rectangle_count,
    )
    dtypes = (
        jnp.complex64,
        jnp.float32,
        jnp.complex64,
        jnp.float64,
        jnp.float32,
        jnp.float32,
        jnp.int32,
        jnp.int32,
        jnp.int32,
        jnp.int32,
    )
    for value, dtype in zip(operands, dtypes, strict=True):
        if not hasattr(value, "dtype") or np.dtype(value.dtype) != np.dtype(dtype):
            raise ValueError(f"native Wavg prefix requires operand dtype {np.dtype(dtype)}")
    if projections.ndim != 3 or raw_rectangle.ndim != 3:
        raise ValueError("projections/raw rectangle must have ranks 3")
    batch, rotations, exact = projections.shape
    raw_batch, translations, rectangle = raw_rectangle.shape
    if min(batch, rotations, exact, translations, rectangle) <= 0 or batch > 65535:
        raise ValueError("native Wavg prefix dimensions outside supported bounds")
    if full_ctf.ndim != 2 or full_ctf.shape[0] != batch or full_ctf.shape[1] <= 0:
        raise ValueError("full CTF must have shape [B,F]")
    if (
        raw_batch != batch
        or rectangle_image_power.shape != (batch, rotations, rectangle)
        or scale.shape != (batch,)
        or posterior.shape != (batch, rotations, translations)
        or exact_positions.shape != (exact,)
        or recon_window_indices.shape != (exact,)
        or logical_exact_count.shape != ()
        or logical_rectangle_count.shape != ()
    ):
        raise ValueError("native Wavg prefix operand geometry differs")
    _ensure_wavg_native_prefix_ffi()
    output = jax.ShapeDtypeStruct((batch, rectangle, 3), jnp.float32)
    if debug:
        result_types = (output, jax.ShapeDtypeStruct((batch, rotations, rectangle, 3), jnp.float32))
        target = _TARGET_RELION_WAVG_NATIVE_PREFIX_DEBUG_F32
    else:
        result_types, target = output, _TARGET_RELION_WAVG_NATIVE_PREFIX_F32
    return jax.ffi.ffi_call(target, result_types, vmap_method="sequential")(*operands)


@jax.jit
def relion_wavg_rotation_atomic_runtime_triplet_add_f32(
    terms: jax.Array,
    accumulator: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Run Wavg atomics over a logical prefix of fixed-capacity storage."""

    _ensure_ffi()
    terms = jnp.asarray(terms)
    accumulator = jnp.asarray(accumulator)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if terms.dtype != jnp.float32 or terms.ndim != 4 or terms.shape[-1] != 3:
        raise ValueError(
            "runtime RELION Wavg triplets require float32 [batch,rotation,pixel,3] terms"
        )
    if accumulator.dtype != jnp.float32 or accumulator.shape != (
        terms.shape[0],
        terms.shape[2],
        3,
    ):
        raise ValueError(
            "runtime RELION Wavg triplets require a matching [batch,pixel,3] accumulator"
        )
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    output_type = jax.ShapeDtypeStruct(accumulator.shape, jnp.float32)
    return jax.ffi.ffi_call(
        _TARGET_RELION_WAVG_ROTATION_ATOMIC_RUNTIME_TRIPLET_ADD_F32,
        output_type,
        input_output_aliases={1: 0},
        vmap_method="sequential",
    )(terms, accumulator, logical_pixel_count)


@jax.jit
def relion_wavg_sequential_triplet_f32(
    projections: jax.Array,
    raw_ctf: jax.Array,
    scale: jax.Array,
    shifted_images: jax.Array,
    posterior: jax.Array,
) -> jax.Array:
    """Accumulate RELION Wavg triplets in translation-storage order.

    One CUDA thread owns one ``[image, rotation, pixel]`` output and visits
    translations sequentially.  This preserves the float32 arithmetic and
    ordering of RELION's Wavg loop without lowering the translation loop to a
    sequence of separate XLA loop-body kernel launches.
    """

    _ensure_ffi()
    projections = jnp.asarray(projections)
    raw_ctf = jnp.asarray(raw_ctf)
    scale = jnp.asarray(scale)
    shifted_images = jnp.asarray(shifted_images)
    posterior = jnp.asarray(posterior)
    if projections.dtype != jnp.complex64 or projections.ndim != 3:
        raise ValueError(
            "relion_wavg_sequential_triplet_f32 expects complex64 "
            "projections[B,R,P]"
        )
    batch_size, rotation_count, pixel_count = projections.shape
    if raw_ctf.dtype != jnp.float32 or raw_ctf.shape != (batch_size, pixel_count):
        raise ValueError(
            "relion_wavg_sequential_triplet_f32 expects float32 raw_ctf[B,P]"
        )
    if scale.dtype != jnp.float32 or scale.shape != (batch_size,):
        raise ValueError(
            "relion_wavg_sequential_triplet_f32 expects float32 scale[B]"
        )
    if shifted_images.dtype != jnp.complex64 or shifted_images.ndim != 3:
        raise ValueError(
            "relion_wavg_sequential_triplet_f32 expects complex64 "
            "shifted_images[B,T,P]"
        )
    if shifted_images.shape[0] != batch_size or shifted_images.shape[2] != pixel_count:
        raise ValueError(
            "relion_wavg_sequential_triplet_f32 shifted-image batch/pixel axes "
            "must match projections"
        )
    translation_count = shifted_images.shape[1]
    if posterior.dtype != jnp.float32 or posterior.shape != (
        batch_size,
        rotation_count,
        translation_count,
    ):
        raise ValueError(
            "relion_wavg_sequential_triplet_f32 expects float32 posterior[B,R,T]"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("RELION Wavg sequential accumulation requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError(
            "RELION Wavg sequential accumulation was requested but custom CUDA is disabled"
        )

    output_type = jax.ShapeDtypeStruct(
        (batch_size, rotation_count, pixel_count, 3),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_WAVG_SEQUENTIAL_TRIPLET_F32,
        output_type,
        vmap_method="sequential",
    )(projections, raw_ctf, scale, shifted_images, posterior)


@jax.jit
def relion_wavg_sequential_runtime_triplet_f32(
    projections: jax.Array,
    raw_ctf: jax.Array,
    scale: jax.Array,
    shifted_images: jax.Array,
    posterior: jax.Array,
    logical_pixel_count: jax.Array,
) -> jax.Array:
    """Accumulate Wavg translations over a logical compact-prefix only."""

    _ensure_ffi()
    projections = jnp.asarray(projections)
    raw_ctf = jnp.asarray(raw_ctf)
    scale = jnp.asarray(scale)
    shifted_images = jnp.asarray(shifted_images)
    posterior = jnp.asarray(posterior)
    logical_pixel_count = jnp.asarray(logical_pixel_count, dtype=jnp.int32)
    if projections.dtype != jnp.complex64 or projections.ndim != 3:
        raise ValueError(
            "runtime RELION Wavg expects complex64 projections[B,R,P]"
        )
    batch_size, rotation_count, pixel_capacity = projections.shape
    if raw_ctf.dtype != jnp.float32 or raw_ctf.shape != (batch_size, pixel_capacity):
        raise ValueError("runtime RELION Wavg expects float32 raw_ctf[B,P]")
    if scale.dtype != jnp.float32 or scale.shape != (batch_size,):
        raise ValueError("runtime RELION Wavg expects float32 scale[B]")
    if (
        shifted_images.dtype != jnp.complex64
        or shifted_images.ndim != 3
        or shifted_images.shape[0] != batch_size
        or shifted_images.shape[2] != pixel_capacity
    ):
        raise ValueError("runtime RELION Wavg expects complex64 shifted_images[B,T,P]")
    if posterior.dtype != jnp.float32 or posterior.shape != (
        batch_size,
        rotation_count,
        shifted_images.shape[1],
    ):
        raise ValueError("runtime RELION Wavg expects float32 posterior[B,R,T]")
    if logical_pixel_count.shape != ():
        raise ValueError("logical_pixel_count must be an int32 scalar")
    if jax.default_backend() != "gpu":
        raise RuntimeError("runtime RELION Wavg requires a JAX GPU backend")
    if not custom_cuda_requested():
        raise RuntimeError("runtime RELION Wavg requires custom CUDA")
    output_type = jax.ShapeDtypeStruct(
        (batch_size, rotation_count, pixel_capacity, 3),
        jnp.float32,
    )
    return jax.ffi.ffi_call(
        _TARGET_RELION_WAVG_SEQUENTIAL_RUNTIME_TRIPLET_F32,
        output_type,
        vmap_method="sequential",
    )(
        projections,
        raw_ctf,
        scale,
        shifted_images,
        posterior,
        logical_pixel_count,
    )


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


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def project_indexed(
    volume: jax.Array,
    pixel_indices: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    order: int = 1,
    half_volume: bool = False,
    half_image: bool = False,
    max_r: float | None = None,
) -> jax.Array:
    """Project only the requested flattened image pixels.

    ``pixel_indices`` contains flattened pixel positions in the original full
    image grid, or the packed half-image grid when ``half_image=True``. The
    output stores those pixels compactly as ``(n_images, len(pixel_indices))``.
    """
    _ensure_ffi()
    _validate_inputs(volume_shape, image_shape, order, half_volume, half_image, max_r=max_r)
    kw, _, _ = _ffi_kwargs(image_shape, volume_shape, order, half_volume, half_image, max_r)
    pixel_indices = jnp.asarray(pixel_indices, dtype=jnp.int32).reshape(-1)
    n_images = rotation_matrices.shape[0]
    rot6 = _rot_to_compact(rotation_matrices, _volume_real_dtype(volume))
    out_type = jax.ShapeDtypeStruct((n_images, pixel_indices.shape[0]), volume.dtype)

    return jax.ffi.ffi_call(
        _TARGET_PROJECT_INDEXED,
        out_type,
        vmap_method="sequential",
    )(volume, pixel_indices, rot6, **kw)


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
def batch_backproject_interleaved(
    volumes: jax.Array,
    images: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    max_r: float | None = None,
) -> jax.Array:
    """Back-project images into interleaved volumes: output ``(half_vol, batch)``.

    Like ``batch_backproject`` but output is ``(n_voxels_half, batch_size)``
    instead of ``(batch_size, n_voxels_half)``.  All batch entries for the
    same voxel are contiguous → ~30× better L2 cache utilization for large
    batch sizes (e.g. 210 PPCA upper-tri channels).

    Only supports: real data (float32/64), half_volume=True, trilinear, full images.
    """
    _ensure_ffi()
    N0, N1, N2 = volume_shape
    H, W = image_shape
    ups = N0 // H
    max_r2_x4 = -1 if max_r is None else int(4 * max_r * max_r)
    # Ensure images dtype matches volumes (kernel dispatches on volume dtype)
    images = images.astype(volumes.dtype)
    rot6 = _rot_to_compact(rotation_matrices, volumes.dtype)
    out_type = jax.ShapeDtypeStruct(volumes.shape, volumes.dtype)

    return jax.ffi.ffi_call(
        _TARGET_BATCH_BP_INTERLEAVED,
        out_type,
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )(images, rot6, volumes, image_h=H, image_w=W, vol_n0=N0, vol_n1=N1, vol_n2=N2, upsampling=ups, max_r2_x4=max_r2_x4)


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def fused_backproject(
    volumes: jax.Array,
    base_images: jax.Array,
    weight_matrix: jax.Array,
    rotation_matrices: jax.Array,
    image_shape: Tuple[int, int] = (0, 0),
    volume_shape: Tuple[int, int, int] = (0, 0, 0),
    max_r: float | None = None,
) -> jax.Array:
    """Fused backproject: base_images × weight_matrix → interleaved volumes.

    Reads ``base_images[n, pix]`` (e.g. ctf²) and ``weight_matrix[n, ch]``
    (e.g. smz_tri) separately, multiplying inside the CUDA kernel.
    Eliminates the ``(n_ch, n_img, n_pix)`` intermediate tensor.

    Input bandwidth: ~50 MB vs ~3.4 GB for the unfused path at 256³.

    Parameters
    ----------
    volumes : ``(half_vol, n_channels)`` float32 — zero-initialized output
    base_images : ``(n_images, n_pixels)`` float32 — per-pixel per-image values
    weight_matrix : ``(n_images, n_channels)`` float32 — per-image per-channel weights
    rotation_matrices : ``(n_images, 3, 3)`` — shared rotations

    Returns ``(half_vol, n_channels)`` accumulated result.
    """
    _ensure_ffi()
    N0, N1, N2 = volume_shape
    H, W = image_shape
    ups = N0 // H
    max_r2_x4 = -1 if max_r is None else int(4 * max_r * max_r)
    base_images = base_images.astype(volumes.dtype)
    weight_matrix = weight_matrix.astype(volumes.dtype)
    rot6 = _rot_to_compact(rotation_matrices, volumes.dtype)
    out_type = jax.ShapeDtypeStruct(volumes.shape, volumes.dtype)

    return jax.ffi.ffi_call(
        _TARGET_FUSED_BP,
        out_type,
        input_output_aliases={3: 0},
        vmap_method="sequential",
    )(
        base_images,
        weight_matrix,
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
