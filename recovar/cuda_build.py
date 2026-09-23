"""Build, locate, load and register a custom CUDA XLA-FFI library (relax split seam S3).

recovar's pipeline library ``libcuda_backproject.so`` is managed by :mod:`recovar.cuda_backproject`.
Other packages that ship their own translation unit (the EM library ``librelax_cuda.so``) use
:class:`NativeLibrary`, which follows the same rules through the same helpers: cache directory
resolution, the build lock, ``make -C <dir> LIB=<path>``, staleness from source mtimes and required
FFI symbols, the GPU architecture preflight and once-per-process FFI registration.  Headers shared by
both translation units live in :func:`include_dir`.
"""

from __future__ import annotations

import ctypes
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence

import jax

logger = logging.getLogger(__name__)


def include_dir() -> pathlib.Path:
    """Directory of the public CUDA headers (``recovar_cuda_common.cuh``, ``device_scratch.cuh``)."""

    return pathlib.Path(__file__).resolve().parent / "cuda" / "include"


class NativeLibrary:
    """One custom CUDA shared library and its XLA FFI targets.

    Parameters
    ----------
    name:
        Human-readable library name used in messages.
    filename:
        Shared-library file name, e.g. ``librelax_cuda.so``.
    make_dir:
        Directory holding the library's Makefile; builds run ``make -C make_dir LIB=<path>``.
    source_names:
        Build inputs relative to ``make_dir``; a library older than any of them is stale.
    lib_env:
        Environment variable that selects an explicit library path.
    registrations:
        ``(ffi_target, exported_symbol)`` pairs registered eagerly by :meth:`ensure_ffi`; a library
        lacking any of these symbols is stale.
    optional_registrations:
        ``ffi_target -> (exported_symbol, error_message)`` registered on first request only.
    """

    def __init__(
        self,
        *,
        name: str,
        filename: str,
        make_dir: pathlib.Path,
        source_names: Sequence[str],
        lib_env: str,
        registrations: Sequence[tuple[str, str]],
        optional_registrations: Mapping[str, tuple[str, str]],
    ) -> None:
        self.name = name
        self.filename = filename
        self.make_dir = pathlib.Path(make_dir)
        self.source_names = tuple(source_names)
        self.lib_env = lib_env
        self.registrations = tuple(registrations)
        self.optional_registrations = dict(optional_registrations)
        self.handle = None
        self.loaded_path: pathlib.Path | None = None
        self.ffi_registered = False
        self.optional_registered: set[str] = set()
        self.auto_build_attempted = False
        self.auto_build_error: BaseException | None = None
        self._auto_build_lock = threading.Lock()
        self.ffi_lock = threading.Lock()

    # -- locating ---------------------------------------------------------------------------------

    def cached_path(self) -> pathlib.Path:
        from recovar import cuda_backproject

        return cuda_backproject._cache_root() / self.filename

    def configured_path(self) -> pathlib.Path | None:
        override = os.environ.get(self.lib_env)
        return pathlib.Path(override).expanduser() if override else None

    def candidate_paths(self) -> list[pathlib.Path]:
        out = []
        for path in (self.configured_path(), self.cached_path(), self.make_dir / self.filename):
            if path is not None and path.expanduser() not in out:
                out.append(path.expanduser())
        return out

    def missing_required_symbol(self, lib_path: pathlib.Path) -> str | None:
        try:
            lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_LOCAL)
        except OSError:
            return "<dlopen failed>"
        for _target, symbol in self.registrations:
            if not hasattr(lib, symbol):
                return symbol
        return None

    def is_stale(self, lib_path: pathlib.Path) -> bool:
        try:
            lib_mtime = lib_path.stat().st_mtime
        except OSError:
            return False
        for src_name in self.source_names:
            try:
                if (self.make_dir / src_name).stat().st_mtime > lib_mtime:
                    return True
            except OSError:
                continue
        missing = self.missing_required_symbol(lib_path)
        if missing is not None:
            logger.info("%s CUDA library %s is missing symbol '%s' — will rebuild.", self.name, lib_path, missing)
            return True
        return False

    def existing_path(self) -> pathlib.Path | None:
        for candidate in self.candidate_paths():
            if candidate.exists():
                if self.is_stale(candidate):
                    logger.info("%s CUDA library %s is older than its source — will rebuild.", self.name, candidate)
                    continue
                return candidate.resolve()
        return None

    def default_build_path(self) -> pathlib.Path:
        configured = self.configured_path()
        return (configured if configured is not None else self.cached_path()).expanduser().resolve()

    # -- building ---------------------------------------------------------------------------------

    def build(self, output_path: str | os.PathLike[str] | None = None, force: bool = False) -> pathlib.Path:
        """Build the library (``make -C make_dir``) and return its path."""
        from recovar import cuda_backproject

        lib_path = pathlib.Path(output_path).expanduser().resolve() if output_path else self.default_build_path()
        stale = lib_path.exists() and self.is_stale(lib_path)
        if lib_path.exists() and not force and not stale:
            self.auto_build_attempted = True
            self.auto_build_error = None
            return lib_path
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Building %s", lib_path)
        make_env = os.environ.copy()
        nvcc_visible = (
            make_env.get("NVCC")
            or make_env.get("CUDACXX")
            or shutil.which("nvcc")
            or any(
                os.access(os.path.join(make_env.get(var, ""), "bin", "nvcc"), os.X_OK)
                for var in ("LOCAL_CUDA_PATH", "CUDA_HOME", "CUDA_PATH")
                if make_env.get(var)
            )
        )
        if not nvcc_visible:
            discovered = cuda_backproject._discover_system_nvcc()
            if discovered is not None:
                make_env["NVCC"] = discovered
        make_cmd = ["make"]
        if force or stale:
            make_cmd.append("-B")
        make_cmd += [
            "-C",
            str(self.make_dir),
            f"PYTHON={sys.executable}",
            f"LIB={lib_path}",
            f"RECOVAR_CUDA_INCLUDE={include_dir()}",
        ]
        subprocess.check_call(make_cmd, env=make_env)
        if not lib_path.exists():
            raise RuntimeError(f"Build failed — {lib_path} not found")
        self.auto_build_attempted = True
        self.auto_build_error = None
        self.ffi_registered = False
        self.handle = None
        self.loaded_path = None
        return lib_path

    def ensure_path(self) -> pathlib.Path | None:
        from recovar import cuda_backproject

        existing = self.existing_path()
        if existing is not None:
            return existing
        target = self.default_build_path()
        with self._auto_build_lock:
            existing = self.existing_path()
            if existing is not None:
                return existing
            if self.auto_build_attempted:
                return None
            self.auto_build_attempted = True
            with cuda_backproject._build_file_lock(target.parent / cuda_backproject._BUILD_LOCKFILE):
                existing = self.existing_path()
                if existing is not None:
                    self.auto_build_error = None
                    return existing
                try:
                    built = self.build(output_path=target)
                except Exception as exc:  # pragma: no cover - exercised in GPU envs
                    self.auto_build_error = exc
                    logger.debug("Automatic %s CUDA build failed", self.name, exc_info=True)
                    return None
            self.auto_build_error = None
            return built.resolve()

    def unavailable_error(self, exc: BaseException | None = None) -> RuntimeError:
        searched = ", ".join(str(p) for p in self.candidate_paths())
        detail = f" Last error: {exc!s}." if exc is not None else ""
        return RuntimeError(
            f"The {self.name} custom CUDA library is unavailable.{detail} Set {self.lib_env} to a built "
            f"library or fix the CUDA compiler setup so it can be built automatically. Searched: {searched}"
        )

    # -- loading and registration -----------------------------------------------------------------

    def get_lib(self):
        if self.ffi_registered and self.handle is not None and self.loaded_path is not None:
            configured = self.configured_path()
            if configured is not None and configured.resolve() != self.loaded_path:
                raise RuntimeError(
                    f"{self.lib_env}={configured} asks for a different CUDA library than the one this "
                    f"process's XLA FFI handlers are bound to ({self.loaded_path}); restart the process to switch"
                )
            return self.handle
        lib_path = self.existing_path() or self.ensure_path()
        if lib_path is None:
            raise self.unavailable_error(self.auto_build_error)
        lib_path = pathlib.Path(lib_path).resolve()
        if self.handle is None or self.loaded_path != lib_path:
            self.handle = ctypes.CDLL(str(lib_path))
            self.loaded_path = lib_path
        return self.handle

    def ensure_ffi(self) -> None:
        from recovar import cuda_backproject

        if self.ffi_registered:
            return
        with self.ffi_lock:
            if self.ffi_registered:
                return
            lib = self.get_lib()
            if self.loaded_path:
                cuda_backproject._preflight_check(pathlib.Path(self.loaded_path))
            for target, symbol in self.registrations:
                jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(getattr(lib, symbol)), platform="CUDA")
            self.ffi_registered = True
            logger.debug("Registered %s CUDA FFI targets", self.name)

    def ensure_optional_ffi(self, target: str) -> None:
        self.ensure_ffi()
        if target in self.optional_registered:
            return
        with self.ffi_lock:
            if target in self.optional_registered:
                return
            symbol_name, error = self.optional_registrations[target]
            symbol = getattr(self.get_lib(), symbol_name, None)
            if symbol is None:
                raise RuntimeError(error)
            jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(symbol), platform="CUDA")
            self.optional_registered.add(target)
