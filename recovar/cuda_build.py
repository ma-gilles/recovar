"""Build, locate, load and register a custom CUDA XLA-FFI library (relax split seam S3).

recovar's pipeline library ``libcuda_backproject.so`` is managed by :mod:`recovar.cuda_backproject`.
Other packages that ship their own translation unit (the EM library ``librelax_cuda.so``) use
:class:`NativeLibrary`, which follows the same rules through the same helpers: cache directory
resolution, the build lock, ``make -C <dir> LIB=<path>``, staleness from a source content hash and
required FFI symbols, the GPU architecture preflight and once-per-process FFI registration.  Headers
shared by both translation units live in :func:`include_dir`.

A library selected explicitly through the library's environment variable (``RECOVAR_CUDA_LIB``,
``RELAX_CUDA_LIB``) is pinned: it is used if it exists and exports the required symbols, otherwise
loading fails with an error naming the file.  It is never rebuilt automatically, because another
process may be running from it.  Unpinned libraries are rebuilt when the sha256 of their build inputs
differs from the one recorded beside the library (``<library>.sources.sha256``) at build time; source
mtimes are ignored, since a fresh checkout makes identical sources look newer.  Every build writes a
temporary file in the target directory and renames it over the target, so a library is never
rewritten in place.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence

import jax

logger = logging.getLogger(__name__)


def include_dir() -> pathlib.Path:
    """Directory of the public CUDA headers (``recovar_cuda_common.cuh``, ``device_scratch.cuh``)."""

    return pathlib.Path(__file__).resolve().parent / "cuda" / "include"


SOURCE_DIGEST_SUFFIX = ".sources.sha256"


def source_digest(sources: Sequence[tuple[str, pathlib.Path]]) -> str:
    """sha256 over the ``(name, path)`` build inputs, in order; a missing input raises ``OSError``."""

    digest = hashlib.sha256()
    for name, path in sources:
        data = pathlib.Path(path).read_bytes()
        digest.update(name.encode() + b"\0" + len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def digest_path(lib_path: pathlib.Path) -> pathlib.Path:
    """File beside ``lib_path`` that records the source digest the library was built from."""

    lib_path = pathlib.Path(lib_path)
    return lib_path.with_name(lib_path.name + SOURCE_DIGEST_SUFFIX)


def built_from(lib_path: pathlib.Path, digest: str) -> bool:
    """True when ``lib_path`` records ``digest`` as its build inputs' hash."""

    try:
        return digest_path(lib_path).read_text().strip() == digest
    except OSError:
        return False


class PinnedLibraryError(RuntimeError):
    """A library pinned through its environment variable is missing or lacks a required symbol."""


def check_pinned(lib_path: pathlib.Path, lib_env: str, missing_symbol) -> pathlib.Path:
    """Return a pinned library's resolved path, or raise naming the file; never builds it.

    ``missing_symbol(path)`` returns the first required symbol the library lacks, or ``None``.
    """

    lib_path = pathlib.Path(lib_path).expanduser()
    if not lib_path.is_file():
        raise PinnedLibraryError(
            f"{lib_env}={lib_path} does not exist. A library selected through {lib_env} is never built "
            f"automatically; build it explicitly (make -C <cuda dir> LIB={lib_path}) or unset {lib_env}."
        )
    missing = missing_symbol(lib_path)
    if missing is not None:
        raise PinnedLibraryError(
            f"{lib_env}={lib_path} lacks the required symbol {missing!r} (built from older sources?). "
            f"A library selected through {lib_env} is never rebuilt automatically; rebuild it explicitly "
            f"or point {lib_env} at a current build."
        )
    return lib_path.resolve()


def make_atomically(make_cmd: Sequence[str], lib_path: pathlib.Path, env: Mapping[str, str], digest: str) -> None:
    """Run ``make_cmd + [LIB=<temporary>]`` in ``lib_path``'s directory and rename the result over it.

    The source digest is recorded beside the library the same way, so a reader sees either the old pair
    or the new library; the target file itself is never opened for writing.
    """

    lib_path = pathlib.Path(lib_path)
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{lib_path.name}.", suffix=".tmp", dir=lib_path.parent)
    os.close(fd)
    tmp = pathlib.Path(tmp_name)
    tmp.unlink()  # make must create it; an existing empty target would look up to date
    tmp_digest = digest_path(tmp)
    try:
        subprocess.check_call([*make_cmd, f"LIB={tmp}"], env=dict(env))
        if not tmp.is_file():
            raise RuntimeError(f"Build failed — {tmp} not found (target {lib_path})")
        tmp_digest.write_text(digest + "\n")
        os.replace(tmp, lib_path)
        os.replace(tmp_digest, digest_path(lib_path))
    finally:
        for leftover in (tmp, tmp_digest):
            leftover.unlink(missing_ok=True)


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
        Build inputs, relative to ``make_dir`` or absolute (e.g. under :func:`include_dir`); an unpinned
        library whose recorded digest of them differs is stale, and a missing input is an error.
    lib_env:
        Environment variable that selects an explicit, pinned library path (never rebuilt).
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

        return cuda_backproject.cache_root() / self.filename

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

    def source_digest(self) -> str:
        return source_digest([(name, self.make_dir / name) for name in self.source_names])

    def is_stale(self, lib_path: pathlib.Path) -> bool:
        if not lib_path.exists():
            return False
        if not built_from(lib_path, self.source_digest()):
            logger.info(
                "%s CUDA library %s was not built from the current sources — will rebuild.", self.name, lib_path
            )
            return True
        missing = self.missing_required_symbol(lib_path)
        if missing is not None:
            logger.info("%s CUDA library %s is missing symbol '%s' — will rebuild.", self.name, lib_path, missing)
            return True
        return False

    def existing_path(self) -> pathlib.Path | None:
        """The pinned library (checked, never rebuilt), else the first current unpinned candidate."""

        configured = self.configured_path()
        if configured is not None:
            return check_pinned(configured, self.lib_env, self.missing_required_symbol)
        for candidate in self.candidate_paths():
            if candidate.exists() and not self.is_stale(candidate):
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
        make_cmd = [
            "make",
            "-B",
            "-C",
            str(self.make_dir),
            f"PYTHON={sys.executable}",
            f"RECOVAR_CUDA_INCLUDE={include_dir()}",
        ]
        make_atomically(make_cmd, lib_path, make_env, self.source_digest())
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
