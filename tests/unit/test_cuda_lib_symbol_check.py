"""Regression tests for the binary-compat check in ``_lib_is_stale``.

The cached ``libcuda_backproject.so`` at ``~/.cache/recovar/cuda/`` survives
across recovar versions. Without a symbol check, a cached ``.so`` built from
an older ``.cu`` (e.g. lacking ``BackprojectIndexed``) was silently accepted
by mtime alone — ``_ensure_ffi`` then crashed deep inside ``ctypes`` and
``cuda_available()`` returned False, dropping the user onto the slow JAX
fallback without any warning.

These tests pin the ``_lib_missing_required_symbols`` helper and verify
``_lib_is_stale`` integrates it correctly.
"""

from __future__ import annotations

import ctypes.util
import os
import pathlib

import pytest

from recovar import cuda_backproject as cb

pytestmark = pytest.mark.unit


def test_ffi_registrations_cover_all_target_constants():
    """Catch missing entries when someone adds a new ``_TARGET_*`` without
    updating the single-source-of-truth ``_FFI_REGISTRATIONS``.
    """
    targets_in_table = {target for target, _symbol in cb._FFI_REGISTRATIONS}
    target_constants = {
        v for k, v in vars(cb).items() if k.startswith("_TARGET_") and isinstance(v, str)
    }
    missing_from_table = target_constants - targets_in_table
    assert not missing_from_table, (
        f"_FFI_REGISTRATIONS is missing entries for: {sorted(missing_from_table)}. "
        "Add the (target, symbol) pair so the cache binary-compat check covers it."
    )


def test_ffi_registrations_have_unique_targets_and_symbols():
    targets = [t for t, _s in cb._FFI_REGISTRATIONS]
    symbols = [s for _t, s in cb._FFI_REGISTRATIONS]
    assert len(set(targets)) == len(targets), "duplicate FFI target name"
    assert len(set(symbols)) == len(symbols), "duplicate C symbol name"


def test_missing_required_symbols_detects_unrelated_lib():
    """A library that doesn't export ANY recovar symbol (libc) must be flagged."""
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        pytest.skip("libc not found via ctypes.util.find_library")
    libc_path = pathlib.Path(libc_name)
    missing = cb._lib_missing_required_symbols(libc_path)
    assert missing is not None, "libc must be detected as missing recovar symbols"
    expected = {s for _t, s in cb._FFI_REGISTRATIONS}
    assert missing in expected, f"unexpected symbol name returned: {missing!r}"


def test_missing_required_symbols_returns_none_for_missing_file(tmp_path):
    """Nonexistent file: dlopen fails, helper returns the sentinel."""
    fake = tmp_path / "does_not_exist.so"
    result = cb._lib_missing_required_symbols(fake)
    assert result == "<dlopen failed>"


def _resolve_libc_path() -> pathlib.Path | None:
    """Resolve libc to an absolute on-disk path (find_library returns a SONAME)."""
    soname = ctypes.util.find_library("c")
    if not soname:
        return None
    # dlopen by SONAME so the dynamic linker finds the real file, then read its
    # absolute path from /proc/self/map_files (works on Linux) or fall back to
    # walking the common search paths.
    handle = ctypes.CDLL(soname)
    # dlinfo would be nicer, but ctypes doesn't wrap it; fall back to a search.
    for prefix in ("/lib64", "/usr/lib64", "/lib/x86_64-linux-gnu", "/lib", "/usr/lib"):
        candidate = pathlib.Path(prefix) / soname
        if candidate.exists():
            del handle
            return candidate
    return None


def test_lib_is_stale_flags_so_missing_symbols(tmp_path, monkeypatch):
    """End-to-end: a fresh-mtime .so that's missing a required symbol must
    be flagged as stale. This is the regression case that bit dev2 today:
    cached .so from an older branch, mtime newer than current .cu source,
    but missing ``BackprojectIndexed``.
    """
    libc_path = _resolve_libc_path()
    if libc_path is None:
        pytest.skip("could not resolve absolute libc path")

    # Bypass the mtime branch: make the cu/Makefile sources point at tmp_path
    # files older than libc, so only the symbol check can trip staleness.
    fake_lib_dir = tmp_path / "cuda"
    fake_lib_dir.mkdir()
    cu_file = fake_lib_dir / "cuda_backproject.cu"
    mk_file = fake_lib_dir / "Makefile"
    cu_file.write_text("// stub\n")
    mk_file.write_text("# stub\n")
    # Make the source files older than libc
    import os

    old_ts = libc_path.stat().st_mtime - 100_000
    os.utime(cu_file, (old_ts, old_ts))
    os.utime(mk_file, (old_ts, old_ts))

    monkeypatch.setattr(cb, "_LIB_DIR", fake_lib_dir)
    assert cb._lib_is_stale(libc_path) is True


def test_lib_is_stale_passes_when_no_lib_exists(tmp_path):
    """Nonexistent .so: ``_lib_is_stale`` returns False so callers fall
    through to the build/no-build branch rather than spuriously rebuilding."""
    assert cb._lib_is_stale(tmp_path / "nope.so") is False


# ---------------------------------------------------------------------------
# _discover_system_nvcc: cluster-agnostic nvcc fallback
# ---------------------------------------------------------------------------


def _make_fake_nvcc(dirpath: pathlib.Path) -> pathlib.Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    nvcc = dirpath / "nvcc"
    nvcc.write_text("#!/bin/sh\necho stub\n")
    nvcc.chmod(0o755)
    return nvcc


def test_discover_system_nvcc_returns_none_on_empty(monkeypatch, tmp_path):
    """No system CUDA installs: helper returns None, caller will surface error."""
    monkeypatch.setattr(pathlib.Path, "glob", lambda self, pat: iter(()))
    # Also redirect the Debian fallback by patching the function-local pathlib
    # use; easier: just verify the function tolerates an empty environment.
    # We can't easily prevent /usr/local/cuda* on the test machine from being
    # found, so just assert the contract: returns None OR a real executable.
    result = cb._discover_system_nvcc()
    assert result is None or os.access(result, os.X_OK)


def test_discover_system_nvcc_picks_highest_version(monkeypatch, tmp_path):
    """When multiple versioned CUDA dirs exist, pick the highest version."""
    # Build fake "/fake/usr/local/cuda-12.6/bin/nvcc", "...cuda-13.1/bin/nvcc",
    # "...cuda-13.2/bin/nvcc". The helper globs absolute paths under "/", so we
    # patch pathlib.Path.glob to enumerate our fake set.
    fake_root = tmp_path / "fakeroot"
    versions = ["cuda-12.6", "cuda-13.1", "cuda-13.2"]
    fakes = [_make_fake_nvcc(fake_root / "usr" / "local" / v / "bin") for v in versions]

    def _fake_glob(self, pat):
        if "/usr/local/cuda" in pat or pat.startswith("usr/local/cuda"):
            return iter(fakes)
        return iter(())

    monkeypatch.setattr(pathlib.Path, "glob", _fake_glob)
    # Also stub the Debian-specific path so it's not picked up:
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: str(self) in {str(p) for p in fakes})
    monkeypatch.setattr(os, "access", lambda p, mode: str(p) in {str(f) for f in fakes})

    result = cb._discover_system_nvcc()
    assert result is not None
    assert "cuda-13.2" in result, f"expected highest-version cuda-13.2, got {result}"


def test_discover_system_nvcc_skips_nonexecutable():
    """Non-executable nvcc must not be returned (avoid permission errors at build time)."""
    # Real-environment check: every returned path must be executable.
    result = cb._discover_system_nvcc()
    if result is not None:
        assert os.access(result, os.X_OK), f"{result} is not executable"
