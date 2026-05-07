"""CPU-only tests for `_cache_root` fallback when XDG_CACHE_HOME / HOME
are unwritable (issue #136).

The reporter's cluster has ``XDG_CACHE_HOME=/home/levans`` while the
real home is ``/mnt/home/levans`` — ``/home/levans`` exists but is not
writable, so ``recovar build_custom_cuda`` previously crashed with
``PermissionError``. ``_cache_root`` must now probe the candidate
directory and fall through to the next one.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from recovar import cuda_backproject


@pytest.fixture(autouse=True)
def _reset_cache_root_state(monkeypatch):
    """Each test starts with a clean warning flag and no inherited
    RECOVAR_CUDA_CACHE_DIR. Other env vars are set per test."""
    monkeypatch.setattr(cuda_backproject, "_CACHE_ROOT_FALLBACK_WARNED", False)
    monkeypatch.delenv("RECOVAR_CUDA_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def _make_unwritable_dir(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir()
    d.chmod(0o000)
    return d


def _restore(*paths: Path) -> None:
    """Restore 0o755 perms so pytest's tmp_path cleanup can rmtree."""
    for p in paths:
        try:
            p.chmod(0o755)
        except OSError:
            pass


def test_xdg_unwritable_falls_back_to_home(tmp_path, monkeypatch, caplog):
    """When XDG_CACHE_HOME points at a chmod-000 directory but HOME is
    writable, _cache_root() returns the HOME-based path and warns."""
    bad_xdg = _make_unwritable_dir(tmp_path, "xdg_unwritable")
    home = tmp_path / "home"
    home.mkdir()

    try:
        monkeypatch.setenv("XDG_CACHE_HOME", str(bad_xdg))
        monkeypatch.setenv("HOME", str(home))

        caplog.set_level(logging.WARNING, logger=cuda_backproject.logger.name)
        result = cuda_backproject._cache_root()

        # 1. Returned path is under HOME, not the bad XDG dir
        assert str(result).startswith(str(home)), result
        assert str(bad_xdg) not in str(result), result
        assert result == home / ".cache" / "recovar" / "cuda"

        # 2. The chosen path is actually writable
        assert cuda_backproject._path_is_writable(result)

        # 3. A WARNING was emitted naming the bad XDG path and the
        #    RECOVAR_CUDA_CACHE_DIR env var
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "expected a fallback warning"
        msg = warnings[0].getMessage()
        assert "XDG_CACHE_HOME" in msg
        assert str(bad_xdg) in msg
        assert "RECOVAR_CUDA_CACHE_DIR" in msg
    finally:
        _restore(bad_xdg)


def test_override_wins_over_unwritable_xdg(tmp_path, monkeypatch, caplog):
    """RECOVAR_CUDA_CACHE_DIR is the explicit user override and must be
    returned as-is, regardless of XDG state — no probe, no warning."""
    bad_xdg = _make_unwritable_dir(tmp_path, "xdg_unwritable")
    override = tmp_path / "explicit" / "cuda"

    try:
        monkeypatch.setenv("XDG_CACHE_HOME", str(bad_xdg))
        monkeypatch.setenv("RECOVAR_CUDA_CACHE_DIR", str(override))

        caplog.set_level(logging.WARNING, logger=cuda_backproject.logger.name)
        result = cuda_backproject._cache_root()

        assert result == override
        # Override wins even when its parent is missing — no probe runs
        assert not result.exists()
        # No fallback warning fired
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, f"unexpected fallback warnings: {warnings}"
    finally:
        _restore(bad_xdg)


def test_both_xdg_and_home_unwritable_falls_back_to_tempdir(tmp_path, monkeypatch, caplog):
    """If neither XDG nor HOME is writable, fall back to
    `tempfile.gettempdir()/recovar_cuda_cache/cuda` and emit a WARNING
    naming both rejected candidates."""
    bad_xdg = _make_unwritable_dir(tmp_path, "xdg_unwritable")
    bad_home = _make_unwritable_dir(tmp_path, "home_unwritable")

    try:
        monkeypatch.setenv("XDG_CACHE_HOME", str(bad_xdg))
        monkeypatch.setenv("HOME", str(bad_home))

        caplog.set_level(logging.WARNING, logger=cuda_backproject.logger.name)
        result = cuda_backproject._cache_root()

        expected = Path(tempfile.gettempdir()) / "recovar_cuda_cache" / "cuda"
        assert result == expected, result
        # Tempdir fallback is itself writable
        assert cuda_backproject._path_is_writable(result)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "expected a fallback warning"
        msg = warnings[0].getMessage()
        assert "XDG_CACHE_HOME" in msg
        assert "HOME" in msg
        assert str(expected) in msg
        assert "RECOVAR_CUDA_CACHE_DIR" in msg
    finally:
        _restore(bad_xdg, bad_home)


def test_path_is_writable_handles_chmod000_existing_dir(tmp_path):
    """The probe correctly rejects an existing-but-unwritable directory
    (the exact failure mode in #136 where /home/levans existed but was
    not writable)."""
    bad = _make_unwritable_dir(tmp_path, "bad")
    try:
        # Try a subpath that doesn't exist yet — mkdir(parents=True)
        # should fail at the chmod-000 parent
        target = bad / "recovar" / "cuda"
        assert not cuda_backproject._path_is_writable(target)
    finally:
        _restore(bad)

    # Sanity: a fresh writable directory passes
    good = tmp_path / "good" / "recovar" / "cuda"
    assert cuda_backproject._path_is_writable(good)
    assert good.is_dir()
    # Probe file was cleaned up
    assert not (good / ".recovar_write_test").exists()
