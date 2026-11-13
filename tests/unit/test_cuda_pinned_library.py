"""A pinned CUDA library is never rebuilt; unpinned staleness is by source content; builds are atomic.

Two agents' jobs were broken when a fresh checkout (identical sources, newer mtimes) made the loader
run ``make -B LIB=<pinned path>`` over a library other processes were using. These tests pin the
loader contract for recovar's own library and for :class:`recovar.cuda_build.NativeLibrary`, with
fake sources and a fake ``make``; no CUDA is needed.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from recovar import cuda_backproject as cb
from recovar import cuda_build

pytestmark = pytest.mark.unit


def _no_build(*_args, **_kwargs):
    pytest.fail("the loader must not build here")


def _fake_make_writing(content):
    calls = []

    def fake_check_call(cmd, *, env):
        calls.append(cmd)
        pathlib.Path(cmd[-1].removeprefix("LIB=")).write_text(content)

    return fake_check_call, calls


# -- recovar's own library -------------------------------------------------------------------------


@pytest.fixture
def recovar_sources(tmp_path, monkeypatch):
    lib_dir = tmp_path / "cuda"
    lib_dir.mkdir()
    (lib_dir / "cuda_backproject.cu").write_text("// kernels\n")
    (lib_dir / "Makefile").write_text("# make\n")
    monkeypatch.setattr(cb, "_LIB_DIR", lib_dir)
    monkeypatch.setattr(cb, "_PACKAGE_LIB_PATH", lib_dir / "libcuda_backproject.so")
    monkeypatch.setenv("RECOVAR_CUDA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cb, "_auto_build_attempted", False)
    monkeypatch.setattr(cb, "_auto_build_error", None)
    return lib_dir


def test_pinned_recovar_library_with_symbols_is_used_without_any_source_check(tmp_path, monkeypatch, recovar_sources):
    pinned = tmp_path / "pinned" / "libcuda_backproject.so"
    pinned.parent.mkdir()
    pinned.write_text("pinned build")  # no digest recorded, sources "newer"
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(pinned))
    monkeypatch.setattr(cb, "_lib_missing_required_symbols", lambda _path: None)
    monkeypatch.setattr(cb.subprocess, "check_call", _no_build)

    assert cb._existing_lib_path() == pinned.resolve()
    assert pinned.read_text() == "pinned build"


def test_missing_pinned_recovar_library_raises_naming_it_and_does_not_build(tmp_path, monkeypatch, recovar_sources):
    pinned = tmp_path / "pinned" / "libcuda_backproject.so"
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(pinned))
    monkeypatch.setattr(cb.subprocess, "check_call", _no_build)

    with pytest.raises(RuntimeError, match=f"RECOVAR_CUDA_LIB={pinned} does not exist"):
        cb._existing_lib_path()
    assert not pinned.exists()


def test_pinned_recovar_library_lacking_a_symbol_raises_and_is_left_alone(tmp_path, monkeypatch, recovar_sources):
    pinned = tmp_path / "libcuda_backproject.so"
    pinned.write_text("old build")
    # A current cached build must not be substituted for the pinned one either.
    cached = cb._cached_lib_path()
    cached.parent.mkdir(parents=True)
    cached.write_text("cached build")
    cuda_build.digest_path(cached).write_text(cb._source_digest())
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(pinned))
    monkeypatch.setattr(cb, "_lib_missing_required_symbols", lambda path: None if path == cached else "Backproject")
    monkeypatch.setattr(cb.subprocess, "check_call", _no_build)

    with pytest.raises(RuntimeError, match="lacks the required symbol 'Backproject'"):
        cb._get_lib()
    assert pinned.read_text() == "old build"


def test_cuda_available_raises_for_an_unusable_pinned_library_instead_of_falling_back(
    tmp_path, monkeypatch, recovar_sources
):
    class FakeGpu:
        platform = "gpu"

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(tmp_path / "absent.so"))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cb.jax, "devices", lambda: [FakeGpu()])
    monkeypatch.setattr(cb, "_cuda_ok", None)
    monkeypatch.setattr(cb, "_ffi_registered", False)
    monkeypatch.setattr(cb.subprocess, "check_call", _no_build)

    with pytest.raises(cuda_build.PinnedLibraryError, match="RECOVAR_CUDA_LIB="):
        cb.cuda_available()


def test_unpinned_cache_is_rebuilt_atomically_once_sources_change(tmp_path, monkeypatch, recovar_sources):
    monkeypatch.delenv("RECOVAR_CUDA_LIB", raising=False)
    monkeypatch.setattr(cb, "_lib_missing_required_symbols", lambda _path: None)
    fake_make, calls = _fake_make_writing("first build")
    monkeypatch.setattr(cb.subprocess, "check_call", fake_make)

    built = cb._ensure_lib_path()
    assert built == cb._cached_lib_path().resolve() and built.read_text() == "first build"
    assert cb._existing_lib_path() == built  # current: no rebuild
    assert len(calls) == 1

    (recovar_sources / "cuda_backproject.cu").write_text("// kernels, fixed\n")
    assert cb._existing_lib_path() is None
    fake_make, calls = _fake_make_writing("second build")
    monkeypatch.setattr(cb.subprocess, "check_call", fake_make)
    with open(built) as reader:
        cb.build_custom_cuda(output_path=built)
        assert reader.read() == "first build"  # renamed over, not rewritten in place
    assert built.read_text() == "second build"
    assert sorted(p.name for p in built.parent.iterdir() if p.name.endswith(".tmp")) == []


def test_failed_build_leaves_target_and_no_temporaries(tmp_path, monkeypatch, recovar_sources):
    target = tmp_path / "out" / "libcuda_backproject.so"
    target.parent.mkdir()
    target.write_text("working build")

    def failing_make(cmd, *, env):
        pathlib.Path(cmd[-1].removeprefix("LIB=")).write_text("partial")
        raise cb.subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(cb.subprocess, "check_call", failing_make)
    with pytest.raises(cb.subprocess.CalledProcessError):
        cb.build_custom_cuda(output_path=target, force=True)
    assert target.read_text() == "working build"
    assert sorted(p.name for p in target.parent.iterdir()) == ["libcuda_backproject.so"]


# -- NativeLibrary (used by relax for librelax_cuda.so) ----------------------------------------------


@pytest.fixture
def native(tmp_path, monkeypatch):
    make_dir = tmp_path / "em_cuda"
    make_dir.mkdir()
    (make_dir / "kernels.cu").write_text("// em kernels\n")
    (make_dir / "Makefile").write_text("# make\n")
    monkeypatch.setenv("RECOVAR_CUDA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TEST_EM_CUDA_LIB", raising=False)
    library = cuda_build.NativeLibrary(
        name="test EM",
        filename="libtest_em.so",
        make_dir=make_dir,
        source_names=("kernels.cu", "Makefile"),
        lib_env="TEST_EM_CUDA_LIB",
        registrations=(("em_target", "EmSymbol"),),
        optional_registrations={},
    )
    monkeypatch.setattr(library, "missing_required_symbol", lambda _path: None)
    return library


def test_native_pinned_library_is_never_rebuilt(tmp_path, monkeypatch, native):
    pinned = tmp_path / "pinned" / "libtest_em.so"
    pinned.parent.mkdir()
    pinned.write_text("pinned")
    old = pinned.stat().st_mtime - 10_000
    os.utime(pinned, (old, old))  # sources look newer, as after a fresh checkout
    monkeypatch.setenv("TEST_EM_CUDA_LIB", str(pinned))
    monkeypatch.setattr(cuda_build.subprocess, "check_call", _no_build)

    assert native.existing_path() == pinned.resolve()
    assert native.ensure_path() == pinned.resolve()
    assert pinned.read_text() == "pinned"


def test_native_missing_or_incomplete_pinned_library_raises(tmp_path, monkeypatch, native):
    pinned = tmp_path / "libtest_em.so"
    monkeypatch.setenv("TEST_EM_CUDA_LIB", str(pinned))
    monkeypatch.setattr(cuda_build.subprocess, "check_call", _no_build)
    with pytest.raises(RuntimeError, match=f"TEST_EM_CUDA_LIB={pinned} does not exist"):
        native.get_lib()

    pinned.write_text("old")
    monkeypatch.setattr(native, "missing_required_symbol", lambda _path: "EmSymbol")
    with pytest.raises(RuntimeError, match="lacks the required symbol 'EmSymbol'"):
        native.ensure_path()
    assert pinned.read_text() == "old"


def test_native_unpinned_staleness_follows_source_content(monkeypatch, native):
    fake_make, calls = _fake_make_writing("built")
    monkeypatch.setattr(cuda_build.subprocess, "check_call", fake_make)
    built = native.ensure_path()
    assert built == native.cached_path().resolve() and built.read_text() == "built"
    assert calls[0][:3] == ["make", "-B", "-C"] and calls[0][-1] != f"LIB={built}"

    (native.make_dir / "kernels.cu").write_text("// em kernels\n")  # identical bytes, newer mtime
    assert native.existing_path() == built
    (native.make_dir / "kernels.cu").write_text("// em kernels, changed\n")
    assert native.is_stale(built) is True
    assert native.existing_path() is None
