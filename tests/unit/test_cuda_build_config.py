import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAKE = shutil.which("make")
_MAKEFILE = _REPO_ROOT / "recovar" / "cuda" / "Makefile"
_CUDA_DIR = _MAKEFILE.parent
_SYSTEM_PATH = [Path("/usr/bin"), Path("/bin")]


@pytest.fixture(autouse=True)
def _reset_cuda_backproject_module_state():
    import recovar.cuda_backproject as cb

    cb._auto_build_attempted = False
    cb._auto_build_error = None
    cb._cuda_ok = None
    cb._ffi_registered = False
    cb._lib_handle = None
    cb._loaded_lib_path = None
    yield
    cb._auto_build_attempted = False
    cb._auto_build_error = None
    cb._cuda_ok = None
    cb._ffi_registered = False
    cb._lib_handle = None
    cb._loaded_lib_path = None


def _write_fake_executable(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _make_env(path_entries, **overrides):
    env = {}
    for key in ("HOME", "LANG", "LC_ALL", "TERM"):
        if key in os.environ:
            env[key] = os.environ[key]
    env["PATH"] = os.pathsep.join(str(entry) for entry in [*path_entries, *_SYSTEM_PATH])
    env["PYTHON"] = sys.executable
    env["PYTHONNOUSERSITE"] = "1"
    env.update({key: value for key, value in overrides.items() if value is not None})
    return env


def _make_wrapper(tmp_path: Path):
    wrapper = tmp_path / "print_nvcc.mk"
    wrapper.write_text(
        f"include {_MAKEFILE}\n\n"
        "print-nvcc:\n"
        "\t@printf '%s\\n' '$(NVCC)'\n"
    )
    return wrapper


def _run_make(wrapper: Path, *targets: str, env: dict[str, str]):
    assert _MAKE is not None
    return subprocess.run(
        [_MAKE, "-f", str(wrapper), *targets],
        cwd=_CUDA_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_makefile_prefers_cudacxx_over_path_and_roots(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    fake_path_nvcc = _write_fake_executable(tmp_path / "path-bin" / "nvcc")
    fake_local_nvcc = _write_fake_executable(tmp_path / "local-cuda" / "bin" / "nvcc")
    fake_home_nvcc = _write_fake_executable(tmp_path / "cuda-home" / "bin" / "nvcc")
    fake_path_root_nvcc = _write_fake_executable(tmp_path / "cuda-path" / "bin" / "nvcc")
    fake_cudacxx = _write_fake_executable(tmp_path / "custom-bin" / "my-nvcc")

    env = _make_env(
        [fake_path_nvcc.parent],
        CUDACXX=str(fake_cudacxx),
        LOCAL_CUDA_PATH=str(fake_local_nvcc.parents[1]),
        CUDA_HOME=str(fake_home_nvcc.parents[1]),
        CUDA_PATH=str(fake_path_root_nvcc.parents[1]),
    )

    result = _run_make(wrapper, "print-nvcc", env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(fake_cudacxx)


def test_makefile_prefers_path_over_root_fallbacks(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    fake_path_nvcc = _write_fake_executable(tmp_path / "path-bin" / "nvcc")
    fake_local_nvcc = _write_fake_executable(tmp_path / "local-cuda" / "bin" / "nvcc")
    fake_home_nvcc = _write_fake_executable(tmp_path / "cuda-home" / "bin" / "nvcc")
    fake_path_root_nvcc = _write_fake_executable(tmp_path / "cuda-path" / "bin" / "nvcc")

    env = _make_env(
        [fake_path_nvcc.parent],
        LOCAL_CUDA_PATH=str(fake_local_nvcc.parents[1]),
        CUDA_HOME=str(fake_home_nvcc.parents[1]),
        CUDA_PATH=str(fake_path_root_nvcc.parents[1]),
    )

    result = _run_make(wrapper, "print-nvcc", env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(fake_path_nvcc)


def test_makefile_root_fallback_order_prefers_local_cuda_path_then_cuda_home(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    fake_local_nvcc = _write_fake_executable(tmp_path / "local-cuda" / "bin" / "nvcc")
    fake_home_nvcc = _write_fake_executable(tmp_path / "cuda-home" / "bin" / "nvcc")
    fake_path_root_nvcc = _write_fake_executable(tmp_path / "cuda-path" / "bin" / "nvcc")

    env = _make_env(
        [tmp_path / "empty-bin"],
        LOCAL_CUDA_PATH=str(fake_local_nvcc.parents[1]),
        CUDA_HOME=str(fake_home_nvcc.parents[1]),
        CUDA_PATH=str(fake_path_root_nvcc.parents[1]),
    )

    result = _run_make(wrapper, "print-nvcc", env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(fake_local_nvcc)


def test_make_clean_does_not_require_nvcc(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    env = _make_env([tmp_path / "empty-bin"])

    result = _run_make(wrapper, "clean", env=env)

    assert result.returncode == 0, result.stderr


def test_check_nvcc_reports_clear_error_when_no_compiler(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    env = _make_env([tmp_path / "empty-bin"])

    result = _run_make(wrapper, "check-nvcc", env=env)

    assert result.returncode != 0
    assert "Could not find nvcc" in result.stderr
    assert "CUDACXX" in result.stderr
    assert "PATH" in result.stderr


def test_build_recipe_compiles_cuda_source_not_check_target(tmp_path):
    wrapper = _make_wrapper(tmp_path)
    env = _make_env([tmp_path / "empty-bin"], NVCC="/bin/echo")

    result = _run_make(wrapper, "-n", "all", env=env)

    assert result.returncode == 0, result.stderr
    assert " -o libcuda_backproject.so cuda_backproject.cu" in result.stdout
    assert "check-nvcc" not in result.stdout.splitlines()[-1]


def test_custom_cuda_requested_defaults_to_true_unless_disabled(monkeypatch):
    import recovar.cuda_backproject as cb

    monkeypatch.delenv("RECOVAR_ENABLE_CUSTOM_CUDA", raising=False)
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    assert cb.custom_cuda_requested() is True

    monkeypatch.setenv("RECOVAR_ENABLE_CUSTOM_CUDA", "1")
    assert cb.custom_cuda_requested() is True

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    assert cb.custom_cuda_requested() is False


def test_build_custom_cuda_writes_requested_output(monkeypatch, tmp_path):
    import recovar.cuda_backproject as cb

    target = (tmp_path / "cache" / "libcuda_backproject.so").resolve()
    captured = {}

    def fake_check_call(cmd):
        captured["cmd"] = cmd
        target.write_text("stub")

    monkeypatch.setattr(cb.subprocess, "check_call", fake_check_call)
    result = cb.build_custom_cuda(output_path=target)

    assert result == target
    assert target.exists()
    assert captured["cmd"][-1] == f"LIB={target}"


def test_build_custom_cuda_force_rebuilds_even_when_output_exists(monkeypatch, tmp_path):
    import recovar.cuda_backproject as cb

    target = (tmp_path / "cache" / "libcuda_backproject.so").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old")
    captured = {}

    def fake_check_call(cmd):
        captured["cmd"] = cmd
        target.write_text("new")

    monkeypatch.setattr(cb.subprocess, "check_call", fake_check_call)
    result = cb.build_custom_cuda(output_path=target, force=True)

    assert result == target
    assert target.read_text() == "new"
    assert captured["cmd"][:3] == ["make", "-B", "-C"]
    assert captured["cmd"][-1] == f"LIB={target}"


def test_ensure_lib_path_autobuilds_when_missing(monkeypatch, tmp_path):
    import recovar.cuda_backproject as cb

    target = (tmp_path / "cache" / "libcuda_backproject.so").resolve()
    calls = []

    monkeypatch.setattr(cb, "_auto_build_attempted", False)
    monkeypatch.setattr(cb, "_auto_build_error", None)
    monkeypatch.setattr(cb, "_default_build_lib_path", lambda: target)
    monkeypatch.setattr(cb, "_existing_lib_path", lambda: target if target.exists() else None)

    def fake_build_custom_cuda(output_path=None, force=False):
        calls.append((output_path, force))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub")
        return target

    monkeypatch.setattr(cb, "build_custom_cuda", fake_build_custom_cuda)

    assert cb._ensure_lib_path() == target
    assert calls == [(target, False)]


def test_cuda_available_records_load_error_for_followup_failure_message(monkeypatch):
    import recovar.cuda_backproject as cb

    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cb, "_cuda_ok", None)
    monkeypatch.setattr(cb, "_auto_build_error", None)
    monkeypatch.setattr(cb.jax, "devices", lambda: [types.SimpleNamespace(platform="gpu")])
    error = OSError("dlopen failed")

    def _fail_ensure_ffi():
        raise error

    monkeypatch.setattr(cb, "_ensure_ffi", _fail_ensure_ffi)

    assert cb.cuda_available() is False
    assert cb._auto_build_error is error
    assert "dlopen failed" in str(cb.cuda_unavailable_error())


def test_slicing_uses_custom_cuda_by_default_on_gpu(monkeypatch):
    import recovar.core.slicing as core_slicing
    import recovar.cuda_backproject as cb

    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: True)
    monkeypatch.delenv("RECOVAR_ENABLE_CUSTOM_CUDA", raising=False)
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cb, "cuda_available", lambda: True)

    assert core_slicing._use_cuda(1) is True


def test_slicing_respects_explicit_cuda_disable(monkeypatch):
    import recovar.core.slicing as core_slicing
    import recovar.cuda_backproject as cb

    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: True)
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")

    def _forbidden_cuda_available():
        raise AssertionError("cuda_available should not be called when RECOVAR_DISABLE_CUDA=1")

    monkeypatch.setattr(cb, "cuda_available", _forbidden_cuda_available)

    assert core_slicing._use_cuda(1) is False


def test_slicing_raises_clear_error_when_default_custom_cuda_is_unavailable(monkeypatch):
    import recovar.core.slicing as core_slicing
    import recovar.cuda_backproject as cb

    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: True)
    monkeypatch.delenv("RECOVAR_ENABLE_CUSTOM_CUDA", raising=False)
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cb, "cuda_available", lambda: False)
    error = RuntimeError("Run `recovar build_custom_cuda` or set `RECOVAR_DISABLE_CUDA=1`.")
    monkeypatch.setattr(cb, "cuda_unavailable_error", lambda: error)

    with pytest.raises(RuntimeError, match="build_custom_cuda"):
        core_slicing._use_cuda(1)
