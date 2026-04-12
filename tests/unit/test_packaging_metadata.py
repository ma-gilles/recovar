import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from setup_helpers import remove_stale_fast_marching_build_artifacts

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_psutil_declared_for_runtime_installs():
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")
    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("psutil") for dep in dependencies)


def test_psutil_declared_for_pixi_dev_env():
    pixi = _load_toml(REPO_ROOT / "pixi.toml")
    dependencies = pixi["dependencies"]

    assert "psutil" in dependencies


def test_gpu_extra_matches_cuda_alias():
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["gpu"] == extras["cuda"]


def test_gpu_flexible_extra_matches_cuda_flexible_alias():
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["gpu-flexible"] == extras["cuda-flexible"]


def test_all_extra_prefers_gpu_alias():
    pyproject = _load_toml(REPO_ROOT / "pyproject.toml")
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["all"] == ["recovar[gpu,interactive,gui,dev]"]


def test_pixi_declares_build_tools_for_optional_native_extensions():
    pixi = _load_toml(REPO_ROOT / "pixi.toml")
    dependencies = pixi["dependencies"]

    assert "cxx-compiler" in dependencies
    assert "make" in dependencies


def test_remove_stale_fast_marching_build_artifacts_only_removes_legacy_root_entries(tmp_path):
    build_lib = tmp_path / "build-lib"
    build_temp = tmp_path / "build-temp"
    stale_cpp = build_lib / "recovar" / "_fast_marching_native.cpp"
    stale_so = build_lib / "recovar" / "_fast_marching_native.cpython-311-x86_64-linux-gnu.so"
    keep_cpp = build_lib / "recovar" / "trajectory" / "_fast_marching_native.cpp"
    stale_obj = build_temp / "recovar" / "_fast_marching_native.o"

    for path in (stale_cpp, stale_so, keep_cpp, stale_obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub")

    removed = remove_stale_fast_marching_build_artifacts(build_lib=build_lib, build_temp=build_temp)

    assert set(removed) == {stale_cpp, stale_so, stale_obj}
    assert not stale_cpp.exists()
    assert not stale_so.exists()
    assert not stale_obj.exists()
    assert keep_cpp.exists()


@pytest.fixture(scope="module")
def built_package_artifacts(tmp_path_factory):
    dist_dir = tmp_path_factory.mktemp("package-dist")
    env = dict(os.environ, PYTHONNOUSERSITE="1")

    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "-w", str(dist_dir)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    sdist = next(dist_dir.glob("*.tar.gz"))
    wheel = next(dist_dir.glob("*.whl"))
    return sdist, wheel


def test_sdist_and_wheel_include_cuda_build_files(built_package_artifacts):
    sdist, wheel = built_package_artifacts

    with tarfile.open(sdist, "r:gz") as tf:
        sdist_names = set(tf.getnames())

    with zipfile.ZipFile(wheel) as zf:
        wheel_names = set(zf.namelist())

    assert any(name.endswith("/recovar/cuda/Makefile") for name in sdist_names)
    assert any(name.endswith("/recovar/cuda/cuda_backproject.cu") for name in sdist_names)
    assert any(name.endswith("/setup_helpers.py") for name in sdist_names)

    assert "recovar/cuda/Makefile" in wheel_names
    assert "recovar/cuda/cuda_backproject.cu" in wheel_names


def test_wheel_excludes_legacy_root_fast_marching_entries(built_package_artifacts):
    _, wheel = built_package_artifacts

    with zipfile.ZipFile(wheel) as zf:
        wheel_names = set(zf.namelist())

    assert any(name.startswith("recovar/trajectory/_fast_marching_native") for name in wheel_names)
    assert not any(name.startswith("recovar/_fast_marching_native") for name in wheel_names)
