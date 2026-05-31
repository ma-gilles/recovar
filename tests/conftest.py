import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

_REQUIRE_CUSTOM_CUDA_FOR_TESTS_ENV = "RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS"
_CUSTOM_CUDA_LIB_UNSET = object()
_custom_cuda_test_lib = _CUSTOM_CUDA_LIB_UNSET
_custom_cuda_test_error = None


def _env_flag(name):
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _pick_most_free_gpu_index():
    """Best-effort selection of the GPU with the most free memory."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return None

    best_idx = None
    best_free = -1
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            free_mb = int(parts[1])
        except ValueError:
            continue
        if free_mb > best_free:
            best_free = free_mb
            best_idx = idx
    return best_idx


def gpu_subprocess_env():
    """Environment dict for GPU subprocesses spawned by integration tests.

    - Prepends the repo root to PYTHONPATH so the subprocess imports the
      local ``recovar`` package rather than whatever is ``pip install -e``'d.
    - Sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so the subprocess does
      not try to grab most of GPU memory (the main pytest process may
      already hold a large chunk).
    - Pins ``XLA_PYTHON_CLIENT_MEM_FRACTION=.90`` so regression baselines
      are not perturbed by a developer shell override such as ``.50``.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    # Ensure subprocesses prefer CUDA backend for custom-call kernels.
    env["JAX_PLATFORMS"] = "cuda,cpu"
    env["JAX_PLATFORM_NAME"] = "gpu"
    env["PYTHONNOUSERSITE"] = "1"
    assigned_visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    if not assigned_visible_devices:
        gpu_idx = _pick_most_free_gpu_index()
        if gpu_idx is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    if not _env_flag("RECOVAR_DISABLE_CUDA"):
        lib_path = _resolve_custom_cuda_test_lib(require=_env_flag(_REQUIRE_CUSTOM_CUDA_FOR_TESTS_ENV))
        if lib_path is not None:
            env["RECOVAR_CUDA_LIB"] = str(lib_path)
            env["RECOVAR_ENABLE_CUSTOM_CUDA"] = "1"
            env.pop("RECOVAR_DISABLE_CUDA", None)
    return env


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked slow")
    parser.addoption("--run-gpu", action="store_true", default=False, help="run tests marked gpu")
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked integration",
    )
    parser.addoption(
        "--run-tiny-metrics",
        action="store_true",
        default=False,
        help="run tiny end-to-end metrics/outliers integration tests (no large dataset required)",
    )
    parser.addoption(
        "--long-test",
        action="store_true",
        default=False,
        help=(
            "run long quality-regression tests (cryo-EM SPA, cryo-ET, pipeline with "
            "outliers, pipeline with --ind/--particle-ind). Implies --run-slow, "
            "--run-gpu, --run-integration. Volumes are generated synthetically so no "
            "external data is required. Baselines auto-created in tests/baselines/ on "
            "first run. Set LONG_METRICS_OUTPUT_BASE=/scratch/... to redirect large outputs."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, isolated unit tests")
    config.addinivalue_line("markers", "integration: multi-module integration tests")
    config.addinivalue_line("markers", "gpu: tests requiring CUDA/GPU runtime")
    config.addinivalue_line("markers", "slow: long-running tests")
    config.addinivalue_line("markers", "io: filesystem/network-like I/O tests")
    config.addinivalue_line("markers", "tiny_metrics: tiny end-to-end metrics/outliers tests")
    config.addinivalue_line(
        "markers",
        "long_test: long quality regression tests (cryo-EM SPA, cryo-ET, outliers, "
        "with/without indices); requires --long-test flag; volumes generated synthetically",
    )
    config.addinivalue_line(
        "markers",
        "gpu_memory_matrix: 14-cell GPU memory matrix (7 budgets x 2 backends); "
        "runs under --long-test or via scripts/run_gpu_memory_matrix.sh",
    )


def pytest_collection_modifyitems(config, items):
    run_long_test = config.getoption("--long-test")
    # --long-test implies all the sub-flags so long tests are not doubly skipped
    run_slow = config.getoption("--run-slow") or run_long_test
    # Auto-detect GPU: run gpu-marked tests whenever a GPU is available,
    # even without --run-gpu.  The flag still works as an explicit override.
    gpu_available = False
    try:
        import jax

        gpu_available = any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        pass
    run_gpu = config.getoption("--run-gpu") or run_long_test or gpu_available
    run_integration = config.getoption("--run-integration") or run_long_test
    run_tiny_metrics = config.getoption("--run-tiny-metrics")

    skip_slow = pytest.mark.skip(reason="need --run-slow to run")
    skip_gpu = pytest.mark.skip(reason="need --run-gpu to run")
    skip_integration = pytest.mark.skip(reason="need --run-integration to run")
    skip_tiny_metrics = pytest.mark.skip(reason="need --run-tiny-metrics to run")
    skip_long_test = pytest.mark.skip(reason="need --long-test to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "tiny_metrics" in item.keywords and not run_tiny_metrics:
            item.add_marker(skip_tiny_metrics)
        if "long_test" in item.keywords and not run_long_test:
            item.add_marker(skip_long_test)


@pytest.fixture(autouse=True)
def _set_deterministic_seed():
    # Keep stochastic tests deterministic by default.
    import numpy as np

    np.random.seed(0)


def _first_gpu_or_skip():
    """Return the first available GPU device or skip the test."""
    jax = pytest.importorskip("jax")
    for backend in ("cuda", "gpu"):
        try:
            gpus = jax.devices(backend)
            if gpus:
                return gpus[0]
        except RuntimeError:
            continue
    pytest.skip("No GPU device available")


@pytest.fixture
def gpu_device():
    """Pytest fixture that provides a GPU device or skips the test."""
    return _first_gpu_or_skip()


def _candidate_nvcc_paths():
    candidates = []
    for candidate in (
        os.environ.get("NVCC"),
        shutil.which("nvcc"),
        "/usr/local/cuda-13.1/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and str(path) not in candidates:
            candidates.append(str(path))
    return candidates


def _custom_cuda_test_output_path():
    return ROOT / ".tmp" / "pytest_custom_cuda" / "libcuda_backproject.so"


def _custom_cuda_test_error_detail():
    if _custom_cuda_test_error is None:
        return ""
    return f": {_custom_cuda_test_error}"


def _resolve_custom_cuda_test_lib(*, require=False):
    global _custom_cuda_test_error, _custom_cuda_test_lib

    if _custom_cuda_test_lib is _CUSTOM_CUDA_LIB_UNSET:
        import recovar.cuda_backproject as cuda_backproject

        configured = os.environ.get("RECOVAR_CUDA_LIB")
        if configured and Path(configured).exists():
            _custom_cuda_test_lib = Path(configured).resolve()
            _custom_cuda_test_error = None
        else:
            target = _custom_cuda_test_output_path()
            if target.exists():
                _custom_cuda_test_lib = target.resolve()
                _custom_cuda_test_error = None
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                last_exc = None
                old_nvcc = os.environ.get("NVCC")
                old_cache_dir = os.environ.get("RECOVAR_CUDA_CACHE_DIR")
                os.environ["RECOVAR_CUDA_CACHE_DIR"] = str(target.parent)
                try:
                    for nvcc_path in _candidate_nvcc_paths():
                        os.environ["NVCC"] = nvcc_path
                        try:
                            _custom_cuda_test_lib = cuda_backproject.build_custom_cuda(output_path=target)
                            _custom_cuda_test_error = None
                            break
                        except Exception as exc:  # pragma: no cover - exercised in GPU test envs
                            last_exc = exc
                            cuda_backproject._cuda_ok = None
                    else:
                        _custom_cuda_test_lib = None
                        _custom_cuda_test_error = last_exc
                finally:
                    if old_nvcc is None:
                        os.environ.pop("NVCC", None)
                    else:
                        os.environ["NVCC"] = old_nvcc
                    if old_cache_dir is None:
                        os.environ.pop("RECOVAR_CUDA_CACHE_DIR", None)
                    else:
                        os.environ["RECOVAR_CUDA_CACHE_DIR"] = old_cache_dir

    if _custom_cuda_test_lib is None and require:
        raise RuntimeError(f"Could not build RECOVAR custom CUDA test library{_custom_cuda_test_error_detail()}")

    return _custom_cuda_test_lib


@pytest.fixture(scope="session")
def custom_cuda_lib():
    """Build the optional RECOVAR CUDA extension for tests that need it."""
    _first_gpu_or_skip()
    lib_path = _resolve_custom_cuda_test_lib(require=_env_flag(_REQUIRE_CUSTOM_CUDA_FOR_TESTS_ENV))
    if lib_path is None:
        pytest.skip(f"Could not build RECOVAR custom CUDA test library{_custom_cuda_test_error_detail()}")
    yield lib_path
