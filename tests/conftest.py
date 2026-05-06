import os
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
      not try to grab 75 % of GPU memory (the main pytest process may
      already hold a large chunk).
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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
    parser.addoption(
        "--em-parity-long",
        action="store_true",
        default=False,
        help=(
            "run EM-long parity regression tests (256² 50k full ab-initio K=1 / K=4 "
            "vs RELION). Disjoint from --long-test by design: those tests measure "
            "SPA/ET pipeline metrics that EM-only changes do not move, and cost "
            "hours of GPU time per run. Implies --run-slow, --run-gpu, and "
            "--run-integration."
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
        "em_parity_long: EM-long parity regression tests (256² 50k full ab-initio "
        "K=1 / K=4 vs RELION); requires --em-parity-long flag and a GPU; ~2-4 hr per case",
    )


def pytest_collection_modifyitems(config, items):
    run_long_test = config.getoption("--long-test")
    run_em_parity_long = config.getoption("--em-parity-long")
    # --long-test implies all the sub-flags so long tests are not doubly skipped
    run_slow = config.getoption("--run-slow") or run_long_test or run_em_parity_long
    # Auto-detect GPU: run gpu-marked tests whenever a GPU is available,
    # even without --run-gpu.  The flag still works as an explicit override.
    gpu_available = False
    try:
        import jax

        gpu_available = any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        pass
    run_gpu = config.getoption("--run-gpu") or run_long_test or run_em_parity_long or gpu_available
    run_integration = config.getoption("--run-integration") or run_long_test or run_em_parity_long
    run_tiny_metrics = config.getoption("--run-tiny-metrics")

    skip_slow = pytest.mark.skip(reason="need --run-slow to run")
    skip_gpu = pytest.mark.skip(reason="need --run-gpu to run")
    skip_integration = pytest.mark.skip(reason="need --run-integration to run")
    skip_tiny_metrics = pytest.mark.skip(reason="need --run-tiny-metrics to run")
    skip_long_test = pytest.mark.skip(reason="need --long-test to run")
    skip_em_parity_long = pytest.mark.skip(reason="need --em-parity-long to run")

    for item in items:
        # item.keywords also contains package/path names like tests/long_test;
        # use explicit marks so EM-long parity remains disjoint from --long-test.
        has_long_test_marker = item.get_closest_marker("long_test") is not None
        has_em_parity_long_marker = item.get_closest_marker("em_parity_long") is not None
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "tiny_metrics" in item.keywords and not run_tiny_metrics:
            item.add_marker(skip_tiny_metrics)
        if has_long_test_marker and not run_long_test:
            item.add_marker(skip_long_test)
        if has_em_parity_long_marker and not run_em_parity_long:
            item.add_marker(skip_em_parity_long)


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
