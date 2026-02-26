import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


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


def pytest_collection_modifyitems(config, items):
    run_long_test = config.getoption("--long-test")
    # --long-test implies all the sub-flags so long tests are not doubly skipped
    run_slow = config.getoption("--run-slow") or run_long_test
    run_gpu = config.getoption("--run-gpu") or run_long_test
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
