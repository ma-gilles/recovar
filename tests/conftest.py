import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked slow")
    parser.addoption("--run-gpu", action="store_true", default=False, help="run tests marked gpu")
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked integration",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, isolated unit tests")
    config.addinivalue_line("markers", "integration: multi-module integration tests")
    config.addinivalue_line("markers", "gpu: tests requiring CUDA/GPU runtime")
    config.addinivalue_line("markers", "slow: long-running tests")
    config.addinivalue_line("markers", "io: filesystem/network-like I/O tests")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    run_gpu = config.getoption("--run-gpu")
    run_integration = config.getoption("--run-integration")

    skip_slow = pytest.mark.skip(reason="need --run-slow to run")
    skip_gpu = pytest.mark.skip(reason="need --run-gpu to run")
    skip_integration = pytest.mark.skip(reason="need --run-integration to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def _set_deterministic_seed():
    # Keep stochastic tests deterministic by default.
    import numpy as np

    np.random.seed(0)
