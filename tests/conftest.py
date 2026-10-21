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
    _prepend_repo_root_to_pythonpath(env)
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

    if _env_flag("RECOVAR_DISABLE_CUDA"):
        # An explicit disable must override inherited custom-CUDA settings.
        env.pop("RECOVAR_CUDA_LIB", None)
        env.pop("RECOVAR_ENABLE_CUSTOM_CUDA", None)
    else:
        lib_path = _resolve_custom_cuda_test_lib(require=_env_flag(_REQUIRE_CUSTOM_CUDA_FOR_TESTS_ENV))
        if lib_path is not None:
            env["RECOVAR_CUDA_LIB"] = str(lib_path)
            env["RECOVAR_ENABLE_CUSTOM_CUDA"] = "1"
            env.pop("RECOVAR_DISABLE_CUDA", None)
    return env


def _prepend_repo_root_to_pythonpath(env):
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")


def repo_subprocess_env(env=None):
    """Environment for a child interpreter that must import this checkout's ``recovar``.

    The device-neutral sibling of ``gpu_subprocess_env``: it pins the import
    root and leaves device, backend and allocator settings as the caller made
    them. A shared environment's editable finder can map ``recovar`` to another
    checkout, and a child sees this repo root on ``sys.path`` only through its
    working directory (``-c``/``-m``) or not at all (``python scripts/x.py``
    puts ``scripts/`` there), so an unpinned child can silently test the other
    checkout. Launch the child with ``repo_python_command`` so the import root
    is asserted inside it. ``RECOVAR_EXPECTED_REPO_ROOT`` is deliberately left
    alone: ``relax.commands.initial_model`` prints its own provenance check on
    stdout when that is set.
    """
    env = dict(os.environ if env is None else env)
    _prepend_repo_root_to_pythonpath(env)
    env["PYTHONNOUSERSITE"] = "1"
    return env


# Exit status of a child whose ``recovar`` import resolved outside the repo root.
# It differs from the statuses the launched entry points use, so a test that
# expects its child to fail cannot pass on a child that failed in another checkout.
REPO_IMPORT_ROOT_FAILURE_STATUS = 86

# Runs one ``python`` invocation (``-c CODE``, ``-m MODULE`` or ``SCRIPT``, then
# its arguments) as ``python`` would, then, however it ended, exits with
# REPO_IMPORT_ROOT_FAILURE_STATUS unless ``recovar`` was imported from under the
# root given as the first argument.
_REPO_IMPORT_ROOT_LAUNCHER = f"""\
import os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1]).resolve()
if sys.argv[2] in ("-c", "-m"):
    mode, target, sys.argv = sys.argv[2], sys.argv[3], [sys.argv[2], *sys.argv[4:]]
else:
    mode, target, sys.argv = "script", sys.argv[2], sys.argv[2:]
try:
    if mode == "-c":
        exec(compile(target, "<string>", "exec"), {{"__name__": "__main__"}})
    elif mode == "-m":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        sys.path[0] = os.path.dirname(os.path.realpath(target))
        runpy.run_path(target, run_name="__main__")
finally:
    module = sys.modules.get("recovar")
    origin = pathlib.Path(module.__file__).resolve() if module is not None else None
    if origin is None or not origin.is_relative_to(root):
        sys.stdout.flush()
        sys.stderr.write(f"child imported recovar from {{origin}}, not from under {{root}}\\n")
        sys.stderr.flush()
        os._exit({REPO_IMPORT_ROOT_FAILURE_STATUS})
"""


def repo_python_command(*args):
    """``python *args`` for a child that must fail unless it imported this checkout's ``recovar``.

    ``args`` are what would follow ``python``: ``"-c", code, ...``,
    ``"-m", module, ...`` or ``script, ...``. The child runs them unchanged and
    then exits with ``REPO_IMPORT_ROOT_FAILURE_STATUS`` if ``recovar`` was not
    imported from under the repo root, whatever the invocation itself returned.
    Pair it with ``repo_subprocess_env``, which is what makes that import
    resolve here.
    """
    return [sys.executable, "-c", _REPO_IMPORT_ROOT_LAUNCHER, str(ROOT), *args]


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
        "gpu_memory_matrix: 14-cell GPU memory matrix (7 budgets x 2 backends); "
        "runs under --long-test or via scripts/run_gpu_memory_matrix.sh",
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

    import recovar.cuda_backproject as cuda_backproject

    if cuda_backproject._ffi_registered and cuda_backproject._loaded_lib_path is not None:
        # This process's XLA FFI handlers are bound to the library it loaded
        # first, and the loader refuses any other; tests that pin
        # RECOVAR_CUDA_LIB and their subprocesses must use that same one.
        return Path(cuda_backproject._loaded_lib_path)

    if _custom_cuda_test_lib is _CUSTOM_CUDA_LIB_UNSET:
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


@pytest.fixture(autouse=True)
def _strict_em_operand_precision(monkeypatch):
    """Fail tests that carry EM operands wider than the precision policy.

    Production EM precision is float32; a single float64 factor upstream can
    silently promote reconstruction and M-step rows with no visible effect on
    results. Production only warns (see
    ``relax.helpers.dtype_policy``); tests are strict
    unless a test opts out by setting the variable itself.
    """

    monkeypatch.setenv("RECOVAR_EM_OPERAND_PRECISION_CHECK", "raise")
