import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


_CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"
_SPEC = importlib.util.spec_from_file_location("tests_conftest_module", _CONFTEST_PATH)
_CONFTEST = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_CONFTEST)


def test_gpu_subprocess_env_preserves_assigned_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    called = {"count": 0}

    def _forbidden_picker():
        called["count"] += 1
        raise AssertionError("_pick_most_free_gpu_index should not run when a shard already assigned a GPU")

    monkeypatch.setattr(_CONFTEST, "_pick_most_free_gpu_index", _forbidden_picker)

    env = _CONFTEST.gpu_subprocess_env()

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert called["count"] == 0


def test_gpu_subprocess_env_picks_gpu_when_unassigned(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(_CONFTEST, "_pick_most_free_gpu_index", lambda: 2)

    env = _CONFTEST.gpu_subprocess_env()

    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_gpu_subprocess_env_preserves_existing_xla_flags(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

    env = _CONFTEST.gpu_subprocess_env()
    tokens = env["XLA_FLAGS"].split()

    assert "--xla_force_host_platform_device_count=8" in tokens
