import subprocess
import sys
from types import SimpleNamespace

import pytest

from helpers import perf_regression

pytestmark = pytest.mark.unit

_GB = 1024**3


def test_run_tracked_subprocess_records_polled_peak(monkeypatch):
    recorded = []

    monkeypatch.setattr(perf_regression, "_read_nvidia_smi_gpu_bytes", lambda env=None: 5 * _GB)
    monkeypatch.setattr(perf_regression, "_record_tracked_gpu_peak_bytes", lambda peak: recorded.append(peak))
    monkeypatch.setattr(
        perf_regression.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr=""),
    )

    result = perf_regression.run_tracked_subprocess(["echo", "hi"], capture_output=True, text=True)

    assert recorded == [5 * _GB]
    assert result.stdout == "ok"
    assert result.gpu_peak_bytes == 5 * _GB


def test_perf_snapshot_consumes_tracked_gpu_peak(monkeypatch):
    class _FakeDevice:
        @staticmethod
        def memory_stats():
            return {}

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(local_devices=lambda: [_FakeDevice()]))
    monkeypatch.setattr(perf_regression, "_read_nvidia_smi_gpu_bytes", lambda env=None: 1 * _GB)

    perf_regression._record_tracked_gpu_peak_bytes(6 * _GB)
    first = perf_regression.perf_snapshot()
    second = perf_regression.perf_snapshot()

    assert first["tracked_gpu_peak_bytes"] == 6 * _GB
    assert second["tracked_gpu_peak_bytes"] == 1 * _GB


def test_stage_perf_uses_tracked_gpu_peak_bytes():
    before = {
        "wall_time": 0.0,
        "cpu_rss_bytes": 100,
        "gpu_bytes_in_use": 0,
        "gpu_peak_bytes": 0,
    }
    after = {
        "wall_time": 2.0,
        "cpu_rss_bytes": 200,
        "gpu_bytes_in_use": 1 * _GB,
        "gpu_peak_bytes": 0,
        "tracked_gpu_peak_bytes": 6 * _GB,
    }

    perf = perf_regression.stage_perf(before, after)

    assert perf["wall_seconds"] == 2.0
    assert perf["peak_cpu_memory_gb"] == pytest.approx(0.0)
    assert perf["peak_gpu_memory_gb"] == pytest.approx(round((6 * _GB) / 1e9, 3))
