from __future__ import annotations

import pytest

from recovar.em.dense_single_volume import local_caches

pytestmark = pytest.mark.unit


class _FakeDevice:
    platform = "gpu"

    def __init__(self, bytes_limit: int):
        self._bytes_limit = int(bytes_limit)

    def memory_stats(self):
        return {"bytes_limit": self._bytes_limit}


@pytest.mark.parametrize(
    ("bytes_limit", "expected_gb"),
    [
        (40_000_000_000, 6.0),
        (80_000_000_000, 12.0),
        (120_000_000_000, 12.0),
    ],
)
def test_sparse_big_jit_default_cap_scales_with_device_memory(monkeypatch, bytes_limit, expected_gb):
    import jax

    monkeypatch.setattr(jax, "local_devices", lambda: [_FakeDevice(bytes_limit)])
    local_caches._default_sparse_big_jit_mstep_max_gb.cache_clear()
    assert local_caches._default_sparse_big_jit_mstep_max_gb() == pytest.approx(expected_gb)


def test_sparse_big_jit_explicit_cap_does_not_query_device(monkeypatch):
    monkeypatch.setenv(local_caches.EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB_ENV, "7.25")

    def fail_if_called():
        raise AssertionError("device default must not be evaluated for an explicit override")

    monkeypatch.setattr(local_caches, "_default_sparse_big_jit_mstep_max_gb", fail_if_called)
    _estimated, cap = local_caches._sparse_big_jit_mstep_tensors_memory_gb(
        image_count=2,
        rotation_count=3,
        n_recon_windowed=5,
        use_float64_scoring=False,
    )
    assert cap == pytest.approx(7.25)
