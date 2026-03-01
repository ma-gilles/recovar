from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.utils import multi_gpu as mgu

pytestmark = pytest.mark.unit


def test_split_indices_for_gpus_even_and_remainder():
    splits = mgu.split_indices_for_gpus(10, 3)
    assert len(splits) == 3
    # np.array_split distributes extra items across first chunks: [4, 3, 3]
    assert [len(s) for s in splits] == [4, 3, 3]
    merged = np.concatenate(splits)
    assert np.all(merged == np.arange(10))

    # Even split
    splits_even = mgu.split_indices_for_gpus(12, 3)
    assert [len(s) for s in splits_even] == [4, 4, 4]
    merged_even = np.concatenate(splits_even)
    assert np.all(merged_even == np.arange(12))


def test_reduce_results_sums_across_devices():
    Hs = [np.ones((2, 2), dtype=np.float32), np.full((2, 2), 2.0, dtype=np.float32)]
    Bs = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    H, B = mgu.reduce_results(Hs, Bs)
    assert np.allclose(H, np.full((2, 2), 3.0))
    assert np.allclose(B, np.array([4.0, 6.0]))


def test_estimate_multi_gpu_speedup_basic_monotonic():
    s1 = mgu.estimate_multi_gpu_speedup(n_images=1000, n_gpus=1)
    s4 = mgu.estimate_multi_gpu_speedup(n_images=1000, n_gpus=4)
    assert s1["expected_speedup"] > 0
    assert s4["expected_speedup"] > s1["expected_speedup"]
    assert s4["images_per_gpu"] == 250


def test_compute_H_B_multi_gpu_single_device_path(monkeypatch):
    class _FakeCtx:
        def __init__(self, _d):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mgu, "get_available_gpus", lambda: [SimpleNamespace(id=0)])
    monkeypatch.setattr(mgu.jax, "default_device", lambda d: _FakeCtx(d))

    called = {}

    def compute_fn(ds, **kwargs):
        called["n_images"] = ds.n_images
        called["kwargs"] = kwargs
        return np.array([[1.0]], dtype=np.float32), np.array([2.0], dtype=np.float32)

    ds = SimpleNamespace(n_images=5)
    H, B = mgu.compute_H_B_multi_gpu(compute_fn, ds, n_gpus=1, foo=3)
    assert np.allclose(H, np.array([[1.0]], dtype=np.float32))
    assert np.allclose(B, np.array([2.0], dtype=np.float32))
    assert called["n_images"] == 5
    assert called["kwargs"]["foo"] == 3


def test_compute_H_B_multi_gpu_multi_device_reduction(monkeypatch):
    monkeypatch.setattr(mgu, "get_available_gpus", lambda: [SimpleNamespace(id=0), SimpleNamespace(id=1)])
    monkeypatch.setattr(mgu, "split_indices_for_gpus", lambda n_items, n_gpus: [np.array([0, 1]), np.array([2, 3])])

    def fake_parallel(_fn, _splits, _devices, *_args, **_kwargs):
        return [np.array([[1.0]], dtype=np.float32), np.array([[2.0]], dtype=np.float32)], [
            np.array([10.0], dtype=np.float32),
            np.array([20.0], dtype=np.float32),
        ]

    monkeypatch.setattr(mgu, "compute_on_gpus_parallel", fake_parallel)
    ds = SimpleNamespace(n_images=4)

    H, B = mgu.compute_H_B_multi_gpu(lambda *_a, **_k: None, ds, n_gpus=2)
    assert np.allclose(H, np.array([[3.0]], dtype=np.float32))
    assert np.allclose(B, np.array([30.0], dtype=np.float32))
