"""The Python-loop fallback in ``backproject_weight_sets_from_fft`` is ~15-100x
slower than the per-image CUDA path. It must not be taken silently on a GPU
where CUDA should be available (e.g. a stale/unbuilt ``libcuda_backproject.so``).
These tests pin the guard logic.
"""

from __future__ import annotations

import types

import pytest

from recovar.heterogeneity import kernel_regression_reconstruction as kr

pytestmark = pytest.mark.unit

_STUB_CONFIG = types.SimpleNamespace(disc_type="cubic")  # order 3 -> CUDA per-image unsupported


def test_loop_allowed_on_cpu(monkeypatch):
    monkeypatch.setattr(kr.jax, "default_backend", lambda: "cpu")
    monkeypatch.delenv(kr._ALLOW_LOOP_FALLBACK_ENV, raising=False)
    assert kr._loop_fallback_allowed(_STUB_CONFIG) is True


def test_loop_blocked_on_gpu_with_cuda(monkeypatch):
    monkeypatch.setattr(kr.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(kr, "custom_cuda_requested", lambda: True)
    monkeypatch.delenv(kr._ALLOW_LOOP_FALLBACK_ENV, raising=False)
    assert kr._loop_fallback_allowed(_STUB_CONFIG) is False


def test_loop_allowed_on_gpu_when_cuda_disabled(monkeypatch):
    monkeypatch.setattr(kr.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(kr, "custom_cuda_requested", lambda: False)
    monkeypatch.delenv(kr._ALLOW_LOOP_FALLBACK_ENV, raising=False)
    assert kr._loop_fallback_allowed(_STUB_CONFIG) is True


def test_loop_allowed_on_gpu_with_explicit_optin(monkeypatch):
    monkeypatch.setattr(kr.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(kr, "custom_cuda_requested", lambda: True)
    monkeypatch.setenv(kr._ALLOW_LOOP_FALLBACK_ENV, "1")
    assert kr._loop_fallback_allowed(_STUB_CONFIG) is True


def test_dispatcher_raises_on_blocked_fallback(monkeypatch):
    monkeypatch.setattr(kr.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(kr, "custom_cuda_requested", lambda: True)
    monkeypatch.delenv(kr._ALLOW_LOOP_FALLBACK_ENV, raising=False)
    with pytest.raises(RuntimeError, match="slow Python loop"):
        kr.backproject_weight_sets_from_fft(_STUB_CONFIG, None, None, None, None, None, [[1.0]])
