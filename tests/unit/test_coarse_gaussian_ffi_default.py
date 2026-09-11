"""The fresh-InitialModel coarse Gaussian FFI default applies only with its RELION projector operands.

A dense K=1 pass without a supplied RELION projector (for example a cold start
on the host NumPy image backend routed through the adaptive engine) keeps the
JAX coarse path instead of failing on the FFI operand check.
"""

from __future__ import annotations

import inspect

import pytest

from recovar.em.dense_single_volume.helpers import significance

pytestmark = pytest.mark.unit


def test_default_needs_projector_and_texture_or_float64():
    f = significance._coarse_gaussian_ffi_default
    assert f(True, use_relion_projector=True, use_float64_scoring=False, coarse_texture_interp=True) is True
    assert f(True, use_relion_projector=True, use_float64_scoring=True, coarse_texture_interp=False) is True
    assert f(True, use_relion_projector=False, use_float64_scoring=False, coarse_texture_interp=True) is False
    assert f(True, use_relion_projector=True, use_float64_scoring=False, coarse_texture_interp=False) is False
    assert f(False, use_relion_projector=True, use_float64_scoring=True, coarse_texture_interp=True) is False


def test_explicit_environment_request_still_wins(monkeypatch):
    monkeypatch.setenv(significance._K1_COARSE_GAUSSIAN_FFI_ENV, "1")
    assert significance._k1_coarse_gaussian_ffi_enabled(default=False) is True
    monkeypatch.setenv(significance._K1_COARSE_GAUSSIAN_FFI_ENV, "0")
    assert significance._k1_coarse_gaussian_ffi_enabled(default=True) is False


def test_significance_resolves_the_default_through_the_rule():
    source = inspect.getsource(significance._compute_k_class_significance_batched)
    assert source.count("default=_coarse_gaussian_ffi_default(") == 1
    assert "default=relion_coarse_gaussian_default,\n    )\n    coarse_gaussian_ffi_enabled" not in source
