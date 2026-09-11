"""Both sparse pass-2 scorers select their RELION powerClass noise terms through one owner."""

from __future__ import annotations

import inspect

import pytest

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sp

from recovar.em.dense_single_volume.helpers import sparse_pass2_scoring

pytestmark = pytest.mark.unit


def _with_fakes(monkeypatch):
    calls = []
    monkeypatch.setattr(sparse_pass2_scoring, "_relion_cuda_powerclass_highres_xi2_half", lambda x, **kw: calls.append("xi2") or "xi2")
    monkeypatch.setattr(sparse_pass2_scoring, "_relion_cuda_powerclass_spectrum_highres_norm_units", lambda x, **kw: calls.append("spectrum") or "spectrum")
    monkeypatch.setattr(sparse_pass2_scoring, "_relion_powerclass_highres_xi2_half_to_norm_units", lambda v, shape: calls.append("convert") or ("norm", v))
    return calls


def test_nothing_is_computed_without_exact_scoring_or_noise_accumulation(monkeypatch):
    calls = _with_fakes(monkeypatch)
    out = sp._relion_powerclass_noise_terms("x", image_shape=(8, 8), current_size=8, use_exact_relion_gaussian=False, accumulate_noise=False, source_faithful_spectrum_norm=True)
    assert out == (None, None) and calls == []


def test_exact_scoring_without_current_size_only_needs_highres_xi2(monkeypatch):
    calls = _with_fakes(monkeypatch)
    out = sp._relion_powerclass_noise_terms("x", image_shape=(8, 8), current_size=None, use_exact_relion_gaussian=True, accumulate_noise=True, source_faithful_spectrum_norm=False)
    assert out == ("xi2", None) and calls == ["xi2"]


@pytest.mark.parametrize("source_faithful,expected,calls_expected", [(True, "spectrum", ["xi2", "spectrum"]), (False, ("norm", "xi2"), ["xi2", "convert"])])
def test_noise_accumulation_selects_the_norm_term(monkeypatch, source_faithful, expected, calls_expected):
    calls = _with_fakes(monkeypatch)
    out = sp._relion_powerclass_noise_terms("x", image_shape=(8, 8), current_size=8, use_exact_relion_gaussian=False, accumulate_noise=True, source_faithful_spectrum_norm=source_faithful)
    assert out == ("xi2", expected) and calls == calls_expected


def test_both_sparse_scorers_use_the_owner():
    for name in ("compute_pass2_stats_sparse_bucketed", "compute_k_class_pass2_stats_sparse_fused"):
        source = inspect.getsource(getattr(sp, name))
        assert source.count("_relion_powerclass_noise_terms(") == 1
        assert "_relion_cuda_powerclass_highres_xi2_half(" not in source
