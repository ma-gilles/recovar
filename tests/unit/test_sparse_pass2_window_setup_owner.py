"""Both sparse pass-2 scorers build their forward model and Fourier windows through one owner."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import pytest

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sp

pytestmark = pytest.mark.unit


def _setup(monkeypatch, *, use_window, recon, x_half, prepare):
    calls = []
    monkeypatch.setattr(sp.ForwardModelConfig, "from_dataset", lambda ds, *, disc_type, process_fn: calls.append(("config", disc_type)) or "config")
    monkeypatch.setattr(sp, "make_fourier_window_spec", lambda *a, **kw: SimpleNamespace(use_window=use_window, score_indices_np="sinp", score_indices="si", recon_indices=recon, n_score=11, n_recon=7))
    monkeypatch.setattr(sp, "centered_half_indices_to_fftw_half_indices", lambda shape, idx: calls.append(("fftw", idx if not hasattr(idx, "shape") else int(idx.shape[0]))) or "xhalf")
    monkeypatch.setattr(sp, "_windowed_prepare_enabled_for_pass", lambda use_window: prepare)
    out = sp._sparse_pass2_window_setup(
        SimpleNamespace(process_images=None), disc_type="linear", image_shape=(8, 8), current_size=6, n_half=40, mstep_current_size=8,
        square_window=False, window_spec_kwargs={}, use_relion_x_half_mstep=x_half, log_label="Sparse pass-2",
    )
    return out, calls


def test_setup_returns_windows_and_indices(monkeypatch):
    out, calls = _setup(monkeypatch, use_window=True, recon="recon", x_half=False, prepare=True)
    assert out.config == "config" and out.use_window is True and out.window_indices_np == "sinp" and out.window_indices == "si"
    assert out.recon_window_indices == "recon" and out.relion_x_half_recon_indices is None and out.windowed_prepare is True
    assert (out.n_windowed, out.n_recon_windowed) == (11, 7) and calls == [("config", "linear")]


def test_x_half_mstep_converts_the_reconstruction_window_or_the_full_half(monkeypatch):
    out, calls = _setup(monkeypatch, use_window=True, recon="recon", x_half=True, prepare=False)
    assert out.relion_x_half_recon_indices == "xhalf" and calls[-1] == ("fftw", "recon")
    out, calls = _setup(monkeypatch, use_window=False, recon=None, x_half=True, prepare=False)
    assert out.relion_x_half_recon_indices == "xhalf" and calls[-1] == ("fftw", 40)


def test_both_sparse_scorers_use_the_owner():
    for name in ("compute_pass2_stats_sparse_bucketed", "compute_k_class_pass2_stats_sparse_fused"):
        source = inspect.getsource(getattr(sp, name))
        assert source.count("_sparse_pass2_window_setup(") == 1
        # The scorers keep only their separate budget-planning window; the scoring/reconstruction window lives in the owner.
        assert re.search(r"^\s*window_spec = make_fourier_window_spec\(", source, re.MULTILINE) is None
        # The budget-planning window is resolved by the shared pass-2 window setup owner.
        assert source.count("= _pass2_window_setup(") == 1
        assert "budget_window_spec = make_fourier_window_spec(" not in source
        assert "ForwardModelConfig.from_dataset(" not in source
        assert "centered_half_indices_to_fftw_half_indices(" not in source

def test_pass2_window_setup_owner_builds_the_budget_window():
    assert inspect.getsource(sp._pass2_window_setup).count("budget_window_spec = make_fourier_window_spec(") == 1
