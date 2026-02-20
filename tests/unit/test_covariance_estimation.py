import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.covariance_estimation as cov_est

pytestmark = pytest.mark.unit


def test_default_covariance_options_has_expected_keys(monkeypatch):
    monkeypatch.setattr(cov_est.utils, "get_gpu_memory_total", lambda: 8)
    opts = cov_est.get_default_covariance_computation_options(grid_size=32)
    assert isinstance(opts, dict)
    assert "n_pcs_to_compute" in opts
    assert opts["n_pcs_to_compute"] >= 1
    assert opts["covariance_fn"] == "kernel"


def test_greedy_column_choice_validation():
    sampling_vec = np.ones(64, dtype=np.float32)
    volume_shape = (4, 4, 4)

    with pytest.raises(ValueError):
        cov_est.greedy_column_choice(sampling_vec, n_samples=0, volume_shape=volume_shape, avoid_in_radius=1)
    with pytest.raises(ValueError):
        cov_est.greedy_column_choice(sampling_vec, n_samples=2, volume_shape=volume_shape, avoid_in_radius=-1)


def test_greedy_column_choice_basic_properties():
    sampling_vec = np.ones(64, dtype=np.float32)
    volume_shape = (4, 4, 4)
    picked, freqs = cov_est.greedy_column_choice(
        sampling_vec=sampling_vec,
        n_samples=5,
        volume_shape=volume_shape,
        avoid_in_radius=0,
        keep_only_below_freq=32,
    )
    assert picked.shape == (5,)
    assert freqs.shape == (5, 3)
    assert len(np.unique(picked)) == 5
    assert np.all(freqs[:, 0] >= 0)


def test_randomized_column_choice_basic_properties():
    np.random.seed(0)
    sampling_vec = np.ones(64, dtype=np.float64)
    volume_shape = (4, 4, 4)
    picked, freqs = cov_est.randomized_column_choice(
        sampling_vec=sampling_vec,
        n_samples=4,
        volume_shape=volume_shape,
        avoid_in_radius=0,
    )
    assert picked.shape == (4,)
    assert freqs.shape == (4, 3)
    assert len(np.unique(picked)) == 4
