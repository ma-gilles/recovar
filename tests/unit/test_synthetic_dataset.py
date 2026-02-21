import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import synthetic_dataset

pytestmark = pytest.mark.unit


def _make_hvd():
    volumes = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, 0.2, 0.1, 0.0],
            [2.0, 1.0, 4.0, 3.0, 0.7, 0.3, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    image_assignments = np.array([0, 1, 1, -1], dtype=np.int32)
    contrasts = np.array([1.0, 1.2, 0.8, 1.1], dtype=np.float32)
    return synthetic_dataset.HeterogeneousVolumeDistribution(
        volumes=volumes,
        image_assignments=image_assignments,
        contrasts=contrasts,
        valid_indices=np.ones(volumes.shape[-1], dtype=np.float32),
        vol_batch_size=1,
    )


def test_hvd_probabilities_mean_and_outlier_fraction():
    hvd = _make_hvd()
    probs = hvd.get_probs_of_state()
    np.testing.assert_allclose(probs, np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float32))
    assert hvd.percent_outliers == pytest.approx(0.25)

    expected_mean = hvd.volumes[0] * (1.0 / 3.0) + hvd.volumes[1] * (2.0 / 3.0)
    np.testing.assert_allclose(hvd.get_mean(), expected_mean, atol=1e-7, rtol=1e-7)


def test_hvd_covariance_square_root_and_columns():
    hvd = _make_hvd()
    sqrt_cov = hvd.get_covariance_square_root(contrasted=False)
    assert sqrt_cov.shape[0] == hvd.volumes.shape[-1]
    assert sqrt_cov.shape[1] == hvd.volumes.shape[0]

    picked = np.array([0, 3, 7], dtype=np.int32)
    cols = hvd.get_covariance_columns(picked, contrasted=False)
    manual = sqrt_cov @ np.conj(sqrt_cov[picked, :]).T
    np.testing.assert_allclose(cols, manual, atol=1e-7, rtol=1e-7)


def test_hvd_variance_helpers_match_when_idft_is_identity(monkeypatch):
    hvd = _make_hvd()
    monkeypatch.setattr(
        synthetic_dataset.linalg,
        "batch_idft3",
        lambda vols, volume_shape, vol_batch_size: vols,
    )
    fourier_vars = hvd.get_fourier_variances(contrasted=False)
    spatial_vars = hvd.get_spatial_variances(contrasted=False)
    np.testing.assert_allclose(spatial_vars, fourier_vars, atol=1e-7, rtol=1e-7)


def test_load_heterogeneous_reconstruction_dict_path_with_scaling(monkeypatch):
    simulation_info = {
        "volumes_path_root": "/tmp/vol_",
        "grid_size": 2,
        "trailing_zero_format_in_vol_name": True,
        "image_assignment": np.array([0, 1, 0], dtype=np.int32),
        "per_image_contrast": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "scale_vol": 3.0,
    }
    monkeypatch.setattr(
        synthetic_dataset.simulator,
        "load_volumes_from_folder",
        lambda *args, **kwargs: np.ones((2, 8), dtype=np.float32) * 2.0,
    )

    hvd = synthetic_dataset.load_heterogeneous_reconstruction(simulation_info, load_volumes=True)
    assert hvd.volumes.shape == (2, 8)
    # Constructor applies a default radial mask; only valid frequency entries remain non-zero.
    assert np.max(hvd.volumes) == pytest.approx(6.0)
    assert np.count_nonzero(hvd.volumes) > 0


def test_get_col_covariance_matches_manual_computation():
    xs = np.array(
        [
            [1 + 1j, 2 + 0j, 3 + 0j],
            [2 + 0j, 1 + 1j, 4 + 0j],
        ],
        dtype=np.complex64,
    )
    probs = np.array([0.25, 0.75], dtype=np.float32)
    mean = np.sum(xs * probs[:, None], axis=0)
    vec_indices = np.array([0, 2], dtype=np.int32)

    out = synthetic_dataset.get_col_covariance(xs, mean, vec_indices, probs)
    expected = np.zeros((3, 2), dtype=np.complex64)
    for j, v in enumerate(vec_indices):
        expected[:, j] = np.sum(
            probs[:, None] * (xs - mean) * np.conj(xs[:, v : v + 1] - mean[v]),
            axis=0,
        )
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6, rtol=1e-6)
