import numpy as np
import pytest

from recovar.em.ppca_refinement.initialization import (
    covariance_from_loading_matrix,
    empirical_weighted_covariance,
    initialize_ppca_from_gt_volumes,
    initialize_ppca_from_kclass_volumes,
    load_volume_stack,
)
from recovar.em.ppca_refinement.schedule import loading_subspace_agreement
from recovar.utils import helpers as utils


pytestmark = pytest.mark.unit


def _synthetic_volumes():
    mu = np.arange(8, dtype=np.float32).reshape(2, 2, 2) / 10.0
    pc1 = np.zeros((2, 2, 2), dtype=np.float32)
    pc1[0, 0, 0] = 1.0
    pc2 = np.zeros((2, 2, 2), dtype=np.float32)
    pc2[1, 1, 1] = -2.0
    scores = np.array([[-1.0, -0.5], [0.0, 1.0], [1.0, -0.5]], dtype=np.float32)
    volumes = np.stack([mu + a * pc1 + b * pc2 for a, b in scores], axis=0)
    return mu, np.stack([pc1, pc2], axis=0), volumes


def test_kclass_initialization_covariance_from_w_matches_weighted_covariance():
    _mu, _pcs, volumes = _synthetic_volumes()
    weights = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    init = initialize_ppca_from_kclass_volumes(volumes, q=2, class_weights=weights, frame="recovar")

    expected_cov = empirical_weighted_covariance(init.aligned_volumes, weights)
    actual_cov = covariance_from_loading_matrix(init.W)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=1e-5, atol=1e-6)
    expected_mean = np.sum((weights / weights.sum())[:, None, None, None] * volumes, axis=0)
    np.testing.assert_allclose(init.mu, expected_mean, rtol=1e-6, atol=1e-6)
    assert init.diagnostics["latent_prior"] == "identity"
    assert init.diagnostics["W_stores_covariance_scale"] is True


def test_relion_to_recovar_volume_conversion_is_applied_once():
    vol_recovar = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    vol_relion = utils.recovar_volume_to_relion(vol_recovar)

    loaded_from_relion = load_volume_stack([vol_relion], frame="relion")[0]
    loaded_from_recovar = load_volume_stack([vol_recovar], frame="recovar")[0]

    np.testing.assert_allclose(loaded_from_relion, vol_recovar)
    np.testing.assert_allclose(loaded_from_recovar, vol_recovar)
    assert not np.allclose(utils.relion_volume_to_recovar(loaded_from_relion), vol_recovar)


def test_gt_initialization_requires_explicit_frame_and_preserves_scale():
    _mu, _pcs, volumes = _synthetic_volumes()
    with pytest.raises(ValueError, match="explicit frame"):
        initialize_ppca_from_gt_volumes(volumes, q=1, frame=None)

    init = initialize_ppca_from_gt_volumes(volumes, q=1, frame="recovar", amplitude_scale=2.5)
    scaled = volumes * 2.5
    np.testing.assert_allclose(init.mu, np.mean(scaled, axis=0), rtol=1e-6, atol=1e-6)
    assert init.diagnostics["amplitude_scale"] == 2.5


def test_synthetic_known_pcs_recovered_up_to_subspace_rotation():
    _mu, pcs, volumes = _synthetic_volumes()
    init = initialize_ppca_from_gt_volumes(volumes, q=2, frame="recovar")
    agreement = loading_subspace_agreement(init.W, pcs)
    assert agreement > 1.0 - 1e-6
