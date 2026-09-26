import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from recovar import utils
from recovar.ppca import prior_estimation

pytestmark = pytest.mark.unit


def _radial_volume(shell_values, volume_shape):
    return np.asarray(utils.make_radial_image(jnp.array(shell_values, dtype=jnp.float32), volume_shape)).reshape(-1)


def test_make_estimated_prior_from_combined_matches_gt_when_shells_are_reliable():
    volume_shape = (16, 16, 16)
    npc = 2
    shell_total = np.array([16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25], dtype=np.float32)
    mean_sq_shells = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125], dtype=np.float32)

    combined = _radial_volume(shell_total, volume_shape)
    mean_estimate = np.sqrt(_radial_volume(mean_sq_shells, volume_shape)).astype(np.float32)

    gt_prior = prior_estimation.make_gt_prior_from_variance_total(combined, npc, volume_shape)
    est_prior = prior_estimation.make_estimated_prior_from_combined(combined, mean_estimate, npc, volume_shape)

    np.testing.assert_allclose(est_prior["raw_shell_total"], shell_total, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(est_prior["repaired_shell_total"], shell_total, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(est_prior["radial_raw"], gt_prior["radial_raw"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(est_prior["radial_used"], gt_prior["radial_used"], atol=1e-6, rtol=1e-6)
    assert not np.any(est_prior["tail_fallback_mask"])
    assert est_prior["last_reliable_shell"] == shell_total.size - 1


def test_make_estimated_prior_from_combined_repairs_unreliable_tail_only():
    volume_shape = (16, 16, 16)
    npc = 1
    shell_total = np.array([10.0, 8.0, 6.0, 4.0, 2.0, -1.0, -2.0], dtype=np.float32)
    mean_sq_shells = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.25, 0.125], dtype=np.float32)

    combined = _radial_volume(shell_total, volume_shape)
    mean_estimate = np.sqrt(_radial_volume(mean_sq_shells, volume_shape)).astype(np.float32)

    est_prior = prior_estimation.make_estimated_prior_from_combined(combined, mean_estimate, npc, volume_shape)

    expected_ratio = 2.0
    expected_tail = mean_sq_shells[-2:] * expected_ratio

    np.testing.assert_allclose(est_prior["repaired_shell_total"][:5], shell_total[:5], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(est_prior["repaired_shell_total"][-2:], expected_tail, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(est_prior["tail_fallback_mask"], np.array([False, False, False, False, False, True, True]))
    assert est_prior["last_reliable_shell"] == 4
    assert np.isclose(est_prior["median_ratio"], expected_ratio)


def test_repair_shell_total_with_mean_sq_falls_back_when_no_shell_is_reliable():
    shell_total = np.array([0.0, -1.0, np.nan, -2.0], dtype=np.float32)
    mean_sq_shells = np.array([4.0, 2.0, 1.0, 0.5], dtype=np.float32)

    repaired = prior_estimation.repair_shell_total_with_mean_sq(shell_total, mean_sq_shells)

    np.testing.assert_allclose(repaired["repaired_shell_total"], np.maximum(mean_sq_shells, 1e-8), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(repaired["tail_fallback_mask"], np.ones(shell_total.size, dtype=bool))
    assert repaired["last_reliable_shell"] == -1
    assert repaired["median_ratio"] == 1.0
