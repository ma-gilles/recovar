import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.reconstruction.noise as noise

pytestmark = pytest.mark.unit


def test_noise_upper_bound_ignores_zero_nonfinite_and_negative_shells():
    """Unsupported shells must not be converted into zero-noise shells."""
    original = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    upper_bound = np.array([5.0, 0.0, np.nan, np.inf, -1.0], dtype=np.float32)

    capped = noise._cap_noise_with_valid_upper_bound(original, upper_bound)

    np.testing.assert_array_equal(capped, np.array([5.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32))
    assert np.all(np.isfinite(capped))
    assert np.all(capped > 0)


def test_observed_shell_percentile_ignores_voxels_without_data():
    observed = np.linspace(80.0, 120.0, 96)
    shell = np.concatenate([observed, np.zeros(4)])  # 4% of the shell has no data

    assert noise._observed_shell_percentile(shell, 5) == np.percentile(observed, 5)
    assert noise._observed_shell_percentile(np.zeros(10), 5) == 0.0

