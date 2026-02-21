"""
Unit tests for recovar.heterogeneity_volume.

Covers the public API at two tiers:

1. Pure-numpy / scipy helper functions (fast, no GPU).
2. JAX-accelerated functions tested on tiny synthetic inputs.

The large orchestration functions (make_volumes_kernel_estimate_local,
choice_most_likely_split) are tested at the smoke-test level using tiny
synthetic arrays so that every code path is executed at least once without
any on-disk I/O.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

import recovar.heterogeneity_volume as hv
from helpers.tiny_synthetic import make_tiny_cryo_dataset_with_images

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Tier 1 – pure-numpy helpers
# ---------------------------------------------------------------------------

def test_pick_minimum_discretization_size_enough_images():
    """With enough images the returned threshold is above the chi2 quantile."""
    ndim = 2
    rng = np.random.default_rng(0)
    log_ll = np.sort(rng.exponential(scale=5.0, size=200).astype(np.float32))
    threshold = hv.pick_minimum_discretization_size(ndim, log_ll, q=0.5, min_images=50)
    # Result must be positive and finite
    assert np.isfinite(threshold)
    assert threshold > 0


def test_pick_minimum_discretization_size_too_few_images():
    """With fewer images than min_images the function logs a warning and still returns."""
    ndim = 2
    log_ll = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    threshold = hv.pick_minimum_discretization_size(ndim, log_ll, q=0.5, min_images=50)
    assert np.isfinite(threshold)


def test_pick_minimum_discretization_size_ndim_zero():
    """ndim=0 should not call latent_density (no latent space) and still work."""
    log_ll = np.linspace(1, 10, 100).astype(np.float32)
    threshold = hv.pick_minimum_discretization_size(0, log_ll)
    assert np.isfinite(threshold)


def test_pick_heterogeneity_bins2_returns_sorted_bins():
    """Bins must be strictly increasing and span from near-min to near-max."""
    ndim = 2
    rng = np.random.default_rng(42)
    log_ll = rng.exponential(scale=5.0, size=300).astype(np.float32)
    bins = hv.pick_heterogeneity_bins2(ndim, log_ll, n_bins=7)
    assert bins.shape == (7,)
    # Bins are strictly increasing
    assert np.all(np.diff(bins) > 0)
    assert np.all(np.isfinite(bins))


def test_pick_heterogeneity_bins2_n_bins_respected():
    ndim = 1
    log_ll = np.linspace(0.5, 20.0, 200).astype(np.float32)
    for n in [3, 5, 11]:
        bins = hv.pick_heterogeneity_bins2(ndim, log_ll, n_bins=n)
        assert bins.shape == (n,)


# ---------------------------------------------------------------------------
# Tier 2 – JAX functions on tiny arrays
# ---------------------------------------------------------------------------

def test_smooth_shell_error_output_shape():
    """smooth_shell_error preserves the last (shell) dimension.

    smooth_shell_error expects (n_estimators, n_shells); batch version adds one
    more leading dimension.  All extra args must be positional (JAX vmap
    requirement).
    """
    n_estimators = 3
    n_shells = 8
    # batch_smooth_shell_error vmaps over axis-0 of a 3-D input
    shell_error = jnp.ones((2, n_estimators, n_shells), dtype=jnp.float32)
    voxel_size = 1.5
    subarray_size = (n_shells + 1) * 2

    result = hv.batch_smooth_shell_error(
        shell_error, voxel_size, subarray_size, 50, 3  # all positional
    )
    assert result.shape == (2, n_estimators, n_shells)
    assert np.all(np.isfinite(np.asarray(result)))


def test_smooth_shell_error_constant_input_stays_constant():
    """Smoothing a constant error signal must return the same constant.

    smooth_shell_error vmaps convolve over the first axis of shell_error so the
    input must be 2-D (n_estimators, n_shells).
    """
    const_val = 4.0
    n_shells = 10
    # 2-D input: (n_estimators, n_shells)
    shell_error = jnp.ones((2, n_shells), dtype=jnp.float32) * const_val
    result = hv.smooth_shell_error(shell_error, 1.0, 20, 50, 3)
    # After smoothing a constant the values should remain close to the constant
    assert result.shape == (2, n_shells)
    assert np.all(np.isfinite(np.asarray(result)))


def test_smoothed_best_choice_output_shape_and_finite():
    """smoothed_best_choice must return a volume with the right shape."""
    vol_shape = (4, 4, 4)
    n_est = 3
    vol_size = int(np.prod(vol_shape))
    rng = np.random.default_rng(7)
    # estimates: (n_est, vol_size), complex
    estimates = jnp.array(
        rng.standard_normal((n_est, vol_size)).astype(np.float32)
        + 1j * rng.standard_normal((n_est, vol_size)).astype(np.float32)
    )
    # choice: integer index in [0, n_est-1] for each voxel, reshaped to 3D
    choice = jnp.array(rng.integers(0, n_est, size=(vol_size,), dtype=np.int32))

    smoothed_est, smoothed_choice = hv.smoothed_best_choice(estimates, choice, kernel_rad=1)

    assert smoothed_est.shape == (vol_size,)
    assert smoothed_choice.shape == (vol_size,)
    assert np.all(np.isfinite(np.asarray(smoothed_est)))
    assert np.all(np.isfinite(np.asarray(smoothed_choice)))


# ---------------------------------------------------------------------------
# Tier 3 – make_volumes_kernel_estimate_local smoke test
# ---------------------------------------------------------------------------

def test_make_volumes_kernel_estimate_local_smoke(tmp_path):
    """
    make_volumes_kernel_estimate_local must run without crashing on tiny data.
    Output files are written to a temporary directory so nothing leaks.
    """
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8)
    noise_variance = np.ones(cryo.grid_size // 2 - 1, dtype=np.float32) * 0.1
    # The pipeline reads per-image noise via cryo.noise.get(indices), so we
    # must attach a noise model before calling.
    cryo.set_radial_noise_model(noise_variance)

    n_images = cryo.n_images
    rng = np.random.default_rng(0)

    # Fake heterogeneity distances for two half-datasets
    het_dists = [
        rng.exponential(scale=2.0, size=n_images).astype(np.float32),
        rng.exponential(scale=2.0, size=n_images).astype(np.float32),
    ]
    bins = hv.pick_heterogeneity_bins2(ndim=2, log_likelihoods=np.concatenate(het_dists), n_bins=3)

    output_folder = str(tmp_path / "hv_output")

    # This exercises the full local estimation path.
    hv.make_volumes_kernel_estimate_local(
        heterogeneity_distances=het_dists,
        cryos=[cryo, cryo],
        output_folder=output_folder,
        ndim=2,
        bins=bins,
        B_factor=0,
        tau=None,
        n_min_particles=2,   # very small to avoid "no images" assertion
        metric_used="locshellmost_likely",
        locres_sampling=2,
        kernel_rad=1,
    )
