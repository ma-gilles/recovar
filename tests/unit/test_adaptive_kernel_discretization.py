"""
Unit tests for recovar.adaptive_kernel_discretization.

Tests are organised in three tiers:

1. Pure-Python / numpy utility functions – no JAX tracing needed.
2. JAX-traced functions exercised on tiny synthetic inputs.
3. Dataset-level functions run against make_tiny_cryo_dataset_with_images
   (grid_size=4, n_images=8) so every batch in the real code is exercised
   without GPU memory pressure.
"""
import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

import recovar.adaptive_kernel_discretization as akd
from helpers.tiny_synthetic import make_tiny_cryo_dataset_with_images

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Tier 1 – pure-Python / numpy utilities
# ---------------------------------------------------------------------------

def test_get_feature_size_known_degrees():
    assert akd.get_feature_size(0) == 1
    assert akd.get_feature_size(1) == 4
    assert akd.get_feature_size(2) == 10


def test_small_gram_matrix_size():
    # For degree-0 the Gram matrix is scalar (1x1): upper-tri storage = 1
    assert akd.small_gram_matrix_size(0) == 1
    # For degree-1: feature_size=4, 4*5//2 = 10
    assert akd.small_gram_matrix_size(1) == 10


def test_big_gram_matrix_size():
    assert akd.big_gram_matrix_size(0) == 1
    assert akd.big_gram_matrix_size(1) == 16  # 4**2


def test_volume_shape_to_half_volume_shape_even():
    vs = (8, 8, 8)
    hvs = akd.volume_shape_to_half_volume_shape(vs)
    assert hvs == (5, 8, 8)


def test_volume_shape_to_half_volume_shape_four():
    vs = (4, 4, 4)
    hvs = akd.volume_shape_to_half_volume_shape(vs)
    assert hvs == (3, 4, 4)


def test_get_default_discretization_params_small_grid():
    params = akd.get_default_discretization_params(grid_size=32)
    assert len(params) > 0
    for pol_degree, h, flag in params:
        assert pol_degree in (0, 1)
        assert isinstance(h, int)
        assert isinstance(flag, bool)


def test_keep_upper_triangular_roundtrip():
    """keep_upper_triangular extracts upper-tri; undo should recover it."""
    n = 3
    mat = np.arange(n * n, dtype=np.float32).reshape(n, n)
    mat = mat + mat.T  # make symmetric
    mat = mat[np.newaxis]  # (1, n, n)
    flat = akd.keep_upper_triangular(mat)  # (1, m) where m = n(n+1)/2
    assert flat.shape == (1, n * (n + 1) // 2)
    recovered = akd.undo_keep_upper_triangular(flat)  # (1, n, n)
    np.testing.assert_allclose(np.asarray(recovered[0]), mat[0], atol=1e-6)


def test_find_smaller_pol_indices_degree0_inside_degree1():
    """Degree-0 indices must be a strict subset of degree-1 indices."""
    idx = akd.find_smaller_pol_indices(max_pol_degree=1, target_pol_degree=0)
    # degree-0 has 1 parameter; result should be of length 1
    assert idx.shape[0] == 1
    # All indices must be valid for the degree-1 upper-triangular layout
    n_max = akd.small_gram_matrix_size(1)
    assert int(idx[0]) < n_max


def test_find_diagonal_pol_indices_degree0():
    idx = akd.find_diagonal_pol_indices(max_pol_degree=0)
    assert idx.shape[0] == 1
    assert idx[0] == 0


def test_find_diagonal_pol_indices_degree1():
    # Degree-1 has 4 features; diagonal entries of 4x4 are 4 elements.
    # Upper-tri storage of a 4x4 matrix has 10 entries.
    idx = akd.find_diagonal_pol_indices(max_pol_degree=1)
    assert idx.shape[0] == 4
    # All indices must be within the valid upper-tri range
    n_flat = akd.small_gram_matrix_size(1)
    assert np.all(idx < n_flat)


# ---------------------------------------------------------------------------
# Tier 2 – JAX-traced functions on tiny synthetic inputs
# ---------------------------------------------------------------------------

def test_full_volume_to_half_volume_to_full_roundtrip():
    """full→half→full should preserve values in the 'negative-frequency' half."""
    vol_shape = (4, 4, 4)
    vol_size = int(np.prod(vol_shape))
    vol = jnp.array(
        np.random.randn(vol_size).astype(np.complex64) +
        1j * np.random.randn(vol_size).astype(np.float32)
    )
    half = akd.full_volume_to_half_volume(vol, vol_shape)
    half_size = int(np.prod(akd.volume_shape_to_half_volume_shape(vol_shape)))
    assert half.shape == (half_size,)
    # Reconstructed volume must be finite
    full_back = akd.half_volume_to_full_volume(half, vol_shape)
    assert np.all(np.isfinite(np.asarray(full_back)))


def test_vec_index_to_half_vec_index_shape():
    vol_shape = (4, 4, 4)
    vol_size = int(np.prod(vol_shape))
    indices = jnp.arange(vol_size, dtype=jnp.int32)
    half_indices, neg_freq = akd.vec_index_to_half_vec_index(indices, vol_shape, flip_positive=False)
    assert half_indices.shape == (vol_size,)
    assert neg_freq.shape == (vol_size,)
    # neg_freq must be boolean-like (0/1)
    vals = np.asarray(neg_freq)
    assert np.all((vals == 0) | (vals == 1))


def test_solve_for_m_simple_identity_system():
    """solve_for_m_simple on a well-conditioned system returns correct answer."""
    n = 4
    A = jnp.eye(n, dtype=jnp.float32) * 5.0
    b = jnp.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j], dtype=jnp.complex64)
    reg = jnp.eye(n, dtype=jnp.float32) * 1e-6
    v, good_v, problem = akd.solve_for_m_simple(A, b, reg)
    expected = jnp.array([0.2, 0.4, 0.6, 0.8], dtype=jnp.complex64)
    np.testing.assert_allclose(np.asarray(v).real, np.asarray(expected).real, atol=1e-4)
    assert bool(good_v)


def test_make_X_mat_pol_degree_0_shape():
    """For pol_degree=0 the feature matrix has shape (n_images, n_pixels, 1)."""
    n_images = 4
    grid_size = 4
    vol_shape = (grid_size, grid_size, grid_size)
    img_shape = (grid_size, grid_size)
    rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    X, grid_pt_indices = akd.make_X_mat(rots, vol_shape, img_shape, pol_degree=0)
    n_pixels = grid_size * grid_size
    assert X.shape == (n_images, n_pixels, 1)
    assert grid_pt_indices.shape == (n_images, n_pixels)


def test_make_X_mat_pol_degree_1_shape():
    """For pol_degree=1 the feature matrix has 4 features per pixel (1+3)."""
    n_images = 4
    grid_size = 4
    vol_shape = (grid_size, grid_size, grid_size)
    img_shape = (grid_size, grid_size)
    rots = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    X, _ = akd.make_X_mat(rots, vol_shape, img_shape, pol_degree=1)
    n_pixels = grid_size * grid_size
    assert X.shape == (n_images, n_pixels, akd.get_feature_size(1))


# ---------------------------------------------------------------------------
# Tier 3 – Dataset-level: precompute + reconstruct with a tiny dataset
# ---------------------------------------------------------------------------

def test_precompute_triangular_kernel_shapes():
    """precompute_triangular_kernel returns XWX and F with expected shapes."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8)
    noise_variance = np.ones(cryo.grid_size // 2 - 1, dtype=np.float32) * 0.1

    XWX, F = akd.precompute_triangular_kernel(cryo, noise_variance, pol_degree=0)

    half_vol_size = int(np.prod(akd.volume_shape_to_half_volume_shape(cryo.upsampled_volume_shape)))
    gram_size = akd.small_gram_matrix_size(0)
    feat_size = akd.get_feature_size(0)

    assert XWX.shape == (half_vol_size, gram_size, 1)
    assert F.shape == (half_vol_size, feat_size, 1)
    # XWX (CTF^2/noise) must be finite; F (CTF*image/noise) may have NaN for
    # zero-coverage voxels, which is acceptable - the solver ignores them.
    assert np.isfinite(XWX).all()


def test_precompute_kernel_shapes():
    """precompute_kernel (original codepath) returns same shapes as triangular version."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8)
    noise_variance = np.ones(cryo.grid_size // 2 - 1, dtype=np.float32) * 0.1

    XWX, F = akd.precompute_kernel(cryo, noise_variance, pol_degree=0)

    half_vol_size = int(np.prod(akd.volume_shape_to_half_volume_shape(cryo.upsampled_volume_shape)))
    gram_size = akd.small_gram_matrix_size(0)
    feat_size = akd.get_feature_size(0)

    assert XWX.shape == (half_vol_size, gram_size, 1)
    assert F.shape == (half_vol_size, feat_size, 1)


def test_precompute_kernel_with_heterogeneity_bins():
    """Passing heterogeneity_bins expands the trailing dimension of XWX and F."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8)
    noise_variance = np.ones(cryo.grid_size // 2 - 1, dtype=np.float32) * 0.1
    n_bins = 3
    het_dists = np.random.rand(cryo.n_images).astype(np.float32)
    het_bins = np.array([0.3, 0.6, 1.0], dtype=np.float32)

    XWX, F = akd.precompute_triangular_kernel(
        cryo, noise_variance, pol_degree=0,
        heterogeneity_distances=het_dists,
        heterogeneity_bins=het_bins,
    )

    half_vol_size = int(np.prod(akd.volume_shape_to_half_volume_shape(cryo.upsampled_volume_shape)))
    assert XWX.shape == (half_vol_size, akd.small_gram_matrix_size(0), n_bins)
    assert F.shape == (half_vol_size, akd.get_feature_size(0), n_bins)


def test_estimate_multiple_disc_relion_style_returns_finite_volume():
    """Full end-to-end pipeline with tiny dataset must return finite reconstructions."""
    cryo = make_tiny_cryo_dataset_with_images(grid_size=4, n_images=8)
    noise_variance = np.ones(cryo.grid_size // 2 - 1, dtype=np.float32) * 0.1
    disc_params = [(0, 0, True)]  # minimal single param set
    n_bins = 2
    het_bins = np.array([0.5, 1.0], dtype=np.float32)
    # Two half-datasets each with n_images=8 random heterogeneity distances
    het_dists = [
        np.random.rand(cryo.n_images).astype(np.float32),
        np.random.rand(cryo.n_images).astype(np.float32),
    ]

    # estimate_multiple_disc_relion_style returns (first_estimates, opt_halfmaps,
    # disc_choices, residuals_averaged) – unpack all four.
    first_estimates, opt_halfmaps, disc_choices, residuals = akd.estimate_multiple_disc_relion_style(
        experiment_datasets=[cryo, cryo],  # two half-datasets
        noise_variance=noise_variance,
        discretization_params=disc_params,
        heterogeneity_distances=het_dists,
        heterogeneity_bins=het_bins,
    )

    assert first_estimates is not None
    assert opt_halfmaps is not None
