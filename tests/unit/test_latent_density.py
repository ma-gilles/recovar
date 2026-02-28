import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.heterogeneity import latent_density as ld

pytestmark = pytest.mark.unit


def test_grid_mapping_roundtrip():
    bounds = np.array([[-2.0, 2.0], [0.0, 10.0]], dtype=np.float32)
    num_points = 11
    x = np.array([0.4, 6.0], dtype=np.float32)
    g = ld.pca_coord_to_grid(x, bounds, num_points, to_int=False)
    x_rt = ld.grid_to_pca_coord(g, bounds, num_points)
    assert np.allclose(x_rt, x, atol=1e-6)

    gi = ld.pca_coord_to_grid(x, bounds, num_points, to_int=True)
    assert gi.dtype.kind in ("i", "u")
    assert gi.shape == (2,)


def test_get_grid_z_mappings_callable():
    bounds = np.array([[-1.0, 1.0]], dtype=np.float32)
    grid_to_z, z_to_grid = ld.get_grid_z_mappings(bounds, num_points=5)
    z = np.array([0.25], dtype=np.float32)
    g = z_to_grid(z, to_int=False)
    z_back = grid_to_z(g)
    assert np.allclose(z_back, z, atol=1e-6)


def test_make_latent_space_grid_from_bounds_shape():
    bounds = np.array([[-1.0, 1.0], [0.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    grid = ld.make_latent_space_grid_from_bounds(bounds, num_points=4)
    assert grid.shape == (4**3, 3)
    assert np.all(np.isfinite(np.asarray(grid)))


def test_compute_latent_space_bounds_percentile():
    zs = np.array(
        [
            [-10.0, 1.0],
            [0.0, 2.0],
            [10.0, 3.0],
            [100.0, 4.0],
        ],
        dtype=np.float32,
    )
    b = ld.compute_latent_space_bounds(zs, percentile=25)
    assert b.shape == (2, 2)
    assert b[0, 0] >= -10.0
    assert b[0, 1] <= 100.0
    assert b[1, 0] >= 1.0
    assert b[1, 1] <= 4.0


def test_compute_weights_of_conformation_thresholding():
    latent_points = np.array([[0.0, 0.0]], dtype=np.float32)
    zs = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    cov = np.stack([np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)], axis=0)
    w = ld.compute_weights_of_conformation_2(latent_points, zs, cov, likelihood_threshold=4.0)
    assert w.shape == (2, 1)
    assert w[0, 0] == 1.0
    assert w[1, 0] == 0.0


def test_compute_det_cov_xs_normalized_max_is_one():
    cov = np.stack(
        [
            np.eye(2, dtype=np.float32),
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        ],
        axis=0,
    )
    d = np.asarray(ld.compute_det_cov_xs(cov))
    assert d.shape == (2,)
    assert np.isclose(np.max(d), 1.0)
    assert np.all(d > 0)
