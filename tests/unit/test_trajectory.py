import numpy as np
import pytest

pytest.importorskip("skfmm")

from recovar.heterogeneity import trajectory

pytestmark = pytest.mark.unit



def test_get_cum_curvelength_monotonic():
    pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    d = trajectory.get_cum_curvelength(pts)
    assert np.all(d[1:] >= d[:-1])
    assert np.isclose(d[0], 0.0)
    assert np.isclose(d[-1], 3.0)


def test_resample_at_uniform_pts_shapes():
    pts = np.stack([np.linspace(0, 3, 6), np.zeros(6)], axis=1).astype(np.float32)
    out = trajectory.resample_at_uniform_pts(pts, n_vols_along_path=4)
    assert out.shape == (4, 2)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[-1, 0], 3.0)



def test_get_grid_spacing():
    bounds = [(-2.0, 2.0), (0.0, 1.0), (4.0, 8.0)]
    density = np.zeros((8, 4, 2), dtype=np.float32)
    dx = trajectory.get_grid_spacing(bounds, density)
    assert np.allclose(dx, [0.5, 0.25, 2.0])


def test_gradient_descent_nd_stays_finite_and_moves_toward_start(monkeypatch):
    x_st = np.array([0.0, 0.0], dtype=np.float32)
    x_end = np.array([3.0, 3.0], dtype=np.float32)
    travel_time = np.zeros((8, 8), dtype=np.float32)

    # Use a convex surrogate objective over candidate points.
    monkeypatch.setattr(
        trajectory,
        "evaluate_function_off_grid",
        lambda _tt, pts: np.sum((pts - x_st[None, :]) ** 2, axis=1),
    )

    path = trajectory.gradient_descent_nd(
        travel_time=travel_time,
        x_st=x_st,
        x_end=x_end,
        dx=[1.0, 1.0],
        step_size=0.5,
        n_theta=5,
        max_steps=100,
    )

    assert path is not None
    assert path.ndim == 2 and path.shape[1] == 2
    assert np.all(np.isfinite(path))
    # Returned path is [start ... end] by construction.
    assert np.linalg.norm(path[0] - x_st) < 1e-6
    assert np.linalg.norm(path[-1] - x_end) < 1e-6


def test_compute_fixed_dimensional_path_runs_on_smooth_density():
    n = 32
    axis = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    density = np.exp(-(xx**2 + yy**2))
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float32)

    z_st = np.array([-1.5, -1.0], dtype=np.float32)
    z_end = np.array([1.5, 1.0], dtype=np.float32)
    path = trajectory.compute_fixed_dimensional_path(
        z_st=z_st,
        z_end=z_end,
        density_low_dim=density,
        latent_space_bounds=bounds,
        density_eps=1e-4,
        use_log_density=False,
    )

    assert path.ndim == 2 and path.shape[1] == 2
    assert path.shape[0] > 10
    assert np.all(np.isfinite(path))
    # Endpoint coordinates are quantized to grid; check they land near the requested endpoints.
    assert np.linalg.norm(path[0] - z_st) < 0.2
    assert np.linalg.norm(path[-1] - z_end) < 0.2


def test_compute_high_dimensional_path_base_case_low_dim_only():
    n = 32
    axis = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    density_2d = np.exp(-(xx**2 + yy**2))

    rng = np.random.default_rng(0)
    zs = rng.normal(size=(120, 3)).astype(np.float32)
    cov_zs = np.tile(np.eye(3, dtype=np.float32)[None], (120, 1, 1)) * 0.1

    path = trajectory.compute_high_dimensional_path(
        zs=zs,
        cov_zs=cov_zs,
        z_st=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
        z_end=np.array([1.0, 1.0, 0.5], dtype=np.float32),
        density_low_dim=density_2d,
        density_eps=1e-4,
        max_dim=2,
        percentile_bound=5,
        use_log_density=False,
    )

    assert path.ndim == 2 and path.shape[1] == 2
    assert path.shape[0] > 10
    assert np.all(np.isfinite(path))
