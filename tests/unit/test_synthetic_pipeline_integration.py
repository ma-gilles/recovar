import numpy as np
import pytest

pytest.importorskip("jax")

import recovar.output.metrics as metrics
from helpers.tiny_synthetic import make_tiny_hvd_from_simulation, make_tiny_simulation

pytestmark = pytest.mark.unit


def test_simulator_to_synthetic_dataset_to_metrics_tiny():
    grid_size = 4
    n_images = 8
    volume_size = int(grid_size**3)
    main_image_stack, ctf_params, rots, trans, _, _, _ = make_tiny_simulation(
        grid_size=grid_size, n_images=n_images, seed=0
    )

    assert main_image_stack.shape == (n_images, grid_size, grid_size)
    assert ctf_params.shape[0] == n_images
    assert rots.shape == (n_images, 3, 3)
    assert trans.shape == (n_images, 2)

    hvd, _, _ = make_tiny_hvd_from_simulation(grid_size=grid_size, n_images=n_images, seed=0)

    mean = hvd.get_mean()
    assert mean.shape == (volume_size,)
    assert np.all(np.isfinite(mean))

    picked = np.array([0, 1, 2], dtype=np.int32)
    cov_cols = hvd.get_covariance_columns(picked, contrasted=False)
    assert cov_cols.shape == (volume_size, picked.size)

    u, s, _ = hvd.get_vol_svd(contrasted=False, real_space=False, random_svd_pcs=None)
    assert u.shape[0] == volume_size
    assert s.ndim == 1
    assert np.all(s >= 0)

    _, rel_var, norm_var = metrics.get_all_variance_scores(u, u, s)
    assert rel_var[-1] == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert np.all(norm_var <= 1.0 + 1e-6)
    angles = metrics.subspace_angles(u, u, max_rank=min(5, u.shape[1]))
    assert np.all(angles <= 1e-5)
