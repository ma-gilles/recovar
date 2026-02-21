import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("jax")

from helpers import tiny_synthetic

pytestmark = pytest.mark.unit


def test_make_tiny_fourier_volumes_shape_and_dtype():
    vols = tiny_synthetic.make_tiny_fourier_volumes(grid_size=4)
    assert vols.shape == (2, 64)
    assert vols.dtype == np.complex64
    assert np.all(np.isfinite(vols))


def test_tiny_ctf_pose_generator_shapes():
    ctf, rots, trans = tiny_synthetic.tiny_ctf_pose_generator(n_images=5, grid_size=4)
    assert ctf.shape == (5, 9)
    assert rots.shape == (5, 3, 3)
    assert trans.shape == (5, 2)


def test_make_tiny_simulation_and_hvd_end_to_end():
    stack, ctf_params, rots, trans, sim_info, _, _ = tiny_synthetic.make_tiny_simulation(grid_size=4, n_images=6, seed=0)
    assert stack.shape == (6, 4, 4)
    assert ctf_params.shape[0] == 6
    assert rots.shape == (6, 3, 3)
    assert trans.shape == (6, 2)
    assert "image_assignment" in sim_info
    assert "per_image_contrast" in sim_info

    hvd, sim_info2, vols = tiny_synthetic.make_tiny_hvd_from_simulation(grid_size=4, n_images=6, seed=0)
    assert hvd.volumes.shape == vols.shape
    assert sim_info2["image_assignment"].shape[0] == 6
    assert np.isfinite(hvd.get_mean()).all()


def test_make_tiny_loader_files(tmp_path):
    files = tiny_synthetic.make_tiny_loader_files(tmp_path, grid_size=8, n_images=6, n_particles=3)
    assert set(["particles_mrcs", "particles_star", "poses_pkl", "ctf_pkl"]).issubset(files.keys())
    for key in ["particles_mrcs", "particles_star", "poses_pkl", "ctf_pkl"]:
        assert (tmp_path / Path(files[key]).name).exists()
