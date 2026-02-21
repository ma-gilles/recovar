import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import csparc_bind
from recovar import utils


def _write_particles_cs(path, z0, z1):
    arr = np.zeros(
        z0.shape[0],
        dtype=[("components_mode_0/value", np.float32), ("components_mode_1/value", np.float32)],
    )
    arr["components_mode_0/value"] = z0
    arr["components_mode_1/value"] = z1
    with open(path, "wb") as f:
        np.save(f, arr)


def test_load_3dva_results_roundtrip(tmp_path):
    root = str(tmp_path / "job")
    mean = np.ones((4, 4, 4), dtype=np.float32)
    c0 = np.full((4, 4, 4), 2.0, dtype=np.float32)
    c1 = np.full((4, 4, 4), 3.0, dtype=np.float32)
    utils.write_mrc(root + "_map.mrc", mean)
    utils.write_mrc(root + "_component_0.mrc", c0)
    utils.write_mrc(root + "_component_1.mrc", c1)
    _write_particles_cs(root + "_particles.cs", np.array([0.1, 0.2]), np.array([0.3, 0.4]))

    out_mean, components, zs = csparc_bind.load_3dva_results(root, dft=False)
    assert out_mean.shape == mean.shape
    assert components.shape == (4 * 4 * 4, 2)
    assert zs.shape == (2, 2)


def test_load_3dflex_results_dft_shapes(tmp_path):
    root = str(tmp_path / "job2")
    mean = np.ones((4, 4, 4), dtype=np.float32)
    c0 = np.full((4, 4, 4), 2.0, dtype=np.float32)
    utils.write_mrc(root + "_map.mrc", mean)
    utils.write_mrc(root + "_component_0.mrc", c0)
    _write_particles_cs(root + "_particles.cs", np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.0]))

    out_mean, components_dft, zs = csparc_bind.load_3dflex_results(root, dft=True)
    assert out_mean.ndim == 1
    assert components_dft.ndim == 2
    assert zs.shape[0] == 3

