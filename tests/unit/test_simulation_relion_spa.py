"""Simulated multi-optics-group RELION SPA datasets: files, optics table and images."""

import mrcfile
import numpy as np
import pytest

from recovar import utils
from recovar.data_io import starfile
from recovar.simulation import relion_spa

GRID = 32
VOXEL = 4.0
GROUPS = (
    dict(voltage=300.0, cs=2.7, amp_contrast=0.1, noise_scale=1.0),
    dict(voltage=200.0, cs=1.4, amp_contrast=0.07, noise_scale=1.5, pixel_size=VOXEL * GRID / 24, box_size=28),
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("relion_spa")
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-((xx - 12) ** 2 + yy**2 + zz**2) / 200) + 0.5 * np.exp(-(xx**2 + (yy + 16) ** 2 + zz**2) / 100)
    with mrcfile.new(root / "vol0000.mrc") as mrc:
        mrc.set_data(vol.astype(np.float32))
        mrc.voxel_size = VOXEL
    out = root / "project"
    result = relion_spa.generate_relion_spa_dataset(
        str(out), str(root / "vol"), VOXEL, 40, grid_size=GRID, optics_groups=GROUPS, micrographs_per_group=3, seed=5
    )
    return out, result


def test_optics_table_and_stacks(dataset):
    out, result = dataset
    particles, optics = starfile.read_star(result["particles"])
    assert list(optics["_rlnImageSize"].astype(int)) == [GRID, 28]
    np.testing.assert_allclose(optics["_rlnImagePixelSize"].astype(float), [VOXEL, VOXEL * GRID / 24])
    np.testing.assert_allclose(optics["_rlnVoltage"].astype(float), [300.0, 200.0])
    groups = particles["_rlnOpticsGroup"].astype(int).values
    assert np.array_equal(groups, np.arange(40) % 2 + 1)
    for g, box in ((1, GRID), (2, 28)):
        names = particles["_rlnImageName"].values[groups == g]
        assert all(name.endswith(f"Particles/opticsGroup{g}.mrcs") for name in names)
        assert sorted(int(name.split("@")[0]) for name in names) == list(range(1, names.size + 1))
        with mrcfile.open(out / f"Particles/opticsGroup{g}.mrcs") as mrc:
            stack = np.asarray(mrc.data)
            assert stack.shape == (names.size, box, box)
            assert np.isclose(float(mrc.voxel_size.x), float(optics["_rlnImagePixelSize"].values[g - 1]), rtol=1e-4)
        # relion_preprocess --norm: background mean 0 and standard deviation 1 per image.
        # Background as simulator.normalize_particles_relion_style defines it.
        c = np.arange(box) - (box / 2 - 0.5)
        background = np.hypot(*np.meshgrid(c, c, indexing="ij")) > round(0.375 * box)
        np.testing.assert_allclose(stack[:, background].mean(axis=1), 0.0, atol=1e-4)
        np.testing.assert_allclose(stack[:, background].std(axis=1), 1.0, rtol=1e-3)
    micrographs = particles["_rlnMicrographName"].values
    assert all(f"opticsGroup{g}_" in m for g, m in zip(groups, micrographs))
    assert len(set(micrographs)) <= 6


def test_star_angles_and_ctf_match_simulation(dataset):
    _, result = dataset
    particles, _ = starfile.read_star(result["particles"])
    info = result["simulation_info"]
    eulers = particles[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float)
    np.testing.assert_allclose(utils.R_from_relion(eulers, degrees=True), info["rots"], atol=1e-12)
    np.testing.assert_allclose(particles["_rlnDefocusU"].astype(float), info["ctf_params"][:, 0])
    np.testing.assert_allclose(info["ctf_params"][:, 3], np.where(np.arange(40) % 2, 200.0, 300.0))
    assert info["atomic_solvent_correction"]["enabled"]
