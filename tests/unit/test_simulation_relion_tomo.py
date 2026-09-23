"""Simulated RELION 5 subtomogram (2D-stack) datasets: files, optics groups and geometry."""

import os

import mrcfile
import numpy as np
import pytest

from recovar import utils
from recovar.data_io import starfile
from recovar.simulation import relion_tomo

GRID = 32
VOXEL = 4.0


def _relion_euler_matrix(rot, tilt, psi):
    """RELION's Euler_angles2matrix (src/euler.cpp), written out independently."""
    a, b, g = np.deg2rad([rot, tilt, psi])
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.array(
        [
            [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb],
        ]
    )


def _axis_rotation(axis, deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m = np.eye(3)
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("relion_tomo")
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-((xx - 12) ** 2 + yy**2 + zz**2) / 200) + 0.5 * np.exp(-(xx**2 + (yy + 16) ** 2 + zz**2) / 100)
    with mrcfile.new(root / "vol0000.mrc") as mrc:
        mrc.set_data(vol.astype(np.float32))
        mrc.voxel_size = VOXEL
    out = root / "project"
    result = relion_tomo.generate_relion5_tomo_dataset(
        str(out),
        str(root / "vol"),
        VOXEL,
        n_particles=6,
        grid_size=GRID,
        n_tomograms=2,
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        hidden_tilt_fraction=0.3,
        snr=0.05,
        seed=3,
    )
    return out, result


def test_dose_symmetric_tilt_scheme():
    angles, order = relion_tomo.dose_symmetric_tilt_scheme(max_tilt=12.0, tilt_step=3.0)
    np.testing.assert_array_equal(angles, np.arange(-12.0, 13.0, 3.0))
    acquisition = angles[np.argsort(order)]
    np.testing.assert_array_equal(acquisition, [0, 3, 6, -3, -6, 9, 12, -9, -12])


def test_relion_files_and_optics_groups(dataset):
    out, result = dataset
    text = open(out / "tomograms.star").read()
    assert "data_global" in text
    assert "data_TS_01" in open(out / "tilt_series" / "TS_01.star").read()
    assert "rlnTomoParticlesFile" in open(result["optimisation_set"]).read()

    particles_text = open(result["particles"]).read()
    assert "data_general" in particles_text and "_rlnTomoSubTomosAre2DStacks" in particles_text
    particles, optics = starfile.read_star(result["particles"])
    assert optics["_rlnVoltage"].astype(float).tolist() == [300.0, 200.0]
    assert set(particles["_rlnOpticsGroup"].astype(int)) == {1, 2}

    tomograms, _ = starfile.read_star(str(out / "tomograms.star"))
    tomo_optics = dict(zip(tomograms["_rlnTomoName"], tomograms["_rlnOpticsGroupName"]))
    for name, og in zip(particles["_rlnTomoName"], particles["_rlnOpticsGroup"]):
        assert tomo_optics[name] == f"opticsGroup{og}"

    flat, flat_optics = starfile.read_star(result["particles_2d"])
    assert len(flat_optics) == 2
    n_visible = 0
    for _, p in particles.iterrows():
        visible = sum(parse == "1" for parse in p["_rlnTomoVisibleFrames"].strip("[]").split(","))
        n_visible += visible
        with mrcfile.open(out / p["_rlnImageName"]) as mrc:
            assert mrc.data.shape == (visible, GRID, GRID)
        rows = flat[flat["_rlnGroupName"] == p["_rlnTomoParticleName"]]
        assert len(rows) == visible
        assert set(rows["_rlnOpticsGroup"]) == {p["_rlnOpticsGroup"]}
    assert len(flat) == n_visible


def test_per_tilt_pose_and_defocus_follow_relion(dataset):
    """Flattened pose = Aproj A_euler (exp_model.cpp) and depth defocus (tomogram.cpp getCtf)."""
    out, result = dataset
    particles, _ = starfile.read_star(result["particles"])
    flat, _ = starfile.read_star(result["particles_2d"])
    info = result["simulation_info"]
    hand = -1
    for _, p in particles.iterrows():
        tilts, _ = starfile.read_star(str(out / "tilt_series" / f"{p['_rlnTomoName']}.star"))
        visible = np.nonzero(np.array(p["_rlnTomoVisibleFrames"].strip("[]").split(",")) == "1")[0]
        a_euler = _relion_euler_matrix(*p[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].astype(float))
        pos = p[["_rlnCenteredCoordinateXAngst", "_rlnCenteredCoordinateYAngst", "_rlnCenteredCoordinateZAngst"]]
        pos = pos.values.astype(float)
        rows = flat[flat["_rlnGroupName"] == p["_rlnTomoParticleName"]]
        for idx, row in rows.iterrows():
            tilt = tilts.iloc[visible[int(row["_rlnImageName"].split("@")[0]) - 1]]
            r_zyx = (
                _axis_rotation(2, float(tilt["_rlnTomoZRot"]))
                @ _axis_rotation(1, float(tilt["_rlnTomoYTilt"]))
                @ _axis_rotation(0, float(tilt["_rlnTomoXTilt"]))
            )
            a_row = _relion_euler_matrix(*row[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].astype(float))
            np.testing.assert_allclose(a_row, r_zyx @ a_euler, atol=1e-5)
            dz = hand * (r_zyx @ pos)[2]
            np.testing.assert_allclose(float(row["_rlnDefocusU"]), float(tilt["_rlnDefocusU"]) + dz, atol=1e-3)
            np.testing.assert_allclose(
                info["flat_rows_ctf_params"][idx, 9], float(tilt["_rlnMicrographPreExposure"]), atol=1e-6
            )
            np.testing.assert_allclose(
                info["flat_rows_rots"][idx],
                utils.R_from_relion(row[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float))[0],
                atol=1e-6,
            )


def test_optics_group_noise_scale(dataset):
    """Group 2 has noise_scale 1.5, so its low-SNR images carry about 2.25x the power."""
    out, result = dataset
    particles, _ = starfile.read_star(result["particles"])
    power = {1: [], 2: []}
    for _, p in particles.iterrows():
        with mrcfile.open(out / p["_rlnImageName"]) as mrc:
            power[int(p["_rlnOpticsGroup"])].append(np.mean(mrc.data.astype(np.float64) ** 2))
    ratio = np.mean(power[2]) / np.mean(power[1])
    assert 1.7 < ratio < 2.8, ratio
    assert os.path.isfile(out / "simulation_info.pkl")
