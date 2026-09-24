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


def _write_volume(root):
    x = (np.arange(GRID) - GRID / 2) * VOXEL
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-((xx - 12) ** 2 + yy**2 + zz**2) / 200) + 0.5 * np.exp(-(xx**2 + (yy + 16) ** 2 + zz**2) / 100)
    with mrcfile.new(root / "vol0000.mrc") as mrc:
        mrc.set_data(vol.astype(np.float32))
        mrc.voxel_size = VOXEL
    return vol


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("relion_tomo")
    _write_volume(root)
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
    """Groups share one per-pixel noise variance times noise_scale (1.5 for group 2): about 2.25x the power."""
    out, result = dataset
    particles, _ = starfile.read_star(result["particles"])
    power = {1: [], 2: []}
    for _, p in particles.iterrows():
        with mrcfile.open(out / p["_rlnImageName"]) as mrc:
            power[int(p["_rlnOpticsGroup"])].append(np.mean(mrc.data.astype(np.float64) ** 2))
    ratio = np.mean(power[2]) / np.mean(power[1])
    assert 1.7 < ratio < 2.8, ratio
    assert os.path.isfile(out / "simulation_info.pkl")


def test_relion_dose_weight_matches_ctf_h():
    """exp(-0.5 dose / (0.245 u2^-0.8325 + 2.81)) with weight 1 at u2 = 0, as in RELION's CTF::getCTF."""
    u2 = np.array([0.0, 1e-4, 0.01, 0.0625])
    dose = np.array([0.0, 3.0, 60.0])
    got = np.asarray(relion_tomo.relion_dose_weight(u2, dose))
    with np.errstate(divide="ignore"):
        expected = np.exp(-0.5 * dose[:, None] / (0.245 * u2[None, :] ** -0.8325 + 2.81))
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert np.all(got[:, 0] == 1.0) and np.all(got[0] == 1.0)


def test_relion_tomo_ctf_is_spa_ctf_times_dose_weight():
    from recovar import core
    from recovar.core import fourier_transform_utils as ftu

    params = np.zeros((2, 11))
    params[:, :6] = [[15000, 14500, 30, 300, 2.7, 0.1], [22000, 22000, 0, 200, 1.4, 0.07]]
    params[:, core.CTFParamIndex.CONTRAST] = [1.0, 0.5]
    params[:, core.CTFParamIndex.DOSE] = [0.0, 45.0]
    ctf = np.asarray(relion_tomo.relion_tomo_ctf(params, (16, 16), 3.0))
    freqs = np.asarray(ftu.get_k_coordinate_of_each_pixel((16, 16), 3.0, scaled=True))
    spa = np.asarray(core.evaluate_ctf(freqs, params[:, :9]))
    np.testing.assert_allclose(ctf[0], spa[0], rtol=1e-4, atol=1e-5)  # float32 grid vs float64 reference
    weight = np.asarray(relion_tomo.relion_dose_weight((freqs**2).sum(-1), params[1:, 9]))[0]
    np.testing.assert_allclose(ctf[1], spa[1] * weight, rtol=1e-4, atol=1e-5)  # float32 grid vs float64 reference
    half = np.asarray(relion_tomo.relion_tomo_ctf(params, (16, 16), 3.0, half_image=True))
    np.testing.assert_allclose(
        half, np.asarray(ftu.full_image_to_half_image(ctf, (16, 16))), rtol=1e-4, atol=1e-5
    )  # float32 grid vs float64 reference


def test_group_volumes_resample_and_pad(tmp_path):
    from recovar.core import fourier_transform_utils as ftu

    _write_volume(tmp_path)
    loaded = relion_tomo._group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, VOXEL, GRID)[0]
    vol = np.real(np.asarray(ftu.get_idft3(loaded.reshape((GRID,) * 3))))
    padded = relion_tomo._group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, VOXEL, GRID + 8)[0]
    real = np.real(np.asarray(ftu.get_idft3(padded.reshape((GRID + 8,) * 3))))
    np.testing.assert_allclose(real[4:-4, 4:-4, 4:-4], vol, atol=1e-4)
    assert np.abs(real[:4]).max() < 1e-10
    coarse = relion_tomo._group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, 2 * VOXEL, GRID // 2)[0]
    assert coarse.shape == ((GRID // 2) ** 3,)
    with pytest.raises(ValueError, match="even integer"):
        relion_tomo._group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, 3 * VOXEL, GRID)


def test_optics_groups_with_different_pixel_and_box_sizes(tmp_path):
    _write_volume(tmp_path)
    groups = (
        {"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0},
        {"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0, "pixel_size": 2 * VOXEL, "box_size": 24},
    )
    result = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "project"),
        str(tmp_path / "vol"),
        VOXEL,
        n_particles=4,
        grid_size=GRID,
        n_tomograms=2,
        optics_groups=groups,
        max_tilt=20.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        seed=1,
        atomic_solvent_correction=False,
    )
    assert result["simulation_info"]["atomic_solvent_correction"] == {"enabled": False}
    particles, optics = starfile.read_star(result["particles"])
    assert optics["_rlnImagePixelSize"].astype(float).tolist() == [VOXEL, 2 * VOXEL]
    assert optics["_rlnImageSize"].astype(int).tolist() == [GRID, 24]
    tomograms, _ = starfile.read_star(result["tomograms"])
    assert tomograms["_rlnTomoTiltSeriesPixelSize"].astype(float).tolist() == [VOXEL, 2 * VOXEL]
    for _, p in particles.iterrows():
        box, pixel = (GRID, VOXEL) if p["_rlnOpticsGroup"] == "1" else (24, 2 * VOXEL)
        with mrcfile.open(tmp_path / "project" / p["_rlnImageName"]) as mrc:
            assert mrc.data.shape == (5, box, box)
            assert np.isclose(float(mrc.voxel_size.x), pixel)
            assert np.all(np.isfinite(mrc.data)) and mrc.data.std() > 0


def test_em_development_preset_is_default_and_recorded_for_ground_truth(dataset):
    from recovar.simulation import synthetic_dataset

    out, result = dataset
    info = utils.pickle_load(str(out / "simulation_info.pkl"))
    record = info["atomic_solvent_correction"]
    assert record["enabled"] and (record["a"], record["B"], record["B_atomic"]) == (0.8, 2000.0, 100.0)
    flat, _ = starfile.read_star(result["particles_2d"])
    assert info["image_assignment"].shape == info["per_image_contrast"].shape == (len(flat),)
    gt = synthetic_dataset.load_heterogeneous_reconstruction(info)
    assert gt.volumes.shape == (1, GRID**3)


def test_cli_preset_on_by_default_with_opt_out():
    import argparse

    from recovar.commands import make_relion_tomo_dataset
    from recovar.simulation import solvent_contrast

    parser = argparse.ArgumentParser()
    make_relion_tomo_dataset.add_args(parser)
    base = ["vol", "4.25", "10", "-o", "out"]
    assert solvent_contrast.kwargs_from_cli_args(parser.parse_args(base))["atomic_solvent_correction"] is True
    off = parser.parse_args(base + ["--no-atomic-solvent-correction"])
    assert solvent_contrast.kwargs_from_cli_args(off)["atomic_solvent_correction"] is False
