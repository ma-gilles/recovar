"""Simulated RELION 5 subtomogram (2D-stack) datasets: files, optics groups and geometry."""

import os

import mrcfile
import numpy as np
import pytest

from recovar import utils
from recovar.data_io import starfile
from recovar.simulation import optics_groups, relion_tomo

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
        optics_groups=optics_groups.DEFAULT_OPTICS_GROUPS,
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
    with mrcfile.open(out / "tilt_series" / "TS_01_001.mrc") as mrc:  # header read by relion_refine
        assert mrc.data.shape == (512, 512)
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


def test_optics_group_noise_scale_and_normalised_stacks(dataset):
    """Groups share one per-pixel noise variance times noise_scale**2 (1.5 for group 2).

    The stacks are then normalised as relion_preprocess --norm does: background mean 0
    and standard deviation 1 per tilt image, so the group difference is one of SNR.
    """
    out, result = dataset
    info = result["simulation_info"]
    nv = info["noise_variance_per_optics_group"]
    np.testing.assert_allclose(np.mean(nv[1]) / np.mean(nv[0]), 1.0, rtol=0.15)  # one shared level; noise_scale on top
    assert info["relion_normalize"] is True
    particles, _ = starfile.read_star(result["particles"])
    c = np.arange(GRID) - (GRID / 2 - 0.5)
    background = np.hypot(*np.meshgrid(c, c, indexing="ij")) > round(0.375 * GRID)
    for _, p in particles.iterrows():
        with mrcfile.open(out / p["_rlnImageName"]) as mrc:
            stack = np.asarray(mrc.data, dtype=np.float64)
        np.testing.assert_allclose(stack[:, background].mean(axis=1), 0.0, atol=1e-4)
        np.testing.assert_allclose(stack[:, background].std(axis=1), 1.0, rtol=1e-3)
    assert os.path.isfile(out / "simulation_info.pkl")


@pytest.mark.parametrize("per_tomogram", [True, False])
def test_one_optics_group_per_tomogram_by_default(tmp_path, per_tomogram):
    """RELION 5's tomogram import writes one optics group per tomogram: the default.

    Two settings over four tomograms (tomogram t uses setting t % 2): per tomogram gives four groups
    with their settings' values and group t + 1 for tomogram t's particles; shared gives one group per
    setting. Images and noise are simulated per setting either way, so groups with one setting share
    its noise variance.
    """
    _write_volume(tmp_path)
    result = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "project"),
        str(tmp_path / "vol"),
        VOXEL,
        n_particles=8,
        grid_size=GRID,
        n_tomograms=4,
        optics_groups=optics_groups.DEFAULT_OPTICS_GROUPS,
        optics_group_per_tomogram=per_tomogram,
        max_tilt=20.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        seed=2,
    )
    particles, optics = starfile.read_star(result["particles"])
    tomograms, _ = starfile.read_star(result["tomograms"])
    tomo_index = {name: t for t, name in enumerate(tomograms["_rlnTomoName"])}
    groups = 4 if per_tomogram else 2
    assert optics["_rlnOpticsGroup"].astype(int).tolist() == list(range(1, groups + 1))
    assert optics["_rlnVoltage"].astype(float).tolist() == [300.0, 200.0] * (groups // 2)
    for name, group in zip(particles["_rlnTomoName"], particles["_rlnOpticsGroup"].astype(int)):
        t = tomo_index[name]
        assert group == (t + 1 if per_tomogram else t % 2 + 1)
        assert tomograms["_rlnOpticsGroupName"][t] == f"opticsGroup{group}"
    noise = result["simulation_info"]["noise_variance_per_optics_group"]
    assert len(noise) == groups
    if per_tomogram:
        for g in (2, 3):  # the same setting as group g - 2: one noise variance
            np.testing.assert_allclose(noise[g], noise[g - 2], rtol=1e-12)


def test_default_is_one_shared_optics_setting(tmp_path):
    """Without optics settings every tomogram gets RELION's first default optics, in its own group."""
    _write_volume(tmp_path)
    result = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "project"),
        str(tmp_path / "vol"),
        VOXEL,
        n_particles=3,
        grid_size=GRID,
        n_tomograms=3,
        max_tilt=10.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        seed=1,
    )
    _, optics = starfile.read_star(result["particles"])
    assert optics["_rlnOpticsGroupName"].tolist() == ["opticsGroup1", "opticsGroup2", "opticsGroup3"]
    for column, value in (("_rlnVoltage", 300.0), ("_rlnSphericalAberration", 2.7), ("_rlnAmplitudeContrast", 0.1)):
        assert optics[column].astype(float).tolist() == [value] * 3


def test_ground_truth_offsets_project_into_every_tilt(tmp_path):
    """rlnOrigin{X,Y,Z}Angst are written and each tilt is shifted by Aproj_i[:2] o / pixel."""
    _write_volume(tmp_path)
    result = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "project"),
        str(tmp_path / "vol"),
        VOXEL,
        n_particles=4,
        grid_size=GRID,
        n_tomograms=2,
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        origin_std_angstrom=3.0,
        seed=5,
    )
    info = result["simulation_info"]
    particles, _ = starfile.read_star(result["particles"])
    origins = particles[["_rlnOriginXAngst", "_rlnOriginYAngst", "_rlnOriginZAngst"]].values.astype(float)
    np.testing.assert_allclose(origins, info["particle_origins_angstrom"])
    assert np.all(np.abs(origins) > 0)
    flat, _ = starfile.read_star(str(tmp_path / "project" / "particles_2d.star"))
    names = list(particles["_rlnTomoParticleName"].values)
    pose = {n: _relion_euler_matrix(*r) for n, r in zip(names, particles[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float))}
    origin = dict(zip(names, origins))
    for k, (_, row) in enumerate(flat.iterrows()):
        a_i = _relion_euler_matrix(*row[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float))
        a_proj = a_i @ pose[row["_rlnGroupName"]].T
        np.testing.assert_allclose(
            info["flat_rows_translations_px"][k], a_proj[:2] @ origin[row["_rlnGroupName"]] / VOXEL, atol=1e-9
        )


def test_given_particle_eulers_are_written_and_simulated(tmp_path):
    """particle_eulers replaces the uniform draw; STAR poses and per-tilt rotations follow it, the rest is unchanged."""
    _write_volume(tmp_path)
    eulers = np.array([[10.0, 5.0, -30.0], [-120.0, 12.0, 45.0], [170.0, 2.0, 100.0], [60.0, 20.0, -170.0]])
    kwargs = dict(
        n_particles=4,
        grid_size=GRID,
        n_tomograms=2,
        max_tilt=30.0,
        tilt_step=10.0,
        tomogram_size=(512, 512, 128),
        origin_std_angstrom=2.0,
        seed=7,
    )
    given = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "given"), str(tmp_path / "vol"), VOXEL, particle_eulers=eulers, **kwargs
    )
    uniform = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "uniform"), str(tmp_path / "vol"), VOXEL, **kwargs
    )
    particles, _ = starfile.read_star(given["particles"])
    written = particles[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float)
    np.testing.assert_allclose(written, eulers, atol=1e-6)
    # Per-tilt rotation Aproj_i = A_row A_pose^T depends only on the tilt, so it matches the uniform-pose dataset row by row.
    uniform_particles, _ = starfile.read_star(uniform["particles"])
    a_proj = {}
    for label, ps in (("given", particles), ("uniform", uniform_particles)):
        flat, _ = starfile.read_star(str(tmp_path / label / "particles_2d.star"))
        pose = dict(
            zip(
                ps["_rlnTomoParticleName"].values,
                ps[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float),
            )
        )
        a_proj[label] = np.stack(
            [
                _relion_euler_matrix(*row[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float))
                @ _relion_euler_matrix(*pose[row["_rlnGroupName"]]).T
                for _, row in flat.iterrows()
            ]
        )
    np.testing.assert_allclose(a_proj["given"], a_proj["uniform"], atol=1e-5)
    np.testing.assert_allclose(
        given["simulation_info"]["particle_origins_angstrom"], uniform["simulation_info"]["particle_origins_angstrom"]
    )
    with pytest.raises(ValueError, match="particle_eulers"):
        relion_tomo.generate_relion5_tomo_dataset(
            str(tmp_path / "bad"), str(tmp_path / "vol"), VOXEL, particle_eulers=eulers[:3], **kwargs
        )


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
    loaded = optics_groups.group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, VOXEL, GRID)[0]
    vol = np.real(np.asarray(ftu.get_idft3(loaded.reshape((GRID,) * 3))))
    padded = optics_groups.group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, VOXEL, GRID + 8)[0]
    real = np.real(np.asarray(ftu.get_idft3(padded.reshape((GRID + 8,) * 3))))
    np.testing.assert_allclose(real[4:-4, 4:-4, 4:-4], vol, atol=1e-4)
    assert np.abs(real[:4]).max() < 1e-10
    coarse = optics_groups.group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, 2 * VOXEL, GRID // 2)[0]
    assert coarse.shape == ((GRID // 2) ** 3,)
    with pytest.raises(ValueError, match="even integer"):
        optics_groups.group_volumes(str(tmp_path / "vol"), True, VOXEL, GRID, 3 * VOXEL, GRID)


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


def test_zernike_indices_and_beam_tilt_follow_relion():
    """Zernike::oddIndexToMN / evenIndexToMN and TiltHelper::insertTilt, written out from RELION's code."""
    assert [relion_tomo.odd_index_to_mn(i) for i in range(6)] == [(-1, 1), (1, 1), (-3, 3), (-1, 3), (1, 3), (3, 3)]
    assert [relion_tomo.even_index_to_mn(i) for i in range(9)] == [
        (0, 0), (-2, 2), (0, 2), (2, 2), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)
    ]
    lam = 12.2643247 / np.sqrt(300e3 * (1.0 + 300e3 * 0.978466e-6))
    scale = 2.7 * 20000 * lam * lam * 3.141592654
    got = relion_tomo.odd_coefficients_with_beam_tilt([0, 0, 40, 0, 0, -30], (1.6, -1.2), 2.7, 300)
    z3x, z3y = -scale * 1.6 / 3.0, -scale * -1.2 / 3.0
    np.testing.assert_allclose(got, [2 * z3y, 2 * z3x, 40, z3y, z3x, -30], rtol=1e-12)
    # Z_3^-3 = rho^3 sin(3 phi), Z_4^0 = 6 rho^4 - 6 rho^2 + 1 at (x, y) in 1/A.
    freqs = np.array([[0.1, 0.05], [-0.02, 0.2], [0.0, 0.0]])
    rho, phi = np.hypot(freqs[:, 0], freqs[:, 1]), np.arctan2(freqs[:, 1], freqs[:, 0])
    np.testing.assert_allclose(
        relion_tomo.zernike_phase([0, 0, 2.0], relion_tomo.odd_index_to_mn, freqs), 2.0 * rho**3 * np.sin(3 * phi)
    )
    np.testing.assert_allclose(
        relion_tomo.zernike_phase([0, 0, 0, 0, 0, 0, 1.5], relion_tomo.even_index_to_mn, freqs),
        1.5 * (6 * rho**4 - 6 * rho**2 + 1),
    )


def test_relion_tomo_ctf_adds_even_phase_at_the_magnified_frequency():
    """CTF::getCTF: every term at M k, with the even Zernike phase added to gamma."""
    from recovar import core
    from recovar.core import fourier_transform_utils as ftu

    params = np.zeros((2, 11))
    params[:, :6] = [[15000, 14500, 30, 300, 2.7, 0.1], [22000, 22000, 0, 300, 2.7, 0.1]]
    params[:, core.CTFParamIndex.CONTRAST] = 1.0
    params[:, core.CTFParamIndex.DOSE] = [0.0, 45.0]
    mag = np.array([[1.015, 0.004], [0.004, 0.99]])
    even = [0, 0, 0, 0, 400, 0, 0, 0, -300]
    plain = np.asarray(relion_tomo.relion_tomo_ctf(params, (16, 16), 3.0))
    np.testing.assert_array_equal(
        np.asarray(relion_tomo.relion_tomo_ctf(params, (16, 16), 3.0, even_zernike=[0.0] * 9)), plain
    )
    got = np.asarray(relion_tomo.relion_tomo_ctf(params, (16, 16), 3.0, mag_matrix=mag, even_zernike=even))
    freqs = np.asarray(ftu.get_k_coordinate_of_each_pixel((16, 16), 3.0, scaled=True), dtype=np.float64) @ mag.T
    gamma = relion_tomo.zernike_phase(even, relion_tomo.even_index_to_mn, freqs)
    expected = np.asarray(core.evaluate_ctf(freqs, params[:, :9], gamma)) * np.asarray(
        relion_tomo.relion_dose_weight((freqs**2).sum(-1), params[:, 9])
    )
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)  # float32 grid vs float64 reference
    assert np.max(np.abs(got - plain)) > 0.1


def test_odd_modulation_is_a_real_unit_phase_that_demodulation_undoes():
    """Images carry exp(i phase); multiplying their transform by exp(-i phase) gives the input back."""
    from recovar.core import fourier_transform_utils as ftu

    rng = np.random.default_rng(0)
    images = rng.normal(size=(3, 16, 16)).astype(np.float64)
    freqs = np.asarray(ftu.get_k_coordinate_of_each_pixel((16, 16), 3.0, scaled=True), dtype=np.float64)
    phase = relion_tomo.zernike_phase([0.0, 0.0, 40.0, 0.0, 0.0, -30.0], relion_tomo.odd_index_to_mn, freqs)
    out = relion_tomo.modulate_odd_aberrations(images, freqs, phase)
    assert out.dtype == images.dtype
    ft = np.asarray(ftu.get_dft2(out)) * np.exp(-1j * phase).reshape(16, 16)[None]
    back = np.asarray(ftu.get_idft2(ft)).real
    # Only the unpaired Nyquist row and column (index 0 of the centred grid) lose their imaginary part.
    mask = np.ones((16, 16), bool)
    mask[0, :] = mask[:, 0] = False
    np.testing.assert_allclose(
        np.asarray(ftu.get_dft2(back))[:, mask], np.asarray(ftu.get_dft2(images))[:, mask], atol=1e-9
    )
    assert np.max(np.abs(out - images)) > 0.1 * np.std(images)


def test_optics_aberrations_are_written_and_applied(tmp_path):
    """Optics settings with aberrations write RELION's columns and change only their own images."""
    _write_volume(tmp_path)
    base = dict(voltage=300.0, cs=2.7, amp_contrast=0.1, noise_scale=1.0)
    aberrated = dict(
        base,
        beam_tilt=(1.6, -1.2),
        odd_zernike=[0, 0, 40, 0, 0, -30],
        even_zernike=[0, 0, 0, 0, 400, 0, 0, 0, -300],
        mag_matrix=[[1.015, 0.004], [0.004, 0.99]],
    )
    kwargs = dict(
        n_particles=4, grid_size=GRID, n_tomograms=2, max_tilt=30.0, tilt_step=10.0, tomogram_size=(512, 512, 128),
        snr=1e6, seed=5,
    )
    plain = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "plain"), str(tmp_path / "vol"), VOXEL, optics_groups=[base, base], **kwargs
    )
    mixed = relion_tomo.generate_relion5_tomo_dataset(
        str(tmp_path / "mixed"), str(tmp_path / "vol"), VOXEL, optics_groups=[base, aberrated], **kwargs
    )
    _, plain_optics = starfile.read_star(plain["particles"])
    assert not any(c.startswith(("_rlnOddZernike", "_rlnEvenZernike", "_rlnBeamTilt", "_rlnMagMat")) for c in plain_optics)
    particles, optics = starfile.read_star(mixed["particles"])
    assert list(optics["_rlnOddZernike"]) == ["[0,0,0,0,0,0]", "[0,0,40,0,0,-30]"]
    assert list(optics["_rlnEvenZernike"]) == ["[0,0,0,0,0,0,0,0,0]", "[0,0,0,0,400,0,0,0,-300]"]
    np.testing.assert_allclose(optics["_rlnBeamTiltX"].astype(float), [0.0, 1.6])
    np.testing.assert_allclose(optics["_rlnMagMat00"].astype(float), [1.0, 1.015])
    np.testing.assert_allclose(optics["_rlnMagMat01"].astype(float), [0.0, 0.004])
    for name, group in zip(particles["_rlnImageName"], particles["_rlnOpticsGroup"].astype(int)):
        with mrcfile.open(tmp_path / "plain" / name) as a, mrcfile.open(tmp_path / "mixed" / name) as b:
            difference = np.max(np.abs(a.data - b.data))
        assert (difference > 1e-2) == (group == 2), (name, group, difference)
