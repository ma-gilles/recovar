"""Filament-attachment scenes (recovar.simulation.filament_attachments): placement, frame, truth tables, image path."""

import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
import pytest

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.core import cubic_interpolation
from recovar.data_io import starfile
from recovar.simulation import filament_attachments as fa
from recovar.simulation import optics_groups, relion_tomo

pytestmark = pytest.mark.unit

VOXEL = 8.0


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


def _rz(phi_deg):
    c, s = np.cos(np.deg2rad(phi_deg)), np.sin(np.deg2rad(phi_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _placed_atom_centres(model, a, phi):
    """Reference-frame centres of the model's atoms placed by T(a, phi): R_z(phi) (r0 e_x + c) + a e_z."""
    return (model.centres + np.array([model.r0, 0.0, 0.0])) @ _rz(phi).T + np.array([0.0, 0.0, a])


def _centre_of_mass(volume, voxel_size):
    """Centre of mass (A, reference frame) of a volume whose axes 0, 1, 2 are x, y, z."""
    n = volume.shape[0]
    x = (np.arange(n) - n // 2) * voxel_size
    total = volume.sum()
    return (
        np.array(
            [
                (volume.sum(axis=(1, 2)) * x).sum(),
                (volume.sum(axis=(0, 2)) * x).sum(),
                (volume.sum(axis=(0, 1)) * x).sum(),
            ]
        )
        / total
    )


def _mass_weighted_centre(model, weights, a, phi):
    masses = weights * model.sigmas**3
    return masses @ _placed_atom_centres(model, a, phi) / masses.sum()


@pytest.fixture(scope="module")
def stage0(tmp_path_factory):
    """A tiny stage-0 project: landmark only, no core, no noise, random azimuth origins, no astigmatism.

    The rendering box is the automatic one (172 px for a 48 px image at +-45 degrees), so content cut
    off by the box projects outside every image, as in the presets."""
    out = tmp_path_factory.mktemp("filament_stage0")
    config = fa.stage_config(
        "stage0",
        grid_size=48,
        max_tilt=45.0,
        tilt_step=22.5,
        n_filaments=2,
        windows_per_filament=3,
        n_tomograms=2,
        tomogram_size=(1024, 1024, 256),
        astigmatism=0.0,
        azimuth_origin="random",
        n_probe_windows=1,
        seed=4,
    )
    return out, config, fa.generate_filament_attachment_dataset(str(out), config)


# ---- (1) placement operator T(a, phi) ----


@pytest.mark.parametrize("a, phi", [(0.0, 0.0), (13.7, 37.3), (-41.25, 211.9), (100.0, -95.5)])
def test_placed_blob_centre_of_mass_off_grid(a, phi):
    """A blob placed by T(a, phi) has its centre of mass at R_z(phi) (r0 e_x + c) + a e_z within 0.01 voxel."""
    config = fa.FilamentAttachmentConfig(
        grid_size=48, voxel_size=VOXEL, r0=140.0, blobs=((6.0, -9.0, 7.0, 10.0, 1.0),), q=0
    )
    model = fa.build_attachment_model(config)
    density = np.asarray(fa.render_objects(model, 48, VOXEL, a, phi, model.mean_weights))
    expected = _placed_atom_centres(model, a, phi)[0]
    np.testing.assert_allclose(_centre_of_mass(density, VOXEL), expected, atol=0.01 * VOXEL)


def test_modes_are_orthonormal_and_shift_mode_is_a_linearised_shift():
    """U_orth is voxel-orthonormal at T(0, 0); the shift atom is d/de of the blob moved along e (finite difference)."""
    config = fa.FilamentAttachmentConfig(grid_size=64, voxel_size=VOXEL, q=2)
    model = fa.build_attachment_model(config)
    basis = np.stack(
        [np.asarray(fa.render_objects(model, 64, VOXEL, 0.0, 0.0, w)).ravel() for w in model.basis_weights], axis=1
    )
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
    # Rotation and axial shift keep it orthonormal up to grid sampling.
    moved = np.stack(
        [np.asarray(fa.render_objects(model, 64, VOXEL, 23.0, 71.0, w)).ravel() for w in model.basis_weights], axis=1
    )
    np.testing.assert_allclose(moved.T @ moved, np.eye(2), atol=1e-3)
    assert np.all(model.mode_scales > 0)

    blob, e = 1, np.array([0.0, 0.0, 1.0])
    shift_atom = int(np.nonzero(model.kinds == 1)[0][0])
    one_hot = np.zeros(len(model.kinds))
    one_hot[shift_atom] = 1.0
    analytic = np.asarray(fa.render_objects(model, 64, VOXEL, 0.0, 0.0, one_hot))
    eps = 1e-3
    shifted = []
    for sign in (1, -1):
        moved_config = fa.FilamentAttachmentConfig(
            grid_size=64,
            voxel_size=VOXEL,
            blobs=(tuple(np.r_[np.asarray(config.blobs[blob][:3]) + sign * eps * e, config.blobs[blob][3:]]),),
        )
        moved_model = fa.build_attachment_model(moved_config)
        shifted.append(np.asarray(fa.render_objects(moved_model, 64, VOXEL, 0.0, 0.0, [1.0])))
    finite_difference = (shifted[0] - shifted[1]) / (2 * eps)
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-6 * np.abs(analytic).max())


# ---- (2) frame consistency against the STAR geometry (stage 0) ----


def _particle_rows(out, name):
    flat, _ = starfile.read_star(str(out / "particles_2d.star"))
    return flat.index[flat["_rlnGroupName"] == name].to_numpy(), flat


def test_stage0_projection_centre_of_mass_follows_star_geometry(stage0):
    """Projecting a window's band attachment at the per-tilt recovar rotation puts its centre of mass at
    (A_row p)[:2] / pixel from the image centre, A_row the RELION matrix of the row's rlnAngle* in
    particles_2d.star and p the attachment's mass centre; error < 0.25 px at every tilt (plan stage 0).
    Uses the generator's spline-prefiltered cubic slicing; linear interpolation misses by 1-2.5 px here."""
    out, config, result = stage0
    model = result["model"]
    render = result["record"]["render_grid_size"]
    windows = pd.read_csv(out / "truth" / "windows.csv")
    attachments = pd.read_csv(out / "truth" / "attachments.csv").set_index("attachment_id")
    weights = model.mean_weights
    n_checked = 0
    for _, w in windows[windows["band_occupied"]].head(3).iterrows():
        a, phi = float(w["a_angst"]), float(w["phi_deg"])
        assert phi == attachments.loc[int(w["band_attachment_id"]), "phi_deg"]
        density = np.asarray(fa.render_objects(model, render, VOXEL, a, phi, weights))
        assert config.disc_type == "cubic"
        volume_ft = ftu.get_dft3(jnp.asarray(-density))  # recovar frame
        coefficients = cubic_interpolation.calculate_spline_coefficients(volume_ft)
        rows, flat = _particle_rows(out, w["rlnTomoParticleName"])
        rots = utils.R_from_relion(
            flat.loc[rows, ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float)
        )
        slices = core.slice_volume(coefficients, jnp.asarray(rots), (render, render), (render,) * 3, "cubic")
        images = -np.real(np.asarray(ftu.get_idft2(slices.reshape(-1, render, render))))
        p = _mass_weighted_centre(model, weights, a, phi)
        rr, cc = np.indices((render, render))
        for image, (_, row) in zip(images, flat.loc[rows].iterrows()):
            a_row = _relion_euler_matrix(*row[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].astype(float))
            expected = (a_row @ p)[:2] / VOXEL  # (x = column, y = row) in px from the centre
            com = np.array([(image * cc).sum(), (image * rr).sum()]) / image.sum() - render // 2
            np.testing.assert_allclose(com, expected, atol=0.25)
            n_checked += 1
    assert n_checked == 3 * 5


def _relion_model_image(model, window_links, attachments, a_row, ctf_params_row, grid, render, *, mirror=False):
    """RELION's image of a window's attachments, built independently of the simulator's projector.

    Each Gaussian atom projects to a 2D Gaussian at ``(A_row p)[:2]`` (column x, row y); the
    image is ``CTF_relion * projection`` with ``CTF_relion = -relion_tomo_ctf`` (RELION's
    ``CTF::getCTF`` is ``-sin(gamma_relion)`` with ``gamma_relion = gamma_recovar - atan(Q0 / sqrt(1 - Q0^2))``,
    i.e. minus recovar's ``evaluate_ctf``). Like the simulator it is formed on the 2x padded
    rendering box and cropped to the image box.
    """
    big = 2 * render
    x = (np.arange(big) - big // 2) * VOXEL
    xx, yy = np.meshgrid(x, x, indexing="xy")  # xx varies along columns
    image = np.zeros((big, big))
    for _, link in window_links.iterrows():
        phi = attachments.loc[int(link["attachment_id"]), "phi_deg"]
        centres = _placed_atom_centres(model, float(link["a_angst"]), -phi if mirror else phi)
        for c, s, wgt in zip(centres, model.sigmas, model.mean_weights):
            px, py = (a_row @ c)[:2]
            image += wgt * s * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * s**2))
    ctf = -np.asarray(relion_tomo.relion_tomo_ctf(jnp.asarray(ctf_params_row[None]), (big, big), VOXEL)).reshape(
        big, big
    )
    image = np.real(np.asarray(ftu.get_idft2(ftu.get_dft2(jnp.asarray(image)) * ctf)))
    lo = (big - grid) // 2
    return image[lo : lo + grid, lo : lo + grid]


def test_stage0_written_stacks_match_relions_image_model(stage0):
    """End to end: every written tilt image is an increasing affine function of RELION's model image
    (positive density, RELION CTF sign, STAR geometry); the mirrored azimuth does not fit."""
    out, config, result = stage0
    model = result["model"]
    info = result["simulation_info"]
    render = result["record"]["render_grid_size"]
    windows = pd.read_csv(out / "truth" / "windows.csv")
    links = pd.read_csv(out / "truth" / "window_attachments.csv")
    attachments = pd.read_csv(out / "truth" / "attachments.csv").set_index("attachment_id")
    particles, _ = starfile.read_star(str(out / "particles.star"))
    for p, (_, w) in enumerate(windows.iterrows()):
        assert particles["_rlnTomoParticleName"].iloc[p] == w["rlnTomoParticleName"]
        with mrcfile.open(out / particles["_rlnImageName"].iloc[p]) as mrc:
            stack = np.asarray(mrc.data, dtype=np.float64)
        rows, flat = _particle_rows(out, w["rlnTomoParticleName"])
        slices = np.array([int(name.split("@")[0]) - 1 for name in flat.loc[rows, "_rlnImageName"]])
        window_links = links[links["window_id"] == w["window_id"]]
        for row_index, slice_index in zip(rows, slices):
            row = flat.loc[row_index]
            a_row = _relion_euler_matrix(*row[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].astype(float))
            ctf_row = info["flat_rows_ctf_params"][row_index]
            image = stack[slice_index].ravel()
            corr = {}
            for mirror in (False, True):
                model_image = _relion_model_image(
                    model, window_links, attachments, a_row, ctf_row, config.grid_size, render, mirror=mirror
                ).ravel()
                corr[mirror] = np.corrcoef(image, model_image)[0, 1]
            assert corr[False] > 0.99, corr
            assert corr[True] < corr[False] - 0.05, corr


# ---- (3) truth tables ----


@pytest.mark.parametrize("eta", [0.05, 0.10, 0.30, 1.0])
def test_truth_tables_are_consistent_and_occupancy_matches_eta(eta):
    config = fa.FilamentAttachmentConfig(
        grid_size=96,
        render_grid_size=96,
        n_filaments=400,
        windows_per_filament=20,
        n_tomograms=4,
        eta=eta,
        q=2,
        seed=11,
    )
    scene = fa.build_scene(config)
    windows, attachments, links = scene["windows"], scene["attachments"], scene["window_attachments"]
    band = config.band_length
    occupied = windows["band_occupied"].mean()
    if eta == 1.0:
        assert occupied == 1.0
    else:
        assert abs(occupied - eta) < 0.03, occupied
    # Minimum spacing on each filament, so at most one attachment per band.
    for _, group in attachments.groupby("filament_id"):
        assert np.all(np.diff(np.sort(group["s_angst"].to_numpy())) >= band - 1e-9)
    # Band truth recomputed from the physical list.
    att = attachments.set_index("attachment_id")
    for _, w in windows.iterrows():
        on_filament = attachments[attachments["filament_id"] == w["filament_id"]]
        a = on_filament["s_angst"].to_numpy() - w["s_angst"]
        hits = on_filament["attachment_id"].to_numpy()[(a >= -band / 2) & (a < band / 2)]
        assert len(hits) <= 1
        if len(hits):
            j = int(hits[0])
            assert w["band_occupied"] and w["band_attachment_id"] == j
            assert np.isclose(w["a_angst"], att.loc[j, "s_angst"] - w["s_angst"])
            assert w["phi_deg"] == att.loc[j, "phi_deg"]
            np.testing.assert_array_equal(w[["z0", "z1"]].to_numpy(float), att.loc[j, ["z0", "z1"]].to_numpy(float))
        else:
            assert not w["band_occupied"] and w["band_attachment_id"] == -1
    # One physical attachment in several overlapping windows: consecutive windows of one filament, one id.
    multi = attachments[attachments["n_band_windows"] > 1]
    assert len(multi) > 0
    for _, j in multi.iterrows():
        sharing = windows[windows["band_attachment_id"] == j["attachment_id"]]
        assert set(sharing["filament_id"]) == {j["filament_id"]}
        assert np.all(np.diff(np.sort(sharing["window_index"].to_numpy())) == 1)
    expected_windows = round(band / config.segment_spacing)
    assert attachments.loc[attachments["n_band_windows"] > 0, "n_band_windows"].max() == expected_windows
    # Every link's a is s_j - s_i, and band links are exactly the windows' band truth.
    s_window = windows.set_index("window_id")["s_angst"]
    np.testing.assert_allclose(
        links["a_angst"].to_numpy(),
        att.loc[links["attachment_id"], "s_angst"].to_numpy() - s_window.loc[links["window_id"]].to_numpy(),
    )
    band_links = links[links["in_band"]].set_index("window_id")["attachment_id"]
    np.testing.assert_array_equal(
        band_links.reindex(windows["window_id"]).fillna(-1).astype(int).to_numpy(),
        windows["band_attachment_id"].to_numpy(),
    )
    # Halves per filament.
    assert windows.groupby("filament_id")["random_subset"].nunique().max() == 1


def test_project_rows_match_truth_windows(stage0):
    """particles.star rows carry the truth table's names, coordinates, poses, filament ids and halves."""
    out, config, result = stage0
    windows = pd.read_csv(out / "truth" / "windows.csv")
    particles, _ = starfile.read_star(str(out / "particles.star"))
    assert particles["_rlnTomoParticleName"].tolist() == windows["rlnTomoParticleName"].tolist()
    assert particles["_rlnTomoName"].tolist() == windows["tomogram"].tolist()
    coords = particles[["_rlnCenteredCoordinateXAngst", "_rlnCenteredCoordinateYAngst", "_rlnCenteredCoordinateZAngst"]]
    np.testing.assert_allclose(
        coords.values.astype(float), windows[["centre_x_angst", "centre_y_angst", "centre_z_angst"]].values, atol=1e-6
    )
    np.testing.assert_allclose(
        particles[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(float),
        windows[["rot", "tilt", "psi"]].values,
        atol=1e-6,
    )
    np.testing.assert_array_equal(particles["_rlnHelicalTubeID"].astype(int), windows["filament_id"] + 1)
    np.testing.assert_array_equal(particles["_rlnRandomSubset"].astype(int), windows["random_subset"])
    # Window centres lie on the filament line, d = A e_z.
    filaments = pd.read_csv(out / "truth" / "filaments.csv").set_index("filament_id")
    for _, w in windows.iterrows():
        f = filaments.loc[w["filament_id"]]
        d = _relion_euler_matrix(f["rot"], f["tilt"], f["psi"])[:, 2]
        anchor = f[["anchor_x_angst", "anchor_y_angst", "anchor_z_angst"]].to_numpy(float)
        np.testing.assert_allclose(
            w[["centre_x_angst", "centre_y_angst", "centre_z_angst"]].to_numpy(float), anchor + w["s_angst"] * d
        )
    # Truth volumes are RELION MRCs: loading into recovar's frame flips the sign.
    mu = utils.load_relion_volume(str(out / "truth" / "volumes" / "mu.mrc"))
    assert mu.shape == (config.grid_size,) * 3 and mu.min() < 0 and abs(mu.max()) < 1e-6
    assert result["record"]["frame_convention"] == fa.FRAME_CONVENTION


# ---- (4) image path ----


def _ctf_params(n, rng):
    params = np.zeros((n, int(core.CTFParamIndex.TILT_ANGLE) + 1))
    params[:, core.CTFParamIndex.DFU] = rng.uniform(15000, 25000, n)
    params[:, core.CTFParamIndex.DFV] = params[:, core.CTFParamIndex.DFU] - 200
    params[:, core.CTFParamIndex.DFANG] = 30.0
    params[:, core.CTFParamIndex.VOLT] = 300.0
    params[:, core.CTFParamIndex.CS] = 2.7
    params[:, core.CTFParamIndex.W] = 0.1
    params[:, core.CTFParamIndex.CONTRAST] = 1.0
    params[:, core.CTFParamIndex.DOSE] = rng.uniform(0, 60, n)
    return params


def test_per_particle_path_reproduces_shared_volume_images():
    """With every particle showing one of two shared volumes, the per-particle path gives the shared path's
    noise-free images within float32 machine epsilon of the image scale (the images are float32,
    simulate_data's dtype; only the batching differs, which moves float32 intermediates by an ulp)."""
    n = 32
    rng = np.random.default_rng(0)
    x = (np.arange(n) - n // 2) * 4.0
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    reals = [np.exp(-((xx - 12) ** 2 + yy**2 + zz**2) / 50), np.exp(-(xx**2 + (yy + 8) ** 2 + (zz - 4) ** 2) / 30)]
    volumes = np.stack([np.asarray(ftu.get_dft3(jnp.asarray(v))).ravel() for v in reals])
    n_particles, n_tilts = 6, 4
    row_particle = np.repeat(np.arange(n_particles), n_tilts)
    particle_volume = np.array([0, 1, 1, 0, 1, 0])
    rots = np.asarray(utils.R_from_relion(rng.uniform(-180, 180, (row_particle.size, 3))))
    ctf_params = _ctf_params(row_particle.size, rng)
    contrast = 1 + 0.1 * rng.standard_normal(row_particle.size)
    group = [{"pixel_size": 4.0, "box_size": n, "noise_scale": 1.0}]
    common = dict(ctf_evaluator=relion_tomo.relion_tomo_ctf, seed=0, disc_type="linear_interp", premultiplied_ctf=False)
    shared, _ = optics_groups.simulate_optics_groups(
        volumes,
        group,
        np.zeros(row_particle.size, int),
        rots,
        ctf_params,
        particle_volume[row_particle],
        contrast,
        volumes_path_root=None,
        trailing_zero_format_in_vol_name=True,
        voxel_size=4.0,
        grid_size=n,
        scale_vol=1.0,
        solvent_record={"enabled": False},
        snr=np.inf,
        noise_model="white",
        n_probe=8,
        **common,
    )
    per_particle, _, calibration = optics_groups.simulate_per_particle_volumes(
        lambda p: volumes[particle_volume[np.asarray(p)]],
        n,
        group,
        np.zeros(row_particle.size, int),
        row_particle,
        rots,
        ctf_params,
        contrast,
        voxel_size=4.0,
        grid_size=n,
        snr=np.inf,
        noise_model="white",
        n_probe_particles=8,
        max_volume_bytes=2 * 16 * n**3,
        **common,
    )
    assert calibration["noiseless"]
    shared = np.stack(shared)
    assert shared.dtype == per_particle.dtype == np.float32
    np.testing.assert_allclose(per_particle, shared, rtol=0, atol=np.finfo(np.float32).eps * np.abs(shared).max())
    assert np.mean(per_particle == shared) > 0.99


def test_per_particle_path_rejects_wrong_volume_shapes_and_solvent_preset(tmp_path):
    group = [{"pixel_size": 4.0, "box_size": 16, "noise_scale": 1.0}]
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="returned shape"):
        optics_groups.simulate_per_particle_volumes(
            lambda p: np.zeros((len(p), 8**3), complex),
            16,
            group,
            np.zeros(2, int),
            np.zeros(2, int),
            np.asarray(utils.R_from_relion(np.zeros((2, 3)))),
            _ctf_params(2, rng),
            np.ones(2),
            ctf_evaluator=relion_tomo.relion_tomo_ctf,
            voxel_size=4.0,
            grid_size=16,
            snr=None,
            noise_model="white",
            n_probe_particles=8,
            seed=0,
            disc_type="linear_interp",
            premultiplied_ctf=False,
        )
    with pytest.raises(ValueError, match="atomic_solvent_correction"):
        relion_tomo.generate_relion5_tomo_dataset(
            str(tmp_path / "p"), None, 4.0, 2, grid_size=16, n_tomograms=1, particle_volume_fn=lambda p: None
        )
