"""Simulate cryo-ET subtomogram datasets in RELION 5's 2D-stack format.

The output directory is a RELION project that ``relion_refine --ios
optimisation_set.star`` reads directly:

- ``tomograms.star`` (``data_global``) and ``tilt_series/<tomo>.star``
  (``data_<tomo>``): per-tilt geometry (``rlnTomoYTilt``, ``rlnTomoZRot``),
  defocus, cumulative dose and CTF scale;
- ``particles.star``: ``data_general`` with ``rlnTomoSubTomosAre2DStacks 1``,
  one optics row per optics group, and one row per particle with centred 3D
  coordinates, Euler angles and ``rlnTomoVisibleFrames``;
- ``Subtomograms/<tomo>/<n>_stack2d.mrcs``: one slice per visible tilt, in
  tilt-table order (what ``relion_tomo_subtomo --stack2d`` writes);
- ``particles_2d.star``: the same data flattened to one row per particle-tilt
  (``recovar pipeline --tilt-series`` input).

Each tomogram belongs to one optics group, and groups may differ in voltage,
Cs, amplitude contrast and noise level. Per-tilt poses and depth-corrected
defocus are not computed here: the RELION STAR files are written first and
then read back with :func:`recovar.commands.parse_relion5_tomo.convert`, so the
simulator and recovar's RELION 5 reader share one geometry implementation.
Images use the ``CRYO_ET`` CTF (``rlnCtfScalefactor`` as contrast,
``rlnMicrographPreExposure`` as dose). The RELION conventions are listed in
the cryo-ET port plan (``pr179_coordination/cryoet_plan_20260923/PLAN.md``).
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import recovar.utils as utils
from recovar import core
from recovar.commands import parse_relion5_tomo
from recovar.data_io import cryoem_dataset, metadata_readers, starfile
from recovar.simulation import simulator

logger = logging.getLogger(__name__)

DEFAULT_OPTICS_GROUPS = (
    {"voltage": 300.0, "cs": 2.7, "amp_contrast": 0.1, "noise_scale": 1.0},
    {"voltage": 200.0, "cs": 1.4, "amp_contrast": 0.07, "noise_scale": 1.5},
)


def dose_symmetric_tilt_scheme(max_tilt=60.0, tilt_step=3.0, group_size=2):
    """Tilt angles in ascending order and their acquisition order (Hagen scheme).

    Acquisition starts at 0 degrees and alternates groups of ``group_size`` tilts
    between the positive and negative branches.
    """
    n_side = int(round(max_tilt / tilt_step))
    positive = list(np.arange(1, n_side + 1) * tilt_step)
    negative = list(-np.arange(1, n_side + 1) * tilt_step)
    order = [0.0]
    sign = 1
    while positive or negative:
        branch = positive if sign > 0 else negative
        order += branch[:group_size]
        del branch[:group_size]
        sign = -sign
    angles = np.sort(np.array(order))
    acquisition_index = np.array([order.index(a) for a in angles])
    return angles, acquisition_index


def _visible_frames_string(mask):
    return "[" + ",".join(str(int(v)) for v in mask) + "]"


def generate_relion5_tomo_dataset(
    output_folder,
    volumes_path_root,
    voxel_size,
    n_particles,
    *,
    grid_size=128,
    n_tomograms=4,
    optics_groups=DEFAULT_OPTICS_GROUPS,
    max_tilt=60.0,
    tilt_step=3.0,
    dose_per_tilt=3.0,
    tilt_axis_angle=85.0,
    defocus_range=(20000.0, 40000.0),
    astigmatism=300.0,
    tomogram_size=(1024, 1024, 256),
    hand=-1,
    snr=0.05,
    noise_model="white",
    contrast_std=0.1,
    hidden_tilt_fraction=0.0,
    volume_distribution=None,
    premultiplied_ctf=False,
    trailing_zero_format_in_vol_name=True,
    disc_type="linear_interp",
    seed=0,
):
    """Write a simulated RELION 5 subtomogram (2D-stack) project to ``output_folder``.

    Parameters
    ----------
    volumes_path_root, voxel_size, grid_size, trailing_zero_format_in_vol_name
        Ground-truth volumes, as for :func:`simulator.generate_synthetic_dataset`.
        The images use the volume voxel size (bin 1) and a ``grid_size`` box.
    n_particles : int
        Particles in total, spread evenly over ``n_tomograms``.
    optics_groups : sequence of dict
        One dict per optics group with ``voltage`` (kV), ``cs`` (mm),
        ``amp_contrast`` and ``noise_scale`` (noise standard-deviation factor).
        Tomogram ``t`` belongs to group ``t % len(optics_groups)``.
    snr : float
        Mean per-pixel signal power over noise power for a noise scale of 1.
    hidden_tilt_fraction : float
        Probability that a tilt is marked invisible for a particle (the zero
        tilt is always visible). Hidden tilts get no slice in the stack.
    tomogram_size : tuple of int
        ``rlnTomoSizeX/Y/Z`` in bin-1 pixels; must be even.

    Returns
    -------
    dict
        Paths of the written STAR files and the simulation ground truth, also
        saved as ``simulation_info.pkl``.
    """
    rng = np.random.default_rng(seed)
    if any(s % 2 for s in tomogram_size):
        raise ValueError(f"tomogram_size must be even (RELION centres at int(size/2)), got {tomogram_size}")
    os.makedirs(os.path.join(output_folder, "tilt_series"), exist_ok=True)

    volumes = simulator.load_volumes_from_folder(
        volumes_path_root, grid_size, trailing_zero_format_in_vol_name, normalize=False
    )
    scale_vol = 1 / np.mean(np.linalg.norm(volumes, axis=-1))
    volumes = volumes * scale_vol
    volume_distribution = (
        np.ones(volumes.shape[0]) / volumes.shape[0] if volume_distribution is None else volume_distribution
    )

    # ---- Tomograms and tilt series ----
    tilt_angles, acquisition_index = dose_symmetric_tilt_scheme(max_tilt, tilt_step)
    n_tilts = tilt_angles.size
    tomo_names = [f"TS_{t + 1:02d}" for t in range(n_tomograms)]
    tomo_optics = np.arange(n_tomograms) % len(optics_groups)
    tomo_rows = []
    for t, name in enumerate(tomo_names):
        og = optics_groups[tomo_optics[t]]
        defocus = rng.uniform(*defocus_range) + rng.normal(0, 200.0, n_tilts)
        tilt_df = pd.DataFrame(
            {
                "_rlnMicrographName": [f"tilt_series/{name}_{k + 1:03d}.mrc" for k in range(n_tilts)],
                "_rlnTomoNominalStageTiltAngle": tilt_angles,
                "_rlnMicrographPreExposure": acquisition_index * dose_per_tilt,
                "_rlnDefocusU": defocus + astigmatism / 2,
                "_rlnDefocusV": defocus - astigmatism / 2,
                "_rlnDefocusAngle": rng.uniform(-90.0, 90.0),
                "_rlnTomoXTilt": 0.0,
                "_rlnTomoYTilt": tilt_angles,
                "_rlnTomoZRot": tilt_axis_angle,
                "_rlnTomoXShiftAngst": rng.normal(0, 20.0, n_tilts),
                "_rlnTomoYShiftAngst": rng.normal(0, 20.0, n_tilts),
                "_rlnCtfScalefactor": np.cos(np.deg2rad(tilt_angles)),
            }
        )
        starfile.write_star_blocks(
            os.path.join(output_folder, "tilt_series", f"{name}.star"), {f"data_{name}": tilt_df}
        )
        tomo_rows.append(
            {
                "_rlnTomoName": name,
                "_rlnVoltage": og["voltage"],
                "_rlnSphericalAberration": og["cs"],
                "_rlnAmplitudeContrast": og["amp_contrast"],
                "_rlnMicrographOriginalPixelSize": voxel_size,
                "_rlnTomoHand": hand,
                "_rlnTomoTiltSeriesPixelSize": voxel_size,
                "_rlnTomoTiltSeriesStarFile": f"tilt_series/{name}.star",
                "_rlnTomoSizeX": tomogram_size[0],
                "_rlnTomoSizeY": tomogram_size[1],
                "_rlnTomoSizeZ": tomogram_size[2],
                "_rlnOpticsGroupName": f"opticsGroup{tomo_optics[t] + 1}",
            }
        )
    tomograms_path = os.path.join(output_folder, "tomograms.star")
    starfile.write_star_blocks(tomograms_path, {"data_global": pd.DataFrame(tomo_rows)})

    # ---- Particles ----
    particle_tomo = np.arange(n_particles) % n_tomograms
    particle_index = np.array([np.sum(particle_tomo[:p] == particle_tomo[p]) + 1 for p in range(n_particles)])
    half_extent = 0.4 * np.asarray(tomogram_size, dtype=float) * voxel_size
    half_extent[2] = 0.3 * tomogram_size[2] * voxel_size
    coords = rng.uniform(-half_extent, half_extent, size=(n_particles, 3))
    eulers = Rotation.random(n_particles, random_state=seed).as_euler("ZYZ", degrees=True)
    visible = rng.random((n_particles, n_tilts)) >= hidden_tilt_fraction
    visible[:, np.argmin(np.abs(tilt_angles))] = True
    particle_names = [f"{tomo_names[t]}/{i}" for t, i in zip(particle_tomo, particle_index)]
    stack_names = [f"Subtomograms/{tomo_names[t]}/{i}_stack2d.mrcs" for t, i in zip(particle_tomo, particle_index)]
    particles_df = pd.DataFrame(
        {
            "_rlnTomoName": [tomo_names[t] for t in particle_tomo],
            "_rlnCenteredCoordinateXAngst": coords[:, 0],
            "_rlnCenteredCoordinateYAngst": coords[:, 1],
            "_rlnCenteredCoordinateZAngst": coords[:, 2],
            "_rlnOpticsGroup": tomo_optics[particle_tomo] + 1,
            "_rlnTomoParticleName": particle_names,
            "_rlnImageName": stack_names,
            "_rlnOriginXAngst": 0.0,
            "_rlnOriginYAngst": 0.0,
            "_rlnOriginZAngst": 0.0,
            "_rlnAngleRot": eulers[:, 0],
            "_rlnAngleTilt": eulers[:, 1],
            "_rlnAnglePsi": eulers[:, 2],
            "_rlnRandomSubset": rng.permutation(np.arange(n_particles) % 2) + 1,
            "_rlnTomoVisibleFrames": [_visible_frames_string(v) for v in visible],
        }
    )
    optics_df = pd.DataFrame(
        {
            "_rlnOpticsGroup": np.arange(len(optics_groups)) + 1,
            "_rlnOpticsGroupName": [f"opticsGroup{g + 1}" for g in range(len(optics_groups))],
            "_rlnVoltage": [og["voltage"] for og in optics_groups],
            "_rlnSphericalAberration": [og["cs"] for og in optics_groups],
            "_rlnAmplitudeContrast": [og["amp_contrast"] for og in optics_groups],
            "_rlnTomoTiltSeriesPixelSize": voxel_size,
            "_rlnCtfDataAreCtfPremultiplied": int(premultiplied_ctf),
            "_rlnImageDimensionality": 2,
            "_rlnTomoSubtomogramBinning": 1.0,
            "_rlnImagePixelSize": voxel_size,
            "_rlnImageSize": grid_size,
        }
    )
    particles_path = os.path.join(output_folder, "particles.star")
    starfile.write_star_blocks(
        particles_path,
        {
            "data_general": pd.DataFrame({"_rlnTomoSubTomosAre2DStacks": [1]}),
            "data_optics": optics_df,
            "data_particles": particles_df,
        },
    )
    optimisation_set_path = os.path.join(output_folder, "optimisation_set.star")
    starfile.write_star_blocks(
        optimisation_set_path,
        {
            "data_": pd.DataFrame(
                {"_rlnTomoParticlesFile": ["particles.star"], "_rlnTomoTomogramsFile": ["tomograms.star"]}
            )
        },
    )

    # ---- Per-tilt poses and CTFs, read back through recovar's RELION 5 reader ----
    flat_path = os.path.join(output_folder, "particles_2d.star")
    parse_relion5_tomo.convert(tomograms_path, particles_path, flat_path)
    flat_df, _ = starfile.read_star(flat_path)
    euler_flat = flat_df[["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]].values.astype(np.float64)
    rots = utils.R_from_relion(euler_flat, degrees=True)
    ctf_params = np.zeros((len(flat_df), int(core.CTFParamIndex.TILT_ANGLE) + 1))
    ctf_params[:, : int(core.CTFParamIndex.BFACTOR)] = metadata_readers.parse_ctf_from_star(flat_path, grid_size)[:, 1:]
    ctf_params[:, core.CTFParamIndex.CONTRAST] = flat_df["_rlnCtfScalefactor"].values.astype(float)
    ctf_params[:, core.CTFParamIndex.DOSE] = flat_df["_rlnMicrographPreExposure"].values.astype(float)

    row_particle = pd.Index(particle_names).get_indexer(flat_df["_rlnGroupName"].values)
    row_slice = np.array([int(name.split("@")[0]) - 1 for name in flat_df["_rlnImageName"].values])
    particle_volume = rng.choice(volumes.shape[0], size=n_particles, p=volume_distribution)
    particle_contrast = 1 + rng.normal(0, contrast_std, n_particles)
    group_noise_scale = np.array([og["noise_scale"] for og in optics_groups])

    def make_dataset(rows):
        return cryoem_dataset.CryoEMDataset(
            None,
            voxel_size,
            cryoem_dataset.ImageMetadata(rots[rows], np.zeros((rows.size, 2)), ctf_params[rows]),
            ctf_evaluator=core.CTFEvaluator(mode=core.CTFMode.CRYO_ET),
            grid_size=grid_size,
        )

    batch_size = int(5 * utils.get_image_batch_size(grid_size, utils.get_gpu_memory_total()))

    def simulate(rows, noise_variance, contrast, noise_scale, seed_offset):
        return simulator.simulate_data(
            make_dataset(rows),
            volumes,
            noise_variance,
            batch_size,
            particle_volume[row_particle[rows]],
            contrast,
            noise_scale,
            seed=seed + seed_offset,
            disc_type=disc_type,
            premultiplied_ctf=premultiplied_ctf,
        )

    # Calibrate the noise level on a probe of the first rows: noise-free signal
    # power against the power of unit-model noise alone.
    noise_shape = simulator.get_noise_model(noise_model, grid_size)
    probe = np.arange(min(len(flat_df), 8 * n_tilts))
    ones, zeros = np.ones(probe.size), np.zeros(probe.size)
    signal_power = np.mean(simulate(probe, 0 * noise_shape, ones, ones, 1) ** 2)
    noise_power = np.mean(simulate(probe, noise_shape, zeros, ones, 2) ** 2)
    noise_variance = noise_shape * signal_power / (noise_power * snr)

    all_rows = np.arange(len(flat_df))
    row_optics = optics_df["_rlnOpticsGroup"].searchsorted(flat_df["_rlnOpticsGroup"].values.astype(int))
    images = simulate(
        all_rows,
        noise_variance,
        particle_contrast[row_particle],
        group_noise_scale[row_optics],
        0,
    )

    for p, stack_name in enumerate(stack_names):
        rows = np.nonzero(row_particle == p)[0]
        rows = rows[np.argsort(row_slice[rows])]
        os.makedirs(os.path.dirname(os.path.join(output_folder, stack_name)), exist_ok=True)
        utils.write_mrc_stack(os.path.join(output_folder, stack_name), images[rows], voxel_size=voxel_size)

    simulation_info = {
        "scale_vol": scale_vol,
        "volumes_path_root": volumes_path_root,
        "voxel_size": voxel_size,
        "grid_size": grid_size,
        "noise_variance": noise_variance.astype(np.float32),
        "snr": snr,
        "optics_groups": [dict(og) for og in optics_groups],
        "premultiplied_ctf": premultiplied_ctf,
        "particle_volume": particle_volume,
        "particle_contrast": particle_contrast,
        "particle_names": particle_names,
        "flat_rows_particle": row_particle,
        "flat_rows_ctf_params": ctf_params,
        "flat_rows_rots": rots,
    }
    utils.pickle_dump(simulation_info, os.path.join(output_folder, "simulation_info.pkl"))
    logger.info("Wrote %d particles x %d tilts (%d images) to %s", n_particles, n_tilts, len(flat_df), output_folder)
    return {
        "tomograms": tomograms_path,
        "particles": particles_path,
        "optimisation_set": optimisation_set_path,
        "particles_2d": flat_path,
        "simulation_info": simulation_info,
    }
