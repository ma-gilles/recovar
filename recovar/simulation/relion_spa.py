"""Simulate single-particle datasets with several RELION optics groups.

The output directory is a RELION project that ``relion_refine --i particles.star``
reads directly:

- ``particles.star``: ``data_optics`` with one row per optics group (voltage, Cs,
  amplitude contrast, ``rlnImagePixelSize``, ``rlnImageSize``) and ``data_particles``
  with ``rlnImageName``, ``rlnMicrographName``, defocus, Euler angles and
  ``rlnOpticsGroup``;
- ``Particles/opticsGroup<g>.mrcs``: the images of group ``g``, on its own pixel size
  and box.

Groups may differ in voltage, Cs, amplitude contrast, noise level, pixel size and box
size (:mod:`recovar.simulation.optics_groups`). RELION puts the reference on group 1's
grid, so group 1 should use ``voxel_size`` and ``grid_size``. Images are normalised as
``relion_preprocess --norm`` does, per group, since relion_refine expects that.
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import recovar.utils as utils
from recovar import core
from recovar.data_io import starfile
from recovar.simulation import optics_groups as optics_groups_sim
from recovar.simulation import simulator, solvent_contrast
from recovar.simulation.optics_groups import DEFAULT_OPTICS_GROUPS

logger = logging.getLogger(__name__)


def generate_relion_spa_dataset(
    output_folder,
    volumes_path_root,
    voxel_size,
    n_particles,
    *,
    grid_size=128,
    optics_groups=DEFAULT_OPTICS_GROUPS,
    micrographs_per_group=10,
    defocus_range=(10000.0, 30000.0),
    astigmatism=300.0,
    snr=0.05,
    noise_model="white",
    contrast_std=0.1,
    volume_distribution=None,
    trailing_zero_format_in_vol_name=True,
    disc_type="linear_interp",
    seed=0,
    atomic_solvent_correction=True,
    solvent_contrast_a=None,
    solvent_contrast_B=None,
    atomic_bfactor=None,
):
    """Write a simulated multi-optics-group RELION SPA project to ``output_folder``.

    Parameters are those of
    :func:`recovar.simulation.relion_tomo.generate_relion5_tomo_dataset` where they
    share a name: ``optics_groups`` dicts carry ``voltage``, ``cs``, ``amp_contrast``,
    ``noise_scale`` and optionally ``pixel_size`` and ``box_size``; ``snr`` is the
    first group's mean noise-free signal power over the shared noise power; the
    EM-development atomic-volume transform is on by default. Particle ``p`` belongs
    to group ``p % len(optics_groups)`` and to one of ``micrographs_per_group``
    micrographs of that group (RELION's scale-correction groups).

    Returns
    -------
    dict
        The particles STAR path and the simulation ground truth, also saved as
        ``simulation_info.pkl``.
    """
    rng = np.random.default_rng(seed)
    optics_groups = [{"pixel_size": voxel_size, "box_size": grid_size, **og} for og in optics_groups]
    os.makedirs(os.path.join(output_folder, "Particles"), exist_ok=True)

    solvent_record = solvent_contrast.record_from_options(
        atomic_solvent_correction, voxel_size, grid_size, solvent_contrast_a, solvent_contrast_B, atomic_bfactor
    )
    volumes = simulator.load_volumes_from_folder(
        volumes_path_root, grid_size, trailing_zero_format_in_vol_name, normalize=False
    )
    scale_vol = 1 / np.mean(np.linalg.norm(volumes, axis=-1))
    volumes = volumes * scale_vol
    if solvent_record["enabled"]:
        volumes = solvent_contrast.apply_record(volumes, solvent_record)
    volume_distribution = (
        np.ones(volumes.shape[0]) / volumes.shape[0] if volume_distribution is None else volume_distribution
    )

    particle_optics = np.arange(n_particles) % len(optics_groups)
    particle_micrograph = rng.integers(0, micrographs_per_group, n_particles)
    eulers = Rotation.random(n_particles, random_state=seed).as_euler("ZYZ", degrees=True)
    rots = utils.R_from_relion(eulers, degrees=True)
    defocus = rng.uniform(*defocus_range, n_particles)
    defocus_angle = rng.uniform(-90.0, 90.0, n_particles)
    ctf_params = np.zeros((n_particles, int(core.CTFParamIndex.TILT_ANGLE) + 1))
    ctf_params[:, core.CTFParamIndex.DFU] = defocus + astigmatism / 2
    ctf_params[:, core.CTFParamIndex.DFV] = defocus - astigmatism / 2
    ctf_params[:, core.CTFParamIndex.DFANG] = defocus_angle
    ctf_params[:, core.CTFParamIndex.VOLT] = [optics_groups[g]["voltage"] for g in particle_optics]
    ctf_params[:, core.CTFParamIndex.CS] = [optics_groups[g]["cs"] for g in particle_optics]
    ctf_params[:, core.CTFParamIndex.W] = [optics_groups[g]["amp_contrast"] for g in particle_optics]
    ctf_params[:, core.CTFParamIndex.CONTRAST] = 1.0
    particle_volume = rng.choice(volumes.shape[0], size=n_particles, p=volume_distribution)
    particle_contrast = 1 + rng.normal(0, contrast_std, n_particles)

    images, noise_variances = optics_groups_sim.simulate_optics_groups(
        volumes,
        optics_groups,
        particle_optics,
        rots,
        ctf_params,
        particle_volume,
        particle_contrast,
        ctf_evaluator=None,
        volumes_path_root=volumes_path_root,
        trailing_zero_format_in_vol_name=trailing_zero_format_in_vol_name,
        voxel_size=voxel_size,
        grid_size=grid_size,
        scale_vol=scale_vol,
        solvent_record=solvent_record,
        snr=snr,
        noise_model=noise_model,
        n_probe=256,
        seed=seed,
        disc_type=disc_type,
        premultiplied_ctf=False,
    )

    image_names = [""] * n_particles
    for g, og in enumerate(optics_groups):
        members = np.nonzero(particle_optics == g)[0]
        if members.size == 0:
            continue
        stack_name = f"Particles/opticsGroup{g + 1}.mrcs"
        stack = np.stack([images[p] for p in members]).astype(np.float32)
        stack, _, _ = simulator.normalize_particles_relion_style(stack, int(round(0.375 * og["box_size"])))
        utils.write_mrc_stack(os.path.join(output_folder, stack_name), stack, voxel_size=og["pixel_size"])
        for k, p in enumerate(members):
            image_names[p] = f"{k + 1}@{stack_name}"

    optics_df = pd.DataFrame(
        {
            "_rlnOpticsGroup": np.arange(len(optics_groups)) + 1,
            "_rlnOpticsGroupName": [f"opticsGroup{g + 1}" for g in range(len(optics_groups))],
            "_rlnVoltage": [og["voltage"] for og in optics_groups],
            "_rlnSphericalAberration": [og["cs"] for og in optics_groups],
            "_rlnAmplitudeContrast": [og["amp_contrast"] for og in optics_groups],
            "_rlnImagePixelSize": [og["pixel_size"] for og in optics_groups],
            "_rlnImageSize": [og["box_size"] for og in optics_groups],
            "_rlnImageDimensionality": 2,
            "_rlnCtfDataAreCtfPremultiplied": 0,
        }
    )
    particles_df = pd.DataFrame(
        {
            "_rlnImageName": image_names,
            "_rlnMicrographName": [
                f"Micrographs/opticsGroup{g + 1}_{m + 1:03d}.mrc" for g, m in zip(particle_optics, particle_micrograph)
            ],
            "_rlnDefocusU": ctf_params[:, core.CTFParamIndex.DFU],
            "_rlnDefocusV": ctf_params[:, core.CTFParamIndex.DFV],
            "_rlnDefocusAngle": defocus_angle,
            "_rlnOpticsGroup": particle_optics + 1,
            "_rlnAngleRot": eulers[:, 0],
            "_rlnAngleTilt": eulers[:, 1],
            "_rlnAnglePsi": eulers[:, 2],
            "_rlnOriginXAngst": 0.0,
            "_rlnOriginYAngst": 0.0,
        }
    )
    particles_path = os.path.join(output_folder, "particles.star")
    starfile.write_star_blocks(particles_path, {"data_optics": optics_df, "data_particles": particles_df})

    simulation_info = {
        "scale_vol": scale_vol,
        "volumes_path_root": volumes_path_root,
        "trailing_zero_format_in_vol_name": trailing_zero_format_in_vol_name,
        "voxel_size": voxel_size,
        "grid_size": grid_size,
        solvent_contrast.METADATA_KEY: solvent_record,
        "image_assignment": particle_volume,
        "per_image_contrast": particle_contrast,
        "noise_variance_per_optics_group": noise_variances,
        "snr": snr,
        "optics_groups": [dict(og) for og in optics_groups],
        "rots": rots,
        "ctf_params": ctf_params,
    }
    utils.pickle_dump(simulation_info, os.path.join(output_folder, "simulation_info.pkl"))
    logger.info("Wrote %d particles in %d optics groups to %s", n_particles, len(optics_groups), output_folder)
    return {"particles": particles_path, "simulation_info": simulation_info}
