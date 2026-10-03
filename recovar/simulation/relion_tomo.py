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
- ``tilt_series/<tomo>_001.mrc``: an empty image of the tilt-image size. No
  tilt-series micrographs are simulated, so programs that read them
  (``relion_tomo_subtomo``, ``relion_tomo_reconstruct_particle``) do not apply;
- ``particles_2d.star``: the same data flattened to one row per particle-tilt
  (``recovar pipeline --tilt-series`` input).

Each tomogram belongs to one optics group, and groups may differ in voltage,
Cs, amplitude contrast and noise level. Per-tilt poses and depth-corrected
defocus are not computed here: the RELION STAR files are written first and
then read back with :func:`recovar.commands.parse_relion5_tomo.convert`, so the
simulator and recovar's RELION 5 reader share one geometry implementation.
Images use the CTF relion_refine applies to a tomo image (:func:`relion_tomo_ctf`):
the SPA CTF scaled by ``rlnCtfScalefactor`` times RELION's dose weight for the
tilt's ``rlnMicrographPreExposure``. This is deliberately not recovar's
``CRYO_ET`` dose filter, which has a cutoff and a 200 kV factor RELION lacks.
The RELION conventions are listed in the cryo-ET port plan
(``pr179_coordination/cryoet_plan_20260923/PLAN.md``).
"""

import functools
import logging
import math
import os

import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import recovar.core.fourier_transform_utils as fourier_transform_utils
import recovar.utils as utils
from recovar import core
from recovar.commands import parse_relion5_tomo
from recovar.data_io import metadata_readers, starfile
from recovar.simulation import optics_groups as optics_groups_sim
from recovar.simulation import simulator, solvent_contrast
from recovar.simulation.optics_groups import DEFAULT_OPTICS_GROUPS

logger = logging.getLogger(__name__)


def relion_dose_weight(freq_sq, dose):
    """RELION's dose weight ``exp(-0.5 N / Ne(k))``, ``Ne = 0.245 k^-1.665 + 2.81`` (``src/ctf.h:219-233``).

    ``freq_sq`` is ``k^2`` in 1/A^2 per pixel and ``dose`` the cumulative dose ``N``
    in e/A^2 per image. At ``k = 0`` ``Ne`` is infinite and the weight is 1.
    """
    critical_exposure = 0.245 * jnp.power(freq_sq, -0.8325) + 2.81
    return jnp.exp(-0.5 * jnp.asarray(dose)[:, None] / critical_exposure[None, :])


def relion_tomo_ctf(ctf_params, image_shape, voxel_size, *, half_image=False, mag_matrix=None, even_zernike=None):
    """CTF of a RELION tomo image: SPA CTF (with ``CONTRAST`` = ``rlnCtfScalefactor``) times the dose weight.

    ``mag_matrix`` (``rlnMagMat``, 2x2) and ``even_zernike`` (``rlnEvenZernike``) are the
    image's optics-group terms as ``CTF::getCTF`` applies them: every term, damping
    included, is evaluated at the magnified frequency ``M k``, and the even Zernike
    phase ``sum_i c_i Z_i(M k)`` (:func:`zernike_phase`) is added to ``gamma``.
    """
    grid_fn = (
        fourier_transform_utils.get_k_coordinate_of_each_pixel_half
        if half_image
        else fourier_transform_utils.get_k_coordinate_of_each_pixel
    )
    ctf_params = jnp.asarray(ctf_params)
    freqs = grid_fn(image_shape, voxel_size, scaled=True, dtype=jnp.result_type(ctf_params, jnp.float32))
    if mag_matrix is not None:
        freqs = freqs @ jnp.asarray(np.asarray(mag_matrix).T, dtype=freqs.dtype)
    gamma_offset = None
    if even_zernike is not None and np.any(np.asarray(even_zernike) != 0):
        gamma_offset = zernike_phase(even_zernike, even_index_to_mn, freqs)
    ctf = core.evaluate_ctf(freqs, ctf_params[:, : int(core.CTFParamIndex.DOSE)], gamma_offset)
    return ctf * relion_dose_weight(jnp.sum(freqs**2, axis=-1), ctf_params[:, core.CTFParamIndex.DOSE])


# RELION's optics-group aberrations (src/jaz/gravis/Zernike.cpp, src/jaz/single_particle/obs_model.cpp,
# tilt_helper.cpp), as relion_refine applies them to every tilt image. These are the simulator's own
# implementation, an independent forward model for the codes that refine the data.


def odd_index_to_mn(index):
    """``Zernike::oddIndexToMN``: the ``(m, n)`` of odd Zernike coefficient ``index``."""
    k = int((np.sqrt(1 + 4 * index) - 1.0) / 2.0)
    n = 2 * k + 1
    return 2 * (index - (k * k + k)) - n, n


def even_index_to_mn(index):
    """``Zernike::evenIndexToMN``: the ``(m, n)`` of even Zernike coefficient ``index``."""
    k = int(np.sqrt(float(index)))
    return 2 * (index - k * k - k), 2 * k


def _zernike_cartesian(xp, m, n, x, y):
    """``Zernike::Z_cart``: ``R_n^|m|(rho)`` times ``cos(m phi)`` (m >= 0) or ``sin(-m phi)``."""
    rho = xp.hypot(x, y)
    phi = xp.where((x == 0) & (y == 0), 0.0, xp.arctan2(y, x))
    a = abs(m)
    radial = xp.zeros_like(rho)
    if (n - a) % 2 == 0:
        for k in range((n - a) // 2 + 1):
            c = (-1) ** k * math.factorial(n - k)
            c /= math.factorial(k) * math.factorial((n + a) // 2 - k) * math.factorial((n - a) // 2 - k)
            radial = radial + c * rho ** (n - 2 * k)
    return radial * (xp.cos(m * phi) if m >= 0 else xp.sin(-m * phi))


def zernike_phase(coefficients, index_to_mn, freqs):
    """``sum_i c_i Z_i(k)`` at frequencies ``freqs`` ``(n, 2)`` in 1/A (``getGammaOffset``, ``getPhaseCorrection``).

    NumPy frequencies give a NumPy phase; JAX frequencies (e.g. inside a traced CTF evaluator) a JAX phase.
    """
    xp = np if isinstance(freqs, np.ndarray) else jnp
    x, y = freqs[:, 0], freqs[:, 1]
    phase = xp.zeros_like(x)
    for index, c in enumerate(coefficients):
        if c != 0:
            m, n = index_to_mn(index)
            phase = phase + float(c) * _zernike_cartesian(xp, m, n, x, y)
    return phase


def relion_wavelength(voltage_kv):
    """Electron wavelength in A (``ObservationModel``, obs_model.cpp)."""
    volts = float(voltage_kv) * 1e3
    return 12.2643247 / np.sqrt(volts * (1.0 + volts * 0.978466e-6))


def odd_coefficients_with_beam_tilt(odd_zernike, beam_tilt_mrad, cs_mm, voltage_kv):
    """``TiltHelper::insertTilt``: the odd Zernike coefficients with a beam tilt ``(x, y)`` in mrad added."""
    coefficients = [float(c) for c in (odd_zernike or [])]
    if beam_tilt_mrad is None:
        return coefficients
    coefficients += [0.0] * max(0, 6 - len(coefficients))
    lam = relion_wavelength(voltage_kv)
    scale = float(cs_mm) * 20000 * lam * lam * 3.141592654
    z3x = -scale * float(beam_tilt_mrad[0]) / 3.0
    z3y = -scale * float(beam_tilt_mrad[1]) / 3.0
    coefficients[1] += 2.0 * z3x
    coefficients[0] += 2.0 * z3y
    coefficients[4] += z3x
    coefficients[3] += z3y
    return coefficients


def modulate_odd_aberrations(images, phase_freqs, odd_phase):
    """Images ``[n, N, N]`` with their Fourier transform times ``exp(i phase)``.

    relion_refine demodulates every image by ``exp(-i phase)`` of its optics group's odd
    aberrations before scoring and backprojection (``ObservationModel::demodulatePhase``),
    so recorded images carry ``exp(i phase)``. ``odd_phase`` is the phase at the centred
    frequency grid ``phase_freqs`` of :func:`fourier_transform_utils.get_k_coordinate_of_each_pixel`.
    """
    images = np.asarray(images)
    shape = images.shape[-2:]
    factor = np.exp(1j * np.asarray(odd_phase)).reshape(shape)
    ft = np.asarray(fourier_transform_utils.get_dft2(jnp.asarray(images)))
    return np.asarray(fourier_transform_utils.get_idft2(jnp.asarray(ft * factor[None]))).real.astype(images.dtype)


def _setting_aberrations(setting):
    """``(mag_matrix, even_zernike, odd coefficients)`` of an optics setting; None where absent."""
    mag = setting.get("mag_matrix")
    even = setting.get("even_zernike")
    odd = odd_coefficients_with_beam_tilt(
        setting.get("odd_zernike"), setting.get("beam_tilt"), setting["cs"], setting["voltage"]
    )
    return (
        None if mag is None else np.asarray(mag, dtype=np.float64),
        None if even is None or not np.any(np.asarray(even) != 0) else list(even),
        odd if np.any(np.asarray(odd) != 0) else None,
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


def _relion_euler_matrix(eulers):
    """RELION's Euler_angles2matrix (src/euler.cpp) for ``[..., (rot, tilt, psi)]`` in degrees."""

    a, b, g = np.deg2rad(np.asarray(eulers, dtype=np.float64)).T
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(g), np.sin(g)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    return np.stack(
        [
            np.stack([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb], -1),
            np.stack([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb], -1),
            np.stack([sc, ss, cb], -1),
        ],
        -2,
    )


def _visible_frames_string(mask):
    return "[" + ",".join(str(int(v)) for v in mask) + "]"


def _relion_vector(values):
    return "[" + ",".join(f"{float(v):.10g}" for v in values) + "]"


def _write_optics_aberrations(optics_df, optics_groups):
    """Add the groups' aberration columns to the optics table (only the features some group uses)."""
    if any(og.get("odd_zernike") is not None for og in optics_groups):
        optics_df["_rlnOddZernike"] = [_relion_vector(og.get("odd_zernike") or [0.0] * 6) for og in optics_groups]
    if any(og.get("beam_tilt") is not None for og in optics_groups):
        tilts = [og.get("beam_tilt") or (0.0, 0.0) for og in optics_groups]
        optics_df["_rlnBeamTiltX"] = [float(t[0]) for t in tilts]
        optics_df["_rlnBeamTiltY"] = [float(t[1]) for t in tilts]
    if any(og.get("even_zernike") is not None for og in optics_groups):
        optics_df["_rlnEvenZernike"] = [_relion_vector(og.get("even_zernike") or [0.0] * 9) for og in optics_groups]
    if any(og.get("mag_matrix") is not None for og in optics_groups):
        mags = [
            np.eye(2) if og.get("mag_matrix") is None else np.asarray(og["mag_matrix"], dtype=np.float64)
            for og in optics_groups
        ]
        for i in range(2):
            for j in range(2):
                optics_df[f"_rlnMagMat{i}{j}"] = [float(m[i, j]) for m in mags]


def generate_relion5_tomo_dataset(
    output_folder,
    volumes_path_root,
    voxel_size,
    n_particles,
    *,
    grid_size=128,
    n_tomograms=4,
    optics_groups=DEFAULT_OPTICS_GROUPS[:1],
    optics_group_per_tomogram=True,
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
    origin_std_angstrom=0.0,
    particle_eulers=None,
    volume_distribution=None,
    premultiplied_ctf=False,
    trailing_zero_format_in_vol_name=True,
    disc_type="linear_interp",
    seed=0,
    atomic_solvent_correction=True,
    solvent_contrast_a=None,
    solvent_contrast_B=None,
    atomic_bfactor=None,
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
        The optics settings: one dict per setting with ``voltage`` (kV), ``cs`` (mm),
        ``amp_contrast`` and ``noise_scale`` (noise standard-deviation factor),
        and optionally ``pixel_size`` (A) and ``box_size`` (px), which default to
        ``voxel_size`` and ``grid_size``. Optional RELION optics-group aberrations, written
        to the optics table and applied to the images as relion_refine models them:
        ``beam_tilt`` (``(x, y)`` mrad, ``rlnBeamTiltX/Y``), ``odd_zernike`` and
        ``even_zernike`` (coefficient lists, ``rlnOddZernike``/``rlnEvenZernike``) and
        ``mag_matrix`` (2x2, ``rlnMagMat00..11``). Projections use ``inv(M3) A`` (RELION's
        ``applyAnisoMag``), the CTF is evaluated at ``M k`` with the even Zernike phase
        (:func:`relion_tomo_ctf`), and each image is multiplied by ``exp(i phase)`` of the
        odd terms (:func:`modulate_odd_aberrations`). A setting's volume is Fourier-resampled to
        its pixel size, so ``grid_size * voxel_size / pixel_size`` must be an even
        integer. RELION puts the reference on group 1's grid.
        Tomogram ``t`` is imaged with setting ``t % len(optics_groups)``. The default
        is one setting, so every tomogram has the same optics.
    optics_group_per_tomogram : bool
        Write one optics group per tomogram (``opticsGroup<t+1>`` with its setting's
        values), as RELION 5's tomogram import does; this is the default. False writes
        one optics group per setting, shared by its tomograms. Images and noise are
        simulated per setting either way, so tomograms with one setting share one
        noise level.
    snr : float
        Mean per-pixel noise-free signal power of the first optics group's images
        over the per-pixel noise power shared by all groups (before ``noise_scale``).
    premultiplied_ctf : bool
        Write CTF-premultiplied tilt images (``rlnCtfDataAreCtfPremultiplied 1``, the default of RELION 5's
        ``relion_tomo_subtomo``): each image, noise included, multiplied by RELION's CTF of its tilt.
    hidden_tilt_fraction : float
        Probability that a tilt is marked invisible for a particle (the zero
        tilt is always visible). Hidden tilts get no slice in the stack.
    origin_std_angstrom : float
        Standard deviation of each particle's ground-truth 3D offset
        (``rlnOriginX/Y/ZAngst``, tomogram frame). Every tilt image of the particle is
        shifted by its projection, ``Aproj_i[:2] o`` (RELION's
        ``Experiment::getTranslationInTiltSeries``); 0 writes centred particles.
    particle_eulers : array of shape (n_particles, 3) or None
        Particle poses as RELION Euler angles (rot, tilt, psi) in degrees (ZYZ,
        ``rlnAngleRot/Tilt/Psi``), e.g. a preferred-orientation distribution. None draws
        uniform random rotations from ``seed``. The other random draws do not depend on it.
    tomogram_size : tuple of int
        ``rlnTomoSizeX/Y/Z`` in bin-1 pixels; must be even.
    atomic_solvent_correction, solvent_contrast_a, solvent_contrast_B, atomic_bfactor
        The shared atomic-volume transform (:mod:`recovar.simulation.solvent_contrast`).
        This is an EM/VDAM development simulator, so the EM-development preset is
        on by default; pass ``atomic_solvent_correction=False`` for experimental
        or already-corrected maps. Each optics group gets the operator on its own
        grid, and the record is stored in ``simulation_info`` for ground-truth loading.

    Returns
    -------
    dict
        Paths of the written STAR files and the simulation ground truth, also
        saved as ``simulation_info.pkl``.
    """
    rng = np.random.default_rng(seed)
    settings = [{"pixel_size": voxel_size, "box_size": grid_size, **og} for og in optics_groups]
    setting_aberrations = [_setting_aberrations(setting) for setting in settings]
    # Each tomogram's setting, each tomogram's STAR optics group and each group's setting.
    tomo_settings = np.arange(n_tomograms) % len(settings)
    tomo_optics = np.arange(n_tomograms) if optics_group_per_tomogram else tomo_settings
    group_settings = tomo_settings if optics_group_per_tomogram else np.arange(len(settings))
    optics_groups = [settings[s] for s in group_settings]
    if any(s % 2 for s in tomogram_size):
        raise ValueError(f"tomogram_size must be even (RELION centres at int(size/2)), got {tomogram_size}")
    os.makedirs(os.path.join(output_folder, "tilt_series"), exist_ok=True)

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

    # ---- Tomograms and tilt series ----
    tilt_angles, acquisition_index = dose_symmetric_tilt_scheme(max_tilt, tilt_step)
    n_tilts = tilt_angles.size
    tomo_names = [f"TS_{t + 1:02d}" for t in range(n_tomograms)]
    tomo_rows = []
    for t, name in enumerate(tomo_names):
        og = optics_groups[tomo_optics[t]]
        defocus = rng.uniform(*defocus_range) + rng.normal(0, 200.0, n_tilts)
        # No tilt-series micrographs are simulated. relion_refine only reads the
        # header of the first tilt image, for its size (TomogramSet::loadTomogram,
        # tomogram_set.cpp:264-270), so only that file exists, as an empty image.
        micrographs = [f"tilt_series/{name}_{k + 1:03d}.mrc" for k in range(n_tilts)]
        with mrcfile.new(os.path.join(output_folder, micrographs[0]), overwrite=True) as mrc:
            mrc.set_data(np.zeros(tomogram_size[1::-1], dtype=np.int8))
            mrc.voxel_size = og["pixel_size"]
        tilt_df = pd.DataFrame(
            {
                "_rlnMicrographName": micrographs,
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
                "_rlnMicrographOriginalPixelSize": og["pixel_size"],
                "_rlnTomoHand": hand,
                "_rlnTomoTiltSeriesPixelSize": og["pixel_size"],
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
    if particle_eulers is None:
        eulers = Rotation.random(n_particles, random_state=seed).as_euler("ZYZ", degrees=True)
    else:
        eulers = np.asarray(particle_eulers, dtype=np.float64)
        if eulers.shape != (n_particles, 3) or not np.all(np.isfinite(eulers)):
            raise ValueError(f"particle_eulers must be finite with shape ({n_particles}, 3), got {eulers.shape}")
    visible = rng.random((n_particles, n_tilts)) >= hidden_tilt_fraction
    visible[:, np.argmin(np.abs(tilt_angles))] = True
    particle_names = [f"{tomo_names[t]}/{i}" for t, i in zip(particle_tomo, particle_index)]
    stack_names = [f"Subtomograms/{tomo_names[t]}/{i}_stack2d.mrcs" for t, i in zip(particle_tomo, particle_index)]
    origins = (
        rng.normal(0.0, origin_std_angstrom, size=(n_particles, 3))
        if origin_std_angstrom > 0
        else np.zeros((n_particles, 3))
    )
    particles_df = pd.DataFrame(
        {
            "_rlnTomoName": [tomo_names[t] for t in particle_tomo],
            "_rlnCenteredCoordinateXAngst": coords[:, 0],
            "_rlnCenteredCoordinateYAngst": coords[:, 1],
            "_rlnCenteredCoordinateZAngst": coords[:, 2],
            "_rlnOpticsGroup": tomo_optics[particle_tomo] + 1,
            "_rlnTomoParticleName": particle_names,
            "_rlnImageName": stack_names,
            "_rlnOriginXAngst": origins[:, 0],
            "_rlnOriginYAngst": origins[:, 1],
            "_rlnOriginZAngst": origins[:, 2],
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
            "_rlnTomoTiltSeriesPixelSize": [og["pixel_size"] for og in optics_groups],
            "_rlnCtfDataAreCtfPremultiplied": int(premultiplied_ctf),
            "_rlnImageDimensionality": 2,
            "_rlnTomoSubtomogramBinning": 1.0,
            "_rlnImagePixelSize": [og["pixel_size"] for og in optics_groups],
            "_rlnImageSize": [og["box_size"] for og in optics_groups],
        }
    )
    _write_optics_aberrations(optics_df, optics_groups)
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
    row_optics = optics_df["_rlnOpticsGroup"].searchsorted(flat_df["_rlnOpticsGroup"].values.astype(int))
    row_settings = group_settings[row_optics]
    # The tilt's projection matrix: the flattened per-tilt matrix is Aproj_i times the pose.
    row_projection = np.einsum(
        "iab,icb->iac",
        _relion_euler_matrix(euler_flat),
        _relion_euler_matrix(eulers)[row_particle],
    )
    row_pixel = np.array([optics_groups[g]["pixel_size"] for g in row_optics])
    row_translations = np.einsum("iab,ib->ia", row_projection[:, :2, :], origins[row_particle]) / row_pixel[:, None]
    # Anisotropic magnification: recovar's projection matrices are inv(A)^T, so RELION's inv(M3) A
    # (ObservationModel::applyAnisoMag) is M3^T inv(A)^T.
    row_projection_rots = np.array(rots, dtype=np.float64, copy=True)
    for setting, (mag, _even, _odd) in enumerate(setting_aberrations):
        if mag is not None:
            mag3 = np.eye(3)
            mag3[:2, :2] = mag
            rows = row_settings == setting
            row_projection_rots[rows] = np.einsum("ij,njk->nik", mag3.T, row_projection_rots[rows])
    row_images, setting_noise_variances = optics_groups_sim.simulate_optics_groups(
        volumes,
        settings,
        row_settings,
        row_projection_rots.astype(np.asarray(rots).dtype),
        ctf_params,
        particle_volume[row_particle],
        particle_contrast[row_particle],
        ctf_evaluator=[
            functools.partial(relion_tomo_ctf, mag_matrix=mag, even_zernike=even)
            for mag, even, _odd in setting_aberrations
        ],
        volumes_path_root=volumes_path_root,
        trailing_zero_format_in_vol_name=trailing_zero_format_in_vol_name,
        voxel_size=voxel_size,
        grid_size=grid_size,
        scale_vol=scale_vol,
        solvent_record=solvent_record,
        snr=snr,
        noise_model=noise_model,
        n_probe=8 * n_tilts,
        seed=seed,
        disc_type=disc_type,
        premultiplied_ctf=premultiplied_ctf,
        row_translations=row_translations,
    )

    if premultiplied_ctf:
        # simulate_data multiplies by recovar's CTF, which is minus RELION's (CTF::getCTF returns
        # -sin(gamma)); relion_tomo_subtomo premultiplies by RELION's, so a RELION reader sees CTF^2 X.
        row_images = [-image for image in row_images]
    for setting, (mag, _even, odd) in enumerate(setting_aberrations):
        rows = np.nonzero(row_settings == setting)[0]
        if odd is None or rows.size == 0:
            continue
        box = settings[setting]["box_size"]
        freqs = np.asarray(
            fourier_transform_utils.get_k_coordinate_of_each_pixel(
                (box, box), settings[setting]["pixel_size"], scaled=True, dtype=jnp.float64
            )
        )
        if mag is not None:
            freqs = freqs @ mag.T
        phase = zernike_phase(odd, odd_index_to_mn, freqs)
        for start in range(0, rows.size, 1024):
            block = rows[start : start + 1024]
            for row, image in zip(
                block, modulate_odd_aberrations(np.stack([row_images[r] for r in block]), freqs, phase)
            ):
                row_images[row] = image

    # Tilt images are normalised as relion_preprocess --norm does (background mean 0 and
    # standard deviation 1 per image), as the SPA writer does; RELION's refinement has
    # absolute thresholds (e.g. the sigma2_noise < 1e-14 fill in maximizationOtherParameters)
    # that images on a tiny greyscale trip.
    row_bg_mean = np.zeros(len(flat_df))
    row_bg_std = np.ones(len(flat_df))
    for p, stack_name in enumerate(stack_names):
        rows = np.nonzero(row_particle == p)[0]
        rows = rows[np.argsort(row_slice[rows])]
        box = optics_groups[row_optics[rows[0]]]["box_size"]
        stack, row_bg_mean[rows], row_bg_std[rows] = simulator.normalize_particles_relion_style(
            np.stack([row_images[r] for r in rows]), int(round(0.375 * box))
        )
        os.makedirs(os.path.dirname(os.path.join(output_folder, stack_name)), exist_ok=True)
        utils.write_mrc_stack(
            os.path.join(output_folder, stack_name),
            np.asarray(stack, dtype=np.float32),
            voxel_size=optics_groups[row_optics[rows[0]]]["pixel_size"],
        )

    # The keys load_heterogeneous_reconstruction reads describe the images of
    # particles_2d.star, row by row, on the (voxel_size, grid_size) reference grid.
    simulation_info = {
        "scale_vol": scale_vol,
        "volumes_path_root": volumes_path_root,
        "trailing_zero_format_in_vol_name": trailing_zero_format_in_vol_name,
        "voxel_size": voxel_size,
        "grid_size": grid_size,
        solvent_contrast.METADATA_KEY: solvent_record,
        "image_assignment": particle_volume[row_particle],
        "per_image_contrast": particle_contrast[row_particle],
        # Per STAR optics group (a group's noise is its setting's).
        "noise_variance_per_optics_group": [setting_noise_variances[g] for g in group_settings],
        "optics_settings": [dict(og) for og in settings],
        "snr": snr,
        "optics_groups": [dict(og) for og in optics_groups],
        "premultiplied_ctf": premultiplied_ctf,
        "particle_volume": particle_volume,
        "particle_contrast": particle_contrast,
        "particle_names": particle_names,
        "flat_rows_particle": row_particle,
        "flat_rows_ctf_params": ctf_params,
        "flat_rows_rots": rots,
        "particle_origins_angstrom": origins,
        "flat_rows_translations_px": row_translations,
        "relion_normalize": True,
        "flat_rows_bg_mean": row_bg_mean,
        "flat_rows_bg_std": row_bg_std,
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
