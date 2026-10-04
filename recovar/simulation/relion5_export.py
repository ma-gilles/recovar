"""Export the simulator's grouped projections as native RELION 5 cryo-ET data.

Each simulated particle has its own synthetic tomogram: the simulator can draw
different CTF parameters for every image, so sharing a real tomogram catalogue
would silently change the forward model. These are already extracted particle
stacks, not full fields of view suitable for motion correction or re-extraction.

The exported images have the opposite sign to RECOVAR images. Native stack
frames and tilt-series rows are both sorted by nominal tilt angle. Particle
poses, shifts, dose, CTF parameters, and frame provenance follow that same order.
RELION's Grant--Grigorieff dose envelope omits RECOVAR's hard high-dose cutoff;
this remaining modelling difference is reported explicitly in the manifest.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy.spatial.transform import Rotation

from recovar.core.ctf import CTFParamIndex as C
from recovar.utils import R_from_relion, R_to_relion

logger = logging.getLogger(__name__)
_ARTIFACTS = (
    "particles.star",
    "tomograms.star",
    "optimisation_set.star",
    "particles",
    "tilt_series",
    "frame_rows.npy",
    "particle_groups.npy",
    "relion5_export.json",
)


def validate_output_directory(output_folder):
    """Fail before simulation if a RELION export could overwrite existing data."""
    root = Path(output_folder).absolute()
    if root.is_symlink():
        raise FileExistsError(f"RELION5 output must be a new directory, not a symlink: {root}")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"RELION5 output directory must be absent or empty: {root}")
    return root


def native_ctf_generator(original, forward_bfactor_snapshots):
    """Wrap a generator to record its physical B-factor without changing values.

    The legacy simulator overwrites B-factors with WARP dose bookkeeping after
    projection. Native export needs the values that were actually used. This
    wrapper returns the original arrays unchanged, including their column count.
    """

    def generate(n_images, grid_size):
        ctf_params, rotations, translations = original(n_images, grid_size)
        params = np.asarray(ctf_params)
        if params.ndim != 2 or params.shape[0] != n_images or params.shape[1] < 9:
            raise ValueError("Native export requires at least nine CTF parameters per image")
        forward_bfactor_snapshots.append(params[:, C.BFACTOR].copy())
        return ctf_params, rotations, translations

    return generate


def _tilt_angles(n_tilts, angle_per_tilt):
    half = np.arange(n_tilts // 2 + 1, dtype=float) * angle_per_tilt
    angles = np.zeros(n_tilts)
    angles[::2] = -half[:-1] if n_tilts % 2 == 0 else -half
    angles[1::2] = half[1:]
    return angles


def _finite_array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}; got {array.shape}")
    return array


def _write_star(path, blocks):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    # Nine significant decimal places preserve Euler angles and physical units.
    starfile.write(blocks, path, overwrite=False, float_format="%.9f")


def export_relion5(
    output_folder,
    images,
    ctf_params,
    rotations,
    translations,
    voxel_size,
    tilt_groups,
    *,
    n_tilts,
    dose_per_tilt,
    angle_per_tilt,
    image_dtype=np.float32,
    simulation_info=None,
):
    """Write native RELION 5 particle/tomogram/optimisation STARs and stacks.

    ``images`` is ``(N,D,D)`` (a memmap is supported), rotations are RECOVAR
    matrices, and translations are in image pixels. Only complete standard
    simulator tilt groups are supported. Metadata and all output paths are
    validated before the first output is created. A completion manifest is
    written last; its absence indicates an incomplete export.

    ``simulation_info['forward_ctf_bfactor']`` is the actual per-image B-factor
    used during projection, before the legacy WARP dose encoding overwrites
    column 7. ``premultiplied_ctf=True`` is deliberately unsupported.
    """
    info = {} if simulation_info is None else simulation_info
    if info.get("premultiplied_ctf", False):
        raise ValueError("RELION5 export does not support CTF-premultiplied simulation")
    if not isinstance(n_tilts, (int, np.integer)) or n_tilts < 1:
        raise ValueError("RELION5 export requires a positive integer n_tilts")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    if dose_per_tilt is None or not np.isfinite(dose_per_tilt) or dose_per_tilt < 0:
        raise ValueError("dose_per_tilt must be finite and nonnegative")
    if angle_per_tilt is None or not np.isfinite(angle_per_tilt) or angle_per_tilt < 0:
        raise ValueError("angle_per_tilt must be finite and nonnegative")
    dtype = np.dtype(image_dtype)
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("RELION5 stacks require float16 or float32 image_dtype")
    if not hasattr(images, "shape") or len(images.shape) != 3:
        raise ValueError("images must have shape (N,D,D)")
    n_images, grid_size, width = images.shape
    if n_images < 1 or grid_size < 2 or grid_size != width or grid_size % 2:
        raise ValueError("RELION5 export requires nonempty, even-sized square images")
    if n_images % n_tilts:
        raise ValueError("RELION5 export requires complete n_tilts-image particles")
    ctf = _finite_array(ctf_params, (n_images, 11), "ctf_params").astype(float)
    rots = _finite_array(rotations, (n_images, 3, 3), "rotations")
    trans = _finite_array(translations, (n_images, 2), "translations")
    groups = np.asarray(tilt_groups)
    if groups.shape != (n_images,) or groups.dtype.kind not in "iu":
        raise ValueError("tilt_groups must contain one integer group ID per image")
    if not np.allclose(rots @ rots.swapaxes(-1, -2), np.eye(3), atol=2e-5, rtol=0):
        raise ValueError("rotations must be orthogonal matrices")
    if not np.allclose(np.linalg.det(rots), 1, atol=2e-5, rtol=0):
        raise ValueError("rotations must have determinant +1")

    group_ids, first_indices = np.unique(groups, return_index=True)
    group_ids = group_ids[np.argsort(first_indices)]
    n_particles = len(group_ids)
    rows = [np.flatnonzero(groups == group) for group in group_ids]
    if any(len(row) != n_tilts for row in rows):
        raise ValueError("Every physical particle must have exactly n_tilts images")
    acquisition_rows = np.stack(rows)
    angles = _tilt_angles(n_tilts, angle_per_tilt)
    if np.max(np.abs(angles)) >= 90:
        raise ValueError("RELION5 export requires all simulated tilt angles to be below 90 degrees")
    native_order = np.argsort(angles, kind="stable")
    frame_rows = acquisition_rows[:, native_order]
    tilt_rotations = Rotation.from_euler("x", angles, degrees=True).as_matrix()
    particle_rotations = rots[acquisition_rows[:, 0]]
    expected_rotations = particle_rotations[:, None] @ tilt_rotations[None]
    if not np.allclose(rots[acquisition_rows], expected_rotations, rtol=0, atol=2e-5):
        raise ValueError("Rotations do not match the standard simulator tilt geometry R0 @ Rx(angle)")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        eulers = R_to_relion(particle_rotations)
    if not np.allclose(R_from_relion(eulers), particle_rotations, rtol=0, atol=2e-5):
        raise ValueError("RECOVAR-to-RELION particle Euler-angle roundtrip failed")

    # RELION refines A_image = P_tilt @ A_particle; the simulator constructs
    # R_image = R_zero @ Rx(angle). Conjugating the tilt rotations is essential
    # because these matrix products do not commute. Each synthetic tomogram
    # therefore has its own tilt-axis orientation, derived from the actual poses.
    projection_rotations = expected_rotations @ particle_rotations[:, None].swapaxes(-1, -2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        projection_angles = (
            Rotation.from_matrix(projection_rotations.reshape(-1, 3, 3))
            .as_euler("xyz", degrees=True)
            .reshape(n_particles, n_tilts, 3)
        )
    rebuilt_projection = (
        Rotation.from_euler("xyz", projection_angles.reshape(-1, 3), degrees=True)
        .as_matrix()
        .reshape(n_particles, n_tilts, 3, 3)
    )
    if not np.allclose(rebuilt_projection @ particle_rotations[:, None], rots[acquisition_rows], rtol=0, atol=2e-5):
        raise ValueError("Native tomography projection composition does not reproduce simulator rotations")

    # Native origin is a 3-D shift in the tomogram coordinate system. Require
    # that its projected x/y components reproduce every simulated pixel shift.
    design = projection_rotations[:, :, :2, :].reshape(n_particles, -1, 3)
    projected_origins = (trans[acquisition_rows] * voxel_size).reshape(n_particles, -1)
    origins = (np.linalg.pinv(design) @ projected_origins[..., None])[..., 0]
    if not np.allclose(
        (design @ origins[..., None])[..., 0],
        projected_origins,
        rtol=0,
        atol=1e-4,
    ):
        raise ValueError("Per-image translations cannot be represented by a native 3-D particle origin")

    expected_dose = (np.arange(n_tilts) + 0.5) * dose_per_tilt
    actual_dose = ctf[acquisition_rows, C.DOSE]
    if np.any(actual_dose < 0) or np.any(np.diff(actual_dose, axis=1) < 0):
        raise ValueError("Actual CTF doses must be nonnegative and nondecreasing in acquisition order")
    dose_matches_schedule = bool(np.allclose(actual_dose, expected_dose[None], rtol=1e-6, atol=1e-6))
    # In legacy simulator versions, an 11-column generator receives two extra
    # appended dose columns, while the forward model still reads columns 9/10.
    # Preserve the actual forward-used dose rather than silently changing images
    # or exporting nominal doses that were never applied to the pixels.
    if "dose_indices" in info and info["dose_indices"] is not None:
        dose_indices = _finite_array(info["dose_indices"], (n_images,), "dose_indices")
        if not np.array_equal(
            dose_indices[acquisition_rows], np.broadcast_to(np.arange(n_tilts), acquisition_rows.shape)
        ):
            raise ValueError("dose_indices do not match the particle acquisition order")
    if np.any(ctf[:, C.VOLT] <= 0) or np.any(ctf[:, C.CS] < 0):
        raise ValueError("Invalid microscope voltage or spherical aberration")
    if np.any(ctf[:, C.W] < 0) or np.any(ctf[:, C.W] > 1):
        raise ValueError(
            "RELION5 requires amplitude contrast in [0, 1]; the RECOVAR no-CTF convention W=-1 is unsupported"
        )
    if np.any(ctf[:, C.DOSE] > 0) and not np.allclose(ctf[:, C.VOLT], 300.0):
        raise ValueError(
            "Native RELION dose weighting requires 300 kV for this export; use RECOVAR format for other voltages"
        )
    optics_values = ctf[:, [C.VOLT, C.CS, C.W]]
    group_optics = optics_values[acquisition_rows[:, 0]]
    if not np.allclose(optics_values[acquisition_rows], group_optics[:, None], rtol=0, atol=1e-7):
        raise ValueError("Microscope optics must be constant within each physical particle")
    optics_unique, optics_ids = np.unique(group_optics, axis=0, return_inverse=True)

    if "forward_ctf_bfactor" in info:
        bfactors = _finite_array(info["forward_ctf_bfactor"], (n_images,), "forward_ctf_bfactor")
        bfactor_source = "simulation_info.forward_ctf_bfactor"
    else:
        warp_encoded = np.allclose(ctf[:, C.BFACTOR], -4 * ctf[:, C.DOSE], rtol=1e-6, atol=1e-6)
        if not warp_encoded and not np.allclose(ctf[:, C.BFACTOR], 0):
            raise ValueError(
                "Supply simulation_info['forward_ctf_bfactor']; the saved WARP B-factor is not a physical envelope"
            )
        bfactors = np.zeros(n_images)
        bfactor_source = "assumed zero; saved WARP dose encoding discarded"
    if np.any(bfactors < 0):
        raise ValueError("Negative forward B-factors are not supported by the RELION5 simulator exporter")

    root = Path(output_folder).absolute()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise FileExistsError(f"Invalid RELION5 output directory: {root}")
    for name in _ARTIFACTS:
        target = root / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite RELION5 export artifact: {target}")
    # Scan images per particle, so validation works without materializing a
    # multi-gigabyte memory-mapped dataset or allocating a full-size Boolean mask.
    for row in acquisition_rows:
        block = np.asarray(images[row])
        if not np.isfinite(block).all() or np.max(np.abs(block)) > np.finfo(dtype).max:
            raise ValueError("Images contain nonfinite values or overflow the requested MRC dtype")

    root.mkdir(parents=True, exist_ok=True)
    (root / "particles").mkdir()
    (root / "tilt_series").mkdir()
    halves = 1 + np.arange(n_particles) % 2
    np.random.default_rng(0).shuffle(halves)
    particle_rows, tomogram_rows = [], []
    for particle_idx, image_rows in enumerate(frame_rows):
        tomo_name = f"sim_{particle_idx:06d}"
        stack_path = root / "particles" / f"particle_{particle_idx:06d}_stack2d.mrcs"
        tilt_path = root / "tilt_series" / f"{tomo_name}.star"
        signed_frames = np.asarray(-np.asarray(images[image_rows], dtype=np.float32), dtype=dtype)
        with mrcfile.new(stack_path, overwrite=False) as handle:
            handle.set_data(signed_frames)
            handle.voxel_size = voxel_size
            handle.set_image_stack()
        with mrcfile.open(stack_path, mode="r") as handle:
            if not np.array_equal(handle.data.reshape(signed_frames.shape), signed_frames):
                raise IOError(f"Signed native stack readback failed: {stack_path}")
        p = {
            "rlnTomoName": tomo_name,
            "rlnTomoParticleName": f"{tomo_name}/particle_{particle_idx:06d}",
            "rlnOpticsGroup": int(optics_ids[particle_idx] + 1),
            "rlnTomoVisibleFrames": "[" + ",".join(["1"] * n_tilts) + "]",
            "rlnImageName": str(stack_path),
            "rlnRandomSubset": int(halves[particle_idx]),
        }
        p.update(zip(("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"), eulers[particle_idx]))
        p.update({f"rlnCenteredCoordinate{axis}Angst": 0.0 for axis in "XYZ"})
        p.update({f"rlnOrigin{axis}Angst": origins[particle_idx, j] for j, axis in enumerate("XYZ")})
        particle_rows.append(p)
        voltage, cs, amplitude = group_optics[particle_idx]
        tomogram_rows.append(
            {
                "rlnTomoName": tomo_name,
                "rlnTomoTiltSeriesStarFile": str(tilt_path),
                "rlnTomoTiltSeriesName": str(stack_path),
                "rlnTomoFrameCount": n_tilts,
                "rlnTomoSizeX": grid_size,
                "rlnTomoSizeY": grid_size,
                "rlnTomoSizeZ": grid_size,
                "rlnTomoHand": 1,
                "rlnTomoTiltSeriesPixelSize": float(voxel_size),
                "rlnMicrographOriginalPixelSize": float(voxel_size),
                "rlnVoltage": voltage,
                "rlnSphericalAberration": cs,
                "rlnAmplitudeContrast": amplitude,
                "rlnTomoImportFractionalDose": float(np.median(np.diff(actual_dose[particle_idx])))
                if n_tilts > 1
                else 0.0,
                "rlnTomoDefocusSlope": 0.0,
            }
        )
        c = ctf[image_rows]
        tilt_table = pd.DataFrame(
            {
                "rlnMicrographName": [f"{j + 1}@{stack_path}" for j in range(n_tilts)],
                "rlnMicrographPreExposure": c[:, C.DOSE],
                "rlnTomoNominalStageTiltAngle": angles[native_order],
                "rlnTomoXTilt": projection_angles[particle_idx, native_order, 0],
                "rlnTomoYTilt": projection_angles[particle_idx, native_order, 1],
                "rlnTomoZRot": projection_angles[particle_idx, native_order, 2],
                "rlnTomoXShiftAngst": 0.0,
                "rlnTomoYShiftAngst": 0.0,
                "rlnDefocusU": c[:, C.DFU],
                "rlnDefocusV": c[:, C.DFV],
                "rlnDefocusAngle": c[:, C.DFANG],
                "rlnPhaseShift": c[:, C.PHASE_SHIFT],
                "rlnCtfScalefactor": c[:, C.CONTRAST] * np.cos(np.deg2rad(c[:, C.TILT_ANGLE])),
                "rlnCtfBfactor": bfactors[image_rows],
            }
        )
        _write_star(tilt_path, {tomo_name: tilt_table})

    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": np.arange(1, len(optics_unique) + 1),
            "rlnOpticsGroupName": [f"opticsGroup{i + 1}" for i in range(len(optics_unique))],
            "rlnVoltage": optics_unique[:, 0],
            "rlnSphericalAberration": optics_unique[:, 1],
            "rlnAmplitudeContrast": optics_unique[:, 2],
            "rlnImagePixelSize": float(voxel_size),
            "rlnImageSize": grid_size,
            "rlnImageDimensionality": 2,
            "rlnTomoTiltSeriesPixelSize": float(voxel_size),
            "rlnTomoSubtomogramBinning": 1.0,
            "rlnCtfDataAreCtfPremultiplied": 0,
        }
    )
    particles_path = root / "particles.star"
    tomograms_path = root / "tomograms.star"
    optimisation_path = root / "optimisation_set.star"
    _write_star(
        particles_path,
        {
            "general": {"rlnTomoSubTomosAre2DStacks": 1},
            "optics": optics,
            "particles": pd.DataFrame(particle_rows),
        },
    )
    _write_star(tomograms_path, {"global": pd.DataFrame(tomogram_rows)})
    _write_star(
        optimisation_path,
        {
            "optimisation_set": {
                "rlnTomoParticlesFile": str(particles_path),
                "rlnTomoTomogramsFile": str(tomograms_path),
            }
        },
    )
    np.save(root / "frame_rows.npy", frame_rows, allow_pickle=False)
    np.save(root / "particle_groups.npy", group_ids, allow_pickle=False)
    dose_note = (
        (
            "At 300 kV RELION's native Grant-Grigorieff dose envelope reproduces "
            "RECOVAR's exponential attenuation, but does not implement RECOVAR's "
            "dose < 2.51284 * critical_exposure frequency cutoff. Exported images "
            "retain that cutoff; format conversion does not change their signal/noise."
        )
        if np.any(actual_dose > 0)
        else (
            "Actual forward-used dose is zero for every image, so native RELION "
            "metadata applies no dose attenuation. The simulator's nominal dose "
            "schedule may differ; format export preserves the original simulation."
        )
    )
    manifest = {
        "schema_version": 1,
        "status": "validated_complete",
        "output_format": "relion5",
        "n_particles": n_particles,
        "n_images": n_images,
        "n_tilts": int(n_tilts),
        "grid_size": grid_size,
        "voxel_size": float(voxel_size),
        "image_dtype": dtype.name,
        "intensity_multiplier": -1,
        "files": {
            "particles": str(particles_path),
            "tomograms": str(tomograms_path),
            "optimisation_set": str(optimisation_path),
            "frame_rows": str(root / "frame_rows.npy"),
            "particle_groups": str(root / "particle_groups.npy"),
        },
        "frame_order": "increasing simulator tilt angle; identical stack and tilt STAR row order",
        "frame_rows_description": "zero-based original image rows for each native particle stack",
        "geometry": "one synthetic tomogram per particle; coordinates at centre; P_tilt = R_zero @ Rx(angle) @ R_zero.T so P_tilt @ R_particle = R_image",
        "halfsets": "balanced per-particle labels shuffled with numpy.default_rng(0)",
        "bfactor_source": bfactor_source,
        "dose_model_difference": dose_note,
        "actual_dose_matches_nominal_schedule": dose_matches_schedule,
        "actual_dose_range": [float(actual_dose.min()), float(actual_dose.max())],
        "nominal_dose_range": [float(expected_dose.min()), float(expected_dose.max())],
        "dose_provenance": "Forward-used ctf_params column 9, not any appended legacy bookkeeping columns",
        "recommended_nr_parts_sigma2noise": int(min(n_particles, 100) * n_tilts),
        "initialisation_note": "Some RELION versions count tilt images for --nr_parts_sigma2noise. Use enough complete particles for every class and inspect initial class maps.",
    }
    with (root / "relion5_export.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    logger.warning(dose_note)
    if not dose_matches_schedule:
        logger.warning(
            "Actual forward-used doses differ from the nominal simulator dose schedule. Export preserved actual doses; see relion5_export.json."
        )
    logger.info("Exported %d native RELION5 particles (%d tilt images): %s", n_particles, n_images, optimisation_path)
    return manifest
