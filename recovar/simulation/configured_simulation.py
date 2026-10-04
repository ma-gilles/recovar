"""Explicit, dataset-independent cryo-ET simulations; legacy defaults are unused.

The JSON configuration specifies every scientific input. Only computational
batch sizes and a uniform mixture distribution have defaults. Existing
``simulator.generate_synthetic_dataset`` is intentionally not called or changed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation

_REQUIRED = {
    "volumes",
    "grid_size",
    "n_particles",
    "n_tilts",
    "angle_per_tilt_deg",
    "dose_per_tilt",
    "seed",
    "signal_scale",
    "poses",
    "noise",
    "ctf",
    "tilt_amplitude_weighting",
}
_OPTIONAL = {
    "volume_weights",
    "volume_assignments_file",
    "batch_size",
    "noise_rng_batch_size",
}
_CTF_FIELDS = (
    "defocus_u",
    "defocus_v",
    "defocus_angle_deg",
    "voltage_kv",
    "cs_mm",
    "amplitude_contrast",
    "phase_shift_deg",
    "bfactor",
)


def _keys(value, required, optional, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{name}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")


def _number(value, name, *, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")
    return value


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _path(value, base, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty file path")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise ValueError(f"{name} does not exist or is not a file: {path}")
    return str(path.resolve())


def _validate_config(config, base):
    _keys(config, _REQUIRED, _OPTIONAL, "simulation config")
    config = copy.deepcopy(config)
    for key, minimum in (("grid_size", 4), ("n_particles", 1), ("n_tilts", 1), ("seed", 0)):
        _integer(config[key], key, minimum)
    if config["grid_size"] % 2:
        raise ValueError("grid_size must be even")
    if config["seed"] > 2**32 - 1:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    for key, section in (("angle_per_tilt_deg", "poses"), ("dose_per_tilt", "ctf")):
        file_mode = isinstance(config[section], dict) and config[section].get("mode") == "per_image_file"
        if file_mode:
            if config[key] is not None:
                raise ValueError(f"{key} must be null when {section}.mode is 'per_image_file'")
        else:
            _number(config[key], key, minimum=0)
    _number(config["signal_scale"], "signal_scale", minimum=0, strict=True)
    if config["tilt_amplitude_weighting"] not in ("none", "cosine"):
        raise ValueError("tilt_amplitude_weighting must be 'none' or 'cosine'")
    if not isinstance(config["volumes"], list) or not config["volumes"]:
        raise ValueError("volumes must be a nonempty list of explicit MRC paths")
    config["volumes"] = [_path(v, base, "volume") for v in config["volumes"]]
    if "volume_assignments_file" in config:
        if "volume_weights" in config:
            raise ValueError("volume_assignments_file and volume_weights are mutually exclusive")
        config["volume_assignments_file"] = _path(config["volume_assignments_file"], base, "volume_assignments_file")
    else:
        weights = config.get("volume_weights", [1 / len(config["volumes"])] * len(config["volumes"]))
        if not isinstance(weights, list) or len(weights) != len(config["volumes"]):
            raise ValueError("volume_weights must have one value per volume")
        for weight in weights:
            _number(weight, "volume_weights", minimum=0)
        if not np.isclose(sum(weights), 1, rtol=0, atol=1e-8):
            raise ValueError("volume_weights must sum to one; values are not silently normalized")
        config["volume_weights"] = weights
    for key, default in (("batch_size", 32), ("noise_rng_batch_size", 128)):
        config.setdefault(key, default)
        _integer(config[key], key, 1)

    poses = config["poses"]
    if not isinstance(poses, dict) or poses.get("mode") not in ("uniform", "file", "per_image_file"):
        raise ValueError("poses.mode must be 'uniform', 'file', or 'per_image_file'")
    pose_fields = {"uniform": set(), "file": {"file"}, "per_image_file": {"rotations_file", "translations_file"}}
    _keys(poses, {"mode"} | pose_fields[poses["mode"]], set(), "poses")
    if poses["mode"] == "file":
        poses["file"] = _path(poses["file"], base, "poses.file")
    elif poses["mode"] == "per_image_file":
        for key in ("rotations_file", "translations_file"):
            poses[key] = _path(poses[key], base, f"poses.{key}")

    noise = config["noise"]
    noise_fields = {
        "none": {"model"},
        "white": {"model", "std"},
        "radial": {"model", "spectrum_file", "variance_scale"},
        "radial_per_tilt": {"model", "spectrum_file", "variance_scale"},
    }
    if not isinstance(noise, dict) or noise.get("model") not in noise_fields:
        raise ValueError("noise.model must be 'none', 'white', 'radial', or 'radial_per_tilt'")
    _keys(noise, noise_fields[noise["model"]], set(), "noise")
    if noise["model"] == "white":
        _number(noise["std"], "noise.std", minimum=0)
    if noise["model"] in ("radial", "radial_per_tilt"):
        _number(noise["variance_scale"], "noise.variance_scale", minimum=0)
        noise["spectrum_file"] = _path(noise["spectrum_file"], base, "noise.spectrum_file")

    ctf = config["ctf"]
    if not isinstance(ctf, dict) or ctf.get("mode") not in ("constant", "sample", "per_image_file"):
        raise ValueError("ctf.mode must be 'constant', 'sample', or 'per_image_file'")
    fields = set(_CTF_FIELDS) if ctf["mode"] == "constant" else {"file"}
    _keys(ctf, {"mode"} | fields, set(), "ctf")
    if ctf["mode"] in ("sample", "per_image_file"):
        ctf["file"] = _path(ctf["file"], base, "ctf.file")
    else:
        for field in _CTF_FIELDS:
            _number(ctf[field], f"ctf.{field}")
    if (ctf["mode"] == "per_image_file" or poses["mode"] == "per_image_file") and config[
        "tilt_amplitude_weighting"
    ] != "none":
        raise ValueError(
            "Per-image metadata replay requires tilt_amplitude_weighting='none'; supplied CTF scales are authoritative"
        )
    return config


def load_simulation_config(path):
    """Read strict JSON; resolve all input paths relative to the JSON file."""
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        config = json.load(handle)
    return _validate_config(config, path.parent)


def _array_file(path, name):
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite numeric .npy array")
    return values


def _inspect_inputs(config, output_format):
    if output_format not in ("recovar", "relion5"):
        raise ValueError("output_format must be 'recovar' or 'relion5'")
    from recovar.simulation.relion5_export import _tilt_angles

    grid_size = config["grid_size"]
    per_image_poses = config["poses"]["mode"] == "per_image_file"
    per_image_ctf = config["ctf"]["mode"] == "per_image_file"
    if output_format == "relion5" and (per_image_poses or per_image_ctf):
        raise ValueError(
            "Per-image metadata replay currently requires RECOVAR output; native RELION5 export requires standard simulator geometry and dose schedule"
        )
    angles = None if per_image_poses else _tilt_angles(config["n_tilts"], config["angle_per_tilt_deg"])
    if angles is not None and np.max(np.abs(angles)) >= 90:
        raise ValueError("Configured tilt angles must be strictly below 90 degrees")
    volume_grid, input_voxel = None, None
    for path in config["volumes"]:
        with mrcfile.open(path, mode="r") as handle:
            data = handle.data
            apix = np.array([handle.voxel_size.x, handle.voxel_size.y, handle.voxel_size.z], dtype=float)
            if data is None or data.ndim != 3 or len(set(data.shape)) != 1 or data.shape[0] % 2:
                raise ValueError(f"Volume must be cubic and even-sized: {path}")
            if not np.isfinite(data).all():
                raise ValueError(f"Volume contains nonfinite values: {path}")
            if not np.isfinite(apix).all() or np.any(apix <= 0) or not np.allclose(apix, apix[0], rtol=1e-6):
                raise ValueError(f"Volume must have a positive, isotropic voxel size: {path}")
            if volume_grid is not None and (
                volume_grid != data.shape[0] or not np.isclose(input_voxel, apix[0], rtol=1e-6)
            ):
                raise ValueError("All input volumes must have the same dimensions and voxel size")
            volume_grid, input_voxel = data.shape[0], float(apix[0])
    if grid_size > volume_grid:
        raise ValueError("grid_size must not exceed the input volume size; upsampling is not implicit")
    particle_assignments = None
    if "volume_assignments_file" in config:
        particle_assignments = _array_file(config["volume_assignments_file"], "volume_assignments_file")
        if particle_assignments.dtype.kind not in "iu":
            raise ValueError("volume_assignments_file must contain integer volume indices")
        if particle_assignments.shape != (config["n_particles"],):
            raise ValueError("volume_assignments_file must have shape (n_particles,)")
        if np.any(particle_assignments < 0) or np.any(particle_assignments >= len(config["volumes"])):
            raise ValueError("volume_assignments_file indices must be in [0, number of volumes)")
        particle_assignments = particle_assignments.astype(np.int64, copy=False)
    poses, translations = None, None
    if config["poses"]["mode"] == "file":
        poses = _array_file(config["poses"]["file"], "poses.file")
        if poses.shape != (config["n_particles"], 3, 3):
            raise ValueError("poses.file must have shape (n_particles, 3, 3), one RECOVAR base rotation per particle")
    elif per_image_poses:
        poses = _array_file(config["poses"]["rotations_file"], "poses.rotations_file")
        translations = _array_file(config["poses"]["translations_file"], "poses.translations_file")
        leading = (config["n_particles"], config["n_tilts"])
        if poses.shape != (*leading, 3, 3):
            raise ValueError("poses.rotations_file must have shape (n_particles, n_tilts, 3, 3)")
        if translations.shape != (*leading, 2):
            raise ValueError("poses.translations_file must have shape (n_particles, n_tilts, 2), in image pixels")
    if poses is not None and (
        not np.allclose(poses @ poses.swapaxes(-1, -2), np.eye(3), atol=2e-5, rtol=0)
        or not np.allclose(np.linalg.det(poses), 1, atol=2e-5, rtol=0)
    ):
        raise ValueError("Pose arrays must contain orthogonal rotations with determinant +1")
    ctf_config = config["ctf"]
    if ctf_config["mode"] == "constant":
        ctf_source = np.array([[ctf_config[key] for key in _CTF_FIELDS] + [1.0]], dtype=float)
    else:
        ctf_source = _array_file(ctf_config["file"], "ctf.file")
    if per_image_ctf:
        if ctf_source.shape != (config["n_particles"], config["n_tilts"], 11):
            raise ValueError("Per-image ctf.file must have shape (n_particles, n_tilts, 11)")
        if np.any(ctf_source[..., 9] < 0) or np.any(np.diff(ctf_source[..., 9], axis=1) < 0):
            raise ValueError("Per-image CTF doses must be nonnegative and nondecreasing in acquisition order")
        if np.any(ctf_source[..., 10] != 0):
            raise ValueError(
                "RECOVAR grouped STAR cannot preserve nonzero per-image CTF TILT_ANGLE; replay requires column 10 to be zero"
            )
        ctf_source = ctf_source.reshape(-1, 11)
    elif ctf_source.ndim != 2 or ctf_source.shape[1] != 9 or len(ctf_source) == 0:
        raise ValueError(
            "ctf.file must have shape (N, 9): DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT, BFACTOR, CONTRAST"
        )
    if np.any(ctf_source[:, 3] <= 0) or np.any(ctf_source[:, 4] < 0) or np.any(ctf_source[:, 7] < 0):
        raise ValueError("CTF voltage must be positive; Cs and physical B-factor must be nonnegative")
    if np.any(ctf_source[:, 5] < 0) or np.any(ctf_source[:, 5] > 1) or np.any(ctf_source[:, 8] < 0):
        raise ValueError("CTF amplitude contrast must be in [0, 1] and scale factors must be nonnegative")
    if not np.allclose(ctf_source[:, [3, 4, 5]], ctf_source[0, [3, 4, 5]], rtol=0, atol=1e-7):
        raise ValueError("Sampled CTF microscope voltage, Cs, and amplitude contrast must be constant")
    if output_format == "recovar" and np.any(ctf_source[:, 7] != 0):
        raise ValueError(
            "Configured RECOVAR grouped STAR cannot preserve a nonzero physical B-factor; "
            "use bfactor=0 or choose RELION5 output. Legacy readers are unchanged."
        )
    has_dose = bool(np.any(ctf_source[:, 9] > 0)) if per_image_ctf else config["dose_per_tilt"] > 0
    if output_format == "relion5" and has_dose and not np.allclose(ctf_source[:, 3], 300.0):
        raise ValueError("RELION5 native dose metadata requires 300 kV; choose RECOVAR for other voltages")
    if has_dose and not (np.allclose(ctf_source[:, 3], 200.0) or np.allclose(ctf_source[:, 3], 300.0)):
        raise ValueError("The implemented dose model supports 200 or 300 kV only")
    n_bins = grid_size // 2 - 1
    noise = config["noise"]
    if noise["model"] == "none":
        spectrum = np.zeros(n_bins)
    elif noise["model"] == "white":
        # make_noise_batch draws N(0, 1)/sqrt(D*D), then filters by sqrt(P).
        # The simulator's 2x padding applies P*4, preserving real-space std.
        spectrum = np.full(n_bins, noise["std"] ** 2 * grid_size**2)
    else:
        spectrum = _array_file(noise["spectrum_file"], "noise.spectrum_file")
        expected_shape = (config["n_tilts"], n_bins) if noise["model"] == "radial_per_tilt" else (n_bins,)
        if spectrum.shape != expected_shape or np.any(spectrum < 0):
            raise ValueError(
                f"noise.spectrum_file must have shape {expected_shape} with nonnegative radial Fourier powers"
            )
        spectrum = spectrum * noise["variance_scale"]
    if not np.isfinite(spectrum).all():
        raise ValueError("Scaled noise power is nonfinite")
    return (
        input_voxel * volume_grid / grid_size,
        volume_grid,
        poses,
        ctf_source,
        spectrum,
        angles,
        translations,
        particle_assignments,
    )


def run_configured_simulation(output_folder, config, output_format="relion5", dry_run=False):
    """Run explicit physics, or return/print resolved configuration without writes.

    Returned images always use RECOVAR's internal sign/acquisition order;
    RELION5 export alone negates and reorders frames. White-noise ``std`` is
    measured in the same real-space intensity units as the simulated images.
    Radial spectra are Fourier powers in the low-level simulator convention.
    """
    from recovar import core, utils
    from recovar.data_io import cryoem_dataset
    from recovar.simulation import simulator
    from recovar.simulation.relion5_export import export_relion5, validate_output_directory

    config = _validate_config(config, Path.cwd())
    root = validate_output_directory(output_folder)
    (
        voxel_size,
        input_grid,
        base_rotations,
        ctf_source,
        spectrum,
        angles,
        file_translations,
        file_assignments,
    ) = _inspect_inputs(config, output_format)
    per_image_poses = config["poses"]["mode"] == "per_image_file"
    per_image_ctf = config["ctf"]["mode"] == "per_image_file"
    per_tilt_noise = config["noise"]["model"] == "radial_per_tilt"
    noise_seeds_per_tilt = None
    if per_tilt_noise:
        noise_seeds_per_tilt = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in np.random.SeedSequence(config["seed"]).spawn(config["n_tilts"])
        ]
    resolved = copy.deepcopy(config)
    resolved["derived"] = {
        "output_format": output_format,
        "voxel_size_angstrom": voxel_size,
        "input_grid_size": input_grid,
        "n_images": config["n_particles"] * config["n_tilts"],
        "tilt_angles_acquisition_order_deg": None if angles is None else angles.tolist(),
        "dose_model": "Grant-Grigorieff exponential with RECOVAR critical-exposure cutoff",
        "dose_source": "ordered CTF file column 9, used verbatim" if per_image_ctf else "midpoint cumulative doses",
        "ctf_convention": "RECOVAR; native RELION5 export negates image intensities",
        "noise_units": "white std in real-space image units; radial spectrum in Fourier power units",
        "volume_scaling": "Fourier crop without norm calibration; multiply only by signal_scale",
        "translations": "ordered per-image file, in image pixels" if per_image_poses else "zero",
        "rotations": "ordered per-image file" if per_image_poses else "base rotation @ dose-symmetric x-tilt",
        "interpolation": "linear_interp",
        "image_dtype": "float32",
        "premultiplied_ctf": False,
        "outliers": False,
        "contrast_jitter": False,
        "mixture_assignment": (
            "ordered zero-based volume index loaded verbatim for each particle"
            if file_assignments is not None
            else "independent categorical draw per particle using volume_weights"
        ),
        "volume_assignment_source": (
            config["volume_assignments_file"] if file_assignments is not None else "volume_weights"
        ),
        "volume_assignment_counts": (
            np.bincount(file_assignments, minlength=len(config["volumes"])).astype(int).tolist()
            if file_assignments is not None
            else None
        ),
        "ctf_sampling": "ordered file rows used verbatim"
        if per_image_ctf
        else "independent row draw per tilt image when mode=sample",
        "noise_rng_seed_scheme": (
            "one NumPy SeedSequence(seed).spawn(n_tilts) child per acquisition rank; "
            "child.generate_state(1, dtype=uint32)[0] seeds the low-level simulator"
        )
        if per_tilt_noise
        else "the explicit config seed seeds the low-level simulator",
        "noise_seeds_per_tilt": noise_seeds_per_tilt,
    }
    print(json.dumps(resolved, indent=2, allow_nan=False))
    if dry_run:
        return resolved

    rng = np.random.default_rng(config["seed"])
    n_particles, n_tilts = config["n_particles"], config["n_tilts"]
    n_images, grid_size = n_particles * n_tilts, config["grid_size"]
    if per_image_poses:
        rotations = base_rotations.reshape(n_images, 3, 3).astype(np.float32)
        translations = file_translations.reshape(n_images, 2).astype(np.float32)
    else:
        if base_rotations is None:
            base_rotations = Rotation.random(n_particles, random_state=rng).as_matrix()
        tilt_rotations = Rotation.from_euler("x", angles, degrees=True).as_matrix()
        rotations = (base_rotations[:, None] @ tilt_rotations[None]).reshape(n_images, 3, 3).astype(np.float32)
        translations = np.zeros((n_images, 2), dtype=np.float32)
    groups = np.repeat(np.arange(n_particles), n_tilts)
    dose_indices = np.tile(np.arange(n_tilts), n_particles)
    assignment_weights = config.get("volume_weights", [1 / len(config["volumes"])] * len(config["volumes"]))
    sampled_assignments = rng.choice(len(config["volumes"]), size=n_particles, p=assignment_weights)
    # Preserve the categorical RNG step so a replay file does not perturb later
    # seeded CTF sampling; the ordered file remains authoritative.
    particle_assignments = sampled_assignments if file_assignments is None else file_assignments.copy()
    assignments = np.repeat(particle_assignments, n_tilts)
    if per_image_ctf:
        ctf_params = ctf_source.astype(np.float32)
    else:
        ctf_params = np.zeros((n_images, 11), dtype=np.float32)
        ctf_params[:, :9] = ctf_source[rng.integers(len(ctf_source), size=n_images)]
        ctf_params[:, 9] = (dose_indices + 0.5) * config["dose_per_tilt"]
        if config["tilt_amplitude_weighting"] == "cosine":
            ctf_params[:, 8] *= np.tile(np.cos(np.deg2rad(angles)), n_particles)
    if not np.isfinite(ctf_params).all():
        raise ValueError("CTF parameters or cumulative doses overflow float32")
    volumes, loaded_voxel = simulator.generate_volumes_from_mrcs(config["volumes"], grid_size)
    if not np.isclose(loaded_voxel, voxel_size, rtol=1e-6):
        raise ValueError("Loaded volume voxel size differs from validated metadata")
    volumes = volumes * config["signal_scale"]
    if not np.isfinite(volumes).all():
        raise ValueError("Scaled Fourier volumes contain nonfinite values")
    dataset = cryoem_dataset.CryoEMDataset(
        None,
        voxel_size,
        cryoem_dataset.ImageMetadata(rotations, translations, ctf_params),
        ctf_evaluator=core.CTFEvaluator(mode=core.CTFMode.CRYO_ET),
        grid_size=grid_size,
    )
    ones = np.ones(n_images, dtype=np.float32)
    if per_tilt_noise:
        images = np.empty((n_images, grid_size, grid_size), dtype=np.float32)
        for rank in range(n_tilts):
            rows = np.arange(rank, n_images, n_tilts)
            rank_dataset = cryoem_dataset.CryoEMDataset(
                None,
                voxel_size,
                cryoem_dataset.ImageMetadata(rotations[rows], translations[rows], ctf_params[rows]),
                ctf_evaluator=core.CTFEvaluator(mode=core.CTFMode.CRYO_ET),
                grid_size=grid_size,
            )
            images[rows] = simulator.simulate_data(
                rank_dataset,
                volumes,
                spectrum[rank],
                config["batch_size"],
                assignments[rows],
                ones[rows],
                ones[rows],
                seed=noise_seeds_per_tilt[rank],
                disc_type="linear_interp",
                premultiplied_ctf=False,
                noise_rng_batch_size=config["noise_rng_batch_size"],
            )
    else:
        images = simulator.simulate_data(
            dataset,
            volumes,
            spectrum,
            config["batch_size"],
            assignments,
            ones,
            ones,
            seed=config["seed"],
            disc_type="linear_interp",
            premultiplied_ctf=False,
            noise_rng_batch_size=config["noise_rng_batch_size"],
        )
    images = np.asarray(images, dtype=np.float32)
    if images.shape != (n_images, grid_size, grid_size) or not np.isfinite(images).all():
        raise ValueError("Simulator returned invalid images; no output was written")
    info = {
        "ctf_params": ctf_params,
        "rots": rotations,
        "trans": translations,
        "per_image_contrast": ones.copy(),
        "per_image_noise_scale": ones.copy(),
        "per_image_offset": np.zeros(n_images, dtype=np.float32),
        "image_assignment": assignments,
        "tilt_series_assignment": particle_assignments,
        "tilt_groups": groups,
        "noise_variance": spectrum,
        "voxel_size": voxel_size,
        "forward_ctf_bfactor": ctf_params[:, 7].copy(),
        "dose_indices": dose_indices,
        "dose_per_tilt": config["dose_per_tilt"],
        "angle_per_tilt": config["angle_per_tilt_deg"],
        "n_tilts": n_tilts,
        "output_format": output_format,
        "simulation_config": resolved,
        "premultiplied_ctf": False,
        "simulation_batch_size": config["batch_size"],
        "simulation_noise_rng_batch_size": config["noise_rng_batch_size"],
        "noise_seeds_per_tilt": noise_seeds_per_tilt,
    }
    # Recheck after computation; never silently replace another run's files.
    validate_output_directory(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "simulation_config.json").open("x") as handle:
        json.dump(resolved, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if output_format == "relion5":
        info["relion5_export"] = export_relion5(
            root,
            images,
            ctf_params,
            rotations,
            translations,
            voxel_size,
            groups,
            n_tilts=n_tilts,
            dose_per_tilt=config["dose_per_tilt"],
            angle_per_tilt=config["angle_per_tilt_deg"],
            image_dtype=np.float32,
            simulation_info=info,
        )
    else:
        stack_name = f"particles.{grid_size}.mrcs"
        with mrcfile.new(root / stack_name, overwrite=False) as handle:
            handle.set_data(images)
            handle.voxel_size = voxel_size
            handle.set_image_stack()
        utils.pickle_dump((rotations, translations), str(root / "poses.pkl"))
        simulator.save_ctf_params(str(root), grid_size, ctf_params, voxel_size)
        star_ctf = ctf_params.copy()
        # WARP-style grouped STAR uses this field for dose bookkeeping, not
        # the physical envelope. Actual forward B remains in simulation_info.
        if not per_image_ctf:
            star_ctf[:, 7] = -4 * star_ctf[:, 9]
        utils.write_starfile(
            star_ctf,
            rotations,
            translations * voxel_size,
            voxel_size,
            grid_size,
            stack_name,
            str(root / "particles.star"),
            halfset_indices=None,
            tilt_groups=groups,
        )
    utils.pickle_dump(info, str(root / "simulation_info.pkl"))
    return images, info
