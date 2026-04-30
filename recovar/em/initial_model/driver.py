"""Native InitialModel / ab-initio K-class driver.

This module owns the executable recovar path behind ``scripts/run_ab_initio``.
The script remains a thin argparse and RELION-command-snapshot layer; all
data loading, denovo seeding, dense K-class E-step wiring, VDAM iteration, and
artifact writing lives here so InitialModel does not grow a second EM stack.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from recovar.core import mask as core_mask
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.data_io.starfile import read_star
from recovar.em import sampling
from recovar.reconstruction.noise import make_radial_noise
from recovar.utils.helpers import write_relion_mrc

from .avg_unaligned import compute_avg_unaligned_and_sigma2
from .bootstrap_iref import compute_bootstrap_iref_via_cpp, initial_low_pass_filter_references
from .dense_adapter import DenseInitialModelEstepConfig, run_dense_initial_model_estep
from .init import initialise_denovo_state, seed_noise_from_mavg
from .iteration_loop import run_vdam_iterations
from .schedules import DEFAULT_GRAD_EM_ITERS, DEFAULT_GRAD_MU, default_subset_sizes_for_3d_initial_model
from .state import InitialModelState
from .subset import RndUnifFn


DEFAULT_WIDTH_MASK_EDGE_PX = 5.0
DEFAULT_HEALPIX_ORDER = 1
DEFAULT_OFFSET_RANGE_PX = 6.0
DEFAULT_OFFSET_STEP_PX = 2.0
DEFAULT_RANDOM_SEED = 0
DEFAULT_OVERSAMPLING = 1
DEFAULT_PERTURBATION_FACTOR = 0.5


@dataclass(frozen=True, kw_only=True)
class NativeInitialModelOptions:
    """Options for a native InitialModel run.

    Defaults mirror the GUI InitialModel command where the dense path already
    has a native implementation. Unsupported RELION branches fail loudly rather
    than silently diverging.
    """

    fn_img: str
    outputname: str = "ab_initio/run"
    nr_iter: int = 200
    nr_classes: int = 1
    tau2_fudge: float = 4.0
    sym_name: str = "C1"
    do_run_C1: bool = True
    particle_diameter: float = 200.0
    do_zero_mask: bool = True
    do_ctf_correction: bool = True
    random_seed: int = DEFAULT_RANDOM_SEED
    width_mask_edge_px: float = DEFAULT_WIDTH_MASK_EDGE_PX
    healpix_order: int = DEFAULT_HEALPIX_ORDER
    oversampling: int = DEFAULT_OVERSAMPLING
    perturbation_factor: float = DEFAULT_PERTURBATION_FACTOR
    random_perturbation: float | None = None
    offset_range_px: float = DEFAULT_OFFSET_RANGE_PX
    offset_step_px: float = DEFAULT_OFFSET_STEP_PX
    image_batch_size: int = 500
    rotation_block_size: int = 5000
    bootstrap_min_particles: int = 1000
    sigma2_min_particles: int = 1000
    padding_factor: int = 1
    lazy: bool = True
    datadir: str | None = None
    strip_prefix: str | None = None
    translation_sigma_angstrom: float | None = None
    write_iter_artifacts: bool = True
    run_relion_align_symmetry: bool = False


@dataclass(frozen=True)
class NativeInitialModelResult:
    """Summary returned by ``run_native_initial_model``."""

    state: InitialModelState
    output_prefix: str
    final_model_star: str
    final_mrc: str
    class_mrcs: tuple[str, ...]


@dataclass(frozen=True)
class NativeSamplingPlan:
    """Dense trial grid used by one native InitialModel E-step."""

    rotations: np.ndarray
    translations: np.ndarray
    random_perturbation: float


def _relion_rnd_unif_factory(seed: int) -> RndUnifFn:
    """Return a RELION ``rnd_unif`` source using the local C++ binding."""

    from recovar.relion_bind import _relion_bind_core as bind

    cache = np.asarray(bind.vdam_rnd_unif_sequence(int(seed), 1024), dtype=np.float64)

    def _rnd(call_idx: int) -> float:
        nonlocal cache
        if call_idx >= cache.size:
            new_size = max(call_idx + 1, cache.size * 2)
            cache = np.asarray(bind.vdam_rnd_unif_sequence(int(seed), int(new_size)), dtype=np.float64)
        return float(cache[call_idx])

    return _rnd


def _output_dir_from_prefix(outputname: str) -> Path:
    parent = Path(outputname).parent
    return Path(".") if str(parent) == "" else parent


def _initial_model_mrc_from_prefix(outputname: str) -> str:
    """Mirror RELION's GUI ``outputname.rstrip("run") + initial_model.mrc``."""

    return outputname.rstrip("run") + "initial_model.mrc"


def _micrograph_sort_order(main_star) -> np.ndarray:
    """RELION reads InitialModel particles in stable micrograph-name order."""

    n = len(main_star)
    if "_rlnMicrographName" not in main_star.columns:
        return np.arange(n, dtype=np.int64)
    names = np.asarray(main_star["_rlnMicrographName"].astype(str).to_numpy())
    return np.argsort(names, kind="stable").astype(np.int64, copy=False)


def _optics_group_indices(main_star) -> np.ndarray:
    if "_rlnOpticsGroup" not in main_star.columns:
        return np.zeros(len(main_star), dtype=np.int64)
    raw = main_star["_rlnOpticsGroup"].to_numpy()
    try:
        numeric = np.asarray(raw, dtype=np.int64)
        unique = {value: i for i, value in enumerate(sorted(np.unique(numeric).tolist()))}
        return np.asarray([unique[int(value)] for value in numeric], dtype=np.int64)
    except (TypeError, ValueError):
        labels = np.asarray(raw, dtype=str)
        unique = {value: i for i, value in enumerate(sorted(np.unique(labels).tolist()))}
        return np.asarray([unique[str(value)] for value in labels], dtype=np.int64)


def _single_optics_scalars(main_star, optics_star, ds) -> tuple[float, float, float, float]:
    """Return voltage, Cs, amplitude contrast, and pixel size.

    The current C++ bootstrap binding takes scalar optics parameters. To avoid
    wrong native output, reject genuinely multi-optics inputs until the binding
    grows per-particle voltage/Cs/Q0 support.
    """

    pixel_size = float(ds.voxel_size)
    if optics_star is None:
        required = ("_rlnVoltage", "_rlnSphericalAberration", "_rlnAmplitudeContrast")
        missing = [name for name in required if name not in main_star.columns]
        if missing:
            raise ValueError(
                "native InitialModel needs voltage/Cs/amplitude contrast in the STAR file; "
                f"missing {', '.join(missing)}"
            )
        values = tuple(float(main_star[name].astype(float).iloc[0]) for name in required)
        return values[0], values[1], values[2], pixel_size

    groups = _optics_group_indices(main_star)
    if np.unique(groups).size != 1 or len(optics_star) != 1:
        raise NotImplementedError(
            "native InitialModel bootstrap currently supports one optics group; "
            "multi-optics support needs per-particle optics in the RELION bootstrap binding"
        )
    row = optics_star.iloc[0]
    return (
        float(row["_rlnVoltage"]),
        float(row["_rlnSphericalAberration"]),
        float(row["_rlnAmplitudeContrast"]),
        pixel_size,
    )


def _phase_shift(main_star) -> np.ndarray:
    if "_rlnPhaseShift" not in main_star.columns:
        return np.zeros(len(main_star), dtype=np.float64)
    return np.asarray(main_star["_rlnPhaseShift"].astype(float).to_numpy(), dtype=np.float64)


def _star_column(main_star, name: str):
    if name in main_star.columns:
        return main_star[name]
    no_prefix = name[1:] if name.startswith("_") else name
    if no_prefix in main_star.columns:
        return main_star[no_prefix]
    return None


def _image_pre_shifts_from_star(main_star, dataset) -> np.ndarray:
    """Return RELION old-offset image pre-shifts in pixel units.

    Dense EM uses these as the per-image pre-centering base before applying
    the sampled translation grid. RELION rounds old offsets and applies an
    integer zero-filled real-space shift in its accelerated path; pass the
    rounded values so the shared dense engine uses the same fast path.
    """

    n_images = int(len(main_star))
    origin_x_ang = _star_column(main_star, "_rlnOriginXAngst")
    origin_y_ang = _star_column(main_star, "_rlnOriginYAngst")
    if (origin_x_ang is None) != (origin_y_ang is None):
        raise ValueError("STAR file must provide both _rlnOriginXAngst and _rlnOriginYAngst")
    if origin_x_ang is not None and origin_y_ang is not None:
        pixel_size = float(dataset.voxel_size)
        if pixel_size <= 0.0:
            raise ValueError("dataset voxel_size must be positive to convert STAR origins from Angstroms")
        shifts = np.stack(
            [
                np.asarray(origin_x_ang.astype(float).to_numpy(), dtype=np.float64) / pixel_size,
                np.asarray(origin_y_ang.astype(float).to_numpy(), dtype=np.float64) / pixel_size,
            ],
            axis=1,
        )
    else:
        origin_x_px = _star_column(main_star, "_rlnOriginX")
        origin_y_px = _star_column(main_star, "_rlnOriginY")
        if (origin_x_px is None) != (origin_y_px is None):
            raise ValueError("STAR file must provide both _rlnOriginX and _rlnOriginY")
        if origin_x_px is None or origin_y_px is None:
            return np.zeros((n_images, 2), dtype=np.float32)
        shifts = np.stack(
            [
                np.asarray(origin_x_px.astype(float).to_numpy(), dtype=np.float64),
                np.asarray(origin_y_px.astype(float).to_numpy(), dtype=np.float64),
            ],
            axis=1,
        )

    if shifts.shape != (n_images, 2):
        raise ValueError(f"STAR origin shifts must have shape ({n_images}, 2), got {shifts.shape}")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("STAR origin shifts must be finite")
    return np.rint(shifts).astype(np.float32)


def _load_raw_images(dataset, image_indices: np.ndarray, *, batch_size: int) -> np.ndarray:
    """Load raw real-space particle images through ``CryoEMDataset`` I/O."""

    images: list[np.ndarray] = []
    for batch_images, _particle_indices, _local_indices in dataset.image_source.iter_batches(
        batch_size=batch_size,
        batch_mode="images",
        subset_indices=np.asarray(image_indices, dtype=np.int64),
    ):
        images.append(np.asarray(batch_images))
    if not images:
        return np.empty((0, dataset.grid_size, dataset.grid_size), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(images, axis=0))


def _image_sigma2_iter(
    dataset,
    image_indices: np.ndarray,
    optics_group_by_particle: np.ndarray,
    *,
    batch_size: int,
) -> Iterable[tuple[int, np.ndarray]]:
    for batch_images, _particle_indices, local_indices in dataset.image_source.iter_batches(
        batch_size=batch_size,
        batch_mode="images",
        subset_indices=np.asarray(image_indices, dtype=np.int64),
    ):
        batch_images = np.asarray(batch_images)
        local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        for image, local_idx in zip(batch_images, local_indices):
            yield int(optics_group_by_particle[int(local_idx)]), image


def _configure_relion_image_mask(dataset, opts: NativeInitialModelOptions) -> None:
    """Configure dataset preprocessing to match InitialModel scoring masks."""

    source = dataset.image_source
    backend = getattr(source, "backend", source)
    if backend is None:
        return
    image_mask = core_mask.relion_soft_image_mask(
        int(dataset.grid_size),
        float(dataset.voxel_size),
        float(opts.particle_diameter),
        float(opts.width_mask_edge_px),
    )
    if hasattr(backend, "image_mask"):
        backend.image_mask = image_mask
    if hasattr(backend, "mask"):
        backend.mask = image_mask
    if hasattr(backend, "image_mask_mode"):
        backend.image_mask_mode = "relion_background_fill"


def _build_sampling_plan(opts: NativeInitialModelOptions, *, iteration: int = 1) -> NativeSamplingPlan:
    healpix_order = int(opts.healpix_order)
    oversampling = int(opts.oversampling)
    if oversampling < 0:
        raise ValueError("oversampling must be >= 0")

    random_perturbation = _random_perturbation_for_iteration(opts, iteration)

    coarse_translations = sampling.get_translation_grid(
        max_pixel=float(opts.offset_range_px),
        pixel_offset=float(opts.offset_step_px),
    ).astype(np.float32)
    if oversampling == 0:
        rotations = sampling.get_rotation_grid(
            nside_level=healpix_order,
            n_in_planes=sampling.rotation_grid_n_in_planes(healpix_order),
            matrices=True,
        ).astype(np.float32)
        translations = coarse_translations
        if abs(random_perturbation) > 1e-12:
            rotations = sampling.apply_relion_rotation_perturbation(
                rotations,
                random_perturbation,
                sampling.relion_angular_sampling_deg(healpix_order),
            ).astype(np.float32)
            translations = sampling.apply_relion_translation_perturbation(
                translations,
                random_perturbation,
                float(opts.offset_step_px),
            ).astype(np.float32)
        return NativeSamplingPlan(rotations=rotations, translations=translations, random_perturbation=random_perturbation)

    coarse_indices = np.arange(sampling.rotation_grid_size(healpix_order), dtype=np.int64)
    rotations, _rotation_parent = sampling.get_oversampled_rotation_grid_from_samples(
        coarse_indices,
        parent_nside_level=healpix_order,
        oversampling_order=oversampling,
        random_perturbation=random_perturbation,
    )
    translations, _translation_parent = sampling.get_oversampled_translation_grid(
        coarse_translations,
        pixel_offset=float(opts.offset_step_px),
        oversampling_order=oversampling,
    )
    translations = sampling.apply_relion_translation_perturbation(
        translations.astype(np.float32, copy=False),
        random_perturbation,
        offset_step_pixels=float(opts.offset_step_px),
    )
    return NativeSamplingPlan(
        rotations=np.asarray(rotations, dtype=np.float32),
        translations=np.asarray(translations, dtype=np.float32),
        random_perturbation=random_perturbation,
    )


def _translation_log_prior(
    translations: np.ndarray,
    *,
    voxel_size: float,
    sigma_angstrom: float | None,
) -> np.ndarray | None:
    if sigma_angstrom is None:
        return None
    sigma_angstrom = float(sigma_angstrom)
    if sigma_angstrom <= 0.0:
        raise ValueError("translation_sigma_angstrom must be positive when provided")
    translations = np.asarray(translations, dtype=np.float32)
    dist2_angstrom = np.sum(translations[:, :2] ** 2, axis=1) * float(voxel_size) ** 2
    return (-0.5 * dist2_angstrom / (sigma_angstrom**2)).astype(np.float32)


def _random_perturbation_for_iteration(opts: NativeInitialModelOptions, iteration: int) -> float:
    if opts.random_perturbation is not None:
        return float(opts.random_perturbation)
    return _random_perturbation_sequence(
        int(opts.random_seed),
        float(opts.perturbation_factor),
        max(1, int(iteration)),
    )


def _random_perturbation_sequence(random_seed: int, perturbation_factor: float, n_steps: int) -> float:
    if perturbation_factor <= 0.0:
        return 0.0
    rnd = _relion_rnd_unif_factory(int(random_seed))
    pf = float(perturbation_factor)
    value = 0.0
    for step in range(max(1, int(n_steps))):
        value += 0.5 * pf + (pf - 0.5 * pf) * rnd(step)
        while value > pf:
            value -= 2.0 * pf
        while value < -pf:
            value += 2.0 * pf
    return float(value)


def _noise_variance_from_sigma2(sigma2_noise: np.ndarray, ori_size: int) -> np.ndarray:
    # RELION stores normalized 2D FFT shell power. The dense engine scores
    # unnormalized particle FFTs, matching the existing InitialModel fixture.
    n4 = int(ori_size) ** 4
    return np.asarray(make_radial_noise(np.asarray(sigma2_noise)[0] * n4, (ori_size, ori_size))).astype(
        np.float32,
        copy=False,
    ).reshape(-1)


def _dense_estep_config(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    sampling_plan: NativeSamplingPlan,
    image_pre_shifts: np.ndarray,
) -> DenseInitialModelEstepConfig:
    translation_log_prior = _translation_log_prior(
        sampling_plan.translations,
        voxel_size=float(dataset.voxel_size),
        sigma_angstrom=opts.translation_sigma_angstrom,
    )

    engine_kwargs = {
        "score_with_masked_images": True,
        "relion_firstiter_score_mode": "gaussian",
        "image_pre_shifts": np.asarray(image_pre_shifts, dtype=np.float32),
        "sparse_pass2": False,
    }
    if translation_log_prior is not None:
        engine_kwargs["translation_log_prior"] = translation_log_prior

    return DenseInitialModelEstepConfig(
        noise_variance=noise_variance,
        rotations=sampling_plan.rotations,
        translations=sampling_plan.translations,
        image_batch_size=int(opts.image_batch_size),
        rotation_block_size=int(opts.rotation_block_size),
        padding_factor=int(opts.padding_factor),
        relion_bpref_frame=True,
        engine_kwargs=engine_kwargs,
    )


def _native_expectation_step(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    image_pre_shifts: np.ndarray,
):
    def _expectation_step(state: InitialModelState, particle_ids: np.ndarray, halfset_ids: np.ndarray):
        sampling_plan = _build_sampling_plan(opts, iteration=max(1, int(state.iter)))
        config = _dense_estep_config(dataset, opts, noise_variance, sampling_plan, image_pre_shifts)
        result = run_dense_initial_model_estep(
            dataset,
            state,
            config,
            particle_ids=particle_ids,
            halfset_ids=halfset_ids,
        )
        result.meta["random_perturbation"] = float(sampling_plan.random_perturbation)
        result.meta["n_rotations"] = int(sampling_plan.rotations.shape[0])
        result.meta["n_translations"] = int(sampling_plan.translations.shape[0])
        return result.accumulators, result.meta

    return _expectation_step


def _initial_state_from_particles(
    dataset,
    main_star,
    optics_star,
    opts: NativeInitialModelOptions,
    rotations: np.ndarray,
) -> tuple[InitialModelState, np.ndarray]:
    ori_size = int(dataset.grid_size)
    pixel_size = float(dataset.voxel_size)
    order = _micrograph_sort_order(main_star)
    optics_group_by_particle = _optics_group_indices(main_star)
    nr_optics_groups = int(np.unique(optics_group_by_particle).size)
    if nr_optics_groups != 1:
        raise NotImplementedError("native InitialModel currently supports one optics group")

    Mavg, sigma2_per_group = compute_avg_unaligned_and_sigma2(
        _image_sigma2_iter(
            dataset,
            order,
            optics_group_by_particle,
            batch_size=max(1, int(opts.image_batch_size)),
        ),
        ori_size=ori_size,
        pixel_size=pixel_size,
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=int(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        nr_optics_groups=nr_optics_groups,
        minimum_nr_particles=int(opts.sigma2_min_particles),
    )

    bootstrap_count = min(len(order), int(opts.bootstrap_min_particles))
    bootstrap_order = order[:bootstrap_count]
    images = _load_raw_images(dataset, bootstrap_order, batch_size=max(1, int(opts.image_batch_size)))
    sorted_star = main_star.iloc[bootstrap_order]
    voltage, Cs, Q0, pixel_size = _single_optics_scalars(sorted_star, optics_star, dataset)

    iref = compute_bootstrap_iref_via_cpp(
        images=images,
        defU=np.asarray(sorted_star["_rlnDefocusU"].astype(float).to_numpy(), dtype=np.float64),
        defV=np.asarray(sorted_star["_rlnDefocusV"].astype(float).to_numpy(), dtype=np.float64),
        defAngle=np.asarray(sorted_star["_rlnDefocusAngle"].astype(float).to_numpy(), dtype=np.float64),
        phase_shift=_phase_shift(sorted_star),
        voltage=voltage,
        Cs=Cs,
        Q0=Q0,
        pixel_size=pixel_size,
        ori_size=ori_size,
        nr_classes=int(opts.nr_classes),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
        do_zero_mask=bool(opts.do_zero_mask),
        do_ctf_correction=bool(opts.do_ctf_correction),
        random_seed=int(opts.random_seed),
        padding_factor=int(opts.padding_factor),
        current_size=-1,
        minimum_nr_particles=int(opts.bootstrap_min_particles),
    )

    state = initialise_denovo_state(
        ori_size=ori_size,
        pixel_size=pixel_size,
        K=int(opts.nr_classes),
        nr_iter=int(opts.nr_iter),
        n_directions=int(rotations.shape[0]),
        nr_optics_groups=nr_optics_groups,
        pseudo_halfsets=True,
        padding_factor=int(opts.padding_factor),
    )
    state = seed_noise_from_mavg(state, sigma2_per_group)
    state.Mavg = Mavg
    state.Iref = initial_low_pass_filter_references(
        iref,
        ori_size=ori_size,
        pixel_size=pixel_size,
        ini_high_ang=float(state.ini_high),
    )
    return state, optics_group_by_particle


def _class_mrc_paths(output_prefix: str, iteration: int, K: int) -> tuple[str, ...]:
    return tuple(f"{output_prefix}_it{iteration:03d}_class{k + 1:03d}.mrc" for k in range(K))


def _write_model_star(path: str, state: InitialModelState, class_mrcs: tuple[str, ...]) -> None:
    with open(path, "w") as f:
        f.write("# Created by recovar native InitialModel\n\n")
        f.write("data_model_classes\n\n")
        f.write("loop_\n")
        f.write("_rlnReferenceImage #1\n")
        f.write("_rlnClassDistribution #2\n")
        f.write("_rlnEstimatedResolution #3\n")
        for class_mrc, probability in zip(class_mrcs, np.asarray(state.pdf_class)):
            f.write(f"{class_mrc} {float(probability):.12g} 0\n")

        f.write("\n\ndata_model_optics_group_1\n\n")
        f.write("loop_\n")
        f.write("_rlnSpectralIndex #1\n")
        f.write("_rlnResolution #2\n")
        f.write("_rlnSigma2Noise #3\n")
        for shell, sigma2 in enumerate(np.asarray(state.sigma2_noise)[0]):
            f.write(f"{int(shell)} 0 {float(sigma2):.12g}\n")


def _write_iteration_artifacts(output_prefix: str, state: InitialModelState, iteration: int, meta: dict) -> None:
    out_dir = _output_dir_from_prefix(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    for k, class_mrc in enumerate(class_mrcs):
        write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    model_star = f"{output_prefix}_it{iteration:03d}_model.star"
    _write_model_star(model_star, state, class_mrcs)
    meta_path = f"{output_prefix}_it{iteration:03d}_recovar_meta.json"
    with open(meta_path, "w") as f:
        json.dump(_json_ready(meta), f, indent=2, sort_keys=True)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_final_outputs(output_prefix: str, state: InitialModelState) -> tuple[str, tuple[str, ...]]:
    iteration = int(state.iter)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    out_dir = _output_dir_from_prefix(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, class_mrc in enumerate(class_mrcs):
        if not os.path.exists(class_mrc):
            write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    final_mrc = _initial_model_mrc_from_prefix(output_prefix)
    best_class = int(np.argmax(np.asarray(state.pdf_class)))
    write_relion_mrc(final_mrc, np.asarray(state.Iref[best_class]), voxel_size=float(state.pixel_size))
    return final_mrc, class_mrcs


def run_native_initial_model(opts: NativeInitialModelOptions) -> NativeInitialModelResult:
    """Run native recovar InitialModel refinement."""

    if opts.nr_classes < 1:
        raise ValueError("nr_classes must be >= 1")
    if opts.nr_iter < 1:
        raise ValueError("nr_iter must be >= 1")
    if opts.padding_factor != 1:
        raise NotImplementedError("native InitialModel currently supports RELION GUI --pad 1 only")
    if opts.run_relion_align_symmetry:
        raise NotImplementedError("native post-run relion_align_symmetry execution is not wired yet")

    main_star, optics_star = read_star(opts.fn_img)
    dataset = load_dataset(
        opts.fn_img,
        lazy=bool(opts.lazy),
        datadir=opts.datadir,
        strip_prefix=opts.strip_prefix,
    )
    if getattr(dataset, "tilt_series_flag", False):
        raise NotImplementedError("native InitialModel currently supports SPA particle STAR files, not tilt-series")

    _configure_relion_image_mask(dataset, opts)
    sampling_plan = _build_sampling_plan(opts, iteration=1)
    image_pre_shifts = _image_pre_shifts_from_star(main_star, dataset)
    state, optics_group_by_particle = _initial_state_from_particles(
        dataset,
        main_star,
        optics_star,
        opts,
        sampling_plan.rotations,
    )
    noise_variance = _noise_variance_from_sigma2(state.sigma2_noise, int(state.ori_size))
    expectation_step = _native_expectation_step(dataset, opts, noise_variance, image_pre_shifts)
    grad_ini_subset_size, grad_fin_subset_size = default_subset_sizes_for_3d_initial_model(int(dataset.n_images))

    if opts.write_iter_artifacts:
        _output_dir_from_prefix(opts.outputname).mkdir(parents=True, exist_ok=True)
        config_path = f"{opts.outputname}_native_options.json"
        with open(config_path, "w") as f:
            json.dump(asdict(opts), f, indent=2, sort_keys=True)

    artifact_sink = (
        (lambda current, iteration, meta: _write_iteration_artifacts(opts.outputname, current, iteration, meta))
        if opts.write_iter_artifacts
        else (lambda *args, **kwargs: None)
    )

    final_state = run_vdam_iterations(
        state,
        nr_particles=int(dataset.n_images),
        optics_group_by_particle=optics_group_by_particle,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        tau2_fudge_arg=float(opts.tau2_fudge),
        grad_em_iters=DEFAULT_GRAD_EM_ITERS,
        random_seed=int(opts.random_seed),
        rnd_unif_factory=_relion_rnd_unif_factory,
        expectation_step=expectation_step,
        iter_artifact_sink=artifact_sink,
        mu=DEFAULT_GRAD_MU,
    )
    final_mrc, class_mrcs = _write_final_outputs(opts.outputname, final_state)
    final_model_star = f"{opts.outputname}_it{final_state.iter:03d}_model.star"
    if not os.path.exists(final_model_star):
        _write_model_star(final_model_star, final_state, class_mrcs)
    return NativeInitialModelResult(
        state=final_state,
        output_prefix=opts.outputname,
        final_model_star=final_model_star,
        final_mrc=final_mrc,
        class_mrcs=class_mrcs,
    )


__all__ = [
    "NativeInitialModelOptions",
    "NativeInitialModelResult",
    "run_native_initial_model",
]
