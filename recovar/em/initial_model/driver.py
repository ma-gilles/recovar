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
from recovar.data_io.starfile import read_star, write_star
from recovar.em import sampling
from recovar.em.dense_single_volume.helpers.orientation_priors import make_relion_translation_log_prior
from recovar.reconstruction.noise import make_radial_noise
from recovar.utils.helpers import write_relion_mrc

from .avg_unaligned import compute_avg_unaligned_and_sigma2
from .bootstrap_iref import compute_bootstrap_iref_via_cpp, postprocess_bootstrap_iref_via_cpp
from .dense_adapter import DenseInitialModelEstepConfig, run_dense_initial_model_estep
from .init import initialise_denovo_state, seed_noise_from_mavg
from .iteration_loop import relion_solvent_flatten_state, relion_solvent_mask, run_vdam_iterations
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
    do_solvent: bool = True
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
    coarse_translations: np.ndarray | None = None
    coarse_prior_translations: np.ndarray | None = None
    translation_parent: np.ndarray | None = None


@dataclass
class NativeParticleState:
    """Per-particle metadata carried between native InitialModel iterations."""

    translation_offsets: np.ndarray
    class_assignments: np.ndarray
    max_posterior: np.ndarray


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


def _image_origin_offsets_pixels_from_star(main_star, dataset) -> np.ndarray:
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
    return shifts.astype(np.float32)


def _image_pre_shifts_from_star(main_star, dataset) -> np.ndarray:
    """Return RELION rounded old-offset image pre-shifts in pixel units.

    Dense EM uses these as the per-image pre-centering base before applying
    the sampled translation grid. RELION rounds old offsets and applies an
    integer zero-filled real-space shift in its accelerated path; pass the
    rounded values so the shared dense engine uses the same fast path.
    """

    return np.rint(_image_origin_offsets_pixels_from_star(main_star, dataset)).astype(np.float32)


def _particle_state_from_star(main_star, dataset) -> NativeParticleState:
    n_images = int(getattr(dataset, "n_images", len(main_star)))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but dataset has {n_images} images")
    class_col = _star_column(main_star, "_rlnClassNumber")
    if class_col is None:
        class_assignments = np.zeros(n_images, dtype=np.int32)
    else:
        class_assignments = np.asarray(class_col.astype(int).to_numpy(), dtype=np.int32) - 1
        if np.any(class_assignments < 0):
            raise ValueError("_rlnClassNumber values must be one-indexed positive class ids")

    pmax_col = _star_column(main_star, "_rlnMaxValueProbDistribution")
    if pmax_col is None:
        max_posterior = np.zeros(n_images, dtype=np.float32)
    else:
        max_posterior = np.asarray(pmax_col.astype(float).to_numpy(), dtype=np.float32)
        if not np.all(np.isfinite(max_posterior)):
            raise ValueError("_rlnMaxValueProbDistribution values must be finite")
    return NativeParticleState(
        translation_offsets=_image_origin_offsets_pixels_from_star(main_star, dataset),
        class_assignments=class_assignments,
        max_posterior=max_posterior,
    )


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
    coarse_pass1_translations = coarse_translations
    if abs(random_perturbation) > 1e-12:
        coarse_pass1_translations = sampling.apply_relion_translation_perturbation(
            coarse_translations.astype(np.float32, copy=False),
            random_perturbation,
            float(opts.offset_step_px),
        ).astype(np.float32)
    if oversampling == 0:
        rotations = sampling.get_relion_hidden_rotation_grid(
            healpix_order,
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
        return NativeSamplingPlan(
            rotations=rotations,
            translations=translations,
            random_perturbation=random_perturbation,
            coarse_translations=coarse_pass1_translations,
            coarse_prior_translations=coarse_translations,
            translation_parent=None,
        )

    coarse_indices = np.arange(sampling.rotation_grid_size(healpix_order), dtype=np.int64)
    rotations, _rotation_parent = sampling.get_oversampled_relion_hidden_rotation_grid_from_samples(
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
        coarse_translations=coarse_pass1_translations,
        coarse_prior_translations=coarse_translations,
        translation_parent=np.asarray(_translation_parent, dtype=np.int64),
    )


def _translation_log_prior(
    translations: np.ndarray,
    *,
    voxel_size: float,
    sigma_angstrom: float | None,
    centers: np.ndarray | None = None,
) -> np.ndarray | None:
    if sigma_angstrom is None:
        return None
    sigma_angstrom = float(sigma_angstrom)
    if sigma_angstrom <= 0.0:
        raise ValueError("translation_sigma_angstrom must be positive when provided")
    translations = np.asarray(translations, dtype=np.float32)
    centers_arr = None
    if centers is not None:
        centers_arr = np.asarray(centers, dtype=np.float32)
        if centers_arr.ndim != 2 or centers_arr.shape[1] != 2:
            raise ValueError(f"translation prior centers must have shape (N, 2), got {centers_arr.shape}")
    return make_relion_translation_log_prior(
        translations[:, :2],
        voxel_size=float(voxel_size),
        sigma_offset_angstrom=sigma_angstrom,
        prior_centers=centers_arr,
    )


def _random_perturbation_for_iteration(opts: NativeInitialModelOptions, iteration: int) -> float:
    env_override = os.environ.get("RECOVAR_RANDOM_PERTURBATION")
    if env_override is not None:
        return float(env_override)
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
    pf = float(perturbation_factor)
    value = 0.0
    n_steps = max(1, int(n_steps))
    for step in range(n_steps + 1):
        # RELION advances once during sampling.initialise() before
        # initialiseWorkLoad() calls init_random_generator(random_seed). C's
        # default rand state is the same as srand(1) on glibc. Then each
        # expectationSetup() reinitializes with random_seed + iter.
        seed = 1 if step == 0 else int(random_seed) + step
        rnd = _relion_rnd_unif_factory(seed)
        value += 0.5 * pf + (pf - 0.5 * pf) * rnd(0)
        while value > pf:
            value -= 2.0 * pf
        while value < -pf:
            value += 2.0 * pf
    return float(value)


def _noise_variance_from_sigma2(sigma2_noise: np.ndarray, ori_size: int) -> np.ndarray:
    # RELION stores normalized 2D FFT shell power. The dense engine scores
    # unnormalized particle FFTs, matching the existing InitialModel fixture.
    n4 = int(ori_size) ** 4
    return (
        np.asarray(make_radial_noise(np.asarray(sigma2_noise)[0] * n4, (ori_size, ori_size)))
        .astype(
            np.float32,
            copy=False,
        )
        .reshape(-1)
    )


def _dense_estep_config(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    sampling_plan: NativeSamplingPlan,
    translation_offsets: np.ndarray,
) -> DenseInitialModelEstepConfig:
    image_pre_shifts = np.rint(np.asarray(translation_offsets, dtype=np.float32)).astype(np.float32)
    coarse_translations = (
        np.asarray(sampling_plan.coarse_translations, dtype=np.float32)
        if sampling_plan.coarse_translations is not None
        else np.asarray(sampling_plan.translations, dtype=np.float32)
    )
    coarse_prior_translations = (
        np.asarray(sampling_plan.coarse_prior_translations, dtype=np.float32)
        if sampling_plan.coarse_prior_translations is not None
        else coarse_translations
    )
    # RELION InitialModel default: _rlnSigmaOffsetsAngst = 10.0 Å (set at
    # iter000 in run_it000_model.star, used for the Gaussian translation
    # prior in iter-1 E-step). When `opts.translation_sigma_angstrom` is
    # unset, default to 10.0 to match RELION's iter-1 prior. Without this
    # the prior is None (uniform) and posteriors diverge from RELION.
    sigma_angstrom = opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    coarse_translation_log_prior = _translation_log_prior(
        coarse_prior_translations,
        voxel_size=float(dataset.voxel_size),
        sigma_angstrom=sigma_angstrom,
        centers=-image_pre_shifts,
    )
    translation_log_prior = _translation_log_prior(
        sampling_plan.translations,
        voxel_size=float(dataset.voxel_size),
        sigma_angstrom=sigma_angstrom,
        centers=-image_pre_shifts,
    )

    engine_kwargs = {
        "score_with_masked_images": True,
        "reconstruct_with_masked_images": False,
        # RELION's --grad mode (ml_optimiser.cpp:10092-10105) accumulates
        # Fimg_store = Fimg_shift_nomask - Frefctf into BPref. Without this
        # flag the engine accumulates Fimg_shift_nomask directly, matching
        # standard EM but NOT VDAM. Production was missing this; fixing
        # lifts bp_data CC from +0.91 to +0.996 (per BPref test fixture).
        "reconstruction_subtract_projected_reference": True,
        "relion_firstiter_score_mode": "gaussian",
        "image_pre_shifts": np.asarray(image_pre_shifts, dtype=np.float32),
        "sparse_pass2": int(opts.oversampling) > 0,
    }
    if int(opts.oversampling) > 0:
        engine_kwargs.update(
            {
                "healpix_order": int(opts.healpix_order),
                "oversampling_order": int(opts.oversampling),
                "translation_step": float(opts.offset_step_px),
                "random_perturbation": float(sampling_plan.random_perturbation),
                "coarse_translations": coarse_translations,
                "particle_diameter_ang": float(opts.particle_diameter),
                "return_profile": True,
            }
        )
        # Allow env-override of adaptive_fraction for K-class parity tuning
        _af_env = os.environ.get("RECOVAR_ADAPTIVE_FRACTION")
        if _af_env:
            engine_kwargs["adaptive_fraction"] = float(_af_env)
    if os.environ.get("RECOVAR_USE_FLOAT64_SCORING"):
        engine_kwargs["use_float64_scoring"] = True
    if os.environ.get("RECOVAR_HALF_SPECTRUM_SCORING"):
        engine_kwargs["half_spectrum_scoring"] = True
    if os.environ.get("RECOVAR_SQUARE_WINDOW"):
        engine_kwargs["square_window"] = True
    # Independent RECON window override: SCORE uses square_window, RECON uses
    # recon_square_window. Set this to True to backproject all pixels of the
    # current_size FFTW half (matching RELION's set2DFourierTransform sweep
    # without the radial r<=r_max cutoff). Default behavior follows
    # square_window via dense_adapter group_kwargs.
    _recon_sq = os.environ.get("RECOVAR_RECON_SQUARE_WINDOW")
    if _recon_sq is not None:
        engine_kwargs["recon_square_window"] = bool(int(_recon_sq))
    # Disable VDAM gradient subtraction (Fimg_store = Fimg - Frefctf). Setting
    # RECOVAR_DISABLE_SUBTRACT_PROJECTED_REFERENCE=1 makes the M-step accumulate
    # raw posterior-weighted images (standard EM), bypassing the VDAM gradient
    # form. Useful for diagnosing whether the Frefctf subtraction is the c2
    # parity bottleneck (the proj_weighted=-N²×Fref scaling makes the subtraction
    # add instead of subtract — see m_step.py and dense_adapter.py).
    if os.environ.get("RECOVAR_DISABLE_SUBTRACT_PROJECTED_REFERENCE"):
        engine_kwargs["reconstruction_subtract_projected_reference"] = False
    if translation_log_prior is not None:
        engine_kwargs["translation_log_prior"] = translation_log_prior
    if coarse_translation_log_prior is not None:
        engine_kwargs["coarse_translation_log_prior"] = coarse_translation_log_prior

    return DenseInitialModelEstepConfig(
        noise_variance=noise_variance,
        rotations=sampling_plan.rotations,
        translations=sampling_plan.translations,
        image_batch_size=int(opts.image_batch_size),
        rotation_block_size=int(opts.rotation_block_size),
        padding_factor=int(opts.padding_factor),
        relion_bpref_frame=True,
        relion_projector_frame=True,
        engine_kwargs=engine_kwargs,
    )


def _native_expectation_step(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    particle_state: NativeParticleState | np.ndarray,
):
    if not isinstance(particle_state, NativeParticleState):
        particle_state = NativeParticleState(
            translation_offsets=np.asarray(particle_state, dtype=np.float32).copy(),
            class_assignments=np.zeros(int(dataset.n_images), dtype=np.int32),
            max_posterior=np.zeros(int(dataset.n_images), dtype=np.float32),
        )

    def _expectation_step(state: InitialModelState, particle_ids: np.ndarray, halfset_ids: np.ndarray):
        iteration = max(1, int(state.iter))
        sampling_plan = _build_sampling_plan(opts, iteration=iteration)
        config = _dense_estep_config(dataset, opts, noise_variance, sampling_plan, particle_state.translation_offsets)
        config.engine_kwargs["debug_iteration"] = iteration
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
        _update_particle_state_from_estep_meta(
            particle_state,
            result.meta,
            sampling_plan.translations,
        )
        return result.accumulators, result.meta

    return _expectation_step


def _update_translation_offsets_from_estep_meta(
    translation_offsets: np.ndarray,
    meta: dict,
    translations: np.ndarray,
) -> None:
    particle_state = NativeParticleState(
        translation_offsets=translation_offsets,
        class_assignments=np.zeros(translation_offsets.shape[0], dtype=np.int32),
        max_posterior=np.zeros(translation_offsets.shape[0], dtype=np.float32),
    )
    _update_particle_state_from_estep_meta(particle_state, meta, translations)


def _update_particle_state_from_estep_meta(
    particle_state: NativeParticleState,
    meta: dict,
    translations: np.ndarray,
) -> None:
    selected_particle_ids = meta.get("selected_particle_ids")
    if selected_particle_ids is None:
        return

    particle_ids = np.asarray(selected_particle_ids, dtype=np.int64).reshape(-1)
    if particle_ids.size == 0:
        return
    if np.any(particle_ids < 0) or np.any(particle_ids >= particle_state.translation_offsets.shape[0]):
        raise ValueError("selected_particle_ids contains entries outside the particle state table")

    pose_assignments = meta.get("pose_assignments")
    if pose_assignments is not None:
        assignments = np.asarray(pose_assignments, dtype=np.int64).reshape(-1)
        translations = np.asarray(translations, dtype=np.float32)
        if particle_ids.shape != assignments.shape:
            raise ValueError(
                "selected_particle_ids and pose_assignments must have matching shape, "
                f"got {particle_ids.shape} and {assignments.shape}"
            )
        if translations.ndim != 2 or translations.shape[1] < 2:
            raise ValueError(f"translations must have shape (T, >=2), got {translations.shape}")
        n_trans = int(translations.shape[0])
        translation_ids = np.mod(assignments, n_trans)
        base = np.rint(particle_state.translation_offsets[particle_ids]).astype(np.float32)
        particle_state.translation_offsets[particle_ids] = base + translations[translation_ids, :2]

    class_assignments = meta.get("class_assignments")
    if class_assignments is not None:
        classes = np.asarray(class_assignments, dtype=np.int32).reshape(-1)
        if particle_ids.shape != classes.shape:
            raise ValueError(
                "selected_particle_ids and class_assignments must have matching shape, "
                f"got {particle_ids.shape} and {classes.shape}"
            )
        particle_state.class_assignments[particle_ids] = classes

    max_posterior = meta.get("max_posterior_per_image")
    if max_posterior is not None:
        pmax = np.asarray(max_posterior, dtype=np.float32).reshape(-1)
        if particle_ids.shape != pmax.shape:
            raise ValueError(
                "selected_particle_ids and max_posterior_per_image must have matching shape, "
                f"got {particle_ids.shape} and {pmax.shape}"
            )
        particle_state.max_posterior[particle_ids] = pmax


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
    # RECOVAR_INITIAL_IREF_OVERRIDE lets a parity caller swap in RELION's
    # iter000 ref directly when isolating E/M-step behavior from bootstrap.
    override_path = os.environ.get("RECOVAR_INITIAL_IREF_OVERRIDE")
    if override_path:
        from recovar.utils.helpers import load_relion_volume

        # Comma-separated paths for K-class parity (e.g.
        # "run_it000_class001.mrc,run_it000_class002.mrc"). When a single
        # path is provided, broadcast to all classes (back-compat).
        # If the path contains the literal "{k}" or "{k:03d}" template,
        # expand for k=1..K.
        paths = [p.strip() for p in override_path.split(",") if p.strip()]
        if len(paths) == 1 and ("{k" in paths[0]):
            template = paths[0]
            paths = [template.format(k=k + 1) for k in range(int(opts.nr_classes))]
        K = int(opts.nr_classes)
        if len(paths) == 1:
            loaded = np.asarray(load_relion_volume(paths[0]), dtype=np.float64)
            if loaded.shape != (ori_size, ori_size, ori_size):
                raise ValueError(
                    f"RECOVAR_INITIAL_IREF_OVERRIDE volume shape {loaded.shape} != ({ori_size}, {ori_size}, {ori_size})"
                )
            state.Iref = np.broadcast_to(loaded, (K, ori_size, ori_size, ori_size)).copy()
        elif len(paths) == K:
            stacked = np.stack(
                [np.asarray(load_relion_volume(p), dtype=np.float64) for p in paths],
                axis=0,
            )
            if stacked.shape != (K, ori_size, ori_size, ori_size):
                raise ValueError(
                    f"RECOVAR_INITIAL_IREF_OVERRIDE stacked shape {stacked.shape} != "
                    f"({K}, {ori_size}, {ori_size}, {ori_size})"
                )
            state.Iref = stacked
        else:
            raise ValueError(
                f"RECOVAR_INITIAL_IREF_OVERRIDE expects 1 or K={K} comma-separated paths, got {len(paths)}"
            )
    else:
        state.Iref = postprocess_bootstrap_iref_via_cpp(
            iref,
            pixel_size=pixel_size,
            ini_high_ang=float(state.ini_high),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
            do_init_blobs=True,
            is_helical_segment=False,
        )
    return state, optics_group_by_particle


def _class_mrc_paths(output_prefix: str, iteration: int, K: int) -> tuple[str, ...]:
    return tuple(f"{output_prefix}_it{iteration:03d}_class{k + 1:03d}.mrc" for k in range(K))


def _write_model_star(path: str, state: InitialModelState, class_mrcs: tuple[str, ...]) -> None:
    with open(path, "w") as f:
        f.write("# Created by recovar native InitialModel\n\n")
        f.write("data_model_general\n\n")
        f.write(f"_rlnCurrentResolution {float(state.current_resolution):.12g}\n")
        f.write(f"_rlnCurrentImageSize {int(state.current_size)}\n")
        f.write(f"_rlnCurrentIteration {int(state.iter)}\n")
        f.write(f"_rlnNrClasses {int(state.K)}\n")
        f.write(f"_rlnTau2FudgeFactor {float(state.tau2_fudge_factor):.12g}\n")
        f.write(f"_rlnAveragePmax {float(state.ave_Pmax):.12g}\n\n")

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


def _set_star_column(table, column: str, values) -> None:
    target = column
    no_prefix = column[1:] if column.startswith("_") else column
    if target not in table.columns and no_prefix in table.columns:
        target = no_prefix
    table[target] = values


def _format_float_column(values: np.ndarray, precision: int = 6) -> list[str]:
    return [f"{float(value):.{precision}f}" for value in np.asarray(values).reshape(-1)]


def _write_data_star(path: str, main_star, optics_star, dataset, particle_state: NativeParticleState) -> None:
    n_images = int(getattr(dataset, "n_images", len(main_star)))
    if len(main_star) != n_images:
        raise ValueError(f"STAR table has {len(main_star)} particles but dataset has {n_images} images")
    if particle_state.translation_offsets.shape != (n_images, 2):
        raise ValueError(
            f"translation_offsets must have shape ({n_images}, 2), got {particle_state.translation_offsets.shape}"
        )
    if particle_state.class_assignments.shape != (n_images,):
        raise ValueError(
            f"class_assignments must have shape ({n_images},), got {particle_state.class_assignments.shape}"
        )
    if particle_state.max_posterior.shape != (n_images,):
        raise ValueError(f"max_posterior must have shape ({n_images},), got {particle_state.max_posterior.shape}")

    table = main_star.copy()
    offsets_angstrom = np.asarray(particle_state.translation_offsets, dtype=np.float64) * float(dataset.voxel_size)
    _set_star_column(table, "_rlnOriginXAngst", _format_float_column(offsets_angstrom[:, 0]))
    _set_star_column(table, "_rlnOriginYAngst", _format_float_column(offsets_angstrom[:, 1]))
    has_legacy_x = _star_column(table, "_rlnOriginX") is not None
    has_legacy_y = _star_column(table, "_rlnOriginY") is not None
    if has_legacy_x or has_legacy_y:
        offsets_pixels = np.asarray(particle_state.translation_offsets, dtype=np.float64)
        _set_star_column(table, "_rlnOriginX", _format_float_column(offsets_pixels[:, 0]))
        _set_star_column(table, "_rlnOriginY", _format_float_column(offsets_pixels[:, 1]))
    _set_star_column(table, "_rlnClassNumber", (np.asarray(particle_state.class_assignments, dtype=np.int32) + 1))
    _set_star_column(table, "_rlnMaxValueProbDistribution", _format_float_column(particle_state.max_posterior))

    out_path = Path(path)
    if str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    write_star(str(out_path), table, optics_star.copy() if optics_star is not None else None)


def _write_iteration_artifacts(
    output_prefix: str,
    state: InitialModelState,
    iteration: int,
    meta: dict,
    *,
    main_star=None,
    optics_star=None,
    dataset=None,
    particle_state: NativeParticleState | None = None,
) -> None:
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
    if main_star is not None and dataset is not None and particle_state is not None:
        _write_data_star(
            f"{output_prefix}_it{iteration:03d}_data.star",
            main_star,
            optics_star,
            dataset,
            particle_state,
        )


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
    if opts.padding_factor not in (1, 2):
        raise NotImplementedError("native InitialModel currently supports RELION GUI --pad 1 or 2 only")
    if opts.run_relion_align_symmetry:
        raise NotImplementedError("native post-run relion_align_symmetry execution is not wired yet")

    main_star, optics_star = read_star(opts.fn_img)
    particle_order = _micrograph_sort_order(main_star)
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
    particle_state = _particle_state_from_star(main_star, dataset)
    state, optics_group_by_particle = _initial_state_from_particles(
        dataset,
        main_star,
        optics_star,
        opts,
        sampling_plan.rotations,
    )
    noise_variance = _noise_variance_from_sigma2(state.sigma2_noise, int(state.ori_size))
    expectation_step = _native_expectation_step(dataset, opts, noise_variance, particle_state)
    grad_ini_subset_size, grad_fin_subset_size = default_subset_sizes_for_3d_initial_model(int(dataset.n_images))

    if opts.write_iter_artifacts:
        _output_dir_from_prefix(opts.outputname).mkdir(parents=True, exist_ok=True)
        config_path = f"{opts.outputname}_native_options.json"
        with open(config_path, "w") as f:
            json.dump(asdict(opts), f, indent=2, sort_keys=True)

    artifact_sink = (
        (
            lambda current, iteration, meta: _write_iteration_artifacts(
                opts.outputname,
                current,
                iteration,
                meta,
                main_star=main_star,
                optics_star=optics_star,
                dataset=dataset,
                particle_state=particle_state,
            )
        )
        if opts.write_iter_artifacts
        else (lambda *args, **kwargs: None)
    )
    post_mstep_update = None
    if opts.do_solvent:
        solvent_mask = relion_solvent_mask(
            ori_size=int(state.ori_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
        )
        post_mstep_update = lambda current, _iteration, _meta: relion_solvent_flatten_state(
            current,
            mask=solvent_mask,
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
        post_mstep_update=post_mstep_update,
        particle_order=particle_order,
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
