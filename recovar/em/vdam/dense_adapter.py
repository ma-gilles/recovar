"""InitialModel E-step adapter on the dense K-class engine.

The hidden variable axis is ``class x pose``; pseudo-halfsets share one E-step
(reconstruction accumulators are split per halfset) so projection/scoring isn't
duplicated while the VDAM M-step still gets independent halfset BackProjectors.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from recovar.em.classification.k_class import run_dense_k_class_em
from recovar.em.helpers.orientation_priors import (
    relion_round_away_from_zero,
    relion_sigma_offset_prior_center,
    relion_translation_prior_center,
)
from recovar.em.relion import relion_projector_setup
from recovar.em.relion.relion_projector_setup import ProjectorSetupBackend
from recovar.em.vdam import native_sampling
from recovar.em.vdam.estep_common import (
    _PARTICLE_RESULT_FIELDS,
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
    _add_accumulator_weight_meta,
    _arrays_to_accumulators,
    _empty_accumulator,
    _estep_meta,
    _group_local_kwargs,
    _relion_projector_dense_rotations,
)
from recovar.em.vdam.native_options import NativeInitialModelOptions
from recovar.em.vdam.native_sampling import NativeSamplingPlan
from recovar.em.vdam.sparse_pass2_estep import _run_sparse_pass2_initial_model_estep
from recovar.em.vdam.state import InitialModelState, VdamAccumulator
from recovar.reconstruction.noise import make_radial_noise
from recovar.utils.helpers import get_gpu_memory_total

INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE = 256
INITIAL_MODEL_LOCAL_BATCH_REFERENCE_COUNT_40GB = 32

_INACTIVE_CLASS_LOG_PRIOR = -1.0e30
_EXACT_RELION_PROJECTOR_ENV = "RECOVAR_INITIAL_MODEL_EXACT_RELION_PROJECTOR"
_RELION_PROJECTOR_DUMP_DIR_ENV = "RECOVAR_INITIAL_MODEL_PROJECTOR_DUMP_DIR"


logger = logging.getLogger(__name__)


@dataclass
class _IterationProjectorContext:
    """One refresh-to-E-step handoff; never a cache across iterations."""

    projector_setup_backend: Literal["native", "jax"] = "native"
    prepared: tuple | None = None
    reference: np.ndarray | None = None
    geometry: tuple | None = None

    def refresh(self, state, *, padding_factor, interpolator):
        # Clear even if construction fails, so stale data cannot survive a retry.
        self.prepared = self.reference = self.geometry = None
        inputs, power = prepare_relion_projector_class_inputs_and_power(
            state, padding_factor=padding_factor, interpolator=interpolator,
            projector_setup_backend=self.projector_setup_backend,
        )
        self.prepared = inputs
        self.reference = state.Iref
        self.geometry = (
            int(state.iter), int(state.ori_size), int(state.current_size),
            int(state.K), int(padding_factor), int(interpolator),
        )
        return replace(state, tau2_class=power)

    def take(self, state, *, padding_factor, interpolator=1):
        if self.prepared is None:
            return None  # No refresh callback: preserve standalone/disabled behavior.
        inputs, reference, geometry = self.prepared, self.reference, self.geometry
        self.prepared = self.reference = self.geometry = None
        expected = (
            int(state.iter), int(state.ori_size), int(state.current_size),
            int(state.K), int(padding_factor), int(interpolator),
        )
        if reference is not state.Iref or geometry != expected:
            raise ValueError("projector refresh/E-step reference or geometry changed")
        return inputs


def _configure_relion_image_mask(dataset, opts: NativeInitialModelOptions) -> None:
    """Configure dataset preprocessing to match InitialModel scoring masks."""

    backend = dataset.image_source.backend
    backend.set_relion_image_mask(
        pixel_size=float(dataset.voxel_size),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
    )
    backend.set_relion_fourier_backend(opts.image_fourier_backend)


def _noise_variance_from_sigma2(sigma2_noise: np.ndarray, ori_size: int) -> np.ndarray:
    """Convert RELION normalized shell power to engine-frame radial noise (unnormalised FFT)."""
    n4 = int(ori_size) ** 4
    # Keep RELION's RFLOAT shell spectrum through the reciprocal used by the
    # guarded exact coarse path.  The downstream float32 kernels already cast
    # their ordinary operands explicitly; narrowing here first loses up to a
    # few ULP in Minvsigma2 and changes near-threshold candidate weights.
    return np.asarray(
        make_radial_noise(
            np.asarray(sigma2_noise, dtype=np.float64)[0] * n4,
            (ori_size, ori_size),
        ),
        dtype=np.float64,
    ).reshape(-1)


def _effective_initial_model_image_batch_size(
    requested: int,
    *,
    grid_size: int,
    gpu_memory_gb: float,
) -> int:
    """Conservatively cap exact-local batches for large InitialModel grids.

    Exact fine search has a transient that scales approximately with
    ``batch * grid_size**2`` in addition to its resident projector/cache
    state.  The user-facing batch remains an upper bound; 128-pixel jobs keep
    their established behavior, while 256+ grids scale from 32 images on a
    40 GB accelerator.
    """

    if requested < 1:
        raise ValueError(f"image_batch_size must be positive, got {requested}")
    if grid_size < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if gpu_memory_gb <= 0:
        raise ValueError(f"gpu_memory_gb must be positive, got {gpu_memory_gb}")
    if grid_size < INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE:
        return int(requested)
    scaled_cap = int(
        INITIAL_MODEL_LOCAL_BATCH_REFERENCE_COUNT_40GB
        * (INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE / float(grid_size)) ** 2
        * (float(gpu_memory_gb) / 40.0)
    )
    return min(int(requested), max(1, scaled_cap))


def _dense_estep_config(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    sampling_plan: NativeSamplingPlan,
    translation_offsets: np.ndarray,
    sigma_offset_angstrom: float,
    class_log_priors: np.ndarray,
    pass1_healpix_order: int,
) -> DenseInitialModelEstepConfig:
    image_pre_shifts = relion_round_away_from_zero(translation_offsets)
    coarse_translations = np.asarray(
        sampling_plan.coarse_translations
        if sampling_plan.coarse_translations is not None
        else sampling_plan.translations,
        dtype=np.float32,
    )
    coarse_prior_translations = np.asarray(
        sampling_plan.coarse_prior_translations
        if sampling_plan.coarse_prior_translations is not None
        else coarse_translations,
        dtype=np.float32,
    )
    sigma_angstrom = float(sigma_offset_angstrom)
    # InitialModel uses the same accelerated ``pdf_offset`` convention as the
    # supplied-map EM path: the sampling grid is represented in projection
    # pixels, while RELION applies its source-faithful pixel_size**4 scale.
    translation_prior_centers = relion_translation_prior_center(
        translation_offsets,
        float(dataset.voxel_size),
    )
    _prior_kwargs = dict(
        voxel_size=float(dataset.voxel_size),
        sigma_angstrom=sigma_angstrom,
        centers=translation_prior_centers,
    )
    coarse_translation_log_prior = native_sampling._translation_log_prior(coarse_prior_translations, **_prior_kwargs)
    translation_log_prior = native_sampling._translation_log_prior(sampling_plan.translations, **_prior_kwargs)

    sparse_pass2_enabled = os.environ.get("RECOVAR_DISABLE_SPARSE_PASS2", "") not in (
        "1",
        "true",
        "TRUE",
    )
    if sampling_plan.rotations is None and not sparse_pass2_enabled:
        raise ValueError("Deferred fine rotations require sparse pass 2")
    engine_kwargs: dict = {
        "score_with_masked_images": True,
        "reconstruct_with_masked_images": False,
        # VDAM --grad subtracts Frefctf (ml_optimiser.cpp:10092-10105); lifts BPref CC +0.91→+0.996.
        "reconstruction_subtract_projected_reference": True,
        "relion_firstiter_score_mode": "gaussian",
        "image_pre_shifts": image_pre_shifts,
        "translation_prior_centers": relion_sigma_offset_prior_center(translation_offsets),
        # RECOVAR_DISABLE_SPARSE_PASS2=1 forces dense path (cuFFT plan OOM at 256²+).
        # Oversampling zero is still RELION's adaptive two-pass algorithm: its
        # fine children are the coarse samples themselves.  Keep it on the
        # same exact significance/local route as positive oversampling instead
        # of falling back to RECOVAR's algebraic dense engine.
        "sparse_pass2": sparse_pass2_enabled,
    }
    if sparse_pass2_enabled or int(sampling_plan.oversampling) > 0:
        engine_kwargs.update(
            healpix_order=int(sampling_plan.healpix_order),
            oversampling_order=int(sampling_plan.oversampling),
            translation_step=float(sampling_plan.offset_step_px),
            random_perturbation=float(sampling_plan.random_perturbation),
            coarse_translations=coarse_translations,
            particle_diameter_ang=float(opts.particle_diameter),
            pass1_healpix_order=int(pass1_healpix_order),
            return_profile=bool(os.environ.get("RECOVAR_INITIAL_MODEL_PROFILE")),
        )
        if _af := os.environ.get("RECOVAR_ADAPTIVE_FRACTION"):
            engine_kwargs["adaptive_fraction"] = float(_af)
    for env_var, kwarg in (
        ("RECOVAR_USE_FLOAT64_SCORING", "use_float64_scoring"),
        ("RECOVAR_HALF_SPECTRUM_SCORING", "half_spectrum_scoring"),
        ("RECOVAR_SQUARE_WINDOW", "square_window"),
    ):
        if os.environ.get(env_var):
            engine_kwargs[kwarg] = True
    if (_recon_sq := os.environ.get("RECOVAR_RECON_SQUARE_WINDOW")) is not None:
        engine_kwargs["recon_square_window"] = bool(int(_recon_sq))
    if os.environ.get("RECOVAR_DISABLE_SUBTRACT_PROJECTED_REFERENCE"):
        engine_kwargs["reconstruction_subtract_projected_reference"] = False
    if translation_log_prior is not None:
        engine_kwargs["translation_log_prior"] = translation_log_prior
    if coarse_translation_log_prior is not None:
        engine_kwargs["coarse_translation_log_prior"] = coarse_translation_log_prior

    grid_size = int(dataset.image_shape[0])
    gpu_memory_gb = (
        float(get_gpu_memory_total())
        if grid_size >= INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE
        else 40.0
    )
    effective_image_batch_size = _effective_initial_model_image_batch_size(
        int(opts.image_batch_size),
        grid_size=grid_size,
        gpu_memory_gb=gpu_memory_gb,
    )
    return DenseInitialModelEstepConfig(
        noise_variance=noise_variance,
        rotations=sampling_plan.rotations,
        translations=sampling_plan.translations,
        image_batch_size=effective_image_batch_size,
        rotation_block_size=int(opts.rotation_block_size),
        pass2_engine=str(opts.pass2_engine),
        relion_wavg_sequential_cuda=bool(opts.relion_wavg_sequential_cuda),
        exact_local_bucket_radix=int(opts.exact_local_bucket_radix),
        exact_local_physical_order_chunk_size=int(
            opts.exact_local_physical_order_chunk_size
        ),
        stable_fourier_window_shapes=bool(opts.stable_fourier_window_shapes),
        padding_factor=int(opts.padding_factor),
        projector_setup_backend=opts.projector_setup_backend,
        relion_bpref_frame=True,
        relion_projector_frame=True,
        class_log_priors=class_log_priors,
        engine_kwargs=engine_kwargs,
    )


def class_log_priors_from_state(state: InitialModelState) -> np.ndarray:
    """Log class priors from ``state.pdf_class`` (collapsed classes get a finite sentinel)."""
    weights = np.asarray(state.pdf_class, dtype=np.float64)
    if weights.shape != (state.K,):
        raise ValueError(f"state.pdf_class must have shape ({state.K},), got {weights.shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("state.pdf_class must contain non-negative finite class probabilities")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("state.pdf_class must contain at least one positive class probability")
    out = np.full(state.K, _INACTIVE_CLASS_LOG_PRIOR, dtype=np.float64)
    positive = weights > 0.0
    out[positive] = np.log(weights[positive] / total)
    return out


def _image_groups(
    particle_ids: np.ndarray | None,
    halfset_ids: np.ndarray | None,
    *,
    n_images: int,
    pseudo_halfsets: bool,
) -> list[tuple[int, np.ndarray]]:
    ids = np.arange(n_images, dtype=np.int64) if particle_ids is None else np.asarray(particle_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"particle_ids must be 1D, got {ids.shape}")
    if np.any(ids < 0) or np.any(ids >= n_images):
        raise ValueError("particle_ids contains entries outside the dataset")

    if not pseudo_halfsets:
        return [(0, ids)]

    if halfset_ids is None:
        h0, h1 = ids[0::2], ids[1::2]
    else:
        halves = np.asarray(halfset_ids, dtype=np.int8)
        if halves.shape != ids.shape:
            raise ValueError(f"halfset_ids shape {halves.shape} must match particle_ids shape {ids.shape}")
        if np.any((halves != 0) & (halves != 1)):
            raise ValueError("halfset_ids must contain only 0/1 values")
        h0, h1 = ids[halves == 0], ids[halves == 1]
    return [(0, h0), (1, h1)]


def _dense_engine_kwargs(state: InitialModelState, config: DenseInitialModelEstepConfig) -> dict[str, Any]:
    engine_kwargs = {
        "current_size": None if state.current_size <= 0 else state.current_size,
        "projection_padding_factor": config.padding_factor,
        "reconstruction_padding_factor": config.padding_factor,
        "half_spectrum_scoring": True,
        "score_with_masked_images": True,
        "reconstruct_with_masked_images": True,
        "sparse_pass2": False,
        # RELION InitialModel BPref uses the rounded radial reconstruction support
        # encoded by Minvsigma2, not the full square Fourier crop.
        "recon_square_window": False,
        "recon_exact_radius": False,
        "reconstruction_subtract_projected_reference": True,
        # RELION InitialModel scores the full rounded Fourier crop emitted by its
        # CUDA projector, including the few crop-corner pixels outside r_max.
        "projection_mask_current_image_disk": False,
    }
    engine_kwargs.update(config.engine_kwargs)

    controlled = ("image_indices", "reconstruction_group_ids", "reconstruction_group_count")
    present = sorted(name for name in controlled if name in config.engine_kwargs)
    if present:
        raise ValueError(f"InitialModel dense E-step controls these dense-engine arguments: {', '.join(present)}")
    if engine_kwargs["projection_padding_factor"] != engine_kwargs["reconstruction_padding_factor"]:
        raise ValueError("InitialModel dense E-step requires matching projection/reconstruction padding factors")
    return engine_kwargs


_DENSE_RUN_EM_REJECT = frozenset(
    {
        # InitialModel-only kwargs the dense run_em wrapper doesn't accept.
        "reconstruct_with_masked_images",
        "recon_square_window",
        "recon_exact_radius",
        "reconstruction_subtract_projected_reference",
        "projection_mask_current_image_disk",
        "relion_projector_shape",
        # Sparse/local engine kwargs that run_dense_k_class_em rejects.
        "return_profile",
        "return_best_pose_details",
        "return_stats",
        "disable_adjoint_y",
        "disable_adjoint_ctf",
        "normalization_log_evidence",
    }
)


def _dense_run_em_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs unsupported by the dense run_em wrapper (sparse-only escape hatch)."""
    return {k: v for k, v in kwargs.items() if k not in _DENSE_RUN_EM_REJECT}


def _relion_projector_to_dense_volume(projector_data: np.ndarray, ori_size: int) -> np.ndarray:
    """Embed cropped RELION ``Projector::data`` half-complex slab into dense full-centered Fourier cube.

    Out-of-range coordinates are clipped (any-size input accepted; high frequencies truncated).
    """
    import recovar.core.fourier_transform_utils as ftu

    ppref = np.asarray(projector_data, dtype=np.complex128)
    if ppref.ndim != 3:
        raise ValueError(f"projector_data must be 3D, got {ppref.shape}")
    n = int(ori_size)
    center = n // 2
    half = np.zeros((n, n, center + 1), dtype=np.complex128)
    slab = ppref[::-1, :, :]
    source_slices, target_slices = [], []
    for slab_size in slab.shape[:2]:
        offset = center - slab_size // 2
        start, stop = max(0, offset), min(n, offset + slab_size)
        source_slices.append(slice(start - offset, stop - offset))
        target_slices.append(slice(start, stop))
    x_slice = (slice(0, min(slab.shape[2], center + 1)),)
    half[tuple(target_slices) + x_slice] = slab[tuple(source_slices) + x_slice]
    return np.asarray(ftu.half_volume_to_full_volume(half, (n, n, n)), dtype=np.complex128)


def relion_projector_half_maps_to_dense_means(projector_half_maps: np.ndarray, ori_size: int) -> np.ndarray:
    """Embed RELION ``Projector::data`` maps into dense recovar scoring volumes."""

    n = int(ori_size)
    means = []
    for projector_data in np.asarray(projector_half_maps):
        dense = _relion_projector_to_dense_volume(np.asarray(projector_data), n)
        # RECOVAR_DENSE_MEANS_SCALE diag override (see project_k2_c2_cc_root_cause_2026_05_03).
        tok = (os.environ.get("RECOVAR_DENSE_MEANS_SCALE") or "-N2").strip()
        scale = {"-N2": -(n**2), "N2": float(n**2)}.get(tok)
        if scale is None:
            scale = float(tok)
        means.append(dense.reshape(-1) * scale)
    return np.asarray(means, dtype=np.complex64)


def _dense_rotations_for_config(rotations: Any, config: DenseInitialModelEstepConfig) -> np.ndarray:
    rotations_np = np.asarray(rotations, dtype=np.float32)
    if not config.relion_projector_frame:
        return rotations_np
    return _relion_projector_dense_rotations(rotations_np)


# (attr_name_on_result, dtype) for fields harvested per halfset and concatenated.


def reference_to_dense_means(references: np.ndarray) -> np.ndarray:
    """Convert recovar-frame InitialModel references to unnormalised centered FFTs for dense scoring."""
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.reconstruction.relion_functions import griddingCorrect

    refs = np.asarray(references)
    if refs.ndim != 4:
        raise ValueError(f"references must have shape (K, N, N, N), got {refs.shape}")
    n = int(refs.shape[-1])
    means = []
    for ref in refs:
        corrected, _ = griddingCorrect(jnp.asarray(ref), n, padding_factor=1, order=1)
        means.append(np.asarray(ftu.get_dft3(corrected).reshape(-1)))
    return np.asarray(means, dtype=np.complex64)


def prepare_relion_projector_class_inputs(
    state: InitialModelState,
    *,
    padding_factor: int,
    projector_setup_backend: ProjectorSetupBackend = "native",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build InitialModel's production RELION projector once per iteration."""
    projector_half_by_class, projector_r_max = relion_projector_setup.reference_to_relion_projector_half_maps(
        state.Iref,
        current_size=state.current_size if state.current_size > 0 else state.ori_size,
        padding_factor=padding_factor,
        projector_setup_backend=projector_setup_backend,
    )
    return _finish_relion_projector_class_inputs(
        state, padding_factor, projector_half_by_class, projector_r_max
    )


def prepare_relion_projector_class_inputs_and_power(
    state: InitialModelState,
    *,
    padding_factor: int,
    projector_setup_backend: ProjectorSetupBackend = "native",
    interpolator: int = 1,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
    """Produce scoring operands and tau2 from the identical corrected FFT."""
    half_maps, power, r_max = relion_projector_setup.reference_to_relion_projector_half_maps_and_power(
        state.Iref,
        current_size=state.current_size if state.current_size > 0 else state.ori_size,
        padding_factor=padding_factor,
        projector_setup_backend=projector_setup_backend,
        interpolator=interpolator,
    )
    inputs = _finish_relion_projector_class_inputs(state, padding_factor, half_maps, r_max)
    return inputs, power


def _finish_relion_projector_class_inputs(
    state: InitialModelState,
    padding_factor: int,
    projector_half_by_class: np.ndarray,
    projector_r_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    projector_dump_dir = os.environ.get(_RELION_PROJECTOR_DUMP_DIR_ENV, "").strip()
    if projector_dump_dir:
        os.makedirs(projector_dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(projector_dump_dir, f"iter{int(state.iter):03d}_relion_projector_half.npz"),
            projector_half=np.asarray(projector_half_by_class),
            projector_r_max=np.int64(projector_r_max),
            current_size=np.int64(
                state.current_size if state.current_size > 0 else state.ori_size
            ),
            padding_factor=np.int64(padding_factor),
            iteration=np.int64(state.iter),
        )
    means = relion_projector_half_maps_to_dense_means(
        projector_half_by_class,
        int(state.ori_size),
    )
    mean_variance = np.abs(np.asarray(means)) ** 2
    return means, mean_variance, projector_half_by_class, int(projector_r_max)


def _resolve_class_inputs(
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
) -> tuple[Any, Any, np.ndarray | None, int | None]:
    mean_variance = config.mean_variance
    relion_projector_half_by_class = None
    relion_projector_r_max = None
    if config.relion_projector_half_by_class is not None:
        if config.relion_projector_r_max is None:
            raise ValueError(
                "relion_projector_r_max is required with relion_projector_half_by_class"
            )
        relion_projector_half_by_class = np.asarray(config.relion_projector_half_by_class)
        relion_projector_r_max = int(config.relion_projector_r_max)
        means = (
            config.means
            if config.means is not None
            else relion_projector_half_maps_to_dense_means(
                relion_projector_half_by_class,
                int(state.ori_size),
            )
        )
    elif config.means is not None:
        means = config.means
    elif config.relion_projector_frame:
        means, prepared_variance, projector_half_by_class, projector_r_max = (
            prepare_relion_projector_class_inputs(
                state,
                padding_factor=config.padding_factor,
                projector_setup_backend=config.projector_setup_backend,
            )
        )
        if mean_variance is None:
            mean_variance = prepared_variance
        exact_projector_setting = os.environ.get(_EXACT_RELION_PROJECTOR_ENV, "1").strip().lower()
        if exact_projector_setting not in {"0", "false", "no", "off"}:
            relion_projector_half_by_class = projector_half_by_class
            relion_projector_r_max = projector_r_max
    else:
        means = reference_to_dense_means(state.Iref)
    if mean_variance is None:
        mean_variance = np.abs(np.asarray(means)) ** 2
    return means, mean_variance, relion_projector_half_by_class, relion_projector_r_max


def run_dense_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    particle_ids: np.ndarray | None = None,
    halfset_ids: np.ndarray | None = None,
) -> DenseInitialModelEstepResult:
    """Run the InitialModel E-step with RELION-compatible pseudo-halfset routing."""
    class_log_priors = (
        class_log_priors_from_state(state) if config.class_log_priors is None else np.asarray(config.class_log_priors)
    )
    groups = _image_groups(
        particle_ids,
        halfset_ids,
        n_images=int(experiment_dataset.n_images),
        pseudo_halfsets=state.pseudo_halfsets,
    )
    engine_kwargs = _dense_engine_kwargs(state, config)
    means, mean_variance, relion_projector_half_by_class, relion_projector_r_max = _resolve_class_inputs(
        state,
        config,
    )
    if bool(engine_kwargs.get("sparse_pass2", False)):
        selected_particle_ids = (
            np.arange(int(experiment_dataset.n_images), dtype=np.int64)
            if particle_ids is None
            else np.asarray(particle_ids, dtype=np.int64)
        )
        if state.pseudo_halfsets:
            selected_halfset_ids = (
                np.arange(selected_particle_ids.size, dtype=np.int32) % 2
                if halfset_ids is None
                else np.asarray(halfset_ids, dtype=np.int32)
            )
        else:
            selected_halfset_ids = None
        return _run_sparse_pass2_initial_model_estep(
            experiment_dataset,
            state,
            config,
            class_log_priors=class_log_priors,
            groups=groups,
            joint_particle_ids=selected_particle_ids,
            joint_halfset_ids=selected_halfset_ids,
            means=means,
            relion_projector_half_by_class=relion_projector_half_by_class,
            relion_projector_r_max=relion_projector_r_max,
            engine_kwargs=engine_kwargs,
        )

    # Sparse execution constructs its own coarse/local rotation operands.
    if config.rotations is None:
        raise ValueError("Dense execution requires materialized rotations")
    dense_rotations = _dense_rotations_for_config(config.rotations, config)
    halfset_results: dict[int, Any] = {}
    accumulators: list[VdamAccumulator] = []
    for halfset_idx, image_indices in groups:
        if image_indices.size == 0:
            accumulators.extend(_empty_accumulator(state, k, halfset_idx) for k in range(state.K))
            continue
        result = run_dense_k_class_em(
            experiment_dataset,
            means,
            mean_variance,
            config.noise_variance,
            dense_rotations,
            config.translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            image_batch_size=config.image_batch_size,
            rotation_block_size=config.rotation_block_size,
            image_indices=image_indices,
            accumulate_noise=True,
            **_dense_run_em_kwargs(
                _group_local_kwargs(
                    engine_kwargs,
                    image_indices,
                    n_images=int(experiment_dataset.n_images),
                )
            ),
        )
        halfset_results[halfset_idx] = result
        accumulators.extend(
            _arrays_to_accumulators(
                result.Ft_y,
                result.Ft_ctf,
                state,
                halfset_idx=halfset_idx,
                relion_bpref_frame=config.relion_bpref_frame,
                relion_projector_frame=config.relion_projector_frame,
                padding_factor=config.padding_factor,
            )
        )

    selected_particle_ids: list[np.ndarray] = []
    field_lists: dict[str, list[np.ndarray]] = {attr: [] for attr, _ in _PARTICLE_RESULT_FIELDS}
    max_posterior: list[np.ndarray] = []
    for halfset_idx, image_indices in groups:
        result = halfset_results.get(halfset_idx)
        if result is None:
            continue
        attrs = {attr: getattr(result, attr, None) for attr, _ in _PARTICLE_RESULT_FIELDS}
        stats = getattr(result, "stats", None)
        pmax = None if stats is None else getattr(stats, "max_posterior_per_image", None)
        if pmax is None and all(v is None for v in attrs.values()):
            continue
        selected_particle_ids.append(np.asarray(image_indices, dtype=np.int64))
        for attr, dtype in _PARTICLE_RESULT_FIELDS:
            if attrs[attr] is not None:
                field_lists[attr].append(np.asarray(attrs[attr], dtype=dtype))
        if pmax is not None:
            max_posterior.append(np.asarray(pmax, dtype=np.float32))

    meta = _estep_meta(halfset_results)
    _add_accumulator_weight_meta(meta, accumulators, state.K)
    if selected_particle_ids:
        meta["selected_particle_ids"] = np.concatenate(selected_particle_ids).astype(np.int64, copy=False)
    for attr, dtype in _PARTICLE_RESULT_FIELDS:
        if field_lists[attr]:
            meta[attr] = np.concatenate(field_lists[attr]).astype(dtype, copy=False)
    if max_posterior:
        meta["max_posterior_per_image"] = np.concatenate(max_posterior).astype(np.float32, copy=False)
    return DenseInitialModelEstepResult(
        accumulators=accumulators,
        meta=meta,
        halfset_results=halfset_results,
    )
