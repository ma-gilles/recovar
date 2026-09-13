"""Native InitialModel / ab-initio K-class driver.

This module owns the executable recovar path behind ``scripts/run_ab_initio``.
The script remains a thin argparse and RELION-command-snapshot layer; all
data loading, denovo seeding, dense K-class E-step wiring, VDAM iteration and
artifact writing are coordinated here through their implementation owners.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Literal

import numpy as np

from recovar.core import mask as core_mask
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.data_io.starfile import read_star
from recovar.em import sampling
from recovar.em.helpers.batch_planning import maybe_cache_raw_image_loaders
from recovar.em.helpers.orientation_priors import (
    make_relion_translation_log_prior,
    relion_round_away_from_zero,
    relion_sigma_offset_prior_center,
    relion_translation_prior_center,
)
from recovar.em.relion import vdam_checkpoint
from recovar.em.relion.initial_noise import compute_avg_unaligned_and_sigma2
from recovar.em.vdam import estep_meta_updates, star_io
from recovar.em.vdam.bootstrap_iref import compute_bootstrap_iref_via_cpp, postprocess_bootstrap_iref_via_cpp
from recovar.em.vdam.dense_adapter import (
    prepare_relion_projector_class_inputs,
    prepare_relion_projector_class_inputs_and_power,
    run_dense_initial_model_estep,
)
from recovar.em.vdam.estep_common import DenseInitialModelEstepConfig
from recovar.em.vdam.init import initialise_data_vs_prior_from_references, initialise_denovo_state, seed_noise_from_mavg
from recovar.em.vdam.iteration_loop import relion_solvent_flatten_state, relion_solvent_mask, run_vdam_iterations
from recovar.em.vdam.native_options import NativeInitialModelOptions
from recovar.em.vdam.native_sampling import (
    NativeSamplingPlan,
    NativeSamplingState,
    _build_sampling_plan,
    _estimate_native_sampling_accuracy,
    _initial_sampling_state,
    _isolate_native_sampling_accuracy_diagnostic,
    _prepare_native_sampling_for_iteration,
    _record_native_sampling_assignment_changes,
    _record_native_sampling_post_iteration,
    _relion_rnd_unif_factory,
)
from recovar.em.vdam.schedules import (
    DEFAULT_GRAD_EM_ITERS,
    DEFAULT_SIGMA2_FUDGE,
    default_subset_sizes_for_3d_initial_model,
    phase_lengths_from_effective_fractions,
)
from recovar.em.vdam.star_io import (
    NativeOpticsState,
    _experiment_read_order,
    _micrograph_sort_order,
    _output_dir_from_prefix,
    _particle_state_from_star,
    _write_final_outputs,
    _write_iteration_artifacts,
    _write_model_star,
)
from recovar.em.vdam.state import InitialModelState, NativeParticleState
from recovar.em.vdam.subset_schedule import restore_subset_order_for_continuation
from recovar.reconstruction.noise import make_radial_noise
from recovar.utils.helpers import (
    get_gpu_memory_total,
)

RELION_INITIALMODEL_MAX_NR_ITER_WO_LARGE_HIDDEN_VARIABLE_CHANGES = 1
RELION_INITIALMODEL_3D_GRADIENT_MAX_SIGNIFICANTS_PER_CLASS = 100
INITIAL_MODEL_LOCAL_BATCH_REFERENCE_SIZE = 256
INITIAL_MODEL_LOCAL_BATCH_REFERENCE_COUNT_40GB = 32
INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV = "RECOVAR_INITIALMODEL_IREF_REPLAY_TEMPLATE"
INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV = "RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY"


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


@dataclass(frozen=True)
class NativeInitialModelResult:
    """Summary returned by ``run_native_initial_model``."""

    state: InitialModelState
    output_prefix: str
    final_model_star: str
    final_mrc: str
    class_mrcs: tuple[str, ...]


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
    if hasattr(backend, "set_relion_image_mask"):
        backend.set_relion_image_mask(
            pixel_size=float(dataset.voxel_size),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
        )
    else:
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

    if hasattr(backend, "set_relion_fourier_backend"):
        backend.set_relion_fourier_backend(opts.image_fourier_backend)
    elif opts.image_fourier_backend != "host_numpy":
        raise ValueError(
            "InitialModel image_fourier_backend requires a compatible image backend; "
            f"got {opts.image_fourier_backend!r}",
        )


def _native_initialmodel_do_grad(
    state: InitialModelState,
    iteration: int,
    *,
    grad_em_iters: int = DEFAULT_GRAD_EM_ITERS,
) -> bool:
    return ((int(state.nr_iter) - int(iteration)) >= int(grad_em_iters)) and not bool(state.has_converged)


def _active_relion_initialmodel_max_significants(state: InitialModelState, *, do_grad: bool) -> int:
    """Runtime maximum_significants used by RELION gradient InitialModel."""

    if not bool(do_grad):
        return -1
    return int(RELION_INITIALMODEL_3D_GRADIENT_MAX_SIGNIFICANTS_PER_CLASS) * int(state.K)


def _should_record_native_sampling_changes(*, iteration: int, nr_iter: int, do_grad: bool) -> bool:
    """Per-iteration ``monitorHiddenVariableChanges`` cadence (must NOT defer to autosampling)."""
    return int(iteration) <= int(nr_iter)


def _should_estimate_native_sampling_accuracy(*, iteration: int, nr_iter: int, do_grad: bool) -> bool:
    """RELION's ``calculateExpectedAngularErrors`` cadence."""
    iteration = int(iteration)
    if iteration <= 1:
        return True
    if bool(do_grad) and iteration % 10 != 0:
        return False
    return iteration <= int(nr_iter)


def _skip_native_sampling_accuracy_diagnostic() -> bool:
    """Return whether the focused controller discriminator skips accuracy estimation."""
    value = os.environ.get(INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV, "").strip()
    if value not in {"", "0", "1"}:
        raise ValueError(f"{INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV} must be 0 or 1")
    return value == "1"


def _translation_log_prior(
    translations: np.ndarray,
    *,
    voxel_size: float,
    sigma_angstrom: float | None,
    centers: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build InitialModel's RELION accelerated-path ``pdf_offset`` values."""

    if sigma_angstrom is None:
        return None
    translations = np.asarray(translations, dtype=np.float32)
    shared = centers is None
    centers_arr = np.zeros(2, dtype=np.float32) if shared else np.asarray(centers, dtype=np.float32)
    log_prior = make_relion_translation_log_prior(
        translations,
        voxel_size=float(voxel_size),
        sigma_offset_angstrom=float(sigma_angstrom),
        prior_centers=centers_arr,
    )
    return np.asarray(log_prior, dtype=np.float32)


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


def _n_directions_for_healpix_order(healpix_order: int) -> int:
    return int(
        sampling.rotation_grid_size(int(healpix_order)) // sampling.rotation_grid_n_in_planes(int(healpix_order))
    )


def _class_direction_rotation_log_prior(state: InitialModelState, healpix_order: int) -> np.ndarray:
    """Return RELION's class-specific direction prior over coarse rotations.

    RELION copies ``pdf_direction`` into an ``RFLOAT`` buffer and its CUDA
    ``initOrientations`` kernel stores ``log(pdf)`` directly in ``XFLOAT``.
    Do not remove the class-common scale before taking the logarithm.  Although
    that scale cancels analytically, changing it changes float32 addition and
    adaptive-significance ties.
    """

    n_psi = int(sampling.rotation_grid_n_in_planes(int(healpix_order)))
    n_dir = _n_directions_for_healpix_order(int(healpix_order))
    n_rot = int(n_dir * n_psi)
    pdf_direction = np.asarray(state.pdf_direction, dtype=np.float64)
    if pdf_direction.shape != (int(state.K), n_dir):
        pdf_direction = np.full((int(state.K), n_dir), 1.0 / float(int(state.K) * n_dir), dtype=np.float64)
    direction_ids = np.arange(n_rot, dtype=np.int64) // n_psi
    values = pdf_direction[:, direction_ids]
    out = np.full(values.shape, -1.0e30, dtype=np.float64)
    positive = values > 0.0
    out[positive] = np.log(values[positive])
    return out.astype(np.float32)


def _class_rotation_log_prior_for_sampling(
    state: InitialModelState,
    sampling_state: NativeSamplingState | None,
    healpix_order: int,
) -> np.ndarray:
    """Select the live RELION orientation-prior source for this sampling state."""

    if sampling_state is not None and bool(sampling_state.uniform_local_orientation_prior):
        n_rot = int(sampling.rotation_grid_size(int(healpix_order)))
        return np.zeros((int(state.K), n_rot), dtype=np.float32)
    return _class_direction_rotation_log_prior(state, int(healpix_order))


def _expand_class_rotation_log_prior_for_dense_fine_grid(
    class_rotation_log_prior: np.ndarray,
    sampling_plan: NativeSamplingPlan,
) -> np.ndarray:
    """Broadcast coarse direction priors onto dense oversampled rotations."""

    prior = np.asarray(class_rotation_log_prior, dtype=np.float32)
    if int(sampling_plan.oversampling) <= 0:
        return prior
    if prior.ndim != 2:
        raise ValueError(f"class_rotation_log_prior must be 2D, got {prior.ndim} dimensions")

    _rotations, parent_map = sampling.get_oversampled_relion_hidden_rotation_grid_from_samples(
        np.arange(prior.shape[1], dtype=np.int64),
        parent_nside_level=int(sampling_plan.healpix_order),
        oversampling_order=int(sampling_plan.oversampling),
        random_perturbation=float(sampling_plan.random_perturbation),
    )
    parent_map = np.asarray(parent_map, dtype=np.int64)
    expected = int(np.asarray(sampling_plan.rotations).shape[0])
    if parent_map.shape != (expected,):
        raise ValueError(
            "oversampled rotation parent map shape does not match dense rotations: "
            f"got {parent_map.shape}, expected ({expected},)",
        )
    return prior[:, parent_map].astype(np.float32, copy=False)


def _dense_estep_config(
    dataset,
    opts: NativeInitialModelOptions,
    noise_variance: np.ndarray,
    sampling_plan: NativeSamplingPlan,
    translation_offsets: np.ndarray,
    sigma_offset_angstrom: float | None = None,
    class_log_priors: np.ndarray | None = None,
    pass1_healpix_order: int | None = None,
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
    # Default σ_offset = 10 Å matches RELION's _rlnSigmaOffsetsAngst at iter000.
    if sigma_offset_angstrom is None:
        sigma_angstrom = opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    else:
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
    coarse_translation_log_prior = _translation_log_prior(coarse_prior_translations, **_prior_kwargs)
    translation_log_prior = _translation_log_prior(sampling_plan.translations, **_prior_kwargs)

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
            pass1_healpix_order=(
                int(sampling_plan.healpix_order)
                if pass1_healpix_order is None
                else int(pass1_healpix_order)
            ),
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

    dataset_image_shape = getattr(dataset, "image_shape", None)
    if dataset_image_shape is None:
        # Lightweight test/adaptor datasets may intentionally expose only the
        # metadata consumed by this function.  The cap is an execution-policy
        # guard for real image stacks, so an unknown grid keeps the requested
        # value rather than inventing a size or probing a device.
        effective_image_batch_size = int(opts.image_batch_size)
    else:
        grid_size = int(dataset_image_shape[0])
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


def _native_expectation_step(
    dataset,
    opts: NativeInitialModelOptions,
    particle_state: NativeParticleState,
    sampling_state: NativeSamplingState | None = None,
    optics_state: NativeOpticsState | None = None,
    *,
    projector_context: _IterationProjectorContext | None = None,
):
    def _expectation_step(state: InitialModelState, particle_ids: np.ndarray, halfset_ids: np.ndarray):
        defer_token = os.environ.get("RECOVAR_VDAM_DEFER_SPARSE_ROTATIONS", "0").strip()
        if defer_token not in {"0", "1"}:
            raise ValueError("RECOVAR_VDAM_DEFER_SPARSE_ROTATIONS must be 0 or 1")
        sampling_kwargs = {"defer_fine_rotations": True} if defer_token == "1" else {}
        iteration = max(1, int(state.iter))
        do_grad = _native_initialmodel_do_grad(
            state,
            iteration,
            grad_em_iters=int(opts.grad_em_iters),
        )
        sampling_updated = False
        accuracy_meta = None
        prepared_projector_inputs = (
            None if projector_context is None else projector_context.take(
                state, padding_factor=int(opts.padding_factor)
            )
        )
        pass1_healpix_order = (
            int(opts.healpix_order)
            if sampling_state is None
            else int(sampling_state.healpix_order)
        )
        if sampling_state is None:
            sampling_plan = _build_sampling_plan(opts, iteration=iteration, **sampling_kwargs)
        else:
            skip_expected_accuracy = _skip_native_sampling_accuracy_diagnostic()
            if (
                optics_state is not None
                and not skip_expected_accuracy
                and _should_estimate_native_sampling_accuracy(
                    iteration=iteration,
                    nr_iter=int(state.nr_iter),
                    do_grad=do_grad,
                )
            ):
                # RELION expectationSetup constructs the production PPref
                # before calculateExpectedAngularErrors and reuses that PPref
                # for scoring. Build RECOVAR's production projector in the
                # same order and pass it through the shared E-step adapter so
                # the accuracy helper cannot perturb a later rebuild.
                if prepared_projector_inputs is None:
                    prepared_projector_inputs = prepare_relion_projector_class_inputs(
                        state,
                        padding_factor=int(opts.padding_factor),
                        projector_setup_backend=opts.projector_setup_backend,
                    )
                accuracy_meta = _estimate_native_sampling_accuracy(
                    sampling_state,
                    state,
                    particle_state,
                    optics_state,
                    particle_order=np.asarray(particle_ids, dtype=np.int64),
                    random_seed=int(opts.random_seed),
                    padding_factor=int(opts.padding_factor),
                    sigma2_fudge=DEFAULT_SIGMA2_FUDGE,
                )
            sampling_updated = _prepare_native_sampling_for_iteration(
                sampling_state,
                state,
                iteration=iteration,
                do_grad=do_grad,
            )
            sampling_plan = _build_sampling_plan(
                opts,
                iteration=iteration,
                sampling_state=sampling_state,
                **sampling_kwargs,
            )
        sigma_offset_angstrom = float(np.sqrt(max(float(state.sigma2_offset), 0.0)))
        current_noise_variance = _noise_variance_from_sigma2(state.sigma2_noise, int(state.ori_size))
        config = _dense_estep_config(
            dataset,
            opts,
            current_noise_variance,
            sampling_plan,
            particle_state.translation_offsets,
            sigma_offset_angstrom=sigma_offset_angstrom,
            class_log_priors=np.zeros(int(state.K), dtype=np.float64),
            pass1_healpix_order=pass1_healpix_order,
        )
        if prepared_projector_inputs is not None:
            prepared_means, prepared_variance, prepared_half, prepared_r_max = (
                prepared_projector_inputs
            )
            config = replace(
                config,
                means=prepared_means,
                mean_variance=prepared_variance,
                relion_projector_half_by_class=prepared_half,
                relion_projector_r_max=prepared_r_max,
            )
        class_rotation_log_prior = _class_rotation_log_prior_for_sampling(
            state,
            sampling_state,
            int(sampling_plan.healpix_order),
        )
        if not bool(config.engine_kwargs.get("sparse_pass2", False)):
            class_rotation_log_prior = _expand_class_rotation_log_prior_for_dense_fine_grid(
                class_rotation_log_prior,
                sampling_plan,
            )
        config.engine_kwargs["class_rotation_log_prior"] = class_rotation_log_prior
        config.engine_kwargs.setdefault(
            "max_significants",
            _active_relion_initialmodel_max_significants(state, do_grad=do_grad),
        )
        config.engine_kwargs["debug_iteration"] = iteration
        previous_translations = np.asarray(particle_state.translation_offsets, dtype=np.float64).copy()
        previous_rotations = (
            None
            if particle_state.best_pose_rotations is None
            else np.asarray(particle_state.best_pose_rotations, dtype=np.float64).copy()
        )
        previous_classes = np.asarray(particle_state.class_assignments, dtype=np.int32).copy()
        if particle_state.visited is not None:
            # RELION's old class number is zero until the first visit. Our
            # scorer uses zero-based classes, so preserve that distinct old
            # state only in this change-monitor snapshot, before visits update.
            previous_classes[~np.asarray(particle_state.visited, dtype=bool)] = -1
        result = run_dense_initial_model_estep(
            dataset, state, config, particle_ids=particle_ids, halfset_ids=halfset_ids
        )
        result.meta.update(
            random_perturbation=float(sampling_plan.random_perturbation),
            n_rotations=sampling_plan.n_rotations,
            n_translations=int(sampling_plan.translations.shape[0]),
            requested_image_batch_size=int(opts.image_batch_size),
            effective_image_batch_size=int(config.image_batch_size),
            healpix_order=int(sampling_plan.healpix_order),
            oversampling=int(sampling_plan.oversampling),
            offset_range_px=float(sampling_plan.offset_range_px),
            offset_step_px=float(sampling_plan.offset_step_px),
            offset_range_angstrom=float(sampling_plan.offset_range_angstrom),
            offset_step_angstrom=float(sampling_plan.offset_step_angstrom),
            max_significants=int(config.engine_kwargs.get("max_significants", -1)),
            sigma_offset_angstrom=sigma_offset_angstrom,
            sigma2_offset_before=float(state.sigma2_offset),
        )
        if sampling_state is not None:
            result.meta["sampling_accuracy_estimated"] = accuracy_meta is not None
            result.meta["sampling_accuracy_skipped_by_diagnostic"] = bool(skip_expected_accuracy)
            result.meta["sampling_accuracy_isolated_by_diagnostic"] = bool(
                _isolate_native_sampling_accuracy_diagnostic()
            )
            if accuracy_meta is not None:
                result.meta.update(accuracy_meta)
            result.meta.update(
                sampling_updated=bool(sampling_updated),
                effective_offset_step_angstrom=float(sampling_state.effective_offset_step_angstrom),
                sampling_acc_rot=float(sampling_state.acc_rot),
                sampling_acc_trans_angstrom=float(sampling_state.acc_trans_angstrom),
                sampling_nr_iter_wo_resol_gain=int(sampling_state.nr_iter_wo_resol_gain),
                sampling_has_fine_enough_angular_sampling=bool(sampling_state.has_fine_enough_angular_sampling),
                orientational_prior_mode=int(sampling_state.orientational_prior_mode),
                uniform_local_orientation_prior=bool(sampling_state.uniform_local_orientation_prior),
            )
        estep_meta_updates._update_particle_state_from_estep_meta(
            particle_state,
            result.meta,
            (
                sampling_plan.translations
                if sampling_plan.metadata_translations is None
                else sampling_plan.metadata_translations
            ),
        )
        if sampling_state is not None and _should_record_native_sampling_changes(
            iteration=iteration,
            nr_iter=int(state.nr_iter),
            do_grad=do_grad,
        ):
            _record_native_sampling_assignment_changes(
                sampling_state,
                particle_ids=result.meta.get("selected_particle_ids"),
                previous_translations=previous_translations,
                current_translations=particle_state.translation_offsets,
                previous_rotations=previous_rotations,
                current_rotations=particle_state.best_pose_rotations,
                previous_classes=previous_classes,
                current_classes=particle_state.class_assignments,
            )
        if sampling_state is not None:
            result.meta["current_changes_optimal_offsets_angstrom"] = float(
                sampling_state.current_changes_optimal_offsets_angstrom
            )
            result.meta["current_changes_optimal_orientations"] = float(
                sampling_state.current_changes_optimal_orientations
            )
            result.meta["current_changes_optimal_classes"] = float(sampling_state.current_changes_optimal_classes)
            result.meta["sampling_nr_iter_wo_large_hidden_variable_changes"] = int(
                sampling_state.nr_iter_wo_large_hidden_variable_changes
            )
            result.meta["sampling_smallest_changes_optimal_offsets_angstrom"] = float(
                sampling_state.smallest_changes_optimal_offsets_angstrom
            )
            result.meta["sampling_smallest_changes_optimal_orientations"] = float(
                sampling_state.smallest_changes_optimal_orientations
            )
            result.meta["sampling_smallest_changes_optimal_classes"] = float(
                sampling_state.smallest_changes_optimal_classes
            )
        return result.accumulators, result.meta

    return _expectation_step


def _initial_state_from_particles(
    dataset,
    main_star,
    optics_star,
    opts: NativeInitialModelOptions,
    rotations: np.ndarray,
) -> tuple[InitialModelState, np.ndarray]:
    profile_initial_state = bool(os.environ.get("RECOVAR_INITIAL_MODEL_PROFILE"))
    initial_state_started = time.perf_counter()
    stage_started = initial_state_started
    initial_state_profile: dict[str, float] = {}

    def _record_initial_state_stage(name: str) -> None:
        nonlocal stage_started
        if not profile_initial_state:
            return
        now = time.perf_counter()
        initial_state_profile[f"{name}_time_s"] = float(now - stage_started)
        stage_started = now

    ori_size = int(dataset.grid_size)
    pixel_size = float(dataset.voxel_size)
    order = _experiment_read_order(main_star)
    optics_group_by_particle = star_io._optics_group_indices(main_star)
    nr_optics_groups = int(np.unique(optics_group_by_particle).size)
    if nr_optics_groups != 1:
        raise NotImplementedError("native InitialModel currently supports one optics group")
    _record_initial_state_stage("setup")

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
    _record_initial_state_stage("average_unaligned")

    bootstrap_count = min(len(order), int(opts.bootstrap_min_particles))
    bootstrap_order = order[:bootstrap_count]
    images = _load_raw_images(dataset, bootstrap_order, batch_size=max(1, int(opts.image_batch_size)))
    _record_initial_state_stage("raw_images")
    sorted_star = main_star.iloc[bootstrap_order]
    voltage, Cs, Q0, pixel_size = star_io._single_optics_scalars(sorted_star, optics_star, dataset)
    _record_initial_state_stage("optics_metadata")

    iref = compute_bootstrap_iref_via_cpp(
        images=images,
        defU=np.asarray(sorted_star["_rlnDefocusU"].astype(float).to_numpy(), dtype=np.float64),
        defV=np.asarray(sorted_star["_rlnDefocusV"].astype(float).to_numpy(), dtype=np.float64),
        defAngle=np.asarray(sorted_star["_rlnDefocusAngle"].astype(float).to_numpy(), dtype=np.float64),
        phase_shift=star_io._phase_shift(sorted_star),
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
    _record_initial_state_stage("bootstrap")

    state = initialise_denovo_state(
        ori_size=ori_size,
        pixel_size=pixel_size,
        K=int(opts.nr_classes),
        nr_iter=int(opts.nr_iter),
        n_directions=_n_directions_for_healpix_order(int(opts.healpix_order)),
        nr_optics_groups=nr_optics_groups,
        pseudo_halfsets=True,
        padding_factor=int(opts.padding_factor),
    )
    state = seed_noise_from_mavg(state, sigma2_per_group)
    init_sigma_offset_angstrom = (
        opts.translation_sigma_angstrom if opts.translation_sigma_angstrom is not None else 10.0
    )
    state.sigma2_offset = float(init_sigma_offset_angstrom) ** 2
    state.Mavg = Mavg
    _record_initial_state_stage("state_init")
    # RECOVAR_INITIAL_IREF_OVERRIDE lets a parity caller swap in RELION's
    # iter000 ref directly when isolating E/M-step behavior from bootstrap.
    override_path = os.environ.get("RECOVAR_INITIAL_IREF_OVERRIDE")
    if override_path:
        # Parity hook: load Iref directly. Comma-separated paths for K-class,
        # single path broadcast across K, or a "{k}" template expanded k=1..K.
        from recovar.utils.helpers import load_relion_volume

        K = int(opts.nr_classes)
        paths = [p.strip() for p in override_path.split(",") if p.strip()]
        if len(paths) == 1 and "{k" in paths[0]:
            paths = [paths[0].format(k=k + 1) for k in range(K)]
        if len(paths) not in (1, K):
            raise ValueError(f"RECOVAR_INITIAL_IREF_OVERRIDE expects 1 or K={K} paths, got {len(paths)}")
        vols = np.stack(
            [np.asarray(load_relion_volume(p), dtype=np.float64) for p in paths],
            axis=0,
        )
        if vols.shape[1:] != (ori_size, ori_size, ori_size):
            raise ValueError(f"RECOVAR_INITIAL_IREF_OVERRIDE volume shape {vols.shape[1:]} != {(ori_size,) * 3}")
        state.Iref = np.broadcast_to(vols, (K, ori_size, ori_size, ori_size)).copy() if len(paths) == 1 else vols
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
    _record_initial_state_stage("initial_reference")
    state = initialise_data_vs_prior_from_references(
        state,
        nr_particles=len(main_star),
        fix_tau=False,
    )
    _record_initial_state_stage("data_vs_prior")
    if profile_initial_state:
        initial_state_profile["total_time_s"] = float(time.perf_counter() - initial_state_started)
        print(
            f"VDAM initial state profile: {json.dumps(initial_state_profile, sort_keys=True)}",
            flush=True,
        )
    return state, optics_group_by_particle


def _maybe_replay_iteration_references(
    state: InitialModelState,
    *,
    iteration: int,
    meta: dict,
) -> InitialModelState:
    """Replace post-M-step references from an explicit diagnostic template.

    This fail-closed hook is used only for causal trajectory boundaries.  A
    template may contain ``{iteration}`` and ``{k}`` format fields, where
    ``k`` is RELION's one-based class number.  A comma-separated list supplies
    one path per class; a single path is broadcast only for K=1.
    """

    template = os.environ.get(INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV, "").strip()
    if not template:
        return state

    tokens = [token.strip() for token in template.split(",") if token.strip()]
    if len(tokens) == 1 and "{k" in tokens[0]:
        paths = [
            tokens[0].format(iteration=int(iteration), k=class_index + 1)
            for class_index in range(int(state.K))
        ]
    elif len(tokens) == 1 and int(state.K) == 1:
        paths = [tokens[0].format(iteration=int(iteration), k=1)]
    elif len(tokens) == int(state.K):
        paths = [
            token.format(iteration=int(iteration), k=class_index + 1)
            for class_index, token in enumerate(tokens)
        ]
    else:
        raise ValueError(
            f"{INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV} expects one path for K=1 "
            f"or K={int(state.K)} comma-separated paths, got {len(tokens)}"
        )

    from recovar.utils.helpers import load_relion_volume

    references = np.stack(
        [np.asarray(load_relion_volume(path), dtype=np.float64) for path in paths],
        axis=0,
    )
    expected_shape = (int(state.K), int(state.ori_size), int(state.ori_size), int(state.ori_size))
    if references.shape != expected_shape:
        raise ValueError(
            f"iteration reference replay shape {references.shape} != {expected_shape}"
        )
    if not np.all(np.isfinite(references)):
        raise ValueError("iteration reference replay contains non-finite values")

    out = replace(state)
    out.Iref = references
    meta["diagnostic_iref_replay_paths"] = paths
    meta["diagnostic_iref_replay_iteration"] = int(iteration)
    return out


def _should_write_iteration_artifacts(iteration: int, nr_iter: int, grad_write_iter: int) -> bool:
    """Match RELION's gradient-output cadence, including the final iteration."""

    if grad_write_iter < 1:
        raise ValueError("grad_write_iter must be >= 1")
    return (iteration % grad_write_iter) == 0 or iteration == nr_iter


def _prepare_mstep_state_precision(state, mstep_compute_dtype):
    """Convert M-owned numerical state once after bootstrap or continuation.

    FSC, authoritative tau2, noise and priors retain their existing precision.
    Bootstrap and projector refresh are not part of this F32 transaction route.
    """
    from recovar.em.vdam.mstep_single_class import _MSTEP_F32_STATE_DTYPES

    if mstep_compute_dtype == "float64":
        return state
    if mstep_compute_dtype != "float32":
        raise ValueError(f"Unknown mstep_compute_dtype: {mstep_compute_dtype!r}")
    return replace(
        state,
        **{
            name: np.asarray(getattr(state, name)).astype(dtype, copy=True)
            for name, dtype in _MSTEP_F32_STATE_DTYPES.items()
        },
    )


def run_native_initial_model(opts: NativeInitialModelOptions) -> NativeInitialModelResult:
    """Run native recovar InitialModel refinement."""

    profile_driver = bool(os.environ.get("RECOVAR_INITIAL_MODEL_PROFILE"))
    driver_started = time.perf_counter()
    stage_started = driver_started
    driver_profile: dict[str, float] = {}

    def _record_driver_stage(name: str) -> None:
        nonlocal stage_started
        if not profile_driver:
            return
        now = time.perf_counter()
        driver_profile[f"{name}_time_s"] = float(now - stage_started)
        stage_started = now

    from recovar.em.vdam.mstep_single_class import _validate_mstep_precision_route

    _validate_mstep_precision_route(opts.mstep_compute_dtype, opts.mstep_backend)
    if opts.mstep_compute_dtype == "float32" and os.environ.get(
        INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV, ""
    ).strip():
        raise ValueError("float32 M-step is incompatible with iteration reference replay")
    if opts.nr_classes < 1:
        raise ValueError("nr_classes must be >= 1")
    if opts.nr_iter < 1:
        raise ValueError("nr_iter must be >= 1")
    if opts.grad_write_iter < 1:
        raise ValueError("grad_write_iter must be >= 1")
    if int(opts.exact_local_bucket_radix) not in (2, 4):
        raise ValueError("exact_local_bucket_radix must be 2 or 4")
    if int(opts.exact_local_physical_order_chunk_size) not in (0,) and int(
        opts.exact_local_physical_order_chunk_size
    ) < 3:
        raise ValueError(
            "exact_local_physical_order_chunk_size must be 0 (disabled) or at least 3"
        )
    if opts.diagnostic_stop_after_iteration is not None and not (
        1 <= int(opts.diagnostic_stop_after_iteration) <= int(opts.nr_iter)
    ):
        raise ValueError("diagnostic_stop_after_iteration must be between 1 and nr_iter")
    if (
        opts.diagnostic_continue_optimiser is not None
        and opts.diagnostic_stop_after_iteration is None
    ):
        raise ValueError(
            "diagnostic_continue_optimiser requires diagnostic_stop_after_iteration; "
            "unbounded continuation is intentionally unsupported"
        )
    if opts.padding_factor not in (1, 2):
        raise NotImplementedError("native InitialModel currently supports RELION GUI --pad 1 or 2 only")
    if opts.run_relion_align_symmetry:
        raise NotImplementedError("native post-run relion_align_symmetry execution is not wired yet")
    if not opts.do_run_C1 and opts.sym_name.lower() != "c1":
        raise NotImplementedError(
            "native InitialModel direct refinement currently supports C1 only; "
            "use the GUI-default do_run_C1 mode until symmetry-restricted sampling is implemented"
        )
    _record_driver_stage("validation")

    main_star, optics_star = read_star(opts.fn_img)
    particle_order = _micrograph_sort_order(main_star)
    _record_driver_stage("input_star")
    dataset = load_dataset(
        opts.fn_img,
        lazy=bool(opts.lazy),
        datadir=opts.datadir,
        strip_prefix=opts.strip_prefix,
    )
    if getattr(dataset, "tilt_series_flag", False):
        raise NotImplementedError("native InitialModel currently supports SPA particle STAR files, not tilt-series")
    _record_driver_stage("dataset_load")
    maybe_cache_raw_image_loaders((dataset,))
    _record_driver_stage("raw_cache_setup")

    _configure_relion_image_mask(dataset, opts)
    optics_state = star_io._native_optics_state(main_star, optics_star, dataset)
    continuation = None
    if opts.diagnostic_continue_optimiser is not None:
        continuation = vdam_checkpoint._load_native_vdam_continuation(
            opts.diagnostic_continue_optimiser,
            expected_data_star=opts.fn_img,
            opts=opts,
            dataset=dataset,
        )
        if int(opts.diagnostic_stop_after_iteration) != int(continuation.iteration) + 1:
            raise ValueError(
                "diagnostic native VDAM continuation must execute exactly one next iteration: "
                f"checkpoint={continuation.iteration}, "
                f"stop={opts.diagnostic_stop_after_iteration}"
            )
    particle_state = _particle_state_from_star(
        main_star,
        dataset,
        allow_unvisited_class_zero=continuation is not None,
        nr_classes=int(opts.nr_classes),
    )
    if continuation is None:
        grad_ini_subset_size, grad_fin_subset_size = (
            default_subset_sizes_for_3d_initial_model(int(dataset.n_images))
        )
        grad_ini_frac = float(opts.grad_ini_frac)
        grad_fin_frac = float(opts.grad_fin_frac)
        continuation_phase_lengths = None
        sampling_state = _initial_sampling_state(opts, pixel_size=float(dataset.voxel_size))
        sampling_plan = _build_sampling_plan(opts, iteration=1, sampling_state=sampling_state)
        state, optics_group_by_particle = _initial_state_from_particles(
            dataset,
            main_star,
            optics_star,
            opts,
            sampling_plan.rotations,
        )
        sampling_state.last_current_resolution = float(state.current_resolution)
    else:
        vdam_checkpoint._validate_continuation_order_replay(continuation)
        grad_ini_subset_size = int(continuation.grad_ini_subset_size)
        grad_fin_subset_size = int(continuation.grad_fin_subset_size)
        grad_ini_frac = float(continuation.grad_ini_frac)
        grad_fin_frac = float(continuation.grad_fin_frac)
        continuation_phase_lengths = phase_lengths_from_effective_fractions(
            int(continuation.state.nr_iter),
            grad_ini_frac,
            grad_fin_frac,
        )
        optics_group_by_particle = star_io._optics_group_indices(main_star)
        if int(np.unique(optics_group_by_particle).size) != 1:
            raise NotImplementedError(
                "diagnostic native VDAM continuation currently supports one optics group"
            )
        state = restore_subset_order_for_continuation(
            continuation.state,
            through_iteration=int(continuation.iteration),
            nr_particles=int(dataset.n_images),
            optics_group_by_particle=optics_group_by_particle,
            grad_ini_subset_size=grad_ini_subset_size,
            grad_fin_subset_size=grad_fin_subset_size,
            random_seed=int(opts.random_seed),
            rnd_unif_factory=_relion_rnd_unif_factory,
            particle_order=particle_order,
            grad_ini_frac=grad_ini_frac,
            grad_fin_frac=grad_fin_frac,
            grad_em_iters=int(opts.grad_em_iters),
            phase_lengths=continuation_phase_lengths,
        )
        sampling_state = continuation.sampling_state
    state = _prepare_mstep_state_precision(state, opts.mstep_compute_dtype)
    _record_driver_stage("state_setup")
    exact_projector_setting = os.environ.get(
        "RECOVAR_INITIAL_MODEL_EXACT_RELION_PROJECTOR", "1"
    ).strip().lower()
    projector_context = (
        None if exact_projector_setting in {"0", "false", "no", "off"}
        else _IterationProjectorContext(projector_setup_backend=opts.projector_setup_backend)
    )
    expectation_step = _native_expectation_step(
        dataset,
        opts,
        particle_state,
        sampling_state,
        optics_state,
        projector_context=projector_context,
    )
    _record_driver_stage("expectation_setup")

    if opts.write_iter_artifacts:
        _output_dir_from_prefix(opts.outputname).mkdir(parents=True, exist_ok=True)
        config_path = f"{opts.outputname}_native_options.json"
        native_options = asdict(opts)
        native_options["resolved_cuda_allocator"] = os.environ.get(
            "TF_GPU_ALLOCATOR",
            "default",
        )
        native_options["jax_compilation_cache_enabled"] = bool(
            os.environ.get("JAX_COMPILATION_CACHE_DIR")
        )
        native_options["jax_compilation_cache_dir"] = os.environ.get(
            "JAX_COMPILATION_CACHE_DIR"
        )
        native_options["jax_persistent_cache_min_compile_time_secs"] = os.environ.get(
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
        )
        with open(config_path, "w") as f:
            json.dump(native_options, f, indent=2, sort_keys=True)
        if continuation is None:
            _write_iteration_artifacts(
                opts.outputname,
                state,
                0,
                {"checkpoint_iteration": 0, "phase": "bootstrap"},
                main_star=main_star,
                optics_star=optics_star,
                dataset=dataset,
                particle_state=particle_state,
            )
        else:
            continuation_path = f"{opts.outputname}_diagnostic_continuation.json"
            with open(continuation_path, "w") as f:
                json.dump(
                    {
                        "classification": "diagnostic_performance_only",
                        "exactly_one_next_iteration": True,
                        "iteration": int(continuation.iteration),
                        "optimiser_star": str(continuation.optimiser_star),
                        "model_star": str(continuation.model_star),
                        "data_star": str(continuation.data_star),
                        "sampling_star": str(continuation.sampling_star),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")
    _record_driver_stage("initial_artifacts")

    def artifact_sink(current, iteration, meta):
        _record_native_sampling_post_iteration(
            sampling_state,
            current,
            iteration=iteration,
            meta=meta,
        )
        if not opts.write_iter_artifacts or not _should_write_iteration_artifacts(
            iteration, int(opts.nr_iter), int(opts.grad_write_iter)
        ):
            return
        _write_iteration_artifacts(
            opts.outputname,
            current,
            iteration,
            meta,
            main_star=main_star,
            optics_star=optics_star,
            dataset=dataset,
            particle_state=particle_state,
        )

    post_mstep_update = None
    solvent_mask = None
    if opts.do_solvent:
        solvent_mask = relion_solvent_mask(
            ori_size=int(state.ori_size),
            pixel_size=float(state.pixel_size),
            particle_diameter_ang=float(opts.particle_diameter),
            width_mask_edge_px=float(opts.width_mask_edge_px),
        )
    if opts.do_solvent or os.environ.get(INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV, "").strip():

        def post_mstep_update(current, _iteration, _meta):
            if opts.mstep_compute_dtype == "float32" and os.environ.get(
                INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV, ""
            ).strip():
                raise ValueError("float32 M-step is incompatible with iteration reference replay")
            if solvent_mask is not None:
                current = relion_solvent_flatten_state(
                    current,
                    mask=solvent_mask,
                    compute_dtype=opts.mstep_compute_dtype,
                )
            return _maybe_replay_iteration_references(
                current,
                iteration=int(_iteration),
                meta=_meta,
            )
    _record_driver_stage("iteration_setup")

    final_state = run_vdam_iterations(
        state,
        nr_particles=int(dataset.n_images),
        optics_group_by_particle=optics_group_by_particle,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        tau2_fudge_arg=float(opts.tau2_fudge),
        grad_em_iters=int(opts.grad_em_iters),
        random_seed=int(opts.random_seed),
        rnd_unif_factory=_relion_rnd_unif_factory,
        expectation_step=expectation_step,
        iter_artifact_sink=artifact_sink,
        post_mstep_update=post_mstep_update,
        particle_order=particle_order,
        grad_ini_frac=grad_ini_frac,
        grad_fin_frac=grad_fin_frac,
        phase_lengths=continuation_phase_lengths,
        grad_stepsize=float(opts.stepsize),
        mu=float(opts.mu),
        projector_padding_factor=int(opts.padding_factor),
        mstep_backend=opts.mstep_backend,
        mstep_compute_dtype=opts.mstep_compute_dtype,
        projector_refresh_fn=None if projector_context is None else projector_context.refresh,
        start_iteration=int(state.iter),
        diagnostic_stop_after_iteration=opts.diagnostic_stop_after_iteration,
    )
    _record_driver_stage("iterations")
    final_mrc, class_mrcs = _write_final_outputs(opts.outputname, final_state)
    final_model_star = f"{opts.outputname}_it{final_state.iter:03d}_model.star"
    if not os.path.exists(final_model_star):
        _write_model_star(final_model_star, final_state, class_mrcs)
    _record_driver_stage("final_artifacts")
    if profile_driver:
        driver_profile["total_time_s"] = float(time.perf_counter() - driver_started)
        print(f"VDAM driver profile: {json.dumps(driver_profile, sort_keys=True)}", flush=True)
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
