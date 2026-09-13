"""Native InitialModel / ab-initio K-class driver.

This module owns the executable path behind ``recovar.commands.initial_model``; all
data loading, denovo seeding, dense K-class E-step wiring, VDAM iteration and
artifact writing are coordinated here through their implementation owners.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from recovar.data_io.cryoem_dataset import load_dataset
from recovar.data_io.starfile import read_star
from recovar.em.diagnostics.vdam_mstep_replay import (
    INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV,
    _maybe_replay_iteration_references,
)
from recovar.em.helpers.batch_planning import maybe_cache_raw_image_loaders
from recovar.em.relion import vdam_checkpoint
from recovar.em.vdam import dense_adapter, estep_meta_updates, native_sampling, schedules, star_io
from recovar.em.vdam.bootstrap_iref import _initial_state_from_particles
from recovar.em.vdam.dense_adapter import (
    prepare_relion_projector_class_inputs,
    prepare_relion_projector_class_inputs_and_power,
    run_dense_initial_model_estep,
)
from recovar.em.vdam.iteration_loop import run_vdam_iterations
from recovar.em.vdam.m_step import relion_solvent_flatten_state, relion_solvent_mask
from recovar.em.vdam.mstep_single_class import _prepare_mstep_state_precision
from recovar.em.vdam.native_options import NativeInitialModelOptions
from recovar.em.vdam.native_sampling import (
    NativeSamplingState,
    _build_sampling_plan,
    _estimate_native_sampling_accuracy,
    _initial_sampling_state,
    _isolate_native_sampling_accuracy_diagnostic,
    _prepare_native_sampling_for_iteration,
    _record_native_sampling_assignment_changes,
    _record_native_sampling_post_iteration,
)
from recovar.em.vdam.schedules import (
    DEFAULT_SIGMA2_FUDGE,
    default_subset_sizes_for_3d_initial_model,
    phase_lengths_from_effective_fractions,
)
from recovar.em.vdam.star_io import (
    NativeOpticsState,
    _experiment_read_order,
    _particle_state_from_star,
    _write_final_outputs,
    _write_iteration_artifacts,
    _write_model_star,
)
from recovar.em.vdam.state import InitialModelState, NativeParticleState
from recovar.em.vdam.subset_schedule import restore_subset_order_for_continuation
from recovar.reconstruction.noise import make_radial_noise

INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV = "RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY"


@dataclass(frozen=True)
class NativeInitialModelResult:
    """Summary returned by ``run_native_initial_model``."""

    state: InitialModelState
    output_prefix: str
    final_model_star: str
    final_mrc: str
    class_mrcs: tuple[str, ...]


def _configure_relion_image_mask(dataset, opts: NativeInitialModelOptions) -> None:
    """Configure dataset preprocessing to match InitialModel scoring masks."""

    backend = dataset.image_source.backend
    backend.set_relion_image_mask(
        pixel_size=float(dataset.voxel_size),
        particle_diameter_ang=float(opts.particle_diameter),
        width_mask_edge_px=float(opts.width_mask_edge_px),
    )
    backend.set_relion_fourier_backend(opts.image_fourier_backend)


def _skip_native_sampling_accuracy_diagnostic() -> bool:
    """Return whether the focused controller discriminator skips accuracy estimation."""
    value = os.environ.get(INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV, "").strip()
    if value not in {"", "0", "1"}:
        raise ValueError(f"{INITIAL_MODEL_SKIP_EXPECTED_ACCURACY_ENV} must be 0 or 1")
    return value == "1"


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
    sampling_state: NativeSamplingState,
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
        do_grad = schedules._native_initialmodel_do_grad(
            state,
            iteration,
            grad_em_iters=int(opts.grad_em_iters),
        )
        accuracy_meta = None
        prepared_projector_inputs = (
            None if projector_context is None else projector_context.take(
                state, padding_factor=int(opts.padding_factor)
            )
        )
        pass1_healpix_order = int(sampling_state.healpix_order)
        skip_expected_accuracy = _skip_native_sampling_accuracy_diagnostic()
        if (
            optics_state is not None
            and not skip_expected_accuracy
            and schedules._should_estimate_native_sampling_accuracy(
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
        config = dense_adapter._dense_estep_config(
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
        class_rotation_log_prior = native_sampling._class_rotation_log_prior_for_sampling(
            state,
            sampling_state,
            int(sampling_plan.healpix_order),
        )
        if not bool(config.engine_kwargs.get("sparse_pass2", False)):
            class_rotation_log_prior = native_sampling._expand_class_rotation_log_prior_for_dense_fine_grid(
                class_rotation_log_prior,
                sampling_plan,
            )
        config.engine_kwargs["class_rotation_log_prior"] = class_rotation_log_prior
        config.engine_kwargs.setdefault(
            "max_significants",
            schedules._active_relion_initialmodel_max_significants(state, do_grad=do_grad),
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
        # Monitor hidden-variable changes every iteration, independently of autosampling.
        if int(iteration) <= int(state.nr_iter):
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


def _should_write_iteration_artifacts(iteration: int, nr_iter: int, grad_write_iter: int) -> bool:
    """Match RELION's gradient-output cadence, including the final iteration."""

    if grad_write_iter < 1:
        raise ValueError("grad_write_iter must be >= 1")
    return (iteration % grad_write_iter) == 0 or iteration == nr_iter


def run_native_initial_model(opts: NativeInitialModelOptions) -> NativeInitialModelResult:
    """Run native recovar InitialModel refinement."""

    profile = star_io._StageProfile()

    from recovar.em.vdam.mstep_single_class import _validate_mstep_precision_route

    _validate_mstep_precision_route(opts.mstep_compute_dtype, opts.mstep_backend)
    if opts.mstep_compute_dtype == "float32" and os.environ.get(
        INITIAL_MODEL_IREF_REPLAY_TEMPLATE_ENV, ""
    ).strip():
        raise ValueError("float32 M-step is incompatible with iteration reference replay")
    opts.validate_run()
    profile.record("validation")

    main_star, optics_star = read_star(opts.fn_img)
    particle_order = _experiment_read_order(main_star)
    profile.record("input_star")
    dataset = load_dataset(
        opts.fn_img,
        lazy=bool(opts.lazy),
        datadir=opts.datadir,
        strip_prefix=opts.strip_prefix,
    )
    if getattr(dataset, "tilt_series_flag", False):
        raise NotImplementedError("native InitialModel currently supports SPA particle STAR files, not tilt-series")
    profile.record("dataset_load")
    maybe_cache_raw_image_loaders((dataset,))
    profile.record("raw_cache_setup")

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
        state, optics_group_by_particle = _initial_state_from_particles(
            dataset,
            main_star,
            optics_star,
            opts,
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
            particle_order=particle_order,
            grad_ini_frac=grad_ini_frac,
            grad_fin_frac=grad_fin_frac,
            grad_em_iters=int(opts.grad_em_iters),
            phase_lengths=continuation_phase_lengths,
        )
        sampling_state = continuation.sampling_state
    state = _prepare_mstep_state_precision(state, opts.mstep_compute_dtype)
    profile.record("state_setup")
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
    profile.record("expectation_setup")

    if opts.write_iter_artifacts:
        star_io._write_initial_run_metadata(opts, continuation)
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
    profile.record("initial_artifacts")

    def artifact_sink(current, iteration, meta):
        _record_native_sampling_post_iteration(
            sampling_state,
            current,
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
    profile.record("iteration_setup")

    final_state = run_vdam_iterations(
        state,
        nr_particles=int(dataset.n_images),
        optics_group_by_particle=optics_group_by_particle,
        grad_ini_subset_size=grad_ini_subset_size,
        grad_fin_subset_size=grad_fin_subset_size,
        tau2_fudge_arg=float(opts.tau2_fudge),
        grad_em_iters=int(opts.grad_em_iters),
        random_seed=int(opts.random_seed),
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
    profile.record("iterations")
    final_mrc, class_mrcs = _write_final_outputs(opts.outputname, final_state)
    final_model_star = f"{opts.outputname}_it{final_state.iter:03d}_model.star"
    if not os.path.exists(final_model_star):
        _write_model_star(final_model_star, final_state, class_mrcs)
    profile.record("final_artifacts")
    profile.report("driver")
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
