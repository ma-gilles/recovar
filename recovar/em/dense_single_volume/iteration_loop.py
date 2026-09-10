"""Orchestrate dense single-volume and K-class EM refinement.

``refine_single_volume`` resolves the entry-point configuration, and
``_run_relion_iteration_loop`` manages refinement state and dispatch.
``half_scoring`` owns the per-half dense/local engine calls; ``scoring_policy``
owns their shared execution defaults and diagnostic selectors. Local chunks
are implemented in ``local_search_iteration``; state-swap diagnostics belong
to ``helpers.state_swap_runtime``.
See ``docs/math/relion_refinement_algorithm.md`` for the algorithm map.
"""

import gc
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core import fourier_transform_utils
from recovar.data_io import cryoem_dataset
from recovar.em.dense_single_volume import parity_dump as _parity_dump
from recovar.em.dense_single_volume.batch_planning import (
    _estimate_relion_em_batch_sizes,
    _image_backend,
    _plan_adaptive_dense_batch_sizes,
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
    maybe_cache_raw_image_loaders,
)
from recovar.em.dense_single_volume.frozen_boundary import (
    _assert_frozen_scoring_state_unchanged,
    _frozen_scoring_state_arrays,
    _restore_diagnostic_frozen_boundary_state,
)
from recovar.em.dense_single_volume.half_scoring import (
    _score_half_dense_in_bpref_scope,
    _score_half_local_in_bpref_scope,
)
from recovar.em.dense_single_volume.helpers import bpref_diagnostics, reconstruction_diagnostics
from recovar.em.dense_single_volume.helpers.convergence import (
    RefinementState,
    _apply_relion_healpix_order_oracle,
    _approx_acc_rot_policy_for_convergence,
    _direction_prior_healpix_order_for_scoring,
    _exhaustive_grid_order_for_state,
    _final_local_sampling_orders,
    _native_final_perturbation_healpix_order,
    calculate_expected_angular_errors,
    concatenate_pose_stacks_or_none,
    check_convergence,
    healpix_angular_step,
    update_angular_sampling,
    update_refinement_state,
)
from recovar.em.dense_single_volume.helpers.dtype_policy import _local_search_precision_flags
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag_or_false
from recovar.em.dense_single_volume.helpers.expected_accuracy import (
    estimate_relion_expected_accuracy,
    prepare_relion_half1_trial_order,
)
from recovar.em.dense_single_volume.helpers.fourier_window import quantize_current_size
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    half_volume_accumulator_shape,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
)
from recovar.em.dense_single_volume.helpers.iteration_history import RefinementHistory
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
    infer_direction_prior_healpix_order,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
    relion_sigma_offset_prior_center,
    relion_translation_prior_center,
    relion_translation_search_base,
    remap_half_direction_prior_to_healpix_order,
)
from recovar.em.dense_single_volume.helpers.resolution import (
    _bootstrap_current_size_relion,
    _firstiter_cc_ini_high_tau2_taper,
    _firstiter_cc_scheduling_resolution_shell,
    _k1_data_vs_prior_for_scheduling,
    _truncate_data_vs_prior_for_current_size,
    _truncate_fsc_for_current_size_growth,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    initialize_resolution_from_firstiter_ini_high,
    initialize_resolution_from_fsc,
    relion_expectation_coarse_size_order,
    relion_local_pass1_current_size,
    relion_optics_image_current_sizes,
    shell_index_to_resolution_angstrom,
)
from recovar.em.dense_single_volume.helpers.state_swap_runtime import (
    _apply_state_swap_probe,
    _copy_half_pair,
    _copy_optional_float_pair,
    _snapshot_state_swap_inputs,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats, make_relion_stats
from recovar.em.dense_single_volume.local_layout import _selected_rotation_matrices
from recovar.em.dense_single_volume.local_search_iteration import _precompute_exact_local_fine_grid_enabled
from recovar.em.dense_single_volume.mean_helpers import (
    _class_weights_from_posterior,
    _combined_class_direction_prior_from_halves,
    _initialize_class_log_priors,
    _mean_noise_variance,
    _mean_variance_for_scoring_half,
    _merged_mean_from_halves,
    _normalize_initial_means,
    _normalize_noise_variance_per_half,
    _reconstruct_and_postprocess_means,
    _reconstruct_volume_eager,
    _relion_optimizer_average_pmax,
    _updated_mean_variance_per_half,
    compute_unregularized_halfmaps_and_align_signs,
    prepare_initial_mean_variance,
    update_c1_sigma_offset_from_posterior,
    update_posterior_noise_variance,
)
from recovar.em.dense_single_volume.projector_preparation import (
    _relion_projector_half_maps_for_scoring,
    _validate_captured_relion_projector_for_iteration,
    prepare_initial_real_references,
)
from recovar.em.dense_single_volume.refinement_options import RefinementOptions, with_validated_sampling_schedule
from recovar.em.dense_single_volume.relion_metadata import (
    _radial_profile_from_noise_variance,
    _relion_metadata_translations,
    _relion_rotation_grid_float32,
)
from recovar.em.dense_single_volume.relion_normalization import update_relion_norm_scale_corrections
from recovar.em.dense_single_volume.relion_replay import (
    select_final_sampling_star,
    _apply_replay_correction_overrides,
    _as_sigma_offset_half_pair,
    _has_numbered_replay_iteration_overrides,
    _maybe_debug_replay_relion_references,
    _mean_sigma_offset_per_half,
    _normalize_sigma_offset_per_half,
    _perturbation_restart_state_iteration,
    _RelionHalfInputState,
    _resolve_replay_random_perturbation,
    _restore_convergence_state_from_replay_restart,
    _sealed_direction_log_prior,
    _sealed_sampling_base_grids,
    _sealed_sampling_rotation_ids,
    _validate_bpref_particle_order_scope,
    apply_iter_replay_overrides,
    apply_optimiser_convergence_replay,
)
from recovar.em.dense_single_volume.relion_worker_scale import (
    _dispatch_relion_follower_scale_for_final_all_data,
    _dispatch_relion_follower_scale_for_numbered_iteration,
    _finalize_relion_follower_scale_replay_telemetry,
    _format_relion_correction_range,
    _update_relion_follower_corrections,
    setup_relion_follower_scale_state,
)
from recovar.em.dense_single_volume.score_outputs import (
    HalfScoreResult,
    PerHalfOutputs,
    _combine_optional_half_accumulators,
    _maybe_host_offload_half0_local_accumulators,
    _record_score_profile,
    _resolve_mstep_accumulator_shape,
    _resolve_mstep_full_half_axis,
)
from recovar.em.dense_single_volume.scoring_policy import (
    _DENSE_EM_STATIC_KWARGS,
    _TRUE_ENV_VALUES,
    PADDING_FACTOR,
    PROJECTION_PADDING_FACTOR,
    _dense_global_scoring_dtype,
    _k1_relion_x_half_mstep_enabled,
)
from recovar.em.sampling import (
    _get_relion_rotation_grid_eulers_float64,
    _relion_adaptive_pass1_rotations,
    _translation_grid_for_class_count,
    advance_relion_perturbation,
    advance_relion_perturbation_from_seed,
    apply_relion_rotation_perturbation,
    apply_relion_rotation_perturbation_to_eulers,
    apply_relion_translation_perturbation,
    build_local_search_grid_metadata,
    read_relion_optimiser_metadata,
    read_relion_sampling_metadata,
    relion_angular_sampling_deg,
    relion_sampling_perturbation_for_iteration,
    rotation_grid_size,
)
from recovar.reconstruction.regularization import (
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    resolution_from_data_vs_prior,
    update_relion_growth_state_from_fsc,
)

logger = logging.getLogger(__name__)


_FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV = "RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE"
_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV = "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE"
_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV = "RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE"
_FINAL_ALL_DATA_GRID_CORRECT_ENV = "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT"
_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV = "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER"
_KCLASS_REPLAY_TAU2_ENV = "RECOVAR_KCLASS_REPLAY_TAU2"
_KCLASS_REPLAY_TAU2_SAME_ITER_ENV = "RECOVAR_KCLASS_REPLAY_TAU2_SAME_ITER"


def _final_all_data_grid_correct_enabled() -> bool:
    """Return whether final all-data output applies RELION gridding correction.

    The GUI-quality path keeps final all-data gridding correction off by
    default because enabling it can shift final-map FSC-AUC.  Strict RELION
    replay can still enable it explicitly with
    ``RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=1``.
    """

    return parse_env_flag_or_false(_FINAL_ALL_DATA_GRID_CORRECT_ENV, logger=logger)


def _final_all_data_after_max_iter_enabled() -> bool:
    """Return whether diagnostics force final all-data after iteration-cap exit."""

    return parse_env_flag_or_false(_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV, logger=logger)


def _fresh_k1_spectrum_norm_default(
    *,
    preserve_bpref_particle_order: bool,
    allow_replayed_bpref_particle_order: bool,
) -> bool:
    """Enable source-faithful powerClass normalization only for a fresh run."""

    return bool(
        preserve_bpref_particle_order and not allow_replayed_bpref_particle_order
    )


def _should_run_final_all_data_iteration(
    *,
    has_converged: bool,
    iteration: int,
    max_iter: int,
    force_max_iter_after_convergence: bool,
    k_class_enabled: bool = False,
) -> bool:
    """Return whether to run RELION's final all-data reconstruction pass."""

    if force_max_iter_after_convergence:
        return False
    if bool(has_converged):
        return True
    if not (_final_all_data_after_max_iter_enabled() and int(iteration) >= int(max_iter)):
        return False
    if bool(k_class_enabled):
        logger.warning(
            "Ignoring %s=1 for K-class after max_iter exhaustion; final all-data "
            "is only valid for K-class after convergence",
            _FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
        )
        return False
    return True


def _kclass_replay_tau2_enabled() -> bool:
    """Diagnostic switch: use RELION replayed Class3D tau2 spectra directly."""

    return parse_env_flag_or_false(_KCLASS_REPLAY_TAU2_ENV, logger=logger)


def _kclass_replay_tau2_same_iter_enabled() -> bool:
    """Compatibility switch for same-numbered Class3D tau2 replay.

    ``RECOVAR_KCLASS_REPLAY_TAU2`` now uses same-numbered model.star tau2 by
    default, matching RELION's expectation-setup timing.  Keep this parser so
    existing diagnostic scripts that also set the flag continue to work.
    """

    return parse_env_flag_or_false(_KCLASS_REPLAY_TAU2_SAME_ITER_ENV, logger=logger)


from recovar.em.dense_single_volume.debug_dumps import (  # noqa: F401
    _bpref_device_signature_active_for_numbered_half,
    _maybe_dump_noise_update_debug,
    _save_bpref_accumulators,
    _save_iteration_intermediates,
    _significance_dump_half_indices,
)

# RELION's --minres_map default: do not add the Wiener prior term to the
# lowest Fourier shells during MAP reconstruction.
RELION_MINRES_MAP = 5


def _sigma_offset_for_half(current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half, half_index):
    if current_sigma_offset_angstrom_per_half is None:
        return float(current_sigma_offset_angstrom)
    return float(current_sigma_offset_angstrom_per_half[int(half_index)])


def refine_single_volume(
    experiment_datasets: list[cryoem_dataset.CryoEMDataset],
    init_volume: list[jnp.ndarray] | jnp.ndarray,
    init_noise_variance: jnp.ndarray,
    init_mean_variance: jnp.ndarray,
    rotations: jnp.ndarray | None,
    translations: jnp.ndarray | None,
    options: RefinementOptions | None = None,
) -> dict:
    """Multi-iteration RELION-parity EM refinement.

    This API always runs the RELION-parity refinement loop.

    Parameters
    ----------
    experiment_datasets : list of 2 dataset objects
        Half-set datasets (same format as split_E_M_v2 expects).
    init_volume : list of 2 jnp.ndarray, shape (volume_size,) or jnp.ndarray, shape (volume_size,)
        Initial volume in Fourier space for each half-set.
    init_noise_variance : jnp.ndarray, shape (2,image_size)
        Initial per-pixel noise variance for each half-set.
    init_mean_variance : jnp.ndarray, shape (volume_size,)
        Initial signal prior (tau^2).
    rotations : np.ndarray, shape (n_rot, 3, 3)
        Optional initial rotation grid for compatibility. RELION mode
        regenerates grids from the HEALPix refinement state.
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    options : `RefinementOptions` struct that bundles the schedule / adaptive / parity
        / local-search / K-class / replay / debug / batching kwarg groups.
        Defaults to ``RefinementOptions()`` when omitted.

    Returns
    -------
    dict with keys:
        mean : jnp.ndarray -- final merged mean volume
        means : list of 2 jnp.ndarray -- per-half-set means
        fsc : jnp.ndarray -- final FSC curve
        hard_assignments : list of 2 np.ndarray -- per-half-set assignments
        current_sizes : list of int -- current_size at each iteration
        fsc_history : list of jnp.ndarray -- FSC curve at each iteration
        pixel_resolutions : list of float -- pixel resolution at each iter
        wall_times : list of float -- wall time per iteration
        significant_counts : list of (jnp.ndarray or None) -- per-image
            significant sample counts at each iteration (None when
            adaptive_oversampling=0).

    RELION-specific keys:
        convergence_state : RefinementState -- final convergence state
        data_vs_prior_trajectory : list of jnp.ndarray -- per-iteration
            data_vs_prior curves
        healpix_order_trajectory : list of int -- HEALPix order per iter
        ave_Pmax_trajectory : list of float -- average Pmax per iter
    """
    if options is None:
        options = RefinementOptions()

    options = with_validated_sampling_schedule(options)

    return _run_relion_iteration_loop(
        experiment_datasets=experiment_datasets,
        init_volume=init_volume,
        init_reference_real=options.replay.init_reference_real,
        init_noise_variance=init_noise_variance,
        init_mean_variance=init_mean_variance,
        rotations=rotations,
        translations=translations,
        options=options,
    )


# ---------------------------------------------------------------------------
# RELION-parity refinement mode
# ---------------------------------------------------------------------------


def _numbered_relion_iteration(init_relion_iteration: int, local_iteration: int) -> int:
    """Map a restart-local zero-based loop index to RELION's numbered iteration."""

    return int(init_relion_iteration) + int(local_iteration) + 1


def _past_perturb_replay_max_iter(iteration: int, perturb_replay_max_iter: int | None) -> bool:
    """Return whether ``iteration`` (0-indexed) is past the diagnostic replay cutoff.

    ``perturb_replay_max_iter`` is 1-indexed to match
    ``scripts/run_multi_iter_parity.py``'s ``--replay-override-max-iter``
    (and ``replay_iteration_overrides``, which the same flag also gates).
    ``None`` means "no cutoff": every iteration stays in range.
    """

    if perturb_replay_max_iter is None:
        return False
    return (int(iteration) + 1) > int(perturb_replay_max_iter)


def _native_sampling_boundary_for_iteration(
    *,
    iteration: int,
    perturb_replay_relion_dir: str | None,
    perturb_replay_max_iter: int | None,
    sealed_sampling_state,
) -> bool:
    """Return whether this physical iteration owns sampling and convergence.

    A diagnostic replay cutoff is a real ownership boundary, not merely a
    guard around STAR reads. Once crossed, RECOVAR must resume its native
    expected-accuracy, angular-sampling, and convergence transitions.
    """

    replay_active = perturb_replay_relion_dir is not None and not _past_perturb_replay_max_iter(
        iteration,
        perturb_replay_max_iter,
    )
    return not replay_active and sealed_sampling_state is None


def _run_relion_iteration_loop(
    experiment_datasets,
    init_volume,
    init_reference_real,
    init_noise_variance,
    init_mean_variance,
    rotations,
    translations,
    options,
):
    """RELION-parity refinement loop with convergence detection.

    This implements the full RELION auto-refine algorithm:
    1. Convergence-driven iteration (not fixed max_iter)
    2. data_vs_prior for resolution instead of FSC < 0.143
    3. Angular step refinement (HEALPix order increments)
    4. Local angular search when HEALPix order reaches auto_local_healpix_order
    5. Per-image best assignment tracking
    6. Average Pmax computation for adaptive current_size growth

    Corresponds to RELION's autoRefine iteration loop.
    See docs/relion5_auto_refine_algorithm.md.
    """
    from recovar.reconstruction import regularization

    schedule = options.schedule
    adaptive = options.adaptive
    parity = options.parity
    local_search = options.local_search
    k_class = options.k_class
    replay = options.replay
    debug = options.debug
    batching = options.batching
    expected_accuracy = debug.expected_accuracy

    particle_diameter_ang = schedule.particle_diameter_ang
    tau2_fudge = parity.tau2_fudge
    perturb_replay_relion_dir = parity.perturb_replay_relion_dir
    perturb_replay_relion_prefix = parity.perturb_replay_relion_prefix
    perturb_replay_max_iter = parity.perturb_replay_max_iter
    init_relion_iteration = schedule.init_relion_iteration
    final_replay_override = replay.final_replay_override
    n_classes = k_class.n_classes
    stop_after_local_search = debug.stop_after_local_search
    sealed_sampling_state = debug.sealed_sampling_state

    if options.parity.perturb_replay_restart_state_iterations:
        logger.info(
            "Perturbation replay restart provenance: saved-state iterations=%s",
            list(options.parity.perturb_replay_restart_state_iterations),
        )

    setup_t0 = time.time()
    setup_phase_seconds = {}

    def _mark_setup_phase(name: str) -> None:
        setup_phase_seconds[name] = time.time() - setup_t0

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape
    grid_size = cryo.image_shape[0]  # ori_size in RELION terms
    n_classes = int(n_classes)
    k_class_enabled = n_classes > 1
    if (parity.relion_optics_image_sizes is None) != (parity.relion_optics_pixel_sizes is None):
        raise ValueError(
            "relion_optics_image_sizes and relion_optics_pixel_sizes must be supplied together",
        )
    optics_image_sizes = None
    optics_pixel_sizes = None
    if parity.relion_optics_image_sizes is not None:
        optics_image_sizes = np.asarray(parity.relion_optics_image_sizes, dtype=np.int64).reshape(-1)
        optics_pixel_sizes = np.asarray(parity.relion_optics_pixel_sizes, dtype=np.float64).reshape(-1)
        if optics_image_sizes.shape != optics_pixel_sizes.shape or optics_image_sizes.size == 0:
            raise ValueError("RELION optics image geometry arrays must be non-empty and aligned")
    model_pixel_size = (
        float(cryo.voxel_size)
        if parity.relion_model_pixel_size is None
        else float(parity.relion_model_pixel_size)
    )
    if not np.isfinite(model_pixel_size) or model_pixel_size <= 0.0:
        raise ValueError(f"RELION model pixel size must be positive, got {model_pixel_size}")
    _validate_bpref_particle_order_scope(
        preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
        n_classes=n_classes,
        init_relion_iteration=init_relion_iteration,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        replay_iteration_overrides=replay.replay_iteration_overrides,
        sealed_sampling_state=sealed_sampling_state,
        sealed_scoring_context=debug.sealed_scoring_context,
        allow_replayed_bpref_particle_order=parity.allow_replayed_bpref_particle_order,
        allow_state_swap_fresh_bpref_particle_order=debug.state_swap_probe is not None,
    )
    source_faithful_spectrum_norm = _fresh_k1_spectrum_norm_default(
        preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
        allow_replayed_bpref_particle_order=parity.allow_replayed_bpref_particle_order,
    )
    class_log_priors, class_weights = _initialize_class_log_priors(
        n_classes,
        k_class.init_class_log_priors,
        replay.init_direction_prior,
    )

    # --- RELION image mask (softMaskOutsideMap on particles) ---
    # RELION masks images to particle_diameter/(2*pixel_size) with a 5-pixel
    # cosine taper before E-step scoring (ml_optimiser.cpp:6288).  The default
    # edge-taper mask (window_mask(D, 0.85, 0.99)) is too tight — it tapers
    # at 54 px vs RELION's 64 px for a 128-px box.
    RELION_WIDTH_MASK_EDGE = 5
    # Width of the Fourier-mask raised-cosine for ``initialLowPassFilterReferences``
    # / ``ini_high`` (RELION's ``WIDTH_FMASK_EDGE`` macro at
    # ``ml_optimiser.h:91``). This is a Fourier-shell width, NOT to be conflated
    # with the real-space ``--maskedge`` mask edge above. They have different
    # semantic units and different RELION defaults (2 vs 5).
    RELION_WIDTH_FMASK_EDGE = 2


    for ds in experiment_datasets:
        backend = _image_backend(ds)
        if backend is None:
            continue
        if hasattr(backend, "set_relion_fourier_backend"):
            backend.set_relion_fourier_backend(parity.image_fourier_backend)
        if particle_diameter_ang is not None and particle_diameter_ang > 0:
            backend.set_relion_image_mask(
                pixel_size=cryo.voxel_size,
                particle_diameter_ang=particle_diameter_ang,
                width_mask_edge_px=RELION_WIDTH_MASK_EDGE,
            )
            logger.info(
                "RELION mode: image mask radius=%.1f px (particle_diameter=%.1f A, edge=%d px)",
                particle_diameter_ang / (2.0 * cryo.voxel_size),
                particle_diameter_ang,
                RELION_WIDTH_MASK_EDGE,
            )

    maybe_cache_raw_image_loaders(experiment_datasets)
    _mark_setup_phase("mask_and_image_cache")

    # --- Initialize RefinementState ---
    # Corresponds to RELION's initialiseSamplingVectors + initialLowPassFilterReferences
    state = RefinementState(
        iteration=0,
        healpix_order=schedule.init_healpix_order,
        adaptive_oversampling=adaptive.adaptive_oversampling,
        translation_range=schedule.init_translation_range,
        translation_step=schedule.init_translation_step,
        max_healpix_order=schedule.max_healpix_order,
        auto_local_healpix_order=local_search.auto_local_healpix_order,
        current_resolution=float("inf"),
        voxel_size_angstrom=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
        particle_diameter_angstrom=float(particle_diameter_ang or 0.0),
    )
    # RELION's convergence counters are not initialized against an infinite
    # previous resolution.  They resume from the previous optimiser/model STAR
    # in replay mode, or from the initial FSC/ini_high state in a fresh run.
    if (
        sealed_sampling_state is None
        and perturb_replay_relion_dir is not None
        and int(init_relion_iteration) > 0
    ):
        _restore_convergence_state_from_replay_restart(state, options)
    elif schedule.init_fsc is not None:
        initialize_resolution_from_fsc(
            state, options, grid_size=grid_size, voxel_size=cryo.voxel_size,
            dtype=_dense_global_scoring_dtype(),
        )
    elif init_relion_iteration == 0 and parity.relion_firstiter_ini_high_angstrom is not None:
        initialize_resolution_from_firstiter_ini_high(state, options, grid_size=grid_size, voxel_size=cryo.voxel_size)
    if replay.init_refinement_state_fields is not None:
        _restore_diagnostic_frozen_boundary_state(state, options)
    _mark_setup_phase("state_init")

    # RELION mode owns the coarse HEALPix grid. When coarse-grid metadata is
    # provided, regenerate the matching coarse grid here instead of inheriting
    # any finer caller-supplied rotation table.
    current_healpix_order = int(schedule.init_healpix_order)
    if adaptive.nside_level is not None and int(adaptive.nside_level) != current_healpix_order:
        logger.info(
            "RELION mode: ignoring caller nside_level=%d and regenerating initial coarse grid at healpix_order=%d",
            int(adaptive.nside_level),
            current_healpix_order,
        )
    elif rotations is not None:
        logger.info(
            "RELION mode: ignoring caller-provided rotation table and regenerating initial coarse grid at healpix_order=%d",
            current_healpix_order,
        )
    if sealed_sampling_state is not None:
        current_rotations, current_rotation_eulers, current_translations = _sealed_sampling_base_grids(
            sealed_sampling_state,
            voxel_size_angstrom=cryo.voxel_size,
            dtype=_dense_global_scoring_dtype(),
        )
        base_translations = np.asarray(current_translations, dtype=np.float64)
        current_healpix_order = int(sealed_sampling_state["healpix_order_original"])
        if current_healpix_order != int(schedule.init_healpix_order):
            raise ValueError(
                "sealed sampling HEALPix order does not match initialized boundary: "
                f"sealed={current_healpix_order} init={schedule.init_healpix_order}"
            )
        logger.info(
            "Frozen-boundary v3 directly materialized %d Euler rows and %d translations",
            int(current_rotation_eulers.shape[0]),
            int(current_translations.shape[0]),
        )
    elif translations is None:
        current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(
            current_healpix_order, dtype=_dense_global_scoring_dtype()
        )
        base_translations = _translation_grid_for_class_count(
            schedule.init_translation_range,
            schedule.init_translation_step,
            n_classes=n_classes,
            source_units_per_pixel=(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
        ).astype(np.float64, copy=False)
        current_translations = jnp.asarray(
            base_translations,
            dtype=_dense_global_scoring_dtype(),
        )
    else:
        current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(
            current_healpix_order, dtype=_dense_global_scoring_dtype()
        )
        base_translations = np.asarray(translations, dtype=np.float64)
        current_translations = jnp.asarray(translations, dtype=_dense_global_scoring_dtype())
    # Unperturbed base grid — `current_translations` may be replaced per-iter by
    # a perturbed copy (SamplingPerturbation). Keep the base so each iter
    # perturbs a fresh copy rather than compounding prior perturbations.
    # Keep RELION's host-RFLOAT base grid separate so each perturbation starts
    # from the unrounded coordinates.  In double mode the score/pose grid is
    # also RFLOAT; explicit CUDA-f32 helpers cast only at their ABI boundary.
    if debug.save_intermediates_dir is not None:
        os.makedirs(debug.save_intermediates_dir, exist_ok=True)

    collect_local_search_profile = (
        debug.save_intermediates_dir is not None if local_search.local_search_profile_mode == "auto" else local_search.local_search_profile_mode == "on"
    )
    if debug.stop_after_local_search_profile:
        collect_local_search_profile = True
    if debug.stop_after_local_search_score_only:
        stop_after_local_search = True
    _mark_setup_phase("sampling_grid")

    padded_volume_shape = tuple(d * PADDING_FACTOR for d in volume_shape)

    def _safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        """Reduce batch sizes for large pose grids to avoid GPU OOM."""
        plan = _estimate_relion_em_batch_sizes(
            requested_image_batch_size=batching.image_batch_size,
            requested_rotation_block_size=batching.rotation_block_size,
            n_rot=n_rot,
            n_trans=n_trans,
            image_shape=image_shape_for_batch or cryo.image_shape,
            volume_shape=volume_shape,
            padding_factor=PADDING_FACTOR,
            n_classes=n_classes if classes is None else classes,
            current_size=current_size_for_batch,
        )
        plan.log_adjustment(
            requested_image_batch_size=batching.image_batch_size,
            requested_rotation_block_size=batching.rotation_block_size,
            n_rot=n_rot,
            n_trans=n_trans,
            n_classes=n_classes if classes is None else classes,
            logger=logger,
        )
        return plan.image_batch_size, plan.rotation_block_size

    # State: two half-set references.  For K-class refinement each half stores
    # an explicit leading class axis; single-class callers keep the historical
    # flat per-half reference layout.
    means = _normalize_initial_means(init_volume, n_classes)
    initial_real_references_by_half = prepare_initial_real_references(
        init_reference_real, volume_shape=volume_shape, n_classes=n_classes, log=logger
    )
    noise_variance_per_half = _normalize_noise_variance_per_half(
        init_noise_variance,
        n_halves=2,
    )
    noise_variance = _mean_noise_variance(noise_variance_per_half)
    initial_mean_variance = jnp.array(init_mean_variance)
    mean_variance, mean_variance_per_half = prepare_initial_mean_variance(
        initial_mean_variance,
        use_per_half_mean_variance=parity.use_per_half_mean_variance,
        k_class_enabled=k_class_enabled,
        log=logger,
    )
    _mark_setup_phase("initial_arrays")

    # History tracking: one RefinementHistory instance accumulates every
    # per-iteration trajectory (see helpers/iteration_history.py).
    history = RefinementHistory()
    per_half = PerHalfOutputs.empty()
    hard_assignments = per_half.hard_assignments
    previous_assignments = [None, None]
    class_assignments = per_half.class_assignments
    previous_class_assignments = [None, None]
    previous_best_rotations = [None, None]
    relion_half_inputs = _RelionHalfInputState.from_initial_values(
        previous_best_translations=replay.init_previous_best_translations,
        previous_best_rotation_eulers=replay.init_previous_best_rotation_eulers,
        image_corrections=replay.init_image_corrections,
        scale_corrections=replay.init_scale_corrections,
        group_ids=replay.init_group_ids,
        group_count=replay.init_group_count,
    )
    max_posterior_per_half = per_half.max_posterior
    rotation_posterior_per_half = per_half.rotation_posterior
    class_rotation_posterior_per_half = per_half.class_rotation_posterior
    previous_data_vs_prior_for_scheduling = None
    tau2_update_details = None
    tau2_update_details_per_half = None

    # C1 (RELION-parity): per-iter sigma2_offset update from data. Initialized
    # from `init_translation_sigma_angstrom`; updated from RELION's
    # posterior-weighted offset moment when the E-step path propagates it.
    # RELION stores and updates this quantity in Angstrom², and its default
    # lower bound is min_sigma2_offset=2 Å² (ml_optimiser.cpp).
    current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(schedule.init_translation_sigma_angstrom)
    current_sigma_offset_angstrom = _mean_sigma_offset_per_half(current_sigma_offset_angstrom_per_half)
    expected_accuracy_trial_local_indices = None
    expected_accuracy_trial_particle_ids = None
    relion_incr_size = int(schedule.init_relion_incr_size)
    if relion_incr_size <= 0:
        raise ValueError("init_relion_incr_size must be positive")
    relion_has_high_fsc_at_limit = bool(schedule.init_has_high_fsc_at_limit) if schedule.init_has_high_fsc_at_limit is not None else False
    global_direction_prior_per_half = [None, None]
    global_direction_prior_order_per_half = [None, None]
    class_direction_prior_per_half = [None, None]
    class_direction_prior_order_per_half = [None, None]

    # --- Direction prior from snapshot ---
    # When starting from a RELION snapshot, the previous iteration's
    # pdf_orientation is a non-uniform prior over HEALPix directions.
    # RELION applies this in the next E-step.  recovar must do the same.
    if replay.init_direction_prior is not None and k_class_enabled:
        class_direction_prior_per_half = normalize_class_direction_prior_per_half(
            replay.init_direction_prior, n_classes, dtype=_dense_global_scoring_dtype()
        )
        for k in range(2):
            if class_direction_prior_per_half[k] is None:
                continue
            prior_k = np.asarray(class_direction_prior_per_half[k], dtype=_dense_global_scoring_dtype())
            class_direction_prior_per_half[k] = prior_k
            class_direction_prior_order_per_half[k] = infer_direction_prior_healpix_order(prior_k[0])
            logger.info(
                "RELION mode: loaded init class direction priors half-%d: %d classes, %d directions",
                k + 1,
                prior_k.shape[0],
                prior_k.shape[1],
            )
    elif replay.init_direction_prior is not None:
        global_direction_prior_per_half = normalize_direction_prior_per_half(
            replay.init_direction_prior, dtype=_dense_global_scoring_dtype()
        )
        for k in range(2):
            if global_direction_prior_per_half[k] is None:
                continue
            prior_k = np.asarray(global_direction_prior_per_half[k], dtype=_dense_global_scoring_dtype())
            global_direction_prior_per_half[k] = prior_k
            global_direction_prior_order_per_half[k] = infer_direction_prior_healpix_order(prior_k)
            logger.info(
                "RELION mode: loaded init direction prior half-%d: %d directions, range=[%.6f, %.6f], %d zero-probability",
                k + 1,
                len(prior_k),
                prior_k.min(),
                prior_k.max(),
                int(np.sum(prior_k == 0)),
            )
    _mark_setup_phase("direction_prior")

    # Extract per-shell radial profiles from the input pixel-array noise
    # variances for diagnostic logging ("noise update per shell: old=... new=...").
    previous_noise_radial_per_half = [
        _radial_profile_from_noise_variance(noise_k, cryo.image_shape) for noise_k in noise_variance_per_half
    ]
    previous_noise_radial = jnp.asarray(
        np.mean(np.stack(previous_noise_radial_per_half, axis=0), axis=0),
        dtype=_dense_global_scoring_dtype(),
    )
    _mark_setup_phase("noise_radial_init")

    # RELION randomises each half once at the first iteration and then uses
    # the first 100 half-1 particles for calculateExpectedAngularErrors.
    # Build that immutable local order once.  A missing/rebuilt-without-this-
    # helper binding is handled fail-closed below: acc_rot stays infinite and
    # cannot trigger convergence.
    effective_optimizer_random_seed = (
        parity.perturb_seed if parity.optimizer_random_seed is None else parity.optimizer_random_seed
    )
    expected_accuracy_trial_order = prepare_relion_half1_trial_order(
        expected_accuracy=expected_accuracy,
        half1_dataset=experiment_datasets[0],
        optimizer_random_seed=effective_optimizer_random_seed,
        init_relion_iteration=init_relion_iteration,
        log=logger,
    )

    follower_setup = setup_relion_follower_scale_state(
        options,
        relion_half_inputs=relion_half_inputs,
        experiment_datasets=experiment_datasets,
        k_class_enabled=k_class_enabled,
    )

    # --- RELION SamplingPerturbation state (healpix_sampling.cpp:167-174) ---
    # RELION applies a random rigid rotation of the entire SO(3) trial grid at
    # each iteration: A -> A @ R_perturb with R_perturb = R_from_relion([m,m,m])
    # and m = random_perturbation * angular_sampling. The random_perturbation
    # is advanced per iter via realWRAP(prev + rnd_unif(0.5*pf, pf), -pf, +pf).
    # For exact parity replay, read _rlnSamplingPerturbInstance from RELION's
    # per-iter sampling.star.
    if parity.perturb_factor > 0 and parity.perturb_seed is not None:
        random_perturbation = relion_sampling_perturbation_for_iteration(
            parity.perturb_factor,
            parity.perturb_seed,
            init_relion_iteration,
        )
        logger.info(
            "Perturbation init: relion_iter=%d random_seed=%d rp=%+.5f",
            int(init_relion_iteration),
            int(parity.perturb_seed),
            random_perturbation,
        )
    else:
        random_perturbation = 0.0
    perturb_rng = None if parity.perturb_seed is not None else np.random.default_rng()
    iteration = 0
    _mark_setup_phase("before_iterations")
    logger.info(
        "RELION mode setup timing before iteration loop: %s",
        ", ".join(f"{key}={value:.1f}s" for key, value in setup_phase_seconds.items()),
    )
    native_sampling_boundary = _native_sampling_boundary_for_iteration(
        iteration=iteration,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        perturb_replay_max_iter=perturb_replay_max_iter,
        sealed_sampling_state=sealed_sampling_state,
    )
    # A numbered RELION sampling STAR is the state *after* the expectation
    # transition that produced it.  The next expectation computes
    # image_coarse_size from that saved state before updateAngularSampling.
    # Keep this replay boundary separate from RECOVAR's end-of-iteration
    # RefinementState, which may already have advanced one order.
    replay_saved_healpix_order = (
        None if native_sampling_boundary else int(state.healpix_order)
    )
    frozen_initial_scoring_state = None
    frozen_initial_scoring_state_sha256 = None
    if debug.assert_initial_scoring_state_immutable:
        if k_class_enabled:
            raise RuntimeError(
                "Frozen scoring-state immutability assertion currently supports K=1 only"
            )
        frozen_initial_scoring_state = _frozen_scoring_state_arrays(
            means=means,
            mean_variance=mean_variance,
            mean_variance_per_half=(
                mean_variance_per_half if parity.use_per_half_mean_variance else None
            ),
            relion_half_inputs=relion_half_inputs,
            noise_variance_per_half=noise_variance_per_half,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            global_direction_prior_per_half=global_direction_prior_per_half,
            experiment_datasets=experiment_datasets,
            sealed_sampling_state=sealed_sampling_state,
            sealed_scoring_context=debug.sealed_scoring_context,
        )
    while (schedule.force_max_iter_after_convergence or not state.has_converged) and iteration < schedule.max_iter:
        if perturb_replay_relion_dir is not None and _past_perturb_replay_max_iter(
            iteration, perturb_replay_max_iter
        ):
            logger.info(
                "Replay override: disabling RELION per-iteration STAR replay from "
                "iteration %d onward (--replay-override-max-iter %d)",
                iteration + 1,
                perturb_replay_max_iter,
            )
            perturb_replay_relion_dir = None
            replay_saved_healpix_order = None
        native_sampling_boundary = _native_sampling_boundary_for_iteration(
            iteration=iteration,
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            perturb_replay_max_iter=perturb_replay_max_iter,
            sealed_sampling_state=sealed_sampling_state,
        )
        # RELION checks convergence at the top of iteration n from the
        # completed n-1 statistics and the fine-enough decision latched during
        # expectation n-1.  If true, iteration n is the unnumbered joined
        # all-data pass rather than another numbered half-set iteration.
        if (
            native_sampling_boundary
            and not schedule.force_max_iter_after_convergence
            and iteration > 0
            and check_convergence(state)
        ):
            state.has_converged = True
            logger.info(
                "Convergence reached after numbered iteration %d. "
                "Entering RELION final all-data iteration.",
                iteration,
            )
            break
        t0 = time.time()
        _parity_dump.start_iteration(iteration)
        iter_replay_override = None
        if replay.replay_iteration_overrides is not None and iteration < len(replay.replay_iteration_overrides):
            iter_replay_override = replay.replay_iteration_overrides[iteration]
        relion_firstiter_cc_this_iter = bool(
            parity.emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_normalized_cc_this_iter = bool(
            parity.first_iteration_score_mode == "normalized_cc" and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_hard_reconstruction_this_iter = bool(
            parity.first_iteration_reconstruction_mode == "hard" and init_relion_iteration == 0 and iteration == 0
        )
        firstiter_score_mode_this_iter = (
            "normalized_cc" if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter) else "gaussian"
        )
        firstiter_winner_take_all_this_iter = bool(
            relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
        )
        numbered_relion_iteration = _numbered_relion_iteration(init_relion_iteration, iteration)

        if follower_setup.follower_scale_state is not None:
            _dispatch_relion_follower_scale_for_numbered_iteration(
                follower_setup,
                history,
                iteration=iteration,
                numbered_relion_iteration=numbered_relion_iteration,
                relion_half_inputs=relion_half_inputs,
                relion_follower_scale_replay_source=replay.relion_follower_scale_replay,
                dtype=_dense_global_scoring_dtype(),
                logger=logger,
            )

        # --- Determine current_size using RELION's FSC-derived SSNR (C4/C5) ---
        # At iteration 0, no previous half-map FSC exists yet; use the initial
        # resolution plus RELION's bootstrap image-size growth. After that,
        # mimic RELION's auto-refine update:
        # 1. zero FSC beyond the previous current_size limit
        # 2. convert FSC -> SSNR (= data_vs_prior in split-half auto-refine)
        # 3. grow current_size using ave_Pmax, FSC at the current limit, and
        #    RELION's dynamic incr_size heuristic.
        if iteration == 0:
            if init_relion_iteration == 0:
                seeded_cs = bootstrap_current_size_from_ini_high_relion(
                    grid_size,
                    float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
                    parity.relion_firstiter_ini_high_angstrom,
                    incr_size=relion_incr_size,
                )
            else:
                seeded_cs = None
            if seeded_cs is not None:
                current_size = int(seeded_cs)
                data_vs_prior_iter = None
                logger.info(
                    "RELION init bootstrap: seeding iter-1 current_size from ini_high=%.2f A -> %d",
                    float(parity.relion_firstiter_ini_high_angstrom),
                    current_size,
                )
            elif schedule.init_fsc is not None:
                prev_cs = int(schedule.init_current_size)
                fsc_prev = _truncate_fsc_for_current_size_growth(
                    schedule.init_fsc,
                    current_size=prev_cs,
                    grid_size=grid_size,
                    dtype=_dense_global_scoring_dtype(),
                )
                data_vs_prior_iter = np.asarray(
                    fsc_to_relion_ssnr(fsc_prev, tau2_fudge=tau2_fudge),
                )
                previous_data_vs_prior_for_scheduling = data_vs_prior_iter
                res_shell = resolution_from_data_vs_prior(
                    data_vs_prior_iter,
                    allow_high_res_recovery=True,
                )
                relion_incr_size, relion_has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
                    fsc_prev,
                    prev_cs,
                    incr_size=relion_incr_size,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                )
                _init_pmax = float(schedule.init_ave_Pmax) if schedule.init_ave_Pmax is not None else 0.0
                raw_cs = compute_current_size_relion(
                    res_shell,
                    grid_size,
                    ave_Pmax=_init_pmax,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    incr_size=relion_incr_size,
                )
                current_size = quantize_current_size(raw_cs, ori_size=grid_size)
            else:
                current_size = _bootstrap_current_size_relion(schedule.init_current_size, grid_size)
                data_vs_prior_iter = None
        else:
            prev_cs = history.current_sizes[-1]
            if k_class_enabled:
                if previous_data_vs_prior_for_scheduling is None:
                    raise RuntimeError("K-class current-size scheduling requires a previous data_vs_prior curve")
                data_vs_prior_prev_raw = np.asarray(
                    previous_data_vs_prior_for_scheduling,
                    dtype=_dense_global_scoring_dtype(),
                ).copy()
                data_vs_prior_prev = data_vs_prior_prev_raw.copy()
                if prev_cs < grid_size:
                    data_vs_prior_prev[..., min(data_vs_prior_prev.shape[-1], prev_cs // 2 + 1) :] = 0.0
                per_class_res_shell = np.asarray(
                    [
                        resolution_from_data_vs_prior(dvp_class, allow_high_res_recovery=False)
                        for dvp_class in np.asarray(data_vs_prior_prev)
                    ],
                    dtype=np.int32,
                )
                res_shell = int(np.max(per_class_res_shell))
                scheduling_res_shell = _firstiter_cc_scheduling_resolution_shell(
                    res_shell,
                    emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
                    ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
                    relion_iteration=int(init_relion_iteration) + int(iteration),
                    grid_size=grid_size,
                    voxel_size=cryo.voxel_size,
                )
                if scheduling_res_shell != res_shell:
                    res_shell = scheduling_res_shell
                    logger.info(
                        "RELION firstiter_cc scheduling: using ini_high=%.2f A shell %d "
                        "for next K-class current_size",
                        float(parity.relion_firstiter_ini_high_angstrom),
                        int(res_shell),
                    )
                raw_cs = compute_current_size_relion(
                    res_shell,
                    grid_size,
                    ave_Pmax=state.ave_Pmax,
                    has_high_fsc_at_limit=False,
                    incr_size=relion_incr_size,
                )
                computed_cs = quantize_current_size(raw_cs, ori_size=grid_size)
                _kclass_dump_dir = os.environ.get("RECOVAR_KCLASS_DUMP_DIR")
                if _kclass_dump_dir:
                    reconstruction_diagnostics.write_kclass_current_size(
                        output_dir=_kclass_dump_dir,
                        computed_cs=computed_cs,
                        data_vs_prior_prev=data_vs_prior_prev,
                        data_vs_prior_prev_raw=data_vs_prior_prev_raw,
                        grid_size=grid_size,
                        iteration=iteration,
                        per_class_res_shell=per_class_res_shell,
                        prev_cs=prev_cs,
                        raw_cs=raw_cs,
                        relion_has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                        relion_incr_size=relion_incr_size,
                        res_shell=res_shell,
                        state=state,
                    )
                current_size = computed_cs
            else:
                fsc_prev_raw = np.asarray(
                    history.fsc_history[-1],
                    dtype=_dense_global_scoring_dtype(),
                ).copy()
                fsc_prev_for_growth = _truncate_fsc_for_current_size_growth(
                    history.fsc_for_growth_history[-1] if history.fsc_for_growth_history else fsc_prev_raw,
                    current_size=prev_cs,
                    grid_size=grid_size,
                    dtype=_dense_global_scoring_dtype(),
                )

                data_vs_prior_iter = _k1_data_vs_prior_for_scheduling(
                    raw_fsc=fsc_prev_raw,
                    corrected_data_vs_prior=previous_data_vs_prior_for_scheduling,
                    current_size=prev_cs,
                    grid_size=grid_size,
                    tau2_fudge=tau2_fudge,
                    dtype=_dense_global_scoring_dtype(),
                )
                previous_data_vs_prior_for_scheduling = data_vs_prior_iter
                res_shell = resolution_from_data_vs_prior(
                    data_vs_prior_iter,
                    allow_high_res_recovery=True,
                )
                relion_incr_size, relion_has_high_fsc_at_limit = update_relion_growth_state_from_fsc(
                    fsc_prev_for_growth,
                    prev_cs,
                    incr_size=relion_incr_size,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                )
                scheduling_res_shell = _firstiter_cc_scheduling_resolution_shell(
                    res_shell,
                    emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
                    ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
                    relion_iteration=int(init_relion_iteration) + int(iteration),
                    grid_size=grid_size,
                    voxel_size=cryo.voxel_size,
                )
                if scheduling_res_shell != res_shell:
                    res_shell = scheduling_res_shell
                    logger.info(
                        "RELION firstiter_cc scheduling: using ini_high=%.2f A shell %d for next current_size",
                        float(parity.relion_firstiter_ini_high_angstrom),
                        int(res_shell),
                    )

                raw_cs = compute_current_size_relion(
                    res_shell,
                    grid_size,
                    ave_Pmax=state.ave_Pmax,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    incr_size=relion_incr_size,
                )
                current_size = quantize_current_size(raw_cs, ori_size=grid_size)

        current_size = quantize_current_size(current_size, ori_size=grid_size)
        if iteration > 0:
            logger.info(
                "RELION current-size decision: iter=%d prev=%d res_shell=%d "
                "incr_size=%d high_fsc_at_limit=%s ave_Pmax=%.6f raw=%d quantized=%d",
                iteration + 1,
                int(prev_cs),
                int(res_shell),
                int(relion_incr_size),
                bool(relion_has_high_fsc_at_limit),
                float(state.ave_Pmax),
                int(raw_cs),
                int(current_size),
            )
        if adaptive.relion_current_sizes is not None:
            if iteration < len(adaptive.relion_current_sizes):
                oracle_cs = int(adaptive.relion_current_sizes[iteration])
            else:
                oracle_cs = int(adaptive.relion_current_sizes[-1])
            if oracle_cs <= 0:
                oracle_cs = int(schedule.init_current_size)
            current_size = quantize_current_size(oracle_cs, ori_size=grid_size)
            logger.info(
                "Current-size oracle: iteration %d using current_size=%d",
                iteration + 1,
                current_size,
            )

        # RELION updates image_coarse_size before updateAngularSampling at the
        # start of expectation(). Preserve that incoming sampling order even
        # when replay/native scheduling advances state.healpix_order below.
        coarse_size_healpix_order = relion_expectation_coarse_size_order(
            state_healpix_order=state.healpix_order,
            replay_saved_healpix_order=replay_saved_healpix_order,
        )

        # --- Replay override: force recovar's sampling state to mirror RELION ---
        # When replaying, RELION's per-iter sampling.star / model.star /
        # iter_replay_override dict dictate the actual hp_order, sigma priors,
        # translation grid, current_size, direction priors, noise, etc. used
        # at this iteration. Helper mutates state + relion_half_inputs +
        # direction-prior lists in place; returns explicit new values for
        # everything else.
        recovar_state_swap_snapshot = None
        state_swap_target_this_iteration = (
            debug.state_swap_probe is not None
            and int(debug.state_swap_probe.get("iteration", -1)) == int(iteration)
        )
        if state_swap_target_this_iteration:
            recovar_state_swap_snapshot = _snapshot_state_swap_inputs(
                state=state,
                cs=current_size,
                means=means,
                mean_variance=mean_variance,
                noise_variance_per_half=noise_variance_per_half,
                noise_variance=noise_variance,
                previous_noise_radial_per_half=previous_noise_radial_per_half,
                previous_noise_radial=previous_noise_radial,
                relion_half_inputs=relion_half_inputs,
                previous_best_rotations=previous_best_rotations,
                current_sigma_offset_angstrom=current_sigma_offset_angstrom,
                current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
                class_direction_prior_per_half=class_direction_prior_per_half,
                class_direction_prior_order_per_half=class_direction_prior_order_per_half,
                global_direction_prior_per_half=global_direction_prior_per_half,
                global_direction_prior_order_per_half=global_direction_prior_order_per_half,
            )
        replay_result = apply_iter_replay_overrides(
            iter_replay_override=iter_replay_override,
            perturb_replay_relion_dir=perturb_replay_relion_dir,
            perturb_replay_relion_prefix=perturb_replay_relion_prefix,
            init_relion_iteration=init_relion_iteration,
            iteration=iteration,
            state=state,
            cs=current_size,
            cryo=cryo,
            k_class_enabled=k_class_enabled,
            n_classes=n_classes,
            relion_half_inputs=relion_half_inputs,
            previous_best_rotations=previous_best_rotations,
            noise_variance_per_half=noise_variance_per_half,
            noise_variance=noise_variance,
            previous_noise_radial_per_half=previous_noise_radial_per_half,
            previous_noise_radial=previous_noise_radial,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            class_direction_prior_per_half=class_direction_prior_per_half,
            class_direction_prior_order_per_half=class_direction_prior_order_per_half,
            global_direction_prior_per_half=global_direction_prior_per_half,
            global_direction_prior_order_per_half=global_direction_prior_order_per_half,
            preserve_existing_direction_prior=replay.preserve_initial_direction_prior,
            sealed_sampling_state=sealed_sampling_state,
            dtype=_dense_global_scoring_dtype(),
        )
        current_size = replay_result.cs
        _replay_prior_translations = replay_result.prior_translations
        _replay_meta = replay_result.replay_meta
        previous_best_rotations = replay_result.previous_best_rotations
        noise_variance_per_half = replay_result.noise_variance_per_half
        noise_variance = replay_result.noise_variance
        previous_noise_radial_per_half = replay_result.previous_noise_radial_per_half
        previous_noise_radial = replay_result.previous_noise_radial
        replay_mean_variance = (
            None
            if iter_replay_override is None
            else iter_replay_override.get("mean_variance")
        )
        if replay_mean_variance is not None:
            replay_mean_variance = np.asarray(replay_mean_variance, dtype=np.float64).reshape(-1)
            expected_mean_variance_shape = tuple(mean_variance.shape)
            if replay_mean_variance.shape != expected_mean_variance_shape:
                raise ValueError(
                    "K=1 replay mean_variance shape mismatch: "
                    f"expected {expected_mean_variance_shape}, "
                    f"got {replay_mean_variance.shape}"
                )
            mean_variance = jnp.asarray(replay_mean_variance)
            logger.info("Replay override: K=1 tau2/mean_variance <- model.star")
        current_sigma_offset_angstrom = replay_result.current_sigma_offset_angstrom
        current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
            replay_result.current_sigma_offset_angstrom_per_half
        )
        if replay_saved_healpix_order is not None:
            replay_saved_healpix_order = int(state.healpix_order)
        if k_class_enabled and replay_result.class_weights is not None:
            class_weights = np.asarray(replay_result.class_weights, dtype=np.float64)
            class_log_priors = np.log(class_weights)
            logger.info(
                "Replay override: class priors <- direction-prior row sums (%s)",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_weights)),
            )

        means = _maybe_debug_replay_relion_references(
            means=means,
            perturb_replay_relion_dir=(
                None
                if sealed_sampling_state is not None and not state_swap_target_this_iteration
                else perturb_replay_relion_dir
            ),
            perturb_replay_relion_prefix=perturb_replay_relion_prefix,
            init_relion_iteration=init_relion_iteration,
            iteration=iteration,
            volume_shape=volume_shape,
            n_classes=n_classes,
            force=(
                state_swap_target_this_iteration
                and bool(debug.state_swap_probe.get("replay_relion_references", False))
            ),
        )

        (
            current_size,
            means,
            mean_variance,
            noise_variance_per_half,
            noise_variance,
            previous_noise_radial_per_half,
            previous_noise_radial,
            previous_best_rotations,
            current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half,
            class_direction_prior_per_half,
            class_direction_prior_order_per_half,
            global_direction_prior_per_half,
            global_direction_prior_order_per_half,
        ) = _apply_state_swap_probe(
            probe=debug.state_swap_probe,
            iteration=iteration,
            recovar_snapshot=recovar_state_swap_snapshot,
            state=state,
            cs=current_size,
            volume_shape=volume_shape,
            means=means,
            mean_variance=mean_variance,
            noise_variance_per_half=noise_variance_per_half,
            noise_variance=noise_variance,
            previous_noise_radial_per_half=previous_noise_radial_per_half,
            previous_noise_radial=previous_noise_radial,
            relion_half_inputs=relion_half_inputs,
            previous_best_rotations=previous_best_rotations,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            class_direction_prior_per_half=class_direction_prior_per_half,
            class_direction_prior_order_per_half=class_direction_prior_order_per_half,
            global_direction_prior_per_half=global_direction_prior_per_half,
            global_direction_prior_order_per_half=global_direction_prior_order_per_half,
        )
        if not parity.use_per_half_mean_variance:
            # State-swap diagnostics historically replace the one shared tau2.
            # Do not leave the scorer pointing at pre-swap aliases.
            mean_variance_per_half = _updated_mean_variance_per_half(
                mean_variance,
                mean_variance_per_half,
                use_per_half_mean_variance=False,
            )
        if state_swap_target_this_iteration:
            history.record_state_swap_probe_iteration(int(init_relion_iteration) + int(iteration) + 1)
        if frozen_initial_scoring_state is not None and iteration == 0:
            frozen_initial_scoring_state_sha256 = _assert_frozen_scoring_state_unchanged(
                frozen_initial_scoring_state,
                _frozen_scoring_state_arrays(
                    means=means,
                    mean_variance=mean_variance,
                    mean_variance_per_half=(
                        mean_variance_per_half if parity.use_per_half_mean_variance else None
                    ),
                    relion_half_inputs=relion_half_inputs,
                    noise_variance_per_half=noise_variance_per_half,
                    current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
                    global_direction_prior_per_half=global_direction_prior_per_half,
                    experiment_datasets=experiment_datasets,
                    sealed_sampling_state=sealed_sampling_state,
                    sealed_scoring_context=debug.sealed_scoring_context,
                ),
            )
            logger.info(
                "Frozen scoring-state ownership verified immediately before physical iteration %d scoring",
                int(init_relion_iteration) + iteration + 1,
            )

        exact_acc_rot_this_iter = None
        exact_acc_trans_this_iter = None
        exact_acc_rot_per_class_this_iter = None
        exact_acc_trans_per_class_this_iter = None
        exact_accuracy_class_counts_this_iter = None
        exact_accuracy_status_this_iter = "skipped_firstiter_cc"
        should_estimate_exact_accuracy = not relion_firstiter_cc_this_iter
        if native_sampling_boundary and should_estimate_exact_accuracy:
            previous_eulers_half1 = relion_half_inputs.previous_best_rotation_eulers[0]
            if expected_accuracy_trial_order is None or previous_eulers_half1 is None:
                exact_accuracy_status_this_iter = "unavailable_inputs"
                state.acc_rot = float("inf")
                state.acc_trans = float("inf")
                logger.warning(
                    "RELION exact expected accuracy unavailable at iteration %d; "
                    "convergence remains fail-closed",
                    iteration + 1,
                )
            else:
                if k_class_enabled:
                    accuracy_class_ids = class_assignments[0]
                    if accuracy_class_ids is None:
                        accuracy_class_ids = np.zeros(int(experiment_datasets[0].n_units), dtype=np.int32)
                else:
                    accuracy_class_ids = np.zeros(int(experiment_datasets[0].n_units), dtype=np.int32)
                try:
                    accuracy = estimate_relion_expected_accuracy(
                        reference_fourier=means[0],
                        volume_shape=tuple(volume_shape),
                        best_eulers_deg=previous_eulers_half1,
                        class_ids=accuracy_class_ids,
                        class_weights=class_weights,
                        sigma2_noise_native=previous_noise_radial_per_half[0],
                        dataset=experiment_datasets[0],
                        trial_order_local=expected_accuracy_trial_order,
                        current_image_size=int(current_size),
                        padding_factor=PROJECTION_PADDING_FACTOR,
                        sigma2_fudge=float(tau2_fudge),
                        random_seed=int(effective_optimizer_random_seed),
                        random_seed_particle_ids=expected_accuracy.half1_particle_ids,
                        ctf_params_override=expected_accuracy.half1_ctf_params,
                        do_ctf_correction=expected_accuracy.do_ctf_correction,
                    )
                    exact_acc_rot_this_iter = float(accuracy.acc_rot)
                    exact_acc_trans_this_iter = float(accuracy.acc_trans_angstrom)
                    exact_acc_rot_per_class_this_iter = np.asarray(
                        accuracy.acc_rot_per_class,
                        dtype=np.float64,
                    ).copy()
                    exact_acc_trans_per_class_this_iter = np.asarray(
                        accuracy.acc_trans_per_class_angstrom,
                        dtype=np.float64,
                    ).copy()
                    exact_accuracy_class_counts_this_iter = np.asarray(
                        accuracy.class_counts,
                        dtype=np.int64,
                    ).copy()
                    expected_accuracy_trial_local_indices = np.asarray(
                        accuracy.trial_local_indices,
                        dtype=np.int64,
                    ).copy()
                    expected_accuracy_trial_particle_ids = np.asarray(
                        accuracy.trial_particle_ids,
                        dtype=np.int64,
                    ).copy()
                    exact_accuracy_status_this_iter = "ok"
                    state.acc_rot = exact_acc_rot_this_iter
                    state.acc_trans = exact_acc_trans_this_iter
                    logger.info(
                        "RELION exact expected accuracy: acc_rot=%.3f deg, acc_trans=%.4f A "
                        "(trials=%d, first_particle_ids=%s)",
                        exact_acc_rot_this_iter,
                        exact_acc_trans_this_iter,
                        int(accuracy.trial_local_indices.size),
                        accuracy.trial_particle_ids[:5].tolist(),
                    )
                except Exception as exc:
                    exact_accuracy_status_this_iter = f"error:{type(exc).__name__}:{exc}"
                    state.acc_rot = float("inf")
                    state.acc_trans = float("inf")
                    logger.warning(
                        "RELION exact expected-accuracy estimation failed at iteration %d; "
                        "convergence remains fail-closed: %s",
                        iteration + 1,
                        exc,
                    )

        # RELION evaluates accuracy and updates sampling at the beginning of
        # Expectation (iterations > 1), using the previous iteration's stall
        # counters.  Sampling must therefore be prepared here, not after this
        # iteration's M-step statistics are recorded.
        if native_sampling_boundary and iteration > 0 and adaptive.relion_healpix_orders is None:
            state = update_angular_sampling(state)
        if adaptive.relion_healpix_orders is not None:
            target_healpix_order = int(adaptive.relion_healpix_orders[iteration])
            state = _apply_relion_healpix_order_oracle(
                state,
                target_healpix_order,
                iteration_number=iteration + 1,
            )
            logger.info(
                "HEALPix-order oracle: iteration %d using healpix_order=%d",
                iteration + 1,
                target_healpix_order,
            )

        history.record_scheduling(
            current_size,
            state.healpix_order,
            float(current_sigma_offset_angstrom),
            _copy_optional_float_pair(current_sigma_offset_angstrom_per_half),
        )
        scoring_current_size = int(current_size)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            iteration + 1,
            schedule.max_iter,
            scoring_current_size,
            state.healpix_order,
            state.do_local_search,
        )

        # --- Angular step refinement: regenerate rotation grid if needed ---
        # When update_refinement_state incremented healpix_order, we need
        # a new rotation grid at the finer level.
        # IMPORTANT: At order >= 5, the full grid has 2.4M+ rotations which
        # OOMs the GPU.  Instead, keep the order-4 grid as the "base" and
        # rely on local search + oversampling to achieve finer angular steps.
        # The order is still tracked for sigma calculation.
        if state.healpix_order != current_healpix_order:
            new_order = _exhaustive_grid_order_for_state(state)
            if new_order != current_healpix_order:
                logger.info(
                    "Regenerating rotation grid: order %d -> %d",
                    current_healpix_order,
                    new_order,
                )
                current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(
                    new_order, dtype=_dense_global_scoring_dtype()
                )
                current_healpix_order = new_order
            else:
                logger.info(
                    "Angular step refined to order %d (exhaustive grid stays at order %d — local search handles finer sampling)",
                    state.healpix_order,
                    current_healpix_order,
                )

            # Regenerate translation grid based on updated parameters
            base_translations = _translation_grid_for_class_count(
                state.translation_range,
                state.translation_step,
                n_classes=n_classes,
                source_units_per_pixel=(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
            ).astype(np.float64, copy=False)
            current_translations = jnp.asarray(base_translations, dtype=_dense_global_scoring_dtype())
            logger.info(
                "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
                current_rotations.shape[0],
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
        elif perturb_replay_relion_dir is not None and sealed_sampling_state is None:
            # Translation params may have changed under replay without an
            # hp_order bump. Regenerate the translation grid to match RELION.
            _new_t_source = _translation_grid_for_class_count(
                state.translation_range,
                state.translation_step,
                n_classes=n_classes,
                source_units_per_pixel=(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
            ).astype(np.float64, copy=False)
            _new_t = jnp.asarray(_new_t_source, dtype=_dense_global_scoring_dtype())
            if _new_t.shape != base_translations.shape or not jnp.allclose(
                _new_t,
                np.asarray(base_translations, dtype=_dense_global_scoring_dtype()),
            ):
                current_translations = _new_t
                base_translations = _new_t_source
                logger.info(
                    "Replay: regenerated translation grid: %d translations (range=%.2f px, step=%.2f px)",
                    current_translations.shape[0],
                    state.translation_range,
                    state.translation_step,
                )

        # --- Local angular search bookkeeping ---
        # Once RELION enters local search, each image should search around its
        # own previous orientation on the true current HEALPix order. Use the
        # exact rotations selected in the previous iteration, not the nearest
        # snapped grid indices.
        effective_rotations = current_rotations
        effective_rotation_eulers = np.asarray(
            current_rotation_eulers,
            dtype=_dense_global_scoring_dtype(),
        )
        effective_mstep_rotations = None
        adaptive_pass1_rotations = None
        rotation_log_prior_per_half = [None, None]
        class_rotation_log_prior_per_half = [None, None]
        use_local = state.do_local_search and all(
            eulers is not None for eulers in relion_half_inputs.previous_best_rotation_eulers
        )
        adaptive_pass1_source_eulers = np.asarray(effective_rotation_eulers, dtype=np.float64)
        # --- Apply RELION SamplingPerturbation to the trial grid for this iter ---
        # healpix_sampling.cpp:1909-1934 (rotations) + 1810-1820 (translations)
        # Perturbation is a rigid rotation of SO(3): A := A @ R_perturb applied
        # AFTER oversampling. At adaptive_oversampling=0 (os0 RELION runs),
        # the coarse grid IS the trial grid so we apply directly here.
        if _replay_meta is not None and _replay_meta.get("sealed_v3", False):
            random_perturbation = float(_replay_meta["random_perturbation"])
            replay_perturbation_source = "sealed_frozen_boundary_v3"
            logger.info(
                "Perturbation replay: iter=%d rp=%+.12g pf=%.3f relion_hp_order=%d source=%s",
                iteration + 1,
                random_perturbation,
                float(_replay_meta["perturbation_factor"]),
                int(_replay_meta["healpix_order"]),
                replay_perturbation_source,
            )
        elif _replay_meta is not None:
            replay_relion_iteration = int(init_relion_iteration) + int(iteration) + 1
            replay_restart_state_iteration = _perturbation_restart_state_iteration(
                parity.perturb_replay_restart_state_iterations,
                replay_relion_iteration,
            )
            random_perturbation, replay_perturbation_source = _resolve_replay_random_perturbation(
                star_value=float(_replay_meta["random_perturbation"]),
                perturbation_factor=float(_replay_meta["perturbation_factor"]),
                relion_iteration=replay_relion_iteration,
                replay_dir=str(perturb_replay_relion_dir),
                replay_prefix=perturb_replay_relion_prefix,
                explicit_seed=parity.perturb_seed,
                precision_mode=str(parity.perturb_replay_precision),
                restart_state_iteration=replay_restart_state_iteration,
            )
            logger.info(
                "Perturbation replay: iter=%d rp=%+.12g pf=%.3f relion_hp_order=%d source=%s",
                iteration + 1,
                random_perturbation,
                float(_replay_meta["perturbation_factor"]),
                int(_replay_meta["healpix_order"]),
                replay_perturbation_source,
            )
        elif parity.perturb_factor > 0:
            relion_iter = int(init_relion_iteration) + iteration + 1
            if parity.perturb_seed is not None:
                seed = int(parity.perturb_seed) + relion_iter
                random_perturbation = advance_relion_perturbation_from_seed(
                    random_perturbation,
                    parity.perturb_factor,
                    seed=seed,
                )
                logger.info(
                    "Perturbation advance: iter=%d relion_iter=%d seed=%d rp=%+.5f",
                    iteration + 1,
                    relion_iter,
                    seed,
                    random_perturbation,
                )
            else:
                random_perturbation = advance_relion_perturbation(random_perturbation, parity.perturb_factor, perturb_rng)
                logger.info("Perturbation advance: iter=%d rp=%+.5f", iteration + 1, random_perturbation)
        if _replay_meta is not None or parity.perturb_factor > 0:
            # Use RELION's actual hp_order when replaying (recovar's current
            # grid order may be capped at MAX_FULL_GRID_ORDER=4 for memory).
            _angsamp_order = int(_replay_meta["healpix_order"]) if _replay_meta is not None else current_healpix_order
            angsamp_deg = relion_angular_sampling_deg(_angsamp_order, adaptive_oversampling=0)
            if effective_rotation_eulers is not None:
                mstep_source_eulers = (
                    np.asarray(effective_rotation_eulers, dtype=np.float64)
                    if sealed_sampling_state is not None
                    else _get_relion_rotation_grid_eulers_float64(_angsamp_order)
                )
                if int(mstep_source_eulers.shape[0]) != int(effective_rotation_eulers.shape[0]):
                    mstep_source_eulers = np.asarray(effective_rotation_eulers, dtype=np.float64)
                effective_rotations, effective_rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
                    effective_rotation_eulers,
                    random_perturbation,
                    angsamp_deg,
                    dtype=_dense_global_scoring_dtype(),
                )
                _, _, effective_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
                    mstep_source_eulers,
                    random_perturbation,
                    angsamp_deg,
                    return_mstep_rotations=True,
                    dtype=_dense_global_scoring_dtype(),
                )
            else:
                effective_rotations = apply_relion_rotation_perturbation(
                    np.asarray(effective_rotations),
                    random_perturbation,
                    angsamp_deg,
                ).astype(_dense_global_scoring_dtype(), copy=False)
                effective_rotation_eulers = utils.R_to_relion(np.asarray(effective_rotations), degrees=True).astype(
                    _dense_global_scoring_dtype()
                )
            _perturbed_translations = apply_relion_translation_perturbation(
                np.asarray(base_translations),
                random_perturbation,
                float(state.translation_step),
            )
            current_translations = jnp.asarray(
                _perturbed_translations,
                dtype=_dense_global_scoring_dtype(),
            )
        if not use_local and int(state.adaptive_oversampling) > 0:
            adaptive_pass1_order = (
                int(_replay_meta["healpix_order"])
                if _replay_meta is not None
                else int(current_healpix_order)
            )
            adaptive_pass1_use_float64 = bool(_DENSE_EM_STATIC_KWARGS["use_float64_scoring"])
            adaptive_pass1_rotations = _relion_adaptive_pass1_rotations(
                adaptive_pass1_source_eulers,
                random_perturbation if (_replay_meta is not None or parity.perturb_factor > 0) else 0.0,
                relion_angular_sampling_deg(adaptive_pass1_order, adaptive_oversampling=0),
                use_float64=adaptive_pass1_use_float64,
            )
            if adaptive_pass1_rotations is not None:
                logger.info(
                    "RELION adaptive pass 1: using %s-built coarse scorer rotations; "
                    "fine/M-step rotations remain host-generated",
                    "double-precision CUDA" if adaptive_pass1_use_float64 else "CUDA",
                )
        # NOTE: previously this branch restricted the translation grid to a single
        # perturbed shift at iter 1 with --firstiter_cc. That was a misguided
        # emulation; RELION's ml_optimiser.cpp:9181-9207 evaluates the FULL
        # translation grid at iter 1 then binarizes exp_Mweight to the single
        # best (class, pose) afterward. The restriction broke the K-class adaptive
        # engine's trans_parent_map (oversampled fine→coarse map) because the
        # restricted grid had length 1 while the parent_map values reached 28.
        # run_k_class_parity (the working 0.998 single-step path) does NOT
        # restrict translations either. Keeping the full grid here.
        local_search_order = None
        local_search_rotations = None
        local_search_rotation_eulers = None
        local_search_mstep_rotations = None
        model_current_size_for_engine = (
            scoring_current_size if scoring_current_size < cryo.image_shape[0] else None
        )
        image_current_size = int(scoring_current_size)
        if optics_image_sizes is not None:
            remapped_image_sizes = relion_optics_image_current_sizes(
                scoring_current_size,
                model_ori_size=grid_size,
                model_pixel_size=model_pixel_size,
                optics_image_sizes=optics_image_sizes,
                optics_pixel_sizes=optics_pixel_sizes,
            )
            unique_remapped_sizes = np.unique(remapped_image_sizes)
            if unique_remapped_sizes.size != 1:
                raise NotImplementedError(
                    "K=1 parity currently requires all optics groups to share one remapped "
                    f"image current size; got {remapped_image_sizes.tolist()}",
                )
            image_current_size = int(unique_remapped_sizes[0])
        cs_for_engine = image_current_size if image_current_size < cryo.image_shape[0] else None
        if image_current_size != int(scoring_current_size) and model_current_size_for_engine is None:
            # ``None`` normally means full-box support, but the EM engines also
            # interpret it as "reuse the particle-image score cutoff". Keep
            # the full model size explicit when optics remapping makes those
            # two cutoffs differ.
            model_current_size_for_engine = int(scoring_current_size)
        if image_current_size != int(scoring_current_size):
            logger.info(
                "RELION optics current-size remap: model_current_size=%d "
                "image_current_size=%d model_pixel_size=%.9g",
                int(scoring_current_size),
                image_current_size,
                model_pixel_size,
            )
        sigma_rot = state.sigma_rot
        sigma_psi = state.sigma_psi if state.sigma_psi > 0 else sigma_rot
        if use_local and sigma_rot <= 0:
            step_rad = np.deg2rad(healpix_angular_step(state.healpix_order) / (2**state.adaptive_oversampling))
            sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
            sigma_psi = sigma_rot

        if use_local:
            local_search_order = state.healpix_order + state.adaptive_oversampling
            local_pass1_current_size = cs_for_engine
            local_search_random_perturbation = 0.0
            local_search_angular_sampling_deg = None
            use_parent_expanded_local = state.adaptive_oversampling > 0
            if effective_rotations.shape[0] != rotation_grid_size(local_search_order):
                logger.info(
                    "Using lazy fine local-search grid: order=%d (%d rotations) from capped base order=%d",
                    local_search_order,
                    rotation_grid_size(local_search_order),
                    current_healpix_order,
                )
                local_search_angular_sampling_deg = relion_angular_sampling_deg(
                    local_search_order,
                    adaptive_oversampling=0,
                )
                if (not use_parent_expanded_local) and _precompute_exact_local_fine_grid_enabled(local_search_order):
                    _, local_search_rotation_eulers = _relion_rotation_grid_float32(local_search_order)
                    local_search_rotations, local_search_rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
                        local_search_rotation_eulers,
                        float(random_perturbation),
                        local_search_angular_sampling_deg,
                    )
                    _, _, local_search_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
                        _get_relion_rotation_grid_eulers_float64(local_search_order),
                        float(random_perturbation),
                        local_search_angular_sampling_deg,
                        return_mstep_rotations=True,
                    )
                    local_search_random_perturbation = 0.0
                else:
                    local_search_rotations = None
                    local_search_rotation_eulers = None
                    local_search_random_perturbation = float(random_perturbation)
                    if use_parent_expanded_local:
                        logger.info(
                            "RELION local search: expanding selected coarse parents by oversampling_order=%d",
                            int(state.adaptive_oversampling),
                        )
                        parent_order = local_search_order - int(state.adaptive_oversampling)
                        local_pass1_current_size = relion_local_pass1_current_size(
                            pre_update_healpix_order=coarse_size_healpix_order,
                            pixel_size=(
                                float(optics_pixel_sizes[0])
                                if optics_pixel_sizes is not None
                                else model_pixel_size
                            ),
                            ori_size=(
                                int(optics_image_sizes[0])
                                if optics_image_sizes is not None
                                else grid_size
                            ),
                            particle_diameter=particle_diameter_ang,
                            current_size=image_current_size if cs_for_engine is not None else None,
                        )
                        logger.info(
                            "Local adaptive oversampling: pass 1 at coarse_size=%s, "
                            "pass 2 at current_size=%s (size_order=%d, parent_order=%d, oversampling=%d)",
                            local_pass1_current_size,
                            cs_for_engine,
                            coarse_size_healpix_order,
                            parent_order,
                            int(state.adaptive_oversampling),
                        )
            else:
                local_search_rotations = effective_rotations
                local_search_rotation_eulers = None
                if effective_mstep_rotations is not None:
                    local_search_mstep_rotations = effective_mstep_rotations
                else:
                    mstep_source_eulers = _get_relion_rotation_grid_eulers_float64(local_search_order)
                    if int(mstep_source_eulers.shape[0]) != int(effective_rotation_eulers.shape[0]):
                        mstep_source_eulers = np.asarray(effective_rotation_eulers, dtype=np.float64)
                    _, _, local_search_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
                        mstep_source_eulers,
                        0.0,
                        relion_angular_sampling_deg(local_search_order, adaptive_oversampling=0),
                        return_mstep_rotations=True,
                    )
            logger.info(
                "Local search (batched exact): fine_order=%d, sigma_rot=%.4f rad (%.2f deg), sigma_psi=%.4f rad",
                local_search_order,
                sigma_rot,
                np.rad2deg(sigma_rot),
                sigma_psi,
            )
        direction_prior_healpix_order = _direction_prior_healpix_order_for_scoring(
            use_local=use_local,
            current_healpix_order=current_healpix_order,
            state_healpix_order=state.healpix_order,
            adaptive_oversampling=state.adaptive_oversampling,
            local_search_order=local_search_order,
        )
        coarse_rotation_ids_for_scoring = (
            _sealed_sampling_rotation_ids(sealed_sampling_state)
            if sealed_sampling_state is not None and not use_local
            else None
        )
        if (
            coarse_rotation_ids_for_scoring is not None
            and coarse_rotation_ids_for_scoring.shape != (int(effective_rotations.shape[0]),)
        ):
            raise RuntimeError(
                "sealed captured rotation IDs do not match the directly materialized scorer grid"
            )

        for _half_idx in range(2):
            if use_local:
                continue
            if k_class_enabled:
                class_prior_k = class_direction_prior_per_half[_half_idx]
                class_prior_order_k = class_direction_prior_order_per_half[_half_idx]
                if class_prior_k is None and global_direction_prior_per_half[_half_idx] is not None:
                    shared_prior = np.asarray(
                        global_direction_prior_per_half[_half_idx], dtype=_dense_global_scoring_dtype()
                    )
                    class_prior_k = np.broadcast_to(shared_prior[None, :], (n_classes, shared_prior.size)).copy()
                    class_prior_order_k = global_direction_prior_order_per_half[_half_idx]
                if class_prior_k is not None and class_prior_order_k == direction_prior_healpix_order:
                    class_rotation_log_prior_per_half[_half_idx] = np.stack(
                        [
                            make_relion_direction_log_prior(
                                class_prior_k[class_idx],
                                direction_prior_healpix_order,
                                dtype=_dense_global_scoring_dtype(),
                            )
                            for class_idx in range(n_classes)
                        ],
                        axis=0,
                    )
                    logger.info(
                        "Using learned per-class global direction prior half-%d: %d classes, %d directions at healpix_order=%d",
                        _half_idx + 1,
                        n_classes,
                        class_prior_k.shape[1],
                        direction_prior_healpix_order,
                    )
                    continue
            prior_k = global_direction_prior_per_half[_half_idx]
            prior_order_k = global_direction_prior_order_per_half[_half_idx]
            if prior_k is None or prior_order_k != direction_prior_healpix_order:
                continue
            rotation_log_prior_per_half[_half_idx] = (
                _sealed_direction_log_prior(prior_k, sealed_sampling_state, dtype=_dense_global_scoring_dtype())
                if sealed_sampling_state is not None
                else make_relion_direction_log_prior(
                    prior_k,
                    direction_prior_healpix_order,
                    dtype=_dense_global_scoring_dtype(),
                )
            )
            logger.info(
                "Using learned global direction prior half-%d: %d directions at healpix_order=%d",
                _half_idx + 1,
                prior_k.shape[0],
                direction_prior_healpix_order,
            )

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        iter_sig_counts = None
        iter_sig_count_parts: list[np.ndarray] = []
        iter_recorded_sig_counts = None
        iter_recorded_sig_count_parts: list[np.ndarray] = []
        use_adaptive = state.adaptive_oversampling > 0 and not use_local and effective_rotations.shape[0] > 16
        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        per_half = PerHalfOutputs.empty()
        hard_assignments = per_half.hard_assignments
        class_assignments = per_half.class_assignments
        class_posterior_per_half = per_half.class_posterior
        class_full_posterior_per_half = per_half.class_full_posterior
        max_posterior_per_half = per_half.max_posterior
        rotation_posterior_per_half = per_half.rotation_posterior
        class_rotation_posterior_per_half = per_half.class_rotation_posterior
        pose_rotations = per_half.pose_rotations  # rotations to use with ha for poses
        pose_rotation_eulers = per_half.pose_rotation_eulers
        best_pose_rotations = per_half.best_pose_rotations
        best_pose_rotation_eulers = per_half.best_pose_rotation_eulers
        best_pose_translations = per_half.best_pose_translations
        translation_search_bases = per_half.translation_search_bases
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = per_half.coarse_ha
        class_posterior_per_half = per_half.class_posterior

        if use_adaptive:
            # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
            # Pass 1: coarse E-step at reduced resolution to find
            #         significant orientations.
            # Pass 2: oversampled E+M at full current_size for significant
            #         orientations only.

            # RELION sizes pass 1 before updating angular sampling. Keep the
            # incoming order for Fourier sizing; the updated order still
            # controls effective_rotations and the oversampled candidate grid.
            effective_step_deg = healpix_angular_step(coarse_size_healpix_order)
            pixel_size = cryo.voxel_size if cryo.voxel_size > 0 else 1.0
            coarse_size = compute_coarse_image_size(
                effective_step_deg,
                (
                    float(optics_pixel_sizes[0])
                    if optics_pixel_sizes is not None
                    else pixel_size
                ),
                (
                    int(optics_image_sizes[0])
                    if optics_image_sizes is not None
                    else grid_size
                ),
                particle_diameter=particle_diameter_ang,
            )
            coarse_size = clamp_relion_coarse_image_size(
                coarse_size,
                image_current_size if cs_for_engine is not None else None,
                grid_size,
            )
            if sealed_sampling_state is not None:
                coarse_size = int(sealed_sampling_state["coarse_size"])
                if coarse_size > int(current_size):
                    raise ValueError(
                        "sealed sampling coarse_size exceeds active current_size: "
                        f"coarse={coarse_size} current={current_size}"
                    )
                logger.info(
                    "Frozen-boundary v3 directly owns adaptive pass-1 coarse_size=%d",
                    coarse_size,
                )
            coarse_cs = coarse_size if coarse_size < grid_size else None

            logger.info(
                "Adaptive oversampling: pass 1 at coarse_size=%s, "
                "pass 2 at current_size=%s (oversampling=%d, particle_diameter=%s)",
                coarse_cs,
                cs_for_engine,
                state.adaptive_oversampling,
                (f"{float(particle_diameter_ang):.1f} A" if particle_diameter_ang is not None else "box_size"),
            )

        # D.2: per-class noise stats (K-tuple of NoiseStats per half) for the
        # per-class sigma_offset C1 update at end-of-iter. K=1 paths leave
        # this None; K-class paths populate from k_class_result.noise_stats.
        noise_stats_per_half = per_half.noise_stats
        noise_stats_per_half_per_class = per_half.noise_stats_per_class

        relion_projector_half_by_half = [None, None]
        relion_projector_r_max_by_half = [None, None]
        captured_projector_state = replay_result.relion_projector_state
        if captured_projector_state is not None and not (use_local or use_adaptive):
            raise RuntimeError(
                "captured RELION Projector::data was supplied but this iteration has no projector scoring path"
            )
        if use_local or use_adaptive:
            projector_t0 = time.time()
            if captured_projector_state is not None:
                (
                    relion_projector_half_by_half,
                    relion_projector_r_max_by_half,
                ) = _validate_captured_relion_projector_for_iteration(
                    captured_projector_state,
                    current_size=model_current_size_for_engine,
                    volume_shape=volume_shape,
                    padding_factor=PROJECTION_PADDING_FACTOR,
                    n_classes=n_classes,
                )
                logger.info(
                    "RELION mode: using captured exact Projector::data at current_size=%s "
                    "r_max=%s manifest=%s",
                    model_current_size_for_engine,
                    relion_projector_r_max_by_half[0],
                    captured_projector_state.source_manifest_sha256,
                )
            else:
                for _half_idx in range(2):
                    if experiment_datasets[_half_idx].n_units == 0:
                        logger.info(
                            "RELION mode: skipping Projector::data build for empty half-%d dataset",
                            _half_idx + 1,
                        )
                        continue
                    projector_half, projector_r_max = _relion_projector_half_maps_for_scoring(
                        means[_half_idx],
                        volume_shape=volume_shape,
                        current_size=model_current_size_for_engine,
                        padding_factor=PROJECTION_PADDING_FACTOR,
                        n_classes=n_classes,
                        real_references=(
                            initial_real_references_by_half[_half_idx]
                            if iteration == 0
                            else None
                        ),
                        dump_label=f"iter{iteration:03d}_half{_half_idx}",
                    )
                    relion_projector_half_by_half[_half_idx] = projector_half
                    relion_projector_r_max_by_half[_half_idx] = projector_r_max
                logger.info(
                    "RELION mode: built exact Projector::data for scoring at current_size=%s r_max=%s in %.2fs",
                    model_current_size_for_engine,
                    relion_projector_r_max_by_half[0],
                    time.time() - projector_t0,
                )

        # Freeze the exact iteration-start curve used by RELION's scale XA/AA
        # shell gate.  The scheduling variable is updated again after the
        # reconstruction, before parity diagnostics are written.
        scale_correction_data_vs_prior_this_iter = previous_data_vs_prior_for_scheduling

        diagnostic_half_indices = _significance_dump_half_indices(
            numbered_iteration=numbered_relion_iteration,
            n_classes=n_classes,
            experiment_datasets=experiment_datasets,
        )
        for k in diagnostic_half_indices:
            bpref_diagnostics.set_bpref_contribution_dump_context(
                iteration=iteration + 1,
                half=k + 1,
            )
            bpref_device_signature_active = (
                _bpref_device_signature_active_for_numbered_half(
                    iteration=iteration + 1,
                    half=k + 1,
                )
            )
            logger.info(
                "BPREF_DEVICE_SIGNATURE_ACTIVATION iteration=%d half=%d "
                "final_all_data=false active=%s",
                iteration + 1,
                k + 1,
                str(bpref_device_signature_active).lower(),
            )
            noise_variance_k = noise_variance_per_half[k]
            mean_variance_k = _mean_variance_for_scoring_half(mean_variance_per_half, k)
            rotation_log_prior_k = rotation_log_prior_per_half[k]
            class_rotation_log_prior_k = class_rotation_log_prior_per_half[k]
            previous_translations_k = relion_half_inputs.previous_best_translations[k]
            translation_search_base = relion_translation_search_base(
                previous_translations_k, dtype=_dense_global_scoring_dtype()
            )
            translation_search_bases[k] = translation_search_base
            sigma_offset_k = _sigma_offset_for_half(
                current_sigma_offset_angstrom,
                current_sigma_offset_angstrom_per_half,
                k,
            )
            current_translation_range = float(state.translation_range)
            k_class_image_batch_size = batching.image_batch_size
            dense_k_class_rotation_block_size = batching.rotation_block_size
            significance_image_batch_size = None
            significance_rotation_block_size = None
            if use_adaptive:
                adaptive_batch_plan = _plan_adaptive_dense_batch_sizes(
                    n_rot=effective_rotations.shape[0],
                    n_trans=current_translations.shape[0],
                    n_classes=n_classes,
                    image_shape=experiment_datasets[k].image_shape,
                    cs_for_engine=cs_for_engine,
                    coarse_cs=coarse_cs,
                    k_class_enabled=k_class_enabled,
                    safe_batch_sizes=_safe_batch_sizes,
                )
                k_class_image_batch_size = adaptive_batch_plan.pass2_image_batch_size
                dense_k_class_rotation_block_size = adaptive_batch_plan.pass2_rotation_block_size
                significance_image_batch_size = adaptive_batch_plan.significance_image_batch_size
                significance_rotation_block_size = adaptive_batch_plan.significance_rotation_block_size
            elif k_class_enabled:
                k_class_image_batch_size, dense_k_class_rotation_block_size = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                    classes=n_classes,
                    image_shape_for_batch=experiment_datasets[k].image_shape,
                    current_size_for_batch=cs_for_engine,
                )
                k_class_image_batch_size = min(
                    k_class_image_batch_size,
                    _safe_firstiter_cc_image_batch_size(
                        current_translations.shape[0],
                        experiment_datasets[k].image_shape,
                    ),
                )
                dense_k_class_rotation_block_size = min(
                    dense_k_class_rotation_block_size,
                    _safe_dense_k_class_rotation_block_size(
                        current_translations.shape[0],
                        k_class_image_batch_size,
                    ),
                )
            if k_class_enabled:
                if k_class_image_batch_size != batching.image_batch_size:
                    logger.info(
                        "STRICT-PARITY: clamping dense K-class image_batch_size from %d to %d",
                        batching.image_batch_size,
                        k_class_image_batch_size,
                    )
                if dense_k_class_rotation_block_size != batching.rotation_block_size:
                    logger.info(
                        "STRICT-PARITY: clamping dense K-class rotation_block_size from %d to %d",
                        batching.rotation_block_size,
                        dense_k_class_rotation_block_size,
                    )
                if use_adaptive and (
                    significance_image_batch_size != k_class_image_batch_size
                    or significance_rotation_block_size != dense_k_class_rotation_block_size
                ):
                    logger.info(
                        "RELION adaptive pass-1 significance batch sizing: "
                        "image_batch_size=%d rotation_block_size=%d "
                        "(pass2 image_batch_size=%d rotation_block_size=%d, "
                        "coarse_current_size=%s fine_current_size=%s)",
                        significance_image_batch_size,
                        significance_rotation_block_size,
                        k_class_image_batch_size,
                        dense_k_class_rotation_block_size,
                        coarse_cs,
                        cs_for_engine,
                    )
            # RELION translation prior sigma (ml_optimiser.cpp:7737-7746):
            # RELION checks `offset_range_x` (rlnOffsetRangeX in optimiser.star),
            # NOT the search-grid `offset_range` (rlnOffsetRange in sampling.star).
            # When offset_range_x > 0: sigma² = range_x²/9 (per-axis override)
            # When offset_range_x <= 0: sigma² = model.sigma2_offset (learned)
            # For this dataset, rlnOffsetRangeX = -1 → model sigma is used.
            # In split-half auto-refine, RELION keeps this in each half-model.
            #
            # Evaluate scoring and sigma-offset priors with their separate
            # RELION source formulas. `pdf_offset` scores the unperturbed
            # coarse sampling grid, while `wsum_sigma2_offset` accumulates
            # getTranslationsInPixel() shifts in storeWeightedSums.
            trans_prior_center = relion_translation_prior_center(
                previous_translations_k,
                cryo.voxel_size,
                dtype=_dense_global_scoring_dtype(),
            )
            local_trans_prior_center = relion_translation_prior_center(
                previous_translations_k,
                cryo.voxel_size,
                dtype=_dense_global_scoring_dtype(),
            )
            trans_sigma_center = relion_sigma_offset_prior_center(
                previous_translations_k, dtype=_dense_global_scoring_dtype()
            )
            # A.1 fix: at iter 1 cold-start `previous_translations_k` is None, so
            # `trans_sigma_center` is None and em_engine's wsum_sigma2_offset
            # accumulator (em_engine.py:1636) is gated off. RELION still computes
            # wsum_sigma2_offset = sum_i E[||t_i||²] at iter 1 using the implicit
            # zero prior center, which seeds iter-2's sigma_offset ~ 1.6 Å (vs
            # default 10 Å). Pass a zero-centered prior to the engine so the
            # noise accumulator fires. Keep the score log-prior path separate:
            # prior_centers=None means RELION's cold-start flat offset prior,
            # while an explicit zero center means a real Gaussian offset prior.
            trans_prior_center_for_engine = (
                np.zeros(2, dtype=_dense_global_scoring_dtype())
                if trans_sigma_center is None
                else trans_sigma_center
            )
            translation_prior_translations = np.asarray(base_translations, dtype=_dense_global_scoring_dtype())
            if current_translations.shape[0] != base_translations.shape[0]:
                if current_translations.shape[0] == 1 and base_translations.shape[0] > 1:
                    center_idx = int(base_translations.shape[0] // 2)
                    translation_prior_translations = np.asarray(
                        base_translations[center_idx : center_idx + 1],
                        dtype=_dense_global_scoring_dtype(),
                    )
                else:
                    translation_prior_translations = np.asarray(
                        current_translations, dtype=_dense_global_scoring_dtype()
                    )
            translation_log_prior = None
            if not use_local:
                translation_log_prior = make_relion_translation_log_prior(
                    translation_prior_translations,
                    cryo.voxel_size,
                    sigma_offset_k,
                    trans_prior_center,
                    offset_range_pixels=None,
                    dtype=_dense_global_scoring_dtype(),
                )
            if experiment_datasets[k].n_units == 0:
                logger.info("Skipping E-step/M-step accumulation for empty half-%d dataset", k + 1)
                n_shells = int(cryo.image_shape[0] // 2 + 1)
                n_rot_for_stats = int(
                    rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]
                )
                empty_k1_x_half_mstep = (
                    (not k_class_enabled)
                    and (use_local or use_adaptive)
                    and _k1_relion_x_half_mstep_enabled()
                )
                empty_mstep_accumulator_shape = (
                    relion_backprojector_volume_shape(
                        experiment_datasets[k].volume_shape,
                        PADDING_FACTOR,
                        current_size=(
                            cs_for_engine
                            if model_current_size_for_engine is None
                            else model_current_size_for_engine
                        ),
                    )
                    if empty_k1_x_half_mstep
                    else None
                )
                if k_class_enabled:
                    Ft_y_k = None
                    Ft_ctf_k = None
                elif empty_k1_x_half_mstep:
                    empty_x_half_shape = half_volume_accumulator_shape(empty_mstep_accumulator_shape)
                    empty_x_half_accumulator_shape = (int(np.prod(empty_x_half_shape)),)
                    Ft_y_x_half = jnp.zeros(empty_x_half_accumulator_shape, dtype=jnp.complex128)
                    Ft_ctf_x_half = jnp.zeros(empty_x_half_accumulator_shape, dtype=jnp.complex128)
                    Ft_y_k, Ft_ctf_k = relion_x_half_accumulators_to_public_layout(
                        Ft_y_x_half,
                        Ft_ctf_x_half,
                        empty_mstep_accumulator_shape,
                    )
                else:
                    accumulator_shape = (int(np.prod(padded_volume_shape)),)
                    Ft_y_k = jnp.zeros(accumulator_shape, dtype=jnp.complex128)
                    Ft_ctf_k = jnp.zeros(accumulator_shape, dtype=jnp.complex128)
                ha_k = np.zeros(0, dtype=np.int32)
                class_assignments[k] = np.zeros(0, dtype=np.int32)
                class_posterior_per_half[k] = np.zeros(n_classes, dtype=np.float32)
                class_full_posterior_per_half[k] = np.zeros(n_classes, dtype=np.float32)
                class_rotation_posterior_per_half[k] = np.zeros((n_classes, n_rot_for_stats), dtype=np.float32)
                em_stats_k = make_relion_stats(
                    log_evidence_per_image=jnp.zeros(0, dtype=jnp.float32),
                    best_log_score_per_image=jnp.zeros(0, dtype=jnp.float32),
                    max_posterior_per_image=jnp.zeros(0, dtype=jnp.float32),
                    rotation_posterior_sums=jnp.zeros(n_rot_for_stats, dtype=jnp.float32),
                )
                noise_stats_k = make_noise_stats(
                    wsum_sigma2_noise=jnp.zeros(n_shells, dtype=jnp.float32),
                    wsum_img_power=jnp.zeros(n_shells, dtype=jnp.float32),
                    wsum_sigma2_offset=0.0,
                    sumw=0.0,
                )
                coarse_ha[k] = ha_k
                empty_result = HalfScoreResult(
                    ha=ha_k,
                    Ft_y=Ft_y_k,
                    Ft_ctf=Ft_ctf_k,
                    em_stats=em_stats_k,
                    noise_stats=noise_stats_k,
                    best_pose_rotations=np.zeros((0, 3, 3), dtype=np.float32),
                    best_pose_rotation_eulers=np.zeros((0, 3), dtype=np.float32),
                    best_pose_translations=np.zeros((0, current_translations.shape[1]), dtype=np.float32),
                    mstep_full_half_axis=0 if empty_k1_x_half_mstep else None,
                    mstep_accumulator_shape=empty_mstep_accumulator_shape,
                )
                per_half.update_from(k, empty_result, dtype=_dense_global_scoring_dtype())
                if k == 0:
                    Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
                else:
                    Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k
                _parity_dump.collect_e_step(
                    half=k,
                    em_stats=em_stats_k,
                    hard_assignment=ha_k,
                    coarse_hard_assignment=coarse_ha[k],
                    noise_stats=noise_stats_k,
                    Ft_y=Ft_y_k,
                    Ft_ctf=Ft_ctf_k,
                    pose_rotation_eulers=pose_rotation_eulers[k],
                    best_pose_rotation_eulers=best_pose_rotation_eulers[k],
                    best_pose_translations=best_pose_translations[k],
                    translation_search_base=translation_search_bases[k],
                    original_image_indices=np.zeros(0, dtype=np.int64),
                )
                continue
            if use_local:
                local_parent_oversampling_order = int(state.adaptive_oversampling) if state.adaptive_oversampling > 0 else 0
                local_result = _score_half_local_in_bpref_scope(
                    bpref_device_signature_active=bpref_device_signature_active,
                    k=k,
                    experiment_dataset=experiment_datasets[k],
                    means_k=means[k],
                    mean_variance=mean_variance_k,
                    noise_variance_k=noise_variance_k,
                    previous_best_rotation_eulers_k=relion_half_inputs.previous_best_rotation_eulers[k],
                    local_search_rotations=local_search_rotations,
                    local_search_rotation_eulers=local_search_rotation_eulers,
                    local_search_mstep_rotations=local_search_mstep_rotations,
                    local_search_order=local_search_order,
                    sigma_rot=sigma_rot,
                    sigma_psi=sigma_psi,
                    current_translations=current_translations,
                    base_translations=base_translations,
                    trans_prior_center=local_trans_prior_center,
                    trans_prior_center_for_engine=trans_prior_center_for_engine,
                    current_sigma_offset_angstrom=sigma_offset_k,
                    current_translation_range=current_translation_range,
                    disc_type=options.disc_type,
                    cs_for_engine=cs_for_engine,
                    model_current_size_for_engine=model_current_size_for_engine,
                    local_pass1_current_size=local_pass1_current_size,
                    image_corrections_k=relion_half_inputs.image_corrections[k],
                    scale_corrections_k=relion_half_inputs.scale_corrections[k],
                    group_ids_k=follower_setup.scale_stats_group_ids_per_half[k],
                    group_count_k=follower_setup.scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                    translation_search_base=translation_search_base,
                    disable_adjoint_y=debug.disable_adjoint_y,
                    disable_adjoint_ctf=debug.disable_adjoint_ctf,
                    max_significants=adaptive.max_significants,
                    iteration=iteration,
                    debug_iteration=numbered_relion_iteration,
                    save_intermediates_dir=debug.save_intermediates_dir,
                    local_search_random_perturbation=local_search_random_perturbation,
                    local_search_angular_sampling_deg=local_search_angular_sampling_deg,
                    local_parent_oversampling_order=local_parent_oversampling_order,
                    local_search_translation_prior_mode=local_search.local_search_translation_prior_mode,
                    replay_prior_translations=_replay_prior_translations,
                    class_log_priors=class_log_priors,
                    k_class_enabled=k_class_enabled,
                    collect_local_search_profile=collect_local_search_profile,
                    diagnostic_score_only=bool(debug.stop_after_local_search_score_only),
                    safe_batch_sizes=_safe_batch_sizes,
                    outputs=per_half,
                    local_profile_history=history.local_profile_history,
                    relion_projector_half=relion_projector_half_by_half[k],
                    relion_projector_r_max=relion_projector_r_max_by_half[k],
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                )
                ha_k = local_result.ha
                Ft_y_k = local_result.Ft_y
                Ft_ctf_k = local_result.Ft_ctf
                em_stats_k = local_result.em_stats
                noise_stats_k = local_result.noise_stats
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = None
                coarse_ha[k] = ha_k
                score_result = local_result

            elif use_adaptive:
                adaptive_result = _score_half_dense_in_bpref_scope(
                    bpref_device_signature_active=bpref_device_signature_active,
                    k=k,
                    experiment_dataset=experiment_datasets[k],
                    means_k=means[k],
                    mean_variance=mean_variance_k,
                    noise_variance_k=noise_variance_k,
                    effective_rotations=(
                        adaptive_pass1_rotations
                        if adaptive_pass1_rotations is not None
                        else effective_rotations
                    ),
                    current_translations=current_translations,
                    base_translations=base_translations,
                    current_healpix_order=current_healpix_order,
                    state=state,
                    random_perturbation=random_perturbation,
                    disc_type=options.disc_type,
                    image_batch_size=batching.image_batch_size,
                    rotation_log_prior_k=rotation_log_prior_k,
                    class_rotation_log_prior_k=class_rotation_log_prior_k,
                    translation_log_prior=translation_log_prior,
                    translation_search_base=translation_search_base,
                    trans_prior_center_for_engine=trans_prior_center_for_engine,
                    image_corrections_k=relion_half_inputs.image_corrections[k],
                    scale_corrections_k=relion_half_inputs.scale_corrections[k],
                    group_ids_k=follower_setup.scale_stats_group_ids_per_half[k],
                    group_count_k=follower_setup.scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                    firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
                    firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
                    cs_for_engine=cs_for_engine,
                    model_current_size_for_engine=model_current_size_for_engine,
                    class_log_priors=class_log_priors,
                    k_class_enabled=k_class_enabled,
                    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                    disable_adjoint_y=debug.disable_adjoint_y,
                    disable_adjoint_ctf=debug.disable_adjoint_ctf,
                    safe_batch_sizes=_safe_batch_sizes,
                    max_significants=adaptive.max_significants,
                    outputs=per_half,
                    # Adaptive-specific:
                    k_class_image_batch_size_override=k_class_image_batch_size,
                    k_class_rotation_block_size_override=dense_k_class_rotation_block_size,
                    significance_image_batch_size_override=significance_image_batch_size,
                    significance_rotation_block_size_override=significance_rotation_block_size,
                    firstiter_coarse_current_size=coarse_cs,
                    firstiter_fine_current_size=cs_for_engine,
                    firstiter_log_label="",
                    firstiter_updates_em_kwargs_ibs=True,
                    relion_projector_half=relion_projector_half_by_half[k],
                    relion_projector_r_max=relion_projector_r_max_by_half[k],
                    debug_iteration=numbered_relion_iteration,
                    coarse_rotation_ids=coarse_rotation_ids_for_scoring,
                    preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                )
                ha_k = adaptive_result.ha
                Ft_y_k = adaptive_result.Ft_y
                Ft_ctf_k = adaptive_result.Ft_ctf
                em_stats_k = adaptive_result.em_stats
                noise_stats_k = adaptive_result.noise_stats
                noise_stats_per_half[k] = noise_stats_k
                if adaptive_result.pose_rotations is not None:
                    pose_rotations[k] = adaptive_result.pose_rotations
                    pose_rotation_eulers[k] = adaptive_result.pose_rotation_eulers
                else:
                    pose_rotations[k] = effective_rotations
                    pose_rotation_eulers[k] = effective_rotation_eulers
                coarse_ha[k] = adaptive_result.coarse_ha if adaptive_result.coarse_ha is not None else ha_k
                score_result = adaptive_result

            else:
                # --- SINGLE-PASS E+M (no adaptive oversampling) ---
                single_pass_result = _score_half_dense_in_bpref_scope(
                    bpref_device_signature_active=bpref_device_signature_active,
                    k=k,
                    experiment_dataset=experiment_datasets[k],
                    means_k=means[k],
                    mean_variance=mean_variance_k,
                    noise_variance_k=noise_variance_k,
                    effective_rotations=effective_rotations,
                    current_translations=current_translations,
                    base_translations=base_translations,
                    current_healpix_order=current_healpix_order,
                    state=state,
                    random_perturbation=random_perturbation,
                    disc_type=options.disc_type,
                    image_batch_size=batching.image_batch_size,
                    rotation_log_prior_k=rotation_log_prior_k,
                    class_rotation_log_prior_k=class_rotation_log_prior_k,
                    translation_log_prior=translation_log_prior,
                    translation_search_base=translation_search_base,
                    trans_prior_center_for_engine=trans_prior_center_for_engine,
                    image_corrections_k=relion_half_inputs.image_corrections[k],
                    scale_corrections_k=relion_half_inputs.scale_corrections[k],
                    group_ids_k=follower_setup.scale_stats_group_ids_per_half[k],
                    group_count_k=follower_setup.scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                    firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
                    firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
                    cs_for_engine=cs_for_engine,
                    model_current_size_for_engine=model_current_size_for_engine,
                    class_log_priors=class_log_priors,
                    k_class_enabled=k_class_enabled,
                    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                    disable_adjoint_y=debug.disable_adjoint_y,
                    disable_adjoint_ctf=debug.disable_adjoint_ctf,
                    safe_batch_sizes=_safe_batch_sizes,
                    max_significants=adaptive.max_significants,
                    outputs=per_half,
                    preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
                    source_faithful_spectrum_norm=source_faithful_spectrum_norm,
                    relion_projector_half=relion_projector_half_by_half[k],
                    relion_projector_r_max=relion_projector_r_max_by_half[k],
                    debug_iteration=numbered_relion_iteration,
                    coarse_rotation_ids=coarse_rotation_ids_for_scoring,
                )
                ha_k = single_pass_result.ha
                Ft_y_k = single_pass_result.Ft_y
                Ft_ctf_k = single_pass_result.Ft_ctf
                em_stats_k = single_pass_result.em_stats
                noise_stats_k = single_pass_result.noise_stats
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = effective_rotations
                pose_rotation_eulers[k] = effective_rotation_eulers
                coarse_ha[k] = ha_k  # same grid, no oversampling
                score_result = single_pass_result

                # --- Manifest dump for deterministic replay (Phase 0.1) ---
                if debug.save_intermediates_dir is not None:
                    _manifest_path = os.path.join(
                        debug.save_intermediates_dir,
                        f"manifest_iter{iteration}_half{k}.npz",
                    )
                    _manifest = {
                        "effective_rotations": np.asarray(effective_rotations, dtype=np.float32),
                        "current_translations": np.asarray(current_translations, dtype=np.float32),
                        "rotation_log_prior": np.asarray(rotation_log_prior_k, dtype=np.float64)
                        if rotation_log_prior_k is not None
                        else np.array([]),
                        "translation_log_prior": np.asarray(translation_log_prior, dtype=np.float64)
                        if translation_log_prior is not None
                        else np.array([]),
                        "image_corrections": np.asarray(relion_half_inputs.image_corrections[k], dtype=np.float64)
                        if relion_half_inputs.image_corrections[k] is not None
                        else np.array([]),
                        "scale_corrections": np.asarray(relion_half_inputs.scale_corrections[k], dtype=np.float64)
                        if relion_half_inputs.scale_corrections[k] is not None
                        else np.array([]),
                        "image_pre_shifts": np.asarray(translation_search_base, dtype=np.float32)
                        if translation_search_base is not None
                        else np.array([]),
                        "absolute_previous_translations": np.asarray(previous_translations_k, dtype=np.float32)
                        if previous_translations_k is not None
                        else np.array([]),
                        "mean_vol_ft": np.asarray(means[k]),
                        "mean_variance": np.asarray(mean_variance),
                        "noise_variance": np.asarray(noise_variance_k),
                        "current_size": np.int32(cs_for_engine) if cs_for_engine is not None else np.int32(-1),
                        "half_spectrum_scoring": np.bool_(True),
                        "use_float64_scoring": np.bool_(False),
                        "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                        "reconstruction_padding_factor": np.int32(PADDING_FACTOR),
                        "score_with_masked_images": np.bool_(True),
                        "perturbation_instance": np.float64(random_perturbation),
                        "perturbation_factor": np.float64(parity.perturb_factor),
                        "iteration": np.int32(iteration),
                        "half_index": np.int32(k),
                        "ave_Pmax": np.float64(float(np.mean(em_stats_k.max_posterior_per_image))),
                    }
                    np.savez(_manifest_path, **_manifest)
                    logger.info("Manifest dumped: %s", _manifest_path)

            # NOTE: means[k] reconstruction is DEFERRED until after the
            # low_resol_join_halves step below — we need both halves'
            # Ft_y / Ft_ctf accumulators in hand before we can average
            # the low-frequency shells across the two halves.
            score_result = _maybe_host_offload_half0_local_accumulators(
                half_index=k,
                use_local=use_local,
                k_class_enabled=k_class_enabled,
                score_result=score_result,
                log=logger,
            )
            Ft_y_k = score_result.Ft_y
            Ft_ctf_k = score_result.Ft_ctf
            per_half.update_from(k, score_result, dtype=_dense_global_scoring_dtype())
            _record_score_profile(
                history.global_profile_history,
                score_result,
                phase="iteration",
                iteration=iteration,
                relion_iteration=iteration + 1,
                half_index=k,
                current_size=cs_for_engine,
                healpix_order=current_healpix_order,
                k_class_enabled=k_class_enabled,
            )
            if score_result.significant_counts is not None:
                score_sig_counts = np.asarray(score_result.significant_counts, dtype=np.int32)
                iter_recorded_sig_count_parts.append(score_sig_counts)
                if not k_class_enabled:
                    iter_sig_count_parts.append(score_sig_counts)

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

            _device_signature_target_iteration = os.environ.get(
                "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION"
            )
            _device_signature_target_half = os.environ.get(
                "RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF"
            )
            if _device_signature_target_half and int(_device_signature_target_half) not in {1, 2}:
                raise ValueError("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF must be 1 or 2")
            if (
                os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR")
                and (
                    not _device_signature_target_iteration
                    or int(_device_signature_target_iteration) == iteration + 1
                )
                and (
                    not _device_signature_target_half
                    or int(_device_signature_target_half) == k + 1
                )
            ):
                bpref_diagnostics.flush_bpref_device_panel_accumulator(
                    iteration=iteration + 1,
                    half=k + 1,
                )

            # Capture original-stack image indices for the half so dumps can be
            # matched to RELION's data.star image_name ordering.
            try:
                _half_orig_idx = np.asarray(
                    experiment_datasets[k]._index_layout.original_image_indices_for_local(
                        np.arange(experiment_datasets[k].n_images, dtype=np.int32)
                    ),
                    dtype=np.int64,
                )
            except Exception:
                _half_orig_idx = None
            _parity_dump.collect_e_step(
                half=k,
                em_stats=em_stats_k,
                hard_assignment=ha_k,
                coarse_hard_assignment=coarse_ha[k],
                noise_stats=noise_stats_per_half[k],
                Ft_y=Ft_y_k,
                Ft_ctf=Ft_ctf_k,
                pose_rotation_eulers=pose_rotation_eulers[k],
                best_pose_rotation_eulers=best_pose_rotation_eulers[k],
                best_pose_translations=best_pose_translations[k],
                translation_search_base=translation_search_bases[k],
                original_image_indices=_half_orig_idx,
            )

        if diagnostic_half_indices != (0, 1):
            raise RuntimeError(
                "targeted half-only significance diagnostic returned without writing its "
                "complete target set; refusing to continue with one half missing"
            )

        # E-step + per-half M-step accumulators are now both populated.
        _parity_dump.mark_stage(iteration, "e_step")
        if iter_sig_count_parts:
            iter_sig_counts = np.concatenate(iter_sig_count_parts, axis=0)
        if iter_recorded_sig_count_parts:
            iter_recorded_sig_counts = np.concatenate(iter_recorded_sig_count_parts, axis=0)
        if (debug.stop_after_local_search_profile or stop_after_local_search) and use_local:
            elapsed = time.time() - t0
            logger.info(
                "Stopping after local-search diagnostic at iteration %d: profiles=%d score_only=%s wall=%.1fs",
                iteration + 1,
                len(history.local_profile_history),
                bool(debug.stop_after_local_search_score_only),
                elapsed,
            )
            merged_mean, merged_class_means = _merged_mean_from_halves(
                means,
                class_weights if k_class_enabled else None,
            )
            (
                replay_requested_iterations,
                replay_applied_iterations,
            ) = _finalize_relion_follower_scale_replay_telemetry(
                replay.relion_follower_scale_replay,
                applied_iterations=history.relion_follower_scale_replay_applied_iterations,
                logger=logger,
            )
            return {
                "profile_only": True,
                "mean": merged_mean,
                "means": means,
                "class_means": merged_class_means,
                "class_weights": class_weights if k_class_enabled else None,
                "class_assignments": class_assignments if k_class_enabled else None,
                "relion_follower_scale_replay_requested_iterations": replay_requested_iterations,
                "relion_follower_scale_replay_applied_iterations": replay_applied_iterations,
                "hard_assignments": hard_assignments,
                "convergence_state": state,
                "frozen_initial_scoring_state_sha256": frozen_initial_scoring_state_sha256,
                "expected_accuracy_trial_local_indices": expected_accuracy_trial_local_indices,
                "expected_accuracy_trial_particle_ids": expected_accuracy_trial_particle_ids,
                "final_all_data_ran": False,
                "stop_after_local_search_score_only": bool(debug.stop_after_local_search_score_only),
                "setup_phase_seconds": setup_phase_seconds,
                **history.to_dict(),
                "wall_times": [elapsed],
                "significant_counts": [iter_recorded_sig_counts],
            }
        if k_class_enabled:
            class_weights = _class_weights_from_posterior(
                class_posterior_per_half,
                n_classes,
                class_weights,
            )
            class_log_priors = np.log(class_weights)
            history.record_class_weights(
                class_weights.copy(),
                class_weights.copy(),
                _class_weights_from_posterior(
                    class_full_posterior_per_half,
                    n_classes,
                    class_weights,
                ).copy(),
            )
            logger.info(
                "K-class occupancies: %s",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_weights)),
            )
        mstep_accumulator_shape = _resolve_mstep_accumulator_shape(
            per_half.mstep_accumulator_shape,
            padded_volume_shape,
        )
        mstep_full_half_axis = _resolve_mstep_full_half_axis(
            per_half.mstep_full_half_axis,
            default_axis=-1,
        )

        _bpref_prejoin_dir = os.environ.get("RECOVAR_BPREF_PREJOIN_DUMP_DIR")
        _bpref_boundary_target_iteration = os.environ.get("RECOVAR_BPREF_BOUNDARY_DUMP_ITERATION")
        _bpref_boundary_iteration_matches = (
            not _bpref_boundary_target_iteration
            or iteration + 1 == int(_bpref_boundary_target_iteration)
        )
        if _bpref_prejoin_dir and not k_class_enabled and _bpref_boundary_iteration_matches:
            _save_bpref_accumulators(
                _bpref_prejoin_dir,
                stage="prejoin",
                iteration=iteration,
                current_size=current_size,
                padding_factor=PADDING_FACTOR,
                grid_size=grid_size,
                voxel_size=cryo.voxel_size,
                volume_shape=volume_shape,
                accumulator_shape=mstep_accumulator_shape,
                Ft_y_0=Ft_y_0,
                Ft_y_1=Ft_y_1,
                Ft_ctf_0=Ft_ctf_0,
                Ft_ctf_1=Ft_ctf_1,
            )

        # --- RELION's --low_resol_join_halves: average the low-resolution
        # shells of the per-half Fourier accumulators between the two halves
        # BEFORE the Wiener solve. This forces the two half-maps to share
        # their low-frequency content, preventing them from diverging in
        # orientation space at SNR-poor low shells. RELION mirrors this in
        # ml_optimiser_mpi.cpp::joinTwoHalvesAtLowResolution; without it
        # recovar's iter-N FSC drops gradually from shell ~2 while RELION's
        # stays at 1.0 through shell 13 (= 40 A for a 128/4.25 dataset),
        # which directly translates to a ~5-shell deficit in
        # ``first_shell_below_0.5`` and a ~10-pixel/iter deficit in
        # ``current_size`` growth (the dominant convergence-speed gap
        # observed in the 2026-04 5k normalized parity benchmark).
        #
        # Use the previous iteration's resolution to cap the join radius
        # (so we never join shells beyond the actual resolution of the
        # map). Mirrors the ``XMIPP_MAX(low_resol_join_halves,
        # 1./mymodel.current_resolution)`` in RELION's source.
        if k_class_enabled:
            Ft_y_combined = _combine_optional_half_accumulators(Ft_y_0, Ft_y_1, label="Ft_y")
            Ft_ctf_combined = _combine_optional_half_accumulators(Ft_ctf_0, Ft_ctf_1, label="Ft_ctf")
        elif parity.low_resol_join_halves_angstrom is not None and parity.low_resol_join_halves_angstrom > 0:
            prev_res_angstrom = None
            if history.pixel_resolutions:
                prev_pixel_res = history.pixel_resolutions[-1]
                if prev_pixel_res > 0:
                    prev_res_angstrom = shell_index_to_resolution_angstrom(
                        prev_pixel_res,
                        grid_size,
                        cryo.voxel_size,
                    )
            elif np.isfinite(float(getattr(state, "current_resolution", float("inf")))):
                prev_res_angstrom = float(state.current_resolution)
            Ft_y_0, Ft_y_1, Ft_ctf_0, Ft_ctf_1 = regularization.join_halves_at_low_resolution(
                Ft_y_0,
                Ft_y_1,
                Ft_ctf_0,
                Ft_ctf_1,
                mstep_accumulator_shape,
                cryo.voxel_size,
                grid_size,
                parity.low_resol_join_halves_angstrom,
                current_resolution_angstrom=prev_res_angstrom,
                padding_factor=PADDING_FACTOR,
            )

        # --- RELION-exact M-step ordering ---
        # K=1 stays on RELION's split-half auto-refine path
        # (compareTwoHalves -> updateSSNRarrays -> reconstruct).
        # K>1 switches to RELION Class3D semantics:
        #   1. combine the two half accumulators per class
        #   2. carry the previous Iref power spectrum forward as tau2
        #   3. run one Wiener solve per class
        #
        # Snapshot the previous-iter means BEFORE the reconstruction so sign
        # alignment has a reference at iter 1.
        if k_class_enabled:
            # K-class 256px maps are large enough that materializing both
            # previous class stacks on the host immediately after pass 2 can
            # SIGBUS under Slurm/tmp quota pressure.  JAX arrays are immutable;
            # keep device references here and let the later per-class tau2/sign
            # code transfer only the slices it actually needs.
            previous_means = [jnp.asarray(mean) if mean is not None else None for mean in means]
        else:
            previous_means = [np.asarray(mean).copy() if mean is not None else None for mean in means]

        _t_unreg_first = time.time()
        if k_class_enabled:
            tau2_update_details_per_class = []
            mean_signal_variance_per_class = []
            mean_signal_variance_shells_per_class = []
            data_vs_prior_per_class = []
            # Dense RECOVAR accumulators live in the historical unnormalised
            # image frame: RELION BPref weight = Ft_ctf * N^4. Equivalently,
            # keep Ft_y/Ft_ctf in RECOVAR frame and scale RELION tau2 by N^4
            # before the Wiener solve. See initial_model/gpu_pipeline.py's
            # bp_weight_frame_scale for the same frame conversion.
            kclass_tau2_frame_scale = float(grid_size) ** 4
            kclass_tau2_source = "previous Iref power spectra"
            replay_class_tau2 = None
            replay_tau2_enabled = _kclass_replay_tau2_enabled()
            tau2_replay_override = iter_replay_override
            tau2_replay_label = "current replay override"
            if replay_tau2_enabled:
                # RELION updates mymodel.tau2_class during expectation setup
                # from the current Iref, then uses that same model state for
                # maximization. Therefore run_itNNN_model.star contains the
                # tau2 prior used by iteration NNN, not the prior for NNN+1.
                _kclass_replay_tau2_same_iter_enabled()
                same_iter_index = iteration + 1
                tau2_replay_override = None
                if replay.replay_iteration_overrides is not None and same_iter_index < len(replay.replay_iteration_overrides):
                    tau2_replay_override = replay.replay_iteration_overrides[same_iter_index]
                    tau2_replay_label = f"same-iteration replay override index={same_iter_index}"
                if tau2_replay_override is None or tau2_replay_override.get("class_tau2") is None:
                    logger.warning(
                        "Diagnostic %s=1 requested same-numbered Class3D tau2 at iter=%d, "
                        "but replay override index %d is unavailable; falling back to current override",
                        _KCLASS_REPLAY_TAU2_ENV,
                        iteration + 1,
                        same_iter_index,
                    )
                    tau2_replay_override = iter_replay_override
                    tau2_replay_label = "current replay override fallback"
            if tau2_replay_override is not None and tau2_replay_override.get("class_tau2") is not None:
                replay_class_tau2 = np.asarray(tau2_replay_override["class_tau2"], dtype=np.float64)
                replay_class_tau2_shape = replay_class_tau2.shape
                if len(replay_class_tau2_shape) != 2 or replay_class_tau2_shape[0] != n_classes:
                    raise ValueError(
                        "class_tau2 replay override must have shape "
                        f"({n_classes}, n_shells), got {replay_class_tau2_shape}",
                    )
                if replay_tau2_enabled:
                    kclass_tau2_source = f"RELION replay class_tau2 ({tau2_replay_label})"
                    logger.info(
                        "Diagnostic %s=1: Class3D tau2 replay override used at iter=%d from %s with shape=%s",
                        _KCLASS_REPLAY_TAU2_ENV,
                        iteration + 1,
                        tau2_replay_label,
                        replay_class_tau2_shape,
                    )
                else:
                    logger.info(
                        "Class3D tau2 replay override available at iter=%d with shape=%s; "
                        "M-step tau2 is recomputed from previous Iref power spectra",
                        iteration + 1,
                        replay_class_tau2_shape,
                    )
            if iteration == 0:
                mean_variance_arr = jnp.asarray(mean_variance)
                expected_shape = (n_classes, int(np.prod(volume_shape)))
                if tuple(mean_variance_arr.shape) == expected_shape:
                    logger.info(
                        "Class3D initial per-class tau2 volume available at iter=%d with shape=%s; "
                        "M-step tau2 is recomputed from previous Iref power spectra",
                        iteration + 1,
                        tuple(mean_variance_arr.shape),
                    )
            for class_idx in range(n_classes):
                logger.info(
                    "Class3D tau2 update start: iter=%d class=%d/%d current_size=%d source=%s",
                    iteration + 1,
                    class_idx + 1,
                    n_classes,
                    int(current_size),
                    kclass_tau2_source,
                )
                if replay_class_tau2 is not None and replay_tau2_enabled:
                    tau2_shells_recovar_frame_k = jnp.asarray(
                        replay_class_tau2[class_idx],
                        dtype=jnp.float32,
                    )
                    mean_signal_variance_k = jnp.asarray(
                        utils.make_radial_image(
                            tau2_shells_recovar_frame_k,
                            volume_shape,
                            extend_last_frequency=True,
                        ),
                        dtype=jnp.float32,
                    ).reshape(-1)
                    tau2_shells_relion_frame_k = tau2_shells_recovar_frame_k / jnp.asarray(
                        kclass_tau2_frame_scale,
                        dtype=tau2_shells_recovar_frame_k.dtype,
                    )
                else:
                    mean_signal_variance_relion_k, tau2_update_details_k = (
                        regularization.compute_relion_tau2_from_iref_power_spectrum(
                            previous_means[0][class_idx],
                            volume_shape,
                            padding_factor=PADDING_FACTOR,
                            current_size=current_size,
                            return_details=True,
                        )
                    )
                    mean_signal_variance_k = mean_signal_variance_relion_k * jnp.asarray(
                        kclass_tau2_frame_scale,
                        dtype=mean_signal_variance_relion_k.dtype,
                    )
                    tau2_shells_relion_frame_k = jnp.asarray(
                        tau2_update_details_k["tau2_shells"],
                        dtype=mean_signal_variance_k.dtype,
                    )
                    tau2_shells_recovar_frame_k = tau2_shells_relion_frame_k * jnp.asarray(
                        kclass_tau2_frame_scale,
                        dtype=mean_signal_variance_k.dtype,
                    )
                shell_stats_k = regularization._compute_relion_weight_shell_stats(
                    Ft_ctf_combined[class_idx],
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    r_max=current_size // 2,
                    shell_rounding="round",
                    full_half_axis=mstep_full_half_axis,
                    accumulator_volume_shape=mstep_accumulator_shape,
                )
                reconstruct_floor_stats_k = regularization._compute_relion_weight_shell_stats(
                    Ft_ctf_combined[class_idx],
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    r_max=current_size // 2,
                    shell_rounding="floor",
                    full_half_axis=mstep_full_half_axis,
                    accumulator_volume_shape=mstep_accumulator_shape,
                )
                data_vs_prior_k = regularization.compute_data_vs_prior(
                    Ft_ctf_combined[class_idx],
                    tau2_shells_recovar_frame_k,
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    tau2_fudge=tau2_fudge,
                    current_size=current_size,
                    full_half_axis=mstep_full_half_axis,
                    accumulator_volume_shape=mstep_accumulator_shape,
                )
                mean_signal_variance_per_class.append(mean_signal_variance_k)
                mean_signal_variance_shells_per_class.append(tau2_shells_recovar_frame_k)
                data_vs_prior_per_class.append(data_vs_prior_k)
                tau2_update_details_per_class.append(
                    {
                        "prior_shells": np.asarray(tau2_shells_recovar_frame_k, dtype=np.float64),
                        "sigma2_shells": np.asarray(
                            jnp.where(
                                shell_stats_k["avg_weight_shells"] > 0,
                                1.0 / (PADDING_FACTOR**3 * shell_stats_k["avg_weight_shells"]),
                                0.0,
                            ),
                            dtype=np.float64,
                        ),
                        "avg_weight_shells": np.asarray(shell_stats_k["avg_weight_shells"], dtype=np.float64),
                        "shell_sum": np.asarray(shell_stats_k["shell_sum"], dtype=np.float64),
                        "shell_count": np.asarray(shell_stats_k["shell_count"], dtype=np.float64),
                        "fsc_shells": None,
                        "ssnr_shells": np.asarray(data_vs_prior_k, dtype=np.float64),
                    }
                )
                _kclass_dump_dir = os.environ.get("RECOVAR_KCLASS_DUMP_DIR")
                if _kclass_dump_dir:
                    reconstruction_diagnostics.write_kclass_mstep(
                        Ft_ctf_0=Ft_ctf_0,
                        Ft_ctf_1=Ft_ctf_1,
                        Ft_ctf_combined=Ft_ctf_combined,
                        Ft_y_combined=Ft_y_combined,
                        PADDING_FACTOR=PADDING_FACTOR,
                        output_dir=_kclass_dump_dir,
                        class_idx=class_idx,
                        current_size=current_size,
                        data_vs_prior_k=data_vs_prior_k,
                        grid_size=grid_size,
                        iteration=iteration,
                        kclass_tau2_frame_scale=kclass_tau2_frame_scale,
                        kclass_tau2_source=kclass_tau2_source,
                        mstep_accumulator_shape=mstep_accumulator_shape,
                        mstep_full_half_axis=mstep_full_half_axis,
                        previous_means=previous_means,
                        reconstruct_floor_stats_k=reconstruct_floor_stats_k,
                        shell_stats_k=shell_stats_k,
                        tau2_fudge=tau2_fudge,
                        tau2_shells_recovar_frame_k=tau2_shells_recovar_frame_k,
                        tau2_shells_relion_frame_k=tau2_shells_relion_frame_k,
                    )
                logger.info(
                    "Class3D tau2 update done: iter=%d class=%d/%d elapsed=%.1fs",
                    iteration + 1,
                    class_idx + 1,
                    n_classes,
                    time.time() - _t_unreg_first,
                )
            mean_signal_variance = jnp.stack(mean_signal_variance_per_class, axis=0)
            mean_signal_variance_shells = jnp.stack(mean_signal_variance_shells_per_class, axis=0)
            data_vs_prior_iter = np.stack(
                [np.asarray(dvp, dtype=_dense_global_scoring_dtype()) for dvp in data_vs_prior_per_class],
                axis=0,
            )
            history.record_data_vs_prior(data_vs_prior_iter)
            previous_data_vs_prior_for_scheduling = data_vs_prior_iter
            tau2_update_details = {
                key: np.stack([detail[key] for detail in tau2_update_details_per_class], axis=0)
                if key not in {"fsc_shells"}
                else None
                for key in [
                    "prior_shells",
                    "sigma2_shells",
                    "avg_weight_shells",
                    "shell_sum",
                    "shell_count",
                    "fsc_shells",
                    "ssnr_shells",
                ]
            }
            logger.info(
                "Computed iter-%d Class3D tau2 from %s: %.1fs",
                iteration + 1,
                kclass_tau2_source,
                time.time() - _t_unreg_first,
            )
        else:
            mean_signal_variance_shells = None
            # Optional dump of post-join Ft_y, Ft_ctf for shell-by-shell parity
            # comparison against RELION's RECOVAR_MSTEP_DUMP_DIR. Activated by
            # RECOVAR_BPREF_ACCUM_DUMP_DIR. One npz per iteration.
            _bpref_accum_dir = os.environ.get("RECOVAR_BPREF_ACCUM_DUMP_DIR")
            if _bpref_accum_dir and _bpref_boundary_iteration_matches:
                _save_bpref_accumulators(
                    _bpref_accum_dir,
                    stage="accum",
                    iteration=iteration,
                    current_size=current_size,
                    padding_factor=PADDING_FACTOR,
                    grid_size=grid_size,
                    voxel_size=cryo.voxel_size,
                    volume_shape=volume_shape,
                    accumulator_shape=mstep_accumulator_shape,
                    Ft_y_0=Ft_y_0,
                    Ft_y_1=Ft_y_1,
                    Ft_ctf_0=Ft_ctf_0,
                    Ft_ctf_1=Ft_ctf_1,
                )
            current_iter_fsc = regularization.compute_relion_fsc_from_backprojector(
                Ft_y_0,
                Ft_y_1,
                Ft_ctf_0,
                Ft_ctf_1,
                volume_shape,
                padding_factor=PADDING_FACTOR,
                r_max=current_size // 2,
                accumulator_volume_shape=mstep_accumulator_shape,
                output_dtype=_dense_global_scoring_dtype(),
            )
            logger.info(
                "Computed iter-%d FSC for tau2 (RELION backprojector path): %.1fs",
                iteration + 1,
                time.time() - _t_unreg_first,
            )
            raw_backprojector_fsc = current_iter_fsc
            tau2_fsc_for_update = current_iter_fsc
            if parity.do_solvent_fsc_correction and particle_diameter_ang is not None and particle_diameter_ang > 0:
                from recovar.core import mask as _mask

                _t_solvent_fsc = time.time()
                unfiltered_half_maps = []
                for Ft_ctf_half, Ft_y_half in ((Ft_ctf_0, Ft_y_0), (Ft_ctf_1, Ft_y_1)):
                    unfiltered_real = _reconstruct_volume_eager(
                        Ft_ctf_half,
                        Ft_y_half,
                        volume_shape,
                        PADDING_FACTOR,
                        tau=None,
                        tau2_fudge=tau2_fudge,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        # RELION's BackProjector::reconstruct(do_map=false)
                        # still calls softMaskOutsideMap inside
                        # windowToOridimRealSpace.
                        use_spherical_mask=True,
                        minres_map=RELION_MINRES_MAP,
                        current_size=int(current_size),
                        return_real_space=True,
                        accumulator_volume_shape=mstep_accumulator_shape,
                    )
                    unfiltered_real = np.asarray(
                        jnp.asarray(unfiltered_real).reshape(volume_shape),
                        dtype=np.float64,
                    ).real
                    unfiltered_half_maps.append(unfiltered_real)

                flatten_radius = particle_diameter_ang / (2.0 * cryo.voxel_size)
                solvent_mask = np.asarray(
                    _mask.raised_cosine_mask(
                        volume_shape,
                        radius=flatten_radius,
                        radius_p=flatten_radius + RELION_WIDTH_MASK_EDGE,
                        offset=jnp.zeros(3),
                        dtype=jnp.float64,
                    ),
                    dtype=np.float64,
                )
                tau2_fsc_for_update, solvent_fsc_details = regularization.compute_relion_solvent_corrected_true_fsc(
                    unfiltered_half_maps[0],
                    unfiltered_half_maps[1],
                    solvent_mask,
                    current_size=int(current_size),
                    rng_seed=int(1775735620 + iteration),
                    return_details=True,
                )
                randomize_at = int(solvent_fsc_details["randomize_at"])
                probe_shell = max(1, randomize_at) if randomize_at > 0 else min(len(solvent_fsc_details["fsc_true"]) - 1, 1)
                corrected_shell = min(len(solvent_fsc_details["fsc_true"]) - 1, max(probe_shell, randomize_at + 2))
                logger.info(
                    "Computed iter-%d solvent-corrected true FSC for tau2: randomize_at=%d "
                    "raw_fsc[%d]=%.4f masked=%.4f random_masked=%.4f true=%.4f; "
                    "formula_shell[%d]: masked=%.4f random_masked=%.4f true=%.4f elapsed=%.1fs",
                    iteration + 1,
                    randomize_at,
                    probe_shell,
                    float(np.asarray(raw_backprojector_fsc)[probe_shell]),
                    float(solvent_fsc_details["fsc_masked"][probe_shell]),
                    float(solvent_fsc_details["fsc_random_masked"][probe_shell]),
                    float(solvent_fsc_details["fsc_true"][probe_shell]),
                    corrected_shell,
                    float(solvent_fsc_details["fsc_masked"][corrected_shell]),
                    float(solvent_fsc_details["fsc_random_masked"][corrected_shell]),
                    float(solvent_fsc_details["fsc_true"][corrected_shell]),
                    time.time() - _t_solvent_fsc,
                )
            elif parity.do_solvent_fsc_correction:
                logger.warning(
                    "RELION solvent FSC correction requested but particle_diameter_ang is unset; using raw FSC for tau2"
                )

            # RELION calls BackProjector::updateSSNRarrays independently for each
            # half-map BPref.  The gold-standard FSC is shared, but sigma2/tau2
            # come from each half's own Fourier weight outside the joined shells.
            tau2_update_details_per_half = []
            mean_signal_variance_per_half = []
            for half_idx, Ft_ctf_half in enumerate((Ft_ctf_0, Ft_ctf_1)):
                full_half_axis = per_half.mstep_full_half_axis[half_idx]
                mean_signal_variance_k, _, tau2_update_details_k = regularization.compute_relion_tau2_from_weights(
                    Ft_ctf_half,
                    Ft_ctf_half,
                    tau2_fsc_for_update,
                    volume_shape,
                    tau2_fudge=tau2_fudge,
                    padding_factor=PADDING_FACTOR,
                    r_max=current_size // 2,
                    return_details=True,
                    full_half_axis=-1 if full_half_axis is None else int(full_half_axis),
                    accumulator_volume_shape=mstep_accumulator_shape,
                    output_dtype=_dense_global_scoring_dtype(),
                )
                mean_signal_variance_per_half.append(mean_signal_variance_k)
                tau2_update_details_per_half.append(tau2_update_details_k)
            mean_signal_variance = 0.5 * (mean_signal_variance_per_half[0] + mean_signal_variance_per_half[1])
            # Keep the single tau2 diagnostic fields aligned with RELION's half1
            # model.star, which is what the parity diff script reports.
            tau2_update_details = tau2_update_details_per_half[0]
            logger.info(
                "tau2 update from THIS-iter FSC: old_max=%.4e new_max=%.4e half_max=(%.4e, %.4e)",
                float(jnp.max(jnp.abs(mean_variance))),
                float(jnp.max(jnp.abs(mean_signal_variance))),
                float(jnp.max(jnp.abs(mean_signal_variance_per_half[0]))),
                float(jnp.max(jnp.abs(mean_signal_variance_per_half[1]))),
            )
        mean_variance = mean_signal_variance
        if not k_class_enabled:
            mean_variance_per_half = _updated_mean_variance_per_half(
                mean_variance,
                mean_signal_variance_per_half,
                use_per_half_mean_variance=parity.use_per_half_mean_variance,
            )
        else:
            mean_variance_per_half = [mean_variance, mean_variance]

        # --- Free previous-iteration means to reclaim GPU memory ---
        # (previous_means already snapshotted earlier for FSC sign alignment)
        for k in range(2):
            means[k] = None

        # --- Now reconstruct the regularized means ---
        _reconstruct_and_postprocess_means(
            means,
            Ft_y_0=Ft_y_0,
            Ft_y_1=Ft_y_1,
            Ft_ctf_0=Ft_ctf_0,
            Ft_ctf_1=Ft_ctf_1,
            Ft_y_combined=Ft_y_combined if k_class_enabled else None,
            Ft_ctf_combined=Ft_ctf_combined if k_class_enabled else None,
            mean_signal_variance=mean_signal_variance if k_class_enabled else None,
            mean_signal_variance_shells=mean_signal_variance_shells if k_class_enabled else None,
            mean_signal_variance_per_half=mean_signal_variance_per_half if not k_class_enabled else None,
            n_classes=n_classes,
            k_class_enabled=k_class_enabled,
            cs=current_size,
            iteration=iteration,
            grid_size=grid_size,
            cryo=cryo,
            volume_shape=volume_shape,
            tau2_fudge=tau2_fudge,
            padding_factor=PADDING_FACTOR,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            relion_minres_map=RELION_MINRES_MAP,
            particle_diameter_ang=particle_diameter_ang,
            relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
            relion_firstiter_ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
            relion_width_mask_edge=RELION_WIDTH_MASK_EDGE,
            relion_fmask_edge=RELION_WIDTH_FMASK_EDGE,
            accumulator_volume_shape=mstep_accumulator_shape,
        )

        # RELION reconstructs the first-iteration CC maps with the untapered
        # updateSSNRarrays tau2.  Only afterwards does
        # initialLowPassFilterReferences taper tau2/data_vs_prior for the
        # model state and reporting; that tapered spectrum is explicitly not
        # used in the reconstruction calculation (ml_optimiser.cpp:5296-5328).
        if (
            not k_class_enabled
            and relion_firstiter_cc_this_iter
            and parity.relion_firstiter_ini_high_angstrom is not None
        ):
            tau2_taper = _firstiter_cc_ini_high_tau2_taper(
                len(tau2_update_details_per_half[0]["prior_shells"]),
                grid_size,
                cryo.voxel_size,
                parity.relion_firstiter_ini_high_angstrom,
                filter_edgewidth=RELION_WIDTH_FMASK_EDGE,
            )
            radial_shells = np.asarray(
                fourier_transform_utils.get_grid_of_radial_distances(
                    volume_shape,
                    scaled=False,
                    frequency_shift=0,
                ),
                dtype=np.int32,
            ).reshape(-1)
            radial_shells = np.minimum(radial_shells, len(tau2_taper) - 1)
            tau2_taper_volume = jnp.asarray(
                tau2_taper[radial_shells],
                dtype=_dense_global_scoring_dtype(),
            )
            for half_idx in range(2):
                mean_signal_variance_per_half[half_idx] = (
                    mean_signal_variance_per_half[half_idx] * tau2_taper_volume
                )
                for field in ("prior_shells", "ssnr_shells"):
                    field_values = tau2_update_details_per_half[half_idx][field]
                    tau2_update_details_per_half[half_idx][field] = field_values * jnp.asarray(
                        tau2_taper,
                        dtype=field_values.dtype,
                    )
            mean_signal_variance = 0.5 * (
                mean_signal_variance_per_half[0] + mean_signal_variance_per_half[1]
            )
            mean_variance = mean_signal_variance
            mean_variance_per_half = _updated_mean_variance_per_half(
                mean_variance,
                mean_signal_variance_per_half,
                use_per_half_mean_variance=parity.use_per_half_mean_variance,
            )
            tau2_update_details = tau2_update_details_per_half[0]
            logger.info(
                "RELION iter-1 CC emulation: tapered post-reconstruction tau2/data-vs-prior "
                "with ini_high=%.2f A",
                float(parity.relion_firstiter_ini_high_angstrom),
            )
        _parity_dump.mark_stage(iteration, "recon")

        history.record_significant_counts(iter_recorded_sig_counts)

        history.record_rotation_posterior(
            [
                None if value is None else np.asarray(value, dtype=np.float64).copy()
                for value in rotation_posterior_per_half
            ]
        )
        if all(rot_sum is not None for rot_sum in rotation_posterior_per_half):
            k1_direction_prior_order = current_healpix_order
            if use_local:
                k1_direction_prior_order = (
                    int(state.healpix_order)
                    if int(state.adaptive_oversampling) > 0
                    else int(local_search_order)
                )
            k1_direction_prior_size = rotation_grid_size(k1_direction_prior_order)
            if (
                not k_class_enabled
                and all(np.asarray(rot_sum).shape[0] == k1_direction_prior_size for rot_sum in rotation_posterior_per_half)
            ):
                for k in range(2):
                    direction_prior_k = collapse_rotation_posterior_to_direction_prior(
                        np.asarray(rotation_posterior_per_half[k], dtype=np.float64),
                        k1_direction_prior_order,
                        dtype=_dense_global_scoring_dtype(),
                    )
                    try:
                        make_relion_direction_log_prior(direction_prior_k, k1_direction_prior_order)
                    except ValueError as exc:
                        logger.warning(
                            "Skipping K=1 direction prior update for half-%d at healpix_order=%d: "
                            "%s",
                            k + 1,
                            k1_direction_prior_order,
                            exc,
                        )
                        continue
                    global_direction_prior_per_half[k] = direction_prior_k
                    global_direction_prior_order_per_half[k] = k1_direction_prior_order
            elif (
                not use_local
                and k_class_enabled
                and effective_rotations.shape[0] == rotation_grid_size(current_healpix_order)
                and all(rot_sum is not None for rot_sum in class_rotation_posterior_per_half)
            ):
                combined_class_direction_prior = _combined_class_direction_prior_from_halves(
                    class_rotation_posterior_per_half,
                    n_classes,
                    current_healpix_order,
                    dtype=_dense_global_scoring_dtype(),
                )
                for k in range(2):
                    class_direction_prior_per_half[k] = combined_class_direction_prior.copy()
                    class_direction_prior_order_per_half[k] = current_healpix_order

        if k_class_enabled:
            direction_prior_snapshot = [
                None
                if class_direction_prior_per_half[k] is None
                else np.asarray(class_direction_prior_per_half[k][0], dtype=np.float64).copy()
                for k in range(2)
            ]
        else:
            direction_prior_snapshot = [
                None
                if global_direction_prior_per_half[k] is None
                else np.asarray(global_direction_prior_per_half[k], dtype=np.float64).copy()
                for k in range(2)
            ]
        history.record_direction_prior(direction_prior_snapshot)

        # --- Compute unregularized half-maps only when diagnostics need them ---
        # K=1 FSC was already computed above directly from the BackProjector
        # accumulators (current_iter_fsc), matching RELION ordering. For K>1
        # the shared class3D prior is from the previous Iref power spectrum.
        # Reconstructing unreg here is only needed for saved intermediates /
        # parity dumps.
        need_unreg_means = (
            (debug.save_intermediates_dir is not None and not debug.save_intermediates_skip_unregularized)
            or _parity_dump.is_active()
        )
        unreg_result = compute_unregularized_halfmaps_and_align_signs(
            means=means,
            previous_means=previous_means,
            Ft_y_per_half=(Ft_y_0, Ft_y_1),
            Ft_ctf_per_half=(Ft_ctf_0, Ft_ctf_1),
            Ft_y_combined=Ft_y_combined if k_class_enabled else None,
            Ft_ctf_combined=Ft_ctf_combined if k_class_enabled else None,
            volume_shape=volume_shape,
            n_classes=n_classes,
            k_class_enabled=k_class_enabled,
            tau2_fudge=tau2_fudge,
            padding_factor=PADDING_FACTOR,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            minres_map=RELION_MINRES_MAP,
            need_unreg_means=need_unreg_means,
            accumulator_volume_shape=mstep_accumulator_shape,
        )
        unreg_means = unreg_result.unregularized_means

        # K>1 uses the shared per-class data_vs_prior curve to drive growth;
        # K=1 keeps the split-half FSC history.
        if k_class_enabled:
            fsc = None
            history.record_fsc(fsc, None)
            _parity_dump.mark_stage(iteration, "fsc")
        else:
            # FSC was already computed above in the RELION-exact ordering block
            # (current_iter_fsc) and used to derive tau2 BEFORE the Wiener solve.
            # Reuse it here — recomputing would give the same value (same
            # underlying unreg accumulators).
            fsc = current_iter_fsc
            history.record_fsc(fsc, tau2_fsc_for_update)
            _parity_dump.mark_stage(iteration, "fsc")

        # --- Save intermediate volumes if requested ---
        if debug.save_intermediates_dir is not None:
            _save_iteration_intermediates(
                debug.save_intermediates_dir,
                iteration=iteration,
                Ft_y_0=Ft_y_0,
                Ft_y_1=Ft_y_1,
                Ft_ctf_0=Ft_ctf_0,
                Ft_ctf_1=Ft_ctf_1,
                means=means,
                unreg_means=unreg_means,
                fsc=fsc,
                noise_variance=noise_variance,
                noise_variance_per_half=noise_variance_per_half,
                mean_variance=mean_variance,
                hard_assignments=hard_assignments,
                coarse_ha=coarse_ha,
                effective_rotations=effective_rotations,
                current_translations=current_translations,
                use_local=use_local,
                local_search_order=local_search_order,
                cs=current_size,
                state=state,
                n_classes=n_classes,
                k_class_enabled=k_class_enabled,
                volume_shape=volume_shape,
                voxel_size=cryo.voxel_size,
            )

        # --- Compute ave_Pmax from the actual E-step maxima ---
        if any(pmax is None for pmax in max_posterior_per_half):
            raise RuntimeError(
                "RELION mode expected per-image posterior maxima from the EM engine",
            )
        if k_class_enabled:
            pmax_normalization_mass_per_half = [
                float(np.sum(np.asarray(mass, dtype=np.float64), dtype=np.float64))
                for mass in class_posterior_per_half
            ]
        else:
            pmax_normalization_mass_per_half = [
                None if stats is None else float(np.asarray(stats.sumw, dtype=np.float64))
                for stats in noise_stats_per_half
            ]
        combined_max_posterior, ave_pmax, ave_pmax_denominator = _relion_optimizer_average_pmax(
            max_posterior_per_half,
            pmax_normalization_mass_per_half,
        )
        if k_class_enabled:
            logger.info(
                "Class3D optimizer Pmax: value=%.9f numerator=%.9f "
                "half1_mstep_posterior_mass=%.9f half1_particle_count=%d",
                ave_pmax,
                float(np.sum(np.asarray(max_posterior_per_half[0]), dtype=np.float64)),
                ave_pmax_denominator,
                int(np.asarray(max_posterior_per_half[0]).size),
            )
        history.record_pmax(ave_pmax, ave_pmax_denominator, combined_max_posterior.copy())

        # --- Track per-image best assignments for convergence detection ---
        # Combine both half-sets' assignments into a single array for
        # update_refinement_state.  Use coarse_ha (indexed into
        # effective_rotations) for consistent convergence tracking.
        current_combined_ha = np.concatenate(
            [np.asarray(ha, dtype=np.int32) for ha in coarse_ha],
            axis=0,
        )
        if all(ha is not None for ha in previous_assignments):
            previous_combined_ha = np.concatenate(
                [np.asarray(ha, dtype=np.int32) for ha in previous_assignments],
                axis=0,
            )
        else:
            previous_combined_ha = None
        if k_class_enabled:
            current_combined_classes = np.concatenate(
                [np.asarray(cls, dtype=np.int32) for cls in class_assignments],
                axis=0,
            )
            history.record_class_assignment(current_combined_classes.copy())
            if all(cls is not None for cls in previous_class_assignments):
                previous_combined_classes = np.concatenate(
                    [np.asarray(cls, dtype=np.int32) for cls in previous_class_assignments],
                    axis=0,
                )
            else:
                previous_combined_classes = None
        else:
            current_combined_classes = None
            previous_combined_classes = None

        # tau2 was already updated BEFORE the Wiener solve (matching RELION's
        # reconstruct() which calls updateSSNRarrays before the filter).

        # --- Resolution from updated FSC-derived SSNR (RELION auto-refine) ---
        # K=1: data_vs_prior comes from the half-map FSC.
        # K>1: data_vs_prior comes from the shared per-class prior and the
        # combined class accumulators.
        if k_class_enabled:
            dvp_iter = _truncate_data_vs_prior_for_current_size(
                history.data_vs_prior_trajectory[-1],
                current_size=current_size,
                grid_size=grid_size,
                dtype=_dense_global_scoring_dtype(),
            )
            dvp_res_shell = max(
                resolution_from_data_vs_prior(dvp_class, allow_high_res_recovery=False)
                for dvp_class in np.asarray(dvp_iter)
            )
        else:
            if tau2_update_details is not None and tau2_update_details.get("ssnr_shells") is not None:
                dvp_iter = np.asarray(
                    tau2_update_details["ssnr_shells"],
                    dtype=_dense_global_scoring_dtype(),
                ).copy()
            else:
                dvp_iter = np.asarray(
                    fsc_to_relion_ssnr(
                        np.asarray(fsc, dtype=_dense_global_scoring_dtype()),
                        tau2_fudge=tau2_fudge,
                    ),
                    dtype=_dense_global_scoring_dtype(),
                )
            dvp_iter = _truncate_data_vs_prior_for_current_size(
                dvp_iter,
                current_size=current_size,
                grid_size=grid_size,
                dtype=_dense_global_scoring_dtype(),
            )
            dvp_res_shell = resolution_from_data_vs_prior(
                dvp_iter,
                allow_high_res_recovery=True,
            )
        pixel_res = float(
            _firstiter_cc_scheduling_resolution_shell(
                dvp_res_shell,
                emulate_relion_firstiter_cc=parity.emulate_relion_firstiter_cc,
                ini_high_angstrom=parity.relion_firstiter_ini_high_angstrom,
                relion_iteration=int(init_relion_iteration) + int(iteration) + 1,
                grid_size=grid_size,
                voxel_size=cryo.voxel_size,
            )
        )
        if int(pixel_res) != int(dvp_res_shell):
            logger.info(
                "RELION firstiter_cc resolution state: using ini_high=%.2f A shell %d "
                "instead of live data-vs-prior shell %d",
                float(parity.relion_firstiter_ini_high_angstrom),
                int(pixel_res),
                int(dvp_res_shell),
            )
        _tau2_debug_dump_dir = os.environ.get("RECOVAR_RELION_TAU2_DEBUG_DUMP_DIR")
        if _tau2_debug_dump_dir:
            reconstruction_diagnostics.write_tau2_update(
                _replay_meta=_replay_meta,
                output_dir=_tau2_debug_dump_dir,
                voxel_size=cryo.voxel_size,
                current_size=current_size,
                dvp_iter=dvp_iter,
                fsc=locals().get("fsc"),
                grid_size=grid_size,
                iteration=iteration,
                mstep_accumulator_shape=mstep_accumulator_shape,
                perturb_replay_relion_dir=perturb_replay_relion_dir,
                perturb_replay_relion_prefix=perturb_replay_relion_prefix,
                pixel_res=pixel_res,
                sealed_sampling_state=sealed_sampling_state,
                tau2_update_details=tau2_update_details,
                tau2_update_details_per_half=tau2_update_details_per_half,
                logger=logger,
            )
        history.record_pixel_resolution(pixel_res)

        # --- Update poses and noise ---
        # Snapshot the iter K-1 best rotations / translations BEFORE the
        # loop overwrites them, so update_refinement_state below can compute
        # the RELION-exact change metrics (B3) between iter K-1 and iter K.
        prior_iter_best_rotations = [
            np.asarray(rot).copy() if rot is not None else None for rot in previous_best_rotations
        ]
        prior_iter_best_translations = [
            np.asarray(trans).copy() if trans is not None else None
            for trans in relion_half_inputs.previous_best_translations
        ]
        new_iter_best_rotations = [None, None]
        new_iter_best_rotation_eulers = [None, None]
        new_iter_best_translations = [None, None]
        # Cross-iteration pose/translation state: RELION carries this in
        # ``exp_metadata`` (``MultidimArray<RFLOAT>``) and writes it back via
        # ``EMDL_ORIENT_ORIGIN_X/Y_ANGSTROM`` (registered ``EMDL_DOUBLE``,
        # backed by ``std::vector<double>`` in ``MetaDataContainer``) -- never
        # narrowed to float, so this snapshot must follow the same dtype as
        # the rest of the dense/global-path float64-sensitive operands
        # (``_dense_global_scoring_dtype``). The Euler columns live in the
        # same ``MultidimArray<RFLOAT> exp_metadata`` as translations in
        # RELION, so ACC_DOUBLE_PRECISION also keeps them double internally.
        _pose_state_dtype = _dense_global_scoring_dtype()
        for k in range(2):
            if best_pose_rotations[k] is not None:
                best_rots = np.asarray(best_pose_rotations[k], dtype=_pose_state_dtype)
                best_eulers = (
                    np.asarray(best_pose_rotation_eulers[k], dtype=np.float64)
                    if best_pose_rotation_eulers[k] is not None
                    else utils.R_to_relion(best_rots, degrees=True).astype(_pose_state_dtype)
                )
                best_trans = np.asarray(best_pose_translations[k], dtype=_pose_state_dtype)
            elif use_local:
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                trans_idx = hard_assignments[k] % current_translations.shape[0]
                if local_search_rotations is None:
                    local_grid_metadata = build_local_search_grid_metadata(local_search_order)
                    best_rots = _selected_rotation_matrices(
                        rot_idx,
                        None,
                        local_grid_metadata,
                        random_perturbation=local_search_random_perturbation,
                        angular_sampling_deg=local_search_angular_sampling_deg,
                    )
                    best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(_pose_state_dtype)
                else:
                    best_rots = np.asarray(local_search_rotations, dtype=_pose_state_dtype)[rot_idx]
                    if local_search_rotation_eulers is not None:
                        best_eulers = np.asarray(local_search_rotation_eulers, dtype=_pose_state_dtype)[rot_idx]
                    else:
                        best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(_pose_state_dtype)
                best_trans = np.asarray(current_translations)[trans_idx]
            else:
                # Global search uses the dense grid in pose_rotations[k].
                # All dense EM / K-class paths report the flattened
                # rotation-translation row index here.
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                trans_idx = hard_assignments[k] % current_translations.shape[0]
                best_rots = np.asarray(pose_rotations[k], dtype=_pose_state_dtype)[rot_idx]
                best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(_pose_state_dtype)
                best_trans = np.asarray(current_translations)[trans_idx]
            new_iter_best_rotations[k] = best_rots
            new_iter_best_rotation_eulers[k] = best_eulers
            new_iter_best_translations[k] = _relion_metadata_translations(
                prior_iter_best_translations[k],
                best_trans,
                dtype=_pose_state_dtype,
            )
        previous_best_rotations = new_iter_best_rotations
        relion_half_inputs.previous_best_rotation_eulers = new_iter_best_rotation_eulers
        relion_half_inputs.previous_best_translations = new_iter_best_translations
        history.record_pose_history(
            [np.asarray(e).copy() if e is not None else None for e in new_iter_best_rotation_eulers],
            [np.asarray(t).copy() if t is not None else None for t in new_iter_best_translations],
        )

        current_rotation_matrices_combined = concatenate_pose_stacks_or_none(
            new_iter_best_rotations,
            trailing_shape=(3, 3),
            label="current rotation",
            dtype=_dense_global_scoring_dtype(),
            logger=logger,
        )
        previous_rotation_matrices_combined = concatenate_pose_stacks_or_none(
            prior_iter_best_rotations,
            trailing_shape=(3, 3),
            label="previous rotation",
            dtype=_dense_global_scoring_dtype(),
            logger=logger,
        )
        current_translations_pixel_combined = concatenate_pose_stacks_or_none(
            new_iter_best_translations,
            trailing_shape=(2,),
            label="current translation",
            dtype=_dense_global_scoring_dtype(),
            logger=logger,
        )
        previous_translations_pixel_combined = concatenate_pose_stacks_or_none(
            prior_iter_best_translations,
            trailing_shape=(2,),
            label="previous translation",
            dtype=_dense_global_scoring_dtype(),
            logger=logger,
        )

        if not k_class_enabled:
            history.record_data_vs_prior(np.asarray(dvp_iter, dtype=_dense_global_scoring_dtype()))
            previous_data_vs_prior_for_scheduling = np.asarray(
                dvp_iter,
                dtype=_dense_global_scoring_dtype(),
            )

        # RELION-style posterior-weighted noise update. Helper folds the
        # K-class (shared) / K=1 (per-half) / firstiter_cc-skip variants;
        # returns updated radial sigma2_noise + the unrolled
        # ``noise_variance`` representation consumed by the engine.
        noise_update = update_posterior_noise_variance(
            noise_stats_per_half=noise_stats_per_half,
            noise_variance_per_half=noise_variance_per_half,
            previous_noise_radial_per_half=previous_noise_radial_per_half,
            previous_noise_radial=previous_noise_radial,
            cryo=cryo,
            k_class_enabled=k_class_enabled,
            relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
            iteration=iteration,
            cs=current_size,
            maybe_dump_noise_update_debug=_maybe_dump_noise_update_debug,
        )
        noise_from_res = noise_update.noise_from_res
        noise_from_res_per_half = noise_update.noise_from_res_per_half
        noise_variance_per_half = noise_update.noise_variance_per_half
        noise_variance = noise_update.noise_variance
        previous_noise_radial = noise_update.previous_noise_radial
        previous_noise_radial_per_half = noise_update.previous_noise_radial_per_half
        if not relion_firstiter_cc_this_iter:
            _parity_dump.mark_stage(iteration, "noise_update")

        group_scale_corrections_for_dump = [None, None]
        norm_corrections_for_dump = [None, None]
        avg_norm_corrections_for_dump = [None, None]
        zero_norm_residual_counts_for_dump = [None, None]
        can_update_norm_scale = (
            noise_stats_per_half is not None
            and all(
                stats_k is not None
                and (
                    getattr(stats_k, "wsum_norm_correction", None) is not None
                    or int(experiment_datasets[_half_idx].n_units) == 0
                )
                for _half_idx, stats_k in enumerate(noise_stats_per_half)
            )
        )
        if follower_setup.follower_scale_state is not None and not can_update_norm_scale:
            raise RuntimeError(
                "Strict RELION follower-scale topology requires per-half norm/scale "
                "statistics at every numbered M-step"
            )
        if can_update_norm_scale:
            group_ids_per_half = [
                np.zeros(int(experiment_datasets[_half_idx].n_units), dtype=np.int64)
                if group_ids_k is None
                else group_ids_k
                for _half_idx, group_ids_k in enumerate(relion_half_inputs.group_ids)
            ]
            norm_scale_update = update_relion_norm_scale_corrections(
                noise_stats_per_half=noise_stats_per_half,
                image_corrections_per_half=relion_half_inputs.image_corrections,
                scale_corrections_per_half=relion_half_inputs.scale_corrections,
                group_ids_per_half=group_ids_per_half,
                group_count_per_half=relion_half_inputs.group_count,
                relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                do_norm_correction=True,
                do_scale_correction=follower_setup.follower_scale_state is None,
                dtype=_dense_global_scoring_dtype(),
            )
            if follower_setup.follower_scale_state is None:
                relion_half_inputs.image_corrections = norm_scale_update.image_corrections_per_half
                relion_half_inputs.scale_corrections = norm_scale_update.scale_corrections_per_half
                group_scale_corrections_for_dump = norm_scale_update.group_scale_corrections_per_half
            else:
                group_scale_corrections_for_dump = _update_relion_follower_corrections(
                    follower_setup,
                    noise_stats_per_half=noise_stats_per_half,
                    norm_scale_update=norm_scale_update,
                    relion_half_inputs=relion_half_inputs,
                    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                    dtype=_dense_global_scoring_dtype(),
                    logger=logger,
                )
            norm_corrections_for_dump = norm_scale_update.norm_corrections_per_half
            avg_norm_corrections_for_dump = norm_scale_update.avg_norm_correction_per_half
            zero_norm_residual_counts_for_dump = norm_scale_update.zero_norm_residual_counts
            if any(int(count) > 0 for count in norm_scale_update.zero_norm_residual_counts):
                logger.warning(
                    "RELION norm correction preserved previous image normalization for zero/tiny norm residuals: "
                    "half1=%d half2=%d",
                    int(norm_scale_update.zero_norm_residual_counts[0]),
                    int(norm_scale_update.zero_norm_residual_counts[1]),
                )
            logger.info(
                "RELION norm correction update: avg_norm half1=%.6g half2=%.6g; "
                "image_corr ranges half1=%s half2=%s; scale_corr ranges half1=%s half2=%s",
                float(norm_scale_update.avg_norm_correction_per_half[0]),
                float(norm_scale_update.avg_norm_correction_per_half[1]),
                _format_relion_correction_range(norm_scale_update.image_corrections_per_half[0]),
                _format_relion_correction_range(norm_scale_update.image_corrections_per_half[1]),
                _format_relion_correction_range(norm_scale_update.scale_corrections_per_half[0]),
                _format_relion_correction_range(norm_scale_update.scale_corrections_per_half[1]),
            )
        if follower_setup.follower_scale_state is not None:
            history.record_follower_scale_post_mstep(
                np.asarray(follower_setup.follower_scale_state.scales, dtype=np.float64).copy()
            )

        # Save per-iter per-shell sigma2 (after this iter's noise update) and
        # the exact shell-wise tau2 ingredients used in the Wiener update.
        history.record_noise_and_tau2(
            np.asarray(noise_from_res, dtype=np.float64),
            np.stack([np.asarray(noise_k, dtype=np.float64) for noise_k in noise_from_res_per_half], axis=0),
            None
            if tau2_update_details is None
            else {
                "prior_shells": np.asarray(tau2_update_details["prior_shells"], dtype=np.float64),
                "sigma2_shells": np.asarray(tau2_update_details["sigma2_shells"], dtype=np.float64),
                "avg_weight_shells": np.asarray(tau2_update_details["avg_weight_shells"], dtype=np.float64),
                "shell_sum": np.asarray(tau2_update_details["shell_sum"], dtype=np.float64),
                "shell_count": np.asarray(tau2_update_details["shell_count"], dtype=np.float64),
                "fsc_shells": None if k_class_enabled else np.asarray(tau2_update_details["fsc_shells"], dtype=np.float64),
                "ssnr_shells": np.asarray(tau2_update_details["ssnr_shells"], dtype=np.float64),
            },
            k_class_enabled=k_class_enabled,
        )

        # --- Update convergence state ---
        # This checks assignment changes, resolution stalls, and may trigger
        # angular step refinement or convergence.
        n_trans_current = current_translations.shape[0]

        # ``update_refinement_state`` expects ``new_resolution`` in
        # Angstroms (lower = better resolution), matching RELION's
        # ``mymodel.current_resolution``.  Convert from the shell index
        # ``pixel_res`` to Å here so the resol_gain stall detection
        # compares apples to apples (not shell-vs-shell with the wrong
        # sign).
        new_res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res,
            cryo.image_shape[0],
            cryo.voxel_size,
        )

        # This is a cheap support-width proxy, not RELION's full
        # map-perturbation calculateExpectedAngularErrors implementation.
        # Keep it in the output trajectory for diagnostics, but do not let it
        # stop K=1 refinements by default; collapsed one-sample support can
        # otherwise declare HEALPix-3 sampling "fine enough" too early.
        iter_acc_rot = exact_acc_rot_this_iter
        iter_acc_trans = exact_acc_trans_this_iter
        convergence_acc_rot = None
        convergence_acc_trans = None
        if iter_sig_counts is not None and len(iter_sig_counts) > 0:
            approx_acc_rot, _ = calculate_expected_angular_errors(
                state.healpix_order,
                iter_sig_counts,
                n_translations=n_trans_current,
            )
            approx_for_convergence, approx_convergence_reason = _approx_acc_rot_policy_for_convergence(
                logger=logger,
                state=state,
                iteration_number=iteration + 1,
                ave_pmax=ave_pmax,
                new_resolution_angstrom=new_res_angstrom,
            )
            if approx_for_convergence and exact_acc_rot_this_iter is None:
                convergence_acc_rot = approx_acc_rot
            logger.info(
                "approx_acc_rot=%.3f deg (from %d images, mean n_sig=%.1f, convergence=%s)",
                approx_acc_rot,
                len(iter_sig_counts),
                float(np.mean(iter_sig_counts)),
                approx_convergence_reason,
            )

        _optimiser_meta = None
        _optimiser_star = None
        if perturb_replay_relion_dir is not None and sealed_sampling_state is None:
            _optimiser_iter = int(init_relion_iteration) + iteration + 1
            _optimiser_star = os.path.join(
                perturb_replay_relion_dir,
                f"{perturb_replay_relion_prefix}_it{_optimiser_iter:03d}_optimiser.star",
            )
            if os.path.exists(_optimiser_star):
                try:
                    _optimiser_meta = read_relion_optimiser_metadata(_optimiser_star)
                    _relion_acc_rot = _optimiser_meta.get("overall_accuracy_rotations")
                    _relion_acc_trans_angst = _optimiser_meta.get("overall_accuracy_translations_angst")
                    if _relion_acc_rot is not None and np.isfinite(float(_relion_acc_rot)):
                        iter_acc_rot = float(_relion_acc_rot)
                        convergence_acc_rot = iter_acc_rot
                    if _relion_acc_trans_angst is not None and np.isfinite(float(_relion_acc_trans_angst)):
                        iter_acc_trans = float(_relion_acc_trans_angst)
                        convergence_acc_trans = iter_acc_trans
                    logger.info(
                        "Replay override: optimiser accuracy <- %s (acc_rot=%.3f deg, acc_trans=%s Å)",
                        _optimiser_star,
                        float(iter_acc_rot) if iter_acc_rot is not None else float("nan"),
                        f"{iter_acc_trans:.3f}" if iter_acc_trans is not None else "unset",
                    )
                except Exception as exc:
                    logger.warning(
                        "Replay override: failed to read optimiser metadata from %s: %s", _optimiser_star, exc
                    )

        state = update_refinement_state(
            state,
            current_assignments=current_combined_ha,
            previous_assignments=previous_combined_ha,
            n_translations=n_trans_current,
            translations=np.asarray(current_translations),
            new_resolution=new_res_angstrom,
            max_posterior_per_image=combined_max_posterior,
            acc_rot=convergence_acc_rot,
            acc_trans=convergence_acc_trans,
            current_rotation_matrices=current_rotation_matrices_combined,
            previous_rotation_matrices=previous_rotation_matrices_combined,
            current_translations_pixel=current_translations_pixel_combined,
            previous_translations_pixel=previous_translations_pixel_combined,
            current_classes=current_combined_classes,
            previous_classes=previous_combined_classes,
            ave_pmax_override=ave_pmax,
            voxel_size_angstrom=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
            update_sampling=not native_sampling_boundary,
            check_convergence_now=not native_sampling_boundary,
        )
        if _optimiser_meta is not None:
            apply_optimiser_convergence_replay(
                state,
                metadata=_optimiser_meta,
                optimiser_star=_optimiser_star,
                optimiser_iteration=_optimiser_iter,
                replay_dir=perturb_replay_relion_dir,
                replay_prefix=perturb_replay_relion_prefix,
                sealed_sampling_state=sealed_sampling_state,
                logger=logger,
            )

        # Reuse the assignment statistic computed by update_refinement_state.
        # Sampling transitions and optimiser replay preserve this field.
        frac_changed = state.fraction_changed
        state._last_frac_changed = frac_changed
        history.record_frac_changed(float(frac_changed))

        # --- C1 (RELION-parity): update sigma2_offset from data ---
        # Posterior-weighted RELION update with fallback to hard-assignment
        # proxy; see ``update_c1_sigma_offset_from_posterior`` for details.
        sigma_offset_result = update_c1_sigma_offset_from_posterior(
            noise_stats_per_half=noise_stats_per_half,
            noise_stats_per_half_per_class=noise_stats_per_half_per_class,
            current_sigma_offset_angstrom=current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half=current_sigma_offset_angstrom_per_half,
            n_classes=n_classes,
            k_class_enabled=k_class_enabled,
            state_fallback_offsets_angstrom=state.current_changes_optimal_offsets_angstrom,
        )
        current_sigma_offset_angstrom = sigma_offset_result.current_sigma_offset_angstrom
        current_sigma_offset_angstrom_per_half = _normalize_sigma_offset_per_half(
            sigma_offset_result.current_sigma_offset_angstrom_per_half
        )
        per_class_sigma_offset = sigma_offset_result.per_class_sigma_offset_angstrom
        history.record_sigma_offset_update(
            float(current_sigma_offset_angstrom),
            _copy_optional_float_pair(current_sigma_offset_angstrom_per_half),
            None if per_class_sigma_offset is None else per_class_sigma_offset.tolist(),
        )
        history.record_pose_accuracy_diagnostics(
            float(iter_acc_rot) if iter_acc_rot is not None else np.nan,
            float(iter_acc_trans) if iter_acc_trans is not None else np.nan,
            np.full(n_classes, np.nan, dtype=np.float64)
            if exact_acc_rot_per_class_this_iter is None
            else exact_acc_rot_per_class_this_iter,
            np.full(n_classes, np.nan, dtype=np.float64)
            if exact_acc_trans_per_class_this_iter is None
            else exact_acc_trans_per_class_this_iter,
            np.full(n_classes, -1, dtype=np.int64)
            if exact_accuracy_class_counts_this_iter is None
            else exact_accuracy_class_counts_this_iter,
            exact_accuracy_status_this_iter,
            float(state.current_changes_optimal_orientations),
            float(state.current_changes_optimal_offsets_angstrom),
        )

        # Save assignments for next iteration's change tracking.
        # Use coarse_ha (indexed into effective_rotations/current_rotations)
        # so that local search and convergence detection work correctly
        # regardless of whether adaptive oversampling was used.
        previous_assignments = [ha.copy() if ha is not None else None for ha in coarse_ha]
        previous_class_assignments = [cls.copy() if cls is not None else None for cls in class_assignments]
        _parity_dump.mark_stage(iteration, "convergence")

        if _parity_dump.is_active():
            try:
                _parity_dump.dump_iteration(
                    iteration=iteration,
                    init_relion_iteration=int(init_relion_iteration),
                    current_size=int(current_size),
                    sigma_offset=float(current_sigma_offset_angstrom),
                    translation_step=float(state.translation_step),
                    translation_range=float(state.translation_range),
                    random_perturbation=float(random_perturbation) if random_perturbation is not None else 0.0,
                    random_perturbation_instance=int(state.perturbation_instance)
                    if hasattr(state, "perturbation_instance")
                    else 0,
                    tau2_fudge=float(tau2_fudge),
                    voxel_size=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
                    grid_size=int(grid_size),
                    volume_shape=tuple(volume_shape),
                    ave_pmax=float(ave_pmax),
                    fsc=np.asarray(fsc, dtype=np.float64),
                    sigma2_noise=np.asarray(noise_variance, dtype=np.float64),
                    means=means,
                    unreg_means=unreg_means,
                    new_iter_best_rotation_eulers=new_iter_best_rotation_eulers,
                    new_iter_best_translations=new_iter_best_translations,
                    image_corrections=relion_half_inputs.image_corrections,
                    scale_corrections=relion_half_inputs.scale_corrections,
                    group_ids=relion_half_inputs.group_ids,
                    group_counts=relion_half_inputs.group_count,
                    group_scale_corrections=group_scale_corrections_for_dump,
                    norm_corrections=norm_corrections_for_dump,
                    avg_norm_corrections=avg_norm_corrections_for_dump,
                    zero_norm_residual_counts=zero_norm_residual_counts_for_dump,
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                )
            except Exception as exc:
                logger.warning("parity_dump.dump_iteration failed at iter %d: %s", iteration, exc)
        elif _parity_dump.timing_is_active():
            try:
                _parity_dump.dump_timing_iteration(
                    iteration=iteration,
                    init_relion_iteration=int(init_relion_iteration),
                    iteration_start=t0,
                )
            except Exception as exc:
                logger.warning("parity_dump.dump_timing_iteration failed at iter %d: %s", iteration, exc)

        # --- Timing ---
        elapsed = time.time() - t0
        history.record_wall_time(elapsed)

        res_angstrom = shell_index_to_resolution_angstrom(
            pixel_res,
            cryo.image_shape[0],
            cryo.voxel_size,
        )
        logger.info(
            "RELION Iteration %d: current_size=%d, pixel_res=%.1f, "
            "res=%.2f A, ave_Pmax=%.4f, healpix_order=%d, "
            "converged=%s, time=%.1fs",
            iteration + 1,
            current_size,
            pixel_res,
            res_angstrom,
            ave_pmax,
            state.healpix_order,
            state.has_converged,
            elapsed,
        )

        # End-of-iteration memory boundary.  The next iteration immediately
        # pads each half-map to the projection grid; keeping previous
        # backprojector accumulators or unregularized diagnostic maps live can
        # make high-resolution runs OOM before the batch-size estimator can act.
        try:
            jax.block_until_ready(means)
        except Exception:
            pass
        Ft_y_0 = Ft_y_1 = None
        Ft_ctf_0 = Ft_ctf_1 = None
        Ft_y_combined = Ft_ctf_combined = None
        unreg_means = previous_means = None
        mean_signal_variance_per_half = tau2_update_details_per_half = None
        noise_stats_per_half = noise_stats_per_half_per_class = None
        gc.collect()
        if os.environ.get("RECOVAR_RELION_CLEAR_JAX_CACHES_BETWEEN_ITERS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            jax.clear_caches()

        if state.has_converged and not schedule.force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d. Final resolution: %.2f A (pixel_res=%.1f)",
                iteration + 1,
                res_angstrom,
                pixel_res,
            )
            break
        if state.has_converged and schedule.force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d, continuing because force_max_iter_after_convergence=True",
                iteration + 1,
            )

        iteration += 1

    # Numbered contribution/native dump identities must never leak into the
    # return path or RELION's unnumbered final all-data pass.
    bpref_diagnostics.clear_bpref_contribution_dump_context()

    # RELION can enter final all-data only when checkConvergence() ran at the
    # top of a permitted loop iteration.  If the last numbered iteration
    # merely makes the state convergence-ready, ``iter <= nr_iter`` ends and
    # RELION does not synthesize another boundary after the cap.
    should_run_final_iteration = _should_run_final_all_data_iteration(
        has_converged=state.has_converged,
        iteration=iteration,
        max_iter=schedule.max_iter,
        force_max_iter_after_convergence=schedule.force_max_iter_after_convergence,
        k_class_enabled=k_class_enabled,
    )
    if schedule.skip_final_iteration or not should_run_final_iteration:
        if not schedule.skip_final_iteration and not should_run_final_iteration:
            logger.info(
                "Skipping RELION final all-data iteration: has_converged=%s, "
                "iteration=%d, max_iter=%d, force_max_iter_after_convergence=%s",
                state.has_converged,
                iteration,
                schedule.max_iter,
                schedule.force_max_iter_after_convergence,
            )
        merged_mean, merged_class_means = _merged_mean_from_halves(
            means,
            class_weights if k_class_enabled else None,
        )
        (
            replay_requested_iterations,
            replay_applied_iterations,
        ) = _finalize_relion_follower_scale_replay_telemetry(
            replay.relion_follower_scale_replay,
            applied_iterations=history.relion_follower_scale_replay_applied_iterations,
            logger=logger,
        )
        return {
            "mean": merged_mean,
            "means": means,
            "class_means": merged_class_means,
            "class_weights": class_weights if k_class_enabled else None,
            "class_assignments": class_assignments if k_class_enabled else None,
            "relion_follower_scale_replay_requested_iterations": replay_requested_iterations,
            "relion_follower_scale_replay_applied_iterations": replay_applied_iterations,
            **follower_setup.to_result_dict(history),
            "hard_assignments": hard_assignments,
            "convergence_state": state,
            "frozen_initial_scoring_state_sha256": frozen_initial_scoring_state_sha256,
            "expected_accuracy_trial_local_indices": expected_accuracy_trial_local_indices,
            "expected_accuracy_trial_particle_ids": expected_accuracy_trial_particle_ids,
            "final_all_data_ran": False,
            "setup_phase_seconds": setup_phase_seconds,
            **history.to_dict(),
        }
    if not state.has_converged:
        logger.info(
            "Diagnostic %s=1: running RELION final all-data iteration after max_iter exhaustion "
            "(iteration=%d, max_iter=%d)",
            _FINAL_ALL_DATA_AFTER_MAX_ITER_ENV,
            iteration,
            schedule.max_iter,
        )
    final_expected_accuracy = None
    final_expected_accuracy_status = "not_run"
    # --- RELION's final iteration: do_join_random_halves + do_use_all_data ---
    # After convergence, RELION runs ONE more iter with:
    #   - current_size = ori_size (Nyquist, all shells)
    #   - joined weighted sums for reconstruction
    #   - each half still scored against its own half-map
    # See ml_optimiser.cpp:10157-10160 (sets do_join_random_halves and
    # do_use_all_data) and ml_optimiser.cpp:5707-5708 (forces current_size to
    # ori_size when do_use_all_data is true).
    #
    # Implementation: run one more E+M at full Nyquist for each half, using
    # that half's own reference map, then join the weighted sums into one final
    # reconstruction.
    final_join_means = [means[0], means[1]]
    if (
        not k_class_enabled
        and os.environ.get(_FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV, "").strip().lower() in _TRUE_ENV_VALUES
    ):
        final_merged_reference, _ = _merged_mean_from_halves(means)
        final_join_means = [final_merged_reference, final_merged_reference]
        logger.info(
            "Diagnostic %s=1: final all-data K=1 E-step uses merged reference for both halves",
            _FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV,
        )
    final_replay_forced = (
        os.environ.get(_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV, "").strip().lower() in _TRUE_ENV_VALUES
    )
    final_replay_disabled = (
        os.environ.get(_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV, "").strip().lower() in _TRUE_ENV_VALUES
    )
    final_replay_has_overrides = replay.replay_iteration_overrides is not None and len(replay.replay_iteration_overrides) > 0
    final_replay_has_numbered_overrides = _has_numbered_replay_iteration_overrides(
        replay.replay_iteration_overrides
    )
    diagnostic_final_replay_override = final_replay_override
    if (
        (
            diagnostic_final_replay_override is not None
            or replay.final_replay_reference_maps is not None
        )
        and replay.final_replay_source_iteration is not None
        and int(len(history.current_sizes)) != int(replay.final_replay_source_iteration)
    ):
        raise RuntimeError(
            "Diagnostic final-only substitution source does not match autonomous convergence boundary: "
            f"source_iteration={int(replay.final_replay_source_iteration)} "
            f"numbered_iteration_count={int(len(history.current_sizes))}"
        )
    if replay.final_replay_reference_maps is not None:
        if k_class_enabled:
            raise RuntimeError(
                "Diagnostic final-only RELION reference substitution is currently K=1 only"
            )
        if len(replay.final_replay_reference_maps) != 2:
            raise ValueError(
                "Diagnostic final-only RELION reference substitution requires exactly two half maps"
            )
        expected_reference_shape = tuple(np.asarray(means[0]).shape)
        candidate_reference_shapes = [
            tuple(np.asarray(reference).shape)
            for reference in replay.final_replay_reference_maps
        ]
        if any(shape != expected_reference_shape for shape in candidate_reference_shapes):
            raise ValueError(
                "Diagnostic final-only RELION reference shape mismatch: "
                f"expected={expected_reference_shape} got={candidate_reference_shapes}"
            )
        final_join_means = [
            jnp.asarray(reference, dtype=means[half_idx].dtype)
            for half_idx, reference in enumerate(replay.final_replay_reference_maps)
        ]
        logger.info(
            "Diagnostic final-only RELION reference substitution at numbered boundary %d",
            int(len(history.current_sizes)),
        )
    final_replay_last_numbered_state = (
        diagnostic_final_replay_override is not None
        or (
            not final_replay_disabled
            and (final_replay_forced or final_replay_has_numbered_overrides)
        )
    )
    if final_replay_last_numbered_state:
        final_replay_requested_index = int(len(history.current_sizes))
        final_replay_override_index = final_replay_requested_index
        if diagnostic_final_replay_override is not None:
            final_replay_override = diagnostic_final_replay_override
            logger.info(
                "Diagnostic final-only RELION state substitution at previous-state index %d (fields=%s)",
                final_replay_requested_index,
                ",".join(sorted(final_replay_override)) or "<none>",
            )
        else:
            final_replay_override = None
        if diagnostic_final_replay_override is None and final_replay_has_overrides:
            final_replay_override_index = min(
                final_replay_requested_index,
                int(len(replay.replay_iteration_overrides)) - 1,
            )
            final_replay_override = replay.replay_iteration_overrides[final_replay_override_index]
            if final_replay_override_index != final_replay_requested_index:
                logger.info(
                    "RELION replay: final all-data requested previous-state index %d, "
                    "using last available numbered replay override index %d",
                    final_replay_requested_index,
                    final_replay_override_index,
                )
        if final_replay_override is None:
            if final_replay_has_overrides:
                raise RuntimeError(
                    "Strict RELION final all-data replay is missing the requested "
                    f"previous-state override at index {final_replay_requested_index}"
                )
            logger.info(
                "RELION replay: final all-data requested last numbered state replay, "
                "but no replay override exists for previous-state index %d",
                final_replay_requested_index,
            )
        else:
            _final_replay_fields = []
            _final_replay_sigma_per_half = final_replay_override.get("translation_sigma_angstrom_per_half")
            if _final_replay_sigma_per_half is not None:
                current_sigma_offset_angstrom_per_half = _normalize_sigma_offset_per_half(
                    _final_replay_sigma_per_half
                )
                current_sigma_offset_angstrom = _mean_sigma_offset_per_half(
                    current_sigma_offset_angstrom_per_half
                )
                _final_replay_fields.append("translation_sigma_angstrom_per_half")
            _final_replay_sigma = final_replay_override.get("translation_sigma_angstrom")
            if _final_replay_sigma is not None and _final_replay_sigma_per_half is None:
                current_sigma_offset_angstrom = float(_final_replay_sigma)
                current_sigma_offset_angstrom_per_half = _as_sigma_offset_half_pair(
                    current_sigma_offset_angstrom
                )
                _final_replay_fields.append("translation_sigma_angstrom")
            _final_replay_prev_trans = final_replay_override.get("previous_best_translations")
            if _final_replay_prev_trans is not None:
                relion_half_inputs.previous_best_translations = _copy_half_pair(_final_replay_prev_trans)
                _final_replay_fields.append("previous_best_translations")
            _final_replay_prev_eulers = final_replay_override.get("previous_best_rotation_eulers")
            if _final_replay_prev_eulers is not None:
                relion_half_inputs.previous_best_rotation_eulers = _copy_half_pair(_final_replay_prev_eulers)
                _final_replay_fields.append("previous_best_rotation_eulers")
            _final_replay_fields.extend(
                _apply_replay_correction_overrides(
                    relion_half_inputs=relion_half_inputs,
                    replay_override=final_replay_override,
                )
            )
            _final_replay_noise = final_replay_override.get("noise_variance")
            if _final_replay_noise is not None:
                noise_variance_per_half = _normalize_noise_variance_per_half(
                    _final_replay_noise,
                    n_halves=2,
                )
                noise_variance = _mean_noise_variance(noise_variance_per_half)
                previous_noise_radial_per_half = [
                    _radial_profile_from_noise_variance(noise_k, cryo.image_shape)
                    for noise_k in noise_variance_per_half
                ]
                previous_noise_radial = jnp.asarray(
                    np.mean(np.stack(previous_noise_radial_per_half, axis=0), axis=0),
                    dtype=_dense_global_scoring_dtype(),
                )
                _final_replay_fields.append("noise_variance")
            _final_replay_dir_prior = final_replay_override.get("direction_prior")
            if _final_replay_dir_prior is not None:
                _final_replay_prior_dtype = _dense_global_scoring_dtype()
                if k_class_enabled:
                    _final_replay_priors = normalize_class_direction_prior_per_half(
                        _final_replay_dir_prior,
                        n_classes,
                        dtype=_final_replay_prior_dtype,
                    )
                else:
                    _final_replay_priors = normalize_direction_prior_per_half(
                        _final_replay_dir_prior, dtype=_final_replay_prior_dtype
                    )
                for _half_idx in range(2):
                    if _final_replay_priors[_half_idx] is None:
                        continue
                    _prior_k = np.asarray(_final_replay_priors[_half_idx], dtype=_final_replay_prior_dtype)
                    _prior_order_k = infer_direction_prior_healpix_order(
                        _prior_k[0] if k_class_enabled else _prior_k
                    )
                    if _prior_order_k != state.healpix_order:
                        _prior_k = remap_half_direction_prior_to_healpix_order(
                            _prior_k,
                            _prior_order_k,
                            state.healpix_order,
                            n_classes=n_classes if k_class_enabled else None,
                            dtype=_final_replay_prior_dtype,
                        )
                        _prior_order_k = state.healpix_order
                    if k_class_enabled:
                        class_direction_prior_per_half[_half_idx] = normalize_class_direction_prior_per_half(
                            [_prior_k, None] if _half_idx == 0 else [None, _prior_k],
                            n_classes,
                            dtype=_final_replay_prior_dtype,
                        )[_half_idx]
                        class_direction_prior_order_per_half[_half_idx] = _prior_order_k
                    else:
                        global_direction_prior_per_half[_half_idx] = _prior_k
                        global_direction_prior_order_per_half[_half_idx] = _prior_order_k
                _final_replay_fields.append("direction_prior")
            logger.info(
                "RELION replay: final all-data replays last numbered RELION state "
                "(previous_state_index=%d, fields=%s)",
                final_replay_override_index,
                ",".join(_final_replay_fields) if _final_replay_fields else "<none>",
            )
    elif not k_class_enabled and final_replay_disabled and final_replay_has_overrides:
        logger.info(
            "Diagnostic %s=1: final all-data skips automatic last-numbered RELION state replay",
            _FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV,
        )
    if follower_setup.follower_scale_state is not None:
        _dispatch_relion_follower_scale_for_final_all_data(
            follower_setup,
            init_relion_iteration=int(init_relion_iteration),
            numbered_iteration_count=int(len(history.current_sizes)),
            relion_half_inputs=relion_half_inputs,
            dtype=_dense_global_scoring_dtype(),
            logger=logger,
        )
    final_noise_variance_per_half = noise_variance_per_half
    if not k_class_enabled:
        # RELION joins the half-set weighted sums after the post-convergence
        # expectation step.  During that E-step each MPI follower still owns
        # its numbered-iteration half model, so particles from random subset
        # 1 and 2 are scored with sigma2_noise from half 1 and 2 respectively.
        logger.info(
            "RELION final all-data: scoring each particle half with its own sigma2_noise",
        )
    final_iter_t0 = time.time()
    final_current_size = int(grid_size)  # = ori_size, full Nyquist
    if native_sampling_boundary:
        final_eulers_half1 = relion_half_inputs.previous_best_rotation_eulers[0]
        if expected_accuracy_trial_order is None or final_eulers_half1 is None:
            final_expected_accuracy_status = "unavailable_inputs"
            state.acc_rot = float("inf")
            state.acc_trans = float("inf")
            logger.warning(
                "RELION final all-data expected accuracy unavailable; "
                "final expectation remains fail-closed",
            )
        else:
            final_accuracy_class_ids = (
                class_assignments[0]
                if k_class_enabled and class_assignments[0] is not None
                else np.zeros(int(experiment_datasets[0].n_units), dtype=np.int32)
            )
            try:
                final_expected_accuracy = estimate_relion_expected_accuracy(
                    reference_fourier=final_join_means[0],
                    volume_shape=tuple(volume_shape),
                    best_eulers_deg=final_eulers_half1,
                    class_ids=final_accuracy_class_ids,
                    class_weights=class_weights,
                    sigma2_noise_native=previous_noise_radial_per_half[0],
                    dataset=experiment_datasets[0],
                    trial_order_local=expected_accuracy_trial_order,
                    current_image_size=final_current_size,
                    padding_factor=PROJECTION_PADDING_FACTOR,
                    sigma2_fudge=float(tau2_fudge),
                    random_seed=int(effective_optimizer_random_seed),
                    random_seed_particle_ids=expected_accuracy.half1_particle_ids,
                    ctf_params_override=expected_accuracy.half1_ctf_params,
                    do_ctf_correction=expected_accuracy.do_ctf_correction,
                )
                state.acc_rot = float(final_expected_accuracy.acc_rot)
                state.acc_trans = float(final_expected_accuracy.acc_trans_angstrom)
                state = update_angular_sampling(state)
                final_expected_accuracy_status = "ok"
                logger.info(
                    "RELION final all-data expected accuracy: acc_rot=%.3f deg, "
                    "acc_trans=%.4f A",
                    state.acc_rot,
                    state.acc_trans,
                )
            except Exception as exc:
                final_expected_accuracy_status = f"error:{type(exc).__name__}:{exc}"
                state.acc_rot = float("inf")
                state.acc_trans = float("inf")
                logger.warning(
                    "RELION final all-data expected-accuracy estimation failed: %s",
                    exc,
                )
    final_current_healpix_order = _exhaustive_grid_order_for_state(state)
    if final_current_healpix_order == current_healpix_order:
        final_current_rotations = current_rotations
        final_current_rotation_eulers = current_rotation_eulers
    else:
        final_current_rotations, final_current_rotation_eulers = _relion_rotation_grid_float32(
            final_current_healpix_order, dtype=_dense_global_scoring_dtype()
        )
    final_effective_rotations = final_current_rotations
    final_effective_rotation_eulers = np.asarray(
        final_current_rotation_eulers,
        dtype=_dense_global_scoring_dtype(),
    )
    final_effective_mstep_rotations = None
    final_base_translations = jnp.asarray(
        _translation_grid_for_class_count(
            state.translation_range,
            state.translation_step,
            n_classes=n_classes,
            source_units_per_pixel=(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
        ).astype(_dense_global_scoring_dtype(), copy=False),
        dtype=_dense_global_scoring_dtype(),
    )
    final_current_translations = final_base_translations
    final_translation_range = float(state.translation_range)
    final_translation_step = float(state.translation_step)
    # RELION's final all-data iteration is logged as the next iteration after
    # the last split-half refinement step. Prefer an explicit numbered final
    # STAR when available; otherwise use the unnumbered final run_sampling.star.
    # Current RELION writes final all-data metadata from run_sampling.star.
    final_sampling_relion_iteration = int(init_relion_iteration) + int(len(history.current_sizes)) + 1
    final_numbered_sampling_relion_iteration = int(init_relion_iteration) + int(len(history.current_sizes))
    final_sampling_star = None
    final_sampling_star_source = None
    final_random_perturbation = 0.0
    final_perturbation_factor = float(parity.perturb_factor)
    final_perturbation_healpix_order = _native_final_perturbation_healpix_order(
        state,
        final_current_healpix_order,
    )
    final_perturbation_applied = False
    final_sampling_replay_dir = (
        parity.final_sampling_replay_relion_dir
        if parity.final_sampling_replay_relion_dir is not None
        else perturb_replay_relion_dir
    )
    if final_sampling_replay_dir is not None:
        final_sampling_star, final_sampling_star_source, final_sampling_candidates = select_final_sampling_star(
            final_sampling_replay_dir,
            perturb_replay_relion_prefix,
            final_iteration=final_sampling_relion_iteration,
            previous_iteration=final_numbered_sampling_relion_iteration,
            require_final_state=replay.replay_iteration_overrides is not None,
        )
        if final_sampling_star is not None:
            final_replay_meta = read_relion_sampling_metadata(final_sampling_star)
            final_perturbation_factor = float(final_replay_meta.get("perturbation_factor", parity.perturb_factor))
            final_replay_relion_iteration = (
                final_numbered_sampling_relion_iteration
                if final_sampling_star_source == "last-numbered"
                else final_sampling_relion_iteration
            )
            final_restart_state_iteration = _perturbation_restart_state_iteration(
                parity.perturb_replay_restart_state_iterations,
                final_replay_relion_iteration,
            )
            final_random_perturbation, final_perturbation_source = _resolve_replay_random_perturbation(
                star_value=float(final_replay_meta["random_perturbation"]),
                perturbation_factor=final_perturbation_factor,
                relion_iteration=final_replay_relion_iteration,
                replay_dir=str(final_sampling_replay_dir),
                replay_prefix=perturb_replay_relion_prefix,
                explicit_seed=parity.perturb_seed,
                precision_mode=str(parity.perturb_replay_precision),
                restart_state_iteration=final_restart_state_iteration,
            )
            final_perturbation_healpix_order = int(final_replay_meta["healpix_order"])
            px = float(cryo.voxel_size) if cryo.voxel_size > 0 else 1.0
            final_translation_range = float(final_replay_meta["offset_range"]) / px
            final_translation_step = float(final_replay_meta["offset_step"]) / px
            numbered_sampling_path = os.path.join(
                final_sampling_replay_dir,
                f"{perturb_replay_relion_prefix}_it{final_numbered_sampling_relion_iteration:03d}_sampling.star",
            )
            if (
                final_sampling_star_source == "final"
                and os.path.exists(numbered_sampling_path)
            ):
                numbered_meta = read_relion_sampling_metadata(numbered_sampling_path)
                numbered_range = float(numbered_meta["offset_range"]) / px
                numbered_step = float(numbered_meta["offset_step"]) / px
                numbered_grid = _translation_grid_for_class_count(
                    numbered_range,
                    numbered_step,
                    n_classes=n_classes,
                    source_units_per_pixel=px,
                ).astype(np.float32)
                final_grid_preview = _translation_grid_for_class_count(
                    final_translation_range,
                    final_translation_step,
                    n_classes=n_classes,
                    source_units_per_pixel=px,
                ).astype(np.float32)
                same_shape = numbered_grid.shape == final_grid_preview.shape
                same_grid = bool(
                    same_shape
                    and np.allclose(
                        numbered_grid,
                        final_grid_preview,
                        rtol=0.0,
                        atol=1e-6,
                    )
                )
                if not same_grid:
                    logger.info(
                        "RELION final all-data sampling grid differs from last numbered sampling: "
                        "numbered n=%d range=%.9g step=%.9g hp=%d; final n=%d range=%.9g step=%.9g hp=%d",
                        int(numbered_grid.shape[0]),
                        numbered_range,
                        numbered_step,
                        int(numbered_meta["healpix_order"]),
                        int(final_grid_preview.shape[0]),
                        final_translation_range,
                        final_translation_step,
                        final_perturbation_healpix_order,
                    )
            final_base_translations = jnp.asarray(
                _translation_grid_for_class_count(
                    final_translation_range,
                    final_translation_step,
                    n_classes=n_classes,
                    source_units_per_pixel=px,
                ).astype(_dense_global_scoring_dtype(), copy=False),
                dtype=_dense_global_scoring_dtype(),
            )
            final_current_translations = final_base_translations
            logger.info(
                "Perturbation replay: final all-data relion_iter=%d rp=%+.12g pf=%.3f "
                "relion_hp_order=%d offset_range=%.3f px offset_step=%.3f px source=%s/%s",
                final_replay_relion_iteration,
                final_random_perturbation,
                final_perturbation_factor,
                final_perturbation_healpix_order,
                final_translation_range,
                final_translation_step,
                final_sampling_star_source,
                final_perturbation_source,
            )
        else:
            missing_sampling_stars = ", ".join(path for path, _source in final_sampling_candidates)
            logger.info(
                "Perturbation replay: final all-data sampling STAR missing for relion_iter=%d (%s); "
                "leaving final trial grid unperturbed",
                final_sampling_relion_iteration,
                missing_sampling_stars,
            )
            final_sampling_star = None
    elif parity.perturb_factor > 0:
        if parity.perturb_seed is not None:
            seed = int(parity.perturb_seed) + final_sampling_relion_iteration
            final_random_perturbation = advance_relion_perturbation_from_seed(
                random_perturbation,
                parity.perturb_factor,
                seed=seed,
            )
            logger.info(
                "Perturbation advance: final all-data relion_iter=%d seed=%d rp=%+.5f",
                final_sampling_relion_iteration,
                seed,
                final_random_perturbation,
            )
        else:
            final_random_perturbation = advance_relion_perturbation(
                random_perturbation,
                parity.perturb_factor,
                perturb_rng,
            )
            logger.info(
                "Perturbation advance: final all-data relion_iter=%d rp=%+.5f",
                final_sampling_relion_iteration,
                final_random_perturbation,
            )
    if (
        final_sampling_star is not None
        or (perturb_replay_relion_dir is None and parity.perturb_factor > 0)
    ):
        final_angsamp_deg = relion_angular_sampling_deg(
            final_perturbation_healpix_order,
            adaptive_oversampling=0,
        )
        final_mstep_source_eulers = _get_relion_rotation_grid_eulers_float64(
            final_perturbation_healpix_order
        )
        if int(final_mstep_source_eulers.shape[0]) != int(final_effective_rotation_eulers.shape[0]):
            final_mstep_source_eulers = np.asarray(final_effective_rotation_eulers, dtype=np.float64)
        final_effective_rotations, final_effective_rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
            final_effective_rotation_eulers,
            final_random_perturbation,
            final_angsamp_deg,
            dtype=_dense_global_scoring_dtype(),
        )
        _, _, final_effective_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
            final_mstep_source_eulers,
            final_random_perturbation,
            final_angsamp_deg,
            return_mstep_rotations=True,
            dtype=_dense_global_scoring_dtype(),
        )
        final_current_translations = jnp.asarray(
            apply_relion_translation_perturbation(
                np.asarray(final_base_translations),
                final_random_perturbation,
                final_translation_step,
            ),
            dtype=_dense_global_scoring_dtype(),
        )
        final_perturbation_applied = True
    final_use_local = bool(
        (not k_class_enabled)
        and state.do_local_search
        and all(eulers is not None for eulers in relion_half_inputs.previous_best_rotation_eulers)
        and all(trans is not None for trans in relion_half_inputs.previous_best_translations)
    )
    final_local_search_order = None
    final_local_search_rotations = None
    final_local_search_rotation_eulers = None
    final_local_search_mstep_rotations = None
    final_local_search_random_perturbation = 0.0
    final_local_search_angular_sampling_deg = None
    final_local_parent_oversampling_order = int(state.adaptive_oversampling)
    final_local_pass1_current_size = final_current_size
    final_adaptive_pass1_current_size = None
    final_adaptive_pass2_current_size = None
    final_sigma_rot = state.sigma_rot
    final_sigma_psi = state.sigma_psi if state.sigma_psi > 0 else final_sigma_rot
    if final_use_local:
        if final_sigma_rot <= 0:
            step_rad = np.deg2rad(healpix_angular_step(state.healpix_order) / (2**state.adaptive_oversampling))
            final_sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
            final_sigma_psi = final_sigma_rot
        final_local_parent_order, final_local_search_order = _final_local_sampling_orders(
            state_healpix_order=int(state.healpix_order),
            adaptive_oversampling=int(state.adaptive_oversampling),
            final_sampling_healpix_order=(
                int(final_perturbation_healpix_order)
                if final_sampling_star is not None
                else None
            ),
        )
        use_parent_expanded_final_local = int(state.adaptive_oversampling) > 0
        if final_effective_rotations.shape[0] != rotation_grid_size(final_local_search_order):
            final_local_search_angular_sampling_deg = relion_angular_sampling_deg(
                final_local_search_order,
                adaptive_oversampling=0,
            )
            if (not use_parent_expanded_final_local) and _precompute_exact_local_fine_grid_enabled(
                final_local_search_order
            ):
                _final_local_search_use_float64_scoring, _final_local_search_use_float64_projections = (
                    _local_search_precision_flags(final_sampling_relion_iteration, pass_index=2)
                )
                final_local_search_rotations, final_local_search_rotation_eulers = _relion_rotation_grid_float32(
                    final_local_search_order,
                    dtype=(
                        np.float64
                        if (_final_local_search_use_float64_scoring or _final_local_search_use_float64_projections)
                        else np.float32
                    ),
                )
                if final_perturbation_applied:
                    final_local_search_rotations, final_local_search_rotation_eulers = (
                        apply_relion_rotation_perturbation_to_eulers(
                            final_local_search_rotation_eulers,
                            final_random_perturbation,
                            final_local_search_angular_sampling_deg,
                        )
                    )
                    _, _, final_local_search_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
                        _get_relion_rotation_grid_eulers_float64(final_local_search_order),
                        final_random_perturbation,
                        final_local_search_angular_sampling_deg,
                        return_mstep_rotations=True,
                    )
                else:
                    _, _, final_local_search_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
                        _get_relion_rotation_grid_eulers_float64(final_local_search_order),
                        0.0,
                        final_local_search_angular_sampling_deg,
                        return_mstep_rotations=True,
                    )
            else:
                final_local_search_rotations = None
                final_local_search_rotation_eulers = None
                if final_perturbation_applied:
                    final_local_search_random_perturbation = float(final_random_perturbation)
                if use_parent_expanded_final_local:
                    parent_order = final_local_parent_order
                    parent_step_deg = healpix_angular_step(parent_order)
                    local_coarse_size = compute_coarse_image_size(
                        parent_step_deg,
                    cryo.voxel_size if cryo.voxel_size > 0 else 1.0,
                    grid_size,
                    particle_diameter=particle_diameter_ang,
                )
                local_coarse_size = clamp_relion_coarse_image_size(
                    local_coarse_size,
                    final_current_size,
                    grid_size,
                )
                final_local_pass1_current_size = local_coarse_size if local_coarse_size < grid_size else None
        else:
            final_local_search_rotations = final_effective_rotations
            final_local_search_rotation_eulers = final_effective_rotation_eulers
            if final_effective_mstep_rotations is not None:
                final_local_search_mstep_rotations = final_effective_mstep_rotations
            else:
                final_mstep_source_eulers = _get_relion_rotation_grid_eulers_float64(final_local_search_order)
                if int(final_mstep_source_eulers.shape[0]) != int(final_effective_rotation_eulers.shape[0]):
                    final_mstep_source_eulers = np.asarray(final_effective_rotation_eulers, dtype=np.float64)
                _, _, final_local_search_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
                    final_mstep_source_eulers,
                    0.0,
                    relion_angular_sampling_deg(final_local_search_order, adaptive_oversampling=0),
                    return_mstep_rotations=True,
                )
        logger.info(
            "RELION final all-data iteration using local search: parent_order=%d fine_order=%d, "
            "sigma_rot=%.4f rad (%.2f deg), sigma_psi=%.4f rad, perturbation=%+.5f",
            final_local_parent_order,
            final_local_search_order,
            final_sigma_rot,
            np.rad2deg(final_sigma_rot),
            final_sigma_psi,
            final_random_perturbation if final_perturbation_applied else 0.0,
        )
    else:
        logger.info(
            "RELION final all-data iteration using dense global scoring: healpix_order=%d, perturbation=%+.5f",
            final_current_healpix_order,
            final_random_perturbation if final_perturbation_applied else 0.0,
        )
        if k_class_enabled and int(state.adaptive_oversampling) > 0:
            final_coarse_size = compute_coarse_image_size(
                healpix_angular_step(final_current_healpix_order),
                cryo.voxel_size if cryo.voxel_size > 0 else 1.0,
                grid_size,
                particle_diameter=particle_diameter_ang,
            )
            final_adaptive_pass1_current_size = clamp_relion_coarse_image_size(
                final_coarse_size,
                final_current_size,
                grid_size,
            )
            final_adaptive_pass2_current_size = final_current_size
            logger.info(
                "RELION final all-data adaptive K-class pass-1: coarse_current_size=%d "
                "fine_current_size=%d oversampling=%d",
                final_adaptive_pass1_current_size,
                final_adaptive_pass2_current_size,
                int(state.adaptive_oversampling),
            )
    final_relion_projector_half_by_half = [None, None]
    final_relion_projector_r_max_by_half = [None, None]
    if final_use_local or int(state.adaptive_oversampling) > 0:
        projector_t0 = time.time()
        for _half_idx in range(2):
            projector_half, projector_r_max = _relion_projector_half_maps_for_scoring(
                final_join_means[_half_idx],
                volume_shape=volume_shape,
                current_size=final_current_size,
                padding_factor=PROJECTION_PADDING_FACTOR,
                n_classes=n_classes,
                dump_label=f"final_half{_half_idx}",
            )
            final_relion_projector_half_by_half[_half_idx] = projector_half
            final_relion_projector_r_max_by_half[_half_idx] = projector_r_max
        logger.info(
            "RELION final all-data: built exact Projector::data for scoring at current_size=%d r_max=%s in %.2fs",
            final_current_size,
            final_relion_projector_r_max_by_half[0],
            time.time() - projector_t0,
        )
    logger.info("=== RELION final all-data Nyquist iteration ===")
    final_use_float64_scoring, final_use_float64_projections = _local_search_precision_flags(
        final_sampling_relion_iteration,
        pass_index=2,
        static_em_kwargs=_DENSE_EM_STATIC_KWARGS,
    )
    final_outs = PerHalfOutputs.empty()
    for k in range(2):
        bpref_diagnostics.clear_bpref_contribution_dump_context()
        final_half_t0 = time.time()
        logger.info(
            "BPREF_DEVICE_SIGNATURE_ACTIVATION iteration=%d half=%d "
            "final_all_data=true active=false",
            iteration + 1,
            k + 1,
        )
        logger.info(
            "RELION final all-data half-%d start: images=%d current_size=%d "
            "healpix_order=%d n_rot=%d n_trans=%d local_search=%s",
            k + 1,
            experiment_datasets[k].n_units,
            final_current_size,
            final_current_healpix_order,
            final_current_rotations.shape[0],
            final_current_translations.shape[0],
            final_use_local,
        )
        # Pass the merged mean as input (both halves get the same projection source).
        # Run on each half-set's particles (avoids loading all particles at once),
        # then accumulate Ft_y/Ft_ctf and noise stats from BOTH halves.
        previous_translations_k = relion_half_inputs.previous_best_translations[k]
        translation_search_base = relion_translation_search_base(
            previous_translations_k, dtype=_dense_global_scoring_dtype()
        )
        final_outs.translation_search_bases[k] = translation_search_base
        final_sigma_offset_k = _sigma_offset_for_half(
            current_sigma_offset_angstrom,
            current_sigma_offset_angstrom_per_half,
            k,
        )
        final_trans_prior_center = relion_translation_prior_center(
            previous_translations_k,
            cryo.voxel_size,
            dtype=_dense_global_scoring_dtype(),
        )
        final_local_trans_prior_center = relion_translation_prior_center(
            previous_translations_k,
            cryo.voxel_size,
            dtype=_dense_global_scoring_dtype(),
        )
        final_trans_sigma_center = relion_sigma_offset_prior_center(
            previous_translations_k, dtype=_dense_global_scoring_dtype()
        )
        final_trans_prior_center_for_engine = (
            np.zeros(2, dtype=_dense_global_scoring_dtype())
            if final_trans_sigma_center is None
            else final_trans_sigma_center
        )
        final_translation_prior_translations = np.asarray(
            final_base_translations, dtype=_dense_global_scoring_dtype()
        )
        if final_current_translations.shape[0] != final_base_translations.shape[0]:
            if final_current_translations.shape[0] == 1 and final_base_translations.shape[0] > 1:
                center_idx = int(final_base_translations.shape[0] // 2)
                final_translation_prior_translations = np.asarray(
                    final_base_translations[center_idx : center_idx + 1],
                    dtype=_dense_global_scoring_dtype(),
                )
            else:
                final_translation_prior_translations = np.asarray(
                    final_current_translations, dtype=_dense_global_scoring_dtype()
                )
        final_translation_log_prior = make_relion_translation_log_prior(
            final_translation_prior_translations,
            cryo.voxel_size,
            final_sigma_offset_k,
            final_trans_prior_center,
            offset_range_pixels=None,
            dtype=_dense_global_scoring_dtype(),
        )
        final_rotation_log_prior_k = None
        final_class_rotation_log_prior_k = None
        final_direction_prior_healpix_order = None
        if not final_use_local:
            final_direction_prior_healpix_order = _direction_prior_healpix_order_for_scoring(
                use_local=False,
                current_healpix_order=final_current_healpix_order,
                state_healpix_order=state.healpix_order,
                adaptive_oversampling=final_local_parent_oversampling_order,
                local_search_order=None,
            )
        if (
            not final_use_local
            and final_direction_prior_healpix_order is not None
            and k_class_enabled
            and class_direction_prior_per_half[k] is not None
            and class_direction_prior_order_per_half[k] == final_direction_prior_healpix_order
        ):
            final_class_rotation_log_prior_k = np.stack(
                [
                    make_relion_direction_log_prior(
                        class_direction_prior_per_half[k][class_idx],
                        final_direction_prior_healpix_order,
                        dtype=_dense_global_scoring_dtype(),
                    )
                    for class_idx in range(n_classes)
                ],
                axis=0,
            )
        elif (
            not final_use_local
            and final_direction_prior_healpix_order is not None
            and not k_class_enabled
            and global_direction_prior_per_half[k] is not None
            and global_direction_prior_order_per_half[k] == final_direction_prior_healpix_order
        ):
            final_rotation_log_prior_k = make_relion_direction_log_prior(
                global_direction_prior_per_half[k],
                final_direction_prior_healpix_order,
                dtype=_dense_global_scoring_dtype(),
            )
        if final_use_local:
            final_result = _score_half_local_in_bpref_scope(
                bpref_device_signature_active=False,
                k=k,
                experiment_dataset=experiment_datasets[k],
                means_k=final_join_means[k],
                mean_variance=mean_variance,
                noise_variance_k=final_noise_variance_per_half[k],
                previous_best_rotation_eulers_k=relion_half_inputs.previous_best_rotation_eulers[k],
                local_search_rotations=final_local_search_rotations,
                local_search_rotation_eulers=final_local_search_rotation_eulers,
                local_search_mstep_rotations=final_local_search_mstep_rotations,
                local_search_order=final_local_search_order,
                sigma_rot=final_sigma_rot,
                sigma_psi=final_sigma_psi,
                current_translations=final_current_translations,
                base_translations=final_base_translations,
                trans_prior_center=final_local_trans_prior_center,
                trans_prior_center_for_engine=final_trans_prior_center_for_engine,
                current_sigma_offset_angstrom=final_sigma_offset_k,
                current_translation_range=final_translation_range,
                disc_type=options.disc_type,
                cs_for_engine=final_current_size,
                local_pass1_current_size=final_local_pass1_current_size,
                image_corrections_k=relion_half_inputs.image_corrections[k],
                scale_corrections_k=relion_half_inputs.scale_corrections[k],
                group_ids_k=follower_setup.scale_stats_group_ids_per_half[k],
                group_count_k=follower_setup.scale_stats_group_count_per_half[k],
                scale_correction_data_vs_prior=previous_data_vs_prior_for_scheduling,
                translation_search_base=translation_search_base,
                disable_adjoint_y=debug.disable_adjoint_y,
                disable_adjoint_ctf=debug.disable_adjoint_ctf,
                max_significants=adaptive.max_significants,
                iteration=iteration + 1,
                debug_iteration=final_sampling_relion_iteration,
                save_intermediates_dir=debug.save_intermediates_dir,
                local_search_random_perturbation=final_local_search_random_perturbation,
                local_search_angular_sampling_deg=final_local_search_angular_sampling_deg,
                local_parent_oversampling_order=final_local_parent_oversampling_order,
                local_search_translation_prior_mode=local_search.local_search_translation_prior_mode,
                replay_prior_translations=None,
                class_log_priors=class_log_priors,
                k_class_enabled=False,
                collect_local_search_profile=collect_local_search_profile,
                diagnostic_score_only=False,
                safe_batch_sizes=_safe_batch_sizes,
                outputs=final_outs,
                local_profile_history=history.local_profile_history,
                relion_projector_half=final_relion_projector_half_by_half[k],
                relion_projector_r_max=final_relion_projector_r_max_by_half[k],
            )
        else:
            final_result = _score_half_dense_in_bpref_scope(
                bpref_device_signature_active=False,
                k=k,
            experiment_dataset=experiment_datasets[k],
            means_k=final_join_means[k],
            mean_variance=mean_variance,
            noise_variance_k=final_noise_variance_per_half[k],
            effective_rotations=final_effective_rotations,
            current_translations=final_current_translations,
            base_translations=final_base_translations,
            current_healpix_order=final_current_healpix_order,
            state=state,
            random_perturbation=final_random_perturbation if final_perturbation_applied else 0.0,
            disc_type=options.disc_type,
            image_batch_size=batching.image_batch_size,
                rotation_log_prior_k=final_rotation_log_prior_k,
                class_rotation_log_prior_k=final_class_rotation_log_prior_k,
                translation_log_prior=final_translation_log_prior,
                translation_search_base=translation_search_base,
                trans_prior_center_for_engine=final_trans_prior_center_for_engine,
                image_corrections_k=relion_half_inputs.image_corrections[k],
                scale_corrections_k=relion_half_inputs.scale_corrections[k],
                group_ids_k=follower_setup.scale_stats_group_ids_per_half[k],
                group_count_k=follower_setup.scale_stats_group_count_per_half[k],
                scale_correction_data_vs_prior=previous_data_vs_prior_for_scheduling,
                firstiter_score_mode_this_iter="gaussian",
                firstiter_winner_take_all_this_iter=False,
                cs_for_engine=final_current_size,
                class_log_priors=class_log_priors,
                k_class_enabled=k_class_enabled,
                relion_firstiter_cc_this_iter=False,
                disable_adjoint_y=debug.disable_adjoint_y,
                disable_adjoint_ctf=debug.disable_adjoint_ctf,
                safe_batch_sizes=_safe_batch_sizes,
                max_significants=adaptive.max_significants,
                outputs=final_outs,
                relion_projector_half=final_relion_projector_half_by_half[k],
                relion_projector_r_max=final_relion_projector_r_max_by_half[k],
                firstiter_coarse_current_size=final_adaptive_pass1_current_size,
                firstiter_fine_current_size=final_adaptive_pass2_current_size,
                firstiter_log_label="final all-data ",
                firstiter_updates_em_kwargs_ibs=True,
                return_best_pose_details=not k_class_enabled,
                debug_iteration=final_sampling_relion_iteration,
                preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
                source_faithful_spectrum_norm=source_faithful_spectrum_norm,
            )
        if final_result.best_pose_translations is not None:
            final_result.best_pose_translations = _relion_metadata_translations(
                relion_half_inputs.previous_best_translations[k],
                final_result.best_pose_translations,
                dtype=_dense_global_scoring_dtype(),
            )
        final_outs.update_from(k, final_result, dtype=_dense_global_scoring_dtype())
        _record_score_profile(
            history.global_profile_history,
            final_result,
            phase="final_all_data",
            iteration=iteration + 1,
            relion_iteration=final_sampling_relion_iteration,
            half_index=k,
            current_size=final_current_size,
            healpix_order=final_current_healpix_order,
            k_class_enabled=k_class_enabled,
        )
        logger.info(
            "RELION final all-data half-%d done: wall=%.1fs",
            k + 1,
            time.time() - final_half_t0,
        )
        # --- Manifest dump for final all-data iteration (Phase 0.1) ---
        if debug.save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                debug.save_intermediates_dir,
                f"manifest_final_half{k}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(final_effective_rotations, dtype=np.float32),
                "current_translations": np.asarray(final_current_translations, dtype=np.float32),
                "rotation_log_prior": np.asarray(final_rotation_log_prior_k, dtype=np.float64)
                if final_rotation_log_prior_k is not None
                else np.array([]),
                "translation_log_prior": np.asarray(final_translation_log_prior, dtype=np.float64),
                "translation_prior_centers": np.asarray(final_trans_prior_center_for_engine, dtype=np.float64),
                "image_corrections": np.asarray(relion_half_inputs.image_corrections[k], dtype=np.float64)
                if relion_half_inputs.image_corrections[k] is not None
                else np.array([]),
                "scale_corrections": np.asarray(relion_half_inputs.scale_corrections[k], dtype=np.float64)
                if relion_half_inputs.scale_corrections[k] is not None
                else np.array([]),
                "image_pre_shifts": np.asarray(translation_search_base, dtype=np.float32)
                if translation_search_base is not None
                else np.array([]),
                "absolute_previous_translations": np.asarray(
                    relion_half_inputs.previous_best_translations[k],
                    dtype=np.float32,
                )
                if relion_half_inputs.previous_best_translations[k] is not None
                else np.array([]),
                "mean_vol_ft": np.asarray(final_join_means[k]),
                "mean_variance": np.asarray(mean_variance),
                "noise_variance": np.asarray(final_noise_variance_per_half[k]),
                "current_size": np.int32(final_current_size),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(final_use_float64_scoring),
                "use_float64_projections": np.bool_(final_use_float64_projections),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(final_random_perturbation if final_perturbation_applied else 0.0),
                "perturbation_factor": np.float64(final_perturbation_factor),
                "perturbation_applied": np.bool_(final_perturbation_applied),
                "perturbation_relion_iteration": np.int32(final_sampling_relion_iteration),
                "local_search": np.bool_(final_use_local),
                "iteration": np.int32(-1),
                "half_index": np.int32(k),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Final manifest dumped: %s", _manifest_path)

    final_Ft_y_0 = final_outs.Ft_y[0]
    final_Ft_y_1 = final_outs.Ft_y[1]
    final_Ft_ctf_0 = final_outs.Ft_ctf[0]
    final_Ft_ctf_1 = final_outs.Ft_ctf[1]
    # RELION writes the converged half BackProjectors to temporary files before
    # joinTwoHalvesAtLowResolution mutates their low-frequency voxels.  Those
    # saved, pre-join arrays are later used for run_half*_class001_unfil.mrc.
    # Keep both boundaries explicit: the joined arrays below drive FSC/tau2 and
    # the final joined reconstruction, while these arrays drive unfiltered
    # half-map output.
    final_unfiltered_Ft_y_0 = final_Ft_y_0
    final_unfiltered_Ft_y_1 = final_Ft_y_1
    final_unfiltered_Ft_ctf_0 = final_Ft_ctf_0
    final_unfiltered_Ft_ctf_1 = final_Ft_ctf_1
    final_mstep_accumulator_shape = _resolve_mstep_accumulator_shape(
        final_outs.mstep_accumulator_shape,
        padded_volume_shape,
    )
    if not k_class_enabled and parity.low_resol_join_halves_angstrom is not None and parity.low_resol_join_halves_angstrom > 0:
        final_prev_res_angstrom = None
        if history.pixel_resolutions:
            final_prev_pixel_res = history.pixel_resolutions[-1]
            if final_prev_pixel_res > 0:
                final_prev_res_angstrom = shell_index_to_resolution_angstrom(
                    final_prev_pixel_res,
                    grid_size,
                    cryo.voxel_size,
                )
        elif np.isfinite(float(getattr(state, "current_resolution", float("inf")))):
            final_prev_res_angstrom = float(state.current_resolution)
        final_Ft_y_0, final_Ft_y_1, final_Ft_ctf_0, final_Ft_ctf_1 = regularization.join_halves_at_low_resolution(
            final_Ft_y_0,
            final_Ft_y_1,
            final_Ft_ctf_0,
            final_Ft_ctf_1,
            final_mstep_accumulator_shape,
            cryo.voxel_size,
            grid_size,
            parity.low_resol_join_halves_angstrom,
            current_resolution_angstrom=final_prev_res_angstrom,
            padding_factor=PADDING_FACTOR,
        )

    final_ft_y = final_Ft_y_0 + final_Ft_y_1
    final_ft_ctf = final_Ft_ctf_0 + final_Ft_ctf_1
    final_mean_variance = mean_variance
    final_iter_fsc = None
    final_tau2_update_details = None
    final_mstep_full_half_axis = _resolve_mstep_full_half_axis(
        final_outs.mstep_full_half_axis,
        default_axis=-1,
    )
    if k_class_enabled:
        class_weights = _class_weights_from_posterior(
            final_outs.class_posterior,
            n_classes,
            class_weights,
        )
        class_log_priors = np.log(class_weights)
        history.record_class_weights(
            class_weights.copy(),
            class_weights.copy(),
            _class_weights_from_posterior(
                final_outs.class_full_posterior,
                n_classes,
                class_weights,
            ).copy(),
        )
        _t_final_tau2 = time.time()
        kclass_tau2_frame_scale = float(grid_size) ** 4
        final_mean_variance_per_class = []
        final_mean_variance_shells_per_class = []
        final_data_vs_prior_per_class = []
        final_tau2_update_details_per_class = []
        for class_idx in range(n_classes):
            mean_signal_variance_relion_k, tau2_update_details_k = (
                regularization.compute_relion_tau2_from_iref_power_spectrum(
                    final_join_means[0][class_idx],
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    current_size=final_current_size,
                    return_details=True,
                )
            )
            mean_signal_variance_k = mean_signal_variance_relion_k * jnp.asarray(
                kclass_tau2_frame_scale,
                dtype=mean_signal_variance_relion_k.dtype,
            )
            tau2_shells_relion_frame_k = jnp.asarray(
                tau2_update_details_k["tau2_shells"],
                dtype=mean_signal_variance_k.dtype,
            )
            tau2_shells_recovar_frame_k = tau2_shells_relion_frame_k * jnp.asarray(
                kclass_tau2_frame_scale,
                dtype=mean_signal_variance_k.dtype,
            )
            shell_stats_k = regularization._compute_relion_weight_shell_stats(
                final_ft_ctf[class_idx],
                volume_shape,
                padding_factor=PADDING_FACTOR,
                r_max=final_current_size // 2,
                shell_rounding="round",
                full_half_axis=final_mstep_full_half_axis,
                accumulator_volume_shape=final_mstep_accumulator_shape,
            )
            data_vs_prior_k = regularization.compute_data_vs_prior(
                final_ft_ctf[class_idx],
                tau2_shells_recovar_frame_k,
                volume_shape,
                padding_factor=PADDING_FACTOR,
                tau2_fudge=tau2_fudge,
                current_size=final_current_size,
                full_half_axis=final_mstep_full_half_axis,
                accumulator_volume_shape=final_mstep_accumulator_shape,
            )
            final_mean_variance_per_class.append(mean_signal_variance_k)
            final_mean_variance_shells_per_class.append(tau2_shells_recovar_frame_k)
            final_data_vs_prior_per_class.append(data_vs_prior_k)
            final_tau2_update_details_per_class.append(
                {
                    "prior_shells": np.asarray(tau2_shells_recovar_frame_k, dtype=np.float64),
                    "sigma2_shells": np.asarray(
                        jnp.where(
                            shell_stats_k["avg_weight_shells"] > 0,
                            1.0 / (PADDING_FACTOR**3 * shell_stats_k["avg_weight_shells"]),
                            0.0,
                        ),
                        dtype=np.float64,
                    ),
                    "avg_weight_shells": np.asarray(shell_stats_k["avg_weight_shells"], dtype=np.float64),
                    "shell_sum": np.asarray(shell_stats_k["shell_sum"], dtype=np.float64),
                    "shell_count": np.asarray(shell_stats_k["shell_count"], dtype=np.float64),
                    "fsc_shells": None,
                    "ssnr_shells": np.asarray(data_vs_prior_k, dtype=np.float64),
                }
            )
        final_mean_variance = jnp.stack(final_mean_variance_per_class, axis=0)
        final_mean_variance_shells = jnp.stack(final_mean_variance_shells_per_class, axis=0)
        final_data_vs_prior = np.stack(
            [np.asarray(dvp, dtype=np.float32) for dvp in final_data_vs_prior_per_class],
            axis=0,
        )
        final_tau2_update_details = {
            key: (
                None
                if key == "fsc_shells"
                else np.stack([detail[key] for detail in final_tau2_update_details_per_class], axis=0)
            )
            for key in [
                "prior_shells",
                "sigma2_shells",
                "avg_weight_shells",
                "shell_sum",
                "shell_count",
                "fsc_shells",
                "ssnr_shells",
            ]
        }
        tau2_update_details = final_tau2_update_details
        logger.info(
            "RELION final all-data Class3D tau2 from Iref power spectra: old_max=%.4e new_max=%.4e "
            "dvp_shell_1=%.4f wall=%.1fs",
            float(jnp.max(jnp.abs(mean_variance))),
            float(jnp.max(jnp.abs(final_mean_variance))),
            float(np.asarray(final_data_vs_prior)[0, 1]) if np.asarray(final_data_vs_prior).shape[-1] > 1 else float("nan"),
            time.time() - _t_final_tau2,
        )
    else:
        _t_final_tau2 = time.time()
        final_iter_fsc = regularization.compute_relion_fsc_from_backprojector(
            final_Ft_y_0,
            final_Ft_y_1,
            final_Ft_ctf_0,
            final_Ft_ctf_1,
            volume_shape,
            padding_factor=PADDING_FACTOR,
            r_max=final_current_size // 2,
            accumulator_volume_shape=final_mstep_accumulator_shape,
            output_dtype=_dense_global_scoring_dtype(),
        )
        # RELION's joined-half final reconstruction combines the two half
        # BackProjectors before updateSSNRarrays, then applies the whole-data
        # FSC conversion.
        final_mean_variance, _, final_tau2_update_details = regularization.compute_relion_tau2_from_weights(
            final_Ft_ctf_0,
            final_Ft_ctf_1,
            final_iter_fsc,
            volume_shape,
            tau2_fudge=tau2_fudge,
            padding_factor=PADDING_FACTOR,
            r_max=final_current_size // 2,
            is_whole_instead_of_half=True,
            return_details=True,
            full_half_axis=final_mstep_full_half_axis,
            accumulator_volume_shape=final_mstep_accumulator_shape,
            weight_combination="sum",
            output_dtype=_dense_global_scoring_dtype(),
        )
        logger.info(
            "RELION final all-data tau2 from joined FSC: old_max=%.4e new_max=%.4e "
            "fsc_shell_1=%.4f wall=%.1fs",
            float(jnp.max(jnp.abs(mean_variance))),
            float(jnp.max(jnp.abs(final_mean_variance))),
            float(np.asarray(final_iter_fsc)[1]) if np.asarray(final_iter_fsc).size > 1 else float("nan"),
            time.time() - _t_final_tau2,
        )
        tau2_update_details = final_tau2_update_details

    final_grid_correct = _final_all_data_grid_correct_enabled()
    if final_grid_correct:
        logger.info("RELION final all-data reconstruction gridding correction enabled")
    else:
        logger.info(
            "RELION final all-data reconstruction gridding correction disabled by explicit %s override",
            _FINAL_ALL_DATA_GRID_CORRECT_ENV,
        )

    _final_bpref_accum_dir = os.environ.get("RECOVAR_FINAL_BPREF_ACCUM_DUMP_DIR")
    if _final_bpref_accum_dir:
        reconstruction_diagnostics.write_final_bpref_accumulators(
            PADDING_FACTOR=PADDING_FACTOR,
            PROJECTION_PADDING_FACTOR=PROJECTION_PADDING_FACTOR,
            output_dir=_final_bpref_accum_dir,
            voxel_size=cryo.voxel_size,
            final_Ft_ctf_0=final_Ft_ctf_0,
            final_Ft_ctf_1=final_Ft_ctf_1,
            final_Ft_y_0=final_Ft_y_0,
            final_Ft_y_1=final_Ft_y_1,
            final_current_size=final_current_size,
            final_ft_ctf=final_ft_ctf,
            final_ft_y=final_ft_y,
            final_grid_correct=final_grid_correct,
            final_iter_fsc=final_iter_fsc,
            final_mstep_accumulator_shape=final_mstep_accumulator_shape,
            final_mstep_full_half_axis=final_mstep_full_half_axis,
            final_tau2_update_details=final_tau2_update_details,
            final_unfiltered_Ft_ctf_0=final_unfiltered_Ft_ctf_0,
            final_unfiltered_Ft_ctf_1=final_unfiltered_Ft_ctf_1,
            final_unfiltered_Ft_y_0=final_unfiltered_Ft_y_0,
            final_unfiltered_Ft_y_1=final_unfiltered_Ft_y_1,
            grid_size=grid_size,
            k_class_enabled=k_class_enabled,
            tau2_fudge=tau2_fudge,
            volume_shape=volume_shape,
            logger=logger,
        )

    # Reconstruct the final volume from the COMBINED Ft_y/Ft_ctf accumulators
    # at the full Nyquist resolution. Skip the join_halves step (we're already
    # combining the two halves into one dataset for this final iter).
    final_reconstruct_t0 = time.time()
    logger.info(
        "RELION final all-data reconstruction start: current_size=%d n_classes=%d",
        final_current_size,
        n_classes,
    )
    final_unfiltered_means_for_output = None
    if k_class_enabled:
        final_class_means = jnp.stack(
            [
                _reconstruct_volume_eager(
                    final_ft_ctf[class_idx],
                    final_ft_y[class_idx],
                    volume_shape,
                    PADDING_FACTOR,
                    tau=final_mean_variance_shells[class_idx],
                    tau2_fudge=tau2_fudge,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    grid_correct=final_grid_correct,
                    minres_map=RELION_MINRES_MAP,
                    current_size=final_current_size,
                    accumulator_volume_shape=final_mstep_accumulator_shape,
                    tau_is_1d=True,
                ).reshape(-1)
                for class_idx in range(n_classes)
            ],
            axis=0,
        )
        merged_mean = jnp.sum(
            jnp.asarray(class_weights, dtype=final_class_means.real.dtype)[:, None] * final_class_means, axis=0
        )
        final_means_for_output = [final_class_means, final_class_means]
        class_assignments = final_outs.class_assignments
    else:
        final_class_means = None
        merged_mean = _reconstruct_volume_eager(
            final_ft_ctf,
            final_ft_y,
            volume_shape,
            PADDING_FACTOR,
            tau=final_mean_variance,
            tau2_fudge=tau2_fudge,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            grid_correct=final_grid_correct,
            minres_map=RELION_MINRES_MAP,
            current_size=final_current_size,
            accumulator_volume_shape=final_mstep_accumulator_shape,
        ).reshape(-1)
        final_means_for_output = [
            _reconstruct_volume_eager(
                half_ctf,
                half_y,
                volume_shape,
                PADDING_FACTOR,
                tau=final_mean_variance,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
                grid_correct=final_grid_correct,
                minres_map=RELION_MINRES_MAP,
                current_size=final_current_size,
                accumulator_volume_shape=final_mstep_accumulator_shape,
            ).reshape(-1)
            for half_ctf, half_y in (
                (final_Ft_ctf_0, final_Ft_y_0),
                (final_Ft_ctf_1, final_Ft_y_1),
            )
        ]
        # RELION writes run_half{1,2}_class001_unfil.mrc from each final
        # BackProjector with do_map=false.  Keep this separate from the
        # Wiener-regularized half maps above so parity audits compare like
        # products.  Default RELION refinement sets BackProjector's
        # skip_gridding=true, so reconstruct uses the radial denominator floor
        # and direct division path before its final real-space gridding
        # correction and always applies softMaskOutsideMap inside
        # windowToOridimRealSpace; do_map=false only omits the tau2 prior.
        final_unfiltered_means_for_output = [
            _reconstruct_volume_eager(
                half_ctf,
                half_y,
                volume_shape,
                PADDING_FACTOR,
                tau=None,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
                use_spherical_mask=True,
                grid_correct=True,
                minres_map=RELION_MINRES_MAP,
                current_size=final_current_size,
                accumulator_volume_shape=final_mstep_accumulator_shape,
            ).reshape(-1)
            for half_ctf, half_y in (
                (final_unfiltered_Ft_ctf_0, final_unfiltered_Ft_y_0),
                (final_unfiltered_Ft_ctf_1, final_unfiltered_Ft_y_1),
            )
        ]
    logger.info(
        "RELION final all-data reconstruction done: wall=%.1fs",
        time.time() - final_reconstruct_t0,
    )
    final_iter_elapsed = time.time() - final_iter_t0
    logger.info(
        "Final iter complete: current_size=%d (Nyquist), wall=%.1fs",
        final_current_size,
        final_iter_elapsed,
    )
    history.record_wall_time(final_iter_elapsed)

    (
        replay_requested_iterations,
        replay_applied_iterations,
    ) = _finalize_relion_follower_scale_replay_telemetry(
        replay.relion_follower_scale_replay,
        applied_iterations=history.relion_follower_scale_replay_applied_iterations,
        logger=logger,
    )

    return {
        "mean": merged_mean,
        "means": final_means_for_output,
        "unfiltered_means": final_unfiltered_means_for_output,
        "class_means": final_class_means,
        "class_weights": class_weights if k_class_enabled else None,
        "class_assignments": class_assignments if k_class_enabled else None,
        "relion_follower_scale_replay_requested_iterations": replay_requested_iterations,
        "relion_follower_scale_replay_applied_iterations": replay_applied_iterations,
        **follower_setup.to_result_dict(history),
        "hard_assignments": hard_assignments,
        # RELION-mode specific outputs
        "convergence_state": state,
        "frozen_initial_scoring_state_sha256": frozen_initial_scoring_state_sha256,
        "expected_accuracy_trial_local_indices": expected_accuracy_trial_local_indices,
        "expected_accuracy_trial_particle_ids": expected_accuracy_trial_particle_ids,
        **history.to_dict(),
        "final_all_data_ran": True,
        "final_all_data_expected_accuracy_status": final_expected_accuracy_status,
        "final_all_data_acc_rot": (
            None if final_expected_accuracy is None else float(final_expected_accuracy.acc_rot)
        ),
        "final_all_data_acc_trans": (
            None if final_expected_accuracy is None else float(final_expected_accuracy.acc_trans_angstrom)
        ),
        "final_all_data_acc_rot_per_class": (
            None if final_expected_accuracy is None else final_expected_accuracy.acc_rot_per_class
        ),
        "final_all_data_acc_trans_per_class": (
            None if final_expected_accuracy is None else final_expected_accuracy.acc_trans_per_class_angstrom
        ),
        "final_all_data_expected_accuracy_class_counts": (
            None if final_expected_accuracy is None else final_expected_accuracy.class_counts
        ),
        # -1 preserves the legacy scalar sentinel for a per-half source.
        "final_all_data_noise_source_half": -1,
        "final_all_data_noise_source_halves": (0, 1),
        "final_all_data_fsc": final_iter_fsc,
        "tau2_radial_final_all_data": (
            None
            if final_tau2_update_details is None
            else np.asarray(final_tau2_update_details["prior_shells"], dtype=np.float64)
        ),
        "tau2_fsc_used_final_all_data": (
            None
            if final_tau2_update_details is None or final_tau2_update_details.get("fsc_shells") is None
            else np.asarray(final_tau2_update_details["fsc_shells"], dtype=np.float64)
        ),
        "tau2_ssnr_final_all_data": (
            None
            if final_tau2_update_details is None
            else np.asarray(final_tau2_update_details["ssnr_shells"], dtype=np.float64)
        ),
        "tau2_weight_combination_final_all_data": "class_iref" if k_class_enabled else "sum",
        "final_all_data_best_rotation_eulers": final_outs.best_pose_rotation_eulers,
        "final_all_data_best_translations": final_outs.best_pose_translations,
        "final_all_data_max_posterior": final_outs.max_posterior,
        "final_all_data_class_assignments": final_outs.class_assignments if k_class_enabled else None,
        "final_all_data_sampling_perturbation": final_random_perturbation if final_perturbation_applied else 0.0,
        "final_all_data_sampling_perturbation_applied": final_perturbation_applied,
        "final_all_data_sampling_relion_iteration": final_sampling_relion_iteration,
        "final_all_data_sampling_star": final_sampling_star,
        "final_all_data_sampling_star_source": final_sampling_star_source,
        "final_all_data_sampling_offset_range": final_translation_range,
        "final_all_data_sampling_offset_step": final_translation_step,
        "final_all_data_grid_correct": final_grid_correct,
        "final_all_data_gridding_correct": "radial",
        "setup_phase_seconds": setup_phase_seconds,
    }
