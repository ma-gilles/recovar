"""Core refinement loop for dense single-volume EM.

This file contains the three core algorithm functions:
- ``refine_single_volume`` — public entry point
- ``_run_relion_iteration_loop`` — RELION-parity iteration loop
- ``_run_local_search_iteration`` — exact local angular search

All supporting helpers live in ``helpers/``.
See ``docs/math/relion_refinement_algorithm.md`` for the full algorithm map.
"""

import gc
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.em.dense_single_volume import parity_dump as _parity_dump
from recovar.em.dense_single_volume.batch_planning import (
    _estimate_relion_em_batch_sizes,
    _image_backend,
    _maybe_cache_raw_image_loaders,
)
from recovar.em.dense_single_volume.em_engine import run_em
from recovar.em.dense_single_volume.firstiter_cc import (
    _build_firstiter_cc_pass2_grids,
    _safe_dense_k_class_rotation_block_size,
    _safe_firstiter_cc_image_batch_size,
)
from recovar.em.dense_single_volume.helpers.convergence import (
    LOCAL_SEARCH_HEALPIX_ORDER,
    RefinementState,
    calculate_expected_angular_errors,
    healpix_angular_step,
    update_refinement_state,
)
from recovar.em.dense_single_volume.helpers.fourier_window import quantize_current_size
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    class_weights_from_direction_prior,
    collapse_rotation_posterior_to_direction_prior,
    infer_direction_prior_healpix_order,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    normalize_class_direction_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
    relion_sigma_offset_prior_center,
    relion_translation_prior_center,
    relion_translation_search_base,
    remap_direction_prior_to_healpix_order,
)
from recovar.em.dense_single_volume.helpers.resolution import (
    _bootstrap_current_size_relion,
    bootstrap_current_size_from_ini_high_relion,
    clamp_relion_coarse_image_size,
    compute_coarse_image_size,
    shell_index_to_resolution_angstrom,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats, make_relion_stats
from recovar.em.dense_single_volume.k_class import (
    run_dense_k_class_em,
    run_dense_k_class_em_adaptive,
)

# Re-exports kept for test back-compat: tests monkeypatch these names at the
# ``iteration_loop`` module level (``monkeypatch.setattr(iteration_loop, ...)``)
# even though the call sites now live in the focused submodules. The submodules
# resolve the symbols through ``recovar.em.dense_single_volume.iteration_loop``
# at call time, so keeping the bindings here lets the existing monkeypatches
# continue to win without test churn.
from recovar.em.dense_single_volume.k_class import (  # noqa: F401
    run_local_k_class_em as run_local_k_class_em,
)
from recovar.em.dense_single_volume.local_em_engine import (  # noqa: F401
    run_local_em_exact as run_local_em_exact,
)
from recovar.em.dense_single_volume.local_layout import (
    _selected_rotation_matrices,
)
from recovar.em.dense_single_volume.local_layout import (  # noqa: F401
    build_local_hypothesis_layout as build_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_search_iteration import (
    _precompute_exact_local_fine_grid_enabled,
    _run_local_search_iteration,
)
from recovar.em.dense_single_volume.mean_helpers import (
    _align_fourier_volume_sign_to_reference,
    _class_weights_from_posterior,
    _combined_class_direction_prior_from_halves,
    _combined_noise_stats,
    _mean_noise_variance,
    _merged_mean_from_halves,
    _normalize_class_log_priors,
    _normalize_initial_means,
    _normalize_noise_variance_per_half,
    _reconstruct_and_postprocess_means,
    _reconstruct_volume_eager,
)
from recovar.em.dense_single_volume.relion_metadata import (
    _radial_profile_from_noise_variance,
    _relion_metadata_translations,
    _relion_rotation_grid_float32,
    _rotation_eulers_for_canonical_or_custom_grid,
)
from recovar.em.dense_single_volume.relion_replay import (
    _optional_float32_half_pair,
    _RelionHalfInputState,
    _replay_control_model_iteration,
)
from recovar.em.sampling import (
    advance_relion_perturbation,
    advance_relion_perturbation_from_seed,
    apply_relion_rotation_perturbation,
    apply_relion_rotation_perturbation_to_eulers,
    apply_relion_translation_perturbation,
    build_local_search_grid_metadata,
    get_translation_grid,
    read_relion_direction_prior,
    read_relion_direction_priors,
    read_relion_model_metadata,
    read_relion_optimiser_metadata,
    read_relion_sampling_metadata,
    relion_angular_sampling_deg,
    relion_sampling_perturbation_for_iteration,
    rotation_grid_size,
)
from recovar.em.sampling import (  # noqa: F401
    get_oversampled_rotation_grid_from_samples as get_oversampled_rotation_grid_from_samples,
)
from recovar.em.sampling import (
    get_oversampled_translation_grid as get_oversampled_translation_grid,
)
from recovar.em.sampling import (
    get_relion_rotation_grid as get_relion_rotation_grid,
)
from recovar.em.sampling import (
    get_relion_rotation_grid_eulers as get_relion_rotation_grid_eulers,
)
from recovar.reconstruction.regularization import (
    compute_current_size_relion,
    fsc_to_relion_ssnr,
    resolution_from_data_vs_prior,
    update_relion_growth_state_from_fsc,
)
from recovar.reconstruction.regularization import (  # noqa: F401
    compute_data_vs_prior as compute_data_vs_prior,
)

_EM_RAW_IMAGE_CACHE_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE"
_EM_RAW_IMAGE_CACHE_MAX_GB_ENV = "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB"
_EM_RAW_IMAGE_CACHE_DEFAULT_MAX_GB = 16.0

logger = logging.getLogger(__name__)

RELION_SCORE_TENSOR_FLOAT_BUDGET = 200_000_000
RELION_FIRSTITER_RECON_COMPLEX_BUDGET = 8_000_000
RELION_DENSE_K_CLASS_HYPOTHESES_BUDGET = 2_000_000
RELION_MAX_FULL_GRID_ORDER = 4
EXACT_LOCAL_PRECOMPUTE_FINE_GRID_MAX_ROTATIONS = 3_000_000
_RELION_EM_BATCH_DEFAULT_GPU_GB = 80.0
_RELION_EM_BATCH_USABLE_FRACTION = 0.65
_RELION_EM_BATCH_PROJECTION_FRACTION = 0.20
_RELION_EM_BATCH_SCORE_FRACTION = 0.20
_RELION_EM_BATCH_MAX_PROJECTION_GB = 10.0
_RELION_EM_BATCH_MIN_PROJECTION_GB = 0.5
_RELION_EM_BATCH_PROJECTION_LIVE_FACTOR = 1.5
_RELION_EM_BATCH_TRANSLATION_TILE_FRACTION = 0.35
_RELION_EM_BATCH_RUNTIME_TRANSLATION_TILE_FRACTION = 0.17
_RELION_EM_BATCH_MAX_TRANSLATION_TILE_GB = 14.0
_RELION_EM_BATCH_MIN_TRANSLATION_TILE_GB = 0.5
_RELION_EM_BATCH_RUNTIME_FREE_FRACTION = 0.80


def _exhaustive_grid_order_for_state(state: RefinementState) -> int:
    """Return the global exhaustive HEALPix order for the current state.

    Once RELION enables local angular searches, it no longer scores the full
    HEALPix grid for that order. Keep the global base at the last exhaustive
    order and let the local-search path build image-specific neighborhoods.
    """
    if state.do_local_search:
        return min(
            state.healpix_order,
            max(0, state.auto_local_healpix_order - 1),
            RELION_MAX_FULL_GRID_ORDER,
        )
    return min(state.healpix_order, RELION_MAX_FULL_GRID_ORDER)


from recovar.em.dense_single_volume.debug_dumps import (  # noqa: F401
    _maybe_dump_noise_update_debug,
    _save_iteration_intermediates,
)
from recovar.em.dense_single_volume.ppca_bridge import (  # noqa: F401
    PPCAKClassScheduleBridge,
    run_dense_ppca_refinement_with_kclass_schedule,
    run_local_ppca_refinement_with_kclass_schedule,
)

# RELION stores windowFourierTransform(in, out, current_size) as a rectangular
# FFTW half image, but the likelihood support is the nonzero Minvsigma2 mask:
# rounded radial shells, no DC, no redundant negative-row kx=0 entries.
RELION_FOURIER_WINDOW_SQUARE = False
# RELION's --minres_map default: do not add the Wiener prior term to the
# lowest Fourier shells during MAP reconstruction.
RELION_MINRES_MAP = 5
# RELION uses pf=2 for both projection and reconstruction (--pad 2).
# Projection: real-space zero-pad N³→(2N)³, DFT, then trilinear slice.
# Reconstruction: backproject into (2N)³ Fourier grid, Wiener solve,
# iDFT at (2N)³, crop real-space to N³.
PADDING_FACTOR = 2
PROJECTION_PADDING_FACTOR = 2


# Dense ``run_em`` kwargs that are identical for every E-step in RELION mode.
# Per-iter and per-half values are layered on top at each call site via
# ``{**_DENSE_EM_STATIC_KWARGS, ...}``.
_DENSE_EM_STATIC_KWARGS: dict = {
    "score_with_masked_images": True,
    "half_spectrum_scoring": True,
    "projection_padding_factor": PROJECTION_PADDING_FACTOR,
    "reconstruction_padding_factor": PADDING_FACTOR,
    "use_float64_scoring": False,
    "use_float64_projections": False,
    "do_gridding_correction": True,
    "square_window": RELION_FOURIER_WINDOW_SQUARE,
    "sparse_pass2": False,
}


def refine_single_volume(
    experiment_datasets,
    init_volume,
    init_noise_variance,
    init_mean_variance,
    rotations,
    translations,
    disc_type="linear_interp",
    max_iter=10,
    image_batch_size=500,
    rotation_block_size=5000,
    relion_current_sizes=None,
    init_current_size=32,
    fsc_threshold=1.0 / 7.0,
    adaptive_oversampling=0,
    max_significants=500,
    nside_level=None,
    translation_pixel_offset=None,
    # --- RELION-mode parameters ---
    init_healpix_order=2,
    max_healpix_order=7,
    auto_local_healpix_order=LOCAL_SEARCH_HEALPIX_ORDER,
    init_translation_range=10.0,
    init_translation_step=2.0,
    init_translation_sigma_angstrom=10.0,
    particle_diameter_ang=None,
    save_intermediates_dir=None,
    low_resol_join_halves_angstrom=40.0,
    tau2_fudge=1.0,
    perturb_factor=0.0,
    perturb_seed=None,
    perturb_replay_relion_dir=None,
    init_fsc=None,
    init_ave_Pmax=None,
    init_has_high_fsc_at_limit=None,
    init_relion_iteration=0,
    init_image_corrections=None,
    init_scale_corrections=None,
    init_direction_prior=None,
    init_previous_best_translations=None,
    init_previous_best_rotation_eulers=None,
    replay_iteration_overrides=None,
    skip_final_iteration=False,
    local_search_profile_mode="auto",
    local_search_translation_prior_mode="coarse",
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    emulate_relion_firstiter_cc=False,
    relion_firstiter_ini_high_angstrom=None,
    first_iteration_score_mode="gaussian",
    first_iteration_reconstruction_mode="soft",
    force_max_iter_after_convergence=False,
    n_classes=1,
    init_class_log_priors=None,
    options=None,
):
    """Multi-iteration RELION-parity EM refinement.

    This API always runs the RELION-parity refinement loop.

    ``options`` accepts a :class:`recovar.em.dense_single_volume.refinement_options.RefinementOptions`
    struct that bundles the schedule / adaptive / parity / local-search /
    K-class / replay / debug / batching kwarg groups. When provided, its
    fields override the individual kwargs below. Existing callers that
    pass individual kwargs continue to work unchanged.

    Parameters
    ----------
    experiment_datasets : list of 2 dataset objects
        Half-set datasets (same format as split_E_M_v2 expects).
    init_volume : jnp.ndarray, shape (volume_size,)
        Initial volume in Fourier space.
    init_noise_variance : jnp.ndarray, shape (image_size,)
        Initial per-pixel noise variance.
    init_mean_variance : jnp.ndarray, shape (volume_size,)
        Initial signal prior (tau^2).
    rotations : np.ndarray, shape (n_rot, 3, 3)
        Optional initial rotation grid for compatibility. RELION mode
        regenerates grids from the HEALPix refinement state.
    translations : jnp.ndarray, shape (n_trans, 2)
        Translation grid.
    disc_type : str
        Discretization type for forward/adjoint slicing.
    max_iter : int
        Maximum number of iterations.
    image_batch_size : int
        Number of images per GPU batch.
    rotation_block_size : int
        Number of rotations per block in em_engine.
    relion_current_sizes : list of int or None
        Oracle mode: if provided, use these current_sizes instead of
        computing RELION-style current sizes from the FSC/data-vs-prior
        trajectory. relion_current_sizes[i] is used at iteration i.
    init_current_size : int
        Starting current_size for the first iteration (when no FSC is
        available yet).  Ignored if relion_current_sizes is provided.
    fsc_threshold : float
        FSC threshold for resolution estimation.
    adaptive_oversampling : int
        Number of HEALPix subdivision levels for pass 2 (0=disabled,
        1=2x finer = 4 children, 2=4x finer = 16 children).
    max_significants : int
        Maximum significant (rotation x translation) samples per image.
        Matches RELION's --maxsig semantics (counts SAMPLES, not just
        orientations; see C5 in plan_relion_parity.md).
    nside_level : int or None
        Compatibility keyword for older callers. RELION mode derives the
        coarse rotation grid from ``init_healpix_order``.
    translation_pixel_offset : float or None
        Step size between coarse translation grid points (pixels).
        Required when adaptive_oversampling > 0.
    init_healpix_order : int
        Starting HEALPix order for RELION mode (default 2, ~14.7 deg).
    max_healpix_order : int
        Maximum HEALPix order (finest angular sampling, default 7).
    auto_local_healpix_order : int
        RELION ``--auto_local_healpix_order`` threshold for switching from
        global to local angular searches.
    init_translation_range : float
        Initial translation search range in pixels (RELION mode).
    init_translation_step : float
        Initial translation step size in pixels (RELION mode).
    init_translation_sigma_angstrom : float
        Initial RELION-style translation prior width in Angstrom.
    particle_diameter_ang : float or None
        RELION particle diameter in Angstrom for the adaptive coarse-image-size
        formula. When None, fall back to ``ori_size * pixel_size``.

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
    if options is not None:
        # Pull from RefinementOptions struct. Per-field unpacking lets old kwargs
        # remain authoritative when no struct is passed.
        schedule = options.schedule
        max_iter = schedule.max_iter
        init_current_size = schedule.init_current_size
        fsc_threshold = schedule.fsc_threshold
        init_healpix_order = schedule.init_healpix_order
        max_healpix_order = schedule.max_healpix_order
        init_translation_range = schedule.init_translation_range
        init_translation_step = schedule.init_translation_step
        init_translation_sigma_angstrom = schedule.init_translation_sigma_angstrom
        particle_diameter_ang = schedule.particle_diameter_ang
        init_relion_iteration = schedule.init_relion_iteration
        init_fsc = schedule.init_fsc
        init_ave_Pmax = schedule.init_ave_Pmax
        init_has_high_fsc_at_limit = schedule.init_has_high_fsc_at_limit
        force_max_iter_after_convergence = schedule.force_max_iter_after_convergence
        skip_final_iteration = schedule.skip_final_iteration

        adaptive = options.adaptive
        adaptive_oversampling = adaptive.adaptive_oversampling
        max_significants = adaptive.max_significants
        nside_level = adaptive.nside_level
        translation_pixel_offset = adaptive.translation_pixel_offset
        relion_current_sizes = adaptive.relion_current_sizes

        parity = options.parity
        low_resol_join_halves_angstrom = parity.low_resol_join_halves_angstrom
        tau2_fudge = parity.tau2_fudge
        perturb_factor = parity.perturb_factor
        perturb_seed = parity.perturb_seed
        perturb_replay_relion_dir = parity.perturb_replay_relion_dir
        emulate_relion_firstiter_cc = parity.emulate_relion_firstiter_cc
        relion_firstiter_ini_high_angstrom = parity.relion_firstiter_ini_high_angstrom
        first_iteration_score_mode = parity.first_iteration_score_mode
        first_iteration_reconstruction_mode = parity.first_iteration_reconstruction_mode

        local_search = options.local_search
        auto_local_healpix_order = local_search.auto_local_healpix_order
        local_search_profile_mode = local_search.local_search_profile_mode
        local_search_translation_prior_mode = local_search.local_search_translation_prior_mode

        debug_opts = options.debug
        disable_adjoint_y = debug_opts.disable_adjoint_y
        disable_adjoint_ctf = debug_opts.disable_adjoint_ctf
        save_intermediates_dir = debug_opts.save_intermediates_dir

        k_class_opts = options.k_class
        n_classes = k_class_opts.n_classes
        init_class_log_priors = k_class_opts.init_class_log_priors

        replay = options.replay
        init_image_corrections = replay.init_image_corrections
        init_scale_corrections = replay.init_scale_corrections
        init_direction_prior = replay.init_direction_prior
        init_previous_best_translations = replay.init_previous_best_translations
        init_previous_best_rotation_eulers = replay.init_previous_best_rotation_eulers
        replay_iteration_overrides = replay.replay_iteration_overrides

        batching = options.batching
        image_batch_size = batching.image_batch_size
        rotation_block_size = batching.rotation_block_size

        disc_type = options.disc_type

    if relion_current_sizes is not None and len(relion_current_sizes) == 0:
        raise ValueError("relion_current_sizes must be non-empty when provided")

    return _run_relion_iteration_loop(
        experiment_datasets=experiment_datasets,
        init_volume=init_volume,
        init_noise_variance=init_noise_variance,
        init_mean_variance=init_mean_variance,
        rotations=rotations,
        translations=translations,
        disc_type=disc_type,
        max_iter=max_iter,
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        init_current_size=init_current_size,
        fsc_threshold=fsc_threshold,
        adaptive_oversampling=adaptive_oversampling,
        max_significants=max_significants,
        relion_current_sizes=relion_current_sizes,
        init_healpix_order=init_healpix_order,
        max_healpix_order=max_healpix_order,
        auto_local_healpix_order=auto_local_healpix_order,
        init_translation_range=init_translation_range,
        init_translation_step=init_translation_step,
        init_translation_sigma_angstrom=init_translation_sigma_angstrom,
        particle_diameter_ang=particle_diameter_ang,
        nside_level=nside_level,
        save_intermediates_dir=save_intermediates_dir,
        low_resol_join_halves_angstrom=low_resol_join_halves_angstrom,
        tau2_fudge=tau2_fudge,
        perturb_factor=perturb_factor,
        perturb_seed=perturb_seed,
        perturb_replay_relion_dir=perturb_replay_relion_dir,
        init_fsc=init_fsc,
        init_ave_Pmax=init_ave_Pmax,
        init_has_high_fsc_at_limit=init_has_high_fsc_at_limit,
        init_relion_iteration=init_relion_iteration,
        init_image_corrections=init_image_corrections,
        init_scale_corrections=init_scale_corrections,
        init_direction_prior=init_direction_prior,
        init_previous_best_translations=init_previous_best_translations,
        init_previous_best_rotation_eulers=init_previous_best_rotation_eulers,
        replay_iteration_overrides=replay_iteration_overrides,
        skip_final_iteration=skip_final_iteration,
        local_search_profile_mode=local_search_profile_mode,
        local_search_translation_prior_mode=local_search_translation_prior_mode,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        emulate_relion_firstiter_cc=emulate_relion_firstiter_cc,
        relion_firstiter_ini_high_angstrom=relion_firstiter_ini_high_angstrom,
        first_iteration_score_mode=first_iteration_score_mode,
        first_iteration_reconstruction_mode=first_iteration_reconstruction_mode,
        force_max_iter_after_convergence=force_max_iter_after_convergence,
        n_classes=n_classes,
        init_class_log_priors=init_class_log_priors,
    )


# ---------------------------------------------------------------------------
# RELION-parity refinement mode
# ---------------------------------------------------------------------------


def _run_relion_iteration_loop(
    experiment_datasets,
    init_volume,
    init_noise_variance,
    init_mean_variance,
    rotations,
    translations,
    disc_type,
    max_iter,
    image_batch_size,
    rotation_block_size,
    init_current_size,
    fsc_threshold,
    adaptive_oversampling,
    max_significants,
    relion_current_sizes,
    init_healpix_order,
    max_healpix_order,
    auto_local_healpix_order,
    init_translation_range,
    init_translation_step,
    init_translation_sigma_angstrom,
    particle_diameter_ang,
    nside_level,
    save_intermediates_dir=None,
    low_resol_join_halves_angstrom=40.0,
    tau2_fudge=1.0,
    perturb_factor=0.0,
    perturb_seed=None,
    perturb_replay_relion_dir=None,
    init_fsc=None,
    init_ave_Pmax=None,
    init_has_high_fsc_at_limit=None,
    init_relion_iteration=0,
    init_image_corrections=None,
    init_scale_corrections=None,
    init_direction_prior=None,
    init_previous_best_translations=None,
    init_previous_best_rotation_eulers=None,
    replay_iteration_overrides=None,
    skip_final_iteration=False,
    local_search_profile_mode="auto",
    local_search_translation_prior_mode="coarse",
    disable_adjoint_y=False,
    disable_adjoint_ctf=False,
    emulate_relion_firstiter_cc=False,
    relion_firstiter_ini_high_angstrom=None,
    first_iteration_score_mode="gaussian",
    first_iteration_reconstruction_mode="soft",
    force_max_iter_after_convergence=False,
    n_classes=1,
    init_class_log_priors=None,
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
    from recovar.reconstruction import noise, regularization

    setup_t0 = time.time()
    setup_phase_seconds = {}

    def _mark_setup_phase(name: str) -> None:
        setup_phase_seconds[name] = time.time() - setup_t0

    cryo = experiment_datasets[0]
    volume_shape = cryo.volume_shape
    grid_size = cryo.image_shape[0]  # ori_size in RELION terms
    n_classes = int(n_classes)
    k_class_enabled = n_classes > 1
    class_log_priors = _normalize_class_log_priors(n_classes, init_class_log_priors)
    class_weights = np.exp(class_log_priors)
    if k_class_enabled and init_class_log_priors is None and init_direction_prior is not None:
        inferred_class_weights = class_weights_from_direction_prior(init_direction_prior, n_classes)
        if inferred_class_weights is not None:
            if np.any(inferred_class_weights <= 0.0):
                raise ValueError("RELION direction-prior row sums imply a zero-probability class")
            class_weights = inferred_class_weights
            class_log_priors = np.log(class_weights)

    # --- RELION image mask (softMaskOutsideMap on particles) ---
    # RELION masks images to particle_diameter/(2*pixel_size) with a 5-pixel
    # cosine taper before E-step scoring (ml_optimiser.cpp:6288).  The default
    # edge-taper mask (window_mask(D, 0.85, 0.99)) is too tight — it tapers
    # at 54 px vs RELION's 64 px for a 128-px box.
    RELION_WIDTH_MASK_EDGE = 5

    for ds in experiment_datasets:
        backend = _image_backend(ds)
        if backend is not None and hasattr(backend, "image_mask_mode"):
            backend.image_mask_mode = "multiply"
    if particle_diameter_ang is not None and particle_diameter_ang > 0:
        from recovar.core.mask import relion_soft_image_mask

        relion_mask = relion_soft_image_mask(
            image_size=grid_size,
            pixel_size=cryo.voxel_size,
            particle_diameter_ang=particle_diameter_ang,
            width_mask_edge_px=RELION_WIDTH_MASK_EDGE,
        )
        for ds in experiment_datasets:
            backend = _image_backend(ds)
            if backend is None:
                continue
            backend.image_mask = relion_mask
            if hasattr(backend, "image_mask_mode"):
                backend.image_mask_mode = "relion_background_fill"
        logger.info(
            "RELION mode: image mask radius=%.1f px (particle_diameter=%.1f A, edge=%d px)",
            particle_diameter_ang / (2.0 * cryo.voxel_size),
            particle_diameter_ang,
            RELION_WIDTH_MASK_EDGE,
        )

    _maybe_cache_raw_image_loaders(experiment_datasets)
    _mark_setup_phase("mask_and_image_cache")

    # --- Initialize RefinementState ---
    # Corresponds to RELION's initialiseSamplingVectors + initialLowPassFilterReferences
    state = RefinementState(
        iteration=0,
        healpix_order=init_healpix_order,
        adaptive_oversampling=adaptive_oversampling,
        translation_range=init_translation_range,
        translation_step=init_translation_step,
        max_healpix_order=max_healpix_order,
        auto_local_healpix_order=auto_local_healpix_order,
        current_resolution=float("inf"),
        voxel_size_angstrom=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
        particle_diameter_angstrom=float(particle_diameter_ang or 0.0),
    )
    # RELION's convergence counters are not initialized against an infinite
    # previous resolution.  They resume from the previous optimiser/model STAR
    # in replay mode, or from the initial FSC/ini_high state in a fresh run.
    if perturb_replay_relion_dir is not None and int(init_relion_iteration) > 0:
        _init_opt_star = os.path.join(
            perturb_replay_relion_dir,
            f"run_it{int(init_relion_iteration):03d}_optimiser.star",
        )
        _init_model_star = os.path.join(
            perturb_replay_relion_dir,
            f"run_it{int(init_relion_iteration):03d}_half1_model.star",
        )
        if os.path.exists(_init_model_star):
            _init_model_meta = read_relion_model_metadata(_init_model_star)
            _init_res_angstrom = float(_init_model_meta["current_resolution"])
            if np.isfinite(_init_res_angstrom) and _init_res_angstrom > 0.0:
                state.current_resolution = _init_res_angstrom
                state.previous_resolution = _init_res_angstrom
        if os.path.exists(_init_opt_star):
            _init_opt_meta = read_relion_optimiser_metadata(_init_opt_star)
            state.nr_iter_wo_resol_gain = int(_init_opt_meta.get("number_iter_without_resolution_gain") or 0)
            _hvc = int(_init_opt_meta.get("number_iter_without_changing_assignments") or 0)
            state.nr_iter_wo_large_hidden_variable_changes = _hvc
            state.nr_iter_wo_assignment_changes = _hvc
            if _init_opt_meta.get("overall_accuracy_rotations") is not None:
                state.acc_rot = float(_init_opt_meta["overall_accuracy_rotations"])
            if _init_opt_meta.get("overall_accuracy_translations_angst") is not None:
                state.acc_trans = float(_init_opt_meta["overall_accuracy_translations_angst"])
            if _init_opt_meta.get("smallest_changes_orientations") is not None:
                state.smallest_changes_optimal_orientations = float(_init_opt_meta["smallest_changes_orientations"])
            if _init_opt_meta.get("smallest_changes_offsets") is not None:
                state.smallest_changes_optimal_offsets_angstrom = float(_init_opt_meta["smallest_changes_offsets"])
            if _init_opt_meta.get("smallest_changes_classes") is not None:
                state.smallest_changes_optimal_classes = float(_init_opt_meta["smallest_changes_classes"])
            if _init_opt_meta.get("has_converged") is not None:
                state.has_converged = bool(int(_init_opt_meta["has_converged"]))
        logger.info(
            "Replay convergence init from RELION iter %03d: res=%.2f A, "
            "stalls=(res=%d,hvc=%d), smallest=(rot=%.3f deg, trans=%.3f A, class=%.3f)",
            int(init_relion_iteration),
            state.current_resolution,
            state.nr_iter_wo_resol_gain,
            state.nr_iter_wo_large_hidden_variable_changes,
            state.smallest_changes_optimal_orientations,
            state.smallest_changes_optimal_offsets_angstrom,
            state.smallest_changes_optimal_classes,
        )
    elif init_fsc is not None:
        _init_fsc_for_state = np.asarray(init_fsc, dtype=np.float32).copy()
        _prev_cs_for_state = int(init_current_size)
        if _prev_cs_for_state < grid_size:
            _init_fsc_for_state[min(len(_init_fsc_for_state), _prev_cs_for_state // 2) :] = 0.0
        _init_dvp = np.asarray(fsc_to_relion_ssnr(_init_fsc_for_state, tau2_fudge=tau2_fudge))
        _init_res_shell = resolution_from_data_vs_prior(_init_dvp, allow_high_res_recovery=True)
        _init_res_angstrom = shell_index_to_resolution_angstrom(
            _init_res_shell,
            grid_size,
            cryo.voxel_size,
        )
        if np.isfinite(_init_res_angstrom) and _init_res_angstrom > 0.0:
            state.current_resolution = float(_init_res_angstrom)
            state.previous_resolution = float(_init_res_angstrom)
    elif init_relion_iteration == 0 and relion_firstiter_ini_high_angstrom is not None:
        _px = float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0)
        _init_shell = int(np.floor(grid_size * _px / float(relion_firstiter_ini_high_angstrom) + 0.5))
        _init_shell = max(1, min(grid_size // 2, _init_shell))
        _init_res_angstrom = shell_index_to_resolution_angstrom(_init_shell, grid_size, _px)
        state.current_resolution = float(_init_res_angstrom)
        state.previous_resolution = float(_init_res_angstrom)
    _mark_setup_phase("state_init")

    # RELION mode owns the coarse HEALPix grid. When coarse-grid metadata is
    # provided, regenerate the matching coarse grid here instead of inheriting
    # any finer caller-supplied rotation table.
    current_healpix_order = int(init_healpix_order)
    if nside_level is not None:
        if int(nside_level) != current_healpix_order:
            logger.info(
                "RELION mode: ignoring caller nside_level=%d and regenerating initial coarse grid at healpix_order=%d",
                int(nside_level),
                current_healpix_order,
            )
        current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(current_healpix_order)
    elif rotations is not None:
        current_rotations = np.asarray(rotations, dtype=np.float32)
        current_rotation_eulers = _rotation_eulers_for_canonical_or_custom_grid(
            current_rotations,
            current_healpix_order,
        )
    else:
        current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(current_healpix_order)
    if translations is None:
        current_translations = jnp.asarray(
            get_translation_grid(init_translation_range, init_translation_step), dtype=jnp.float32
        )
    else:
        current_translations = jnp.asarray(translations, dtype=jnp.float32)
    # Unperturbed base grid — `current_translations` may be replaced per-iter by
    # a perturbed copy (SamplingPerturbation). Keep the base so each iter
    # perturbs a fresh copy rather than compounding prior perturbations.
    base_translations = current_translations
    if save_intermediates_dir is not None:
        os.makedirs(save_intermediates_dir, exist_ok=True)
    if local_search_profile_mode not in {"auto", "on", "off"}:
        raise ValueError(
            f"local_search_profile_mode must be one of {{'auto', 'on', 'off'}}, got {local_search_profile_mode!r}",
        )
    collect_local_search_profile = (
        save_intermediates_dir is not None if local_search_profile_mode == "auto" else local_search_profile_mode == "on"
    )
    _mark_setup_phase("sampling_grid")

    padded_volume_shape = tuple(d * PADDING_FACTOR for d in volume_shape)

    def _safe_batch_sizes(n_rot, n_trans, *, classes=None, image_shape_for_batch=None):
        """Reduce batch sizes for large pose grids to avoid GPU OOM."""
        plan = _estimate_relion_em_batch_sizes(
            requested_image_batch_size=image_batch_size,
            requested_rotation_block_size=rotation_block_size,
            n_rot=n_rot,
            n_trans=n_trans,
            image_shape=image_shape_for_batch or cryo.image_shape,
            volume_shape=volume_shape,
            padding_factor=PADDING_FACTOR,
            n_classes=n_classes if classes is None else classes,
        )
        if plan.image_batch_size != image_batch_size or plan.rotation_block_size != rotation_block_size:
            logger.info(
                "RELION EM batch sizing: requested image_batch_size=%d rotation_block_size=%d; "
                "using image_batch_size=%d rotation_block_size=%d "
                "(n_rot=%d n_trans=%d K=%d, score_budget=%.1fM floats, "
                "projection_tile=%.2f/%.2f GB, translation_tile=%.2f/%.2f GB, "
                "persistent_est=%.2f GB, usable_est=%.2f GB, gpu_used_est=%.2f GB)",
                image_batch_size,
                rotation_block_size,
                plan.image_batch_size,
                plan.rotation_block_size,
                int(n_rot),
                int(n_trans),
                int(n_classes if classes is None else classes),
                plan.score_float_budget / 1e6,
                plan.projection_block_gb,
                plan.projection_budget_gb,
                plan.translation_tile_gb,
                plan.translation_tile_budget_gb,
                plan.persistent_estimate_gb,
                plan.usable_estimate_gb,
                plan.gpu_used_estimate_gb,
            )
        return plan.image_batch_size, plan.rotation_block_size

    # State: two half-set references.  For K-class refinement each half stores
    # an explicit leading class axis; single-class callers keep the historical
    # flat per-half reference layout.
    means = _normalize_initial_means(init_volume, n_classes)
    noise_variance_per_half = _normalize_noise_variance_per_half(
        init_noise_variance,
        n_halves=2,
    )
    noise_variance = _mean_noise_variance(noise_variance_per_half)
    mean_variance = jnp.array(init_mean_variance)
    _mark_setup_phase("initial_arrays")

    # History tracking. Keep these plain lists because intermediate outputs
    # serialize them directly.
    current_sizes = []
    fsc_history = []
    pixel_resolutions = []
    wall_times = []
    hard_assignments = [None, None]
    previous_assignments = [None, None]
    class_assignments = [None, None]
    previous_class_assignments = [None, None]
    previous_best_rotations = [None, None]
    relion_half_inputs = _RelionHalfInputState.from_initial_values(
        previous_best_translations=init_previous_best_translations,
        previous_best_rotation_eulers=init_previous_best_rotation_eulers,
        image_corrections=init_image_corrections,
        scale_corrections=init_scale_corrections,
    )
    max_posterior_per_half = [None, None]
    rotation_posterior_per_half = [None, None]
    class_rotation_posterior_per_half = [None, None]
    significant_counts = []
    data_vs_prior_trajectory = []
    previous_data_vs_prior_for_scheduling = None
    healpix_order_trajectory = []
    ave_Pmax_trajectory = []
    pmax_per_image_history = []
    # Per-iter per-shell trajectories for RELION parity diff (added for the
    # 2026-04 audit). noise_radial_trajectory[i] = sigma2_noise per shell after
    # iter i's noise update; tau2_radial_trajectory[i] = recovar's tau2 prior
    # per shell after iter i's signal-prior update.
    noise_radial_trajectory = []
    noise_radial_per_half_trajectory = []
    tau2_radial_trajectory = []
    tau2_sigma2_trajectory = []
    tau2_avg_weight_trajectory = []
    tau2_shell_sum_trajectory = []
    tau2_shell_count_trajectory = []
    tau2_fsc_used_trajectory = []
    tau2_ssnr_trajectory = []
    tau2_update_details = None
    tau2_update_details_per_half = None

    # C1 (RELION-parity): per-iter sigma2_offset update from data. Initialized
    # from `init_translation_sigma_angstrom`; updated from RELION's
    # posterior-weighted offset moment when the E-step path propagates it.
    # RELION stores and updates this quantity in Angstrom², and its default
    # lower bound is min_sigma2_offset=2 Å² (ml_optimiser.cpp).
    current_sigma_offset_angstrom = float(init_translation_sigma_angstrom)
    sigma_offset_used_trajectory = []
    sigma_offset_trajectory = []
    # D.2: per-class sigma_offset trajectory. K=1 leaves entries as
    # [scalar, scalar, ...]; K>1 stores K-vectors. Diagnostic for now —
    # threading per-class sigma into the engine's translation log_prior
    # is the next step (requires k_class.py engine signature change).
    per_class_sigma_offset_trajectory = []
    frac_changed_trajectory = []
    acc_rot_trajectory = []
    smallest_change_angles_trajectory = []
    smallest_change_offsets_trajectory = []
    best_rotation_eulers_history = []
    best_translations_history = []
    class_weight_trajectory = []
    local_profile_history = []
    relion_incr_size = 10  # RELION default
    relion_has_high_fsc_at_limit = bool(init_has_high_fsc_at_limit) if init_has_high_fsc_at_limit is not None else False
    global_direction_prior_per_half = [None, None]
    global_direction_prior_order_per_half = [None, None]
    class_direction_prior_per_half = [None, None]
    class_direction_prior_order_per_half = [None, None]

    # --- Direction prior from snapshot ---
    # When starting from a RELION snapshot, the previous iteration's
    # pdf_orientation is a non-uniform prior over HEALPix directions.
    # RELION applies this in the next E-step.  recovar must do the same.
    if init_direction_prior is not None and k_class_enabled:
        class_direction_prior_per_half = normalize_class_direction_prior_per_half(init_direction_prior, n_classes)
        for k in range(2):
            if class_direction_prior_per_half[k] is None:
                continue
            prior_k = np.asarray(class_direction_prior_per_half[k], dtype=np.float32)
            class_direction_prior_per_half[k] = prior_k
            class_direction_prior_order_per_half[k] = infer_direction_prior_healpix_order(prior_k[0])
            logger.info(
                "RELION mode: loaded init class direction priors half-%d: %d classes, %d directions",
                k + 1,
                prior_k.shape[0],
                prior_k.shape[1],
            )
    elif init_direction_prior is not None:
        global_direction_prior_per_half = normalize_direction_prior_per_half(init_direction_prior)
        for k in range(2):
            if global_direction_prior_per_half[k] is None:
                continue
            prior_k = np.asarray(global_direction_prior_per_half[k], dtype=np.float32)
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
        dtype=jnp.float32,
    )
    _mark_setup_phase("noise_radial_init")

    # --- RELION SamplingPerturbation state (healpix_sampling.cpp:167-174) ---
    # RELION applies a random rigid rotation of the entire SO(3) trial grid at
    # each iteration: A -> A @ R_perturb with R_perturb = R_from_relion([m,m,m])
    # and m = random_perturbation * angular_sampling. The random_perturbation
    # is advanced per iter via realWRAP(prev + rnd_unif(0.5*pf, pf), -pf, +pf).
    # For exact parity replay, read _rlnSamplingPerturbInstance from RELION's
    # per-iter sampling.star.
    if perturb_factor > 0 and perturb_seed is not None:
        random_perturbation = relion_sampling_perturbation_for_iteration(
            perturb_factor,
            perturb_seed,
            init_relion_iteration,
        )
        logger.info(
            "Perturbation init: relion_iter=%d random_seed=%d rp=%+.5f",
            int(init_relion_iteration),
            int(perturb_seed),
            random_perturbation,
        )
    else:
        random_perturbation = 0.0
    perturb_rng = None if perturb_seed is not None else np.random.default_rng()
    iteration = 0
    _mark_setup_phase("before_iterations")
    logger.info(
        "RELION mode setup timing before iteration loop: %s",
        ", ".join(f"{key}={value:.1f}s" for key, value in setup_phase_seconds.items()),
    )
    while (force_max_iter_after_convergence or not state.has_converged) and iteration < max_iter:
        t0 = time.time()
        _parity_dump.start_iteration(iteration)
        iter_replay_override = None
        if replay_iteration_overrides is not None and iteration < len(replay_iteration_overrides):
            iter_replay_override = replay_iteration_overrides[iteration]
        relion_firstiter_cc_this_iter = bool(
            emulate_relion_firstiter_cc and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_normalized_cc_this_iter = bool(
            first_iteration_score_mode == "normalized_cc" and init_relion_iteration == 0 and iteration == 0
        )
        first_iter_hard_reconstruction_this_iter = bool(
            first_iteration_reconstruction_mode == "hard" and init_relion_iteration == 0 and iteration == 0
        )
        firstiter_score_mode_this_iter = (
            "normalized_cc" if (relion_firstiter_cc_this_iter or first_iter_normalized_cc_this_iter) else "gaussian"
        )
        firstiter_winner_take_all_this_iter = bool(
            relion_firstiter_cc_this_iter or first_iter_hard_reconstruction_this_iter
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
                    relion_firstiter_ini_high_angstrom,
                    incr_size=relion_incr_size,
                )
            else:
                seeded_cs = None
            if seeded_cs is not None:
                cs = int(seeded_cs)
                data_vs_prior_iter = None
                logger.info(
                    "RELION init bootstrap: seeding iter-1 current_size from ini_high=%.2f A -> %d",
                    float(relion_firstiter_ini_high_angstrom),
                    cs,
                )
            elif init_fsc is not None:
                fsc_prev = np.asarray(init_fsc, dtype=np.float32).copy()
                prev_cs = int(init_current_size)
                if prev_cs < grid_size:
                    fsc_prev[min(len(fsc_prev), prev_cs // 2) :] = 0.0
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
                _init_pmax = float(init_ave_Pmax) if init_ave_Pmax is not None else 0.0
                raw_cs = compute_current_size_relion(
                    res_shell,
                    grid_size,
                    ave_Pmax=_init_pmax,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    incr_size=relion_incr_size,
                )
                cs = quantize_current_size(raw_cs, ori_size=grid_size)
            else:
                cs = _bootstrap_current_size_relion(init_current_size, grid_size)
                data_vs_prior_iter = None
        else:
            prev_cs = current_sizes[-1]
            if k_class_enabled:
                if previous_data_vs_prior_for_scheduling is None:
                    raise RuntimeError("K-class current-size scheduling requires a previous data_vs_prior curve")
                data_vs_prior_prev_raw = np.asarray(previous_data_vs_prior_for_scheduling, dtype=np.float32).copy()
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
                    import pathlib

                    pathlib.Path(_kclass_dump_dir).mkdir(parents=True, exist_ok=True)
                    np.savez(
                        pathlib.Path(_kclass_dump_dir) / f"recovar_kclass_current_size_it{iteration + 1:03d}.npz",
                        iteration=np.int32(iteration + 1),
                        previous_current_size=np.int32(prev_cs),
                        grid_size=np.int32(grid_size),
                        resolution_shell=np.int32(res_shell),
                        per_class_resolution_shells=np.asarray(per_class_res_shell, dtype=np.int32),
                        ave_Pmax=np.float64(float(state.ave_Pmax)),
                        state_current_resolution=np.float64(float(state.current_resolution)),
                        state_previous_resolution=np.float64(float(state.previous_resolution)),
                        relion_incr_size=np.int32(relion_incr_size),
                        relion_has_high_fsc_at_limit=np.int32(int(relion_has_high_fsc_at_limit)),
                        data_vs_prior_prev_raw=np.asarray(data_vs_prior_prev_raw, dtype=np.float32),
                        data_vs_prior_prev=np.asarray(data_vs_prior_prev, dtype=np.float32),
                        raw_current_size=np.int32(raw_cs),
                        quantized_current_size=np.int32(computed_cs),
                    )
                cs = computed_cs
            else:
                fsc_prev = np.asarray(fsc_history[-1], dtype=np.float32).copy()
                if prev_cs < grid_size:
                    fsc_prev[min(len(fsc_prev), prev_cs // 2) :] = 0.0

                # data_vs_prior = tau2_fudge * fsc / (1 - fsc), matching
                # RELION's updateSSNRarrays at backprojector.cpp:1117-1123
                # for the gold-standard split-half auto-refine path.
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

                raw_cs = compute_current_size_relion(
                    res_shell,
                    grid_size,
                    ave_Pmax=state.ave_Pmax,
                    has_high_fsc_at_limit=relion_has_high_fsc_at_limit,
                    incr_size=relion_incr_size,
                )
                cs = quantize_current_size(raw_cs, ori_size=grid_size)

        cs = quantize_current_size(cs, ori_size=grid_size)
        if relion_current_sizes is not None:
            if iteration < len(relion_current_sizes):
                oracle_cs = int(relion_current_sizes[iteration])
            else:
                oracle_cs = int(relion_current_sizes[-1])
            if oracle_cs <= 0:
                oracle_cs = int(init_current_size)
            cs = quantize_current_size(oracle_cs, ori_size=grid_size)
            logger.info(
                "Current-size oracle: iteration %d using current_size=%d",
                iteration + 1,
                cs,
            )

        # --- Replay override: force recovar's sampling state to mirror RELION ---
        # When replaying, RELION's per-iter sampling.star dictates the actual
        # hp_order, offset_range, and offset_step used at this iteration.
        # Overriding `state.healpix_order`, `state.translation_range` and
        # `state.translation_step` here makes the downstream grid regen code
        # produce the same grid RELION did, so the perturbation applied later
        # is on the correct base grid.
        _replay_meta = None
        _replay_prior_translations = None
        _model_star = None
        _model_meta = None
        if perturb_replay_relion_dir is not None:
            _star = os.path.join(
                perturb_replay_relion_dir,
                f"run_it{init_relion_iteration + iteration + 1:03d}_sampling.star",
            )
            _replay_meta = read_relion_sampling_metadata(_star)
            _relion_hp = int(_replay_meta["healpix_order"])
            _relion_psi_step_deg = float(_replay_meta.get("psi_step", healpix_angular_step(_relion_hp)))
            # RELION stores offset_{range,step} in Angstroms; convert to px.
            _px = float(cryo.voxel_size) if cryo.voxel_size > 0 else 1.0
            _relion_offset_range = float(_replay_meta["offset_range"]) / _px
            _relion_offset_step = float(_replay_meta["offset_step"]) / _px
            _replay_prior_translations = jnp.array(
                get_translation_grid(
                    _relion_offset_range,
                    _relion_offset_step,
                ).astype(np.float32)
            )
            _capped_hp = min(_relion_hp, state.max_healpix_order)
            if state.healpix_order != _capped_hp:
                if _capped_hp < _relion_hp:
                    logger.info(
                        "Replay override: healpix_order %d -> %d (RELION %d capped by max_healpix_order=%d, from %s)",
                        state.healpix_order,
                        _capped_hp,
                        _relion_hp,
                        state.max_healpix_order,
                        _star,
                    )
                else:
                    logger.info(
                        "Replay override: healpix_order %d -> %d (from %s)",
                        state.healpix_order,
                        _capped_hp,
                        _star,
                    )
                state.healpix_order = _capped_hp
            _replay_do_local = bool(state.healpix_order >= state.auto_local_healpix_order)
            if state.do_local_search != _replay_do_local:
                logger.info(
                    "Replay override: local_search %s -> %s (healpix_order=%d, auto_local_healpix_order=%d)",
                    state.do_local_search,
                    _replay_do_local,
                    state.healpix_order,
                    state.auto_local_healpix_order,
                )
                state.do_local_search = _replay_do_local
                if _replay_do_local:
                    state.sigma_rot = 0.0
                    state.sigma_psi = 0.0
            # The model star records the control state for the replayed E-step.
            # Reuse it for both current_size and local-prior sigmas.
            _cs_iter = _replay_control_model_iteration(init_relion_iteration, iteration)
            _model_star_candidates = [
                os.path.join(perturb_replay_relion_dir, f"run_it{_cs_iter:03d}_half1_model.star"),
                os.path.join(perturb_replay_relion_dir, f"run_it{_cs_iter:03d}_model.star"),
            ]
            _model_star = next((path for path in _model_star_candidates if os.path.exists(path)), None)
            if _model_star is not None:
                _model_meta = read_relion_model_metadata(_model_star)
            if _replay_do_local:
                _relion_sigma_rot_deg = None
                _relion_sigma_psi_deg = None
                if _model_meta is not None:
                    _sigma_rot_deg = _model_meta.get("sigma_prior_rot_angle")
                    _sigma_tilt_deg = _model_meta.get("sigma_prior_tilt_angle")
                    _sigma_psi_deg = _model_meta.get("sigma_prior_psi_angle")
                    _dir_candidates = [
                        float(value)
                        for value in (_sigma_rot_deg, _sigma_tilt_deg)
                        if value is not None and float(value) > 0.0
                    ]
                    if _dir_candidates:
                        _relion_sigma_rot_deg = max(_dir_candidates)
                    if _sigma_psi_deg is not None and float(_sigma_psi_deg) > 0.0:
                        _relion_sigma_psi_deg = float(_sigma_psi_deg)
                if _relion_sigma_rot_deg is None:
                    _relion_sigma_rot_deg = _relion_psi_step_deg
                    logger.info(
                        "Replay override: model local prior sigma missing; falling back to RELION psi_step %.3f deg",
                        _relion_psi_step_deg,
                    )
                if _relion_sigma_psi_deg is None:
                    _relion_sigma_psi_deg = _relion_sigma_rot_deg
                _relion_sigma_rot_rad = np.deg2rad(_relion_sigma_rot_deg)
                _relion_sigma_psi_rad = np.deg2rad(_relion_sigma_psi_deg)
                if (
                    abs(float(state.sigma_rot) - _relion_sigma_rot_rad) > 1e-8
                    or abs(float(state.sigma_psi) - _relion_sigma_psi_rad) > 1e-8
                ):
                    logger.info(
                        "Replay override: local prior sigma %.3f/%.3f deg -> %.3f/%.3f deg (from %s)",
                        float(np.rad2deg(state.sigma_rot)),
                        float(np.rad2deg(state.sigma_psi)),
                        _relion_sigma_rot_deg,
                        _relion_sigma_psi_deg,
                        _model_star if _model_star is not None else _star,
                    )
                state.sigma_rot = _relion_sigma_rot_rad
                state.sigma_psi = _relion_sigma_psi_rad
            if (
                abs(float(state.translation_range) - _relion_offset_range) > 1e-6
                or abs(float(state.translation_step) - _relion_offset_step) > 1e-6
            ):
                logger.info(
                    "Replay override: translation_range %.3f -> %.3f px, step %.3f -> %.3f px",
                    float(state.translation_range),
                    _relion_offset_range,
                    float(state.translation_step),
                    _relion_offset_step,
                )
                state.translation_range = _relion_offset_range
                state.translation_step = _relion_offset_step

            # Override current_size from the RELION model star that records the
            # control state for the replayed E-step. Empirically, replaying
            # RELION iter N+1 against the saved benchmark trajectory requires
            # reading run_it{N+1}_model.star, not run_it{N}_model.star:
            # the saved model star already carries the control variables
            # (current_size, sigma_offset) used by that E-step.
            if _model_meta is not None:
                _relion_cs = int(_model_meta["current_image_size"])
                if _relion_cs <= 0:
                    logger.info(
                        "Replay override: ignoring non-positive current_size=%d from %s",
                        _relion_cs,
                        _model_star,
                    )
                elif cs != _relion_cs:
                    logger.info(
                        "Replay override: current_size %d -> %d (from %s)",
                        cs,
                        _relion_cs,
                        _model_star,
                    )
                    cs = _relion_cs

            if iteration > 0:
                _prior_iter = init_relion_iteration + iteration
                if iter_replay_override is None or iter_replay_override.get("direction_prior") is None:
                    for _half_idx in range(2):
                        _prior_star = os.path.join(
                            perturb_replay_relion_dir,
                            f"run_it{_prior_iter:03d}_half{_half_idx + 1}_model.star",
                        )
                        if not os.path.exists(_prior_star):
                            continue
                        _relion_direction_prior = (
                            read_relion_direction_priors(_prior_star, n_classes)
                            if k_class_enabled
                            else read_relion_direction_prior(_prior_star)
                        )
                        _relion_direction_prior_order = infer_direction_prior_healpix_order(
                            _relion_direction_prior[0] if k_class_enabled else _relion_direction_prior
                        )
                        if _relion_direction_prior_order != state.healpix_order:
                            logger.info(
                                "Replay override: remapping half-%d direction prior from healpix_order=%d to %d",
                                _half_idx + 1,
                                _relion_direction_prior_order,
                                state.healpix_order,
                            )
                            if k_class_enabled:
                                _relion_direction_prior = np.stack(
                                    [
                                        remap_direction_prior_to_healpix_order(
                                            _relion_direction_prior[class_idx],
                                            _relion_direction_prior_order,
                                            state.healpix_order,
                                        )
                                        for class_idx in range(n_classes)
                                    ],
                                    axis=0,
                                )
                            else:
                                _relion_direction_prior = remap_direction_prior_to_healpix_order(
                                    _relion_direction_prior,
                                    _relion_direction_prior_order,
                                    state.healpix_order,
                                )
                            _relion_direction_prior_order = state.healpix_order
                        if k_class_enabled:
                            class_direction_prior_per_half[_half_idx] = normalize_class_direction_prior_per_half(
                                [_relion_direction_prior, None] if _half_idx == 0 else [None, _relion_direction_prior],
                                n_classes,
                            )[_half_idx]
                            class_direction_prior_order_per_half[_half_idx] = _relion_direction_prior_order
                            logger.info(
                                "Replay override: class direction prior half-%d <- %s (%d classes, %d directions)",
                                _half_idx + 1,
                                _prior_star,
                                class_direction_prior_per_half[_half_idx].shape[0],
                                class_direction_prior_per_half[_half_idx].shape[1],
                            )
                        else:
                            global_direction_prior_per_half[_half_idx] = _relion_direction_prior
                            global_direction_prior_order_per_half[_half_idx] = _relion_direction_prior_order
                            logger.info(
                                "Replay override: direction prior half-%d <- %s (%d directions, range=[%.6f, %.6f], zeros=%d)",
                                _half_idx + 1,
                                _prior_star,
                                len(_relion_direction_prior),
                                float(_relion_direction_prior.min()),
                                float(_relion_direction_prior.max()),
                                int(np.sum(_relion_direction_prior == 0)),
                            )

        if iter_replay_override is not None:
            _replay_sigma = iter_replay_override.get("translation_sigma_angstrom")
            if _replay_sigma is not None:
                current_sigma_offset_angstrom = float(_replay_sigma)
                logger.info(
                    "Replay override: sigma_offset <- %.4f A (iter=%d)",
                    current_sigma_offset_angstrom,
                    iteration + 1,
                )
            _replay_prev_trans = iter_replay_override.get("previous_best_translations")
            if _replay_prev_trans is not None:
                relion_half_inputs.previous_best_translations = _optional_float32_half_pair(_replay_prev_trans)
                logger.info(
                    "Replay override: previous_best_translations <- half1=%s half2=%s",
                    "set" if relion_half_inputs.previous_best_translations[0] is not None else "none",
                    "set" if relion_half_inputs.previous_best_translations[1] is not None else "none",
                )
            _replay_prev_rots = iter_replay_override.get("previous_best_rotations")
            if _replay_prev_rots is not None:
                previous_best_rotations = _optional_float32_half_pair(_replay_prev_rots)
                logger.info(
                    "Replay override: previous_best_rotations <- half1=%s half2=%s",
                    "set" if previous_best_rotations[0] is not None else "none",
                    "set" if previous_best_rotations[1] is not None else "none",
                )
            _replay_prev_eulers = iter_replay_override.get("previous_best_rotation_eulers")
            if _replay_prev_eulers is not None:
                relion_half_inputs.previous_best_rotation_eulers = _optional_float32_half_pair(_replay_prev_eulers)
                logger.info(
                    "Replay override: previous_best_rotation_eulers <- half1=%s half2=%s",
                    "set" if relion_half_inputs.previous_best_rotation_eulers[0] is not None else "none",
                    "set" if relion_half_inputs.previous_best_rotation_eulers[1] is not None else "none",
                )
            _replay_img_corr = iter_replay_override.get("image_corrections")
            if _replay_img_corr is not None:
                relion_half_inputs.image_corrections = _optional_float32_half_pair(_replay_img_corr)
                logger.info(
                    "Replay override: image_corrections <- half1=%s half2=%s",
                    "set" if relion_half_inputs.image_corrections[0] is not None else "none",
                    "set" if relion_half_inputs.image_corrections[1] is not None else "none",
                )
            _replay_scale_corr = iter_replay_override.get("scale_corrections")
            if _replay_scale_corr is not None:
                relion_half_inputs.scale_corrections = _optional_float32_half_pair(_replay_scale_corr)
                logger.info(
                    "Replay override: scale_corrections <- half1=%s half2=%s",
                    "set" if relion_half_inputs.scale_corrections[0] is not None else "none",
                    "set" if relion_half_inputs.scale_corrections[1] is not None else "none",
                )
            _replay_noise = iter_replay_override.get("noise_variance")
            if _replay_noise is not None:
                noise_variance_per_half = _normalize_noise_variance_per_half(_replay_noise, n_halves=2)
                noise_variance = _mean_noise_variance(noise_variance_per_half)
                previous_noise_radial_per_half = [
                    _radial_profile_from_noise_variance(noise_k, cryo.image_shape)
                    for noise_k in noise_variance_per_half
                ]
                previous_noise_radial = jnp.asarray(
                    np.mean(np.stack(previous_noise_radial_per_half, axis=0), axis=0),
                    dtype=jnp.float32,
                )
                logger.info("Replay override: sigma2_noise <- per-half model.star arrays")
            _replay_dir_prior = iter_replay_override.get("direction_prior")
            if _replay_dir_prior is not None:
                if k_class_enabled:
                    replay_priors = normalize_class_direction_prior_per_half(_replay_dir_prior, n_classes)
                else:
                    replay_priors = normalize_direction_prior_per_half(_replay_dir_prior)
                for _half_idx in range(2):
                    if replay_priors[_half_idx] is None:
                        continue
                    prior_k = np.asarray(replay_priors[_half_idx], dtype=np.float32)
                    prior_order_k = infer_direction_prior_healpix_order(prior_k[0] if k_class_enabled else prior_k)
                    if prior_order_k != state.healpix_order:
                        logger.info(
                            "Replay override: remapping provided half-%d direction prior from healpix_order=%d to %d",
                            _half_idx + 1,
                            prior_order_k,
                            state.healpix_order,
                        )
                        if k_class_enabled:
                            prior_k = np.stack(
                                [
                                    remap_direction_prior_to_healpix_order(
                                        prior_k[class_idx],
                                        prior_order_k,
                                        state.healpix_order,
                                    )
                                    for class_idx in range(n_classes)
                                ],
                                axis=0,
                            )
                        else:
                            prior_k = remap_direction_prior_to_healpix_order(
                                prior_k,
                                prior_order_k,
                                state.healpix_order,
                            )
                        prior_order_k = state.healpix_order
                    if k_class_enabled:
                        class_direction_prior_per_half[_half_idx] = normalize_class_direction_prior(prior_k, n_classes)
                        class_direction_prior_order_per_half[_half_idx] = prior_order_k
                        logger.info(
                            "Replay override: class direction prior half-%d <- provided override (%d classes, %d directions)",
                            _half_idx + 1,
                            class_direction_prior_per_half[_half_idx].shape[0],
                            class_direction_prior_per_half[_half_idx].shape[1],
                        )
                    else:
                        global_direction_prior_per_half[_half_idx] = prior_k
                        global_direction_prior_order_per_half[_half_idx] = prior_order_k
                        logger.info(
                            "Replay override: direction prior half-%d <- provided override (%d directions, range=[%.6f, %.6f], zeros=%d)",
                            _half_idx + 1,
                            len(prior_k),
                            float(prior_k.min()),
                            float(prior_k.max()),
                            int(np.sum(prior_k == 0)),
                        )

        sigma_offset_used_trajectory.append(float(current_sigma_offset_angstrom))
        current_sizes.append(cs)
        healpix_order_trajectory.append(state.healpix_order)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            iteration + 1,
            max_iter,
            cs,
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
                current_rotations, current_rotation_eulers = _relion_rotation_grid_float32(new_order)
                current_healpix_order = new_order
                global_direction_prior_per_half = [None, None]
                global_direction_prior_order_per_half = [None, None]
            else:
                logger.info(
                    "Angular step refined to order %d (exhaustive grid stays at order %d — local search handles finer sampling)",
                    state.healpix_order,
                    current_healpix_order,
                )
                global_direction_prior_per_half = [None, None]
                global_direction_prior_order_per_half = [None, None]
                class_direction_prior_per_half = [None, None]
                class_direction_prior_order_per_half = [None, None]

            # Regenerate translation grid based on updated parameters
            current_translations = jnp.array(
                get_translation_grid(
                    state.translation_range,
                    state.translation_step,
                ).astype(np.float32)
            )
            base_translations = current_translations
            logger.info(
                "New grid: %d rotations, %d translations (range=%.1f, step=%.1f)",
                current_rotations.shape[0],
                current_translations.shape[0],
                state.translation_range,
                state.translation_step,
            )
        elif _replay_meta is not None:
            # Translation params may have changed under replay without an
            # hp_order bump. Regenerate the translation grid to match RELION.
            _new_t = jnp.array(
                get_translation_grid(
                    state.translation_range,
                    state.translation_step,
                ).astype(np.float32)
            )
            if _new_t.shape != base_translations.shape or not jnp.allclose(_new_t, base_translations):
                current_translations = _new_t
                base_translations = _new_t
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
        effective_rotation_eulers = np.asarray(current_rotation_eulers, dtype=np.float32)
        rotation_log_prior_per_half = [None, None]
        class_rotation_log_prior_per_half = [None, None]
        use_local = state.do_local_search and all(
            eulers is not None for eulers in relion_half_inputs.previous_best_rotation_eulers
        )
        # --- Apply RELION SamplingPerturbation to the trial grid for this iter ---
        # healpix_sampling.cpp:1909-1934 (rotations) + 1810-1820 (translations)
        # Perturbation is a rigid rotation of SO(3): A := A @ R_perturb applied
        # AFTER oversampling. At adaptive_oversampling=0 (os0 RELION runs),
        # the coarse grid IS the trial grid so we apply directly here.
        if _replay_meta is not None:
            random_perturbation = float(_replay_meta["random_perturbation"])
            logger.info(
                "Perturbation replay: iter=%d rp=%+.5f pf=%.3f relion_hp_order=%d",
                iteration + 1,
                random_perturbation,
                float(_replay_meta["perturbation_factor"]),
                int(_replay_meta["healpix_order"]),
            )
        elif perturb_factor > 0:
            relion_iter = int(init_relion_iteration) + iteration + 1
            if perturb_seed is not None:
                seed = int(perturb_seed) + relion_iter
                random_perturbation = advance_relion_perturbation_from_seed(
                    random_perturbation,
                    perturb_factor,
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
                random_perturbation = advance_relion_perturbation(random_perturbation, perturb_factor, perturb_rng)
                logger.info("Perturbation advance: iter=%d rp=%+.5f", iteration + 1, random_perturbation)
        if _replay_meta is not None or perturb_factor > 0:
            # Use RELION's actual hp_order when replaying (recovar's current
            # grid order may be capped at MAX_FULL_GRID_ORDER=4 for memory).
            _angsamp_order = int(_replay_meta["healpix_order"]) if _replay_meta is not None else current_healpix_order
            angsamp_deg = relion_angular_sampling_deg(_angsamp_order, adaptive_oversampling=0)
            if effective_rotation_eulers is not None:
                effective_rotations, effective_rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
                    effective_rotation_eulers,
                    random_perturbation,
                    angsamp_deg,
                )
            else:
                effective_rotations = apply_relion_rotation_perturbation(
                    np.asarray(effective_rotations),
                    random_perturbation,
                    angsamp_deg,
                ).astype(np.float32)
                effective_rotation_eulers = utils.R_to_relion(np.asarray(effective_rotations), degrees=True).astype(
                    np.float32
                )
            _perturbed_translations = apply_relion_translation_perturbation(
                np.asarray(base_translations),
                random_perturbation,
                float(state.translation_step),
            )
            current_translations = jnp.asarray(_perturbed_translations, dtype=jnp.float32)
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
        sigma_rot = state.sigma_rot
        sigma_psi = state.sigma_psi if state.sigma_psi > 0 else sigma_rot
        if use_local and sigma_rot <= 0:
            step_rad = np.deg2rad(healpix_angular_step(state.healpix_order) / (2**state.adaptive_oversampling))
            sigma_rot = np.sqrt(2.0 * 2.0) * step_rad
            sigma_psi = sigma_rot

        if use_local:
            local_search_order = state.healpix_order + state.adaptive_oversampling
            local_search_random_perturbation = 0.0
            local_search_angular_sampling_deg = None
            if effective_rotations.shape[0] != rotation_grid_size(local_search_order):
                logger.info(
                    "Using selected-only fine local-search grid: order=%d (%d rotations) from capped base order=%d",
                    local_search_order,
                    rotation_grid_size(local_search_order),
                    current_healpix_order,
                )
                local_search_angular_sampling_deg = relion_angular_sampling_deg(
                    local_search_order,
                    adaptive_oversampling=0,
                )
                if _precompute_exact_local_fine_grid_enabled(local_search_order):
                    _, local_search_rotation_eulers = _relion_rotation_grid_float32(local_search_order)
                    local_search_rotations, local_search_rotation_eulers = apply_relion_rotation_perturbation_to_eulers(
                        local_search_rotation_eulers,
                        float(random_perturbation),
                        local_search_angular_sampling_deg,
                    )
                    local_search_random_perturbation = 0.0
                else:
                    local_search_rotations = None
                    local_search_rotation_eulers = None
                    local_search_random_perturbation = float(random_perturbation)
            else:
                local_search_rotations = effective_rotations
                local_search_rotation_eulers = None
            logger.info(
                "Local search (batched exact): fine_order=%d, sigma_rot=%.4f rad (%.2f deg), sigma_psi=%.4f rad",
                local_search_order,
                sigma_rot,
                np.rad2deg(sigma_rot),
                sigma_psi,
            )
        else:
            for _half_idx in range(2):
                if k_class_enabled:
                    class_prior_k = class_direction_prior_per_half[_half_idx]
                    class_prior_order_k = class_direction_prior_order_per_half[_half_idx]
                    if class_prior_k is None and global_direction_prior_per_half[_half_idx] is not None:
                        shared_prior = np.asarray(global_direction_prior_per_half[_half_idx], dtype=np.float32)
                        class_prior_k = np.broadcast_to(shared_prior[None, :], (n_classes, shared_prior.size)).copy()
                        class_prior_order_k = global_direction_prior_order_per_half[_half_idx]
                    if class_prior_k is not None and class_prior_order_k == current_healpix_order:
                        class_rotation_log_prior_per_half[_half_idx] = np.stack(
                            [
                                make_relion_direction_log_prior(class_prior_k[class_idx], current_healpix_order)
                                for class_idx in range(n_classes)
                            ],
                            axis=0,
                        )
                        logger.info(
                            "Using learned per-class global direction prior half-%d: %d classes, %d directions at healpix_order=%d",
                            _half_idx + 1,
                            n_classes,
                            class_prior_k.shape[1],
                            current_healpix_order,
                        )
                        continue
                prior_k = global_direction_prior_per_half[_half_idx]
                prior_order_k = global_direction_prior_order_per_half[_half_idx]
                if prior_k is None or prior_order_k != current_healpix_order:
                    continue
                rotation_log_prior_per_half[_half_idx] = make_relion_direction_log_prior(
                    prior_k,
                    current_healpix_order,
                )
                logger.info(
                    "Using learned global direction prior half-%d: %d directions at healpix_order=%d",
                    _half_idx + 1,
                    prior_k.shape[0],
                    current_healpix_order,
                )

        cs_for_engine = cs if cs < cryo.image_shape[0] else None

        # --- Run E+M on each half-set ---
        # Two modes: single-pass (adaptive_oversampling=0) or two-pass
        # coarse/fine (adaptive_oversampling>=1).
        iter_sig_counts = None
        use_adaptive = state.adaptive_oversampling > 0 and not use_local and effective_rotations.shape[0] > 16
        # Track the rotation grids used for pose extraction.
        # When adaptive oversampling is active, ha_k indices refer to the
        # oversampled grid (from pass 2), not effective_rotations.
        pose_rotations = [None, None]  # rotations to use with ha for poses
        pose_rotation_eulers = [None, None]
        best_pose_rotations = [None, None]
        best_pose_rotation_eulers = [None, None]
        best_pose_translations = [None, None]
        translation_search_bases = [None, None]
        # Coarse-grid assignments for local search tracking (always indexed
        # into effective_rotations, even when adaptive oversampling is used).
        coarse_ha = [None, None]
        class_posterior_per_half = [None, None]

        if use_adaptive:
            # --- TWO-PASS ADAPTIVE OVERSAMPLING (RELION parity) ---
            # Pass 1: coarse E-step at reduced resolution to find
            #         significant orientations.
            # Pass 2: oversampled E+M at full current_size for significant
            #         orientations only.

            # Compute coarse image size from angular step
            effective_step_deg = healpix_angular_step(current_healpix_order)
            pixel_size = cryo.voxel_size if cryo.voxel_size > 0 else 1.0
            coarse_size = compute_coarse_image_size(
                effective_step_deg,
                pixel_size,
                grid_size,
                particle_diameter=particle_diameter_ang,
            )
            coarse_size = clamp_relion_coarse_image_size(
                coarse_size,
                cs if cs_for_engine is not None else None,
                grid_size,
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

        noise_stats_per_half = [None, None]
        # D.2: per-class noise stats (K-tuple of NoiseStats per half) for the
        # per-class sigma_offset C1 update at end-of-iter. K=1 paths leave
        # this None; K-class paths populate from k_class_result.noise_stats.
        noise_stats_per_half_per_class = [None, None]

        for k in range(2):
            noise_variance_k = noise_variance_per_half[k]
            rotation_log_prior_k = rotation_log_prior_per_half[k]
            class_rotation_log_prior_k = class_rotation_log_prior_per_half[k]
            previous_translations_k = relion_half_inputs.previous_best_translations[k]
            translation_search_base = relion_translation_search_base(previous_translations_k)
            translation_search_bases[k] = translation_search_base
            current_translation_range = float(state.translation_range)
            k_class_image_batch_size = image_batch_size
            dense_k_class_rotation_block_size = rotation_block_size
            if k_class_enabled:
                k_class_base_ibs, k_class_base_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                    classes=n_classes,
                    image_shape_for_batch=experiment_datasets[k].image_shape,
                )
                k_class_image_batch_size = min(
                    k_class_base_ibs,
                    _safe_firstiter_cc_image_batch_size(
                        current_translations.shape[0],
                        experiment_datasets[k].image_shape,
                    ),
                )
                if k_class_image_batch_size != image_batch_size:
                    logger.info(
                        "STRICT-PARITY: clamping dense K-class image_batch_size from %d to %d",
                        image_batch_size,
                        k_class_image_batch_size,
                    )
                dense_k_class_rotation_block_size = min(
                    k_class_base_rbs,
                    _safe_dense_k_class_rotation_block_size(
                        current_translations.shape[0],
                        k_class_image_batch_size,
                    ),
                )
                if dense_k_class_rotation_block_size != rotation_block_size:
                    logger.info(
                        "STRICT-PARITY: clamping dense K-class rotation_block_size from %d to %d",
                        rotation_block_size,
                        dense_k_class_rotation_block_size,
                    )
            # RELION translation prior sigma (ml_optimiser.cpp:7737-7746):
            # RELION checks `offset_range_x` (rlnOffsetRangeX in optimiser.star),
            # NOT the search-grid `offset_range` (rlnOffsetRange in sampling.star).
            # When offset_range_x > 0: sigma² = range_x²/9 (per-axis override)
            # When offset_range_x <= 0: sigma² = model.sigma2_offset (learned)
            # For this dataset, rlnOffsetRangeX = -1 → model sigma is used.
            # We always use current_sigma_offset_angstrom (from model star).
            #
            # Evaluate scoring and sigma-offset priors with their separate
            # RELION source formulas. `pdf_offset` scores the unperturbed
            # coarse sampling grid, while `wsum_sigma2_offset` accumulates
            # getTranslationsInPixel() shifts in storeWeightedSums.
            trans_prior_center = relion_translation_prior_center(
                previous_translations_k,
                cryo.voxel_size,
            )
            trans_sigma_center = relion_sigma_offset_prior_center(previous_translations_k)
            # A.1 fix: at iter 1 cold-start `previous_translations_k` is None, so
            # `trans_sigma_center` is None and em_engine's wsum_sigma2_offset
            # accumulator (em_engine.py:1636) is gated off. RELION still computes
            # wsum_sigma2_offset = sum_i E[||t_i||²] at iter 1 using the implicit
            # zero prior center, which seeds iter-2's sigma_offset ~ 1.6 Å (vs
            # default 10 Å). Pass a zero-centered prior to the engine so the
            # noise accumulator fires; the log-prior path at line 2517 is
            # unaffected because make_relion_translation_log_prior(None) and
            # make_relion_translation_log_prior(zeros(2)) both center at origin.
            trans_prior_center_for_engine = (
                np.zeros(2, dtype=np.float32) if trans_sigma_center is None else trans_sigma_center
            )
            translation_prior_translations = np.asarray(base_translations, dtype=np.float32)
            if current_translations.shape[0] != base_translations.shape[0]:
                if current_translations.shape[0] == 1 and base_translations.shape[0] > 1:
                    center_idx = int(base_translations.shape[0] // 2)
                    translation_prior_translations = np.asarray(
                        base_translations[center_idx : center_idx + 1],
                        dtype=np.float32,
                    )
                else:
                    translation_prior_translations = np.asarray(current_translations, dtype=np.float32)
            translation_log_prior = None
            if not use_local:
                translation_log_prior = make_relion_translation_log_prior(
                    translation_prior_translations,
                    cryo.voxel_size,
                    current_sigma_offset_angstrom,
                    trans_prior_center,
                    offset_range_pixels=None,
                )
            if experiment_datasets[k].n_units == 0:
                logger.info("Skipping E-step/M-step accumulation for empty half-%d dataset", k + 1)
                n_shells = int(cryo.image_shape[0] // 2 + 1)
                n_rot_for_stats = int(
                    rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]
                )
                accumulator_shape = (
                    (n_classes, int(np.prod(padded_volume_shape)))
                    if k_class_enabled
                    else (int(np.prod(padded_volume_shape)),)
                )
                Ft_y_k = jnp.zeros(accumulator_shape, dtype=jnp.complex128)
                Ft_ctf_k = jnp.zeros(accumulator_shape, dtype=jnp.complex128)
                ha_k = np.zeros(0, dtype=np.int32)
                class_assignments[k] = np.zeros(0, dtype=np.int32)
                class_posterior_per_half[k] = np.zeros(n_classes, dtype=np.float32)
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
                noise_stats_per_half[k] = noise_stats_k
                hard_assignments[k] = ha_k
                coarse_ha[k] = ha_k
                max_posterior_per_half[k] = np.zeros(0, dtype=np.float32)
                rotation_posterior_per_half[k] = np.zeros(n_rot_for_stats, dtype=np.float32)
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
                # For local search the per-chunk M-step only sees the
                # cone-restricted rotation set (typically a few thousand
                # rotations per image with high overlap across the chunk)
                # rather than the full ~10⁶-rotation grid at healpix order
                # 5+. Sizing the batch by the full grid produces ibs ≈ 5
                # at order 5 → chunks of 5 images → ~500 chunks per half
                # → ~7 hours per iter on the 5k benchmark.
                #
                # Estimate the per-image cone size from
                #     fraction = (sigma_cutoff * sigma_rot / pi)^2
                # which is the spherical cap area as a fraction of the
                # full SO(3) volume (good to within ~30% for reasonable
                # cones). Use that to compute an effective rotation count
                # equal to ``chunk_size * cone_size``, with a safety
                # factor of 2x for cone-overlap inefficiency.
                _cone_radius = 3.0 * float(sigma_rot)  # sigma_cutoff=3.0
                _cone_fraction = max(
                    (_cone_radius / float(np.pi)) ** 2,
                    1.0 / float(rotation_grid_size(local_search_order)),
                )
                _est_cone_rots = int(np.ceil(rotation_grid_size(local_search_order) * _cone_fraction))
                # Per-chunk effective rotations ≈ 2 * cone_size
                # (after dedup of overlapping cones).
                _eff_n_rot = max(64, 2 * _est_cone_rots)
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    _eff_n_rot,
                    current_translations.shape[0],
                )
                logger.info(
                    "Local search batch sizing: cone_radius=%.3f rad "
                    "(%.2f deg), est_cone_rots=%d, eff_n_rot=%d "
                    "→ image_batch_size=%d, rotation_block_size=%d",
                    _cone_radius,
                    np.rad2deg(_cone_radius),
                    _est_cone_rots,
                    _eff_n_rot,
                    safe_ibs,
                    safe_rbs,
                )
                translation_prior_reference_translations = np.asarray(current_translations, dtype=np.float32)
                if local_search_translation_prior_mode == "coarse":
                    if _replay_prior_translations is not None:
                        translation_prior_reference_translations = np.asarray(
                            _replay_prior_translations, dtype=np.float32
                        )
                    else:
                        translation_prior_reference_translations = np.asarray(base_translations, dtype=np.float32)
                    logger.info(
                        "RELION mode: local translation prior uses coarse base grid (n=%d) while scoring perturbed translations",
                        translation_prior_reference_translations.shape[0],
                    )
                # RELION's accelerated local-search loop still executes the
                # symbolic second pass when adaptive_oversampling == 0. In
                # that case convertAllSquaredDifferencesToWeights sets
                # significant_weight to the minimum fine-pass weight, so
                # storeWeightedSums keeps all local candidates. Do not apply
                # the 0.999 significant-support prune on this os0 path.
                local_reconstruct_significant_only = state.adaptive_oversampling > 0
                local_outputs = _run_local_search_iteration(
                    experiment_datasets[k],
                    means[k],
                    mean_variance,
                    noise_variance_k,
                    relion_half_inputs.previous_best_rotation_eulers[k],
                    local_search_rotations,
                    local_search_rotation_eulers,
                    local_search_order,
                    sigma_rot,
                    sigma_psi,
                    current_translations,
                    trans_prior_center,
                    current_sigma_offset_angstrom,
                    current_translation_range,
                    disc_type,
                    image_batch_size=safe_ibs,
                    rotation_block_size=safe_rbs,
                    current_size=cs_for_engine,
                    accumulate_noise=True,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    reconstruction_padding_factor=PADDING_FACTOR,
                    use_float64_scoring=False,
                    use_float64_projections=False,
                    do_gridding_correction=True,
                    square_window=RELION_FOURIER_WINDOW_SQUARE,
                    half_spectrum_scoring=True,
                    image_corrections=relion_half_inputs.image_corrections[k],
                    scale_corrections=relion_half_inputs.scale_corrections[k],
                    image_pre_shifts=translation_search_base,
                    score_with_masked_images=True,
                    return_profile=collect_local_search_profile,
                    disable_adjoint_y=disable_adjoint_y,
                    disable_adjoint_ctf=disable_adjoint_ctf,
                    adaptive_fraction=1.0,
                    max_significants=max_significants,
                    reconstruct_significant_only=local_reconstruct_significant_only,
                    translation_prior_reference_translations=translation_prior_reference_translations,
                    debug_iteration=iteration + 1,
                    return_best_pose_details=True,
                    translation_prior_centers=trans_prior_center_for_engine,
                    rotation_grid_random_perturbation=local_search_random_perturbation,
                    rotation_grid_angular_sampling_deg=local_search_angular_sampling_deg,
                    class_log_priors=class_log_priors if k_class_enabled else None,
                    return_class_details=k_class_enabled,
                )
                if collect_local_search_profile:
                    if k_class_enabled:
                        (
                            Ft_y_k,
                            Ft_ctf_k,
                            ha_k,
                            best_rots_k,
                            best_trans_k,
                            _best_rot_ids_k,
                            em_stats_k,
                            noise_stats_k,
                            local_profile_k,
                            class_assignments_k,
                            class_posterior_sums_k,
                        ) = local_outputs
                    else:
                        (
                            Ft_y_k,
                            Ft_ctf_k,
                            ha_k,
                            best_rots_k,
                            best_trans_k,
                            _best_rot_ids_k,
                            em_stats_k,
                            noise_stats_k,
                            local_profile_k,
                        ) = local_outputs
                    profile_row = dict(local_profile_k)
                    profile_row["iteration"] = np.int32(iteration)
                    profile_row["half_index"] = np.int32(k)
                    local_profile_history.append(profile_row)
                    if save_intermediates_dir is not None:
                        np.savez_compressed(
                            os.path.join(
                                save_intermediates_dir,
                                f"it{iteration:03d}_half{k + 1}_local_profile.npz",
                            ),
                            **local_profile_k,
                        )
                else:
                    if k_class_enabled:
                        (
                            Ft_y_k,
                            Ft_ctf_k,
                            ha_k,
                            best_rots_k,
                            best_trans_k,
                            _best_rot_ids_k,
                            em_stats_k,
                            noise_stats_k,
                            class_assignments_k,
                            class_posterior_sums_k,
                        ) = local_outputs
                    else:
                        (
                            Ft_y_k,
                            Ft_ctf_k,
                            ha_k,
                            best_rots_k,
                            best_trans_k,
                            _best_rot_ids_k,
                            em_stats_k,
                            noise_stats_k,
                        ) = local_outputs
                    best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                if k_class_enabled:
                    class_assignments[k] = np.asarray(class_assignments_k, dtype=np.int32)
                    class_posterior_per_half[k] = np.asarray(class_posterior_sums_k, dtype=np.float64)
                best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
                best_pose_rotation_eulers[k] = utils.R_to_relion(
                    np.asarray(best_rots_k),
                    degrees=True,
                ).astype(np.float32)
                best_pose_translations[k] = np.asarray(best_trans_k, dtype=np.float32)
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = None
                coarse_ha[k] = ha_k

            elif use_adaptive:
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )
                dense_skip_kwargs = {
                    **_DENSE_EM_STATIC_KWARGS,
                    "image_batch_size": safe_ibs,
                    "rotation_block_size": safe_rbs,
                    "current_size": cs_for_engine,
                    "rotation_log_prior": rotation_log_prior_k,
                    "translation_log_prior": translation_log_prior,
                    "image_corrections": relion_half_inputs.image_corrections[k],
                    "scale_corrections": relion_half_inputs.scale_corrections[k],
                    "image_pre_shifts": translation_search_base,
                    "translation_prior_centers": trans_prior_center_for_engine,
                    "relion_firstiter_score_mode": firstiter_score_mode_this_iter,
                    "relion_firstiter_winner_take_all": firstiter_winner_take_all_this_iter,
                }
                if class_rotation_log_prior_k is not None:
                    dense_skip_kwargs["rotation_log_prior"] = None
                    dense_skip_kwargs["class_rotation_log_prior"] = class_rotation_log_prior_k
                if k_class_enabled:
                    if disable_adjoint_y or disable_adjoint_ctf:
                        raise NotImplementedError("K-class refine does not support adjoint ablation flags")
                    dense_skip_kwargs["image_batch_size"] = k_class_image_batch_size
                    dense_skip_kwargs["rotation_block_size"] = dense_k_class_rotation_block_size
                    # STRICT-PARITY: at iter 1 with --firstiter_cc, route through
                    # run_dense_k_class_em_adaptive with
                    # firstiter_cc_pass2_only_best_coarse=True so the iter-1
                    # path matches RELION's expectationOneParticle pass-2
                    # masked-to-best-coarse semantics (ml_optimiser.cpp:9181-9207
                    # with K>1). This is what run_k_class_parity.py uses to
                    # achieve mean_corr 0.998 at iter 0->1 with the K=4 5k 128
                    # fixture; the basic dense engine evaluates all poses with
                    # soft posteriors, then binarizes only the M-step weights,
                    # which has subtle differences vs RELION's pass2-masked CC.
                    if relion_firstiter_cc_this_iter:
                        adaptive_os_local = int(state.adaptive_oversampling)
                        firstiter_image_batch_size = min(
                            image_batch_size,
                            _safe_firstiter_cc_image_batch_size(
                                current_translations.shape[0],
                                experiment_datasets[k].image_shape,
                            ),
                        )
                        if firstiter_image_batch_size != image_batch_size:
                            logger.info(
                                "STRICT-PARITY: clamping iter-1 winner-take-all image_batch_size from %d to %d",
                                image_batch_size,
                                firstiter_image_batch_size,
                            )
                        dense_skip_kwargs["image_batch_size"] = firstiter_image_batch_size
                        dense_skip_kwargs["rotation_block_size"] = dense_k_class_rotation_block_size
                        logger.info(
                            "STRICT-PARITY: routing iter-1 K-class through run_dense_k_class_em_adaptive (oversampling=%d)",
                            adaptive_os_local,
                        )
                        (
                            _coarse_rot_2954,
                            _coarse_trans_2954,
                            _fine_rot_2954,
                            _fine_trans_2954,
                            _rot_pmap_2954,
                            _trans_pmap_2954,
                        ) = _build_firstiter_cc_pass2_grids(
                            effective_rotations,
                            current_translations,
                            base_translations,
                            int(current_healpix_order),
                            adaptive_os_local,
                            float(state.translation_step),
                            random_perturbation,
                        )
                        _rot_pmap_for_collapse = _rot_pmap_2954
                        k_class_result = run_dense_k_class_em_adaptive(
                            experiment_datasets[k],
                            means[k],
                            mean_variance,
                            noise_variance_k,
                            _coarse_rot_2954,
                            _coarse_trans_2954,
                            _fine_rot_2954,
                            _fine_trans_2954,
                            _rot_pmap_2954,
                            _trans_pmap_2954,
                            disc_type,
                            class_log_priors=class_log_priors,
                            accumulate_noise=True,
                            return_best_pose_details=True,
                            firstiter_cc_pass2_only_best_coarse=True,
                            skip_significance_pruning=True,
                            coarse_current_size=coarse_cs,
                            fine_current_size=cs_for_engine,
                            **dense_skip_kwargs,
                        )
                    else:
                        k_class_result = run_dense_k_class_em(
                            experiment_datasets[k],
                            means[k],
                            mean_variance,
                            noise_variance_k,
                            effective_rotations,
                            current_translations,
                            disc_type,
                            class_log_priors=class_log_priors,
                            accumulate_noise=True,
                            return_best_pose_details=True,
                            **dense_skip_kwargs,
                        )
                    ha_k = np.asarray(k_class_result.pose_assignments, dtype=np.int32)
                    Ft_y_k = k_class_result.Ft_y
                    Ft_ctf_k = k_class_result.Ft_ctf
                    em_stats_k = k_class_result.stats
                    noise_stats_k = k_class_result.aggregate_noise_stats
                    noise_stats_per_half_per_class[k] = k_class_result.noise_stats
                    class_assignments[k] = np.asarray(k_class_result.class_assignments, dtype=np.int32)
                    class_posterior_per_half[k] = np.asarray(
                        k_class_result.class_posterior_sums,
                        dtype=np.float64,
                    )
                    # Collapse fine-grid rotation posteriors to coarse via the parent
                    # map when iter-1 firstiter_cc routes through the adaptive 2-pass
                    # engine with adaptive_oversampling > 0. Downstream
                    # _combined_class_direction_prior_from_halves expects the coarse-grid
                    # shape (n_rot_coarse,).
                    _n_rot_coarse_for_stats = int(effective_rotations.shape[0])
                    _per_class_rot_post_coarse = []
                    for _stats in k_class_result.per_class_stats:
                        _rot_post = np.asarray(_stats.rotation_posterior_sums, dtype=np.float64)
                        if _rot_post.shape[0] == _n_rot_coarse_for_stats:
                            _per_class_rot_post_coarse.append(_rot_post)
                        elif relion_firstiter_cc_this_iter and adaptive_os_local > 0:
                            _coarse_post = np.zeros(_n_rot_coarse_for_stats, dtype=np.float64)
                            np.add.at(
                                _coarse_post,
                                np.asarray(_rot_pmap_for_collapse, dtype=np.int64),
                                _rot_post,
                            )
                            _per_class_rot_post_coarse.append(_coarse_post)
                        else:
                            raise RuntimeError(
                                f"Unexpected K-class rotation_posterior_sums shape {_rot_post.shape}; "
                                f"expected ({_n_rot_coarse_for_stats},)"
                            )
                    class_rotation_posterior_per_half[k] = np.stack(_per_class_rot_post_coarse, axis=0)
                    if k_class_result.best_pose_rotations is None or k_class_result.best_pose_translations is None:
                        raise RuntimeError("Dense K-class path did not return best pose details")
                    best_pose_rotations[k] = np.asarray(k_class_result.best_pose_rotations, dtype=np.float32)
                    best_pose_rotation_eulers[k] = utils.R_to_relion(
                        np.asarray(k_class_result.best_pose_rotations),
                        degrees=True,
                    ).astype(np.float32)
                    best_pose_translations[k] = np.asarray(k_class_result.best_pose_translations, dtype=np.float32)
                else:
                    _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance_k,
                        effective_rotations,
                        current_translations,
                        disc_type,
                        return_stats=True,
                        accumulate_noise=True,
                        disable_adjoint_y=disable_adjoint_y,
                        disable_adjoint_ctf=disable_adjoint_ctf,
                        **dense_skip_kwargs,
                    )
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = effective_rotations
                pose_rotation_eulers[k] = effective_rotation_eulers
                coarse_ha[k] = ha_k

            else:
                # --- SINGLE-PASS E+M (no adaptive oversampling) ---
                safe_ibs, safe_rbs = _safe_batch_sizes(
                    effective_rotations.shape[0],
                    current_translations.shape[0],
                )
                em_kwargs = {
                    **_DENSE_EM_STATIC_KWARGS,
                    "image_batch_size": safe_ibs,
                    "rotation_block_size": safe_rbs,
                    "current_size": cs_for_engine,
                    "rotation_log_prior": rotation_log_prior_k,
                    "translation_log_prior": translation_log_prior,
                    "image_corrections": relion_half_inputs.image_corrections[k],
                    "scale_corrections": relion_half_inputs.scale_corrections[k],
                    "image_pre_shifts": translation_search_base,
                    "translation_prior_centers": trans_prior_center_for_engine,
                    "relion_firstiter_score_mode": firstiter_score_mode_this_iter,
                    "relion_firstiter_winner_take_all": firstiter_winner_take_all_this_iter,
                }
                if class_rotation_log_prior_k is not None:
                    em_kwargs["rotation_log_prior"] = None
                    em_kwargs["class_rotation_log_prior"] = class_rotation_log_prior_k
                if k_class_enabled:
                    if disable_adjoint_y or disable_adjoint_ctf:
                        raise NotImplementedError("K-class refine does not support adjoint ablation flags")
                    # STRICT-PARITY: at iter 1 with --firstiter_cc, route through
                    # run_dense_k_class_em_adaptive with
                    # firstiter_cc_pass2_only_best_coarse=True so the iter-1
                    # path matches RELION's expectationOneParticle pass-2
                    # masked-to-best-coarse semantics. This is the
                    # non-adaptive-oversampling K-class path (use_adaptive=False);
                    # the adaptive_oversampling=0 + K=4 + --firstiter_cc cold-start
                    # lands here.
                    if relion_firstiter_cc_this_iter:
                        adaptive_os_local = int(state.adaptive_oversampling)
                        firstiter_image_batch_size = min(
                            image_batch_size,
                            _safe_firstiter_cc_image_batch_size(
                                current_translations.shape[0],
                                experiment_datasets[k].image_shape,
                            ),
                        )
                        if firstiter_image_batch_size != image_batch_size:
                            logger.info(
                                "STRICT-PARITY: clamping iter-1 winner-take-all image_batch_size from %d to %d",
                                image_batch_size,
                                firstiter_image_batch_size,
                            )
                        logger.info(
                            "STRICT-PARITY (non-adaptive site): routing iter-1 K-class through run_dense_k_class_em_adaptive (oversampling=%d)",
                            adaptive_os_local,
                        )
                        (
                            _coarse_rot_3645,
                            _coarse_trans_3645,
                            _fine_rot_3645,
                            _fine_trans_3645,
                            _rot_pmap_3645,
                            _trans_pmap_3645,
                        ) = _build_firstiter_cc_pass2_grids(
                            effective_rotations,
                            current_translations,
                            base_translations,
                            int(current_healpix_order),
                            adaptive_os_local,
                            float(state.translation_step),
                            random_perturbation,
                        )
                        _rot_pmap_for_collapse = _rot_pmap_3645
                        k_class_result = run_dense_k_class_em_adaptive(
                            experiment_datasets[k],
                            means[k],
                            mean_variance,
                            noise_variance_k,
                            _coarse_rot_3645,
                            _coarse_trans_3645,
                            _fine_rot_3645,
                            _fine_trans_3645,
                            _rot_pmap_3645,
                            _trans_pmap_3645,
                            disc_type,
                            class_log_priors=class_log_priors,
                            accumulate_noise=True,
                            return_best_pose_details=True,
                            firstiter_cc_pass2_only_best_coarse=True,
                            skip_significance_pruning=True,
                            **em_kwargs,
                        )
                    else:
                        k_class_result = run_dense_k_class_em(
                            experiment_datasets[k],
                            means[k],
                            mean_variance,
                            noise_variance_k,
                            effective_rotations,
                            current_translations,
                            disc_type,
                            class_log_priors=class_log_priors,
                            accumulate_noise=True,
                            return_best_pose_details=True,
                            **em_kwargs,
                        )
                    ha_k = np.asarray(k_class_result.pose_assignments, dtype=np.int32)
                    Ft_y_k = k_class_result.Ft_y
                    Ft_ctf_k = k_class_result.Ft_ctf
                    em_stats_k = k_class_result.stats
                    noise_stats_k = k_class_result.aggregate_noise_stats
                    noise_stats_per_half_per_class[k] = k_class_result.noise_stats
                    class_assignments[k] = np.asarray(k_class_result.class_assignments, dtype=np.int32)
                    class_posterior_per_half[k] = np.asarray(
                        k_class_result.class_posterior_sums,
                        dtype=np.float64,
                    )
                    # Collapse fine-grid rotation posteriors to coarse via the parent
                    # map when iter-1 firstiter_cc routes through the adaptive 2-pass
                    # engine with adaptive_oversampling > 0. Downstream
                    # _combined_class_direction_prior_from_halves expects the coarse-grid
                    # shape (n_rot_coarse,).
                    _n_rot_coarse_for_stats = int(effective_rotations.shape[0])
                    _per_class_rot_post_coarse = []
                    for _stats in k_class_result.per_class_stats:
                        _rot_post = np.asarray(_stats.rotation_posterior_sums, dtype=np.float64)
                        if _rot_post.shape[0] == _n_rot_coarse_for_stats:
                            _per_class_rot_post_coarse.append(_rot_post)
                        elif relion_firstiter_cc_this_iter and adaptive_os_local > 0:
                            _coarse_post = np.zeros(_n_rot_coarse_for_stats, dtype=np.float64)
                            np.add.at(
                                _coarse_post,
                                np.asarray(_rot_pmap_for_collapse, dtype=np.int64),
                                _rot_post,
                            )
                            _per_class_rot_post_coarse.append(_coarse_post)
                        else:
                            raise RuntimeError(
                                f"Unexpected K-class rotation_posterior_sums shape {_rot_post.shape}; "
                                f"expected ({_n_rot_coarse_for_stats},)"
                            )
                    class_rotation_posterior_per_half[k] = np.stack(_per_class_rot_post_coarse, axis=0)
                    if k_class_result.best_pose_rotations is None or k_class_result.best_pose_translations is None:
                        raise RuntimeError("Dense K-class path did not return best pose details")
                    best_pose_rotations[k] = np.asarray(k_class_result.best_pose_rotations, dtype=np.float32)
                    best_pose_rotation_eulers[k] = utils.R_to_relion(
                        np.asarray(k_class_result.best_pose_rotations),
                        degrees=True,
                    ).astype(np.float32)
                    best_pose_translations[k] = np.asarray(k_class_result.best_pose_translations, dtype=np.float32)
                else:
                    _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em(
                        experiment_datasets[k],
                        means[k],
                        mean_variance,
                        noise_variance_k,
                        effective_rotations,
                        current_translations,
                        disc_type,
                        return_stats=True,
                        accumulate_noise=True,
                        disable_adjoint_y=disable_adjoint_y,
                        disable_adjoint_ctf=disable_adjoint_ctf,
                        **em_kwargs,
                    )
                noise_stats_per_half[k] = noise_stats_k
                pose_rotations[k] = effective_rotations
                pose_rotation_eulers[k] = effective_rotation_eulers
                coarse_ha[k] = ha_k  # same grid, no oversampling

                # --- Manifest dump for deterministic replay (Phase 0.1) ---
                if save_intermediates_dir is not None:
                    _manifest_path = os.path.join(
                        save_intermediates_dir,
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
                        "perturbation_factor": np.float64(perturb_factor),
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
            hard_assignments[k] = ha_k
            max_posterior_per_half[k] = np.asarray(
                em_stats_k.max_posterior_per_image,
                dtype=np.float32,
            )
            rotation_posterior_per_half[k] = np.asarray(
                em_stats_k.rotation_posterior_sums,
                dtype=np.float32,
            )

            if k == 0:
                Ft_y_0, Ft_ctf_0 = Ft_y_k, Ft_ctf_k
            else:
                Ft_y_1, Ft_ctf_1 = Ft_y_k, Ft_ctf_k

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
                translation_search_base=translation_search_bases[k] if "translation_search_bases" in dir() else None,
                original_image_indices=_half_orig_idx,
            )

        # E-step + per-half M-step accumulators are now both populated.
        _parity_dump.mark_stage(iteration, "e_step")
        if k_class_enabled:
            class_weights = _class_weights_from_posterior(
                class_posterior_per_half,
                n_classes,
                class_weights,
            )
            class_log_priors = np.log(class_weights)
            class_weight_trajectory.append(class_weights.copy())
            logger.info(
                "K-class occupancies: %s",
                ", ".join(f"class {idx + 1}={weight:.4f}" for idx, weight in enumerate(class_weights)),
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
            Ft_y_combined = Ft_y_0 + Ft_y_1
            Ft_ctf_combined = Ft_ctf_0 + Ft_ctf_1
        elif low_resol_join_halves_angstrom is not None and low_resol_join_halves_angstrom > 0:
            prev_res_angstrom = None
            if pixel_resolutions:
                prev_pixel_res = pixel_resolutions[-1]
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
                padded_volume_shape,
                cryo.voxel_size,
                grid_size,
                low_resol_join_halves_angstrom,
                current_resolution_angstrom=prev_res_angstrom,
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
        previous_means = [np.asarray(mean).copy() if mean is not None else None for mean in means]

        _t_unreg_first = time.time()
        if k_class_enabled:
            tau2_update_details_per_class = []
            mean_signal_variance_per_class = []
            data_vs_prior_per_class = []
            # Dense RECOVAR accumulators live in the historical unnormalised
            # image frame: RELION BPref weight = Ft_ctf * N^4. Equivalently,
            # keep Ft_y/Ft_ctf in RECOVAR frame and scale RELION tau2 by N^4
            # before the Wiener solve. See initial_model/gpu_pipeline.py's
            # bp_weight_frame_scale for the same frame conversion.
            kclass_tau2_frame_scale = float(grid_size) ** 4
            for class_idx in range(n_classes):
                mean_signal_variance_relion_k, tau2_update_details_k = (
                    regularization.compute_relion_tau2_from_iref_power_spectrum(
                        previous_means[0][class_idx],
                        volume_shape,
                        padding_factor=PADDING_FACTOR,
                        current_size=cs,
                        return_details=True,
                    )
                )
                mean_signal_variance_k = mean_signal_variance_relion_k * jnp.asarray(
                    kclass_tau2_frame_scale,
                    dtype=mean_signal_variance_relion_k.dtype,
                )
                tau2_shells_recovar_frame_k = jnp.asarray(
                    tau2_update_details_k["tau2_shells"],
                    dtype=mean_signal_variance_k.dtype,
                ) * jnp.asarray(kclass_tau2_frame_scale, dtype=mean_signal_variance_k.dtype)
                shell_stats_k = regularization._compute_relion_weight_shell_stats(
                    Ft_ctf_combined[class_idx],
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    r_max=cs // 2,
                    shell_rounding="round",
                )
                reconstruct_floor_stats_k = regularization._compute_relion_weight_shell_stats(
                    Ft_ctf_combined[class_idx],
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    r_max=cs // 2,
                    shell_rounding="floor",
                )
                data_vs_prior_k = regularization.compute_data_vs_prior(
                    Ft_ctf_combined[class_idx],
                    tau2_shells_recovar_frame_k,
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    tau2_fudge=tau2_fudge,
                    current_size=cs,
                )
                mean_signal_variance_per_class.append(mean_signal_variance_k)
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
                    import pathlib

                    pathlib.Path(_kclass_dump_dir).mkdir(parents=True, exist_ok=True)
                    np.savez(
                        pathlib.Path(_kclass_dump_dir)
                        / f"recovar_kclass_mstep_it{iteration + 1:03d}_c{class_idx + 1:02d}.npz",
                        iteration=np.int32(iteration + 1),
                        class_index=np.int32(class_idx + 1),
                        current_size=np.int32(cs),
                        padding_factor=np.int32(PADDING_FACTOR),
                        grid_size=np.int32(grid_size),
                        tau2_fudge=np.float64(tau2_fudge),
                        tau2_frame_scale=np.float64(kclass_tau2_frame_scale),
                        previous_mean=np.asarray(previous_means[0][class_idx], dtype=np.complex64),
                        previous_mean_half0=np.asarray(previous_means[0][class_idx], dtype=np.complex64),
                        previous_mean_half1=np.asarray(previous_means[1][class_idx], dtype=np.complex64),
                        Ft_y_combined=np.asarray(Ft_y_combined[class_idx], dtype=np.complex64),
                        Ft_ctf_0=np.asarray(Ft_ctf_0[class_idx], dtype=np.complex64),
                        Ft_ctf_1=np.asarray(Ft_ctf_1[class_idx], dtype=np.complex64),
                        Ft_ctf_combined=np.asarray(Ft_ctf_combined[class_idx], dtype=np.complex64),
                        tau2_shells=np.asarray(tau2_shells_recovar_frame_k, dtype=np.float64),
                        tau2_shells_relion=np.asarray(tau2_update_details_k["tau2_shells"], dtype=np.float64),
                        sigma2_shells=np.asarray(
                            jnp.where(
                                shell_stats_k["avg_weight_shells"] > 0,
                                1.0 / (PADDING_FACTOR**3 * shell_stats_k["avg_weight_shells"]),
                                0.0,
                            ),
                            dtype=np.float64,
                        ),
                        avg_weight_shells=np.asarray(shell_stats_k["avg_weight_shells"], dtype=np.float64),
                        shell_sum=np.asarray(shell_stats_k["shell_sum"], dtype=np.float64),
                        shell_count=np.asarray(shell_stats_k["shell_count"], dtype=np.float64),
                        reconstruct_floor_avg_weight_shells=np.asarray(
                            reconstruct_floor_stats_k["avg_weight_shells"],
                            dtype=np.float64,
                        ),
                        reconstruct_floor_shell_count=np.asarray(
                            reconstruct_floor_stats_k["shell_count"],
                            dtype=np.float64,
                        ),
                        data_vs_prior=np.asarray(data_vs_prior_k, dtype=np.float64),
                    )
            mean_signal_variance = jnp.stack(mean_signal_variance_per_class, axis=0)
            data_vs_prior_iter = np.stack(
                [np.asarray(dvp, dtype=np.float32) for dvp in data_vs_prior_per_class], axis=0
            )
            data_vs_prior_trajectory.append(data_vs_prior_iter)
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
                "Computed iter-%d Class3D tau2 from previous Iref power spectra: %.1fs",
                iteration + 1,
                time.time() - _t_unreg_first,
            )
        else:
            # Optional dump of post-join Ft_y, Ft_ctf for shell-by-shell parity
            # comparison against RELION's RECOVAR_MSTEP_DUMP_DIR. Activated by
            # RECOVAR_BPREF_ACCUM_DUMP_DIR. One npz per iteration.
            _bpref_accum_dir = os.environ.get("RECOVAR_BPREF_ACCUM_DUMP_DIR")
            if _bpref_accum_dir:
                import pathlib

                pathlib.Path(_bpref_accum_dir).mkdir(parents=True, exist_ok=True)
                np.savez(
                    pathlib.Path(_bpref_accum_dir) / f"recovar_bpref_accum_it{iteration + 1:03d}.npz",
                    iteration=np.int32(iteration + 1),
                    current_size=np.int32(cs),
                    padding_factor=np.int32(PADDING_FACTOR),
                    grid_size=np.int32(grid_size),
                    voxel_size=np.float32(cryo.voxel_size),
                    volume_shape=np.asarray(volume_shape, dtype=np.int32),
                    Ft_y_0=np.asarray(Ft_y_0, dtype=np.complex64),
                    Ft_y_1=np.asarray(Ft_y_1, dtype=np.complex64),
                    Ft_ctf_0=np.asarray(Ft_ctf_0, dtype=np.complex64).real.astype(np.float32),
                    Ft_ctf_1=np.asarray(Ft_ctf_1, dtype=np.complex64).real.astype(np.float32),
                )
            current_iter_fsc = regularization.compute_relion_fsc_from_backprojector(
                Ft_y_0,
                Ft_y_1,
                Ft_ctf_0,
                Ft_ctf_1,
                volume_shape,
                padding_factor=PADDING_FACTOR,
                r_max=cs // 2,
            )
            logger.info(
                "Computed iter-%d FSC for tau2 (RELION backprojector path): %.1fs",
                iteration + 1,
                time.time() - _t_unreg_first,
            )

            # RELION calls BackProjector::updateSSNRarrays independently for each
            # half-map BPref.  The gold-standard FSC is shared, but sigma2/tau2
            # come from each half's own Fourier weight outside the joined shells.
            tau2_update_details_per_half = []
            mean_signal_variance_per_half = []
            for Ft_ctf_half in (Ft_ctf_0, Ft_ctf_1):
                mean_signal_variance_k, _, tau2_update_details_k = regularization.compute_relion_tau2_from_weights(
                    Ft_ctf_half,
                    Ft_ctf_half,
                    current_iter_fsc,
                    volume_shape,
                    tau2_fudge=tau2_fudge,
                    padding_factor=PADDING_FACTOR,
                    r_max=cs // 2,
                    return_details=True,
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
            mean_signal_variance_per_half=mean_signal_variance_per_half if not k_class_enabled else None,
            n_classes=n_classes,
            k_class_enabled=k_class_enabled,
            cs=cs,
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
            relion_firstiter_ini_high_angstrom=relion_firstiter_ini_high_angstrom,
            relion_width_mask_edge=RELION_WIDTH_MASK_EDGE,
        )
        _parity_dump.mark_stage(iteration, "recon")

        significant_counts.append(iter_sig_counts)

        if (
            not use_local
            and all(rot_sum is not None for rot_sum in rotation_posterior_per_half)
            and effective_rotations.shape[0] == rotation_grid_size(current_healpix_order)
        ):
            if k_class_enabled and all(rot_sum is not None for rot_sum in class_rotation_posterior_per_half):
                combined_class_direction_prior = _combined_class_direction_prior_from_halves(
                    class_rotation_posterior_per_half,
                    n_classes,
                    current_healpix_order,
                )
                for k in range(2):
                    class_direction_prior_per_half[k] = combined_class_direction_prior.copy()
                    class_direction_prior_order_per_half[k] = current_healpix_order
            else:
                for k in range(2):
                    global_direction_prior_per_half[k] = collapse_rotation_posterior_to_direction_prior(
                        np.asarray(rotation_posterior_per_half[k], dtype=np.float64),
                        current_healpix_order,
                    )
                    global_direction_prior_order_per_half[k] = current_healpix_order

        # --- Compute unregularized half-maps only when diagnostics need them ---
        #
        # The K=1 FSC path is already computed above directly from the
        # BackProjector accumulators (`current_iter_fsc`), matching RELION's
        # ordering. For K>1, the shared class3D prior has already been derived
        # from the previous Iref power spectrum. Reconstructing unregularized
        # maps here is only needed for saved intermediates/parity dumps, so
        # skip it in normal timing/production paths.
        _t_unreg = time.time()
        need_unreg_means = save_intermediates_dir is not None or _parity_dump.is_active()
        if need_unreg_means:
            if k_class_enabled:
                unreg_shared = jnp.stack(
                    [
                        _reconstruct_volume_eager(
                            Ft_ctf_combined[class_idx],
                            Ft_y_combined[class_idx],
                            volume_shape,
                            PADDING_FACTOR,
                            tau=None,
                            tau2_fudge=tau2_fudge,
                            projection_padding_factor=PROJECTION_PADDING_FACTOR,
                            minres_map=RELION_MINRES_MAP,
                        ).reshape(-1)
                        for class_idx in range(n_classes)
                    ],
                    axis=0,
                )
                unreg_means = [unreg_shared, unreg_shared]
            else:
                unreg_means = [
                    _reconstruct_volume_eager(
                        Ft_ctf_half,
                        Ft_y_half,
                        volume_shape,
                        PADDING_FACTOR,
                        tau=None,
                        tau2_fudge=tau2_fudge,
                        projection_padding_factor=PROJECTION_PADDING_FACTOR,
                        minres_map=RELION_MINRES_MAP,
                    )
                    for Ft_ctf_half, Ft_y_half in ((Ft_ctf_0, Ft_y_0), (Ft_ctf_1, Ft_y_1))
                ]
        else:
            unreg_means = [None, None]
        if k_class_enabled:
            aligned_classes = []
            unreg_classes = [] if unreg_means[0] is not None else None
            for class_idx in range(n_classes):
                aligned_class, sign_flipped = _align_fourier_volume_sign_to_reference(
                    means[0][class_idx],
                    previous_means[0][class_idx],
                    volume_shape,
                )
                aligned_classes.append(aligned_class)
                if unreg_classes is not None:
                    unreg_classes.append(-unreg_means[0][class_idx] if sign_flipped else unreg_means[0][class_idx])
                if sign_flipped:
                    logger.info("Aligned shared class-%d volume sign to the previous reference", class_idx + 1)
            shared_aligned = jnp.stack(aligned_classes, axis=0)
            means[0] = shared_aligned
            means[1] = shared_aligned
            if unreg_classes is not None:
                shared_unreg = jnp.stack(unreg_classes, axis=0)
                unreg_means = [shared_unreg, shared_unreg]
        else:
            for k in range(2):
                means[k], sign_flipped = _align_fourier_volume_sign_to_reference(
                    means[k],
                    previous_means[k],
                    volume_shape,
                )
                if sign_flipped and unreg_means[k] is not None:
                    unreg_means[k] = -unreg_means[k]
                if sign_flipped:
                    logger.info("Aligned half-%d volume sign to the previous reference", k + 1)
        logger.info(
            "Unregularized reconstruction (2 halves): %.1fs%s",
            time.time() - _t_unreg,
            "" if need_unreg_means else " (skipped; diagnostics disabled)",
        )

        # K>1 uses the shared per-class data_vs_prior curve to drive growth;
        # K=1 keeps the split-half FSC history.
        if k_class_enabled:
            fsc = None
            fsc_history.append(fsc)
            _parity_dump.mark_stage(iteration, "fsc")
        else:
            # FSC was already computed above in the RELION-exact ordering block
            # (current_iter_fsc) and used to derive tau2 BEFORE the Wiener solve.
            # Reuse it here — recomputing would give the same value (same
            # underlying unreg accumulators).
            fsc = current_iter_fsc
            fsc_history.append(fsc)
            _parity_dump.mark_stage(iteration, "fsc")

        # --- Save intermediate volumes if requested ---
        if save_intermediates_dir is not None:
            _save_iteration_intermediates(
                save_intermediates_dir,
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
                cs=cs,
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
        combined_max_posterior = np.concatenate(
            [np.asarray(pmax, dtype=np.float32) for pmax in max_posterior_per_half],
            axis=0,
        )
        ave_pmax = float(np.mean(combined_max_posterior))
        ave_Pmax_trajectory.append(ave_pmax)
        pmax_per_image_history.append(combined_max_posterior.copy())

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
            dvp_iter = np.asarray(data_vs_prior_trajectory[-1], dtype=np.float32).copy()
            if cs < grid_size:
                dvp_iter[..., min(dvp_iter.shape[-1], cs // 2 + 1) :] = 0.0
            dvp_res_shell = max(
                resolution_from_data_vs_prior(dvp_class, allow_high_res_recovery=False)
                for dvp_class in np.asarray(dvp_iter)
            )
            pixel_res = float(dvp_res_shell)
        else:
            dvp_iter = np.asarray(fsc, dtype=np.float32).copy()
            if cs < grid_size:
                dvp_iter[min(len(dvp_iter), cs // 2) :] = 0.0
            dvp_iter = np.asarray(
                fsc_to_relion_ssnr(dvp_iter, tau2_fudge=tau2_fudge),
            )
            dvp_res_shell = resolution_from_data_vs_prior(
                dvp_iter,
                allow_high_res_recovery=True,
            )
            pixel_res = float(dvp_res_shell)
        pixel_resolutions.append(pixel_res)

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
        for k in range(2):
            if best_pose_rotations[k] is not None:
                best_rots = np.asarray(best_pose_rotations[k], dtype=np.float32)
                best_eulers = (
                    np.asarray(best_pose_rotation_eulers[k], dtype=np.float32)
                    if best_pose_rotation_eulers[k] is not None
                    else utils.R_to_relion(best_rots, degrees=True).astype(np.float32)
                )
                best_trans = np.asarray(best_pose_translations[k], dtype=np.float32)
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
                    best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
                else:
                    best_rots = np.asarray(local_search_rotations, dtype=np.float32)[rot_idx]
                    if local_search_rotation_eulers is not None:
                        best_eulers = np.asarray(local_search_rotation_eulers, dtype=np.float32)[rot_idx]
                    else:
                        best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
                best_trans = np.asarray(current_translations)[trans_idx]
            else:
                # Global search uses the dense grid in pose_rotations[k].
                # All dense EM / K-class paths report the flattened
                # rotation-translation row index here.
                rot_idx = hard_assignments[k] // current_translations.shape[0]
                trans_idx = hard_assignments[k] % current_translations.shape[0]
                best_rots = np.asarray(pose_rotations[k], dtype=np.float32)[rot_idx]
                best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
                best_trans = np.asarray(current_translations)[trans_idx]
            new_iter_best_rotations[k] = best_rots
            new_iter_best_rotation_eulers[k] = best_eulers
            new_iter_best_translations[k] = _relion_metadata_translations(
                prior_iter_best_translations[k],
                best_trans,
            )
        previous_best_rotations = new_iter_best_rotations
        relion_half_inputs.previous_best_rotation_eulers = new_iter_best_rotation_eulers
        relion_half_inputs.previous_best_translations = new_iter_best_translations
        best_rotation_eulers_history.append(
            [np.asarray(e).copy() if e is not None else None for e in new_iter_best_rotation_eulers]
        )
        best_translations_history.append(
            [np.asarray(t).copy() if t is not None else None for t in new_iter_best_translations]
        )

        if all(rot is not None for rot in new_iter_best_rotations):
            current_rotation_matrices_combined = np.concatenate(
                [np.asarray(rot, dtype=np.float32) for rot in new_iter_best_rotations],
                axis=0,
            )
        else:
            current_rotation_matrices_combined = None
        if all(rot is not None for rot in prior_iter_best_rotations):
            previous_rotation_matrices_combined = np.concatenate(
                [np.asarray(rot, dtype=np.float32) for rot in prior_iter_best_rotations],
                axis=0,
            )
        else:
            previous_rotation_matrices_combined = None
        if all(trans is not None for trans in new_iter_best_translations):
            current_translations_pixel_combined = np.concatenate(
                [np.asarray(trans, dtype=np.float32) for trans in new_iter_best_translations],
                axis=0,
            )
        else:
            current_translations_pixel_combined = None
        if all(trans is not None for trans in prior_iter_best_translations):
            previous_translations_pixel_combined = np.concatenate(
                [np.asarray(trans, dtype=np.float32) for trans in prior_iter_best_translations],
                axis=0,
            )
        else:
            previous_translations_pixel_combined = None

        if not k_class_enabled:
            data_vs_prior_trajectory.append(np.asarray(dvp_iter, dtype=np.float32))
            previous_data_vs_prior_for_scheduling = np.asarray(dvp_iter, dtype=np.float32)

        # RELION-style posterior-weighted noise update. Sums the wsum/img_power
        # accumulators from both half-sets and normalizes via the M-step formula.
        if noise_stats_per_half[0] is None or noise_stats_per_half[1] is None:
            raise RuntimeError(
                "RELION mode expected per-half NoiseStats from the EM engine; "
                "ensure accumulate_noise=True is plumbed through pass 2.",
            )
        if relion_firstiter_cc_this_iter:
            noise_from_res_per_half = [
                np.asarray(noise_k, dtype=np.float64) for noise_k in previous_noise_radial_per_half
            ]
            noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)
            logger.info(
                "RELION iter-1 CC emulation: keeping previous sigma2_noise (skip first-iter noise update)",
            )
        else:
            if k_class_enabled:
                combined_noise_stats = _combined_noise_stats(noise_stats_per_half)
                if combined_noise_stats is None:
                    raise RuntimeError("K-class noise update expected at least one NoiseStats object")
                noise_shared = noise.normalize_wsum_to_sigma2_noise(
                    np.asarray(combined_noise_stats.wsum_sigma2_noise, dtype=np.float64),
                    np.asarray(combined_noise_stats.wsum_img_power, dtype=np.float64),
                    combined_noise_stats.sumw,
                    cryo.image_shape,
                )
                noise_from_res = np.asarray(noise_shared, dtype=np.float64)
                noise_from_res_per_half = [noise_from_res.copy(), noise_from_res.copy()]
                noise_variance_shared = jnp.asarray(
                    noise.make_radial_noise(noise_shared, cryo.image_shape),
                ).reshape(-1)
                noise_variance_per_half = [noise_variance_shared, noise_variance_shared]
            else:
                noise_from_res_per_half = []
                for k_noise, stats_k in enumerate(noise_stats_per_half):
                    noise_k = noise.normalize_wsum_to_sigma2_noise(
                        np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64),
                        np.asarray(stats_k.wsum_img_power, dtype=np.float64),
                        stats_k.sumw,
                        cryo.image_shape,
                    )
                    noise_from_res_per_half.append(np.asarray(noise_k, dtype=np.float64))
                    noise_variance_per_half[k_noise] = jnp.asarray(
                        noise.make_radial_noise(noise_k, cryo.image_shape),
                    ).reshape(-1)
                noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)

            # Log per-shell noise comparison (first 10 shells) for convergence diagnostics
            old_noise_radial = previous_noise_radial
            n_log = min(10, len(noise_from_res), len(old_noise_radial))
            logger.info(
                "Noise update per shell (first %d): old=[%s] new=[%s]",
                n_log,
                ", ".join(f"{float(x):.3e}" for x in old_noise_radial[:n_log]),
                ", ".join(f"{float(x):.3e}" for x in noise_from_res[:n_log]),
            )
            _maybe_dump_noise_update_debug(
                iteration=iteration,
                current_size=cs,
                image_shape=cryo.image_shape,
                noise_stats_per_half=noise_stats_per_half,
                previous_noise_radial_per_half=previous_noise_radial_per_half,
                noise_from_res_per_half=noise_from_res_per_half,
                noise_from_res=noise_from_res,
            )

            previous_noise_radial_per_half = noise_from_res_per_half
            previous_noise_radial = jnp.asarray(noise_from_res, dtype=jnp.float32)
            noise_variance = _mean_noise_variance(noise_variance_per_half)
            _parity_dump.mark_stage(iteration, "noise_update")

        # Save per-iter per-shell sigma2 (after this iter's noise update) and
        # the exact shell-wise tau2 ingredients used in the Wiener update.
        noise_radial_trajectory.append(np.asarray(noise_from_res, dtype=np.float64))
        noise_radial_per_half_trajectory.append(
            np.stack([np.asarray(noise_k, dtype=np.float64) for noise_k in noise_from_res_per_half], axis=0),
        )
        if tau2_update_details is not None:
            tau2_radial_trajectory.append(np.asarray(tau2_update_details["prior_shells"], dtype=np.float64))
            tau2_sigma2_trajectory.append(np.asarray(tau2_update_details["sigma2_shells"], dtype=np.float64))
            tau2_avg_weight_trajectory.append(np.asarray(tau2_update_details["avg_weight_shells"], dtype=np.float64))
            tau2_shell_sum_trajectory.append(np.asarray(tau2_update_details["shell_sum"], dtype=np.float64))
            tau2_shell_count_trajectory.append(np.asarray(tau2_update_details["shell_count"], dtype=np.float64))
            if k_class_enabled:
                tau2_fsc_used_trajectory.append(None)
                tau2_ssnr_trajectory.append(np.asarray(tau2_update_details["ssnr_shells"], dtype=np.float64))
            else:
                tau2_fsc_used_trajectory.append(np.asarray(tau2_update_details["fsc_shells"], dtype=np.float64))
                tau2_ssnr_trajectory.append(np.asarray(tau2_update_details["ssnr_shells"], dtype=np.float64))
        else:
            tau2_radial_trajectory.append(None)
            tau2_sigma2_trajectory.append(None)
            tau2_avg_weight_trajectory.append(None)
            tau2_shell_sum_trajectory.append(None)
            tau2_shell_count_trajectory.append(None)
            tau2_fsc_used_trajectory.append(None)
            tau2_ssnr_trajectory.append(None)

        # --- Update convergence state ---
        # This checks assignment changes, resolution stalls, and may trigger
        # angular step refinement or convergence.
        n_rot_current = rotation_grid_size(local_search_order) if use_local else effective_rotations.shape[0]
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

        # RELION's calculateExpectedAngularErrors (ml_optimiser.cpp:9534)
        iter_acc_rot = None
        iter_acc_trans = None
        if iter_sig_counts is not None and len(iter_sig_counts) > 0:
            iter_acc_rot, _ = calculate_expected_angular_errors(
                state.healpix_order,
                iter_sig_counts,
                n_translations=n_trans_current,
            )
            logger.info(
                "acc_rot=%.3f deg (from %d images, mean n_sig=%.1f)",
                iter_acc_rot,
                len(iter_sig_counts),
                float(np.mean(iter_sig_counts)),
            )

        if perturb_replay_relion_dir is not None:
            _optimiser_iter = int(init_relion_iteration) + iteration + 1
            _optimiser_star = os.path.join(
                perturb_replay_relion_dir,
                f"run_it{_optimiser_iter:03d}_optimiser.star",
            )
            if os.path.exists(_optimiser_star):
                try:
                    _optimiser_meta = read_relion_optimiser_metadata(_optimiser_star)
                    _relion_acc_rot = _optimiser_meta.get("overall_accuracy_rotations")
                    _relion_acc_trans_angst = _optimiser_meta.get("overall_accuracy_translations_angst")
                    if _relion_acc_rot is not None and np.isfinite(float(_relion_acc_rot)):
                        iter_acc_rot = float(_relion_acc_rot)
                    if _relion_acc_trans_angst is not None and np.isfinite(float(_relion_acc_trans_angst)):
                        iter_acc_trans = float(_relion_acc_trans_angst)
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
            n_rotations=n_rot_current,
            n_translations=n_trans_current,
            translations=np.asarray(current_translations),
            new_resolution=new_res_angstrom,
            max_posterior_per_image=combined_max_posterior,
            acc_rot=iter_acc_rot,
            acc_trans=iter_acc_trans,
            current_rotation_matrices=current_rotation_matrices_combined,
            previous_rotation_matrices=previous_rotation_matrices_combined,
            current_translations_pixel=current_translations_pixel_combined,
            previous_translations_pixel=previous_translations_pixel_combined,
            current_classes=current_combined_classes,
            previous_classes=previous_combined_classes,
            voxel_size_angstrom=float(cryo.voxel_size if cryo.voxel_size > 0 else 1.0),
        )

        # Track frac_changed for local search fallback
        from recovar.em.dense_single_volume.helpers.convergence import compute_assignment_changes

        frac_changed = compute_assignment_changes(
            current_combined_ha,
            previous_combined_ha,
            n_rot_current,
            n_trans_current,
            current_healpix_order,
        )
        state._last_frac_changed = frac_changed
        frac_changed_trajectory.append(float(frac_changed))

        # --- C1 (RELION-parity): update sigma2_offset from data ---
        # Prefer RELION's posterior-weighted sufficient statistic:
        #   sigma2_offset_new = wsum_sigma2_offset / (2 * sum_weight)
        # for 2D single-particle data. Fall back to the older hard-assignment
        # proxy only when a path does not propagate the full posterior moment.
        sigma2_offset_wsum = 0.0
        sigma2_offset_sumw = 0.0
        for stats_k in noise_stats_per_half:
            if stats_k is None:
                continue
            sigma2_offset_wsum += float(getattr(stats_k, "wsum_sigma2_offset", 0.0))
            sigma2_offset_sumw += float(getattr(stats_k, "sumw", 0.0))
        # D.2: per-class sigma_offset diagnostic. RELION Class3D maintains K
        # independent sigma_offset values; recovar currently uses the
        # cross-class aggregate. Compute and log per-class sigmas to gauge
        # whether the per-class refactor is justified for this fixture.
        per_class_sigma_offset = None
        if k_class_enabled:
            per_class_w = np.zeros(n_classes, dtype=np.float64)
            per_class_n = np.zeros(n_classes, dtype=np.float64)
            for half_per_class in noise_stats_per_half_per_class:
                if half_per_class is None:
                    continue
                for c, stats_c in enumerate(half_per_class):
                    if stats_c is None:
                        continue
                    per_class_w[c] += float(getattr(stats_c, "wsum_sigma2_offset", 0.0))
                    per_class_n[c] += float(getattr(stats_c, "sumw", 0.0))
            min_sigma2 = 2.0
            per_class_sigma_offset = np.full(n_classes, current_sigma_offset_angstrom, dtype=np.float64)
            for c in range(n_classes):
                if per_class_w[c] > 0.0 and per_class_n[c] > 0.0:
                    s2 = max(per_class_w[c] / (2.0 * per_class_n[c]), min_sigma2)
                    per_class_sigma_offset[c] = float(np.sqrt(s2))
            logger.info(
                "C1: per-class sigma_offset = [%s] (cross-class aggregate %.3f Å)",
                ", ".join(f"{s:.3f}" for s in per_class_sigma_offset),
                float(np.sqrt(max(sigma2_offset_wsum / max(2.0 * sigma2_offset_sumw, 1e-30), min_sigma2)))
                if sigma2_offset_wsum > 0
                else current_sigma_offset_angstrom,
            )
        if sigma2_offset_wsum > 0.0 and sigma2_offset_sumw > 0.0:
            min_sigma2_angstrom2 = 2.0
            sigma2_offset_angstrom2 = max(
                sigma2_offset_wsum / (2.0 * sigma2_offset_sumw),
                min_sigma2_angstrom2,
            )
            current_sigma_offset_angstrom = float(np.sqrt(sigma2_offset_angstrom2))
            logger.info(
                "C1: sigma_offset updated %.3f Å from posterior variance (clamp sigma^2 >= %.3f Å^2)",
                current_sigma_offset_angstrom,
                min_sigma2_angstrom2,
            )
        else:
            new_sigma_offset_angstrom = state.current_changes_optimal_offsets_angstrom
            if np.isfinite(new_sigma_offset_angstrom) and new_sigma_offset_angstrom > 0:
                min_sigma_angstrom = float(np.sqrt(2.0))  # RELION min_sigma2_offset = 2 Å²
                current_sigma_offset_angstrom = max(
                    float(new_sigma_offset_angstrom),
                    min_sigma_angstrom,
                )
                logger.info(
                    "C1 fallback: sigma_offset updated %.3f Å from hard assignments (clamp >= %.3f Å)",
                    current_sigma_offset_angstrom,
                    min_sigma_angstrom,
                )
        sigma_offset_trajectory.append(float(current_sigma_offset_angstrom))
        per_class_sigma_offset_trajectory.append(
            None if per_class_sigma_offset is None else per_class_sigma_offset.tolist()
        )
        acc_rot_trajectory.append(float(iter_acc_rot) if iter_acc_rot is not None else np.nan)
        smallest_change_angles_trajectory.append(float(state.current_changes_optimal_orientations))
        smallest_change_offsets_trajectory.append(float(state.current_changes_optimal_offsets_angstrom))

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
                    current_size=int(cs),
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
                    sigma2_noise=np.asarray(noise_variance, dtype=np.float64)
                    if "noise_variance" in dir()
                    else np.zeros(0),
                    means=means,
                    unreg_means=unreg_means,
                    new_iter_best_rotation_eulers=new_iter_best_rotation_eulers,
                    new_iter_best_translations=new_iter_best_translations,
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
        wall_times.append(elapsed)

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
            cs,
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

        if state.has_converged and not force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d. Final resolution: %.2f A (pixel_res=%.1f)",
                iteration + 1,
                res_angstrom,
                pixel_res,
            )
            break
        if state.has_converged and force_max_iter_after_convergence:
            logger.info(
                "Convergence reached at iteration %d, continuing because force_max_iter_after_convergence=True",
                iteration + 1,
            )

        iteration += 1

    # RELION's final all-data iteration is a real next iteration after
    # convergence flags are set at the top of the loop. Do not synthesize it
    # after plain max_iter exhaustion, and do not synthesize it when
    # convergence is first detected on the last allowed iteration.
    should_run_final_iteration = bool(
        state.has_converged and not force_max_iter_after_convergence and (iteration + 1) < max_iter
    )
    if skip_final_iteration or not should_run_final_iteration:
        if not skip_final_iteration and not should_run_final_iteration:
            logger.info(
                "Skipping RELION final all-data iteration: has_converged=%s, "
                "iteration=%d, max_iter=%d, force_max_iter_after_convergence=%s",
                state.has_converged,
                iteration,
                max_iter,
                force_max_iter_after_convergence,
            )
        merged_mean, merged_class_means = _merged_mean_from_halves(
            means,
            class_weights if k_class_enabled else None,
        )
        return {
            "mean": merged_mean,
            "means": means,
            "class_means": merged_class_means,
            "class_weights": class_weights if k_class_enabled else None,
            "class_assignments": class_assignments if k_class_enabled else None,
            "class_weight_trajectory": class_weight_trajectory,
            "fsc": fsc_history[-1] if fsc_history else None,
            "hard_assignments": hard_assignments,
            "current_sizes": current_sizes,
            "fsc_history": fsc_history,
            "pixel_resolutions": pixel_resolutions,
            "wall_times": wall_times,
            "significant_counts": significant_counts,
            "convergence_state": state,
            "data_vs_prior_trajectory": data_vs_prior_trajectory,
            "healpix_order_trajectory": healpix_order_trajectory,
            "ave_Pmax_trajectory": ave_Pmax_trajectory,
            "pmax_per_image_history": pmax_per_image_history,
            "noise_radial_trajectory": noise_radial_trajectory,
            "noise_radial_per_half_trajectory": noise_radial_per_half_trajectory,
            "tau2_radial_trajectory": tau2_radial_trajectory,
            "tau2_sigma2_trajectory": tau2_sigma2_trajectory,
            "tau2_avg_weight_trajectory": tau2_avg_weight_trajectory,
            "tau2_shell_sum_trajectory": tau2_shell_sum_trajectory,
            "tau2_shell_count_trajectory": tau2_shell_count_trajectory,
            "tau2_fsc_used_trajectory": tau2_fsc_used_trajectory,
            "tau2_ssnr_trajectory": tau2_ssnr_trajectory,
            "sigma_offset_used_trajectory": sigma_offset_used_trajectory,
            "sigma_offset_trajectory": sigma_offset_trajectory,
            "per_class_sigma_offset_trajectory": per_class_sigma_offset_trajectory,
            "frac_changed_trajectory": frac_changed_trajectory,
            "acc_rot_trajectory": acc_rot_trajectory,
            "smallest_change_angles_trajectory": smallest_change_angles_trajectory,
            "smallest_change_offsets_trajectory": smallest_change_offsets_trajectory,
            "best_rotation_eulers_history": best_rotation_eulers_history,
            "best_translations_history": best_translations_history,
            "local_profile_history": local_profile_history,
            "setup_phase_seconds": setup_phase_seconds,
        }
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
    final_iter_t0 = time.time()
    logger.info("=== RELION final all-data Nyquist iteration (do_join_random_halves=True, do_use_all_data=True) ===")
    final_cs = grid_size  # = ori_size, full Nyquist
    recon_vol_size = int(np.prod([d * PADDING_FACTOR for d in volume_shape]))
    final_accumulator_shape = (n_classes, recon_vol_size) if k_class_enabled else (recon_vol_size,)
    final_ft_y = jnp.zeros(final_accumulator_shape, dtype=cryo.dtype)
    final_ft_ctf = jnp.zeros(final_accumulator_shape, dtype=cryo.dtype)
    final_class_assignments = [None, None]
    final_class_posterior_per_half = [None, None]
    for k in range(2):
        # Pass the merged mean as input (both halves get the same projection source).
        # Run on each half-set's particles (avoids loading all particles at once),
        # then accumulate Ft_y/Ft_ctf and noise stats from BOTH halves.
        safe_ibs, safe_rbs = _safe_batch_sizes(
            current_rotations.shape[0],
            current_translations.shape[0],
        )
        final_em_kwargs = dict(
            image_batch_size=safe_ibs,
            rotation_block_size=safe_rbs,
            current_size=final_cs,  # full Nyquist
            score_with_masked_images=True,
            half_spectrum_scoring=True,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            reconstruction_padding_factor=PADDING_FACTOR,
            image_corrections=relion_half_inputs.image_corrections[k],
            scale_corrections=relion_half_inputs.scale_corrections[k],
            image_pre_shifts=relion_translation_search_base(relion_half_inputs.previous_best_translations[k]),
            use_float64_scoring=False,
            use_float64_projections=False,
            do_gridding_correction=True,
            square_window=RELION_FOURIER_WINDOW_SQUARE,
            sparse_pass2=False,
        )
        if (
            k_class_enabled
            and class_direction_prior_per_half[k] is not None
            and class_direction_prior_order_per_half[k] == current_healpix_order
        ):
            final_em_kwargs["class_rotation_log_prior"] = np.stack(
                [
                    make_relion_direction_log_prior(
                        class_direction_prior_per_half[k][class_idx],
                        current_healpix_order,
                    )
                    for class_idx in range(n_classes)
                ],
                axis=0,
            )
        if k_class_enabled:
            if disable_adjoint_y or disable_adjoint_ctf:
                raise NotImplementedError("K-class final all-data iteration does not support adjoint ablation flags")
            final_k_class_result = run_dense_k_class_em(
                experiment_datasets[k],
                final_join_means[k],
                mean_variance,
                noise_variance_per_half[k],
                current_rotations,
                current_translations,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=True,
                **final_em_kwargs,
            )
            Ft_y_k_final = final_k_class_result.Ft_y
            Ft_ctf_k_final = final_k_class_result.Ft_ctf
            final_class_assignments[k] = np.asarray(final_k_class_result.class_assignments, dtype=np.int32)
            final_class_posterior_per_half[k] = np.asarray(final_k_class_result.class_posterior_sums, dtype=np.float64)
        else:
            _, _, Ft_y_k_final, Ft_ctf_k_final, _, _ = run_em(
                experiment_datasets[k],
                final_join_means[k],
                mean_variance,
                noise_variance_per_half[k],
                current_rotations,
                current_translations,
                disc_type,
                return_stats=True,
                accumulate_noise=True,
                disable_adjoint_y=disable_adjoint_y,
                disable_adjoint_ctf=disable_adjoint_ctf,
                **final_em_kwargs,
            )
        # --- Manifest dump for final all-data iteration (Phase 0.1) ---
        if save_intermediates_dir is not None:
            _manifest_path = os.path.join(
                save_intermediates_dir,
                f"manifest_final_half{k}.npz",
            )
            _manifest = {
                "effective_rotations": np.asarray(current_rotations, dtype=np.float32),
                "current_translations": np.asarray(current_translations, dtype=np.float32),
                "rotation_log_prior": np.array([]),
                "translation_log_prior": np.array([]),
                "image_corrections": np.asarray(relion_half_inputs.image_corrections[k], dtype=np.float64)
                if relion_half_inputs.image_corrections[k] is not None
                else np.array([]),
                "scale_corrections": np.asarray(relion_half_inputs.scale_corrections[k], dtype=np.float64)
                if relion_half_inputs.scale_corrections[k] is not None
                else np.array([]),
                "image_pre_shifts": np.asarray(
                    relion_translation_search_base(relion_half_inputs.previous_best_translations[k]), dtype=np.float32
                )
                if relion_half_inputs.previous_best_translations[k] is not None
                else np.array([]),
                "absolute_previous_translations": np.asarray(
                    relion_half_inputs.previous_best_translations[k],
                    dtype=np.float32,
                )
                if relion_half_inputs.previous_best_translations[k] is not None
                else np.array([]),
                "mean_vol_ft": np.asarray(final_join_means[k]),
                "mean_variance": np.asarray(mean_variance),
                "noise_variance": np.asarray(noise_variance_per_half[k]),
                "current_size": np.int32(final_cs),
                "half_spectrum_scoring": np.bool_(True),
                "use_float64_scoring": np.bool_(False),
                "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
                "reconstruction_padding_factor": np.int32(PADDING_FACTOR),
                "score_with_masked_images": np.bool_(True),
                "perturbation_instance": np.float64(random_perturbation),
                "perturbation_factor": np.float64(perturb_factor),
                "iteration": np.int32(-1),
                "half_index": np.int32(k),
            }
            np.savez(_manifest_path, **_manifest)
            logger.info("Final manifest dumped: %s", _manifest_path)

        final_ft_y = final_ft_y + Ft_y_k_final
        final_ft_ctf = final_ft_ctf + Ft_ctf_k_final
    if k_class_enabled:
        class_weights = _class_weights_from_posterior(
            final_class_posterior_per_half,
            n_classes,
            class_weights,
        )
        class_log_priors = np.log(class_weights)
        class_weight_trajectory.append(class_weights.copy())

    # Reconstruct the final volume from the COMBINED Ft_y/Ft_ctf accumulators
    # at the full Nyquist resolution. Skip the join_halves step (we're already
    # combining the two halves into one dataset for this final iter).
    if k_class_enabled:
        final_class_normalise = np.maximum(np.asarray(class_weights, dtype=np.float64), np.finfo(np.float64).tiny)
        final_class_means = jnp.stack(
            [
                _reconstruct_volume_eager(
                    final_ft_ctf[class_idx] / final_class_normalise[class_idx],
                    final_ft_y[class_idx] / final_class_normalise[class_idx],
                    volume_shape,
                    PADDING_FACTOR,
                    tau=mean_variance[class_idx],
                    tau2_fudge=tau2_fudge,
                    projection_padding_factor=PROJECTION_PADDING_FACTOR,
                    minres_map=RELION_MINRES_MAP,
                ).reshape(-1)
                for class_idx in range(n_classes)
            ],
            axis=0,
        )
        merged_mean = jnp.sum(
            jnp.asarray(class_weights, dtype=final_class_means.real.dtype)[:, None] * final_class_means, axis=0
        )
        class_assignments = final_class_assignments
    else:
        final_class_means = None
        merged_mean = _reconstruct_volume_eager(
            final_ft_ctf,
            final_ft_y,
            volume_shape,
            PADDING_FACTOR,
            tau=mean_variance,
            tau2_fudge=tau2_fudge,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            minres_map=RELION_MINRES_MAP,
        ).reshape(-1)
    final_iter_elapsed = time.time() - final_iter_t0
    logger.info(
        "Final iter complete: current_size=%d (Nyquist), wall=%.1fs",
        final_cs,
        final_iter_elapsed,
    )
    wall_times.append(final_iter_elapsed)

    return {
        "mean": merged_mean,
        "means": means,
        "class_means": final_class_means,
        "class_weights": class_weights if k_class_enabled else None,
        "class_assignments": class_assignments if k_class_enabled else None,
        "class_weight_trajectory": class_weight_trajectory,
        "fsc": fsc_history[-1] if fsc_history else None,
        "hard_assignments": hard_assignments,
        "current_sizes": current_sizes,
        "fsc_history": fsc_history,
        "pixel_resolutions": pixel_resolutions,
        "wall_times": wall_times,
        "significant_counts": significant_counts,
        # RELION-mode specific outputs
        "convergence_state": state,
        "data_vs_prior_trajectory": data_vs_prior_trajectory,
        "healpix_order_trajectory": healpix_order_trajectory,
        "ave_Pmax_trajectory": ave_Pmax_trajectory,
        "pmax_per_image_history": pmax_per_image_history,
        "noise_radial_trajectory": noise_radial_trajectory,
        "noise_radial_per_half_trajectory": noise_radial_per_half_trajectory,
        "tau2_radial_trajectory": tau2_radial_trajectory,
        "tau2_sigma2_trajectory": tau2_sigma2_trajectory,
        "tau2_avg_weight_trajectory": tau2_avg_weight_trajectory,
        "tau2_shell_sum_trajectory": tau2_shell_sum_trajectory,
        "tau2_shell_count_trajectory": tau2_shell_count_trajectory,
        "tau2_fsc_used_trajectory": tau2_fsc_used_trajectory,
        "tau2_ssnr_trajectory": tau2_ssnr_trajectory,
        "sigma_offset_used_trajectory": sigma_offset_used_trajectory,
        "sigma_offset_trajectory": sigma_offset_trajectory,
        "per_class_sigma_offset_trajectory": per_class_sigma_offset_trajectory,
        "frac_changed_trajectory": frac_changed_trajectory,
        "acc_rot_trajectory": acc_rot_trajectory,
        "smallest_change_angles_trajectory": smallest_change_angles_trajectory,
        "smallest_change_offsets_trajectory": smallest_change_offsets_trajectory,
        "best_rotation_eulers_history": best_rotation_eulers_history,
        "best_translations_history": best_translations_history,
        "local_profile_history": local_profile_history,
        "setup_phase_seconds": setup_phase_seconds,
    }
