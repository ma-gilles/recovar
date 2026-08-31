"""Core refinement loop for dense single-volume EM.

This file contains the three core algorithm functions:
- ``refine_single_volume`` — public entry point
- ``_run_relion_iteration_loop`` — RELION-parity iteration loop
- ``_run_local_search_iteration`` — exact local angular search

All supporting helpers live in ``helpers/``.
See ``docs/math/relion_refinement_algorithm.md`` for the full algorithm map.
"""

import dataclasses
import gc
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from recovar import cuda_backproject as _cuda_backproject_diagnostics
from recovar import utils
from recovar.core import fourier_transform_utils
from recovar.data_io import cryoem_dataset
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
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as _sparse_pass2_diagnostics
from recovar.em.dense_single_volume.helpers.convergence import (
    RefinementState,
    calculate_expected_angular_errors,
    check_convergence,
    healpix_angular_step,
    refine_angular_sampling,
    update_angular_sampling,
    update_refinement_state,
)
from recovar.em.dense_single_volume.helpers.expected_accuracy import (
    estimate_relion_expected_accuracy,
    relion_half1_trial_order,
)
from recovar.em.dense_single_volume.helpers.fourier_window import quantize_current_size
from recovar.em.dense_single_volume.helpers.half_volume_mstep import (
    half_volume_accumulator_shape,
    relion_backprojector_volume_shape,
    relion_x_half_accumulators_to_public_layout,
)
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
    infer_direction_prior_healpix_order,
    make_relion_direction_log_prior,
    make_relion_translation_log_prior,
    normalize_class_direction_prior_per_half,
    normalize_direction_prior_per_half,
    relion_local_translation_prior_center,
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
from recovar.em.dense_single_volume.helpers.iteration_history import RefinementHistory
from recovar.em.dense_single_volume.frozen_boundary import (
    _restore_diagnostic_frozen_boundary_state,
)
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
    build_local_adaptive_pass2_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_layout import (  # noqa: F401
    build_local_hypothesis_layout as build_local_hypothesis_layout,
)
from recovar.em.dense_single_volume.local_search_iteration import (
    _precompute_exact_local_fine_grid_enabled,
    _run_local_search_iteration,
)
from recovar.em.dense_single_volume.mean_helpers import (  # noqa: F401  -- imported by tests
    _align_fourier_volume_sign_to_reference as _align_fourier_volume_sign_to_reference,
)
from recovar.em.dense_single_volume.mean_helpers import (
    _class_weights_from_posterior,
    _combined_class_direction_prior_from_halves,
    _initialize_class_log_priors,
    _mean_noise_variance,
    _merged_mean_from_halves,
    _normalize_initial_means,
    _normalize_noise_variance_per_half,
    _reconstruct_and_postprocess_means,
    _reconstruct_volume_eager,
    _relion_optimizer_average_pmax,
    compute_unregularized_halfmaps_and_align_signs,
    update_c1_sigma_offset_from_posterior,
    update_posterior_noise_variance,
    update_relion_norm_scale_corrections,
)
from recovar.em.dense_single_volume.mean_helpers import (
    _combined_noise_stats as _combined_noise_stats,
)
from recovar.em.dense_single_volume.refinement_options import (
    RefinementOptions,
)
from recovar.em.dense_single_volume.relion_metadata import (
    _radial_profile_from_noise_variance,
    _relion_metadata_translations,
    _relion_rotation_grid_float32,
    _rotation_eulers_for_canonical_or_custom_grid,  # noqa: F401 -- test-compatible re-export
)
from recovar.em.dense_single_volume.relion_replay import (
    RelionProjectorReplayState,
    _apply_replay_correction_overrides,
    _as_sigma_offset_half_pair,
    _mean_sigma_offset_per_half,
    _normalize_sigma_offset_per_half,
    _RelionHalfInputState,
    _restore_convergence_state_from_replay_restart,
    apply_iter_replay_overrides,
)
from recovar.em.dense_single_volume.relion_replay import (  # noqa: F401
    _replay_control_model_iteration as _replay_control_model_iteration,
)
from recovar.em.dense_single_volume.relion_worker_scale import (
    RelionFollowerScaleSetup,
    relion_rank1_serialized_scales,
    relion_worker_group_ids,
    select_relion_follower_scales,
    setup_relion_follower_scale_state,
    update_relion_follower_scales,
    validate_relion_follower_scale_replay_application,
)
from recovar.em.dense_single_volume.relion_worker_scale import (  # noqa: F401 -- test back-compat re-export
    _validate_coupled_relion_restart_state as _validate_coupled_relion_restart_state,
)
from recovar.em.sampling import (  # noqa: F401  -- monkeypatched by tests/unit/test_refine_relion_mode.py
    _get_relion_rotation_grid_eulers_float64,
    _relion_adaptive_pass1_rotations,
    advance_relion_perturbation,
    advance_relion_perturbation_from_seed,
    apply_relion_rotation_perturbation,
    apply_relion_rotation_perturbation_to_eulers,
    apply_relion_translation_perturbation,
    build_local_search_grid_metadata,
    get_relion_rotation_grid,
    get_relion_rotation_grid_eulers,
    get_relion_translation_grid,
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
_K1_RELION_EXACT_TRANSLATION_GRID_ENV = "RECOVAR_K1_RELION_EXACT_TRANSLATION_GRID"
_SIGNIFICANCE_DUMP_TARGET_HALF_ENV = "RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF"


def _k1_relion_exact_translation_grid_enabled(environ=None):
    """Return the production-on K=1 grid policy with a diagnostic opt-out."""
    env = os.environ if environ is None else environ
    raw = str(env.get(_K1_RELION_EXACT_TRANSLATION_GRID_ENV, "")).strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_K1_RELION_EXACT_TRANSLATION_GRID_ENV} must be a boolean value, got {raw!r}"
    )


def _translation_grid_for_class_count(max_pixel, pixel_offset, *, n_classes):
    """Use source-exact RELION translation enumeration for K=1 only."""
    if int(n_classes) == 1 and _k1_relion_exact_translation_grid_enabled():
        return get_relion_translation_grid(max_pixel, pixel_offset)
    return get_translation_grid(max_pixel, pixel_offset)


def _significance_dump_half_indices(
    *,
    numbered_iteration: int,
    n_classes: int,
    experiment_datasets,
    environ=None,
) -> tuple[int, ...]:
    """Select one half only at an explicitly terminating coarse-dump boundary."""

    env = os.environ if environ is None else environ
    raw_half = str(env.get(_SIGNIFICANCE_DUMP_TARGET_HALF_ENV, "")).strip()
    if not raw_half:
        return (0, 1)
    if str(env.get("RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET", "")).strip() != "1":
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} requires "
            "RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1"
        )
    if int(n_classes) != 1:
        raise RuntimeError(f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} is K=1 diagnostic-only")
    try:
        target_half = int(raw_half)
    except ValueError as exc:
        raise ValueError(f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} must be 1 or 2") from exc
    if target_half not in {1, 2}:
        raise ValueError(f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} must be 1 or 2")

    raw_iteration = str(env.get("RECOVAR_SIGNIFICANCE_DUMP_ITERATION", "")).strip()
    raw_targets = str(
        env.get("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "")
    ).strip()
    dump_dir = str(env.get("RECOVAR_SIGNIFICANCE_DUMP_DIR", "")).strip()
    if not raw_iteration or not raw_targets or not dump_dir:
        raise RuntimeError(
            f"{_SIGNIFICANCE_DUMP_TARGET_HALF_ENV} requires an explicit dump directory, "
            "iteration, and original-index target set"
        )
    try:
        target_iteration = int(raw_iteration)
        target_indices = {
            int(token) for token in raw_targets.replace(",", " ").split()
        }
    except ValueError as exc:
        raise ValueError("significance dump iteration and original indices must be integers") from exc
    if target_iteration <= 0 or not target_indices:
        raise ValueError("significance dump iteration and original-index target set must be nonempty")
    if int(numbered_iteration) != target_iteration:
        return (0, 1)

    selected_half_indices = set(
        np.asarray(experiment_datasets[target_half - 1].dataset_indices, dtype=np.int64).tolist()
    )
    missing = sorted(target_indices - selected_half_indices)
    if missing:
        raise RuntimeError(
            "significance dump targets are not all present in the selected half: "
            f"half={target_half} missing={missing}"
        )
    logger.info(
        "RECOVAR coarse-significance diagnostic: iteration=%d entering only half=%d "
        "for %d target particles; completion must occur before pass 2/M-step",
        int(numbered_iteration),
        target_half,
        len(target_indices),
    )
    return (target_half - 1,)


def _bpref_device_signature_active_for_numbered_half(
    *,
    iteration: int,
    half: int,
    final_all_data: bool = False,
    environ=None,
) -> bool:
    """Resolve one explicit numbered half boundary for device capture."""

    env = os.environ if environ is None else environ
    if not str(env.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "")).strip():
        return False
    raw_iteration = str(env.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION", "")).strip()
    raw_half = str(env.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF", "")).strip()
    if not raw_iteration or not raw_half:
        raise RuntimeError(
            "Scoped BPref device capture requires explicit positive "
            "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION and half 1 or 2"
        )
    try:
        target_iteration = int(raw_iteration)
        target_half = int(raw_half)
    except ValueError as exc:
        raise ValueError("BPref device capture iteration/half targets must be integers") from exc
    if target_iteration <= 0 or target_half not in {1, 2}:
        raise ValueError("BPref device capture requires target iteration > 0 and half 1 or 2")
    if int(iteration) <= 0 or int(half) not in {1, 2}:
        raise ValueError("BPref device capture context requires numbered iteration > 0 and half 1 or 2")
    if final_all_data:
        return False
    return int(iteration) == target_iteration and int(half) == target_half

logger = logging.getLogger(__name__)

# RELION parses ``--adaptive_fraction 0.999`` through ``textToFloat`` and
# stores that single-precision value in its optimiser state.  Python's literal
# 0.999 is a different binary64 boundary and can change a one-sample support
# cutoff when posterior weights are accumulated in float64.
RELION_ADAPTIVE_FRACTION = float(np.float32("0.999"))

RELION_SCORE_TENSOR_FLOAT_BUDGET = 200_000_000
RELION_FIRSTITER_RECON_COMPLEX_BUDGET = 268_435_456
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
_APPROX_ACC_ROT_CONVERGENCE_ENV = "RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE"
_APPROX_ACC_ROT_CONVERGENCE_DISABLE_ENV = "RECOVAR_EM_DISABLE_APPROX_ACC_ROT_FOR_CONVERGENCE"
_APPROX_ACC_ROT_MAX_AVE_PMAX_ENV = "RECOVAR_EM_APPROX_ACC_ROT_MAX_AVE_PMAX"
_APPROX_ACC_ROT_MIN_ITER_ENV = "RECOVAR_EM_APPROX_ACC_ROT_MIN_ITER"
_FINAL_ALL_DATA_USE_MERGED_REFERENCE_ENV = "RECOVAR_FINAL_ALL_DATA_USE_MERGED_REFERENCE"
_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE_ENV = "RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE"
_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE_ENV = "RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE"
_FINAL_ALL_DATA_GRID_CORRECT_ENV = "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT"
_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV = "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER"
_DEBUG_REPLAY_RELION_REFERENCES_ENV = "RECOVAR_DEBUG_REPLAY_RELION_REFERENCES"
_DEBUG_REPLAY_RELION_REFERENCES_ITERATION_ENV = "RECOVAR_DEBUG_REPLAY_RELION_REFERENCES_ITERATION"
_LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT"
_LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT"
_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY"
_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV = "RECOVAR_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT"
_K1_SKIP_SIGNIFICANCE_PRUNING_ENV = "RECOVAR_K1_SKIP_SIGNIFICANCE_PRUNING"
_K1_RELION_X_HALF_MSTEP_ENV = "RECOVAR_K1_RELION_X_HALF_MSTEP"
_K_CLASS_RELION_X_HALF_MSTEP_ENV = "RECOVAR_K_CLASS_RELION_X_HALF_MSTEP"
_K_CLASS_FULL_VOLUME_MSTEP_ENV = "RECOVAR_K_CLASS_FULL_VOLUME_MSTEP"
_K_CLASS_HALF_VOLUME_MSTEP_ENV = "RECOVAR_K_CLASS_HALF_VOLUME_MSTEP"
_KCLASS_REPLAY_TAU2_ENV = "RECOVAR_KCLASS_REPLAY_TAU2"
_KCLASS_REPLAY_TAU2_SAME_ITER_ENV = "RECOVAR_KCLASS_REPLAY_TAU2_SAME_ITER"
_APPROX_ACC_ROT_DEFAULT_MAX_AVE_PMAX = 0.85
_APPROX_ACC_ROT_DEFAULT_MIN_ITER = 5
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _use_approx_acc_rot_for_convergence() -> bool:
    """Return whether the cheap support-width acc_rot is force-enabled."""
    return os.environ.get(_APPROX_ACC_ROT_CONVERGENCE_ENV, "").strip().lower() in _TRUE_ENV_VALUES


def _disable_approx_acc_rot_for_convergence() -> bool:
    """Return whether all native support-width convergence gating is disabled."""
    return os.environ.get(_APPROX_ACC_ROT_CONVERGENCE_DISABLE_ENV, "").strip().lower() in _TRUE_ENV_VALUES


def _float_env_or_default(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.3f", name, value, default)
        return default


def _int_env_or_default(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default


def _replay_perturbation_seed(
    replay_dir: str,
    relion_iteration: int,
    explicit_seed: int | None,
    replay_prefix: str = "run",
) -> int | None:
    """Return the RELION optimiser seed that generated a sampling state."""
    if explicit_seed is not None:
        return int(explicit_seed)
    candidates = [
        os.path.join(replay_dir, f"{replay_prefix}_it{int(relion_iteration):03d}_optimiser.star"),
        os.path.join(replay_dir, f"{replay_prefix}_optimiser.star"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        seed = read_relion_optimiser_metadata(path).get("random_seed")
        if seed is not None:
            return int(seed)
    return None


def _resolve_replay_random_perturbation(
    *,
    star_value: float,
    perturbation_factor: float,
    relion_iteration: int,
    replay_dir: str,
    replay_prefix: str = "run",
    explicit_seed: int | None,
    precision_mode: str,
    restart_state_iteration: int | None = None,
) -> tuple[float, str]:
    """Recover RELION's live perturbation without STAR decimal truncation."""
    if precision_mode not in {"auto", "seed_exact", "star"}:
        raise ValueError(f"Unsupported perturb_replay_precision={precision_mode!r}")
    if precision_mode == "star":
        return float(star_value), "star"

    seed = _replay_perturbation_seed(
        replay_dir,
        relion_iteration,
        explicit_seed,
        replay_prefix=replay_prefix,
    )
    if seed is None:
        if precision_mode == "seed_exact":
            raise ValueError(
                "perturb_replay_precision='seed_exact' requires perturb_seed or "
                "_rlnRandomSeed in a replay optimiser STAR"
            )
        return float(star_value), "star-fallback"

    exact = relion_sampling_perturbation_for_iteration(
        float(perturbation_factor),
        int(seed),
        int(relion_iteration),
        restart_state_iteration=restart_state_iteration,
    )
    # RELION writes this field with as few as five digits after the decimal.
    # Treat the STAR value as a provenance guard, not as the arithmetic input.
    if not np.isclose(exact, float(star_value), rtol=0.0, atol=5.1e-6):
        raise ValueError(
            "Seed-reconstructed SamplingPerturbation disagrees with replay STAR: "
            f"iteration={relion_iteration} seed={seed} exact={exact:+.12g} "
            f"star={float(star_value):+.12g}"
        )
    source = "seed-exact"
    if restart_state_iteration is not None:
        source = f"seed-exact-restart@{int(restart_state_iteration)}"
    return float(exact), source


def _perturbation_restart_state_iteration(
    restart_state_iterations,
    relion_iteration: int,
) -> int | None:
    """Return the latest explicit restart boundary preceding an iteration."""
    if restart_state_iterations is None:
        return None
    candidates = [
        int(value)
        for value in restart_state_iterations
        if int(value) < int(relion_iteration)
    ]
    return max(candidates) if candidates else None




def _debug_replay_relion_references_enabled(iteration_number: int) -> bool:
    """Return whether this scoring iteration should use RELION half-map references."""

    if os.environ.get(_DEBUG_REPLAY_RELION_REFERENCES_ENV, "").strip().lower() not in _TRUE_ENV_VALUES:
        return False
    requested = os.environ.get(_DEBUG_REPLAY_RELION_REFERENCES_ITERATION_ENV)
    if requested is None or requested.strip() == "":
        return True
    try:
        requested_iterations = {int(token) for token in requested.replace(",", " ").replace(";", " ").split()}
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; RELION reference replay disabled",
            _DEBUG_REPLAY_RELION_REFERENCES_ITERATION_ENV,
            requested,
        )
        return False
    return int(iteration_number) in requested_iterations


def _maybe_debug_replay_relion_references(
    *,
    means,
    perturb_replay_relion_dir,
    perturb_replay_relion_prefix: str = "run",
    init_relion_iteration: int,
    iteration: int,
    volume_shape,
    n_classes: int,
    force: bool = False,
):
    """Debug hook: replace current scoring references with RELION maps."""

    iteration_number = int(iteration) + 1
    if not force and not _debug_replay_relion_references_enabled(iteration_number):
        return means
    if perturb_replay_relion_dir is None:
        logger.warning(
            "%s requested at iteration %d but perturb_replay_relion_dir is unset; keeping RECOVAR references",
            _DEBUG_REPLAY_RELION_REFERENCES_ENV,
            iteration_number,
        )
        return means
    from pathlib import Path

    from recovar.core import fourier_transform_utils
    from recovar.utils.helpers import load_relion_volume as _load_relion_volume

    relion_iter = int(init_relion_iteration) + int(iteration)
    relion_dir = Path(perturb_replay_relion_dir)
    replayed_means = []
    for half_idx in range(2):
        replayed_classes = []
        for class_idx in range(int(n_classes)):
            class_number = class_idx + 1
            map_path = relion_dir / (
                f"{perturb_replay_relion_prefix}_it{relion_iter:03d}_half{half_idx + 1}_"
                f"class{class_number:03d}.mrc"
            )
            if not map_path.exists():
                shared_path = relion_dir / (
                    f"{perturb_replay_relion_prefix}_it{relion_iter:03d}_class{class_number:03d}.mrc"
                )
                if shared_path.exists():
                    map_path = shared_path
            if not map_path.exists():
                raise FileNotFoundError(
                    f"{_DEBUG_REPLAY_RELION_REFERENCES_ENV}=1 requested RELION reference "
                    f"for scoring iteration {iteration_number}, half {half_idx + 1}, "
                    f"class {class_number}, but {map_path} is missing"
                )
            real_volume = np.asarray(_load_relion_volume(str(map_path)), dtype=np.float32)
            if tuple(real_volume.shape) != tuple(volume_shape):
                raise ValueError(
                    f"RELION replay reference {map_path} has shape {real_volume.shape}, "
                    f"expected {tuple(volume_shape)}"
                )
            replayed_classes.append(
                jnp.asarray(fourier_transform_utils.get_dft3(real_volume).reshape(-1))
            )
            logger.info(
                "Debug RELION reference replay: scoring iter %d half %d class %d <- %s",
                iteration_number,
                half_idx + 1,
                class_number,
                map_path,
            )
        if int(n_classes) == 1:
            replayed_means.append(replayed_classes[0])
        else:
            replayed_means.append(jnp.stack(replayed_classes, axis=0))
    return replayed_means


def _final_all_data_grid_correct_enabled() -> bool:
    """Return whether final all-data output applies RELION gridding correction.

    The GUI-quality path keeps final all-data gridding correction off by
    default because enabling it can shift final-map FSC-AUC.  Strict RELION
    replay can still enable it explicitly with
    ``RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=1``.
    """

    value = os.environ.get(_FINAL_ALL_DATA_GRID_CORRECT_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_ENV_VALUES:
        return False
    if normalized in _TRUE_ENV_VALUES:
        return True
    logger.warning("Ignoring invalid %s=%r; using default false", _FINAL_ALL_DATA_GRID_CORRECT_ENV, value)
    return False


def _final_all_data_after_max_iter_enabled() -> bool:
    """Return whether diagnostics force final all-data after iteration-cap exit."""

    value = os.environ.get(_FINAL_ALL_DATA_AFTER_MAX_ITER_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_ENV_VALUES:
        return False
    if normalized in _TRUE_ENV_VALUES:
        return True
    logger.warning("Ignoring invalid %s=%r; using default false", _FINAL_ALL_DATA_AFTER_MAX_ITER_ENV, value)
    return False


def _has_numbered_replay_iteration_overrides(replay_iteration_overrides) -> bool:
    """Return whether replay contains state beyond the cold-start boundary.

    Override slot zero is also used by ``--relion_init_dir`` to seed RELION's
    iter-0 particle/model state for an otherwise autonomous refinement.  That
    cold-start state must not implicitly turn the final all-data iteration
    into a numbered replay.  Genuine trajectory replay populates at least one
    later slot; an explicit final-replay environment override remains handled
    separately by the caller.
    """
    if replay_iteration_overrides is None or len(replay_iteration_overrides) <= 1:
        return False
    return any(override is not None for override in replay_iteration_overrides[1:])


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


def _k_class_relion_half_volume_mstep_enabled() -> bool:
    """Return whether K-class should use the old native half-volume M-step."""

    half_value = os.environ.get(_K_CLASS_HALF_VOLUME_MSTEP_ENV)
    if half_value is not None and half_value.strip() != "":
        normalized = half_value.strip().lower()
        if normalized in _TRUE_ENV_VALUES:
            return True
        if normalized not in _FALSE_ENV_VALUES:
            logger.warning(
                "Ignoring invalid %s=%r; using K-class full-volume M-step default",
                _K_CLASS_HALF_VOLUME_MSTEP_ENV,
                half_value,
            )

    full_value = os.environ.get(_K_CLASS_FULL_VOLUME_MSTEP_ENV)
    if full_value is not None and full_value.strip() != "":
        normalized = full_value.strip().lower()
        if normalized in _FALSE_ENV_VALUES:
            return True
        if normalized in _TRUE_ENV_VALUES:
            return False
        logger.warning(
            "Ignoring invalid %s=%r; using K-class full-volume M-step default",
            _K_CLASS_FULL_VOLUME_MSTEP_ENV,
            full_value,
        )

    return False


def _k_class_relion_x_half_mstep_enabled() -> bool:
    """Return whether K-class should use RELION x-half BPref M-step accumulators."""

    value = os.environ.get(_K_CLASS_RELION_X_HALF_MSTEP_ENV)
    if value is not None and value.strip() != "":
        normalized = value.strip().lower()
        if normalized in _TRUE_ENV_VALUES:
            return True
        if normalized in _FALSE_ENV_VALUES:
            return False
        logger.warning(
            "Ignoring invalid %s=%r; using K-class RELION x-half M-step default",
            _K_CLASS_RELION_X_HALF_MSTEP_ENV,
            value,
        )

    # Preserve the legacy diagnostics as explicit overrides.  ``FULL=1`` means
    # reproduce the old full-volume path; ``FULL=0`` or ``HALF=1`` mean use the
    # native half-volume path.  With neither set, default to the RELION x-half
    # BPref layout used by the K=1 parity path.
    full_value = os.environ.get(_K_CLASS_FULL_VOLUME_MSTEP_ENV)
    if full_value is not None and full_value.strip() != "":
        normalized = full_value.strip().lower()
        if normalized in _TRUE_ENV_VALUES or normalized in _FALSE_ENV_VALUES:
            return False
        logger.warning(
            "Ignoring invalid %s=%r while resolving K-class RELION x-half M-step default",
            _K_CLASS_FULL_VOLUME_MSTEP_ENV,
            full_value,
        )

    half_value = os.environ.get(_K_CLASS_HALF_VOLUME_MSTEP_ENV)
    if half_value is not None and half_value.strip() != "":
        normalized = half_value.strip().lower()
        if normalized in _TRUE_ENV_VALUES:
            return False
        if normalized not in _FALSE_ENV_VALUES:
            logger.warning(
                "Ignoring invalid %s=%r while resolving K-class RELION x-half M-step default",
                _K_CLASS_HALF_VOLUME_MSTEP_ENV,
                half_value,
            )

    return True


def _jax_cpu_forced_from_env() -> bool:
    """Return whether JAX has been forced to CPU by environment."""

    platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
    if platform_name == "cpu":
        return True
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if not platforms:
        return False
    requested = [token.strip() for token in platforms.split(",") if token.strip()]
    return bool(requested) and all(token == "cpu" for token in requested)


def _k1_relion_x_half_mstep_default_available() -> bool:
    """Return whether the default K=1 x-half M-step can use custom CUDA."""

    from recovar.utils.cuda_env import custom_cuda_disabled_from_env

    disabled, _ = custom_cuda_disabled_from_env()
    if disabled or _jax_cpu_forced_from_env():
        return False
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _k1_relion_x_half_mstep_enabled() -> bool:
    """Default-on K=1 RELION x-half BPref accumulation."""

    value = os.environ.get(_K1_RELION_X_HALF_MSTEP_ENV)
    if value is None or value.strip() == "":
        return _k1_relion_x_half_mstep_default_available()
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    logger.warning(
        "Ignoring invalid %s=%r; using K=1 RELION x-half M-step default",
        _K1_RELION_X_HALF_MSTEP_ENV,
        value,
    )
    return _k1_relion_x_half_mstep_default_available()


def _kclass_replay_tau2_enabled() -> bool:
    """Diagnostic switch: use RELION replayed Class3D tau2 spectra directly."""

    value = os.environ.get(_KCLASS_REPLAY_TAU2_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_ENV_VALUES:
        return False
    if normalized in _TRUE_ENV_VALUES:
        return True
    logger.warning("Ignoring invalid %s=%r; using default false", _KCLASS_REPLAY_TAU2_ENV, value)
    return False


def _kclass_replay_tau2_same_iter_enabled() -> bool:
    """Compatibility switch for same-numbered Class3D tau2 replay.

    ``RECOVAR_KCLASS_REPLAY_TAU2`` now uses same-numbered model.star tau2 by
    default, matching RELION's expectation-setup timing.  Keep this parser so
    existing diagnostic scripts that also set the flag continue to work.
    """

    value = os.environ.get(_KCLASS_REPLAY_TAU2_SAME_ITER_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_ENV_VALUES:
        return False
    if normalized in _TRUE_ENV_VALUES:
        return True
    logger.warning("Ignoring invalid %s=%r; using default false", _KCLASS_REPLAY_TAU2_SAME_ITER_ENV, value)
    return False


def _approx_acc_rot_policy_for_convergence(
    *,
    state,
    iteration_number: int,
    ave_pmax: float,
    new_resolution_angstrom: float,
) -> tuple[bool, str]:
    """Return whether native support-width acc_rot may gate convergence.

    The support-width estimate is much cheaper than RELION's map-perturbation
    accuracy calculation, but it overstates certainty when posterior support
    collapses to a few samples.  Keep it diagnostic-only by default; callers
    can opt into the historical convergence gate with
    RECOVAR_EM_USE_APPROX_ACC_ROT_FOR_CONVERGENCE=1.
    """
    if _use_approx_acc_rot_for_convergence():
        return True, "forced-by-env"
    if _disable_approx_acc_rot_for_convergence():
        return False, "disabled-by-env"
    if state.do_local_search:
        return False, "diagnostic-only-local-search"
    if state.healpix_order + 1 < state.auto_local_healpix_order:
        return False, "diagnostic-only-not-prelocal"

    min_iter = max(1, _int_env_or_default(_APPROX_ACC_ROT_MIN_ITER_ENV, _APPROX_ACC_ROT_DEFAULT_MIN_ITER))
    if int(iteration_number) < min_iter:
        return False, f"diagnostic-only-before-iter-{min_iter}"

    max_ave_pmax = _float_env_or_default(_APPROX_ACC_ROT_MAX_AVE_PMAX_ENV, _APPROX_ACC_ROT_DEFAULT_MAX_AVE_PMAX)
    if np.isfinite(ave_pmax) and float(ave_pmax) > max_ave_pmax:
        return False, f"diagnostic-only-high-pmax>{max_ave_pmax:.2f}"

    if np.isfinite(new_resolution_angstrom) and new_resolution_angstrom < state.current_resolution:
        return False, "diagnostic-only-resolution-improving"

    return False, "diagnostic-only-default"


def _k1_data_vs_prior_for_scheduling(
    *,
    raw_fsc,
    corrected_data_vs_prior,
    current_size,
    grid_size,
    tau2_fudge,
):
    """Return the K=1 DVP curve RELION uses for current-resolution updates.

    Auto-refine normally uses raw split-half FSC. If RELION's
    ``--solvent_correct_fsc`` path is enabled, the corrected FSC-derived DVP
    is passed here instead.
    """
    if corrected_data_vs_prior is not None:
        return _truncate_data_vs_prior_for_current_size(
            corrected_data_vs_prior,
            current_size=current_size,
            grid_size=grid_size,
        )

    fsc_prev = np.asarray(raw_fsc, dtype=np.float32).copy()
    if int(current_size) < int(grid_size):
        fsc_prev[min(len(fsc_prev), int(current_size) // 2 + 1) :] = 0.0
    return np.asarray(fsc_to_relion_ssnr(fsc_prev, tau2_fudge=tau2_fudge), dtype=np.float32)


def _truncate_data_vs_prior_for_current_size(data_vs_prior, *, current_size, grid_size):
    """Zero DVP shells beyond RELION's inclusive current-size boundary."""
    truncated = np.asarray(data_vs_prior, dtype=np.float32).copy()
    if int(current_size) < int(grid_size):
        first_unavailable_shell = min(truncated.shape[-1], int(current_size) // 2 + 1)
        truncated[..., first_unavailable_shell:] = 0.0
    return truncated


def _truncate_fsc_for_current_size_growth(fsc, *, current_size, grid_size):
    """Zero FSC shells beyond RELION's inclusive current-size boundary.

    BackProjector includes radii ``R <= current_size / 2``.  The boundary
    shell therefore remains part of RELION's full-array FSC threshold scan;
    only shells starting at ``current_size // 2 + 1`` are unavailable.
    """
    truncated = np.asarray(fsc, dtype=np.float32).copy()
    if int(current_size) < int(grid_size):
        first_unavailable_shell = min(len(truncated), int(current_size) // 2 + 1)
        truncated[first_unavailable_shell:] = 0.0
    return truncated


def _concatenate_pose_stacks_or_none(stacks, *, trailing_shape, label):
    """Concatenate per-half pose stacks, accepting empty replay stacks."""
    arrays = []
    expected_ndim = 1 + len(tuple(trailing_shape))
    for half_idx, stack in enumerate(stacks):
        if stack is None:
            return None
        arr = np.asarray(stack, dtype=np.float32)
        if arr.size == 0:
            arr = arr.reshape((0, *tuple(trailing_shape)))
        if arr.ndim != expected_ndim or tuple(arr.shape[1:]) != tuple(trailing_shape):
            logger.warning(
                "Skipping %s pose-delta stack: half-%d shape %s does not match (*, %s)",
                label,
                half_idx + 1,
                arr.shape,
                ", ".join(str(dim) for dim in trailing_shape),
            )
            return None
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def _firstiter_cc_ini_high_resolution_shell(grid_size, voxel_size, ini_high_angstrom):
    """RELION's firstiter_cc current-resolution shell from ``--ini_high``."""
    px = float(voxel_size if voxel_size > 0 else 1.0)
    shell = int(np.floor(int(grid_size) * px / float(ini_high_angstrom) + 0.5))
    return max(1, min(int(grid_size) // 2, shell))


def _firstiter_cc_scheduling_resolution_shell(
    resolution_shell,
    *,
    emulate_relion_firstiter_cc,
    ini_high_angstrom,
    relion_iteration,
    grid_size,
    voxel_size,
):
    """Apply RELION's iter-1 ``--firstiter_cc`` current-size rule.

    ``MlOptimiser::updateCurrentResolution`` uses the ``--ini_high`` shell at
    physical iteration 1 regardless of whether refinement has one or multiple
    classes. Keep this override independent of the K=1/K-class data-vs-prior
    calculation so both paths schedule iteration 2 identically.
    """

    if (
        emulate_relion_firstiter_cc
        and ini_high_angstrom is not None
        and int(relion_iteration) == 1
    ):
        return _firstiter_cc_ini_high_resolution_shell(
            grid_size,
            voxel_size,
            ini_high_angstrom,
        )
    return int(resolution_shell)


def _firstiter_cc_ini_high_tau2_taper(
    n_shells,
    grid_size,
    voxel_size,
    ini_high_angstrom,
    *,
    filter_edgewidth,
):
    """RELION's squared post-firstiter ``ini_high`` taper for tau2 state."""

    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return np.ones(int(n_shells), dtype=np.float64)
    edge = float(filter_edgewidth)
    radius = float(grid_size) * float(voxel_size) / float(ini_high_angstrom) - edge / 2.0
    radius_p = radius + edge
    shells = np.arange(int(n_shells), dtype=np.float64)
    taper = np.ones(int(n_shells), dtype=np.float64)
    taper[shells > radius_p] = 0.0
    transition = (shells >= radius) & (shells <= radius_p)
    taper[transition] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - shells[transition]) / edge)
    return taper * taper


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


def _relion_local_pass1_current_size(
    *,
    pre_update_healpix_order: int,
    pixel_size: float,
    ori_size: int,
    particle_diameter: float | None,
    current_size: int | None,
) -> int | None:
    """Return RELION's local pass-1 size for the current expectation.

    RELION computes ``image_coarse_size`` before ``updateAngularSampling``.
    When an expectation advances the sampling order, its parent hypotheses
    therefore use the incoming order for Fourier sizing even though the
    updated order controls the parent grid and fine-child expansion.
    """

    coarse_size = compute_coarse_image_size(
        healpix_angular_step(int(pre_update_healpix_order)),
        pixel_size,
        ori_size,
        particle_diameter=particle_diameter,
    )
    coarse_size = clamp_relion_coarse_image_size(
        coarse_size,
        current_size,
        ori_size,
    )
    return coarse_size if coarse_size < int(ori_size) else None


def _relion_expectation_coarse_size_order(
    *,
    state_healpix_order: int,
    replay_saved_healpix_order: int | None,
) -> int:
    """Choose the pre-update sampling order at an expectation boundary.

    In strict replay, RECOVAR's live refinement state may have advanced after
    the preceding M-step.  RELION instead enters the next expectation from
    the preceding numbered sampling STAR and only then updates angular
    sampling, so the saved replay order is authoritative for Fourier sizing.
    """

    if replay_saved_healpix_order is not None:
        return int(replay_saved_healpix_order)
    return int(state_healpix_order)


def _require_relion_follower_owners(
    owners_by_relion_iteration,
    *,
    relion_iteration: int,
    stage: str,
):
    """Return an exact captured owner row, failing closed when it is absent."""

    relion_iteration = int(relion_iteration)
    try:
        return owners_by_relion_iteration[relion_iteration]
    except KeyError as exc:
        raise RuntimeError(
            "RELION dispatch schedule does not contain "
            f"{stage} iteration {relion_iteration}; refusing to reuse stale follower owners"
        ) from exc


def _remap_relion_follower_runtime_inputs(
    *,
    state,
    relion_half_inputs,
    follower_owners_per_half,
    physical_group_count: int,
):
    """Remap live scoring scales and XA/AA group IDs to a new dispatch row."""

    stats_group_ids_per_half = []
    for half_idx in range(2):
        physical_groups = np.asarray(relion_half_inputs.group_ids[half_idx], dtype=np.int64)
        owners = np.asarray(follower_owners_per_half[half_idx], dtype=np.int64)
        old_scales = relion_half_inputs.scale_corrections[half_idx]
        old_image_corrections = relion_half_inputs.image_corrections[half_idx]
        if old_scales is None or old_image_corrections is None:
            raise RuntimeError(
                "RELION follower dispatch remap requires resident image and scale corrections"
            )
        old_scales = np.asarray(old_scales, dtype=np.float64)
        old_image_corrections = np.asarray(old_image_corrections, dtype=np.float64)
        norm_factor = np.ones_like(old_scales, dtype=np.float64)
        np.divide(
            old_image_corrections,
            old_scales,
            out=norm_factor,
            where=old_scales != 0.0,
        )
        selected_scales = select_relion_follower_scales(
            state,
            group_ids=physical_groups,
            follower_owners=owners,
        ).astype(np.float32)
        relion_half_inputs.scale_corrections[half_idx] = selected_scales
        relion_half_inputs.image_corrections[half_idx] = np.asarray(
            norm_factor * selected_scales,
            dtype=np.float32,
        )
        stats_group_ids_per_half.append(
            relion_worker_group_ids(
                physical_groups,
                owners,
                n_groups=physical_group_count,
            )
        )
    return stats_group_ids_per_half


def _dispatch_relion_follower_scale_for_numbered_iteration(
    setup: RelionFollowerScaleSetup,
    history: RefinementHistory,
    *,
    iteration: int,
    init_relion_iteration: int,
    relion_half_inputs,
    relion_follower_scale_replay_source,
) -> None:
    """Recompute this numbered iteration's follower dispatch, apply any
    diagnostic replay override, remap group ids, and record pre-score
    telemetry. Mutates ``setup.follower_owners_per_half``,
    ``setup.follower_scale_state``, and ``setup.scale_stats_group_ids_per_half``
    in place. Only called when ``setup.follower_scale_state is not None``.
    """
    numbered_relion_iteration = _numbered_relion_iteration(init_relion_iteration, iteration)
    setup.follower_owners_per_half = [
        owners.copy()
        for owners in _require_relion_follower_owners(
            setup.follower_owners_by_iteration,
            relion_iteration=numbered_relion_iteration,
            stage="numbered",
        )
    ]
    replayed_follower_scale_state = False
    if numbered_relion_iteration in setup.follower_scale_replay_by_iteration:
        if numbered_relion_iteration in history.relion_follower_scale_replay_applied_iterations:
            raise RuntimeError(
                "RELION follower-scale replay iteration was reached more than once: "
                f"{numbered_relion_iteration}"
            )
        setup.follower_scale_state = type(setup.follower_scale_state)(
            scales=setup.follower_scale_replay_by_iteration[numbered_relion_iteration].copy(),
            group_counts=np.asarray(setup.follower_scale_state.group_counts, dtype=np.float64).copy(),
            n_optics_groups=int(setup.follower_scale_state.n_optics_groups),
        )
        history.record_follower_replay_applied(numbered_relion_iteration)
        replayed_follower_scale_state = True
        logger.info(
            "Diagnostic RELION follower-scale replay: numbered_iter=%d source=%s",
            numbered_relion_iteration,
            relion_follower_scale_replay_source.source,
        )
    if iteration > 0 or replayed_follower_scale_state:
        setup.scale_stats_group_ids_per_half = _remap_relion_follower_runtime_inputs(
            state=setup.follower_scale_state,
            relion_half_inputs=relion_half_inputs,
            follower_owners_per_half=setup.follower_owners_per_half,
            physical_group_count=setup.physical_group_count,
        )
    history.record_follower_scale_pre_score(
        np.asarray(setup.follower_scale_state.scales, dtype=np.float64).copy(),
        setup.follower_owners_per_half[0].copy(),
    )
    logger.info(
        "Strict RELION dynamic dispatch: numbered_iter=%d rank_particle_counts=%s",
        numbered_relion_iteration,
        [
            int(np.count_nonzero(setup.follower_owners_per_half[0] == follower))
            for follower in range(setup.follower_count)
        ],
    )


def _dispatch_relion_follower_scale_for_final_all_data(
    setup: RelionFollowerScaleSetup,
    *,
    init_relion_iteration: int,
    numbered_iteration_count: int,
    relion_half_inputs,
) -> None:
    """Recompute the final all-data dispatch row and remap group ids -- no
    replay override, no telemetry (the final all-data pass is never replayed
    and is not part of the numbered trajectory). Mutates
    ``setup.follower_owners_per_half`` and ``setup.scale_stats_group_ids_per_half``
    in place. Only called when ``setup.follower_scale_state is not None``.
    """
    final_dispatch_relion_iteration = init_relion_iteration + numbered_iteration_count + 1
    setup.follower_owners_per_half = [
        owners.copy()
        for owners in _require_relion_follower_owners(
            setup.follower_owners_by_iteration,
            relion_iteration=final_dispatch_relion_iteration,
            stage="final all-data",
        )
    ]
    setup.scale_stats_group_ids_per_half = _remap_relion_follower_runtime_inputs(
        state=setup.follower_scale_state,
        relion_half_inputs=relion_half_inputs,
        follower_owners_per_half=setup.follower_owners_per_half,
        physical_group_count=setup.physical_group_count,
    )
    logger.info(
        "Strict RELION dynamic dispatch: final_all_data_iter=%d rank_particle_counts=%s",
        final_dispatch_relion_iteration,
        [
            int(np.count_nonzero(setup.follower_owners_per_half[0] == follower))
            for follower in range(setup.follower_count)
        ],
    )


def _final_local_sampling_orders(
    *,
    state_healpix_order: int,
    adaptive_oversampling: int,
    final_sampling_healpix_order: int | None,
) -> tuple[int, int]:
    """Return RELION's final local parent and fine HEALPix orders.

    The unnumbered final ``run_sampling.star`` may advance the parent order
    beyond the last numbered optimiser state.  When that authoritative final
    metadata exists, its order must replace the stale state order before
    applying adaptive oversampling.  Runs without a final sampling STAR keep
    the historical state-derived fallback.
    """

    parent_order = (
        int(state_healpix_order)
        if final_sampling_healpix_order is None
        else int(final_sampling_healpix_order)
    )
    return parent_order, parent_order + int(adaptive_oversampling)


def _native_final_perturbation_healpix_order(state, final_current_healpix_order: int) -> int:
    """Return the angular-step order for native final sampling perturbation.

    Local search deliberately caps the exhaustive trial grid below the active
    sampling order. RELION nevertheless scales SamplingPerturbation by the
    active local-search order recorded in ``sampling.star``, not by that capped
    global-grid order. Global search keeps the historical exhaustive-grid
    behavior.
    """

    if state.do_local_search:
        return int(state.healpix_order)
    return int(final_current_healpix_order)


def _direction_prior_healpix_order_for_scoring(
    *,
    use_local: bool,
    current_healpix_order: int,
    state_healpix_order: int,
    adaptive_oversampling: int,
    local_search_order: int | None,
) -> int:
    """Return the grid order whose global rotations receive ``pdf_direction`` priors.

    RELION's local-search branch scores the explicit local direction/psi
    priors instead of ``mymodel.pdf_direction``. Callers must only use this
    for exhaustive/global scoring.
    """

    if not use_local:
        return int(current_healpix_order)
    if int(adaptive_oversampling) > 0:
        return int(state_healpix_order)
    if local_search_order is None:
        raise ValueError("local_search_order is required when local search is active")
    return int(local_search_order)


def _local_adaptive_pass2_full_parent_enabled() -> bool:
    """Return whether K=1 adaptive local pass-2 expands all parent samples."""

    if os.environ.get(_LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV, "").strip().lower() in _TRUE_ENV_VALUES:
        return False
    value = os.environ.get(_LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    logger.warning(
        "Ignoring invalid %s=%r; using RELION pruned-parent local pass-2 default",
        _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
        value,
    )
    return False


def _local_adaptive_pass2_rotation_only_enabled() -> bool:
    """Diagnostic: expand significant parent rotations to all parent translations."""

    value = os.environ.get(_LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    logger.warning("Ignoring invalid %s=%r; using default false", _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV, value)
    return False


def _local_adaptive_pass2_denominator_support_mode() -> str | None:
    """Diagnostic mode for broad-denominator local adaptive pass 2."""

    value = os.environ.get(_LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV)
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower().replace("-", "_")
    if normalized in _FALSE_ENV_VALUES or normalized in {"none", "default", "pruned", "pruned_parent"}:
        return None
    if normalized in {"rotation", "rotations", "rotation_only", "significant_rotation_full_translation"}:
        return "rotation_only"
    if normalized in {"full", "full_parent", "all", "all_parent"}:
        return "full_parent"
    logger.warning(
        "Ignoring invalid %s=%r; expected rotation_only or full_parent",
        _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
        value,
    )
    return None


def _expand_significant_samples_to_full_parent_translations(
    significant_sample_indices,
    n_parent_translations: int,
):
    """Expand significant parent rotation ids to every parent translation."""

    n_parent_translations = int(n_parent_translations)
    if n_parent_translations <= 0:
        raise ValueError(f"n_parent_translations must be positive, got {n_parent_translations}")
    expanded = []
    parent_translations = np.arange(n_parent_translations, dtype=np.int64)
    for samples in significant_sample_indices:
        if samples is None:
            expanded.append(None)
            continue
        samples_np = np.asarray(samples, dtype=np.int64).reshape(-1)
        if samples_np.size == 0:
            expanded.append(samples_np)
            continue
        parent_rotations = np.unique(samples_np // n_parent_translations).astype(np.int64, copy=False)
        expanded_samples = (parent_rotations[:, None] * n_parent_translations + parent_translations[None, :]).reshape(
            -1
        )
        expanded.append(expanded_samples.astype(np.int64, copy=False))
    return expanded


def _k1_skip_significance_pruning_enabled() -> bool:
    """Diagnostic switch: evaluate the full K=1 adaptive fine grid."""

    value = os.environ.get(_K1_SKIP_SIGNIFICANCE_PRUNING_ENV)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_ENV_VALUES:
        return False
    if normalized in _TRUE_ENV_VALUES:
        return True
    logger.warning("Ignoring invalid %s=%r; using default false", _K1_SKIP_SIGNIFICANCE_PRUNING_ENV, value)
    return False


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
import os as _os_for_f64

_DENSE_EM_STATIC_KWARGS: dict = {
    "score_with_masked_images": True,
    "half_spectrum_scoring": True,
    "projection_padding_factor": PROJECTION_PADDING_FACTOR,
    "reconstruction_padding_factor": PADDING_FACTOR,
    # Default float32. Set ``RECOVAR_USE_FLOAT64_SCORING=1`` /
    # ``RECOVAR_USE_FLOAT64_PROJECTIONS=1`` to upgrade to double precision.
    # Diagnostic: K=4 100k/256² shows growing per-iter drift (8e-4 at it4→it5
    # rising to 19e-4 at it14→it15 vs RELION), pattern consistent with
    # single-precision accumulating in the K-class M-step + projector at high
    # ``current_size``. Flipping these to True for the dense K-class path
    # should remove that precision floor at ~2× wall cost.
    "use_float64_scoring": bool(
        _os_for_f64.environ.get("RECOVAR_USE_FLOAT64_SCORING", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    ),
    "use_float64_projections": bool(
        _os_for_f64.environ.get("RECOVAR_USE_FLOAT64_PROJECTIONS", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    ),
    # Default to RELION's float32 fine-search diff2/minimum ordering. This
    # diagnostic bypass retains the historical algebraic sparse scorer for
    # controlled full-trajectory A/B comparisons.
    "relion_exact_fine_gaussian": not bool(
        _os_for_f64.environ.get(
            "RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN",
            "0",
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    ),
    "do_gridding_correction": True,
    "square_window": RELION_FOURIER_WINDOW_SQUARE,
    "sparse_pass2": False,
}

# Off by default: reproduces RELION's GPU-accelerated projector/backprojector
# narrowing coordinates to float32 before flooring, unconditionally, even under
# ``ACC_DOUBLE_PRECISION`` (see ``recovar.core.relion_project`` module
# docstring). Set ``RECOVAR_RELION_ACC_DOUBLE_FLOORF_QUIRK=1`` to bit-match
# that GPU-double quirk in the local-search fine-pass projector fallback (it
# only has an effect when the texture path is unavailable, e.g. under
# ``use_float64_scoring``/``use_float64_projections``, since CUDA textures
# cannot hold complex128).
RELION_ACC_DOUBLE_FLOORF_QUIRK = bool(
    _os_for_f64.environ.get("RECOVAR_RELION_ACC_DOUBLE_FLOORF_QUIRK", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _diagnostic_float64_pass2_matches(debug_iteration: int | None) -> bool:
    """Select genuine-f64 pass 2 without perturbing an earlier f32 boundary."""

    raw = _os_for_f64.environ.get("RECOVAR_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "")
    if debug_iteration is None or not raw.strip():
        return False
    try:
        requested = {int(token.strip()) for token in raw.split(",") if token.strip()}
    except ValueError as exc:
        raise ValueError(
            "RECOVAR_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS must be comma-separated integers"
        ) from exc
    return int(debug_iteration) in requested


def _dense_global_scoring_dtype() -> np.dtype:
    """Dtype for the dense/global (``use_local=False``) scoring path's
    float64-sensitive operands: the pass-1 rotation grid built by
    ``_relion_rotation_grid_float32``, and the offset/orientation log-prior
    arrays built by ``make_relion_translation_log_prior`` /
    ``make_relion_direction_log_prior`` and their prior-center helpers.

    None of these have a per-iteration diagnostic override (see
    ``_local_search_precision_flags`` for the local-search analog), so this
    collapses the global float64-scoring/-projections switches directly,
    matching RELION's ``ACC_DOUBLE_PRECISION`` build where the corresponding
    host ``RFLOAT`` values are never narrowed to float before the (no-op)
    ``XFLOAT`` cast.
    """

    if _DENSE_EM_STATIC_KWARGS["use_float64_scoring"] or _DENSE_EM_STATIC_KWARGS["use_float64_projections"]:
        return np.float64
    return np.float32


def _local_search_precision_flags(
    debug_iteration: int | None,
    *,
    pass_index: int,
) -> tuple[bool, bool]:
    """Resolve local-search precision without changing production defaults.

    The global float64 switches apply to both local passes.  The targeted
    diagnostic selector upgrades only pass 2 so it remains useful for
    classifying the fine-score/posterior boundary independently of pass 1.
    """

    if int(pass_index) not in (1, 2):
        raise ValueError(f"local-search pass_index must be 1 or 2, got {pass_index}")
    use_float64_scoring = bool(_DENSE_EM_STATIC_KWARGS["use_float64_scoring"])
    use_float64_projections = bool(_DENSE_EM_STATIC_KWARGS["use_float64_projections"])
    if int(pass_index) == 2 and _diagnostic_float64_pass2_matches(debug_iteration):
        use_float64_scoring = True
        use_float64_projections = True
    return use_float64_scoring, use_float64_projections


def _scatter_dense_k_class_result(
    k_class_result,
    *,
    k: int,
    effective_rotations,
    rot_pmap_for_collapse,
    relion_firstiter_cc_this_iter: bool,
    adaptive_os_local: int,
    noise_stats_per_half_per_class,
    class_assignments,
    class_posterior_per_half,
    class_full_posterior_per_half,
    class_rotation_posterior_per_half,
    best_pose_rotations,
    best_pose_rotation_eulers,
    best_pose_translations,
    require_best_pose_details: bool = True,
    pose_dtype: np.dtype = np.float32,
):
    """Scatter ``run_dense_k_class_em*`` result into per-half output lists.

    Returns the five tuple of E-step outputs ``(ha_k, Ft_y_k, Ft_ctf_k,
    em_stats_k, noise_stats_k)`` used downstream by both the adaptive
    pass-2 and single-pass branches.
    """
    ha_k = np.asarray(k_class_result.pose_assignments, dtype=np.int32)
    noise_stats_per_half_per_class[k] = k_class_result.noise_stats
    class_assignments[k] = np.asarray(k_class_result.class_assignments, dtype=np.int32)
    class_mass_for_priors = getattr(k_class_result, "class_mstep_posterior_sums", None)
    if class_mass_for_priors is None:
        class_mass_for_priors = k_class_result.class_posterior_sums
    class_posterior_per_half[k] = np.asarray(class_mass_for_priors, dtype=np.float64)
    if class_full_posterior_per_half is not None:
        class_full_posterior_per_half[k] = np.asarray(k_class_result.class_posterior_sums, dtype=np.float64)
    # Collapse fine-grid rotation posteriors to coarse via the parent map
    # when iter-1 firstiter_cc routes through the adaptive 2-pass engine
    # with adaptive_oversampling > 0; downstream
    # _combined_class_direction_prior_from_halves expects the coarse-grid
    # shape (n_rot_coarse,).
    n_rot_coarse = int(effective_rotations.shape[0])
    per_class_rot_post_coarse = []
    for stats in k_class_result.per_class_stats:
        rot_post = np.asarray(stats.rotation_posterior_sums, dtype=np.float64)
        if rot_post.shape[0] == n_rot_coarse:
            per_class_rot_post_coarse.append(rot_post)
        elif rot_pmap_for_collapse is not None and adaptive_os_local > 0:
            coarse_post = np.zeros(n_rot_coarse, dtype=np.float64)
            np.add.at(
                coarse_post,
                np.asarray(rot_pmap_for_collapse, dtype=np.int64),
                rot_post,
            )
            per_class_rot_post_coarse.append(coarse_post)
        else:
            raise RuntimeError(
                f"Unexpected K-class rotation_posterior_sums shape {rot_post.shape}; expected ({n_rot_coarse},)"
            )
    class_rotation_posterior_per_half[k] = np.stack(per_class_rot_post_coarse, axis=0)
    if require_best_pose_details:
        if k_class_result.best_pose_rotations is None or k_class_result.best_pose_translations is None:
            raise RuntimeError("Dense K-class path did not return best pose details")
        # See the cross-iteration pose-state comment at the
        # ``new_iter_best_rotations``/``new_iter_best_translations`` call
        # site: RELION's exp_metadata is never narrowed to float, so
        # rotations/translations follow ``pose_dtype``
        # (``_dense_global_scoring_dtype()``); Euler angles stay float32.
        best_rots = np.asarray(k_class_result.best_pose_rotations, dtype=pose_dtype)
        best_pose_rotations[k] = best_rots
        best_pose_rotation_eulers[k] = utils.R_to_relion(best_rots, degrees=True).astype(np.float32)
        best_pose_translations[k] = np.asarray(k_class_result.best_pose_translations, dtype=pose_dtype)
    return (
        ha_k,
        k_class_result.Ft_y,
        k_class_result.Ft_ctf,
        k_class_result.stats,
        k_class_result.aggregate_noise_stats,
    )


def _collapse_fine_pose_assignments_to_coarse(
    pose_assignments,
    *,
    rot_parent_map,
    trans_parent_map,
    n_trans_coarse: int,
    n_trans_fine: int,
):
    pose = np.asarray(pose_assignments, dtype=np.int64)
    rot_idx = pose // int(n_trans_fine)
    trans_idx = pose % int(n_trans_fine)
    coarse_rot = np.asarray(rot_parent_map, dtype=np.int64)[rot_idx]
    coarse_trans = np.asarray(trans_parent_map, dtype=np.int64)[trans_idx]
    return (coarse_rot * int(n_trans_coarse) + coarse_trans).astype(np.int32, copy=False)


def _select_single_class_accumulator(value, *, label: str):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2 or int(shape[0]) != 1:
        raise RuntimeError(f"K=1 adaptive {label} accumulator must have leading class axis 1; got {shape}")
    return value[0]


def _collapse_single_class_stats_to_coarse(stats, *, rot_parent_map, n_rot_coarse: int):
    rot_post = np.asarray(stats.rotation_posterior_sums, dtype=np.float64)
    n_rot_coarse = int(n_rot_coarse)
    if rot_post.shape == (n_rot_coarse,):
        return stats
    if rot_parent_map is None:
        raise RuntimeError(
            f"K=1 adaptive rotation_posterior_sums has shape {rot_post.shape}; expected ({n_rot_coarse},)"
        )
    rot_parent = np.asarray(rot_parent_map, dtype=np.int64)
    if rot_post.shape != rot_parent.shape:
        raise RuntimeError(
            "K=1 adaptive rotation posterior and parent map disagree: "
            f"{rot_post.shape} vs {rot_parent.shape}"
        )
    coarse_post = np.zeros(n_rot_coarse, dtype=np.float64)
    np.add.at(coarse_post, rot_parent, rot_post)
    return make_relion_stats(
        log_evidence_per_image=np.asarray(stats.log_evidence_per_image, dtype=np.float32),
        best_log_score_per_image=np.asarray(stats.best_log_score_per_image, dtype=np.float32),
        max_posterior_per_image=np.asarray(stats.max_posterior_per_image, dtype=np.float32),
        rotation_posterior_sums=coarse_post.astype(np.float32),
    )


@dataclass
class HalfScoreResult:
    """Per-half scoring output shared by the dense scoring branches."""

    # Always populated by every scoring branch.
    ha: np.ndarray
    Ft_y: object
    Ft_ctf: object
    em_stats: object
    noise_stats: object

    # Local-search and explicit best-pose paths.
    best_pose_rotations: np.ndarray | None = None
    best_pose_rotation_eulers: np.ndarray | None = None
    best_pose_translations: np.ndarray | None = None

    # Diagnostics emitted by some branches.
    coarse_ha: np.ndarray | None = None
    pose_rotations: object | None = None
    pose_rotation_eulers: object | None = None
    significant_counts: np.ndarray | None = None
    profile_summary: dict | None = None
    mstep_full_half_axis: int | None = None
    mstep_accumulator_shape: tuple[int, int, int] | None = None


def _record_score_profile(
    profile_history: list,
    score_result,
    *,
    phase: str,
    iteration: int,
    relion_iteration: int,
    half_index: int,
    current_size: int | None,
    healpix_order: int | None,
    k_class_enabled: bool,
) -> None:
    profile = getattr(score_result, "profile_summary", None)
    if not profile:
        return
    row = dict(profile)
    row.update(
        {
            "phase": str(phase),
            "iteration": np.int32(iteration),
            "relion_iteration": np.int32(relion_iteration),
            "half_index": np.int32(half_index),
            "current_size": np.int32(-1 if current_size is None else int(current_size)),
            "healpix_order": np.int32(-1 if healpix_order is None else int(healpix_order)),
            "k_class_enabled": bool(k_class_enabled),
        }
    )
    profile_history.append(row)


@dataclass(frozen=True)
class _AdaptiveDenseBatchSizes:
    """Separate dense batch plans for adaptive pass 1 and pass 2."""

    pass2_image_batch_size: int
    pass2_rotation_block_size: int
    significance_image_batch_size: int
    significance_rotation_block_size: int


def _plan_adaptive_dense_batch_sizes(
    *,
    n_rot: int,
    n_trans: int,
    n_classes: int,
    image_shape,
    cs_for_engine,
    coarse_cs,
    k_class_enabled: bool,
    safe_batch_sizes,
) -> _AdaptiveDenseBatchSizes:
    """Plan adaptive dense microbatches from each pass' Fourier window."""

    pass2_image_batch_size, pass2_rotation_block_size = safe_batch_sizes(
        n_rot,
        n_trans,
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=cs_for_engine,
    )
    if k_class_enabled:
        pass2_image_batch_size = min(
            pass2_image_batch_size,
            _safe_firstiter_cc_image_batch_size(
                n_trans,
                image_shape,
            ),
        )
        pass2_rotation_block_size = min(
            pass2_rotation_block_size,
            _safe_dense_k_class_rotation_block_size(
                n_trans,
                pass2_image_batch_size,
            ),
        )

    significance_image_batch_size, significance_rotation_block_size = safe_batch_sizes(
        n_rot,
        n_trans,
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=coarse_cs,
    )
    return _AdaptiveDenseBatchSizes(
        pass2_image_batch_size=int(pass2_image_batch_size),
        pass2_rotation_block_size=int(pass2_rotation_block_size),
        significance_image_batch_size=int(significance_image_batch_size),
        significance_rotation_block_size=int(significance_rotation_block_size),
    )


def _plan_kclass_adaptive_grid_batch_sizes(
    *,
    coarse_rotations,
    coarse_translations,
    fine_rotations,
    fine_translations,
    n_classes: int,
    image_shape,
    coarse_current_size,
    fine_current_size,
    safe_batch_sizes,
) -> _AdaptiveDenseBatchSizes:
    """Plan K-class adaptive pass-1/pass-2 batches from the actual grids."""

    pass2_image_batch_size, pass2_rotation_block_size = safe_batch_sizes(
        int(np.asarray(fine_rotations).shape[0]),
        int(np.asarray(fine_translations).shape[0]),
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=fine_current_size,
    )
    pass2_image_batch_size = min(
        pass2_image_batch_size,
        _safe_firstiter_cc_image_batch_size(
            int(np.asarray(fine_translations).shape[0]),
            image_shape,
        ),
    )
    if int(n_classes) > 1:
        pass2_rotation_block_size = min(
            pass2_rotation_block_size,
            _safe_dense_k_class_rotation_block_size(
                int(np.asarray(fine_translations).shape[0]),
                pass2_image_batch_size,
            ),
        )

    significance_image_batch_size, significance_rotation_block_size = safe_batch_sizes(
        int(np.asarray(coarse_rotations).shape[0]),
        int(np.asarray(coarse_translations).shape[0]),
        classes=n_classes,
        image_shape_for_batch=image_shape,
        current_size_for_batch=coarse_current_size,
    )
    significance_image_batch_size = min(
        significance_image_batch_size,
        _safe_firstiter_cc_image_batch_size(
            int(np.asarray(coarse_translations).shape[0]),
            image_shape,
        ),
    )
    if int(n_classes) > 1:
        significance_rotation_block_size = min(
            significance_rotation_block_size,
            _safe_dense_k_class_rotation_block_size(
                int(np.asarray(coarse_translations).shape[0]),
                significance_image_batch_size,
            ),
        )

    return _AdaptiveDenseBatchSizes(
        pass2_image_batch_size=int(pass2_image_batch_size),
        pass2_rotation_block_size=int(pass2_rotation_block_size),
        significance_image_batch_size=int(significance_image_batch_size),
        significance_rotation_block_size=int(significance_rotation_block_size),
    )


@dataclass(frozen=True)
class PerHalfOutputs:
    """Mutable per-half E-step outputs grouped behind one owner."""

    hard_assignments: list
    Ft_y: list
    Ft_ctf: list
    coarse_ha: list
    max_posterior: list
    rotation_posterior: list
    class_assignments: list
    class_posterior: list
    class_full_posterior: list
    class_rotation_posterior: list
    noise_stats: list
    noise_stats_per_class: list
    best_pose_rotations: list
    best_pose_rotation_eulers: list
    best_pose_translations: list
    translation_search_bases: list
    pose_rotations: list
    pose_rotation_eulers: list
    mstep_full_half_axis: list
    mstep_accumulator_shape: list

    @classmethod
    def empty(cls) -> "PerHalfOutputs":
        return cls(
            hard_assignments=[None, None],
            Ft_y=[None, None],
            Ft_ctf=[None, None],
            coarse_ha=[None, None],
            max_posterior=[None, None],
            rotation_posterior=[None, None],
            class_assignments=[None, None],
            class_posterior=[None, None],
            class_full_posterior=[None, None],
            class_rotation_posterior=[None, None],
            noise_stats=[None, None],
            noise_stats_per_class=[None, None],
            best_pose_rotations=[None, None],
            best_pose_rotation_eulers=[None, None],
            best_pose_translations=[None, None],
            translation_search_bases=[None, None],
            pose_rotations=[None, None],
            pose_rotation_eulers=[None, None],
            mstep_full_half_axis=[None, None],
            mstep_accumulator_shape=[None, None],
        )

    def update_from(self, idx: int, hs: HalfScoreResult) -> None:
        self.hard_assignments[idx] = hs.ha
        self.Ft_y[idx] = hs.Ft_y
        self.Ft_ctf[idx] = hs.Ft_ctf
        self.noise_stats[idx] = hs.noise_stats
        self.max_posterior[idx] = np.asarray(
            hs.em_stats.max_posterior_per_image,
            dtype=np.float32,
        )
        self.rotation_posterior[idx] = np.asarray(
            hs.em_stats.rotation_posterior_sums,
            dtype=np.float32,
        )
        if hs.best_pose_rotations is not None:
            self.best_pose_rotations[idx] = hs.best_pose_rotations
        if hs.best_pose_rotation_eulers is not None:
            self.best_pose_rotation_eulers[idx] = hs.best_pose_rotation_eulers
        if hs.best_pose_translations is not None:
            self.best_pose_translations[idx] = hs.best_pose_translations
        if hs.coarse_ha is not None:
            self.coarse_ha[idx] = hs.coarse_ha
        if hs.pose_rotations is not None:
            self.pose_rotations[idx] = hs.pose_rotations
        if hs.pose_rotation_eulers is not None:
            self.pose_rotation_eulers[idx] = hs.pose_rotation_eulers
        self.mstep_full_half_axis[idx] = hs.mstep_full_half_axis
        self.mstep_accumulator_shape[idx] = hs.mstep_accumulator_shape


def _host_offload_array(value):
    """Copy a retained accumulator to host memory and release its device buffer."""

    if isinstance(value, np.ndarray):
        return value
    host_value = np.asarray(jax.device_get(value))
    delete = getattr(value, "delete", None)
    if callable(delete):
        try:
            delete()
        except RuntimeError:
            pass
    return host_value


def _maybe_host_offload_half0_local_accumulators(
    *,
    half_index: int,
    use_local: bool,
    k_class_enabled: bool,
    score_result: HalfScoreResult,
) -> HalfScoreResult:
    """Keep finished half-0 local accumulators off GPU while half 1 scores."""

    if int(half_index) != 0 or not use_local or k_class_enabled:
        return score_result
    if score_result.mstep_full_half_axis is None:
        return score_result

    ft_y_nbytes = int(np.size(score_result.Ft_y)) * int(np.dtype(getattr(score_result.Ft_y, "dtype")).itemsize)
    ft_ctf_nbytes = int(np.size(score_result.Ft_ctf)) * int(np.dtype(getattr(score_result.Ft_ctf, "dtype")).itemsize)
    score_result.Ft_y = _host_offload_array(score_result.Ft_y)
    score_result.Ft_ctf = _host_offload_array(score_result.Ft_ctf)
    gc.collect()
    logger.info(
        "Offloaded half-1 local RELION M-step accumulators to host before scoring half-2 "
        "(Ft_y=%.2f GB, Ft_ctf=%.2f GB)",
        ft_y_nbytes / 1e9,
        ft_ctf_nbytes / 1e9,
    )
    return score_result


def _combine_optional_half_accumulators(left, right, *, label: str):
    """Combine half accumulators, treating an empty Class3D half as absent."""

    if left is None:
        if right is None:
            raise RuntimeError(f"{label} accumulators are missing for both halves")
        return right
    if right is None:
        return left
    return left + right


def _resolve_mstep_accumulator_shape(per_half_shapes, default_shape):
    """Return the common M-step accumulator shape for this iteration."""

    present = [tuple(int(v) for v in shape) for shape in per_half_shapes if shape is not None]
    if not present:
        return tuple(int(v) for v in default_shape)
    first = present[0]
    if any(shape != first for shape in present[1:]):
        raise RuntimeError(f"Per-half M-step accumulator shapes disagree: {present}")
    return first


def _resolve_mstep_full_half_axis(per_half_axes, default_axis=-1):
    """Return the common RELION half-complex axis for shell statistics."""

    present = [int(axis) for axis in per_half_axes if axis is not None]
    if not present:
        return int(default_axis)
    first = present[0]
    if any(axis != first for axis in present[1:]):
        raise RuntimeError(f"Per-half M-step full-half axes disagree: {present}")
    return first


def _score_kclass_firstiter_cc_pass2(
    *,
    experiment_dataset,
    mean,
    mean_variance,
    noise_variance_k,
    effective_rotations,
    current_translations,
    base_translations,
    current_healpix_order: int,
    state,
    random_perturbation,
    disc_type,
    class_log_priors,
    image_batch_size: int,
    image_shape_k,
    em_kwargs: dict,
    safe_batch_sizes=None,
    coarse_current_size: int | None = None,
    fine_current_size: int | None = None,
    log_label: str = "",
    update_em_kwargs_image_batch_size: bool = False,
    bpref_device_signature_active: bool = False,
    debug_iteration: int | None = None,
    coarse_rotation_ids=None,
):
    """RELION iter-1 ``--firstiter_cc`` K-class adaptive 2-pass dispatch.

    Builds the firstiter_cc pass-2 (coarse, fine, parent-map) grids and
    invokes ``run_dense_k_class_em_adaptive`` with normalized-CC scoring
    through the global coarse winner subset.  RELION's iter-1
    ``--firstiter_cc`` route is winner-take-all; default-GUI parity depends
    on applying that to the M-step support, not just reporting ``Pmax=1``.

    Shared between the half-set loop's adaptive (``elif use_adaptive``)
    and single-pass (``else``) branches with three controlled
    differences:

    1. ``update_em_kwargs_image_batch_size`` — adaptive site overrides
       em_kwargs["image_batch_size"] with the firstiter clamp; single-pass
       leaves em_kwargs untouched.
    2. ``coarse_current_size`` / ``fine_current_size`` — adaptive site
       passes ``coarse_cs`` / ``cs_for_engine`` to the engine; single-pass
       omits both (engine resolves from ``current_size`` in em_kwargs).
    3. ``log_label`` — "" for adaptive site, "(non-adaptive site) " for
       single-pass.

    Returns ``(k_class_result, rot_pmap, trans_pmap, n_trans_fine, adaptive_os)``.
    """

    adaptive_os_local = int(state.adaptive_oversampling)
    (
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        fine_mstep_rot,
    ) = _build_firstiter_cc_pass2_grids(
        effective_rotations,
        current_translations,
        base_translations,
        int(current_healpix_order),
        adaptive_os_local,
        float(state.translation_step),
        random_perturbation,
        return_mstep_rotations=True,
        **(
            {"coarse_rotation_ids": coarse_rotation_ids}
            if coarse_rotation_ids is not None
            else {}
        ),
    )
    coarse_translation_phase_source = apply_relion_translation_perturbation(
        np.asarray(base_translations, dtype=np.float64),
        float(random_perturbation),
        float(state.translation_step),
    )
    n_classes = int(np.asarray(mean).shape[0]) if np.asarray(mean).ndim >= 2 else 1
    firstiter_significance_image_batch_size = None
    firstiter_significance_rotation_block_size = None
    firstiter_sparse_pass2 = not bool(
        _os_for_f64.environ.get("RECOVAR_K_CLASS_DENSE_PASS2", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if safe_batch_sizes is not None:
        batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
            coarse_rotations=coarse_rot,
            coarse_translations=coarse_trans,
            fine_rotations=fine_rot,
            fine_translations=fine_trans,
            n_classes=n_classes,
            image_shape=image_shape_k,
            coarse_current_size=coarse_current_size if coarse_current_size is not None else em_kwargs.get("current_size"),
            fine_current_size=fine_current_size if fine_current_size is not None else em_kwargs.get("current_size"),
            safe_batch_sizes=safe_batch_sizes,
        )
        if firstiter_sparse_pass2:
            requested_firstiter_image_batch_size = int(em_kwargs.get("image_batch_size", image_batch_size))
            firstiter_image_batch_size = min(
                requested_firstiter_image_batch_size,
                _safe_firstiter_cc_image_batch_size(
                    fine_trans.shape[0],
                    image_shape_k,
                ),
            )
            firstiter_rotation_block_size = min(
                int(em_kwargs.get("rotation_block_size", batch_plan.pass2_rotation_block_size)),
                _safe_dense_k_class_rotation_block_size(
                    fine_trans.shape[0],
                    firstiter_image_batch_size,
                ),
            )
        else:
            firstiter_image_batch_size = batch_plan.pass2_image_batch_size
            firstiter_rotation_block_size = batch_plan.pass2_rotation_block_size
        firstiter_significance_image_batch_size = batch_plan.significance_image_batch_size
        firstiter_significance_rotation_block_size = batch_plan.significance_rotation_block_size
        logger.info(
            "STRICT-PARITY: iter-1 K-class adaptive batch sizing "
            "coarse image_batch_size=%d rotation_block_size=%d; "
            "fine image_batch_size=%d rotation_block_size=%d (%s pass2)",
            firstiter_significance_image_batch_size,
            firstiter_significance_rotation_block_size,
            firstiter_image_batch_size,
            firstiter_rotation_block_size,
            "sparse" if firstiter_sparse_pass2 else "dense",
        )
    else:
        requested_firstiter_image_batch_size = int(em_kwargs.get("image_batch_size", image_batch_size))
        firstiter_image_batch_size = min(
            requested_firstiter_image_batch_size,
            _safe_firstiter_cc_image_batch_size(
                fine_trans.shape[0],
                image_shape_k,
            ),
        )
        firstiter_rotation_block_size = int(em_kwargs.get("rotation_block_size", 5000))
        if firstiter_image_batch_size != requested_firstiter_image_batch_size:
            logger.info(
                "STRICT-PARITY: clamping iter-1 winner-take-all image_batch_size from %d to %d",
                requested_firstiter_image_batch_size,
                firstiter_image_batch_size,
            )
    if update_em_kwargs_image_batch_size:
        em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs = dict(em_kwargs)
    firstiter_em_kwargs["image_batch_size"] = firstiter_image_batch_size
    firstiter_em_kwargs["rotation_block_size"] = firstiter_rotation_block_size
    firstiter_em_kwargs["sparse_pass2"] = firstiter_sparse_pass2
    logger.info(
        "STRICT-PARITY %srouting iter-1 K-class through %s run_dense_k_class_em_adaptive "
        "(oversampling=%d, relion_x_half_mstep=%s, best_coarse_subset=True)",
        log_label,
        "sparse" if firstiter_sparse_pass2 else "dense",
        adaptive_os_local,
        bool(firstiter_em_kwargs.get("mstep_relion_x_half", False)),
    )
    extra: dict = {}
    if coarse_current_size is not None:
        extra["coarse_current_size"] = coarse_current_size
    if fine_current_size is not None:
        extra["fine_current_size"] = fine_current_size
    k_class_result = run_dense_k_class_em_adaptive(
        experiment_dataset,
        mean,
        mean_variance,
        noise_variance_k,
        coarse_rot,
        coarse_trans,
        fine_rot,
        fine_trans,
        rot_pmap,
        trans_pmap,
        disc_type,
        class_log_priors=class_log_priors,
        accumulate_noise=True,
        return_best_pose_details=True,
        firstiter_cc_pass2_only_best_coarse=True,
        skip_significance_pruning=False,
        relion_fine_mstep_prune=True,
        significance_image_batch_size=firstiter_significance_image_batch_size,
        significance_rotation_block_size=firstiter_significance_rotation_block_size,
        coarse_healpix_order=int(current_healpix_order),
        coarse_rotation_ids=coarse_rotation_ids,
        oversampling_order=int(adaptive_os_local),
        fine_mstep_rotations_override=(fine_mstep_rot if firstiter_sparse_pass2 else None),
        bpref_device_signature_active=bpref_device_signature_active,
        debug_iteration=debug_iteration,
        coarse_translation_phase_source=(
            coarse_translation_phase_source if n_classes == 1 else None
        ),
        **extra,
        **firstiter_em_kwargs,
    )
    return k_class_result, rot_pmap, trans_pmap, int(fine_trans.shape[0]), adaptive_os_local


def _relion_projector_half_maps_for_scoring(
    means_k,
    *,
    volume_shape,
    current_size: int | None,
    padding_factor: int,
    n_classes: int,
    real_references=None,
    dump_label: str | None = None,
) -> tuple[np.ndarray, int]:
    """Build RELION ``Projector::data`` slabs from current Fourier references."""

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.initial_model.dense_adapter import reference_to_relion_projector_half_maps

    refs_ft = np.asarray(means_k)
    if int(n_classes) == 1 and refs_ft.ndim == 1:
        refs_ft = refs_ft[None, :]
    if refs_ft.ndim != 2 or int(refs_ft.shape[0]) != int(n_classes):
        raise ValueError(
            "means_k must be a flat reference or a per-class reference array; "
            f"got shape {refs_ft.shape} for n_classes={n_classes}",
        )
    refs_real_override = None
    if real_references is not None:
        refs_real_override = np.asarray(real_references, dtype=np.float64)
        expected_shape = (int(n_classes),) + tuple(int(value) for value in volume_shape)
        if refs_real_override.shape != expected_shape:
            raise ValueError(
                "real_references must have one real-space volume per class; "
                f"got {refs_real_override.shape}, expected {expected_shape}",
            )
    resolved_current_size = int(current_size) if current_size is not None else int(volume_shape[0])
    cache_dir = os.environ.get("RECOVAR_RELION_PROJECTOR_CACHE_DIR", "").strip()
    cache_path = None
    if cache_dir:
        refs_for_hash = np.ascontiguousarray(
            refs_ft if refs_real_override is None else refs_real_override
        )
        hasher = hashlib.sha256()
        hasher.update(b"recovar-relion-projector-cache-v1")
        hasher.update(b"fourier-reference" if refs_real_override is None else b"real-reference")
        hasher.update(str(refs_for_hash.dtype).encode("utf-8"))
        hasher.update(np.asarray(refs_for_hash.shape, dtype=np.int64).tobytes())
        hasher.update(np.asarray(volume_shape, dtype=np.int64).tobytes())
        cache_params = np.asarray(
            [resolved_current_size, int(padding_factor), int(n_classes)],
            dtype=np.int64,
        )
        hasher.update(cache_params.tobytes())
        hasher.update(refs_for_hash.view(np.uint8))
        cache_path = os.path.join(cache_dir, f"projector_{hasher.hexdigest()[:24]}.npz")
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    projector_half = np.asarray(cached["projector_half"])
                    projector_r_max = int(np.asarray(cached["projector_r_max"]))
                    if (
                        int(np.asarray(cached["current_size"])) != resolved_current_size
                        or int(np.asarray(cached["padding_factor"])) != int(padding_factor)
                        or int(np.asarray(cached["n_classes"])) != int(n_classes)
                        or tuple(np.asarray(cached["volume_shape"], dtype=np.int64).tolist()) != tuple(volume_shape)
                    ):
                        raise ValueError("metadata mismatch")
                logger.info("RELION mode: loaded cached Projector::data from %s", cache_path)
                return projector_half, projector_r_max
            except Exception as exc:
                logger.warning("Ignoring unreadable RELION projector cache %s: %s", cache_path, exc)
    if refs_real_override is None:
        refs_real = []
        for class_index in range(int(n_classes)):
            ref_ft = jnp.asarray(refs_ft[class_index]).reshape(volume_shape)
            refs_real.append(np.asarray(ftu.get_idft3(ref_ft)).real)
        refs_real = np.asarray(refs_real, dtype=np.float64)
    else:
        refs_real = refs_real_override
    projector_half, projector_r_max = reference_to_relion_projector_half_maps(
        refs_real,
        current_size=resolved_current_size,
        padding_factor=int(padding_factor),
    )
    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(os.path.join(cache_dir, "SAFE_TO_DELETE"), "a", encoding="utf-8"):
                pass
            tmp_path = f"{cache_path}.{os.getpid()}.tmp.npz"
            np.savez(
                tmp_path,
                projector_half=np.asarray(projector_half),
                projector_r_max=np.int64(projector_r_max),
                current_size=np.int64(resolved_current_size),
                padding_factor=np.int64(padding_factor),
                volume_shape=np.asarray(volume_shape, dtype=np.int64),
                n_classes=np.int64(n_classes),
            )
            os.replace(tmp_path, cache_path)
            logger.info("RELION mode: saved Projector::data cache to %s", cache_path)
        except Exception as exc:
            logger.warning("Could not write RELION projector cache %s: %s", cache_path, exc)
    dump_dir = os.environ.get("RECOVAR_RELION_PROJECTOR_DUMP_DIR")
    if dump_dir:
        label = dump_label or "projector"
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
        os.makedirs(dump_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(dump_dir, f"{safe_label}_relion_projector_half.npz"),
            projector_half=np.asarray(projector_half),
            projector_r_max=np.int64(projector_r_max),
            current_size=np.int64(resolved_current_size),
            padding_factor=np.int64(padding_factor),
            volume_shape=np.asarray(volume_shape, dtype=np.int64),
            n_classes=np.int64(n_classes),
        )
    return projector_half, projector_r_max


def _validate_captured_relion_projector_for_iteration(
    replay_state: RelionProjectorReplayState,
    *,
    current_size: int | None,
    volume_shape,
    padding_factor: int,
    n_classes: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Bind a captured projector state to one exact live replay geometry."""

    resolved_current_size = int(current_size) if current_size is not None else int(volume_shape[0])
    expected_volume_shape = tuple(int(value) for value in volume_shape)
    mismatches = []
    if replay_state.current_size != resolved_current_size:
        mismatches.append(
            f"current_size captured={replay_state.current_size} replay={resolved_current_size}"
        )
    if replay_state.padding_factor != int(padding_factor):
        mismatches.append(
            f"padding_factor captured={replay_state.padding_factor} replay={int(padding_factor)}"
        )
    if replay_state.volume_shape != expected_volume_shape:
        mismatches.append(
            f"volume_shape captured={replay_state.volume_shape} replay={expected_volume_shape}"
        )
    if replay_state.n_classes != int(n_classes):
        mismatches.append(
            f"n_classes captured={replay_state.n_classes} replay={int(n_classes)}"
        )
    if mismatches:
        raise ValueError(
            "captured RELION Projector::data does not match the live replay boundary: "
            + "; ".join(mismatches)
        )
    return (
        list(replay_state.projector_half_by_half),
        [int(value) for value in replay_state.projector_r_max_by_half],
    )


def _score_half_dense(
    *,
    k: int,
    experiment_dataset,
    means_k,
    mean_variance,
    noise_variance_k,
    effective_rotations,
    current_translations,
    base_translations,
    current_healpix_order: int,
    state,
    random_perturbation,
    disc_type,
    image_batch_size: int,
    rotation_log_prior_k,
    class_rotation_log_prior_k,
    translation_log_prior,
    translation_search_base,
    trans_prior_center_for_engine,
    image_corrections_k,
    scale_corrections_k,
    firstiter_score_mode_this_iter: str,
    firstiter_winner_take_all_this_iter: bool,
    cs_for_engine,
    class_log_priors,
    k_class_enabled: bool,
    relion_firstiter_cc_this_iter: bool,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    safe_batch_sizes,
    max_significants,
    # K-class scatter targets (mutated in place):
    noise_stats_per_half_per_class,
    class_assignments,
    class_posterior_per_half,
    class_full_posterior_per_half,
    class_rotation_posterior_per_half,
    best_pose_rotations,
    best_pose_rotation_eulers,
    best_pose_translations,
    group_ids_k=None,
    group_count_k=None,
    scale_correction_data_vs_prior=None,
    # Mode-specific overrides (adaptive site sets these; single-pass uses
    # the defaults):
    k_class_image_batch_size_override: int | None = None,
    k_class_rotation_block_size_override: int | None = None,
    significance_image_batch_size_override: int | None = None,
    significance_rotation_block_size_override: int | None = None,
    firstiter_coarse_current_size: int | None = None,
    firstiter_fine_current_size: int | None = None,
    firstiter_log_label: str = "(non-adaptive site) ",
    firstiter_updates_em_kwargs_ibs: bool = False,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
    return_best_pose_details: bool = True,
    bpref_device_signature_active: bool = False,
    debug_iteration: int | None = None,
    coarse_rotation_ids=None,
    preserve_bpref_particle_order: bool = False,
) -> HalfScoreResult:
    """Dense (non-local-search) E+M scoring for one half-set.

    Used by both the single-pass (``else``) and adaptive-2-pass
    (``elif use_adaptive``) branches of the half-set loop. The two modes
    differ in four places, all controlled by trailing parameters:

    1. ``k_class_image_batch_size_override`` /
       ``k_class_rotation_block_size_override`` — adaptive overrides
       em_kwargs ibs/rbs to K-class values before firstiter_cc check.
    2. ``significance_*_override`` — adaptive pass 1 may use a smaller
       Fourier window than pass 2, so it needs its own memory-sized batches.
    3. ``firstiter_coarse_current_size`` / ``firstiter_fine_current_size``
       — adaptive passes ``coarse_cs`` / ``cs_for_engine`` through to the
       adaptive 2-pass engine; single-pass omits them.
    4. ``firstiter_log_label`` — single-pass uses
       ``"(non-adaptive site) "`` for the routing log message.
    5. ``firstiter_updates_em_kwargs_ibs`` — adaptive overrides
       em_kwargs["image_batch_size"] with the firstiter clamp; single-pass
       leaves em_kwargs untouched.

    Mutates the per-half K-class scatter lists when ``k_class_enabled`` is
    True (via ``_scatter_dense_k_class_result``); the K=1 per-half lists
    are still owned by the caller. ``safe_batch_sizes`` is the
    closure-bound batch sizer from ``_run_relion_iteration_loop``.
    """

    safe_ibs, safe_rbs = safe_batch_sizes(
        effective_rotations.shape[0],
        current_translations.shape[0],
        current_size_for_batch=cs_for_engine,
    )
    em_kwargs = {
        **_DENSE_EM_STATIC_KWARGS,
        "image_batch_size": safe_ibs,
        "rotation_block_size": safe_rbs,
        "current_size": cs_for_engine,
        "rotation_log_prior": rotation_log_prior_k,
        "translation_log_prior": translation_log_prior,
        "image_corrections": image_corrections_k,
        "scale_corrections": scale_corrections_k,
        "group_ids": group_ids_k,
        "scale_correction_group_count": group_count_k,
        "scale_correction_data_vs_prior": scale_correction_data_vs_prior,
        "image_pre_shifts": translation_search_base,
        "translation_prior_centers": trans_prior_center_for_engine,
        "relion_firstiter_score_mode": firstiter_score_mode_this_iter,
        "relion_firstiter_winner_take_all": firstiter_winner_take_all_this_iter,
    }
    if preserve_bpref_particle_order and k_class_enabled:
        raise ValueError("RELION BPref particle-order preservation is K=1-only")
    if preserve_bpref_particle_order:
        em_kwargs["preserve_bpref_particle_order"] = True
    diagnostic_float64_pass2 = _diagnostic_float64_pass2_matches(debug_iteration)
    if diagnostic_float64_pass2:
        logger.info(
            "Diagnostic genuine-float64 adaptive pass 2 at iteration %d; pass 1 and prior boundaries remain f32",
            int(debug_iteration),
        )
    if k_class_image_batch_size_override is not None:
        em_kwargs["image_batch_size"] = k_class_image_batch_size_override
    if k_class_rotation_block_size_override is not None:
        em_kwargs["rotation_block_size"] = k_class_rotation_block_size_override
    if class_rotation_log_prior_k is not None:
        em_kwargs["rotation_log_prior"] = None
        em_kwargs["class_rotation_log_prior"] = class_rotation_log_prior_k
    if relion_projector_half is not None:
        em_kwargs["relion_projector_half"] = relion_projector_half
        em_kwargs["relion_projector_r_max"] = relion_projector_r_max
    logger.info(
        "Dense half-set projector handoff: supplied_ppref=%s state_oversampling=%d",
        relion_projector_half is not None,
        int(state.adaptive_oversampling),
    )
    if k_class_enabled:
        if disable_adjoint_y or disable_adjoint_ctf:
            raise NotImplementedError("K-class refine does not support adjoint ablation flags")
        # K-class should use RELION's x-half BackProjector accumulator layout,
        # matching the K=1 parity path.  The old full-volume and native
        # half-volume paths remain available as diagnostics via
        # RECOVAR_K_CLASS_RELION_X_HALF_MSTEP=0 together with the legacy
        # RECOVAR_K_CLASS_FULL_VOLUME_MSTEP / RECOVAR_K_CLASS_HALF_VOLUME_MSTEP
        # switches.
        k_class_relion_x_half_mstep = _k_class_relion_x_half_mstep_enabled()
        em_kwargs["mstep_relion_x_half"] = bool(k_class_relion_x_half_mstep)
        em_kwargs["relion_half_volume_mstep"] = (
            False if k_class_relion_x_half_mstep else _k_class_relion_half_volume_mstep_enabled()
        )
        k_class_mstep_full_half_axis_this_score = None
        rot_pmap_for_collapse = None
        trans_pmap_for_collapse = None
        n_trans_fine_for_collapse = None
        fine_rotations_for_pose = None
        adaptive_os_local = 0
        # STRICT-PARITY: at iter 1 with --firstiter_cc, route through the
        # adaptive 2-pass engine with normalized-CC scoring. Pass 2 retains the
        # oversampled children of the single best coarse class/pose, matching
        # RELION's firstiter-CC binarized coarse support.
        if relion_firstiter_cc_this_iter:
            (
                k_class_result,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                n_trans_fine_for_collapse,
                adaptive_os_local,
            ) = _score_kclass_firstiter_cc_pass2(
                experiment_dataset=experiment_dataset,
                mean=means_k,
                mean_variance=mean_variance,
                noise_variance_k=noise_variance_k,
                effective_rotations=effective_rotations,
                current_translations=current_translations,
                base_translations=base_translations,
                current_healpix_order=current_healpix_order,
                state=state,
                random_perturbation=random_perturbation,
                disc_type=disc_type,
                class_log_priors=class_log_priors,
                image_batch_size=image_batch_size,
                image_shape_k=experiment_dataset.image_shape,
                em_kwargs=em_kwargs,
                safe_batch_sizes=safe_batch_sizes,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                log_label=firstiter_log_label,
                update_em_kwargs_image_batch_size=firstiter_updates_em_kwargs_ibs,
                bpref_device_signature_active=bpref_device_signature_active,
                debug_iteration=debug_iteration,
                coarse_rotation_ids=coarse_rotation_ids,
            )
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        elif firstiter_coarse_current_size is not None and int(state.adaptive_oversampling) > 0:
            adaptive_os_local = int(state.adaptive_oversampling)
            (
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                fine_mstep_rot,
            ) = _build_firstiter_cc_pass2_grids(
                effective_rotations,
                current_translations,
                base_translations,
                int(current_healpix_order),
                adaptive_os_local,
                float(state.translation_step),
                random_perturbation,
                return_mstep_rotations=True,
                **(
                    {"coarse_rotation_ids": coarse_rotation_ids}
                    if coarse_rotation_ids is not None
                    else {}
                ),
            )
            coarse_translation_phase_source = apply_relion_translation_perturbation(
                np.asarray(base_translations, dtype=np.float64),
                float(random_perturbation),
                float(state.translation_step),
            )
            n_trans_fine_for_collapse = int(fine_trans.shape[0])
            adaptive_em_kwargs = dict(em_kwargs)
            n_classes_local = int(np.asarray(means_k).shape[0]) if np.asarray(means_k).ndim >= 2 else 1
            grid_batch_plan = _plan_kclass_adaptive_grid_batch_sizes(
                coarse_rotations=coarse_rot,
                coarse_translations=coarse_trans,
                fine_rotations=fine_rot,
                fine_translations=fine_trans,
                n_classes=n_classes_local,
                image_shape=experiment_dataset.image_shape,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                safe_batch_sizes=safe_batch_sizes,
            )
            adaptive_em_kwargs["image_batch_size"] = grid_batch_plan.pass2_image_batch_size
            adaptive_em_kwargs["rotation_block_size"] = grid_batch_plan.pass2_rotation_block_size
            significance_image_batch_size_override = grid_batch_plan.significance_image_batch_size
            significance_rotation_block_size_override = grid_batch_plan.significance_rotation_block_size
            logger.info(
                "RELION adaptive K-class grid batch sizing: "
                "coarse image_batch_size=%d rotation_block_size=%d; "
                "fine image_batch_size=%d rotation_block_size=%d",
                significance_image_batch_size_override,
                significance_rotation_block_size_override,
                adaptive_em_kwargs["image_batch_size"],
                adaptive_em_kwargs["rotation_block_size"],
            )
            # ``RECOVAR_K_CLASS_DENSE_PASS2=1`` swaps K-class adaptive
            # oversampling from sparse-bucketed pass-2 to dense pass-2.
            # Diagnostic: tests whether the sparse-bucket reduction order
            # carries a structural bias vs the dense in-place reduction.
            kclass_sparse_pass2 = not bool(
                _os_for_f64.environ.get("RECOVAR_K_CLASS_DENSE_PASS2", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            adaptive_em_kwargs["sparse_pass2"] = kclass_sparse_pass2
            logger.info(
                "RELION adaptive K-class routing through run_dense_k_class_em_adaptive "
                "(oversampling=%d, pass2_backend=%s, fine_mstep_prune=%s)",
                adaptive_os_local,
                "sparse" if kclass_sparse_pass2 else "dense",
                bool(kclass_sparse_pass2),
            )
            k_class_result = run_dense_k_class_em_adaptive(
                experiment_dataset,
                means_k,
                mean_variance,
                noise_variance_k,
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=True,
                adaptive_fraction=RELION_ADAPTIVE_FRACTION,
                max_significants=-1 if max_significants is None else int(max_significants),
                relion_fine_mstep_prune=bool(kclass_sparse_pass2),
                significance_image_batch_size=significance_image_batch_size_override,
                significance_rotation_block_size=significance_rotation_block_size_override,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                coarse_healpix_order=int(current_healpix_order),
                oversampling_order=int(adaptive_os_local),
                fine_mstep_rotations_override=(fine_mstep_rot if kclass_sparse_pass2 else None),
                return_best_pose_details=return_best_pose_details,
                bpref_device_signature_active=bpref_device_signature_active,
                debug_iteration=debug_iteration,
                **adaptive_em_kwargs,
            )
            k_class_mstep_full_half_axis_this_score = k_class_result.mstep_full_half_axis
        else:
            dense_em_kwargs = dict(em_kwargs)
            # The direct dense K-class wrapper delegates to run_em, which does
            # not implement RELION x-half accumulators. Keep that branch on its
            # historical layout and avoid tagging its full-volume output as
            # x-half-expanded.
            dense_em_kwargs.pop("mstep_relion_x_half", None)
            dense_em_kwargs.pop("group_ids", None)
            dense_em_kwargs.pop("scale_correction_group_count", None)
            dense_em_kwargs.pop("scale_correction_data_vs_prior", None)
            # Exact fine-Gaussian scoring is implemented only by sparse pass 2.
            # A non-adaptive dense iteration has no fine pass to select.
            dense_em_kwargs.pop("relion_exact_fine_gaussian", None)
            k_class_result = run_dense_k_class_em(
                experiment_dataset,
                means_k,
                mean_variance,
                noise_variance_k,
                effective_rotations,
                current_translations,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=True,
                return_best_pose_details=return_best_pose_details,
                **dense_em_kwargs,
            )
            k_class_mstep_full_half_axis_this_score = None
        ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = _scatter_dense_k_class_result(
            k_class_result,
            k=k,
            effective_rotations=effective_rotations,
            rot_pmap_for_collapse=rot_pmap_for_collapse,
            relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
            adaptive_os_local=adaptive_os_local,
            noise_stats_per_half_per_class=noise_stats_per_half_per_class,
            class_assignments=class_assignments,
            class_posterior_per_half=class_posterior_per_half,
            class_full_posterior_per_half=class_full_posterior_per_half,
            class_rotation_posterior_per_half=class_rotation_posterior_per_half,
            best_pose_rotations=best_pose_rotations,
            best_pose_rotation_eulers=best_pose_rotation_eulers,
            best_pose_translations=best_pose_translations,
            require_best_pose_details=return_best_pose_details,
            pose_dtype=_dense_global_scoring_dtype(),
        )
        coarse_ha_k = None
        if trans_pmap_for_collapse is not None and n_trans_fine_for_collapse is not None:
            coarse_ha_k = _collapse_fine_pose_assignments_to_coarse(
                ha_k,
                rot_parent_map=rot_pmap_for_collapse,
                trans_parent_map=trans_pmap_for_collapse,
                n_trans_coarse=current_translations.shape[0],
                n_trans_fine=n_trans_fine_for_collapse,
            )
        return HalfScoreResult(
            ha=ha_k,
            Ft_y=Ft_y_k,
            Ft_ctf=Ft_ctf_k,
            em_stats=em_stats_k,
            noise_stats=noise_stats_k,
            coarse_ha=coarse_ha_k,
            significant_counts=(
                None
                if k_class_result.significant_counts is None
                else np.asarray(k_class_result.significant_counts, dtype=np.int32)
            ),
            profile_summary=k_class_result.profile_summary,
            mstep_full_half_axis=k_class_mstep_full_half_axis_this_score,
            mstep_accumulator_shape=getattr(k_class_result, "mstep_accumulator_shape", None),
        )

    if int(state.adaptive_oversampling) > 0:
        if disable_adjoint_y or disable_adjoint_ctf:
            raise NotImplementedError("K=1 adaptive oversampling does not support adjoint ablation flags")
        adaptive_os_local = int(state.adaptive_oversampling)
        k1_relion_x_half_mstep = _k1_relion_x_half_mstep_enabled()
        means_single = jnp.asarray(means_k)[None, :]
        rot_pmap_for_collapse = None
        trans_pmap_for_collapse = None
        n_trans_fine_for_collapse = None
        fine_rotations_for_pose = None
        if relion_firstiter_cc_this_iter:
            (
                k1_adaptive_result,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                n_trans_fine_for_collapse,
                adaptive_os_local,
            ) = _score_kclass_firstiter_cc_pass2(
                experiment_dataset=experiment_dataset,
                mean=means_single,
                mean_variance=mean_variance,
                noise_variance_k=noise_variance_k,
                effective_rotations=effective_rotations,
                current_translations=current_translations,
                base_translations=base_translations,
                current_healpix_order=current_healpix_order,
                state=state,
                random_perturbation=random_perturbation,
                disc_type=disc_type,
                class_log_priors=class_log_priors,
                image_batch_size=image_batch_size,
                image_shape_k=experiment_dataset.image_shape,
                em_kwargs=(
                    {**em_kwargs, "mstep_relion_x_half": True}
                    if k1_relion_x_half_mstep
                    else em_kwargs
                ),
                safe_batch_sizes=safe_batch_sizes,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                log_label="K=1 ",
                update_em_kwargs_image_batch_size=firstiter_updates_em_kwargs_ibs,
                bpref_device_signature_active=bpref_device_signature_active,
                debug_iteration=debug_iteration,
            )
        else:
            (
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                fine_mstep_rot,
            ) = _build_firstiter_cc_pass2_grids(
                effective_rotations,
                current_translations,
                base_translations,
                int(current_healpix_order),
                adaptive_os_local,
                float(state.translation_step),
                random_perturbation,
                return_mstep_rotations=True,
                **(
                    {"coarse_rotation_ids": coarse_rotation_ids}
                    if coarse_rotation_ids is not None
                    else {}
                ),
            )
            coarse_translation_phase_source = apply_relion_translation_perturbation(
                np.asarray(base_translations, dtype=np.float64),
                float(random_perturbation),
                float(state.translation_step),
            )
            n_trans_fine_for_collapse = int(fine_trans.shape[0])
            fine_rotations_for_pose = fine_rot
            adaptive_em_kwargs = dict(em_kwargs)
            k1_sparse_pass2 = not bool(
                _os_for_f64.environ.get("RECOVAR_K1_DENSE_PASS2", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            k1_skip_significance_pruning = _k1_skip_significance_pruning_enabled()
            adaptive_em_kwargs["sparse_pass2"] = k1_sparse_pass2
            if group_ids_k is not None:
                adaptive_em_kwargs["group_ids"] = group_ids_k
            if k1_relion_x_half_mstep:
                adaptive_em_kwargs["mstep_relion_x_half"] = True
            logger.info(
                "RELION adaptive K=1 routing through run_dense_k_class_em_adaptive "
                "(oversampling=%d, pass2_backend=%s, skip_significance_pruning=%s, "
                "fine_mstep_prune=%s, relion_x_half_mstep=%s, supplied_ppref=%s, "
                "engine_ppref=%s)",
                adaptive_os_local,
                "sparse" if k1_sparse_pass2 else "dense",
                bool(k1_skip_significance_pruning),
                bool(k1_sparse_pass2),
                bool(k1_relion_x_half_mstep),
                relion_projector_half is not None,
                adaptive_em_kwargs.get("relion_projector_half") is not None,
            )
            k1_adaptive_result = run_dense_k_class_em_adaptive(
                experiment_dataset,
                means_single,
                mean_variance,
                noise_variance_k,
                coarse_rot,
                coarse_trans,
                fine_rot,
                fine_trans,
                rot_pmap_for_collapse,
                trans_pmap_for_collapse,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=True,
                adaptive_fraction=RELION_ADAPTIVE_FRACTION,
                max_significants=-1 if max_significants is None else int(max_significants),
                skip_significance_pruning=k1_skip_significance_pruning,
                relion_fine_mstep_prune=bool(k1_sparse_pass2),
                significance_image_batch_size=significance_image_batch_size_override,
                significance_rotation_block_size=significance_rotation_block_size_override,
                coarse_current_size=firstiter_coarse_current_size,
                fine_current_size=firstiter_fine_current_size,
                coarse_healpix_order=int(current_healpix_order),
                oversampling_order=int(adaptive_os_local),
                fine_mstep_rotations_override=(fine_mstep_rot if k1_sparse_pass2 else None),
                return_best_pose_details=return_best_pose_details,
                bpref_device_signature_active=bpref_device_signature_active,
                debug_iteration=debug_iteration,
                pass2_use_float64_scoring=True if diagnostic_float64_pass2 else None,
                pass2_use_float64_projections=True if diagnostic_float64_pass2 else None,
                coarse_translation_phase_source=coarse_translation_phase_source,
                **adaptive_em_kwargs,
            )
        ha_k = np.asarray(k1_adaptive_result.pose_assignments, dtype=np.int32)
        Ft_y_k = _select_single_class_accumulator(k1_adaptive_result.Ft_y, label="Ft_y")
        Ft_ctf_k = _select_single_class_accumulator(k1_adaptive_result.Ft_ctf, label="Ft_ctf")
        em_stats_k = _collapse_single_class_stats_to_coarse(
            k1_adaptive_result.stats,
            rot_parent_map=rot_pmap_for_collapse,
            n_rot_coarse=effective_rotations.shape[0],
        )
        noise_stats_k = k1_adaptive_result.aggregate_noise_stats
        if noise_stats_k is None and k1_adaptive_result.noise_stats is not None:
            noise_stats_k = k1_adaptive_result.noise_stats[0]
        if noise_stats_k is None:
            raise RuntimeError("K=1 adaptive path did not return noise statistics")
        coarse_ha_k = None
        if trans_pmap_for_collapse is not None and n_trans_fine_for_collapse is not None:
            coarse_ha_k = _collapse_fine_pose_assignments_to_coarse(
                ha_k,
                rot_parent_map=rot_pmap_for_collapse,
                trans_parent_map=trans_pmap_for_collapse,
                n_trans_coarse=current_translations.shape[0],
                n_trans_fine=n_trans_fine_for_collapse,
            )
        if return_best_pose_details:
            if (
                k1_adaptive_result.best_pose_rotations is None
                or k1_adaptive_result.best_pose_translations is None
            ):
                raise RuntimeError("K=1 adaptive path did not return best pose details")
            best_rots = np.asarray(k1_adaptive_result.best_pose_rotations, dtype=np.float32)
            best_pose_rotations[k] = best_rots
            best_pose_rotation_eulers[k] = utils.R_to_relion(best_rots, degrees=True).astype(np.float32)
            best_pose_translations[k] = np.asarray(k1_adaptive_result.best_pose_translations, dtype=np.float32)
        if fine_rotations_for_pose is None and rot_pmap_for_collapse is not None:
            fine_rotations_for_pose = _build_firstiter_cc_pass2_grids(
                effective_rotations,
                current_translations,
                base_translations,
                int(current_healpix_order),
                adaptive_os_local,
                float(state.translation_step),
                random_perturbation,
                **(
                    {"coarse_rotation_ids": coarse_rotation_ids}
                    if coarse_rotation_ids is not None
                    else {}
                ),
            )[2]
        fine_rotation_eulers_for_pose = None
        if fine_rotations_for_pose is not None and _parity_dump.is_active():
            fine_rotation_eulers_for_pose = utils.R_to_relion(
                np.asarray(fine_rotations_for_pose, dtype=np.float32),
                degrees=True,
            ).astype(np.float32)
        return HalfScoreResult(
            ha=ha_k,
            Ft_y=Ft_y_k,
            Ft_ctf=Ft_ctf_k,
            em_stats=em_stats_k,
            noise_stats=noise_stats_k,
            best_pose_rotations=best_pose_rotations[k],
            best_pose_rotation_eulers=best_pose_rotation_eulers[k],
            best_pose_translations=best_pose_translations[k],
            coarse_ha=coarse_ha_k,
            pose_rotations=fine_rotations_for_pose,
            pose_rotation_eulers=fine_rotation_eulers_for_pose,
            significant_counts=(
                None
                if k1_adaptive_result.significant_counts is None
                else np.asarray(k1_adaptive_result.significant_counts, dtype=np.int32)
            ),
            profile_summary=k1_adaptive_result.profile_summary,
            mstep_full_half_axis=k1_adaptive_result.mstep_full_half_axis,
            mstep_accumulator_shape=getattr(k1_adaptive_result, "mstep_accumulator_shape", None),
        )

    if group_ids_k is not None:
        raise RuntimeError(
            "RELION native group-scale correction requires the sparse/adaptive or local "
            "M-step; direct dense K=1 does not accumulate group XA/AA statistics"
        )
    direct_em_kwargs = dict(em_kwargs)
    direct_em_kwargs.pop("group_ids", None)
    direct_em_kwargs.pop("scale_correction_group_count", None)
    direct_em_kwargs.pop("scale_correction_data_vs_prior", None)
    # Exact fine-Gaussian scoring is implemented only by sparse pass 2.  This
    # branch is the single dense pass used when adaptive oversampling is off.
    direct_em_kwargs.pop("relion_exact_fine_gaussian", None)
    _, ha_k, Ft_y_k, Ft_ctf_k, em_stats_k, noise_stats_k = run_em(
        experiment_dataset,
        means_k,
        mean_variance,
        noise_variance_k,
        effective_rotations,
        current_translations,
        disc_type,
        return_stats=True,
        accumulate_noise=True,
        disable_adjoint_y=disable_adjoint_y,
        disable_adjoint_ctf=disable_adjoint_ctf,
        **direct_em_kwargs,
    )
    return HalfScoreResult(
        ha=ha_k,
        Ft_y=Ft_y_k,
        Ft_ctf=Ft_ctf_k,
        em_stats=em_stats_k,
        noise_stats=noise_stats_k,
        mstep_accumulator_shape=None,
    )


def _score_half_dense_in_bpref_scope(
    *,
    bpref_device_signature_active: bool,
    **kwargs,
) -> HalfScoreResult:
    """Keep all authoritative dense-half work outside diagnostic CUDA scope."""

    with _cuda_backproject_diagnostics.bpref_device_signature_scope(False):
        return _score_half_dense(
            bpref_device_signature_active=bpref_device_signature_active,
            **kwargs,
        )


def _local_translation_prior_reference_translations(
    *,
    current_translations,
    base_translations,
    replay_prior_translations,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, str, bool]:
    """Choose a local-search translation-prior grid compatible with scoring."""

    current = np.asarray(current_translations, dtype=dtype)
    base = np.asarray(base_translations, dtype=dtype)
    if replay_prior_translations is not None:
        candidate = np.asarray(replay_prior_translations, dtype=dtype)
        source = "replay"
    else:
        candidate = base
        source = "base"

    if candidate.shape == current.shape:
        return candidate, source, False
    if base.shape == current.shape:
        return base, "base", True
    return current, "current", True


def _relion_coarse_significant_counts(significant_sample_indices):
    """Count explicit retained pass-1 samples using RELION metadata semantics."""

    if any(indices is None for indices in significant_sample_indices):
        return None
    return np.asarray(
        [np.asarray(indices).size for indices in significant_sample_indices],
        dtype=np.int32,
    )


def _score_half_local(
    *,
    k: int,
    experiment_dataset,
    means_k,
    mean_variance,
    noise_variance_k,
    previous_best_rotation_eulers_k,
    local_search_rotations,
    local_search_rotation_eulers,
    local_search_order: int,
    sigma_rot,
    sigma_psi,
    current_translations,
    base_translations,
    trans_prior_center,
    trans_prior_center_for_engine,
    current_sigma_offset_angstrom: float,
    current_translation_range: float,
    disc_type,
    cs_for_engine,
    local_pass1_current_size,
    image_corrections_k,
    scale_corrections_k,
    translation_search_base,
    disable_adjoint_y: bool,
    disable_adjoint_ctf: bool,
    max_significants,
    state,
    iteration: int,
    debug_iteration: int | None = None,
    save_intermediates_dir,
    local_search_random_perturbation,
    local_search_angular_sampling_deg,
    local_parent_oversampling_order: int,
    local_search_translation_prior_mode: str,
    replay_prior_translations,
    rotation_log_prior_k,
    class_log_priors,
    k_class_enabled: bool,
    collect_local_search_profile: bool,
    diagnostic_score_only: bool,
    safe_batch_sizes,
    # Scatter targets (mutated in place):
    class_assignments,
    class_posterior_per_half,
    class_full_posterior_per_half,
    best_pose_rotations,
    best_pose_rotation_eulers,
    best_pose_translations,
    local_profile_history,
    local_search_mstep_rotations=None,
    group_ids_k=None,
    group_count_k=None,
    scale_correction_data_vs_prior=None,
    relion_projector_half=None,
    relion_projector_r_max: int | None = None,
) -> HalfScoreResult:
    """Local-search E+M scoring for one half-set.

    Sizes the per-chunk M-step batches against the cone-restricted
    rotation count (not the full HEALPix grid) so chunk_size doesn't
    collapse at high HEALPix orders. Routes through
    ``_run_local_search_iteration`` which itself handles K-class /
    K=1 internally via ``return_class_details=k_class_enabled``.

    Caller handles ``noise_stats_per_half[k]``, ``pose_rotations[k] = None``,
    and ``coarse_ha[k] = ha_k`` from the returned ``HalfScoreResult``.
    """

    # RELION's convertAllSquaredDifferencesToWeights uses mymodel.pdf_direction
    # only when orientational_prior_mode == NOPRIOR. Local searches run through
    # PRIOR_ROTTILT_PSI and score the local direction/psi priors in the
    # hypothesis layout, so adding the learned global direction prior here
    # biases both support selection and final weights.
    relion_local_rotation_log_prior_k = None
    if diagnostic_score_only and k_class_enabled:
        raise NotImplementedError("score-only local-search diagnostics are currently K=1-only")

    # For local search the per-chunk M-step only sees the cone-restricted
    # rotation set (typically a few thousand rotations per image with high
    # overlap across the chunk) rather than the full ~10⁶-rotation grid at
    # healpix order 5+. Estimate per-image cone size from
    #     fraction = (sigma_cutoff * sigma_rot / pi)^2
    # (spherical cap area as a fraction of full SO(3) volume; good to
    # within ~30% for reasonable cones). Use that for an effective rotation
    # count equal to ``chunk_size * cone_size`` with a 2x safety factor.
    cone_radius = 3.0 * float(sigma_rot)  # sigma_cutoff=3.0
    cone_fraction = max(
        (cone_radius / float(np.pi)) ** 2,
        1.0 / float(rotation_grid_size(local_search_order)),
    )
    est_cone_rots = int(np.ceil(rotation_grid_size(local_search_order) * cone_fraction))
    eff_n_rot = max(64, 2 * est_cone_rots)
    local_n_trans = int(current_translations.shape[0])
    if int(local_parent_oversampling_order) > 0:
        local_n_trans *= int(4 ** int(local_parent_oversampling_order))
    local_debug_iteration = iteration + 1 if debug_iteration is None else int(debug_iteration)
    parent_use_float64_scoring, parent_use_float64_projections = _local_search_precision_flags(
        local_debug_iteration,
        pass_index=1,
    )
    fine_use_float64_scoring, fine_use_float64_projections = _local_search_precision_flags(
        local_debug_iteration,
        pass_index=2,
    )
    # Adaptive pass-2 (fine, oversampled) hypothesis layout precision; see
    # ``parent_local_layout_dtype`` below for the matching pass-1 value.
    fine_local_layout_dtype = (
        np.float64 if (fine_use_float64_scoring or fine_use_float64_projections) else np.float32
    )
    if fine_use_float64_scoring or fine_use_float64_projections:
        logger.info(
            "Local-search precision iteration %d: pass1 scoring/projections=%s/%s "
            "pass2 scoring/projections=%s/%s",
            local_debug_iteration,
            parent_use_float64_scoring,
            parent_use_float64_projections,
            fine_use_float64_scoring,
            fine_use_float64_projections,
        )

    safe_ibs, safe_rbs = safe_batch_sizes(
        eff_n_rot,
        local_n_trans,
        image_shape_for_batch=experiment_dataset.image_shape,
        current_size_for_batch=cs_for_engine,
    )
    logger.info(
        "Local search batch sizing: cone_radius=%.3f rad (%.2f deg), est_cone_rots=%d, eff_n_rot=%d "
        "n_trans=%d → image_batch_size=%d, rotation_block_size=%d",
        cone_radius,
        np.rad2deg(cone_radius),
        est_cone_rots,
        eff_n_rot,
        local_n_trans,
        safe_ibs,
        safe_rbs,
    )
    # Keep the local-search hypothesis grid genuinely double precision end to
    # end when either flag requests it; default stays float32 to match
    # RELION's accelerated-GPU precision.
    parent_local_layout_dtype = (
        np.float64 if (parent_use_float64_scoring or parent_use_float64_projections) else np.float32
    )
    translation_prior_reference_translations = np.asarray(
        current_translations, dtype=parent_local_layout_dtype
    )
    if local_search_translation_prior_mode == "coarse":
        translation_prior_reference_translations, prior_grid_source, prior_grid_shape_mismatch = (
            _local_translation_prior_reference_translations(
                current_translations=current_translations,
                base_translations=base_translations,
                replay_prior_translations=replay_prior_translations,
                dtype=parent_local_layout_dtype,
            )
        )
        if prior_grid_shape_mismatch:
            logger.warning(
                "RELION mode: local translation prior grid from replay/base did not match scoring grid; "
                "using %s grid shape=%s for scoring grid shape=%s",
                prior_grid_source,
                translation_prior_reference_translations.shape,
                np.asarray(current_translations).shape,
            )
        logger.info(
            "RELION mode: local translation prior uses coarse %s grid (n=%d) while scoring perturbed translations",
            prior_grid_source,
            translation_prior_reference_translations.shape[0],
        )
    if int(local_parent_oversampling_order) > 0:
        logger.info(
            "RELION local search: expanding translations by oversampling_order=%d (coarse n=%d -> fine n=%d)",
            int(local_parent_oversampling_order),
            int(current_translations.shape[0]),
            int(local_n_trans),
        )
    pass2_layout = None
    relion_significant_counts_k = None
    local_adaptive_pass2_parent_mode = "none"
    local_adaptive_pass2_denominator_layout = None
    local_normalization_log_evidence = None
    if int(local_parent_oversampling_order) > 0 and not k_class_enabled:
        local_adaptive_pass2_full_parent = _local_adaptive_pass2_full_parent_enabled()
        local_adaptive_pass2_rotation_only = _local_adaptive_pass2_rotation_only_enabled()
        local_adaptive_pass2_denominator_mode = _local_adaptive_pass2_denominator_support_mode()
        local_adaptive_pass2_parent_mode = "full_parent" if local_adaptive_pass2_full_parent else "pruned_parent"
        parent_prior_translations = trans_prior_center
        if parent_prior_translations is None:
            parent_prior_translations = np.zeros(
                (np.asarray(previous_best_rotation_eulers_k).shape[0], np.asarray(current_translations).shape[1]),
                dtype=parent_local_layout_dtype,
            )
        parent_order = int(local_search_order) - int(local_parent_oversampling_order)
        if parent_order < 0:
            raise ValueError(
                "local_search_order must be >= local_parent_oversampling_order; "
                f"got {local_search_order} and {local_parent_oversampling_order}",
            )
        parent_grid_metadata = build_local_search_grid_metadata(parent_order)
        parent_layout = build_local_hypothesis_layout(
            previous_best_rotation_eulers_k,
            None,
            sigma_rot,
            sigma_psi,
            parent_order,
            current_translations,
            parent_prior_translations,
            current_sigma_offset_angstrom,
            None,
            experiment_dataset.voxel_size,
            grid_metadata=parent_grid_metadata,
            translation_prior_reference_translations=translation_prior_reference_translations,
            rotation_log_prior=relion_local_rotation_log_prior_k,
            rotation_grid_random_perturbation=local_search_random_perturbation,
            rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(parent_order, adaptive_oversampling=0),
            dtype=parent_local_layout_dtype,
        )
        parent_local_rot_max = (
            int(np.max(np.asarray(parent_layout.rotation_counts, dtype=np.int64)))
            if int(np.asarray(parent_layout.rotation_counts).size)
            else 1
        )
        parent_ibs, parent_rbs = safe_batch_sizes(
            max(64, parent_local_rot_max),
            int(current_translations.shape[0]),
        )
        logger.info(
            "RELION local adaptive pass 1: parent_order=%d local_rot_max=%d n_trans=%d current_size=%s",
            parent_order,
            parent_local_rot_max,
            int(current_translations.shape[0]),
            local_pass1_current_size,
        )
        logger.info("RELION local adaptive pass 1: using manual supplied-PPref interpolation")
        parent_outputs = _run_local_search_iteration(
            experiment_dataset,
            means_k,
            mean_variance,
            noise_variance_k,
            previous_best_rotation_eulers_k,
            None,
            None,
            parent_order,
            sigma_rot,
            sigma_psi,
            current_translations,
            trans_prior_center,
            current_sigma_offset_angstrom,
            current_translation_range,
            disc_type,
            image_batch_size=parent_ibs,
            rotation_block_size=parent_rbs,
            current_size=local_pass1_current_size,
            accumulate_noise=False,
            projection_padding_factor=PROJECTION_PADDING_FACTOR,
            reconstruction_padding_factor=PADDING_FACTOR,
            relion_projector_half=relion_projector_half,
            relion_projector_r_max=relion_projector_r_max,
            projection_relion_texture_interp=False,
            projection_relion_acc_double_floorf_quirk=RELION_ACC_DOUBLE_FLOORF_QUIRK,
            use_float64_scoring=parent_use_float64_scoring,
            use_float64_projections=parent_use_float64_projections,
            do_gridding_correction=True,
            square_window=RELION_FOURIER_WINDOW_SQUARE,
            half_spectrum_scoring=True,
            relion_exact_score_translation=bool(
                _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"]
                and not parent_use_float64_scoring
            ),
            image_corrections=image_corrections_k,
            scale_corrections=scale_corrections_k,
            group_ids=group_ids_k,
            scale_correction_group_count=group_count_k,
            scale_correction_data_vs_prior=scale_correction_data_vs_prior,
            image_pre_shifts=translation_search_base,
            score_with_masked_images=True,
            return_profile=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            adaptive_fraction=RELION_ADAPTIVE_FRACTION,
            max_significants=max_significants,
            reconstruct_significant_only=True,
            translation_prior_reference_translations=translation_prior_reference_translations,
            debug_iteration=local_debug_iteration,
            debug_pass_label="pass1_parent",
            pass2_layout=parent_layout,
            return_best_pose_details=False,
            translation_prior_centers=trans_prior_center_for_engine,
            rotation_log_prior=relion_local_rotation_log_prior_k,
            return_reconstruction_sample_indices=True,
            apply_max_significants_to_support=True,
            score_only=True,
        )
        parent_profile = parent_outputs[-1]
        significant_sample_indices = parent_profile["reconstruction_sample_indices_by_image"]
        pruned_parent_significant_sample_indices = significant_sample_indices
        # RELION's rlnNrOfSignificantSamples records the number of retained
        # coarse hypotheses from pass 1, not the number of fine hypotheses
        # used for reconstruction in pass 2. Preserve this before any
        # diagnostic expansion of the pass-2 parent support.
        relion_significant_counts_k = _relion_coarse_significant_counts(
            pruned_parent_significant_sample_indices
        )
        if relion_significant_counts_k is None:
            logger.warning(
                "RELION local adaptive pass 1 did not return explicit retained support; "
                "rlnNrOfSignificantSamples-compatible counts are unavailable"
            )
        if local_adaptive_pass2_full_parent:
            significant_sample_indices = [None] * len(significant_sample_indices)
            logger.info(
                "RELION local adaptive pass 2: expanding all parent samples; "
                "set %s=0 or %s=1 for diagnostic pruned-parent support",
                _LOCAL_ADAPTIVE_PASS2_FULL_PARENT_ENV,
                _LOCAL_ADAPTIVE_PASS2_DISABLE_FULL_PARENT_ENV,
            )
        elif local_adaptive_pass2_rotation_only:
            significant_sample_indices = _expand_significant_samples_to_full_parent_translations(
                significant_sample_indices,
                int(current_translations.shape[0]),
            )
            local_adaptive_pass2_parent_mode = "significant_rotation_full_translation"
            logger.info(
                "RELION local adaptive pass 2 diagnostic: expanding significant parent rotations to all "
                "parent translations via %s=1",
                _LOCAL_ADAPTIVE_PASS2_ROTATION_ONLY_ENV,
            )
        pass2_layout = build_local_adaptive_pass2_hypothesis_layout(
            parent_layout,
            significant_sample_indices,
            parent_order,
            oversampling_order=int(local_parent_oversampling_order),
            random_perturbation=float(local_search_random_perturbation),
            dtype=fine_local_layout_dtype,
        )
        if local_adaptive_pass2_denominator_mode is not None:
            if local_adaptive_pass2_denominator_mode == "full_parent":
                denominator_significant_sample_indices = [None] * len(pruned_parent_significant_sample_indices)
            elif local_adaptive_pass2_denominator_mode == "rotation_only":
                denominator_significant_sample_indices = _expand_significant_samples_to_full_parent_translations(
                    pruned_parent_significant_sample_indices,
                    int(current_translations.shape[0]),
                )
            else:  # Defensive only; parser restricts values.
                raise AssertionError(f"unexpected denominator mode {local_adaptive_pass2_denominator_mode!r}")
            local_adaptive_pass2_denominator_layout = build_local_adaptive_pass2_hypothesis_layout(
                parent_layout,
                denominator_significant_sample_indices,
                parent_order,
                oversampling_order=int(local_parent_oversampling_order),
                random_perturbation=float(local_search_random_perturbation),
                dtype=fine_local_layout_dtype,
            )
            if local_adaptive_pass2_denominator_layout.sample_mask_flat is None:
                denominator_valid_samples_per_image = (
                    np.asarray(local_adaptive_pass2_denominator_layout.rotation_counts, dtype=np.int64)
                    * int(local_adaptive_pass2_denominator_layout.translation_grid.shape[0])
                )
            else:
                denominator_valid_samples_per_image = np.asarray(
                    [
                        int(np.count_nonzero(local_adaptive_pass2_denominator_layout.sample_mask_flat[start:stop]))
                        for start, stop in zip(
                            local_adaptive_pass2_denominator_layout.rotation_offsets[:-1],
                            local_adaptive_pass2_denominator_layout.rotation_offsets[1:],
                        )
                    ],
                    dtype=np.int64,
                )
            logger.info(
                "RELION local adaptive pass 2 diagnostic: denominator support mode=%s "
                "fine valid candidates median=%d max=%d via %s",
                local_adaptive_pass2_denominator_mode,
                int(np.median(denominator_valid_samples_per_image))
                if denominator_valid_samples_per_image.size
                else 0,
                int(np.max(denominator_valid_samples_per_image)) if denominator_valid_samples_per_image.size else 0,
                _LOCAL_ADAPTIVE_PASS2_DENOMINATOR_SUPPORT_ENV,
            )
        parent_samples_per_image = np.asarray(
            [
                (
                    (
                        int(np.count_nonzero(parent_layout.sample_mask_flat[start:stop]))
                        if parent_layout.sample_mask_flat is not None
                        else int(stop - start) * int(current_translations.shape[0])
                    )
                    if sig is None
                    else int(np.asarray(sig).size)
                )
                for sig, start, stop in zip(
                    significant_sample_indices,
                    parent_layout.rotation_offsets[:-1],
                    parent_layout.rotation_offsets[1:],
                )
            ],
            dtype=np.int64,
        )
        if pass2_layout.sample_mask_flat is None:
            valid_samples_per_image = (
                np.asarray(pass2_layout.rotation_counts, dtype=np.int64) * int(pass2_layout.translation_grid.shape[0])
            )
        else:
            valid_samples_per_image = np.asarray(
                [
                    int(np.count_nonzero(pass2_layout.sample_mask_flat[start:stop]))
                    for start, stop in zip(pass2_layout.rotation_offsets[:-1], pass2_layout.rotation_offsets[1:])
                ],
                dtype=np.int64,
            )
        logger.info(
            "RELION local adaptive pass 2 mask: parent significant samples median=%d max=%d; "
            "fine valid candidates median=%d max=%d",
            int(np.median(parent_samples_per_image)) if parent_samples_per_image.size else 0,
            int(np.max(parent_samples_per_image)) if parent_samples_per_image.size else 0,
            int(np.median(valid_samples_per_image)) if valid_samples_per_image.size else 0,
            int(np.max(valid_samples_per_image)) if valid_samples_per_image.size else 0,
        )
    elif int(local_parent_oversampling_order) > 0:
        local_adaptive_pass2_parent_mode = "k_class_parent_expanded"
        logger.info(
            "Adaptive local coarse-pair masking is currently K=1-only; K-class local search keeps the existing parent-expanded support"
        )
    local_relion_x_half_mstep = (
        _k_class_relion_x_half_mstep_enabled()
        if k_class_enabled
        else _k1_relion_x_half_mstep_enabled()
    )
    if diagnostic_score_only:
        local_relion_x_half_mstep = False
    if local_relion_x_half_mstep:
        logger.info(
            "RELION local %s M-step: using x-half BPref-layout backprojection",
            "K-class" if k_class_enabled else "K=1",
        )
    if local_adaptive_pass2_denominator_layout is not None:
        logger.info(
            "RELION local adaptive pass 2 diagnostic: running score-only broad-denominator probe"
        )
        local_debug_env_names = [
            name
            for name in os.environ
            if name.startswith("RECOVAR_LOCAL_SCORE_DUMP_")
            or name.startswith("RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_")
            or name.startswith("RECOVAR_LOCAL_NOISE_COMPONENT_DUMP_")
        ]
        saved_local_debug_env = {name: os.environ.pop(name) for name in local_debug_env_names}
        try:
            denominator_outputs = _run_local_search_iteration(
                experiment_dataset,
                means_k,
                mean_variance,
                noise_variance_k,
                previous_best_rotation_eulers_k,
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
                accumulate_noise=False,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
                reconstruction_padding_factor=PADDING_FACTOR,
                relion_projector_half=relion_projector_half,
                relion_projector_r_max=relion_projector_r_max,
                use_float64_scoring=fine_use_float64_scoring,
                use_float64_projections=fine_use_float64_projections,
                do_gridding_correction=True,
                square_window=RELION_FOURIER_WINDOW_SQUARE,
                half_spectrum_scoring=True,
                relion_exact_score_translation=bool(
                    _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"]
                    and not fine_use_float64_scoring
                ),
                image_corrections=image_corrections_k,
                scale_corrections=scale_corrections_k,
                group_ids=group_ids_k,
                scale_correction_group_count=group_count_k,
                scale_correction_data_vs_prior=scale_correction_data_vs_prior,
                image_pre_shifts=translation_search_base,
                score_with_masked_images=True,
                return_profile=False,
                disable_adjoint_y=True,
                disable_adjoint_ctf=True,
                adaptive_fraction=RELION_ADAPTIVE_FRACTION,
                max_significants=max_significants,
                reconstruct_significant_only=False,
                translation_prior_reference_translations=translation_prior_reference_translations,
                debug_iteration=None,
                pass2_layout=local_adaptive_pass2_denominator_layout,
                return_best_pose_details=False,
                translation_prior_centers=trans_prior_center_for_engine,
                rotation_grid_random_perturbation=local_search_random_perturbation,
                rotation_grid_angular_sampling_deg=relion_angular_sampling_deg(
                    local_search_order,
                    adaptive_oversampling=0,
                ),
                score_only=True,
            )
        finally:
            os.environ.update(saved_local_debug_env)
        denominator_stats = denominator_outputs[3]
        local_normalization_log_evidence = np.asarray(
            denominator_stats.log_evidence_per_image,
            dtype=np.float64,
        )
        logger.info(
            "RELION local adaptive pass 2 diagnostic: broad-denominator evidence ready "
            "(finite=%d/%d)",
            int(np.count_nonzero(np.isfinite(local_normalization_log_evidence))),
            int(local_normalization_log_evidence.size),
        )
    # RELION's accelerated local-search loop still executes the symbolic
    # second pass when adaptive_oversampling == 0. In that case
    # convertAllSquaredDifferencesToWeights sets significant_weight to the
    # minimum fine-pass weight, so storeWeightedSums keeps all local
    # candidates. Do not apply the 0.999 significant-support prune on
    # this os0 path.
    local_reconstruct_significant_only = int(local_parent_oversampling_order) > 0
    local_accumulate_noise = not diagnostic_score_only
    local_disable_adjoint_y = bool(disable_adjoint_y or diagnostic_score_only)
    local_disable_adjoint_ctf = bool(disable_adjoint_ctf or diagnostic_score_only)
    logger.info(
        "RELION local fine pass 2: supplied-PPref interpolation follows "
        "RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP (default texture)"
    )
    local_outputs = _run_local_search_iteration(
        experiment_dataset,
        means_k,
        mean_variance,
        noise_variance_k,
        previous_best_rotation_eulers_k,
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
        accumulate_noise=local_accumulate_noise,
        projection_padding_factor=PROJECTION_PADDING_FACTOR,
        reconstruction_padding_factor=PADDING_FACTOR,
        relion_projector_half=relion_projector_half,
        relion_projector_r_max=relion_projector_r_max,
        # RELION's local adaptive path is intentionally hybrid: parent pass 1
        # uses the manual supplied-PPref projector above, while fine pass 2
        # follows the user-switchable texture default.
        projection_relion_texture_interp=None,
        projection_relion_acc_double_floorf_quirk=RELION_ACC_DOUBLE_FLOORF_QUIRK,
        use_float64_scoring=fine_use_float64_scoring,
        use_float64_projections=fine_use_float64_projections,
        do_gridding_correction=True,
        square_window=RELION_FOURIER_WINDOW_SQUARE,
        half_spectrum_scoring=True,
        relion_exact_score_translation=bool(
            _DENSE_EM_STATIC_KWARGS["relion_exact_fine_gaussian"]
            and not fine_use_float64_scoring
        ),
        image_corrections=image_corrections_k,
        scale_corrections=scale_corrections_k,
        group_ids=group_ids_k,
        scale_correction_group_count=group_count_k,
        scale_correction_data_vs_prior=scale_correction_data_vs_prior,
        image_pre_shifts=translation_search_base,
        score_with_masked_images=True,
        mstep_relion_x_half=local_relion_x_half_mstep,
        return_profile=collect_local_search_profile,
        disable_adjoint_y=local_disable_adjoint_y,
        disable_adjoint_ctf=local_disable_adjoint_ctf,
        adaptive_fraction=RELION_ADAPTIVE_FRACTION,
        max_significants=max_significants,
        reconstruct_significant_only=local_reconstruct_significant_only,
        stats_use_reconstruction_probs=local_reconstruct_significant_only,
        translation_prior_reference_translations=translation_prior_reference_translations,
        debug_iteration=local_debug_iteration,
        debug_pass_label="pass2_final",
        pass2_layout=pass2_layout,
        return_best_pose_details=True,
        normalization_log_evidence=local_normalization_log_evidence,
        translation_prior_centers=trans_prior_center_for_engine,
        rotation_grid_random_perturbation=local_search_random_perturbation,
        rotation_grid_angular_sampling_deg=local_search_angular_sampling_deg,
        local_parent_oversampling_order=local_parent_oversampling_order,
        rotation_log_prior=None if pass2_layout is not None else relion_local_rotation_log_prior_k,
        class_log_priors=class_log_priors if k_class_enabled else None,
        return_class_details=k_class_enabled,
        # The engine count describes fine-pass reconstruction support. It is
        # deliberately not exposed as RELION's coarse pass-1 metadata count.
        return_significant_counts=False,
        score_only=diagnostic_score_only,
        rotation_grid_mstep_rotations=local_search_mstep_rotations,
        generate_relion_mstep_rotations=True,
    )
    _local_cursor = 0
    Ft_y_k, Ft_ctf_k, ha_k = local_outputs[_local_cursor : _local_cursor + 3]
    _local_cursor += 3
    best_rots_k, best_trans_k, _best_rot_ids_k = local_outputs[_local_cursor : _local_cursor + 3]
    _local_cursor += 3
    em_stats_k = local_outputs[_local_cursor]
    _local_cursor += 1
    if local_accumulate_noise:
        noise_stats_k = local_outputs[_local_cursor]
        _local_cursor += 1
    else:
        noise_stats_k = None
    _local_tail = local_outputs[_local_cursor:]
    _tail_idx = 0
    if collect_local_search_profile:
        local_profile_k = _local_tail[_tail_idx]
        _tail_idx += 1
        profile_row = dict(local_profile_k)
        profile_row["iteration"] = np.int32(iteration)
        profile_row["half_index"] = np.int32(k)
        profile_row["local_adaptive_pass2_parent_mode"] = local_adaptive_pass2_parent_mode
        profile_row["local_adaptive_pass2_full_parent"] = np.bool_(local_adaptive_pass2_parent_mode == "full_parent")
        profile_row["diagnostic_score_only"] = np.bool_(diagnostic_score_only)
        local_profile_history.append(profile_row)
        if save_intermediates_dir is not None:
            np.savez_compressed(
                os.path.join(
                    save_intermediates_dir,
                    f"it{iteration:03d}_half{k + 1}_local_profile.npz",
                ),
                **local_profile_k,
            )
    if k_class_enabled:
        class_assignments_k, class_posterior_sums_k = _local_tail[_tail_idx : _tail_idx + 2]
        _tail_idx += 2
        if len(_local_tail) > _tail_idx:
            class_full_posterior_sums_k = _local_tail[_tail_idx]
            _tail_idx += 1
        else:
            class_full_posterior_sums_k = class_posterior_sums_k
        class_assignments[k] = np.asarray(class_assignments_k, dtype=np.int32)
        class_posterior_per_half[k] = np.asarray(class_posterior_sums_k, dtype=np.float64)
        if class_full_posterior_per_half is not None:
            class_full_posterior_per_half[k] = np.asarray(class_full_posterior_sums_k, dtype=np.float64)
    best_pose_rotations[k] = np.asarray(best_rots_k, dtype=np.float32)
    best_pose_rotation_eulers[k] = utils.R_to_relion(
        np.asarray(best_rots_k),
        degrees=True,
    ).astype(np.float32)
    best_pose_translations[k] = np.asarray(best_trans_k, dtype=np.float32)
    return HalfScoreResult(
        ha=ha_k,
        Ft_y=Ft_y_k,
        Ft_ctf=Ft_ctf_k,
        em_stats=em_stats_k,
        noise_stats=noise_stats_k,
        best_pose_rotations=best_pose_rotations[k],
        best_pose_rotation_eulers=best_pose_rotation_eulers[k],
        best_pose_translations=best_pose_translations[k],
        significant_counts=relion_significant_counts_k,
        mstep_full_half_axis=0 if local_relion_x_half_mstep else None,
        mstep_accumulator_shape=(
            # Must match the current-size BPref grid allocated by the local
            # engine above; downstream join/reconstruct calls infer layout
            # from this shape.
            relion_backprojector_volume_shape(
                experiment_dataset.volume_shape,
                PADDING_FACTOR,
                current_size=cs_for_engine,
            )
            if local_relion_x_half_mstep
            else None
        ),
    )


_STATE_SWAP_VARIANT_COMPONENTS = {
    "all_relion": set(),
    "all_recovar": {
        "state",
        "maps",
        "tau2_noise",
        "image_scale",
        "direction_prior",
        "sigma_offset",
        "current_size",
        "poses",
    },
    "recovar_maps": {"maps"},
    "recovar_tau2_noise": {"tau2_noise"},
    "recovar_image_scale": {"image_scale"},
    "recovar_direction_prior": {"direction_prior"},
    # Preserve every live RECOVAR boundary component except the learned
    # direction prior.  The replayed RELION prior then becomes the sole
    # intervention for first-divergence diagnostics.
    "relion_direction_prior": {
        "state",
        "maps",
        "tau2_noise",
        "image_scale",
        "sigma_offset",
        "current_size",
        "poses",
    },
    "recovar_sigma_offset": {"sigma_offset"},
    "recovar_current_size": {"current_size"},
    "recovar_poses": {"poses"},
    "recovar_state": {"state"},
    "recovar_state_sampling_grid": {"state_sampling_grid"},
    "recovar_state_local_priors": {"state_local_priors"},
    "recovar_state_convergence_only": {"state_convergence_only"},
    "recovar_state_no_grid": {"state_no_grid"},
    "recovar_tau2_only": {"tau2"},
    "recovar_noise_variance_only": {"noise_variance"},
    "recovar_previous_noise_radial_only": {"previous_noise_radial"},
    "recovar_image_correction_only": {"image_correction"},
    "recovar_scale_correction_only": {"scale_correction"},
    "recovar_maps_tau2_noise": {"maps", "tau2_noise"},
    "recovar_tau2_noise_image_scale": {"tau2_noise", "image_scale"},
    "recovar_state_poses": {"state", "poses"},
    "recovar_image_scale_poses": {"image_scale", "poses"},
    "recovar_maps_global_scale_to_relion": {"map_scale"},
    "recovar_maps_shell_scale_to_relion": {"map_scale"},
    "relion_maps_global_scale_to_recovar": {"map_scale"},
    "relion_maps_shell_scale_to_recovar": {"map_scale"},
}

_STATE_SWAP_MAP_SCALE_VARIANTS = {
    "recovar_maps_global_scale_to_relion": ("recovar", "relion", "global"),
    "recovar_maps_shell_scale_to_relion": ("recovar", "relion", "shell"),
    "relion_maps_global_scale_to_recovar": ("relion", "recovar", "global"),
    "relion_maps_shell_scale_to_recovar": ("relion", "recovar", "shell"),
}

_STATE_SWAP_STATE_FIELD_GROUPS = {
    "state_sampling_grid": {
        "translation_range",
        "translation_step",
        "adaptive_oversampling",
    },
    "state_local_priors": {
        "do_local_search",
        "sigma_rot",
        "sigma_psi",
    },
    "state_convergence_only": {
        "current_resolution",
        "previous_resolution",
        "ave_Pmax",
        "acc_rot",
        "acc_trans",
        "fraction_changed",
        "changes_optimal_offsets",
        "current_changes_optimal_orientations",
        "current_changes_optimal_offsets_angstrom",
        "current_changes_optimal_classes",
        "smallest_changes_optimal_orientations",
        "smallest_changes_optimal_offsets_angstrom",
        "smallest_changes_optimal_classes",
        "nr_iter_wo_resol_gain",
        "nr_iter_wo_assignment_changes",
        "nr_iter_wo_large_hidden_variable_changes",
        "mpi_leader_hidden_variable_angular_step_deg",
        "mpi_leader_hidden_variable_translation_step_angstrom",
        "suppress_hidden_variable_increment_once",
        "has_converged",
    },
}
_STATE_SWAP_STATE_NO_GRID_EXCLUDE = (
    _STATE_SWAP_STATE_FIELD_GROUPS["state_sampling_grid"]
    | _STATE_SWAP_STATE_FIELD_GROUPS["state_local_priors"]
)


def _score_half_local_in_bpref_scope(
    *,
    bpref_device_signature_active: bool,
    **kwargs,
) -> HalfScoreResult:
    """Run local scoring with device-capture flags disabled or fail closed."""

    if bpref_device_signature_active:
        raise RuntimeError(
            "BPref device signature capture is supported only by sparse adaptive pass 2"
        )
    with _cuda_backproject_diagnostics.bpref_device_signature_scope(False):
        return _score_half_local(**kwargs)


def _copy_optional_array(value):
    if value is None:
        return None
    return np.asarray(value).copy()


def _copy_half_pair(values):
    return [_copy_optional_array(value) for value in values]


def _format_relion_correction_range(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return "empty"
    return f"[{float(np.min(arr)):.6g}, {float(np.max(arr)):.6g}]"


def _copy_direction_prior_state(values, orders):
    return _copy_half_pair(values), [None if order is None else int(order) for order in orders]


def _copy_optional_float_pair(values):
    if values is None:
        return None
    return [float(values[0]), float(values[1])]


def _frozen_scoring_state_arrays(
    *,
    means,
    mean_variance,
    mean_variance_per_half=None,
    relion_half_inputs,
    noise_variance_per_half,
    current_sigma_offset_angstrom_per_half,
    global_direction_prior_per_half,
    experiment_datasets=None,
    sealed_sampling_state=None,
    sealed_scoring_context=None,
):
    """Materialize every K=1 scoring primitive owned by a frozen restart."""

    pair_fields = {
        "mean": means,
        "previous_best_rotation_eulers": relion_half_inputs.previous_best_rotation_eulers,
        "previous_best_translations": relion_half_inputs.previous_best_translations,
        "image_corrections": relion_half_inputs.image_corrections,
        "scale_corrections": relion_half_inputs.scale_corrections,
        "noise_variance": noise_variance_per_half,
        "direction_prior": global_direction_prior_per_half,
    }
    arrays = {}
    for field_name, values in pair_fields.items():
        if values is None or len(values) != 2:
            raise RuntimeError(
                f"Frozen scoring-state ownership requires two {field_name} arrays"
            )
        for half_index, value in enumerate(values):
            if value is None:
                raise RuntimeError(
                    "Frozen scoring-state ownership requires "
                    f"{field_name} for half {half_index + 1}"
                )
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number):
                raise RuntimeError(
                    f"Frozen scoring-state field {field_name} half {half_index + 1} "
                    f"must be numeric, got {array.dtype}"
                )
            if not np.all(np.isfinite(array)):
                raise RuntimeError(
                    f"Frozen scoring-state field {field_name} half {half_index + 1} "
                    "contains non-finite values"
                )
            arrays[f"{field_name}.half{half_index + 1}"] = np.ascontiguousarray(array).copy()

    if mean_variance_per_half is None:
        tau2 = np.asarray(mean_variance)
        if not np.issubdtype(tau2.dtype, np.number) or not np.all(np.isfinite(tau2)):
            raise RuntimeError("Frozen scoring-state mean_variance must be a finite numeric array")
        arrays["mean_variance"] = np.ascontiguousarray(tau2).copy()
    else:
        if len(mean_variance_per_half) != 2:
            raise RuntimeError("Frozen scoring-state ownership requires two mean_variance arrays")
        for half_index, value in enumerate(mean_variance_per_half):
            tau2 = np.asarray(value)
            if not np.issubdtype(tau2.dtype, np.number) or not np.all(np.isfinite(tau2)):
                raise RuntimeError(
                    f"Frozen scoring-state mean_variance half {half_index + 1} must be finite numeric"
                )
            arrays[f"mean_variance.half{half_index + 1}"] = np.ascontiguousarray(tau2).copy()

    sigma_pair = np.asarray(current_sigma_offset_angstrom_per_half, dtype=np.float64)
    if sigma_pair.shape != (2,) or not np.all(np.isfinite(sigma_pair)):
        raise RuntimeError(
            "Frozen scoring-state ownership requires two finite translation sigmas"
        )
    arrays["translation_sigma_angstrom_per_half"] = sigma_pair
    if experiment_datasets is not None:
        if len(experiment_datasets) != 2:
            raise RuntimeError("Frozen scoring-state ownership requires two datasets")
        for half_index, dataset in enumerate(experiment_datasets):
            ctf_params = np.asarray(dataset.CTF_params)
            if not np.issubdtype(ctf_params.dtype, np.number) or not np.all(np.isfinite(ctf_params)):
                raise RuntimeError(
                    f"Frozen scoring-state CTF parameters half {half_index + 1} must be finite numeric"
                )
            arrays[f"ctf_params.half{half_index + 1}"] = np.ascontiguousarray(ctf_params).copy()
    if sealed_sampling_state is not None:
        for key, value in sorted(sealed_sampling_state.items()):
            if isinstance(value, np.ndarray):
                array = np.asarray(value)
            elif isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                array = np.asarray(value)
            else:
                array = np.frombuffer(str(value).encode("utf-8"), dtype=np.uint8)
            arrays[f"sampling.{key}"] = np.ascontiguousarray(array).copy()
    if sealed_scoring_context is not None:
        context_bytes = json.dumps(
            sealed_scoring_context,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        arrays["sealed_scoring_context_json"] = np.frombuffer(context_bytes, dtype=np.uint8).copy()
    return arrays


def _assert_frozen_scoring_state_unchanged(expected, actual):
    """Fail closed if replay or setup mutated a sealed scoring primitive."""

    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "Frozen scoring-state fields changed: "
            f"missing={missing}, extra={extra}"
        )
    hashes = {}
    for field_name in sorted(expected):
        expected_array = np.asarray(expected[field_name])
        actual_array = np.asarray(actual[field_name])
        if expected_array.shape != actual_array.shape:
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} shape changed: "
                f"{expected_array.shape} != {actual_array.shape}"
            )
        if expected_array.dtype != actual_array.dtype:
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} dtype changed: "
                f"{expected_array.dtype} != {actual_array.dtype}"
            )
        if not np.all(np.isfinite(actual_array)):
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} contains non-finite values"
            )
        if not np.array_equal(expected_array, actual_array):
            raise RuntimeError(
                f"Frozen scoring-state field {field_name} was overwritten before scoring"
            )
        hashes[field_name] = {
            "dtype": str(actual_array.dtype),
            "shape": list(actual_array.shape),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(actual_array).view(np.uint8)
            ).hexdigest(),
        }
    return hashes


def _sigma_offset_for_half(current_sigma_offset_angstrom, current_sigma_offset_angstrom_per_half, half_index):
    if current_sigma_offset_angstrom_per_half is None:
        return float(current_sigma_offset_angstrom)
    return float(current_sigma_offset_angstrom_per_half[int(half_index)])


def _restore_state_fields(state, state_fields, fields):
    for field_name in fields:
        if field_name in state_fields:
            setattr(state, field_name, state_fields[field_name])


def _snapshot_state_swap_inputs(
    *,
    state,
    cs,
    means,
    mean_variance,
    noise_variance_per_half,
    noise_variance,
    previous_noise_radial_per_half,
    previous_noise_radial,
    relion_half_inputs,
    previous_best_rotations,
    current_sigma_offset_angstrom,
    current_sigma_offset_angstrom_per_half,
    class_direction_prior_per_half,
    class_direction_prior_order_per_half,
    global_direction_prior_per_half,
    global_direction_prior_order_per_half,
):
    class_priors, class_prior_orders = _copy_direction_prior_state(
        class_direction_prior_per_half,
        class_direction_prior_order_per_half,
    )
    global_priors, global_prior_orders = _copy_direction_prior_state(
        global_direction_prior_per_half,
        global_direction_prior_order_per_half,
    )
    return {
        "state_fields": dict(state.__dict__),
        "cs": int(cs),
        "means": [_copy_optional_array(mean) for mean in means],
        "mean_variance": _copy_optional_array(mean_variance),
        "noise_variance_per_half": _copy_half_pair(noise_variance_per_half),
        "noise_variance": _copy_optional_array(noise_variance),
        "previous_noise_radial_per_half": _copy_half_pair(previous_noise_radial_per_half),
        "previous_noise_radial": _copy_optional_array(previous_noise_radial),
        "image_corrections": _copy_half_pair(relion_half_inputs.image_corrections),
        "scale_corrections": _copy_half_pair(relion_half_inputs.scale_corrections),
        "previous_best_translations": _copy_half_pair(relion_half_inputs.previous_best_translations),
        "previous_best_rotation_eulers": _copy_half_pair(relion_half_inputs.previous_best_rotation_eulers),
        "previous_best_rotations": _copy_half_pair(previous_best_rotations),
        "current_sigma_offset_angstrom": float(current_sigma_offset_angstrom),
        "current_sigma_offset_angstrom_per_half": _copy_optional_float_pair(
            current_sigma_offset_angstrom_per_half
        ),
        "class_direction_prior_per_half": class_priors,
        "class_direction_prior_order_per_half": class_prior_orders,
        "global_direction_prior_per_half": global_priors,
        "global_direction_prior_order_per_half": global_prior_orders,
    }


def _state_swap_return_tuple(
    cs,
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
):
    return (
        cs,
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
    )


def _state_swap_map_shell_labels(volume_shape):
    """Return unshifted integer-radius labels for a full Fourier volume."""

    shape = tuple(int(size) for size in volume_shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(
            "State-swap map scaling requires a positive three-dimensional "
            f"volume shape, got {shape}"
        )
    axes = [np.fft.fftfreq(size) * size for size in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.rint(np.sqrt(sum(grid * grid for grid in grids))).astype(np.int32).reshape(-1)


def _scale_state_swap_reference_maps(
    source_means,
    target_means,
    *,
    mode,
    volume_shape,
):
    """Scale source-map amplitudes toward target maps without changing phase."""

    if mode not in {"global", "shell"}:
        raise ValueError(f"Unknown state-swap map scaling mode {mode!r}")
    if len(source_means) != len(target_means) or not source_means:
        raise ValueError(
            "State-swap map scaling requires equal non-empty source and target map lists"
        )

    expected_size = int(np.prod(tuple(int(size) for size in volume_shape)))
    shell_labels = (
        _state_swap_map_shell_labels(volume_shape)
        if mode == "shell"
        else None
    )
    scaled_means = []
    summaries = []
    for map_index, (source_value, target_value) in enumerate(
        zip(source_means, target_means, strict=True)
    ):
        if source_value is None or target_value is None:
            raise ValueError(
                f"State-swap map scaling requires map {map_index + 1} in both states"
            )
        source = np.asarray(source_value)
        target = np.asarray(target_value)
        if source.shape != target.shape or source.size != expected_size:
            raise ValueError(
                "State-swap map scaling shape mismatch for map "
                f"{map_index + 1}: source={source.shape}, target={target.shape}, "
                f"expected_size={expected_size}"
            )
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"State-swap map scaling requires finite map {map_index + 1} values"
            )

        source_flat = source.reshape(-1)
        target_flat = target.reshape(-1)
        before_norm = float(np.linalg.norm(source_flat - target_flat))
        target_norm = float(np.linalg.norm(target_flat))
        relative_before = before_norm / target_norm if target_norm > 0.0 else before_norm

        if mode == "global":
            denominator = float(np.vdot(source_flat, source_flat).real)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(
                    f"State-swap map {map_index + 1} has zero or invalid source energy"
                )
            scale = float(np.vdot(source_flat, target_flat).real / denominator)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"State-swap map {map_index + 1} has invalid global scale {scale}"
                )
            scaled_flat = source_flat * scale
            scale_values = np.asarray([scale], dtype=np.float64)
        else:
            scaled_flat = source_flat.copy()
            scale_values = np.ones(int(np.max(shell_labels)) + 1, dtype=np.float64)
            for shell in np.unique(shell_labels):
                shell_mask = shell_labels == shell
                source_shell = source_flat[shell_mask]
                target_shell = target_flat[shell_mask]
                denominator = float(np.vdot(source_shell, source_shell).real)
                target_energy = float(np.vdot(target_shell, target_shell).real)
                if denominator <= 0.0:
                    if target_energy > 0.0:
                        raise ValueError(
                            "State-swap map scaling cannot create target energy in "
                            f"empty source shell {int(shell)} for map {map_index + 1}"
                        )
                    scale = 1.0
                else:
                    scale = float(np.vdot(source_shell, target_shell).real / denominator)
                    if not np.isfinite(scale) or scale <= 0.0:
                        raise ValueError(
                            "State-swap map scaling found invalid scale "
                            f"{scale} in shell {int(shell)} for map {map_index + 1}"
                        )
                scaled_flat[shell_mask] = source_shell * scale
                scale_values[int(shell)] = scale

        scaled = scaled_flat.reshape(source.shape).astype(source.dtype, copy=False)
        after_norm = float(np.linalg.norm(scaled.reshape(-1) - target_flat))
        relative_after = after_norm / target_norm if target_norm > 0.0 else after_norm
        scaled_means.append(scaled)
        summaries.append(
            {
                "map_index": map_index,
                "mode": mode,
                "scale_min": float(np.min(scale_values)),
                "scale_max": float(np.max(scale_values)),
                "relative_l2_before": relative_before,
                "relative_l2_after": relative_after,
            }
        )
    return scaled_means, summaries


def _apply_state_swap_probe(
    *,
    probe,
    iteration,
    recovar_snapshot,
    state,
    cs,
    volume_shape,
    means,
    mean_variance,
    noise_variance_per_half,
    noise_variance,
    previous_noise_radial_per_half,
    previous_noise_radial,
    relion_half_inputs,
    previous_best_rotations,
    current_sigma_offset_angstrom,
    current_sigma_offset_angstrom_per_half,
    class_direction_prior_per_half,
    class_direction_prior_order_per_half,
    global_direction_prior_per_half,
    global_direction_prior_order_per_half,
):
    """Restore selected RECOVAR-produced state after RELION replay override."""

    if not probe or recovar_snapshot is None:
        return _state_swap_return_tuple(
            cs,
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
        )
    target_iteration = int(probe.get("iteration", 1))
    if int(iteration) != target_iteration:
        return _state_swap_return_tuple(
            cs,
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
        )

    variant = str(probe.get("variant", "all_relion"))
    components = _STATE_SWAP_VARIANT_COMPONENTS.get(variant)
    if components is None:
        raise ValueError(
            f"Unknown state_swap_probe variant {variant!r}; expected one of "
            f"{sorted(['all_relion', *_STATE_SWAP_VARIANT_COMPONENTS])}",
        )
    logger.warning(
        "STATE-SWAP diagnostic: iteration=%d variant=%s restoring RECOVAR components=%s after RELION replay",
        int(iteration) + 1,
        variant,
        ",".join(sorted(components)),
    )

    recovar_state_fields = recovar_snapshot["state_fields"]
    if "state" in components:
        state.__dict__.update(recovar_state_fields)
    if "state_sampling_grid" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            _STATE_SWAP_STATE_FIELD_GROUPS["state_sampling_grid"],
        )
    if "state_local_priors" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            _STATE_SWAP_STATE_FIELD_GROUPS["state_local_priors"],
        )
    if "state_convergence_only" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            _STATE_SWAP_STATE_FIELD_GROUPS["state_convergence_only"],
        )
    if "state_no_grid" in components:
        _restore_state_fields(
            state,
            recovar_state_fields,
            set(recovar_state_fields) - _STATE_SWAP_STATE_NO_GRID_EXCLUDE,
        )
    if "maps" in components:
        means = [jnp.asarray(mean) if mean is not None else None for mean in recovar_snapshot["means"]]
    if "map_scale" in components:
        source_name, target_name, scale_mode = _STATE_SWAP_MAP_SCALE_VARIANTS[variant]
        relion_means = means
        recovar_means = recovar_snapshot["means"]
        source_means = recovar_means if source_name == "recovar" else relion_means
        target_means = recovar_means if target_name == "recovar" else relion_means
        scaled_means, scale_summaries = _scale_state_swap_reference_maps(
            source_means,
            target_means,
            mode=scale_mode,
            volume_shape=volume_shape,
        )
        means = [jnp.asarray(mean) for mean in scaled_means]
        for summary in scale_summaries:
            logger.warning(
                "STATE-SWAP map amplitude: variant=%s map=%d mode=%s "
                "scale=[%.9g, %.9g] relative_l2=%.9g->%.9g",
                variant,
                int(summary["map_index"]) + 1,
                summary["mode"],
                summary["scale_min"],
                summary["scale_max"],
                summary["relative_l2_before"],
                summary["relative_l2_after"],
            )
    if "tau2_noise" in components or "tau2" in components:
        mean_variance = jnp.asarray(recovar_snapshot["mean_variance"])
    if "tau2_noise" in components or "noise_variance" in components:
        noise_variance_per_half = [
            jnp.asarray(noise_k) if noise_k is not None else None
            for noise_k in recovar_snapshot["noise_variance_per_half"]
        ]
        noise_variance = jnp.asarray(recovar_snapshot["noise_variance"])
    if "tau2_noise" in components or "previous_noise_radial" in components:
        previous_noise_radial_per_half = _copy_half_pair(recovar_snapshot["previous_noise_radial_per_half"])
        previous_noise_radial = _copy_optional_array(recovar_snapshot["previous_noise_radial"])
    if "image_scale" in components or "image_correction" in components:
        relion_half_inputs.image_corrections = _copy_half_pair(recovar_snapshot["image_corrections"])
    if "image_scale" in components or "scale_correction" in components:
        relion_half_inputs.scale_corrections = _copy_half_pair(recovar_snapshot["scale_corrections"])
    if "poses" in components:
        relion_half_inputs.previous_best_translations = _copy_half_pair(
            recovar_snapshot["previous_best_translations"],
        )
        relion_half_inputs.previous_best_rotation_eulers = _copy_half_pair(
            recovar_snapshot["previous_best_rotation_eulers"],
        )
        previous_best_rotations = _copy_half_pair(recovar_snapshot["previous_best_rotations"])
    if "direction_prior" in components:
        class_direction_prior_per_half = _copy_half_pair(recovar_snapshot["class_direction_prior_per_half"])
        class_direction_prior_order_per_half = list(recovar_snapshot["class_direction_prior_order_per_half"])
        global_direction_prior_per_half = _copy_half_pair(recovar_snapshot["global_direction_prior_per_half"])
        global_direction_prior_order_per_half = list(recovar_snapshot["global_direction_prior_order_per_half"])
    if "sigma_offset" in components:
        current_sigma_offset_angstrom = float(recovar_snapshot["current_sigma_offset_angstrom"])
        current_sigma_offset_angstrom_per_half = _copy_optional_float_pair(
            recovar_snapshot["current_sigma_offset_angstrom_per_half"]
        )
    if "current_size" in components:
        cs = int(recovar_snapshot["cs"])

    return _state_swap_return_tuple(
        cs,
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
    )


def _validate_relion_healpix_orders(orders, *, max_iter, init_healpix_order, max_healpix_order):
    if orders is None:
        return None
    orders = tuple(int(order) for order in orders)
    if len(orders) < int(max_iter):
        raise ValueError(
            "relion_healpix_orders must provide at least max_iter entries "
            f"({len(orders)} < {int(max_iter)})"
        )
    if any(right < left for left, right in zip(orders, orders[1:])):
        raise ValueError("relion_healpix_orders must be monotone nondecreasing")
    if orders[0] < int(init_healpix_order):
        raise ValueError(
            "relion_healpix_orders cannot coarsen below init_healpix_order "
            f"({orders[0]} < {int(init_healpix_order)})"
        )
    if orders[-1] > int(max_healpix_order):
        raise ValueError(
            "relion_healpix_orders exceeds max_healpix_order "
            f"({orders[-1]} > {int(max_healpix_order)})"
        )
    return orders

def _with_validated_relion_healpix_orders(options: RefinementOptions) -> RefinementOptions:
    """Return a copy of `options` with validated `adaptive.relion_healpix_orders`.
    """
    adaptive = options.adaptive
    if adaptive.relion_current_sizes is not None and len(adaptive.relion_current_sizes) == 0:
        raise ValueError("relion_current_sizes must be non-empty when provided")
    validated_orders = _validate_relion_healpix_orders(
        adaptive.relion_healpix_orders,
        max_iter=options.schedule.max_iter,
        init_healpix_order=options.schedule.init_healpix_order,
        max_healpix_order=options.schedule.max_healpix_order,
    )
    return dataclasses.replace(
        options, adaptive=dataclasses.replace(adaptive, relion_healpix_orders=validated_orders)
    )

def _apply_relion_healpix_order_oracle(state, target_order, *, iteration_number):
    target_order = int(target_order)
    if target_order < int(state.healpix_order):
        raise ValueError(
            "relion_healpix_orders cannot coarsen the active state: "
            f"iteration={int(iteration_number)} target={target_order} "
            f"active={int(state.healpix_order)}"
        )
    while int(state.healpix_order) < target_order:
        state = refine_angular_sampling(state)
    return state


def _sealed_sampling_base_grids(sealed_sampling_state, *, voxel_size_angstrom, dtype: np.dtype = np.float32):
    """Construct scorer grids directly from a schema-v3 captured sampling state.

    ``dtype`` controls only the returned rotation matrices, matching
    ``_relion_rotation_grid_float32``'s policy: pass ``np.float64`` under
    float64 scoring/projections so a restart from a sealed boundary keeps the
    same coarse-grid precision as a fresh (non-restarted) run.
    """

    state = sealed_sampling_state
    directions = np.asarray(state["directions_ipix"], dtype=np.int64)
    rot = np.asarray(state["rot_angles_deg"], dtype=np.float64)
    tilt = np.asarray(state["tilt_angles_deg"], dtype=np.float64)
    psi = np.asarray(state["psi_angles_deg"], dtype=np.float64)
    if directions.ndim != 1 or directions.size < 1:
        raise ValueError("sealed sampling directions must be a nonempty vector")
    if rot.shape != directions.shape or tilt.shape != directions.shape or psi.ndim != 1 or psi.size < 1:
        raise ValueError("sealed sampling Euler component shapes are inconsistent")
    source_eulers = np.stack(
        [
            np.tile(rot, psi.size),
            np.tile(tilt, psi.size),
            np.repeat(psi, directions.size),
        ],
        axis=1,
    )
    from recovar.em.sampling import _relion_mstep_rotations_from_eulers

    rotations = _relion_mstep_rotations_from_eulers(source_eulers, dtype=dtype)
    eulers = source_eulers.astype(np.float32)
    voxel_size = float(voxel_size_angstrom)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("sealed sampling requires a finite positive voxel size")
    tx = np.asarray(state["translations_x_angstrom"], dtype=np.float64)
    ty = np.asarray(state["translations_y_angstrom"], dtype=np.float64)
    if tx.shape != ty.shape or tx.ndim != 1 or tx.size < 1:
        raise ValueError("sealed sampling translation component shapes are inconsistent")
    translations = np.stack([tx / voxel_size, ty / voxel_size], axis=1).astype(dtype)
    return rotations, eulers, jnp.asarray(translations, dtype=dtype)


def _sealed_sampling_rotation_ids(sealed_sampling_state):
    """Map captured direction/psi rows to canonical coarse rotation IDs."""

    direction_ids = np.asarray(sealed_sampling_state["directions_ipix"], dtype=np.int64)
    n_psi = int(np.asarray(sealed_sampling_state["psi_angles_deg"]).size)
    order = int(sealed_sampling_state["healpix_order_original"])
    n_pixels = 12 * (4**order)
    return np.concatenate(
        [direction_ids + psi_index * n_pixels for psi_index in range(n_psi)]
    ).astype(np.int64, copy=False)


def _sealed_direction_log_prior(direction_prior, sealed_sampling_state, *, dtype: np.dtype = np.float32):
    """Expand a full direction prior onto the exact captured direction rows."""

    prior = np.asarray(direction_prior, dtype=dtype).reshape(-1)
    direction_ids = np.asarray(sealed_sampling_state["directions_ipix"], dtype=np.int64)
    n_psi = int(np.asarray(sealed_sampling_state["psi_angles_deg"]).size)
    selected = np.tile(prior[direction_ids], n_psi)
    result = np.full(selected.shape, -np.inf, dtype=dtype)
    positive = selected > 0.0
    result[positive] = np.log(selected[positive]).astype(dtype)
    return result


def _mean_variance_for_scoring_half(mean_variance_per_half, half_index):
    """Select the exact half-owned K=1 tau2 prior passed to the scorer."""

    if len(mean_variance_per_half) != 2 or int(half_index) not in (0, 1):
        raise ValueError("per-half scoring tau2 requires exactly two halves and index 0 or 1")
    return mean_variance_per_half[int(half_index)]


def _updated_mean_variance_per_half(
    shared_mean_variance,
    updated_mean_variance_per_half,
    *,
    use_per_half_mean_variance,
):
    """Keep historical K=1 scoring on shared tau2 unless explicitly enabled."""

    if use_per_half_mean_variance:
        if len(updated_mean_variance_per_half) != 2:
            raise ValueError("per-half scoring tau2 update requires exactly two halves")
        return [
            jnp.asarray(updated_mean_variance_per_half[0]),
            jnp.asarray(updated_mean_variance_per_half[1]),
        ]
    return [shared_mean_variance, shared_mean_variance]


def _init_resolution_from_fsc(
    state: RefinementState, options: RefinementOptions, *, grid_size: int, voxel_size: float
) -> None:
    """Seed current/previous resolution from a caller-provided initial FSC curve."""
    schedule = options.schedule
    fsc = np.asarray(schedule.init_fsc, dtype=np.float32).copy()
    previous_current_size = int(schedule.init_current_size)
    if previous_current_size < grid_size:
        fsc[min(len(fsc), previous_current_size // 2) :] = 0.0
    data_vs_prior = np.asarray(fsc_to_relion_ssnr(fsc, tau2_fudge=options.parity.tau2_fudge))
    resolution_shell = resolution_from_data_vs_prior(data_vs_prior, allow_high_res_recovery=True)
    resolution_angstrom = shell_index_to_resolution_angstrom(resolution_shell, grid_size, voxel_size)
    if np.isfinite(resolution_angstrom) and resolution_angstrom > 0.0:
        state.current_resolution = float(resolution_angstrom)
        state.previous_resolution = float(resolution_angstrom)


def _init_resolution_from_firstiter_ini_high(
    state: RefinementState, options: RefinementOptions, *, grid_size: int, voxel_size: float
) -> None:
    """Seed current/previous resolution from the first-iteration ini_high lowpass."""
    ini_high_angstrom = options.parity.relion_firstiter_ini_high_angstrom
    pixel_size = float(voxel_size if voxel_size > 0 else 1.0)
    shell = int(np.floor(grid_size * pixel_size / float(ini_high_angstrom) + 0.5))
    shell = max(1, min(grid_size // 2, shell))
    resolution_angstrom = shell_index_to_resolution_angstrom(shell, grid_size, pixel_size)
    state.current_resolution = float(resolution_angstrom)
    state.previous_resolution = float(resolution_angstrom)



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

    if options.adaptive.relion_current_sizes is not None and len(options.adaptive.relion_current_sizes) == 0:
        raise ValueError("relion_current_sizes must be non-empty when provided")
    options = _with_validated_relion_healpix_orders(options)

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
    class_log_priors, class_weights = _initialize_class_log_priors(n_classes, k_class.init_class_log_priors, replay.init_direction_prior)

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

    _maybe_cache_raw_image_loaders(experiment_datasets)
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
        _init_resolution_from_fsc(state, options, grid_size=grid_size, voxel_size=cryo.voxel_size)
    elif init_relion_iteration == 0 and parity.relion_firstiter_ini_high_angstrom is not None:
        _init_resolution_from_firstiter_ini_high(state, options, grid_size=grid_size, voxel_size=cryo.voxel_size)
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
        if plan.image_batch_size != batching.image_batch_size or plan.rotation_block_size != batching.rotation_block_size:
            logger.info(
                "RELION EM batch sizing: requested image_batch_size=%d rotation_block_size=%d; "
                "using image_batch_size=%d rotation_block_size=%d "
                "(n_rot=%d n_trans=%d K=%d, score_budget=%.1fM floats, score_pixels=%d, "
                "projection_tile=%.2f/%.2f GB, active_score_tile=%.2f/%.2f GB, "
                "pose_pixel_tile=%.2f GB, translation_tile=%.2f/%.2f GB, "
                "persistent_est=%.2f GB, usable_est=%.2f GB, gpu_used_est=%.2f GB)",
                batching.image_batch_size,
                batching.rotation_block_size,
                plan.image_batch_size,
                plan.rotation_block_size,
                int(n_rot),
                int(n_trans),
                int(n_classes if classes is None else classes),
                plan.score_float_budget / 1e6,
                plan.score_pixel_count,
                plan.projection_block_gb,
                plan.projection_budget_gb,
                plan.active_score_tile_gb,
                plan.active_score_tile_budget_gb,
                plan.pose_pixel_tile_gb,
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
    initial_real_references_by_half = [None, None]
    if init_reference_real is not None:
        expected_volume_shape = tuple(int(value) for value in volume_shape)

        def _as_class_real_references(value):
            array = np.asarray(value, dtype=np.float64)
            if n_classes == 1 and array.shape == expected_volume_shape:
                return array[None, ...]
            expected_class_shape = (n_classes,) + expected_volume_shape
            if array.shape == expected_class_shape:
                return array
            raise ValueError(
                "init_reference_real must be a shared real volume, a per-class "
                f"array, or a two-half collection; got {array.shape}, expected "
                f"{expected_volume_shape} or {expected_class_shape}",
            )

        if isinstance(init_reference_real, (list, tuple)) and len(init_reference_real) == 2:
            initial_real_references_by_half = [
                _as_class_real_references(init_reference_real[0]),
                _as_class_real_references(init_reference_real[1]),
            ]
        else:
            real_array = np.asarray(init_reference_real)
            per_half_shape = (2, n_classes) + expected_volume_shape
            if n_classes == 1 and real_array.shape == (2,) + expected_volume_shape:
                initial_real_references_by_half = [
                    _as_class_real_references(real_array[0]),
                    _as_class_real_references(real_array[1]),
                ]
            elif real_array.shape == per_half_shape:
                initial_real_references_by_half = [
                    _as_class_real_references(real_array[0]),
                    _as_class_real_references(real_array[1]),
                ]
            else:
                shared_real = _as_class_real_references(real_array)
                initial_real_references_by_half = [shared_real, shared_real]
        logger.info(
            "RELION initial projector: preserving direct float64 real-reference handoff"
        )
    noise_variance_per_half = _normalize_noise_variance_per_half(
        init_noise_variance,
        n_halves=2,
    )
    noise_variance = _mean_noise_variance(noise_variance_per_half)
    initial_mean_variance = jnp.array(init_mean_variance)
    if parity.use_per_half_mean_variance:
        if k_class_enabled:
            raise ValueError("per-half scoring tau2 is supported only for K=1")
        if initial_mean_variance.ndim != 2 or initial_mean_variance.shape[0] != 2:
            raise ValueError(
                "per-half scoring tau2 requires init_mean_variance with leading half axis 2"
            )
        mean_variance_per_half = [
            jnp.asarray(initial_mean_variance[0]),
            jnp.asarray(initial_mean_variance[1]),
        ]
        mean_variance = jnp.asarray(
            0.5
            * (
                mean_variance_per_half[0].astype(jnp.float64)
                + mean_variance_per_half[1].astype(jnp.float64)
            ),
            dtype=jnp.float32,
        )
        logger.info("Initialized exact per-half K=1 tau2 priors")
    else:
        mean_variance = initial_mean_variance
        mean_variance_per_half = [mean_variance, mean_variance]
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
        dtype=jnp.float32,
    )
    _mark_setup_phase("noise_radial_init")

    # RELION randomises each half once at the first iteration and then uses
    # the first 100 half-1 particles for calculateExpectedAngularErrors.
    # Build that immutable local order once.  A missing/rebuilt-without-this-
    # helper binding is handled fail-closed below: acc_rot stays infinite and
    # cannot trigger convergence.
    expected_accuracy_trial_order = None
    effective_optimizer_random_seed = (
        parity.perturb_seed if parity.optimizer_random_seed is None else parity.optimizer_random_seed
    )
    if expected_accuracy.half1_trial_order_local is not None:
        expected_accuracy_trial_order = np.asarray(
            expected_accuracy.half1_trial_order_local,
            dtype=np.int64,
        ).reshape(-1)
        expected_trial_count = int(experiment_datasets[0].n_units)
        if expected_accuracy_trial_order.shape != (expected_trial_count,):
            raise ValueError(
                "expected_accuracy.half1_trial_order_local must have shape "
                f"({expected_trial_count},), got {expected_accuracy_trial_order.shape}"
            )
        if not np.array_equal(
            np.sort(expected_accuracy_trial_order),
            np.arange(expected_trial_count, dtype=np.int64),
        ):
            raise ValueError(
                "expected_accuracy.half1_trial_order_local must be a permutation "
                "of half-1 local particle indices"
            )
        logger.info(
            "RELION expected accuracy consumes an explicit physical trial order "
            "(%d particles)",
            expected_trial_count,
        )
    elif effective_optimizer_random_seed is not None and int(experiment_datasets[0].n_units) > 0:
        try:
            expected_accuracy_trial_order = relion_half1_trial_order(
                int(experiment_datasets[0].n_units),
                int(effective_optimizer_random_seed),
                first_iteration=max(1, int(init_relion_iteration) + 1),
                base_order_local=expected_accuracy.half1_base_order_local,
                optics_group_ids=expected_accuracy.half1_optics_group_ids,
            )
            if (
                expected_accuracy.half1_optics_group_ids is not None
                and np.unique(np.asarray(expected_accuracy.half1_optics_group_ids)).size > 1
            ):
                raise NotImplementedError(
                    "exact expected accuracy currently supports one RELION optics group; "
                    "per-optics image size/noise/CTF scaling is not yet implemented",
                )
        except Exception as exc:
            expected_accuracy_trial_order = None
            logger.warning("RELION exact expected-accuracy particle order unavailable: %s", exc)

    follower_setup = setup_relion_follower_scale_state(
        options,
        relion_half_inputs=relion_half_inputs,
        experiment_datasets=experiment_datasets,
        k_class_enabled=k_class_enabled,
    )
    relion_follower_scale_state = follower_setup.follower_scale_state
    relion_follower_owners_per_half = follower_setup.follower_owners_per_half
    relion_scale_stats_group_ids_per_half = follower_setup.scale_stats_group_ids_per_half
    relion_scale_stats_group_count_per_half = follower_setup.scale_stats_group_count_per_half

    def _finalize_relion_follower_scale_replay_telemetry():
        if replay.relion_follower_scale_replay is None:
            return None, None
        requested, applied = validate_relion_follower_scale_replay_application(
            replay.relion_follower_scale_replay,
            applied_iterations=history.relion_follower_scale_replay_applied_iterations,
        )
        logger.info(
            "Diagnostic RELION follower-scale replay complete: source=%s requested=%s applied=%s",
            replay.relion_follower_scale_replay.source,
            requested.tolist(),
            applied.tolist(),
        )
        return requested, applied

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
    native_sampling_boundary = (
        perturb_replay_relion_dir is None and sealed_sampling_state is None
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

        if relion_follower_scale_state is not None:
            _dispatch_relion_follower_scale_for_numbered_iteration(
                follower_setup,
                history,
                iteration=iteration,
                init_relion_iteration=int(init_relion_iteration),
                relion_half_inputs=relion_half_inputs,
                relion_follower_scale_replay_source=replay.relion_follower_scale_replay,
            )
            relion_follower_scale_state = follower_setup.follower_scale_state
            relion_follower_owners_per_half = follower_setup.follower_owners_per_half
            relion_scale_stats_group_ids_per_half = follower_setup.scale_stats_group_ids_per_half

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
                cs = int(seeded_cs)
                data_vs_prior_iter = None
                logger.info(
                    "RELION init bootstrap: seeding iter-1 current_size from ini_high=%.2f A -> %d",
                    float(parity.relion_firstiter_ini_high_angstrom),
                    cs,
                )
            elif schedule.init_fsc is not None:
                prev_cs = int(schedule.init_current_size)
                fsc_prev = _truncate_fsc_for_current_size_growth(
                    schedule.init_fsc,
                    current_size=prev_cs,
                    grid_size=grid_size,
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
                cs = quantize_current_size(raw_cs, ori_size=grid_size)
            else:
                cs = _bootstrap_current_size_relion(schedule.init_current_size, grid_size)
                data_vs_prior_iter = None
        else:
            prev_cs = history.current_sizes[-1]
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
                fsc_prev_raw = np.asarray(history.fsc_history[-1], dtype=np.float32).copy()
                fsc_prev_for_growth = _truncate_fsc_for_current_size_growth(
                    history.fsc_for_growth_history[-1] if history.fsc_for_growth_history else fsc_prev_raw,
                    current_size=prev_cs,
                    grid_size=grid_size,
                )

                data_vs_prior_iter = _k1_data_vs_prior_for_scheduling(
                    raw_fsc=fsc_prev_raw,
                    corrected_data_vs_prior=previous_data_vs_prior_for_scheduling,
                    current_size=prev_cs,
                    grid_size=grid_size,
                    tau2_fudge=tau2_fudge,
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
                cs = quantize_current_size(raw_cs, ori_size=grid_size)

        cs = quantize_current_size(cs, ori_size=grid_size)
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
                int(cs),
            )
        if adaptive.relion_current_sizes is not None:
            if iteration < len(adaptive.relion_current_sizes):
                oracle_cs = int(adaptive.relion_current_sizes[iteration])
            else:
                oracle_cs = int(adaptive.relion_current_sizes[-1])
            if oracle_cs <= 0:
                oracle_cs = int(schedule.init_current_size)
            cs = quantize_current_size(oracle_cs, ori_size=grid_size)
            logger.info(
                "Current-size oracle: iteration %d using current_size=%d",
                iteration + 1,
                cs,
            )

        # RELION updates image_coarse_size before updateAngularSampling at the
        # start of expectation(). Preserve that incoming sampling order even
        # when replay/native scheduling advances state.healpix_order below.
        coarse_size_healpix_order = _relion_expectation_coarse_size_order(
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
                cs=cs,
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
            cs=cs,
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
        )
        cs = replay_result.cs
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
            cs,
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
            cs=cs,
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
                        current_image_size=int(cs),
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
            cs,
            state.healpix_order,
            float(current_sigma_offset_angstrom),
            _copy_optional_float_pair(current_sigma_offset_angstrom_per_half),
        )
        current_size = int(cs)

        logger.info(
            "=== RELION Iteration %d/%d: current_size=%d, healpix_order=%d, local_search=%s ===",
            iteration + 1,
            schedule.max_iter,
            current_size,
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
        effective_rotation_eulers = np.asarray(current_rotation_eulers, dtype=np.float32)
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
                )
                _, _, effective_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
                    mstep_source_eulers,
                    random_perturbation,
                    angsamp_deg,
                    return_mstep_rotations=True,
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
                    "double-precision host" if adaptive_pass1_use_float64 else "CUDA",
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
        cs_for_engine = current_size if current_size < cryo.image_shape[0] else None
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
                        local_pass1_current_size = _relion_local_pass1_current_size(
                            pre_update_healpix_order=coarse_size_healpix_order,
                            pixel_size=cryo.voxel_size if cryo.voxel_size > 0 else 1.0,
                            ori_size=grid_size,
                            particle_diameter=particle_diameter_ang,
                            current_size=cs if cs_for_engine is not None else None,
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
            if sealed_sampling_state is not None:
                coarse_size = int(sealed_sampling_state["coarse_size"])
                if coarse_size > int(cs):
                    raise ValueError(
                        "sealed sampling coarse_size exceeds active current_size: "
                        f"coarse={coarse_size} current={cs}"
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
                    current_size=cs_for_engine,
                    volume_shape=volume_shape,
                    padding_factor=PROJECTION_PADDING_FACTOR,
                    n_classes=n_classes,
                )
                logger.info(
                    "RELION mode: using captured exact Projector::data at current_size=%s "
                    "r_max=%s manifest=%s",
                    cs_for_engine,
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
                        current_size=cs_for_engine,
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
                    cs_for_engine,
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
            _sparse_pass2_diagnostics.set_bpref_contribution_dump_context(
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
            local_trans_prior_center = relion_local_translation_prior_center(
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
                        current_size=cs_for_engine,
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
                per_half.update_from(k, empty_result)
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
                    local_pass1_current_size=local_pass1_current_size,
                    image_corrections_k=relion_half_inputs.image_corrections[k],
                    scale_corrections_k=relion_half_inputs.scale_corrections[k],
                    group_ids_k=relion_scale_stats_group_ids_per_half[k],
                    group_count_k=relion_scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                    translation_search_base=translation_search_base,
                    disable_adjoint_y=debug.disable_adjoint_y,
                    disable_adjoint_ctf=debug.disable_adjoint_ctf,
                    max_significants=adaptive.max_significants,
                    state=state,
                    iteration=iteration,
                    debug_iteration=numbered_relion_iteration,
                    save_intermediates_dir=debug.save_intermediates_dir,
                    local_search_random_perturbation=local_search_random_perturbation,
                    local_search_angular_sampling_deg=local_search_angular_sampling_deg,
                    local_parent_oversampling_order=local_parent_oversampling_order,
                    local_search_translation_prior_mode=local_search.local_search_translation_prior_mode,
                    replay_prior_translations=_replay_prior_translations,
                    rotation_log_prior_k=rotation_log_prior_k,
                    class_log_priors=class_log_priors,
                    k_class_enabled=k_class_enabled,
                    collect_local_search_profile=collect_local_search_profile,
                    diagnostic_score_only=bool(debug.stop_after_local_search_score_only),
                    safe_batch_sizes=_safe_batch_sizes,
                    class_assignments=class_assignments,
                    class_posterior_per_half=class_posterior_per_half,
                    class_full_posterior_per_half=class_full_posterior_per_half,
                    best_pose_rotations=best_pose_rotations,
                    best_pose_rotation_eulers=best_pose_rotation_eulers,
                    best_pose_translations=best_pose_translations,
                    local_profile_history=history.local_profile_history,
                    relion_projector_half=relion_projector_half_by_half[k],
                    relion_projector_r_max=relion_projector_r_max_by_half[k],
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
                    group_ids_k=relion_scale_stats_group_ids_per_half[k],
                    group_count_k=relion_scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                    firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
                    firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
                    cs_for_engine=cs_for_engine,
                    class_log_priors=class_log_priors,
                    k_class_enabled=k_class_enabled,
                    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                    disable_adjoint_y=debug.disable_adjoint_y,
                    disable_adjoint_ctf=debug.disable_adjoint_ctf,
                    safe_batch_sizes=_safe_batch_sizes,
                    max_significants=adaptive.max_significants,
                    noise_stats_per_half_per_class=noise_stats_per_half_per_class,
                    class_assignments=class_assignments,
                    class_posterior_per_half=class_posterior_per_half,
                    class_full_posterior_per_half=class_full_posterior_per_half,
                    class_rotation_posterior_per_half=class_rotation_posterior_per_half,
                    best_pose_rotations=best_pose_rotations,
                    best_pose_rotation_eulers=best_pose_rotation_eulers,
                    best_pose_translations=best_pose_translations,
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
                    group_ids_k=relion_scale_stats_group_ids_per_half[k],
                    group_count_k=relion_scale_stats_group_count_per_half[k],
                    scale_correction_data_vs_prior=scale_correction_data_vs_prior_this_iter,
                    firstiter_score_mode_this_iter=firstiter_score_mode_this_iter,
                    firstiter_winner_take_all_this_iter=firstiter_winner_take_all_this_iter,
                    cs_for_engine=cs_for_engine,
                    class_log_priors=class_log_priors,
                    k_class_enabled=k_class_enabled,
                    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                    disable_adjoint_y=debug.disable_adjoint_y,
                    disable_adjoint_ctf=debug.disable_adjoint_ctf,
                    safe_batch_sizes=_safe_batch_sizes,
                    max_significants=adaptive.max_significants,
                    noise_stats_per_half_per_class=noise_stats_per_half_per_class,
                    class_assignments=class_assignments,
                    class_posterior_per_half=class_posterior_per_half,
                    class_full_posterior_per_half=class_full_posterior_per_half,
                    class_rotation_posterior_per_half=class_rotation_posterior_per_half,
                    best_pose_rotations=best_pose_rotations,
                    best_pose_rotation_eulers=best_pose_rotation_eulers,
                    best_pose_translations=best_pose_translations,
                    preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
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
            )
            Ft_y_k = score_result.Ft_y
            Ft_ctf_k = score_result.Ft_ctf
            per_half.update_from(k, score_result)
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
                _sparse_pass2_diagnostics.flush_bpref_device_panel_accumulator(
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
            ) = _finalize_relion_follower_scale_replay_telemetry()
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
            import pathlib

            pathlib.Path(_bpref_prejoin_dir).mkdir(parents=True, exist_ok=True)
            np.savez(
                pathlib.Path(_bpref_prejoin_dir)
                / f"recovar_bpref_prejoin_it{iteration + 1:03d}.npz",
                schema=np.asarray("recovar-bpref-prejoin-v2"),
                run_id=np.asarray(os.environ.get("RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID", "unset")),
                iteration=np.int32(iteration + 1),
                current_size=np.int32(cs),
                padding_factor=np.int32(PADDING_FACTOR),
                grid_size=np.int32(grid_size),
                voxel_size=np.float32(cryo.voxel_size),
                volume_shape=np.asarray(volume_shape, dtype=np.int32),
                mstep_accumulator_shape=np.asarray(mstep_accumulator_shape, dtype=np.int32),
                Ft_y_0=np.asarray(Ft_y_0),
                Ft_y_1=np.asarray(Ft_y_1),
                Ft_ctf_0=np.asarray(Ft_ctf_0).real,
                Ft_ctf_1=np.asarray(Ft_ctf_1).real,
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
                    int(cs),
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
                            current_size=cs,
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
                    r_max=cs // 2,
                    shell_rounding="round",
                    full_half_axis=mstep_full_half_axis,
                    accumulator_volume_shape=mstep_accumulator_shape,
                )
                reconstruct_floor_stats_k = regularization._compute_relion_weight_shell_stats(
                    Ft_ctf_combined[class_idx],
                    volume_shape,
                    padding_factor=PADDING_FACTOR,
                    r_max=cs // 2,
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
                    current_size=cs,
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
                    import pathlib

                    pathlib.Path(_kclass_dump_dir).mkdir(parents=True, exist_ok=True)
                    _preserve_kclass_dump_dtype = os.environ.get(
                        "RECOVAR_KCLASS_DUMP_PRESERVE_DTYPE", ""
                    ).strip().lower() not in {"", "0", "false", "no", "off"}
                    np.savez(
                        pathlib.Path(_kclass_dump_dir)
                        / f"recovar_kclass_mstep_it{iteration + 1:03d}_c{class_idx + 1:02d}.npz",
                        iteration=np.int32(iteration + 1),
                        class_index=np.int32(class_idx + 1),
                        current_size=np.int32(cs),
                        padding_factor=np.int32(PADDING_FACTOR),
                        grid_size=np.int32(grid_size),
                        mstep_accumulator_shape=np.asarray(mstep_accumulator_shape, dtype=np.int32),
                        mstep_full_half_axis=np.int32(mstep_full_half_axis),
                        tau2_fudge=np.float64(tau2_fudge),
                        tau2_frame_scale=np.float64(kclass_tau2_frame_scale),
                        previous_mean=np.asarray(previous_means[0][class_idx], dtype=np.complex64),
                        previous_mean_half0=np.asarray(previous_means[0][class_idx], dtype=np.complex64),
                        previous_mean_half1=np.asarray(previous_means[1][class_idx], dtype=np.complex64),
                        Ft_y_combined=(
                            np.asarray(Ft_y_combined[class_idx])
                            if _preserve_kclass_dump_dtype
                            else np.asarray(Ft_y_combined[class_idx], dtype=np.complex64)
                        ),
                        Ft_ctf_0=(
                            (
                                np.asarray(Ft_ctf_0[class_idx])
                                if _preserve_kclass_dump_dtype
                                else np.asarray(Ft_ctf_0[class_idx], dtype=np.complex64)
                            )
                            if Ft_ctf_0 is not None
                            else np.empty(0, dtype=np.complex64)
                        ),
                        Ft_ctf_1=(
                            (
                                np.asarray(Ft_ctf_1[class_idx])
                                if _preserve_kclass_dump_dtype
                                else np.asarray(Ft_ctf_1[class_idx], dtype=np.complex64)
                            )
                            if Ft_ctf_1 is not None
                            else np.empty(0, dtype=np.complex64)
                        ),
                        Ft_ctf_combined=(
                            np.asarray(Ft_ctf_combined[class_idx])
                            if _preserve_kclass_dump_dtype
                            else np.asarray(Ft_ctf_combined[class_idx], dtype=np.complex64)
                        ),
                        dump_preserve_dtype=np.int32(int(_preserve_kclass_dump_dtype)),
                        tau2_shells=np.asarray(tau2_shells_recovar_frame_k, dtype=np.float64),
                        tau2_shells_relion=np.asarray(tau2_shells_relion_frame_k, dtype=np.float64),
                        tau2_source=np.asarray(kclass_tau2_source),
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
                [np.asarray(dvp, dtype=np.float32) for dvp in data_vs_prior_per_class], axis=0
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
                import pathlib

                pathlib.Path(_bpref_accum_dir).mkdir(parents=True, exist_ok=True)
                np.savez(
                    pathlib.Path(_bpref_accum_dir) / f"recovar_bpref_accum_it{iteration + 1:03d}.npz",
                    schema=np.asarray("recovar-bpref-accum-v2"),
                    run_id=np.asarray(os.environ.get("RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID", "unset")),
                    iteration=np.int32(iteration + 1),
                    current_size=np.int32(cs),
                    padding_factor=np.int32(PADDING_FACTOR),
                    grid_size=np.int32(grid_size),
                    voxel_size=np.float32(cryo.voxel_size),
                    volume_shape=np.asarray(volume_shape, dtype=np.int32),
                    mstep_accumulator_shape=np.asarray(mstep_accumulator_shape, dtype=np.int32),
                    Ft_y_0=np.asarray(Ft_y_0),
                    Ft_y_1=np.asarray(Ft_y_1),
                    Ft_ctf_0=np.asarray(Ft_ctf_0).real,
                    Ft_ctf_1=np.asarray(Ft_ctf_1).real,
                )
            current_iter_fsc = regularization.compute_relion_fsc_from_backprojector(
                Ft_y_0,
                Ft_y_1,
                Ft_ctf_0,
                Ft_ctf_1,
                volume_shape,
                padding_factor=PADDING_FACTOR,
                r_max=cs // 2,
                accumulator_volume_shape=mstep_accumulator_shape,
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
                        current_size=int(cs),
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
                    ),
                    dtype=np.float64,
                )
                tau2_fsc_for_update, solvent_fsc_details = regularization.compute_relion_solvent_corrected_true_fsc(
                    unfiltered_half_maps[0],
                    unfiltered_half_maps[1],
                    solvent_mask,
                    current_size=int(cs),
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
                    r_max=cs // 2,
                    return_details=True,
                    full_half_axis=-1 if full_half_axis is None else int(full_half_axis),
                    accumulator_volume_shape=mstep_accumulator_shape,
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
            tau2_taper_volume = jnp.asarray(tau2_taper[radial_shells], dtype=jnp.float32)
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
                current_size=cs,
                grid_size=grid_size,
            )
            dvp_res_shell = max(
                resolution_from_data_vs_prior(dvp_class, allow_high_res_recovery=False)
                for dvp_class in np.asarray(dvp_iter)
            )
        else:
            if tau2_update_details is not None and tau2_update_details.get("ssnr_shells") is not None:
                dvp_iter = np.asarray(tau2_update_details["ssnr_shells"], dtype=np.float32).copy()
            else:
                dvp_iter = np.asarray(
                    fsc_to_relion_ssnr(np.asarray(fsc, dtype=np.float32), tau2_fudge=tau2_fudge),
                    dtype=np.float32,
                )
            dvp_iter = _truncate_data_vs_prior_for_current_size(
                dvp_iter,
                current_size=cs,
                grid_size=grid_size,
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
            import pathlib

            _tau2_dump = {
                "iteration": np.int32(iteration + 1),
                "relion_iteration": np.int32(iteration + 1),
                "current_size": np.int32(cs),
                "grid_size": np.int32(grid_size),
                "voxel_size": np.float64(cryo.voxel_size),
                "pixel_res": np.float64(pixel_res),
                "res_angstrom": np.float64(
                    shell_index_to_resolution_angstrom(pixel_res, grid_size, cryo.voxel_size)
                    if pixel_res > 0.0
                    else np.inf
                ),
                "dvp_iter": np.asarray(dvp_iter, dtype=np.float64),
                "mstep_accumulator_shape": np.asarray(mstep_accumulator_shape, dtype=np.int32),
            }
            if tau2_update_details is not None:
                for _key in (
                    "fsc_shells",
                    "ssnr_shells",
                    "prior_shells",
                    "sigma2_shells",
                    "avg_weight_shells",
                    "shell_sum",
                    "shell_count",
                ):
                    if _key in tau2_update_details and tau2_update_details[_key] is not None:
                        _tau2_dump[f"tau2_{_key}"] = np.asarray(tau2_update_details[_key], dtype=np.float64)
            if tau2_update_details_per_half is not None:
                for _half_idx, _detail in enumerate(tau2_update_details_per_half):
                    if _detail is None:
                        continue
                    for _key in ("fsc_shells", "ssnr_shells", "prior_shells", "sigma2_shells"):
                        if _key in _detail and _detail[_key] is not None:
                            _tau2_dump[f"half{_half_idx + 1}_{_key}"] = np.asarray(
                                _detail[_key],
                                dtype=np.float64,
                            )
            if "fsc" in locals() and fsc is not None:
                _tau2_dump["current_iter_fsc"] = np.asarray(fsc, dtype=np.float64)
            if perturb_replay_relion_dir is not None and sealed_sampling_state is None:
                _model_path = os.path.join(
                    str(perturb_replay_relion_dir),
                    f"{perturb_replay_relion_prefix}_it{iteration + 1:03d}_half1_model.star",
                )
                _tau2_dump["relion_model_path"] = np.asarray(_model_path)
                _tau2_dump["relion_model_exists"] = np.bool_(os.path.exists(_model_path))
            if _replay_meta is not None:
                for _key, _value in _replay_meta.items():
                    try:
                        _tau2_dump[f"replay_meta_{_key}"] = np.asarray(_value)
                    except Exception:
                        _tau2_dump[f"replay_meta_{_key}"] = np.asarray(str(_value))
            pathlib.Path(_tau2_debug_dump_dir).mkdir(parents=True, exist_ok=True)
            _tau2_dump_path = (
                pathlib.Path(_tau2_debug_dump_dir) / f"recovar_tau2_debug_it{iteration + 1:03d}.npz"
            )
            np.savez(_tau2_dump_path, **_tau2_dump)
            logger.info("RELION tau2 debug dump written: %s", _tau2_dump_path)
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
        # (``_dense_global_scoring_dtype``). Euler angles are left at float32:
        # they are RELION's public per-particle metadata angles and, unlike
        # the rotation matrix/translation used directly in scoring
        # arithmetic, degree-valued float32 has ample precision relative to
        # any HEALPix grid spacing used in practice.
        _pose_state_dtype = _dense_global_scoring_dtype()
        for k in range(2):
            if best_pose_rotations[k] is not None:
                best_rots = np.asarray(best_pose_rotations[k], dtype=_pose_state_dtype)
                best_eulers = (
                    np.asarray(best_pose_rotation_eulers[k], dtype=np.float32)
                    if best_pose_rotation_eulers[k] is not None
                    else utils.R_to_relion(best_rots, degrees=True).astype(np.float32)
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
                    best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
                else:
                    best_rots = np.asarray(local_search_rotations, dtype=_pose_state_dtype)[rot_idx]
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
                best_rots = np.asarray(pose_rotations[k], dtype=_pose_state_dtype)[rot_idx]
                best_eulers = utils.R_to_relion(np.asarray(best_rots), degrees=True).astype(np.float32)
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

        current_rotation_matrices_combined = _concatenate_pose_stacks_or_none(
            new_iter_best_rotations,
            trailing_shape=(3, 3),
            label="current rotation",
        )
        previous_rotation_matrices_combined = _concatenate_pose_stacks_or_none(
            prior_iter_best_rotations,
            trailing_shape=(3, 3),
            label="previous rotation",
        )
        current_translations_pixel_combined = _concatenate_pose_stacks_or_none(
            new_iter_best_translations,
            trailing_shape=(2,),
            label="current translation",
        )
        previous_translations_pixel_combined = _concatenate_pose_stacks_or_none(
            prior_iter_best_translations,
            trailing_shape=(2,),
            label="previous translation",
        )

        if not k_class_enabled:
            history.record_data_vs_prior(np.asarray(dvp_iter, dtype=np.float32))
            previous_data_vs_prior_for_scheduling = np.asarray(dvp_iter, dtype=np.float32)

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
            cs=cs,
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
        if relion_follower_scale_state is not None and not can_update_norm_scale:
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
                do_scale_correction=relion_follower_scale_state is None,
                dtype=_dense_global_scoring_dtype(),
            )
            if relion_follower_scale_state is None:
                relion_half_inputs.image_corrections = norm_scale_update.image_corrections_per_half
                relion_half_inputs.scale_corrections = norm_scale_update.scale_corrections_per_half
                group_scale_corrections_for_dump = norm_scale_update.group_scale_corrections_per_half
            else:
                scale_xa = noise_stats_per_half[0].wsum_scale_correction_xa
                scale_aa = noise_stats_per_half[0].wsum_scale_correction_aa
                if not relion_firstiter_cc_this_iter and (scale_xa is None or scale_aa is None):
                    raise RuntimeError("Follower-local scale update requires expanded XA/AA statistics")
                relion_follower_scale_state = update_relion_follower_scales(
                    relion_follower_scale_state,
                    wsum_signal_product=scale_xa,
                    wsum_reference_power=scale_aa,
                    relion_firstiter_cc_this_iter=relion_firstiter_cc_this_iter,
                )
                follower_setup.follower_scale_state = relion_follower_scale_state
                for half_idx in range(2):
                    physical_groups = np.asarray(relion_half_inputs.group_ids[half_idx], dtype=np.int64)
                    selected_scales = select_relion_follower_scales(
                        relion_follower_scale_state,
                        group_ids=physical_groups,
                        follower_owners=relion_follower_owners_per_half[half_idx],
                    ).astype(_dense_global_scoring_dtype(), copy=False)
                    normcorr = np.asarray(
                        norm_scale_update.norm_corrections_per_half[half_idx],
                        dtype=np.float64,
                    )
                    avg_norm = float(norm_scale_update.avg_norm_correction_per_half[half_idx])
                    norm_factor = np.ones_like(normcorr, dtype=np.float64)
                    np.divide(avg_norm, normcorr, out=norm_factor, where=normcorr > 0.0)
                    relion_half_inputs.scale_corrections[half_idx] = selected_scales
                    relion_half_inputs.image_corrections[half_idx] = np.asarray(
                        norm_factor * selected_scales,
                        dtype=_dense_global_scoring_dtype(),
                    )
                group_scale_corrections_for_dump = [
                    relion_rank1_serialized_scales(relion_follower_scale_state),
                    None,
                ]
                logger.info(
                    "Strict RELION follower-scale update: rank1=%s rank2=%s; "
                    "rank1 model-STAR diagnostic retained separately",
                    _format_relion_correction_range(relion_follower_scale_state.scales[0]),
                    _format_relion_correction_range(relion_follower_scale_state.scales[1])
                    if relion_follower_scale_state.n_followers > 1
                    else "none",
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
        elif relion_follower_scale_state is not None:
            raise RuntimeError(
                "Strict RELION follower-scale mode requires normalization statistics "
                "for every numbered iteration"
            )
        if relion_follower_scale_state is not None:
            history.record_follower_scale_post_mstep(
                np.asarray(relion_follower_scale_state.scales, dtype=np.float64).copy()
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
            n_rotations=n_rot_current,
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
            _relion_res_stalls = _optimiser_meta.get("number_iter_without_resolution_gain")
            _relion_hvc_stalls = _optimiser_meta.get("number_iter_without_changing_assignments")
            if _relion_res_stalls is not None:
                state.nr_iter_wo_resol_gain = int(_relion_res_stalls)
            if _relion_hvc_stalls is not None:
                _hvc = int(_relion_hvc_stalls)
                state.nr_iter_wo_large_hidden_variable_changes = _hvc
                state.nr_iter_wo_assignment_changes = _hvc
            _relion_changes = (
                ("changes_optimal_orientations", "current_changes_optimal_orientations", float),
                ("changes_optimal_offsets", "current_changes_optimal_offsets_angstrom", float),
                ("changes_optimal_classes", "current_changes_optimal_classes", float),
                ("smallest_changes_orientations", "smallest_changes_optimal_orientations", float),
                ("smallest_changes_offsets", "smallest_changes_optimal_offsets_angstrom", float),
                ("smallest_changes_classes", "smallest_changes_optimal_classes", float),
            )
            for _meta_key, _state_attr, _cast in _relion_changes:
                _value = _optimiser_meta.get(_meta_key)
                if _value is not None:
                    setattr(state, _state_attr, _cast(_value))
            _relion_has_converged = _optimiser_meta.get("has_converged")
            if _relion_has_converged is not None:
                state.has_converged = bool(int(_relion_has_converged))

            # RELION's final all-data pass is stored as unnumbered
            # run_sampling.star/run_optimiser.star.  Numbered strict-replay
            # streams therefore end one iteration before the final pass; do not
            # request run_it{N+1}_sampling.star when RELION already recorded the
            # final convergence state in run_optimiser.star.
            if (
                perturb_replay_relion_dir is not None
                and sealed_sampling_state is None
                and not state.has_converged
            ):
                _next_sampling_star = os.path.join(
                    perturb_replay_relion_dir,
                    f"{perturb_replay_relion_prefix}_it{_optimiser_iter + 1:03d}_sampling.star",
                )
                _final_sampling_star = os.path.join(
                    perturb_replay_relion_dir,
                    f"{perturb_replay_relion_prefix}_sampling.star",
                )
                _final_optimiser_star = os.path.join(
                    perturb_replay_relion_dir,
                    f"{perturb_replay_relion_prefix}_optimiser.star",
                )
                if (
                    not os.path.exists(_next_sampling_star)
                    and os.path.exists(_final_sampling_star)
                    and os.path.exists(_final_optimiser_star)
                ):
                    try:
                        _final_optimiser_meta = read_relion_optimiser_metadata(_final_optimiser_star)
                        _final_has_converged = _final_optimiser_meta.get("has_converged")
                        if _final_has_converged is not None and bool(int(_final_has_converged)):
                            state.has_converged = True
                            logger.info(
                                "Replay override: RELION final optimiser convergence <- %s "
                                "(numbered replay ended after %s)",
                                _final_optimiser_star,
                                _optimiser_star,
                            )
                    except Exception as exc:
                        logger.warning(
                            "Replay override: failed to read final optimiser metadata from %s: %s",
                            _final_optimiser_star,
                            exc,
                        )
            logger.info(
                "Replay override: optimiser control <- %s "
                "(res_stalls=%d, hvc_stalls=%d, changes=(rot=%.3f deg, trans=%.3f A, class=%.0f), "
                "smallest=(rot=%.3f deg, trans=%.3f A, class=%.0f), converged=%s)",
                _optimiser_star,
                state.nr_iter_wo_resol_gain,
                state.nr_iter_wo_large_hidden_variable_changes,
                state.current_changes_optimal_orientations,
                state.current_changes_optimal_offsets_angstrom,
                state.current_changes_optimal_classes,
                state.smallest_changes_optimal_orientations,
                state.smallest_changes_optimal_offsets_angstrom,
                state.smallest_changes_optimal_classes,
                state.has_converged,
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
    _sparse_pass2_diagnostics.clear_bpref_contribution_dump_context()

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
        ) = _finalize_relion_follower_scale_replay_telemetry()
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
                    dtype=jnp.float32,
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
                        if k_class_enabled:
                            _prior_k = np.stack(
                                [
                                    remap_direction_prior_to_healpix_order(
                                        _prior_k[class_idx],
                                        _prior_order_k,
                                        state.healpix_order,
                                        dtype=_final_replay_prior_dtype,
                                    )
                                    for class_idx in range(n_classes)
                                ],
                                axis=0,
                            )
                        else:
                            _prior_k = remap_direction_prior_to_healpix_order(
                                _prior_k,
                                _prior_order_k,
                                state.healpix_order,
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
    if relion_follower_scale_state is not None:
        _dispatch_relion_follower_scale_for_final_all_data(
            follower_setup,
            init_relion_iteration=int(init_relion_iteration),
            numbered_iteration_count=int(len(history.current_sizes)),
            relion_half_inputs=relion_half_inputs,
        )
        relion_follower_owners_per_half = follower_setup.follower_owners_per_half
        relion_scale_stats_group_ids_per_half = follower_setup.scale_stats_group_ids_per_half
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
    final_effective_rotation_eulers = np.asarray(final_current_rotation_eulers, dtype=np.float32)
    final_effective_mstep_rotations = None
    final_base_translations = jnp.asarray(
        _translation_grid_for_class_count(
            state.translation_range,
            state.translation_step,
            n_classes=n_classes,
        ).astype(np.float32),
        dtype=jnp.float32,
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
        if replay.replay_iteration_overrides is not None:
            required_final_state = [
                os.path.join(
                    final_sampling_replay_dir,
                    f"{perturb_replay_relion_prefix}_sampling.star",
                ),
                os.path.join(
                    final_sampling_replay_dir,
                    f"{perturb_replay_relion_prefix}_optimiser.star",
                ),
            ]
            missing_final_state = [
                path for path in required_final_state if not os.path.isfile(path)
            ]
            if missing_final_state:
                raise RuntimeError(
                    "Strict RELION final all-data replay requires the unnumbered final "
                    "sampling and optimiser state; missing "
                    + ", ".join(missing_final_state)
                )
        final_sampling_candidates = [
            (
                os.path.join(
                    final_sampling_replay_dir,
                    f"{perturb_replay_relion_prefix}_it{final_sampling_relion_iteration:03d}_sampling.star",
                ),
                "final-numbered",
            ),
            (
                os.path.join(
                    final_sampling_replay_dir,
                    f"{perturb_replay_relion_prefix}_sampling.star",
                ),
                "final",
            ),
            (
                os.path.join(
                    final_sampling_replay_dir,
                    f"{perturb_replay_relion_prefix}_it{final_numbered_sampling_relion_iteration:03d}_sampling.star",
                ),
                "last-numbered",
            ),
        ]
        for candidate_path, candidate_source in final_sampling_candidates:
            if os.path.exists(candidate_path):
                final_sampling_star = candidate_path
                final_sampling_star_source = candidate_source
                break
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
                ).astype(np.float32)
                final_grid_preview = _translation_grid_for_class_count(
                    final_translation_range,
                    final_translation_step,
                    n_classes=n_classes,
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
                ).astype(np.float32),
                dtype=jnp.float32,
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
        )
        _, _, final_effective_mstep_rotations = apply_relion_rotation_perturbation_to_eulers(
            final_mstep_source_eulers,
            final_random_perturbation,
            final_angsamp_deg,
            return_mstep_rotations=True,
        )
        final_current_translations = jnp.asarray(
            apply_relion_translation_perturbation(
                np.asarray(final_base_translations),
                final_random_perturbation,
                final_translation_step,
            ),
            dtype=jnp.float32,
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
    )
    final_outs = PerHalfOutputs.empty()
    for k in range(2):
        _sparse_pass2_diagnostics.clear_bpref_contribution_dump_context()
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
        final_local_trans_prior_center = relion_local_translation_prior_center(
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
                group_ids_k=relion_scale_stats_group_ids_per_half[k],
                group_count_k=relion_scale_stats_group_count_per_half[k],
                scale_correction_data_vs_prior=previous_data_vs_prior_for_scheduling,
                translation_search_base=translation_search_base,
                disable_adjoint_y=debug.disable_adjoint_y,
                disable_adjoint_ctf=debug.disable_adjoint_ctf,
                max_significants=adaptive.max_significants,
                state=state,
                iteration=iteration + 1,
                debug_iteration=final_sampling_relion_iteration,
                save_intermediates_dir=debug.save_intermediates_dir,
                local_search_random_perturbation=final_local_search_random_perturbation,
                local_search_angular_sampling_deg=final_local_search_angular_sampling_deg,
                local_parent_oversampling_order=final_local_parent_oversampling_order,
                local_search_translation_prior_mode=local_search.local_search_translation_prior_mode,
                replay_prior_translations=None,
                rotation_log_prior_k=final_rotation_log_prior_k,
                class_log_priors=class_log_priors,
                k_class_enabled=False,
                collect_local_search_profile=collect_local_search_profile,
                diagnostic_score_only=False,
                safe_batch_sizes=_safe_batch_sizes,
                class_assignments=final_outs.class_assignments,
                class_posterior_per_half=final_outs.class_posterior,
                class_full_posterior_per_half=final_outs.class_full_posterior,
                best_pose_rotations=final_outs.best_pose_rotations,
                best_pose_rotation_eulers=final_outs.best_pose_rotation_eulers,
                best_pose_translations=final_outs.best_pose_translations,
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
                group_ids_k=relion_scale_stats_group_ids_per_half[k],
                group_count_k=relion_scale_stats_group_count_per_half[k],
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
                noise_stats_per_half_per_class=final_outs.noise_stats_per_class,
                class_assignments=final_outs.class_assignments,
                class_posterior_per_half=final_outs.class_posterior,
                class_full_posterior_per_half=final_outs.class_full_posterior,
                class_rotation_posterior_per_half=final_outs.class_rotation_posterior,
                best_pose_rotations=final_outs.best_pose_rotations,
                best_pose_rotation_eulers=final_outs.best_pose_rotation_eulers,
                best_pose_translations=final_outs.best_pose_translations,
                relion_projector_half=final_relion_projector_half_by_half[k],
                relion_projector_r_max=final_relion_projector_r_max_by_half[k],
                firstiter_coarse_current_size=final_adaptive_pass1_current_size,
                firstiter_fine_current_size=final_adaptive_pass2_current_size,
                firstiter_log_label="final all-data ",
                firstiter_updates_em_kwargs_ibs=True,
                return_best_pose_details=not k_class_enabled,
                debug_iteration=final_sampling_relion_iteration,
                preserve_bpref_particle_order=parity.preserve_bpref_particle_order,
            )
        if final_result.best_pose_translations is not None:
            final_result.best_pose_translations = _relion_metadata_translations(
                relion_half_inputs.previous_best_translations[k],
                final_result.best_pose_translations,
                dtype=_dense_global_scoring_dtype(),
            )
        final_outs.update_from(k, final_result)
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
        import pathlib

        pathlib.Path(_final_bpref_accum_dir).mkdir(parents=True, exist_ok=True)
        _final_dump = {
            "current_size": np.int32(final_current_size),
            "padding_factor": np.int32(PADDING_FACTOR),
            "projection_padding_factor": np.int32(PROJECTION_PADDING_FACTOR),
            "grid_size": np.int32(grid_size),
            "voxel_size": np.float32(cryo.voxel_size),
            "volume_shape": np.asarray(volume_shape, dtype=np.int32),
            "mstep_accumulator_shape": np.asarray(final_mstep_accumulator_shape, dtype=np.int32),
            "tau2_fudge": np.float64(tau2_fudge),
            "k_class_enabled": np.bool_(k_class_enabled),
            "grid_correct": np.bool_(final_grid_correct),
            "tau2_weight_combination": np.asarray("class_iref" if k_class_enabled else "sum"),
            "mstep_full_half_axis": np.int32(final_mstep_full_half_axis),
            "Ft_y_0": np.asarray(final_Ft_y_0),
            "Ft_y_1": np.asarray(final_Ft_y_1),
            "Ft_ctf_0": np.asarray(final_Ft_ctf_0).real,
            "Ft_ctf_1": np.asarray(final_Ft_ctf_1).real,
            "Ft_y_0_pre_lowres_join": np.asarray(final_unfiltered_Ft_y_0),
            "Ft_y_1_pre_lowres_join": np.asarray(final_unfiltered_Ft_y_1),
            "Ft_ctf_0_pre_lowres_join": np.asarray(final_unfiltered_Ft_ctf_0).real,
            "Ft_ctf_1_pre_lowres_join": np.asarray(final_unfiltered_Ft_ctf_1).real,
            "Ft_y": np.asarray(final_ft_y),
            "Ft_ctf": np.asarray(final_ft_ctf).real,
        }
        if final_iter_fsc is not None:
            _final_dump["fsc_shells"] = np.asarray(final_iter_fsc, dtype=np.float64)
        if final_tau2_update_details is not None:
            for _key in (
                "prior_shells",
                "sigma2_shells",
                "avg_weight_shells",
                "shell_sum",
                "shell_count",
                "fsc_shells",
                "ssnr_shells",
            ):
                if _key in final_tau2_update_details and final_tau2_update_details[_key] is not None:
                    _final_dump[f"tau2_{_key}"] = np.asarray(final_tau2_update_details[_key], dtype=np.float64)
        _final_dump_path = pathlib.Path(_final_bpref_accum_dir) / "recovar_final_bpref_accum.npz"
        np.savez(_final_dump_path, **_final_dump)
        logger.info("Final all-data BPref accumulators dumped: %s", _final_dump_path)

    # Reconstruct the final volume from the COMBINED Ft_y/Ft_ctf accumulators
    # at the full Nyquist resolution. Skip the join_halves step (we're already
    # combining the two halves into one dataset for this final iter).
    final_reconstruct_t0 = time.time()
    logger.info(
        "RELION final all-data reconstruction start: current_size=%d n_classes=%d",
        final_current_size,
        n_classes,
    )
    final_means_for_output = means
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
                final_Ft_ctf_0,
                final_Ft_y_0,
                volume_shape,
                PADDING_FACTOR,
                tau=final_mean_variance,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
                grid_correct=final_grid_correct,
                minres_map=RELION_MINRES_MAP,
                current_size=final_current_size,
                accumulator_volume_shape=final_mstep_accumulator_shape,
            ).reshape(-1),
            _reconstruct_volume_eager(
                final_Ft_ctf_1,
                final_Ft_y_1,
                volume_shape,
                PADDING_FACTOR,
                tau=final_mean_variance,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=PROJECTION_PADDING_FACTOR,
                grid_correct=final_grid_correct,
                minres_map=RELION_MINRES_MAP,
                current_size=final_current_size,
                accumulator_volume_shape=final_mstep_accumulator_shape,
            ).reshape(-1),
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
                final_unfiltered_Ft_ctf_0,
                final_unfiltered_Ft_y_0,
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
            ).reshape(-1),
            _reconstruct_volume_eager(
                final_unfiltered_Ft_ctf_1,
                final_unfiltered_Ft_y_1,
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
            ).reshape(-1),
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
    ) = _finalize_relion_follower_scale_replay_telemetry()

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
