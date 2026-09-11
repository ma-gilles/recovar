"""Sparse pass-2 E-step of the native InitialModel driver.

The RELION-style coarse-then-local pass 2 for InitialModel: its environment
switches, pass-2 layout, coarse-diagnostic scopes, the run itself and its
metadata/profile summaries. ``dense_adapter`` routes to it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import replace
from typing import Any

import numpy as np

from recovar.em.dense_single_volume.batch_planning import RELION_SCORE_TENSOR_FLOAT_BUDGET
from recovar.em.dense_single_volume.helpers import bpref_diagnostics
from recovar.em.dense_single_volume.helpers.coarse_score_diagnostics import (
    _coarse_selector_audit_from_full_stats,
    _with_coarse_significance_diagnostics,
)
from recovar.em.dense_single_volume.helpers.convergence import healpix_angular_step
from recovar.em.dense_single_volume.helpers.preprocessing import uses_relion_cuda_image_preprocessing
from recovar.em.dense_single_volume.helpers.resolution import compute_coarse_image_size
from recovar.em.dense_single_volume.helpers.significance import (
    CoarseGaussianGemmDiagnosticScope,
    _compute_k_class_significance_batched,
)
from recovar.em.dense_single_volume.k_class import _run_sparse_k_class_adaptive_pass2, run_local_k_class_em
from recovar.em.dense_single_volume.local_layout import build_pass2_hypothesis_layout
from recovar.em.initial_model.estep_common import (
    DenseInitialModelEstepConfig,
    DenseInitialModelEstepResult,
    _add_accumulator_weight_meta,
    _arrays_to_accumulators,
    _empty_accumulator,
    _estep_meta,
    _group_local_kwargs,
    _relion_projector_dense_rotations,
    _select_image_rows,
)
from recovar.em.sampling import (
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    get_translation_grid,
    relion_angular_sampling_deg,
    rotation_grid_n_in_planes,
    rotation_grid_size,
)

from .m_step import VdamAccumulator
from .state import InitialModelState

logger = logging.getLogger(__name__)


_EXACT_RELION_FINE_DIFF2_ENV = "RECOVAR_INITIAL_MODEL_EXACT_FINE_DIFF2"


_FLAT_LOCAL_ROWS_ENV = "RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS"


_STABLE_FLAT_ROW_CAPACITY_ENV = "RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY"


_PACKED_LOCAL_PROJECTION_ENV = "RECOVAR_INITIAL_MODEL_PACKED_LOCAL_PROJECTION"


_FUSED_PAIR_FINE_SCORE_ENV = "RECOVAR_EXACT_LOCAL_FUSED_PAIR_FINE_SCORE"


_DEFER_PACKED_VDAM_ENV = "RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM"


_PACKED_FINAL_NOISE_ENV = "RECOVAR_INITIAL_MODEL_PACKED_FINAL_NOISE"


_UNIFY_LOCAL_BUCKET_SIZES_ENV = "RECOVAR_INITIAL_MODEL_UNIFY_LOCAL_BUCKET_SIZES"


_COMPACT_SPARSE_PASS2_ENV = "RECOVAR_INITIAL_MODEL_COMPACT_SPARSE_PASS2"


_RELION_F32_COARSE_TIE_ULPS_ENV = "RECOVAR_INITIAL_MODEL_RELION_F32_COARSE_TIE_ULPS"


def _safe_coarse_significance_image_batch_size(
    requested_image_batch_size: int,
    *,
    n_classes: int,
    n_rotations: int,
    n_translations: int,
) -> int:
    """Cap InitialModel pass-1 batches by their materialized pose tensor.

    The RELION float32 coarse-posterior kernel consumes a dense
    ``image x class x rotation x translation`` tensor.  VDAM can increase its
    coarse Healpix order late in a run, so a batch that was safe on the
    previous iteration can otherwise become several times larger without any
    change to the user-selected image batch size.  Splitting only the image
    axis leaves every particle's score and reduction order unchanged.
    """

    requested = max(1, int(requested_image_batch_size))
    poses_per_image = max(
        1,
        int(n_classes) * int(n_rotations) * int(n_translations),
    )
    pose_tensor_cap = max(1, RELION_SCORE_TENSOR_FLOAT_BUDGET // poses_per_image)
    return min(requested, pose_tensor_cap)


def _exact_relion_fine_diff2_enabled() -> bool:
    """Use RELION's CUDA fine-score arithmetic unless explicitly disabled."""

    setting = os.environ.get(_EXACT_RELION_FINE_DIFF2_ENV, "1").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _flat_local_rows_enabled() -> bool:
    """Enable the packed-row exact scorer for controlled InitialModel A/B runs."""

    setting = os.environ.get(_FLAT_LOCAL_ROWS_ENV, "0").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _stable_flat_row_capacity_enabled() -> bool:
    """Use the mature dense bucket ABI as the packed-row physical capacity."""

    setting = os.environ.get(_STABLE_FLAT_ROW_CAPACITY_ENV, "0").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _packed_local_projection_enabled() -> bool:
    """Project packed fine rows directly for controlled InitialModel A/B runs."""

    setting = os.environ.get(_PACKED_LOCAL_PROJECTION_ENV, "0").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _fused_pair_fine_score_enabled() -> bool:
    """Enable the shared selected-pair exact-local scorer for controlled A/B runs."""

    setting = os.environ.get(_FUSED_PAIR_FINE_SCORE_ENV, "0").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _defer_packed_vdam_enabled() -> bool:
    """Defer VDAM noise/M-step work onto final packed support for A/B runs."""

    setting = os.environ.get(_DEFER_PACKED_VDAM_ENV, "0").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _packed_final_noise_enabled() -> bool:
    """Reduce deferred VDAM noise on final nonzero rows for controlled A/B runs."""

    setting = os.environ.get(_PACKED_FINAL_NOISE_ENV, "0").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _unify_local_bucket_sizes_enabled() -> bool:
    """Keep the proven single-shape policy unless a performance probe disables it."""

    setting = os.environ.get(_UNIFY_LOCAL_BUCKET_SIZES_ENV, "1").strip().lower()
    return setting not in {"0", "false", "no", "off"}


def _initial_model_relion_f32_coarse_tie_ulps() -> int:
    """Resolve the scoped coarse-cutoff ULP envelope for diagnostic A/B runs."""

    setting = os.environ.get(_RELION_F32_COARSE_TIE_ULPS_ENV, "0").strip()
    try:
        value = int(setting)
    except ValueError as error:
        raise ValueError(f"{_RELION_F32_COARSE_TIE_ULPS_ENV} must be an integer") from error
    if value < 0 or value > 16:
        raise ValueError(f"{_RELION_F32_COARSE_TIE_ULPS_ENV} must be in [0, 16]")
    return value


def _initial_model_relion_f32_fine_posterior_enabled(
    *,
    n_classes: int,
    relion_bpref_frame: bool,
    oversampling_order: int,
    backend_enabled: bool,
) -> bool:
    """Use the pruned native fine posterior only for an oversampled pass 2.

    RELION still executes a symbolic second pass when adaptive oversampling is
    zero, but that pass reconstructs every coarse sample selected by pass 1.
    Its coarse normalization is already supplied separately, so the pruned
    float32 fine-posterior kernel is neither required nor compatible there.
    """

    return bool(
        int(n_classes) == 1
        and relion_bpref_frame
        and int(oversampling_order) > 0
        and backend_enabled
    )


def _compact_sparse_pass2_enabled(n_classes: int, pass2_engine: str = "auto") -> bool:
    """Resolve the InitialModel pass-2 engine without changing K=1 defaults.

    ``auto`` keeps the source-faithful local K=1 reduction and uses the joint
    class-by-pose compact engine for K>1.  The legacy environment variable is
    retained as an explicit diagnostic override only when ``auto`` is used.
    """

    engine = str(pass2_engine).strip().lower()
    if engine not in {"auto", "local", "compact"}:
        raise ValueError(
            "InitialModel pass2_engine must be one of 'auto', 'local', or "
            f"'compact', got {pass2_engine!r}"
        )
    if engine != "auto":
        return engine == "compact"
    setting = os.environ.get(_COMPACT_SPARSE_PASS2_ENV)
    if setting is not None and setting.strip():
        return setting.strip().lower() not in {"0", "false", "no", "off"}
    return int(n_classes) > 1


_SPARSE_PASS2_CONTROL_KEYS = {
    "adaptive_fraction",
    "max_significants",
    "healpix_order",
    "oversampling_order",
    "translation_step",
    "random_perturbation",
    "coarse_translations",
    "coarse_translation_log_prior",
    "particle_diameter_ang",
    "pass1_healpix_order",
    "pass1_current_size",
    "return_profile",
}


def _initial_model_coarse_gemm_diagnostic_scopes(
    processing_groups,
    *,
    debug_iteration: int | None,
    current_size: int | None,
    n_classes: int,
) -> dict[int, CoarseGaussianGemmDiagnosticScope]:
    """Name every non-empty InitialModel pass-1 call without collisions."""

    nonempty_groups = []
    digest = hashlib.sha256()
    for processing_index, (halfset_idx, image_indices, reconstruction_group_ids) in enumerate(
        processing_groups
    ):
        image_indices = np.asarray(image_indices, dtype=np.int64).reshape(-1)
        if image_indices.size == 0:
            continue
        digest.update(np.asarray([processing_index, halfset_idx], dtype="<i8").tobytes())
        digest.update(image_indices.astype("<i8", copy=False).tobytes())
        if reconstruction_group_ids is not None:
            digest.update(
                np.asarray(reconstruction_group_ids, dtype="<i4").reshape(-1).tobytes()
            )
            halfset_label = "joint"
        else:
            halfset_label = f"h{int(halfset_idx):02d}"
        nonempty_groups.append((processing_index, halfset_label))
    if not nonempty_groups:
        return {}

    iteration = -1 if debug_iteration is None else int(debug_iteration)
    iteration_token = f"m{-iteration:04d}" if iteration < 0 else f"{iteration:04d}"
    size = -1 if current_size is None else int(current_size)
    size_token = f"m{-size:04d}" if size < 0 else f"{size:04d}"
    run_id = (
        f"initial_model_it{iteration_token}_cs{size_token}_k{int(n_classes):03d}_"
        f"particles{digest.hexdigest()[:16]}"
    )
    call_ids = tuple(
        f"call{call_ordinal:04d}_group{processing_index:04d}_halfset{halfset_label}"
        for call_ordinal, (processing_index, halfset_label) in enumerate(nonempty_groups)
    )
    return {
        processing_index: CoarseGaussianGemmDiagnosticScope(
            run_id=run_id,
            call_id=call_ids[call_ordinal],
            expected_call_ids=call_ids,
            finalize=call_ordinal == len(call_ids) - 1,
        )
        for call_ordinal, (processing_index, _halfset_label) in enumerate(nonempty_groups)
    }


def _pop_sparse_pass2_options(engine_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split shared dense/local-engine kwargs from InitialModel pass-2 controls."""

    cleaned = dict(engine_kwargs)
    options = {name: cleaned.pop(name) for name in list(cleaned) if name in _SPARSE_PASS2_CONTROL_KEYS}
    cleaned.pop("sparse_pass2", None)
    return cleaned, options


def _translation_step_from_grid(translations: np.ndarray) -> float:
    unique_vals = np.unique(np.asarray(translations, dtype=np.float32))
    diffs = np.diff(np.sort(unique_vals))
    diffs = diffs[diffs > 1.0e-6]
    return float(diffs.min()) if diffs.size else 1.0


_uses_relion_cuda_image_preprocessing = uses_relion_cuda_image_preprocessing


def _resolve_sparse_pass1_current_size(
    state: InitialModelState,
    group_kwargs: dict[str, Any],
    options: dict[str, Any],
) -> int | None:
    """RELION's coarse pass-1 scoring size (``image_coarse_size``) for sparse pass 2."""
    explicit = options.get("pass1_current_size")
    if explicit is not None:
        explicit = int(explicit)
        return None if explicit <= 0 or explicit >= int(state.ori_size) else explicit

    current_size = group_kwargs.get("current_size")
    particle_diameter = options.get("particle_diameter_ang")
    if particle_diameter is None:
        return current_size

    coarse_size = int(
        compute_coarse_image_size(
            healpix_angular_step(
                int(options.get("pass1_healpix_order", options.get("healpix_order", 0)))
            ),
            float(state.pixel_size),
            int(state.ori_size),
            particle_diameter=float(particle_diameter),
        )
    )
    current_limit = int(current_size) if current_size is not None else int(state.ori_size)
    coarse_size = min(max(2, coarse_size), current_limit, int(state.ori_size))
    if coarse_size % 2:
        coarse_size += 1
    return None if int(coarse_size) >= int(state.ori_size) else int(coarse_size)


def _coarse_translations_from_config(
    config: DenseInitialModelEstepConfig,
    options: dict[str, Any],
) -> np.ndarray:
    if "coarse_translations" in options:
        return np.asarray(options["coarse_translations"], dtype=np.float32)
    if "translation_step" in options:
        step = float(options["translation_step"])
    else:
        step = _translation_step_from_grid(np.asarray(config.translations, dtype=np.float32))
    max_offset = float(np.max(np.abs(np.asarray(config.translations, dtype=np.float32))))
    return get_translation_grid(max_pixel=max_offset, pixel_offset=step).astype(np.float32)


_SPARSE_PASS2_RESULT_FIELDS: tuple[tuple[str, type], ...] = (
    ("pose_assignments", np.int32),
    ("class_assignments", np.int32),
    ("best_pose_rotations", np.float32),
    ("best_pose_translations", np.float32),
    ("best_pose_rotation_ids", np.int32),
    ("significant_counts", np.int32),
)


def _sparse_pass2_estep_meta(
    halfset_results: dict[int, Any],
    selected_particle_ids_by_halfset: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Meta merger for separate exact-local K-class pseudo-halfset passes."""

    meta = _estep_meta(halfset_results)
    source_euler_rows = []
    source_euler_valid = []
    selected_particle_ids: list[np.ndarray] = []
    max_posterior: list[np.ndarray] = []
    field_lists: dict[str, list[np.ndarray]] = {attr: [] for attr, _ in _SPARSE_PASS2_RESULT_FIELDS}

    for halfset_idx, result in sorted(halfset_results.items()):
        image_ids = np.asarray(selected_particle_ids_by_halfset[int(halfset_idx)], dtype=np.int64)
        selected_particle_ids.append(image_ids)
        source = getattr(result, "best_pose_eulers_deg", None)
        if source is not None:
            source = np.asarray(source)
            if source.dtype != np.float64 or source.shape != (image_ids.size, 3) or not np.all(np.isfinite(source)):
                raise ValueError("source Euler rows must match their pseudo-halfset particle IDs")
        source_euler_rows.append(np.zeros((image_ids.size, 3), dtype=np.float64) if source is None else source)
        source_euler_valid.append(np.full(image_ids.size, source is not None, dtype=bool))
        for attr, dtype in _SPARSE_PASS2_RESULT_FIELDS:
            value = getattr(result, attr, None)
            if value is not None:
                field_lists[attr].append(np.asarray(value, dtype=dtype))
        stats = getattr(result, "stats", None)
        if stats is not None and getattr(stats, "max_posterior_per_image", None) is not None:
            max_posterior.append(np.asarray(stats.max_posterior_per_image, dtype=np.float32))
            meta[f"halfset_{halfset_idx}_pmax_mean"] = (
                float(np.mean(np.asarray(stats.max_posterior_per_image))) if image_ids.size else 0.0
            )

    def _merge(arrays: list[np.ndarray], key: str, dtype) -> None:
        if arrays:
            meta[key] = np.concatenate(arrays).astype(dtype, copy=False)

    if any(np.any(valid) for valid in source_euler_valid):
        meta["best_pose_eulers_deg"] = np.concatenate(source_euler_rows)
        meta["best_pose_eulers_valid"] = np.concatenate(source_euler_valid)
    _merge(selected_particle_ids, "selected_particle_ids", np.int64)
    for attr, dtype in _SPARSE_PASS2_RESULT_FIELDS:
        _merge(field_lists[attr], attr, dtype)
    _merge(max_posterior, "max_posterior_per_image", np.float32)
    meta["sparse_pass2"] = True
    return meta


def _with_initial_model_coarse_diagnostics(
    result,
    *,
    full_stats: dict[str, Any] | None,
    selector_audit: dict[str, Any] | None,
):
    """Carry the shared coarse result diagnostics through InitialModel pass 2."""

    stats = {} if full_stats is None else full_stats
    significant_counts = stats.get("significant_cutoff_counts")
    if significant_counts is not None:
        counts = np.asarray(significant_counts, dtype=np.int32)
        n_images = int(np.asarray(result.pose_assignments).size)
        if counts.shape != (n_images,):
            raise RuntimeError(
                "InitialModel coarse significant counts do not match pass-2 images: "
                f"{counts.shape} vs ({n_images},)",
            )
        result = result._replace(significant_counts=counts)
    return _with_coarse_significance_diagnostics(
        result,
        selector_audit=selector_audit,
        support_audit=stats.get("coarse_significance_support_audit"),
        hybrid_stats=stats.get("coarse_gaussian_gemm_hybrid"),
        exact_coarse_operand_assembly=stats.get(
            "exact_coarse_operand_assembly",
        ),
    )


def _sparse_pass2_profile_summary(
    pass1_time_s: float,
    pass2_time_s: float,
    n_significant_by_image: list[np.ndarray],
) -> dict[str, object]:
    all_counts = (
        np.concatenate([np.asarray(counts, dtype=np.int32).reshape(-1) for counts in n_significant_by_image])
        if n_significant_by_image
        else np.zeros(0, dtype=np.int32)
    )
    return {
        "pass1_time_s": float(pass1_time_s),
        "pass2_time_s": float(pass2_time_s),
        "mean_significant_samples": float(np.mean(all_counts)) if all_counts.size else 0.0,
        "max_significant_samples": int(np.max(all_counts)) if all_counts.size else 0,
    }


def _initial_model_pass2_layout(layout):
    """Scatter pass-2 posterior mass into RELION coarse direction bins."""

    parent_ids = getattr(layout, "rotation_posterior_ids_flat", None)
    if parent_ids is None:
        return layout

    n_parent_rotations = int(layout.n_global_rotations)
    n_psi = int(getattr(layout, "n_psi", 0))
    if n_parent_rotations <= 0 or n_psi <= 0 or n_parent_rotations % n_psi != 0:
        raise ValueError(
            "InitialModel pass-2 posterior grid expects parent rotations "
            f"to be direction-major with n_psi={n_psi}, got {n_parent_rotations} rotations",
        )
    parent_ids = np.asarray(parent_ids, dtype=np.int32)
    if np.any(parent_ids < 0) or np.any(parent_ids >= n_parent_rotations):
        raise ValueError("InitialModel pass-2 layout contains an invalid parent posterior id")
    direction_ids = (parent_ids // n_psi).astype(np.int32, copy=False)
    return replace(
        layout,
        n_global_rotations=n_parent_rotations // n_psi,
        rotation_posterior_ids_flat=direction_ids,
    )


def _collapse_compact_pass2_rotation_stats_to_directions(result, n_psi: int):
    """Convert shared-EM coarse orientation statistics to VDAM direction bins."""

    n_psi = int(n_psi)
    if n_psi <= 0:
        raise ValueError(f"n_psi must be positive, got {n_psi}")

    def _collapse(stats):
        values = np.asarray(stats.rotation_posterior_sums, dtype=np.float64)
        if values.size % n_psi:
            raise ValueError(
                "compact pass-2 rotation posterior count is not divisible by "
                f"n_psi={n_psi}: {values.size}"
            )
        direction_sums = values.reshape(-1, n_psi).sum(axis=1)
        return stats._replace(rotation_posterior_sums=direction_sums)

    per_class_stats = tuple(_collapse(stats) for stats in result.per_class_stats)
    return result._replace(
        per_class_stats=per_class_stats,
        stats=_collapse(result.stats),
    )


def _class_pass2_rotation_log_prior(group_kwargs: dict[str, Any], class_index: int) -> np.ndarray | None:
    class_prior = group_kwargs.get("class_rotation_log_prior")
    if class_prior is None:
        return group_kwargs.get("rotation_log_prior")
    prior = np.asarray(class_prior, dtype=np.float32)
    if prior.ndim != 2 or int(class_index) >= int(prior.shape[0]):
        raise ValueError(
            f"class_rotation_log_prior must have shape (n_classes, n_coarse_rotations), got {prior.shape}",
        )
    return prior[int(class_index)]


def _restore_zero_oversampling_coarse_metadata(
    result,
    *,
    hard_assignment: np.ndarray,
    class_assignment: np.ndarray,
    full_stats: dict[str, Any],
    coarse_rotations: np.ndarray,
    coarse_translations: np.ndarray,
    coarse_source_eulers: np.ndarray | None = None,
):
    """Keep RELION's pass-1 argmax/Pmax metadata for an os0 pass-2 M-step."""

    hard = np.asarray(hard_assignment, dtype=np.int32)
    classes = np.asarray(class_assignment, dtype=np.int32)
    n_images = int(hard.size)
    n_classes = len(result.per_class_stats)
    n_translations = int(np.asarray(coarse_translations).shape[0])
    if (
        hard.shape != (n_images,)
        or classes.shape != (n_images,)
        or n_translations <= 0
        or np.any(classes < 0)
        or np.any(classes >= n_classes)
    ):
        raise ValueError("invalid coarse assignments for zero-oversampling metadata")
    rotation_ids = hard.astype(np.int64) // n_translations
    translation_ids = hard.astype(np.int64) % n_translations
    rotations = np.asarray(coarse_rotations, dtype=np.float32)[rotation_ids]
    translations = np.asarray(coarse_translations, dtype=np.float32)[translation_ids]

    def _coarse_scalars(stats):
        return stats._replace(
            log_evidence_per_image=np.asarray(
                full_stats["log_evidence_per_image"],
                dtype=np.float32,
            ),
            best_log_score_per_image=np.asarray(
                full_stats["best_log_score_per_image"],
                dtype=np.float32,
            ),
            max_posterior_per_image=np.asarray(
                full_stats["max_posterior_per_image"],
                dtype=np.float32,
            ),
        )

    aggregate_stats = _coarse_scalars(result.stats)
    replace_kwargs = dict(
        class_assignments=classes,
        pose_assignments=hard,
        stats=aggregate_stats,
        best_pose_rotations=rotations,
        best_pose_translations=translations,
        best_pose_rotation_ids=rotation_ids.astype(np.int32),
        best_pose_eulers_deg=(
            None if coarse_source_eulers is None else np.asarray(coarse_source_eulers, dtype=np.float64)[rotation_ids]
        ),
        per_class_best_pose_eulers_deg=None,
    )
    if n_classes == 1:
        replace_kwargs.update(
            per_class_hard_assignments=hard[None, :],
            per_class_stats=(_coarse_scalars(result.per_class_stats[0]),),
            per_class_best_pose_rotations=(rotations,),
            per_class_best_pose_translations=(translations,),
            per_class_best_pose_rotation_ids=(rotation_ids.astype(np.int32),),
        )
    return result._replace(**replace_kwargs)


def _run_sparse_pass2_initial_model_estep(
    experiment_dataset,
    state: InitialModelState,
    config: DenseInitialModelEstepConfig,
    *,
    class_log_priors,
    groups: list[tuple[int, np.ndarray]],
    joint_particle_ids: np.ndarray,
    joint_halfset_ids: np.ndarray | None,
    means,
    mean_variance,
    relion_projector_half_by_class: np.ndarray | None = None,
    relion_projector_r_max: int | None = None,
    engine_kwargs: dict[str, Any],
) -> DenseInitialModelEstepResult:
    """Run RELION-style coarse significance plus exact-local K-class pass-2."""

    base_kwargs, options = _pop_sparse_pass2_options(engine_kwargs)
    healpix_order = int(options.get("healpix_order", 1))
    oversampling_order = int(options.get("oversampling_order", 1))
    if oversampling_order < 0:
        raise ValueError("sparse pass-2 requires oversampling_order >= 0")
    adaptive_fraction = float(options.get("adaptive_fraction", 0.999))
    max_significants = int(options.get("max_significants", -1))
    random_perturbation = float(options.get("random_perturbation", 0.0))
    return_profile = bool(options.get("return_profile", False))
    use_compact_sparse_pass2 = _compact_sparse_pass2_enabled(
        state.K,
        config.pass2_engine,
    )
    if int(config.exact_local_bucket_radix) not in (2, 4):
        raise ValueError("InitialModel exact_local_bucket_radix must be 2 or 4")
    if int(config.exact_local_physical_order_chunk_size) not in (0,) and int(
        config.exact_local_physical_order_chunk_size
    ) < 3:
        raise ValueError(
            "InitialModel exact_local_physical_order_chunk_size must be 0 (disabled) or at least 3"
        )
    pass1_time_s = 0.0
    pass2_time_s = 0.0
    exact_local_runtime_policy_active = False
    requested_stable_flat_row_capacity = _stable_flat_row_capacity_enabled()
    effective_stable_flat_row_capacity = False
    requested_fused_pair_fine_score = _fused_pair_fine_score_enabled()
    effective_fused_pair_fine_score = False
    coarse_gemm_aggregate_manifest_path = None
    coarse_gemm_stream_aggregate_manifest_path = None
    n_significant_by_image: list[np.ndarray] = []
    use_exact_relion_projector = relion_projector_half_by_class is not None
    if use_exact_relion_projector and relion_projector_r_max is None:
        raise ValueError("relion_projector_r_max is required with relion_projector_half_by_class")
    if config.stable_fourier_window_shapes and (
        state.K != 1
        or use_compact_sparse_pass2
        or not config.relion_bpref_frame
        or not use_exact_relion_projector
        or not config.relion_wavg_sequential_cuda
        or not _exact_relion_fine_diff2_enabled()
    ):
        raise ValueError(
            "stable Fourier-window shapes are supported only by K=1 local "
            "pass-2 with the exact RELION projector/fine scorer, CUDA Wavg, "
            "and RELION BPref output"
        )

    joint_halfset_stream = bool(
        state.pseudo_halfsets
        and state.K == 1
        and config.relion_bpref_frame
        and use_exact_relion_projector
        and not use_compact_sparse_pass2
        and joint_halfset_ids is not None
        and _uses_relion_cuda_image_preprocessing(experiment_dataset)
    )
    if joint_halfset_stream:
        joint_particle_ids = np.asarray(joint_particle_ids, dtype=np.int64)
        joint_halfset_ids = np.asarray(joint_halfset_ids, dtype=np.int32)
        if joint_halfset_ids.shape != joint_particle_ids.shape:
            raise ValueError("joint halfset ids must match the selected particle stream")
        if np.any((joint_halfset_ids != 0) & (joint_halfset_ids != 1)):
            raise ValueError("joint halfset ids must contain only 0/1 values")
        processing_groups = [(0, joint_particle_ids, joint_halfset_ids)]
    else:
        processing_groups = [
            (int(halfset_idx), np.asarray(image_indices, dtype=np.int64), None)
            for halfset_idx, image_indices in groups
        ]

    coarse_translations = _coarse_translations_from_config(config, options)
    translation_step = float(options.get("translation_step", _translation_step_from_grid(coarse_translations)))
    coarse_translation_log_prior = options.get("coarse_translation_log_prior")
    n_coarse_rotations = rotation_grid_size(healpix_order)
    if config.rotations is not None and int(np.asarray(config.rotations).shape[0]) == n_coarse_rotations:
        coarse_rotations = np.asarray(config.rotations, dtype=np.float32)
        coarse_metadata_rotations = coarse_rotations
    else:
        from recovar.em import sampling

        coarse_rotations = sampling.get_relion_hidden_rotation_grid(
            healpix_order,
            matrices=True,
        ).astype(np.float32)
        coarse_rotations = sampling.apply_relion_rotation_perturbation(
            coarse_rotations,
            random_perturbation,
            relion_angular_sampling_deg(healpix_order),
        ).astype(np.float32, copy=False)
        coarse_metadata_rotations = coarse_rotations
    # AccProjectorPlan builds coarse scorer matrices on the accelerator even
    # when adaptive_oversampling == 0 and config.rotations already contains
    # exactly the coarse grid.  Do not let that equal-size fast path retain
    # the nearby host-double matrices.  Fine scoring and weighted-sum
    # backprojection deliberately continue to use their separate host path.
    if use_exact_relion_projector:
        from recovar.em import sampling

        coarse_source_eulers = sampling.get_relion_rotation_grid_eulers(
            healpix_order,
            rotation_index_order="relion",
        )
        device_coarse_rotations = sampling._relion_adaptive_pass1_rotations(
            coarse_source_eulers,
            random_perturbation,
            relion_angular_sampling_deg(healpix_order),
        )
        if device_coarse_rotations is not None:
            coarse_rotations = device_coarse_rotations
    coarse_rotations_for_dense = np.asarray(coarse_rotations, dtype=np.float32)
    if config.relion_projector_frame and not use_exact_relion_projector:
        coarse_rotations_for_dense = _relion_projector_dense_rotations(coarse_rotations)
    accumulators: list[VdamAccumulator] = []
    halfset_results: dict[int, Any] = {}
    significance_image_batch_size = _safe_coarse_significance_image_batch_size(
        config.image_batch_size,
        n_classes=state.K,
        n_rotations=int(coarse_rotations_for_dense.shape[0]),
        n_translations=int(coarse_translations.shape[0]),
    )
    if significance_image_batch_size != int(config.image_batch_size):
        logger.info(
            "InitialModel coarse significance batch sizing: requested "
            "image_batch_size=%d; using image_batch_size=%d "
            "(rotations=%d translations=%d K=%d, pose-float budget=%.1fM)",
            int(config.image_batch_size),
            significance_image_batch_size,
            int(coarse_rotations_for_dense.shape[0]),
            int(coarse_translations.shape[0]),
            int(state.K),
            RELION_SCORE_TENSOR_FLOAT_BUDGET / 1e6,
        )

    coarse_gemm_diagnostic_requested = bool(
        os.environ.get("RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR", "").strip()
        or os.environ.get(
            "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR",
            "",
        ).strip()
    )
    coarse_gemm_diagnostic_scopes = (
        _initial_model_coarse_gemm_diagnostic_scopes(
            processing_groups,
            debug_iteration=base_kwargs.get("debug_iteration"),
            current_size=(
                base_kwargs.get("current_size")
                if oversampling_order == 0
                else _resolve_sparse_pass1_current_size(
                    state,
                    base_kwargs,
                    options,
                )
            ),
            n_classes=state.K,
        )
        if coarse_gemm_diagnostic_requested
        else {}
    )
    if not coarse_gemm_diagnostic_scopes and coarse_gemm_diagnostic_requested:
        raise ValueError(
            "coarse GEMM InitialModel diagnostic has no non-empty particle group"
        )

    for processing_index, (
        halfset_idx,
        image_indices,
        reconstruction_group_ids,
    ) in enumerate(processing_groups):
        image_indices = np.asarray(image_indices, dtype=np.int64)
        if image_indices.size == 0:
            accumulators.extend(_empty_accumulator(state, k, int(halfset_idx)) for k in range(state.K))
            continue
        group_kwargs = _group_local_kwargs(
            base_kwargs,
            image_indices,
            n_images=int(experiment_dataset.n_images),
        )
        coarse_rotations_for_pass1 = coarse_rotations_for_dense
        group_dataset = experiment_dataset.subset(image_indices)
        if coarse_translation_log_prior is not None:
            coarse_prior_array = np.asarray(coarse_translation_log_prior)
            pass1_translation_log_prior = (
                coarse_translation_log_prior
                if coarse_prior_array.ndim == 1
                else _select_image_rows(
                    coarse_translation_log_prior,
                    image_indices,
                    n_images=int(experiment_dataset.n_images),
                    name="coarse_translation_log_prior",
                )
            )
        else:
            pass1_translation_log_prior = group_kwargs.get("translation_log_prior")
        # RELION only uses its reduced coarse image size when adaptive
        # oversampling is active. At os0, pass 0 and the symbolic pass 1 both
        # score the full current image size (``exp_current_image_size``).
        pass1_current_size = (
            group_kwargs.get("current_size")
            if oversampling_order == 0
            else _resolve_sparse_pass1_current_size(state, group_kwargs, options)
        )

        t0 = time.time()
        sig_result = _compute_k_class_significance_batched(
            group_dataset,
            means,
            config.noise_variance,
            coarse_rotations_for_pass1,
            coarse_translations,
            config.disc_type,
            class_log_priors=class_log_priors,
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            image_batch_size=significance_image_batch_size,
            rotation_block_size=config.rotation_block_size,
            current_size=pass1_current_size,
            score_with_masked_images=bool(group_kwargs.get("score_with_masked_images", False)),
            rotation_log_prior=group_kwargs.get("class_rotation_log_prior", group_kwargs.get("rotation_log_prior")),
            translation_log_prior=pass1_translation_log_prior,
            image_corrections=group_kwargs.get("image_corrections"),
            scale_corrections=group_kwargs.get("scale_corrections"),
            image_pre_shifts=group_kwargs.get("image_pre_shifts"),
            half_spectrum_scoring=bool(group_kwargs.get("half_spectrum_scoring", False)),
            projection_padding_factor=int(group_kwargs.get("projection_padding_factor", 1)),
            do_gridding_correction=bool(group_kwargs.get("do_gridding_correction", False)),
            square_window=bool(group_kwargs.get("square_window", False)),
            use_float64_scoring=bool(group_kwargs.get("use_float64_scoring", False)),
            relion_projector_half=relion_projector_half_by_class,
            relion_projector_r_max=relion_projector_r_max,
            debug_iteration=group_kwargs.get("debug_iteration"),
            # The supplied-map EM path enables RELION's exact CUDA coarse
            # operands/support for fresh InitialModel runs. InitialModel reaches the
            # same significance engine through this adapter, so opt into the
            # shared guarded path whenever its exact projector is active.
            relion_coarse_gaussian_default=bool(
                use_exact_relion_projector
                and _uses_relion_cuda_image_preprocessing(group_dataset)
            ),
            # Production follows RELION's exact threshold comparison.  The
            # optional score-ULP envelope is diagnostic-only: native CUDA
            # launches can straddle a near-tied cutoff, but expanding every
            # cutoff changes stable supports in other particles.
            relion_f32_coarse_tie_ulps=(
                _initial_model_relion_f32_coarse_tie_ulps()
                if state.K == 1
                and use_exact_relion_projector
                and _uses_relion_cuda_image_preprocessing(group_dataset)
                else 0
            ),
            # VDAM changes its subset size almost every iteration. Keep the
            # coarse scorer's image axis fixed so JAX reuses one executable
            # instead of compiling each tail mini-batch shape.
            pad_final_image_batch=True,
            # Reuse the same physical/logical Fourier-window policy in the
            # shared coarse significance path and exact-local pass 2.  The
            # flag remains default-off until trajectory parity is qualified.
            stable_fourier_window_shapes=bool(
                config.stable_fourier_window_shapes
            ),
            coarse_gemm_diagnostic_scope=coarse_gemm_diagnostic_scopes.get(
                processing_index,
            ),
        )
        (
            _sig_rot_any,
            _n_sig_all,
            _hard_assignment,
            _class_assignment,
            significant_sample_indices,
            _full_stats,
        ) = sig_result
        coarse_selector_audit = _coarse_selector_audit_from_full_stats(_full_stats)
        if (
            _full_stats is not None
            and "coarse_gaussian_gemm_aggregate_manifest_path" in _full_stats
        ):
            coarse_gemm_aggregate_manifest_path = _full_stats[
                "coarse_gaussian_gemm_aggregate_manifest_path"
            ]
        if (
            _full_stats is not None
            and "coarse_gaussian_gemm_stream_aggregate_manifest_path" in _full_stats
        ):
            coarse_gemm_stream_aggregate_manifest_path = _full_stats[
                "coarse_gaussian_gemm_stream_aggregate_manifest_path"
            ]
        zero_oversampling = oversampling_order == 0
        k1_zero_oversampling = state.K == 1 and zero_oversampling
        pass1_time_s += time.time() - t0
        n_significant_by_image.append(np.asarray(_n_sig_all, dtype=np.int32))

        # RELION reuses the coarse pass-1 pdf_offset for all oversampled pass-2 children.
        pass2_translation_log_prior = pass1_translation_log_prior
        pass2_fine_translation_log_prior = None
        if pass2_translation_log_prior is None and (fallback := group_kwargs.get("translation_log_prior")) is not None:
            fallback_np = np.asarray(fallback)
            if fallback_np.ndim > 0 and fallback_np.shape[-1] == int(coarse_translations.shape[0]):
                pass2_translation_log_prior = fallback
            else:
                pass2_fine_translation_log_prior = fallback

        if use_compact_sparse_pass2 and k1_zero_oversampling:
            raise ValueError("InitialModel compact sparse pass 2 does not yet support oversampling_order=0")

        local_layout = None
        fine_source_eulers = None
        fine_rotations = None
        fine_rotation_parent = None
        fine_translations = None
        fine_translation_parent = None
        if use_compact_sparse_pass2:
            fine_rotations, fine_rotation_parent, _fine_rotation_ids, fine_source_eulers = (
                get_oversampled_rotation_grid_from_samples(
                    np.arange(n_coarse_rotations, dtype=np.int64),
                    healpix_order,
                    oversampling_order=oversampling_order,
                    random_perturbation=random_perturbation,
                    return_rotation_indices=True,
                    return_source_eulers=True,
                    rotation_index_order="relion_hidden",
                )
            )
            fine_translations, fine_translation_parent = get_oversampled_translation_grid(
                coarse_translations,
                translation_step,
                oversampling_order=oversampling_order,
            )
        else:
            local_layouts = []
            for class_index in range(state.K):
                class_layout = build_pass2_hypothesis_layout(
                    significant_sample_indices[class_index],
                    n_coarse_rotations,
                    int(coarse_translations.shape[0]),
                    healpix_order,
                    coarse_translations,
                    oversampling_order=oversampling_order,
                    translation_step=translation_step,
                    rotation_log_prior=_class_pass2_rotation_log_prior(group_kwargs, class_index),
                    translation_log_prior=pass2_translation_log_prior,
                    fine_translation_log_prior=pass2_fine_translation_log_prior,
                    random_perturbation=random_perturbation,
                    rotation_index_order="relion_hidden",
                    allow_empty=True,
                )
                if config.relion_projector_frame and not use_exact_relion_projector:
                    class_layout = replace(
                        class_layout,
                        rotations_flat=_relion_projector_dense_rotations(class_layout.rotations_flat),
                    )
                local_layouts.append(_initial_model_pass2_layout(class_layout))
            local_layout = tuple(local_layouts)

        t0 = time.time()
        from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse_diagnostics

        use_exact_local_relion_operands = bool(
            state.K == 1
            and use_exact_relion_projector
            and _uses_relion_cuda_image_preprocessing(group_dataset)
        )
        exact_local_runtime_policy_active = bool(
            exact_local_runtime_policy_active
            or (not use_compact_sparse_pass2 and use_exact_local_relion_operands)
        )
        use_exact_fine_diff2 = bool(
            state.K == 1
            and use_exact_local_relion_operands
            and _exact_relion_fine_diff2_enabled()
        )
        use_flat_local_rows = bool(
            use_exact_fine_diff2 and _flat_local_rows_enabled()
        )
        if requested_fused_pair_fine_score and not use_flat_local_rows:
            raise ValueError(
                "shared fused-pair fine scoring requires exact fine diff2 and flat local rows"
            )
        use_stable_flat_row_capacity = bool(
            use_flat_local_rows and requested_stable_flat_row_capacity
        )
        effective_stable_flat_row_capacity = bool(
            effective_stable_flat_row_capacity or use_stable_flat_row_capacity
        )
        use_packed_local_projection = bool(
            use_flat_local_rows and _packed_local_projection_enabled()
        )
        use_fused_pair_fine_score = bool(
            use_flat_local_rows and requested_fused_pair_fine_score
        )
        effective_fused_pair_fine_score = bool(
            effective_fused_pair_fine_score or use_fused_pair_fine_score
        )
        bpref_diagnostics.set_bpref_contribution_dump_context(
            iteration=int(group_kwargs.get("debug_iteration", -1)),
            half=int(halfset_idx) + 1,
        )
        try:
            if use_compact_sparse_pass2:
                if pass2_fine_translation_log_prior is not None:
                    compact_translation_prior = pass2_fine_translation_log_prior
                else:
                    compact_translation_prior = pass2_translation_log_prior
                compact_engine_kwargs = dict(group_kwargs)
                compact_engine_kwargs.update(
                    {
                        "translation_log_prior": compact_translation_prior,
                        "mstep_relion_x_half": bool(config.relion_bpref_frame),
                        "mstep_subtract_ctf_projection": bool(
                            group_kwargs.get("reconstruction_subtract_projected_reference", False)
                        ),
                        "relion_fine_mstep_prune": oversampling_order > 0,
                        "relion_fine_mstep_keep_all": oversampling_order == 0,
                        "adaptive_fraction": adaptive_fraction,
                        "relion_projector_half": relion_projector_half_by_class,
                        "relion_projector_r_max": relion_projector_r_max,
                        # InitialModel's small changing subsets benefit from a
                        # stable compact-pair shape policy. Environment
                        # overrides remain authoritative for diagnostics.
                        "compact_pair_min_bucket_size_default": 1,
                        "compact_pair_tail_coalesce_max_images_default": 1024,
                        "compact_pair_tail_coalesce_max_inflation_default": 8.0,
                        "compact_pair_tail_coalesce_min_bucket_size_default": 1,
                    }
                )
                if oversampling_order == 0:
                    if "relion_f32_sum_weight" in _full_stats:
                        compact_engine_kwargs[
                            "relion_f32_normalization_sum_weight"
                        ] = np.asarray(
                            _full_stats["relion_f32_sum_weight"],
                            dtype=np.float32,
                        )
                    else:
                        # CPU/reference scorers do not expose the native CUDA
                        # denominator. Preserve their mathematically
                        # equivalent log-evidence normalization fallback.
                        compact_engine_kwargs["normalization_log_evidence"] = np.asarray(
                            _full_stats["normalization_log_evidence"],
                            dtype=np.float64,
                        )
                compact_engine_kwargs["fine_source_eulers_override"] = fine_source_eulers
                result = _run_sparse_k_class_adaptive_pass2(
                    group_dataset,
                    means,
                    mean_variance,
                    config.noise_variance,
                    coarse_rotations_for_pass1,
                    coarse_translations,
                    np.asarray(fine_rotations, dtype=np.float32),
                    None,
                    np.asarray(fine_rotation_parent, dtype=np.int64),
                    np.asarray(fine_translations),
                    np.asarray(fine_translation_parent, dtype=np.int64),
                    significant_sample_indices,
                    config.disc_type,
                    class_log_priors=class_log_priors,
                    accumulate_noise=True,
                    return_best_pose_details=True,
                    coarse_healpix_order=healpix_order,
                    oversampling_order=oversampling_order,
                    random_perturbation=random_perturbation,
                    engine_kwargs=compact_engine_kwargs,
                )
                result = _collapse_compact_pass2_rotation_stats_to_directions(
                    result,
                    rotation_grid_n_in_planes(healpix_order),
                )
            else:
                result = run_local_k_class_em(
                    group_dataset,
                    means,
                    mean_variance,
                    config.noise_variance,
                    local_layout,
                    config.disc_type,
                    class_log_priors=class_log_priors,
                    class_log_evidence=(
                        np.asarray(_full_stats["class_log_evidence_per_image"], dtype=np.float64)
                        if k1_zero_oversampling
                        else None
                    ),
                    normalization_max_posterior=(
                        np.asarray(_full_stats["max_posterior_per_image"], dtype=np.float64)
                        if k1_zero_oversampling
                        else None
                    ),
                    image_batch_size=config.image_batch_size,
                    rotation_block_size=config.rotation_block_size,
                    current_size=group_kwargs.get("current_size"),
                    accumulate_noise=True,
                    projection_padding_factor=int(group_kwargs.get("projection_padding_factor", 1)),
                    reconstruction_padding_factor=int(group_kwargs.get("reconstruction_padding_factor", 1)),
                    score_with_masked_images=bool(group_kwargs.get("score_with_masked_images", False)),
                    half_spectrum_scoring=bool(group_kwargs.get("half_spectrum_scoring", False)),
                    use_float64_scoring=bool(group_kwargs.get("use_float64_scoring", False)),
                    use_float64_normalization=True,
                    use_float64_projections=bool(group_kwargs.get("use_float64_projections", False)),
                    do_gridding_correction=bool(group_kwargs.get("do_gridding_correction", False)),
                    square_window=bool(group_kwargs.get("square_window", False)),
                    recon_exact_radius=bool(group_kwargs.get("recon_exact_radius", True)),
                    image_corrections=group_kwargs.get("image_corrections"),
                    scale_corrections=group_kwargs.get("scale_corrections"),
                    image_pre_shifts=group_kwargs.get("image_pre_shifts"),
                    # InitialModel adds the unmodeled image-power spectrum once per particle.
                    unweighted_high_shell_image_power=True,
                    mstep_subtract_ctf_projection=bool(
                        group_kwargs.get("reconstruction_subtract_projected_reference", False)
                    ),
                    mstep_relion_x_half=bool(config.relion_bpref_frame),
                    # InitialModel consumes these accumulators on the host. Host
                    # x=0 enforcement and layout expansion avoid compiling two
                    # new volume-shaped JAX programs at every resolution step.
                    host_accumulator_finalize=True,
                    relion_f32_fine_posterior=_initial_model_relion_f32_fine_posterior_enabled(
                        n_classes=state.K,
                        relion_bpref_frame=config.relion_bpref_frame,
                        oversampling_order=oversampling_order,
                        backend_enabled=sparse_diagnostics.relion_x_half_f32_fine_posterior_enabled(default=True),
                    ),
                    # RELION's symbolic pass 2 at os0 retains every sample selected
                    # by pass 1. It does not apply another adaptive-fraction prune.
                    reconstruct_significant_only=not k1_zero_oversampling,
                    adaptive_fraction=adaptive_fraction,
                    debug_iteration=int(group_kwargs.get("debug_iteration", -1)),
                    # RELION's gradient InitialModel cap defines the coarse pass-1
                    # support only. Fine pass-2 reconstruction uses adaptive_fraction
                    # without reapplying maximum_significants.
                    max_significants=-1,
                    # A joint halfset stream must remain one physical bucket
                    # sequence; changing bucket shapes would restart RELION's
                    # pool-of-three phase at an artificial FFI boundary.
                    unify_local_bucket_sizes=(
                        int(config.exact_local_physical_order_chunk_size) == 0
                        if reconstruction_group_ids is not None
                        else _unify_local_bucket_sizes_enabled()
                    ),
                    stats_use_reconstruction_probs=True,
                    class_posterior_sums_from_noise=False,
                    return_profile=return_profile,
                    return_best_pose_details=True,
                    translation_prior_centers=group_kwargs.get("translation_prior_centers"),
                    relion_projector_half=relion_projector_half_by_class,
                    relion_projector_r_max=relion_projector_r_max,
                    projection_mask_current_image_disk=bool(
                        group_kwargs.get("projection_mask_current_image_disk", True)
                    ),
                    relion_exact_bpref_operands=bool(use_exact_local_relion_operands),
                    preserve_bpref_particle_order=bool(
                        use_exact_local_relion_operands and config.relion_bpref_frame
                    ),
                    reconstruction_group_ids=reconstruction_group_ids,
                    reconstruction_group_count=(
                        2 if reconstruction_group_ids is not None else None
                    ),
                    relion_exact_fine_diff2=use_exact_fine_diff2,
                    relion_exact_score_translation=use_exact_fine_diff2,
                    _flat_local_rows_enabled=use_flat_local_rows,
                    _stable_flat_row_capacity_enabled=(
                        use_stable_flat_row_capacity
                    ),
                    _packed_local_projection_enabled=use_packed_local_projection,
                    fused_pair_fine_score=use_fused_pair_fine_score,
                    _defer_packed_vdam_enabled=bool(
                        use_packed_local_projection
                        and _defer_packed_vdam_enabled()
                    ),
                    _packed_final_noise_enabled=bool(
                        use_exact_fine_diff2
                        and _flat_local_rows_enabled()
                        and _packed_local_projection_enabled()
                        and _defer_packed_vdam_enabled()
                        and _packed_final_noise_enabled()
                    ),
                    relion_wavg_sequential_cuda=(
                        bool(config.relion_wavg_sequential_cuda)
                        if use_exact_local_relion_operands
                        else None
                    ),
                    exact_local_bucket_radix=(
                        int(config.exact_local_bucket_radix)
                        if use_exact_local_relion_operands
                        else None
                    ),
                    stable_fourier_window_shapes=bool(
                        config.stable_fourier_window_shapes
                        and use_exact_local_relion_operands
                        and use_exact_fine_diff2
                    ),
                    consecutive_mixed_bucket_size=(
                        int(config.exact_local_physical_order_chunk_size)
                        if reconstruction_group_ids is not None
                        and int(config.exact_local_physical_order_chunk_size) > 0
                        else None
                    ),
                )
        finally:
            bpref_diagnostics.clear_bpref_contribution_dump_context()
        pass2_time_s += time.time() - t0
        if zero_oversampling:
            result = _restore_zero_oversampling_coarse_metadata(
                result,
                hard_assignment=_hard_assignment,
                class_assignment=_class_assignment,
                full_stats=_full_stats,
                coarse_source_eulers=get_oversampled_rotation_grid_from_samples(
                    np.arange(n_coarse_rotations),
                    healpix_order,
                    oversampling_order=0,
                    random_perturbation=random_perturbation,
                    rotation_index_order="relion_hidden",
                    return_source_eulers=True,
                )[-1],
                coarse_rotations=coarse_metadata_rotations,
                coarse_translations=coarse_translations,
            )
        result = _with_initial_model_coarse_diagnostics(
            result,
            full_stats=_full_stats,
            selector_audit=coarse_selector_audit,
        )
        halfset_results[int(halfset_idx)] = result
        accumulators.extend(
            _arrays_to_accumulators(
                result.Ft_y,
                result.Ft_ctf,
                state,
                halfset_idx=(
                    None if reconstruction_group_ids is not None else int(halfset_idx)
                ),
                reconstruction_group_count=(
                    2 if reconstruction_group_ids is not None else None
                ),
                relion_bpref_frame=config.relion_bpref_frame,
                relion_projector_frame=config.relion_projector_frame,
                padding_factor=config.padding_factor,
            )
        )

    selected_particle_ids_by_result = (
        {0: np.asarray(joint_particle_ids, dtype=np.int64)}
        if joint_halfset_stream
        else {
            int(group_index): np.asarray(image_ids, dtype=np.int64)
            for group_index, image_ids in groups
        }
    )
    meta = _sparse_pass2_estep_meta(
        halfset_results,
        selected_particle_ids_by_result,
    )
    if joint_halfset_stream:
        meta["halfset_ids"] = (0, 1)
        meta["joint_halfset_particle_stream"] = True
    _add_accumulator_weight_meta(meta, accumulators, state.K)
    meta["pass2_engine"] = "compact" if use_compact_sparse_pass2 else "local"
    meta["requested_relion_wavg_sequential_cuda"] = bool(
        config.relion_wavg_sequential_cuda
    )
    meta["requested_exact_local_bucket_radix"] = int(
        config.exact_local_bucket_radix
    )
    meta["requested_stable_fourier_window_shapes"] = bool(
        config.stable_fourier_window_shapes
    )
    meta["requested_stable_flat_row_capacity"] = bool(
        requested_stable_flat_row_capacity
    )
    meta["requested_fused_pair_fine_score"] = bool(
        requested_fused_pair_fine_score
    )
    meta["requested_exact_local_physical_order_chunk_size"] = int(
        config.exact_local_physical_order_chunk_size
    )
    meta["effective_relion_wavg_sequential_cuda"] = bool(
        exact_local_runtime_policy_active and config.relion_wavg_sequential_cuda
    )
    meta["effective_exact_local_bucket_radix"] = (
        int(config.exact_local_bucket_radix)
        if exact_local_runtime_policy_active
        else None
    )
    meta["effective_stable_fourier_window_shapes"] = bool(
        exact_local_runtime_policy_active
        and config.stable_fourier_window_shapes
    )
    meta["effective_stable_flat_row_capacity"] = bool(
        effective_stable_flat_row_capacity
    )
    meta["effective_fused_pair_fine_score"] = bool(
        effective_fused_pair_fine_score
    )
    meta["effective_exact_local_physical_order_chunk_size"] = (
        int(config.exact_local_physical_order_chunk_size)
        if exact_local_runtime_policy_active and joint_halfset_stream
        else None
    )
    if coarse_gemm_aggregate_manifest_path is not None:
        meta["coarse_gaussian_gemm_aggregate_manifest_path"] = (
            coarse_gemm_aggregate_manifest_path
        )
    if coarse_gemm_stream_aggregate_manifest_path is not None:
        meta["coarse_gaussian_gemm_stream_aggregate_manifest_path"] = (
            coarse_gemm_stream_aggregate_manifest_path
        )
    out = DenseInitialModelEstepResult(
        accumulators=accumulators,
        meta=meta,
        halfset_results=halfset_results,
    )
    if return_profile:
        out.meta["sparse_pass2_profile_summary"] = _sparse_pass2_profile_summary(
            pass1_time_s,
            pass2_time_s,
            n_significant_by_image,
        )
    return out
