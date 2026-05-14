"""K-class EM orchestration for dense and exact-local single-volume engines."""

from __future__ import annotations

import logging
import os
import time
import inspect
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.utils.nvtx_shim import nvtx

from .em_engine import run_em
from .helpers.types import NoiseStats, RelionStats, make_noise_stats, make_relion_stats
from .local_em_engine import run_local_em_exact
from .local_layout import LocalHypothesisLayout

logger = logging.getLogger(__name__)
NVTX_DOMAIN_EM = "recovar_em"
_RUN_EM_ALLOWED_KWARGS = frozenset(inspect.signature(run_em).parameters)


class KClassEMResult(NamedTuple):
    """K-class EM sufficient statistics normalized over class and pose axes."""

    new_means: jax.Array | None
    Ft_y: jax.Array
    Ft_ctf: jax.Array
    per_class_hard_assignments: jax.Array
    class_assignments: jax.Array
    pose_assignments: jax.Array
    class_responsibilities: jax.Array
    class_posterior_sums: jax.Array
    stats: RelionStats
    per_class_stats: tuple[RelionStats, ...]
    noise_stats: tuple[NoiseStats, ...] | None
    aggregate_noise_stats: NoiseStats | None
    per_class_best_pose_rotations: tuple[jax.Array, ...] | None = None
    per_class_best_pose_translations: tuple[jax.Array, ...] | None = None
    per_class_best_pose_rotation_ids: tuple[jax.Array, ...] | None = None
    best_pose_rotations: jax.Array | None = None
    best_pose_translations: jax.Array | None = None
    best_pose_rotation_ids: jax.Array | None = None
    profile_summary: dict | None = None


class _DenseKClassScoreProbeResult(NamedTuple):
    """Score-only dense K-class probe output used before class-normalized M-steps."""

    class_log_evidence: np.ndarray
    per_class_hard_assignments: np.ndarray
    per_class_stats: tuple[RelionStats, ...]
    class_assignments: np.ndarray


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    # Guard against the all-(-inf) case which would otherwise propagate NaN
    # through ``values - max_value`` (since -inf - (-inf) = NaN).
    safe_max = np.where(np.isfinite(max_value), max_value, np.zeros_like(max_value))
    diff = np.where(np.isfinite(values), values - safe_max, -np.inf)
    sum_exp = np.sum(np.exp(diff), axis=axis)
    with np.errstate(divide="ignore"):
        log_sum = np.log(sum_exp)
    return np.squeeze(max_value, axis=axis) + log_sum


def _global_reconstruction_probability_thresholds(
    support_values_by_class: list[tuple[np.ndarray, ...]],
    class_log_evidence: np.ndarray,
    global_log_evidence: np.ndarray,
    adaptive_fraction: float,
) -> np.ndarray:
    """RELION pass-2 support threshold over the global class x pose posterior."""

    n_classes, n_images = class_log_evidence.shape
    if len(support_values_by_class) != n_classes:
        raise ValueError("support value class count does not match class_log_evidence")
    thresholds = np.full(n_images, np.inf, dtype=np.float64)
    target = float(adaptive_fraction)
    for image_index in range(n_images):
        values = []
        for class_index in range(n_classes):
            if not np.isfinite(class_log_evidence[class_index, image_index]) or not np.isfinite(
                global_log_evidence[image_index]
            ):
                continue
            class_values = np.asarray(support_values_by_class[class_index][image_index], dtype=np.float64)
            if class_values.size == 0:
                continue
            scale = np.exp(class_log_evidence[class_index, image_index] - global_log_evidence[image_index])
            scaled = class_values[class_values > 0.0] * scale
            if scaled.size:
                values.append(scaled)
        if not values:
            continue
        sorted_values = np.sort(np.concatenate(values))[::-1]
        cumulative = np.cumsum(sorted_values, dtype=np.float64)
        threshold_index = int(np.searchsorted(cumulative, target, side="right"))
        if threshold_index >= sorted_values.size:
            threshold_index = sorted_values.size - 1
        thresholds[image_index] = sorted_values[threshold_index]
    return thresholds


def _class_log_priors(n_classes: int, class_log_priors) -> np.ndarray:
    if class_log_priors is None:
        return np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)
    priors = np.asarray(class_log_priors, dtype=np.float64)
    if priors.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {priors.shape}")
    if not np.all(np.isfinite(priors)):
        raise ValueError("class_log_priors must be finite")
    return priors


def _as_class_means(means) -> jax.Array:
    means_array = jnp.asarray(means)
    if means_array.ndim != 2:
        raise ValueError(f"means must have shape (n_classes, volume_size), got {means_array.shape}")
    if int(means_array.shape[0]) < 1:
        raise ValueError("means must contain at least one class")
    return means_array


def _select_class_value(value, class_index: int, n_classes: int):
    value_array = jnp.asarray(value)
    if value_array.ndim >= 2 and int(value_array.shape[0]) == n_classes:
        return value_array[class_index]
    return value


def _select_required_class_value(value, class_index: int, n_classes: int, name: str):
    value_array = jnp.asarray(value)
    if value_array.ndim < 2 or int(value_array.shape[0]) != n_classes:
        raise ValueError(f"{name} must have leading class axis of length {n_classes}, got {value_array.shape}")
    return value_array[class_index]


def _is_class_lazy_mask(value) -> bool:
    return hasattr(value, "for_class") and hasattr(value, "shape")


def _dense_engine_kwargs_for_class(engine_kwargs: dict, class_index: int, n_classes: int) -> dict:
    kwargs = dict(engine_kwargs)
    coarse_translation_log_prior = kwargs.pop("coarse_translation_log_prior", None)
    if coarse_translation_log_prior is not None and kwargs.get("translation_log_prior") is None:
        kwargs["translation_log_prior"] = coarse_translation_log_prior
    class_rotation_log_prior = kwargs.pop("class_rotation_log_prior", None)
    if class_rotation_log_prior is not None:
        if "rotation_log_prior" in kwargs and kwargs["rotation_log_prior"] is not None:
            raise ValueError("Provide only one of rotation_log_prior or class_rotation_log_prior")
        kwargs["rotation_log_prior"] = _select_required_class_value(
            class_rotation_log_prior,
            class_index,
            n_classes,
            "class_rotation_log_prior",
        )
    class_rotation_translation_mask = kwargs.pop("class_rotation_translation_mask", None)
    if class_rotation_translation_mask is not None:
        if kwargs.get("rotation_translation_mask") is not None:
            raise ValueError(
                "Provide only one of rotation_translation_mask or class_rotation_translation_mask",
            )
        if _is_class_lazy_mask(class_rotation_translation_mask):
            if tuple(class_rotation_translation_mask.shape[:1]) != (n_classes,):
                raise ValueError(
                    "class_rotation_translation_mask must have leading class axis of length "
                    f"{n_classes}, got {class_rotation_translation_mask.shape}",
                )
            kwargs["rotation_translation_mask"] = class_rotation_translation_mask.for_class(class_index)
        else:
            mask_array = np.asarray(class_rotation_translation_mask)
            if mask_array.ndim < 3 or int(mask_array.shape[0]) != n_classes:
                raise ValueError(
                    "class_rotation_translation_mask must have leading class axis of length "
                    f"{n_classes}, got {mask_array.shape}",
                )
            kwargs["rotation_translation_mask"] = mask_array[class_index]
    # Drop any leftover InitialModel/VDAM-specific engine kwargs that run_em
    # doesn't accept (e.g. ``debug_iteration``, ``adaptive_fraction``,
    # ``recon_square_window``, ``reconstruction_subtract_projected_reference``).
    # The adaptive K-class wrapper consumes these higher up; the non-adaptive
    # path forwards directly to run_em so they have to be filtered here.
    kwargs = {k: v for k, v in kwargs.items() if k in _RUN_EM_ALLOWED_KWARGS}
    return kwargs


def _local_engine_kwargs_for_class(engine_kwargs: dict, class_index: int, n_classes: int) -> dict:
    """Select class-indexed local-engine kwargs before calling the single-class kernel."""

    kwargs = dict(engine_kwargs)
    projector_half = kwargs.get("relion_projector_half")
    if projector_half is not None:
        projector_half_arr = jnp.asarray(projector_half)
        if projector_half_arr.ndim >= 4 and int(projector_half_arr.shape[0]) == n_classes:
            kwargs["relion_projector_half"] = projector_half_arr[class_index]
    return kwargs


class _DenseScoreDumpClassLabel:
    """Temporarily label env-gated dense score dumps by K-class index."""

    def __init__(self, class_index: int):
        self._label = f"class{int(class_index):03d}"
        self._old = None

    def __enter__(self):
        self._old = os.environ.get("RECOVAR_DEBUG_PER_POSE_DUMP_LABEL")
        os.environ["RECOVAR_DEBUG_PER_POSE_DUMP_LABEL"] = _append_dense_score_dump_label(
            self._old,
            self._label,
        )

    def __exit__(self, exc_type, exc, tb):
        if self._old is None:
            os.environ.pop("RECOVAR_DEBUG_PER_POSE_DUMP_LABEL", None)
        else:
            os.environ["RECOVAR_DEBUG_PER_POSE_DUMP_LABEL"] = self._old


class _DenseScoreDumpPhaseLabel:
    """Temporarily label env-gated dense score dumps by adaptive pass."""

    def __init__(self, label: str):
        self._label = label
        self._old = None

    def __enter__(self):
        self._old = os.environ.get("RECOVAR_DEBUG_PER_POSE_DUMP_LABEL")
        os.environ["RECOVAR_DEBUG_PER_POSE_DUMP_LABEL"] = _append_dense_score_dump_label(
            self._old,
            self._label,
        )

    def __exit__(self, exc_type, exc, tb):
        if self._old is None:
            os.environ.pop("RECOVAR_DEBUG_PER_POSE_DUMP_LABEL", None)
        else:
            os.environ["RECOVAR_DEBUG_PER_POSE_DUMP_LABEL"] = self._old


class _LocalDebugDumpPhaseLabel:
    """Temporarily label env-gated exact-local debug dumps by K-class phase."""

    _ENV_NAMES = (
        "RECOVAR_LOCAL_SCORE_DUMP_LABEL",
        "RECOVAR_LOCAL_FUSED_POSTERIOR_DUMP_LABEL",
    )

    def __init__(self, label: str):
        self._label = label
        self._old: dict[str, str | None] = {}

    def __enter__(self):
        for name in self._ENV_NAMES:
            old = os.environ.get(name)
            self._old[name] = old
            if old:
                os.environ[name] = f"{old}_{self._label}"
            else:
                os.environ[name] = self._label

    def __exit__(self, exc_type, exc, tb):
        for name, old in self._old.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


def _append_dense_score_dump_label(old_label: str | None, suffix: str) -> str:
    if not old_label:
        return suffix
    return f"{old_label}_{suffix}"


def _local_layout_for_class(
    local_layout: LocalHypothesisLayout,
    class_local_rotation_log_prior,
    class_index: int,
    n_classes: int,
) -> LocalHypothesisLayout:
    if class_local_rotation_log_prior is None:
        return local_layout
    class_prior = np.asarray(
        _select_required_class_value(
            class_local_rotation_log_prior,
            class_index,
            n_classes,
            "class_local_rotation_log_prior",
        ),
        dtype=np.float32,
    ).reshape(-1)
    if class_prior.shape != local_layout.rotation_log_priors_flat.shape:
        raise ValueError(
            "class_local_rotation_log_prior per-class values must have shape "
            f"{local_layout.rotation_log_priors_flat.shape}, got {class_prior.shape}",
        )
    return LocalHypothesisLayout(
        n_global_rotations=local_layout.n_global_rotations,
        n_pixels=local_layout.n_pixels,
        n_psi=local_layout.n_psi,
        rotation_offsets=local_layout.rotation_offsets,
        rotation_ids_flat=local_layout.rotation_ids_flat,
        rotations_flat=local_layout.rotations_flat,
        rotation_log_priors_flat=class_prior,
        rotation_counts=local_layout.rotation_counts,
        translation_grid=local_layout.translation_grid,
        translation_log_priors=local_layout.translation_log_priors,
        rotation_posterior_ids_flat=local_layout.rotation_posterior_ids_flat,
        sample_mask_flat=local_layout.sample_mask_flat,
    )


def _select_local_layout_for_class(
    local_layout,
    class_local_rotation_log_prior,
    class_index: int,
    n_classes: int,
) -> LocalHypothesisLayout:
    if isinstance(local_layout, (list, tuple)):
        if len(local_layout) != n_classes:
            raise ValueError(f"local_layout must contain {n_classes} per-class layouts, got {len(local_layout)}")
        if class_local_rotation_log_prior is not None:
            raise ValueError("class_local_rotation_log_prior is redundant with per-class local layouts")
        return local_layout[class_index]
    return _local_layout_for_class(
        local_layout,
        class_local_rotation_log_prior,
        class_index,
        n_classes,
    )


def _dataset_image_count(experiment_dataset, fallback: int | None = None) -> int:
    if hasattr(experiment_dataset, "n_units"):
        return int(experiment_dataset.n_units)
    if hasattr(experiment_dataset, "n_images"):
        return int(experiment_dataset.n_images)
    if fallback is not None:
        return int(fallback)
    raise AttributeError("experiment_dataset must expose n_units or n_images")


def _reject_kwargs(kwargs: dict, names: tuple[str, ...], caller: str) -> None:
    present = sorted(name for name in names if name in kwargs)
    if present:
        raise ValueError(f"{caller} controls these arguments directly: {', '.join(present)}")


def _dense_outputs(output, *, accumulate_noise: bool):
    new_mean, hard_assignment, Ft_y, Ft_ctf, stats = output[:5]
    noise_stats = output[5] if accumulate_noise else None
    return new_mean, hard_assignment, Ft_y, Ft_ctf, stats, noise_stats


def _local_outputs(output, *, accumulate_noise: bool, return_best_pose_details: bool, return_profile: bool = False):
    Ft_y, Ft_ctf, hard_assignment = output[:3]
    next_index = 3
    best_pose_rotations = None
    best_pose_translations = None
    best_pose_rotation_ids = None
    if return_best_pose_details:
        best_pose_rotations = output[next_index]
        best_pose_translations = output[next_index + 1]
        best_pose_rotation_ids = output[next_index + 2]
        next_index += 3
    stats = output[next_index]
    next_index += 1
    noise_stats = output[next_index] if accumulate_noise else None
    if accumulate_noise:
        next_index += 1
    profile_summary = output[next_index] if return_profile else None
    return (
        Ft_y,
        Ft_ctf,
        hard_assignment,
        best_pose_rotations,
        best_pose_translations,
        best_pose_rotation_ids,
        stats,
        noise_stats,
        profile_summary,
    )


def _stack_or_none(values):
    if not values:
        return None
    return jnp.stack([jnp.asarray(value) for value in values], axis=0)


def _selected_by_class(per_class_values, class_assignments: np.ndarray):
    stacked = _stack_or_none(per_class_values)
    if stacked is None:
        return None
    image_indices = jnp.arange(class_assignments.shape[0])
    return stacked[jnp.asarray(class_assignments, dtype=jnp.int32), image_indices]


def _decode_dense_best_pose_details(hard_assignment, rotations: np.ndarray, translations: np.ndarray):
    """Decode dense flat pose IDs into the pose fields expected by RELION state."""

    hard_np = np.asarray(hard_assignment, dtype=np.int64)
    n_trans = int(np.asarray(translations).shape[0])
    if n_trans <= 0:
        raise ValueError("translations must contain at least one pose")
    rot_idx = hard_np // n_trans
    trans_idx = hard_np % n_trans
    rotations_np = np.asarray(rotations, dtype=np.float32)
    translations_np = np.asarray(translations, dtype=np.float32)
    return (
        jnp.asarray(rotations_np[rot_idx], dtype=jnp.float32),
        jnp.asarray(translations_np[trans_idx], dtype=jnp.float32),
        jnp.asarray(rot_idx, dtype=jnp.int32),
    )


def _infer_healpix_order_from_rotation_count(n_rot: int) -> int:
    from recovar.em.sampling import rotation_grid_size

    n_rot = int(n_rot)
    for order in range(16):
        if rotation_grid_size(order) == n_rot:
            return order
    raise ValueError(f"Cannot infer RELION HEALPix order from {n_rot} rotations")


def _rotation_prior_with_class_log_prior(rotation_log_prior, class_log_prior: float, n_rot: int):
    if rotation_log_prior is None:
        return np.full(int(n_rot), float(class_log_prior), dtype=np.float32)
    prior = np.asarray(rotation_log_prior, dtype=np.float32)
    return prior + np.asarray(float(class_log_prior), dtype=np.float32)


def _sparse_pose_ids_to_fine_grid(hard_assignment, best_rotation_ids, n_fine_trans: int) -> np.ndarray:
    trans_ids = np.asarray(hard_assignment, dtype=np.int64) % int(n_fine_trans)
    rot_ids = np.asarray(best_rotation_ids, dtype=np.int64)
    return (rot_ids * int(n_fine_trans) + trans_ids).astype(np.int32, copy=False)


def _run_sparse_k_class_adaptive_pass2(
    experiment_dataset,
    means_array,
    mean_variance,
    noise_variance,
    coarse_rotations_np,
    coarse_translations_np,
    fine_rotations_np,
    rot_parent_map_np,
    fine_translations_np,
    trans_parent_map_np,
    sig_sample_indices_by_class,
    disc_type: str,
    *,
    class_log_priors,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    oversampling_order: int,
    random_perturbation: float,
    engine_kwargs: dict,
) -> KClassEMResult:
    """Run K-class adaptive pass-2 over RELION significant sparse support."""

    from recovar.em.dense_single_volume.helpers.oversampling import compute_pass2_stats_sparse

    n_classes = int(means_array.shape[0])
    n_rot_coarse = int(coarse_rotations_np.shape[0])
    n_fine_trans = int(fine_translations_np.shape[0])
    healpix_order = _infer_healpix_order_from_rotation_count(n_rot_coarse)
    base_engine_kwargs = dict(engine_kwargs)

    def _class_rotation_prior(class_index: int):
        class_prior = base_engine_kwargs.get("class_rotation_log_prior")
        if class_prior is not None:
            rot_prior = _select_required_class_value(
                class_prior,
                class_index,
                n_classes,
                "class_rotation_log_prior",
            )
        else:
            rot_prior = base_engine_kwargs.get("rotation_log_prior")
        return _rotation_prior_with_class_log_prior(rot_prior, float(class_log_priors[class_index]), n_rot_coarse)

    common = dict(
        nside_level=healpix_order,
        disc_type=disc_type,
        oversampling_order=int(oversampling_order),
        current_size=base_engine_kwargs.get("current_size"),
        translation_step=None,
        score_with_masked_images=bool(base_engine_kwargs.get("score_with_masked_images", False)),
        return_stats=True,
        translation_log_prior=base_engine_kwargs.get("translation_log_prior"),
        half_spectrum_scoring=bool(base_engine_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(base_engine_kwargs.get("projection_padding_factor", 1)),
        reconstruction_padding_factor=int(base_engine_kwargs.get("reconstruction_padding_factor", 1)),
        image_corrections=base_engine_kwargs.get("image_corrections"),
        scale_corrections=base_engine_kwargs.get("scale_corrections"),
        image_pre_shifts=base_engine_kwargs.get("image_pre_shifts"),
        use_float64_scoring=bool(base_engine_kwargs.get("use_float64_scoring", False)),
        translation_prior_centers=base_engine_kwargs.get("translation_prior_centers"),
        do_gridding_correction=bool(base_engine_kwargs.get("do_gridding_correction", False)),
        square_window=bool(base_engine_kwargs.get("square_window", False)),
        relion_half_volume_mstep=bool(base_engine_kwargs.get("relion_half_volume_mstep", False)),
        random_perturbation=float(random_perturbation),
        fine_rotations_override=fine_rotations_np,
        fine_rotation_parent_override=rot_parent_map_np,
        fine_translations_override=fine_translations_np,
        fine_translation_parent_override=trans_parent_map_np,
    )

    class_log_evidence = [None] * n_classes
    class_score_log_z = [None] * n_classes
    last_class_index = n_classes - 1
    probe_t0 = time.time()
    for class_index in range(last_class_index):
        output = compute_pass2_stats_sparse(
            experiment_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            _select_class_value(noise_variance, class_index, n_classes),
            coarse_translations_np,
            sig_sample_indices_by_class[class_index],
            rotation_log_prior=_class_rotation_prior(class_index),
            accumulate_noise=False,
            return_score_log_z_only=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            **common,
        )
        log_evidence, score_log_z = output
        class_log_evidence[class_index] = np.asarray(log_evidence, dtype=np.float64)
        class_score_log_z[class_index] = np.asarray(score_log_z, dtype=np.float64)
    if last_class_index > 0:
        other_score_log_z = _logsumexp_np(np.stack(class_score_log_z[:last_class_index], axis=0), axis=0)
    else:
        other_score_log_z = np.full(_dataset_image_count(experiment_dataset), -np.inf, dtype=np.float64)
    probe_s = time.time() - probe_t0

    Ft_y = [None] * n_classes
    Ft_ctf = [None] * n_classes
    hard_assignments = [None] * n_classes
    per_class_stats = [None] * n_classes
    per_class_noise = [None] * n_classes if accumulate_noise else None
    per_class_best_pose_rotations = [None] * n_classes if return_best_pose_details else None
    per_class_best_pose_translations = [None] * n_classes if return_best_pose_details else None
    per_class_best_pose_rotation_ids = [None] * n_classes if return_best_pose_details else None

    def _store_mstep_output(class_index: int, output, *, includes_score_log_z: bool = False):
        class_Ft_y, class_Ft_ctf, hard_assignment, best_rots, best_trans, best_rot_ids, stats = output[:7]
        next_index = 7
        score_log_z = None
        if includes_score_log_z:
            score_log_z = np.asarray(output[next_index], dtype=np.float64)
            next_index += 1
        noise = output[next_index] if accumulate_noise else None
        Ft_y[class_index] = _as_host_accumulator(class_Ft_y)
        Ft_ctf[class_index] = _as_host_accumulator(class_Ft_ctf)
        hard_assignments[class_index] = _sparse_pose_ids_to_fine_grid(hard_assignment, best_rot_ids, n_fine_trans)
        per_class_stats[class_index] = stats
        if per_class_noise is not None:
            per_class_noise[class_index] = noise
        if return_best_pose_details:
            per_class_best_pose_rotations[class_index] = best_rots
            per_class_best_pose_translations[class_index] = best_trans
            per_class_best_pose_rotation_ids[class_index] = best_rot_ids
        return stats, score_log_z

    mstep_t0 = time.time()
    output = compute_pass2_stats_sparse(
        experiment_dataset,
        means_array[last_class_index],
        _select_class_value(mean_variance, last_class_index, n_classes),
        _select_class_value(noise_variance, last_class_index, n_classes),
        coarse_translations_np,
        sig_sample_indices_by_class[last_class_index],
        rotation_log_prior=_class_rotation_prior(last_class_index),
        accumulate_noise=accumulate_noise,
        normalization_other_score_log_z=other_score_log_z,
        return_score_log_z=True,
        **common,
    )
    last_stats, last_score_log_z = _store_mstep_output(last_class_index, output, includes_score_log_z=True)
    class_log_evidence[last_class_index] = np.asarray(last_stats.log_evidence_per_image, dtype=np.float64)
    class_score_log_z[last_class_index] = last_score_log_z
    global_score_log_z = np.logaddexp(other_score_log_z, last_score_log_z)

    for class_index in range(last_class_index):
        output = compute_pass2_stats_sparse(
            experiment_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            _select_class_value(noise_variance, class_index, n_classes),
            coarse_translations_np,
            sig_sample_indices_by_class[class_index],
            rotation_log_prior=_class_rotation_prior(class_index),
            accumulate_noise=accumulate_noise,
            normalization_log_z=global_score_log_z,
            **common,
        )
        _store_mstep_output(class_index, output)
    mstep_s = time.time() - mstep_t0
    logger.info(
        "Sparse adaptive K-class pass2 profile: classes=%d probe_classes=%d images=%d "
        "probe=%.1fs mstep=%.1fs total=%.1fs",
        n_classes,
        last_class_index,
        _dataset_image_count(experiment_dataset),
        probe_s,
        mstep_s,
        probe_s + mstep_s,
    )
    class_log_evidence_np = np.stack(class_log_evidence, axis=0)

    return _assemble_result(
        class_log_evidence=class_log_evidence_np,
        new_means=None,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        per_class_hard_assignments=np.stack(hard_assignments, axis=0),
        per_class_stats=tuple(per_class_stats),
        noise_stats=None if per_class_noise is None else tuple(per_class_noise),
        per_class_best_pose_rotations=per_class_best_pose_rotations,
        per_class_best_pose_translations=per_class_best_pose_translations,
        per_class_best_pose_rotation_ids=per_class_best_pose_rotation_ids,
        profile_summary={
            "sparse_adaptive_probe_s": np.float64(probe_s),
            "sparse_adaptive_mstep_s": np.float64(mstep_s),
        },
        host_accumulators=True,
    )


def _sum_noise_stats(noise_stats: tuple[NoiseStats, ...] | None) -> NoiseStats | None:
    if not noise_stats:
        return None

    def _sum_field(name: str):
        values = [getattr(stats, name) for stats in noise_stats]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Cannot aggregate mixed missing/present noise field {name}")
        return jnp.sum(jnp.stack([jnp.asarray(value) for value in values], axis=0), axis=0)

    return make_noise_stats(
        wsum_sigma2_noise=_sum_field("wsum_sigma2_noise"),
        wsum_img_power=_sum_field("wsum_img_power"),
        wsum_sigma2_offset=sum(float(stats.wsum_sigma2_offset) for stats in noise_stats),
        sumw=sum(float(stats.sumw) for stats in noise_stats),
        wsum_noise_a2=_sum_field("wsum_noise_a2"),
        wsum_noise_xa=_sum_field("wsum_noise_xa"),
    )


def _as_host_accumulator(value):
    """Copy a full-volume accumulator off GPU before retaining it."""

    return np.asarray(jax.device_get(value))


def _stack_accumulators(values, *, host: bool):
    if host:
        return np.stack([np.asarray(value) for value in values], axis=0)
    return jnp.stack([jnp.asarray(value) for value in values], axis=0)


def _sum_k_class_noise_stats(
    noise_stats: tuple[NoiseStats, ...] | None,
    class_posterior_sums: np.ndarray,
) -> NoiseStats | None:
    """Aggregate Class3D noise stats with RELION's single global sum_weight.

    Each per-class ``run_em`` call normalizes posteriors over poses within one
    class and reports ``sumw == n_images``.  RELION normalizes over the joint
    class x pose grid, so ``sum_weight`` is the sum of class responsibilities
    over images, not ``n_classes * n_images``.
    """

    aggregate = _sum_noise_stats(noise_stats)
    if aggregate is None:
        return None
    responsibilities = np.asarray(class_posterior_sums, dtype=np.float64).reshape(-1)
    relion_sumw = float(np.sum(responsibilities))
    raw_sumw = np.asarray([float(stats.sumw) for stats in noise_stats], dtype=np.float64)
    if responsibilities.shape != raw_sumw.shape:
        raise ValueError(
            "class_posterior_sums and noise_stats disagree on class count: "
            f"{responsibilities.shape[0]} vs {raw_sumw.shape[0]}",
        )
    image_power = np.zeros_like(np.asarray(aggregate.wsum_img_power, dtype=np.float64))
    for stats, responsibility, class_sumw in zip(noise_stats, responsibilities, raw_sumw, strict=True):
        if class_sumw <= 0.0:
            continue
        image_power += np.asarray(stats.wsum_img_power, dtype=np.float64) * (responsibility / class_sumw)
    return aggregate._replace(
        wsum_img_power=jnp.asarray(image_power, dtype=aggregate.wsum_img_power.dtype), sumw=relion_sumw
    )


def _assemble_result(
    *,
    class_log_evidence: np.ndarray,
    new_means,
    Ft_y,
    Ft_ctf,
    per_class_hard_assignments,
    per_class_stats: tuple[RelionStats, ...],
    noise_stats: tuple[NoiseStats, ...] | None,
    per_class_best_pose_rotations=None,
    per_class_best_pose_translations=None,
    per_class_best_pose_rotation_ids=None,
    profile_summary: dict | None = None,
    class_posterior_sums_override=None,
    host_accumulators: bool = False,
) -> KClassEMResult:
    global_log_evidence = _logsumexp_np(class_log_evidence, axis=0).astype(np.float64)
    # Guard against -inf - (-inf) = NaN when an entire (image, class) had all
    # poses masked out (e.g., RELION firstiter_cc_pass2_only_best_coarse where
    # the losing class is fully excluded by the significance mask). Treat
    # those entries as zero responsibility, matching RELION's binarized
    # weight pattern.
    diff = np.where(
        np.isfinite(global_log_evidence)[None, :] & np.isfinite(class_log_evidence),
        class_log_evidence - global_log_evidence[None, :],
        -np.inf,
    )
    class_responsibilities = np.exp(diff)
    class_posterior_sums = np.sum(class_responsibilities, axis=1)
    if class_posterior_sums_override is not None:
        class_posterior_sums = np.asarray(class_posterior_sums_override, dtype=np.float64)
        if class_posterior_sums.shape != (class_log_evidence.shape[0],):
            raise ValueError(
                "class_posterior_sums_override must have shape "
                f"({class_log_evidence.shape[0]},), got {class_posterior_sums.shape}",
            )
        if not np.all(np.isfinite(class_posterior_sums)) or np.any(class_posterior_sums < 0.0):
            raise ValueError("class_posterior_sums_override must be finite and non-negative")

    best_scores = np.stack(
        [np.asarray(stats.best_log_score_per_image, dtype=np.float64) for stats in per_class_stats],
        axis=0,
    )
    pmax = np.stack(
        [np.asarray(stats.max_posterior_per_image, dtype=np.float64) for stats in per_class_stats],
        axis=0,
    )
    class_assignments = np.argmax(best_scores, axis=0).astype(np.int32)
    image_indices = np.arange(class_assignments.shape[0])
    pose_assignments = np.asarray(per_class_hard_assignments)[class_assignments, image_indices]
    rotation_posterior_sums = jnp.sum(
        jnp.stack([jnp.asarray(stats.rotation_posterior_sums) for stats in per_class_stats], axis=0),
        axis=0,
    )
    stats = make_relion_stats(
        log_evidence_per_image=global_log_evidence,
        best_log_score_per_image=np.max(best_scores, axis=0),
        max_posterior_per_image=np.max(pmax, axis=0),
        rotation_posterior_sums=rotation_posterior_sums,
        image_dtype=jnp.float32,
    )
    best_pose_rotations = _selected_by_class(per_class_best_pose_rotations, class_assignments)
    best_pose_translations = _selected_by_class(per_class_best_pose_translations, class_assignments)
    best_pose_rotation_ids = _selected_by_class(per_class_best_pose_rotation_ids, class_assignments)
    if new_means is None or all(mean is None for mean in new_means):
        stacked_new_means = None
    elif any(mean is None for mean in new_means):
        raise ValueError("Cannot stack mixed missing/present per-class new_means")
    else:
        stacked_new_means = jnp.stack([jnp.asarray(mean) for mean in new_means], axis=0)

    aggregate_noise_stats = _sum_k_class_noise_stats(noise_stats, class_posterior_sums)

    return KClassEMResult(
        new_means=stacked_new_means,
        Ft_y=_stack_accumulators(Ft_y, host=host_accumulators),
        Ft_ctf=_stack_accumulators(Ft_ctf, host=host_accumulators),
        per_class_hard_assignments=jnp.asarray(per_class_hard_assignments, dtype=jnp.int32),
        class_assignments=jnp.asarray(class_assignments, dtype=jnp.int32),
        pose_assignments=jnp.asarray(pose_assignments, dtype=jnp.int32),
        class_responsibilities=jnp.asarray(class_responsibilities, dtype=jnp.float32),
        class_posterior_sums=jnp.asarray(class_posterior_sums, dtype=jnp.float32),
        stats=stats,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        aggregate_noise_stats=aggregate_noise_stats,
        per_class_best_pose_rotations=(
            None if per_class_best_pose_rotations is None else tuple(per_class_best_pose_rotations)
        ),
        per_class_best_pose_translations=(
            None if per_class_best_pose_translations is None else tuple(per_class_best_pose_translations)
        ),
        per_class_best_pose_rotation_ids=(
            None if per_class_best_pose_rotation_ids is None else tuple(per_class_best_pose_rotation_ids)
        ),
        best_pose_rotations=best_pose_rotations,
        best_pose_translations=best_pose_translations,
        best_pose_rotation_ids=best_pose_rotation_ids,
        profile_summary=profile_summary,
    )


def _run_dense_k_class_score_probe(
    experiment_dataset,
    means_array,
    mean_variance,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    *,
    class_log_priors=None,
    **engine_kwargs,
) -> _DenseKClassScoreProbeResult:
    """Run the shared dense K-class score-only pass.

    This evaluates each class independently and returns the same coarse
    hard assignments and best-score class assignments that the full dense
    K-class wrapper uses before its M-step.  Callers that only need those
    assignments can avoid the second reconstruction pass.
    """

    means_array = _as_class_means(means_array)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)
    base_engine_kwargs = dict(engine_kwargs)

    if (
        base_engine_kwargs.get("relion_firstiter_score_mode") == "normalized_cc"
        and bool(base_engine_kwargs.get("relion_firstiter_winner_take_all", False))
    ):
        return _run_dense_k_class_joint_firstiter_score_probe(
            experiment_dataset,
            means_array,
            noise_variance,
            rotations,
            translations,
            disc_type,
            class_log_priors=log_priors,
            engine_kwargs=base_engine_kwargs,
        )

    class_log_evidence = []
    hard_assignments = []
    per_class_stats = []
    for class_index in range(n_classes):
        class_engine_kwargs = _dense_engine_kwargs_for_class(base_engine_kwargs, class_index, n_classes)
        with _DenseScoreDumpClassLabel(class_index):
            probe = run_em(
                experiment_dataset,
                means_array[class_index],
                _select_class_value(mean_variance, class_index, n_classes),
                _select_class_value(noise_variance, class_index, n_classes),
                rotations,
                translations,
                disc_type,
                return_stats=True,
                accumulate_noise=False,
                class_log_prior=float(log_priors[class_index]),
                disable_adjoint_y=True,
                disable_adjoint_ctf=True,
                score_only=True,
                **class_engine_kwargs,
            )
        hard_assignments.append(np.asarray(probe[1], dtype=np.int32))
        stats = probe[4]
        per_class_stats.append(stats)
        class_log_evidence.append(np.asarray(stats.log_evidence_per_image, dtype=np.float64))

    per_class_hard = np.stack(hard_assignments, axis=0)
    per_class_stats_tuple = tuple(per_class_stats)
    best_scores = np.stack(
        [np.asarray(stats.best_log_score_per_image, dtype=np.float64) for stats in per_class_stats_tuple],
        axis=0,
    )
    class_assignments = np.argmax(best_scores, axis=0).astype(np.int32)

    return _DenseKClassScoreProbeResult(
        class_log_evidence=np.stack(class_log_evidence, axis=0),
        per_class_hard_assignments=per_class_hard,
        per_class_stats=per_class_stats_tuple,
        class_assignments=class_assignments,
    )


def _run_dense_k_class_joint_firstiter_score_probe(
    experiment_dataset,
    means_array,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    *,
    class_log_priors: np.ndarray,
    engine_kwargs: dict,
) -> _DenseKClassScoreProbeResult:
    """Score RELION firstiter-CC K-class coarse poses in one shared pass."""

    from .helpers.significance import _compute_k_class_significance_batched

    means_array = _as_class_means(means_array)
    n_classes = int(means_array.shape[0])
    n_rot = int(np.asarray(rotations).shape[0])
    n_images = _dataset_image_count(experiment_dataset)

    rotation_prior = engine_kwargs.get("class_rotation_log_prior")
    if rotation_prior is None:
        rotation_prior = engine_kwargs.get("rotation_log_prior")
    full_stats = _compute_k_class_significance_batched(
        experiment_dataset,
        means_array,
        noise_variance,
        rotations,
        translations,
        disc_type,
        class_log_priors=class_log_priors,
        adaptive_fraction=1.0,
        max_significants=1,
        image_batch_size=int(engine_kwargs.get("image_batch_size", 500)),
        rotation_block_size=int(engine_kwargs.get("rotation_block_size", 5000)),
        current_size=engine_kwargs.get("current_size"),
        score_with_masked_images=bool(engine_kwargs.get("score_with_masked_images", False)),
        rotation_log_prior=rotation_prior,
        translation_log_prior=engine_kwargs.get("translation_log_prior"),
        image_corrections=engine_kwargs.get("image_corrections"),
        scale_corrections=engine_kwargs.get("scale_corrections"),
        image_pre_shifts=engine_kwargs.get("image_pre_shifts"),
        half_spectrum_scoring=bool(engine_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(engine_kwargs.get("projection_padding_factor", 1)),
        do_gridding_correction=bool(engine_kwargs.get("do_gridding_correction", False)),
        square_window=bool(engine_kwargs.get("square_window", False)),
        use_float64_scoring=bool(engine_kwargs.get("use_float64_scoring", False)),
        relion_projector_half=engine_kwargs.get("relion_projector_half"),
        relion_projector_r_max=engine_kwargs.get("relion_projector_r_max"),
        score_mode="normalized_cc",
        collect_significance=False,
        return_class_best=True,
    )[-1]

    class_log_evidence = np.asarray(full_stats["class_log_evidence_per_image"], dtype=np.float64)
    per_class_hard = np.asarray(full_stats["class_hard_assignments"], dtype=np.int32)
    class_best_log_score = np.asarray(full_stats["class_best_log_score_per_image"], dtype=np.float32)
    class_assignments = np.asarray(full_stats["class_assignments"], dtype=np.int32)
    per_class_stats = tuple(
        make_relion_stats(
            log_evidence_per_image=np.asarray(class_log_evidence[class_index], dtype=np.float32),
            best_log_score_per_image=np.asarray(class_best_log_score[class_index], dtype=np.float32),
            max_posterior_per_image=np.ones(n_images, dtype=np.float32),
            rotation_posterior_sums=np.zeros(n_rot, dtype=np.float32),
        )
        for class_index in range(n_classes)
    )

    return _DenseKClassScoreProbeResult(
        class_log_evidence=class_log_evidence,
        per_class_hard_assignments=per_class_hard,
        per_class_stats=per_class_stats,
        class_assignments=class_assignments,
    )


_IMAGE_AXIS_ENGINE_KWARGS = (
    "image_corrections",
    "scale_corrections",
    "image_pre_shifts",
    "translation_prior_centers",
    "translation_log_prior",
    "rotation_log_prior",
    "normalization_log_evidence",
)


def _subset_image_axis_engine_kwargs(kwargs: dict, image_indices: np.ndarray, n_images: int) -> dict:
    """Slice image-axis kwargs when running a class-specific dataset subset."""

    out = dict(kwargs)
    image_indices = np.asarray(image_indices, dtype=np.int64)
    for name in _IMAGE_AXIS_ENGINE_KWARGS:
        value = out.get(name)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim > 0 and int(array.shape[0]) == int(n_images):
            out[name] = array[image_indices]
    return out


def _full_stats_from_subset(
    subset_stats: RelionStats,
    image_indices: np.ndarray,
    n_images: int,
    *,
    class_log_evidence: np.ndarray,
) -> RelionStats:
    image_indices = np.asarray(image_indices, dtype=np.int64)
    best = np.full(int(n_images), -np.inf, dtype=np.float32)
    pmax = np.zeros(int(n_images), dtype=np.float32)
    best[image_indices] = np.asarray(subset_stats.best_log_score_per_image, dtype=np.float32)
    pmax[image_indices] = np.asarray(subset_stats.max_posterior_per_image, dtype=np.float32)
    return make_relion_stats(
        log_evidence_per_image=np.asarray(class_log_evidence, dtype=np.float32),
        best_log_score_per_image=best,
        max_posterior_per_image=pmax,
        rotation_posterior_sums=subset_stats.rotation_posterior_sums,
    )


def _run_firstiter_global_winner_subset_pass2(
    experiment_dataset,
    means_array,
    mean_variance,
    noise_variance,
    fine_rotations_np,
    fine_translations_np,
    sig_sample_indices_by_class,
    disc_type: str,
    *,
    coarse_result: _DenseKClassScoreProbeResult,
    coarse_class_assignments: np.ndarray,
    n_rot_coarse: int,
    n_trans_coarse: int,
    n_rot_fine: int,
    n_trans_fine: int,
    rot_parent_map_np: np.ndarray,
    trans_parent_map_np: np.ndarray,
    class_log_priors,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    pass2_kwargs: dict,
) -> KClassEMResult:
    """Fine pass-2 for RELION firstiter-CC after coarse global class winners.

    RELION's firstiter-CC binarization chooses one global class x pose winner
    per image at the coarse step. The fine pass only needs to refine that
    winning class's pose, so evaluating every class for every image is pure
    overhead. This keeps the same M-step contract while reducing the expensive
    dense fine pass by roughly the number of classes.
    """

    n_classes = int(means_array.shape[0])
    n_images = int(coarse_class_assignments.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)
    rotations_np = np.asarray(fine_rotations_np, dtype=np.float32)
    translations_np = np.asarray(fine_translations_np, dtype=np.float32)

    Ft_y = []
    Ft_ctf = []
    hard_assignments = []
    per_class_stats = []
    per_class_noise = [] if accumulate_noise else None
    per_class_best_pose_rotations = [] if return_best_pose_details else None
    per_class_best_pose_translations = [] if return_best_pose_details else None
    per_class_best_pose_rotation_ids = [] if return_best_pose_details else None

    subset_counts = []
    t0 = time.time()
    for class_index in range(n_classes):
        image_indices = np.nonzero(coarse_class_assignments == class_index)[0].astype(np.int64, copy=False)
        subset_counts.append(int(image_indices.size))
        if image_indices.size == 0:
            zero = jnp.zeros_like(means_array[class_index])
            Ft_y.append(_as_host_accumulator(zero))
            Ft_ctf.append(_as_host_accumulator(jnp.zeros_like(jnp.real(means_array[class_index]))))
            hard_assignments.append(np.zeros(n_images, dtype=np.int32))
            per_class_stats.append(
                make_relion_stats(
                    log_evidence_per_image=np.asarray(coarse_result.class_log_evidence[class_index], dtype=np.float32),
                    best_log_score_per_image=np.full(n_images, -np.inf, dtype=np.float32),
                    max_posterior_per_image=np.zeros(n_images, dtype=np.float32),
                    rotation_posterior_sums=np.zeros(n_rot_fine, dtype=np.float32),
                ),
            )
            if per_class_noise is not None:
                per_class_noise.append(
                    make_noise_stats(
                        wsum_sigma2_noise=np.zeros_like(np.asarray(noise_variance), dtype=np.float32),
                        wsum_img_power=np.zeros_like(np.asarray(noise_variance), dtype=np.float32),
                        wsum_sigma2_offset=0.0,
                        sumw=0.0,
                    ),
                )
            if return_best_pose_details:
                per_class_best_pose_rotations.append(np.zeros((n_images, 3, 3), dtype=np.float32))
                per_class_best_pose_translations.append(np.zeros((n_images, 2), dtype=np.float32))
                per_class_best_pose_rotation_ids.append(np.zeros(n_images, dtype=np.int32))
            continue

        subset_dataset = experiment_dataset.subset(image_indices)
        subset_sig = [sig_sample_indices_by_class[class_index][int(i)] for i in image_indices]
        class_kwargs = _dense_engine_kwargs_for_class(pass2_kwargs, class_index, n_classes)
        class_kwargs = _subset_image_axis_engine_kwargs(class_kwargs, image_indices, n_images)
        class_kwargs["rotation_translation_mask"] = _PerClassFineGridSignificanceMask(
            significant_sample_indices=subset_sig,
            n_rot_coarse=n_rot_coarse,
            n_trans_coarse=n_trans_coarse,
            n_rot_fine=n_rot_fine,
            n_trans_fine=n_trans_fine,
            rot_parent_map=rot_parent_map_np,
            trans_parent_map=trans_parent_map_np,
            n_images=int(image_indices.size),
            class_index=class_index,
            global_winner=None,
        )
        with _DenseScoreDumpClassLabel(class_index):
            output = run_em(
                subset_dataset,
                means_array[class_index],
                _select_class_value(mean_variance, class_index, n_classes),
                _select_class_value(noise_variance, class_index, n_classes),
                rotations_np,
                translations_np,
                disc_type,
                return_stats=True,
                accumulate_noise=accumulate_noise,
                class_log_prior=float(log_priors[class_index]),
                **class_kwargs,
            )
        _new_mean, hard_subset, class_Ft_y, class_Ft_ctf, stats_subset, noise = _dense_outputs(
            output,
            accumulate_noise=accumulate_noise,
        )
        hard_full = np.zeros(n_images, dtype=np.int32)
        hard_full[image_indices] = np.asarray(hard_subset, dtype=np.int32)
        Ft_y.append(class_Ft_y)
        Ft_ctf.append(class_Ft_ctf)
        hard_assignments.append(hard_full)
        per_class_stats.append(
            _full_stats_from_subset(
                stats_subset,
                image_indices,
                n_images,
                class_log_evidence=coarse_result.class_log_evidence[class_index],
            ),
        )
        if per_class_noise is not None:
            per_class_noise.append(noise)
        if return_best_pose_details:
            best_rots, best_trans, best_rot_ids = _decode_dense_best_pose_details(
                hard_subset,
                rotations_np,
                translations_np,
            )
            best_rots_full = np.zeros((n_images, 3, 3), dtype=np.float32)
            best_trans_full = np.zeros((n_images, 2), dtype=np.float32)
            best_rot_ids_full = np.zeros(n_images, dtype=np.int32)
            best_rots_full[image_indices] = best_rots
            best_trans_full[image_indices] = best_trans
            best_rot_ids_full[image_indices] = best_rot_ids
            per_class_best_pose_rotations.append(best_rots_full)
            per_class_best_pose_translations.append(best_trans_full)
            per_class_best_pose_rotation_ids.append(best_rot_ids_full)

    logger.info(
        "Firstiter-CC global-winner subset pass2: classes=%d images=%d subset_counts=%s total=%.1fs",
        n_classes,
        n_images,
        subset_counts,
        time.time() - t0,
    )
    return _assemble_result(
        class_log_evidence=coarse_result.class_log_evidence,
        new_means=None,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        per_class_hard_assignments=np.stack(hard_assignments, axis=0),
        per_class_stats=tuple(per_class_stats),
        noise_stats=None if per_class_noise is None else tuple(per_class_noise),
        per_class_best_pose_rotations=per_class_best_pose_rotations,
        per_class_best_pose_translations=per_class_best_pose_translations,
        per_class_best_pose_rotation_ids=per_class_best_pose_rotation_ids,
        class_posterior_sums_override=np.asarray(subset_counts, dtype=np.float64),
        profile_summary={"firstiter_subset_pass2_s": np.float64(time.time() - t0)},
    )


def _run_sparse_firstiter_global_winner_subset_pass2(
    experiment_dataset,
    means_array,
    mean_variance,
    noise_variance,
    coarse_translations_np,
    fine_rotations_np,
    fine_translations_np,
    rot_parent_map_np: np.ndarray,
    trans_parent_map_np: np.ndarray,
    sig_sample_indices_by_class,
    disc_type: str,
    *,
    coarse_result: _DenseKClassScoreProbeResult,
    coarse_class_assignments: np.ndarray,
    n_rot_coarse: int,
    n_fine_trans: int,
    healpix_order: int,
    oversampling_order: int,
    class_log_priors,
    accumulate_noise: bool,
    return_best_pose_details: bool,
    pass2_kwargs: dict,
) -> KClassEMResult:
    """Sparse RELION firstiter_cc fine pass over global-winner image subsets."""

    from recovar.em.dense_single_volume.helpers.oversampling import compute_pass2_stats_sparse

    n_classes = int(means_array.shape[0])
    n_images = int(coarse_class_assignments.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)

    def _class_rotation_prior(class_index: int):
        class_prior = pass2_kwargs.get("class_rotation_log_prior")
        if class_prior is not None:
            rot_prior = _select_required_class_value(
                class_prior,
                class_index,
                n_classes,
                "class_rotation_log_prior",
            )
        else:
            rot_prior = pass2_kwargs.get("rotation_log_prior")
        return _rotation_prior_with_class_log_prior(rot_prior, float(log_priors[class_index]), n_rot_coarse)

    common = dict(
        nside_level=int(healpix_order),
        disc_type=disc_type,
        oversampling_order=int(oversampling_order),
        current_size=pass2_kwargs.get("current_size"),
        translation_step=None,
        score_with_masked_images=bool(pass2_kwargs.get("score_with_masked_images", False)),
        return_stats=True,
        half_spectrum_scoring=bool(pass2_kwargs.get("half_spectrum_scoring", False)),
        projection_padding_factor=int(pass2_kwargs.get("projection_padding_factor", 1)),
        reconstruction_padding_factor=int(pass2_kwargs.get("reconstruction_padding_factor", 1)),
        use_float64_scoring=bool(pass2_kwargs.get("use_float64_scoring", False)),
        do_gridding_correction=bool(pass2_kwargs.get("do_gridding_correction", False)),
        square_window=bool(pass2_kwargs.get("square_window", False)),
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations_np,
        fine_rotation_parent_override=rot_parent_map_np,
        fine_translations_override=fine_translations_np,
        fine_translation_parent_override=trans_parent_map_np,
        relion_half_volume_mstep=bool(pass2_kwargs.get("relion_half_volume_mstep", False)),
        relion_firstiter_score_mode="normalized_cc",
        relion_firstiter_winner_take_all=True,
    )

    Ft_y = []
    Ft_ctf = []
    hard_assignments = []
    per_class_stats = []
    per_class_noise = [] if accumulate_noise else None
    per_class_best_pose_rotations = [] if return_best_pose_details else None
    per_class_best_pose_translations = [] if return_best_pose_details else None
    per_class_best_pose_rotation_ids = [] if return_best_pose_details else None

    subset_counts = []
    t0 = time.time()
    for class_index in range(n_classes):
        image_indices = np.nonzero(coarse_class_assignments == class_index)[0].astype(np.int64, copy=False)
        subset_counts.append(int(image_indices.size))
        if image_indices.size == 0:
            zero = jnp.zeros_like(means_array[class_index])
            Ft_y.append(zero)
            Ft_ctf.append(jnp.zeros_like(jnp.real(means_array[class_index])))
            hard_assignments.append(np.zeros(n_images, dtype=np.int32))
            per_class_stats.append(
                make_relion_stats(
                    log_evidence_per_image=np.asarray(coarse_result.class_log_evidence[class_index], dtype=np.float32),
                    best_log_score_per_image=np.full(n_images, -np.inf, dtype=np.float32),
                    max_posterior_per_image=np.zeros(n_images, dtype=np.float32),
                    rotation_posterior_sums=np.zeros(n_rot_coarse, dtype=np.float32),
                ),
            )
            if per_class_noise is not None:
                class_noise = np.asarray(_select_class_value(noise_variance, class_index, n_classes), dtype=np.float32)
                per_class_noise.append(
                    make_noise_stats(
                        wsum_sigma2_noise=np.zeros_like(class_noise, dtype=np.float32),
                        wsum_img_power=np.zeros_like(class_noise, dtype=np.float32),
                        wsum_sigma2_offset=0.0,
                        sumw=0.0,
                    ),
                )
            if return_best_pose_details:
                per_class_best_pose_rotations.append(np.zeros((n_images, 3, 3), dtype=np.float32))
                per_class_best_pose_translations.append(np.zeros((n_images, 2), dtype=np.float32))
                per_class_best_pose_rotation_ids.append(np.zeros(n_images, dtype=np.int32))
            continue

        subset_dataset = experiment_dataset.subset(image_indices)
        subset_sig = [sig_sample_indices_by_class[class_index][int(i)] for i in image_indices]
        class_kwargs = _dense_engine_kwargs_for_class(pass2_kwargs, class_index, n_classes)
        class_kwargs = _subset_image_axis_engine_kwargs(class_kwargs, image_indices, n_images)

        output = compute_pass2_stats_sparse(
            subset_dataset,
            means_array[class_index],
            _select_class_value(mean_variance, class_index, n_classes),
            _select_class_value(noise_variance, class_index, n_classes),
            coarse_translations_np,
            subset_sig,
            rotation_log_prior=_class_rotation_prior(class_index),
            translation_log_prior=class_kwargs.get("translation_log_prior"),
            accumulate_noise=accumulate_noise,
            image_corrections=class_kwargs.get("image_corrections"),
            scale_corrections=class_kwargs.get("scale_corrections"),
            image_pre_shifts=class_kwargs.get("image_pre_shifts"),
            translation_prior_centers=class_kwargs.get("translation_prior_centers"),
            **common,
        )
        class_Ft_y, class_Ft_ctf, hard_subset, best_rots, best_trans, best_rot_ids, stats_subset = output[:7]
        noise = output[7] if accumulate_noise else None
        hard_full = np.zeros(n_images, dtype=np.int32)
        hard_full[image_indices] = _sparse_pose_ids_to_fine_grid(hard_subset, best_rot_ids, n_fine_trans)
        Ft_y.append(_as_host_accumulator(class_Ft_y))
        Ft_ctf.append(_as_host_accumulator(class_Ft_ctf))
        hard_assignments.append(hard_full)
        per_class_stats.append(
            _full_stats_from_subset(
                stats_subset,
                image_indices,
                n_images,
                class_log_evidence=coarse_result.class_log_evidence[class_index],
            ),
        )
        if per_class_noise is not None:
            per_class_noise.append(noise)
        if return_best_pose_details:
            best_rots_full = np.zeros((n_images, 3, 3), dtype=np.float32)
            best_trans_full = np.zeros((n_images, 2), dtype=np.float32)
            best_rot_ids_full = np.zeros(n_images, dtype=np.int32)
            best_rots_full[image_indices] = best_rots
            best_trans_full[image_indices] = best_trans
            best_rot_ids_full[image_indices] = best_rot_ids
            per_class_best_pose_rotations.append(best_rots_full)
            per_class_best_pose_translations.append(best_trans_full)
            per_class_best_pose_rotation_ids.append(best_rot_ids_full)

    logger.info(
        "Sparse firstiter-CC global-winner subset pass2: classes=%d images=%d subset_counts=%s total=%.1fs",
        n_classes,
        n_images,
        subset_counts,
        time.time() - t0,
    )
    return _assemble_result(
        class_log_evidence=coarse_result.class_log_evidence,
        new_means=None,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        per_class_hard_assignments=np.stack(hard_assignments, axis=0),
        per_class_stats=tuple(per_class_stats),
        noise_stats=None if per_class_noise is None else tuple(per_class_noise),
        per_class_best_pose_rotations=per_class_best_pose_rotations,
        per_class_best_pose_translations=per_class_best_pose_translations,
        per_class_best_pose_rotation_ids=per_class_best_pose_rotation_ids,
        class_posterior_sums_override=np.asarray(subset_counts, dtype=np.float64),
        profile_summary={"sparse_firstiter_subset_pass2_s": np.float64(time.time() - t0)},
        host_accumulators=True,
    )


@nvtx.annotate("kclass.run_dense_k_class_em", color="cyan", domain=NVTX_DOMAIN_EM)
def run_dense_k_class_em(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    rotations,
    translations,
    disc_type: str,
    *,
    class_log_priors=None,
    accumulate_noise: bool = False,
    return_best_pose_details: bool = False,
    **engine_kwargs,
) -> KClassEMResult:
    """Run dense K-class EM using ``run_em`` as the only scoring/M-step kernel."""

    _reject_kwargs(
        engine_kwargs,
        (
            "return_stats",
            "accumulate_noise",
            "class_log_prior",
            "normalization_log_evidence",
            "disable_adjoint_y",
            "disable_adjoint_ctf",
            "return_profile",
            "return_best_pose_details",
        ),
        "run_dense_k_class_em",
    )
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)
    base_engine_kwargs = dict(engine_kwargs)
    rotations_np = np.asarray(rotations, dtype=np.float32)
    translations_np = np.asarray(translations, dtype=np.float32)

    overall_t0 = time.time()
    if n_classes == 1:
        class_engine_kwargs = _dense_engine_kwargs_for_class(base_engine_kwargs, 0, n_classes)
        output = run_em(
            experiment_dataset,
            means_array[0],
            _select_class_value(mean_variance, 0, n_classes),
            _select_class_value(noise_variance, 0, n_classes),
            rotations,
            translations,
            disc_type,
            return_stats=True,
            accumulate_noise=accumulate_noise,
            class_log_prior=float(log_priors[0]),
            **class_engine_kwargs,
        )
        new_mean, hard_assignment, class_Ft_y, class_Ft_ctf, stats, noise = _dense_outputs(
            output,
            accumulate_noise=accumulate_noise,
        )
        best_pose_rotations = None
        best_pose_translations = None
        best_pose_rotation_ids = None
        if return_best_pose_details:
            best_pose_rotations, best_pose_translations, best_pose_rotation_ids = _decode_dense_best_pose_details(
                hard_assignment,
                rotations_np,
                translations_np,
            )
        logger.info(
            "Dense K-class EM profile: classes=1 images=%d rotations=%d translations=%d single_pass=%.1fs",
            _dataset_image_count(experiment_dataset),
            int(rotations_np.shape[0]),
            int(translations_np.shape[0]),
            time.time() - overall_t0,
        )
        return _assemble_result(
            class_log_evidence=np.asarray(stats.log_evidence_per_image, dtype=np.float64)[None, :],
            new_means=[new_mean],
            Ft_y=[class_Ft_y],
            Ft_ctf=[class_Ft_ctf],
            per_class_hard_assignments=np.asarray(hard_assignment, dtype=np.int32)[None, :],
            per_class_stats=(stats,),
            noise_stats=None if noise is None else (noise,),
            per_class_best_pose_rotations=None if best_pose_rotations is None else [best_pose_rotations],
            per_class_best_pose_translations=None if best_pose_translations is None else [best_pose_translations],
            per_class_best_pose_rotation_ids=None if best_pose_rotation_ids is None else [best_pose_rotation_ids],
        )

    probe_t0 = time.time()
    score_probe = _run_dense_k_class_score_probe(
        experiment_dataset,
        means_array,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        class_log_priors=log_priors,
        **base_engine_kwargs,
    )
    probe_s = time.time() - probe_t0

    class_log_evidence_np = score_probe.class_log_evidence
    global_log_evidence = _logsumexp_np(class_log_evidence_np, axis=0)

    new_means = []
    Ft_y = []
    Ft_ctf = []
    hard_assignments = []
    per_class_stats = []
    per_class_noise = [] if accumulate_noise else None
    per_class_best_pose_rotations = [] if return_best_pose_details else None
    per_class_best_pose_translations = [] if return_best_pose_details else None
    per_class_best_pose_rotation_ids = [] if return_best_pose_details else None
    mstep_t0 = time.time()
    for class_index in range(n_classes):
        class_engine_kwargs = _dense_engine_kwargs_for_class(base_engine_kwargs, class_index, n_classes)
        with _DenseScoreDumpClassLabel(class_index):
            output = run_em(
                experiment_dataset,
                means_array[class_index],
                _select_class_value(mean_variance, class_index, n_classes),
                _select_class_value(noise_variance, class_index, n_classes),
                rotations,
                translations,
                disc_type,
                return_stats=True,
                accumulate_noise=accumulate_noise,
                class_log_prior=float(log_priors[class_index]),
                normalization_log_evidence=global_log_evidence,
                **class_engine_kwargs,
            )
        new_mean, hard_assignment, class_Ft_y, class_Ft_ctf, stats, noise = _dense_outputs(
            output,
            accumulate_noise=accumulate_noise,
        )
        new_means.append(new_mean)
        Ft_y.append(class_Ft_y)
        Ft_ctf.append(class_Ft_ctf)
        hard_assignments.append(np.asarray(hard_assignment, dtype=np.int32))
        per_class_stats.append(stats)
        if per_class_noise is not None:
            per_class_noise.append(noise)
        if return_best_pose_details:
            best_rots, best_trans, best_rot_ids = _decode_dense_best_pose_details(
                hard_assignment,
                rotations_np,
                translations_np,
            )
            per_class_best_pose_rotations.append(best_rots)
            per_class_best_pose_translations.append(best_trans)
            per_class_best_pose_rotation_ids.append(best_rot_ids)
    mstep_s = time.time() - mstep_t0
    logger.info(
        "Dense K-class EM profile: classes=%d images=%d rotations=%d translations=%d probe=%.1fs mstep=%.1fs total=%.1fs",
        n_classes,
        _dataset_image_count(experiment_dataset),
        int(rotations_np.shape[0]),
        int(translations_np.shape[0]),
        probe_s,
        mstep_s,
        time.time() - overall_t0,
    )

    return _assemble_result(
        class_log_evidence=class_log_evidence_np,
        new_means=new_means,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        per_class_hard_assignments=np.stack(hard_assignments, axis=0),
        per_class_stats=tuple(per_class_stats),
        noise_stats=None if per_class_noise is None else tuple(per_class_noise),
        per_class_best_pose_rotations=per_class_best_pose_rotations,
        per_class_best_pose_translations=per_class_best_pose_translations,
        per_class_best_pose_rotation_ids=per_class_best_pose_rotation_ids,
    )


def run_local_k_class_em(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    local_layout: LocalHypothesisLayout,
    disc_type: str,
    *,
    class_log_priors=None,
    accumulate_noise: bool = False,
    return_best_pose_details: bool = False,
    class_log_evidence=None,
    normalization_log_evidence=None,
    stats_use_reconstruction_probs: bool = False,
    class_posterior_sums_from_noise: bool = False,
    **engine_kwargs,
) -> KClassEMResult:
    """Run exact-local K-class EM using ``run_local_em_exact`` for all kernels."""

    _reject_kwargs(
        engine_kwargs,
        (
            "accumulate_noise",
            "class_log_prior",
            "normalization_log_z",
            "disable_adjoint_y",
            "disable_adjoint_ctf",
            "return_best_pose_details",
        ),
        "run_local_k_class_em",
    )
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    fallback_n_images = local_layout[0].n_images if isinstance(local_layout, (list, tuple)) else local_layout.n_images
    n_images = _dataset_image_count(experiment_dataset, fallback=fallback_n_images)
    log_priors = _class_log_priors(n_classes, class_log_priors)
    base_engine_kwargs = dict(engine_kwargs)
    return_profile = bool(base_engine_kwargs.pop("return_profile", False))
    class_local_rotation_log_prior = base_engine_kwargs.pop("class_local_rotation_log_prior", None)

    def _class_posterior_sums_override(noise_values: tuple[NoiseStats, ...] | None):
        if not class_posterior_sums_from_noise:
            return None
        if noise_values is None:
            raise ValueError("class_posterior_sums_from_noise requires accumulate_noise=True")
        return np.asarray([float(stats.sumw) for stats in noise_values], dtype=np.float64)

    class_log_evidence_np = None
    if class_log_evidence is not None:
        class_log_evidence_np = np.asarray(class_log_evidence, dtype=np.float64)
        if class_log_evidence_np.shape != (n_classes, n_images):
            raise ValueError(
                f"class_log_evidence must have shape ({n_classes}, {n_images}), got {class_log_evidence_np.shape}",
            )
    normalization_log_evidence_np = None
    if normalization_log_evidence is not None:
        normalization_log_evidence_np = np.asarray(normalization_log_evidence, dtype=np.float64)
        if normalization_log_evidence_np.shape != (n_images,):
            raise ValueError(
                f"normalization_log_evidence must have shape ({n_images},), got {normalization_log_evidence_np.shape}",
            )
    if class_log_evidence_np is not None:
        if normalization_log_evidence_np is None:
            normalization_log_evidence_np = _logsumexp_np(class_log_evidence_np, axis=0)
        if class_log_evidence_np.shape[1] != normalization_log_evidence_np.shape[0]:
            raise ValueError(
                "class_log_evidence and normalization_log_evidence image axes disagree: "
                f"{class_log_evidence_np.shape[1]} vs {normalization_log_evidence_np.shape[0]}",
            )

    if class_log_evidence_np is None:
        if n_classes == 1 and normalization_log_evidence_np is None:
            class_layout = _select_local_layout_for_class(
                local_layout,
                class_local_rotation_log_prior,
                0,
                n_classes,
            )
            class_engine_kwargs = _local_engine_kwargs_for_class(base_engine_kwargs, 0, n_classes)
            with _LocalDebugDumpPhaseLabel("single_class"):
                output = run_local_em_exact(
                    experiment_dataset,
                    means_array[0],
                    _select_class_value(mean_variance, 0, n_classes),
                    _select_class_value(noise_variance, 0, n_classes),
                    class_layout,
                    disc_type,
                    accumulate_noise=accumulate_noise,
                    return_profile=return_profile,
                    return_best_pose_details=return_best_pose_details,
                    class_log_prior=float(log_priors[0]),
                    stats_use_reconstruction_probs=stats_use_reconstruction_probs,
                    **class_engine_kwargs,
                )
            (
                class_Ft_y,
                class_Ft_ctf,
                hard_assignment,
                best_pose_rotations,
                best_pose_translations,
                best_pose_rotation_ids,
                stats,
                noise,
                profile_summary,
            ) = _local_outputs(
                output,
                accumulate_noise=accumulate_noise,
                return_best_pose_details=return_best_pose_details,
                return_profile=return_profile,
            )
            return _assemble_result(
                class_log_evidence=np.asarray(stats.log_evidence_per_image, dtype=np.float64)[None, :],
                new_means=None,
                Ft_y=[class_Ft_y],
                Ft_ctf=[class_Ft_ctf],
                per_class_hard_assignments=np.asarray(hard_assignment, dtype=np.int32)[None, :],
                per_class_stats=(stats,),
                noise_stats=None if noise is None else (noise,),
                per_class_best_pose_rotations=None if best_pose_rotations is None else [best_pose_rotations],
                per_class_best_pose_translations=None if best_pose_translations is None else [best_pose_translations],
                per_class_best_pose_rotation_ids=None if best_pose_rotation_ids is None else [best_pose_rotation_ids],
                profile_summary=profile_summary,
                class_posterior_sums_override=_class_posterior_sums_override(
                    None if noise is None else (noise,),
                ),
            )

        collect_global_reconstruction_threshold = bool(
            base_engine_kwargs.get("reconstruct_significant_only", False)
            and base_engine_kwargs.get("reconstruction_probability_threshold") is None
        )
        class_log_evidence = []
        support_values_by_class = [] if collect_global_reconstruction_threshold else None
        for class_index in range(n_classes):
            class_layout = _select_local_layout_for_class(
                local_layout,
                class_local_rotation_log_prior,
                class_index,
                n_classes,
            )
            class_engine_kwargs = _local_engine_kwargs_for_class(base_engine_kwargs, class_index, n_classes)
            with _LocalDebugDumpPhaseLabel(f"probe_class{class_index:03d}"):
                probe = run_local_em_exact(
                    experiment_dataset,
                    means_array[class_index],
                    _select_class_value(mean_variance, class_index, n_classes),
                    _select_class_value(noise_variance, class_index, n_classes),
                    class_layout,
                    disc_type,
                    accumulate_noise=False,
                    return_best_pose_details=False,
                    class_log_prior=float(log_priors[class_index]),
                    disable_adjoint_y=True,
                    disable_adjoint_ctf=True,
                    stats_use_reconstruction_probs=stats_use_reconstruction_probs,
                    return_profile=return_profile or collect_global_reconstruction_threshold,
                    return_reconstruction_probability_values=collect_global_reconstruction_threshold,
                    **class_engine_kwargs,
                )
            class_log_evidence.append(np.asarray(probe[3].log_evidence_per_image, dtype=np.float64))
            if support_values_by_class is not None:
                profile = probe[-1]
                support_values_by_class.append(tuple(profile["reconstruction_probability_values_by_image"]))
        class_log_evidence_np = np.stack(class_log_evidence, axis=0)
        normalization_log_evidence_np = _logsumexp_np(class_log_evidence_np, axis=0)
        if support_values_by_class is not None:
            base_engine_kwargs["reconstruction_probability_threshold"] = _global_reconstruction_probability_thresholds(
                support_values_by_class,
                class_log_evidence_np,
                normalization_log_evidence_np,
                float(base_engine_kwargs.get("adaptive_fraction", 0.999)),
            )
    else:
        for class_index in range(n_classes):
            _select_local_layout_for_class(
                local_layout,
                class_local_rotation_log_prior,
                class_index,
                n_classes,
            )

    global_log_evidence = _logsumexp_np(class_log_evidence_np, axis=0)
    if normalization_log_evidence_np is not None:
        global_log_evidence = normalization_log_evidence_np

    Ft_y = []
    Ft_ctf = []
    hard_assignments = []
    per_class_stats = []
    per_class_noise = [] if accumulate_noise else None
    per_class_best_pose_rotations = [] if return_best_pose_details else None
    per_class_best_pose_translations = [] if return_best_pose_details else None
    per_class_best_pose_rotation_ids = [] if return_best_pose_details else None
    per_class_profile_summaries = [] if return_profile else None
    for class_index in range(n_classes):
        class_layout = _select_local_layout_for_class(
            local_layout,
            class_local_rotation_log_prior,
            class_index,
            n_classes,
        )
        class_engine_kwargs = _local_engine_kwargs_for_class(base_engine_kwargs, class_index, n_classes)
        with _LocalDebugDumpPhaseLabel(f"mstep_class{class_index:03d}"):
            output = run_local_em_exact(
                experiment_dataset,
                means_array[class_index],
                _select_class_value(mean_variance, class_index, n_classes),
                _select_class_value(noise_variance, class_index, n_classes),
                class_layout,
                disc_type,
                accumulate_noise=accumulate_noise,
                return_profile=return_profile,
                return_best_pose_details=return_best_pose_details,
                class_log_prior=float(log_priors[class_index]),
                normalization_log_evidence=global_log_evidence,
                stats_use_reconstruction_probs=stats_use_reconstruction_probs,
                **class_engine_kwargs,
            )
        (
            class_Ft_y,
            class_Ft_ctf,
            hard_assignment,
            best_pose_rotations,
            best_pose_translations,
            best_pose_rotation_ids,
            stats,
            noise,
            profile_summary,
        ) = _local_outputs(
            output,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            return_profile=return_profile,
        )
        Ft_y.append(class_Ft_y)
        Ft_ctf.append(class_Ft_ctf)
        hard_assignments.append(np.asarray(hard_assignment, dtype=np.int32))
        per_class_stats.append(stats)
        if per_class_noise is not None:
            per_class_noise.append(noise)
        if return_best_pose_details:
            per_class_best_pose_rotations.append(best_pose_rotations)
            per_class_best_pose_translations.append(best_pose_translations)
            per_class_best_pose_rotation_ids.append(best_pose_rotation_ids)
        if per_class_profile_summaries is not None:
            per_class_profile_summaries.append(profile_summary)

    profile_summary = None
    if per_class_profile_summaries is not None:
        profile_summary = {
            "per_class_profile_summary": tuple(per_class_profile_summaries),
            "em_time_s": np.float64(
                sum(
                    float(summary.get("em_time_s", 0.0))
                    for summary in per_class_profile_summaries
                    if summary is not None
                )
            ),
        }

    return _assemble_result(
        class_log_evidence=class_log_evidence_np,
        new_means=None,
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        per_class_hard_assignments=np.stack(hard_assignments, axis=0),
        per_class_stats=tuple(per_class_stats),
        noise_stats=None if per_class_noise is None else tuple(per_class_noise),
        per_class_best_pose_rotations=per_class_best_pose_rotations,
        per_class_best_pose_translations=per_class_best_pose_translations,
        per_class_best_pose_rotation_ids=per_class_best_pose_rotation_ids,
        profile_summary=profile_summary,
        class_posterior_sums_override=_class_posterior_sums_override(
            None if per_class_noise is None else tuple(per_class_noise),
        ),
    )


def _build_fine_grid_significance_mask(
    significant_sample_indices_per_class,
    n_rot_coarse: int,
    n_trans_coarse: int,
    n_rot_fine: int,
    n_trans_fine: int,
    rot_oversampling_factor: int,
    trans_oversampling_factor: int,
    rot_parent_map: np.ndarray,
    trans_parent_map: np.ndarray,
    n_images: int,
) -> np.ndarray:
    """Expand pass-1 coarse significance to a per-image fine-grid mask.

    For each image, ``significant_sample_indices_per_class[i]`` holds the
    flat coarse pose indices ``r_coarse * n_trans_coarse + t_coarse`` that
    survived adaptive_fraction pruning at the coarse grid (or ``None``
    when every coarse pose was significant).

    Each coarse pose ``(r_coarse, t_coarse)`` expands to
    ``rot_oversampling_factor * trans_oversampling_factor`` fine poses,
    where the parent of fine rotation ``r_fine`` is
    ``rot_parent_map[r_fine]`` and likewise for translations.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_images, n_rot_fine, n_trans_fine)
        True at fine pose positions whose coarse parent was significant.

    Mirrors RELION's pass-2 significance mask in
    ``ml_optimiser.cpp::expectationOneParticle`` (line 5022 onward), where
    only oversampled children of pass-1 significant coarse samples are
    evaluated in pass-2.
    """
    if rot_parent_map.shape != (n_rot_fine,):
        raise ValueError(
            f"rot_parent_map must have shape ({n_rot_fine},), got {rot_parent_map.shape}",
        )
    if trans_parent_map.shape != (n_trans_fine,):
        raise ValueError(
            f"trans_parent_map must have shape ({n_trans_fine},), got {trans_parent_map.shape}",
        )

    mask = np.zeros((n_images, n_rot_fine, n_trans_fine), dtype=bool)
    for image_index in range(n_images):
        sig = significant_sample_indices_per_class[image_index]
        if sig is None:
            mask[image_index] = True
            continue
        sig = np.asarray(sig, dtype=np.int64)
        if sig.size == 0:
            continue
        coarse_rot_idx = sig // n_trans_coarse
        coarse_trans_idx = sig % n_trans_coarse
        coarse_pair = np.zeros((n_rot_coarse, n_trans_coarse), dtype=bool)
        coarse_pair[coarse_rot_idx, coarse_trans_idx] = True
        # Broadcast parent significance to the fine grid.
        mask[image_index] = coarse_pair[rot_parent_map][:, trans_parent_map]
    return mask


@dataclass(frozen=True)
class _PerClassFineGridSignificanceMask:
    significant_sample_indices: object
    n_rot_coarse: int
    n_trans_coarse: int
    n_rot_fine: int
    n_trans_fine: int
    rot_parent_map: np.ndarray
    trans_parent_map: np.ndarray
    n_images: int
    class_index: int
    global_winner: np.ndarray | None = None

    @property
    def shape(self):
        return (self.n_images, self.n_rot_fine, self.n_trans_fine)

    @property
    def size(self):
        return int(self.n_images * self.n_rot_fine * self.n_trans_fine)

    def __array__(self, dtype=None):
        raise TypeError("_PerClassFineGridSignificanceMask is lazy; call block_mask instead")

    def block_mask(self, *, r0: int, r1: int, start: int, end: int, batch_count: int, rotation_block_size: int):
        actual_count = int(end - start)
        batch_count = int(batch_count)
        r0 = int(r0)
        actual_rot = max(0, min(int(rotation_block_size), self.n_rot_fine - r0))
        mask = np.zeros((batch_count, int(rotation_block_size), self.n_trans_fine), dtype=bool)
        if actual_count <= 0 or actual_rot <= 0:
            return jnp.asarray(mask)

        rot_parent_block = self.rot_parent_map[r0 : r0 + actual_rot]
        for local_image, image_index in enumerate(range(int(start), int(end))):
            if self.global_winner is not None and int(self.global_winner[image_index]) != int(self.class_index):
                continue
            sig = self.significant_sample_indices[image_index]
            if sig is None:
                mask[local_image, :actual_rot, :] = True
                continue
            sig = np.asarray(sig, dtype=np.int64).reshape(-1)
            if sig.size == 0:
                continue
            coarse_rot_idx = sig // self.n_trans_coarse
            coarse_trans_idx = sig % self.n_trans_coarse
            coarse_pair = np.zeros((self.n_rot_coarse, self.n_trans_coarse), dtype=bool)
            coarse_pair[coarse_rot_idx, coarse_trans_idx] = True
            mask[local_image, :actual_rot, :] = coarse_pair[rot_parent_block][:, self.trans_parent_map]
        return jnp.asarray(mask)


@dataclass(frozen=True)
class _ClassFineGridSignificanceMask:
    significant_sample_indices_by_class: object
    n_rot_coarse: int
    n_trans_coarse: int
    n_rot_fine: int
    n_trans_fine: int
    rot_parent_map: np.ndarray
    trans_parent_map: np.ndarray
    n_images: int
    n_classes: int
    global_winner: np.ndarray | None = None

    @property
    def shape(self):
        return (self.n_classes, self.n_images, self.n_rot_fine, self.n_trans_fine)

    @property
    def size(self):
        return int(self.n_classes * self.n_images * self.n_rot_fine * self.n_trans_fine)

    def __array__(self, dtype=None):
        raise TypeError("_ClassFineGridSignificanceMask is lazy; call for_class instead")

    def for_class(self, class_index: int) -> _PerClassFineGridSignificanceMask:
        return _PerClassFineGridSignificanceMask(
            significant_sample_indices=self.significant_sample_indices_by_class[int(class_index)],
            n_rot_coarse=self.n_rot_coarse,
            n_trans_coarse=self.n_trans_coarse,
            n_rot_fine=self.n_rot_fine,
            n_trans_fine=self.n_trans_fine,
            rot_parent_map=self.rot_parent_map,
            trans_parent_map=self.trans_parent_map,
            n_images=self.n_images,
            class_index=int(class_index),
            global_winner=self.global_winner,
        )


@nvtx.annotate("kclass.run_dense_k_class_em_adaptive", color="red", domain=NVTX_DOMAIN_EM)
def run_dense_k_class_em_adaptive(
    experiment_dataset,
    means,
    mean_variance,
    noise_variance,
    coarse_rotations,
    coarse_translations,
    fine_rotations,
    fine_translations,
    rot_parent_map,
    trans_parent_map,
    disc_type: str,
    *,
    class_log_priors=None,
    accumulate_noise: bool = False,
    adaptive_fraction: float = 0.999,
    max_significants: int = -1,
    significance_image_batch_size: int | None = None,
    significance_rotation_block_size: int | None = None,
    coarse_current_size: int | None = None,
    fine_current_size: int | None = None,
    coarse_translation_log_prior=None,
    coarse_rotation_log_prior=None,
    coarse_class_rotation_log_prior=None,
    skip_significance_pruning: bool = False,
    firstiter_cc_pass2_only_best_coarse: bool = False,
    return_best_pose_details: bool = False,
    **engine_kwargs,
) -> KClassEMResult:
    """K-class adaptive 2-pass EM: coarse pass-1 significance + fine pass-2 masked.

    Mirrors RELION's adaptive 2-pass logic in
    ``ml_optimiser.cpp::expectationOneParticle`` (line 5022).  Pass-1 evaluates
    the coarse grid and produces a per-particle significance mask retaining
    ``adaptive_fraction`` of the posterior mass.  Pass-2 evaluates the fine
    (oversampled) grid but masks out fine poses whose coarse parent was not
    significant, recovering the same pose marginal as RELION's true 2-pass
    while keeping JIT compilation simple.

    Parameters
    ----------
    coarse_rotations, coarse_translations : np.ndarray
        Pass-1 coarse pose grids.
    fine_rotations, fine_translations : np.ndarray
        Pass-2 fine (oversampled) pose grids.
    rot_parent_map : np.ndarray of int, shape (n_rot_fine,)
        Index into ``coarse_rotations`` for each fine rotation.
    trans_parent_map : np.ndarray of int, shape (n_trans_fine,)
        Index into ``coarse_translations`` for each fine translation.
    coarse_current_size, fine_current_size : int or None
        Per-pass Fourier window radii.  Pass-1 typically uses a smaller
        ``coarse_current_size`` per RELION's ``image_coarse_size`` semantics.
        When ``None``, both passes use the same ``current_size``.
    coarse_*_log_prior : optional priors used only at pass-1.  ``engine_kwargs``
        carries the priors used at pass-2.
    skip_significance_pruning : bool
        When True, skip the pass-1 coarse significance computation entirely
        and evaluate the full fine grid with no mask.  This matches RELION's
        ``--firstiter_cc`` + adaptive_oversampling behavior at iter 1
        (ml_optimiser.cpp:9181-9207): RELION still runs the 2-pass loop but
        binarizes weights to a single best pose post-pass-2, so pass-1
        significance pruning is a no-op for that iteration.
    """
    # Lazy import to avoid the formatter stripping a top-level name that is
    # only referenced inside this function.
    from .helpers.significance import _compute_k_class_significance_batched

    overall_t0 = time.time()
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)

    coarse_rotations_np = np.asarray(coarse_rotations, dtype=np.float32)
    coarse_translations_np = np.asarray(coarse_translations, dtype=np.float32)
    fine_rotations_np = np.asarray(fine_rotations, dtype=np.float32)
    fine_translations_np = np.asarray(fine_translations, dtype=np.float32)
    rot_parent_map_np = np.asarray(rot_parent_map, dtype=np.int64)
    trans_parent_map_np = np.asarray(trans_parent_map, dtype=np.int64)

    n_rot_coarse = int(coarse_rotations_np.shape[0])
    n_trans_coarse = int(coarse_translations_np.shape[0])
    n_rot_fine = int(fine_rotations_np.shape[0])
    n_trans_fine = int(fine_translations_np.shape[0])

    if rot_parent_map_np.shape != (n_rot_fine,):
        raise ValueError(
            f"rot_parent_map must have shape ({n_rot_fine},), got {rot_parent_map_np.shape}",
        )
    if trans_parent_map_np.shape != (n_trans_fine,):
        raise ValueError(
            f"trans_parent_map must have shape ({n_trans_fine},), got {trans_parent_map_np.shape}",
        )
    if int(rot_parent_map_np.max(initial=-1)) >= n_rot_coarse:
        raise ValueError("rot_parent_map values must be < n_rot_coarse")
    if int(trans_parent_map_np.max(initial=-1)) >= n_trans_coarse:
        raise ValueError("trans_parent_map values must be < n_trans_coarse")
    n_images = _dataset_image_count(experiment_dataset)
    image_batch_size = int(engine_kwargs.get("image_batch_size", 500))
    rotation_block_size = int(engine_kwargs.get("rotation_block_size", 5000))
    sig_ibs = int(significance_image_batch_size or image_batch_size)
    sig_rbs = int(significance_rotation_block_size or rotation_block_size)

    # Pass-1 priors fall back to pass-2 priors when not supplied separately.
    if coarse_translation_log_prior is None:
        coarse_translation_log_prior = engine_kwargs.get("translation_log_prior")
    if coarse_rotation_log_prior is None:
        coarse_rotation_log_prior = engine_kwargs.get("rotation_log_prior")
    if coarse_class_rotation_log_prior is None:
        coarse_class_rotation_log_prior = engine_kwargs.get("class_rotation_log_prior")

    pass1_rotation_prior = (
        coarse_class_rotation_log_prior if coarse_class_rotation_log_prior is not None else coarse_rotation_log_prior
    )

    coarse_class_assignments_for_override = None
    pass1_t0 = time.time()
    if firstiter_cc_pass2_only_best_coarse:
        # RELION ml_optimiser.cpp:9181-9207: at iter 1 with --firstiter_cc,
        # pass-1 binarizes exp_Mweight to a single best (class, pose). Pass-2
        # then refines only that pose's children. With recovar's K-class
        # winner_take_all M-step (each class accumulates one-hot weight at
        # its own per-class best), the cleanest mirror is: for each class
        # independently, restrict pass-2 to children of class k's per-class
        # coarse-best pose. This matches the per-class M-step contribution
        # pattern recovar's existing K-class baseline already uses, while
        # giving each class the benefit of fine-grid refinement around its
        # own coarse winner.
        coarse_probe_kwargs = dict(engine_kwargs)
        coarse_probe_kwargs.pop("rotation_translation_mask", None)
        coarse_probe_kwargs.pop("class_rotation_translation_mask", None)
        coarse_probe_kwargs["relion_firstiter_score_mode"] = "normalized_cc"
        coarse_probe_kwargs["relion_firstiter_winner_take_all"] = True
        coarse_probe_kwargs["current_size"] = (
            coarse_current_size if coarse_current_size is not None else fine_current_size
        )
        with _DenseScoreDumpPhaseLabel("coarse"):
            with nvtx.annotate("kclass.adaptive.coarse_probe", color="yellow", domain=NVTX_DOMAIN_EM):
                coarse_result = _run_dense_k_class_score_probe(
                    experiment_dataset,
                    means_array,
                    mean_variance,
                    noise_variance,
                    coarse_rotations_np,
                    coarse_translations_np,
                    disc_type,
                    class_log_priors=class_log_priors,
                    **coarse_probe_kwargs,
                )
        # ``per_class_hard_assignments[k, i]`` is class k's best coarse pose
        # (independently scored per class). For each class, restrict pass-2
        # to that single pose's children.
        coarse_per_class_assn = np.asarray(coarse_result.per_class_hard_assignments, dtype=np.int64)
        # Preserve the K-class assignment from the coarse pass: RELION
        # decides class membership at the coarse grid binarization step
        # (the fine refinement only touches the single winning class's pose),
        # so the per-particle class assignment should reflect coarse argmax,
        # not per-class fine-best argmax.
        coarse_class_assignments_for_override = np.asarray(
            coarse_result.class_assignments,
            dtype=np.int32,
        )
        sig_sample_indices_by_class = [
            [np.array([int(coarse_per_class_assn[k, i])], dtype=np.int32) for i in range(n_images)]
            for k in range(n_classes)
        ]
    elif skip_significance_pruning:
        # Trivial mask: every fine pose is significant (None means all-True).
        sig_sample_indices_by_class = [[None] * n_images for _ in range(n_classes)]
    else:
        sig_kwargs = dict(
            adaptive_fraction=adaptive_fraction,
            max_significants=max_significants,
            image_batch_size=sig_ibs,
            rotation_block_size=sig_rbs,
            current_size=(coarse_current_size if coarse_current_size is not None else fine_current_size),
            score_with_masked_images=engine_kwargs.get("score_with_masked_images", True),
            rotation_log_prior=pass1_rotation_prior,
            translation_log_prior=coarse_translation_log_prior,
            image_corrections=engine_kwargs.get("image_corrections"),
            scale_corrections=engine_kwargs.get("scale_corrections"),
            image_pre_shifts=engine_kwargs.get("image_pre_shifts"),
            half_spectrum_scoring=engine_kwargs.get("half_spectrum_scoring", False),
            projection_padding_factor=engine_kwargs.get("projection_padding_factor", 1),
            do_gridding_correction=engine_kwargs.get("do_gridding_correction", False),
            square_window=engine_kwargs.get("square_window", False),
            use_float64_scoring=engine_kwargs.get("use_float64_scoring", False),
        )

        with nvtx.annotate("kclass.adaptive.significance", color="orange", domain=NVTX_DOMAIN_EM):
            (
                _sig_rot_any_by_class,
                _n_sig_per_image,
                _coarse_hard_assignment,
                _coarse_class_assignment,
                sig_sample_indices_by_class,
                _full_coarse_stats,
            ) = _compute_k_class_significance_batched(
                experiment_dataset,
                means_array,
                noise_variance,
                coarse_rotations_np,
                coarse_translations_np,
                disc_type,
                class_log_priors=log_priors,
                **sig_kwargs,
            )
    pass1_s = time.time() - pass1_t0

    mask_t0 = time.time()
    pass2_kwargs = dict(engine_kwargs)
    # Build a per-particle, per-class fine-grid mask from the coarse significance.
    pass2_kwargs.pop("rotation_translation_mask", None)
    sparse_pass2_requested = bool(pass2_kwargs.pop("sparse_pass2", False))
    # The explicit bucketed sparse pass-2 path consumes ``sparse_pass2`` above.
    # Dense fallback calls must keep run_em's block-skipping optimization off:
    # otherwise omitting the kwarg silently re-enables run_em's default.
    pass2_kwargs["sparse_pass2"] = False
    if "current_size" not in pass2_kwargs and fine_current_size is not None:
        pass2_kwargs["current_size"] = fine_current_size

    if (
        sparse_pass2_requested
        and firstiter_cc_pass2_only_best_coarse
        and coarse_class_assignments_for_override is not None
        and hasattr(experiment_dataset, "subset")
    ):
        pass2_t0 = time.time()
        result = _run_sparse_firstiter_global_winner_subset_pass2(
            experiment_dataset,
            means_array,
            mean_variance,
            noise_variance,
            coarse_translations_np,
            fine_rotations_np,
            fine_translations_np,
            rot_parent_map_np,
            trans_parent_map_np,
            sig_sample_indices_by_class,
            disc_type,
            coarse_result=coarse_result,
            coarse_class_assignments=coarse_class_assignments_for_override,
            n_rot_coarse=n_rot_coarse,
            n_fine_trans=n_trans_fine,
            healpix_order=_infer_healpix_order_from_rotation_count(n_rot_coarse),
            oversampling_order=max(
                0,
                _infer_healpix_order_from_rotation_count(n_rot_fine)
                - _infer_healpix_order_from_rotation_count(n_rot_coarse),
            ),
            class_log_priors=class_log_priors,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            pass2_kwargs=pass2_kwargs,
        )
        pass2_s = time.time() - pass2_t0
        logger.info(
            "Adaptive K-class EM profile: classes=%d images=%d coarse=(rot=%d,trans=%d) sparse_firstiter_fine=(rot=%d,trans=%d) pass1=%.1fs mask=%.1fs pass2=%.1fs total=%.1fs",
            n_classes,
            n_images,
            n_rot_coarse,
            n_trans_coarse,
            n_rot_fine,
            n_trans_fine,
            pass1_s,
            time.time() - mask_t0,
            pass2_s,
            time.time() - overall_t0,
        )
        return result

    if sparse_pass2_requested and not firstiter_cc_pass2_only_best_coarse and not skip_significance_pruning:
        result = _run_sparse_k_class_adaptive_pass2(
            experiment_dataset,
            means_array,
            mean_variance,
            noise_variance,
            coarse_rotations_np,
            coarse_translations_np,
            fine_rotations_np,
            rot_parent_map_np,
            fine_translations_np,
            trans_parent_map_np,
            sig_sample_indices_by_class,
            disc_type,
            class_log_priors=log_priors,
            accumulate_noise=accumulate_noise,
            return_best_pose_details=return_best_pose_details,
            oversampling_order=max(
                0,
                _infer_healpix_order_from_rotation_count(n_rot_fine)
                - _infer_healpix_order_from_rotation_count(n_rot_coarse),
            ),
            random_perturbation=0.0,
            engine_kwargs=pass2_kwargs,
        )
        logger.info(
            "Adaptive K-class EM profile: classes=%d images=%d coarse=(rot=%d,trans=%d) sparse_fine=(rot=%d,trans=%d) pass1=%.1fs mask=%.1fs total=%.1fs",
            n_classes,
            n_images,
            n_rot_coarse,
            n_trans_coarse,
            n_rot_fine,
            n_trans_fine,
            pass1_s,
            time.time() - mask_t0,
            time.time() - overall_t0,
        )
        return result

    # Expand priors from coarse to fine grid by parent broadcasting.
    # Mirrors RELION's pushback semantics where each oversampled child inherits
    # its parent's prior weight (sampling_ml.cpp, ml_optimiser.cpp:5478 etc.).
    rotation_log_prior_in = pass2_kwargs.get("rotation_log_prior")
    if rotation_log_prior_in is not None:
        prior_np = np.asarray(rotation_log_prior_in, dtype=np.float32)
        if prior_np.ndim == 1:
            if prior_np.shape != (n_rot_coarse,):
                raise ValueError(
                    f"rotation_log_prior must have shape ({n_rot_coarse},), got {prior_np.shape}",
                )
            pass2_kwargs["rotation_log_prior"] = prior_np[rot_parent_map_np]
        elif prior_np.ndim == 2:
            if prior_np.shape != (n_images, n_rot_coarse):
                raise ValueError(
                    f"rotation_log_prior must have shape ({n_images}, {n_rot_coarse}), got {prior_np.shape}",
                )
            pass2_kwargs["rotation_log_prior"] = prior_np[:, rot_parent_map_np]
    class_rotation_log_prior_in = pass2_kwargs.get("class_rotation_log_prior")
    if class_rotation_log_prior_in is not None:
        prior_np = np.asarray(class_rotation_log_prior_in, dtype=np.float32)
        if prior_np.ndim != 2 or prior_np.shape != (n_classes, n_rot_coarse):
            raise ValueError(
                f"class_rotation_log_prior must have shape ({n_classes}, {n_rot_coarse}), got {prior_np.shape}",
            )
        pass2_kwargs["class_rotation_log_prior"] = prior_np[:, rot_parent_map_np]
    translation_log_prior_in = pass2_kwargs.get("translation_log_prior")
    if translation_log_prior_in is not None:
        prior_np = np.asarray(translation_log_prior_in, dtype=np.float32)
        if prior_np.ndim == 1:
            if prior_np.shape != (n_trans_coarse,):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_trans_coarse},), got {prior_np.shape}",
                )
            pass2_kwargs["translation_log_prior"] = prior_np[trans_parent_map_np]
        elif prior_np.ndim == 2:
            if prior_np.shape != (n_images, n_trans_coarse):
                raise ValueError(
                    f"translation_log_prior must have shape ({n_images}, {n_trans_coarse}), got {prior_np.shape}",
                )
            pass2_kwargs["translation_log_prior"] = prior_np[:, trans_parent_map_np]

    global_winner = None
    if coarse_class_assignments_for_override is not None:
        # RELION ml_optimiser.cpp:9181-9207 with K>1: at iter 1 with --firstiter_cc,
        # binarization sets ONE entry to 1 in the global (class × pose) grid.
        # Only the globally-winning class accumulates weight=1 from each image;
        # the K-1 losing classes contribute zero. Without this gate, recovar's
        # per-class winner-take-all M-step gives every class weight=1 from every
        # image, which over-mixes images across classes (especially harmful for
        # the lowest-occupancy class — see K=4 chained iter-1 class-1 corr 0.876
        # vs RELION class-1, where class 1 receives ~80% off-class images).
        #
        # Mask out images where global winner != class_index so class k's M-step
        # only sees its global-winner images.
        global_winner = np.asarray(coarse_class_assignments_for_override, dtype=np.int64)
    if global_winner is not None and hasattr(experiment_dataset, "subset"):
        with _DenseScoreDumpPhaseLabel("fine"):
            with nvtx.annotate("kclass.adaptive.fine_subset_em", color="green", domain=NVTX_DOMAIN_EM):
                pass2_t0 = time.time()
                result = _run_firstiter_global_winner_subset_pass2(
                    experiment_dataset,
                    means_array,
                    mean_variance,
                    noise_variance,
                    fine_rotations_np,
                    fine_translations_np,
                    sig_sample_indices_by_class,
                    disc_type,
                    coarse_result=coarse_result,
                    coarse_class_assignments=global_winner,
                    n_rot_coarse=n_rot_coarse,
                    n_trans_coarse=n_trans_coarse,
                    n_rot_fine=n_rot_fine,
                    n_trans_fine=n_trans_fine,
                    rot_parent_map_np=rot_parent_map_np,
                    trans_parent_map_np=trans_parent_map_np,
                    class_log_priors=class_log_priors,
                    accumulate_noise=accumulate_noise,
                    return_best_pose_details=return_best_pose_details,
                    pass2_kwargs=pass2_kwargs,
                )
                pass2_s = time.time() - pass2_t0
        coarse_assn = jnp.asarray(global_winner, dtype=jnp.int32)
        n_imgs = int(coarse_assn.shape[0])
        image_indices = jnp.arange(n_imgs)
        per_class_hard = result.per_class_hard_assignments
        new_pose_assn = per_class_hard[coarse_assn, image_indices]
        replace_kwargs = dict(
            class_assignments=coarse_assn,
            pose_assignments=new_pose_assn,
        )
        if return_best_pose_details:
            best_rots, best_trans, best_rot_ids = _decode_dense_best_pose_details(
                np.asarray(new_pose_assn, dtype=np.int64),
                fine_rotations_np,
                fine_translations_np,
            )
            replace_kwargs.update(
                best_pose_rotations=best_rots,
                best_pose_translations=best_trans,
                best_pose_rotation_ids=best_rot_ids,
            )
        result = result._replace(**replace_kwargs)
        logger.info(
            "Adaptive K-class EM profile: classes=%d images=%d coarse=(rot=%d,trans=%d) fine_subset=(rot=%d,trans=%d) pass1=%.1fs mask=%.1fs pass2=%.1fs total=%.1fs",
            n_classes,
            n_images,
            n_rot_coarse,
            n_trans_coarse,
            n_rot_fine,
            n_trans_fine,
            pass1_s,
            time.time() - mask_t0,
            pass2_s,
            time.time() - overall_t0,
        )
        return result
    if not (skip_significance_pruning and global_winner is None):
        pass2_kwargs["class_rotation_translation_mask"] = _ClassFineGridSignificanceMask(
            significant_sample_indices_by_class=sig_sample_indices_by_class,
            n_rot_coarse=n_rot_coarse,
            n_trans_coarse=n_trans_coarse,
            n_rot_fine=n_rot_fine,
            n_trans_fine=n_trans_fine,
            rot_parent_map=rot_parent_map_np,
            trans_parent_map=trans_parent_map_np,
            n_images=n_images,
            n_classes=n_classes,
            global_winner=global_winner,
        )
    mask_s = time.time() - mask_t0

    with _DenseScoreDumpPhaseLabel("fine"):
        with nvtx.annotate("kclass.adaptive.fine_dense_em", color="green", domain=NVTX_DOMAIN_EM):
            pass2_t0 = time.time()
            result = run_dense_k_class_em(
                experiment_dataset,
                means_array,
                mean_variance,
                noise_variance,
                fine_rotations_np,
                fine_translations_np,
                disc_type,
                class_log_priors=class_log_priors,
                accumulate_noise=accumulate_noise,
                return_best_pose_details=return_best_pose_details,
                **pass2_kwargs,
            )
            pass2_s = time.time() - pass2_t0
    if coarse_class_assignments_for_override is not None:
        # RELION binarization picks the global-best (class, pose) at the
        # COARSE grid; the fine refinement only repositions the pose within
        # the winning class. Reflect this by replacing the K-class
        # ``class_assignments`` with the coarse-pass argmax. The per-class
        # M-step accumulators already encode each class's fine-refined best
        # pose, so reconstruction quality is preserved.
        coarse_assn = jnp.asarray(coarse_class_assignments_for_override, dtype=jnp.int32)
        n_imgs = int(coarse_assn.shape[0])
        image_indices = jnp.arange(n_imgs)
        per_class_hard = result.per_class_hard_assignments
        new_pose_assn = per_class_hard[coarse_assn, image_indices]
        replace_kwargs = dict(
            class_assignments=coarse_assn,
            pose_assignments=new_pose_assn,
        )
        if return_best_pose_details:
            best_rots, best_trans, best_rot_ids = _decode_dense_best_pose_details(
                np.asarray(new_pose_assn, dtype=np.int64),
                fine_rotations_np,
                fine_translations_np,
            )
            replace_kwargs.update(
                best_pose_rotations=best_rots,
                best_pose_translations=best_trans,
                best_pose_rotation_ids=best_rot_ids,
            )
        result = result._replace(**replace_kwargs)
    logger.info(
        "Adaptive K-class EM profile: classes=%d images=%d coarse=(rot=%d,trans=%d) fine=(rot=%d,trans=%d) pass1=%.1fs mask=%.1fs pass2=%.1fs total=%.1fs",
        n_classes,
        n_images,
        n_rot_coarse,
        n_trans_coarse,
        n_rot_fine,
        n_trans_fine,
        pass1_s,
        mask_s,
        pass2_s,
        time.time() - overall_t0,
    )
    return result
