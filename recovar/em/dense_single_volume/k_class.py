"""K-class EM orchestration for dense and exact-local single-volume engines."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .dense_k_class_engine import run_dense_k_class_em_native
from .helpers.types import NoiseStats, RelionStats, make_noise_stats, make_relion_stats
from .local_k_class_engine import run_local_k_class_em_native
from .local_layout import LocalHypothesisLayout


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


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_value, axis=axis) + np.log(
        np.sum(np.exp(values - max_value), axis=axis),
    )


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


def _reject_kwargs(kwargs: dict, names: tuple[str, ...], caller: str) -> None:
    present = sorted(name for name in names if name in kwargs)
    if present:
        raise ValueError(f"{caller} controls these arguments directly: {', '.join(present)}")


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
) -> KClassEMResult:
    global_log_evidence = _logsumexp_np(class_log_evidence, axis=0).astype(np.float64)
    class_responsibilities = np.exp(class_log_evidence - global_log_evidence[None, :])
    class_posterior_sums = np.sum(class_responsibilities, axis=1)

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

    return KClassEMResult(
        new_means=stacked_new_means,
        Ft_y=jnp.stack([jnp.asarray(value) for value in Ft_y], axis=0),
        Ft_ctf=jnp.stack([jnp.asarray(value) for value in Ft_ctf], axis=0),
        per_class_hard_assignments=jnp.asarray(per_class_hard_assignments, dtype=jnp.int32),
        class_assignments=jnp.asarray(class_assignments, dtype=jnp.int32),
        pose_assignments=jnp.asarray(pose_assignments, dtype=jnp.int32),
        class_responsibilities=jnp.asarray(class_responsibilities, dtype=jnp.float32),
        class_posterior_sums=jnp.asarray(class_posterior_sums, dtype=jnp.float32),
        stats=stats,
        per_class_stats=per_class_stats,
        noise_stats=noise_stats,
        aggregate_noise_stats=_sum_noise_stats(noise_stats),
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
    )


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
    **engine_kwargs,
) -> KClassEMResult:
    """Run dense K-class EM over the joint class x pose grid."""

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
        ),
        "run_dense_k_class_em",
    )
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)
    if bool(engine_kwargs.get("sparse_pass2", False)):
        raise NotImplementedError("Dense K-class sparse pass-2 must use a joint class x pose native implementation")
    native = run_dense_k_class_em_native(
        experiment_dataset,
        means_array,
        mean_variance,
        noise_variance,
        rotations,
        translations,
        disc_type,
        class_log_priors=log_priors,
        accumulate_noise=accumulate_noise,
        **engine_kwargs,
    )
    return _assemble_result(
        class_log_evidence=native.class_log_evidence,
        new_means=native.new_means,
        Ft_y=native.Ft_y,
        Ft_ctf=native.Ft_ctf,
        per_class_hard_assignments=native.hard_assignments,
        per_class_stats=native.per_class_stats,
        noise_stats=native.noise_stats,
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
    **engine_kwargs,
) -> KClassEMResult:
    """Run exact-local K-class EM over the joint class x local-pose grid."""

    _reject_kwargs(
        engine_kwargs,
        (
            "accumulate_noise",
            "class_log_prior",
            "normalization_log_evidence",
            "normalization_log_z",
            "disable_adjoint_y",
            "disable_adjoint_ctf",
            "return_profile",
            "return_best_pose_details",
        ),
        "run_local_k_class_em",
    )
    means_array = _as_class_means(means)
    n_classes = int(means_array.shape[0])
    log_priors = _class_log_priors(n_classes, class_log_priors)

    native = run_local_k_class_em_native(
        experiment_dataset,
        means_array,
        mean_variance,
        noise_variance,
        local_layout,
        disc_type,
        class_log_priors=log_priors,
        accumulate_noise=accumulate_noise,
        return_best_pose_details=return_best_pose_details,
        **engine_kwargs,
    )
    return _assemble_result(
        class_log_evidence=native.class_log_evidence,
        new_means=native.new_means,
        Ft_y=native.Ft_y,
        Ft_ctf=native.Ft_ctf,
        per_class_hard_assignments=native.hard_assignments,
        per_class_stats=native.per_class_stats,
        noise_stats=native.noise_stats,
        per_class_best_pose_rotations=native.per_class_best_pose_rotations,
        per_class_best_pose_translations=native.per_class_best_pose_translations,
        per_class_best_pose_rotation_ids=native.per_class_best_pose_rotation_ids,
    )
