"""K-class EM orchestration for dense and exact-local single-volume engines."""

from __future__ import annotations

import logging
import os
import time
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
    import inspect as _inspect

    _allowed = set(_inspect.signature(run_em).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
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
                    **base_engine_kwargs,
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
            )

        class_log_evidence = []
        for class_index in range(n_classes):
            class_layout = _select_local_layout_for_class(
                local_layout,
                class_local_rotation_log_prior,
                class_index,
                n_classes,
            )
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
                    **base_engine_kwargs,
                )
            class_log_evidence.append(np.asarray(probe[3].log_evidence_per_image, dtype=np.float64))
        class_log_evidence_np = np.stack(class_log_evidence, axis=0)
        normalization_log_evidence_np = _logsumexp_np(class_log_evidence_np, axis=0)
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
                **base_engine_kwargs,
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
    if "current_size" not in pass2_kwargs and fine_current_size is not None:
        pass2_kwargs["current_size"] = fine_current_size

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
