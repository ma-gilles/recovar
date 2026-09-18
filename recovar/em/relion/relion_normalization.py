"""RELION image-normalization and group-scale updates from M-step statistics.

Independent of iteration scheduling, follower dispatch and mean reconstruction.
Inputs and outputs retain the two-half convention, including an empty half in
Class3D. The update computes new arrays without installing runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class NormScaleCorrectionUpdateResult:
    """Native RELION norm/scale correction state for the next iteration."""

    norm_corrections_per_half: list
    avg_norm_correction_per_half: list
    group_scale_corrections_per_half: list
    image_corrections_per_half: list
    scale_corrections_per_half: list
    zero_norm_residual_counts: list


def _half_list_or_none(values, *, n_halves: int, name: str):
    if values is None:
        return [None] * n_halves
    if not isinstance(values, (list, tuple)) or len(values) != n_halves:
        raise ValueError(f"{name} must be a {n_halves}-element list/tuple or None")
    return list(values)


def _derive_group_scale_from_image_scale(scale_per_image, group_ids, n_groups):
    group_scale = np.ones(n_groups, dtype=np.float64)
    if scale_per_image is None:
        return group_scale
    sums = np.bincount(group_ids, weights=np.asarray(scale_per_image, dtype=np.float64), minlength=n_groups)
    counts = np.bincount(group_ids, minlength=n_groups)
    present = counts > 0
    group_scale[present] = sums[present] / counts[present]
    return group_scale


def update_relion_norm_scale_corrections(
    *,
    noise_stats_per_half,
    image_corrections_per_half=None,
    scale_corrections_per_half=None,
    group_ids_per_half=None,
    group_count_per_half=None,
    group_scale_corrections_per_half=None,
    avg_norm_correction_per_half=None,
    relion_firstiter_cc_this_iter: bool = False,
    do_norm_correction: bool = True,
    do_scale_correction: bool = True,
    scale_relaxation_mu: float = 0.0,
    eps: float = 1e-30,
    dtype: np.dtype = np.float32,
) -> NormScaleCorrectionUpdateResult:
    """Update RELION-style per-image norm and per-group scale corrections.

    Compute host statistics in float64 and cast returned arrays to ``dtype``.
    The default float32 arrays serve production EM; an explicit float64 dtype
    supports diagnostic replay. Preserve the higher-precision host arithmetic.

    See ``docs/math/relion_refinement_algorithm.md`` for the refinement state
    and M-step contracts. This helper computes corrections; the controller
    owns installation of the next iteration's state.

    The existing scoring paths consume two per-image arrays:
    ``image_corrections = (avg_norm / normcorr) * scale[group_id]`` and
    ``scale_corrections = scale[group_id]``.  This helper updates the native
    RELION state behind those arrays from M-step sufficient statistics.

    Scale sufficient statistics are expected to match RELION's
    ``wsum_model.wsum_signal_product`` and ``wsum_model.wsum_reference_power``:
    the collection site must divide XA by the old scale and AA by its square
    before accumulation.

    RELION's average norm-correction numerator contains one updated
    ``normcorr`` per particle, without posterior weighting.  Its denominator
    is ``wsum_model.pdf_class.sum()``, i.e. the retained significant-support
    posterior mass.  That mass is normally slightly smaller than the particle
    count, so using a conventional arithmetic mean introduces a systematic
    normalization drift.
    """

    stats_per_half = _half_list_or_none(noise_stats_per_half, n_halves=2, name="noise_stats_per_half")
    image_corr_in = _half_list_or_none(image_corrections_per_half, n_halves=2, name="image_corrections_per_half")
    scale_corr_in = _half_list_or_none(scale_corrections_per_half, n_halves=2, name="scale_corrections_per_half")
    group_ids_in = _half_list_or_none(group_ids_per_half, n_halves=2, name="group_ids_per_half")
    group_count_in = _half_list_or_none(group_count_per_half, n_halves=2, name="group_count_per_half")
    group_scale_in = _half_list_or_none(
        group_scale_corrections_per_half,
        n_halves=2,
        name="group_scale_corrections_per_half",
    )
    avg_norm_in = (
        [1.0, 1.0]
        if avg_norm_correction_per_half is None
        else list(avg_norm_correction_per_half)
        if isinstance(avg_norm_correction_per_half, (list, tuple)) and len(avg_norm_correction_per_half) == 2
        else None
    )
    if avg_norm_in is None:
        raise ValueError("avg_norm_correction_per_half must be a 2-element list/tuple or None")
    if not (0.0 <= float(scale_relaxation_mu) <= 1.0):
        raise ValueError(f"scale_relaxation_mu must be in [0, 1], got {scale_relaxation_mu}")

    out_norm = []
    out_avg_norm = []
    out_group_scale = []
    out_image_corr = []
    out_scale_corr = []
    out_zero_norm_counts = []

    for half_idx, stats in enumerate(stats_per_half):
        if stats is None:
            raise RuntimeError("RELION norm/scale update expected per-half NoiseStats")

        norm_stats = getattr(stats, "wsum_norm_correction", None)
        if norm_stats is None:
            if image_corr_in[half_idx] is None and scale_corr_in[half_idx] is None and group_ids_in[half_idx] is None:
                raise RuntimeError("Cannot infer image count without norm stats or correction arrays")
            n_images = int(
                len(
                    next(
                        value
                        for value in (image_corr_in[half_idx], scale_corr_in[half_idx], group_ids_in[half_idx])
                        if value is not None
                    )
                )
            )
        else:
            n_images = int(np.asarray(norm_stats).reshape(-1).shape[0])

        group_ids = (
            np.zeros(n_images, dtype=np.int64)
            if group_ids_in[half_idx] is None
            else np.asarray(group_ids_in[half_idx], dtype=np.int64).reshape(-1)
        )
        if group_ids.shape != (n_images,):
            raise ValueError(
                f"group_ids_per_half[{half_idx}] has shape {group_ids.shape}, expected ({n_images},)",
            )
        if group_ids.size and int(np.min(group_ids)) < 0:
            raise ValueError("group IDs must be non-negative")

        scale_per_image_in = (
            None
            if scale_corr_in[half_idx] is None
            else np.asarray(scale_corr_in[half_idx], dtype=np.float64).reshape(-1)
        )
        image_corr = (
            None
            if image_corr_in[half_idx] is None
            else np.asarray(image_corr_in[half_idx], dtype=np.float64).reshape(-1)
        )
        if scale_per_image_in is not None and scale_per_image_in.shape != (n_images,):
            raise ValueError(
                f"scale_corrections_per_half[{half_idx}] has shape {scale_per_image_in.shape}, expected ({n_images},)",
            )
        if image_corr is not None and image_corr.shape != (n_images,):
            raise ValueError(
                f"image_corrections_per_half[{half_idx}] has shape {image_corr.shape}, expected ({n_images},)",
            )

        n_groups_from_ids = int(np.max(group_ids)) + 1 if group_ids.size else 1
        explicit_group_count = 0
        if group_count_in[half_idx] is not None:
            explicit_group_count = int(group_count_in[half_idx])
            if (
                explicit_group_count < 0
                or not np.isfinite(float(group_count_in[half_idx]))
                or float(group_count_in[half_idx]) != float(explicit_group_count)
            ):
                raise ValueError(
                    f"group_count_per_half[{half_idx}] must be a non-negative integer, "
                    f"got {group_count_in[half_idx]!r}"
                )
        required_group_count = max(explicit_group_count, n_groups_from_ids)
        if group_scale_in[half_idx] is None:
            n_groups = required_group_count
            group_scale_old = _derive_group_scale_from_image_scale(scale_per_image_in, group_ids, n_groups)
        else:
            group_scale_old = np.asarray(group_scale_in[half_idx], dtype=np.float64).reshape(-1)
            n_groups = int(group_scale_old.shape[0])
            if n_groups < required_group_count:
                raise ValueError(
                    f"group_scale_corrections_per_half[{half_idx}] has {n_groups} groups, "
                    f"but group IDs / explicit count require {required_group_count}",
                )
        if np.any(group_scale_old <= 0.0):
            raise ValueError("group scale corrections must be positive")

        scale_per_image_old = group_scale_old[group_ids]
        if scale_per_image_in is not None:
            scale_per_image_old = scale_per_image_in
        if image_corr is None:
            image_corr = scale_per_image_old.copy()
        if np.any(image_corr <= 0.0):
            raise ValueError("image corrections must be positive")

        image_norm_factor = image_corr / np.maximum(scale_per_image_old, eps)
        avg_norm_old = float(avg_norm_in[half_idx])
        if avg_norm_old <= 0.0:
            raise ValueError("avg_norm corrections must be positive")

        if do_norm_correction and norm_stats is not None:
            norm_residual = np.asarray(norm_stats, dtype=np.float64).reshape(-1)
            if norm_residual.shape != (n_images,):
                raise ValueError(
                    f"wsum_norm_correction for half {half_idx} has shape {norm_residual.shape}, "
                    f"expected ({n_images},)",
                )
            if not np.all(np.isfinite(norm_residual)):
                raise ValueError("wsum_norm_correction must be finite")
            if np.any(norm_residual < -1e-12):
                raise ValueError("wsum_norm_correction must be non-negative")
            old_norm_over_avg = scale_per_image_old / image_corr
            valid_norm = norm_residual > eps
            zero_norm_count = int(n_images - np.count_nonzero(valid_norm))
            normcorr_from_stats = old_norm_over_avg * np.sqrt(np.maximum(2.0 * norm_residual, 0.0))
            retained_sum_weight = float(getattr(stats, "sumw", 0.0))
            if retained_sum_weight > 0.0:
                target_avg_norm = float(np.sum(normcorr_from_stats[valid_norm]) / retained_sum_weight)
            elif np.any(valid_norm):
                target_avg_norm = float(np.mean(normcorr_from_stats[valid_norm]))
            else:
                target_avg_norm = avg_norm_old
            avg_norm_new = float(scale_relaxation_mu) * avg_norm_old + (1.0 - float(scale_relaxation_mu)) * target_avg_norm
            image_norm_factor_new = image_norm_factor.copy()
            image_norm_factor_new[valid_norm] = avg_norm_new / np.maximum(normcorr_from_stats[valid_norm], eps)
            image_norm_factor = image_norm_factor_new
            normcorr_new = avg_norm_new / np.maximum(image_norm_factor, eps)
        else:
            normcorr_new = avg_norm_old / np.maximum(image_norm_factor, eps)
            avg_norm_new = avg_norm_old
            zero_norm_count = 0

        scale_xa = getattr(stats, "wsum_scale_correction_xa", None)
        scale_aa = getattr(stats, "wsum_scale_correction_aa", None)
        if (
            do_scale_correction
            and not relion_firstiter_cc_this_iter
            and scale_xa is not None
            and scale_aa is not None
        ):
            xa = np.asarray(scale_xa, dtype=np.float64).reshape(-1)
            aa = np.asarray(scale_aa, dtype=np.float64).reshape(-1)
            if xa.shape != (n_groups,) or aa.shape != (n_groups,):
                raise ValueError(
                    f"scale stats for half {half_idx} have shapes {xa.shape}/{aa.shape}, expected ({n_groups},)",
                )
            scale_target = np.ones_like(xa, dtype=np.float64)
            np.divide(xa, aa, out=scale_target, where=aa > 0.0)
            scale_new = float(scale_relaxation_mu) * group_scale_old + (1.0 - float(scale_relaxation_mu)) * scale_target
            sorted_scale = np.sort(scale_new)
            median = float(sorted_scale[n_groups // 2])
            if np.isfinite(median) and median > 0.0:
                scale_new = np.clip(scale_new, median / 5.0, 5.0 * median)
            counts = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
            count_sum = float(np.sum(counts))
            if count_sum > 0.0:
                avg_scale = float(np.sum(counts * scale_new) / count_sum)
                if avg_scale > 0.0 and np.isfinite(avg_scale):
                    scale_new = scale_new / avg_scale
        else:
            scale_new = group_scale_old.copy()

        scale_per_image_new = scale_new[group_ids]
        image_corr_new = image_norm_factor * scale_per_image_new

        out_norm.append(jnp.asarray(normcorr_new, dtype=dtype))
        out_avg_norm.append(avg_norm_new)
        out_group_scale.append(jnp.asarray(scale_new, dtype=dtype))
        out_scale_corr.append(jnp.asarray(scale_per_image_new, dtype=dtype))
        out_image_corr.append(jnp.asarray(image_corr_new, dtype=dtype))
        out_zero_norm_counts.append(zero_norm_count)

    return NormScaleCorrectionUpdateResult(
        norm_corrections_per_half=out_norm,
        avg_norm_correction_per_half=out_avg_norm,
        group_scale_corrections_per_half=out_group_scale,
        image_corrections_per_half=out_image_corr,
        scale_corrections_per_half=out_scale_corr,
        zero_norm_residual_counts=out_zero_norm_counts,
    )
