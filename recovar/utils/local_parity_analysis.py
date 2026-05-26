"""Helpers for comparing RECOVAR local score dumps against RELION operands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _load_npz_dict(obj):
    if isinstance(obj, (str, bytes)):
        with np.load(obj, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if hasattr(obj, "files"):
        return {key: obj[key] for key in obj.files}
    return dict(obj)


def _finite_scores(scores):
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(arr)
    return arr[mask]


def _normalize_probs(probs):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs = probs[np.isfinite(probs) & (probs >= 0.0)]
    if probs.size == 0:
        raise ValueError("No finite probabilities provided")
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("Probability vector has zero mass")
    return probs / total


def _softmax_probabilities_from_scores(scores, mask=None):
    score_arr = np.asarray(scores, dtype=np.float64)
    finite_mask = np.isfinite(score_arr)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != score_arr.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match scores shape {score_arr.shape}")
        finite_mask &= mask_arr
    finite = score_arr[finite_mask]
    if finite.size == 0:
        raise ValueError("No finite scores provided")
    shift = float(np.max(finite))
    weights = np.exp(finite - shift)
    norm = float(np.sum(weights))
    probs = np.zeros(score_arr.shape, dtype=np.float64)
    probs[finite_mask] = weights / norm
    return probs


def _align_mask_to_score_shape(mask, score_shape, *, name="mask"):
    mask_arr = np.asarray(mask, dtype=bool)
    score_shape = tuple(int(dim) for dim in score_shape)
    if mask_arr.shape == score_shape:
        return mask_arr
    if len(mask_arr.shape) != len(score_shape):
        raise ValueError(f"{name} shape {mask_arr.shape} does not match scores shape {score_shape}")
    if mask_arr.shape[1:] != score_shape[1:]:
        raise ValueError(f"{name} shape {mask_arr.shape} does not match scores shape {score_shape}")
    if mask_arr.shape[0] > score_shape[0]:
        trailing = mask_arr[score_shape[0] :, ...]
        if np.any(trailing):
            raise ValueError(f"{name} shape {mask_arr.shape} exceeds scores shape {score_shape}")
        return mask_arr[: score_shape[0], ...]
    padded = np.zeros(score_shape, dtype=bool)
    padded[: mask_arr.shape[0], ...] = mask_arr
    return padded


def _optional_recovar_array(recovar, key, shape, *, fill_value=np.nan, dtype=np.float64):
    if key in recovar:
        return np.asarray(recovar[key], dtype=dtype)
    return np.full(shape, fill_value, dtype=dtype)


def _probability_summary_from_probs(probs, topk=(1, 2, 5, 10)):
    probs = _normalize_probs(probs)
    sorted_probs = np.sort(probs)[::-1]
    positive = sorted_probs > 0.0
    entropy = -np.sum(sorted_probs[positive] * np.log(sorted_probs[positive]))
    topk_mass = {}
    for k in topk:
        topk_mass[f"top{k}_mass"] = float(np.sum(sorted_probs[: min(int(k), sorted_probs.size)]))
    return {
        "pmax": float(sorted_probs[0]),
        "entropy": float(entropy),
        "effective_support": float(np.exp(entropy)),
        "support_size": int(sorted_probs.size),
        **topk_mass,
    }


def probability_summary_from_weights(weights, topk=(1, 2, 5, 10)):
    """Summarize a non-negative weight vector via its normalized distribution."""
    raw = np.asarray(weights, dtype=np.float64).reshape(-1)
    raw = raw[np.isfinite(raw) & (raw > 0.0)]
    if raw.size == 0:
        raise ValueError("No positive finite weights provided")
    probs = raw / np.sum(raw)
    return _probability_summary_from_probs(probs, topk=topk)


def score_summary_from_log_scores(scores, topk=(1, 2, 5, 10), mask=None):
    """Summarize log-scores via their normalized softmax distribution."""
    score_arr = np.asarray(scores, dtype=np.float64)
    finite_mask = np.isfinite(score_arr)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != score_arr.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match scores shape {score_arr.shape}")
        finite_mask &= mask_arr
    finite = score_arr[finite_mask]
    if finite.size == 0:
        raise ValueError("No finite scores provided")
    centered = finite - np.max(finite)
    weights = np.exp(centered)
    probs = weights / np.sum(weights)
    summary = _probability_summary_from_probs(probs, topk=topk)
    sorted_scores = np.sort(finite)[::-1]
    summary.update(
        {
            "best_score": float(sorted_scores[0]),
            "runner_up_score": float(sorted_scores[1]) if sorted_scores.size > 1 else float(sorted_scores[0]),
            "best_minus_runner_up": float(sorted_scores[0] - sorted_scores[1])
            if sorted_scores.size > 1
            else float("inf"),
            "support_size": int(finite.size),
        }
    )
    return summary


def masked_score_mass_summary(scores, normalization_mask, support_mask=None, topk=(1, 2, 5, 10)):
    """Summarize support mass and renormalized probabilities under a chosen denominator.

    ``normalization_mask`` defines the denominator used for ``log_Z``.
    ``support_mask`` defines the subset whose mass is reported. When omitted,
    the support equals the denominator.
    """
    score_arr = np.asarray(scores, dtype=np.float64)
    norm_mask = np.asarray(normalization_mask, dtype=bool)
    if norm_mask.shape != score_arr.shape:
        raise ValueError(f"normalization mask shape {norm_mask.shape} does not match scores shape {score_arr.shape}")
    if support_mask is None:
        support_mask = norm_mask
    support_mask = np.asarray(support_mask, dtype=bool)
    if support_mask.shape != score_arr.shape:
        raise ValueError(f"support mask shape {support_mask.shape} does not match scores shape {score_arr.shape}")
    if np.any(support_mask & ~norm_mask):
        raise ValueError("support_mask must be a subset of normalization_mask")

    finite_mask = np.isfinite(score_arr)
    norm_scores = score_arr[finite_mask & norm_mask]
    support_scores = score_arr[finite_mask & support_mask]
    if norm_scores.size == 0:
        raise ValueError("No finite denominator scores provided")
    if support_scores.size == 0:
        raise ValueError("No finite support scores provided")

    shift = float(np.max(norm_scores))
    norm_weights = np.exp(norm_scores - shift)
    support_weights = np.exp(support_scores - shift)
    norm_sum = float(np.sum(norm_weights))
    support_sum = float(np.sum(support_weights))
    normalized_probs = support_weights / norm_sum
    renormalized_probs = support_weights / support_sum
    renorm_summary = _probability_summary_from_probs(renormalized_probs, topk=topk)
    result = {
        "normalization_support_size": int(norm_scores.size),
        "support_size": int(support_scores.size),
        "support_mass": float(support_sum / norm_sum),
        "normalized_pmax": float(np.max(normalized_probs)),
        "normalized_top1_mass": float(np.max(normalized_probs)),
        "normalized_top2_mass": float(np.sum(np.sort(normalized_probs)[::-1][: min(2, normalized_probs.size)])),
        "normalized_top5_mass": float(np.sum(np.sort(normalized_probs)[::-1][: min(5, normalized_probs.size)])),
        "normalized_top10_mass": float(np.sum(np.sort(normalized_probs)[::-1][: min(10, normalized_probs.size)])),
        "log_Z": float(np.log(norm_sum) + shift),
        "score_shift": shift,
        "renormalized": renorm_summary,
    }
    result["pmax"] = result["normalized_pmax"] if np.isclose(result["support_mass"], 1.0) else float(
        renorm_summary["pmax"]
    )
    return result


def solve_scale_for_target_pmax(scores, target_pmax, *, tol=1e-8, max_iter=80):
    """Find a scalar alpha with softmax(alpha * scores).pmax ~= target_pmax."""
    target = float(target_pmax)
    finite = _finite_scores(scores)
    if finite.size == 0:
        raise ValueError("No finite scores provided")
    min_pmax = 1.0 / float(finite.size)
    if not (min_pmax <= target <= 1.0):
        raise ValueError(f"Target Pmax {target} outside [{min_pmax}, 1]")
    centered = finite - np.max(finite)

    def _pmax(alpha):
        weights = np.exp(alpha * centered)
        return float(np.max(weights) / np.sum(weights))

    if abs(_pmax(1.0) - target) <= tol:
        return 1.0

    lo, hi = 0.0, 1.0
    if _pmax(hi) < target:
        while _pmax(hi) < target and hi < 1e6:
            hi *= 2.0
    else:
        while hi > 1e-9 and _pmax(hi) > target:
            hi *= 0.5
        lo, hi = hi, hi * 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pmax_mid = _pmax(mid)
        if abs(pmax_mid - target) <= tol:
            return mid
        if pmax_mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def summarize_relion_operands(relion_npz, topk=(1, 2, 5, 10)):
    """Summarize a RELION operand dump parsed into NPZ form."""
    relion = _load_npz_dict(relion_npz)
    raw_weights = np.asarray(relion["exp_Mweight_posterior"], dtype=np.float64).reshape(-1)
    raw_weights = raw_weights[np.isfinite(raw_weights) & (raw_weights >= 0.0)]
    exp_sum_weight = float(np.asarray(relion["exp_sum_weight"], dtype=np.float64).reshape(()))
    pmax_stored = float(np.asarray(relion["Pmax"], dtype=np.float64).reshape(()))
    pmax_from_raw = float(np.max(raw_weights) / exp_sum_weight)
    summary = probability_summary_from_weights(raw_weights, topk=topk)
    denom_mask = np.asarray(relion.get("candidate_in_denominator_set", np.ones(raw_weights.size, dtype=np.int32)), dtype=bool)
    significant_weight = float(
        np.asarray(relion.get("exp_significant_weight", np.array(0.0, dtype=np.float64)), dtype=np.float64).reshape(())
    )
    fine_threshold_mask = np.asarray(
        relion.get("candidate_in_fine_threshold_set", raw_weights >= significant_weight),
        dtype=bool,
    )
    reconstruction_mask = np.asarray(
        relion.get("candidate_in_reconstruction_set", fine_threshold_mask),
        dtype=bool,
    )
    normalized_weight = np.asarray(
        relion.get("candidate_weight_normalized", raw_weights / exp_sum_weight),
        dtype=np.float64,
    ).reshape(-1)
    cumulative_fraction = np.asarray(
        relion.get("candidate_weight_cumulative_fraction", np.full(raw_weights.size, np.nan, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    sorted_rank = np.asarray(
        relion.get("candidate_sorted_rank", np.arange(raw_weights.size, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    summary.update(
        {
            "exp_sum_weight": exp_sum_weight,
            "exp_max_weight": float(np.max(raw_weights)),
            "stored_pmax": pmax_stored,
            "pmax_from_raw": pmax_from_raw,
            "pmax_semantics_gap": float(pmax_stored - pmax_from_raw),
            "nr_dir": int(np.asarray(relion.get("header_nr_dir", 0)).reshape(())),
            "nr_psi": int(np.asarray(relion.get("header_nr_psi", 0)).reshape(())),
            "nr_trans": int(np.asarray(relion.get("header_nr_trans", 0)).reshape(())),
            "nr_oversampled_rot": int(np.asarray(relion.get("header_nr_oversampled_rot", 0)).reshape(())),
            "nr_oversampled_trans": int(np.asarray(relion.get("header_nr_oversampled_trans", 0)).reshape(())),
            "current_size": int(np.asarray(relion.get("header_current_size", 0)).reshape(())),
            "ori_size": int(np.asarray(relion.get("header_ori_size", 0)).reshape(())),
            "adaptive_fraction": float(np.asarray(relion.get("adaptive_fraction", np.array(np.nan))).reshape(())),
            "maximum_significants": int(np.asarray(relion.get("maximum_significants", np.array(-1))).reshape(())),
            "denominator_count": int(np.count_nonzero(denom_mask)),
            "fine_threshold_count": int(np.count_nonzero(fine_threshold_mask)),
            "reconstruction_count": int(np.count_nonzero(reconstruction_mask)),
            "candidate_threshold_idx": int(np.asarray(relion.get("candidate_threshold_idx", np.array(0))).reshape(())),
            "candidate_threshold_count": int(np.asarray(relion.get("candidate_threshold_count", np.array(np.count_nonzero(fine_threshold_mask)))).reshape(())),
            "candidate_denominator_count": int(np.asarray(relion.get("candidate_denominator_count", np.array(raw_weights.size))).reshape(())),
            "normalized_weight": normalized_weight,
            "cumulative_fraction": cumulative_fraction,
            "sorted_rank": sorted_rank,
            "denominator_mask": denom_mask,
            "fine_threshold_mask": fine_threshold_mask,
            "reconstruction_mask": reconstruction_mask,
        }
    )
    return summary


def summarize_recovar_score_dump(recovar_npz, image_position=0, topk=(1, 2, 5, 10)):
    """Summarize one selected image from a RECOVAR local-search score dump."""
    recovar = _load_npz_dict(recovar_npz)
    image_position = int(image_position)
    raw_scores = np.asarray(recovar["pass2_scores_raw"][image_position], dtype=np.float64)
    total_scores = np.asarray(recovar["pass2_scores_total"][image_position], dtype=np.float64)
    rotation_prior = np.asarray(recovar["rotation_log_prior"][image_position], dtype=np.float64)
    translation_prior = np.asarray(recovar["translation_log_prior"][image_position], dtype=np.float64)
    candidate_mask = np.asarray(recovar["rotation_candidate_mask"][image_position], dtype=bool)

    raw_summary = score_summary_from_log_scores(raw_scores, topk=topk)
    full_summary = score_summary_from_log_scores(total_scores, topk=topk)

    raw_plus_rot = raw_scores + rotation_prior[:, None]
    raw_plus_rot = np.where(candidate_mask[:, None], raw_plus_rot, -np.inf)
    raw_plus_rot_summary = score_summary_from_log_scores(raw_plus_rot, topk=topk)

    saved_pmax = float(np.asarray(recovar["max_posterior"][image_position], dtype=np.float64).reshape(()))
    saved_log_z = float(np.asarray(recovar["log_Z"][image_position], dtype=np.float64).reshape(()))
    saved_best_score = float(np.asarray(recovar["best_score"][image_position], dtype=np.float64).reshape(()))

    cross_norm_consistency = None
    if "pass2_cross_term" in recovar and "pass2_norm_term" in recovar:
        cross = np.asarray(recovar["pass2_cross_term"][image_position], dtype=np.float64)
        norms = np.asarray(recovar["pass2_norm_term"][image_position], dtype=np.float64)
        reconstructed = -0.5 * (cross + norms)
        cross_norm_consistency = float(np.max(np.abs(reconstructed - raw_scores)))

    pass_consistency = None
    if "pass1_scores_raw" in recovar:
        pass1 = np.asarray(recovar["pass1_scores_raw"][image_position], dtype=np.float64)
        finite_mask = np.isfinite(pass1) & np.isfinite(raw_scores)
        pass_consistency = float(np.max(np.abs(pass1[finite_mask] - raw_scores[finite_mask]))) if np.any(finite_mask) else 0.0

    summary = {
        "image_position": image_position,
        "selected_global_image_index": int(np.asarray(recovar["selected_global_image_indices"][image_position])),
        "saved_pmax": saved_pmax,
        "saved_log_Z": saved_log_z,
        "saved_best_score": saved_best_score,
        "recomputed_pmax": full_summary["pmax"],
        "recomputed_log_Z": float(
            np.log(np.sum(np.exp(_finite_scores(total_scores) - np.max(_finite_scores(total_scores)))))
            + np.max(_finite_scores(total_scores))
        ),
        "support_rotations": int(np.count_nonzero(candidate_mask)),
        "support_pairs_upper_bound": int(np.count_nonzero(candidate_mask) * total_scores.shape[1]),
        "pass_raw_max_abs_diff": pass_consistency,
        "cross_norm_max_abs_diff": cross_norm_consistency,
        "raw_only": raw_summary,
        "raw_plus_rotation_prior": raw_plus_rot_summary,
        "full": full_summary,
        "rotation_prior_range": (
            float(np.min(rotation_prior[candidate_mask])) if np.any(candidate_mask) else float("nan"),
            float(np.max(rotation_prior[candidate_mask])) if np.any(candidate_mask) else float("nan"),
        ),
        "translation_prior_range": (
            float(np.min(translation_prior)),
            float(np.max(translation_prior)),
        ),
    }
    return summary


@dataclass
class CandidateMapping:
    relion_rot_id: np.ndarray
    relion_trans_idx: np.ndarray
    relion_coarse_trans_idx: np.ndarray
    relion_translation_x: np.ndarray
    relion_translation_y: np.ndarray
    recovar_rot_slot: np.ndarray
    recovar_trans_idx: np.ndarray
    denominator_mask: np.ndarray
    fine_threshold_mask: np.ndarray
    reconstruction_mask: np.ndarray
    relion_local_rot_id: np.ndarray
    recovar_local_rot_id: np.ndarray
    pixel_support_equal: bool
    psi_support_equal: bool


def _fit_sorted_axis_affine(src_values, dst_values):
    src_unique = np.unique(np.asarray(src_values, dtype=np.float64).reshape(-1))
    dst_unique = np.unique(np.asarray(dst_values, dtype=np.float64).reshape(-1))
    if src_unique.size != dst_unique.size:
        raise ValueError(
            f"Cannot align translation axes with different unique counts: "
            f"{src_unique.size} vs {dst_unique.size}",
        )
    if src_unique.size == 1:
        return 1.0, float(dst_unique[0] - src_unique[0])
    scale = float((dst_unique[-1] - dst_unique[0]) / (src_unique[-1] - src_unique[0]))
    shift = float(np.mean(dst_unique - scale * src_unique))
    return scale, shift


def _translation_snap_tolerance(translations):
    translations = np.asarray(translations, dtype=np.float64).reshape(-1, 2)
    steps = []
    for axis in range(2):
        unique = np.unique(np.round(translations[:, axis], decimals=12))
        if unique.size > 1:
            diffs = np.diff(unique)
            positive = diffs[diffs > 1e-12]
            if positive.size:
                steps.append(float(np.min(positive)))
    if not steps:
        return 1e-3
    return 0.5 * max(steps) + 1e-6


def _map_relion_translation_indices(relion, recovar):
    relion_trans_idx = np.asarray(relion["acc_trans_idx"], dtype=np.int64).reshape(-1)
    relion_coarse_trans_idx = np.asarray(
        relion.get("candidate_coarse_trans_idx", relion_trans_idx),
        dtype=np.int64,
    ).reshape(-1)
    recovar_translations = np.asarray(recovar["translations"], dtype=np.float64).reshape(-1, 2)
    relion_tx = np.asarray(relion["translations_x"], dtype=np.float64).reshape(-1)
    relion_ty = np.asarray(relion["translations_y"], dtype=np.float64).reshape(-1)

    scale_x, shift_x = _fit_sorted_axis_affine(relion_tx, recovar_translations[:, 0])
    scale_y, shift_y = _fit_sorted_axis_affine(relion_ty, recovar_translations[:, 1])
    candidate_tolerance = _translation_snap_tolerance(recovar_translations)
    relion_coarse_coords = np.column_stack(
        [
            scale_x * relion_tx + shift_x,
            scale_y * relion_ty + shift_y,
        ]
    )

    coarse_to_recovar = np.empty(relion_coarse_coords.shape[0], dtype=np.int64)
    max_mapping_error = 0.0
    for idx, coord in enumerate(relion_coarse_coords):
        sqdist = np.sum((recovar_translations - coord[None, :]) ** 2, axis=1)
        best = int(np.argmin(sqdist))
        coarse_to_recovar[idx] = best
        max_mapping_error = max(max_mapping_error, float(np.sqrt(sqdist[best])))

    has_candidate_translation = "candidate_translation_x" in relion and "candidate_translation_y" in relion
    if (not has_candidate_translation) and max_mapping_error > 1e-3:
        raise ValueError(
            f"RELION->RECOVAR translation mapping error too large: {max_mapping_error:.6f}",
        )

    if has_candidate_translation:
        relion_candidate_coords = np.column_stack(
            [
                np.asarray(relion["candidate_translation_x"], dtype=np.float64).reshape(-1),
                np.asarray(relion["candidate_translation_y"], dtype=np.float64).reshape(-1),
            ]
        )
        recovar_trans_idx = np.empty(relion_candidate_coords.shape[0], dtype=np.int64)
        max_candidate_error = 0.0
        for idx, coord in enumerate(relion_candidate_coords):
            sqdist = np.sum((recovar_translations - coord[None, :]) ** 2, axis=1)
            best = int(np.argmin(sqdist))
            recovar_trans_idx[idx] = best
            max_candidate_error = max(max_candidate_error, float(np.sqrt(sqdist[best])))
        if max_candidate_error > candidate_tolerance:
            raise ValueError(
                f"RELION candidate->RECOVAR translation mapping error too large: {max_candidate_error:.6f}",
            )
    else:
        relion_candidate_coords = relion_coarse_coords[relion_coarse_trans_idx]
        recovar_trans_idx = coarse_to_recovar[relion_coarse_trans_idx]

    return (
        relion_trans_idx,
        relion_coarse_trans_idx,
        relion_candidate_coords[:, 0],
        relion_candidate_coords[:, 1],
        recovar_trans_idx,
    )


def build_relion_recovar_candidate_mapping(relion_npz, recovar_npz, image_position=0):
    """Map RELION factorized candidates onto one RECOVAR local score dump."""
    relion = _load_npz_dict(relion_npz)
    recovar = _load_npz_dict(recovar_npz)
    image_position = int(image_position)

    required = {
        "local_rotation_pixel_indices",
        "local_rotation_psi_indices",
        "grid_n_psi",
        "rotation_candidate_mask",
        "pass2_scores_total",
    }
    missing = [key for key in required if key not in recovar]
    if missing:
        raise KeyError(f"RECOVAR dump missing factorized support metadata: {missing}")

    recovar_pixel = np.asarray(recovar["local_rotation_pixel_indices"], dtype=np.int64).reshape(-1)
    recovar_psi = np.asarray(recovar["local_rotation_psi_indices"], dtype=np.int64).reshape(-1)
    relion_pixel = np.asarray(relion["pointer_dir_nonzeroprior"], dtype=np.int64).reshape(-1)
    relion_psi = np.asarray(relion["pointer_psi_nonzeroprior"], dtype=np.int64).reshape(-1)
    npsi = relion_psi.size
    pixel_support_equal = np.array_equal(np.unique(recovar_pixel), relion_pixel)
    psi_support_equal = np.array_equal(np.unique(recovar_psi), relion_psi)

    pixel_pos = {int(value): idx for idx, value in enumerate(relion_pixel)}
    psi_pos = {int(value): idx for idx, value in enumerate(relion_psi)}
    recovar_local_rot_id = np.array(
        [pixel_pos[int(pixel)] * npsi + psi_pos[int(psi)] for pixel, psi in zip(recovar_pixel, recovar_psi)],
        dtype=np.int64,
    )
    rot_id_to_slot = {int(rot_id): idx for idx, rot_id in enumerate(recovar_local_rot_id)}

    relion_rot_id = np.asarray(relion["acc_rot_id"], dtype=np.int64).reshape(-1)
    (
        relion_trans_idx,
        relion_coarse_trans_idx,
        relion_translation_x,
        relion_translation_y,
        recovar_trans_idx,
    ) = _map_relion_translation_indices(relion, recovar)
    recovar_rot_slot = np.array([rot_id_to_slot[int(rot_id)] for rot_id in relion_rot_id], dtype=np.int64)
    score_shape = np.asarray(recovar["pass2_scores_total"][image_position]).shape
    denominator_mask = np.zeros(score_shape, dtype=bool)
    fine_threshold_mask = np.zeros(score_shape, dtype=bool)
    reconstruction_mask = np.zeros(score_shape, dtype=bool)

    relion_summary = summarize_relion_operands(relion)
    denominator_member = np.asarray(relion_summary["denominator_mask"], dtype=bool).reshape(-1)
    fine_threshold_member = np.asarray(relion_summary["fine_threshold_mask"], dtype=bool).reshape(-1)
    reconstruction_member = np.asarray(relion_summary["reconstruction_mask"], dtype=bool).reshape(-1)
    for idx, (rot_slot, trans_idx) in enumerate(zip(recovar_rot_slot, recovar_trans_idx)):
        denominator_mask[int(rot_slot), int(trans_idx)] = denominator_member[idx]
        fine_threshold_mask[int(rot_slot), int(trans_idx)] = fine_threshold_member[idx]
        reconstruction_mask[int(rot_slot), int(trans_idx)] = reconstruction_member[idx]

    return CandidateMapping(
        relion_rot_id=relion_rot_id,
        relion_trans_idx=relion_trans_idx,
        relion_coarse_trans_idx=relion_coarse_trans_idx,
        relion_translation_x=relion_translation_x,
        relion_translation_y=relion_translation_y,
        recovar_rot_slot=recovar_rot_slot,
        recovar_trans_idx=recovar_trans_idx,
        denominator_mask=denominator_mask,
        fine_threshold_mask=fine_threshold_mask,
        reconstruction_mask=reconstruction_mask,
        relion_local_rot_id=relion_rot_id,
        recovar_local_rot_id=recovar_local_rot_id,
        pixel_support_equal=pixel_support_equal,
        psi_support_equal=psi_support_equal,
    )


def summarize_recovar_mask_ladder(recovar_npz, mapping: CandidateMapping, image_position=0, topk=(1, 2, 5, 10)):
    """Compute full, denominator, and threshold summaries for one RECOVAR dump."""
    recovar = _load_npz_dict(recovar_npz)
    image_position = int(image_position)
    total_scores = np.asarray(recovar["pass2_scores_total"][image_position], dtype=np.float64)
    raw_scores = np.asarray(recovar["pass2_scores_raw"][image_position], dtype=np.float64)
    denominator_mask = _align_mask_to_score_shape(mapping.denominator_mask, total_scores.shape, name="denominator_mask")
    fine_threshold_mask = _align_mask_to_score_shape(
        mapping.fine_threshold_mask,
        total_scores.shape,
        name="fine_threshold_mask",
    )
    reconstruction_mask = _align_mask_to_score_shape(
        mapping.reconstruction_mask,
        total_scores.shape,
        name="reconstruction_mask",
    )
    full_mask = np.isfinite(total_scores)
    full = masked_score_mass_summary(total_scores, full_mask, full_mask, topk=topk)
    denominator_under_full = masked_score_mass_summary(total_scores, full_mask, denominator_mask, topk=topk)
    threshold_under_full = masked_score_mass_summary(total_scores, full_mask, fine_threshold_mask, topk=topk)
    reconstruction_under_full = masked_score_mass_summary(
        total_scores,
        full_mask,
        reconstruction_mask,
        topk=topk,
    )
    denominator = masked_score_mass_summary(total_scores, denominator_mask, denominator_mask, topk=topk)
    threshold_denorm = masked_score_mass_summary(
        total_scores,
        denominator_mask,
        fine_threshold_mask,
        topk=topk,
    )
    threshold_renorm = masked_score_mass_summary(
        total_scores,
        fine_threshold_mask,
        fine_threshold_mask,
        topk=topk,
    )
    raw_denominator = masked_score_mass_summary(raw_scores, denominator_mask, denominator_mask, topk=topk)
    return {
        "full": full,
        "denominator_under_full": denominator_under_full,
        "threshold_under_full": threshold_under_full,
        "reconstruction_under_full": reconstruction_under_full,
        "denominator": denominator,
        "threshold_denorm": threshold_denorm,
        "threshold_renorm": threshold_renorm,
        "raw_denominator": raw_denominator,
    }


def compare_shared_subset_score_deltas(relion_npz, recovar_npz, mapping: CandidateMapping, image_position=0):
    """Compare RELION log-weights and RECOVAR scores on the mapped denominator set."""
    relion = _load_npz_dict(relion_npz)
    recovar = _load_npz_dict(recovar_npz)
    relion_weights = np.asarray(relion["exp_Mweight_posterior"], dtype=np.float64).reshape(-1)
    relion_log = np.log(relion_weights)
    denominator_member = np.asarray(
        summarize_relion_operands(relion)["denominator_mask"],
        dtype=bool,
    ).reshape(-1)
    relion_log = relion_log[denominator_member]
    relion_rot_id = np.asarray(relion["acc_rot_id"], dtype=np.int64).reshape(-1)[denominator_member]
    relion_trans_idx = np.asarray(mapping.relion_trans_idx, dtype=np.int64).reshape(-1)[denominator_member]
    recovar_trans_idx = np.asarray(mapping.recovar_trans_idx, dtype=np.int64).reshape(-1)[denominator_member]

    mapped_slots = mapping.recovar_rot_slot[denominator_member]
    recovar_scores = np.asarray(recovar["pass2_scores_total"][int(image_position)], dtype=np.float64)
    mapped_scores = np.array(
        [recovar_scores[int(slot), int(trans)] for slot, trans in zip(mapped_slots, recovar_trans_idx)],
        dtype=np.float64,
    )
    best_idx = int(np.argmax(relion_log))
    relion_delta = relion_log - relion_log[best_idx]
    recovar_delta = mapped_scores - mapped_scores[best_idx]
    delta_error = recovar_delta - relion_delta
    coeff = np.polyfit(recovar_delta, relion_delta, 1)
    pred = coeff[0] * recovar_delta + coeff[1]
    ss_res = float(np.sum((relion_delta - pred) ** 2))
    ss_tot = float(np.sum((relion_delta - np.mean(relion_delta)) ** 2))
    return {
        "best_relion_rot_id": int(relion_rot_id[best_idx]),
        "best_relion_trans_idx": int(relion_trans_idx[best_idx]),
        "corr": float(np.corrcoef(recovar_delta, relion_delta)[0, 1]),
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "mean_abs_delta_error": float(np.mean(np.abs(delta_error))),
        "max_abs_delta_error": float(np.max(np.abs(delta_error))),
    }


def build_shared_subset_candidate_table(relion_npz, recovar_npz, mapping: CandidateMapping, image_position=0):
    """Return per-candidate RELION-vs-RECOVAR data on the exact RELION denominator set."""
    relion = _load_npz_dict(relion_npz)
    recovar = _load_npz_dict(recovar_npz)
    image_position = int(image_position)

    relion_summary = summarize_relion_operands(relion)
    denominator_member = np.asarray(relion_summary["denominator_mask"], dtype=bool).reshape(-1)
    fine_threshold_member = np.asarray(relion_summary["fine_threshold_mask"], dtype=bool).reshape(-1)[denominator_member]
    reconstruction_member = np.asarray(relion_summary["reconstruction_mask"], dtype=bool).reshape(-1)[denominator_member]

    relion_rot_id_all = np.asarray(relion["acc_rot_id"], dtype=np.int64).reshape(-1)
    relion_trans_idx_all = np.asarray(mapping.relion_trans_idx, dtype=np.int64).reshape(-1)
    relion_translation_x_all = np.asarray(mapping.relion_translation_x, dtype=np.float64).reshape(-1)
    relion_translation_y_all = np.asarray(mapping.relion_translation_y, dtype=np.float64).reshape(-1)
    relion_raw_weight_all = np.asarray(relion["exp_Mweight_posterior"], dtype=np.float64).reshape(-1)
    relion_norm_weight_all = np.asarray(relion_summary["normalized_weight"], dtype=np.float64).reshape(-1)
    relion_rank_all = np.asarray(relion_summary["sorted_rank"], dtype=np.int64).reshape(-1)
    relion_orientation_prior_all = np.asarray(
        relion.get("candidate_orientation_log_prior", np.zeros_like(relion_raw_weight_all)),
        dtype=np.float64,
    ).reshape(-1)
    relion_offset_prior_all = np.asarray(
        relion.get("candidate_offset_log_prior", np.zeros_like(relion_raw_weight_all)),
        dtype=np.float64,
    ).reshape(-1)
    relion_combined_prior_all = np.asarray(
        relion.get("candidate_combined_log_prior", relion_orientation_prior_all + relion_offset_prior_all),
        dtype=np.float64,
    ).reshape(-1)
    relion_coarse_trans_idx_all = np.asarray(mapping.relion_coarse_trans_idx, dtype=np.int64).reshape(-1)
    recovar_trans_idx_all = np.asarray(mapping.recovar_trans_idx, dtype=np.int64).reshape(-1)

    relion_rot_id = relion_rot_id_all[denominator_member]
    relion_trans_idx = relion_trans_idx_all[denominator_member]
    relion_translation_x = relion_translation_x_all[denominator_member]
    relion_translation_y = relion_translation_y_all[denominator_member]
    relion_raw_weight = relion_raw_weight_all[denominator_member]
    relion_norm_weight = relion_norm_weight_all[denominator_member]
    relion_rank = relion_rank_all[denominator_member]
    relion_orientation_prior = relion_orientation_prior_all[denominator_member]
    relion_offset_prior = relion_offset_prior_all[denominator_member]
    relion_combined_prior = relion_combined_prior_all[denominator_member]
    relion_coarse_trans_idx = relion_coarse_trans_idx_all[denominator_member]
    recovar_trans_idx = recovar_trans_idx_all[denominator_member]
    relion_log_weight = np.log(relion_raw_weight)
    relion_data_log_weight_centered = relion_log_weight - relion_combined_prior
    relion_dir_list = np.asarray(relion["pointer_dir_nonzeroprior"], dtype=np.int64).reshape(-1)
    relion_psi_list = np.asarray(relion["pointer_psi_nonzeroprior"], dtype=np.int64).reshape(-1)
    relion_n_psi = relion_psi_list.size
    relion_dir_pos = relion_rot_id // relion_n_psi
    relion_psi_pos = relion_rot_id % relion_n_psi
    relion_pixel_index = relion_dir_list[relion_dir_pos]
    relion_psi_index = relion_psi_list[relion_psi_pos]

    recovar_total_scores = np.asarray(recovar["pass2_scores_total"][image_position], dtype=np.float64)
    recovar_raw_scores = np.asarray(recovar["pass2_scores_raw"][image_position], dtype=np.float64)
    recovar_cross_term_all = _optional_recovar_array(
        recovar,
        "pass2_cross_term",
        recovar["pass2_scores_total"].shape[:2] + recovar["pass2_scores_total"].shape[2:],
    )[image_position]
    recovar_norm_term_all = _optional_recovar_array(
        recovar,
        "pass2_norm_term",
        recovar["pass2_scores_total"].shape[:2] + (1,),
        fill_value=0.0,
    )[image_position]
    recovar_rotation_prior = _optional_recovar_array(
        recovar,
        "rotation_log_prior",
        recovar["pass2_scores_total"].shape[:2],
        fill_value=0.0,
    )[image_position]
    recovar_translation_prior = _optional_recovar_array(
        recovar,
        "translation_log_prior",
        (recovar["pass2_scores_total"].shape[0], recovar["pass2_scores_total"].shape[2]),
        fill_value=0.0,
    )[image_position]
    recovar_pixel_indices = np.asarray(recovar["local_rotation_pixel_indices"], dtype=np.int64).reshape(-1)
    recovar_psi_indices = np.asarray(recovar["local_rotation_psi_indices"], dtype=np.int64).reshape(-1)
    recovar_psi_deg_all = _optional_recovar_array(
        recovar,
        "local_rotation_psi_deg",
        recovar_pixel_indices.shape,
    ).reshape(-1)
    recovar_dir_vecs_all = _optional_recovar_array(
        recovar,
        "local_rotation_dir_vecs",
        (recovar_pixel_indices.shape[0], 3),
    ).reshape(-1, 3)
    recovar_eulers_all = _optional_recovar_array(
        recovar,
        "local_rotation_eulers",
        (recovar_pixel_indices.shape[0], 3),
    ).reshape(-1, 3)
    recovar_translations_all = np.asarray(recovar["translations"], dtype=np.float64).reshape(-1, 2)

    recovar_full_prob = _softmax_probabilities_from_scores(recovar_total_scores)
    recovar_denominator_prob = _softmax_probabilities_from_scores(
        recovar_total_scores,
        mask=_align_mask_to_score_shape(mapping.denominator_mask, recovar_total_scores.shape, name="denominator_mask"),
    )

    mapped_slots = mapping.recovar_rot_slot[denominator_member]
    recovar_total = np.array(
        [recovar_total_scores[int(slot), int(trans)] for slot, trans in zip(mapped_slots, recovar_trans_idx)],
        dtype=np.float64,
    )
    recovar_raw = np.array(
        [recovar_raw_scores[int(slot), int(trans)] for slot, trans in zip(mapped_slots, recovar_trans_idx)],
        dtype=np.float64,
    )
    recovar_cross = np.array(
        [recovar_cross_term_all[int(slot), int(trans)] for slot, trans in zip(mapped_slots, recovar_trans_idx)],
        dtype=np.float64,
    )
    recovar_norm = np.array(
        [recovar_norm_term_all[int(slot), 0] for slot in mapped_slots],
        dtype=np.float64,
    )
    recovar_rot_prior = np.array(
        [recovar_rotation_prior[int(slot)] for slot in mapped_slots],
        dtype=np.float64,
    )
    recovar_trans_prior = np.array(
        [recovar_translation_prior[int(trans)] for trans in recovar_trans_idx],
        dtype=np.float64,
    )
    recovar_full_prob_selected = np.array(
        [recovar_full_prob[int(slot), int(trans)] for slot, trans in zip(mapped_slots, recovar_trans_idx)],
        dtype=np.float64,
    )
    recovar_denominator_prob_selected = np.array(
        [recovar_denominator_prob[int(slot), int(trans)] for slot, trans in zip(mapped_slots, recovar_trans_idx)],
        dtype=np.float64,
    )
    recovar_pixel_index = recovar_pixel_indices[mapped_slots]
    recovar_psi_index = recovar_psi_indices[mapped_slots]
    recovar_psi_deg = recovar_psi_deg_all[mapped_slots]
    recovar_dir_vec = recovar_dir_vecs_all[mapped_slots]
    recovar_eulers = recovar_eulers_all[mapped_slots]
    recovar_translation = recovar_translations_all[recovar_trans_idx]

    relion_best_idx = int(np.argmax(relion_log_weight))
    recovar_best_idx = int(np.argmax(recovar_total))
    relion_delta = relion_log_weight - relion_log_weight[relion_best_idx]
    recovar_delta_relion_best = recovar_total - recovar_total[relion_best_idx]
    recovar_delta_recovar_best = recovar_total - recovar_total[recovar_best_idx]
    relion_prior_delta = relion_combined_prior - relion_combined_prior[relion_best_idx]
    recovar_prior_total = recovar_rot_prior + recovar_trans_prior
    recovar_prior_delta = recovar_prior_total - recovar_prior_total[relion_best_idx]
    recovar_cross_score_delta_relion_best = -0.5 * (recovar_cross - recovar_cross[relion_best_idx])
    recovar_norm_score_delta_relion_best = -0.5 * (recovar_norm - recovar_norm[relion_best_idx])
    recovar_data_delta_relion_best = recovar_raw - recovar_raw[relion_best_idx]
    relion_data_delta_relion_best = relion_data_log_weight_centered - relion_data_log_weight_centered[relion_best_idx]

    return {
        "candidate_index": np.arange(relion_rot_id.size, dtype=np.int64),
        "relion_rot_id": relion_rot_id,
        "relion_trans_idx": relion_trans_idx,
        "relion_translation_x": relion_translation_x,
        "relion_translation_y": relion_translation_y,
        "relion_pixel_index": relion_pixel_index,
        "relion_psi_index": relion_psi_index,
        "relion_raw_weight": relion_raw_weight,
        "relion_normalized_weight": relion_norm_weight,
        "relion_log_weight": relion_log_weight,
        "relion_orientation_log_prior": relion_orientation_prior,
        "relion_offset_log_prior": relion_offset_prior,
        "relion_combined_log_prior": relion_combined_prior,
        "relion_data_log_weight_centered": relion_data_log_weight_centered,
        "relion_prior_delta_to_relion_best": relion_prior_delta,
        "relion_data_delta_to_relion_best": relion_data_delta_relion_best,
        "relion_sorted_rank": relion_rank,
        "relion_coarse_trans_idx": relion_coarse_trans_idx,
        "recovar_trans_idx": recovar_trans_idx,
        "relion_in_fine_threshold_set": fine_threshold_member.astype(np.int32),
        "relion_in_reconstruction_set": reconstruction_member.astype(np.int32),
        "recovar_rot_slot": mapped_slots,
        "recovar_pixel_index": recovar_pixel_index,
        "recovar_psi_index": recovar_psi_index,
        "recovar_psi_deg": recovar_psi_deg,
        "recovar_dir_x": recovar_dir_vec[:, 0],
        "recovar_dir_y": recovar_dir_vec[:, 1],
        "recovar_dir_z": recovar_dir_vec[:, 2],
        "recovar_euler_rot": recovar_eulers[:, 0],
        "recovar_euler_tilt": recovar_eulers[:, 1],
        "recovar_euler_psi": recovar_eulers[:, 2],
        "recovar_translation_x": recovar_translation[:, 0],
        "recovar_translation_y": recovar_translation[:, 1],
        "recovar_cross_term": recovar_cross,
        "recovar_norm_term": recovar_norm,
        "recovar_raw_score": recovar_raw,
        "recovar_rotation_log_prior": recovar_rot_prior,
        "recovar_translation_log_prior": recovar_trans_prior,
        "recovar_combined_log_prior": recovar_prior_total,
        "recovar_total_score": recovar_total,
        "recovar_full_probability": recovar_full_prob_selected,
        "recovar_denominator_probability": recovar_denominator_prob_selected,
        "relion_delta_to_best": relion_delta,
        "recovar_data_delta_to_relion_best": recovar_data_delta_relion_best,
        "recovar_cross_score_delta_to_relion_best": recovar_cross_score_delta_relion_best,
        "recovar_norm_score_delta_to_relion_best": recovar_norm_score_delta_relion_best,
        "recovar_prior_delta_to_relion_best": recovar_prior_delta,
        "recovar_delta_to_relion_best": recovar_delta_relion_best,
        "recovar_delta_to_recovar_best": recovar_delta_recovar_best,
        "prior_error": recovar_prior_total - relion_combined_prior,
        "prior_delta_error_to_relion_best": recovar_prior_delta - relion_prior_delta,
        "data_delta_error_to_relion_best": recovar_data_delta_relion_best - relion_data_delta_relion_best,
        "delta_error_to_relion_best": recovar_delta_relion_best - relion_delta,
        "best_relion_candidate_index": np.int64(relion_best_idx),
        "best_recovar_candidate_index": np.int64(recovar_best_idx),
    }


def summarize_candidate_table_components(candidate_table):
    """Summarize total-score, prior, and data-term agreement on the common set."""
    table = _load_npz_dict(candidate_table)

    def _stats(values):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return {"mean_abs": float("nan"), "max_abs": float("nan"), "rms": float("nan")}
        return {
            "mean_abs": float(np.mean(np.abs(arr))),
            "max_abs": float(np.max(np.abs(arr))),
            "rms": float(np.sqrt(np.mean(arr**2))),
        }

    relion_total_delta = np.asarray(table["relion_delta_to_best"], dtype=np.float64)
    recovar_total_delta = np.asarray(table["recovar_delta_to_relion_best"], dtype=np.float64)
    relion_data_delta = np.asarray(table["relion_data_delta_to_relion_best"], dtype=np.float64)
    recovar_data_delta = np.asarray(table["recovar_data_delta_to_relion_best"], dtype=np.float64)
    relion_prior_delta = np.asarray(table["relion_prior_delta_to_relion_best"], dtype=np.float64)
    recovar_prior_delta = np.asarray(table["recovar_prior_delta_to_relion_best"], dtype=np.float64)

    total_delta_error = recovar_total_delta - relion_total_delta
    data_delta_error = recovar_data_delta - relion_data_delta
    prior_delta_error = recovar_prior_delta - relion_prior_delta
    prior_level_error = np.asarray(table["prior_error"], dtype=np.float64)

    relion_best_idx = int(np.asarray(table["best_relion_candidate_index"]).reshape(()))
    recovar_best_idx = int(np.asarray(table["best_recovar_candidate_index"]).reshape(()))

    return {
        "candidate_count": int(relion_total_delta.size),
        "best_relion_candidate_index": relion_best_idx,
        "best_recovar_candidate_index": recovar_best_idx,
        "same_best_candidate": bool(relion_best_idx == recovar_best_idx),
        "total_delta_error": _stats(total_delta_error),
        "data_delta_error": _stats(data_delta_error),
        "prior_delta_error": _stats(prior_delta_error),
        "prior_level_error": _stats(prior_level_error),
    }


def compare_relion_recovar_pmax(relion_summary, recovar_summary):
    """Return a compact comparison between RELION and RECOVAR summaries."""
    relion_pmax = float(relion_summary["pmax"])
    recovar_pmax = float(recovar_summary["full"]["pmax"])
    raw_pmax = float(recovar_summary["raw_only"]["pmax"])
    return {
        "relion_pmax": relion_pmax,
        "recovar_pmax": recovar_pmax,
        "gap": float(recovar_pmax - relion_pmax),
        "recovar_raw_only_pmax": raw_pmax,
        "prior_delta_pmax": float(recovar_pmax - raw_pmax),
        "relion_effective_support": float(relion_summary["effective_support"]),
        "recovar_effective_support": float(recovar_summary["full"]["effective_support"]),
    }
