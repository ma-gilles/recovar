"""Debug-only dump helpers for the exact-local RELION refinement path."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recovar import utils
from recovar.em.dense_single_volume.helpers.env_flags import parse_int_set
from recovar.em.dense_single_volume.helpers.half_spectrum import bin_shell_values_np


@dataclass(frozen=True)
class DensePerPoseScoreDumpRequest:
    """Dense/global per-pose score dump request parsed from environment."""

    dump_dir: Path | None = None
    target: int | None = None
    dump_preprior: bool = False
    target_is_original: bool = False

    @property
    def enabled(self) -> bool:
        return self.dump_dir is not None and self.target is not None


def parse_debug_score_dump_request():
    """Return the optional debug score-dump request from the environment."""

    dump_dir = os.environ.get("RECOVAR_LOCAL_SCORE_DUMP_DIR")
    dump_indices = os.environ.get("RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES")
    dump_current_size = os.environ.get("RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE")
    dump_iterations = os.environ.get("RECOVAR_LOCAL_SCORE_DUMP_ITERATION")
    if not dump_dir or not dump_indices:
        return None, set(), None, None
    targets = parse_int_set(dump_indices) or set()
    if not targets:
        return None, set(), None, None
    requested_current_sizes = parse_int_set(dump_current_size)
    requested_iterations = parse_int_set(dump_iterations)
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    return dump_path, targets, requested_current_sizes, requested_iterations


def parse_debug_noise_component_dump_request():
    """Return optional per-particle local noise component dump settings."""

    dump_dir = os.environ.get("RECOVAR_LOCAL_NOISE_COMPONENT_DUMP_DIR")
    dump_indices = os.environ.get("RECOVAR_LOCAL_NOISE_COMPONENT_DUMP_GLOBAL_INDICES")
    dump_current_size = os.environ.get("RECOVAR_LOCAL_NOISE_COMPONENT_DUMP_CURRENT_SIZE")
    dump_iterations = os.environ.get("RECOVAR_LOCAL_NOISE_COMPONENT_DUMP_ITERATION")
    if not dump_dir or not dump_indices:
        return None, set(), None, None
    targets = parse_int_set(dump_indices) or set()
    if not targets:
        return None, set(), None, None
    requested_current_sizes = parse_int_set(dump_current_size)
    requested_iterations = parse_int_set(dump_iterations)
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    return dump_path, targets, requested_current_sizes, requested_iterations


def parse_dense_noise_component_dump_request():
    """Return optional per-particle dense noise component dump settings."""

    dump_dir = os.environ.get("RECOVAR_DENSE_NOISE_COMPONENT_DUMP_DIR")
    dump_indices = os.environ.get("RECOVAR_DENSE_NOISE_COMPONENT_DUMP_GLOBAL_INDICES")
    dump_current_size = os.environ.get("RECOVAR_DENSE_NOISE_COMPONENT_DUMP_CURRENT_SIZE")
    if not dump_dir or not dump_indices:
        return None, set(), None
    targets = parse_int_set(dump_indices) or set()
    if not targets:
        return None, set(), None
    requested_current_sizes = parse_int_set(dump_current_size)
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    return dump_path, targets, requested_current_sizes


def parse_dense_per_pose_score_dump_request() -> DensePerPoseScoreDumpRequest:
    """Return optional dense/global per-pose score dump settings."""

    dump_dir = os.environ.get("RECOVAR_DEBUG_PER_POSE_DUMP_DIR")
    dump_target = os.environ.get("RECOVAR_DEBUG_PER_POSE_DUMP_TARGET")
    if not dump_dir or dump_target is None:
        return DensePerPoseScoreDumpRequest()
    try:
        target = int(dump_target)
    except ValueError:
        return DensePerPoseScoreDumpRequest()
    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_preprior = os.environ.get("RECOVAR_DEBUG_PER_POSE_DUMP_PREPRIOR")
    target_is_original = os.environ.get("RECOVAR_DEBUG_PER_POSE_DUMP_TARGET_IS_ORIGINAL")
    return DensePerPoseScoreDumpRequest(
        dump_dir=dump_path,
        target=target,
        dump_preprior=bool(dump_preprior and dump_preprior != "0"),
        target_is_original=bool(target_is_original and target_is_original != "0"),
    )


def _dense_score_dump_label_suffix() -> str:
    """Return a sanitized optional label suffix for dense score dumps."""

    label = os.environ.get("RECOVAR_DEBUG_PER_POSE_DUMP_LABEL")
    if not label:
        return ""
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return f"_{label}" if label else ""


def dense_score_dump_label_suffix() -> str:
    """Return the optional label suffix shared by dense score diagnostics."""

    return _dense_score_dump_label_suffix()


def maybe_write_dense_per_pose_score_dump(
    *,
    request: DensePerPoseScoreDumpRequest,
    indices,
    scores,
    block_index: int,
    preprior: bool = False,
    original_indices=None,
) -> None:
    """Dump one dense/global score block for a targeted input image."""

    if not request.enabled:
        return
    if preprior and not request.dump_preprior:
        return
    try:
        match_indices = original_indices if request.target_is_original else indices
        hits = np.where(np.asarray(match_indices, dtype=np.int64) == int(request.target))[0]
        if len(hits) == 0:
            return
        row = int(hits[0])
        suffix = "_preprior" if preprior else ""
        label_suffix = _dense_score_dump_label_suffix()
        scores_target = np.asarray(scores[row], dtype=np.float64)
        np.save(
            request.dump_dir / f"target{int(request.target):06d}{label_suffix}_block{int(block_index):04d}{suffix}.npy",
            scores_target,
        )
    except Exception:
        return


def noise_split_diagnostics_requested() -> bool:
    """Return whether per-shell A2/XA noise split diagnostics are needed."""

    return bool(
        os.environ.get("RECOVAR_NOISE_DEBUG_DUMP_DIR")
        or os.environ.get("RECOVAR_LOCAL_NOISE_COMPONENT_DUMP_DIR")
    )


def maybe_write_debug_noise_component_dump(
    *,
    experiment_dataset,
    bucket,
    support_mass,
    processed_noise_power_half,
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs,
    noise_variance_for_noise,
    shell_indices_half,
    shell_indices_noise,
    n_shells,
    current_size,
    debug_iteration,
    reconstruction_sample_mask,
    n_significant_samples,
    dump_dir: Path | None,
    pending_targets: set[int],
    requested_current_sizes: set[int] | None = None,
    requested_iterations: set[int] | None = None,
):
    """Dump per-particle RELION-style noise components for selected images."""

    if dump_dir is None or not pending_targets:
        return pending_targets
    if requested_current_sizes is not None and int(current_size or -1) not in requested_current_sizes:
        return pending_targets
    if requested_iterations is not None and int(debug_iteration or -1) not in requested_iterations:
        return pending_targets

    original_image_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(bucket.image_indices),
        dtype=np.int64,
    )
    target_rows = [
        row
        for row, original_idx in enumerate(original_image_indices.tolist())
        if int(original_idx) in pending_targets
    ]
    if not target_rows:
        return pending_targets

    support_mass_np = np.asarray(support_mass, dtype=np.float64)
    processed_noise_power_np = np.asarray(processed_noise_power_half)
    proj_np = np.asarray(proj_for_noise)
    proj_abs2_np = (
        np.abs(proj_np) ** 2
        if proj_abs2_for_noise is None
        else np.asarray(proj_abs2_for_noise, dtype=np.float64)
    )
    summed_np = np.asarray(summed_masked_noise)
    ctf_probs_np = np.asarray(ctf_probs, dtype=np.float64)
    noise_variance_np = np.asarray(noise_variance_for_noise, dtype=np.float64)
    shell_indices_half_np = np.asarray(shell_indices_half, dtype=np.int64)
    shell_indices_noise_np = np.asarray(shell_indices_noise, dtype=np.int64)
    reconstruction_sample_mask_np = np.asarray(reconstruction_sample_mask, dtype=bool)
    n_significant_samples_np = np.asarray(n_significant_samples, dtype=np.int32)

    for row in target_rows:
        original_idx = int(original_image_indices[row])
        local_idx = int(bucket.image_indices[row])
        p_img_pixel = (np.abs(processed_noise_power_np[row]) ** 2) * support_mass_np[row]
        p_img_shells = bin_shell_values_np(p_img_pixel, shell_indices_half_np, n_shells)

        ctf_probs_raw = ctf_probs_np[row] * noise_variance_np[None, :]
        a2_pixel = np.sum(proj_abs2_np[row] * ctf_probs_raw, axis=0)
        xa_pixel = noise_variance_np * np.real(np.sum(proj_np[row] * np.conj(summed_np[row]), axis=0))
        a2_shells = bin_shell_values_np(a2_pixel, shell_indices_noise_np, n_shells)
        xa_shells = bin_shell_values_np(xa_pixel, shell_indices_noise_np, n_shells)
        total_shells = p_img_shells + a2_shells - 2.0 * xa_shells

        significant = reconstruction_sample_mask_np[row, : int(bucket.actual_rotation_counts[row]), :]
        dump_path = dump_dir / f"local_noise_components_it{int(debug_iteration or -1):03d}_image_{original_idx}.npz"
        np.savez_compressed(
            dump_path,
            selected_global_image_indices=np.array([original_idx], dtype=np.int64),
            selected_local_image_indices=np.array([local_idx], dtype=np.int64),
            current_size=np.array([int(current_size) if current_size is not None else -1], dtype=np.int32),
            debug_iteration=np.array([int(debug_iteration or -1)], dtype=np.int32),
            support_mass=np.array([support_mass_np[row]], dtype=np.float64),
            n_significant_samples=np.array([int(n_significant_samples_np[row])], dtype=np.int32),
            significant_count=np.array([int(np.sum(significant))], dtype=np.int32),
            p_img_shells=p_img_shells.astype(np.float64),
            a2_shells=a2_shells.astype(np.float64),
            xa_shells=xa_shells.astype(np.float64),
            total_shells=total_shells.astype(np.float64),
            shell_indices_half=shell_indices_half_np.astype(np.int32),
            shell_indices_noise=shell_indices_noise_np.astype(np.int32),
        )
        if requested_iterations is None:
            pending_targets.remove(original_idx)

    return pending_targets


def _child_ordinals_from_parent_ids(parent_ids: np.ndarray) -> np.ndarray:
    """Return RELION-style child ordinal within each repeated parent id."""

    parent_ids = np.asarray(parent_ids, dtype=np.int32).reshape(-1)
    child_ordinals = np.zeros(parent_ids.shape[0], dtype=np.int32)
    seen: dict[int, int] = {}
    for idx, parent_id in enumerate(parent_ids.tolist()):
        count = seen.get(int(parent_id), 0)
        child_ordinals[idx] = count
        seen[int(parent_id)] = count + 1
    return child_ordinals


def _infer_grouped_child_layout(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Infer parent/child ids for a parent-major oversampled grid.

    The adaptive RELION translation grid is generated parent-major: every
    coarse translation contributes a fixed child-offset pattern. Inferring the
    group size here keeps this helper debug-only instead of adding persistent
    layout fields to the hot path.
    """

    values = np.asarray(values, dtype=np.float64)
    n_values = int(values.shape[0])
    if n_values <= 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty, 1

    candidates = [4**order for order in range(1, 6) if n_values % (4**order) == 0]
    candidates.sort(reverse=True)
    for child_count in candidates:
        grouped = values.reshape(n_values // child_count, child_count, values.shape[-1])
        offsets = grouped - np.mean(grouped, axis=1, keepdims=True)
        if np.allclose(offsets, offsets[0:1], rtol=1e-5, atol=1e-5):
            parent = np.repeat(np.arange(n_values // child_count, dtype=np.int32), child_count)
            child = np.tile(np.arange(child_count, dtype=np.int32), n_values // child_count)
            return parent, child, int(child_count)

    parent = np.arange(n_values, dtype=np.int32)
    child = np.zeros(n_values, dtype=np.int32)
    return parent, child, 1


def maybe_write_debug_score_dump(
    *,
    experiment_dataset,
    local_layout,
    bucket,
    image_pre_shifts,
    scores,
    probs,
    log_Z,
    best_log_score,
    max_posterior,
    reconstruction_sample_mask,
    reconstruction_rotation_mask,
    n_significant_samples,
    current_size,
    debug_iteration,
    shifted_score_split=None,
    shifted_recon_split=None,
    ctf2_over_nv_score=None,
    ctf2_over_nv_recon=None,
    proj_weighted=None,
    proj_for_noise=None,
    proj_abs2_weighted=None,
    dump_dir: Path | None,
    pending_targets: set[int],
    requested_current_sizes: set[int] | None = None,
    requested_iterations: set[int] | None = None,
):
    """Dump one-image local score tensors for the requested original ids."""

    if dump_dir is None or not pending_targets:
        return pending_targets
    if requested_current_sizes is not None and int(current_size or -1) not in requested_current_sizes:
        return pending_targets
    if requested_iterations is not None and int(debug_iteration or -1) not in requested_iterations:
        return pending_targets

    original_image_indices = np.asarray(
        experiment_dataset.original_image_indices_from_local(bucket.image_indices),
        dtype=np.int64,
    )
    target_rows = [
        row
        for row, original_idx in enumerate(original_image_indices.tolist())
        if int(original_idx) in pending_targets
    ]
    if not target_rows:
        return pending_targets

    scores_np = np.asarray(scores, dtype=np.float32)
    probs_np = np.asarray(probs, dtype=np.float32)
    log_Z_np = np.asarray(log_Z, dtype=np.float32)
    best_log_score_np = np.asarray(best_log_score, dtype=np.float32)
    max_posterior_np = np.asarray(max_posterior, dtype=np.float32)
    reconstruction_sample_mask_np = np.asarray(reconstruction_sample_mask, dtype=bool)
    reconstruction_rotation_mask_np = np.asarray(reconstruction_rotation_mask, dtype=bool)
    n_significant_samples_np = np.asarray(n_significant_samples, dtype=np.int32)
    dump_operands = os.environ.get("RECOVAR_LOCAL_SCORE_DUMP_OPERANDS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    shifted_score_np = np.asarray(shifted_score_split) if dump_operands and shifted_score_split is not None else None
    shifted_recon_np = np.asarray(shifted_recon_split) if dump_operands and shifted_recon_split is not None else None
    ctf2_over_nv_np = np.asarray(ctf2_over_nv_score) if dump_operands and ctf2_over_nv_score is not None else None
    ctf2_over_nv_recon_np = np.asarray(ctf2_over_nv_recon) if dump_operands and ctf2_over_nv_recon is not None else None
    proj_weighted_np = np.asarray(proj_weighted) if dump_operands and proj_weighted is not None else None
    proj_for_noise_np = np.asarray(proj_for_noise) if dump_operands and proj_for_noise is not None else None
    proj_abs2_weighted_np = np.asarray(proj_abs2_weighted) if dump_operands and proj_abs2_weighted is not None else None

    for row in target_rows:
        original_idx = int(original_image_indices[row])
        local_idx = int(bucket.image_indices[row])
        actual_count = int(bucket.actual_rotation_counts[row])
        local_rotation_ids = np.asarray(bucket.local_rotation_ids[row, :actual_count], dtype=np.int32)
        local_rotation_parent_ids = (
            np.asarray(bucket.local_rotation_posterior_ids[row, :actual_count], dtype=np.int32)
            if bucket.local_rotation_posterior_ids is not None
            else local_rotation_ids
        )
        local_rotation_child_indices = _child_ordinals_from_parent_ids(local_rotation_parent_ids)
        local_rotation_matrices = np.asarray(bucket.local_rotations[row, :actual_count], dtype=np.float32)
        local_rotation_eulers = np.asarray(
            utils.R_to_relion(local_rotation_matrices, degrees=True),
            dtype=np.float32,
        )
        rotation_mask = np.asarray(bucket.local_rotation_mask[row, :actual_count], dtype=bool)
        rotation_log_prior = np.asarray(bucket.local_rotation_log_prior[row, :actual_count], dtype=np.float32)
        translation_log_prior = np.asarray(bucket.translation_log_prior[row], dtype=np.float32)
        total_scores = np.asarray(scores_np[row, :actual_count, :], dtype=np.float32)
        raw_scores = total_scores - rotation_log_prior[:, None] - translation_log_prior[None, :]
        raw_scores = np.where(rotation_mask[:, None], raw_scores, -np.inf)
        posterior = np.asarray(probs_np[row, :actual_count, :], dtype=np.float32)
        n_trans = int(translation_log_prior.shape[0])
        translation_indices = np.arange(n_trans, dtype=np.int32)
        translation_parent_indices, translation_child_indices, n_trans_over = _infer_grouped_child_layout(
            np.asarray(local_layout.translation_grid, dtype=np.float32)
        )
        n_parent_trans = int(np.max(translation_parent_indices) + 1) if translation_parent_indices.size else n_trans
        n_rot_over = int(np.max(local_rotation_child_indices) + 1) if local_rotation_child_indices.size else 1
        n_hidden_over = int(n_rot_over * n_trans_over)
        candidate_hidden_over_indices = (
            (
                local_rotation_parent_ids[:, None].astype(np.int64) * np.int64(n_parent_trans)
                + translation_parent_indices[None, :].astype(np.int64)
            )
            * np.int64(n_hidden_over)
            + local_rotation_child_indices[:, None].astype(np.int64) * np.int64(n_trans_over)
            + translation_child_indices[None, :].astype(np.int64)
        )
        best_score_flat = int(np.argmax(total_scores))
        best_score_rotation_index, best_score_translation_index = np.unravel_index(
            best_score_flat,
            total_scores.shape,
        )
        best_posterior_flat = int(np.argmax(posterior))
        best_posterior_rotation_index, best_posterior_translation_index = np.unravel_index(
            best_posterior_flat,
            posterior.shape,
        )
        reconstruction_sample_mask_row = np.asarray(
            reconstruction_sample_mask_np[row, :actual_count, :],
            dtype=bool,
        )
        reconstruction_rotation_mask_row = np.asarray(
            reconstruction_rotation_mask_np[row, :actual_count],
            dtype=bool,
        )

        iteration_label = int(debug_iteration or -1)
        dump_path = dump_dir / f"local_score_it{iteration_label:03d}_image_{original_idx}.npz"
        payload = {
            "selected_global_image_indices": np.array([original_idx], dtype=np.int64),
            "selected_local_image_indices": np.array([local_idx], dtype=np.int64),
            "pass2_scores_raw": raw_scores[None, :, :],
            "pass2_scores_total": total_scores[None, :, :],
            "rotation_log_prior": rotation_log_prior[None, :],
            "translation_log_prior": translation_log_prior[None, :],
            "rotation_candidate_mask": rotation_mask[None, :],
            "local_rotation_indices": local_rotation_ids,
            "local_rotation_parent_indices": local_rotation_parent_ids,
            "local_rotation_child_indices": local_rotation_child_indices,
            "local_rotation_pixel_indices": (local_rotation_ids % int(local_layout.n_pixels)).astype(np.int64),
            "local_rotation_psi_indices": (local_rotation_ids // int(local_layout.n_pixels)).astype(np.int64),
            "local_rotation_eulers": local_rotation_eulers,
            "local_rotation_matrices": local_rotation_matrices,
            "translations": np.asarray(local_layout.translation_grid, dtype=np.float32),
            "translation_parent_indices": translation_parent_indices,
            "translation_child_indices": translation_child_indices,
            "n_translation_children": np.array([int(n_trans_over)], dtype=np.int32),
            "n_rotation_children": np.array([int(n_rot_over)], dtype=np.int32),
            "n_hidden_over": np.array([int(n_hidden_over)], dtype=np.int32),
            "candidate_pose_rotation_indices": np.repeat(local_rotation_ids[:, None], n_trans, axis=1),
            "candidate_pose_parent_rotation_indices": np.repeat(
                local_rotation_parent_ids[:, None],
                n_trans,
                axis=1,
            ),
            "candidate_pose_rotation_child_indices": np.repeat(
                local_rotation_child_indices[:, None],
                n_trans,
                axis=1,
            ),
            "candidate_pose_translation_indices": np.broadcast_to(
                translation_indices[None, :],
                (actual_count, n_trans),
            ),
            "candidate_pose_parent_translation_indices": np.broadcast_to(
                translation_parent_indices[None, :],
                (actual_count, n_trans),
            ),
            "candidate_pose_translation_child_indices": np.broadcast_to(
                translation_child_indices[None, :],
                (actual_count, n_trans),
            ),
            "candidate_pose_hidden_over_indices": candidate_hidden_over_indices.astype(np.int64, copy=False),
            "image_pre_shift": (
                np.asarray(image_pre_shifts[local_idx], dtype=np.float32)
                if image_pre_shifts is not None
                else np.array([], dtype=np.float32)
            ),
            "posterior": posterior[None, :, :],
            "reconstruction_sample_mask": reconstruction_sample_mask_row[None, :, :],
            "reconstruction_rotation_mask": reconstruction_rotation_mask_row[None, :],
            "n_significant_samples": np.array([int(n_significant_samples_np[row])], dtype=np.int32),
            "max_posterior": np.array([float(max_posterior_np[row])], dtype=np.float32),
            "log_Z": np.array([float(log_Z_np[row])], dtype=np.float32),
            "best_score": np.array([float(best_log_score_np[row])], dtype=np.float32),
            "best_score_rotation_local_index": np.array([int(best_score_rotation_index)], dtype=np.int32),
            "best_score_translation_index": np.array([int(best_score_translation_index)], dtype=np.int32),
            "best_score_rotation_global_id": np.array(
                [int(local_rotation_ids[int(best_score_rotation_index)])],
                dtype=np.int32,
            ),
            "best_score_translation": np.asarray(
                local_layout.translation_grid[
                    int(best_score_translation_index) : int(best_score_translation_index) + 1
                ],
                dtype=np.float32,
            ),
            "best_posterior_rotation_local_index": np.array([int(best_posterior_rotation_index)], dtype=np.int32),
            "best_posterior_translation_index": np.array([int(best_posterior_translation_index)], dtype=np.int32),
            "best_posterior_rotation_global_id": np.array(
                [int(local_rotation_ids[int(best_posterior_rotation_index)])],
                dtype=np.int32,
            ),
            "best_posterior_translation": np.asarray(
                local_layout.translation_grid[
                    int(best_posterior_translation_index) : int(best_posterior_translation_index) + 1
                ],
                dtype=np.float32,
            ),
            "current_size": np.array([int(current_size) if current_size is not None else -1], dtype=np.int32),
            "debug_iteration": np.array([iteration_label], dtype=np.int32),
            "n_rot": np.array([actual_count], dtype=np.int32),
            "n_trans": np.array([n_trans], dtype=np.int32),
            "grid_n_pixels": np.array([int(local_layout.n_pixels)], dtype=np.int32),
            "grid_n_psi": np.array([int(local_layout.n_psi)], dtype=np.int32),
        }
        if dump_operands:
            if shifted_score_np is not None:
                payload["debug_shifted_score"] = np.asarray(shifted_score_np[row], dtype=np.complex64)
            if shifted_recon_np is not None:
                payload["debug_shifted_recon"] = np.asarray(shifted_recon_np[row], dtype=np.complex64)
            if ctf2_over_nv_np is not None:
                payload["debug_ctf2_over_nv"] = np.asarray(ctf2_over_nv_np[row], dtype=np.float32)
            if ctf2_over_nv_recon_np is not None:
                payload["debug_ctf2_over_nv_recon"] = np.asarray(ctf2_over_nv_recon_np[row], dtype=np.float32)
            if proj_weighted_np is not None:
                payload["debug_proj_weighted"] = np.asarray(
                    proj_weighted_np[row, :actual_count, :],
                    dtype=np.complex64,
                )
            if proj_for_noise_np is not None:
                payload["debug_proj_for_recon"] = np.asarray(
                    proj_for_noise_np[row, :actual_count, :],
                    dtype=np.complex64,
                )
            if proj_abs2_weighted_np is not None:
                payload["debug_proj_abs2_weighted"] = np.asarray(
                    proj_abs2_weighted_np[row, :actual_count, :],
                    dtype=np.float32,
                )
        np.savez_compressed(dump_path, **payload)
        if requested_iterations is None:
            pending_targets.remove(original_idx)

    return pending_targets
