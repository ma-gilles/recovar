"""Top-p pose diagnostic utilities for PPCA refinement."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.ppca_refinement.config import PoseSelectionConfig


@dataclass(frozen=True)
class TopPoseSelection:
    """Padded top-p pose diagnostics."""

    rotation_idx: np.ndarray
    translation_idx: np.ndarray
    log_score: np.ndarray
    posterior: np.ndarray


def pack_pose_ids(rotation_ids, translation_ids, n_translations: int) -> np.ndarray:
    """Pack poses in the K-class convention ``rotation_id * n_trans + trans``."""

    return np.asarray(rotation_ids, dtype=np.int64) * int(n_translations) + np.asarray(
        translation_ids,
        dtype=np.int64,
    )


def top_pose_candidate_count(config: PoseSelectionConfig, n_candidates: int) -> int:
    """Number of raw candidates to retain before CPU-side distinct filtering."""

    n_candidates = int(n_candidates)
    if n_candidates <= 0:
        return 1
    top_p = int(config.top_p_poses)
    if top_p == 1:
        return 1
    return int(min(n_candidates, max(top_p, top_p * int(config.candidate_pool_factor), int(config.min_candidate_pool))))


def top_p_from_score_block(score, *, rotation_offset: int = 0, candidate_count: int = 1):
    """Return raw top candidates from one dense score block.

    ``score`` is ``(B, T, R)`` and the flattened axis is ``T * R`` with
    rotation as the fastest index.
    """

    score = jnp.asarray(score)
    B, T, R = score.shape
    k = max(1, min(int(candidate_count), int(T * R)))
    top_scores, top_flat = jax.lax.top_k(score.reshape(B, T * R), k)
    top_rot = (top_flat % int(R)).astype(jnp.int32) + jnp.asarray(int(rotation_offset), dtype=jnp.int32)
    top_trans = (top_flat // int(R)).astype(jnp.int32)
    return top_scores.astype(jnp.float32), top_rot, top_trans


def pad_top_pose_arrays(
    scores: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    top_p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad or trim per-image top-p arrays to exactly ``top_p`` columns."""

    scores = np.asarray(scores, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.int32)
    translations = np.asarray(translations, dtype=np.int32)
    top_p = int(top_p)
    if scores.ndim == 1:
        scores = scores[:, None]
    if rotations.ndim == 1:
        rotations = rotations[:, None]
    if translations.ndim == 1:
        translations = translations[:, None]
    n_images = int(scores.shape[0])
    out_scores = np.full((n_images, top_p), -np.inf, dtype=np.float32)
    out_rot = np.full((n_images, top_p), -1, dtype=np.int32)
    out_trans = np.full((n_images, top_p), -1, dtype=np.int32)
    width = min(top_p, int(scores.shape[1]))
    if width:
        out_scores[:, :width] = scores[:, :width]
        out_rot[:, :width] = rotations[:, :width]
        out_trans[:, :width] = translations[:, :width]
    return out_scores, out_rot, out_trans


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    cos_angle = (float(np.trace(rel)) - 1.0) * 0.5
    return float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _is_pose_distinct(
    rot_id: int,
    trans_id: int,
    accepted: list[tuple[int, int]],
    *,
    rotations: np.ndarray | None,
    translations: np.ndarray | None,
    min_angle_deg: float,
    min_translation_px: float,
) -> bool:
    for accepted_rot, accepted_trans in accepted:
        if rotations is None or rot_id < 0 or accepted_rot < 0:
            rotation_far = int(rot_id) != int(accepted_rot)
        else:
            rotation_far = _rotation_angle_deg(rotations[int(rot_id)], rotations[int(accepted_rot)]) >= min_angle_deg
        if translations is None or trans_id < 0 or accepted_trans < 0:
            translation_far = int(trans_id) != int(accepted_trans)
        else:
            translation_far = (
                float(np.linalg.norm(translations[int(trans_id)] - translations[int(accepted_trans)]))
                >= min_translation_px
            )
        if not (rotation_far or translation_far):
            return False
    return True


def select_distinct_top_poses(
    candidate_scores,
    candidate_rotations,
    candidate_translations,
    *,
    logZ=None,
    rotations: np.ndarray | None = None,
    candidate_rotation_matrices: np.ndarray | None = None,
    translations: np.ndarray | None = None,
    config: PoseSelectionConfig | None = None,
) -> TopPoseSelection:
    """Sort, distinct-filter, and pad raw top pose candidates per image."""

    config = config if config is not None else PoseSelectionConfig()
    top_p = int(config.top_p_poses)
    scores = np.asarray(candidate_scores, dtype=np.float32)
    rot_ids = np.asarray(candidate_rotations, dtype=np.int32)
    trans_ids = np.asarray(candidate_translations, dtype=np.int32)
    if scores.ndim == 1:
        scores = scores[:, None]
    if rot_ids.ndim == 1:
        rot_ids = rot_ids[:, None]
    if trans_ids.ndim == 1:
        trans_ids = trans_ids[:, None]
    if scores.shape != rot_ids.shape or scores.shape != trans_ids.shape:
        raise ValueError(f"candidate shapes differ: {scores.shape}, {rot_ids.shape}, {trans_ids.shape}")
    candidate_mats = None
    if candidate_rotation_matrices is not None:
        candidate_mats = np.asarray(candidate_rotation_matrices, dtype=np.float32)
        if candidate_mats.shape[:2] != scores.shape or candidate_mats.shape[-2:] != (3, 3):
            raise ValueError(
                "candidate_rotation_matrices must have shape "
                f"{scores.shape} + (3, 3), got {candidate_mats.shape}"
            )

    n_images = int(scores.shape[0])
    out_scores = np.full((n_images, top_p), -np.inf, dtype=np.float32)
    out_rot = np.full((n_images, top_p), -1, dtype=np.int32)
    out_trans = np.full((n_images, top_p), -1, dtype=np.int32)
    min_angle = max(0.0, float(config.top_pose_min_angle_deg))
    min_trans = max(0.0, float(config.top_pose_min_translation_px))
    max_gap = float(config.top_pose_max_log_score_gap)

    for image_idx in range(n_images):
        row_scores = scores[image_idx]
        order = np.lexsort(
            (
                trans_ids[image_idx].astype(np.int64),
                rot_ids[image_idx].astype(np.int64),
                -row_scores.astype(np.float64),
            )
        )
        finite = order[np.isfinite(row_scores[order])]
        if finite.size == 0:
            continue
        best_score = float(row_scores[finite[0]])
        accepted: list[tuple[int, int]] = []
        accepted_mats: list[np.ndarray] = []
        out_col = 0
        for candidate_idx in finite:
            score = float(row_scores[candidate_idx])
            if score < best_score - max_gap:
                break
            rot_id = int(rot_ids[image_idx, candidate_idx])
            trans_id = int(trans_ids[image_idx, candidate_idx])
            if out_col:
                if candidate_mats is None:
                    distinct = _is_pose_distinct(
                        rot_id,
                        trans_id,
                        accepted,
                        rotations=rotations,
                        translations=translations,
                        min_angle_deg=min_angle,
                        min_translation_px=min_trans,
                    )
                else:
                    candidate_mat = candidate_mats[image_idx, int(candidate_idx)]
                    distinct = True
                    for accepted_idx, (_accepted_rot, accepted_trans) in enumerate(accepted):
                        rotation_far = _rotation_angle_deg(candidate_mat, accepted_mats[accepted_idx]) >= min_angle
                        if translations is None or trans_id < 0 or accepted_trans < 0:
                            translation_far = int(trans_id) != int(accepted_trans)
                        else:
                            translation_far = (
                                float(np.linalg.norm(translations[int(trans_id)] - translations[int(accepted_trans)]))
                                >= min_trans
                            )
                        if not (rotation_far or translation_far):
                            distinct = False
                            break
                if not distinct:
                    continue
            out_scores[image_idx, out_col] = score
            out_rot[image_idx, out_col] = rot_id
            out_trans[image_idx, out_col] = trans_id
            accepted.append((rot_id, trans_id))
            if candidate_mats is not None:
                accepted_mats.append(candidate_mats[image_idx, int(candidate_idx)])
            out_col += 1
            if out_col == top_p:
                break

    if logZ is None:
        posterior = np.zeros_like(out_scores, dtype=np.float32)
    else:
        logZ_np = np.asarray(logZ, dtype=np.float64).reshape(n_images)
        with np.errstate(over="ignore", invalid="ignore"):
            posterior = np.exp(out_scores.astype(np.float64) - logZ_np[:, None]).astype(np.float32)
        posterior = np.where(np.isfinite(out_scores), posterior, 0.0).astype(np.float32)
    return TopPoseSelection(
        rotation_idx=out_rot,
        translation_idx=out_trans,
        log_score=out_scores,
        posterior=posterior,
    )


def merge_top_p_pose_scores(
    block_top_scores,
    block_top_rotations,
    block_top_translations,
    logZ,
    *,
    rotations: np.ndarray | None = None,
    translations: np.ndarray | None = None,
    config: PoseSelectionConfig | None = None,
) -> TopPoseSelection:
    """Merge raw top candidates from multiple rotation blocks."""

    config = config if config is not None else PoseSelectionConfig()
    n_images = int(np.asarray(logZ).shape[0])
    if not block_top_scores:
        empty = np.empty((n_images, 0), dtype=np.float32)
        empty_i = np.empty((n_images, 0), dtype=np.int32)
        return select_distinct_top_poses(
            empty,
            empty_i,
            empty_i,
            logZ=logZ,
            rotations=rotations,
            translations=translations,
            config=config,
        )
    scores = np.concatenate([np.asarray(v, dtype=np.float32) for v in block_top_scores], axis=1)
    rotations_raw = np.concatenate([np.asarray(v, dtype=np.int32) for v in block_top_rotations], axis=1)
    translations_raw = np.concatenate([np.asarray(v, dtype=np.int32) for v in block_top_translations], axis=1)
    return select_distinct_top_poses(
        scores,
        rotations_raw,
        translations_raw,
        logZ=logZ,
        rotations=rotations,
        translations=translations,
        config=config,
    )
