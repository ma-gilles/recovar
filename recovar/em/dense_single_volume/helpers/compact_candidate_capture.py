"""Diagnostic-only compact capture of the already-computed K=1 pass-2 state.

This module never selects a scoring, projection, normalization, or M-step
implementation.  Its caller invokes it only after production ``scores`` and
``probs`` exist.  The full-run capture-inertness gate remains mandatory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SCHEMA = "recovar-k1-production-candidate-bucket-v1"
CAPTURE_DIR_ENV = "RECOVAR_COMPACT_CANDIDATE_CAPTURE_DIR"
CAPTURE_ITERATION_ENV = "RECOVAR_COMPACT_CANDIDATE_CAPTURE_ITERATION"
MAX_PARTICLES_PER_RAW_SHARD = 256
MAX_CANDIDATES_PER_RAW_SHARD = 1_000_000
_capture_counter = 0


class CompactCaptureError(RuntimeError):
    pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise CompactCaptureError(f"refusing to overwrite compact capture {path}")
    with partial.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(partial, allow_pickle=False) as check:
        for key in check.files:
            np.asarray(check[key])
    os.replace(partial, path)
    _fsync_directory(path.parent)


def _capture_requested(iteration: int) -> Path | None:
    raw = os.environ.get(CAPTURE_DIR_ENV, "").strip()
    if not raw:
        return None
    target = os.environ.get(CAPTURE_ITERATION_ENV, "").strip()
    if target and int(target) != int(iteration):
        return None
    if int(iteration) < 0:
        raise CompactCaptureError("compact capture requires an explicit EM iteration context")
    return Path(raw)


def maybe_capture_k1_production_bucket(
    *,
    iteration,
    half,
    image_indices,
    original_indices,
    per_image_inputs,
    current_size,
    fine_translations,
    fine_translation_parent,
    scores,
    probs,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    reconstruction_mask,
    log_z,
    best_log_score,
    best_argmax,
    max_posterior,
) -> int:
    """Copy the exact production candidate arrays into one compact raw shard.

    Returns the number of captured particles.  When the env gate is disabled,
    returns before converting or synchronizing any JAX value.
    """

    capture_dir = _capture_requested(int(iteration))
    if capture_dir is None:
        return 0
    if int(half) not in (1, 2):
        raise CompactCaptureError("compact capture requires half 1 or 2")
    if reconstruction_mask is None:
        raise CompactCaptureError("targeted compact capture requires the production significant mask")

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = np.asarray(original_indices, dtype=np.int64)
    score = np.asarray(scores)
    posterior = np.asarray(probs)
    mask = np.asarray(candidate_mask, dtype=bool)
    significant = np.asarray(reconstruction_mask, dtype=bool)
    rot_prior = np.asarray(rotation_log_prior)
    trans_prior = np.asarray(translation_log_prior)
    log_z_np = np.asarray(log_z)
    best_np = np.asarray(best_log_score)
    argmax_np = np.asarray(best_argmax, dtype=np.int64)
    pmax_np = np.asarray(max_posterior)
    translations = np.asarray(fine_translations, dtype=np.float32)
    translation_parent = np.asarray(fine_translation_parent, dtype=np.int32)

    batch = local_indices.size
    if original_indices.shape != (batch,):
        raise CompactCaptureError("original identity topology mismatch")
    if batch == 0:
        raise CompactCaptureError("compact capture received an empty production bucket")
    if score.shape != posterior.shape or score.shape != mask.shape or significant.shape != mask.shape:
        raise CompactCaptureError("score/posterior/mask topology mismatch")
    if score.ndim != 3 or score.shape[0] != batch:
        raise CompactCaptureError("K=1 compact capture expects (batch,rotation,translation) arrays")
    if rot_prior.shape != score.shape[:2] or trans_prior.shape != (batch, score.shape[2]):
        raise CompactCaptureError("prior topology mismatch")
    if translations.shape != (score.shape[2], 2) or translation_parent.shape != (score.shape[2],):
        raise CompactCaptureError("translation topology mismatch")
    if not np.isfinite(translations).all():
        raise CompactCaptureError("non-finite fine translation geometry")
    if np.any(significant & ~mask) or np.any(posterior[~mask] != 0):
        raise CompactCaptureError("candidate/significant/posterior mask closure failed")
    for name, array in (
        ("active score", score[mask]),
        ("active posterior", posterior[mask]),
        ("active rotation prior", np.broadcast_to(rot_prior[:, :, None], mask.shape)[mask]),
        ("active translation prior", np.broadcast_to(trans_prior[:, None, :], mask.shape)[mask]),
        ("log_z", log_z_np),
        ("best score", best_np),
        ("pmax", pmax_np),
    ):
        if not np.isfinite(array).all():
            raise CompactCaptureError(f"non-finite {name}")
    if np.any(posterior < 0):
        raise CompactCaptureError("negative production posterior")

    candidate_offset = np.zeros(batch + 1, dtype=np.int64)
    rotation_offset = np.zeros(batch + 1, dtype=np.int64)
    candidate_rot = []
    candidate_trans = []
    candidate_score = []
    candidate_posterior = []
    candidate_significant = []
    candidate_rot_prior = []
    candidate_trans_prior = []
    rotation_matrix = []
    rotation_global_index = []
    rotation_parent_local = []
    rotation_parent_global = []
    winner_candidate_index = np.empty(batch, dtype=np.int32)
    winner_pose = np.empty((batch, 3, 3), dtype=np.float32)
    winner_translation = np.empty((batch, 2), dtype=np.float32)
    posterior_sum_float32_order = np.empty(batch, dtype=np.float32)
    posterior_sum_float64_exact = np.empty(batch, dtype=np.float64)
    posterior_sum_float32_bound = np.empty(batch, dtype=np.float64)
    significant_count = np.empty(batch, dtype=np.int32)
    significant_threshold = np.empty(batch, dtype=posterior.dtype)

    for row, image_idx in enumerate(local_indices.tolist()):
        rotations = np.asarray(per_image_inputs["oversampled_rots"][image_idx], dtype=np.float32)
        global_rot = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        parent_local = np.asarray(per_image_inputs["parent_map"][image_idx], dtype=np.int32)
        unique_rot = np.asarray(per_image_inputs["unique_rot"][image_idx], dtype=np.int32)
        n_rot = rotations.shape[0]
        if rotations.shape != (n_rot, 3, 3) or global_rot.shape != (n_rot,) or parent_local.shape != (n_rot,):
            raise CompactCaptureError("rotation topology mismatch")
        if not np.isfinite(rotations).all():
            raise CompactCaptureError("non-finite production rotation geometry")
        gram_error = np.max(
            np.abs(rotations @ np.swapaxes(rotations, 1, 2) - np.eye(3, dtype=np.float32)),
            axis=(1, 2),
        )
        determinant_error = np.abs(np.linalg.det(rotations) - np.float32(1.0))
        if np.any(gram_error > 5e-4) or np.any(determinant_error > 5e-4):
            raise CompactCaptureError("production rotation geometry is not a proper orthogonal matrix")
        if np.any((parent_local < 0) | (parent_local >= unique_rot.size)):
            raise CompactCaptureError("rotation parent index is out of range")
        active_rot, active_trans = np.nonzero(mask[row, :n_rot])
        count = active_rot.size
        if count == 0:
            raise CompactCaptureError("particle has no active pass-2 candidate")
        dense_winner = int(argmax_np[row])
        winner_rot, winner_trans = divmod(dense_winner, score.shape[2])
        winner_hits = np.flatnonzero((active_rot == winner_rot) & (active_trans == winner_trans))
        if winner_hits.size != 1:
            raise CompactCaptureError("production winner is absent or ambiguous in candidate support")
        winner_candidate_index[row] = int(winner_hits[0])
        winner_pose[row] = rotations[winner_rot]
        winner_translation[row] = translations[winner_trans]
        selected = (active_rot, active_trans)
        candidate_rot.append(np.asarray(active_rot, dtype=np.int32))
        candidate_trans.append(np.asarray(active_trans, dtype=np.int32))
        candidate_score.append(np.asarray(score[row][selected]))
        candidate_posterior.append(np.asarray(posterior[row][selected]))
        candidate_significant.append(np.asarray(significant[row][selected], dtype=np.uint8))
        candidate_rot_prior.append(np.asarray(rot_prior[row, active_rot]))
        candidate_trans_prior.append(np.asarray(trans_prior[row, active_trans]))
        rotation_matrix.append(rotations)
        rotation_global_index.append(global_rot)
        rotation_parent_local.append(parent_local)
        rotation_parent_global.append(unique_rot[parent_local])
        candidate_offset[row + 1] = candidate_offset[row] + count
        rotation_offset[row + 1] = rotation_offset[row] + n_rot
        posterior_sum_float32_order[row] = np.sum(
            np.asarray(posterior[row][selected], dtype=np.float32), dtype=np.float32
        )
        posterior_sum_float64_exact[row] = np.sum(
            np.asarray(posterior[row][selected], dtype=np.float64), dtype=np.float64
        )
        unit_roundoff = np.finfo(np.float32).eps / 2.0
        gamma_n = count * unit_roundoff / (1.0 - count * unit_roundoff)
        posterior_sum_float32_bound[row] = (
            gamma_n * np.sum(np.abs(posterior[row][selected]), dtype=np.float64)
            + 8.0 * unit_roundoff * max(1.0, abs(posterior_sum_float64_exact[row]))
        )
        if (
            abs(float(posterior_sum_float32_order[row]) - posterior_sum_float64_exact[row])
            > posterior_sum_float32_bound[row]
        ):
            raise CompactCaptureError("posterior native-order float32 sum exceeds its rounding bound")
        significant_count[row] = int(np.count_nonzero(significant[row][selected]))
        if significant_count[row] == 0:
            raise CompactCaptureError("particle has no production-significant candidate")
        significant_threshold[row] = np.min(posterior[row][selected][significant[row][selected]])

    global _capture_counter
    call_index = _capture_counter
    _capture_counter += 1
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    metadata = {
        "schema": SCHEMA,
        "score_semantics": "higher-is-better combined log weight; all priors included",
        "raw_score_includes_prior": True,
        "raw_log_z_convention": "logsumexp(combined production score over candidate_mask)",
        "capture_path": "post-production-normalization tap; does not select dump_this_bucket",
    }
    flat_candidate_arrays = {
        "candidate_local_rotation": np.concatenate(candidate_rot),
        "candidate_translation": np.concatenate(candidate_trans),
        "raw_combined_score": np.concatenate(candidate_score),
        "posterior": np.concatenate(candidate_posterior),
        "significant": np.concatenate(candidate_significant),
        "rotation_log_prior": np.concatenate(candidate_rot_prior),
        "translation_log_prior": np.concatenate(candidate_trans_prior),
    }
    flat_rotation_arrays = {
        "rotation_matrix": np.concatenate(rotation_matrix),
        "rotation_global_index": np.concatenate(rotation_global_index),
        "rotation_parent_local": np.concatenate(rotation_parent_local),
        "rotation_parent_global": np.concatenate(rotation_parent_global),
    }
    shard_ranges = []
    start = 0
    while start < batch:
        stop = start
        candidates = 0
        while stop < batch and stop - start < MAX_PARTICLES_PER_RAW_SHARD:
            next_count = int(candidate_offset[stop + 1] - candidate_offset[stop])
            if next_count > MAX_CANDIDATES_PER_RAW_SHARD:
                raise CompactCaptureError("one particle exceeds the raw-shard candidate bound")
            if stop > start and candidates + next_count > MAX_CANDIDATES_PER_RAW_SHARD:
                break
            candidates += next_count
            stop += 1
        shard_ranges.append((start, stop))
        start = stop

    for shard_index, (row_start, row_stop) in enumerate(shard_ranges):
        candidate_start = int(candidate_offset[row_start])
        candidate_stop = int(candidate_offset[row_stop])
        rotation_start = int(rotation_offset[row_start])
        rotation_stop = int(rotation_offset[row_stop])
        shard_original = original_indices[row_start:row_stop]
        path = capture_dir / (
            f"raw_k1_it{int(iteration):03d}_h{int(half)}_rank{rank:03d}_"
            f"call{call_index:06d}_shard{shard_index:03d}_"
            f"p{int(shard_original.min()):05d}_{int(shard_original.max()) + 1:05d}.npz"
        )
        _atomic_npz(
            path,
            schema=np.asarray(SCHEMA),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            iteration=np.int32(iteration),
            half=np.int8(half),
            rank=np.int32(rank),
            call_index=np.int64(call_index),
            shard_index=np.int32(shard_index),
            current_size=np.int32(-1 if current_size is None else current_size),
            local_indices=local_indices[row_start:row_stop],
            original_indices=shard_original,
            candidate_offset=candidate_offset[row_start : row_stop + 1] - candidate_start,
            **{
                name: array[candidate_start:candidate_stop]
                for name, array in flat_candidate_arrays.items()
            },
            rotation_offset=rotation_offset[row_start : row_stop + 1] - rotation_start,
            **{
                name: array[rotation_start:rotation_stop]
                for name, array in flat_rotation_arrays.items()
            },
            fine_translations=translations,
            fine_translation_parent=translation_parent,
            score_center=best_np[row_start:row_stop],
            raw_log_z=log_z_np[row_start:row_stop],
            pmax=pmax_np[row_start:row_stop],
            posterior_sum_float32_order=posterior_sum_float32_order[row_start:row_stop],
            posterior_sum_float64_exact=posterior_sum_float64_exact[row_start:row_stop],
            posterior_sum_float32_bound=posterior_sum_float32_bound[row_start:row_stop],
            significant_count=significant_count[row_start:row_stop],
            significant_threshold=significant_threshold[row_start:row_stop],
            winner_candidate_index=winner_candidate_index[row_start:row_stop],
            winner_pose_matrix=winner_pose[row_start:row_stop],
            winner_translation=winner_translation[row_start:row_stop],
        )
    return int(batch)
