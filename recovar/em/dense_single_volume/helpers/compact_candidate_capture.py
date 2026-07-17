"""Diagnostic-only compact capture of the already-computed K=1 pass-2 state.

This module never selects a scoring, projection, normalization, or M-step
implementation.  Its caller invokes it only after production ``scores`` and
``probs`` exist.  The full-run capture-inertness gate remains mandatory.
"""

from __future__ import annotations

import hashlib
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise CompactCaptureError(f"refusing to overwrite compact capture {path}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    _fsync_directory(path.parent)


def _require_dtype(array: np.ndarray, dtype, name: str) -> None:
    if array.dtype != np.dtype(dtype):
        raise CompactCaptureError(f"{name} dtype must be {np.dtype(dtype)}, got {array.dtype}")


def validate_raw_capture_shard(path: Path) -> dict[str, object]:
    """Strictly validate one raw production shard and return its inventory."""

    path = Path(path)
    if path.name.endswith(".partial"):
        raise CompactCaptureError(f"partial raw shard is not readable: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema", "metadata_json", "iteration", "half", "rank", "call_index",
                "shard_index", "current_size", "local_indices", "original_indices",
                "candidate_offset", "candidate_local_rotation", "candidate_translation",
                "raw_combined_score", "posterior", "significant", "rotation_log_prior",
                "translation_log_prior", "rotation_offset", "rotation_matrix",
                "rotation_global_index", "rotation_parent_local", "rotation_parent_global",
                "fine_translations", "fine_translation_parent", "score_center", "raw_log_z",
                "pmax", "posterior_sum_float32_order", "posterior_sum_float64_exact",
                "posterior_sum_float32_bound", "significant_count", "significant_threshold",
                "winner_candidate_index", "winner_pose_matrix", "winner_translation",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise CompactCaptureError(f"raw shard is missing fields: {missing}")
            if str(np.asarray(data["schema"]).item()) != SCHEMA:
                raise CompactCaptureError("raw shard schema mismatch")
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            if metadata.get("schema") != SCHEMA:
                raise CompactCaptureError("raw shard metadata schema mismatch")
            arrays = {name: np.asarray(data[name]) for name in required - {"schema", "metadata_json"}}
    except CompactCaptureError:
        raise
    except Exception as exc:
        raise CompactCaptureError(f"invalid raw shard {path}: {exc}") from exc

    for name, dtype in (
        ("iteration", np.int32), ("half", np.int8), ("rank", np.int32),
        ("call_index", np.int64), ("shard_index", np.int32), ("current_size", np.int32),
        ("local_indices", np.int64), ("original_indices", np.int64),
        ("candidate_offset", np.int64), ("candidate_local_rotation", np.int32),
        ("candidate_translation", np.int32), ("significant", np.uint8),
        ("rotation_offset", np.int64), ("rotation_matrix", np.float32),
        ("rotation_global_index", np.int64), ("rotation_parent_local", np.int32),
        ("rotation_parent_global", np.int32), ("fine_translations", np.float32),
        ("fine_translation_parent", np.int32), ("posterior_sum_float32_order", np.float32),
        ("posterior_sum_float64_exact", np.float64),
        ("posterior_sum_float32_bound", np.float64), ("significant_count", np.int32),
        ("winner_candidate_index", np.int32), ("winner_pose_matrix", np.float32),
        ("winner_translation", np.float32),
    ):
        _require_dtype(arrays[name], dtype, name)
    for name in ("iteration", "half", "rank", "call_index", "shard_index", "current_size"):
        if arrays[name].shape != ():
            raise CompactCaptureError(f"{name} must be a scalar")
    if int(arrays["iteration"]) < 0 or int(arrays["half"]) not in (1, 2):
        raise CompactCaptureError("raw shard iteration/half context is invalid")
    for name in ("raw_combined_score", "posterior", "rotation_log_prior", "translation_log_prior"):
        if arrays[name].dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise CompactCaptureError(f"{name} must preserve a float32/float64 production dtype")

    local = arrays["local_indices"]
    original = arrays["original_indices"]
    particle_count = int(local.size)
    if not 0 < particle_count <= MAX_PARTICLES_PER_RAW_SHARD or original.shape != (particle_count,):
        raise CompactCaptureError("raw shard particle topology/bound is invalid")
    if np.unique(local).size != particle_count or np.unique(original).size != particle_count:
        raise CompactCaptureError("raw shard contains duplicate local/original identities")
    candidate_offset = arrays["candidate_offset"]
    rotation_offset = arrays["rotation_offset"]
    if (
        candidate_offset.shape != (particle_count + 1,)
        or rotation_offset.shape != (particle_count + 1,)
        or candidate_offset[0] != 0
        or rotation_offset[0] != 0
        or np.any(np.diff(candidate_offset) <= 0)
        or np.any(np.diff(rotation_offset) <= 0)
    ):
        raise CompactCaptureError("raw shard offsets are invalid")
    candidate_count = int(candidate_offset[-1])
    rotation_count = int(rotation_offset[-1])
    if candidate_count > MAX_CANDIDATES_PER_RAW_SHARD:
        raise CompactCaptureError("raw shard exceeds its candidate bound")
    for name in (
        "candidate_local_rotation", "candidate_translation", "raw_combined_score", "posterior",
        "significant", "rotation_log_prior", "translation_log_prior",
    ):
        if arrays[name].shape != (candidate_count,):
            raise CompactCaptureError(f"{name} does not close over candidate_offset")
    for name in (
        "rotation_global_index", "rotation_parent_local", "rotation_parent_global",
    ):
        if arrays[name].shape != (rotation_count,):
            raise CompactCaptureError(f"{name} does not close over rotation_offset")
    if arrays["rotation_matrix"].shape != (rotation_count, 3, 3):
        raise CompactCaptureError("rotation_matrix does not close over rotation_offset")
    n_trans = arrays["fine_translations"].shape[0]
    if (
        arrays["fine_translations"].shape != (n_trans, 2)
        or arrays["fine_translation_parent"].shape != (n_trans,)
        or n_trans == 0
    ):
        raise CompactCaptureError("fine translation topology is invalid")
    per_particle_shapes = {
        "score_center": (particle_count,), "raw_log_z": (particle_count,),
        "pmax": (particle_count,), "posterior_sum_float32_order": (particle_count,),
        "posterior_sum_float64_exact": (particle_count,),
        "posterior_sum_float32_bound": (particle_count,), "significant_count": (particle_count,),
        "significant_threshold": (particle_count,), "winner_candidate_index": (particle_count,),
        "winner_pose_matrix": (particle_count, 3, 3),
        "winner_translation": (particle_count, 2),
    }
    for name, shape in per_particle_shapes.items():
        if arrays[name].shape != shape:
            raise CompactCaptureError(f"{name} particle topology is invalid")

    finite_names = (
        "raw_combined_score", "posterior", "rotation_log_prior", "translation_log_prior",
        "rotation_matrix", "fine_translations", "score_center", "raw_log_z", "pmax",
        "posterior_sum_float32_order", "posterior_sum_float64_exact",
        "posterior_sum_float32_bound", "significant_threshold", "winner_pose_matrix",
        "winner_translation",
    )
    if any(not np.isfinite(arrays[name]).all() for name in finite_names):
        raise CompactCaptureError("raw shard contains non-finite active values or geometry")
    if np.any((arrays["posterior"] < 0) | (arrays["posterior"] > 1)):
        raise CompactCaptureError("raw shard posterior is outside [0,1]")
    if np.any((arrays["significant"] != 0) & (arrays["significant"] != 1)):
        raise CompactCaptureError("raw shard significant flag is not binary")

    rotations = arrays["rotation_matrix"]
    gram_error = np.max(
        np.abs(rotations @ np.swapaxes(rotations, 1, 2) - np.eye(3, dtype=np.float32)), axis=(1, 2)
    )
    determinant_error = np.abs(np.linalg.det(rotations) - np.float32(1.0))
    if np.any(gram_error > 5e-4) or np.any(determinant_error > 5e-4):
        raise CompactCaptureError("raw shard contains invalid rotation geometry")

    for row in range(particle_count):
        c0, c1 = (int(candidate_offset[row]), int(candidate_offset[row + 1]))
        r0, r1 = (int(rotation_offset[row]), int(rotation_offset[row + 1]))
        local_rot = arrays["candidate_local_rotation"][c0:c1]
        local_trans = arrays["candidate_translation"][c0:c1]
        posterior = arrays["posterior"][c0:c1]
        significant = arrays["significant"][c0:c1].astype(bool)
        if np.any((local_rot < 0) | (local_rot >= r1 - r0)):
            raise CompactCaptureError("raw candidate rotation index is out of range")
        if np.any((local_trans < 0) | (local_trans >= n_trans)):
            raise CompactCaptureError("raw candidate translation index is out of range")
        exact_sum = np.sum(posterior, dtype=np.float64)
        if exact_sum != arrays["posterior_sum_float64_exact"][row]:
            raise CompactCaptureError("raw shard exact posterior sum does not reproduce")
        if (
            abs(float(arrays["posterior_sum_float32_order"][row]) - exact_sum)
            > arrays["posterior_sum_float32_bound"][row]
        ):
            raise CompactCaptureError("raw shard float32 posterior sum exceeds its bound")
        if int(np.count_nonzero(significant)) != int(arrays["significant_count"][row]):
            raise CompactCaptureError("raw shard significant count does not reproduce")
        if not np.any(significant):
            raise CompactCaptureError("raw shard has no significant candidate")
        if arrays["significant_threshold"][row] != np.min(posterior[significant]):
            raise CompactCaptureError("raw shard significant threshold does not reproduce")
        winner = int(arrays["winner_candidate_index"][row])
        if not 0 <= winner < c1 - c0:
            raise CompactCaptureError("raw shard winner index is out of range")
        if arrays["pmax"][row] != posterior[winner]:
            raise CompactCaptureError("raw shard winner posterior does not reproduce Pmax")
        if arrays["score_center"][row] != arrays["raw_combined_score"][c0 + winner]:
            raise CompactCaptureError("raw shard winner score does not reproduce score_center")
        winner_rot = int(local_rot[winner])
        winner_trans = int(local_trans[winner])
        if not np.array_equal(arrays["winner_pose_matrix"][row], rotations[r0 + winner_rot]):
            raise CompactCaptureError("raw shard winner pose does not reproduce candidate geometry")
        if not np.array_equal(
            arrays["winner_translation"][row], arrays["fine_translations"][winner_trans]
        ):
            raise CompactCaptureError("raw shard winner translation does not reproduce candidate geometry")

    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "iteration": int(arrays["iteration"]),
        "half": int(arrays["half"]),
        "particle_count": particle_count,
        "candidate_count": candidate_count,
        "original_indices": original.copy(),
    }


def finalize_raw_capture_directory(
    capture_dir: Path,
    *,
    expected_original_indices_by_half,
    expected_iteration: int,
) -> dict[str, object]:
    """Seal a complete raw capture only after strict identity/readback checks."""

    capture_dir = Path(capture_dir)
    if list(capture_dir.glob("*.partial")):
        raise CompactCaptureError("raw capture directory contains partial files")
    shards = sorted(capture_dir.glob("raw_k1_*.npz"))
    if not shards:
        raise CompactCaptureError("raw capture directory has no shards")
    inventory = [validate_raw_capture_shard(path) for path in shards]
    if any(item["iteration"] != int(expected_iteration) for item in inventory):
        raise CompactCaptureError("raw capture iteration mismatch")
    if not isinstance(expected_original_indices_by_half, dict):
        raise CompactCaptureError("expected original identities must be supplied per half")
    try:
        expected_by_half = {
            int(half): np.asarray(values, dtype=np.int64)
            for half, values in expected_original_indices_by_half.items()
        }
    except Exception as exc:
        raise CompactCaptureError(f"invalid expected half identity mapping: {exc}") from exc
    if set(expected_by_half) != {1, 2}:
        raise CompactCaptureError("expected original identity mapping must contain exactly halves 1 and 2")
    for half, expected in expected_by_half.items():
        if expected.ndim != 1 or expected.size == 0 or np.unique(expected).size != expected.size:
            raise CompactCaptureError(f"expected half-{half} identities must be a nonempty unique vector")
    expected = np.concatenate([expected_by_half[1], expected_by_half[2]])
    if np.unique(expected).size != expected.size:
        raise CompactCaptureError("expected original identities overlap across halves")

    observed = np.concatenate([item["original_indices"] for item in inventory])
    if np.unique(observed).size != observed.size:
        raise CompactCaptureError("raw capture duplicates an original identity across shards")
    if not np.array_equal(np.sort(observed), np.sort(expected)):
        raise CompactCaptureError("raw capture identity set is incomplete or unexpected")
    for half in (1, 2):
        half_shards = [item["original_indices"] for item in inventory if int(item["half"]) == half]
        if not half_shards:
            raise CompactCaptureError(f"raw capture has no half-{half} shards")
        observed_half = np.concatenate(half_shards)
        if not np.array_equal(np.sort(observed_half), np.sort(expected_by_half[half])):
            raise CompactCaptureError(f"raw capture half-{half} identity set is incomplete or unexpected")
    manifest_lines = [f"{item['sha256']}  {Path(item['path']).name}" for item in inventory]
    manifest_payload = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    manifest_path = capture_dir / "RAW_CAPTURE.sha256"
    _atomic_bytes(manifest_path, manifest_payload)
    marker = {
        "schema": SCHEMA,
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256_file(manifest_path),
        "iteration": int(expected_iteration),
        "shard_count": len(inventory),
        "particle_count": int(observed.size),
        "candidate_count": int(sum(item["candidate_count"] for item in inventory)),
        "halves": sorted({int(item["half"]) for item in inventory}),
    }
    _atomic_bytes(
        capture_dir / "RAW_CAPTURE_COMPLETE.json",
        (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return marker


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
