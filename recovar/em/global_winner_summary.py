"""Bounded aggregate diagnostics for K-class first-iteration winners."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "k4_global_winner_summary_v1"
MAX_SUPPORTED_BYTES = 32 * 1024 * 1024
_PATH_ENV = "RECOVAR_GLOBAL_WINNER_SUMMARY_PATH"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{_PATH_ENV} requires {name}")
    return value


def _required_int_env(name: str) -> int:
    value = _required_env(name)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def _original_indices(experiment_dataset, n_images: int) -> np.ndarray:
    resolver = getattr(experiment_dataset, "original_image_indices_from_local", None)
    if not callable(resolver):
        raise RuntimeError(
            "RECOVAR global-winner summary requires experiment_dataset.original_image_indices_from_local()"
        )
    local_indices = np.arange(n_images, dtype=np.int64)
    try:
        indices = np.asarray(resolver(local_indices), dtype=np.int64)
    except Exception as exc:
        raise RuntimeError("RECOVAR global-winner summary could not resolve one original identity per image") from exc
    if indices.shape != (n_images,):
        raise RuntimeError(
            "RECOVAR global-winner summary requires one original identity per image; "
            f"got {indices.shape}, expected {(n_images,)}"
        )
    if np.unique(indices).size != n_images:
        raise RuntimeError("RECOVAR global-winner summary original identities are not unique")
    return indices


def maybe_dump_global_winner_summary(
    *,
    experiment_dataset,
    full_stats: dict,
    n_classes: int,
    n_rotations: int,
    n_translations: int,
    iteration: int | None,
) -> Path | None:
    """Write one fixed-size aggregate artifact when its explicit gate is enabled."""

    path_value = os.environ.get(_PATH_ENV, "").strip()
    if not path_value:
        return None

    target_iteration = _required_int_env("RECOVAR_GLOBAL_WINNER_SUMMARY_ITERATION")
    if iteration is None:
        raise RuntimeError(f"{_PATH_ENV} requires a concrete debug iteration")
    if int(iteration) != target_iteration:
        return None

    expected_particles = _required_int_env("RECOVAR_GLOBAL_WINNER_SUMMARY_EXPECTED_PARTICLES")
    expected_classes = _required_int_env("RECOVAR_GLOBAL_WINNER_SUMMARY_EXPECTED_CLASSES")
    max_bytes = _required_int_env("RECOVAR_GLOBAL_WINNER_SUMMARY_MAX_BYTES")
    if expected_particles <= 0 or expected_classes != 4 or n_classes != expected_classes:
        raise RuntimeError("RECOVAR global-winner summary is bounded to a positive particle count and K=4")
    if not 0 < max_bytes <= MAX_SUPPORTED_BYTES:
        raise RuntimeError(f"RECOVAR_GLOBAL_WINNER_SUMMARY_MAX_BYTES must be in (0, {MAX_SUPPORTED_BYTES}]")

    class_scores = np.asarray(full_stats["class_best_log_score_per_image"])
    class_second_scores = np.asarray(full_stats["class_second_best_log_score_per_image"])
    if class_scores.dtype != np.float32 or class_second_scores.dtype != np.float32:
        raise RuntimeError("RECOVAR global-winner summary requires native float32 best/second class scores")
    class_pose_indices = np.asarray(full_stats["class_hard_assignments"], dtype=np.int32)
    class_second_pose_indices = np.asarray(full_stats["class_second_hard_assignments"], dtype=np.int32)
    class_log_evidence = np.asarray(full_stats["class_log_evidence_per_image"], dtype=np.float64)
    winner_class = np.asarray(full_stats["class_assignments"], dtype=np.int32)
    global_log_z = np.asarray(full_stats["normalization_log_z"], dtype=np.float64)
    expected_shape = (n_classes, expected_particles)
    for name, array in (
        ("class_best_log_score_per_image", class_scores),
        ("class_second_best_log_score_per_image", class_second_scores),
        ("class_hard_assignments", class_pose_indices),
        ("class_second_hard_assignments", class_second_pose_indices),
        ("class_log_evidence_per_image", class_log_evidence),
    ):
        if array.shape != expected_shape:
            raise RuntimeError(f"{name} has shape {array.shape}, expected {expected_shape}")
    if winner_class.shape != (expected_particles,) or global_log_z.shape != (expected_particles,):
        raise RuntimeError("RECOVAR global-winner summary observed an unexpected particle axis")
    if (
        not np.all(np.isfinite(class_scores))
        or not np.all(np.isfinite(class_second_scores))
        or not np.all(np.isfinite(class_log_evidence))
    ):
        raise RuntimeError("RECOVAR global-winner summary refuses non-finite class scores/evidence")
    if np.any((winner_class < 0) | (winner_class >= n_classes)):
        raise RuntimeError("RECOVAR global-winner summary observed an invalid winner class")
    pose_count = int(n_rotations) * int(n_translations)
    if pose_count < 2:
        raise RuntimeError("RECOVAR global-winner summary requires at least two poses per class")
    if np.any((class_pose_indices < 0) | (class_pose_indices >= pose_count)) or np.any(
        (class_second_pose_indices < 0) | (class_second_pose_indices >= pose_count)
    ):
        raise RuntimeError("RECOVAR global-winner summary observed an invalid class-local pose index")
    if np.any(class_pose_indices == class_second_pose_indices):
        raise RuntimeError("RECOVAR global-winner summary best and second-best pose indices must differ")
    class_within_pose_margin = class_scores - class_second_scores
    if np.any(class_within_pose_margin < 0):
        raise RuntimeError("RECOVAR global-winner summary observed a negative within-class pose margin")

    columns = np.arange(expected_particles)
    winner_score = class_scores[winner_class, columns]
    if not np.array_equal(winner_score, np.max(class_scores, axis=0)):
        raise RuntimeError("RECOVAR winner classes are not optimal in the captured class scores")
    # The class-common normalization offset is added before storage as
    # float32, so distinct pre-offset scores can collapse to an exact tie.
    # Preserve the actual pre-offset winner and select the runner-up after
    # excluding it, using stable class order for any remaining tie.
    scores_without_winner = class_scores.copy()
    scores_without_winner[winner_class, columns] = -np.inf
    runner_up_class = np.argmax(scores_without_winner, axis=0).astype(np.int32)
    runner_up_score = class_scores[runner_up_class, columns]
    winner_margin = winner_score - runner_up_score
    class_posterior_mass = np.zeros((n_classes, expected_particles), dtype=np.float32)
    class_posterior_mass[winner_class, columns] = 1.0
    original_index = _original_indices(experiment_dataset, expected_particles)

    metadata = {
        "schema": SCHEMA,
        "engine": "recovar",
        "run_id": _required_env("RECOVAR_GLOBAL_WINNER_SUMMARY_RUN_ID"),
        "source_id": _required_env("RECOVAR_GLOBAL_WINNER_SUMMARY_SOURCE_ID"),
        "executable_sha256": _required_env("RECOVAR_GLOBAL_WINNER_SUMMARY_EXECUTABLE_SHA256"),
        "gpu_uuid": _required_env("RECOVAR_GLOBAL_WINNER_SUMMARY_GPU_UUID"),
        "input_manifest_sha256": _required_env("RECOVAR_GLOBAL_WINNER_SUMMARY_INPUT_MANIFEST_SHA256"),
        "dispatch_oracle_sha256": _required_env("RECOVAR_GLOBAL_WINNER_SUMMARY_DISPATCH_ORACLE_SHA256"),
        "iteration": target_iteration,
        "expected_particles": expected_particles,
        "expected_classes": expected_classes,
        "max_bytes": max_bytes,
        "score_mode": "firstiter_cc_normalized_log_score_higher_is_better",
        "raw_score_semantics": (
            "per-class best normalized-CC log score before class/orientation/translation priors; "
            "includes the class-common per-image normalization offset"
        ),
        "total_score_semantics": ("identical to raw score in firstiter_cc because RELION bypasses priors before WTA"),
        "evidence_semantics": "per-class logsumexp evidence before firstiter_cc one-hot posterior",
        "posterior_semantics": "post-firstiter_cc one-hot class mass",
        "winner_semantics": (
            "actual joint class-pose argmax before WTA; captured float32 scores may tie after "
            "adding the class-common normalization offset"
        ),
        "support_semantics": "post-firstiter_cc exactly one global class-pose sample per particle",
        "pre_wta_support_semantics": "all coarse candidates scored; no posterior threshold before WTA",
        "significant_count_semantics": "post-WTA global class-pose support cardinality; exactly one",
        "within_class_runner_up_semantics": (
            "second-highest distinct class-local coarse pose score before priors and WTA"
        ),
        "score_element_bytes": int(class_scores.dtype.itemsize),
        "class_index_convention": "zero_based",
        "pose_index_convention": "class_local_flat_rotation_translation",
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    output_path = Path(path_value).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_path,
            schema=np.asarray(SCHEMA),
            metadata_json=np.asarray(metadata_json),
            original_index_zero_based=original_index,
            class_best_raw_score_pre_prior=class_scores.T,
            class_best_total_score=class_scores.T,
            class_best_pose_index=class_pose_indices.T,
            class_second_best_raw_score_pre_prior=class_second_scores.T,
            class_second_best_total_score=class_second_scores.T,
            class_second_best_pose_index=class_second_pose_indices.T,
            class_within_pose_margin=class_within_pose_margin.T,
            class_log_evidence=class_log_evidence.T,
            class_posterior_mass=class_posterior_mass.T,
            global_log_z=global_log_z,
            winner_class_zero_based=winner_class,
            runner_up_class_zero_based=runner_up_class,
            winner_score=winner_score,
            runner_up_score=runner_up_score,
            winner_margin=winner_margin,
            significant_count=np.ones(expected_particles, dtype=np.int32),
            n_rotations=np.asarray(n_rotations, dtype=np.int64),
            n_translations=np.asarray(n_translations, dtype=np.int64),
        )
        size = temporary_path.stat().st_size
        if size > max_bytes:
            raise RuntimeError(f"RECOVAR global-winner summary is {size} bytes, exceeding cap {max_bytes}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
